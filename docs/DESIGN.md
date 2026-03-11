# DESIGN: Cost-Aware Research Search Loop v2

Task: `improve-loop-v2`

This document specifies the internal design for five improvements to `src/loop.py`.
The file must remain a single Python file, under 400 lines total.

---

## 0. Problem Summary

From two 4-iteration runs, the current loop has these observed failure modes:

| Problem | Root Cause | Evidence |
|---------|-----------|----------|
| Feedback not reaching code gen | CODE_GEN_PROMPT receives only idea_spec, not judge feedback | Run 2: "increase samples" repeated 4x by judge, never acted on |
| Theme fixation | No mechanism to explore alternative directions | Run 2: all 4 iters were CoT/embedding variants |
| Sample size always too small | No enforcement in prompt or Stage 0 | Run 2: 15-40 samples despite spec saying 300 |
| Weak repair | Error message appended to prompt with no structure | Run 1 iter 1: JSON serialization bug not repaired |
| No search strategy | Pure sequential improvement without exploration | Both runs: no explore/exploit tradeoff |

All five improvements below are designed to fit within the existing single-file
structure by adding small, pure functions and data structures at the top of the file,
then calling them from `run_loop`.

---

## 1. Feedback Propagation

### Current behavior

```
Judge feedback --> IMPROVE_PROMPT --> Claude --> improved idea_spec --> CODE_GEN_PROMPT
                                                                       (no feedback here)
```

The judge says "increase samples to 100+" but CODE_GEN_PROMPT only sees the idea_spec.
Claude code-gen makes its own sample-size decisions, ignoring the judge.

### Desired behavior

```
Judge feedback --+--> IMPROVE_PROMPT --> Claude --> improved idea_spec --+
                 |                                                       |
                 +--> CODE_GEN_PROMPT (new section: PRIOR FEEDBACK) -----+-> Claude --> code
```

### Design

Add a new template variable `{feedback_section}` to `CODE_GEN_PROMPT`. When feedback
exists from the previous iteration, format the judge's `improvement_suggestions` and
`overall.reason` into a clear block:

```python
CODE_GEN_FEEDBACK_BLOCK = """
## MANDATORY: Address These Judge Requirements
The previous iteration scored {score}/10. The judge identified these problems:
{suggestions}

You MUST address these in the generated code. In particular:
- If the judge mentions sample size, use AT LEAST the number they suggest
- If the judge mentions statistical tests, include them
- If the judge mentions baselines, add them
"""
```

In `run_loop`, build `feedback_for_codegen` from `feedback_history[-1]` when it exists,
and pass it as `feedback_section=feedback_for_codegen` to `CODE_GEN_PROMPT.format(...)`.

When `feedback_history` is empty (iteration 1), `feedback_section` is the empty string.

### Data flow

```
feedback_history[-1]  -->  format CODE_GEN_FEEDBACK_BLOCK  -->  CODE_GEN_PROMPT
                                                                  ^
                                                                  |
                                                         idea_spec (as before)
```

### Invariants

- `CODE_GEN_PROMPT` always receives a `feedback_section` argument (empty string or filled).
- The feedback block is inserted BEFORE the idea_spec in the prompt so Claude reads
  constraints before the spec details.

---

## 2. Thompson Sampling for Research Direction Selection

### Current behavior

Iteration 1 generates a random direction. Subsequent iterations improve the same idea.
No mechanism to try alternative directions or return to previously good ones.

### Desired behavior

Maintain a set of "arms" (research directions). Use Thompson Sampling to choose which
direction to explore each iteration. Update arm statistics after judging.

### Design

#### Data structure: `Arms`

```python
# Predefined research directions. Each arm is a dict.
RESEARCH_DIRECTIONS = [
    {"name": "prompt_engineering", "description": "How prompt structure affects LLM outputs"},
    {"name": "embedding_analysis", "description": "Properties and behaviors of text embeddings"},
    {"name": "model_comparison",   "description": "Behavioral differences between LLM models/sizes"},
    {"name": "reasoning_analysis", "description": "Analysis of LLM reasoning patterns and errors"},
    {"name": "text_statistics",    "description": "Statistical properties of LLM-generated text"},
]

# State per arm: Beta(alpha, beta) for Thompson Sampling
# Initial: Beta(1, 1) = uniform prior
arm_states = [{"alpha": 1.0, "beta": 1.0} for _ in RESEARCH_DIRECTIONS]
```

#### Function: `select_arm(arm_states, history, rng) -> int`

1. For each arm, draw `theta_i ~ Beta(alpha_i, beta_i)`.
2. Apply the diversity constraint (see section 5) to mask recently-used arms.
3. Among unmasked arms, pick `argmax(theta_i)`.
4. Return the index.

This function is **pure** (given rng state) and **testable** without any LLM calls.

#### Function: `update_arm(arm_states, arm_idx, score) -> None`

Convert the 1-10 judge score to a reward in [0, 1]:

```python
reward = max(0.0, (score - 2)) / 8.0  # score 2 -> 0.0, score 10 -> 1.0
```

Update:
```python
arm_states[arm_idx]["alpha"] += reward
arm_states[arm_idx]["beta"]  += (1.0 - reward)
```

This is a standard Bernoulli bandit Beta update with a soft reward mapping.

#### Integration into `run_loop`

- Before idea generation, call `select_arm` to choose a direction.
- Pass the direction name and description into `IDEA_GEN_PROMPT` as a new field
  `research_direction`.
- After judging, call `update_arm` with the overall score.
- Log `arm_idx`, `arm_name`, and `arm_states` snapshot in `iter_data`.

### Prompt modification for IDEA_GEN_PROMPT

Add to the prompt:

```
Research direction for this iteration: {direction_name}
Description: {direction_description}

Generate an idea WITHIN this research direction. Do not deviate to other topics.
```

### Invariants

- `arm_states` is never empty (always len == len(RESEARCH_DIRECTIONS)).
- `select_arm` always returns a valid index even if all arms are masked
  (falls back to global argmax).

---

## 3. Sample Size Enforcement

### Current behavior

The idea spec template has `n_samples: 0` as default. Code gen uses whatever the
spec says (often 10-40). The judge then complains about insufficient samples.

### Desired behavior

1. **Stage 0 enforces minimum samples**: reject specs with `n_samples < MIN_SAMPLES`.
2. **Code gen prompt hard-codes a floor**: even if spec says 20, the prompt says "at least 50".
3. **Idea gen prompt sets expectations**: tell Claude the minimum upfront.

### Design

#### Constant

```python
MIN_SAMPLES = 50
```

#### Stage 0 addition

Add a check in `stage0_check`:

```python
n_samples = spec.get("full_evaluation", {}).get("n_samples", 0)
checks["sufficient_samples"] = n_samples >= MIN_SAMPLES
```

This field participates in the `passed` computation.

#### Code gen prompt addition

Add to `CODE_GEN_PROMPT`:

```
- MINIMUM SAMPLE SIZE: {min_samples} data points. Do NOT use fewer samples than this.
  If API costs are a concern, use synthetic/local data generation to reach this minimum.
```

#### Idea gen prompt addition

Add to `IDEA_GEN_SYSTEM`:

```
- Minimum sample size: 50. Set n_samples to at least 50 in full_evaluation.
```

### Invariants

- Every spec that passes Stage 0 has `n_samples >= 50`.
- Every code gen prompt includes `min_samples=50` (or the configured value).

---

## 4. Failure Memory

### Current behavior

When code fails, the error is passed to a single repair attempt. If repair fails,
a dummy score (2/10) is assigned and the loop continues. Past failures are not
available to future iterations.

### Desired behavior

Maintain a list of past failures. Inject a summary of relevant failures into both
idea improvement and code generation prompts, so Claude avoids repeating the same
mistakes.

### Design

#### Data structure

```python
# Maintained across iterations within a run
failure_memory: list[dict] = []

# Each entry:
{
    "iteration": int,
    "error_type": str,          # "runtime_error" | "timeout" | "json_parse" | "import_error"
    "error_summary": str,       # 1-2 line summary
    "direction": str,           # which research direction was being explored
    "code_snippet": str,        # the problematic line(s), max 3 lines
}
```

#### Function: `classify_error(stderr: str) -> str`

Simple keyword-based classification:

```python
def classify_error(stderr: str) -> str:
    s = stderr.lower()
    if "timeout" in s:
        return "timeout"
    if "json" in s or "serializ" in s:
        return "json_parse"
    if "import" in s or "modulenotfound" in s:
        return "import_error"
    if "memory" in s or "oom" in s:
        return "memory_error"
    return "runtime_error"
```

#### Function: `format_failure_memory(failure_memory: list[dict], max_entries: int = 3) -> str`

Returns a formatted string of the most recent `max_entries` failures for prompt injection:

```
## Past Failures (avoid repeating these)
- Iter 1 [runtime_error]: JSON serialization failed on numpy types. Use .item() for numpy scalars.
- Iter 2 [timeout]: API calls took too long. Reduce to batch calls.
```

If `failure_memory` is empty, returns the empty string.

#### Integration

- After any execution failure (first attempt or repair), append to `failure_memory`.
- Pass `format_failure_memory(failure_memory)` into:
  - `IMPROVE_PROMPT` via `{failure_info}` (already exists, currently always "None")
  - `CODE_GEN_PROMPT` via a new `{failure_memory}` section

### Invariants

- `failure_memory` only grows within a run (never truncated).
- `format_failure_memory` always returns a string (empty if no failures).
- Error classification is deterministic and has no external dependencies.

---

## 5. Diversity Constraint

### Current behavior

No mechanism prevents the same direction from being picked repeatedly. In Run 2,
all 4 iterations explored the same CoT/embedding theme.

### Desired behavior

Prevent any single direction from being selected more than 2 times consecutively.
After 2 consecutive picks of the same arm, mask it for the next iteration.

### Design

#### Function: `apply_diversity_mask(history: list[int], n_arms: int, max_consecutive: int = 2) -> list[bool]`

```python
def apply_diversity_mask(history: list[int], n_arms: int, max_consecutive: int = 2) -> list[bool]:
    """Return a boolean mask where True = arm is allowed, False = blocked.

    An arm is blocked if it was selected for the last `max_consecutive` iterations.
    If ALL arms would be blocked (impossible in practice with >= 3 arms and max_consecutive=2),
    return all-True.
    """
    mask = [True] * n_arms
    if len(history) >= max_consecutive:
        tail = history[-max_consecutive:]
        if len(set(tail)) == 1:
            blocked = tail[0]
            mask[blocked] = False
    # Safety: if all blocked (shouldn't happen with n_arms >= 3), unblock all
    if not any(mask):
        mask = [True] * n_arms
    return mask
```

#### Integration with Thompson Sampling

In `select_arm`:

```python
def select_arm(arm_states, history, rng):
    n = len(arm_states)
    mask = apply_diversity_mask(history, n)
    samples = [rng.beta(s["alpha"], s["beta"]) if mask[i] else -1.0
               for i, s in enumerate(arm_states)]
    return int(max(range(n), key=lambda i: samples[i]))
```

### Invariants

- With 5 arms and `max_consecutive=2`, at most 1 arm is blocked at any time.
- The function is pure and deterministic given its inputs (no randomness inside).
- `history` is a list of arm indices, one per past iteration.

---

## 6. Module Boundaries and File Layout

All code stays in `src/loop.py`. The file is organized into these logical sections:

```
src/loop.py  (target: ~350 lines)
|
|-- [Lines 1-35]    Imports, env loading, client setup
|-- [Lines 36-55]   Constants (MIN_SAMPLES, RESEARCH_DIRECTIONS, etc.)
|-- [Lines 56-120]  Prompt templates (IDEA_GEN_SYSTEM, CODE_GEN_PROMPT, etc.)
|-- [Lines 121-160] Pure functions: select_arm, update_arm, apply_diversity_mask
|-- [Lines 161-185] Pure functions: classify_error, format_failure_memory
|-- [Lines 186-215] LLM wrappers: call_claude, call_judge
|-- [Lines 216-240] Execution: run_code, stage0_check
|-- [Lines 241-260] Logging: save_iteration
|-- [Lines 261-400] run_loop (main orchestration)
```

### Function signatures (contracts)

```python
# --- Thompson Sampling ---
def select_arm(arm_states: list[dict], history: list[int], rng) -> int:
    """Choose an arm index via Thompson Sampling with diversity constraint."""

def update_arm(arm_states: list[dict], arm_idx: int, score: int) -> None:
    """Update Beta distribution for the selected arm. Mutates arm_states in place."""

def apply_diversity_mask(history: list[int], n_arms: int, max_consecutive: int = 2) -> list[bool]:
    """Return mask of allowed arms based on recent history."""

# --- Failure Memory ---
def classify_error(stderr: str) -> str:
    """Classify error stderr into a category string."""

def format_failure_memory(failure_memory: list[dict], max_entries: int = 3) -> str:
    """Format recent failures for prompt injection."""

# --- Existing (modified) ---
def stage0_check(idea_spec: str) -> dict:
    """Stage 0 feasibility check. Now also checks n_samples >= MIN_SAMPLES."""

def run_loop(n_iterations: int = 4, max_api_calls: int = 30, time_limit_min: int = 5) -> dict:
    """Main loop. Now uses Thompson Sampling, feedback propagation, failure memory."""
```

---

## 7. Data Flow Diagram (v2)

```
                    RESEARCH_DIRECTIONS
                           |
                    select_arm (Thompson Sampling + diversity mask)
                           |
                    chosen direction
                           |
    +-----------+          v          +-----------------+
    | feedback  |---> IDEA_GEN_PROMPT |                 |
    | history   |     or IMPROVE_PROMPT                 |
    +-----------+          |          |                  |
                           v          |                  |
                     Claude Sonnet    |                  |
                     (idea spec)      |                  |
                           |          |                  |
                     stage0_check     |                  |
                     (+ n_samples     |                  |
                      >= 50 check)    |                  |
                           |          |                  |
                           v          v                  |
                     CODE_GEN_PROMPT                     |
                     + feedback_section  <-- feedback_history[-1]
                     + failure_memory    <-- failure_memory
                     + min_samples=50                    |
                           |                             |
                           v                             |
                     Claude Sonnet                       |
                     (experiment code)                   |
                           |                             |
                     run_code (subprocess)               |
                           |                             |
                      fail?--yes--> classify_error       |
                       |            append to            |
                       |            failure_memory       |
                       |            attempt repair       |
                       |                                 |
                       v                                 |
                     call_judge (GPT-5.4-pro)            |
                           |                             |
                     update_arm(score)                   |
                     append to feedback_history          |
                           |                             |
                           +-----------------------------+
                                  next iteration
```

---

## 8. Changes to Prompt Templates

### IDEA_GEN_SYSTEM -- add at end

```
- Minimum sample size: 50 data points. Always set n_samples >= 50 in full_evaluation.
```

### IDEA_GEN_PROMPT -- add direction constraint

```
Research direction: {direction_name}
Direction description: {direction_description}

Generate an idea strictly WITHIN this research direction.
```

### CODE_GEN_PROMPT -- add two new sections

```
{feedback_section}

{failure_memory}

...existing prompt...

- MINIMUM SAMPLE SIZE: {min_samples} data points. Do NOT generate fewer samples.
```

### IMPROVE_PROMPT -- failure_info is now populated

The existing `{failure_info}` placeholder is already in IMPROVE_PROMPT.
The Coder just needs to pass `format_failure_memory(failure_memory)` instead of
the current `iter_data.get("failure_info", "None")`.

---

## 9. What the Coder Should Implement

1. Add `RESEARCH_DIRECTIONS` and `MIN_SAMPLES` constants.
2. Add the 5 pure functions: `select_arm`, `update_arm`, `apply_diversity_mask`,
   `classify_error`, `format_failure_memory`.
3. Modify `stage0_check` to include the `sufficient_samples` check.
4. Modify prompt templates as described in section 8.
5. Modify `run_loop` to:
   - Initialize `arm_states`, `arm_history`, `failure_memory`.
   - Call `select_arm` before idea generation.
   - Pass direction to idea gen prompt.
   - Build `feedback_for_codegen` and pass to code gen prompt.
   - Build `failure_memory_text` and pass to code gen prompt.
   - On failure, call `classify_error`, append to `failure_memory`.
   - After judging, call `update_arm`.
   - Log arm selection and states.

### Lines budget

| Section | Current lines | New lines (est.) |
|---------|--------------|-----------------|
| Imports + env | 31 | 31 (no change) |
| Constants | 0 | 15 |
| Templates | ~100 | ~120 (+20 for new sections) |
| Pure functions | 0 | 50 |
| LLM wrappers | 20 | 20 (no change) |
| run_code + stage0 | 30 | 35 (+5 for n_samples check) |
| save_iteration | 5 | 5 (no change) |
| run_loop | 120 | 100 (streamlined with helpers) |
| argparse | 8 | 8 (no change) |
| **Total** | **~315** | **~385** |

Fits within the 400-line budget.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Thompson Sampling converges too fast on one arm | Warm prior Beta(1,1) is intentionally flat; 4 iterations is too few for premature convergence |
| Diversity mask starves the best arm | `max_consecutive=2` is lenient; with 5 arms it only blocks 1 at a time |
| Feedback in code gen confuses Claude | Feedback block is clearly labeled "MANDATORY" and placed before the spec |
| MIN_SAMPLES=50 makes experiments too expensive | 50 is low; synthetic data generation costs zero API calls |
| Failure memory grows large | Capped to `max_entries=3` in prompt injection |

---

## 11. Open Questions for the Specifier

None. The requirements in CURRENT_TASK.md are sufficiently clear for this design.
If the Coder finds that 400 lines is too tight, the first thing to extract would be
the prompt templates into `templates/prompts.py`, but this should be avoided unless
necessary.

#!/usr/bin/env python3
"""Cost-Aware Research Search: Main Loop with Thompson Sampling and feedback propagation."""

import json, os, random, subprocess, tempfile, time
from datetime import datetime
from pathlib import Path

ENV_PATH = Path.home() / "Library/CloudStorage/Dropbox/secrets/.env"
for line in ENV_PATH.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI(timeout=600.0)
anthropic_client = Anthropic()
REPO_ROOT = Path(__file__).parent.parent
LOGS_DIR = REPO_ROOT / "logs" / "runs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_SAMPLES = 50

RESEARCH_DIRECTIONS = [
    {"name": "prompt_engineering", "description": "How prompt structure affects LLM outputs"},
    {"name": "embedding_analysis", "description": "Properties and behaviors of text embeddings"},
    {"name": "model_comparison", "description": "Behavioral differences between LLM models/sizes"},
    {"name": "reasoning_analysis", "description": "Analysis of LLM reasoning patterns and errors"},
    {"name": "text_statistics", "description": "Statistical properties of LLM-generated text"},
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
IDEA_SPEC_TEMPLATE = (REPO_ROOT / "templates" / "idea_spec.yaml").read_text()

IDEA_GEN_SYSTEM = """You are an AI research scientist generating structured research ideas
executable 100% locally on a Mac M1 laptop in under 10 minutes with no GPU and NO API calls.

CRITICAL CONSTRAINTS:
- Python only, no GPU, no model training, NO INTERNET/API CALLS
- ALL data must be generated locally (synthetic data, algorithmic generation, local files)
- Must have a clear baseline comparison
- Runtime < 10 min on Apple M1 CPU
- Minimum n_samples: 50 data points (set n_samples >= 50 in the spec)
- ALLOWED LIBRARIES ONLY: json, os, sys, time, math, random, re, collections, statistics,
  numpy, scipy, sklearn, matplotlib, sentence_transformers
- Do NOT use: openai, anthropic, textstat, nltk, spacy, pandas, transformers, torch, requests, httpx
- Do NOT make any HTTP requests or API calls

IMPORTANT: Generate COMPLETELY NEW and DIVERSE ideas. Do NOT repeat any of these past themes:
embedding clustering, syntactic complexity, CoT prompting, silhouette score, paraphrase distance.

Good LOCAL experiment ideas: algorithm comparison on synthetic data, statistical distribution analysis,
optimization algorithm benchmarking, clustering algorithm comparison, dimensionality reduction analysis,
random graph properties, time-series analysis on synthetic signals, feature selection methods comparison,
sampling strategies comparison, numerical precision analysis, etc."""

IDEA_GEN_PROMPT = """Generate a research idea as a YAML idea spec.
Research direction: {direction_name}
Direction description: {direction_description}
Generate an idea strictly WITHIN this research direction.
{feedback_section}
Return ONLY valid YAML (no markdown code fences).
Follow this template:
{template}"""

CODE_GEN_SYSTEM = """You are an expert Python programmer for research experiments.
Write self-contained, executable scripts that run 100% LOCALLY on Mac M1 CPU in <10 min.
NO API calls, NO internet access. All data must be generated locally.

ALLOWED LIBRARIES ONLY (do NOT import anything else):
- Standard library: json, os, sys, time, math, random, re, collections, statistics, pathlib, datetime
- numpy, scipy
- sklearn (scikit-learn)
- matplotlib
- sentence_transformers (SentenceTransformer) - for LOCAL embedding only, no API
Do NOT use: openai, anthropic, requests, httpx, textstat, nltk, spacy, pandas, transformers, torch, tensorflow.
Do NOT make any HTTP requests or API calls.

CRITICAL CODE QUALITY RULES:
- ALL numpy values must be converted to float() before JSON serialization
- Always print progress (e.g. "Processing 5/50...")
- At the end, verify you have >= 50 valid data points before saving
- Use json.dumps with default=str as fallback for serialization
- Print a clear results table at the very end with all conditions
- Use random seeds for reproducibility (e.g. np.random.seed(42))"""

CODE_GEN_PROMPT = """{feedback_section}
{failure_memory}
Write a complete Python experiment script based on this idea spec:
```yaml
{idea_spec}
```
Requirements: save JSON to {results_path}, summary table,
runtime < {time_limit_min} min, NO API calls, NO internet, all data generated locally.
- MINIMUM SAMPLE SIZE: {min_samples} data points. Do NOT generate fewer samples.
- ONLY use allowed libraries (see system prompt). No pip install. No API calls.
- Print progress and final results table to stdout.
- Use random seeds for reproducibility.
Return ONLY Python code, no markdown fences."""

JUDGE_PROMPT = """Evaluate this experiment.
## Idea Spec
```yaml
{idea_spec}
```
## Results
```
{results_stdout}
```
## Data
```json
{results_json}
```
Score 1-10 on novelty, rigor, significance, completeness. Provide overall (1-10), key_finding, improvement_suggestions (3 items).
Return JSON only:
{{"novelty": {{"score": N, "reason": "..."}}, "rigor": {{"score": N, "reason": "..."}}, "significance": {{"score": N, "reason": "..."}}, "completeness": {{"score": N, "reason": "..."}}, "overall": {{"score": N, "reason": "..."}}, "key_finding": "...", "improvement_suggestions": ["...", "...", "..."]}}"""

IMPROVE_PROMPT = """Previously generated idea spec:
```yaml
{prev_spec}
```
Feedback: **Score**: {overall_score}/10 | **Key Finding**: {key_finding}
**Suggestions**:
{suggestions}
**Failure Info**: {failure_info}
Generate an IMPROVED idea spec addressing suggestions, building on findings, CPU-only, <10 min.
Return ONLY valid YAML."""

# ---------------------------------------------------------------------------
# Pure functions: Thompson Sampling + Diversity
# ---------------------------------------------------------------------------
def apply_diversity_mask(history: list[int], n_arms: int, max_consecutive: int = 2) -> list[bool]:
    """Return boolean mask: True = arm allowed, False = blocked by consecutive use."""
    mask = [True] * n_arms
    if len(history) >= max_consecutive:
        tail = history[-max_consecutive:]
        if len(set(tail)) == 1:
            mask[tail[0]] = False
    if not any(mask):
        mask = [True] * n_arms
    return mask


_TS_TEMP = 3.0  # posterior temperature: sharpens sampling toward the mean

def select_arm(arm_states: list[dict], history: list[int], rng: random.Random,
               max_consecutive: int = 2) -> int:
    """Choose arm via Thompson Sampling with diversity constraint.

    Uses posterior temperature scaling (_TS_TEMP) to concentrate samples
    around the posterior mean, improving exploit vs. explore balance.
    """
    n = len(arm_states)
    mask = apply_diversity_mask(history, n, max_consecutive)
    k = _TS_TEMP
    samples = [rng.betavariate(s["alpha"] * k, s["beta"] * k) if mask[i] else -1.0
               for i, s in enumerate(arm_states)]
    return int(max(range(n), key=lambda i: samples[i]))


def update_arm(arm_states: list[dict], arm_idx: int, score: int) -> None:
    """Update Beta distribution for selected arm. reward = max(0,(score-2))/8.

    Both alpha and beta always increase at least by a negligible amount,
    ensuring the posterior never stalls completely on either parameter.
    """
    reward = max(0.0, (score - 2)) / 8.0
    arm_states[arm_idx]["alpha"] += reward + 1e-9
    arm_states[arm_idx]["beta"] += (1.0 - reward) + 1e-9


# ---------------------------------------------------------------------------
# Pure functions: Failure Memory
# ---------------------------------------------------------------------------
def classify_error(stderr: str) -> str:
    """Classify error stderr into a category string."""
    s = stderr.lower()
    if "timeout" in s: return "timeout"
    if "json" in s or "serializ" in s: return "json_parse"
    if "import" in s or "modulenotfound" in s: return "import_error"
    if "memory" in s or "oom" in s: return "memory_error"
    return "runtime_error"


def format_failure_memory(failures: list[dict], max_entries: int = 3) -> str:
    """Format recent failures for prompt injection. Empty string if none."""
    if not failures:
        return ""
    recent = failures[-max_entries:]
    lines = ["## Past Failures (avoid repeating these)"]
    for f in recent:
        lines.append(f"- Iter {f['iteration']} [{f['error_type']}]: {f['error_summary']}. Direction: {f['direction']}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM wrappers and execution
# ---------------------------------------------------------------------------
def call_gpt(system: str, user: str) -> str:
    """Call GPT-5.4-pro via Responses API for idea gen."""
    full_input = system + "\n\n" + user
    resp = openai_client.responses.create(model="gpt-5.4-pro", input=full_input)
    return resp.output_text


def call_claude_code(system: str, user: str, max_tokens: int = 8000) -> str:
    """Call Claude Sonnet 4 for code generation. Fast, reliable, good at coding."""
    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user}])
    return resp.content[0].text


def call_judge(prompt: str, use_pro: bool = True) -> dict:
    full_input = "You are a rigorous research evaluator. Return only valid JSON.\n\n" + prompt
    resp = openai_client.responses.create(model="gpt-5.4-pro", input=full_input)
    raw = resp.output_text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def run_code(code: str, timeout_sec: int = 600) -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code); f.flush()
        try:
            r = subprocess.run(["python3", f.name], capture_output=True, text=True,
                               timeout=timeout_sec, env={**os.environ})
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {timeout_sec} seconds", 1
        finally:
            os.unlink(f.name)


def stage0_check(idea_spec: str) -> dict:
    """Stage 0: Static feasibility check including sample size enforcement."""
    checks = {"valid_yaml": False, "has_hypothesis": False, "has_metric": False,
              "cpu_only": False, "time_ok": False, "sufficient_samples": False}
    try:
        import yaml
        spec = yaml.safe_load(idea_spec)
        checks["valid_yaml"] = True
        checks["has_hypothesis"] = bool(spec.get("hypothesis"))
        checks["has_metric"] = bool(spec.get("proxy_evaluation", {}).get("metric"))
        checks["cpu_only"] = spec.get("implementation_scope", {}).get("compute", "").lower() in ("cpu", "cpu_only")
        checks["time_ok"] = spec.get("implementation_scope", {}).get("time_estimate_min", 999) <= 15
        checks["sufficient_samples"] = spec.get("full_evaluation", {}).get("n_samples", 0) >= MIN_SAMPLES
    except Exception as e:
        checks["error"] = str(e)
    checks["passed"] = all(v for k, v in checks.items() if k != "error" and isinstance(v, bool))
    return checks


def _strip_fences(text: str) -> str:
    if text.strip().startswith("```"):
        text = text.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def _save(iteration: int, data: dict, run_dir: Path):
    (run_dir / f"iter_{iteration:02d}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _dummy_feedback():
    return {"novelty": {"score": 3, "reason": "failed"}, "rigor": {"score": 1, "reason": "failed"},
            "significance": {"score": 3, "reason": "failed"}, "completeness": {"score": 1, "reason": "failed"},
            "overall": {"score": 2, "reason": "Experiment failed to execute"},
            "key_finding": "Experiment failed to execute.",
            "improvement_suggestions": ["Simplify experiment", "Fewer API calls", "Better error handling"]}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_loop(n_iterations: int = 3, max_api_calls: int = 0, time_limit_min: int = 5):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = REPO_ROOT / "results"; results_dir.mkdir(exist_ok=True)
    print(f"{'='*70}\nRESEARCH SEARCH LOOP | {n_iterations} iters | {run_id}\n{'='*70}")

    idea_spec, feedback_history, all_scores = None, [], []
    arm_states = [{"alpha": 1.0, "beta": 1.0} for _ in RESEARCH_DIRECTIONS]
    arm_history, failure_memory, rng = [], [], random.Random()

    for it in range(1, n_iterations + 1):
        is_final = (it == n_iterations)
        d = {"iteration": it, "timestamp": datetime.now().isoformat()}

        # Select direction
        arm_idx = select_arm(arm_states, arm_history, rng)
        direction = RESEARCH_DIRECTIONS[arm_idx]
        arm_history.append(arm_idx)
        d.update(arm_idx=arm_idx, arm_name=direction["name"], arm_states=[dict(s) for s in arm_states])
        print(f"\n{'='*60}")
        print(f"  ITER {it}/{n_iterations} | Direction: {direction['name']}")
        print(f"{'='*60}")

        # Step 1: Generate / improve idea spec
        print("  [1] Generating idea spec...", flush=True)
        prev_arm = arm_history[-2] if len(arm_history) >= 2 else None
        direction_changed = (prev_arm is not None and prev_arm != arm_idx)

        if idea_spec is None or direction_changed:
            # Fresh idea for new direction (don't inherit old spec from different direction)
            fb_context = "First iteration. Generate a fresh idea."
            if direction_changed and feedback_history:
                fb = feedback_history[-1]
                fb_context = (f"Previous direction scored {fb['overall']['score']}/10. "
                              f"Now exploring '{direction['name']}' instead. Generate a COMPLETELY NEW idea.")
            idea_spec = _strip_fences(call_gpt(IDEA_GEN_SYSTEM, IDEA_GEN_PROMPT.format(
                direction_name=direction["name"], direction_description=direction["description"],
                feedback_section=fb_context, template=IDEA_SPEC_TEMPLATE)))
        else:
            # Same direction: improve based on feedback
            fb = feedback_history[-1]
            sug = "\n".join(f"- {s}" for s in fb.get("improvement_suggestions", []))
            fm = format_failure_memory(failure_memory)
            idea_spec = _strip_fences(call_gpt(IDEA_GEN_SYSTEM, IMPROVE_PROMPT.format(
                prev_spec=idea_spec, overall_score=fb["overall"]["score"],
                key_finding=fb.get("key_finding", "N/A"), suggestions=sug, failure_info=fm or "None")))
        d["idea_spec"] = idea_spec
        for line in idea_spec.split("\n"):
            if line.strip().startswith("hypothesis:"):
                print(f"      Hypothesis: {line.split(':', 1)[1].strip()[:90]}", flush=True)
                break

        # Step 2: Stage 0
        print("  [2] Stage 0 check...", end=" ", flush=True)
        s0 = stage0_check(idea_spec); d["stage0"] = s0
        if not s0["passed"]:
            print(f"FAIL {s0}", flush=True)
            print("      Regenerating...", flush=True)
            idea_spec = _strip_fences(call_gpt(IDEA_GEN_SYSTEM, IDEA_GEN_PROMPT.format(
                direction_name=direction["name"], direction_description=direction["description"],
                feedback_section="Previous spec failed validation. Ensure all fields, cpu, time<=10, n_samples>=50.",
                template=IDEA_SPEC_TEMPLATE)))
            s0 = stage0_check(idea_spec); d["idea_spec"] = idea_spec; d["stage0_retry"] = s0
            print(f"      Retry: {'PASS' if s0['passed'] else 'FAIL'}", flush=True)
        else:
            print("PASS", flush=True)

        # Step 3: Code gen with feedback
        results_path = str(results_dir / f"iter_{it:02d}_results.json")
        fb_section = ""
        if feedback_history:
            fb = feedback_history[-1]
            sug = "\n".join(f"- {s}" for s in fb.get("improvement_suggestions", []))
            fb_section = f"## MANDATORY: Address Judge Requirements\nPrevious score: {fb['overall']['score']}/10\n{sug}\nYou MUST address these."
        code_kw = dict(idea_spec=idea_spec, results_path=results_path, max_api_calls=max_api_calls,
                       time_limit_min=time_limit_min, feedback_section=fb_section,
                       failure_memory=format_failure_memory(failure_memory), min_samples=MIN_SAMPLES)
        print(f"  [3] Generating code (Claude)...", end=" ", flush=True)
        code = _strip_fences(call_claude_code(CODE_GEN_SYSTEM, CODE_GEN_PROMPT.format(**code_kw)))
        d["code"] = code; (run_dir / f"iter_{it:02d}_experiment.py").write_text(code)
        print(f"{len(code.splitlines())} lines", flush=True)

        # Step 4: Run
        print(f"  [4] Running experiment...", flush=True)
        t0 = time.time()
        stdout, stderr, rc = run_code(code, timeout_sec=time_limit_min * 60 + 60)
        elapsed = time.time() - t0
        d["execution"] = {"returncode": rc, "elapsed_sec": round(elapsed, 1)}

        if rc != 0:
            print(f"      FAILED (rc={rc}, {time.time()-t0:.0f}s)", flush=True)
            err_lines = stderr.strip().splitlines()[-10:]
            for el in err_lines[-3:]:
                print(f"        {el}", flush=True)
            d["failure_info"] = "\n".join(err_lines)
            failure_memory.append({"iteration": it, "error_type": classify_error(stderr),
                "error_summary": err_lines[-1] if err_lines else "Unknown",
                "direction": direction["name"], "code_snippet": "\n".join(err_lines[-3:])})
            # Repair
            repair_prompt = CODE_GEN_PROMPT.format(**{**code_kw, "failure_memory": format_failure_memory(failure_memory)})
            repair_prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{chr(10).join(err_lines)}\nFix the error."
            code = _strip_fences(call_claude_code(CODE_GEN_SYSTEM, repair_prompt))
            (run_dir / f"iter_{it:02d}_repair.py").write_text(code)
            stdout, stderr, rc = run_code(code, timeout_sec=time_limit_min * 60 + 60)
            if rc != 0:
                failure_memory.append({"iteration": it, "error_type": classify_error(stderr),
                    "error_summary": "Repair failed: " + (stderr.strip().splitlines()[-1] if stderr.strip() else "Unknown"),
                    "direction": direction["name"], "code_snippet": "\n".join(stderr.strip().splitlines()[-3:])})
                feedback = _dummy_feedback()
                feedback_history.append(feedback); all_scores.append(2)
                update_arm(arm_states, arm_idx, score=2); d["judgment"] = feedback
                _save(it, d, run_dir); continue

        print(f"      OK ({time.time()-t0:.0f}s)", flush=True)
        out_lines = stdout.strip().splitlines()
        for ol in out_lines[-5:]:
            print(f"        {ol}", flush=True)

        # Step 5: Judge
        print(f"  [5] Judging (GPT-5.4-pro)...", end=" ", flush=True)
        results_json = Path(results_path).read_text() if Path(results_path).exists() else "{}"
        d["stdout"] = stdout
        try:
            feedback = call_judge(JUDGE_PROMPT.format(idea_spec=idea_spec,
                results_stdout="\n".join(out_lines[-40:]), results_json=results_json[:3000]))
        except Exception:
            feedback = {"novelty": {"score": 4, "reason": "error"}, "rigor": {"score": 4, "reason": "error"},
                        "significance": {"score": 4, "reason": "error"}, "completeness": {"score": 4, "reason": "error"},
                        "overall": {"score": 4, "reason": "error"}, "key_finding": "Judge failed",
                        "improvement_suggestions": ["Clearer output", "More baselines", "More samples"]}

        feedback_history.append(feedback)
        overall = feedback["overall"]["score"]
        all_scores.append(overall)
        update_arm(arm_states, arm_idx, score=overall)
        d["judgment"] = feedback
        print(f"  Score: {overall}/10 | {feedback.get('key_finding', 'N/A')[:100]}")
        _save(it, d, run_dir)

    # Summary
    summary = {"run_id": run_id, "scores": all_scores, "final_idea_spec": idea_spec,
               "final_judgment": feedback_history[-1] if feedback_history else None,
               "score_progression": [{"iteration": i+1, "score": s, "key_finding": fb.get("key_finding", "")}
                                      for i, (s, fb) in enumerate(zip(all_scores, feedback_history))]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDone. Scores: {all_scores}. Logs: {run_dir}/")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--iterations", type=int, default=3)
    p.add_argument("--max-api-calls", type=int, default=0)
    p.add_argument("--time-limit", type=int, default=5)
    a = p.parse_args()
    run_loop(n_iterations=a.iterations, max_api_calls=a.max_api_calls, time_limit_min=a.time_limit)

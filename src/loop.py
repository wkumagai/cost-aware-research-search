#!/usr/bin/env python3
"""
Cost-Aware Research Search: Main Loop

Implements the simplified pipeline:
  1. Idea Spec Generation (Claude)
  2. Stage 0: Static Feasibility Check
  3. Stage 1: Code Generation + Smoke Test
  4. Stage 2: Full Proxy Experiment
  5. Judge (GPT-4o for iterations, GPT-5.4-pro for final)
  6. Feedback → Improved Idea Spec
  7. Repeat
"""

import json
import os
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
ENV_PATH = Path.home() / "Library/CloudStorage/Dropbox/secrets/.env"
for line in ENV_PATH.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI()
anthropic_client = Anthropic()

REPO_ROOT = Path(__file__).parent.parent
LOGS_DIR = REPO_ROOT / "logs" / "runs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
IDEA_SPEC_TEMPLATE = (REPO_ROOT / "templates" / "idea_spec.yaml").read_text()

IDEA_GEN_SYSTEM = """You are an AI research scientist. You generate structured research ideas
that can be executed locally on a laptop in under 10 minutes with no GPU.

CONSTRAINTS:
- Python only, no GPU, no model training
- Can use APIs (OpenAI, Anthropic) but keep calls under 50
- Can use local libraries: numpy, scipy, sklearn, sentence-transformers, matplotlib
- Must produce quantitative results (numbers, tables, charts)
- Must have a clear baseline comparison
- Total runtime < 10 minutes on CPU
"""

IDEA_GEN_PROMPT = """Generate a research idea as a YAML idea spec.

Research domain: LLM behavior analysis, prompt engineering, or text/embedding analysis.
The experiment must run locally on a laptop CPU in under 10 minutes.

{feedback_section}

Return ONLY valid YAML (no markdown code fences, no explanation).
Follow this template structure:

{template}
"""

CODE_GEN_SYSTEM = """You are an expert Python programmer for research experiments.
You write self-contained, executable Python scripts that:
- Run on CPU in under 10 minutes
- Print clear results tables to stdout
- Save results to a JSON file at the path given
- Handle errors gracefully
- Include all imports at the top
"""

CODE_GEN_PROMPT = """Write a complete, self-contained Python experiment script based on this idea spec:

```yaml
{idea_spec}
```

Requirements:
- Load API keys from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY are already set)
- Save results as JSON to: {results_path}
- Print a clear summary table at the end
- Keep API calls under {max_api_calls} total
- Target runtime: under {time_limit_min} minutes
- Include proper error handling
- Use `max_completion_tokens` (not `max_tokens`) for OpenAI chat models

Return ONLY the Python code, no markdown fences, no explanation.
"""

JUDGE_PROMPT = """You are a research quality evaluator. Evaluate this experiment.

## Idea Spec
```yaml
{idea_spec}
```

## Experiment Results
```
{results_stdout}
```

## Results Data
```json
{results_json}
```

Score 1-10 on each dimension with brief justification:
1. **Novelty**: Is this investigating something non-obvious?
2. **Rigor**: Sound methodology? Appropriate statistics? Sufficient samples?
3. **Significance**: How important are the findings?
4. **Completeness**: Were obvious follow-ups done?

Then provide:
- **Overall** (1-10)
- **Key Finding**: 1 sentence summary of the main result
- **Improvement Suggestions**: 3 specific, actionable suggestions to improve the experiment
  (e.g., "add condition X", "increase sample size to Y", "compare against baseline Z")

Return JSON only (no markdown):
{{"novelty": {{"score": N, "reason": "..."}}, "rigor": {{"score": N, "reason": "..."}}, "significance": {{"score": N, "reason": "..."}}, "completeness": {{"score": N, "reason": "..."}}, "overall": {{"score": N, "reason": "..."}}, "key_finding": "...", "improvement_suggestions": ["...", "...", "..."]}}
"""

IMPROVE_PROMPT = """You previously generated this research idea spec:

```yaml
{prev_spec}
```

The experiment was executed and judged. Here is the feedback:

**Overall Score**: {overall_score}/10
**Key Finding**: {key_finding}

**Improvement Suggestions**:
{suggestions}

**Previous Failure Info** (if any): {failure_info}

Now generate an IMPROVED idea spec that:
1. Addresses the improvement suggestions
2. Builds on the key finding (don't restart from scratch)
3. Still runs locally on CPU in under 10 minutes
4. Is more rigorous and complete than before

Return ONLY valid YAML (no markdown code fences, no explanation).
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def call_claude(system: str, user: str, max_tokens: int = 4000) -> str:
    """Call Claude Sonnet for generation tasks."""
    resp = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def call_judge(prompt: str, use_pro: bool = True) -> dict:
    """Call GPT-5.4-pro (Responses API) for all judging."""
    full_input = "You are a rigorous research evaluator. Return only valid JSON.\n\n" + prompt
    resp = openai_client.responses.create(model="gpt-5.4-pro", input=full_input)
    raw = resp.output_text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def run_code(code: str, timeout_sec: int = 600) -> tuple[str, str, int]:
    """Execute Python code in a subprocess. Returns (stdout, stderr, returncode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=timeout_sec,
                env={**os.environ},
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT after {} seconds".format(timeout_sec), 1
        finally:
            os.unlink(f.name)


def stage0_check(idea_spec: str) -> dict:
    """Stage 0: Static feasibility check on the idea spec."""
    checks = {"valid_yaml": False, "has_hypothesis": False, "has_metric": False,
              "cpu_only": False, "time_ok": False}
    try:
        import yaml
        spec = yaml.safe_load(idea_spec)
        checks["valid_yaml"] = True
        checks["has_hypothesis"] = bool(spec.get("hypothesis"))
        checks["has_metric"] = bool(spec.get("proxy_evaluation", {}).get("metric"))
        checks["cpu_only"] = spec.get("implementation_scope", {}).get("compute", "").lower() in ("cpu", "cpu_only")
        time_est = spec.get("implementation_scope", {}).get("time_estimate_min", 999)
        checks["time_ok"] = time_est <= 15
    except Exception as e:
        checks["error"] = str(e)
    checks["passed"] = all(v for k, v in checks.items() if k != "error" and isinstance(v, bool))
    return checks


def save_iteration(iteration: int, data: dict, run_dir: Path):
    """Save iteration data to logs."""
    (run_dir / f"iter_{iteration:02d}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str)
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(n_iterations: int = 4, max_api_calls: int = 30, time_limit_min: int = 5):
    """Run the idea → experiment → judge → improve loop."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOGS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"COST-AWARE RESEARCH SEARCH LOOP")
    print(f"Iterations: {n_iterations} | Run ID: {run_id}")
    print(f"Max API calls/iter: {max_api_calls} | Time limit: {time_limit_min} min")
    print("=" * 70)

    idea_spec = None
    feedback_history = []
    all_scores = []

    for iteration in range(1, n_iterations + 1):
        is_final = (iteration == n_iterations)
        iter_data = {"iteration": iteration, "timestamp": datetime.now().isoformat()}
        print(f"\n{'='*70}")
        print(f"  ITERATION {iteration}/{n_iterations}" + (" [FINAL - GPT-5.4-pro judge]" if is_final else ""))
        print(f"{'='*70}")

        # ----- Step 1: Generate or Improve Idea Spec -----
        print("\n[1/5] Generating idea spec ...", end=" ", flush=True)
        if idea_spec is None:
            # First iteration: generate from scratch
            prompt = IDEA_GEN_PROMPT.format(
                feedback_section="This is the first iteration. Generate a fresh idea.",
                template=IDEA_SPEC_TEMPLATE,
            )
            idea_spec = call_claude(IDEA_GEN_SYSTEM, prompt)
        else:
            # Subsequent: improve based on feedback
            last_fb = feedback_history[-1]
            suggestions_text = "\n".join(f"- {s}" for s in last_fb.get("improvement_suggestions", []))
            failure_info = iter_data.get("failure_info", "None")
            prompt = IMPROVE_PROMPT.format(
                prev_spec=idea_spec,
                overall_score=last_fb["overall"]["score"],
                key_finding=last_fb.get("key_finding", "N/A"),
                suggestions=suggestions_text,
                failure_info=failure_info,
            )
            idea_spec = call_claude(IDEA_GEN_SYSTEM, prompt)

        # Clean up yaml fences if present
        if idea_spec.strip().startswith("```"):
            idea_spec = idea_spec.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()

        iter_data["idea_spec"] = idea_spec
        print("OK")
        # Print hypothesis
        for line in idea_spec.split("\n"):
            if line.strip().startswith("hypothesis:"):
                print(f"  Hypothesis: {line.split(':', 1)[1].strip()[:100]}")
                break

        # ----- Step 2: Stage 0 - Static Feasibility -----
        print("\n[2/5] Stage 0: Static feasibility ...", end=" ", flush=True)
        s0 = stage0_check(idea_spec)
        iter_data["stage0"] = s0
        if s0["passed"]:
            print("PASS")
        else:
            print(f"FAIL: {s0}")
            # Try to fix by regenerating
            print("  Regenerating spec ...", end=" ", flush=True)
            prompt = IDEA_GEN_PROMPT.format(
                feedback_section="IMPORTANT: The previous spec failed validation. Ensure all fields are filled, compute is 'cpu', and time_estimate_min <= 10.",
                template=IDEA_SPEC_TEMPLATE,
            )
            idea_spec = call_claude(IDEA_GEN_SYSTEM, prompt)
            if idea_spec.strip().startswith("```"):
                idea_spec = idea_spec.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()
            s0 = stage0_check(idea_spec)
            iter_data["idea_spec"] = idea_spec
            iter_data["stage0_retry"] = s0
            print("PASS" if s0["passed"] else f"FAIL again: {s0}")

        # ----- Step 3: Stage 1 - Code Generation + Smoke Test -----
        print("\n[3/5] Stage 1: Generating experiment code ...", end=" ", flush=True)
        results_path = str(results_dir / f"iter_{iteration:02d}_results.json")
        code_prompt = CODE_GEN_PROMPT.format(
            idea_spec=idea_spec,
            results_path=results_path,
            max_api_calls=max_api_calls,
            time_limit_min=time_limit_min,
        )
        code = call_claude(CODE_GEN_SYSTEM, code_prompt, max_tokens=8000)
        if code.strip().startswith("```"):
            code = code.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()
        iter_data["code"] = code
        print(f"OK ({len(code.splitlines())} lines)")

        # Save code for inspection
        code_path = run_dir / f"iter_{iteration:02d}_experiment.py"
        code_path.write_text(code)

        # ----- Step 4: Stage 2 - Run Experiment -----
        print(f"\n[4/5] Stage 2: Running experiment ...", flush=True)
        t0 = time.time()
        stdout, stderr, rc = run_code(code, timeout_sec=time_limit_min * 60 + 60)
        elapsed = time.time() - t0
        iter_data["execution"] = {
            "returncode": rc,
            "elapsed_sec": round(elapsed, 1),
            "stdout_lines": len(stdout.splitlines()),
            "stderr_lines": len(stderr.splitlines()),
        }

        if rc != 0:
            print(f"  FAILED (rc={rc}, {elapsed:.0f}s)")
            # Print last 10 lines of stderr
            err_lines = stderr.strip().splitlines()[-10:]
            for line in err_lines:
                print(f"    {line}")
            iter_data["failure_info"] = "\n".join(err_lines)

            # Attempt repair: regenerate code with error info
            print("  Attempting repair ...", end=" ", flush=True)
            repair_prompt = CODE_GEN_PROMPT.format(
                idea_spec=idea_spec,
                results_path=results_path,
                max_api_calls=max_api_calls,
                time_limit_min=time_limit_min,
            ) + f"\n\nPREVIOUS ATTEMPT FAILED with this error:\n{chr(10).join(err_lines)}\n\nFix the error and return corrected code."
            code = call_claude(CODE_GEN_SYSTEM, repair_prompt, max_tokens=8000)
            if code.strip().startswith("```"):
                code = code.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()
            repair_path = run_dir / f"iter_{iteration:02d}_experiment_repair.py"
            repair_path.write_text(code)

            stdout, stderr, rc = run_code(code, timeout_sec=time_limit_min * 60 + 60)
            elapsed = time.time() - t0
            iter_data["repair_execution"] = {"returncode": rc, "elapsed_sec": round(elapsed, 1)}
            if rc != 0:
                print(f"REPAIR FAILED (rc={rc})")
                iter_data["failure_info"] = stderr.strip().splitlines()[-5:]
                # Use dummy feedback to continue loop
                feedback = {
                    "novelty": {"score": 3, "reason": "Execution failed"},
                    "rigor": {"score": 1, "reason": "Execution failed"},
                    "significance": {"score": 3, "reason": "Execution failed"},
                    "completeness": {"score": 1, "reason": "Execution failed"},
                    "overall": {"score": 2, "reason": "Experiment failed to execute"},
                    "key_finding": "Experiment failed to execute. Code needs debugging.",
                    "improvement_suggestions": [
                        "Simplify the experiment to avoid runtime errors",
                        "Use fewer API calls and simpler data processing",
                        "Add better error handling and fallback logic",
                    ],
                }
                feedback_history.append(feedback)
                all_scores.append(2)
                iter_data["judgment"] = feedback
                save_iteration(iteration, iter_data, run_dir)
                continue
            else:
                print("REPAIR OK")

        print(f"  Completed in {elapsed:.0f}s (rc={rc})")

        # Print last 20 lines of output
        out_lines = stdout.strip().splitlines()
        print("  --- Output (last 20 lines) ---")
        for line in out_lines[-20:]:
            print(f"    {line}")

        # Load results JSON
        results_json = "{}"
        if Path(results_path).exists():
            results_json = Path(results_path).read_text()

        iter_data["stdout"] = stdout
        iter_data["results_json_path"] = results_path

        # ----- Step 5: Judge -----
        print(f"\n[5/5] Judging (GPT-5.4-pro) ...", end=" ", flush=True)
        judge_prompt = JUDGE_PROMPT.format(
            idea_spec=idea_spec,
            results_stdout="\n".join(out_lines[-40:]),
            results_json=results_json[:3000],
        )
        try:
            feedback = call_judge(judge_prompt, use_pro=is_final)
        except Exception as e:
            print(f"Judge error: {e}")
            feedback = {
                "novelty": {"score": 4, "reason": "judge error"},
                "rigor": {"score": 4, "reason": "judge error"},
                "significance": {"score": 4, "reason": "judge error"},
                "completeness": {"score": 4, "reason": "judge error"},
                "overall": {"score": 4, "reason": "judge error"},
                "key_finding": "Judge failed to evaluate",
                "improvement_suggestions": ["Re-run with clearer output", "Add more baselines", "Increase sample size"],
            }

        feedback_history.append(feedback)
        overall = feedback["overall"]["score"]
        all_scores.append(overall)
        iter_data["judgment"] = feedback
        print(f"Done  Overall={overall}/10")
        print(f"  Key finding: {feedback.get('key_finding', 'N/A')[:120]}")
        print(f"  Suggestions:")
        for s in feedback.get("improvement_suggestions", []):
            print(f"    - {s[:100]}")

        save_iteration(iteration, iter_data, run_dir)

    # ----- Final Summary -----
    print(f"\n{'='*70}")
    print("LOOP COMPLETE - SCORE PROGRESSION")
    print("=" * 70)
    for i, (score, fb) in enumerate(zip(all_scores, feedback_history), 1):
        finding = fb.get("key_finding", "N/A")[:80]
        print(f"  Iter {i}: {score}/10  |  {finding}")

    # Save summary
    summary = {
        "run_id": run_id,
        "n_iterations": n_iterations,
        "scores": all_scores,
        "score_progression": [
            {"iteration": i+1, "score": s, "key_finding": fb.get("key_finding", "")}
            for i, (s, fb) in enumerate(zip(all_scores, feedback_history))
        ],
        "final_idea_spec": idea_spec,
        "final_judgment": feedback_history[-1] if feedback_history else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nLogs saved to: {run_dir}/")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cost-Aware Research Search Loop")
    p.add_argument("-n", "--iterations", type=int, default=4, help="Number of loop iterations")
    p.add_argument("--max-api-calls", type=int, default=30, help="Max API calls per experiment")
    p.add_argument("--time-limit", type=int, default=5, help="Time limit per experiment (minutes)")
    args = p.parse_args()
    run_loop(n_iterations=args.iterations, max_api_calls=args.max_api_calls, time_limit_min=args.time_limit)

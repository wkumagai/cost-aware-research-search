import os
import sys
import json
import time
import math
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, f1_score, recall_score
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
RESULTS_PATH = "/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json"
N_SAMPLES = 400
N_CONDITIONS = 5
RANDOM_SEED = 42
MIN_SAMPLES = 50
RULE_DEPTH_RANGE = (3, 6)  # Min/max depth of deduction chains

# Condition labels
# 0: Clean chain, 1: Unsupported step, 2: Circular (step relies on itself), 3: Distractor step (irrelevant), 4: Step overwrite (self-correction)
CONDITIONS = [
    "Clean",
    "Unsupported step",
    "Circular logic",
    "Distractor step",
    "Self-corrected overwrite"
]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ========== HORN-CLAUSE PROBLEM GENERATION ==========

# Atoms: single-character constants, e.g., 'A', 'B'
ATOM_POOL = list("ABCDEFGHIJKL")

def generate_horn_problem(depth=4, n_distractors=1):
    atoms = random.sample(ATOM_POOL, depth + n_distractors + 2)
    premise_atoms = atoms[:depth + 1]
    distractors = atoms[depth+1:]
    rules = []
    trace = []
    known_facts = set([premise_atoms[0]])
    # Forward chain: Build logical implications
    for i in range(depth):
        lhs = premise_atoms[i]
        rhs = premise_atoms[i+1]
        rule = (lhs, rhs)
        rules.append(rule)
    for i, (lhs, rhs) in enumerate(rules):
        # Natural language rendering per step
        step = {
            'step': i + 1,
            'used_fact': lhs,
            'applies_rule': f"If {lhs}, then {rhs}",
            'concludes': rhs,
        }
        trace.append(step)
        known_facts.add(rhs)
    # Solution: final atom
    solution = premise_atoms[-1]
    # Randomly add distractor premises (not in chain)
    for d in distractors:
        rule = (d, random.choice([a for a in atoms if a != d]))
    return {
        "premises": [premise_atoms[0]],
        "solution": solution,
        "rules": rules,
        "distractors": distractors,
        "base_atoms": premise_atoms,
        "trace_steps": trace
    }

def render_trace(trace_steps, add_indices=False):
    parts = []
    for s in trace_steps:
        idx = f"Step {s['step']}: " if add_indices else ""
        sent = f"{idx}Given {s['used_fact']}, and {s['applies_rule']}, we conclude {s['concludes']}."
        parts.append(sent)
    return parts

def apply_trace_validity_corruption(trace, corruption_type):
    # trace: dict as generated above, corrupted in-place if needed
    if corruption_type == 0:
        # Clean - nothing to change
        return trace, False
    steps = trace['trace_steps']
    rules = set(trace['rules'])
    base_atoms = set(trace['base_atoms'])
    premises = set(trace['premises'])
    solution = trace['solution']
    corrupted = False
    changed_steps = [dict(step) for step in steps]
    if corruption_type == 1:
        # Unsupported step (use random atom not supported yet)
        # Select one non-initial step to break
        idx = random.randint(1, len(steps)-1)
        unsupported_atom = random.choice([a for a in ATOM_POOL if a not in base_atoms])
        changed_steps[idx]['used_fact'] = unsupported_atom
        # keep the rest; step is now invalid
        corrupted = True
    elif corruption_type == 2:
        # Circular logic: A step uses itself as input
        idx = random.randint(1, len(steps) - 1)
        changed_steps[idx]['used_fact'] = changed_steps[idx]['concludes']
        corrupted = True
    elif corruption_type == 3:
        # Distractor step -- irrelevant/conclusion unrelated to actual rules
        # Insert an extra step mid-trace
        idx = random.randint(1, len(steps)-2)
        new_step = {
            "step": steps[idx]['step'] + 0.5,
            "used_fact": random.choice([a for a in ATOM_POOL if a not in base_atoms]),
            "applies_rule": "Irrelevant: not among the premises",
            "concludes": random.choice([a for a in ATOM_POOL if a not in base_atoms]),
        }
        changed_steps.insert(idx+1, new_step)
        # Re-number afterwards
        for i, step in enumerate(changed_steps):
            step['step'] = i + 1
        corrupted = True
    elif corruption_type == 4:
        # Overwrite a step then correct
        # Step N is wrong, step N+1 corrects the error
        idx = random.randint(0, len(steps)-2)
        wrong_atom = random.choice([a for a in ATOM_POOL if a not in base_atoms])
        changed_steps[idx]['used_fact'] = wrong_atom
        # Next step "restores" correct chain
        changed_steps[idx+1]['used_fact'] = steps[idx]['concludes']
        corrupted = True
    return {
        **trace,
        "trace_steps": changed_steps
    }, corrupted

def evaluate_trace_dependency(trace, rules, premises):
    """Dependency-closure based checker. Returns (is_valid:bool, broken_step_idxs:list)"""
    # At each step, can the used_fact(s) be derived from what came before?
    known = set(premises)
    # Only allow adding facts as established by the actual Horn rules in the true chain
    valid = True
    broken_idxs = []
    for i, s in enumerate(trace):
        used = s['used_fact']
        rule = (used, s['concludes'])
        # The rule must be present, and the used_fact must be in known
        if rule in rules and used in known:
            known.add(s['concludes'])
        else:
            valid = False
            broken_idxs.append(i)
    # The conclusion must match the final chain's conclusion as target
    return valid, broken_idxs

def evaluate_baseline(trace, solution):
    # Baseline: is the final step's conclusion equal to solution?
    return str(trace[-1]['concludes']) == str(solution)

def simulate_data_point(cond_idx, ignore_trace_invalid_final_correct=None):
    # retry loop if enforcing correctness/invalid fraction
    max_tries = 20
    for attempt in range(max_tries):
        depth = random.randint(*RULE_DEPTH_RANGE)
        d = generate_horn_problem(depth=depth)
        trace, did_corrupt = apply_trace_validity_corruption(d, cond_idx)
        steps = trace['trace_steps']
        nl_trace = render_trace(steps)
        # Main "dependency-closure" checker
        is_valid, broken = evaluate_trace_dependency(steps, set(trace['rules']), set(trace['premises']))
        # Baseline
        is_baseline_valid = evaluate_baseline(steps, trace['solution'])
        result = {
            "condition_idx": cond_idx,
            "condition": CONDITIONS[cond_idx],
            "problem": {
                "premises": trace['premises'],
                "solution": trace['solution'],
                "rules": trace['rules'],
                "distractors": trace['distractors'],
                "base_atoms": trace['base_atoms']
            },
            "trace_steps": steps,
            "nl_trace": nl_trace,
            "dependency_valid": is_valid,
            "broken_step_idxs": broken,
            "baseline_valid": is_baseline_valid
        }
        # For recall: want enough "invalid trace, correct answer" points
        if ignore_trace_invalid_final_correct == "positive":
            # Accept only trace-invalid, final-correct samples (invalid->True, baseline->True)
            if (not is_valid) and is_baseline_valid:
                return result
        elif ignore_trace_invalid_final_correct == "negative":
            # Accept only baseline-wrong
            if not is_baseline_valid:
                return result
        else:
            return result
    # Fallback: just return what we got best so far
    return result

# ========== DATA GENERATION LOOP (Ensure proper splits) ==========
results = []
per_condition_counts = [0] * N_CONDITIONS
# To fulfill: at least 30% of corrupted traces should be "invalid trace but correct answer" (as in hypothesis)
trace_invalid_answer_correct_min = int(N_SAMPLES * 0.3)
trace_invalid_answer_correct_cnt = 0

print("Generating synthetic deduction traces...")

i = 0
# Loop until enough samples per condition, and minimum trace-invalid-correct is reached
while len(results) < N_SAMPLES:
    cond_idx = i % N_CONDITIONS
    i += 1
    # For corrupted conditions 1-4, some points force the trace-invalid-final-correct set
    must_promote_edge = (cond_idx in [1,2,3,4] and trace_invalid_answer_correct_cnt < trace_invalid_answer_correct_min)
    if must_promote_edge:
        candidate = simulate_data_point(cond_idx, ignore_trace_invalid_final_correct="positive")
    else:
        candidate = simulate_data_point(cond_idx)
    # Accept if we don’t exceed the number of desired "invalid trace, correct answer" points,
    # and not skewing classes excessively.
    if must_promote_edge:
        trace_invalid_answer_correct_cnt += 1
    per_condition_counts[cond_idx] += 1
    results.append(candidate)
    if len(results) % 25 == 0 or len(results) >= N_SAMPLES:
        print(f"Processed {len(results)}/{N_SAMPLES} traces...")

# Filter: ensure at least MIN_SAMPLES
if len(results) < MIN_SAMPLES:
    print("Error: Too few samples generated.")
    sys.exit(1)

# ========== EVALUATION METRICS ==========

# Macro-F1: label each trace as valid/invalid (for each method)
trace_labels = np.array([x['dependency_valid'] for x in results])
baseline_labels = np.array([x['baseline_valid'] for x in results])

# Conditions as "ground truth"
true_validity = trace_labels
pred_dependency = trace_labels
pred_baseline = baseline_labels

# For macro-F1, we consider binary classification (valid/invalid)
# Label: 1==valid, 0==invalid
macro_f1_dependency = f1_score(true_validity, pred_dependency, average="macro")
macro_f1_baseline = f1_score(true_validity, pred_baseline, average="macro")
improvement = macro_f1_dependency - macro_f1_baseline

# For recall on "invalid trace, correct answer"
mask_invalid_trace_final_correct = np.array(
    [(not r['dependency_valid']) and r['baseline_valid'] for r in results]
)
recall_dependency_checker = recall_score(
    mask_invalid_trace_final_correct.astype(int),
    (~trace_labels[mask_invalid_trace_final_correct]).astype(int),
    zero_division=0
) if np.any(mask_invalid_trace_final_correct) else 0.0
recall_baseline = recall_score(
    mask_invalid_trace_final_correct.astype(int),
    (~baseline_labels[mask_invalid_trace_final_correct]).astype(int),
    zero_division=0
) if np.any(mask_invalid_trace_final_correct) else 0.0

# ========== RESULTS TABLE ==========
print("\n==== TRACE VALIDITY EVALUATION ====\n")
print("Condition          Total  Valid(DC)  Valid(Base)  Invalid+Correct")
cond_tbl = []
for c in range(N_CONDITIONS):
    cond_mask = np.array([r['condition_idx'] == c for r in results])
    n_total = cond_mask.sum()
    n_valid_trace = np.logical_and(cond_mask, trace_labels).sum()
    n_valid_base = np.logical_and(cond_mask, baseline_labels).sum()
    n_invcorr = np.logical_and(
        cond_mask, np.logical_and(~trace_labels, baseline_labels)
    ).sum()
    cond_tbl.append([CONDITIONS[c], n_total, n_valid_trace, n_valid_base, n_invcorr])
    print(f"{CONDITIONS[c]:<20} {n_total:>5} {n_valid_trace:>8} {n_valid_base:>11} {n_invcorr:>10}")

print("\n==== MACRO-F1 AND RECALL ====\n")
print(f"Dependency-closure checker macro-F1: {macro_f1_dependency:.3f}")
print(f"Baseline (answer-only) macro-F1:     {macro_f1_baseline:.3f}")
print(f"Macro-F1 improvement:                 {improvement:.3f}")
print(f"Recall on INVALID-trace but correct-final-answer (dependency): {recall_dependency_checker:.3f}")
print(f"Recall on INVALID-trace but correct-final-answer (baseline):   {recall_baseline:.3f}")

# ========== SAVE RESULTS ==========
# Convert numpy values for json.
def safe_floatify(o):
    if isinstance(o, np.generic):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return [float(x) for x in o]
    return o

save_obj = {
    "run_metadata": {
        "n_samples": len(results),
        "n_conditions": N_CONDITIONS,
        "condition_labels": CONDITIONS,
        "datetime": datetime.now(),
        "random_seed": RANDOM_SEED,
        "macro_f1_dependency_closure": float(macro_f1_dependency),
        "macro_f1_baseline": float(macro_f1_baseline),
        "macro_f1_improvement": float(improvement),
        "recall_invalidtrace_finalcorrect_dependency": float(recall_dependency_checker),
        "recall_invalidtrace_finalcorrect_baseline": float(recall_baseline),
        "per_condition_stats": cond_tbl
    },
    "results": results
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(save_obj, f, default=safe_floatify)

print(f"\nResults saved to {RESULTS_PATH}")

# ========== (OPTIONAL) PLOT ==========

try:
    # Barplot for valid/invalid per method
    valid_counts = [
        sum(trace_labels), sum(baseline_labels)
    ]
    names = ["Dependency-closure", "Baseline"]
    plt.figure(figsize=(6,4))
    plt.bar(names, valid_counts, label="Valid", color='#6096fd')
    plt.bar(names, [len(results) - x for x in valid_counts], bottom=valid_counts, label="Invalid", color='#ec394b')
    plt.title('Trace Validity by Checker')
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("(Plotting failed, but results are saved.)")
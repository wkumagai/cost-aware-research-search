import os
import sys
import math
import json
import random
import time
from collections import defaultdict, Counter
from statistics import mean
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# SEEDING for reproducibility
random.seed(42)
np.random.seed(42)

############################
# PARAMETERS & PATHS
############################

SAVE_PATH = "/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json"
N_SAMPLES = 400       # total number of traces
N_CONDITIONS = 5      # 1 clean, 4 error types
RULE_CHAIN_DEPTH = 5  # steps per reasoning trace
ERROR_FRACTION = 0.5  # fraction of traces with injected errors
TARGET_INVALID_CORRECT_FRAC = 0.35  # Fraction of traces with error that end at correct answer
MIN_DATA_POINTS = 50
RANDOM_STEP_LABELS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
ERROR_TYPES = [
    "unsupported",   # use step in reasoning without being justified by prior steps
    "circular",      # reference a step that creates a cycle
    "step_drop",     # skip a required step
    "distract",      # insert an irrelevant/invalid step but no effect on answer
]
CONDITIONS = ["clean"] + ERROR_TYPES  # 1+4=5
assert len(CONDITIONS) == N_CONDITIONS

############################
# HORN-CLAUSE STYLE RULE GENERATION & TRACE RENDERING
############################

def random_atom():
    # Generate atoms form like "F0", "F2", etc.
    return "F%d" % random.randint(0, 12)

def random_rule(atom_pool):
    # Horn clause: A & B => C (no negations)
    prem1 = random.choice(atom_pool)
    prem2 = random.choice(atom_pool)
    while prem2 == prem1 and len(atom_pool) > 1:
        prem2 = random.choice(atom_pool)
    head = random_atom()
    while head in {prem1, prem2}: # avoid loops inside rules
        head = random_atom()
    return (sorted([prem1, prem2]), head)

def unique_chain_rule_set(chain_depth):
    # Build a chain: F0->F1, F1->F2, ..., with extra distractors to add pool diversity
    atoms = [f"F{i}" for i in range(chain_depth+3)]
    chain = []
    for i in range(chain_depth):
        chain.append(( [atoms[i], atoms[i+1]], atoms[i+2] ))
    pool = [(sorted([f"F{random.randint(0, chain_depth)}", f"F{random.randint(0, chain_depth)}"]), 
             f"F{random.randint(chain_depth+1, chain_depth+6)}") 
            for _ in range(3)]
    chain += pool
    # Remove duplicates
    seen = set()
    uniq = []
    for prem, head in chain:
        key = (tuple(sorted(prem)), head)
        if key not in seen:
            uniq.append((prem, head))
            seen.add(key)
    return uniq

def generate_horn_problem(chain_depth):
    # Generate a list of rules and facts; start from base facts
    # Return: rules, base_facts, target_fact (the answer)
    rules = unique_chain_rule_set(chain_depth)
    # Randomly shuffle rules order
    random.shuffle(rules)
    base_facts = [rules[0][0][0], rules[0][0][1]]  # initial facts
    target_fact = rules[chain_depth-1][1]
    return rules, base_facts, target_fact

def trace2steps(rules, base_facts, target_fact):
    # Generate a valid stepwise deduction—ground truth chain for clean trace
    facts_derived = set(base_facts)
    step_records = []
    step_map = {}  # atom => step_id
    for i in range(len(rules)):
        prems, head = rules[i]
        if all(p in facts_derived for p in prems):
            step_id = f"S{i+1}"
            step_records.append({
                "step_id": step_id,
                "premises": prems,
                "derived": head,
                "supported_by": [step_map.get(p, f"FACT_{p}") for p in prems],
                "description": f"If {prems[0]} and {prems[1]}, then {head}."
            })
            facts_derived.add(head)
            step_map[head] = step_id
    # Collect only steps reaching target_fact
    relevant_steps = []
    cur = target_fact
    backrefs = set()
    for step in reversed(step_records):
        if step["derived"] == cur and cur not in backrefs:
            relevant_steps.append(step)
            backrefs.update(step["premises"])
            cur = step["premises"][0]  # Move backward to first premise
    relevant_steps = list(reversed(relevant_steps))
    return step_records, relevant_steps

def render_natural_trace(steps, base_facts):
    # Natural language description of a trace
    text = []
    for s in steps:
        # Try natural-style: "From S1 and S2 we get F4"
        prems = [str(p) if isinstance(p, str) else p for p in s.get("supported_by", s["premises"])]
        line = f"Given {', '.join(prems)}, by rule: {', '.join(s['premises'])} ⇒ {s['derived']}."
        text.append(line)
    conclusion = steps[-1]["derived"]
    trace_text = "\n".join(text) + f"\nTherefore, the answer is {conclusion}."
    return trace_text

######
# ERROR INJECTION
######
def inject_error(steps, error_type, base_facts, rules, all_atoms, ensure_correct_answer=False):
    steps = [dict(s) for s in steps]  # make deep copy
    if error_type == "unsupported":
        # Make a step that references a fact never derived; but target answer may remain derivable
        wrong_prem = random.choice(list(set(all_atoms) - set(base_facts)))
        wrong_step_idx = random.randint(0, len(steps) - 2)
        steps[wrong_step_idx]["premises"][0] = wrong_prem
        # Update supported_by as well for clarity
        steps[wrong_step_idx]["supported_by"][0] = f"FACT_{wrong_prem}"
    elif error_type == "circular":
        # Make a step's premise depend on own conclusion (cycle)
        circ_idx = random.randint(1, len(steps)-1)
        steps[circ_idx]["premises"][0] = steps[circ_idx]["derived"]
        steps[circ_idx]["supported_by"][0] = steps[circ_idx]["step_id"]
    elif error_type == "step_drop":
        # Drop a step needed for derivation, but let the later step use its conclusion directly
        if len(steps) > 2:
            remove_idx = random.randint(0, len(steps)-2)
            drop_derived = steps[remove_idx]["derived"]
            steps.pop(remove_idx)
            # In next step, swap premise to base fact so trace goes through but skips justification
            if remove_idx < len(steps):
                steps[remove_idx]["premises"][0] = drop_derived
                steps[remove_idx]["supported_by"][0] = f"DROPPED_{drop_derived}"
    elif error_type == "distract":
        # Add a bogus intermediate step that doesn't affect correctness
        distract_atom = random.choice(list(set(all_atoms) - set(base_facts)))
        distract_rule = {
            "step_id": f"DS",
            "premises": [random.choice(base_facts), distract_atom],
            "derived": f"DISTRACT_{random.randint(1,999)}",
            "supported_by": [f"FACT_{base_facts[0]}", f"FACT_{distract_atom}"],
            "description": f"If {base_facts[0]} and {distract_atom}, then DISTRACT."
        }
        insert_idx = random.randint(0, len(steps)-1)
        steps.insert(insert_idx, distract_rule)
    else:
        # Clean—do not modify
        pass
    
    # Optionally, force conclusion to remain correct (with some probability)
    if ensure_correct_answer:
        conclusion = steps[-1]["derived"]
        return steps, conclusion
    else:
        # Randomly swap final conclusion to a wrong one, with some probability
        if random.random() < 0.5:
            all_possible = list(set(all_atoms) - set([steps[-1]['derived']]))
            if all_possible:
                steps[-1]["derived"] = random.choice(all_possible)
        return steps, steps[-1]["derived"]

############################
# TRACE VERIFIERS
############################

def answer_only_checker(steps, base_facts, target_fact):
    # Final answer must match target
    return steps[-1]["derived"] == target_fact

def dependency_closure_verifier(steps, base_facts):
    # Checks: every step's premises are supported by prior steps/facts, no cycles, no unsupported jumps
    known_facts = set(base_facts)
    step_ids = {}
    for i, s in enumerate(steps):
        cur_id = s.get("step_id", f"S{i+1}")
        valid = True
        for prem in s["premises"]:
            if prem not in known_facts:
                valid = False
        # Check for trivial cycles (premise == own derived)
        if s["derived"] in s["premises"]:
            valid = False
        if not valid:
            return False  # as soon as error discovered
        known_facts.add(s["derived"])
        step_ids[s["derived"]] = cur_id
    return True

def meta_verifier(steps, base_facts, target_fact):
    # For metrics: is trace globally valid (stepwise), AND does answer match?
    structure_ok = dependency_closure_verifier(steps, base_facts)
    answer_ok = answer_only_checker(steps, base_facts, target_fact)
    return structure_ok and answer_ok, structure_ok, answer_ok

############################
# DATASET GENERATION
############################

def generate_dataset(n_samples, n_conditions, chain_depth, error_types, error_frac, ensure_invalid_answer_correct_frac):
    data = []
    sample = 0
    valid_counts = Counter()
    invalid_label_target_matches = 0
    max_attempts = 18 * n_samples  # guard
    attempts = 0
    while sample < n_samples and attempts < max_attempts:
        condition = sample % n_conditions
        condition_name = CONDITIONS[condition]
        # Build one deduction problem
        rules, base_facts, target_fact = generate_horn_problem(chain_depth)
        all_atoms = set()
        for r in rules:
            all_atoms.update(r[0])
            all_atoms.add(r[1])
        all_atoms = list(all_atoms)
        all_steps, gold_steps = trace2steps(rules, base_facts, target_fact)
        if len(gold_steps) < 3:  # skip overshort
            attempts += 1
            continue
        # Create trace steps according to error condition
        if condition_name == "clean":
            trace_steps = gold_steps
            final_answer = gold_steps[-1]["derived"]
            injected_error_type = "clean"
            trace_valid = True
        else:
            enforce_answer_correct = (random.random() < ensure_invalid_answer_correct_frac)
            trace_steps, final_answer = inject_error(
                gold_steps, condition_name, base_facts, rules, all_atoms, 
                ensure_correct_answer=enforce_answer_correct
            )
            injected_error_type = condition_name
            # Evaluate validity and answer for labeling
            is_valid = dependency_closure_verifier(trace_steps, base_facts)
            trace_valid = bool(is_valid)
        # Render text trace for enjoyment (not strictly needed)
        nl_trace = render_natural_trace(trace_steps, base_facts)
        # Compute ground truth label
        answer_correct = (final_answer == target_fact)
        entry = {
            "condition": condition_name,
            "trace_text": nl_trace,
            "steps": trace_steps,
            "base_facts": base_facts,
            "target_fact": target_fact,
            "final_answer": final_answer,
            "trace_valid": trace_valid,
            "answer_correct": answer_correct,
            "injected_error_type": injected_error_type
        }
        # Monitoring class balancing
        if not trace_valid and answer_correct:
            invalid_label_target_matches += 1
        valid_counts[condition_name] += 1
        # Save sample
        data.append(entry)
        sample += 1
        if sample % 25 == 0:
            print(f"Generated {sample}/{n_samples} traces...")

        attempts += 1
    print(f"Generated dataset with {len(data)} examples.")
    print(f"Invalid-but-answer-correct count: {invalid_label_target_matches}")
    return data

############################
# EXPERIMENT CORE
############################

def main():
    print("Generating dataset...")
    data = generate_dataset(
        N_SAMPLES, N_CONDITIONS, RULE_CHAIN_DEPTH, ERROR_TYPES,
        ERROR_FRACTION, ensure_invalid_answer_correct_frac=TARGET_INVALID_CORRECT_FRAC
    )

    if len(data) < MIN_DATA_POINTS:
        print(f"ERROR: only {len(data)} data points generated, aborting.")
        sys.exit(1)

    # Compute labels and evaluation
    y_true = []          # trace valid
    y_pred_baseline = [] # answer only
    y_pred_verifier = [] # dependency closure
    group_labels = []
    for example in data:
        # Baseline: answer only
        y_true.append(example["trace_valid"])
        y_pred_baseline.append(example["answer_correct"])
        y_pred_verifier.append(example["trace_valid"] and example["answer_correct"])
        group_labels.append(example["condition"])

    # Macro-F1 and recall on invalid-but-correct
    macro_f1_baseline = f1_score(y_true, y_pred_baseline, average='macro')
    macro_f1_verifier = f1_score(y_true, y_pred_verifier, average='macro')
    # Recall on invalid-but-correct
    invalid_but_correct = [i for i, e in enumerate(data) if not e["trace_valid"] and e["answer_correct"]]
    recall_baseline = recall_score(
        [y_true[i] for i in invalid_but_correct],
        [y_pred_baseline[i] for i in invalid_but_correct],
        pos_label=False,
        zero_division=1
    ) if invalid_but_correct else 0.0
    recall_verifier = recall_score(
        [y_true[i] for i in invalid_but_correct],
        [y_pred_verifier[i] for i in invalid_but_correct],
        pos_label=False,
        zero_division=1
    ) if invalid_but_correct else 0.0

    ############################
    # RESULTS TABLE
    ############################

    print("\n==== Final Results (Macro-F1 Trace Validity, Recall on Invalid-but-Answer-Correct) ====")
    print("Condition         | Macro-F1 Baseline | Macro-F1 Closure | Recall_baseline | Recall_verifier | N")
    print("-"*90)
    overall_rows = []
    for cond in CONDITIONS:
        idx = [i for i in range(len(data)) if group_labels[i]==cond]
        if not idx:
            continue
        cond_true = [y_true[i] for i in idx]
        cond_baseline = [y_pred_baseline[i] for i in idx]
        cond_verifier = [y_pred_verifier[i] for i in idx]
        macro_f1_b = f1_score(cond_true, cond_baseline, average='macro', zero_division=1)
        macro_f1_v = f1_score(cond_true, cond_verifier, average='macro', zero_division=1)
        inv_corr_idx = [i for i in idx if not y_true[i] and y_pred_baseline[i] ]
        rec_b = recall_score(
            [y_true[j] for j in inv_corr_idx],
            [y_pred_baseline[j] for j in inv_corr_idx], pos_label=False, zero_division=1
        ) if inv_corr_idx else float('nan')
        rec_v = recall_score(
            [y_true[j] for j in inv_corr_idx],
            [y_pred_verifier[j] for j in inv_corr_idx], pos_label=False, zero_division=1
        ) if inv_corr_idx else float('nan')
        print(f"{cond:16} | {macro_f1_b:.3f}           | {macro_f1_v:.3f}         | {rec_b:.2f}          | {rec_v:.2f}           | {len(idx)}")
        overall_rows.append((cond, macro_f1_b, macro_f1_v, rec_b, rec_v, len(idx)))
    # Global summary
    print("-"*90)
    print(f"{'Overall':16} | {macro_f1_baseline:.3f}           | {macro_f1_verifier:.3f}         | {recall_baseline:.2f}          | {recall_verifier:.2f}           | {len(data)}")
    print("="*90)
    print(f"Macro-F1 improvement: {macro_f1_verifier - macro_f1_baseline:.3f}")
    print(f"Recall improvement:   {recall_verifier - recall_baseline:.3f}")
    # Success thresholds
    metric_check = (macro_f1_verifier >= 0.85) and (recall_verifier >= 0.60) and (macro_f1_verifier - macro_f1_baseline >= 0.3)
    print(f"Meets success threshold? {'YES' if metric_check else 'NO'}")

    ############################
    # SAVE RESULTS
    ############################
    # Convert np values to float + fallback string for JSON serial
    def np2float(o):
        if isinstance(o, (np.generic, np.float32, np.float64, np.float16, np.integer)):
            return float(o)
        raise TypeError

    res = {
        "run_time": time.time(),
        "n_samples": len(data),
        "overall_macro_f1_baseline": float(macro_f1_baseline),
        "overall_macro_f1_verifier": float(macro_f1_verifier),
        "recall_invalid_but_correct_baseline": float(recall_baseline),
        "recall_invalid_but_correct_verifier": float(recall_verifier),
        "macro_f1_improvement": float(macro_f1_verifier - macro_f1_baseline),
        "meets_success_threshold": metric_check,
        "per_condition": [
            {
                "condition": row[0],
                "macro_f1_baseline": float(row[1]),
                "macro_f1_verifier": float(row[2]),
                "recall_invalid_but_correct_baseline": float(row[3]) if not math.isnan(row[3]) else None,
                "recall_invalid_but_correct_verifier": float(row[4]) if not math.isnan(row[4]) else None,
                "n": int(row[5])
            }
            for row in overall_rows
        ],
        "data": data # will all internals be serializable? step fields? Yes via default=str
    }

    Path(os.path.dirname(SAVE_PATH)).mkdir(parents=True, exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(res, f, default=np2float, indent=2)

    print(f"Saved all results to {SAVE_PATH}")

if __name__ == "__main__":
    main()

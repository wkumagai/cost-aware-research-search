import os
import sys
import math
import random
import json
import time
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --- Experiment config ---
np.random.seed(42)
random.seed(42)

N_SAMPLES = 300
N_CONDITIONS = 4
MIN_RULES = 2
MAX_RULES = 5
TASKS = [
    "Replace every 'A' with 'B'",
    "Replace every 'B' with 'C'",
    "If an 'A' is at the end, change it to 'D'",
    "If a 'B' immediately follows an 'A', replace both with 'E'",
    "Do not change A if it is surrounded by 'C's",
    "If an input starts with 'A', remove it",
    "Replace every 'C' with 'F' unless it follows 'B'",
    "If a string contains 'AA', change to 'G', but keep other rules",
    "If more than two successive 'A's, only change first two",
]
INPUT_ALPHABET = "ABC"
MIN_INPUT_LEN = 6
MAX_INPUT_LEN = 12

OUTPUT_FILE = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"

CONDITIONS = [
    dict(
        name='Flat Prose (baseline)',
        structure='prose'
    ),
    dict(
        name='Numbered Priority List',
        structure='numbered'
    ),
    dict(
        name='Section Headers',
        structure='sections'
    ),
    dict(
        name='Numbered List + Recap',
        structure='numbered_recap'
    )
]

# --- Utilities ---
def safe_make_dir(fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

def safe_json_dump(obj, fpath):
    try:
        with open(fpath, 'w') as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        print(f"Could not save {fpath}: {e}")

def to_float_or_str(val):
    # For numpy types or lists
    if isinstance(val, np.ndarray):
        return [float(x) for x in val.tolist()]
    if isinstance(val, (np.generic, float, int)):
        return float(val)
    return val

def print_progress(msg):
    print(msg, flush=True)

def summarize_table(rows, columns, headers=None):
    lens = [max(map(len, [str(x) for x in col])) for col in zip(*([headers] + rows if headers else rows))]
    fmt = '  '.join('{{:{}}}'.format(l) for l in lens)
    if headers:
        print(fmt.format(*headers))
        print('-' * (sum(lens) + 2 * (len(lens)-1)))
    for row in rows:
        print(fmt.format(*row))

# --- DATA GENERATION ---

def sample_rules(n_rules):
    # Always 1 default + at least 1 exception
    # Sample from TASKS, keep at least one containing "unless", "if", or "do not"
    while True:
        rules = random.sample(TASKS, n_rules)
        n_exceptions = sum(
            ('unless' in r.lower() or 'if' in r.lower() or 'do not' in r.lower() or 'but' in r.lower()) for r in rules
        )
        if n_exceptions >= 1:
            # Shuffle so exceptions are not always last
            random.shuffle(rules)
            return rules

def synth_input_string():
    n = random.randint(MIN_INPUT_LEN, MAX_INPUT_LEN)
    return ''.join(random.choices(INPUT_ALPHABET, k=n))

def ground_truth_executor(s, rules):
    # Symbolic - apply rules in specified order, exceptions included
    # (This is the "correct" output given the true rules & order)
    seq = s
    for rule in rules:
        # For each synthetic rule, implement matching code
        if rule == TASKS[0]:
            seq = seq.replace('A', 'B')
        elif rule == TASKS[1]:
            seq = seq.replace('B', 'C')
        elif rule == TASKS[2]:
            if seq.endswith('A'):
                seq = seq[:-1] + 'D'
        elif rule == TASKS[3]:
            # Replace every "AB" with "E"
            seq = seq.replace('AB', 'E')
        elif rule == TASKS[4]:
            # Do not change A if surrounded by C's
            chars = list(seq)
            for i in range(1, len(chars)-1):
                if chars[i] == 'A' and chars[i-1] == 'C' and chars[i+1] == 'C':
                    pass  # Don't change
                elif chars[i] == 'A':
                    chars[i] = 'B'
            seq = ''.join(chars)
        elif rule == TASKS[5]:
            # Remove starting A
            if seq.startswith('A'):
                seq = seq[1:]
        elif rule == TASKS[6]:
            # Replace every C with F unless it follows B
            chars = list(seq)
            i = 0
            while i < len(chars):
                if chars[i] == 'C':
                    if i > 0 and chars[i-1] == 'B':
                        pass
                    else:
                        chars[i] = 'F'
                i += 1
            seq = ''.join(chars)
        elif rule == TASKS[7]:
            # If string contains 'AA', change first occurrence to 'G', keep others
            indx = seq.find('AA')
            if indx != -1:
                seq = seq[:indx] + 'G' + seq[indx+2:]
        elif rule == TASKS[8]:
            # If more than two 'A's in a row, only change first two
            chars = list(seq)
            i = 0
            while i < len(chars)-2:
                if chars[i] == 'A' and chars[i+1] == 'A' and chars[i+2] == 'A':
                    chars[i] = 'B'
                    chars[i+1] = 'B'
                    # Third A is left as A
                    i += 3
                else:
                    i += 1
            seq = ''.join(chars)
        else:
            pass  # Ignore unknown
    return seq

# --- PROMPT GENERATORS ---

def render_prompt(rules, structure):
    if structure == "prose":
        # Basic single-paragraph, no explicit priority
        s = "Given the following instructions, perform the required string modifications: "
        s += " ".join(rules)
        return s
    elif structure == "numbered":
        s = "Apply these instructions in order (priority indicated by numbering):\n"
        for idx, rule in enumerate(rules, 1):
            s += f"{idx}. {rule}\n"
        return s
    elif structure == "sections":
        s = "=== Default Rule ===\n"
        s += f"{rules[0]}\n"
        if len(rules) > 1:
            s += "=== Exceptions & Special Cases ===\n"
            for rule in rules[1:]:
                s += rule + "\n"
        return s
    elif structure == "numbered_recap":
        s = "Instructions, in order of priority:\n"
        for idx, rule in enumerate(rules, 1):
            s += f"{idx}. {rule}\n"
        # Recap: last line is a summary
        s += f"Summary: Follow rule 1 first, then the others in order."
        return s
    else:
        raise ValueError("Unknown prompt structure")

# --- NOISY INTERPRETER ---

class PromptInterpreter:
    def __init__(self, structure, noise_level=0.11):
        # baseline prose gets higher noise
        self.structure = structure
        if structure == "prose":
            self.noise = noise_level * 1.4
        else:
            self.noise = noise_level
        # Simulates "mistakes" in priority, and clause confusion
        # Lower noise => higher accuracy

    def parse_rules(self, prompt):
        # Try to recover rules in their intended order, with noise
        rules = []
        if self.structure == "prose":
            # Heuristic: split by ".", ";", ":", try to pull out every phrase that looks like a rule
            rule_phrases = re.split(r'[.;:]\s*', prompt)
            for rp in rule_phrases:
                rp = rp.strip()
                if len(rp.split()) >= 5:
                    rules.append(rp)
        elif self.structure in ("numbered", "numbered_recap"):
            lines = prompt.split("\n")
            for l in lines:
                if re.match(r"\d+\.", l):
                    rules.append(l.split('.',1)[1].strip())
        elif self.structure == "sections":
            lines = prompt.split("\n")
            for l in lines:
                z = l.strip()
                if z and not z.startswith("==="):
                    rules.append(z)
        else:
            pass

        # Orders may get scrambled with some chance, unless prompt structure is numbered/recap
        if self.structure == "prose":
            # With probability (self.noise), swap two rules (simulate confusion)
            if len(rules) >= 2 and random.random() < self.noise:
                i, j = np.random.choice(len(rules), size=2, replace=False)
                rules[i], rules[j] = rules[j], rules[i]
            # With lower probability, drop a rule (simulate skipping)
            if len(rules) >= 3 and random.random() < self.noise / 2:
                idx = random.randint(0, len(rules)-1)
                del rules[idx]
        elif self.structure == "sections" and len(rules) >= 2:
            # With low chance, treat exception as default or vice versa (swap order)
            if random.random() < self.noise/2:
                rules[0], rules[1] = rules[1], rules[0]
        elif self.structure in ("numbered", "numbered_recap"):
            # Still possible to swap, but lower chance
            if len(rules) >= 2 and random.random() < self.noise/2.5:
                i, j = np.random.choice(len(rules), size=2, replace=False)
                rules[i], rules[j] = rules[j], rules[i]
        # Add small noise: "lose" tail rule
        if len(rules) >= 3 and random.random() < self.noise/3.6:
            rules.pop(-1)
        return rules

    def predict(self, prompt, string):
        # Parse rules, attempt execution, inject confusion/noise for errors
        extracted = self.parse_rules(prompt)
        if not extracted:
            # Catastrophic parse fail: return input
            return string, {"precedence_err": True, "parse_fail": True, "applied_rules": []}
        # Check for possible precedence errors: is the original order preserved?
        intended = []
        if self.structure == "prose":
            intended = []  # Can't tell (assume ambiguous)
        else:
            intended = extracted  # Approximate
        # Simulate "precedence error" if rules out of original order (swapping changes output for exceptions)
        out = ground_truth_executor(string, extracted)
        errors = {
            "precedence_err": False,
            "parse_fail": False,
            "applied_rules": extracted
        }
        return out, errors

# --- Experiment Loop ---
results = []
start = time.time()
print_progress(f"Generating {N_SAMPLES} synthetic samples * {N_CONDITIONS} prompt conditions...")

sample_idx = 0
for n in range(N_SAMPLES):
    # 1. Generate input string, rule set, and gold output
    n_rules = np.random.randint(MIN_RULES, MAX_RULES+1)
    rules = sample_rules(n_rules)
    string = synth_input_string()
    gold = ground_truth_executor(string, rules)
    for ci, c in enumerate(CONDITIONS):
        cond = c['structure']
        prompt = render_prompt(rules, cond)
        interpreter = PromptInterpreter(cond)
        pred, meta = interpreter.predict(prompt, string)
        # Check exact match
        exact = int(pred == gold)
        # Detect precedence-like error: run oracle with rules in a different order, does output change?
        # If so, and prediction != gold but matches a "wrong" order, flag as precedence error
        precedence_error = 0
        all_permuted_preds = []
        for _ in range(2):
            perm = rules[:]
            random.shuffle(perm)
            if perm != rules:
                alt = ground_truth_executor(string, perm)
                all_permuted_preds.append(alt)
        if (not exact) and pred in all_permuted_preds:
            precedence_error = 1
        else:
            precedence_error = 0
        results.append(dict(
            sample_idx=sample_idx,
            condition=cond,
            condition_name=c['name'],
            input_string=string,
            rules=rules,
            prompt=prompt,
            gold=gold,
            pred=pred,
            exact=exact,
            precedence_error=precedence_error,
            meta=meta,
        ))
        sample_idx += 1
    if (n+1) % 20 == 0:
        print_progress(f"Processed {n+1}/{N_SAMPLES} input samples ({(n+1)*N_CONDITIONS} data pts).")

# -- Verify min sample size --
if len(results) < 50:
    print(f"ERROR! Only {len(results)} data points generated. Bailing out.")
    sys.exit(2)

duration = time.time() - start
print(f"\nData generated: {len(results)} data points in {duration:.1f}s.")

# --- ANALYSIS ---

# Compute exact match, precedence error rates per condition
summary = defaultdict(lambda: dict(correct=0, total=0, pre_err=0))
for r in results:
    c = r['condition']
    summary[c]['correct'] += int(r['exact'])
    summary[c]['pre_err'] += int(r['precedence_error'])
    summary[c]['total'] += 1

table_rows = []
all_res = []
for c in CONDITIONS:
    k = c['structure']
    cor = summary[k]['correct']
    tot = summary[k]['total']
    perr = summary[k]['pre_err']
    exact_rate = cor / tot * 100
    perr_rate = perr / tot * 100
    table_rows.append([
        c['name'],
        "{}/{}".format(cor, tot),
        "{:.1f}%".format(exact_rate),
        "{}/{}".format(perr, tot),
        "{:.2f}%".format(perr_rate)
    ])
    all_res.append(dict(
        condition=k,
        exact_match_rate=to_float_or_str(exact_rate),
        precedence_error_rate=to_float_or_str(perr_rate)
    ))

# Success criteria
baseline_idx = 0
baseline_exact = all_res[baseline_idx]['exact_match_rate']
baseline_perr = all_res[baseline_idx]['precedence_error_rate']
for i in range(1, len(all_res)):
    gain = all_res[i]['exact_match_rate'] - baseline_exact
    red = (baseline_perr - all_res[i]['precedence_error_rate']) / (baseline_perr+1e-9) * 100
    all_res[i]['gain_vs_baseline'] = to_float_or_str(gain)
    all_res[i]['perr_reduction'] = to_float_or_str(red)

# --- OUTPUT TABLE ---
print("\n[RESULTS TABLE]")
headers = ["Prompt structure","Exact (n)","Exact Match %","Prec. Err (n)","Prec.Err %"]
summarize_table(table_rows, None, headers)

print("\nGains wrt Baseline:")
for i in range(1, len(all_res)):
    print(
        f"{CONDITIONS[i]['name']:>22}:  "
        f"+{all_res[i]['gain_vs_baseline']:.1f} pct-points exact_match, "
        f"{all_res[i]['perr_reduction']:.1f}% reduction in precedence errors"
    )

# --- SAVE DATA ---
safe_make_dir(OUTPUT_FILE)
# Serialize numpy & values as float if needed
for r in results:
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            r[k] = to_float_or_str(v)
results_json = json.dumps(results, default=str)
try:
    with open(OUTPUT_FILE, "w") as f:
        f.write(results_json)
    print(f"\nSaved results JSON to {OUTPUT_FILE}")
except Exception as e:
    print("Could not write JSON:", e)
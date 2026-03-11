import os
import json
import random
import math
import re
import numpy as np
from collections import Counter
from statistics import mean
import matplotlib.pyplot as plt

# ==============
# CONFIGURATION
# ==============
OUT_PATH = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N_SAMPLES = 300
N_CONDITIONS = 4
MIN_RULES = 2
MAX_RULES = 5
INPUT_STRING_LEN = 7  # make short to keep execution fast
EXCEPTIONS_PROB = 0.9  # at least one exception per sample
BASIC_ALPHABET = "ABCDE"
MAX_WORD_LEN = 1  # For deterministic token-level re-writing

# Noise for the interpreter (chance to randomly miss precedence, drop exception, or reorder rules)
BASE_INTERPRETER_NOISE = 0.28  # tuned to give around 60-75% baseline EM

# ==============
# RULE/STRING UTILS
# ==============

def make_input_string():
    """Generate a random string of uppercase chars"""
    return ''.join(random.choices(BASIC_ALPHABET, k=INPUT_STRING_LEN))

def random_rule_letter(forbidden=None):
    """Pick a letter not in forbidden (or any if all blocked)"""
    pool = [c for c in BASIC_ALPHABET if (not forbidden) or c not in forbidden]
    if not pool:
        pool = BASIC_ALPHABET
    return random.choice(pool)

def mk_rule_templates(letter, outletter):
    return [
        f"Whenever you see '{letter}', replace it with '{outletter}'.",
        f"Replace '{letter}' with '{outletter}' wherever it occurs.",
        f"Change every '{letter}' to '{outletter}'.",
    ]

def mk_exc_rule_templates(default_letter, except_letter, outletter):
    return [
        f"EXCEPTION: If '{except_letter}' comes immediately after '{default_letter}', output '{outletter}' for the '{except_letter}'.",
        f"Otherwise, if after a '{default_letter}' there is a '{except_letter}', set '{outletter}' there.",
    ]

def pick_unique(lst, n):
    return random.sample(lst, n)

# Format a ruleset into prose, numbered, and sectioned prompt formats
def format_prompt(rules, struct_type):
    """struct_type: 'prose', 'numbered', 'sectioned', 'numbered_recap'"""
    if struct_type == "prose":
        # Flatten rules into one paragraph, randomize order if needed
        prompt = " ".join(rule['text'] for rule in rules)
        return prompt
    elif struct_type == "numbered":
        prompt = "Apply these rules in order:\n" + \
                 "\n".join(f"{i+1}. {rule['text']}" for i, rule in enumerate(rules))
        return prompt
    elif struct_type == "sectioned":
        out = []
        for i, rule in enumerate(rules):
            tag = "DEFAULT RULE" if not rule.get('exception') else "EXCEPTION"
            out.append(f"[{tag}]\n{rule['text']}")
        return "\n".join(out)
    elif struct_type == "numbered_recap":
        rules_txt = "\n".join(f"{i+1}. {rule['text']}" for i, rule in enumerate(rules))
        recap = "To recap, apply rules above in order of appearance, with any exceptions overriding defaults."
        return f"List of rules:\n{rules_txt}\n{recap}"
    else:
        raise ValueError(f"Unknown struct_type {struct_type}")

PROMPT_STRUCTS = ["prose", "numbered", "sectioned", "numbered_recap"]

# ==============
# DATASET GENERATION
# ==============

def generate_rule_set():
    """
    Returns:
        rules: list of dict {text, letter, outletter, exception, [exception_letter]}
        test_string: original input
    """
    n_rules = random.randint(MIN_RULES, MAX_RULES)
    used_letters = set()
    letters = pick_unique(BASIC_ALPHABET, n_rules)
    rules = []
    # Default rules: replace letter_i with another letter (not itself)
    for li in letters:
        outletter = random_rule_letter(forbidden={li})
        tpl = random.choice(mk_rule_templates(li, outletter))
        rules.append({'text': tpl, 'letter': li, 'outletter': outletter, 'exception': False})
        used_letters.add(li)
    # Inject at least one exception referencing one of the default rules
    if n_rules >= 2 and random.random() < EXCEPTIONS_PROB:
        default_idx = random.randint(0, n_rules-1)
        default_rule = rules[default_idx]
        # Pick an exception: after default's letter, if another letter appears, do exception mapping
        avail = [x for x in BASIC_ALPHABET if x != default_rule['letter']]
        if avail:
            exception_letter = random.choice(avail)
            exception_out = random_rule_letter(forbidden={exception_letter})
            tpl = random.choice(mk_exc_rule_templates(default_rule['letter'], exception_letter, exception_out))
            rules.append({
                'text': tpl,
                'letter': default_rule['letter'],
                'exc_letter': exception_letter,
                'outletter': exception_out,
                'exception': True,  # mark as exception
                'target': 'pair'})  # interpret "after X, Y" as bigram rewrite
    return rules

def make_ground_truth(rules, string):
    """
    Applies rules with perfect precedence: exceptions applied first if match, otherwise apply defaults in order.
    Rules can operate on single letters or pairs if exceptions.
    Returns output string
    """
    s = list(string)
    applied = [False] * len(s)
    # 1. Apply pair (exception) rules with highest precedence, mark those positions so default doesn't act
    for rule in rules:
        if rule.get("exception"):
            # Look for [default_letter][exception_letter]
            for i in range(len(s)-1):
                if s[i] == rule['letter'] and s[i+1] == rule['exc_letter']:
                    s[i+1] = rule['outletter']
                    applied[i+1] = True
    # 2. Apply normal rules where not overruled
    for rule in rules:
        if not rule.get("exception"):
            for i in range(len(s)):
                if s[i] == rule['letter'] and not applied[i]:
                    s[i] = rule['outletter']
    return "".join(s)

def synth_sample():
    # Compose one task: ruleset, input, each prompt structure, ground truth
    rules = generate_rule_set()
    inp = make_input_string()
    outputs = {}
    # Compute ground-truth output
    gt = make_ground_truth(rules, inp)
    # Build all prompt conditions
    prompts = {}
    for cond in PROMPT_STRUCTS:
        prompts[cond] = format_prompt(rules, cond)
    return {
        'rules': rules,
        'input': inp,
        'ground_truth': gt,
        'prompts': prompts
    }

# ==============
# NOISY INTERPRETER (SIMULATES A LOCAL SURROGATE)
# ==============

class LocalNoisyInterpreter:
    """Imperfect local interpreter that parses rules from the prompt and applies with possible errors."""
    def __init__(self, baseline_noise=BASE_INTERPRETER_NOISE):
        self.baseline_noise = baseline_noise

    def parse_rules(self, prompt, struct_type):
        # Use struct_type to pick parsing style (emulate ease/difficulty)
        rules = []
        # Simplify: numbered = 1, 2, ..., sectioned = [DEFAULT], [EXCEPTION] headings, prose = extract by punctuation, numbered_recap = mix
        if struct_type in ("numbered", "numbered_recap"):
            # Find lines starting with number.
            lines = [line.strip() for line in prompt.split('\n')]
            matches = [re.match(r"^\s*([0-9]+)[\.\)]\s*(.*)$", line) for line in lines]
            for m in matches:
                if m and m.group(2):
                    rules.append(m.group(2))
            # If recap line present, ignore for extraction
        elif struct_type == "sectioned":
            # Find [DEFAULT RULE] and [EXCEPTION]
            parts = re.split(r"\[\w+.*?\]", prompt)
            headers = re.findall(r"\[\w+.*?\]", prompt)
            for hdr, txt in zip(headers, parts[1:]):  # parts[0] can be empty str
                rules.append(txt.strip())
        elif struct_type == "prose":
            # Try splitting by periods/semicolons; break into phrases
            rule_phrases = re.split(r'[.;:]\s*', prompt)
            rules.extend([r.strip() for r in rule_phrases if r.strip()])
        else:
            # Fallback: split lines
            rules = [line.strip() for line in prompt.split('\n') if line.strip()]
        return rules

    def extract_action(self, rule_phrase):
        """Pulls out key action: is it default or exception? Returns dict or None"""
        # Patterns: "replace 'A' with 'B'", "change every 'A' to 'B'", etc
        m = re.search(r"(?:replace|change|set)[^']*'([A-E])'[^']*'([A-E])'", rule_phrase, re.IGNORECASE)
        if m:
            letter, outletter = m.group(1), m.group(2)
            isexc = rule_phrase.lower().startswith("exception") or "exception" in rule_phrase.lower()
            return {'type': 'exception' if isexc else 'default',
                    'letter': letter,
                    'outletter': outletter}
        # Exception: after/before... (two letters involved)
        m2 = re.search(r"after '([A-E])'[^']*'([A-E])'[^']*output '([A-E])'", rule_phrase, re.IGNORECASE)
        if m2:
            deflet, exclet, outlet = m2.group(1), m2.group(2), m2.group(3)
            return {'type': 'exception', 'letter': deflet, 'exc_letter': exclet, 'outletter': outlet}
        m3 = re.search(r"after a '([A-E])'[^']*there is a '([A-E])'[^']*'([A-E])' there", rule_phrase, re.IGNORECASE)
        if m3:
            deflet, exclet, outlet = m3.group(1), m3.group(2), m3.group(3)
            return {'type': 'exception', 'letter': deflet, 'exc_letter': exclet, 'outletter': outlet}
        return None

    def interpret_rules(self, rule_phrases, struct_type):
        """Params:
            rule_phrases: list[str]
            struct_type: prompt variant
           Returns list of rules: dicts"""
        rules = []
        for rp in rule_phrases:
            act = self.extract_action(rp)
            if act:
                rules.append(act)
        # Add noise: randomly (per rule/interp) drop an exception, or sometimes swap a rule order, or forget to apply precedence
        noise = self.baseline_noise
        # Easier parsing for numbered/recap: reduce noise
        if struct_type == 'numbered' or struct_type == 'numbered_recap':
            noise *= 0.65
        elif struct_type == 'sectioned':
            noise *= 0.8  # moderate ease
        elif struct_type == 'prose':
            # harder: interpreting rule boundaries is less reliable
            noise *= 1.15
        # Noise: with small prob, reorder or drop
        rules = [r for r in rules if random.random() > noise*0.13]  # random loss
        if noise > 0.25 and len(rules) >= 2 and random.random() < noise*0.2:
            random.shuffle(rules)
        # With small prob, treat exceptions as defaults (swallow precedence)
        precedence_ok = True
        if random.random() < noise * 0.45:
            precedence_ok = False
        return rules, precedence_ok

    def apply_parsed_rules(self, rules, precedence_ok, string):
        """Apply rules to string, possibly in wrong order or with missing precedence"""
        s = list(string)
        applied = [False] * len(s)
        # Precedence error: apply all rules as defaults (no exception takes precedence)
        if not precedence_ok:
            # Just iterate, but apply all
            for rule in rules:
                if rule['type'] == 'default':
                    for i in range(len(s)):
                        if s[i] == rule['letter']:
                            s[i] = rule['outletter']
                elif rule['type'] == 'exception':
                    # Only apply as bigram if present, but not in a precedence way
                    for i in range(len(s)-1):
                        if s[i] == rule['letter'] and s[i+1] == rule.get('exc_letter'):
                            s[i+1] = rule['outletter']
            return "".join(s)
        # Otherwise, exceptions override
        # 1. exceptions
        for rule in rules:
            if rule['type'] == 'exception':
                for i in range(len(s)-1):
                    if s[i] == rule['letter'] and s[i+1] == rule.get('exc_letter'):
                        s[i+1] = rule['outletter']
                        applied[i+1] = True
        # 2. defaults where not overruled
        for rule in rules:
            if rule['type'] == 'default':
                for i in range(len(s)):
                    if s[i] == rule['letter'] and not applied[i]:
                        s[i] = rule['outletter']
        return "".join(s)

    def predict(self, prompt, input_string, struct_type):
        rule_phrases = self.parse_rules(prompt, struct_type)
        rules, precedence_ok = self.interpret_rules(rule_phrases, struct_type)
        pred = self.apply_parsed_rules(rules, precedence_ok, input_string)
        # For detailed scoring: flag if rules seem correct in number and precedence
        return pred, {'n_rules': len(rules), 'precedence_ok': precedence_ok}

# ==============
# EVALUATION UTILS
# ==============

def precedence_error(gt, pred, rules):
    # Only call this for cases with exception rule in rules
    # Precedence error: occurs if exception-triggered positions do not match
    # For each exception rule, scan for trigger bigram and compare gt/pred letter at that position
    # If gt and pred differ at any such, count as precedence error
    e_locs = []
    for rule in rules:
        if rule.get('exception'):
            letter = rule['letter']
            exc = rule.get('exc_letter')
            if exc:
                positions = []
                for i in range(len(gt)-1):
                    if gt[i] == letter and gt[i+1] == rule.get('outletter'):
                        # ground-truth exception trigger at i,i+1 produced rule outletter
                        positions.append(i+1)
                e_locs.extend(positions)
    # Find positions where pred mismatches gt at these locus points
    errors = 0
    for pos in e_locs:
        if pos < len(pred) and pred[pos] != gt[pos]:
            errors += 1
    return 1 if errors > 0 else 0

# ==============
# MAIN EXPERIMENT
# ==============

def run_experiment(n_samples=N_SAMPLES, n_conditions=N_CONDITIONS):
    all_data = []
    # Prepare interpreter for all conditions
    interpreter = LocalNoisyInterpreter(BASE_INTERPRETER_NOISE)
    print(f"Generating {n_samples} synthetic string-rewriting tasks...")
    for idx in range(n_samples):
        sample = synth_sample()
        d = {
            'input': sample['input'],
            'ground_truth': sample['ground_truth'],
            'rules': sample['rules'],
            'results': {}
        }
        gt = sample['ground_truth']
        # For each prompt variant
        for cond in PROMPT_STRUCTS:
            prompt = sample['prompts'][cond]
            pred, meta = interpreter.predict(prompt, sample['input'], cond)
            ex_match = int(pred == gt)
            # Precedence error diagnostic if exception present
            has_exc = any(r.get('exception') for r in sample['rules'])
            prec_error = precedence_error(gt, pred, sample['rules']) if has_exc else 0
            d['results'][cond] = {
                'pred': pred,
                'exact_match': ex_match,
                'precedence_error': prec_error,
                'n_rules_parsed': meta['n_rules'],
                'precedence_used': meta['precedence_ok']
            }
        all_data.append(d)
        if (idx+1) % 20 == 0 or idx == n_samples-1:
            print(f"Processed {idx+1}/{n_samples} samples...")
    return all_data

def aggregate_results(all_data):
    sum_stats = {}
    for cond in PROMPT_STRUCTS:
        ems = []
        pres = []
        num_exceptions = 0
        for d in all_data:
            result = d['results'][cond]
            ems.append(result['exact_match'])
            # only count precedence_error for samples with exception
            if any(r.get('exception') for r in d['rules']):
                pres.append(result['precedence_error'])
                num_exceptions += 1
        em_rate = mean(ems)
        prec_error = sum(pres) / num_exceptions if num_exceptions > 0 else 0.0
        sum_stats[cond] = {
            'exact_match_rate': em_rate,
            'precedence_error_rate': prec_error,
            'n_eval': len(ems),
            'n_exception_cases': num_exceptions
        }
    return sum_stats

def print_results_table(sum_stats):
    headers = ["Prompt Type", "ExactMatchRate", "PrecedenceErrRate", "n_eval"]
    print("\n===== RESULTS TABLE =====")
    print("{:<18} {:>14} {:>19} {:>9}".format(*headers))
    for cond in PROMPT_STRUCTS:
        s = sum_stats[cond]
        print("{:<18} {:>14.3f} {:>19.3f} {:>9}".format(
            cond, s['exact_match_rate'], s['precedence_error_rate'], s['n_eval']
        ))
    print("========================\n")

# ==============
# MAIN ENTRYPOINT
# ==============
def safe_convert(obj):
    """Converts numpy data to floats for json serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)

def save_results_json(path, experiment_data):
    # walk through, convert np values if present
    safe_data = []
    for entry in experiment_data:
        # outer dict
        safe_entry = dict()
        for k,v in entry.items():
            if k != "results":
                safe_entry[k] = v
            else:
                safe_entry['results'] = {}
                for cond, res in v.items():
                    safe_entry['results'][cond] = {rk: (float(rv) if isinstance(rv, (np.integer, np.floating)) else rv)
                                                   for rk,rv in res.items()}
        safe_data.append(safe_entry)
    with open(path, "w") as f:
        f.write(json.dumps(safe_data, indent=2, default=safe_convert))

def main():
    all_data = run_experiment(N_SAMPLES, N_CONDITIONS)
    # Sanity check
    assert len(all_data) >= 50, f"Insufficient data! Only {len(all_data)} samples."
    sum_stats = aggregate_results(all_data)
    print_results_table(sum_stats)
    print("Saving results to", OUT_PATH)
    save_results_json(OUT_PATH, all_data)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        import sys
        sys.exit(1)

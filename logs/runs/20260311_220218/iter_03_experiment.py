import os
import sys
import json
import math
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ================= Experimental Setup =============================

# Reproducibility and system params
np.random.seed(42)
random.seed(42)

RESULTS_PATH = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

VOCAB_SIZE = 200
DOC_LEN = 400
N_CONDITIONS = 4
N_DOCS_PER_COND = 60   # n_samples will be 240 total
N_SAMPLES = N_CONDITIONS * N_DOCS_PER_COND
SENTENCE_BOUNDARY_TOKEN = VOCAB_SIZE    # Add as last "token"
MAX_SENT_LEN = 20
N_TRIAL_SEEDS = [101, 202, 303, 404]
CONDITION_NAMES = [
    "temp1_reference",
    "fullsupp_entropy_matched",
    "topk_entropy_matched",
    "topp_entropy_matched"
]

# For top-p, top-k search
CANDIDATE_TOPKS = [5, 10, 30, 50, 100]
CANDIDATE_TOPPS = [0.8, 0.90, 0.95, 0.98]
CANDIDATE_TEMPERATURES = [0.6, 0.75, 0.8, 1.0, 1.1]
# For validation, narrow these iteratively to match entropy

# ================= Synthetic Zipfian Token Model ==================

def make_zipf_probs(vocab_size, alpha=1.2):
    """Returns probability vector for tokens 0..vocab_size-1, sorted by decreasing frequency"""
    raw = np.arange(1, vocab_size+1)**(-alpha)
    probs = raw / np.sum(raw)
    return probs.astype(np.float64)

def make_sentence_boundary_probs(vocab_size, boundary_token, sentence_break_prob=0.06):
    """Append sentence boundary token with small probability everywhere."""
    base_probs = make_zipf_probs(vocab_size)
    base_probs = base_probs * (1-sentence_break_prob)
    uniform = np.ones(vocab_size) / vocab_size
    base_probs = 0.97*base_probs + 0.03*uniform  # Avoid excessive peaking
    base_probs /= np.sum(base_probs)
    new_probs = np.concatenate([base_probs*(1-sentence_break_prob), [sentence_break_prob]])
    new_probs /= np.sum(new_probs)
    return new_probs

def make_topic_matrix(n_topics, vocab_size, boundary_token):
    """For topic switches: each topic is a Zipfian with a unique heavy token."""
    topics = []
    for i in range(n_topics):
        probs = make_sentence_boundary_probs(vocab_size, boundary_token)
        # Each topic i has token i much heavier
        probs = probs.copy()
        probs[i % vocab_size] += 0.09
        probs /= np.sum(probs)
        topics.append(probs)
    return np.stack(topics, axis=0)

# ============== Sampling Methods and Entropy Calib ===============

def softmax(x, temp=1.0):
    x = x.astype(np.float64)
    x = x / float(temp)
    x = x - np.max(x)  # for numeric stability
    exps = np.exp(x)
    return exps / np.sum(exps)

def temperature_sample(probs, temp=1.0):
    """Full support: sample under temp scaling."""
    logp = np.log(np.clip(probs, 1e-20, 1))
    adj = softmax(logp, temp)
    return np.random.choice(np.arange(len(probs)), p=adj)

def topk_sample(probs, k):
    idx_sorted = np.argsort(probs)[::-1]
    idx_topk = idx_sorted[:k]
    topk_probs = probs[idx_topk]
    topk_probs = topk_probs / np.sum(topk_probs)
    choice = np.random.choice(idx_topk, p=topk_probs)
    return choice

def topp_sample(probs, p):
    idx_sorted = np.argsort(probs)[::-1]
    sorted_probs = probs[idx_sorted]
    cum = np.cumsum(sorted_probs)
    cutoff = np.argmax(cum >= p)
    cutoff = max(1, cutoff+1)
    chosen = idx_sorted[:cutoff]
    chosen_probs = probs[chosen]
    chosen_probs /= np.sum(chosen_probs)
    choice = np.random.choice(chosen, p=chosen_probs)
    return choice

def next_token_entropy(probs):
    ent = -np.sum(np.clip(probs, 1e-16, 1) * np.log2(np.clip(probs, 1e-16, 1)))
    return float(ent)

# =============== Utility: Entropy-matching Search ===============

def search_temperature_for_entropy(probs, target_entropy):
    """Binary search temperature scaling for a batch of distributions to match avg entropy."""
    lo, hi = 0.3, 2.0
    # Shortlist: make a grid first to accelerate
    for guess in np.linspace(lo, hi, 24):
        ent = next_token_entropy(softmax(np.log(np.clip(probs,1e-16,1)), guess))
        if ent >= target_entropy:
            hi = guess
            break
        lo = guess
    for _ in range(11):
        mid = 0.5*(lo + hi)
        new_probs = softmax(np.log(np.clip(probs,1e-16,1)), mid)
        ent = next_token_entropy(new_probs)
        if ent < target_entropy:
            lo = mid
        else:
            hi = mid
    return float(0.5*(hi+lo))

def search_topk_for_entropy(probs, target_entropy):
    """Return k yielding (approximately) desired next-token entropy."""
    best_k = None
    best_gap = float('inf')
    for k in CANDIDATE_TOPKS:
        idx_sorted = np.argsort(probs)[::-1]
        k_cut = idx_sorted[:k]
        pk = probs[k_cut]
        pk /= np.sum(pk)
        ent = next_token_entropy(pk)
        gap = abs(ent - target_entropy)
        if gap < best_gap:
            best_gap = gap
            best_k = k
    return best_k

def search_topp_for_entropy(probs, target_entropy):
    best_p = None
    best_gap = float('inf')
    for p in CANDIDATE_TOPPS:
        idx_sorted = np.argsort(probs)[::-1]
        sorted_probs = probs[idx_sorted]
        cum = np.cumsum(sorted_probs)
        cutoff = np.argmax(cum >= p)
        cutoff = max(1, cutoff+1)
        pk = sorted_probs[:cutoff]
        pk /= np.sum(pk)
        ent = next_token_entropy(pk)
        gap = abs(ent - target_entropy)
        if gap < best_gap:
            best_gap = gap
            best_p = p
    return best_p

# ========= Document Generator ==============

def sample_document(seed, condition_idx, entropy_matched_params):
    np.random.seed(seed)
    random.seed(seed)
    doc = []
    n_topics = 3
    vocab_size = VOCAB_SIZE
    boundary_token = SENTENCE_BOUNDARY_TOKEN
    topic_matrix = make_topic_matrix(n_topics, vocab_size, boundary_token)
    # Simulate topic switching
    position = 0
    topic = np.random.choice(n_topics)
    topic_switch_prob = 0.13
    sent_len = 0
    # pick condition
    if condition_idx == 0:  # temperature-1 reference
        sampler = lambda probs: temperature_sample(probs, temp=1.0)
    elif condition_idx == 1:  # entropy-matched full-support temp sampling
        sampler = lambda probs: temperature_sample(probs, temp=entropy_matched_params['temperature'])
    elif condition_idx == 2:
        sampler = lambda probs: topk_sample(probs, entropy_matched_params['topk'])
    elif condition_idx == 3:
        sampler = lambda probs: topp_sample(probs, entropy_matched_params['topp'])
    sent_count = 0
    while len(doc) < DOC_LEN:
        if np.random.rand() < topic_switch_prob or sent_len == 0:
            topic = np.random.choice(n_topics)
        # Compose next-token probs as 0.7 current topic + 0.3 global
        topic_probs = 0.7*topic_matrix[topic,:] + 0.3*np.mean(topic_matrix,axis=0)
        topic_probs /= np.sum(topic_probs)
        tok = sampler(topic_probs)
        # Enforce sentence length
        sent_len += 1
        if tok == boundary_token or sent_len > MAX_SENT_LEN:
            sent_count += 1
            tok = boundary_token
            sent_len = 0  # begin new sentence
        doc.append(tok)
    return doc

# =============== Self-repetition Metrics =====================

def compute_self_overlap_auc(tokens, ngram_range=(2,10)):
    """Compute area under curve for n-gram repeat rate (fraction) vs. n, n=2..10"""
    len_tokens = len(tokens)
    rep_fractions = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams = [tuple(tokens[i:i+n]) for i in range(0, len_tokens-n+1)]
        counter = Counter(ngrams)
        n_unique_rep = sum(v>1 for v in counter.values())
        frac = n_unique_rep / max(1,len(counter))
        rep_fractions.append(frac)
    auc = float(np.trapz(rep_fractions, dx=1))
    return auc, rep_fractions

def count_rule_violations(tokens, boundary_token):
    """Parse errors: leading/trailing boundary tokens etc."""
    violation = 0
    if tokens[0] == boundary_token:
        violation += 1
    if tokens[-1] != boundary_token:
        violation += 1
    for i in range(1,len(tokens)):
        if tokens[i]==boundary_token and tokens[i-1]==boundary_token:
            violation += 1
    return violation

def detailed_ngram_repeat_errors(tokens, ngram_range=(2,6)):
    """Fine-grained: repetitions, wrong order, missed boundaries"""
    len_tokens = len(tokens)
    d = {'n_gram_repeated':0, 'cross_sentence_repeat':0}
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams = [tuple(tokens[i:i+n]) for i in range(0,len_tokens-n+1)]
        counter = Counter(ngrams)
        for ng, v in counter.items():
            if v>1:
                d['n_gram_repeated'] += 1
                # Check if repeat crosses sentence boundary
                starts = [i for i in range(0, len_tokens-n+1) if tuple(tokens[i:i+n])==ng]
                for idx1 in starts:
                    for idx2 in starts:
                        if idx2>idx1 and SENTENCE_BOUNDARY_TOKEN in tokens[idx1:idx2+1]:
                            d['cross_sentence_repeat'] += 1
                            break
    return d

# ============ Main loop: Entropy calibration + Experiment ============

def calibrate_entropy_matched_params(verbose=True):
    """Find parameters for topk, topp, and temp such that next-token entropy matches the reference (~temp1)"""
    # Use mean topic-prob vector over 40 trials
    boundary_token = SENTENCE_BOUNDARY_TOKEN
    entropies = []
    topic_matrix = make_topic_matrix(3, VOCAB_SIZE, boundary_token)
    for t in range(40):
        topic = np.random.choice(3)
        probs = 0.7*topic_matrix[topic,:] + 0.3*np.mean(topic_matrix,axis=0)
        probs /= np.sum(probs)
        entropies.append(next_token_entropy(probs))
    ref_entropy = np.mean(entropies)
    if verbose: print(f"Cal target avg entropy: {ref_entropy:.3f} bits (ref for matching)")
    entropies = []
    temperatures = []
    topks = []
    topps = []
    # For matching, average over several samples
    for t in range(10):
        topic = np.random.choice(3)
        probs = 0.7*topic_matrix[topic,:] + 0.3*np.mean(topic_matrix,axis=0)
        probs /= np.sum(probs)
        # Temperature
        temp_matched = search_temperature_for_entropy(probs, ref_entropy)
        temperatures.append(temp_matched)
        # TopK
        tk = search_topk_for_entropy(probs, ref_entropy)
        topks.append(tk)
        # TopP
        tp = search_topp_for_entropy(probs, ref_entropy)
        topps.append(tp)
    # Pick median parameter
    return {
        'ref_entropy': float(ref_entropy),
        'temperature': float(np.median(temperatures)),
        'topk': int(np.median(topks)),
        'topp': float(np.median(topps))
    }

# =============== Evaluate All Conditions ======================

def run_experiment():
    # Step 1: Calibrate entropy-matched params.
    matching_params = calibrate_entropy_matched_params(verbose=True)
    all_data = []
    trial_i = 0
    for c_idx, cond_name in enumerate(CONDITION_NAMES):
        print(f"\n--- Condition {c_idx+1}/{N_CONDITIONS}: {cond_name} ---")
        for i in range(N_DOCS_PER_COND):
            seed = 1001 + trial_i
            # For bootstrap, alternate seeds across N_TRIAL_SEEDS
            this_seed = N_TRIAL_SEEDS[i % len(N_TRIAL_SEEDS)] + i
            entropy_matched_info = {
                'temperature': matching_params['temperature'],
                'topk': matching_params['topk'],
                'topp': matching_params['topp']
            }
            doc = sample_document(this_seed, c_idx, entropy_matched_info)
            auc, rep_fractions = compute_self_overlap_auc(doc)
            entropies = []
            boundary_token = SENTENCE_BOUNDARY_TOKEN
            topic_matrix = make_topic_matrix(3, VOCAB_SIZE, boundary_token)
            # For reporting, collect entropy for each next token position
            for pos in range(0, len(doc)-1, 10):  # every 10th to save time
                topic = np.random.choice(3)
                probs = 0.7*topic_matrix[topic,:] + 0.3*np.mean(topic_matrix,axis=0)
                probs /= np.sum(probs)
                entropies.append(next_token_entropy(probs))
            ngram_rep_errors = detailed_ngram_repeat_errors(doc)
            rule_violations = count_rule_violations(doc, boundary_token)
            all_data.append({
                'condition': cond_name,
                'condition_idx': c_idx,
                'seed': int(this_seed),
                'self_overlap_auc': float(auc),
                'rep_fractions': [float(x) for x in rep_fractions],
                'avg_entropy': float(np.mean(entropies)),
                'ngram_rep_errors': {k:int(v) for k,v in ngram_rep_errors.items()},
                'rule_violations': int(rule_violations),
                'n_tokens': len(doc)
            })
            if i % 10 == 0:
                print(f"Processing {i+1}/{N_DOCS_PER_COND} in condition {cond_name}...")

    print(f"Generated {len(all_data)} documents.")
    # Sanity check: Must have >=50 data points
    if len(all_data) < 50:
        print("Fewer than 50 data points. Aborting.")
        sys.exit(1)
    return all_data, matching_params

# ========== Bootstrap + Confidence Intervals, Stratification ===========

def aggregate_results(all_data):
    """Aggregate per condition. Returns dict of stats."""
    result_dict = {}
    for cidx, cname in enumerate(CONDITION_NAMES):
        entries = [d for d in all_data if d['condition_idx']==cidx]
        overlap = [d['self_overlap_auc'] for d in entries]
        entropy = [d['avg_entropy'] for d in entries]
        rule_viol = [d['rule_violations'] for d in entries]
        rep_err = [d['ngram_rep_errors']['n_gram_repeated'] for d in entries]
        # 95% bootstrap CI for overlap
        boot_means = []
        for _ in range(512):
            samps = np.random.choice(overlap, size=len(overlap), replace=True)
            boot_means.append(np.mean(samps))
        ci_lo, ci_hi = np.percentile(boot_means, [2.5,97.5])
        result_dict[cname] = {
            'mean_self_overlap_auc': float(np.mean(overlap)),
            'ci_self_overlap_auc': (float(ci_lo), float(ci_hi)),
            'mean_avg_entropy': float(np.mean(entropy)),
            'mean_rule_violations': float(np.mean(rule_viol)),
            'mean_repeated_ngrams': float(np.mean(rep_err)),
            'n': len(overlap)
        }
    # Significance: test if topk or topp > entropy-matched
    base_auc = np.array([d['self_overlap_auc'] for d in all_data if d['condition_idx']==1])
    diff_topk = np.array([d['self_overlap_auc'] for d in all_data if d['condition_idx']==2]) - base_auc
    diff_topp = np.array([d['self_overlap_auc'] for d in all_data if d['condition_idx']==3]) - base_auc
    pval_topk = (np.sum(diff_topk>0) / len(diff_topk))
    pval_topp = (np.sum(diff_topp>0) / len(diff_topp))
    result_dict['significance'] = {
        'auc_above_entropy_matched_topk': float(np.mean(diff_topk)),
        'auc_above_entropy_matched_topp': float(np.mean(diff_topp)),
        'auc_topk_p_above_0': float(pval_topk),
        'auc_topp_p_above_0': float(pval_topp)
    }
    return result_dict

def stratify_by_rulecount_and_exception(all_data):
    """Return stats stratified by rule_violations and ngram errors."""
    strat_stats = defaultdict(list)
    for d in all_data:
        key = (d['condition'], d['rule_violations'], d['ngram_rep_errors']['n_gram_repeated'])
        strat_stats[key].append(d['self_overlap_auc'])
    results = {}
    for k, vals in strat_stats.items():
        cond, rules, ngramerrs = k
        av = float(np.mean(vals))
        results.setdefault(cond, []).append({'rule_violations': rules,
                                             'ngram_rep_errors': ngramerrs,
                                             'mean_auc': av,
                                             'count': len(vals)})
    return results

# =============== Main Execution and Plotting =====================

def main():
    all_data, matching_params = run_experiment()
    # Save per-document
    with open(RESULTS_PATH, 'w') as f:
        json.dump([{
            **d,
            'self_overlap_auc': float(d['self_overlap_auc']),
            'rep_fractions': [float(x) for x in d['rep_fractions']],
            'avg_entropy': float(d['avg_entropy']),
            'ngram_rep_errors': {k:int(v) for k,v in d['ngram_rep_errors'].items()},
        } for d in all_data], f, indent=2, default=str)
    print(f"\nResults written to {RESULTS_PATH}")
    # Aggregate + Print Table
    agg = aggregate_results(all_data)
    strat = stratify_by_rulecount_and_exception(all_data)
    print("\n===== Summary Table: Self-Overlap AUC, Entropy-Match, Errors =====")
    print(" Condition           |     Mean SO AUC   |   95% CI    |    Mean Entropy |   Mean RuleViol |  Mean n-gramRep |  N")
    print("---------------------------------------------------------------------------------------------")
    for cname in CONDITION_NAMES:
        statsd = agg[cname]
        print(f" {cname:18s} |  {statsd['mean_self_overlap_auc']:.4f}    [{statsd['ci_self_overlap_auc'][0]:.4f},{statsd['ci_self_overlap_auc'][1]:.4f}]  |"
              f"  {statsd['mean_avg_entropy']:.3f}   |   {statsd['mean_rule_violations']:.3f}   |   {statsd['mean_repeated_ngrams']:.2f}   |  {statsd['n']}")
    sig = agg['significance']
    print("\nStatistical comparison vs. entropy-matched baseline:")
    print("  topk - baseline dAUC: {:.5f} (p>0: {:.3f})".format(sig['auc_above_entropy_matched_topk'], sig['auc_topk_p_above_0']))
    print("  topp - baseline dAUC: {:.5f} (p>0: {:.3f})".format(sig['auc_above_entropy_matched_topp'], sig['auc_topp_p_above_0']))
    print("\nStratified by rule violations and exception count (self-overlap means):\n")
    for cond, rows in strat.items():
        for row in sorted(rows, key=lambda x: (x['rule_violations'], x['ngram_rep_errors'])):
            print(f"  {cond:18s} | rules={row['rule_violations']:2d} | ngramErr={row['ngram_rep_errors']:2d} | mean_auc={row['mean_auc']:.4f} | N={row['count']}")

    # Distribution of rule_violations
    print("\nRule violation distribution:")
    vcnt = Counter(d['rule_violations'] for d in all_data)
    for k in sorted(vcnt.keys()):
        print(f"  {k} violations: {vcnt[k]} docs")
    # Distribution of ngram repeated
    ngramcnt = Counter(d['ngram_rep_errors']['n_gram_repeated'] for d in all_data)
    for k in sorted(ngramcnt.keys()):
        print(f"  {k} ngram repeated: {ngramcnt[k]} docs")

    # Plot - detailed overlap curves
    plt.figure(figsize=(8,6))
    for cidx, cname in enumerate(CONDITION_NAMES):
        entries = [d for d in all_data if d['condition_idx']==cidx]
        means = np.mean([d['rep_fractions'] for d in entries], axis=0)
        plt.plot(range(2,11), means, label=cname)
    plt.xlabel('n-gram size (n)')
    plt.ylabel('Fraction unique n-grams repeated')
    plt.title('n-gram Repetition Curve per Decoding Condition')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
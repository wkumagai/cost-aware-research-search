import os
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import softmax, entr as entropy_fn
from collections import Counter, defaultdict
from statistics import mean, stdev
from pathlib import Path

# -----------------------------
# 1. Experiment Parameters
# -----------------------------
SAVE_PATH = '/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json'
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
np.random.seed(42)

VOCAB_SIZE = 1000       # Synthetic vocab size (keep tight for runtimes)
ZIPF_A = 1.1            # Zipf parameter: smaller = heavier head, >1 = lighter tail
MAX_DOCS_PER_SEED = 60  # Per-condition per-seed
N_SEEDS = 4
N_CONDITIONS = 4
TOKENS_PER_DOC = 400    # Document length
CONDITION_NAMES = ["temperature_1", "entropy_matched_temp", "top_k", "top_p"]
# For top-k and top-p, initial hyperparams (these will be tuned to match entropy)
TOP_K = 30
TOP_P = 0.9

# For AUC computation
NGRAM_RANGE = (2, 6)  # n-grams for self-overlap
BOOTSTRAP_TRIALS = 500  # For CI

# -----------------------------
# 2. Synthetic AR Model Utils
# -----------------------------

def zipf_probs(vocab_size=VOCAB_SIZE, a=ZIPF_A):
    """Return a Zipfian probability distribution over vocab."""
    ranks = np.arange(1, vocab_size + 1)
    probs = 1.0 / np.power(ranks, a)
    probs = probs / probs.sum()
    return probs

def perturb_distribution(base_probs, topic_shift=0.05, seed=None):
    """Slightly shift Zipf probs to simulate 'topics'."""
    # Pick a block of consecutive indices to 'boost'
    np_random = np.random.RandomState(seed)
    block = np_random.randint(0, len(base_probs) - 30)
    boosted = np.array(base_probs)
    boosted[block:block+30] *= (1 + topic_shift)
    boosted = boosted / boosted.sum()
    return boosted

def sample_sentence_boundary_token(vocab_size):
    # Let's dedicate the last token as EOS/SENT
    return vocab_size - 1

# -----------------------------
# 3. Decoding algorithms
# -----------------------------

def sample_from_probs(probs, temperature=1.0, random_state=None):
    """Full-support, temperature scaling."""
    scaled_logits = np.log(np.clip(probs, 1e-20, 1.0)) / temperature
    sampled_idx = random_state.choice(len(probs), p=softmax(scaled_logits))
    return sampled_idx

def sample_top_k(probs, k, temperature=1.0, random_state=None):
    """Truncation: sample only from top-k tokens, renormalize, applies temperature."""
    top_indices = np.argsort(probs)[-k:]
    top_logits = np.log(probs[top_indices] + 1e-20) / temperature
    local_probs = softmax(top_logits)
    sampled_subidx = random_state.choice(len(top_indices), p=local_probs)
    return top_indices[sampled_subidx]

def sample_top_p(probs, p, temperature=1.0, random_state=None):
    """Truncation: sample minimally many tokens summing >=p probability."""
    sorted_idxs = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idxs]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p)
    keep = sorted_idxs[:cutoff+1]
    trunc_logits = np.log(probs[keep] + 1e-20) / temperature
    local_probs = softmax(trunc_logits)
    sampled_subidx = random_state.choice(len(keep), p=local_probs)
    return keep[sampled_subidx]

# Entropy, for tuning global settings
def token_entropy(probs):
    return float(entropy_fn(probs, base=2).sum()) # bits

# Matched entropy for full-support temperature scaling to match Top-K/Top-P
def tune_temperature_for_entropy(target_entropy, base_probs, n_grid=40):
    lo, hi = 0.2, 3.0   # Only >0 (lower = peaky)
    best_temp, best_gap = None, float('inf')
    for t in np.linspace(lo, hi, n_grid):
        logits = np.log(base_probs + 1e-20) / t
        p = softmax(logits)
        ent = token_entropy(p)
        gap = abs(ent - target_entropy)
        if gap < best_gap: best_gap, best_temp = gap, t
    return best_temp

# -----------------------------
# 4. Self-Overlap Metric
# -----------------------------

def calc_self_overlap_aucs(seq, ngram_range=NGRAM_RANGE):
    """Return average self-overlap fraction vs n-gram size, and all points for AUROC-based stat."""
    L = len(seq)
    overlaps = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        seen = defaultdict(list)
        hits = 0
        for i in range(L - n + 1):
            ng = tuple(seq[i:i+n])
            seen[ng].append(i)
        # For overlap count, only consider those seen 2+ times
        hits += sum((len(poslist) - 1) for poslist in seen.values() if len(poslist) > 1)
        total = max(L - n + 1, 1)
        overlaps.append(hits / total)
    # AUC as average
    auc = float(np.mean(overlaps))
    return auc, overlaps

# Finer-grained error: how long is max repeated n-gram?
def longest_repeat(seq, n_max=15):
    L = len(seq)
    best = 0
    for n in range(2, min(n_max, L//2)+1):
        seen = set()
        for i in range(L - n + 1):
            ng = tuple(seq[i:i+n])
            if ng in seen:
                best = max(best, n)
                break
            seen.add(ng)
    return best

# -----------------------------
# 5. Main Synthetic Corpus Gen
# -----------------------------

def gen_document(
    base_probs, 
    topic_shifts, 
    decoding_fn, 
    decoding_args, 
    n_tokens, 
    vocab_size, 
    sentence_token,
    random_state
    ):
    """Autoregressive: at each token, rare topic switches (shifts in distribution)"""
    doc = []
    cur_probs = np.array(base_probs)
    for t in range(n_tokens):
        # topic shift every 30-90 tokens
        if t == 0 or (t % random_state.randint(30, 90) == 0):
            cur_probs = perturb_distribution(base_probs, topic_shift=topic_shifts, seed=random_state.randint(1e6))
        idx = decoding_fn(cur_probs, **decoding_args, random_state=random_state)
        doc.append(idx)
        # force sentence end at average 20 tokens
        if random_state.rand() < 1/20:
            doc.append(sentence_token)
    # Post-truncate
    return doc[:n_tokens]

# -----------------------------
# 6. Compose Conditions
# -----------------------------

def get_condition_params(cond_name, base_probs, vocab_size,
                        fixed_top_k=TOP_K, fixed_top_p=TOP_P, seed=None):
    """
    For any given base Zipf distribution, return decoding fn + parameters + avg entropy.
    For entropy-matched temp, tune temp to match entropy of top-k or top-p as needed.
    """
    np_random = np.random.RandomState(seed)
    if cond_name == "temperature_1":
        return sample_from_probs, {'temperature':1.0}, token_entropy(base_probs)
    elif cond_name == "top_k":
        # Compute effective entropy with top-k truncation
        top_inds = np.argsort(base_probs)[-fixed_top_k:]
        top_probs = base_probs[top_inds]
        top_probs = top_probs / top_probs.sum()
        ent_k = token_entropy(top_probs)
        return lambda p, temperature, random_state: sample_top_k(p, fixed_top_k, temperature, random_state), \
               {'temperature':1.0}, ent_k
    elif cond_name == "top_p":
        # Compute effective entropy with top-p truncation
        sorted_idxs = np.argsort(base_probs)[::-1]
        sorted_probs = base_probs[sorted_idxs]
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, fixed_top_p)
        keep = sorted_probs[:cutoff+1]
        keep = keep / keep.sum()
        ent_p = token_entropy(keep)
        return lambda p, temperature, random_state: sample_top_p(p, fixed_top_p, temperature, random_state), \
               {'temperature':1.0}, ent_p
    elif cond_name == "entropy_matched_temp":
        # Will be set by main loop, as we have to use the entropy target from k/p/baseline in each split
        # temp will be tuned per-seed
        temp_placeholder = 1.0
        return sample_from_probs, {'temperature':temp_placeholder}, float('nan')
    else:
        raise ValueError(f'Unknown condition: {cond_name}')

# -----------------------------
# 7. Full Generation Loop
# -----------------------------

def run_experiment():
    np.random.seed(42)
    results = []
    base_probs = zipf_probs(VOCAB_SIZE, ZIPF_A)
    sent_tok = sample_sentence_boundary_token(VOCAB_SIZE)
    topic_shift = 0.07
    meta_entropy_stats = defaultdict(list)
    per_condition_aucs = defaultdict(list)
    per_condition_repeats = defaultdict(list)
    per_doc_ngramcurves = defaultdict(list)
    per_seed_stats = defaultdict(list)

    for seed_ndx in range(N_SEEDS):
        print(f"\n===== Random Seed {seed_ndx+1}/{N_SEEDS} =====")
        np_random = np.random.RandomState(seed_ndx * 13 + 57)
        # Obtain decoding functions and entropy targets for conditions for this seed
        cond_info = {}
        # Determine effective entropy of baseline, top-k and top-p
        for cond_ix, cond_name in enumerate(CONDITION_NAMES):
            fn, args, ent = get_condition_params(cond_name, base_probs, VOCAB_SIZE, seed=seed_ndx*100+cond_ix)
            cond_info[cond_name] = {"fn": fn, "args": args, "base_entropy": ent}
        # For entropy-matched-temp, tune temp to match average top-k entropy for this seed
        k_entropy = cond_info["top_k"]["base_entropy"]
        p_entropy = cond_info["top_p"]["base_entropy"]
        avg_target_ent = 0.5 * (k_entropy + p_entropy)
        tuned_temp = tune_temperature_for_entropy(avg_target_ent, base_probs)
        cond_info["entropy_matched_temp"]["args"]["temperature"] = tuned_temp
        cond_info["entropy_matched_temp"]["base_entropy"] = token_entropy(
            softmax(np.log(base_probs+1e-20)/tuned_temp))

        print(f"Entropy values (seed={seed_ndx}):")
        for cname in CONDITION_NAMES:
            print(f"  {cname:22s}: entropy={cond_info[cname]['base_entropy']:.3f}")

        # MAIN loop: generate documents
        total_docs = 0
        for cond_ix, cond_name in enumerate(CONDITION_NAMES):
            n_docs = 0
            for doc_num in range(MAX_DOCS_PER_SEED):
                total_docs += 1
                n_docs += 1
                # Random seed for reproducible generation per doc (and avoid accidental copying)
                doc_seed = seed_ndx * 5000 + cond_ix * 1000 + doc_num * 37 + 915
                doc_random = np.random.RandomState(doc_seed)
                fn = cond_info[cond_name]["fn"]
                args = cond_info[cond_name]["args"]
                doc_tokens = gen_document(
                    base_probs, topic_shift, fn, args, TOKENS_PER_DOC, VOCAB_SIZE, sent_tok, doc_random)
                # Metrics
                auc, overlap_curve = calc_self_overlap_aucs(doc_tokens, NGRAM_RANGE)
                max_ngram = longest_repeat(doc_tokens)
                results.append({
                    "condition": cond_name,
                    "seed": seed_ndx,
                    "doc_id": f"{seed_ndx}_{cond_name}_{doc_num}",
                    "entropy_est": float(cond_info[cond_name]['base_entropy']),
                    "self_overlap_auc": float(auc),
                    "ngram_overlaps": [float(x) for x in overlap_curve],
                    "max_repeated_ngram": int(max_ngram),
                    "tokens": doc_tokens,
                })
                per_condition_aucs[cond_name].append(auc)
                per_condition_repeats[cond_name].append(max_ngram)
                per_doc_ngramcurves[cond_name].append(overlap_curve)
                if total_docs % 12 == 0:
                    print(f"  Processing doc {total_docs} / {N_SEEDS*MAX_DOCS_PER_SEED*N_CONDITIONS} (cond={cond_name}, seed={seed_ndx})...")

    print("\nFinished generation and analysis.")
    print(f"Number of results collected: {len(results)}")
    # Compact to minimal records for saving
    compacted = []
    for rec in results:
        out = {k: v for k, v in rec.items() if k != "tokens"}
        compacted.append(out)

    # 50+ valid points?
    if len(compacted) < 50:
        print("ERROR: Fewer than 50 valid data points. Aborting save.")
        return

    # Save JSON (cast all numpy values to float, for robustness, use default=str)
    with open(SAVE_PATH, 'w') as f:
        json.dump(compacted, f, indent=2, default=str)
    print(f"Results saved to {SAVE_PATH}")

    # -----------------------------
    # 8. Statistics + CI reporting
    # -----------------------------

    def bootstrap_diff(a, b, trials=BOOTSTRAP_TRIALS):
        """Bootstrap CI for mean(a) - mean(b)"""
        a, b = np.array(a), np.array(b)
        n_a, n_b = len(a), len(b)
        diffs = []
        rng = np.random.RandomState(254) # Reproducible
        for _ in range(trials):
            s_a = rng.choice(a, n_a)
            s_b = rng.choice(b, n_b)
            diffs.append(np.mean(s_a) - np.mean(s_b))
        return (np.mean(diffs), np.percentile(diffs, [2.5, 97.5]))

    print("\n==== Summary Table: Self-Overlap AUC ====")
    header = f"{'Condition':24s} | {'AUC mean':>7s} | {'AUC stdev':>7s} | {'MaxRep(mean)':>11s} | {'Entropy(mean)':>10s}"
    print(header)
    print('-'*len(header))
    final_stats = {}
    for cond in CONDITION_NAMES:
        mean_auc = np.mean(per_condition_aucs[cond])
        std_auc = np.std(per_condition_aucs[cond])
        meanrep = np.mean(per_condition_repeats[cond])
        entr = np.mean([rec["entropy_est"] for rec in results if rec["condition"]==cond])
        print(f"{cond:24s} | {mean_auc:.4f} | {std_auc:.4f} | {meanrep:11.3f} | {entr:10.3f}")
        final_stats[cond] = dict(mean_auc=mean_auc, std_auc=std_auc, mean_rep=meanrep, ent=entr)

    print("\n==== Entropy Matching Diagnostics ====")
    for t1, t2 in [("top_k","entropy_matched_temp"), ("top_p","entropy_matched_temp")]:
        diff = abs(final_stats[t1]["ent"] - final_stats[t2]["ent"])
        print(f"Mean entropy diff ({t1} vs {t2}): {diff:.4f} bits")

    # Bootstrap: Is top-k or top-p > entropy-matched?
    for trunc, base in [("top_k","entropy_matched_temp"), ("top_p","entropy_matched_temp")]:
        mean_diff, (lb, ub) = bootstrap_diff(per_condition_aucs[trunc], per_condition_aucs[base])
        pct = 100 * mean_diff/final_stats[base]["mean_auc"]
        print(f"\n{trunc} vs entropy_matched_temp: mean_diff={mean_diff:.4f} ({pct:+.1f}%), 95% CI=({lb:.4f}, {ub:.4f})")
        if lb > 0:
            print("  --> Statistically significant: truncation yields higher repetition.")
        else:
            print("  --> NOT significant.")

    # Plot N-gram overlap curves
    plt.figure(figsize=(8,6))
    ngram_sizes = list(range(NGRAM_RANGE[0], NGRAM_RANGE[1]+1))
    for cond in CONDITION_NAMES:
        curves = np.array(per_doc_ngramcurves[cond])
        means = curves.mean(axis=0)
        stds = curves.std(axis=0)
        plt.plot(ngram_sizes, means, label=cond)
        plt.fill_between(ngram_sizes, means-stds, means+stds, alpha=0.15)
    plt.xlabel("n-gram length")
    plt.ylabel("Self-overlap fraction")
    plt.title("Self-Overlap Curve Across Decoding Conditions")
    plt.legend()
    imgpath = os.path.splitext(SAVE_PATH)[0]+"_curve.png"
    plt.savefig(imgpath)
    print(f"\nPlot saved to {imgpath}")

if __name__ == "__main__":
    run_experiment()

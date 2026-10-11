#!/usr/bin/env python3
"""
Bootstrap confidence intervals for the ACSAC paper.

Run from: ~/agentsafety/ with ncib_env activated.
Inputs:   trustledger_v5_results/results/{model}_{benchmark}.json
Outputs:  prints LaTeX-ready strings to stdout, saves ci_results.json

Usage:
    python compute_paper_cis.py                    # full run
    python compute_paper_cis.py --inspect          # show JSON schema only
    python compute_paper_cis.py --n-boot 1000      # faster, less accurate
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_DIR = Path("trustledger_v5_results/results")
MODELS = ["haiku", "llama", "deepseek", "qwen"]
BENCHMARKS = ["injecagent", "tensortrust", "agentdojo", "asb", "trustbreach"]
N_BOOT_DEFAULT = 10000
SEED = 42

# Field-name candidates (script tries each; lets us handle schema variation)
ATTACK_LABEL_FIELDS = ["is_attack", "label", "is_malicious", "attack", "expected_block"]
L1_SCORE_FIELDS = ["l1_score", "iog_score", "trust_score", "l1"]
L2_SCORE_FIELDS = ["l2_score", "ips_score", "min_sim", "l2"]
L3_FLAG_FIELDS = ["l3_flagged", "l3_flag", "tccr_flagged", "tccr", "l3_detected"]
SCENARIO_LIST_FIELDS = ["scenarios", "results", "samples", "data", "detailed_results"]

# ============================================================
# UTILITIES
# ============================================================
rng = np.random.default_rng(SEED)


def find_field(d: dict, candidates: list):
    """Return the first matching field value from a dict."""
    for k in candidates:
        if k in d:
            return d[k]
    return None


def find_scenario_list(data):
    """Locate the list of per-scenario records inside the JSON."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in SCENARIO_LIST_FIELDS:
            if k in data and isinstance(data[k], list):
                return data[k]
        # Sometimes the file IS a dict of scenarios keyed by ID
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    return None


def load_results(model: str, benchmark: str):
    path = RESULTS_DIR / f"{model}_{benchmark}.json"
    if not path.exists():
        return None, f"missing: {path}"
    with open(path) as f:
        data = json.load(f)
    scenarios = find_scenario_list(data)
    if scenarios is None:
        return None, f"no scenario list found in {path}"
    return scenarios, None


# ============================================================
# BOOTSTRAP FUNCTIONS
# ============================================================
def bootstrap_rate(binary_values, n_boot, ci=95):
    """Bootstrap CI for the mean of a binary array (detection rate, FPR)."""
    x = np.asarray(binary_values, dtype=float)
    if len(x) == 0:
        return None, None, None, 0
    n = len(x)
    point = x.mean()
    boot = np.zeros(n_boot)
    for i in range(n_boot):
        boot[i] = rng.choice(x, size=n, replace=True).mean()
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return point, lo, hi, n


def bootstrap_auroc(scores_atk, scores_ben, n_boot, ci=95, lower_is_attack=True):
    """Bootstrap CI for AUROC.
    
    lower_is_attack=True matches L2/IPS convention: lower min_sim => more suspicious.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None, None, None, 0, 0

    if len(scores_atk) == 0 or len(scores_ben) == 0:
        return None, None, None, len(scores_atk), len(scores_ben)

    y = np.concatenate([np.ones(len(scores_atk)), np.zeros(len(scores_ben))])
    s = np.concatenate([scores_atk, scores_ben]).astype(float)
    if lower_is_attack:
        s = -s  # invert so attacks rank higher

    n = len(y)
    point = roc_auc_score(y, s)
    boot = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            boot[i] = 0.5  # degenerate resample
            continue
        boot[i] = roc_auc_score(y[idx], s[idx])
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return point, lo, hi, len(scores_atk), len(scores_ben)


def wilson_ci(successes, n, ci=95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - ci / 100) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


# ============================================================
# INSPECTION MODE
# ============================================================
def inspect_schema():
    print("=" * 70)
    print(" SCHEMA INSPECTION")
    print("=" * 70)
    for model in MODELS:
        for bench in BENCHMARKS:
            path = RESULTS_DIR / f"{model}_{bench}.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            print(f"\n {path}")
            print(f"   top-level type: {type(data).__name__}")
            if isinstance(data, dict):
                print(f"   top-level keys: {list(data.keys())[:10]}")
            scenarios = find_scenario_list(data)
            if scenarios:
                print(f"   scenario count: {len(scenarios)}")
                if scenarios:
                    print(f"   first scenario keys: {list(scenarios[0].keys())[:15]}")
                    print(f"   first scenario sample: {json.dumps(scenarios[0], indent=2)[:500]}")
            else:
                print(f"   could not locate scenario list")
            print()
            return  # just show one file


# ============================================================
# MAIN ANALYSIS
# ============================================================
def main(n_boot):
    print("=" * 70)
    print(" BOOTSTRAP CONFIDENCE INTERVALS — ACSAC PAPER")
    print(f" n_boot = {n_boot}, seed = {SEED}")
    print("=" * 70)

    output = {"n_boot": n_boot, "seed": SEED, "results": {}}

    # --------------------------------------------------------
    # 1. L3 detection per model (pooled across benchmarks)
    # --------------------------------------------------------
    print("\n--- L3 Detection Rates (pooled across benchmarks, attacks only) ---")
    l3_per_model = defaultdict(list)
    for model in MODELS:
        for bench in BENCHMARKS:
            scenarios, err = load_results(model, bench)
            if scenarios is None:
                print(f"  [skip] {model}/{bench}: {err}")
                continue
            for s in scenarios:
                if not isinstance(s, dict):
                    continue
                is_atk = find_field(s, ATTACK_LABEL_FIELDS)
                l3 = find_field(s, L3_FLAG_FIELDS)
                if is_atk and l3 is not None:
                    l3_per_model[model].append(int(bool(l3)))

    output["results"]["l3_per_model"] = {}
    for model in MODELS:
        flags = l3_per_model[model]
        if not flags:
            print(f"  {model:10s}: NO DATA")
            continue
        p, lo, hi, n = bootstrap_rate(flags, n_boot)
        print(f"  {model:10s}: {p*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]  (n={n})")
        output["results"]["l3_per_model"][model] = {
            "point": p, "lo": lo, "hi": hi, "n": n
        }

    # LaTeX-ready string for paper Section VI.D
    print("\n  LaTeX-ready line for Section VI.D:")
    bits = []
    name_map = {"haiku": "Haiku", "llama": "Llama", "deepseek": "DeepSeek", "qwen": "Qwen"}
    for model in MODELS:
        if model in output["results"]["l3_per_model"]:
            r = output["results"]["l3_per_model"][model]
            bits.append(f"{name_map[model]} {r['point']*100:.1f}\\% [95\\% CI {r['lo']*100:.1f}--{r['hi']*100:.1f}]")
    print("   ", ", ".join(bits))

    # --------------------------------------------------------
    # 2. L2 AUROC per (model, benchmark)
    # --------------------------------------------------------
    print("\n\n--- L2 AUROC CIs per (model, benchmark) ---")
    output["results"]["l2_auroc"] = {}
    for model in MODELS:
        print(f"\n  {model}:")
        for bench in BENCHMARKS:
            scenarios, err = load_results(model, bench)
            if scenarios is None:
                continue
            s_atk, s_ben = [], []
            for s in scenarios:
                if not isinstance(s, dict):
                    continue
                score = find_field(s, L2_SCORE_FIELDS)
                if score is None:
                    continue
                if find_field(s, ATTACK_LABEL_FIELDS):
                    s_atk.append(score)
                else:
                    s_ben.append(score)
            if s_atk and s_ben:
                p, lo, hi, na, nb = bootstrap_auroc(s_atk, s_ben, n_boot)
                if p is not None:
                    print(f"    {bench:15s}: AUROC {p:.3f} [{lo:.3f}, {hi:.3f}]  (n_atk={na}, n_ben={nb})")
                    output["results"]["l2_auroc"][f"{model}_{bench}"] = {
                        "point": p, "lo": lo, "hi": hi, "n_atk": na, "n_ben": nb
                    }
            else:
                print(f"    {bench:15s}: missing scores (atk={len(s_atk)}, ben={len(s_ben)})")

    # --------------------------------------------------------
    # 3. Wilson CIs for small-n FPRs
    # --------------------------------------------------------
    print("\n\n--- Wilson CIs for small-n benign FPR ---")
    print("  (these are the FPR claims most vulnerable to reviewer pushback)")
    cases = [
        ("InjecAgent", 4, [0, 1]),     # 0/4 = 0%, 1/4 = 25%
        ("TrustBreach", 8, [0, 1, 2]), # 0/8, 12.5%, 25%
        ("ASB", 20, [1]),              # 5% reported
    ]
    output["results"]["wilson_ci"] = {}
    for name, n_ben, fp_counts in cases:
        for fp in fp_counts:
            p, lo, hi = wilson_ci(fp, n_ben)
            label = f"{name} {fp}/{n_ben}"
            print(f"  {label:25s}: {p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")
            output["results"]["wilson_ci"][label] = {
                "point": p, "lo": lo, "hi": hi, "n": n_ben, "fp": fp
            }

    # --------------------------------------------------------
    # 4. Save full results
    # --------------------------------------------------------
    with open("ci_results.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    print("\n  Saved full results to: ci_results.json")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                        help="Show JSON schema of one result file and exit")
    parser.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT,
                        help=f"Bootstrap resamples (default {N_BOOT_DEFAULT})")
    args = parser.parse_args()

    if args.inspect:
        inspect_schema()
        sys.exit(0)

    main(args.n_boot)

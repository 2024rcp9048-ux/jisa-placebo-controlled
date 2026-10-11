#!/usr/bin/env python3
"""
confirmatory_analysis.py -- Phase B test of the pre-registration. Run from ~/mob.
Pure analysis on existing result files; no Bedrock.

Implements prereg Section 6:
  PRIMARY  : within- vs between-construct correlation, label-permutation test
  per-pair : Spearman + permutation p (exact if n<=8 else sampled)
  splits   : pilot-only / new-only / pooled  (double-dipping safeguard)
  variance : descriptive model/benchmark/interaction decomposition (NOT independent
             evidence -- it re-encodes the correlations; reported per prereg note)
  falsify  : Section 7 criteria printed explicitly

SELF-GATES (refuses to over-claim):
  1. reproduction gate on pilot AgentDojo (corrected kimi=0.073)
  2. design-completeness gate: PRIMARY H1 test runs only when >=2 constructs have
     >=2 benchmarks AND a within-construct pair exists that is NOT the pilot
     hypothesis pair. Otherwise it prints the gap and stops before H1.
"""
import json, os, sys, random, itertools
from collections import Counter

# ============================ FROZEN DESIGN (edit as you acquire data) =========
PILOT_MODELS = ["deepseek_v3", "qwen_32b", "llama_scout", "mistral_large3", "kimi_k2", "qwen3_next"]
NEW_MODELS = []          # <-- expansion models (vulnerability-INDEPENDENT frozen list)
HYPOTHESIS_PAIR = frozenset(("InjecAgent", "TT-hijack"))   # the pilot pair that generated H1

MASTER = {"deepseek_v3", "qwen_32b", "llama_scout"}
SUITES = ["banking", "slack", "travel", "workspace"]
KNOWN_AJ = {"deepseek_v3": 0.743, "qwen_32b": 0.442, "llama_scout": 0.199,
            "mistral_large3": 0.684, "kimi_k2": 0.073, "qwen3_next": 0.499}

# ---- loaders: each returns undefended ASR for one model ----
def load_agentdojo(AJ, m):
    xs = []
    for s in SUITES:
        if m in MASTER:
            c = AJ[f"{m}_{s}_undefended_static"]; xs.append(c["n_sec"] / c["n"])
        else:
            j = json.load(open(f"msm_b2_{m}_{s}_undefended_static/summary.json"))
            xs.append(j["attack_successes"] / j["total_scenarios"])
    return sum(xs) / 4

def load_injecagent(AJ, m):
    j = json.load(open(f"InjecAgent/results_bedrock/{m}_undefended_base/result.json"))
    s = sum(j[k]["succ"] for k in ("dh", "ds")); v = sum(j[k]["succ"] + j[k]["unsucc"] for k in ("dh", "ds"))
    return s / v if v else float("nan")

def _tt(m, sub):
    j = json.load(open(f"tensortrust_results/{m}_undefended_base/result.json"))[sub]
    v = j["succ"] + j["unsucc"]; return j["succ"] / v if v else float("nan")
def load_tt_hijack(AJ, m):  return _tt(m, "hijacking")
def load_tt_extract(AJ, m): return _tt(m, "extraction")

def load_generic(path_tmpl, top_key, succ="succ", unsucc="unsucc"):
    """Hook for a NEW deterministic-scored benchmark. path_tmpl uses {model};
    valid-only ASR from json[top_key][succ]/(succ+unsucc). Edit to fit the file."""
    def f(AJ, m):
        j = json.load(open(path_tmpl.format(model=m)))[top_key]
        v = j[succ] + j[unsucc]; return j[succ] / v if v else float("nan")
    return f

# benchmark -> (construct, loader).  Construct axis = STRUCTURE (committed).
BENCHMARKS = {
    "Raccoon": ("C2_extraction", load_generic("raccoon_results/{model}_undefended_base/result.json", "raccoon")),
    "AgentDojo":  ("C3_multistep_agentic", load_agentdojo),
    "InjecAgent": ("C1_singleturn_exec",   load_injecagent),
    "TT-hijack":  ("C1_singleturn_exec",   load_tt_hijack),
    "TT-extract": ("C2_extraction",        load_tt_extract),
    # ---- ADD when acquired (deterministic scoring only, no LLM judge): ----
    # "ExtractB":  ("C2_extraction",        load_generic("extractb_results/{model}_undefended/result.json", "extract")),
    # "AgenticB":  ("C3_multistep_agentic", load_generic("agenticb_results/{model}_undefended/result.json", "attack")),
}
# ==============================================================================

def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        a = (i + j) / 2.0 + 1
        for k in range(i, j + 1): r[o[k]] = a
        i = j + 1
    return r

def pear(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5; vb = sum((x - mb) ** 2 for x in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0

def spear(a, b): return pear(rank(a), rank(b))

def perm_p(a, b, cap=40320):
    obs = abs(spear(a, b)); ra = rank(a); rb = rank(b); n = len(b)
    import math
    if math.factorial(n) <= cap:
        cnt = tot = 0
        for p in itertools.permutations(range(n)):
            if abs(pear(ra, [rb[i] for i in p])) >= obs - 1e-9: cnt += 1
            tot += 1
        return cnt / tot
    cnt = 0; T = 20000
    for _ in range(T):
        random.shuffle(rb)
        if abs(pear(ra, rb)) >= obs - 1e-9: cnt += 1
    return cnt / T

def mean(x): return sum(x) / len(x) if x else float("nan")

def primary_test(vecs, constr, n_perm=10000, seed=0):
    """within - between mean Spearman, label-permutation null."""
    benches = list(vecs)
    pairs = [(benches[i], benches[j]) for i in range(len(benches)) for j in range(i + 1, len(benches))]
    rho = {p: spear(vecs[p[0]], vecs[p[1]]) for p in pairs}
    def split(cmap):
        s = [rho[p] for p in pairs if cmap[p[0]] == cmap[p[1]]]
        d = [rho[p] for p in pairs if cmap[p[0]] != cmap[p[1]]]
        return s, d
    s0, d0 = split(constr)
    if not s0 or not d0: return None
    T = mean(s0) - mean(d0)
    labels = [constr[b] for b in benches]; random.seed(seed); cnt = 0; valid = 0
    for _ in range(n_perm):
        sh = labels[:]; random.shuffle(sh)
        cm = {benches[i]: sh[i] for i in range(len(benches))}
        s, d = split(cm)
        if s and d:
            valid += 1
            if (mean(s) - mean(d)) >= T - 1e-12: cnt += 1
    return T, mean(s0), mean(d0), s0, d0, rho, pairs, (cnt / valid if valid else float("nan"))

def variance_decomp(vecs, models):
    """descriptive only: z-score each benchmark over models, two-way (model x bench)
    decomposition. interaction% = ranking-disagreement signal. NOT independent of
    the correlations -- re-encoding, per prereg."""
    benches = list(vecs)
    Z = {}
    for b in benches:
        col = [vecs[b][i] for i in range(len(models))]
        mu = mean(col); sd = (sum((x - mu) ** 2 for x in col) / len(col)) ** 0.5 or 1.0
        Z[b] = [(x - mu) / sd for x in col]
    cells = [Z[b][i] for b in benches for i in range(len(models))]
    grand = mean(cells)
    ss_tot = sum((x - grand) ** 2 for x in cells)
    row_means = [mean([Z[b][i] for b in benches]) for i in range(len(models))]
    ss_model = len(benches) * sum((row_means[i] - grand) ** 2 for i in range(len(models)))
    ss_bench = len(models) * sum((mean(Z[b]) - grand) ** 2 for b in benches)  # ~0 after z
    ss_int = ss_tot - ss_model - ss_bench
    return ss_model / ss_tot, ss_bench / ss_tot, ss_int / ss_tot

def build(models):
    AJ = json.load(open("paper1_MASTER_results.json"))["b2_cells"]
    return {b: {m: ld(AJ, m) for m in models} for b, (c, ld) in BENCHMARKS.items()}, AJ

def vecs_for(asr, models):
    return {b: [asr[b][m] for m in models] for b in asr}

def run_split(name, models, constr):
    if not models:
        print(f"\n[{name}] no models -- skipped."); return
    asr = {b: {m: BENCHMARKS[b][1](json.load(open('paper1_MASTER_results.json'))['b2_cells'], m)
               for m in models} for b in BENCHMARKS}
    V = vecs_for(asr, models)
    res = primary_test(V, {b: BENCHMARKS[b][0] for b in BENCHMARKS})
    print(f"\n[{name}]  n_models={len(models)}")
    if res is None:
        print("  PRIMARY not computable (need both within- and between-construct pairs)."); return
    T, mw, mb, sw, sb, rho, pairs, p = res
    print(f"  within-construct mean rho = {mw:+.3f}  (pairs={len(sw)})")
    print(f"  between-construct mean rho = {mb:+.3f}  (pairs={len(sb)})")
    print(f"  T = within-between = {T:+.3f}   label-permutation p = {p:.4f}"
          f"   {'<-- H1 supported' if p < 0.05 and T > 0 else ''}")
    vm, vb_, vi = variance_decomp(V, models)
    print(f"  variance (z-scored, descriptive): model={vm:.0%}  interaction(ranking-disagreement)={vi:.0%}")

def main():
    # reproduction gate
    AJ = json.load(open("paper1_MASTER_results.json"))["b2_cells"]
    for m in PILOT_MODELS:
        if abs(load_agentdojo(AJ, m) - KNOWN_AJ[m]) > 0.006:
            print(f"REPRODUCTION FAIL: AgentDojo {m}"); sys.exit(1)
    print("Reproduction gate passed (pilot AgentDojo, corrected kimi=0.073).")

    constr = {b: BENCHMARKS[b][0] for b in BENCHMARKS}
    per_construct = Counter(constr.values())
    print("\nDesign:")
    for c, n in per_construct.items():
        print(f"  {c:24s} {n} benchmark(s)  {'OK' if n >= 2 else '<-- need >=2'}")

    # design-completeness gate
    ok_constructs = [c for c, n in per_construct.items() if n >= 2]
    benches = list(BENCHMARKS)
    within_pairs = [frozenset((benches[i], benches[j]))
                    for i in range(len(benches)) for j in range(i + 1, len(benches))
                    if constr[benches[i]] == constr[benches[j]]]
    nonpilot_within = [p for p in within_pairs if p != HYPOTHESIS_PAIR]
    if len(ok_constructs) < 2 or not nonpilot_within:
        print("\n" + "=" * 70)
        print("DESIGN INCOMPLETE -- PRIMARY H1 TEST NOT RUN (prereg gate).")
        print("=" * 70)
        if len(ok_constructs) < 2:
            print(f"  need >=2 constructs with >=2 benchmarks; have {len(ok_constructs)}.")
        if not nonpilot_within:
            print("  the only within-construct pair is the pilot hypothesis pair")
            print("  (InjecAgent,TT-hijack) -- a confirmatory test would double-dip.")
        print("  -> acquire a 2nd C2 (extraction) and a 2nd C3 (agentic) benchmark,")
        print("     deterministic-scored, register them in BENCHMARKS, then re-run.")
        # still show current per-pair correlations descriptively
        V = vecs_for(build(PILOT_MODELS)[0], PILOT_MODELS)
        print("\n  current per-pair Spearman (pilot, descriptive only):")
        for i in range(len(benches)):
            for j in range(i + 1, len(benches)):
                a, b = benches[i], benches[j]
                tag = "within" if constr[a] == constr[b] else "between"
                print(f"    {a:11s}<->{b:11s} rho={spear(V[a],V[b]):+.3f} p={perm_p(V[a],V[b]):.3f} [{tag}]")
        return

    # design complete -> run the three splits (double-dipping safeguard)
    print("\n" + "=" * 70)
    print("PRIMARY TEST (H1): within > between construct correlation")
    print("=" * 70)
    run_split("PILOT-ONLY (hypothesis-generating; not confirmatory)", PILOT_MODELS, constr)
    run_split("NEW-ONLY (CONFIRMATORY)", NEW_MODELS, constr)
    run_split("POOLED (reported for completeness)", PILOT_MODELS + NEW_MODELS, constr)

    # falsification verdict on the confirmatory (new-only) split
    print("\n" + "=" * 70)
    print("FALSIFICATION (prereg Section 7, judged on NEW-ONLY split)")
    print("=" * 70)
    print("  H1 refuted if T<=0 or permutation p>=0.05 on NEW-ONLY, or if any")
    print("  between-construct pair is the single highest correlation.")
    print("  (read the NEW-ONLY block above against these criteria.)")

if __name__ == "__main__":
    main()

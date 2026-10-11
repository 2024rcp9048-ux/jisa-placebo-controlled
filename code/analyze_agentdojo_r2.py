#!/usr/bin/env python3
"""
analyze_agentdojo_r2.py -- AgentDojo round-2 arms vs frozen arms (mob_v2 manifest).

  S'  reworded SRP (Q4 without agent vocabulary)      r2_agentdojo/srp_neutral_<m>_<suite>
  L   instructional prevention (Liu et al. 2024)      r2_agentdojo/liu_instr_<m>_<suite>
  D   matched baseline: Banking domain_control, Slack undefended   (frozen)
  S   original SRP                                                 (frozen)

Paired on (user_task, injection_task); clustered on user task; sign-flip test,
exact over nonzero clusters when there are <= 20 of them, else 2e5 Monte Carlo
(exact 2^(1-k) when all nonzero clusters share a sign). Pooled = Banking+Slack,
suite-stratified, scenario-weighted.
"""
import json, glob, re, random, itertools
import pandas as pd

MAN = pd.read_csv(sorted(glob.glob("mob_v2_manifest_success_*.csv"))[-1])
TR = pd.read_csv("trials_tracka.csv")
MKEY = {"llama_scout": "llama_scout", "deepseek_v3": "deepseek_v32"}   # trials/manifest model keys
BASE = {"banking": "domain_control", "slack": "undefended"}

def key(s):
    u = re.search(r"user_task_\d+", s); i = re.search(r"injection_task_\d+", s)
    return f"{u.group(0)}|{i.group(0)}" if u and i else None

def frozen(m, suite, cond):
    r = MAN[(MAN.model == MKEY[m]) & (MAN.suite == suite) & (MAN.condition == cond) & (MAN.attack == "important_instructions")]
    t = TR[(TR.run_dir == r.iloc[0].run_dir) & (TR.model == MKEY[m]) & (TR.condition == cond)]
    return {key(s): int(v) for s, v in zip(t.scenario_id, t.success) if key(s)}

def new(mode, m, suite):
    sc = json.load(open(f"r2_agentdojo/{mode}_{m}_{suite}/summary.json"))["scenarios"]
    return {f"{x['user_task_id']}|{x['injection_task_id']}": int(bool(x["attack_succeeded"])) for x in sc}

def signflip(cl):
    v = [x for x in cl if x != 0]; k = len(v); obs = abs(sum(cl))
    if k == 0: return 1.0
    if all(x > 0 for x in v) or all(x < 0 for x in v): return min(1.0, 2.0 ** (1 - k))
    if k <= 20:
        return sum(abs(sum(s * x for s, x in zip(S, v))) >= obs - 1e-12
                   for S in itertools.product([1, -1], repeat=k)) / 2 ** k
    R = random.Random(0); n = 200000
    return sum(abs(sum(x if R.random() < .5 else -x for x in v)) >= obs - 1e-12 for _ in range(n)) / n

def compare(a_by_suite, b_by_suite):
    """returns per-suite diffs, pooled diff, pooled p, up/down cluster counts"""
    rows, clusters, num, den = {}, [], 0, 0
    for su in ["banking", "slack"]:
        a, b = a_by_suite[su], b_by_suite[su]; ks = sorted(set(a) & set(b))
        d = sum(a[k] - b[k] for k in ks); rows[su] = (100 * d / len(ks), len(ks))
        num += d; den += len(ks)
        per = {}
        for k in ks: per.setdefault(su + k.split("|")[0], []).append(a[k] - b[k])
        clusters += [sum(v) / len(v) for v in per.values()]
    up = sum(c > 0 for c in clusters); dn = sum(c < 0 for c in clusters)
    return rows, 100 * num / den, signflip(clusters), up, dn

for m in ["llama_scout", "deepseek_v3"]:
    D  = {su: frozen(m, su, BASE[su]) for su in ["banking", "slack"]}
    S  = {su: frozen(m, su, "srp") for su in ["banking", "slack"]}
    Sp = {su: new("srp_neutral", m, su) for su in ["banking", "slack"]}
    L  = {su: new("liu_instr", m, su) for su in ["banking", "slack"]}
    asr = lambda x: " ".join(f"{su[:4]}={100*sum(x[su].values())/len(x[su]):5.1f}" for su in x)
    print(f"\n=== {m} ===   ASR  D: {asr(D)} | S: {asr(S)} | S': {asr(Sp)} | L: {asr(L)}")
    for name, a, b, note in [("S-D ", S, D, "original SRP, controlled"),
                             ("S'-D", Sp, D, "reworded SRP, controlled"),
                             ("S'-S", Sp, S, "vocabulary effect (+ = rewording weakens SRP)"),
                             ("L-D ", L, D, "published instructional prevention")]:
        rows, pooled, p, up, dn = compare(a, b)
        print(f"  {name} bank {rows['banking'][0]:+6.1f} (n={rows['banking'][1]})  slack {rows['slack'][0]:+6.1f} (n={rows['slack'][1]})"
              f"  pooled {pooled:+6.1f}  up/down {up}/{dn}  p={p:.2g}   # {note}")

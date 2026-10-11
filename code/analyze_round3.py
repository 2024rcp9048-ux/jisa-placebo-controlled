#!/usr/bin/env python3
"""
analyze_round3.py -- round-3 arms, all six models.

BIPIA email (clustered on attacks, 5,000 bootstrap resamples):
  Nsec  security-themed non-verificational placebo (500):  Nsec-U, S-Nsec (vs S-N)
  C     SRP content without block structure (500):         C-U, S-C   (reverse order: U->C->S)
  L     instructional prevention, Liu et al. (1,000):      L-B
  W     sandwich prevention, Liu et al. (1,000):           W-B
AgentDojo Banking+Slack (task-clustered sign-flip, pooled, scenario-weighted):
  Nsec-D, S-Nsec, C-D, S-C, L-D (4 new models), SRP repeat vs frozen SRP (A/A flips).
"""
import json, glob, re, random, itertools, os, sys
import pandas as pd
from sample_plan import scenario_key as K

M6 = ["haiku", "sonnet4", "llama_scout", "deepseek_v3", "qwen_32b", "mistral_large3"]
MKEY = {"haiku": "haiku_45", "sonnet4": "sonnet_46", "llama_scout": "llama_scout",
        "deepseek_v3": "deepseek_v32", "qwen_32b": "qwen3_32b", "mistral_large3": "mistral_large_3"}
SUITES = ["banking", "slack"]; GRID = {"banking": 144, "slack": 105}

# ------------------------------------------------------------- completeness
print("== Completeness")
bad = []
for m in M6:
    for mode, n in [("sec_placebo", 500), ("srp_prose", 500), ("liu_instr", 1000), ("sandwich", 1000)]:
        p = f"r3_bipia/{mode}_{m}/per_scenario.jsonl"
        c = sum(1 for _ in open(p)) if os.path.exists(p) else 0
        if c != n: bad.append(f"{p}: {c}/{n}")
    for su in SUITES:
        for d in [f"r3_agentdojo/sec_placebo_{m}_{su}", f"r3_agentdojo/srp_prose_{m}_{su}", f"r3_repeat/srp_{m}_{su}"] + \
                 ([f"r3_agentdojo/liu_instr_{m}_{su}"] if m not in ("llama_scout", "deepseek_v3") else []):
            p = f"{d}/summary.json"
            c = len(json.load(open(p))["scenarios"]) if os.path.exists(p) else 0
            if c != GRID[su]: bad.append(f"{p}: {c}/{GRID[su]}")
print("  all complete" if not bad else "  INCOMPLETE:\n   " + "\n   ".join(bad))
if bad and "--force" not in sys.argv: sys.exit("Fix/re-run incomplete cells first (or pass --force).")

# ------------------------------------------------------------- BIPIA
plan = json.load(open("sample_plan.json"))
P1000 = set(plan["bipia/email"]["scenario_keys"]); P500 = set(plan["bipia/email_n500"]["scenario_keys"])
def bload(d, P):
    o = {}
    for l in open(f"{d}/per_scenario.jsonl"):
        r = json.loads(l); u = r.get("scenario_uid") or K(r, "bipia")
        if u in P and u not in o: o[u] = r
    return o
def find(p): return sorted(x for x in glob.glob(p) if "limit" not in x)[-1]
def bdiff(a, b):
    ks = sorted(set(a) & set(b)); cl = {}
    for k in ks:
        cl.setdefault(a[k].get("attack_str") or a[k].get("attack_name"), []).append(
            int(bool(a[k]["attack_succeeded"])) - int(bool(b[k]["attack_succeeded"])))
    est = 100 * sum(map(sum, cl.values())) / len(ks); A = list(cl); R = random.Random(0); bs = []
    for _ in range(5000):
        p = [cl[R.choice(A)] for _ in A]; bs.append(100 * sum(map(sum, p)) / sum(map(len, p)))
    bs.sort(); return f"{est:+5.1f} [{bs[125]:+5.1f},{bs[4874]:+5.1f}]"

print("\n== BIPIA email")
for m in M6:
    U = bload(f"msm_exp8_{m}_email_none", P500); B = bload(f"msm_exp8_{m}_email_none", P1000)
    N = bload(find(f"msm_exp8_{m}_email_neutral_preamble_2026092?"), P500)
    S = bload(find(f"msm_exp8_{m}_email_srp*"), P500)
    Ns = bload(f"r3_bipia/sec_placebo_{m}", P500); C = bload(f"r3_bipia/srp_prose_{m}", P500)
    L = bload(f"r3_bipia/liu_instr_{m}", P1000); W = bload(f"r3_bipia/sandwich_{m}", P1000)
    print(f"  {m}")
    print(f"    placebos   N-U {bdiff(N,U)}   Nsec-U {bdiff(Ns,U)}   Nsec-N {bdiff(Ns,N)}")
    print(f"    content    S-N {bdiff(S,N)}   S-Nsec {bdiff(S,Ns)}")
    print(f"    order      C-U {bdiff(C,U)}   S-C {bdiff(S,C)}     (U->C->S vs U->N->S)")
    print(f"    published  L-B {bdiff(L,B)}   W-B {bdiff(W,B)}")

# ------------------------------------------------------------- AgentDojo
MAN = pd.read_csv(sorted(glob.glob("mob_v2_manifest_success_*.csv"))[-1]); TR = pd.read_csv("trials_tracka.csv")
BASE = {"banking": "domain_control", "slack": "undefended"}
def key(s):
    u = re.search(r"user_task_\d+", s); i = re.search(r"injection_task_\d+", s)
    return f"{u.group(0)}|{i.group(0)}" if u and i else None
def frozen(m, su, c):
    r = MAN[(MAN.model == MKEY[m]) & (MAN.suite == su) & (MAN.condition == c) & (MAN.attack == "important_instructions")]
    t = TR[(TR.run_dir == r.iloc[0].run_dir) & (TR.model == MKEY[m]) & (TR.condition == c)]
    return {key(s): int(v) for s, v in zip(t.scenario_id, t.success) if key(s)}
def new(d):
    return {f"{x['user_task_id']}|{x['injection_task_id']}": int(bool(x["attack_succeeded"]))
            for x in json.load(open(f"{d}/summary.json"))["scenarios"]}
def signflip(cl):
    v = [x for x in cl if x != 0]; k = len(v); obs = abs(sum(cl))
    if k == 0: return 1.0
    if all(x > 0 for x in v) or all(x < 0 for x in v): return min(1.0, 2.0 ** (1 - k))
    if k <= 20: return sum(abs(sum(s * x for s, x in zip(S_, v))) >= obs - 1e-12 for S_ in itertools.product([1, -1], repeat=k)) / 2 ** k
    R = random.Random(0); n = 200000
    return sum(abs(sum(x if R.random() < .5 else -x for x in v)) >= obs - 1e-12 for _ in range(n)) / n
def adiff(a, b):
    cl, num, den, per_suite = [], 0, 0, []
    for su in SUITES:
        ks = sorted(set(a[su]) & set(b[su])); d = sum(a[su][k] - b[su][k] for k in ks)
        per_suite.append(f"{100*d/len(ks):+6.1f}"); num += d; den += len(ks); g = {}
        for k in ks: g.setdefault(su + k.split("|")[0], []).append(a[su][k] - b[su][k])
        cl += [sum(x) / len(x) for x in g.values()]
    up = sum(c > 0 for c in cl); dn = sum(c < 0 for c in cl)
    return f"bank {per_suite[0]} slack {per_suite[1]} pooled {100*num/den:+6.1f} ({up}/{dn}, p={signflip(cl):.2g})"

print("\n== AgentDojo (Banking + Slack)")
for m in M6:
    D = {su: frozen(m, su, BASE[su]) for su in SUITES}; N = {su: frozen(m, su, "neutral_preamble") for su in SUITES}
    S = {su: frozen(m, su, "srp") for su in SUITES}
    Ns = {su: new(f"r3_agentdojo/sec_placebo_{m}_{su}") for su in SUITES}
    C = {su: new(f"r3_agentdojo/srp_prose_{m}_{su}") for su in SUITES}
    S2 = {su: new(f"r3_repeat/srp_{m}_{su}") for su in SUITES}
    print(f"  {m}")
    print(f"    Nsec-D  {adiff(Ns,D)}")
    print(f"    N-D     {adiff(N,D)}")
    print(f"    S-Nsec  {adiff(S,Ns)}")
    print(f"    C-D     {adiff(C,D)}")
    print(f"    S-C     {adiff(S,C)}")
    ldir = f"r2_agentdojo/liu_instr_{m}" if m in ("llama_scout", "deepseek_v3") else f"r3_agentdojo/liu_instr_{m}"
    L = {su: new(f"{ldir}_{su}") for su in SUITES}
    print(f"    L-D     {adiff(L,D)}")
    flips = sum(1 for su in SUITES for k in set(S[su]) & set(S2[su]) if S[su][k] != S2[su][k])
    n = sum(len(set(S[su]) & set(S2[su])) for su in SUITES)
    print(f"    A/A SRP repeat: flips {flips}/{n} ({100*flips/n:.1f}%)  net {adiff(S2,S)}")

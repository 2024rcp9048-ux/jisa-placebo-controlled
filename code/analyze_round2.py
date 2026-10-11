#!/usr/bin/env python3
"""
analyze_round2.py -- pair the round-2 arms with the frozen arms (same planned-sample
loader as the paper; first occurrence wins) and report paired differences in ASR with
95% cluster-bootstrap CIs over injected attacks (5,000 resamples).

  R2  framing-only F (1,000 pairs):  F-B, compared with P-M (framing given marks)
  R3  domain-neutral SRP S' (500):   S'-U, S-U, S'-S, S'-N
  R4  second placebo N2 (500):       N2-U vs N-U, S-N2 vs S-N
"""
import json, glob, random
from sample_plan import scenario_key as K

P1000 = set(json.load(open("sample_plan.json"))["bipia/email"]["scenario_keys"])
P500  = set(json.load(open("sample_plan.json"))["bipia/email_n500"]["scenario_keys"])

def load(d, plan):
    o = {}
    for l in open(f"{d}/per_scenario.jsonl"):
        r = json.loads(l); u = r.get("scenario_uid") or K(r, "bipia")
        if u in plan and u not in o: o[u] = r
    return o

def find(pattern):
    c = sorted(x for x in glob.glob(pattern) if "limit" not in x and "_r2_" not in x)
    if not c: raise FileNotFoundError(pattern)
    return c[-1]

def diff(a, b):
    """a - b, paired on shared keys, clustered on attack string."""
    ks = sorted(set(a) & set(b)); cl = {}
    for k in ks:
        cl.setdefault(a[k].get("attack_str") or a[k].get("attack_name"), []).append(
            int(bool(a[k]["attack_succeeded"])) - int(bool(b[k]["attack_succeeded"])))
    est = 100 * sum(map(sum, cl.values())) / len(ks)
    A = list(cl); R = random.Random(0); bs = []
    for _ in range(5000):
        p = [cl[R.choice(A)] for _ in A]; bs.append(100 * sum(map(sum, p)) / sum(map(len, p)))
    bs.sort()
    return f"{est:+6.1f} [{bs[125]:+5.1f},{bs[4874]:+5.1f}] n={len(ks)}"

def asr(x): return f"{100*sum(bool(r['attack_succeeded']) for r in x.values())/len(x):5.1f}"

MODELS = ["haiku", "sonnet4", "llama_scout", "deepseek_v3", "qwen_32b", "mistral_large3"]

print("== R2. Framing-only F vs base B (1,000 pairs); compare with P-M (framing given marks)")
for m in MODELS:
    B = load(f"msm_exp8_{m}_email_none", P1000)
    F = load(f"r2_framing/{m}", P1000)
    M = load(find(f"msm_exp8_{m}_email_sham_datamarking_20260822"), P1000)
    P = load(find(f"msm_exp8_{m}_email_spotlighting"), P1000)
    print(f"  {m:15s} B={asr(B)} F={asr(F)}  F-B={diff(F,B)}   P-M={diff(P,M)}")

print("\n== R3. Domain-neutral SRP S' (500 pairs)")
print("== R4. Second placebo N2 (500 pairs)")
for m in ["llama_scout", "deepseek_v3"]:
    U  = load(f"msm_exp8_{m}_email_none", P500)
    N  = load(find(f"msm_exp8_{m}_email_neutral_preamble_2026092?"), P500)
    S  = load(find(f"msm_exp8_{m}_email_srp*"), P500)
    Sn = load(f"r2_srpneutral/{m}", P500)
    N2 = load(f"r2_neutral2/{m}", P500)
    print(f"\n  {m}: U={asr(U)} N={asr(N)} N2={asr(N2)} S={asr(S)} S'={asr(Sn)}")
    print(f"    R3  S-U  ={diff(S,U)}    S'-U ={diff(Sn,U)}")
    print(f"        S'-S ={diff(Sn,S)}   (vocabulary effect: + means rewording weakens SRP)")
    print(f"        S-N  ={diff(S,N)}    S'-N ={diff(Sn,N)}   (content effect, original vs reworded)")
    print(f"    R4  N-U  ={diff(N,U)}    N2-U ={diff(N2,U)}   (block effect, two placebos)")
    print(f"        S-N  ={diff(S,N)}    S-N2 ={diff(S,N2)}   (content effect vs each placebo)")
    print(f"        N2-N ={diff(N2,N)}   (placebo-topic effect)")

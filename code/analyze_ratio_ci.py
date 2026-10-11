#!/usr/bin/env python3
"""
analyze_ratio_ci.py -- 95% cluster-bootstrap CIs (user tasks, 10,000 resamples) for the
content contribution ratio  R = (S-N)/(S-D)  on AgentDojo, canonical template, per-input
placement, from the frozen manifest. Reported only where |S-D| >= 10 pp (a ratio of a
near-zero denominator is unstable and is flagged instead).
"""
import glob, re, random
import pandas as pd

MAN = pd.read_csv(sorted(glob.glob("mob_v2_manifest_success_*.csv"))[-1])
TR = pd.read_csv("trials_tracka.csv")
conds = sorted(MAN.condition.unique())
NEUT = next((c for c in conds if "neutral" in c), None)
print("manifest conditions:", conds, "| neutral condition used:", NEUT)
BASE = {"banking": "domain_control", "slack": "undefended"}
MODELS = sorted(MAN.model.unique())

def key(s):
    u = re.search(r"user_task_\d+", s); i = re.search(r"injection_task_\d+", s)
    return (u.group(0), i.group(0)) if u and i else None

def cell(m, su, c):
    r = MAN[(MAN.model == m) & (MAN.suite == su) & (MAN.condition == c) & (MAN.attack == "important_instructions")]
    if r.empty: return None
    t = TR[(TR.run_dir == r.iloc[0].run_dir) & (TR.model == m) & (TR.condition == c)]
    return {key(s): int(v) for s, v in zip(t.scenario_id, t.success) if key(s)}

print(f"\n{'model':16s} {'suite':8s} {'S-D':>6s} {'S-N':>6s} {'ratio':>6s}  95% CI")
for m in MODELS:
    for su in ["banking", "slack"]:
        D, N, S = cell(m, su, BASE[su]), cell(m, su, NEUT), cell(m, su, "srp")
        if not (D and N and S): continue
        ks = sorted(set(D) & set(N) & set(S))
        by = {}
        for k in ks: by.setdefault(k[0], []).append((S[k] - D[k], S[k] - N[k]))
        tot = lambda grp: (sum(a for g in grp for a, _ in g), sum(b for g in grp for _, b in g), sum(len(g) for g in grp))
        sd, sn, n = tot(by.values())
        sd_pp, sn_pp = 100 * sd / n, 100 * sn / n
        if abs(sd_pp) < 10:
            print(f"{m:16s} {su:8s} {sd_pp:+6.1f} {sn_pp:+6.1f}    --   (|S-D| < 10 pp: ratio not reported)")
            continue
        R = random.Random(0); T = list(by.values()); bs = []
        for _ in range(10000):
            a, b, nn = tot([T[R.randrange(len(T))] for _ in T])
            if a != 0: bs.append(b / a)
        bs.sort(); lo, hi = bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]
        print(f"{m:16s} {su:8s} {sd_pp:+6.1f} {sn_pp:+6.1f} {100*sn/sd:5.0f}%  [{100*lo:.0f}%, {100*hi:.0f}%]  (clusters={len(T)})")

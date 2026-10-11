#!/usr/bin/env python3
"""
verify_paper3.py -- final data items for the paper. Run from ~/mob.
  P  BIPIA single generation: full-2,550 rates (undef/SRP/Spotlight) + placebo
     comparisons paired on their nested subsets
  Q  Cochran's Q for utility cost using cluster-robust (user-task) SEs
  R  two-way crossed bootstrap (user x injection task) for headline + all 6 utility tests
  S  tool-failure accounting: % scenarios with >=1 tool error, ASR among clean scenarios
  T  pooled exact/MC sign-flip tests (utility across 4 models; Llama across suites)
  U  which (template, condition) cells exist for Haiku / Sonnet
  V  trial counts with and without gpt56_sol
Writes paper_verification_report3.txt
"""
import glob, json, math, re, random, itertools
from collections import defaultdict
import pandas as pd

OUT = open("paper_verification_report3.txt", "w")
def say(*a):
    l = " ".join(str(x) for x in a); print(l); OUT.write(l + "\n")
def hdr(t): say("\n" + "=" * 78); say(t); say("=" * 78)
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (100*(c-r)/d, 100*(c+r)/d)
def mcn(b01, b10):
    if b01 + b10 == 0: return float("nan")
    return math.erfc(math.sqrt(((abs(b01-b10)-1)**2/(b01+b10))/2))

man = pd.read_csv(sorted(glob.glob("mob_v2_manifest_success_*.csv"))[-1])
tr = pd.read_csv("trials_tracka.csv")

# ------------------------------------------------------------------ P
hdr("P. BIPIA email -- one generation: full 2,550 for core; nested subsets for placebos")
plan = json.load(open("sample_plan.json"))
try:
    from sample_plan import scenario_key as _key
except Exception:
    _key = None
def load(d):
    rows = {}
    for l in open(f"{d}/per_scenario.jsonl"):
        r = json.loads(l); k = (r.get("attack_name") or r.get("attack_str"), r.get("question"))
        if k in rows: continue
        uid = r.get("scenario_uid") or (_key(r, "bipia") if _key else None)
        rows[k] = (int(bool(r.get("attack_succeeded"))), uid, r.get("attack_str") or r.get("attack_name"))
    return rows
def find(m, sub, cond):
    pats = {"none": [f"msm_exp8_{m}_{sub}_none"],
            "srp": [f"msm_exp8_{m}_{sub}_srp"], "spotlighting": [f"msm_exp8_{m}_{sub}_spotlighting"],
            "sham": [f"msm_exp8_{m}_{sub}_sham_datamarking_2026082?"],
            "neutral": [f"msm_exp8_{m}_{sub}_neutral_preamble_2026092?"]}[cond]
    for p in pats:
        c = [x for x in glob.glob(p) if "limit3" not in x]
        if c: return sorted(c)[-1]
    return None
def clboot(pairs, B=5000):
    cl = defaultdict(list)
    for a, b, g in pairs: cl[g].append(b - a)
    ks = list(cl); rng = random.Random(0); ds = []
    for _ in range(B):
        pick = [cl[rng.choice(ks)] for _ in ks]; n = sum(len(p) for p in pick)
        ds.append(100 * sum(sum(p) for p in pick) / n)
    ds.sort(); p = max(2*min(sum(x <= 0 for x in ds), sum(x >= 0 for x in ds))/B, 1/B)
    return ds[int(.025*B)], ds[int(.975*B)], p
MOD = {"haiku":"Haiku 4.5","sonnet4":"Sonnet 4.6","llama_scout":"Llama Scout","deepseek_v3":"DeepSeek","qwen_32b":"Qwen 32B","mistral_large3":"Mistral L3"}
for m, lab in MOD.items():
    d0 = find(m, "email", "none")
    if not d0: say(f"  {lab}: no undefended email run"); continue
    base = load(d0); k1 = sum(v[0] for v in base.values()); n1 = len(base); lo, hi = wilson(k1, n1)
    say(f"  {lab:11s} undefended  {k1}/{n1} = {100*k1/n1:5.1f}%  Wilson[{lo:.1f},{hi:.1f}]   ({d0})")
    for cond in ["srp", "spotlighting", "sham", "neutral"]:
        d1 = find(m, "email", cond)
        if not d1: say(f"  {lab:11s} {cond:12s} -- not collected"); continue
        trt = load(d1); ks = sorted(set(base) & set(trt))
        a = [base[k][0] for k in ks]; b = [trt[k][0] for k in ks]
        b01 = sum(x == 0 and y == 1 for x, y in zip(a, b)); b10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
        lo, hi, pc = clboot([(base[k][0], trt[k][0], base[k][2]) for k in ks])
        say(f"  {lab:11s} {cond:12s} n={len(ks):5d}  {100*sum(a)/len(ks):5.1f}->{100*sum(b)/len(ks):5.1f}  "
            f"d={100*(sum(b)-sum(a))/len(ks):+6.1f}pp  McNemar p={mcn(b01,b10):.1e}  cluster(attacks) CI[{lo:+.1f},{hi:+.1f}] p~{pc:.4f}")
    # Spotlighting restricted to the sham subset, for a like-for-like placebo comparison
    ds_, dp_ = find(m, "email", "sham"), find(m, "email", "spotlighting")
    if ds_ and dp_:
        sh, sp = load(ds_), load(dp_); ks = sorted(set(base) & set(sh) & set(sp))
        dsh = 100*sum(sh[k][0]-base[k][0] for k in ks)/len(ks); dsp = 100*sum(sp[k][0]-base[k][0] for k in ks)/len(ks)
        say(f"  {lab:11s} same-subset n={len(ks)}: sham d={dsh:+.1f}  spotlight d={dsp:+.1f}  "
            f"sham/spotlight={dsh/dsp*100 if dsp else float('nan'):.0f}%  content (spot-sham)={dsp-dsh:+.1f}pp")

# ---------------------------------------------------------------- helpers for AgentDojo
def cellt(model, suite, cond, col, attack="important_instructions"):
    r = man[(man.model == model) & (man.suite == suite) & (man.condition == cond) & (man.attack == attack)]
    if r.empty: return None
    t = tr[(tr.run_dir == r.iloc[0].run_dir) & (tr.model == model) & (tr.condition == cond)]
    t = t[t[col].astype(str) != ""]
    return t.set_index("scenario_id")[col].astype(int)
def pair(model, suite, base, trt, col):
    a, b = cellt(model, suite, base, col), cellt(model, suite, trt, col)
    if a is None or b is None: return None
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    j["u"] = [re.search(r"user_task_\d+", s).group(0) for s in j.index]
    j["i"] = [(re.search(r"injection_task_\d+", s) or re.search(r"$", s)).group(0) for s in j.index]
    return j

# ------------------------------------------------------------------ Q
hdr("Q. Cochran's Q for SRP utility cost, cluster-robust SEs (user tasks)")
ds, ws = [], []
for m in ["llama_scout", "deepseek_v32", "qwen3_32b", "haiku_45", "sonnet_46", "mistral_large_3"]:
    j = pair(m, "banking", "domain_control", "srp", "utility")
    if j is None: continue
    j["d"] = j.b - j.a; N = len(j); dbar = j.d.mean()
    g = j.groupby("u")["d"].agg(["sum", "count"]); k = len(g)
    se = math.sqrt(k/(k-1) * ((g["sum"] - g["count"]*dbar)**2).sum()) / N * 100
    ds.append(100*dbar); ws.append(1/se**2 if se > 0 else 0)
    say(f"  {m:16s} d={100*dbar:+6.1f}pp  cluster SE={se:.2f}")
def chi2_sf(x, k):
    a, s, t = k/2, 0.0, 1/(k/2)
    for n in range(1, 500): s += t; t *= (x/2)/(a+n)
    return 1 - math.exp(-x/2 + a*math.log(x/2) - math.lgamma(a)) * s
mu = sum(d*w for d, w in zip(ds, ws))/sum(ws); Q = sum(w*(d-mu)**2 for d, w in zip(ds, ws))
say(f"  pooled {mu:+.1f}pp  Q={Q:.2f}  df={len(ds)-1}  p={chi2_sf(Q, len(ds)-1):.3f}")

# ------------------------------------------------------------------ R
hdr("R. Two-way crossed bootstrap (resample user tasks AND injection tasks)")
def twoway(j, B=5000):
    U = sorted(j.u.unique()); I = sorted(j.i.unique()); rng = random.Random(0)
    cell = {(u, i): (b - a) for u, i, a, b in zip(j.u, j.i, j.a, j.b)}
    ds = []
    for _ in range(B):
        us = [rng.choice(U) for _ in U]; is_ = [rng.choice(I) for _ in I]
        v = [cell[(u, i)] for u in us for i in is_ if (u, i) in cell]
        ds.append(100 * sum(v) / len(v))
    ds.sort(); p = max(2*min(sum(x <= 0 for x in ds), sum(x >= 0 for x in ds))/B, 1/B)
    return ds[int(.025*B)], ds[int(.975*B)], p
tests = [("llama_scout","banking","domain_control","srp","success"),
         ("deepseek_v32","banking","domain_control","srp","success"),
         ("llama_scout","slack","undefended","srp","success")] + \
        [(m,"banking","domain_control","srp","utility") for m in
         ["llama_scout","deepseek_v32","qwen3_32b","haiku_45","sonnet_46","mistral_large_3"]]
for t in tests:
    j = pair(*t)
    if j is None: say(f"  {t}: missing"); continue
    lo, hi, p = twoway(j)
    say(f"  {t[0]:15s} {t[1]:8s} {t[4]:7s} d={100*(j.b-j.a).mean():+6.1f}pp  two-way CI[{lo:+.1f},{hi:+.1f}]  p~{p:.4f}")

# ------------------------------------------------------------------ S
hdr("S. Tool-failure accounting (AgentDojo): % scenarios with >=1 tool error; ASR among clean")
def tool_err_map(rd):
    out = {}
    for f in glob.glob(f"{rd}/agentdojo_logs/**/injection_task_*.json", recursive=True):
        try: j = json.load(open(f))
        except Exception: continue
        sid = f"{j.get('user_task_id')}|{j.get('injection_task_id')}"
        err = any(m.get("role") == "tool" and m.get("error") for m in j.get("messages", []))
        out[sid] = (err, int(bool(j.get("security"))), int(bool(j.get("utility"))))
    return out
for su in ["banking", "slack", "travel"]:
    for m in ["llama_scout","deepseek_v32","qwen3_32b","mistral_large_3","haiku_45","sonnet_46"]:
        row = []
        for cond in ["undefended","domain_control","srp","sham_datamarking","spotlighting"]:
            r = man[(man.model == m) & (man.suite == su) & (man.condition == cond) & (man.attack == "important_instructions")]
            if r.empty: continue
            mp = tool_err_map(r.iloc[0].run_dir)
            if not mp: continue
            n = len(mp); e = sum(v[0] for v in mp.values()); clean = [v for v in mp.values() if not v[0]]
            asr_c = 100*sum(v[1] for v in clean)/len(clean) if clean else float("nan")
            row.append(f"{cond[:6]}: err={100*e/n:4.1f}% ASR|clean={asr_c:4.1f}%")
        if row: say(f"  {su:8s} {m:16s} " + "  ".join(row))

# ------------------------------------------------------------------ T
hdr("T. Pooled sign-flip tests")
def flip(vals, label, n):
    g = list(vals); obs = sum(g); k = len(g)
    if k <= 20:
        p = sum(abs(sum(s*v for s, v in zip(S, g))) >= abs(obs)-1e-9 for S in itertools.product([1, -1], repeat=k))/2**k
    else:
        R = random.Random(0); B = 200000
        p = sum(abs(sum(R.choice((1, -1))*v for v in g)) >= abs(obs)-1e-9 for _ in range(B))/B
    say(f"  {label:58s} d={100*obs/n:+6.1f}pp clusters={k} nonzero={sum(v != 0 for v in g)} p={p:.5f}")
tot, N = None, 0
for m in ["llama_scout","deepseek_v32","sonnet_46","haiku_45"]:
    j = pair(m, "banking", "domain_control", "srp", "utility"); g = (j.b-j.a).groupby(j.u).sum()
    tot = g if tot is None else tot.add(g, fill_value=0); N += len(j)
flip(tot.values, "utility, Banking, pooled Llama+DeepSeek+Sonnet+Haiku", N)
for m in ["llama_scout","sonnet_46","haiku_45"]:
    parts, N = [], 0
    for su, b in [("banking","domain_control"),("slack","undefended"),("travel","undefended")]:
        j = pair(m, su, b, "srp", "utility")
        if j is None: continue
        parts.append((j.b-j.a).groupby(su + ":" + j.u).sum()); N += len(j)
    flip(pd.concat(parts).values, f"utility, {m}, Banking+Slack+Travel", N)
parts, N = [], 0
for su, b in [("banking","domain_control"),("slack","undefended"),("travel","undefended")]:
    j = pair("llama_scout", su, b, "srp", "success"); parts.append((j.b-j.a).groupby(su + ":" + j.u).sum()); N += len(j)
flip(pd.concat(parts).values, "Llama SRP backfire, Banking+Slack+Travel", N)

# ------------------------------------------------------------------ U
hdr("U. Template x condition cells present for Anthropic models (Banking)")
for m in ["haiku_45", "sonnet_46"]:
    sub = man[(man.model == m) & (man.suite == "banking")]
    for att, g in sub.groupby("attack"):
        say(f"  {m:10s} {att:24s} conditions: {sorted(g.condition)}")

# ------------------------------------------------------------------ V
hdr("V. Trial counts")
core = man[man.attack == "important_instructions"]
for label, df in [("all models", man), ("excluding gpt56_sol", man[man.model != "gpt56_sol"])]:
    c = df[df.attack == "important_instructions"]
    say(f"  {label:22s} all templates={int(df.n.sum())}  canonical template={int(c.n.sum())}")
OUT.close(); print("\nwrote paper_verification_report3.txt")

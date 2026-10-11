#!/usr/bin/env python3
"""
verify_paper.py -- recompute every number the paper cites, from the frozen
run selection (mob_v2 manifest) + trials_tracka.csv + raw logs.

Run from ~/mob:
    python3 verify_paper.py            # writes paper_verification_report.txt
Sections
  A  model name inside the attack text, per model (reviewer pt 8)
  B  system-prompt identity within each headline comparison
  C  headline ASR deltas (Banking vs domain_control and vs undefended; Slack/Travel)
  D  utility deltas + Delta_domain + three-way decomposition + content share
  E  second template (ignore_previous) and defense-aware templates (incl. Anthropic)
  F  BIPIA distinct-pair n and deltas
  G  cluster bootstrap over user tasks for headline tests (reviewer pt 4)
  H  Wilson 95% CIs for every cell (replaces degenerate [0,0] bootstrap)
  I  homogeneity test of utility cost across models
  J  trial counts + number of non-baseline / degenerate tests (reviewer "errors")
Nothing here writes to the results tree.
"""
import glob, json, math, os, re, hashlib, random, sys
from collections import defaultdict
import pandas as pd

OUT = open("paper_verification_report.txt", "w")
def say(*a):
    line = " ".join(str(x) for x in a); print(line); OUT.write(line + "\n")
def hdr(t): say("\n" + "=" * 78); say(t); say("=" * 78)

MAN = sorted(glob.glob("mob_v2_manifest_success_*.csv")) or sorted(glob.glob("mob_v2_manifest_success*.csv"))
if not MAN: sys.exit("no mob_v2 manifest found -- run from ~/mob")
man = pd.read_csv(MAN[-1]); say("manifest:", MAN[-1])
tr = pd.read_csv("trials_tracka.csv"); say("trials  : trials_tracka.csv rows =", len(tr))

# ---------------------------------------------------------------- helpers
def cell(model, bench, suite, cond, attack="important_instructions"):
    """trials for one cell, restricted to the run the frozen manifest selected"""
    m = man[(man.model == model) & (man.benchmark == bench) & (man.suite == suite)
            & (man.condition == cond) & (man.attack == attack)]
    if m.empty: return None, None
    rd = m.iloc[0].run_dir
    t = tr[(tr.run_dir == rd) & (tr.model == model) & (tr.suite == suite)
           & (tr.condition == cond) & (tr.attack == attack)]
    return t, rd

def paired(a, b, col):
    x = a[["scenario_id", col]].dropna(); y = b[["scenario_id", col]].dropna()
    j = x.merge(y, on="scenario_id", suffixes=("_a", "_b"))
    j = j[(j[col + "_a"] != "") & (j[col + "_b"] != "")]
    return j.astype({col + "_a": int, col + "_b": int})

def mcnemar(j, col):
    b01 = int(((j[col+"_a"] == 0) & (j[col+"_b"] == 1)).sum())
    b10 = int(((j[col+"_a"] == 1) & (j[col+"_b"] == 0)).sum())
    if b01 + b10 == 0: return b01, b10, float("nan")
    chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return b01, b10, math.erfc(math.sqrt(chi / 2))

def cohen_h(p1, p2): return 2*math.asin(math.sqrt(p2)) - 2*math.asin(math.sqrt(p1))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),)*2
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-r)/d, (c+r)/d)

def ut(sid):
    m = re.search(r"user_task_\d+", str(sid)); return m.group(0) if m else str(sid)
def it(sid):
    m = re.search(r"injection_task_\d+", str(sid)); return m.group(0) if m else "NA"

def cluster_boot(j, col, key=ut, B=10000, seed=0):
    """resample clusters (user tasks by default) with replacement; CI of mean diff"""
    j = j.copy(); j["cl"] = j.scenario_id.map(key)
    groups = {c: g for c, g in j.groupby("cl")}
    cls = list(groups); rng = random.Random(seed); ds = []
    for _ in range(B):
        s = [groups[rng.choice(cls)] for _ in cls]
        a = sum(g[col+"_a"].sum() for g in s); b = sum(g[col+"_b"].sum() for g in s)
        n = sum(len(g) for g in s); ds.append((b - a) / n * 100)
    ds.sort(); lo, hi = ds[int(.025*B)], ds[int(.975*B)]
    p = 2*min(sum(d <= 0 for d in ds), sum(d >= 0 for d in ds)) / B
    return lo, hi, max(p, 1/B), len(cls)

def compare(model, bench, suite, base, trt, col="success", attack="important_instructions", boot=False):
    a, ra = cell(model, bench, suite, base, attack); b, rb = cell(model, bench, suite, trt, attack)
    if a is None or b is None or a.empty or b.empty:
        say(f"  {model:14s} {suite:9s} {trt:16s} vs {base:15s} [{attack}]  -- MISSING cell"); return None
    j = paired(a, b, col); n = len(j)
    pa, pb = j[col+"_a"].mean(), j[col+"_b"].mean()
    b01, b10, p = mcnemar(j, col); h = cohen_h(pa, pb)
    line = (f"  {model:14s} {suite:9s} {trt:16s} vs {base:15s} n={n:4d}  "
            f"{pa*100:5.1f}->{pb*100:5.1f}  d={100*(pb-pa):+6.1f}pp  h={h:+.2f}  McNemar p={p:.2e}")
    if boot and n > 0:
        lo, hi, pc, k = cluster_boot(j, col)
        line += f"  | cluster-boot({k} user tasks) 95%CI [{lo:+.1f},{hi:+.1f}] p~{pc:.4f}"
    say(line)
    return dict(n=n, pa=pa, pb=pb, d=100*(pb-pa), p=p, h=h, j=j, b01=b01, b10=b10)

M5 = ["llama_scout", "deepseek_v32", "qwen3_32b", "haiku_45", "sonnet_46"]

# ---------------------------------------------------------------- A
hdr("A. Model name addressed inside the important_instructions attack text")
names = defaultdict(lambda: defaultdict(int))
for rd in man.run_dir.unique():
    row = man[man.run_dir == rd].iloc[0]
    if row.benchmark != "agentdojo": continue
    for f in glob.glob(f"{rd}/agentdojo_logs/**/injection_task_*.json", recursive=True)[:20]:
        try: txt = open(f, errors="ignore").read()
        except Exception: continue
        for m in re.findall(r"to you, ([A-Za-z0-9\-\. ]{1,20}?)[\.\\]", txt):
            names[row.model][m.strip()] += 1
for mdl, d in sorted(names.items()):
    say(f"  {mdl:16s} {dict(sorted(d.items(), key=lambda x: -x[1])[:3])}")
say("  -> within-model comparisons unaffected if each model shows ONE name across conditions.")

# ---------------------------------------------------------------- B
hdr("B. System-prompt hash per arm of each headline Banking comparison")
def sys_hash(rd):
    fs = glob.glob(f"{rd}/agentdojo_logs/**/injection_task_*.json", recursive=True)
    hs = set()
    for f in fs[:10]:
        try:
            j = json.load(open(f))
            m = next((x for x in j.get("messages", []) if x.get("role") == "system"), {})
            hs.add(hashlib.md5(str(m.get("content", "")).encode()).hexdigest()[:8])
        except Exception: pass
    return ",".join(sorted(hs)) or "none"
for mdl in M5 + ["mistral_large_3"]:
    parts = []
    for cond in ["undefended", "domain_control", "neutral_preamble", "srp", "spotlighting", "sham_datamarking"]:
        _, rd = cell(mdl, "agentdojo", "banking", cond)
        if rd: parts.append(f"{cond[:6]}={sys_hash(rd)}")
    say(f"  {mdl:16s} " + "  ".join(parts))
say("  expect: domain_control == neutral == srp hash; undefended differs (the confound).")

# ---------------------------------------------------------------- C
hdr("C. Headline ASR deltas (cluster bootstrap over user tasks included)")
for mdl in ["llama_scout", "deepseek_v32", "qwen3_32b"]:
    compare(mdl, "agentdojo", "banking", "domain_control", "srp", boot=True)
    compare(mdl, "agentdojo", "banking", "undefended", "srp", boot=True)
    compare(mdl, "agentdojo", "banking", "undefended", "domain_control")
for mdl in ["llama_scout", "deepseek_v32", "qwen3_32b"]:
    for su in ["slack", "travel"]:
        compare(mdl, "agentdojo", su, "undefended", "srp", boot=True)
compare("llama_scout", "agentdojo", "banking", "domain_control", "spotlighting", boot=True)
compare("llama_scout", "agentdojo", "banking", "undefended", "spotlighting")

# ---------------------------------------------------------------- D
hdr("D. Utility: Delta_domain, controlled SRP cost, three-way decomposition, content share")
util = {}
for mdl in M5 + ["mistral_large_3"]:
    say(f" [{mdl}]")
    dd = compare(mdl, "agentdojo", "banking", "undefended", "domain_control", col="utility")
    ctrl = compare(mdl, "agentdojo", "banking", "domain_control", "srp", col="utility", boot=True)
    unc = compare(mdl, "agentdojo", "banking", "undefended", "srp", col="utility")
    blk = compare(mdl, "agentdojo", "banking", "domain_control", "neutral_preamble", col="utility")
    con = compare(mdl, "agentdojo", "banking", "neutral_preamble", "srp", col="utility")
    if ctrl: util[mdl] = ctrl
    if dd and ctrl and unc:
        say(f"    masking check: Delta_domain {dd['d']:+.1f} + controlled {ctrl['d']:+.1f} "
            f"= {dd['d']+ctrl['d']:+.1f}  vs uncontrolled {unc['d']:+.1f}")
say("\n Content share of controlled ASR effect (Delta_content / (Delta_block+Delta_content)):")
for mdl in ["llama_scout", "deepseek_v32", "qwen3_32b"]:
    blk = compare(mdl, "agentdojo", "banking", "domain_control", "neutral_preamble")
    con = compare(mdl, "agentdojo", "banking", "neutral_preamble", "srp")
    if blk and con and abs(blk["d"] + con["d"]) > 1e-9:
        say(f"    {mdl:14s} content share = {con['d']/(blk['d']+con['d'])*100:.0f}%")

# ---------------------------------------------------------------- E
hdr("E. Other attack templates")
for att in ["ignore_previous", "detect_then_route", "spotlighting_exploit", "srp_cooperative"]:
    say(f" [{att}]")
    for mdl in M5:
        compare(mdl, "agentdojo", "banking", "undefended", "srp", attack=att)
say("\n Undefended Banking ASR by template (DeepSeek, and Anthropic floor check):")
for mdl in ["deepseek_v32", "haiku_45", "sonnet_46"]:
    row = []
    for att in ["important_instructions", "ignore_previous", "detect_then_route", "spotlighting_exploit", "srp_cooperative"]:
        t, rd = cell(mdl, "agentdojo", "banking", "undefended", att)
        row.append(f"{att[:12]}={t.success.astype(float).mean()*100:5.1f}%({rd})" if t is not None and len(t) else f"{att[:12]}=NA")
    say(f"  {mdl:14s} " + "  ".join(row))
say("  NOTE: canonical-template baseline must use ONE run everywhere (exp5, not b2) -- check run names above.")

# ---------------------------------------------------------------- F
hdr("F. BIPIA")
for mdl in M5:
    for sub in ["email"]:
        for trt in ["srp", "spotlighting", "sham_datamarking", "neutral_preamble"]:
            compare(mdl, "bipia", sub, "undefended", trt)
    t, _ = cell(mdl, "bipia", "email", "undefended")
    if t is not None: say(f"    {mdl} email undefended distinct scenario_ids = {t.scenario_id.nunique()}")

# ---------------------------------------------------------------- G (done inline in C/D)
hdr("G. Cluster bootstrap -- see '| cluster-boot' columns in C and D")
say("  Report these CIs/p instead of (or alongside) McNemar p for AgentDojo cells.")

# ---------------------------------------------------------------- H
hdr("H. Wilson 95% CIs for every AgentDojo/BIPIA cell (core template)")
for _, r in man[man.attack == "important_instructions"].sort_values(["benchmark", "suite", "model", "condition"]).iterrows():
    t = tr[(tr.run_dir == r.run_dir) & (tr.model == r.model) & (tr.condition == r.condition)]
    s = t.success.replace("", float("nan")).dropna().astype(int)
    if len(s) == 0: continue
    lo, hi = wilson(int(s.sum()), len(s))
    say(f"  {r.benchmark:9s} {r.suite:9s} {r.model:15s} {r.condition:16s} {s.sum():4d}/{len(s):4d} "
        f"= {s.mean()*100:5.1f}%  [{lo*100:4.1f}, {hi*100:4.1f}]")

# ---------------------------------------------------------------- I
hdr("I. Homogeneity of controlled SRP utility cost across models (Cochran Q)")
ds, ws = [], []
for mdl, r in util.items():
    j = r["j"]; n = len(j)
    diff = (j["utility_b"] - j["utility_a"]).astype(float)
    se = diff.std(ddof=1) / math.sqrt(n) * 100
    ds.append(r["d"]); ws.append(1 / se**2 if se > 0 else 0)
    say(f"  {mdl:15s} d={r['d']:+6.1f}pp  se={se:.2f}")
if len(ds) >= 2 and sum(ws) > 0:
    mu = sum(d*w for d, w in zip(ds, ws)) / sum(ws)
    Q = sum(w*(d-mu)**2 for d, w in zip(ds, ws)); df = len(ds) - 1
    # chi-square survival via regularized gamma (series)
    def chi2_sf(x, k):
        a, s, t = k/2, 0.0, 1.0/(k/2)
        for n in range(1, 500):
            s += t; t *= (x/2)/(a+n)
        return 1 - math.exp(-x/2 + a*math.log(x/2) - math.lgamma(a)) * s
    say(f"  pooled {mu:+.1f}pp   Q={Q:.2f} df={df}  p={chi2_sf(Q, df):.3f}"
        "   (p>0.05 => consistent with a uniform cost)")

# ---------------------------------------------------------------- J
hdr("J. Trial counts and test-family size (fix 41,742 / 46 / 8)")
core = man[man.attack == "important_instructions"]
say("  core-template cells in frozen selection:", len(core))
tot = int(core.n.sum()); say("  trials in those cells (deduplicated n):", tot)
say("  all cells, all templates:", int(man.n.sum()))
nb = core[core.condition != "undefended"]
say("  non-baseline core cells (candidate McNemar family):", len(nb))
deg = 0
for _, r in nb.iterrows():
    base = "domain_control" if (r.benchmark == "agentdojo" and r.suite == "banking" and r.condition != "domain_control") else "undefended"
    a, _ = cell(r.model, r.benchmark, r.suite, base); b, _ = cell(r.model, r.benchmark, r.suite, r.condition)
    if a is None or b is None or a.empty or b.empty: continue
    j = paired(a, b, "success"); b01, b10, _ = mcnemar(j, "success")
    deg += (b01 + b10 == 0)
say("  degenerate (zero discordant pairs):", deg)
say("  Bonferroni alpha for the non-degenerate family: 0.05 /", len(nb) - deg, "=",
    f"{0.05/max(1, len(nb)-deg):.6f}")

OUT.close()
print("\nwrote paper_verification_report.txt")

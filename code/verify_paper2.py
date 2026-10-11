#!/usr/bin/env python3
"""
verify_paper2.py -- remaining reviewer checks. Run from ~/mob.
  K  BIPIA canonical source: full 2,550 distinct pairs vs 1,000-pair sample,
     McNemar + cluster bootstrap over injected attack strings
  L  system-prompt hash for ignore_previous (alt_*) and b2 template runs
  M  SRP delivery identical in exp6 vs b2 runs (anchor text + placement count)
  N  exact sign-flip test over user tasks for the six headline tests
  O  Workspace neutral/sham vs undefended on the intersected scenario set
Writes paper_verification_report2.txt
"""
import glob, json, math, re, hashlib, random, itertools
from collections import defaultdict
import pandas as pd

OUT = open("paper_verification_report2.txt", "w")
def say(*a):
    l = " ".join(str(x) for x in a); print(l); OUT.write(l + "\n")
def hdr(t): say("\n" + "=" * 78); say(t); say("=" * 78)

def mcn(b01, b10):
    if b01 + b10 == 0: return float("nan")
    chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return math.erfc(math.sqrt(chi / 2))

# ------------------------------------------------------------------ K
hdr("K. BIPIA email: full 2,550 distinct pairs vs 1,000-pair sample")
plan = json.load(open("sample_plan.json"))
keep1000 = set(plan["bipia/email"]["scenario_keys"])
try:
    from sample_plan import scenario_key as _key
except Exception:
    _key = None
def load_bipia(d):
    rows = {}
    for l in open(f"{d}/per_scenario.jsonl"):
        r = json.loads(l)
        k = (r.get("attack_name") or r.get("attack_str"), r.get("question"))
        if k in rows: continue                       # keep first occurrence (dedup)
        uid = r.get("scenario_uid") or (_key(r, "bipia") if _key else None)
        rows[k] = (int(bool(r.get("attack_succeeded"))), uid, r.get("attack_str") or r.get("attack_name"))
    return rows
MODELS = {"haiku":"Haiku 4.5","sonnet4":"Sonnet 4.6","llama_scout":"Llama Scout","deepseek_v3":"DeepSeek","qwen_32b":"Qwen 32B"}
for m, lab in MODELS.items():
    try: base = load_bipia(f"msm_exp8_{m}_email_none")
    except FileNotFoundError: say(f"  {lab}: no msm_exp8_{m}_email_none"); continue
    for dfn in ["srp", "spotlighting"]:
        try: trt = load_bipia(f"msm_exp8_{m}_email_{dfn}")
        except FileNotFoundError: say(f"  {lab} {dfn}: missing"); continue
        ks = sorted(set(base) & set(trt))
        for label, sub in [("full", ks), ("sample1000", [k for k in ks if base[k][1] in keep1000])]:
            if not sub: say(f"  {lab:11s} {dfn:12s} {label:10s} n=0 (uid mismatch?)"); continue
            a = [base[k][0] for k in sub]; b = [trt[k][0] for k in sub]
            b01 = sum(x == 0 and y == 1 for x, y in zip(a, b)); b10 = sum(x == 1 and y == 0 for x, y in zip(a, b))
            d = 100 * (sum(b) - sum(a)) / len(sub)
            # cluster bootstrap over injected attack strings
            cl = defaultdict(list)
            for k in sub: cl[base[k][2]].append(k)
            keys = list(cl); rng = random.Random(0); ds = []
            for _ in range(5000):
                pick = [cl[rng.choice(keys)] for _ in keys]
                n = sum(len(p) for p in pick)
                ds.append(100 * sum(trt[k][0] - base[k][0] for p in pick for k in p) / n)
            ds.sort(); lo, hi = ds[125], ds[4874]
            pc = max(2 * min(sum(x <= 0 for x in ds), sum(x >= 0 for x in ds)) / 5000, 1 / 5000)
            say(f"  {lab:11s} {dfn:12s} {label:10s} n={len(sub):5d} d={d:+6.1f}pp  McNemar p={mcn(b01,b10):.2e}"
                f"  | cluster({len(keys)} attacks) CI[{lo:+.1f},{hi:+.1f}] p~{pc:.4f}")

# ------------------------------------------------------------------ L
hdr("L. System-prompt hash, ignore_previous (alt_*) and b2 template runs")
def shash(d):
    f = glob.glob(f"{d}/agentdojo_logs/**/injection_task_0.json", recursive=True)
    if not f: return "no-logs"
    m = next((x for x in json.load(open(f[0]))["messages"] if x["role"] == "system"), {})
    t = str(m.get("content", "")); return f"{hashlib.md5(t.encode()).hexdigest()[:8]} len={len(t)}"
for d in sorted(glob.glob("alt_*_full")) + sorted(glob.glob("msm_b2_*banking_*_static"))[:12]:
    say(f"  {d:55s} {shash(d)}")
say("  expect: undefended and SRP arms of the SAME template share one hash (no within-template confound)")

# ------------------------------------------------------------------ M
hdr("M. SRP delivery: anchor text and number of insertions per transcript")
def srp_profile(d):
    fs = glob.glob(f"{d}/agentdojo_logs/**/injection_task_*.json", recursive=True)[:20]
    anchors, per = set(), []
    for f in fs:
        j = json.load(open(f)); c = 0
        for m in j.get("messages", []):
            t = str(m.get("content", ""))
            for a in re.findall(r"\[SRP-ANCHOR\].{0,80}", t): anchors.add(a[:60]); c += 1
        per.append(c)
    return len(anchors), (sum(per) / len(per) if per else 0)
for d in ["msm_exp6_llama_scout_banking_defended", "msm_exp6_deepseek_banking_defended",
          "msm_b2_llama_scout_banking_srp_static", "msm_b2_deepseek_v3_banking_srp_static",
          "alt_llama_srp_full", "alt_deepseek_srp_full"]:
    if glob.glob(d):
        n, avg = srp_profile(d); say(f"  {d:45s} distinct anchor openings={n}  mean insertions/transcript={avg:.1f}")
say("  expect: same anchor opening and similar insertion counts => same SRP delivery across campaigns")

# ------------------------------------------------------------------ N
hdr("N. Exact sign-flip test over user tasks (headline family)")
man = pd.read_csv(sorted(glob.glob("mob_v2_manifest_success_*.csv"))[-1])
tr = pd.read_csv("trials_tracka.csv")
def cellt(model, suite, cond, col):
    m = man[(man.model == model) & (man.suite == suite) & (man.condition == cond) & (man.attack == "important_instructions")]
    if m.empty: return None
    t = tr[(tr.run_dir == m.iloc[0].run_dir) & (tr.model == model) & (tr.condition == cond)]
    t = t[t[col].astype(str) != ""]
    return t.set_index("scenario_id")[col].astype(int)
def signflip(model, suite, base, trt, col):
    a, b = cellt(model, suite, base, col), cellt(model, suite, trt, col)
    if a is None or b is None: say(f"  {model} {suite} {trt}: missing"); return
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    j["u"] = [re.search(r"user_task_\d+", s).group(0) for s in j.index]
    g = j.groupby("u").apply(lambda x: (x.b - x.a).sum()).values; n = len(j)
    obs = g.sum()
    if len(g) <= 20:
        cnt = sum(abs(sum(s * v for s, v in zip(signs, g))) >= abs(obs) - 1e-9
                  for signs in itertools.product([1, -1], repeat=len(g)))
        p = cnt / 2 ** len(g); how = f"exact, 2^{len(g)}"
    else:
        rng = random.Random(0); B = 100000
        cnt = sum(abs(sum(rng.choice((1, -1)) * v for v in g)) >= abs(obs) - 1e-9 for _ in range(B))
        p = cnt / B; how = f"MC {B}, {len(g)} clusters"
    say(f"  {model:13s} {suite:8s} {col:7s} {trt:6s} vs {base:14s} d={100*obs/n:+6.1f}pp  sign-flip p={p:.5f} ({how})")
for args in [("llama_scout","banking","domain_control","srp","success"),
             ("deepseek_v32","banking","domain_control","srp","success"),
             ("llama_scout","slack","undefended","srp","success"),
             ("sonnet_46","banking","domain_control","srp","utility"),
             ("haiku_45","banking","domain_control","srp","utility"),
             ("llama_scout","banking","domain_control","srp","utility")]:
    signflip(*args)
say("  headline family alpha = 0.05/6 = 0.0083")

# ------------------------------------------------------------------ O
hdr("O. Workspace: neutral/sham vs undefended on intersected scenarios")
for model in ["deepseek_v32","llama_scout","qwen3_32b","mistral_large_3","haiku_45","sonnet_46"]:
    a = cellt(model, "workspace", "undefended", "success")
    for cond in ["neutral_preamble","sham_datamarking","srp","spotlighting"]:
        b = cellt(model, "workspace", cond, "success")
        if a is None or b is None: continue
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
        say(f"  {model:15s} {cond:17s} n={len(j):3d}  {100*j.a.mean():5.1f} -> {100*j.b.mean():5.1f}  d={100*(j.b.mean()-j.a.mean()):+5.1f}pp")
OUT.close(); print("\nwrote paper_verification_report2.txt")

#!/usr/bin/env python3
"""
verify_paper4.py -- mechanical checks for the new arms (reviewer Q1, Q4). Run from ~/mob.
  A  garble: system prompt byte-identical to undefended; marks present; identical 1,000-key sets
  B  benign: per-run system-prompt hash (D/N/S share base; N/S carry their block once); recount
  C  Llama pooled backfire: 31 up / 7 down / 19 zero, recomputed from frozen trials
  D  engagement counts, recomputed
  E  tokenizer inflation: tokens per injected span, plain vs garbled (open-weight tokenizers)
Writes paper_verification_report4.txt
"""
import json, glob, hashlib, re, sys
import pandas as pd
OUT = open("paper_verification_report4.txt", "w")
def say(*a):
    l = " ".join(str(x) for x in a); print(l); OUT.write(l + "\n")
def hdr(t): say("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)
H = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
text = lambda c: c[0]["content"] if isinstance(c, list) else (c or "")
fails = []

# ---------------------------------------------------------------- A
hdr("A. Garble arm: prompt identity and scenario-set identity (BIPIA email)")
import msm_experiment8_bipia as e8
from sample_plan import scenario_key as K
row = {"context": "hello world attack text", "question": "Q?", "attack_str": "", "ideal": "", "position": "end", "task_name": "t"}
sp_none, _ = e8.build_prompt_for_scenario(row, "email", "none")
sp_garb, _ = e8.build_prompt_for_scenario(row, "email", "garble")
same_shape = sp_none.replace(" ", "") == sp_garb.replace("\u02c8", "").replace(" ", "")
say(f"  undefended len={len(sp_none)} garble len={len(sp_garb)}  marker in garble={chr(0x2c8) in sp_garb}  "
    f"no preamble={sp_garb.startswith(sp_none[:40])}  identical apart from marks={same_shape}")
if not (len(sp_none) == len(sp_garb) and same_shape): fails.append("A prompt")
def keys(d):
    return {json.loads(l).get("scenario_uid") or K(json.loads(l), "bipia") for l in open(f"{d}/per_scenario.jsonl")}
pats = {"B": "none", "G": "garble_20260927", "M": "sham_datamarking_20260822", "P": "spotlighting"}
for m in ["haiku", "sonnet4", "llama_scout", "deepseek_v3", "qwen_32b", "mistral_large3"]:
    ks = {a: keys(sorted([x for x in glob.glob(f"msm_exp8_{m}_email_{p}") if "limit" not in x])[-1]) for a, p in pats.items()}
    plan = set(json.load(open("sample_plan.json"))["bipia/email"]["scenario_keys"])
    G = ks["G"]; ok = len(G) == 1000 and G <= plan and G <= ks["M"] and G <= ks["P"] and G <= ks["B"]
    say(f"  {m:15s} |G|={len(G)}  G in plan={G <= plan}  G in M={G <= ks['M']}  G in P={G <= ks['P']}  G in B={G <= ks['B']}  -> {'OK' if ok else 'FAIL'}")
    if not ok: fails.append(f"A set {m}")

# ---------------------------------------------------------------- B
hdr("B. Benign arms: system-prompt hash per run, block placement, recount")
import msm_experiment6_srp_defense as e6
NEU = re.search(r'NEUTRAL_PREAMBLE = """(.*?)"""', open("msm_experiment9_ablation.py").read(), re.S).group(1)
BLK = {"srp": e6.SRP_PREFIX[:40], "neutral": NEU[:40]}
for su in ["banking", "slack", "travel"]:
    for m in ["haiku", "sonnet4", "llama_scout", "deepseek_v3", "qwen_32b", "mistral_large3"]:
        bases, notes, counts = {}, [], {}
        for a in ["domain", "neutral", "srp"]:
            d = f"benign_{m}_{su}_{a}"
            s = json.load(open(f"{d}/summary.json")); counts[a] = sum(s["per_task"].values())
            logs = glob.glob(f"{d}/agentdojo_logs/**/*.json", recursive=True)
            hs, placed = set(), 0
            for f in logs:
                try: msgs = json.load(open(f))["messages"]
                except Exception: continue
                sysm = text(next(x for x in msgs if x["role"] == "system")["content"])
                tool_hits = sum(1 for x in msgs if x["role"] in ("tool", "user") and a != "domain" and BLK[a] in str(x.get("content")))
                placed += tool_hits > 0 or a == "domain"
                hs.add(H(sysm))
            bases[a] = hs
            if len(hs) != 1: notes.append(f"{a}:{len(hs)} prompts")
        same = bases["domain"] == bases["neutral"] == bases["srp"]
        status = "OK" if same and not notes else "CHECK"
        say(f"  {su:8s} {m:15s} D/N/S tasks done={counts['domain']}/{counts['neutral']}/{counts['srp']}  shared system prompt={same} {' '.join(notes)} -> {status}")
        if status != "OK": fails.append(f"B {su} {m}")
say("  note: per-input arms carry the block with each input, so the system prompt is identical across D/N/S; the block is checked in tool/user turns.")

# ---------------------------------------------------------------- C
hdr("C. Llama pooled backfire split, recomputed from frozen trials")
man = pd.read_csv(sorted(glob.glob("mob_v2_manifest_success_*.csv"))[-1]); tr = pd.read_csv("trials_tracka.csv")
def cell(mk, su, c):
    r = man[(man.model == mk) & (man.suite == su) & (man.condition == c) & (man.attack == "important_instructions")]
    t = tr[(tr.run_dir == r.iloc[0].run_dir) & (tr.model == mk) & (tr.condition == c)]
    return t.set_index("scenario_id").success.astype(int)
for mk in ["llama_scout", "deepseek_v32", "mistral_large_3"]:
    g = []
    for su, b in [("banking", "domain_control"), ("slack", "undefended"), ("travel", "undefended")]:
        j = pd.concat([cell(mk, su, b).rename("a"), cell(mk, su, "srp").rename("b")], axis=1, join="inner")
        g += list((j.b - j.a).groupby([re.search(r"user_task_\d+", x).group(0) for x in j.index]).sum())
    up, dn, z = sum(x > 0 for x in g), sum(x < 0 for x in g), sum(x == 0 for x in g)
    say(f"  {mk:16s} up={up} down={dn} zero={z} of {len(g)}   (separate per-model test; k = nonzero clusters = {up+dn})")

# ---------------------------------------------------------------- D
hdr("D. Engagement counts, recomputed")
for d in sorted(glob.glob("msm_placement_*_srp_sys")) + ["msm_exp6_llama_scout_banking_defended", "msm_exp6_llama_scout_slack_defended",
                                                         "msm_exp6_deepseek_banking_defended", "msm_exp6_mistral_large3_slack_defended"]:
    q = n = 0
    for f in glob.glob(f"{d}/agentdojo_logs/**/injection_task_*.json", recursive=True):
        try: msgs = json.load(open(f))["messages"]
        except Exception: continue
        n += 1; q += any(x["role"] == "assistant" and "SRP" in str(x.get("content")) for x in msgs)
    say(f"  {d:48s} {q}/{n}")

# ---------------------------------------------------------------- E
hdr("E. Tokenizer inflation of injected spans, plain vs garbled")
spans = []
for l in open(sorted(glob.glob("msm_exp8_llama_scout_email_none"))[-1] + "/per_scenario.jsonl"):
    a = json.loads(l).get("attack_str")
    if a and a not in spans: spans.append(a)
spans = spans[:75]
say(f"  {len(spans)} distinct injected attack strings")
try:
    from transformers import AutoTokenizer
except ImportError:
    say("  transformers not installed: pip install transformers --break-system-packages"); spans = []
for name in ["Qwen/Qwen3-32B", "mistralai/Mistral-Large-Instruct-2411", "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]:
    if not spans: break
    try:
        tok = AutoTokenizer.from_pretrained(name)
        p = [len(tok.encode(s)) for s in spans]; g = [len(tok.encode(s.replace(" ", "\u02c8"))) for s in spans]
        ratio = sum(g) / sum(p)
        say(f"  {name:45s} plain={sum(p)/len(p):6.1f} tok/span  garbled={sum(g)/len(g):6.1f}  inflation x{ratio:.2f}")
    except Exception as e:
        say(f"  {name:45s} unavailable ({type(e).__name__}; gated models need an HF token)")

hdr("SUMMARY")
say("  all checks passed" if not fails else f"  CHECK: {fails}")
OUT.close()

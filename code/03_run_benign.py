#!/usr/bin/env python3
"""Benign (no-injection) utility for arms D, N, S on AgentDojo, using the same pipelines
as the attacked runs. Each user task runs once with no injection.
  python3 03_run_benign.py --model llama_scout --suite banking --arm srp --limit 3
  --arm domain | neutral | srp
Writes benign_<model>_<suite>_<arm>/summary.json with per-task utility."""
import argparse, json, re, glob
from pathlib import Path
from agentdojo.benchmark import benchmark_suite_without_injections, get_suite
from agentdojo.logging import OutputLogger
import run_srp_intervention as rsi
exp6 = rsi.exp6
NEUTRAL = re.search(r'NEUTRAL_PREAMBLE = """(.*?)"""', Path("msm_experiment9_ablation.py").read_text(), re.S).group(1)

def system_prompt_for(model, suite):
    # the defended arms' system prompt, read back from an existing SRP run's log
    pats = {"banking": f"msm_exp10_*{model.split('_')[0]}*banking_domain_control"} 
    cands = glob.glob(pats.get(suite, f"msm_exp6_*{model.split('_')[0]}*{suite}_defended"))
    for d in cands:
        for f in glob.glob(f"{d}/agentdojo_logs/**/injection_task_*.json", recursive=True)[:1]:
            m = next(x for x in json.load(open(f))["messages"] if x["role"] == "system")
            c = m["content"]; return c[0]["content"] if isinstance(c, list) else c
    for f in glob.glob(f"*{suite}*/agentdojo_logs/**/injection_task_*.json", recursive=True):
        try:
            m = next(x for x in json.load(open(f))["messages"] if x["role"] == "system")
            c = m["content"]; c = c[0]["content"] if isinstance(c, list) else c
            if "SRP-ANCHOR" not in c and "NEUTRAL-PREAMBLE" not in c and "FORMATTING" not in c and "SPOTLIGHT" not in c:
                print(f"  system prompt from {f.split('/')[0]} ({len(c)} chars)"); return c
        except Exception: pass
    raise SystemExit(f"no log to read the system prompt from for {model}/{suite}")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--suite", required=True)
ap.add_argument("--arm", required=True, choices=["domain", "neutral", "srp"])
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--version", default="v1.2")
a = ap.parse_args()
sp = system_prompt_for(a.model, a.suite)
if a.arm == "domain":
    spec = exp6.MODELS[a.model]; llm = exp6.BedrockLLM(spec)
    pipe = exp6.AgentPipeline([exp6.SystemMessage(sp), exp6.InitQuery(), llm, exp6.ToolsExecutionLoop([exp6.ToolsExecutor(), llm])])
    pipe.name = spec.agentdojo_pipeline_name
else:
    pipe = rsi.build_intervention_pipeline(a.model, sp, exp6.SRP_PREFIX if a.arm == "srp" else NEUTRAL)
suite = get_suite(a.version, a.suite)
tasks = list(suite.user_tasks)[: a.limit or None]
out = Path(f"benign_{a.model}_{a.suite}_{a.arm}" + (f"_limit{a.limit}" if a.limit else "")); logdir = out / "agentdojo_logs"
logdir.mkdir(parents=True, exist_ok=True)
with OutputLogger(str(logdir), live=None):
    res = benchmark_suite_without_injections(pipe, suite, logdir, force_rerun=True, user_tasks=tasks, benchmark_version=a.version)
util = {str(k[0] if isinstance(k, tuple) else k): bool(v) for k, v in res["utility_results"].items()}
json.dump({"model": a.model, "suite": a.suite, "arm": a.arm, "benign": True, "n": len(util),
           "utility_rate": sum(util.values()) / max(len(util), 1), "per_task": util},
          open(out / "summary.json", "w"), indent=2)
print(f"{out}: n={len(util)} benign utility={sum(util.values())/max(len(util),1):.3f}")

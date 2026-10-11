#!/usr/bin/env python3
"""System-prompt-only placement (tests C1-C2). Reuses run_srp_intervention.py unchanged,
swapping only its pipeline builder: the block (SRP or neutral) is appended ONCE to the
system prompt instead of being inserted with every model input.
  python3 02_run_placement.py --model llama_scout --suite banking --variant baseline --limit 3
  --variant baseline = canonical SRP text ; --variant neutral = placebo text
Arm D (domain, no block) is identical to the existing domain-control runs and is reused."""
import sys, re, hashlib
from pathlib import Path
import run_srp_intervention as rsi
exp6 = rsi.exp6
NEUTRAL = re.search(r'NEUTRAL_PREAMBLE = """(.*?)"""', Path("msm_experiment9_ablation.py").read_text(), re.S).group(1)
assert rsi.VARIANTS["baseline"] == exp6.SRP_PREFIX, "baseline variant is not the canonical SRP text -- stop"
rsi.VARIANTS["neutral"] = NEUTRAL

def build_system_only_pipeline(model_key, system_prompt, srp_text):
    spec = exp6.MODELS[model_key]; llm = exp6.BedrockLLM(spec)
    p = exp6.AgentPipeline([exp6.SystemMessage(system_prompt + "\n\n" + srp_text), exp6.InitQuery(), llm,
                            exp6.ToolsExecutionLoop([exp6.ToolsExecutor(), llm])])
    p.name = spec.agentdojo_pipeline_name
    print(f"  [placement=system] block sha256={hashlib.sha256(srp_text.encode()).hexdigest()[:16]} chars={len(srp_text)}")
    return p
rsi.build_intervention_pipeline = build_system_only_pipeline   # looked up at call time
# Guard: never write into an existing srp_intervention output directory.
src = Path(rsi.__file__).read_text()
if '"--output"' not in src and "'--output'" not in src:
    sys.exit("run_srp_intervention.py has no --output flag, so placement runs could overwrite existing "
             "intervention runs. Paste the output of 00_discover.sh and I'll add an output override.")
if "--output" not in sys.argv:
    sys.exit("pass --output msm_placement_<model>_<suite>_<variant> so results land in a new directory")
print("running run_srp_intervention with system-prompt-only placement")
rsi.main()

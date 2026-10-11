#!/usr/bin/env bash
# launch_round2.sh -- run from ~/mob with the venv active.
#   bash launch_round2.sh smoke    # 3-row test of every mode; check output, then:
#   bash launch_round2.sh launch   # commit protocol, start all runs in background
set -euo pipefail
cd ~/mob
MODELS="haiku sonnet4 llama_scout deepseek_v3 qwen_32b mistral_large3"
TWO="llama_scout deepseek_v3"

if [[ "${1:-}" == "smoke" ]]; then
  python3 run_variant.py --show
  for mode in framing neutral2 srp_neutral benign_none benign_garble; do
    BIPIA_PLAN_ENTRY=bipia/email_n500 python3 run_variant.py $mode \
      --model llama_scout --subset email --limit 3 --output r2_smoke/$mode
  done
  python3 - <<'PY'
import json, glob
for d in sorted(glob.glob("r2_smoke/*")):
    rs=[json.loads(l) for l in open(f"{d}/per_scenario.jsonl")]
    r=rs[0]; print(f"{d:28s} rows={len(rs)} defense={r.get('defense')}  resp={r.get('response','')[:70]!r}")
PY
  echo ">>> Check: framing rows have NO U+02C8 in the prompt; benign rows answer without the attack."
  exit 0
fi

if [[ "${1:-}" == "launch" ]]; then
  git add run_variant.py judge_correct.py launch_round2.sh
  git commit -m "protocol r2: benign comprehension, framing-only, domain-neutral SRP, second placebo" || true
  git log --oneline -1
  mkdir -p logs_r2 r2_benign r2_framing r2_srpneutral r2_neutral2
  for m in $MODELS; do
    ( for a in none garble sham spotlighting; do
        BIPIA_PLAN_ENTRY=bipia/email_n500 python3 run_variant.py benign_$a \
          --model $m --subset email --output r2_benign/${m}_${a} --concurrency 5
      done
      BIPIA_PLAN_ENTRY=bipia/email python3 run_variant.py framing \
        --model $m --subset email --output r2_framing/${m} --concurrency 5
    ) > logs_r2/${m}.log 2>&1 &
  done
  for m in $TWO; do
    ( BIPIA_PLAN_ENTRY=bipia/email_n500 python3 run_variant.py srp_neutral \
        --model $m --subset email --output r2_srpneutral/${m} --concurrency 5
      BIPIA_PLAN_ENTRY=bipia/email_n500 python3 run_variant.py neutral2 \
        --model $m --subset email --output r2_neutral2/${m} --concurrency 5
    ) > logs_r2/${m}_srp_n2.log 2>&1 &
  done
  echo "Launched. Monitor: tail -f logs_r2/*.log   |   wc -l r2_*/*/per_scenario.jsonl"
fi

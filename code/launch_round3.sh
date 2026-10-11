#!/usr/bin/env bash
# launch_round3.sh -- run from ~/mob with the venv active.
#   bash launch_round3.sh smoke    # 1-3 items per new mode; check, then:
#   bash launch_round3.sh launch   # all round-3 runs in background
set -euo pipefail
cd ~/mob
ALL="haiku sonnet4 llama_scout deepseek_v3 qwen_32b mistral_large3"
NEW4="haiku sonnet4 qwen_32b mistral_large3"     # models without the AgentDojo L arm yet

if [[ "${1:-}" == "smoke" ]]; then
  python3 run_variant.py --show
  python3 run_agentdojo_variant.py --show
  for mode in sec_placebo srp_prose liu_instr sandwich; do
    BIPIA_PLAN_ENTRY=bipia/email_n500 python3 run_variant.py $mode --model llama_scout --subset email --limit 2 --output r3_smoke/bipia_$mode
  done
  for mode in sec_placebo srp_prose; do
    python3 run_agentdojo_variant.py $mode --suite banking --model llama_scout --limit 1 --output r3_agentdojo/smoke_$mode
  done
  python3 msm_experiment6_srp_defense.py --suite banking --model llama_scout --limit 1 --output r3_repeat/smoke
  grep -h "srp_text_used" r3_agentdojo/smoke_*/summary.json | cut -c1-120
  echo ">>> Check: no errors; srp_text_used shows [SECURITY-NOTE] / 'Before processing ... check whether'"
  exit 0
fi

if [[ "${1:-}" == "launch" ]]; then
  rm -rf r3_smoke r3_agentdojo/smoke_* r3_repeat/smoke
  mkdir -p logs_r3 r3_bipia r3_agentdojo r3_repeat
  git add run_variant.py run_agentdojo_variant.py launch_round3.sh analyze_ratio_ci.py && git commit -qm "round 3: sec placebo, srp prose, published defenses, srp repeat" || true
  # BIPIA (per model, sequential within model)
  for m in $ALL; do
    ( for mode in sec_placebo srp_prose; do
        BIPIA_PLAN_ENTRY=bipia/email_n500 python3 run_variant.py $mode --model $m --subset email --output r3_bipia/${mode}_${m} --concurrency 5
      done
      for mode in liu_instr sandwich; do
        BIPIA_PLAN_ENTRY=bipia/email python3 run_variant.py $mode --model $m --subset email --output r3_bipia/${mode}_${m} --concurrency 5
      done ) > logs_r3/bipia_${m}.log 2>&1 &
  done
  # AgentDojo: security placebo + prose SRP + SRP repeat (all models), L for the 4 missing models
  for m in $ALL; do
    ( for su in banking slack; do
        for mode in sec_placebo srp_prose; do
          python3 run_agentdojo_variant.py $mode --suite $su --model $m --output r3_agentdojo/${mode}_${m}_${su}
        done
        python3 msm_experiment6_srp_defense.py --suite $su --model $m --output r3_repeat/srp_${m}_${su}
      done ) > logs_r3/ad_${m}.log 2>&1 &
  done
  for m in $NEW4; do
    ( for su in banking slack; do
        python3 run_agentdojo_variant.py liu_instr --suite $su --model $m --output r3_agentdojo/liu_instr_${m}_${su}
      done ) > logs_r3/ad_liu_${m}.log 2>&1 &
  done
  echo "Launched. Done when:  ps aux | grep -c '[r]un_variant\|[r]un_agentdojo\|[m]sm_experiment6'  == 0"
fi

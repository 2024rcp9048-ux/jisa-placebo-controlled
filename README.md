# Placebo-Controlled Evaluation of Prompt-Injection Defenses

Artifact for: M. Singh and N. Mittal, "Placebo-Controlled Evaluation of Prompt-Injection Defenses: Attributing Defense Effects Across Models and Benchmarks" (submitted to the *Journal of Information Security and Applications*). Release: **`v1.0-jisa-submission`**.

## Layout
| Folder | Contents |
|---|---|
| `prompts/` | Every prompt text verbatim (SRP, neutral/gardening/security placebos, prose SRP, Spotlighting and sham preambles, framing-only preamble, instructional and sandwich prevention, judge prompts). `prompts_index.json` gives character counts and SHA-256 prefixes matching the paper's Appendix A. |
| `results/` | Frozen tables behind the paper's numbers (mob_v2 manifest, sample plan) and verification reports. |
| `code/` | Runners, control-arm wrappers, analysis (ladder decomposition, sign-flip permutation tests, cluster bootstrap, Holm), mechanical assertion checks, judge re-scoring. Only scripts used for this paper. |
| `logs/` | Run logs for the round-2 and round-3 experiments. |
| `SHA256SUMS` | Checksums of every released file. |

Raw per-scenario outputs (~2.3 GB: every response, AgentDojo transcript, judge verdict) are attached to the GitHub release `v1.0-jisa-submission` (Releases → Assets): `runs_agentdojo.tar.gz`, `runs_bipia.tar.gz`, `trials_tracka.csv`, `RUN_DIRS.txt`, `SHA256SUMS`.

## Environment
| Component | Version |
|---|---|
| Python | 3.10.12 |
| AgentDojo | 0.1.35 (suite v1.2) |
| BIPIA | commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` (https://github.com/microsoft/BIPIA), placed at `external_benchmarks/bipia` |
| Packages | `pip install -r code/requirements.txt` |

Models (Amazon Bedrock, `us-east-1`):

| Model | Role | Bedrock model ID | Temperature |
|---|---|---|---|
| Claude Haiku 4.5 | evaluated; BIPIA judge | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 0 |
| Claude Sonnet 4.6 | evaluated | `us.anthropic.claude-sonnet-4-6` | 0 |
| Llama 4 Scout 17B | evaluated | `us.meta.llama4-scout-17b-instruct-v1:0` | 0 |
| DeepSeek V3.2 | evaluated; second judge | `deepseek.v3.2` | 0 |
| Qwen3 32B | evaluated | `qwen.qwen3-32b-v1:0` | 0 |
| Mistral Large 3 | evaluated | `mistral.mistral-large-3-675b-instruct` | 0 |
| GPT-5.6 Sol | robustness only | `us.openai.gpt-5.6-sol` | provider default |
| GPT-5.6 Terra | independent judge | `us.openai.gpt-5.6-terra` | provider default |

Anthropic and Llama models use account-specific inference profiles; replace `<ACCOUNT_ID>` in `code/msm_experiment5_agentdojo.py` with your AWS account.

## Reproducing the results
Extract the release archives into the repository root, then:

| Paper result | Script |
|---|---|
| Headline numbers, frozen matrix, ladder decompositions | `code/verify_paper.py`, `verify_paper2.py`, `verify_paper3.py` |
| Pre-specified tests C1–C3 | `code/confirmatory_analysis.py` |
| Permutation tests and bootstrap intervals | `code/signflip_test.py`, `code/compute_paper_cis.py` |
| Framing-only, second placebo, reworded SRP (BIPIA) | `code/analyze_round2.py` |
| Reworded SRP and instructional prevention (AgentDojo) | `code/analyze_agentdojo_r2.py` |
| Security placebos, prose SRP, published defenses, repeats | `code/analyze_round3.py` |
| Content-ratio intervals | `code/analyze_ratio_ci.py` |
| Benign comprehension | `code/judge_correct.py report` |
| Independent judge (GPT-5.6 Terra) | `code/rejudge_terra.py report` |
| Second judge (DeepSeek V3.2) | `code/rebuttal_3_second_judge.py` |
| Engagement counts | `code/engagement_canonical.py` |
| Variance decomposition | `code/variance_decomposition.py` |
| Trial table | `code/build_trials_csv.py` |

Mechanical checks (prompt hash, scenario sets, grid counts, unique keys): `python code/verify_paper4.py`

Re-running experiments costs money (Bedrock): `bash code/launch_round3.sh smoke`, then `bash code/launch_round3.sh launch`. Hosted inference is not fully deterministic at temperature 0; repeated runs flip 0–12% of outcomes, but paired effects reproduce within about 2 points.

## License
Code: MIT. Prompts and result tables: CC BY 4.0. AgentDojo and BIPIA remain under their own licenses and are not redistributed.

#!/usr/bin/env python3
"""
msm_experiment11_sham_agentdojo.py — sham-datamarking control on AgentDojo.

WHY A WRAPPER, NOT A NEW RUNNER
-------------------------------
The whole point of this control is that the MECHANICS are identical to
Spotlighting and only the PREAMBLE differs. Reimplementing the pipeline
would risk a second silent divergence of exactly the kind that produced the
Banking system-prompt confound. So this imports msm_experiment7 unchanged,
swaps one module-level constant, and hands off. The SpotlightingDefense
pipeline element, the U+02C8 substitution, the tool-loop placement, the
model specs and the scoring are the exp7 originals, untouched.

The swap works because build_defended_pipeline() reads
SPOTLIGHT_SYSTEM_PREAMBLE from module scope at call time.

After the run, summary.json is rewritten so downstream tooling classifies
the cell correctly:
    defense                   -> "Sham datamarking (matched Spotlighting control)"
    spotlight_system_preamble -> the sham text actually used
    sham_control              -> True
    matched_against           -> the real Spotlighting preamble

USAGE (arguments are exp7's, unchanged)
    python msm_experiment11_sham_agentdojo.py --model haiku --suite banking \
        --limit 3 --output ./smoke_sham_haiku_banking

    python msm_experiment11_sham_agentdojo.py --model llama_scout \
        --suite banking --output ./msm_exp11_llama_scout_banking_sham

ALWAYS --limit 3 first, per standing practice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import msm_experiment7_spotlighting_defense as exp7
from sham_preambles import SHAM_AGENTDOJO, SPOTLIGHT_AGENTDOJO, check

DEFENSE_LABEL = "Sham datamarking (matched Spotlighting control)"


def verify_swap() -> None:
    """Refuse to run unless the constant in exp7 is byte-identical to the
    string this control was matched against. If exp7's preamble is ever
    edited, the sham is no longer matched and the decomposition is invalid."""
    live = getattr(exp7, "SPOTLIGHT_SYSTEM_PREAMBLE", None)
    if live is None:
        sys.exit("FATAL: exp7 has no SPOTLIGHT_SYSTEM_PREAMBLE.")
    if live != SPOTLIGHT_AGENTDOJO:
        print("FATAL: exp7's Spotlighting preamble differs from the string "
              "sham_preambles.py was matched against.\n")
        print(f"  exp7 ({len(live)} ch): {live[:120]!r}...")
        print(f"  ours ({len(SPOTLIGHT_AGENTDOJO)} ch): "
              f"{SPOTLIGHT_AGENTDOJO[:120]!r}...")
        sys.exit("Re-tune SHAM_AGENTDOJO against the live text before running.")
    if not check(SPOTLIGHT_AGENTDOJO, SHAM_AGENTDOJO, "pre-run match check"):
        sys.exit("FATAL: sham no longer structurally matched.")
    if getattr(exp7, "SPOTLIGHT_MARKER", None) != "\u02c8":
        sys.exit("FATAL: exp7 marker is not U+02C8; mechanics would differ.")


def stamp(outdir: Path) -> None:
    f = outdir / "summary.json"
    if not f.exists():
        print(f"  [warn] no summary.json at {f}; not stamped")
        return
    d = json.loads(f.read_text())
    d["defense"] = DEFENSE_LABEL
    d["sham_control"] = True
    d["spotlight_system_preamble"] = SHAM_AGENTDOJO
    d["matched_against"] = SPOTLIGHT_AGENTDOJO
    d["experiment"] = "msm_experiment11_sham_datamarking"
    f.write_text(json.dumps(d, indent=2))
    print(f"  stamped {f} -> defense={DEFENSE_LABEL!r}")


def main() -> int:
    verify_swap()

    # The swap. Everything downstream in exp7 now emits the sham preamble.
    exp7.SPOTLIGHT_SYSTEM_PREAMBLE = SHAM_AGENTDOJO
    assert exp7.SPOTLIGHT_SYSTEM_PREAMBLE == SHAM_AGENTDOJO

    print("\n" + "=" * 66)
    print("SHAM DATAMARKING CONTROL (AgentDojo)")
    print("=" * 66)
    print("  mechanics : exp7 originals, untouched (U+02C8, same pipeline)")
    print("  preamble  : SHAM_AGENTDOJO substituted for SPOTLIGHT_AGENTDOJO")
    print("=" * 66 + "\n")

    args = exp7.parse_args()
    rc = exp7.main() if exp7.main.__code__.co_argcount == 0 else exp7.main(args)
    stamp(Path(args.output))
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())

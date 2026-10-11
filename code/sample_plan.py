#!/usr/bin/env python3
"""
sample_plan.py — deterministic scenario sampling with a recorded manifest.

RULE
    n <= 1000   use the whole grid (small benchmarks are not sampled)
    n >  1000   draw exactly 1000, deterministically, and RECORD which ones

    1,000 is 2x the project's >=500-per-test standing rule. AgentDojo suites
    (105-144) are fixed by the benchmark and fall far below the cap; the rule
    binds only on BIPIA email (2,550 distinct) and code (2,450).

WHY THE MANIFEST MATTERS MORE THAN THE SEED
-------------------------------------------
A fixed seed alone is not reproducible. The drawn set also depends on the
order of the input pool, and that order depends on filesystem globbing,
dict iteration, dedup rules -- all of which have changed during this project.
Same seed + different pool order = different 500 scenarios, and a re-run then
shows a "change" that is pure resampling.

So the sample is stored as an explicit list of scenario IDs, keyed by
(benchmark, suite). Every model, every condition, every attack draws the SAME
500. That makes cells paired by construction:

  * delta = treatment - baseline is computed on identical scenarios
  * cross-model comparisons are on identical scenarios
  * a re-run months later reproduces the same cells exactly

The manifest is content-addressed: the pool itself is hashed, so if the
underlying benchmark data changes the plan refuses to reuse a stale sample
rather than silently drawing from a different population.

BIPIA NOTE
    Sample from the DEDUPLICATED pool -- distinct (attack_str, question).
    BIPIA email has 2,550 distinct content pairs across 3,750 rows; email code
    2,450 across 2,500. Sampling raw rows redraws duplicates and reintroduces
    the effective-n inflation documented in the canonical record.

USAGE
    # build once, from the deduplicated pools
    python sample_plan.py build --benchmark bipia --suite email \
        --pool-from msm_exp8_haiku_email_none/per_scenario.jsonl

    python sample_plan.py build --benchmark agentdojo --suite banking \
        --pool-from msm_exp5_deepseek_banking/summary.json

    python sample_plan.py show
    python sample_plan.py verify --benchmark bipia --suite email \
        --pool-from msm_exp8_qwen_32b_email_none/per_scenario.jsonl

In a runner:
    from sample_plan import load_sample
    keep = load_sample("bipia", "email")        # None => use everything
    if keep is not None:
        scenarios = [s for s in scenarios if scenario_key(s) in keep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PLAN = Path("sample_plan.json")
CAP = 1000
SEED = 20260822


# ---------------------------------------------------------------------------
# Scenario identity
# ---------------------------------------------------------------------------

def scenario_key(row: dict, benchmark: str) -> str | None:
    """Stable content identity for one scenario.

    AgentDojo: (user_task_id, injection_task_id) is genuinely unique.
    BIPIA: NO field combination is unique -- task_name is constant and
    (attack_str, question) yields 2,550 of 3,750 rows. That pair is the
    content identity; rows sharing it are duplicates.
    """
    if benchmark == "agentdojo":
        u, i = row.get("user_task_id"), row.get("injection_task_id")
        return f"{u}|{i}" if u and i else None
    uid = row.get("scenario_uid")
    if uid:
        return str(uid)
    a = row.get("attack_str")
    # BIPIA code rows carry no `question` -- the runner resolves it as
    # `question or error` inside build_prompt_for_scenario, which happens
    # AFTER the sample filter. Mirror that fallback here or plan-time and
    # filter-time keys differ and the filter matches nothing (code plan
    # matched 0 of 1000; email worked only because it has `question`).
    q = row.get("question")
    if q is None:
        q = row.get("error")
    if a is None and q is None:
        return None
    return hashlib.sha1(f"{a}||{q}".encode()).hexdigest()[:16]


def read_pool(path: Path, benchmark: str) -> list[str]:
    """Distinct scenario keys, in a canonical (sorted) order.

    Sorting is deliberate. Relying on file order would make the draw depend
    on collection order, which is exactly the hidden dependency this file
    exists to remove.
    """
    keys, seen = [], set()
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        d = json.loads(path.read_text())
        rows = d.get("scenarios") or []
    for r in rows:
        k = scenario_key(r, benchmark)
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return sorted(keys)


def pool_hash(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------

def load_plan() -> dict:
    return json.loads(PLAN.read_text()) if PLAN.exists() else {}


def save_plan(p: dict) -> None:
    PLAN.write_text(json.dumps(p, indent=2))


def load_sample(benchmark: str, suite: str) -> set[str] | None:
    """Return the scenario-key set for this surface, or None to use all."""
    e = load_plan().get(f"{benchmark}/{suite}")
    if not e or not e.get("sampled"):
        return None
    return set(e["scenario_keys"])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_build(a) -> int:
    surface = f"{a.benchmark}/{a.suite}"
    src = Path(a.pool_from)
    if not src.exists():
        sys.exit(f"pool source not found: {src}")

    keys = read_pool(src, a.benchmark)
    ph = pool_hash(keys)
    n = len(keys)
    plan = load_plan()

    if surface in plan and not a.force:
        old = plan[surface]
        if old["pool_hash"] == ph:
            print(f"{surface}: already planned, pool unchanged -- keeping "
                  f"existing sample (n={len(old['scenario_keys']) or n})")
            return 0
        sys.exit(
            f"{surface}: a plan exists but the POOL HAS CHANGED\n"
            f"  recorded pool_hash {old['pool_hash']}  ({old['pool_size']} keys)\n"
            f"  current  pool_hash {ph}  ({n} keys)\n"
            "Re-sampling would silently change the population. Investigate, "
            "then pass --force if the change is intended.")

    if n <= CAP:
        entry = {"sampled": False, "pool_size": n, "n": n, "pool_hash": ph,
                 "source": str(src), "scenario_keys": [],
                 "note": f"n={n} <= cap {CAP}; full grid used, no sampling"}
        print(f"{surface}: n={n} <= {CAP} -- FULL GRID, no sampling")
    else:
        rng = random.Random(f"{SEED}:{surface}:{ph}")
        chosen = sorted(rng.sample(keys, CAP))
        entry = {"sampled": True, "pool_size": n, "n": CAP, "pool_hash": ph,
                 "seed": SEED, "source": str(src), "scenario_keys": chosen,
                 "sample_hash": pool_hash(chosen)}
        print(f"{surface}: n={n} > {CAP} -- SAMPLED {CAP} "
              f"(sample_hash {entry['sample_hash']})")

    plan[surface] = entry
    save_plan(plan)
    print(f"  wrote {PLAN}")
    return 0


def cmd_show(a) -> int:
    plan = load_plan()
    if not plan:
        print("no plan yet")
        return 0
    print(f"{'surface':22s}{'pool':>8s}{'used':>8s}{'sampled':>9s}  hash")
    print("-" * 68)
    for k in sorted(plan):
        e = plan[k]
        h = e.get("sample_hash", e["pool_hash"])
        print(f"{k:22s}{e['pool_size']:>8d}{e['n']:>8d}"
              f"{str(e['sampled']):>9s}  {h}")
    tot = sum(e["n"] for e in plan.values())
    print(f"\nscenarios per (model, condition) across all surfaces: {tot}")
    return 0


def cmd_verify(a) -> int:
    surface = f"{a.benchmark}/{a.suite}"
    e = load_plan().get(surface)
    if not e:
        sys.exit(f"no plan for {surface}")
    keys = read_pool(Path(a.pool_from), a.benchmark)
    ph = pool_hash(keys)

    ok = ph == e["pool_hash"]
    print(f"{surface}")
    print(f"  pool hash  plan={e['pool_hash']}  this source={ph}  "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if e["sampled"]:
        have = set(keys)
        missing = [k for k in e["scenario_keys"] if k not in have]
        print(f"  sampled {e['n']} keys; missing from this source: {len(missing)}")
        if missing:
            print("  !! this source cannot serve the recorded sample")
            for m in missing[:5]:
                print(f"       {m}")
            return 1
    if not ok:
        print("  !! pool differs -- cells built from this source are NOT "
              "comparable to cells built from the recorded pool")
        return 1
    print("  OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--benchmark", required=True, choices=["agentdojo", "bipia"])
    b.add_argument("--suite", required=True)
    b.add_argument("--pool-from", required=True)
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("show")
    s.set_defaults(fn=cmd_show)

    v = sub.add_parser("verify")
    v.add_argument("--benchmark", required=True, choices=["agentdojo", "bipia"])
    v.add_argument("--suite", required=True)
    v.add_argument("--pool-from", required=True)
    v.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

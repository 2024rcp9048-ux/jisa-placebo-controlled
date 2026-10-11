"""
build_trials_csv.py — flatten ~/mob/msm_*/summary.json into a tidy trial CSV.

Reads metadata from the JSON body (model / suite / defense / subset fields),
NOT from directory names. Path parsing failed on 153/261 dirs; the JSON is
authoritative and self-describing.

Observed schema (AgentDojo):
    model, model_display_name, suite, defense, benchmark_version,
    total_scenarios, attack_successes, attack_success_rate,
    task_completions, utility_rate,
    scenarios: [{user_task_id, injection_task_id, attack_succeeded,
                 task_completed}, ...]

Observed schema (BIPIA, 31 dirs):
    subset, judge_model, valid_scenarios, error_count, by_attack_category

Output columns:
    model family benchmark suite condition attack scenario_id success utility
    run_dir

USAGE
    python build_trials_csv.py --root ~/mob --audit
        Inventory only. Prints every distinct model/suite/defense value found
        and every dir it could not classify. RUN THIS FIRST.

    python build_trials_csv.py --root ~/mob --out trials.csv
        Write the tidy CSV.

    python build_trials_csv.py --root ~/mob --out trials.csv --rectangle
        Also print Track A rectangle coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Normalisation. Extend after running --audit.
# ---------------------------------------------------------------------------

# Attack is encoded ONLY in the directory-name suffix. AgentDojo's log
# `attack_type` field reads "important_instructions" for every b2 run, so
# relying on it pools four distinct attacks into one cell -- and because all
# four have identical n, the run selector then picks between them
# alphabetically, silently making a defense-aware-attack run the baseline.
ATTACK_SUFFIXES = [
    ("_spotlighting_exploit", "spotlighting_exploit"),
    ("_detect_then_route",    "detect_then_route"),
    ("_srp_cooperative",      "srp_cooperative"),
    ("_rotation_static",      "rotation"),
    ("_static",               "important_instructions"),
]

MODEL_MAP = {
    "llama_scout": "llama_scout", "llama_4_scout": "llama_scout",
    "llama4_scout": "llama_scout", "scout": "llama_scout",
    "deepseek": "deepseek_v32", "deepseek_v3": "deepseek_v32",
    "deepseek_v3_2": "deepseek_v32", "deepseek_v32": "deepseek_v32",
    "qwen_32b": "qwen3_32b", "qwen3_32b": "qwen3_32b", "qwen_3_32b": "qwen3_32b",
    "qwen3_next": "qwen3_next",
    "haiku": "haiku_45", "haiku_45": "haiku_45", "claude_haiku_45": "haiku_45",
    "sonnet": "sonnet_46", "sonnet_46": "sonnet_46", "claude_sonnet_46": "sonnet_46",
    "mistral_large3": "mistral_large_3", "mistral_large_3": "mistral_large_3",
    "kimi_k2": "kimi_k2", "kimi_k2_thinking": "kimi_k2",
    "kimi_k2_think": "kimi_k2",
    "gemma_3_27b": "gemma_3_27b",
    "gpt56_sol": "gpt56_sol", "gpt56_terra": "gpt56_terra",
    # --- added after audit ---
    "sonnet4": "sonnet_46", "sonnet_4": "sonnet_46", "sonnet_4_6": "sonnet_46",
    "qwen3_next_80b": "qwen3_next", "qwen_3_next": "qwen3_next",
}

FAMILY = {
    "llama_scout": "meta", "deepseek_v32": "deepseek",
    "qwen3_32b": "alibaba", "qwen3_next": "alibaba",
    "haiku_45": "anthropic", "sonnet_46": "anthropic",
    "mistral_large_3": "mistral", "kimi_k2": "moonshot",
    "gemma_3_27b": "google",
    "gpt56_sol": "openai", "gpt56_terra": "openai",
}

# `defense` is a human-readable string in the summaries. Longest match wins.
DEFENSE_PATTERNS = [
    (r"neutral.?preamble|length.?matched", "neutral_preamble"),
    (r"domain.?prompt.?control|domain.?control|experiment ?10", "domain_control"),
    (r"sham.?datamark|sham", "sham_datamarking"),
    (r"self.?reflect|\bsrp\b", "srp"),
    (r"spotlight", "spotlighting"),
    (r"undefended|no.?defense|baseline|\bnone\b", "undefended"),
]

# Fall back to the directory name only when `defense` is absent from the JSON.
DIR_DEFENSE = [
    # msm_experiment5_agentdojo.py IS the undefended baseline runner. Its
    # summaries carry no `defense` field and its dirnames carry no marker,
    # so it must be recognised by script prefix or 38 baseline runs vanish.
    (r"(^|_)msm_exp(eriment)?5(_|$)", "undefended"),
    (r"_ablation$|neutral", "neutral_preamble"),
    (r"domain.?control|exp(eriment)?10", "domain_control"),
    (r"_sham", "sham_datamarking"),
    (r"_srp", "srp"),
    (r"_spotlighting", "spotlighting"),
    # "_undefended" / "_none" are unambiguous baselines. Order matters:
    # this must come AFTER _srp/_spotlighting so "undefended_static" is not
    # shadowed. "_defended$" alone stays ambiguous and is refused.
    (r"_undefended|_none(_|$)", "undefended"),
]

SUITE_MAP = {
    "banking": "banking", "slack": "slack", "travel": "travel",
    "workspace": "workspace", "email": "email",
}

TRACK_A_MODELS = ["llama_scout", "deepseek_v32", "qwen3_32b",
                  "haiku_45", "sonnet_46", "mistral_large_3", "gpt56_sol"]
# gpt56_terra is NOT in the panel: refuses all AgentDojo Banking scenarios
# (0% utility, so ASR is undefined there) and shows 11.4% run-to-run scenario
# disagreement against Sol's 1.9% on the identical measurement.
# Sample plan integration. When sample_plan.json exists, every surface with a
# recorded 500-key sample is restricted to exactly those keys -- for existing
# cells as well as new ones. Without this, previously-collected BIPIA cells
# (2,550 scenarios) and newly-collected ones (500) would sit in the same
# rectangle at different effective n, and the delta pairing would silently
# intersect down to the overlap without anyone declaring it.
def _load_sample_plan():
    f = Path("sample_plan.json")
    if not f.exists():
        return {}
    try:
        plan = json.loads(f.read_text())
    except Exception:
        return {}
    return {k: set(v["scenario_keys"]) for k, v in plan.items()
            if v.get("sampled") and v.get("scenario_keys")}


SAMPLE_PLAN = _load_sample_plan()


TRACK_A_SURFACES = {"agentdojo/banking": 144, "agentdojo/slack": 105,
                    "agentdojo/travel": 140, "agentdojo/workspace": 560,
                    "bipia/email": 1000,
                    "bipia/code": 1000}
BASE_CONDS = ["undefended", "neutral_preamble", "srp",
              "sham_datamarking", "spotlighting"]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def norm_defense(raw, dirname: str):
    if raw:
        low = str(raw).lower()
        for pat, canon in DEFENSE_PATTERNS:
            if re.search(pat, low):
                return canon, "json"
    low = dirname.lower()
    for pat, canon in DIR_DEFENSE:
        if re.search(pat, low):
            return canon, "dirname"
    return None, "unresolved"


def classify(path: Path, root: Path):
    """Return a record dict, or a reason string on failure."""
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return f"unreadable: {e}"
    if not isinstance(d, dict):
        return "not a dict"

    run_dir = path.parent.name
    raw_model = d.get("model")
    if not raw_model:
        return "no 'model' field"
    model = MODEL_MAP.get(_slug(raw_model), _slug(raw_model))

    subset = d.get("subset")
    if subset:
        benchmark, suite = "bipia", SUITE_MAP.get(_slug(subset), _slug(subset))
    else:
        raw_suite = d.get("suite")
        if not raw_suite:
            return "no 'suite' or 'subset'"
        benchmark, suite = "agentdojo", SUITE_MAP.get(_slug(raw_suite), _slug(raw_suite))

    dir_for_defense = run_dir
    for suf, _ in ATTACK_SUFFIXES:
        if dir_for_defense.lower().endswith(suf):
            dir_for_defense = dir_for_defense[: -len(suf)]
            break
    cond, src = norm_defense(d.get("defense"), dir_for_defense)
    if cond is None:
        return f"unresolved defense (json={d.get('defense')!r}, dir={run_dir})"

    low = run_dir.lower()
    attack = None
    for suf, canon in ATTACK_SUFFIXES:
        if low.endswith(suf):
            attack = canon
            break
    if attack is None:
        if low.startswith("alt_") or low.startswith("smoke_alt_"):
            attack = "ignore_previous"
        else:
            attack = _slug(d.get("attack") or "important_instructions")

    variant = "canonical"
    low = run_dir.lower()
    if low.startswith("srp_intervention"):
        for v in ("historyaware", "history_aware", "structured", "imperative",
                  "repeat", "baseline"):
            if v in low:
                variant = v.replace("history_aware", "historyaware")
                break
        else:
            variant = "intervention_other"

    return {
        "run_dir": run_dir, "model": model, "defense_variant": variant,
        "regenerated": bool(d.get("regenerated")),
        "family": FAMILY.get(model, "unknown"),
        "benchmark": benchmark, "suite": suite,
        "condition": cond, "cond_src": src, "attack": attack,
        "n_reported": d.get("total_scenarios") or d.get("valid_scenarios"),
        "asr": d.get("attack_success_rate"),
        "utility_rate": d.get("utility_rate"),
        "scenarios": d.get("scenarios") or _read_jsonl(path.parent),
        "has_jsonl": (path.parent / "per_scenario.jsonl").exists(),
    }


def _read_jsonl(dirpath: Path):
    """Fallback for runs whose summary.json carries no per-scenario list
    (all 31 BIPIA runs). Returns a list of dicts, or None."""
    f = dirpath / "per_scenario.jsonl"
    if not f.exists():
        return None
    out = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if isinstance(r, dict):
                out.append(r)
        except Exception:
            continue
    return out or None


def _plan_key(row: dict, benchmark: str, sid: str) -> str:
    """Scenario identity as sample_plan.py records it.

    Must match sample_plan.scenario_key exactly or the filter silently drops
    almost everything -- a first attempt reused the dedup tuple here and cut a
    2,400-scenario cell to 6.
    """
    if benchmark == "agentdojo":
        return sid
    uid = row.get("scenario_uid")
    if uid:
        return str(uid)
    a, q = row.get("attack_str"), row.get("question")
    if a is None and q is None:
        return sid
    return hashlib.sha1(f"{a}||{q}".encode()).hexdigest()[:16]

def expand(rec: dict, dedup_bipia: bool = True):
    """One row per trial.

    BIPIA DEDUPLICATION (on by default). BIPIA email holds 75 distinct
    `attack_str` values and 34 distinct `question` values across 3,750 rows;
    75 x 34 = 2,550 distinct content pairs, so ~1,200 rows re-evaluate
    identical content under a constant task_name. Treating them as
    independent inflates effective n and understates every p-value --
    `statistical_analysis.py` correctly used n_pairs=2550. Keep the FIRST
    occurrence of each (attack_str, question) pair; index order is stable
    and verified aligned across condition files.
    """
    rows, scen = [], rec.get("scenarios")
    if not isinstance(scen, list) or not scen:
        return rows, "no per-scenario list"
    surface = f"{rec['benchmark']}/{rec['suite']}"
    plan_keys = SAMPLE_PLAN.get(surface)
    _n500 = SAMPLE_PLAN.get(surface + "_n500")
    if _n500 is not None and len(scen) <= 750:
        plan_keys = _n500
    _n500 = SAMPLE_PLAN.get(surface + "_n500")
    if _n500 is not None and len(scen) <= 750:
        plan_keys = _n500
    seen = set()
    seen_sid = set()
    for i, s in enumerate(scen):
        if dedup_bipia and rec["benchmark"] == "bipia":
            uid = s.get("scenario_uid") or (
                str(s.get("attack_str", "")), str(s.get("question", "")))
            if uid in seen:
                continue
            seen.add(uid)
        if not isinstance(s, dict):
            continue
        sid = (f"{s.get('user_task_id')}|{s.get('injection_task_id')}"
               if "user_task_id" in s
               else str(s.get("scenario_id") or s.get("id") or
                        f"{s.get('attack_name')}|{s.get('question')}" or i))
        succ = s.get("attack_succeeded", s.get("success", s.get("attack_success")))
        util = s.get("task_completed", s.get("utility", s.get("task_completion")))
        if sid in seen_sid:
            continue          # same scenario twice inside one run -- drop
        seen_sid.add(sid)
        if plan_keys is not None and _plan_key(s, rec["benchmark"], sid) \
                not in plan_keys:
            continue          # outside the recorded sample
        rows.append({
            "model": rec["model"], "family": rec["family"],
            "benchmark": rec["benchmark"], "suite": rec["suite"],
            "condition": rec["condition"], "attack": rec["attack"],
            "defense_variant": rec.get("defense_variant", "canonical"),
            "scenario_id": sid,
            "success": int(bool(succ)) if succ is not None else "",
            "utility": int(bool(util)) if util is not None else "",
            "run_dir": rec["run_dir"],
        })
    if plan_keys is not None and rows and len(rows) != len(plan_keys):
        return rows, (f"sample-plan mismatch: kept {len(rows)} of "
                      f"{len(plan_keys)} planned keys")
    return rows, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--rectangle", action="store_true")
    ap.add_argument("--track-a-only", action="store_true",
                    help="restrict output to the 6 Track A models "
                         "(drops kimi_k2, gemma_3_27b, qwen3_next)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep BIPIA duplicate-content rows (NOT recommended; "
                         "inflates effective n -- see audit)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="model slugs to drop, e.g. --exclude kimi_k2 gemma_3_27b")
    a = ap.parse_args()

    root = a.root.expanduser()
    if not root.exists():
        sys.exit(f"root does not exist: {root}")

    # ROOT-LEVEL SUMMARIES ONLY.
    # rglob descends into agentdojo_logs/ and finds NESTED summary.json
    # files. Each nested copy parses to the same (model, suite, condition)
    # cell and was expanded as a second record, DOUBLING rows for affected
    # cells -- haiku/banking/undefended came out at 288 rows for 144
    # scenarios, all from one run_dir, corrupting every rate and delta
    # downstream. Keep only the shallowest summary.json per run directory.
    all_summaries = sorted(root.rglob("summary.json"))
    by_dir: dict = {}
    for f in all_summaries:
        if any(p in ("agentdojo_logs", "logs", "raw", "runs") for p in f.parts):
            continue
        prev = by_dir.get(f.parent)
        if prev is None or len(f.parts) < len(prev.parts):
            by_dir[f.parent] = f
    files = sorted(by_dir.values())
    print(f"scanning {root}")
    print(f"  {len(all_summaries)} summary.json found, {len(files)} kept "
          f"({len(all_summaries) - len(files)} nested/duplicate skipped)\n")

    recs, failures = [], []
    for f in files:
        r = classify(f, root)
        (recs if isinstance(r, dict) else failures).append(
            r if isinstance(r, dict) else (f.relative_to(root), r))

    drop = set(a.exclude)
    if a.track_a_only:
        drop |= {r["model"] for r in recs if r["model"] not in TRACK_A_MODELS}
    if drop:
        before = len(recs)
        recs = [r for r in recs if r["model"] not in drop]
        print(f"  excluded   : {before - len(recs)} runs "
              f"({', '.join(sorted(drop))})")

    print(f"  classified : {len(recs)}")
    print(f"  failed     : {len(failures)}")

    if a.audit or failures:
        print("\n--- distinct values observed ---")
        for field in ("model", "family", "benchmark", "suite", "condition",
                      "defense_variant", "attack"):
            c = Counter(r[field] for r in recs)
            print(f"  {field:11s} {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
        src = Counter(r["cond_src"] for r in recs)
        print(f"  cond_src    {dict(src)}   <- 'dirname' entries are inferred, verify them")
        nojsonl = sum(1 for r in recs if not r["has_jsonl"])
        noscen = sum(1 for r in recs if not isinstance(r.get("scenarios"), list))
        print(f"\n  runs without per_scenario.jsonl : {nojsonl}")
        print(f"  runs without 'scenarios' list   : {noscen}")

    if failures:
        print(f"\n--- UNCLASSIFIED ({len(failures)}) ---")
        byreason = defaultdict(list)
        for p, why in failures:
            byreason[re.sub(r"\(.*\)", "(...)", why)].append(str(p))
        for why, ps in sorted(byreason.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{len(ps):3d}] {why}")
            for p in ps[:4]:
                print(f"        {p}")
        print("\n  -> extend MODEL_MAP / DEFENSE_PATTERNS until this is empty,")
        print("     or confirm these are non-run artefacts and can be ignored.")

    if a.audit:
        return 0

    all_rows, noexp = [], []
    for r in recs:
        rows, why = expand(r, dedup_bipia=not a.no_dedup)
        all_rows.extend(rows)
        if why:
            noexp.append((r["run_dir"], why))

    import collections as _c
    bench_n = _c.Counter(r["benchmark"] for r in all_rows)
    import collections as _cc
    cellcnt = _cc.Counter(
        (r["model"], r["benchmark"], r["suite"], r["condition"], r["attack"],
         r["run_dir"]) for r in all_rows)
    EXP = {("agentdojo", "banking"): 144, ("agentdojo", "slack"): 105,
           ("agentdojo", "travel"): 140, ("agentdojo", "workspace"): 560}
    bad = [(k, v) for k, v in cellcnt.items()
           if EXP.get((k[1], k[2])) and v > EXP[(k[1], k[2])]]
    if bad:
        print(f"\n  !! {len(bad)} run/cell pairs exceed the benchmark grid "
              "-- duplication remains:")
        for (m, b, su, c, at, rd), v in sorted(bad)[:10]:
            print(f"     {m:16s} {b}/{su:10s} {c:18s} n={v} "
                  f"expected {EXP[(b, su)]}  [{rd}]")

    if SAMPLE_PLAN:
        print(f"\n  sample plan active for: {', '.join(sorted(SAMPLE_PLAN))}"
              f"  (cells restricted to the recorded keys)")

    print(f"\n  trial rows extracted : {len(all_rows):,}  "
          f"({dict(bench_n)})")
    if not a.no_dedup and bench_n.get("bipia"):
        print("  BIPIA deduplicated to distinct (attack_str, question) pairs "
              "-- expect ~2,550/cell, not 3,750")
    if noexp:
        print(f"  runs with no expandable trials: {len(noexp)}")
        for rd, why in noexp[:8]:
            print(f"     {rd}: {why}")

    if a.rectangle:
        nmap = {}
        for r in recs:
            k = (r["model"], f"{r['benchmark']}/{r['suite']}", r["condition"])
            nmap[k] = max(nmap.get(k, 0), r.get("n_reported") or 0)
        have = set(nmap)
        need = [(m, s, c) for m in TRACK_A_MODELS for s in TRACK_A_SURFACES
                for c in (BASE_CONDS + (["domain_control"]
                                        if s == "agentdojo/banking" else []))]
        missing = [x for x in need if x not in have]
        print("\n" + "=" * 62)
        print(f"TRACK A RECTANGLE   {len(need)-len(missing)}/{len(need)} cells "
              f"({100*(len(need)-len(missing))/len(need):.0f}%)")
        print("=" * 62)
        for m in TRACK_A_MODELS:
            h = sum(1 for x in need if x[0] == m and x in have)
            n = sum(1 for x in need if x[0] == m)
            print(f"  {m:18s}{h:>4d}/{n:<4d}")
        print()
        for c in BASE_CONDS + ["domain_control"]:
            h = sum(1 for x in need if x[2] == c and x in have)
            n = sum(1 for x in need if x[2] == c)
            print(f"  {c:20s}{h:>4d}/{n:<4d}")
        prio = {"sham_datamarking": 0, "domain_control": 1, "neutral_preamble": 2}
        tot = 0
        print(f"\n  MISSING ({len(missing)}), run order:")
        for i, (m, s, c) in enumerate(sorted(
                missing, key=lambda x: (prio.get(x[2], 9), x[1], x[0])), 1):
            n = TRACK_A_SURFACES[s]
            tot += n
            print(f"  {i:4d}. {c:18s} {m:18s} {s:20s} n={n}")
        print(f"\n  new trials required: {tot:,}")

        thin = [(k, nmap[k], TRACK_A_SURFACES[k[1]]) for k in nmap
                if k in set(need) and nmap[k] < 0.5 * TRACK_A_SURFACES[k[1]]]
        if thin:
            print(f"\n  !! UNDERSIZED CELLS ({len(thin)}) -- counted as covered "
                  "above but far below target.\n     Almost certainly --limit "
                  "smoke runs; re-run at full n before analysis:")
            for (m, srf, c), got, need_n in sorted(thin):
                print(f"       {c:18s} {m:18s} {srf:20s} n={got} (need {need_n})")

    if a.out:
        import csv
        cols = ["model", "family", "benchmark", "suite", "condition",
                "defense_variant", "attack", "scenario_id", "success",
                "utility", "run_dir"]
        with a.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nwrote {a.out}  ({len(all_rows):,} rows)")
        print(f"next: python hierarchical_refit.py --data {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
verify_bipia_key_integrity.py
================================
Checks whether the pairing key used in verify_bipia_numbers.py
-- (task_name, attack_name, position) -- is actually valid:

  1. Does it produce exactly 3750 UNIQUE keys within a single condition
     file? (If fewer, the key is causing collisions -- distinct scenarios
     silently overwriting each other in the dict.)
  2. Do the key SETS match exactly between "none" and "srp" conditions?
     (The McNemar function's "common = set(a) & set(b)" relies on this;
     if the sets don't match, McNemar ran on a smaller/wrong subset.)
  3. Compare against alternate keys (adding "question") to see which
     is actually correct.

USAGE: run from /home/ubuntu/mob/
  python3 verify_bipia_key_integrity.py
"""
import json
from pathlib import Path

FILES = {
    "none": "msm_exp8_haiku_email_none",
    "srp": "msm_exp8_haiku_email_srp",
}

def load_rows(dirname):
    p = Path(dirname) / "per_scenario.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def check_key(rows, key_fields, label):
    keys = [tuple(r.get(f) for f in key_fields) for r in rows]
    n_total = len(keys)
    n_unique = len(set(keys))
    print(f"  {label}: {n_unique} unique / {n_total} total "
          f"{'(COMPLETE, no collisions)' if n_unique == n_total else '*** COLLISIONS -- key is invalid ***'}")
    return set(keys)

print("="*100)
print("STEP 1: is the ORIGINAL key (task_name, attack_name, position) unique within each file?")
print("="*100)
key_sets = {}
for cond, dirname in FILES.items():
    rows = load_rows(dirname)
    print(f"\n--- {cond} ({dirname}) ---")
    key_sets[cond] = check_key(rows, ["task_name", "attack_name", "position"], "original key (task_name, attack_name, position)")
    check_key(rows, ["question", "attack_name", "position"], "alternate key (question, attack_name, position)")
    check_key(rows, ["task_name", "question", "attack_name", "position"], "combined key (task_name, question, attack_name, position)")

print("\n" + "="*100)
print("STEP 2: do the ORIGINAL key sets MATCH between 'none' and 'srp'?")
print("="*100)
common = key_sets["none"] & key_sets["srp"]
only_none = key_sets["none"] - key_sets["srp"]
only_srp = key_sets["srp"] - key_sets["none"]
print(f"  Keys in both: {len(common)}")
print(f"  Keys only in 'none': {len(only_none)}")
print(f"  Keys only in 'srp': {len(only_srp)}")
if len(common) < 3750:
    print(f"  *** McNemar's 'common' set was {len(common)}, NOT the full 3750 -- ")
    print(f"      the earlier p-values were computed on a SUBSET, not the full data. ***")
else:
    print("  Full 3750 pairing confirmed intact.")

print("\nCopy this ENTIRE output back.")

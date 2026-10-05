#!/usr/bin/env python3
"""How repeatable is one evaluation? Compare the first evaluation of seven checkpoints (data/eval_*.log, from
2026-10-04) with the labelled re-evaluation (outputs/eval_labelled/*/eval_info.json, 2026-10-05).

Both are 50 rollouts, batch_size 1. The old logs only keep aggregates (avg_max_reward, pc_success).
Writes outputs/eval_repeatability.txt (a snapshot is committed under results/).
"""
import json
import re
import statistics as st

PAIRS = {"clean_s0": "data/eval_clean_s0_60k.log", "clean_s1": "data/eval_clean_s1.log", "clean_s2": "data/eval_clean_s2.log",
         "corrupted_s0": "data/eval_corrupted_s0.log", "corrupted_s1": "data/eval_corrupted_s1.log",
         "cleaned_s0": "data/eval_cleaned_s0.log", "cleaned_s1": "data/eval_cleaned_s1.log"}
rows = []
for k, f in PAIRS.items():
    t = open(f, errors="ignore").read()
    mr = re.findall(r"avg_max_reward'?\"?: ([0-9.]+)", t)
    sc = re.findall(r"pc_success'?\"?: ([0-9.]+)", t)
    m = json.load(open(f"outputs/eval_labelled/{k}/eval_info.json"))["per_task"][0]["metrics"]
    rows.append((k, float(mr[-1]), st.mean(m["max_rewards"]), float(sc[-1]), 100 * sum(1 for s in m["successes"] if s) / len(m["successes"])))
out = ["checkpoint     old_max_reward new_max_reward  diff   old_succ% new_succ%"]
out += [f"{k:13} {o:14.3f} {n:14.3f} {n - o:+7.3f} {os_:9.1f} {ns:9.1f}" for k, o, n, os_, ns in rows]
d = [abs(n - o) for _, o, n, _, _ in rows]
out.append(f"\nn={len(rows)} checkpoints evaluated twice (50 rollouts each, batch_size 1)")
out.append(f"|diff| in mean max reward: mean {st.mean(d):.3f}, max {max(d):.3f}; success-rate changed for {sum(1 for r in rows if r[3] != r[4])} of {len(rows)}")
print("\n".join(out))
open("outputs/eval_repeatability.txt", "w").write("\n".join(out) + "\n")

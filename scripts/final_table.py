#!/usr/bin/env python3
"""Final DemoDoctor table: per-policy results, per-condition bootstrap CIs, pairwise differences.

Reads outputs/eval_labelled/<condition>_s<seed>/eval_info.json (from eval_all.py). The unit of
replication is the policy (seed), not the rollout, so the bootstrap resamples policies and then
rollouts within them. With 3 seeds per condition the minimum detectable effect is large; the
script prints it.

Usage: python scripts/final_table.py
"""
import glob
import json
import random
import re
from collections import defaultdict
from statistics import mean

random.seed(0)
ORDER = ["clean", "corrupted", "cleaned", "random175"]
runs = defaultdict(dict)  # cond -> seed -> dict(max_rewards, successes)
for f in sorted(glob.glob("outputs/eval_labelled/*/eval_info.json")):
    m = re.search(r"/(\w+)_s(\d)/eval_info.json", f)
    cond, seed = m.group(1), int(m.group(2))
    mt = json.load(open(f))["per_task"][0]["metrics"]
    runs[cond][seed] = {"max_rewards": mt["max_rewards"], "successes": [1 if s else 0 for s in mt["successes"]]}

def boot(cond, key, n=4000):
    seeds = list(runs[cond])
    out = []
    for _ in range(n):
        pol = [random.choice(seeds) for _ in seeds]
        vals = []
        for s in pol:
            xs = runs[cond][s][key]
            vals.append(mean(random.choice(xs) for _ in xs))
        out.append(mean(vals))
    out.sort()
    return mean(mean(runs[cond][s][key]) for s in seeds), out[int(0.025 * n)], out[int(0.975 * n)]

print(f"{'condition':10} {'seed':>4} {'success %':>10} {'max reward':>11}")
for c in ORDER:
    for s in sorted(runs.get(c, {})):
        r = runs[c][s]
        print(f"{c:10} {s:4d} {100*mean(r['successes']):10.1f} {mean(r['max_rewards']):11.3f}")
print("\nper condition (mean over policies, 95% bootstrap CI over policies and rollouts)")
res = {}
for c in ORDER:
    if c not in runs:
        continue
    sr, sl, sh = boot(c, "successes"); rr, rl, rh = boot(c, "max_rewards")
    res[c] = (rr, rl, rh)
    print(f"{c:10} n_policies={len(runs[c])}  success {100*sr:4.1f}% [{100*sl:4.1f}, {100*sh:4.1f}]   max reward {rr:.3f} [{rl:.3f}, {rh:.3f}]")
print("\npairwise difference in mean max reward (A - B), bootstrap 95% CI")
def diff(a, b, n=4000):
    out = []
    for _ in range(n):
        va = mean(mean(random.choice(runs[a][s]['max_rewards']) for _ in runs[a][s]['max_rewards']) for s in [random.choice(list(runs[a])) for _ in runs[a]])
        vb = mean(mean(random.choice(runs[b][s]['max_rewards']) for _ in runs[b][s]['max_rewards']) for s in [random.choice(list(runs[b])) for _ in runs[b]])
        out.append(va - vb)
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]
for a, b in [("clean", "corrupted"), ("corrupted", "cleaned"), ("cleaned", "random175"), ("corrupted", "random175"), ("clean", "cleaned")]:
    if a in runs and b in runs:
        d = res[a][0] - res[b][0]; lo, hi = diff(a, b)
        print(f"{a:10} - {b:10} {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {'(CI excludes 0)' if lo > 0 or hi < 0 else '(CI includes 0)'}")

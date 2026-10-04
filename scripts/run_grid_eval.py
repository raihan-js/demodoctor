#!/usr/bin/env python3
"""Evaluate all policy checkpoints: 50 rollouts each, collect success + reward.

Grid: 3 conditions (clean/corrupted/cleaned) x 3 seeds at fixed step budget.
Writes data/results/policy_grid.json with per-run pc_success and avg_max_reward.

Usage:
  python scripts/run_grid_eval.py --steps 60000
"""
import argparse
import json
import re
import subprocess
from pathlib import Path


CONDITIONS = {
    "clean": "data/policy_clean_s{seed}",
    "corrupted": "data/policy_corrupted_s{seed}",
    "cleaned": "data/policy_cleaned_s{seed}",
}


def eval_one(checkpoint: str, n_episodes: int = 50) -> dict:
    out = subprocess.run(
        [".venv/bin/lerobot-eval",
         f"--policy.path={checkpoint}/checkpoints/last/pretrained_model",
         "--env.type=pusht", f"--eval.n_episodes={n_episodes}",
         "--eval.batch_size=1"],
        capture_output=True, text=True, timeout=3600,
    )
    log = out.stdout + out.stderr
    m_success = re.search(r"'pc_success': ([\d.]+)", log)
    m_reward = re.search(r"'avg_max_reward': ([\d.]+)", log)
    return {
        "pc_success": float(m_success.group(1)) if m_success else None,
        "avg_max_reward": float(m_reward.group(1)) if m_reward else None,
        "returncode": out.returncode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="data/results/policy_grid.json")
    args = ap.parse_args()

    results = []
    for condition, template in CONDITIONS.items():
        for seed in args.seeds:
            ckpt = template.format(seed=seed)
            print(f"=== {condition} seed {seed} ===", flush=True)
            try:
                r = eval_one(ckpt, args.episodes)
            except Exception as e:
                r = {"pc_success": None, "avg_max_reward": None, "error": str(e)[:200]}
            r.update({"condition": condition, "seed": seed, "steps": args.steps})
            print(f"  success={r['pc_success']} max_reward={r['avg_max_reward']}", flush=True)
            results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")

    print(f"\n{'condition':12s} {'seed':>5s} {'success':>8s} {'max_rew':>8s}")
    for r in results:
        print(f"{r['condition']:12s} {r['seed']:5d} "
              f"{str(r['pc_success']):>8s} {str(r['avg_max_reward']):>8s}")


if __name__ == "__main__":
    main()

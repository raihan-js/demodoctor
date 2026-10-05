#!/usr/bin/env python3
"""Evaluate every available policy checkpoint with labelled output (50 rollouts each).

Writes outputs/eval_labelled/<condition>_s<seed>/eval_info.json (lerobot-eval --output_dir), so each
result is tied to a condition and seed. Skips policies whose 60k checkpoint is missing.

Usage:  python scripts/eval_all.py [--episodes 50] [--only random175 clean]
Eval must use --eval.batch_size=1 (async workers lose gym_pusht registration).
"""
import argparse
import subprocess
import sys
from pathlib import Path

POLICIES = {  # condition -> checkpoint dir template (clean seed 0 is the 60k run)
    "clean": ["data/policy_clean_s0_60k", "data/policy_clean_s1", "data/policy_clean_s2"],
    "corrupted": ["data/policy_corrupted_s0", "data/policy_corrupted_s1", "data/policy_corrupted_s2"],
    "cleaned": ["data/policy_cleaned_s0", "data/policy_cleaned_s1", "data/policy_cleaned_s2"],
    "random175": ["data/policy_random175_s0", "data/policy_random175_s1", "data/policy_random175_s2"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for cond, ckpts in POLICIES.items():
        if args.only and cond not in args.only:
            continue
        for seed, ck in enumerate(ckpts):
            model = Path(ck) / "checkpoints" / "060000" / "pretrained_model"
            out = Path(f"outputs/eval_labelled/{cond}_s{seed}")
            if not model.exists():
                print(f"SKIP {cond} s{seed}: no 60k checkpoint at {ck}", flush=True)
                continue
            if (out / "eval_info.json").exists():
                print(f"HAVE {cond} s{seed}", flush=True)
                continue
            print(f"=== {cond} s{seed} ===", flush=True)
            r = subprocess.run([".venv/bin/lerobot-eval", f"--policy.path={model}", "--env.type=pusht",
                                f"--eval.n_episodes={args.episodes}", "--eval.batch_size=1",
                                f"--output_dir={out}"], capture_output=True, text=True, timeout=3600)
            print(f"  returncode {r.returncode}", flush=True)
            if r.returncode != 0:
                print(r.stderr[-400:], file=sys.stderr)


if __name__ == "__main__":
    main()

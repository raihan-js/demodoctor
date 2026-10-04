#!/usr/bin/env python3
"""Build a size-matched random subset of the corrupted set (175 episodes).

Control for the cleaning experiment: if random-175 performs like cleaned-175,
the cleaning deficit is data SIZE, not quality. If it performs like the full
corrupted-206 set, size doesn't matter and cleaning genuinely hurts.

Usage:
  PYTHONPATH=src python scripts/build_random_subset.py
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="data/pusht_corrupted/corrupted")
    ap.add_argument("--src-repo", default="raihan-js/demodoctor-pusht-corrupted")
    ap.add_argument("--out-root", default="data/pusht_random175/random175")
    ap.add_argument("--out-repo", default="raihan-js/demodoctor-pusht-random175")
    ap.add_argument("--n", type=int, default=175)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "scripts")
    from build_corrupted import read_episode, write_episode

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(args.src_repo, root=args.src_root)
    rng = np.random.default_rng(args.seed)
    keep = sorted(rng.choice(src.num_episodes, size=args.n, replace=False).tolist())
    print(f"Keeping {len(keep)}/{src.num_episodes} random episodes")

    new_ds = LeRobotDataset.create(
        repo_id=args.out_repo, fps=10, features=src.features,
        root=args.out_root, use_videos=True)

    manifest = json.loads(Path(args.src_root, "manifest.json").read_text())
    new_manifest = {}
    for new_ep, ep in enumerate(keep):
        data = read_episode(src, ep)
        T = len(data["frames"])
        write_episode(new_ds, new_ep, data["frames"], data["states"],
                      data["actions"],
                      data["rewards"][:T], data["dones"][:T],
                      data["task"], success=data["success"])
        new_ds.save_episode()
        new_manifest[str(new_ep)] = manifest[str(ep)]
        if (new_ep + 1) % 25 == 0:
            print(f"  {new_ep + 1}/{len(keep)}", flush=True)

    new_ds.finalize()
    Path(args.out_root, "manifest.json").write_text(json.dumps(new_manifest, indent=2))
    print(f"Saved random subset to {args.out_root}")


if __name__ == "__main__":
    main()

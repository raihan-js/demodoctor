#!/usr/bin/env python3
"""Build the cleaned dataset: drop episodes flagged by reliable detectors.

Cleaning rule (documented, fixed before seeing policy results): drop an
episode if the stall detector OR the jitter detector flags it. Lag,
truncation and dup detectors are excluded: their precision is too low
(honest nulls reported in detector_scores.json) and using them would gut
the dataset with false positives.

Usage:
  PYTHONPATH=src python scripts/build_cleaned.py
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="data/pusht_corrupted/corrupted")
    ap.add_argument("--src-repo", default="raihan-js/demodoctor-pusht-corrupted")
    ap.add_argument("--out-root", default="data/pusht_cleaned/cleaned")
    ap.add_argument("--out-repo", default="raihan-js/demodoctor-pusht-cleaned")
    ap.add_argument("--video-backend", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "scripts")
    from score_detectors import load_episode_arrays

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from demodoctor.detectors import detect_jitter, detect_stall

    load_kw = {}
    if args.video_backend:
        load_kw["video_backend"] = args.video_backend
    src = LeRobotDataset(args.src_repo, root=args.src_root, **load_kw)
    manifest = json.loads(Path(args.src_root, "manifest.json").read_text())

    # decide drops
    keep, dropped = [], []
    for ep_str, entry in sorted(manifest.items(), key=lambda kv: int(kv[0])):
        ep = int(ep_str)
        data = load_episode_arrays(src, ep)
        _, jit = detect_jitter(data["actions"])
        _, stall, _ = detect_stall(data["actions"])
        (dropped if (jit or stall) else keep).append(ep)

    print(f"Keeping {len(keep)}/{len(manifest)} episodes "
          f"(dropping {len(dropped)} stall/jitter-flagged)", flush=True)
    n_truly_faulty = sum(1 for ep in dropped
                         if manifest[str(ep)]["faults"])
    print(f"  of dropped, truly faulty: {n_truly_faulty}/{len(dropped)}", flush=True)

    # write kept episodes to a new dataset
    create_kw = dict(
        repo_id=args.out_repo, fps=10, features=src.features,
        root=args.out_root, use_videos=True)
    if args.video_backend:
        create_kw["video_backend"] = args.video_backend
    new_ds = LeRobotDataset.create(**create_kw)
    import sys as _sys
    _sys.path.insert(0, "scripts")
    from build_corrupted import write_episode

    new_manifest = {}
    for new_ep, ep in enumerate(keep):
        meta = src.meta.episodes[ep]
        s, e = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        rows = [src[i] for i in range(s, e)]
        frames = np.stack([np.asarray(r["observation.image"]) for r in rows])
        if frames.shape[1] == 3:
            frames = frames.transpose(0, 2, 3, 1)
        if frames.dtype != np.uint8:
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        states = np.stack([np.asarray(r["observation.state"]) for r in rows])
        actions = np.stack([np.asarray(r["action"]) for r in rows])
        rewards = np.array([float(r["next.reward"]) for r in rows])
        dones = np.array([bool(r["next.done"]) for r in rows])
        success = bool(rows[-1]["next.success"])
        task = rows[0]["task"] if "task" in rows[0] else "Push the T block"
        write_episode(new_ds, new_ep, frames, states, actions, rewards, dones,
                      task, success=success)
        new_ds.save_episode()
        new_manifest[str(new_ep)] = manifest[str(ep)]
        if (new_ep + 1) % 25 == 0:
            print(f"  {new_ep + 1}/{len(keep)}", flush=True)

    new_ds.finalize()
    Path(args.out_root, "manifest.json").write_text(json.dumps(new_manifest, indent=2))
    print(f"Saved cleaned set to {args.out_root}", flush=True)


if __name__ == "__main__":
    main()

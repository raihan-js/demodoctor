#!/usr/bin/env python3
"""Score label-free detectors against the ground-truth fault manifest.

For each fault type: precision, recall, F1 of the corresponding detector.
Detectors never see the manifest — this script is the first thing that does.

Usage:
  PYTHONPATH=src python scripts/score_detectors.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from demodoctor.detectors import (action_speed, detect_drop_frames,
                                   detect_duplicate_frames, detect_frame_gaps,
                                   detect_jitter, detect_stall,
                                   detect_truncation, estimate_lag_stable,
                                   motion_energy, progress_monotonicity)


def load_episode_arrays(ds, ep_idx: int) -> dict:
    meta = ds.meta.episodes[ep_idx]
    s, e = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    rows = [ds[i] for i in range(s, e)]
    imgs = np.stack([np.asarray(r["observation.image"]) for r in rows])
    # to HWC uint8
    if imgs.shape[1] == 3:
        imgs = imgs.transpose(0, 2, 3, 1)
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
    return {
        "frames": imgs,
        "actions": np.stack([np.asarray(r["action"]) for r in rows]),
        "timestamps": np.array([float(r["timestamp"]) for r in rows]),
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pusht_corrupted/corrupted")
    ap.add_argument("--repo-id", default="raihan-js/demodoctor-pusht-corrupted")
    ap.add_argument("--out", default="data/results/detector_scores.json")
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_TOKEN", "")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root)
    manifest = json.loads(Path(args.root, "manifest.json").read_text())

    # corpus length statistics for the truncation detector (label-free)
    lengths = []
    for ep_str in manifest:
        meta = ds.meta.episodes[int(ep_str)]
        lengths.append(int(meta["dataset_to_index"]) - int(meta["dataset_from_index"]))
    lengths = np.array(lengths)
    med_len = float(np.median(lengths))
    mad_len = float(np.median(np.abs(lengths - med_len)))

    # per-fault tallies: detector -> {tp, fp, fn}
    tallies = {f: {"tp": 0, "fp": 0, "fn": 0}
               for f in ("lag", "jitter", "stall", "truncation", "dup_frames")}

    for ep_str, entry in sorted(manifest.items(), key=lambda kv: int(kv[0])):
        ep = int(ep_str)
        data = load_episode_arrays(ds, ep)
        truth = {f["type"] for f in entry["faults"]}

        est_lag, lag_conf = estimate_lag_stable(data["frames"], data["actions"])
        _, jit_flag = detect_jitter(data["actions"])
        _, stall_flag, _ = detect_stall(data["actions"])
        _, trunc_flag = detect_truncation(len(data["frames"]), med_len, mad_len)
        _, dup_flag, _ = detect_duplicate_frames(data["frames"])

        preds = {
            "lag": lag_conf,
            "jitter": jit_flag,
            "stall": stall_flag,
            "truncation": trunc_flag,
            "dup_frames": dup_flag,
        }
        for fault in tallies:
            pred, actual = preds[fault], fault in truth
            if pred and actual:
                tallies[fault]["tp"] += 1
            elif pred and not actual:
                tallies[fault]["fp"] += 1
            elif not pred and actual:
                tallies[fault]["fn"] += 1

    out = {f: prf(**t) for f, t in tallies.items()}
    print(f"{'fault':12s} {'P':>6s} {'R':>6s} {'F1':>6s} {'tp':>4s} {'fp':>4s} {'fn':>4s}")
    for f, s in out.items():
        print(f"{f:12s} {s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f} "
              f"{s['tp']:4d} {s['fp']:4d} {s['fn']:4d}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

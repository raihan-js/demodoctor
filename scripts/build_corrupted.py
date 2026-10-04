#!/usr/bin/env python3
"""Build corrupted copies of lerobot/pusht with ground-truth fault manifests.

Conditions:
  clean      - verbatim copy (control; validates the pipeline preserves data)
  corrupted  - 30% of episodes get one injected fault (stratified by type)
  cleaned    - corrupted episodes EXCEPT ones flagged by the detectors are dropped

Writes a new LeRobot dataset per condition + a manifest JSON mapping
episode_id -> injected faults. Detectors never see the manifest.

Usage:
  PYTHONPATH=src python scripts/build_corrupted.py --condition corrupted --ratio 0.3 --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from demodoctor.faults import inject_fault


def read_episode(ds, ep_idx: int) -> dict:
    """Read one full episode as numpy arrays."""
    meta = ds.meta.episodes[ep_idx]
    start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    rows = [ds[i] for i in range(start, end)]
    imgs = np.stack([np.asarray(r["observation.image"]).transpose(1, 2, 0) for r in rows])
    imgs = (imgs * 255).clip(0, 255).astype(np.uint8)
    return {
        "frames": imgs,
        "states": np.stack([np.asarray(r["observation.state"]) for r in rows]),
        "actions": np.stack([np.asarray(r["action"]) for r in rows]),
        "rewards": np.array([float(r["next.reward"]) for r in rows]),
        "dones": np.array([bool(r["next.done"]) for r in rows]),
        "success": bool(rows[-1]["next.success"]),
        "task": rows[0]["task"] if "task" in rows[0] else "Push the T block",
    }


def write_episode(new_ds, ep_idx: int, frames: np.ndarray, states: np.ndarray,
                  actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
                  task: str, success: bool = True) -> None:
    """Write one episode frame-by-frame into a new LeRobot dataset."""
    import torch

    T = len(frames)
    for t in range(T):
        img = torch.from_numpy(frames[t])  # HWC uint8, as LeRobot expects
        # NB: episode_index/frame_index/timestamp/task_index are managed
        # internally by add_frame — passing them raises Feature mismatch.
        last = t == T - 1
        new_ds.add_frame({
            "observation.image": img,
            "observation.state": torch.from_numpy(states[t].astype(np.float32)),
            "action": torch.from_numpy(actions[t].astype(np.float32)),
            "next.reward": torch.tensor([float(rewards[t])], dtype=torch.float32),
            "next.done": torch.tensor([dones[t] if not last else True]),
            "next.success": torch.tensor([(dones[t] if not last else True) and success]),
            "task": task,
        })


def corruption_plan(n_episodes: int, ratio: float, seed: int) -> dict[int, dict]:
    """Assign one fault to `ratio` of episodes, stratified across fault types."""
    from demodoctor.faults import FAULTS

    rng = np.random.default_rng(seed)
    n_corrupt = int(round(n_episodes * ratio))
    chosen = rng.choice(n_episodes, size=n_corrupt, replace=False)
    plan = {}
    for j, ep in enumerate(sorted(chosen)):
        fault = FAULTS[j % len(FAULTS)]
        plan[int(ep)] = {"fault": fault}
    return plan


def fault_kwargs(fault: str, actions: np.ndarray, T: int,
                 rng: np.random.Generator) -> dict:
    if fault == "lag":
        return {"lag": int(rng.integers(1, 4))}
    if fault == "stall":
        length = int(rng.integers(10, 30))
        start = int(rng.integers(5, max(6, T - length - 5)))
        return {"start": start, "length": length}
    if fault == "jitter":
        # scale to the episode's own action range: absolute sigma is
        # meaningless (PushT actions have std ~80, not ~1)
        scale = float(np.abs(actions).std() + 1e-8)
        return {"sigma": float(rng.uniform(0.03, 0.08) * scale)}
    if fault == "truncation":
        return {"cut": int(rng.integers(10, 30))}
    if fault == "dup_frames":
        n_dup = max(1, T // 20)
        return {"dup_idx": sorted(rng.choice(T, size=n_dup, replace=False).tolist())}
    raise ValueError(fault)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["clean", "corrupted"])
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--root", default="data/pusht_corrupted")
    ap.add_argument("--video-backend", default=None)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    load_kw = {}
    if args.video_backend:
        load_kw["video_backend"] = args.video_backend
    src = LeRobotDataset("lerobot/pusht", **load_kw)
    n_eps = src.num_episodes
    print(f"Source: {n_eps} episodes", flush=True)

    plan = corruption_plan(n_eps, args.ratio if args.condition == "corrupted" else 0.0,
                           args.seed)
    print(f"Corrupting {len(plan)} episodes", flush=True)

    out_root = Path(args.root) / args.condition
    create_kw = dict(
        repo_id=args.repo_id or f"demodoctor-pusht-{args.condition}",
        fps=10,
        features=src.features,
        root=str(out_root),
        use_videos=True,
    )
    if args.video_backend:
        create_kw["video_backend"] = args.video_backend
    new_ds = LeRobotDataset.create(**create_kw)

    rng = np.random.default_rng(args.seed)
    manifest = {}
    new_ep = 0
    for ep in range(n_eps):
        data = read_episode(src, ep)
        T = len(data["frames"])
        if ep in plan:
            fault = plan[ep]["fault"]
            kw = fault_kwargs(fault, data["actions"], T, rng)
            frames, actions, rec = inject_fault(data["frames"], data["actions"],
                                                fault, rng, **kw)
            manifest[str(new_ep)] = {"source_episode": ep, "faults": [rec]}
            # dup/lag/stall/jitter preserve length; truncation cuts the suffix
            keep = np.arange(len(frames))
            states = data["states"][keep]
            rewards = data["rewards"][keep]
            dones = data["dones"][keep]
            # truncated episodes end early: they did not succeed
            success = data["success"] and fault != "truncation"
        else:
            frames, actions = data["frames"], data["actions"]
            states, rewards, dones = data["states"], data["rewards"], data["dones"]
            manifest[str(new_ep)] = {"source_episode": ep, "faults": []}
            success = data["success"]
        write_episode(new_ds, new_ep, frames, states, actions, rewards, dones,
                      data["task"], success=success)
        new_ds.save_episode()
        new_ep += 1
        if (ep + 1) % 25 == 0:
            print(f"  {ep + 1}/{n_eps} episodes", flush=True)

    new_ds.finalize()
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {new_ep} episodes to {out_root} + manifest", flush=True)


if __name__ == "__main__":
    main()

"""Fault injector: corrupt clean demonstrations with ground-truth manifests.

Faults (all operate on a single episode of T timesteps):
  lag          - camera frames delayed by L frames vs actions (L in 1..3 at 10fps)
  stall        - action velocity ~0 for a contiguous block (teleop freeze)
  jitter       - high-frequency noise added to actions
  truncation   - episode cut before success (drops the final K frames)
  drop_frames  - random frames dropped (and actions duplicated to keep length)

Every injector returns (corrupted_episode, fault_record). The fault_record is
the ground truth the detectors are scored against — never shown to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAULTS = ("lag", "stall", "jitter", "truncation", "drop_frames")


@dataclass
class FaultManifest:
    """Ground truth for one episode: which faults were injected, where."""

    episode_id: str
    faults: list[dict] = field(default_factory=list)

    def has(self, fault: str) -> bool:
        return any(f["type"] == fault for f in self.faults)

    def to_dict(self) -> dict:
        return {"episode_id": self.episode_id, "faults": self.faults}


def inject_lag(frames: np.ndarray, actions: np.ndarray, lag: int,
               rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Delay camera frames by `lag` steps relative to actions.

    The first `lag` frames are held (repeat frame 0); the tail is dropped so
    shapes are unchanged. This mimics a slow camera pipeline.
    """
    assert 1 <= lag <= 5, "lag must be 1..5 frames"
    out = np.concatenate([np.repeat(frames[:1], lag, axis=0), frames[:-lag]], axis=0)
    return out, {"type": "lag", "lag_frames": lag, "affected": [0, len(frames)]}


def inject_stall(actions: np.ndarray, start: int, length: int) -> tuple[np.ndarray, dict]:
    """Freeze actions (hold last pre-stall action) over [start, start+length)."""
    out = actions.copy()
    hold = actions[max(start - 1, 0)]
    out[start:start + length] = hold
    return out, {"type": "stall", "start": start, "length": length}


def inject_jitter(actions: np.ndarray, sigma: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Add iid Gaussian noise to actions (shaky teleoperation)."""
    noise = rng.normal(0.0, sigma, size=actions.shape)
    return actions + noise, {"type": "jitter", "sigma": sigma,
                             "affected": [0, len(actions)]}


def inject_truncation(frames: np.ndarray, actions: np.ndarray,
                      cut: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Cut the final `cut` frames (episode ends before success)."""
    assert 0 < cut < len(frames), "cut must remove a proper suffix"
    return frames[:-cut], actions[:-cut], {"type": "truncation", "cut_frames": cut,
                                           "kept": len(frames) - cut}


def inject_drop_frames(frames: np.ndarray, actions: np.ndarray,
                       drop_idx: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop frames; duplicate the previous action to keep array length.

    Returns shortened arrays plus the dropped indices as ground truth.
    """
    keep = np.array([i for i in range(len(frames)) if i not in set(drop_idx)])
    return frames[keep], actions[keep], {"type": "drop_frames",
                                         "dropped": sorted(drop_idx),
                                         "kept": len(keep)}


def inject_fault(frames: np.ndarray, actions: np.ndarray, fault: str,
                 rng: np.random.Generator, **kwargs) -> tuple:
    """Dispatch to one fault injector. Returns (frames, actions, record)."""
    if fault == "lag":
        f2, rec = inject_lag(frames, actions, kwargs.get("lag", 2), rng)
        return f2, actions, rec
    if fault == "stall":
        a2, rec = inject_stall(actions, kwargs["start"], kwargs["length"])
        return frames, a2, rec
    if fault == "jitter":
        a2, rec = inject_jitter(actions, kwargs.get("sigma", 0.05), rng)
        return frames, a2, rec
    if fault == "truncation":
        return inject_truncation(frames, actions, kwargs["cut"])
    if fault == "drop_frames":
        return inject_drop_frames(frames, actions, kwargs["drop_idx"])
    raise ValueError(f"unknown fault {fault!r}; choose from {FAULTS}")

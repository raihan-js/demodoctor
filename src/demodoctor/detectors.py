"""Label-free fault detectors: signals only, no ground truth.

Three detectors, one shared contract:

    score, decision = detect_*(signals...) -> (float in [0, 1], bool)

  lag_estimator   - cross-correlate image-motion energy with action-velocity
                    energy across candidate lags; the argmax lag is the
                    estimated camera delay. Score = peak prominence.
  jerk_stall      - jerk (diff of acceleration) flags jitter; long
                    near-zero-velocity runs flag stalls.
  progress_score  - project frame embeddings onto the start→goal direction;
                    non-monotonic backtracking flags truncated or looping demos.

Frame embeddings come from DINOv2-small in the integration scripts; the
detectors themselves only ever see numpy arrays, which is what makes them
unit-testable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate


def motion_energy(frames: np.ndarray) -> np.ndarray:
    """Per-timestep mean absolute frame difference. Shape (T,)."""
    diff = np.abs(frames.astype(np.float32)[1:] - frames.astype(np.float32)[:-1])
    return np.concatenate([[0.0], diff.mean(axis=tuple(range(1, diff.ndim)))])


def action_speed(actions: np.ndarray) -> np.ndarray:
    """Per-timestep L2 action velocity. Shape (T,)."""
    vel = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    return np.concatenate([[0.0], vel])


def estimate_lag(frames: np.ndarray, actions: np.ndarray,
                 max_lag: int = 5) -> tuple[int, float]:
    """Estimate camera-vs-action lag by cross-correlation.

    Returns (best_lag, prominence) where prominence is the peak's margin over
    the mean correlation, normalised to [0, 1].
    """
    m = motion_energy(frames)
    v = action_speed(actions)
    m = (m - m.mean()) / (m.std() + 1e-8)
    v = (v - v.mean()) / (v.std() + 1e-8)
    corrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corrs.append(float(np.dot(m, v) / len(m)))
        else:
            corrs.append(float(np.dot(m[lag:], v[:-lag]) / (len(m) - lag)))
    corrs = np.array(corrs)
    best = int(np.argmax(corrs))
    prominence = float((corrs[best] - corrs.mean()) / (corrs.max() - corrs.min() + 1e-8))
    return best, max(0.0, min(1.0, prominence))


def jerk(actions: np.ndarray) -> np.ndarray:
    """Per-timestep jerk magnitude (diff of acceleration). Shape (T,)."""
    acc = np.diff(actions, axis=0)
    jerk_v = np.linalg.norm(np.diff(acc, axis=0), axis=1)
    return np.concatenate([[0.0, 0.0], jerk_v])


def detect_jitter(actions: np.ndarray, k: float = 5.0) -> tuple[float, bool]:
    """Flag jitter by comparing raw jerk against median-filtered jerk.

    A median filter (window 5) removes iid sensor noise but preserves genuine
    smooth motion, so the filtered jerk is a label-free clean baseline. Flag
    when raw jerk exceeds k× the filtered baseline.
    """
    from scipy.ndimage import median_filter

    j = jerk(actions)
    smooth = median_filter(actions, size=(5, 1))
    base = jerk(smooth)
    # baseline from the least-jerky half of the FILTERED signal
    baseline = np.median(np.sort(base)[: max(1, len(base) // 2)])
    score = float(np.median(j) / (baseline + 1e-8))
    return min(score / (2 * k), 1.0), bool(score > k)


def detect_stall(actions: np.ndarray, eps: float = 1e-3,
                 min_len: int = 5) -> tuple[float, bool, list]:
    """Flag stalls: contiguous runs where action velocity stays below eps."""
    v = action_speed(actions)
    slow = v < eps
    runs, start, n = [], None, 0
    for i, s in enumerate(slow):
        if s:
            if start is None:
                start, n = i, 1
            else:
                n += 1
        else:
            if start is not None and n >= min_len:
                runs.append((start, n))
            start, n = None, 0
    if start is not None and n >= min_len:
        runs.append((start, n))
    longest = max((n for _, n in runs), default=0)
    score = min(longest / (4 * min_len), 1.0)
    return score, bool(runs), runs


def progress_monotonicity(embeddings: np.ndarray) -> tuple[float, bool]:
    """Project frames onto start→goal direction; measure backtracking.

    Returns (monotonicity, is_truncated_like). monotonicity is the fraction
    of steps that move forward along the start→goal axis.
    """
    e = embeddings.astype(np.float64)
    direction = e[-1] - e[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return 0.0, True
    direction /= norm
    proj = e @ direction
    steps = np.diff(proj)
    mono = float((steps > 0).mean()) if len(steps) else 0.0
    return mono, bool(mono < 0.6)

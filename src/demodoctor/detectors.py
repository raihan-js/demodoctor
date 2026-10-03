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


def estimate_lag_stable(frames: np.ndarray, actions: np.ndarray,
                        max_lag: int = 5,
                        min_prominence: float = 0.2) -> tuple[int, bool]:
    """Stable lag estimate: both episode halves must agree on a nonzero lag.

    Single-window argmax fires on noise (flat correlation landscapes give
    arbitrary peaks). A true injected lag is stationary and appears in both
    halves; noise peaks jump between halves. Returns (lag, confident).
    """
    T = len(frames)
    if T < 2 * (max_lag + 10):
        return 0, False
    mid = T // 2
    lag1, p1 = estimate_lag(frames[:mid], actions[:mid], max_lag)
    lag2, prom2 = estimate_lag(frames[mid:], actions[mid:], max_lag)
    p2 = prom2
    confident = (lag1 == lag2 and lag1 > 0 and min(p1, p2) >= min_prominence)
    return (lag1 if confident else 0), confident


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


def detect_truncation(length: int, median_length: float,
                      mad: float) -> tuple[float, bool]:
    """Flag episodes much shorter than the corpus median (robust z-score).

    Truncated demos end before success, so they are length outliers.
    median_length/mad come from the corpus itself — no labels needed.
    """
    if mad < 1e-8:
        return 0.0, False
    z = (median_length - length) / (1.4826 * mad)
    score = min(max(z / 3.0, 0.0), 1.0)
    return score, bool(z > 3.0)


def detect_drop_frames(frames: np.ndarray, k: float = 8.0) -> tuple[float, bool, list]:
    """Flag motion discontinuities: dropped frames appear as sudden jumps.

    Scores the max motion-energy spike relative to the episode median.
    Returns (score, flag, spike_indices).
    """
    m = motion_energy(frames)
    med = np.median(m) + 1e-8
    spikes = [int(i) for i in np.where(m > k * med)[0]]
    score = min(float(m.max() / (k * med)) / 2.0, 1.0) if len(m) else 0.0
    return score, bool(spikes), spikes


def detect_frame_gaps(timestamps: np.ndarray, k: float = 1.5) -> tuple[float, bool, list]:
    """Flag dropped frames via timestamp gaps (needs no motion signal).

    A dropped frame leaves a 2× (or larger) gap in an otherwise regular
    timestamp grid. Returns (score, flag, gap_indices).

    NOTE: only works when timestamps are real. Rebuilt datasets with
    auto-generated uniform timestamps destroy the signal — use
    detect_duplicate_frames for those.
    """
    ts = np.asarray(timestamps, dtype=float)
    if len(ts) < 3:
        return 0.0, False, []
    dt = np.diff(ts)
    med = np.median(dt) + 1e-8
    gaps = [int(i + 1) for i in np.where(dt > k * med)[0]]
    score = min(float(dt.max() / (k * med)) / 3.0, 1.0)
    return score, bool(gaps), gaps


def detect_duplicate_frames(frames: np.ndarray) -> tuple[float, bool, list]:
    """Flag exact-duplicate consecutive frames (stuck camera).

    Real sensors never produce bitwise-identical consecutive frames, so any
    exact duplicate is a certain fault with no threshold to tune.
    Returns (fraction_duplicate, flag, duplicate_indices).
    """
    if len(frames) < 2:
        return 0.0, False, []
    same = [int(i) for i in range(1, len(frames))
            if (frames[i] == frames[i - 1]).all()]
    frac = len(same) / (len(frames) - 1)
    return frac, bool(same), same

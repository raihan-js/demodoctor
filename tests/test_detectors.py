import numpy as np
import pytest

from demodoctor.detectors import (action_speed, detect_jitter, detect_stall,
                                  estimate_lag, jerk, motion_energy,
                                  progress_monotonicity)
from demodoctor.faults import (inject_drop_frames, inject_jitter, inject_lag,
                               inject_stall, inject_truncation)


def smooth_episode(T=120, seed=0):
    """Clean demo: varying-speed random-walk actions, frames that track them.

    Frame brightness is a direct function of action position, so image motion
    correlates with action velocity (with zero lag when clean). Circular
    constant-speed motion is deliberately avoided: it has flat speed and
    carries no lag information.
    """
    rng = np.random.default_rng(seed)
    drive = rng.normal(0, 1.0, (T, 2))
    # smooth with a moving average -> varying speed random walk
    kernel = np.ones(9) / 9
    smooth = np.column_stack([
        np.convolve(drive[:, 0], kernel, mode="same"),
        np.convolve(drive[:, 1], kernel, mode="same"),
    ])
    actions = np.cumsum(smooth, axis=0)
    actions = (actions - actions.mean(axis=0)) / (actions.std(axis=0) + 1e-8)
    actions += rng.normal(0, 0.005, (T, 2))
    # brightness tracks first action dim
    normed = (actions[:, 0] - actions[:, 0].min())
    normed = normed / (normed.max() + 1e-8)
    brightness = (normed * 200 + 20).astype(np.uint8)
    frames = np.zeros((T, 8, 8, 3), dtype=np.uint8)
    for i in range(T):
        frames[i] = brightness[i]
    return frames, actions, rng


class TestMotionEnergy:
    def test_shape(self):
        frames, _, _ = smooth_episode()
        assert motion_energy(frames).shape == (120,)

    def test_static_video_zero(self):
        frames = np.zeros((50, 8, 8, 3), dtype=np.uint8)
        assert motion_energy(frames).max() == 0.0


class TestLagEstimator:
    def test_detects_injected_lag(self):
        frames, actions, rng = smooth_episode()
        # make motion track actions, then lag the frames
        f2, rec = inject_lag(frames, actions, lag=2, rng=rng)
        est, prom = estimate_lag(f2, actions, max_lag=5)
        assert est == 2
        assert prom > 0.3

    def test_clean_episode_lag_zero(self):
        frames, actions, _ = smooth_episode()
        est, _ = estimate_lag(frames, actions, max_lag=5)
        assert est == 0

    def test_lag_range(self):
        frames, actions, rng = smooth_episode()
        for lag in (1, 3):
            f2, _ = inject_lag(frames, actions, lag=lag, rng=rng)
            est, _ = estimate_lag(f2, actions, max_lag=5)
            assert est == lag


class TestJerkStall:
    def test_jitter_detected(self):
        _, actions, rng = smooth_episode()
        a2, _ = inject_jitter(actions, sigma=0.5, rng=rng)
        score, flag = detect_jitter(a2)
        assert flag
        assert score > 0.3

    def test_clean_not_jitter(self):
        _, actions, _ = smooth_episode()
        _, flag = detect_jitter(actions)
        assert not flag

    def test_stall_detected(self):
        _, actions, _ = smooth_episode()
        a2, _ = inject_stall(actions, start=20, length=30)
        score, flag, runs = detect_stall(a2)
        assert flag
        assert runs == [(20, 30)]
        assert score > 0.5

    def test_clean_not_stall(self):
        _, actions, _ = smooth_episode()
        _, flag, runs = detect_stall(actions)
        assert not flag
        assert runs == []

    def test_short_slow_run_ignored(self):
        _, actions, _ = smooth_episode()
        a2, _ = inject_stall(actions, start=10, length=3)
        _, flag, _ = detect_stall(a2, min_len=5)
        assert not flag


class TestProgress:
    def test_monotonic_clean(self):
        rng = np.random.default_rng(0)
        e = np.column_stack([np.linspace(0, 10, 100), rng.normal(0, 0.1, 100)])
        mono, flag = progress_monotonicity(e)
        assert mono > 0.8
        assert not flag

    def test_truncated_like_detected(self):
        rng = np.random.default_rng(1)
        # goes forward then back (looping demo)
        fwd = np.linspace(0, 10, 50)
        back = np.linspace(10, 0, 50)
        e = np.column_stack([np.concatenate([fwd, back]), rng.normal(0, 0.1, 100)])
        mono, flag = progress_monotonicity(e)
        assert flag
        assert mono < 0.6

    def test_degenerate_embeddings(self):
        e = np.zeros((20, 4))
        mono, flag = progress_monotonicity(e)
        assert mono == 0.0 and flag


class TestEndToEnd:
    def test_lag_roundtrip(self):
        frames, actions, rng = smooth_episode()
        f2, _ = inject_lag(frames, actions, lag=2, rng=rng)
        est, prom = estimate_lag(f2, actions)
        assert est == 2 and prom > 0.2

    def test_jitter_roundtrip(self):
        _, actions, rng = smooth_episode()
        a2, _ = inject_jitter(actions, sigma=0.4, rng=rng)
        _, flag = detect_jitter(a2)
        assert flag

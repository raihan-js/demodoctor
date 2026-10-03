import numpy as np
import pytest

from demodoctor.faults import (FAULTS, FaultManifest, inject_dup_frames,
                               inject_fault, inject_jitter, inject_lag,
                               inject_stall, inject_truncation)


def make_episode(T=100, img=(16, 16, 3), act=2, seed=0):
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 255, size=(T,) + img).astype(np.uint8)
    actions = rng.normal(0, 0.5, size=(T, act))
    return frames, actions, rng


class TestLag:
    def test_shape_preserved(self):
        frames, actions, rng = make_episode()
        f2, rec = inject_lag(frames, actions, lag=2, rng=rng)
        assert f2.shape == frames.shape
        assert rec["lag_frames"] == 2

    def test_first_frames_held(self):
        frames, actions, rng = make_episode()
        f2, _ = inject_lag(frames, actions, lag=3, rng=rng)
        assert (f2[:3] == frames[:1]).all()
        assert (f2[3:] == frames[:-3]).all()

    def test_invalid_lag_rejected(self):
        frames, actions, rng = make_episode()
        with pytest.raises(AssertionError):
            inject_lag(frames, actions, lag=0, rng=rng)
        with pytest.raises(AssertionError):
            inject_lag(frames, actions, lag=6, rng=rng)


class TestStall:
    def test_frozen_block(self):
        _, actions, _ = make_episode()
        a2, rec = inject_stall(actions, start=10, length=20)
        assert (a2[10:30] == actions[9]).all()
        assert (a2[:10] == actions[:10]).all()
        assert (a2[30:] == actions[30:]).all()
        assert rec == {"type": "stall", "start": 10, "length": 20}

    def test_stall_at_zero(self):
        _, actions, _ = make_episode()
        a2, _ = inject_stall(actions, start=0, length=5)
        assert (a2[:5] == actions[0]).all()


class TestJitter:
    def test_increases_variance(self):
        _, actions, rng = make_episode()
        a2, rec = inject_jitter(actions, sigma=0.5, rng=rng)
        assert a2.var() > actions.var()
        assert rec["sigma"] == 0.5

    def test_zero_sigma_identity(self):
        _, actions, rng = make_episode()
        a2, _ = inject_jitter(actions, sigma=0.0, rng=rng)
        assert (a2 == actions).all()


class TestTruncation:
    def test_suffix_removed(self):
        frames, actions, _ = make_episode(T=100)
        f2, a2, rec = inject_truncation(frames, actions, cut=15)
        assert len(f2) == 85 and len(a2) == 85
        assert (f2 == frames[:85]).all()
        assert rec["cut_frames"] == 15

    def test_invalid_cut_rejected(self):
        frames, actions, _ = make_episode(T=50)
        with pytest.raises(AssertionError):
            inject_truncation(frames, actions, cut=0)
        with pytest.raises(AssertionError):
            inject_truncation(frames, actions, cut=50)


class TestDupFrames:
    def test_frames_duplicated(self):
        frames, actions, _ = make_episode(T=50)
        f2, a2, rec = inject_dup_frames(frames, actions, dup_idx=[3, 7, 11])
        assert f2.shape == frames.shape
        assert (f2[3] == frames[2]).all()
        assert (f2[7] == frames[6]).all()
        assert (f2[4] == frames[4]).all()  # untouched
        assert (a2 == actions).all()  # actions unchanged
        assert rec["duplicated"] == [3, 7, 11]

    def test_empty_dup_is_identity(self):
        frames, actions, _ = make_episode()
        f2, _, rec = inject_dup_frames(frames, actions, dup_idx=[])
        assert (f2 == frames).all()
        assert rec["duplicated"] == []


class TestManifest:
    def test_has(self):
        m = FaultManifest(episode_id="ep0",
                          faults=[{"type": "lag", "lag_frames": 2}])
        assert m.has("lag")
        assert not m.has("stall")

    def test_dispatch(self):
        frames, actions, rng = make_episode()
        for fault, kw in [("lag", {"lag": 2}), ("stall", {"start": 5, "length": 10}),
                          ("jitter", {"sigma": 0.1}), ("truncation", {"cut": 10}),
                          ("dup_frames", {"dup_idx": [1, 2]})]:
            out = inject_fault(frames, actions, fault, rng, **kw)
            assert out[-1]["type"] == fault

    def test_unknown_fault_rejected(self):
        frames, actions, rng = make_episode()
        with pytest.raises(ValueError):
            inject_fault(frames, actions, "explode", rng)

    def test_fault_registry(self):
        assert set(FAULTS) == {"lag", "stall", "jitter", "truncation", "dup_frames"}

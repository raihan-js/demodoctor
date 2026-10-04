# DemoDoctor: Finding Bad Robot Demonstrations Automatically

*Inject known faults into robot demonstrations, detect them without labels, and measure what they cost a policy.*

---

> Scope note: PushT only, ACT policy, RTX 3060. A data-quality study by an ML engineer learning robotics — not robotics expertise.

## The problem

Robot-learning teams say data quality matters more than model size, but controlled measurements are rare. This project corrupts clean PushT demonstrations with ground-truth faults, detects them from signals alone, then closes the loop: train on clean vs corrupted vs auto-cleaned data and measure simulator success.

## The faults

| Fault | What it mimics | Detectable? |
|---|---|---|
| Lag (1–3 frames) | Slow camera pipeline | Weak — XCorr landscape flat on quasi-static pushing |
| Stall | Teleop freeze | **Yes — F1 0.867** |
| Jitter (scaled) | Shaky teleoperation | **Yes — F1 0.687** |
| Truncation | Episode cut early | Weak — natural length variance swamps cuts |
| Dup frames | Stuck camera | No — re-encoding creates duplicates everywhere |

## The detectors (all label-free)

- **Lag**: cross-correlate image-motion energy with action-velocity energy; stable version requires both halves to agree. Fails honestly on PushT: slow motion gives flat correlation landscapes where noise peaks beat true lag.
- **Jitter**: raw jerk vs median-filtered baseline. Key fix: absolute sigma is meaningless (PushT actions have std ~80, not ~1) — scale to the episode.
- **Stall**: contiguous near-zero-velocity runs. Just works.
- **Truncation**: robust length outlier (median ± MAD). Weak by nature here.
- **Dup frames**: exact-duplicate check confounded by video re-encoding (which creates duplicates in clean data too). Reported as null with the mechanism explained.

## The policy experiment

[TBD — 60k-step baseline running; 3 conditions × 3 seeds grid follows]

## Limitations

- Injected faults are easier than real teleoperation faults; F1 flatters real-world performance.
- PushT only; simulator success has high seed variance.
- 20k-step policies score 0% success (floor effect) — longer training needed.
- Re-encoding destroys timestamp-gap signals; dup detection null is partly pipeline artifact.

## What's next

- DINOv2-small progress-monotonicity scorer (implemented, unevaluated on real embeddings)
- Scan community LeRobot datasets and report flags without claiming ground truth
- ALOHA transfer-cube as stretch

---

*Repo: github.com/raihan-js/demodoctor · Data: huggingface.co/datasets/raihan-js/demodoctor-pusht-corrupted*

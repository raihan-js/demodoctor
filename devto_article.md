# DemoDoctor: Bad Robot Demonstrations Are Detectable. Do They Matter?

*Inject known faults into robot demonstrations, detect them without labels, and measure what they cost a policy. The honest answer so far: I could not measure a cost.*

---

![DemoDoctor results](https://raw.githubusercontent.com/raihan-js/demodoctor/HEAD/images/demodoctor.png)

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

I trained ACT policies on PushT for 60k steps (batch size 32, RTX 3060, about 2 hours each) under four data conditions, 3 seeds each, and evaluated every policy on 50 rollouts:

- **Clean**: all 206 episodes.
- **Corrupted**: 30% of episodes carry an injected fault.
- **Auto-cleaned**: the 175 episodes left after dropping stall- and jitter-flagged ones (26 of the 31 dropped were truly faulty).
- **Random 175**: 175 random episodes of the corrupted set. This is the size-matched control: if cleaning looked worse than corrupted, it separates "less data" from "better data".

| Condition | Success rate | Mean max reward [95% CI] |
|---|---|---|
| Clean | 0.7% | 0.425 [0.369, 0.488] |
| Corrupted | 1.3% | 0.397 [0.344, 0.454] |
| Auto-cleaned | 0.0% | 0.359 [0.302, 0.423] |
| Random 175 | 0.0% | 0.432 [0.386, 0.476] |

Intervals bootstrap over policies, then over rollouts within a policy. **No pairwise difference excludes zero.** The widest gap is auto-cleaned vs random 175 at -0.073 [-0.147, +0.006]. Clean minus corrupted is +0.028 [-0.050, +0.108]. The point estimates even run the wrong way for cleaning, but they are inside the noise.

So this grid shows no effect of corruption, of cleaning, or of data size. It also cannot rule out an effect: it cannot resolve differences smaller than about 0.08 in mean max reward.

### Why I would not read more into it

- **Floor effect.** Success is 0-4% in every cell (at most 2 of 50 rollouts). Max reward is the only usable signal, and it is coarse.
- **Evaluation noise.** I re-evaluated 7 checkpoints a second time. Mean max reward moved by 0.014 on average and by up to 0.037, and the success rate changed for 2 of the 7. That is as large as the smaller condition gaps.
- **Three seeds per condition.** One policy is 2 hours of GPU, so n = 3 was the budget.
- **Mixed training environments.** Some of the original runs were trained on rented pods. Two policies (corrupted and cleaned, seed 2) were retrained locally after their checkpoints were lost; their earlier evaluations (0.373 and 0.349) were close to the retrained ones (0.379 and 0.331).

Two explanations I have not tested: the policies barely solve PushT at this budget, and the faults my detectors can find (stalls, jitter) may be mild for a policy that predicts chunks of actions. Settling it would take more seeds, a longer training budget or a metric with headroom, and faults large enough to hurt.

## Limitations

- Injected faults are easier than real teleoperation faults; detector F1 on them says nothing about real-world fault performance.
- PushT only, ACT only, one training budget (60k steps).
- The policy grid is underpowered (3 seeds per condition, floor-level success), so the null above is "not resolved", not "no effect".
- Re-encoding destroys timestamp-gap signals; dup detection null is partly pipeline artifact.

## What's next

- DINOv2-small progress-monotonicity scorer (implemented, unevaluated on real embeddings)
- Scan community LeRobot datasets and report flags without claiming ground truth
- ALOHA transfer-cube as stretch

---

*Repo: github.com/raihan-js/demodoctor · Data: huggingface.co/datasets/raihan-js/demodoctor-pusht-corrupted*

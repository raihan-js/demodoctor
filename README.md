# DemoDoctor

Find bad robot demonstrations automatically, and measure what they cost a policy.

## The problem

Robot-learning teams say data quality matters more than model size, but controlled measurements are rare. This project injects known faults into clean demonstrations, detects them from signals alone, then measures policy success in simulation on clean vs corrupted vs auto-cleaned data.

## Faults

| Fault | What it mimics |
|---|---|
| `lag` (1–3 frames) | Slow camera pipeline |
| `stall` | Teleop freeze |
| `jitter` | Shaky teleoperation |
| `truncation` | Episode cut before success |
| `drop_frames` | Dropped/duplicated frames |

## Detectors (label-free)

| Detector | Target | Result |
|---|---|---|
| Stall runs | Teleop freeze | **F1 0.867** |
| Jitter vs filtered baseline | Shaky teleop | **F1 0.687** |
| Stable lag XCorr | Camera delay | F1 0.119 (honest null — flat landscape) |
| Length outlier | Truncation | F1 ~0 (honest null — variance swamps cuts) |
| Exact duplicates | Stuck camera | Null (re-encoding confound, explained) |

Cleaning rule (fixed before policy results): drop stall/jitter-flagged episodes → 175/206 kept, 26/31 dropped truly faulty (84% cleaning precision).

## Status

Injector + detectors: 37 tests green. Corrupted + cleaned sets on HF. Policy grid pending (60k-step baseline training).

## Limitations

- Injected faults are easier than real teleoperation faults.
- PushT only; simulator success has high seed variance.
- A data-quality study by an ML engineer learning robotics, not robotics expertise.

## License

MIT

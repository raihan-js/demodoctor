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

- **Lag estimator**: cross-correlate image-motion energy with action-velocity energy
- **Jerk/stall**: jerk vs median-filtered baseline; contiguous near-zero-velocity runs
- **Progress monotonicity**: frame-embedding projection onto start→goal direction

## Status

Injector + detectors: 30 tests green. Policy experiments pending.

## Limitations

- Injected faults are easier than real teleoperation faults.
- PushT only; simulator success has high seed variance.
- A data-quality study by an ML engineer learning robotics, not robotics expertise.

## License

MIT

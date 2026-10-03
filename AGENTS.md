# AGENTS.md — demodoctor

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: DemoDoctor

Find bad robot demonstrations automatically, and measure what they cost a policy. Inject known faults into the LeRobot PushT dataset, build detectors that find them without labels, and measure how much a simulated policy's success drops — and recovers — when the data is cleaned.

## Why this project

Robot-learning teams say data quality matters more than model size, but controlled measurements are rare. This corrupts clean demonstrations with ground-truth faults, detects them from signals alone, then closes the loop with policy success in simulation. Presented as a data-quality study by an ML engineer learning robotics, not robotics expertise.

## Stack

Python, LeRobot (pinned), gym-pusht simulator, ACT policy, DINOv2-small, PyTorch, OpenCV.

## Compute

RTX 3060 12GB. Core grid: 3 data conditions × 3 seeds = 9 policy runs on PushT.

## Key design decisions

- **Ground-truth manifests.** Every injected fault is recorded per episode; detectors never see them.
- **Label-free detectors.** Lag from motion/action cross-correlation, jerk + stall statistics, progress monotonicity on frame embeddings.
- **Closed loop.** ACT trained on clean vs corrupted vs auto-cleaned; 50 rollouts × 3 seeds.
- **Honest scope.** PushT only. Injected faults are easier than real teleop faults — stated plainly.

## Milestones

1. **Environment + baseline** (4d) — pinned LeRobot, ACT on clean PushT, 50 rollouts
2. **Fault injector** (4d) — 5 fault types with manifests (COMPLETE, 30 tests)
3. **Label-free detectors** (6d) — lag/jerk/stall/progress + PR/F1 (COMPLETE, 30 tests)
4. **Policy cost experiment + release** (5d) — 9 runs, HF dataset, CLI, write-up

## Conventions

- Python 3.10+, pytest for all injectors and detectors.
- Seeds fixed; manifests versioned; every claim cites a run.
- Never claim real-world fault performance from injected-fault F1.

## Development

```bash
cd demodoctor
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,robot]"
pytest tests/ -v
```

## Current status

- Fault injector + detectors: 30 tests green
- Pending: LeRobot install, baseline policy, policy cost experiment

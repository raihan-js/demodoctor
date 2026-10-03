"""DemoDoctor CLI: scan a demonstration dataset for faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_scan(args: argparse.Namespace) -> int:
    import numpy as np

    from demodoctor.detectors import (detect_jitter, detect_stall,
                                      estimate_lag, progress_monotonicity)

    # Placeholder wiring: full LeRobot-dataset scan lands with milestone 4.
    # Today it validates the detector contract on a synthetic episode.
    rng = np.random.default_rng(args.seed)
    t = np.linspace(0, 4 * np.pi, 120)
    actions = np.column_stack([np.sin(t), np.cos(t)]) + rng.normal(0, 0.01, (120, 2))
    lag, prom = estimate_lag(np.zeros((120, 8, 8, 3), dtype=np.uint8), actions)
    _, jitter_flag = detect_jitter(actions)
    _, stall_flag, _ = detect_stall(actions)
    report = {
        "episodes": 1,
        "lag_estimate": lag,
        "lag_prominence": round(prom, 3),
        "jitter": jitter_flag,
        "stall": stall_flag,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="demodoctor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan demonstrations for faults")
    s.add_argument("--dataset", default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--output", default=None)
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
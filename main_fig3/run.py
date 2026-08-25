from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compact_measurement.workflows import (  # noqa: E402
    DEFAULT_REPEATS,
    emit_json,
    run_six_panel_compact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the Compact data for Main Figure 3 from raw experimental counts.")
    parser.add_argument("--shots", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--ogm-budget", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    shots = [12, 45] if args.quick and args.shots is None else args.shots
    repeats = 2 if args.quick and args.repeats == DEFAULT_REPEATS else args.repeats
    payload = run_six_panel_compact(
        "experiment",
        "ideal",
        shots,
        repeats,
        args.seed,
        args.ogm_budget,
    )
    emit_json(payload, args.output)


if __name__ == "__main__":
    main()

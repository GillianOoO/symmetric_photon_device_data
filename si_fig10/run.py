from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compact_measurement.workflows import emit_json, run_six_panel_compact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the Compact noisy-reference data for SI Figure 10.")
    parser.add_argument("--shots", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--ogm-budget", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    shots = [12, 45] if args.quick and args.shots is None else args.shots
    repeats = 2 if args.quick and args.repeats == 20 else args.repeats
    emit_json(run_six_panel_compact("experiment", "tomography", shots, repeats, args.seed, args.ogm_budget), args.output)


if __name__ == "__main__":
    main()

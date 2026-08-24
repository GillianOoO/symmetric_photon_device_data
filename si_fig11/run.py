from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compact_measurement.workflows import emit_json, run_si_figure_11  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the Compact variance data for SI Figure 11.")
    parser.add_argument("--shots", nargs="+", type=int, default=[12, 45, 160, 572, 2038, 7259, 25848])
    parser.add_argument("--ogm-budget", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    shots = [12, 45] if args.quick else args.shots
    emit_json(run_si_figure_11(shots, args.ogm_budget), args.output)


if __name__ == "__main__":
    main()

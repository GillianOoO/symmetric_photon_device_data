from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compact_measurement.workflows import emit_json, run_si_figure_1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the Compact and sym_average data for SI Figure 1.")
    parser.add_argument("--shots", nargs="+", type=int, default=[12, 45, 160, 572, 2038, 7259])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--ogm-budget", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    shots = [12, 45] if args.quick else args.shots
    repeats = 2 if args.quick and args.repeats == 20 else args.repeats
    emit_json(run_si_figure_1(shots, repeats, args.seed, args.ogm_budget), args.output)


if __name__ == "__main__":
    main()

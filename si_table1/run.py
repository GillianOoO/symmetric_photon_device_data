from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compact_measurement.workflows import emit_json, run_si_table_1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the Compact column of SI Table 1.")
    parser.add_argument("--ogm-budget", type=int, default=100000)
    parser.add_argument("--output")
    args = parser.parse_args()
    emit_json(run_si_table_1(args.ogm_budget), args.output)


if __name__ == "__main__":
    main()

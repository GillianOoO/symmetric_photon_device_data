from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compact_measurement.hamiltonian import (  # noqa: E402
    load_hamiltonian,
    paper_permutation_twirl,
    save_hamiltonian,
)
from compact_measurement.nonlinear import two_copy_observable  # noqa: E402
from compact_measurement.workflows import (  # noqa: E402
    HAMILTONIAN_ROOT,
    DEFAULT_REPEATS,
    run_nonlinear_properties,
)


def write_summary_csv(path: Path, payload: dict[str, object]) -> None:
    fieldnames = [
        "state",
        "source",
        "shots",
        "reference",
        "mean_estimate",
        "bias",
        "standard_deviation",
        "rmse",
        "uncovered_terms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for state in payload["states"]:
            for source in (
                "ideal_simulation",
                "experimental_counts_ideal_reference",
                "experimental_counts_tomography_reference",
            ):
                for row in state[source]:
                    writer.writerow(
                        {
                            "state": state["state"],
                            "source": source,
                            **{key: row[key] for key in fieldnames[2:]},
                        }
                    )


def write_design_csv(path: Path, design: dict[str, object]) -> None:
    width = len(design["settings"][0]["pauli"])
    fieldnames = ["probability", *[f"q{index + 1}" for index in range(width)]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in design["settings"]:
            writer.writerow(
                {
                    "probability": row["probability"],
                    **{
                        f"q{index + 1}": value
                        for index, value in enumerate(row["pauli"])
                    },
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate nonlinear Compact data for W3 and GHZ3."
    )
    parser.add_argument(
        "--shots", nargs="+", type=int, default=[12, 45, 160, 572, 2038, 7259]
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--ogm-budget", type=int, default=100000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shots = [12, 45] if args.quick else args.shots
    repeats = 2 if args.quick and args.repeats == DEFAULT_REPEATS else args.repeats
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = run_nonlinear_properties(
        shots, repeats, args.seed, args.ogm_budget
    )
    json_path = output_dir / "nonlinear_properties.json"
    csv_path = output_dir / "nonlinear_properties.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(csv_path, payload)

    physical = load_hamiltonian(HAMILTONIAN_ROOT / "H_3.txt")
    original = two_copy_observable(physical)
    compact = two_copy_observable(paper_permutation_twirl(physical))
    original_path = output_dir / "H_swap_3.txt"
    compact_path = output_dir / "sym_H_swap_3.txt"
    save_hamiltonian(original_path, original)
    save_hamiltonian(compact_path, compact)

    original_design_path = output_dir / "OGM_H_swap_3.csv"
    compact_design_path = output_dir / "OGM_sym_H_swap_3.csv"
    write_design_csv(
        original_design_path, payload["measurement_designs"]["original"]
    )
    write_design_csv(
        compact_design_path, payload["measurement_designs"]["compact"]
    )

    for path in (
        json_path,
        csv_path,
        original_path,
        compact_path,
        original_design_path,
        compact_design_path,
    ):
        print(path)


if __name__ == "__main__":
    main()

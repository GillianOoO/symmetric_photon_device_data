"""Merge the two simulation summaries used in Supplementary Figure 9."""

import csv
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = FIGURE_ROOT / "inputs"
OUTPUT_PATH = FIGURE_ROOT / "outputs" / "si_fig9_plot_data.csv"
TARGET_QUBITS = {8, 12, 14}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    output_rows: list[dict[str, str]] = []
    for row in read_csv(INPUT_ROOT / "ogm_summary.csv"):
        num_qubits = int(row["num_qubits"])
        if num_qubits not in TARGET_QUBITS or row["variant"] not in {"H", "sym_H"}:
            continue
        output_rows.append(
            {
                "num_qubits": row["num_qubits"],
                "method": "OGM" if row["variant"] == "H" else "Compact",
                "shots": row["shots"],
                "exact": row["exact"],
                "mean": row["mean"],
                "std": row["std"],
                "rmse": row["deviation"],
                "repeats": row["repeats"],
                "source_file": "ogm_summary.csv",
            }
        )

    method_names = {
        "ShadowGrouping": "SG",
        "Derandomization": "Derand",
        "AdaptivePaulis": "AP",
    }
    for row in read_csv(INPUT_ROOT / "comparison_summary.csv"):
        num_qubits = int(row["num_qubits"])
        if num_qubits not in TARGET_QUBITS or row["method"] not in method_names:
            continue
        output_rows.append(
            {
                "num_qubits": row["num_qubits"],
                "method": method_names[row["method"]],
                "shots": row["shots"],
                "exact": row["exact"],
                "mean": row["mean"],
                "std": row["std"],
                "rmse": row["deviation"],
                "repeats": row["repeats"],
                "source_file": "comparison_summary.csv",
            }
        )

    method_order = {name: index for index, name in enumerate(["SG", "Derand", "OGM", "AP", "Compact"])}
    output_rows.sort(
        key=lambda row: (
            int(row["num_qubits"]),
            method_order[row["method"]],
            int(row["shots"]),
        )
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()

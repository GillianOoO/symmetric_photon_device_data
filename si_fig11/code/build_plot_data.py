"""Merge and filter the variance rows used in Supplementary Figure 11."""

import csv
import math
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = FIGURE_ROOT / "inputs"
OUTPUT_PATH = FIGURE_ROOT / "outputs" / "si_fig11_plot_data.csv"
INPUT_FILES = ["variance_noisy_summary.csv", "variance_compact_summary.csv"]


def main() -> None:
    rows: list[dict[str, str]] = []
    for filename in INPUT_FILES:
        with (INPUT_ROOT / filename).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                variance_t = float(row["variance_T_shots"])
                if math.isnan(variance_t) or int(float(row["shots"])) == 1000:
                    continue
                rows.append(
                    {
                        "case_name": row["case_name"],
                        "state": row["state_label"],
                        "method": row["method"],
                        "shots": row["shots"],
                        "variance_T_shots": row["variance_T_shots"],
                        "variance_single_shot": row["variance_single_shot"],
                        "std_T_shots": row["std_T_shots"],
                        "variance_source": row["variance_source"],
                        "source_file": filename,
                    }
                )

    method_order = {name: index for index, name in enumerate(["SG", "Derand", "OGM", "AP", "Compact"])}
    rows.sort(key=lambda row: (row["case_name"], method_order[row["method"]], int(float(row["shots"]))))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Extract the H8 rows used in Supplementary Figure 1; no plotting is performed."""

import csv
from pathlib import Path


FIGURE_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = FIGURE_ROOT / "inputs" / "summary.csv"
OUTPUT_PATH = FIGURE_ROOT / "outputs" / "si_fig1_plot_data.csv"
METHODS = {"SG", "Derand", "AP", "OGM"}
VARIANTS = {"H", "sym_H", "sym_ave"}


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8", newline="") as handle:
        input_rows = list(csv.DictReader(handle))

    output_rows = []
    for row in input_rows:
        if row["target"] != "H8" or row["method"] not in METHODS:
            continue
        if row["variant"] not in VARIANTS:
            continue
        output_rows.append(
            {
                "panel_method": row["method"],
                "variant": row["variant"],
                "display_label": "compact" if row["variant"] == "sym_H" else row["variant"],
                "shots": row["shots"],
                "exact": row["exact"],
                "mean": row["mean"],
                "std": row["std"],
                "rmse": row["deviation"],
                "repeats": row["repeats"],
            }
        )

    method_order = {name: index for index, name in enumerate(["SG", "Derand", "AP", "OGM"])}
    variant_order = {name: index for index, name in enumerate(["H", "sym_H", "sym_ave"])}
    output_rows.sort(
        key=lambda row: (
            method_order[row["panel_method"]],
            variant_order[row["variant"]],
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

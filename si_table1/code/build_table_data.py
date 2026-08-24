"""Build Supplementary Table 1 from the largest-shot variance rows."""

import csv
from pathlib import Path


TABLE_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = TABLE_ROOT / "inputs"
OUTPUT_ROOT = TABLE_ROOT / "outputs"
METHODS = ["Derand", "OGM", "AP", "SG", "Compact"]
ROW_SPECS = [
    ("W", "SpinH", "H_4_W"),
    ("W", "RandH", "rand_H_3_W"),
    ("W", "nonlinear SpinH", "H_swap_3_W"),
    ("GHZ", "SpinH", "H_4_GHZ"),
    ("GHZ", "RandH", "rand_H_3_GHZ"),
    ("GHZ", "nonlinear SpinH", "H_swap_3_GHZ"),
]


def main() -> None:
    rows: list[dict[str, str]] = []
    for filename in ["variance_noisy_summary.csv", "variance_compact_summary.csv"]:
        with (INPUT_ROOT / filename).open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    best: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["case_name"], row["method"])
        if key not in best or float(row["shots"]) > float(best[key]["shots"]):
            best[key] = row

    wide_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for state, observable, case_name in ROW_SPECS:
        wide_row: dict[str, object] = {"state": state, "observable": observable}
        for method in METHODS:
            source = best.get((case_name, method))
            if source is None:
                wide_row[method] = ""
                continue
            shots = int(round(float(source["shots"])))
            variance_t = float(source["variance_T_shots"])
            variance_single = float(source["variance_single_shot"])
            reconstructed = variance_t * shots
            tolerance = 1e-9 * max(1.0, abs(variance_single))
            if abs(reconstructed - variance_single) > tolerance:
                raise ValueError(f"Variance mismatch for {case_name}/{method}")
            wide_row[method] = f"{variance_single:.12g}"
            source_rows.append(
                {
                    "case_name": case_name,
                    "state": state,
                    "observable": observable,
                    "method": method,
                    "shots_max": shots,
                    "variance_T_at_shots_max": f"{variance_t:.12g}",
                    "variance_single_shot": f"{variance_single:.12g}",
                    "variance_source": source["variance_source"],
                }
            )
        wide_rows.append(wide_row)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "si_table1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state", "observable", *METHODS])
        writer.writeheader()
        writer.writerows(wide_rows)
    with (OUTPUT_ROOT / "si_table1_sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)


if __name__ == "__main__":
    main()

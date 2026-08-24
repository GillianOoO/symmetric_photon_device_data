"""Build the numeric rows used in Supplementary Figure 10; no plotting is performed."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from plot_data_utils import load_estimator_rows, six_panel_specs, write_estimator_csv  # noqa: E402


FIGURE_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = FIGURE_ROOT / "inputs" / "raw_estimates"
OUTPUT_ROOT = FIGURE_ROOT / "outputs"


def main() -> None:
    rows = load_estimator_rows(INPUT_ROOT, six_panel_specs())
    write_estimator_csv(OUTPUT_ROOT / "all_input_rows.csv", rows)
    # The published nonlinear panels omit the auxiliary 1000-shot row.
    write_estimator_csv(
        OUTPUT_ROOT / "si_fig10_plot_data.csv",
        rows,
        keep=lambda row: row["shots"] != 1000,
    )


if __name__ == "__main__":
    main()

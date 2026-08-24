"""Build the numeric rows used in Supplementary Figure 8; no plotting is performed."""

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
    # The MATLAB scripts used for the archived PDF delete row 5 in every series.
    # In the current noiseless inputs, row 5 is the 2038-shot point.
    write_estimator_csv(
        OUTPUT_ROOT / "si_fig8_plot_data_as_published.csv",
        rows,
        keep=lambda row: row["row_index"] != 4,
    )


if __name__ == "__main__":
    main()

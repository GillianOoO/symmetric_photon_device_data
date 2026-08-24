"""Regenerate every plot-data CSV and Supplementary Table 1 CSV."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "main_fig3/code/build_plot_data.py",
    "si_fig1/code/build_plot_data.py",
    "si_fig8/code/build_plot_data.py",
    "si_fig9/code/build_plot_data.py",
    "si_fig10/code/build_plot_data.py",
    "si_fig11/code/build_plot_data.py",
    "si_table1/code/build_table_data.py",
]


def main() -> None:
    for relative_path in SCRIPTS:
        script = ROOT / relative_path
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        print(f"generated data with {relative_path}")


if __name__ == "__main__":
    main()

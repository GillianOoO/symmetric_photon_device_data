"""Shared helpers for turning archived estimator repeats into plot-ready CSV rows."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class EstimatorSeries:
    panel: str
    state: str
    observable: str
    method: str
    relative_path: str


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: list[float], mean: float) -> float:
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def load_estimator_rows(input_root: Path, specs: Iterable[EstimatorSeries]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        source_path = input_root / spec.relative_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        for row_index, line in enumerate(source_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 5:
                raise ValueError(f"Malformed estimator row in {source_path}: {line}")

            shots = int(round(values[0]))
            reference = values[1]
            rmse_reported = values[2]
            repeats = int(round(values[3]))
            estimates = values[4 : 4 + repeats]
            if len(estimates) != repeats:
                raise ValueError(
                    f"Expected {repeats} estimates in {source_path}, found {len(estimates)}"
                )

            mean_estimate = _mean(estimates)
            rmse_recomputed = math.sqrt(
                sum((estimate - reference) ** 2 for estimate in estimates) / repeats
            )
            rows.append(
                {
                    "panel": spec.panel,
                    "state": spec.state,
                    "observable": spec.observable,
                    "method": spec.method,
                    "shots": shots,
                    "reference": reference,
                    "rmse": rmse_reported,
                    "rmse_recomputed": rmse_recomputed,
                    "rmse_rounding_delta": abs(rmse_reported - rmse_recomputed),
                    "mean_estimate": mean_estimate,
                    "std_estimate": _population_std(estimates, mean_estimate),
                    "repeats": repeats,
                    "source_file": spec.relative_path,
                    "source_row": row_index + 1,
                    "row_index": row_index,
                }
            )
    return rows


def write_estimator_csv(
    output_path: Path,
    rows: Iterable[dict[str, object]],
    keep: Callable[[dict[str, object]], bool] | None = None,
) -> None:
    fieldnames = [
        "panel",
        "state",
        "observable",
        "method",
        "shots",
        "reference",
        "rmse",
        "rmse_recomputed",
        "rmse_rounding_delta",
        "mean_estimate",
        "std_estimate",
        "repeats",
        "source_file",
        "source_row",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if keep is None or keep(row):
                writer.writerow({name: row[name] for name in fieldnames})


def six_panel_specs() -> list[EstimatorSeries]:
    specs: list[EstimatorSeries] = []
    linear_panels = [
        ("a", "W", "random_H_3", "rand_H_counts_3"),
        ("b", "GHZ", "random_H_3", "rand_H_counts_3"),
        ("c", "W", "spin_H_4", "H_counts_4"),
        ("d", "GHZ", "spin_H_4", "H_counts_4"),
    ]
    methods = [
        ("SG", "ShadowGrouping"),
        ("Derand", "Derandomization"),
        ("OGM", "OGM_"),
        ("AP", "AdaptivePaulis"),
        ("Compact", "OGM_"),
    ]
    for panel, state, observable, file_tail in linear_panels:
        for method, prefix in methods:
            tail = file_tail
            if method == "Compact":
                tail = "sym_" + file_tail
            specs.append(
                EstimatorSeries(
                    panel=panel,
                    state=state,
                    observable=observable,
                    method=method,
                    relative_path=(
                        f"data_rhoH/haozhaowu_outputs_{state}/{prefix}{tail}.txt"
                    ),
                )
            )

    for panel, state in [("e", "W"), ("f", "GHZ")]:
        for method, filename in [
            ("SG", "ShadowGroupingH_swap_counts_3.txt"),
            ("Derand", "DerandomizationH_swap_counts_3.txt"),
            ("OGM", "OGM_H_swap_counts_3.txt"),
            ("Compact", "OGM_sym_H_swap_counts_3.txt"),
        ]:
            specs.append(
                EstimatorSeries(
                    panel=panel,
                    state=state,
                    observable="tr_rho2_spin_H_3",
                    method=method,
                    relative_path=(
                        f"data_rho^2H/haozhaowu_outputs_{state}/{filename}"
                    ),
                )
            )
    return specs

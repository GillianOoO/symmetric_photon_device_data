from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hamiltonian import Hamiltonian
from .measurement import MeasurementDesign, allocate_counts, coverage_matrix
from .pauli import (
    density_measurement_probabilities,
    fwht,
    ghz_measurement_probabilities,
    support_masks,
)


@dataclass(frozen=True)
class EstimateBatch:
    estimates: np.ndarray
    uncovered_terms: int


def _physical_setting(setting: np.ndarray) -> np.ndarray:
    result = np.asarray(setting, dtype=int).copy()
    result[result == 0] = 3
    return result


def _estimate_from_moments(
    hamiltonian: Hamiltonian,
    setting_counts: dict[tuple[int, ...], int],
    setting_moments: dict[tuple[int, ...], np.ndarray],
) -> tuple[float, int]:
    masks = support_masks(hamiltonian.paulis)
    coverage = np.zeros(hamiltonian.num_terms, dtype=np.int64)
    totals = np.zeros(hamiltonian.num_terms, dtype=float)
    for setting_tuple, count in setting_counts.items():
        setting = np.asarray(setting_tuple, dtype=int)
        hit = np.all((hamiltonian.paulis == 0) | (hamiltonian.paulis == setting), axis=1)
        indices = np.flatnonzero(hit)
        coverage[indices] += count
        totals[indices] += setting_moments[setting_tuple][masks[indices]]
    measured = coverage > 0
    estimate = hamiltonian.offset
    estimate += float(
        np.sum(hamiltonian.coefficients[measured] * totals[measured] / coverage[measured])
    )
    return estimate, int(np.count_nonzero(~measured))


def _distribution_for_state(
    rho: np.ndarray | None,
    setting: np.ndarray,
    state_kind: str,
) -> np.ndarray:
    physical = _physical_setting(setting)
    if state_kind == "ghz":
        return ghz_measurement_probabilities(physical)
    if rho is None:
        raise ValueError("rho is required unless state_kind='ghz'")
    return density_measurement_probabilities(rho, physical)


def simulate_compact(
    hamiltonian: Hamiltonian,
    design: MeasurementDesign,
    shots: list[int],
    repeats: int,
    seed: int,
    rho: np.ndarray | None = None,
    state_kind: str = "density",
) -> dict[int, EstimateBatch]:
    """Simulate the OGM estimator from a density matrix or an analytic GHZ state."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    probability_cache: dict[tuple[int, ...], np.ndarray] = {}
    output: dict[int, EstimateBatch] = {}
    for shot_index, total_shots in enumerate(shots):
        allocated = allocate_counts(design.probabilities, total_shots)
        estimates = np.zeros(repeats, dtype=float)
        uncovered = 0
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 100_000 * shot_index + repeat)
            setting_counts: dict[tuple[int, ...], int] = {}
            setting_moments: dict[tuple[int, ...], np.ndarray] = {}
            for setting, count in zip(design.settings, allocated):
                if count == 0:
                    continue
                physical = _physical_setting(setting)
                key = tuple(int(value) for value in physical)
                probabilities = probability_cache.get(key)
                if probabilities is None:
                    probabilities = _distribution_for_state(rho, physical, state_kind)
                    probability_cache[key] = probabilities
                outcome_counts = rng.multinomial(int(count), probabilities)
                setting_counts[key] = setting_counts.get(key, 0) + int(count)
                moment_vector = fwht(outcome_counts)
                if key in setting_moments:
                    setting_moments[key] += moment_vector
                else:
                    setting_moments[key] = moment_vector
            estimates[repeat], uncovered = _estimate_from_moments(
                hamiltonian, setting_counts, setting_moments
            )
        output[int(total_shots)] = EstimateBatch(estimates, uncovered)
    return output


class ExperimentalArchive:
    """Read detector outcomes from the paper's ZIP archives without extracting files."""

    def __init__(self, zip_path: str | Path, basis_csv: str | Path):
        self.zip_path = Path(zip_path)
        basis_data = np.loadtxt(Path(basis_csv), delimiter=",", skiprows=1, dtype=int)
        self.available_counts = np.asarray(basis_data[:, 0], dtype=int)
        self.basis_to_index = {
            tuple(int(value) for value in row): index + 1
            for index, row in enumerate(np.asarray(basis_data[:, 1:], dtype=int))
        }
        self.num_qubits = int(basis_data.shape[1] - 1)
        self._cache: dict[int, np.ndarray] = {}

    def outcomes(self, setting: np.ndarray) -> np.ndarray:
        physical = _physical_setting(setting)
        basis_key = tuple(int(value) for value in physical)
        if basis_key not in self.basis_to_index:
            raise KeyError(f"Measurement basis {basis_key} is absent from {self.zip_path.name}")
        file_index = self.basis_to_index[basis_key]
        cached = self._cache.get(file_index)
        if cached is not None:
            return cached
        with zipfile.ZipFile(self.zip_path) as archive:
            suffix = f"/{file_index}.bin"
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one member ending {suffix!r} in {self.zip_path}, found {len(matches)}"
                )
            payload = archive.read(matches[0])
        raw = np.frombuffer(payload, dtype=np.uint8)
        bits = raw[(raw == ord("0")) | (raw == ord("1"))] - ord("0")
        usable = (len(bits) // self.num_qubits) * self.num_qubits
        outcomes = bits[:usable].reshape(-1, self.num_qubits)
        expected = int(self.available_counts[file_index - 1])
        if len(outcomes) != expected:
            raise ValueError(
                f"Raw count mismatch for {matches[0]}: CSV={expected}, archive={len(outcomes)}"
            )
        self._cache[file_index] = outcomes
        return outcomes

    def moment_vector(self, setting: np.ndarray, start: int, count: int) -> np.ndarray:
        selected = self.outcome_slice(setting, start, count)
        integer_outcomes = selected @ (1 << np.arange(self.num_qubits, dtype=np.int64))
        histogram = np.bincount(integer_outcomes, minlength=2**self.num_qubits)
        return fwht(histogram)

    def outcome_slice(self, setting: np.ndarray, start: int, count: int) -> np.ndarray:
        outcomes = self.outcomes(setting)
        stop = start + count
        if stop > len(outcomes):
            raise ValueError(
                f"Not enough outcomes for setting {tuple(setting)}: "
                f"need {stop}, have {len(outcomes)}"
            )
        return outcomes[start:stop]


def _unbiased_estimate_from_moments(
    hamiltonian: Hamiltonian,
    design: MeasurementDesign,
    total_shots: int,
    setting_counts: dict[tuple[int, ...], int],
    setting_moments: dict[tuple[int, ...], np.ndarray],
) -> tuple[float, int]:
    physical_design = np.asarray(
        [_physical_setting(setting) for setting in design.settings], dtype=int
    )
    hit_probabilities = (
        coverage_matrix(hamiltonian.paulis, physical_design)
        @ np.asarray(design.probabilities, dtype=float)
    )
    if np.any(hit_probabilities <= 1e-14):
        raise ValueError("Measurement design does not cover every Hamiltonian term")

    masks = support_masks(hamiltonian.paulis)
    totals = np.zeros(hamiltonian.num_terms, dtype=float)
    realized_coverage = np.zeros(hamiltonian.num_terms, dtype=np.int64)
    for setting_tuple, count in setting_counts.items():
        setting = np.asarray(setting_tuple, dtype=int)
        hit = np.all(
            (hamiltonian.paulis == 0) | (hamiltonian.paulis == setting), axis=1
        )
        indices = np.flatnonzero(hit)
        totals[indices] += setting_moments[setting_tuple][masks[indices]]
        realized_coverage[indices] += count

    estimate = hamiltonian.offset + float(
        np.sum(
            hamiltonian.coefficients
            * totals
            / (float(total_shots) * hit_probabilities)
        )
    )
    return estimate, int(np.count_nonzero(realized_coverage == 0))


def simulate_ogm_unbiased(
    hamiltonian: Hamiltonian,
    design: MeasurementDesign,
    shots: list[int],
    repeats: int,
    seed: int,
    rho: np.ndarray,
) -> dict[int, EstimateBatch]:
    """Simulate the inverse-coverage OGM estimator, including schedule randomness."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    probability_cache: dict[tuple[int, ...], np.ndarray] = {}
    output: dict[int, EstimateBatch] = {}
    for shot_index, total_shots in enumerate(shots):
        estimates = np.zeros(repeats, dtype=float)
        max_uncovered = 0
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 100_000 * shot_index + repeat)
            sampled = rng.choice(
                len(design.probabilities),
                size=int(total_shots),
                p=design.probabilities,
            )
            sampled_counts = np.bincount(sampled, minlength=len(design.settings))
            setting_counts: dict[tuple[int, ...], int] = {}
            setting_moments: dict[tuple[int, ...], np.ndarray] = {}
            for setting, count in zip(design.settings, sampled_counts):
                if count == 0:
                    continue
                physical = _physical_setting(setting)
                key = tuple(int(value) for value in physical)
                probabilities = probability_cache.get(key)
                if probabilities is None:
                    probabilities = density_measurement_probabilities(rho, physical)
                    probability_cache[key] = probabilities
                moments = fwht(rng.multinomial(int(count), probabilities))
                setting_counts[key] = setting_counts.get(key, 0) + int(count)
                if key in setting_moments:
                    setting_moments[key] += moments
                else:
                    setting_moments[key] = moments
            estimates[repeat], uncovered = _unbiased_estimate_from_moments(
                hamiltonian,
                design,
                int(total_shots),
                setting_counts,
                setting_moments,
            )
            max_uncovered = max(max_uncovered, uncovered)
        output[int(total_shots)] = EstimateBatch(estimates, max_uncovered)
    return output


def estimate_experimental_nonlinear_unbiased(
    hamiltonian: Hamiltonian,
    design: MeasurementDesign,
    archive: ExperimentalArchive,
    shots: list[int],
    repeats: int,
    seed: int,
) -> dict[int, EstimateBatch]:
    """Estimate a two-copy observable with independent raw outcomes per copy."""

    if hamiltonian.num_qubits != 2 * archive.num_qubits:
        raise ValueError("Nonlinear Hamiltonian must act on two copies of the experimental state")
    half = archive.num_qubits
    archive_offsets: dict[tuple[int, ...], int] = {}
    output: dict[int, EstimateBatch] = {}
    for shot_index, total_shots in enumerate(shots):
        estimates = np.zeros(repeats, dtype=float)
        max_uncovered = 0
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 100_000 * shot_index + repeat)
            sampled = rng.choice(
                len(design.probabilities),
                size=int(total_shots),
                p=design.probabilities,
            )
            sampled_counts = np.bincount(sampled, minlength=len(design.settings))
            setting_counts: dict[tuple[int, ...], int] = {}
            setting_moments: dict[tuple[int, ...], np.ndarray] = {}
            for setting, count in zip(design.settings, sampled_counts):
                if count == 0:
                    continue
                physical = _physical_setting(setting)
                first_setting = physical[:half]
                second_setting = physical[half:]
                first_key = tuple(int(value) for value in first_setting)
                second_key = tuple(int(value) for value in second_setting)
                first_start = archive_offsets.get(first_key, 0)
                first_outcomes = archive.outcome_slice(
                    first_setting, first_start, int(count)
                )
                archive_offsets[first_key] = first_start + int(count)
                second_start = archive_offsets.get(second_key, 0)
                second_outcomes = archive.outcome_slice(
                    second_setting, second_start, int(count)
                )
                archive_offsets[second_key] = second_start + int(count)

                combined = np.hstack((first_outcomes, second_outcomes))
                integer_outcomes = combined @ (
                    1 << np.arange(hamiltonian.num_qubits, dtype=np.int64)
                )
                moments = fwht(
                    np.bincount(integer_outcomes, minlength=2**hamiltonian.num_qubits)
                )
                key = tuple(int(value) for value in physical)
                setting_counts[key] = setting_counts.get(key, 0) + int(count)
                if key in setting_moments:
                    setting_moments[key] += moments
                else:
                    setting_moments[key] = moments
            estimates[repeat], uncovered = _unbiased_estimate_from_moments(
                hamiltonian,
                design,
                int(total_shots),
                setting_counts,
                setting_moments,
            )
            max_uncovered = max(max_uncovered, uncovered)
        output[int(total_shots)] = EstimateBatch(estimates, max_uncovered)
    return output


def estimate_experimental_linear(
    hamiltonian: Hamiltonian,
    design: MeasurementDesign,
    archive: ExperimentalArchive,
    shots: list[int],
    repeats: int,
) -> dict[int, EstimateBatch]:
    output: dict[int, EstimateBatch] = {}
    for total_shots in shots:
        allocated = allocate_counts(design.probabilities, total_shots)
        estimates = np.zeros(repeats, dtype=float)
        uncovered = 0
        for repeat in range(repeats):
            setting_counts: dict[tuple[int, ...], int] = {}
            setting_moments: dict[tuple[int, ...], np.ndarray] = {}
            for setting, count in zip(design.settings, allocated):
                if count == 0:
                    continue
                physical = _physical_setting(setting)
                key = tuple(int(value) for value in physical)
                moment_vector = archive.moment_vector(physical, repeat * int(count), int(count))
                setting_counts[key] = setting_counts.get(key, 0) + int(count)
                if key in setting_moments:
                    setting_moments[key] += moment_vector
                else:
                    setting_moments[key] = moment_vector
            estimates[repeat], uncovered = _estimate_from_moments(
                hamiltonian, setting_counts, setting_moments
            )
        output[int(total_shots)] = EstimateBatch(estimates, uncovered)
    return output


def summarize_batches(
    batches: dict[int, EstimateBatch], reference: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shots, batch in sorted(batches.items()):
        estimates = np.asarray(batch.estimates, dtype=float)
        mean_estimate = float(np.mean(estimates))
        rows.append(
            {
                "shots": int(shots),
                "reference": float(reference),
                "mean_estimate": mean_estimate,
                "bias": mean_estimate - float(reference),
                "standard_deviation": float(np.std(estimates, ddof=0)),
                "rmse": float(np.sqrt(np.mean((estimates - reference) ** 2))),
                "uncovered_terms": int(batch.uncovered_terms),
                "estimates": [float(value) for value in estimates],
            }
        )
    return rows

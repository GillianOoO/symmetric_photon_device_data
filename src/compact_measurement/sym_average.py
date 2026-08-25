from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from .estimator import EstimateBatch
from .hamiltonian import Hamiltonian
from .measurement import MeasurementDesign, sample_schedule
from .pauli import fwht, ghz_measurement_probabilities, support_masks


def _physical_setting(setting: np.ndarray) -> np.ndarray:
    result = np.asarray(setting, dtype=int).copy()
    result[result == 0] = 3
    return result


def _setting_structure(setting: tuple[int, ...]) -> dict[str, object]:
    positions = {
        basis: tuple(index for index, value in enumerate(setting) if value == basis)
        for basis in (1, 2, 3)
    }
    return {
        "positions": positions,
        "masks": {
            basis: sum(1 << index for index in positions[basis]) for basis in (1, 2, 3)
        },
    }


def _orbit_key(mask: int, structure: dict[str, object]) -> tuple[int, int, int]:
    masks = structure["masks"]
    return tuple(bin(mask & int(masks[basis])).count("1") for basis in (1, 2, 3))


def _orbit_masks(setting: tuple[int, ...], key: tuple[int, int, int]) -> np.ndarray:
    structure = _setting_structure(setting)
    positions = structure["positions"]
    masks: list[int] = []
    for x_subset in combinations(positions[1], key[0]):
        x_mask = sum(1 << index for index in x_subset)
        for y_subset in combinations(positions[2], key[1]):
            y_mask = sum(1 << index for index in y_subset)
            for z_subset in combinations(positions[3], key[2]):
                z_mask = sum(1 << index for index in z_subset)
                masks.append(x_mask | y_mask | z_mask)
    return np.asarray(masks, dtype=np.int64)


def _coverage(
    hamiltonian: Hamiltonian,
    setting_counts: dict[tuple[int, ...], int],
) -> np.ndarray:
    result = np.zeros(hamiltonian.num_terms, dtype=np.int64)
    for setting_tuple, count in setting_counts.items():
        setting = np.asarray(setting_tuple, dtype=int)
        hit = np.all((hamiltonian.paulis == 0) | (hamiltonian.paulis == setting), axis=1)
        result[hit] += count
    return result


def symmetry_average_estimate(
    hamiltonian: Hamiltonian,
    setting_counts: dict[tuple[int, ...], int],
    moment_sums: dict[tuple[int, ...], np.ndarray],
) -> tuple[float, int]:
    """Apply measurement-compatible permutation averaging to a fixed protocol."""

    masks = support_masks(hamiltonian.paulis)
    coverage = _coverage(hamiltonian, setting_counts)
    estimate = float(hamiltonian.offset)
    for setting_tuple, count in setting_counts.items():
        setting = np.asarray(setting_tuple, dtype=int)
        hit_indices = np.flatnonzero(
            np.all((hamiltonian.paulis == 0) | (hamiltonian.paulis == setting), axis=1)
        )
        orbit_coefficients: dict[tuple[int, int, int], float] = defaultdict(float)
        structure = _setting_structure(setting_tuple)
        for term_index in hit_indices:
            if coverage[term_index] == 0:
                continue
            beta = (
                float(hamiltonian.coefficients[term_index])
                * float(count)
                / float(coverage[term_index])
            )
            orbit_coefficients[_orbit_key(int(masks[term_index]), structure)] += beta
        for key, coefficient_sum in orbit_coefficients.items():
            compatible_masks = _orbit_masks(setting_tuple, key)
            average_coefficient = coefficient_sum / len(compatible_masks)
            estimate += (
                average_coefficient
                * float(np.sum(moment_sums[setting_tuple][compatible_masks]))
                / float(count)
            )
    return estimate, int(np.count_nonzero(coverage == 0))


def simulate_symmetry_average(
    hamiltonian: Hamiltonian,
    design: MeasurementDesign,
    shots: list[int],
    repeats: int,
    seed: int,
) -> dict[int, EstimateBatch]:
    max_shots = max(shots)
    shot_positions = {int(value): index for index, value in enumerate(shots)}
    estimates = np.zeros((repeats, len(shots)), dtype=float)
    uncovered = np.zeros(len(shots), dtype=int)
    distribution_cache: dict[tuple[int, ...], np.ndarray] = {}

    for repeat in range(repeats):
        schedule_rng = np.random.default_rng(seed + 1000 * repeat)
        measure_rng = np.random.default_rng(seed + 1000 * repeat + 1)
        sequence = sample_schedule(design, max_shots, schedule_rng)
        setting_counts: dict[tuple[int, ...], int] = {}
        moment_sums: dict[tuple[int, ...], np.ndarray] = {}
        previous = 0
        for total_shots in sorted(shots):
            chunk = sequence[previous:total_shots]
            previous = total_shots
            physical_chunk = np.asarray([_physical_setting(setting) for setting in chunk], dtype=int)
            unique_settings, counts = np.unique(physical_chunk, axis=0, return_counts=True)
            for setting, count in zip(unique_settings, counts):
                key = tuple(int(value) for value in setting)
                probabilities = distribution_cache.get(key)
                if probabilities is None:
                    probabilities = ghz_measurement_probabilities(setting)
                    distribution_cache[key] = probabilities
                moments = fwht(measure_rng.multinomial(int(count), probabilities))
                setting_counts[key] = setting_counts.get(key, 0) + int(count)
                if key in moment_sums:
                    moment_sums[key] += moments
                else:
                    moment_sums[key] = moments
            estimate, missed = symmetry_average_estimate(
                hamiltonian, setting_counts, moment_sums
            )
            column = shot_positions[int(total_shots)]
            estimates[repeat, column] = estimate
            uncovered[column] = max(uncovered[column], missed)

    return {
        int(total_shots): EstimateBatch(estimates[:, index], int(uncovered[index]))
        for index, total_shots in enumerate(shots)
    }

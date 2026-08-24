from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .hamiltonian import Hamiltonian


@dataclass(frozen=True)
class MeasurementDesign:
    settings: np.ndarray
    probabilities: np.ndarray
    diagonal_objective: float


def covers(pauli: np.ndarray, setting: np.ndarray) -> bool:
    return bool(np.all((pauli == 0) | (pauli == setting)))


def _compatible(pauli: np.ndarray, setting: np.ndarray) -> bool:
    return bool(np.all((pauli == 0) | (setting == 0) | (pauli == setting)))


def _merge_setting(setting: np.ndarray, pauli: np.ndarray) -> np.ndarray:
    merged = setting.copy()
    fill = (merged == 0) & (pauli != 0)
    merged[fill] = pauli[fill]
    return merged


def coverage_matrix(paulis: np.ndarray, settings: np.ndarray) -> np.ndarray:
    return np.all(
        (paulis[:, :, None] == 0) | (paulis[:, :, None] == settings.T[None, :, :]),
        axis=1,
    )


def _objective(hit: np.ndarray, coefficients: np.ndarray, probabilities: np.ndarray) -> float:
    coverage = hit @ probabilities
    if np.any(coverage <= 1e-15):
        return 1e30
    return float(np.sum(coefficients**2 / coverage))


def design_ogm(
    hamiltonian: Hamiltonian,
    shot_budget: int = 100_000,
    maxiter: int = 200,
) -> MeasurementDesign:
    """Construct and optimize the overlapped-grouping distribution used by Compact."""

    if hamiltonian.num_terms == 0:
        raise ValueError("Cannot design measurements for an identity-only Hamiltonian")
    if shot_budget <= 0:
        raise ValueError("shot_budget must be positive")

    order = np.argsort(-np.abs(hamiltonian.coefficients), kind="stable")
    paulis = hamiltonian.paulis[order]
    coefficients = hamiltonian.coefficients[order]
    added = np.zeros(len(paulis), dtype=bool)
    settings: list[np.ndarray] = []
    initial_weights: list[float] = []

    while not added.all():
        first = int(np.flatnonzero(~added)[0])
        setting = paulis[first].copy()
        added[first] = True
        weight = abs(float(coefficients[first]))
        for index in range(first + 1, len(paulis)):
            # This deliberately follows the paper's OGM implementation: a term
            # already absorbed by an earlier group may still enlarge this setting.
            if _compatible(paulis[index], setting):
                setting = _merge_setting(setting, paulis[index])
                added[index] = True
                weight += abs(float(coefficients[index]))
        for index in range(first):
            if _compatible(paulis[index], setting):
                setting = _merge_setting(setting, paulis[index])
        settings.append(setting)
        initial_weights.append(weight)

    settings_array = np.asarray(settings, dtype=int)
    probabilities = np.asarray(initial_weights, dtype=float)
    probabilities /= probabilities.sum()

    sort_order = np.argsort(-probabilities, kind="stable")
    settings_array = settings_array[sort_order]
    probabilities = probabilities[sort_order]
    cumulative_ceil = np.cumsum(np.ceil(shot_budget * probabilities).astype(int))
    keep = int(np.searchsorted(cumulative_ceil, shot_budget, side="left") + 1)
    candidate_settings = settings_array[:keep]
    candidate_probabilities = probabilities[:keep]
    candidate_probabilities /= candidate_probabilities.sum()
    candidate_hit = coverage_matrix(paulis, candidate_settings).astype(float)
    if np.all(candidate_hit.sum(axis=1) > 0):
        settings_array = candidate_settings
        probabilities = candidate_probabilities

    hit = coverage_matrix(paulis, settings_array).astype(float)
    constraints = ({"type": "eq", "fun": lambda values: np.sum(values) - 1.0},)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.optimize")
        result = minimize(
            lambda values: _objective(hit, coefficients, values),
            probabilities,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(probabilities),
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
        )
    if result.success and np.isfinite(result.fun):
        probabilities = np.clip(result.x, 0.0, None)
        probabilities /= probabilities.sum()

    if np.any((hit @ probabilities) <= 1e-12):
        raise RuntimeError("OGM optimization left at least one Hamiltonian term uncovered")
    order = np.argsort(-probabilities, kind="stable")
    settings_array = settings_array[order]
    probabilities = probabilities[order]
    return MeasurementDesign(
        settings=settings_array,
        probabilities=probabilities,
        diagonal_objective=_objective(hit[:, order], coefficients, probabilities),
    )


def allocate_counts(probabilities: np.ndarray, shots: int) -> np.ndarray:
    if shots <= 0:
        raise ValueError("shots must be positive")
    raw = shots * np.asarray(probabilities, dtype=float)
    counts = np.floor(raw).astype(int)
    remainder = int(shots - counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def sample_schedule(design: MeasurementDesign, shots: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.choice(len(design.probabilities), size=shots, p=design.probabilities)
    return design.settings[indices]

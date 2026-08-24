from __future__ import annotations

import numpy as np

from .hamiltonian import Hamiltonian
from .measurement import MeasurementDesign, allocate_counts, coverage_matrix
from .pauli import pauli_matrix


def single_shot_variance(
    hamiltonian: Hamiltonian,
    rho: np.ndarray,
    settings: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Exact OGM single-shot variance for a fixed measurement distribution."""

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    physical_settings = np.asarray(settings, dtype=int).copy()
    physical_settings[physical_settings == 0] = 3
    coverage = coverage_matrix(hamiltonian.paulis, physical_settings).astype(float)
    chi = coverage @ probabilities
    measured = chi > 1e-14
    if not measured.all():
        raise ValueError(
            f"Variance is undefined: {np.count_nonzero(~measured)} Hamiltonian terms are uncovered"
        )

    matrices = [pauli_matrix(pauli) for pauli in hamiltonian.paulis]
    term_expectations = np.asarray(
        [float(np.trace(rho @ matrix).real) for matrix in matrices], dtype=float
    )
    observable_expectation = float(np.dot(hamiltonian.coefficients, term_expectations))

    second_moment = 0.0
    for first in range(hamiltonian.num_terms):
        for second in range(hamiltonian.num_terms):
            joint_probability = float(
                np.sum(probabilities[(coverage[first] > 0) & (coverage[second] > 0)])
            )
            if joint_probability == 0:
                continue
            pair_expectation = float(
                np.trace(rho @ (matrices[first] @ matrices[second])).real
            )
            second_moment += (
                hamiltonian.coefficients[first]
                * hamiltonian.coefficients[second]
                * joint_probability
                * pair_expectation
                / (chi[first] * chi[second])
            )
    variance = second_moment - observable_expectation**2
    if variance < -1e-8:
        raise ValueError(f"Computed a negative variance ({variance})")
    return float(max(variance, 0.0))


def variance_for_shots(
    hamiltonian: Hamiltonian,
    rho: np.ndarray,
    design: MeasurementDesign,
    shots: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for total_shots in shots:
        counts = allocate_counts(design.probabilities, total_shots)
        keep = counts > 0
        probabilities = counts[keep] / counts[keep].sum()
        try:
            variance_one = single_shot_variance(
                hamiltonian, rho, design.settings[keep], probabilities
            )
            variance_total = variance_one / total_shots
            standard_error = float(np.sqrt(variance_total))
            uncovered = 0
        except ValueError as error:
            if "uncovered" not in str(error):
                raise
            physical = design.settings[keep].copy()
            physical[physical == 0] = 3
            coverage = coverage_matrix(hamiltonian.paulis, physical)
            uncovered = int(np.count_nonzero(~coverage.any(axis=1)))
            variance_one = None
            variance_total = None
            standard_error = None
        rows.append(
            {
                "shots": int(total_shots),
                "single_shot_variance": variance_one,
                "variance_of_mean": variance_total,
                "standard_error": standard_error,
                "uncovered_terms": uncovered,
            }
        )
    return rows

from __future__ import annotations

import numpy as np

from compact_measurement.estimator import ExperimentalArchive
from compact_measurement.hamiltonian import (
    lexicographic_permutation_twirl,
    load_hamiltonian,
    paper_permutation_twirl,
)
from compact_measurement.measurement import coverage_matrix, design_ogm
from compact_measurement.nonlinear import two_copy_observable
from compact_measurement.pauli import expectation, ghz_expectation
from compact_measurement.states import ideal_density, load_density_matrix
from compact_measurement.workflows import EXPERIMENT_ROOT, HAMILTONIAN_ROOT, STATE_ROOT


def test_legacy_compact_h4_and_ogm_design() -> None:
    compact = paper_permutation_twirl(load_hamiltonian(HAMILTONIAN_ROOT / "H_4.txt"))
    assert compact.num_terms == 12
    assert np.allclose(compact.coefficients, 0.091795)
    design = design_ogm(compact)
    expected_settings = {
        (1, 1, 3, 3),
        (3, 3, 1, 3),
        (3, 3, 1, 1),
        (3, 1, 3, 1),
        (1, 3, 3, 1),
    }
    assert {tuple(row) for row in design.settings} == expected_settings
    assert coverage_matrix(compact.paulis, design.settings).any(axis=1).all()


def test_scaling_twirl_preserves_ghz_expectation() -> None:
    original = load_hamiltonian(HAMILTONIAN_ROOT / "H_8.txt")
    compact = lexicographic_permutation_twirl(original)
    assert compact.num_terms == 56
    assert ghz_expectation(original) == ghz_expectation(compact) == 0.0


def test_nonlinear_generation_matches_paper_shape() -> None:
    h3 = load_hamiltonian(HAMILTONIAN_ROOT / "H_3.txt")
    paper_original = two_copy_observable(h3)
    paper_compact = two_copy_observable(paper_permutation_twirl(h3))
    signed_real_component = two_copy_observable(h3, paper_positive_only=False)
    assert paper_original.num_terms == 128
    assert paper_compact.num_terms == 144
    assert signed_real_component.num_terms == 192
    assert np.allclose(paper_compact.coefficients, 0.044464375)


def test_symmetric_state_expectation_is_preserved() -> None:
    for state_name in ("GHZ", "W"):
        rho = ideal_density(state_name, 4)
        original = load_hamiltonian(HAMILTONIAN_ROOT / "H_4.txt")
        compact = paper_permutation_twirl(original)
        assert np.isclose(expectation(rho, original), expectation(rho, compact), atol=1e-12)


def test_tomography_inputs_are_valid() -> None:
    for path in sorted(STATE_ROOT.glob("*.mat")):
        rho = load_density_matrix(path)
        assert np.isclose(np.trace(rho), 1.0)
        assert np.min(np.linalg.eigvalsh(rho)) >= -1e-8


def test_experimental_archive_matches_basis_count() -> None:
    archive = ExperimentalArchive(
        EXPERIMENT_ROOT / "bin_GHZ3.zip",
        EXPERIMENT_ROOT / "pauli_GHZ3.csv",
    )
    outcomes = archive.outcomes(np.array([1, 1, 1]))
    assert outcomes.shape == (1_304_773, 3)
    assert np.isin(outcomes, [0, 1]).all()

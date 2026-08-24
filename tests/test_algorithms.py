from __future__ import annotations

import numpy as np

from compact_measurement.estimator import (
    ExperimentalArchive,
    estimate_experimental_nonlinear_unbiased,
)
from compact_measurement.hamiltonian import (
    Hamiltonian,
    lexicographic_permutation_twirl,
    load_hamiltonian,
    paper_permutation_twirl,
)
from compact_measurement.measurement import MeasurementDesign, coverage_matrix, design_ogm
from compact_measurement.nonlinear import direct_nonlinear_expectation, two_copy_observable
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


def test_nonlinear_generation_keeps_signed_real_coefficients() -> None:
    h3 = load_hamiltonian(HAMILTONIAN_ROOT / "H_3.txt")
    signed_original = two_copy_observable(h3)
    signed_compact = two_copy_observable(paper_permutation_twirl(h3))
    paper_original = two_copy_observable(h3, paper_positive_only=True)
    paper_compact = two_copy_observable(
        paper_permutation_twirl(h3), paper_positive_only=True
    )
    assert signed_original.num_terms == 192
    assert signed_compact.num_terms == 192
    assert np.count_nonzero(signed_original.coefficients < 0) == 64
    assert np.count_nonzero(signed_compact.coefficients < 0) == 48
    assert paper_original.num_terms == 128
    assert paper_compact.num_terms == 144
    assert np.allclose(paper_compact.coefficients, 0.044464375)


def test_signed_nonlinear_expansion_matches_direct_matrix_target() -> None:
    h3 = load_hamiltonian(HAMILTONIAN_ROOT / "H_3.txt")
    observable = two_copy_observable(h3)
    rng = np.random.default_rng(20260824)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = raw @ raw.conj().T
    rho /= np.trace(rho)
    direct = direct_nonlinear_expectation(rho, h3)
    expanded = expectation(np.kron(rho, rho), observable)
    assert np.isclose(expanded, direct, atol=1e-12)


def test_unbiased_experimental_nonlinear_uses_independent_copy_slices() -> None:
    class RecordingArchive:
        num_qubits = 1

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], int, int]] = []

        def outcome_slice(self, setting, start, count):
            self.calls.append((tuple(int(value) for value in setting), start, count))
            return np.zeros((count, 1), dtype=np.uint8)

    observable = Hamiltonian(
        np.asarray([1.0]), np.asarray([[1, 1]], dtype=int)
    )
    design = MeasurementDesign(
        settings=np.asarray([[1, 1]], dtype=int),
        probabilities=np.asarray([1.0]),
        diagonal_objective=1.0,
    )
    archive = RecordingArchive()
    batches = estimate_experimental_nonlinear_unbiased(
        observable, design, archive, shots=[4], repeats=2, seed=123
    )
    assert np.allclose(batches[4].estimates, 1.0)
    assert archive.calls == [
        ((1,), 0, 4),
        ((1,), 4, 4),
        ((1,), 8, 4),
        ((1,), 12, 4),
    ]


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

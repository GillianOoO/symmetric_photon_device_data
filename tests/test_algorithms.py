from __future__ import annotations

import zipfile

import numpy as np

from compact_measurement.estimator import (
    EstimateBatch,
    ExperimentalArchive,
    _estimate_from_moments,
    _unbiased_estimate_from_moments,
    estimate_experimental_nonlinear_unbiased,
    summarize_batches,
)
from compact_measurement.hamiltonian import (
    Hamiltonian,
    generate_spin_hamiltonian,
    lexicographic_permutation_twirl,
    load_hamiltonian,
    paper_permutation_twirl,
)
from compact_measurement.measurement import MeasurementDesign, coverage_matrix, design_ogm
from compact_measurement.nonlinear import direct_nonlinear_expectation, two_copy_observable
from compact_measurement.pauli import expectation, ghz_expectation
from compact_measurement.states import ideal_density, load_density_matrix
from compact_measurement.sym_average import simulate_symmetry_average
from compact_measurement.variance import single_shot_variance
from compact_measurement.workflows import (
    EXPERIMENT_ROOT,
    DEFAULT_REPEATS,
    HAMILTONIAN_ROOT,
    STATE_ROOT,
)


def test_ordered_compact_h4_and_ogm_design() -> None:
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


def test_scaling_hamiltonians_match_generation_seeds() -> None:
    for num_qubits, seed in ((8, 20260508), (12, 20260512), (14, 20260513)):
        stored = load_hamiltonian(HAMILTONIAN_ROOT / f"H_{num_qubits}.txt")
        _, generated = generate_spin_hamiltonian(num_qubits, seed)
        stored_terms = {
            tuple(pauli): coefficient
            for coefficient, pauli in zip(stored.coefficients, stored.paulis)
        }
        generated_terms = {
            tuple(pauli): round(float(coefficient), 6)
            for coefficient, pauli in zip(generated.coefficients, generated.paulis)
        }
        assert stored_terms.keys() == generated_terms.keys()
        assert all(
            np.isclose(stored_terms[key], generated_terms[key], atol=5e-7)
            for key in stored_terms
        )


def test_nonlinear_generation_keeps_both_coefficient_signs() -> None:
    h3 = load_hamiltonian(HAMILTONIAN_ROOT / "H_3.txt")
    original = two_copy_observable(h3)
    compact = two_copy_observable(paper_permutation_twirl(h3))
    assert original.num_terms == 192
    assert compact.num_terms == 192
    assert np.count_nonzero(original.coefficients < 0) == 64
    assert np.count_nonzero(original.coefficients > 0) == 128
    assert np.count_nonzero(compact.coefficients < 0) == 48
    assert np.count_nonzero(compact.coefficients > 0) == 144


def test_nonlinear_expansion_matches_direct_matrix_target() -> None:
    h3 = load_hamiltonian(HAMILTONIAN_ROOT / "H_3.txt")
    observable = two_copy_observable(h3)
    rng = np.random.default_rng(20260824)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = raw @ raw.conj().T
    rho /= np.trace(rho)
    direct = direct_nonlinear_expectation(rho, h3)
    expanded = expectation(np.kron(rho, rho), observable)
    assert np.isclose(expanded, direct, atol=1e-12)


def test_figure_repeat_defaults() -> None:
    assert DEFAULT_REPEATS == 20


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


def test_all_estimators_zero_fill_uncovered_terms_and_use_full_reference() -> None:
    observable = Hamiltonian(
        np.asarray([2.0, -3.0]),
        np.asarray([[1], [3]], dtype=int),
    )
    moments = {(1,): np.asarray([4.0, 4.0])}

    fixed_estimate, fixed_uncovered = _estimate_from_moments(
        observable,
        setting_counts={(1,): 4},
        setting_moments=moments,
    )
    assert fixed_estimate == 2.0
    assert fixed_uncovered == 1

    design = MeasurementDesign(
        settings=np.asarray([[1], [3]], dtype=int),
        probabilities=np.asarray([0.5, 0.5]),
        diagonal_objective=0.0,
    )
    ogm_estimate, ogm_uncovered = _unbiased_estimate_from_moments(
        observable,
        design,
        total_shots=4,
        setting_counts={(1,): 4},
        setting_moments=moments,
    )
    assert ogm_estimate == 4.0
    assert ogm_uncovered == 1

    full_reference = -1.0
    fixed_summary = summarize_batches(
        {4: EstimateBatch(np.asarray([fixed_estimate]), fixed_uncovered)},
        full_reference,
    )[0]
    ogm_summary = summarize_batches(
        {4: EstimateBatch(np.asarray([ogm_estimate]), ogm_uncovered)},
        full_reference,
    )[0]
    assert fixed_summary["rmse"] == 3.0
    assert ogm_summary["rmse"] == 5.0


def test_symmetry_average_reports_maximum_uncovered_terms() -> None:
    observable = Hamiltonian(np.asarray([1.0]), np.asarray([[1]], dtype=int))
    design = MeasurementDesign(
        settings=np.asarray([[1], [3]], dtype=int),
        probabilities=np.asarray([0.5, 0.5]),
        diagonal_objective=0.0,
    )
    batches = simulate_symmetry_average(
        observable, design, shots=[1], repeats=3, seed=1
    )
    assert batches[1].uncovered_terms == 1


def test_variance_uses_coefficient_signs_in_cross_terms() -> None:
    observable = Hamiltonian(
        np.asarray([2.0, -3.0]), np.asarray([[1], [3]], dtype=int)
    )
    rho = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    settings = np.asarray([[1], [3]], dtype=int)
    probabilities = np.asarray([0.25, 0.75])
    expected = 2.0**2 / 0.25 + (-3.0) ** 2 / 0.75 - (-3.0) ** 2
    assert np.isclose(
        single_shot_variance(observable, rho, settings, probabilities), expected
    )


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


def test_experimental_archives_match_basis_counts() -> None:
    for state in ("W3", "GHZ3", "W4", "GHZ4"):
        archive = ExperimentalArchive(
            EXPERIMENT_ROOT / f"bin_{state}.zip",
            EXPERIMENT_ROOT / f"pauli_{state}.csv",
        )
        with zipfile.ZipFile(archive.zip_path) as compressed:
            members = sorted(
                (item for item in compressed.infolist() if not item.is_dir()),
                key=lambda item: int(item.filename.rsplit("/", 1)[-1].split(".")[0]),
            )
        assert len(members) == len(archive.basis_to_index)
        for member, available in zip(members, archive.available_counts):
            assert member.file_size == int(available) * archive.num_qubits

    archive = ExperimentalArchive(
        EXPERIMENT_ROOT / "bin_GHZ3.zip",
        EXPERIMENT_ROOT / "pauli_GHZ3.csv",
    )
    outcomes = archive.outcomes(np.array([1, 1, 1]))
    assert outcomes.shape == (1_304_773, 3)
    assert np.isin(outcomes, [0, 1]).all()

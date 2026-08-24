from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimator import (
    ExperimentalArchive,
    estimate_experimental_linear,
    estimate_experimental_nonlinear,
    simulate_compact,
    summarize_batches,
)
from .hamiltonian import (
    generate_spin_hamiltonian,
    lexicographic_permutation_twirl,
    load_hamiltonian,
    paper_permutation_twirl,
)
from .measurement import design_ogm
from .nonlinear import two_copy_observable
from .pauli import expectation, ghz_expectation
from .states import ideal_density, load_density_matrix
from .sym_average import simulate_symmetry_average
from .variance import single_shot_variance


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUTS_ROOT = REPOSITORY_ROOT / "inputs"
HAMILTONIAN_ROOT = INPUTS_ROOT / "hamiltonians"
STATE_ROOT = INPUTS_ROOT / "states"
EXPERIMENT_ROOT = INPUTS_ROOT / "experimental_counts"

LINEAR_SHOTS = [12, 45, 160, 572, 2038, 7259, 25848]
NONLINEAR_SHOTS = [12, 45, 160, 572, 2038, 7259]


@dataclass(frozen=True)
class PaperCase:
    panel: str
    state: str
    hamiltonian_file: str
    num_qubits: int
    nonlinear: bool = False


SIX_PANEL_CASES = (
    PaperCase("a", "W", "rand_H_3.txt", 3),
    PaperCase("b", "GHZ", "rand_H_3.txt", 3),
    PaperCase("c", "W", "H_4.txt", 4),
    PaperCase("d", "GHZ", "H_4.txt", 4),
    PaperCase("e", "W", "H_3.txt", 3, True),
    PaperCase("f", "GHZ", "H_3.txt", 3, True),
)


def _paper_compact(hamiltonian):
    return paper_permutation_twirl(hamiltonian)


def _observables_for_case(case: PaperCase):
    physical = load_hamiltonian(HAMILTONIAN_ROOT / case.hamiltonian_file)
    if case.nonlinear:
        return two_copy_observable(physical), two_copy_observable(_paper_compact(physical))
    return physical, _paper_compact(physical)


def _state_file(case: PaperCase) -> Path:
    return STATE_ROOT / f"rho_{case.state}{case.num_qubits}.mat"


def _archive(case: PaperCase) -> ExperimentalArchive:
    return ExperimentalArchive(
        EXPERIMENT_ROOT / f"bin_{case.state}{case.num_qubits}.zip",
        EXPERIMENT_ROOT / f"pauli_{case.state}{case.num_qubits}.csv",
    )


def _case_state(case: PaperCase, noisy: bool) -> np.ndarray:
    rho = load_density_matrix(_state_file(case)) if noisy else ideal_density(case.state, case.num_qubits)
    return np.kron(rho, rho) if case.nonlinear else rho


def run_six_panel_compact(
    source: str,
    reference_kind: str,
    shots: list[int] | None,
    repeats: int,
    seed: int,
    ogm_budget: int,
) -> dict[str, object]:
    if source not in {"experiment", "simulation"}:
        raise ValueError("source must be 'experiment' or 'simulation'")
    if reference_kind not in {"ideal", "tomography"}:
        raise ValueError("reference_kind must be 'ideal' or 'tomography'")

    design_cache: dict[str, object] = {}
    archive_cache: dict[tuple[str, int], ExperimentalArchive] = {}
    panels: list[dict[str, object]] = []
    for case_index, case in enumerate(SIX_PANEL_CASES):
        original, compact = _observables_for_case(case)
        design = design_cache.get(case.hamiltonian_file)
        if design is None:
            design = design_ogm(compact, shot_budget=ogm_budget)
            design_cache[case.hamiltonian_file] = design

        case_shots = list(shots) if shots else (
            NONLINEAR_SHOTS if case.nonlinear else LINEAR_SHOTS
        )
        reference_rho = _case_state(case, noisy=reference_kind == "tomography")
        reference_observable = compact if reference_kind == "tomography" else original
        reference = expectation(reference_rho, reference_observable)
        measured_state = _case_state(case, noisy=source == "experiment")
        compact_target = expectation(measured_state, compact)

        if source == "experiment":
            archive_key = (case.state, case.num_qubits)
            archive = archive_cache.get(archive_key)
            if archive is None:
                archive = _archive(case)
                archive_cache[archive_key] = archive
            if case.nonlinear:
                batches = estimate_experimental_nonlinear(
                    compact, design, archive, case_shots, repeats
                )
            else:
                batches = estimate_experimental_linear(
                    compact, design, archive, case_shots, repeats
                )
        else:
            if case.state == "GHZ" and not case.nonlinear:
                batches = simulate_compact(
                    compact,
                    design,
                    case_shots,
                    repeats,
                    seed + 10_000 * case_index,
                    state_kind="ghz",
                )
            else:
                batches = simulate_compact(
                    compact,
                    design,
                    case_shots,
                    repeats,
                    seed + 10_000 * case_index,
                    rho=measured_state,
                )

        panels.append(
            {
                "panel": case.panel,
                "state": case.state,
                "observable": "tr(rho^2 H)" if case.nonlinear else "tr(rho H)",
                "input_hamiltonian": f"inputs/hamiltonians/{case.hamiltonian_file}",
                "input_state": (
                    f"inputs/states/{_state_file(case).name}"
                    if source == "experiment" or reference_kind == "tomography"
                    else f"analytic {case.state}{case.num_qubits} state"
                ),
                "method": "Compact (permutation twirl + OGM)",
                "num_compact_terms": compact.num_terms,
                "num_measurement_settings": int(len(design.settings)),
                "ogm_diagonal_objective": float(design.diagonal_objective),
                "compact_observable_expectation_on_measured_state": compact_target,
                "results": summarize_batches(batches, reference),
            }
        )

    return {
        "source": source,
        "error_reference": reference_kind,
        "repeats": repeats,
        "panels": panels,
    }


def run_si_figure_1(
    shots: list[int], repeats: int, seed: int, ogm_budget: int
) -> dict[str, object]:
    original = load_hamiltonian(HAMILTONIAN_ROOT / "H_8.txt")
    compact = lexicographic_permutation_twirl(original)
    original_design = design_ogm(original, shot_budget=ogm_budget)
    compact_design = design_ogm(compact, shot_budget=ogm_budget)
    exact = ghz_expectation(original)
    compact_batches = simulate_compact(
        compact,
        compact_design,
        shots,
        repeats,
        seed,
        state_kind="ghz",
    )
    sym_average_batches = simulate_symmetry_average(
        original,
        original_design,
        shots,
        repeats,
        seed + 500_000,
    )
    return {
        "input_hamiltonian": "inputs/hamiltonians/H_8.txt",
        "input_state": "analytic GHZ8 state",
        "error_reference": exact,
        "methods": {
            "Compact": summarize_batches(compact_batches, exact),
            "sym_average": summarize_batches(sym_average_batches, exact),
        },
        "note": "Only Compact and sym_average are included; SG, Derand, AP and baseline OGM are intentionally omitted.",
    }


def run_si_figure_9(
    shots: list[int], repeats: int, seed: int, ogm_budget: int
) -> dict[str, object]:
    fixed_seeds = {8: 20260508, 12: 20260512, 14: 20260513}
    systems: list[dict[str, object]] = []
    for index, (num_qubits, generation_seed) in enumerate(fixed_seeds.items()):
        path = HAMILTONIAN_ROOT / f"H_{num_qubits}.txt"
        original = load_hamiltonian(path)
        _, generated = generate_spin_hamiltonian(num_qubits, generation_seed)
        stored_terms = {
            tuple(int(value) for value in pauli): float(coefficient)
            for coefficient, pauli in zip(original.coefficients, original.paulis)
        }
        generated_terms = {
            tuple(int(value) for value in pauli): float(np.round(coefficient, 6))
            for coefficient, pauli in zip(generated.coefficients, generated.paulis)
        }
        if stored_terms.keys() != generated_terms.keys() or any(
            not np.isclose(stored_terms[key], generated_terms[key], atol=5e-7)
            for key in stored_terms
        ):
            raise ValueError(f"Stored H_{num_qubits} does not match seed {generation_seed}")
        compact = lexicographic_permutation_twirl(original)
        design = design_ogm(compact, shot_budget=ogm_budget)
        exact = ghz_expectation(original)
        batches = simulate_compact(
            compact,
            design,
            shots,
            repeats,
            seed + 100_000 * index,
            state_kind="ghz",
        )
        systems.append(
            {
                "num_qubits": num_qubits,
                "hamiltonian_seed": generation_seed,
                "input_hamiltonian": f"inputs/hamiltonians/H_{num_qubits}.txt",
                "method": "Compact (permutation twirl + OGM)",
                "num_terms": compact.num_terms,
                "num_measurement_settings": int(len(design.settings)),
                "results": summarize_batches(batches, exact),
            }
        )
    return {"input_state": "analytic GHZ_n state", "systems": systems}


def _variance_cases(ogm_budget: int) -> list[dict[str, object]]:
    design_cache: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for case in SIX_PANEL_CASES:
        original, compact = _observables_for_case(case)
        design = design_cache.get(case.hamiltonian_file)
        if design is None:
            design = design_ogm(compact, shot_budget=ogm_budget)
            design_cache[case.hamiltonian_file] = design
        rho = _case_state(case, noisy=True)
        variance_one = single_shot_variance(
            compact, rho, design.settings, design.probabilities
        )
        rows.append(
            {
                "panel": case.panel,
                "state": case.state,
                "observable": "nonlinear SpinH" if case.nonlinear else (
                    "SpinH" if case.num_qubits == 4 else "RandH"
                ),
                "input_hamiltonian": f"inputs/hamiltonians/{case.hamiltonian_file}",
                "input_state": f"inputs/states/{_state_file(case).name}",
                "single_shot_variance": variance_one,
            }
        )
    return rows


def run_si_figure_11(shots: list[int], ogm_budget: int) -> dict[str, object]:
    panels = _variance_cases(ogm_budget)
    for panel in panels:
        variance_one = float(panel["single_shot_variance"])
        panel["results"] = [
            {
                "shots": int(total_shots),
                "variance_of_mean": variance_one / total_shots,
                "standard_error": float(np.sqrt(variance_one / total_shots)),
            }
            for total_shots in shots
        ]
    return {
        "method": "Compact (permutation twirl + OGM)",
        "state_model": "experimental tomography density matrices",
        "panels": panels,
    }


def run_si_table_1(ogm_budget: int) -> dict[str, object]:
    return {
        "method": "Compact (permutation twirl + OGM)",
        "reported_quantity": "single-shot variance",
        "rows": _variance_cases(ogm_budget),
        "note": "Only the Compact column is generated, as requested.",
    }


def emit_json(payload: dict[str, object], output: str | None = None) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)

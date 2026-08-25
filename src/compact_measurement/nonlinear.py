from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np

from .hamiltonian import Hamiltonian
from .pauli import pauli_matrix


_PAULI_PRODUCT = {
    (0, 0): (0, 1), (0, 1): (1, 1), (0, 2): (2, 1), (0, 3): (3, 1),
    (1, 0): (1, 1), (1, 1): (0, 1), (1, 2): (3, 1j), (1, 3): (2, -1j),
    (2, 0): (2, 1), (2, 1): (3, -1j), (2, 2): (0, 1), (2, 3): (1, 1j),
    (3, 0): (3, 1), (3, 1): (2, 1j), (3, 2): (1, -1j), (3, 3): (0, 1),
}


def two_copy_observable(
    hamiltonian: Hamiltonian,
    tolerance: float = 1e-12,
) -> Hamiltonian:
    """Expand the Hermitian two-copy observable for ``Tr(rho^2 H)``.

    The returned operator is the Hermitian part of ``(H tensor I) SWAP``.
    Its Pauli coefficients are the real parts of the unsymmetrized expansion.
    Every coefficient whose absolute value exceeds ``tolerance`` is retained.
    """

    num_qubits = hamiltonian.num_qubits
    terms: dict[tuple[int, ...], complex] = defaultdict(complex)
    if abs(hamiltonian.offset) > tolerance:
        for swap_pauli in product(range(4), repeat=num_qubits):
            key = tuple(int(value) for value in swap_pauli) * 2
            terms[key] += hamiltonian.offset / (2**num_qubits)
    for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis):
        for swap_pauli in product(range(4), repeat=num_qubits):
            phase = 1 + 0j
            first_copy: list[int] = []
            for left, right in zip(pauli, swap_pauli):
                product_label, local_phase = _PAULI_PRODUCT[(int(left), int(right))]
                first_copy.append(product_label)
                phase *= local_phase
            key = tuple(first_copy) + tuple(int(value) for value in swap_pauli)
            terms[key] += float(coefficient) * phase / (2**num_qubits)

    rows: list[tuple[tuple[int, ...], float]] = []
    for pauli, coefficient in sorted(terms.items()):
        real = float(coefficient.real)
        if abs(real) > tolerance:
            rows.append((pauli, real))
    paulis = np.asarray([pauli for pauli, _ in rows], dtype=int).reshape(
        -1, 2 * num_qubits
    )
    return Hamiltonian(
        np.asarray([coefficient for _, coefficient in rows], dtype=float),
        paulis,
    )


def direct_nonlinear_expectation(rho: np.ndarray, hamiltonian: Hamiltonian) -> float:
    """Evaluate ``Tr(rho^2 H)`` without constructing the two-copy operator."""

    expected_dimension = 2**hamiltonian.num_qubits
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            f"State dimension {rho.shape} does not match {hamiltonian.num_qubits} qubits"
        )
    rho_squared = rho @ rho
    value = hamiltonian.offset * float(np.trace(rho_squared).real)
    for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis):
        value += coefficient * float(np.trace(rho_squared @ pauli_matrix(pauli)).real)
    return float(value)

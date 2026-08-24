from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np

from .hamiltonian import Hamiltonian


_PAULI_PRODUCT = {
    (0, 0): (0, 1), (0, 1): (1, 1), (0, 2): (2, 1), (0, 3): (3, 1),
    (1, 0): (1, 1), (1, 1): (0, 1), (1, 2): (3, 1j), (1, 3): (2, -1j),
    (2, 0): (2, 1), (2, 1): (3, -1j), (2, 2): (0, 1), (2, 3): (1, 1j),
    (3, 0): (3, 1), (3, 1): (2, 1j), (3, 2): (1, -1j), (3, 3): (0, 1),
}


def two_copy_observable(
    hamiltonian: Hamiltonian,
    paper_positive_only: bool = True,
    tolerance: float = 1e-12,
) -> Hamiltonian:
    """Expand ``(H tensor I) SWAP`` in the product-Pauli basis.

    The publication files retained only positive real coefficients. The default
    preserves that historical behavior. Set ``paper_positive_only=False`` to
    retain the complete signed real component.
    """

    num_qubits = hamiltonian.num_qubits
    terms: dict[tuple[int, ...], complex] = defaultdict(complex)
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
        if abs(coefficient.imag) > tolerance:
            continue
        real = float(coefficient.real)
        if paper_positive_only:
            if real > tolerance:
                rows.append((pauli, real))
        elif abs(real) > tolerance:
            rows.append((pauli, real))
    return Hamiltonian(
        np.asarray([coefficient for _, coefficient in rows], dtype=float),
        np.asarray([pauli for pauli, _ in rows], dtype=int),
    )

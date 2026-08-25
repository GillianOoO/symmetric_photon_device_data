from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Hamiltonian:
    """A real Pauli Hamiltonian using 0=I, 1=X, 2=Y, 3=Z."""

    coefficients: np.ndarray
    paulis: np.ndarray
    offset: float = 0.0

    def __post_init__(self) -> None:
        coeffs = np.asarray(self.coefficients, dtype=float).reshape(-1)
        paulis = np.asarray(self.paulis, dtype=int)
        if paulis.ndim != 2 or len(coeffs) != len(paulis):
            raise ValueError("Hamiltonian coefficients and Pauli rows have incompatible shapes")
        if paulis.size and not np.isin(paulis, [0, 1, 2, 3]).all():
            raise ValueError("Pauli labels must be integers in {0,1,2,3}")
        if not np.isfinite(coeffs).all() or not np.isfinite(self.offset):
            raise ValueError("Hamiltonian contains a non-finite coefficient")
        object.__setattr__(self, "coefficients", coeffs)
        object.__setattr__(self, "paulis", paulis)
        object.__setattr__(self, "offset", float(self.offset))

    @property
    def num_qubits(self) -> int:
        return int(self.paulis.shape[1])

    @property
    def num_terms(self) -> int:
        return int(len(self.coefficients))


def _aggregate(coefficients: np.ndarray, paulis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[tuple[int, ...], float] = defaultdict(float)
    for coefficient, pauli in zip(coefficients, paulis):
        merged[tuple(int(value) for value in pauli)] += float(coefficient)
    rows = [(pauli, coeff) for pauli, coeff in merged.items() if abs(coeff) > 1e-14]
    if not rows:
        width = int(paulis.shape[1]) if paulis.ndim == 2 else 0
        return np.zeros(0, dtype=float), np.zeros((0, width), dtype=int)
    return (
        np.asarray([coeff for _, coeff in rows], dtype=float),
        np.asarray([pauli for pauli, _ in rows], dtype=int),
    )


def load_hamiltonian(path: str | Path) -> Hamiltonian:
    data = np.loadtxt(Path(path), dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"Invalid Hamiltonian file: {path}")
    coefficients = np.asarray(data[:, 0], dtype=float)
    paulis_float = np.asarray(data[:, 1:], dtype=float)
    paulis = np.rint(paulis_float).astype(int)
    if not np.allclose(paulis_float, paulis, atol=1e-12):
        raise ValueError(f"Non-integer Pauli label in {path}")
    identity = np.all(paulis == 0, axis=1)
    offset = float(np.sum(coefficients[identity]))
    coefficients, paulis = _aggregate(coefficients[~identity], paulis[~identity])
    return Hamiltonian(coefficients, paulis, offset)


def save_hamiltonian(path: str | Path, hamiltonian: Hamiltonian) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[np.ndarray] = []
    if abs(hamiltonian.offset) > 1e-14:
        rows.append(np.concatenate(([hamiltonian.offset], np.zeros(hamiltonian.num_qubits))))
    rows.extend(
        np.concatenate(([coefficient], pauli.astype(float)))
        for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis)
    )
    np.savetxt(path, np.asarray(rows), fmt=["%.16g"] + ["%d"] * hamiltonian.num_qubits)


def _orbit_from_counts(num_qubits: int, counts: tuple[int, int, int, int]) -> np.ndarray:
    _, num_x, num_y, num_z = counts
    rows: list[np.ndarray] = []
    all_positions = tuple(range(num_qubits))
    for x_positions in combinations(all_positions, num_x):
        after_x = tuple(pos for pos in all_positions if pos not in x_positions)
        for y_positions in combinations(after_x, num_y):
            after_y = tuple(pos for pos in after_x if pos not in y_positions)
            for z_positions in combinations(after_y, num_z):
                row = np.zeros(num_qubits, dtype=int)
                row[list(x_positions)] = 1
                row[list(y_positions)] = 2
                row[list(z_positions)] = 3
                rows.append(row)
    return np.asarray(rows, dtype=int)


def permutation_twirl(hamiltonian: Hamiltonian) -> Hamiltonian:
    """Return the full qubit-permutation twirl used by Compact."""

    grouped: dict[tuple[int, int, int, int], float] = defaultdict(float)
    for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis):
        counter = Counter(int(value) for value in pauli)
        signature = tuple(counter.get(label, 0) for label in range(4))
        grouped[signature] += float(coefficient)

    out_coefficients: list[np.ndarray] = []
    out_paulis: list[np.ndarray] = []
    for signature, total in sorted(grouped.items()):
        orbit = _orbit_from_counts(hamiltonian.num_qubits, signature)
        out_paulis.append(orbit)
        out_coefficients.append(np.full(len(orbit), total / len(orbit), dtype=float))

    if not out_paulis:
        return Hamiltonian(np.zeros(0), np.zeros((0, hamiltonian.num_qubits), dtype=int), hamiltonian.offset)
    coefficients, paulis = _aggregate(np.concatenate(out_coefficients), np.vstack(out_paulis))
    return Hamiltonian(coefficients, paulis, hamiltonian.offset)


def round_coefficients(hamiltonian: Hamiltonian, decimals: int = 6) -> Hamiltonian:
    """Match the six-decimal text serialization used by the paper pipeline."""

    coefficients = np.round(hamiltonian.coefficients, decimals)
    offset = float(np.round(hamiltonian.offset, decimals))
    keep = np.abs(coefficients) >= 10 ** (-decimals)
    return Hamiltonian(coefficients[keep], hamiltonian.paulis[keep], offset)


def paper_permutation_twirl(hamiltonian: Hamiltonian) -> Hamiltonian:
    """Reproduce the ordered, six-decimal twirl used to generate the paper inputs."""

    merged: dict[tuple[int, ...], float] = defaultdict(float)
    for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis):
        orbit = set(permutations(tuple(int(value) for value in pauli)))
        distributed = float(coefficient) / len(orbit)
        for permuted in orbit:
            merged[permuted] += distributed
    rows = [
        (pauli, round(coefficient, 6))
        for pauli, coefficient in merged.items()
        if abs(round(coefficient, 6)) >= 1e-6
    ]
    return Hamiltonian(
        np.asarray([coefficient for _, coefficient in rows], dtype=float),
        np.asarray([pauli for pauli, _ in rows], dtype=int),
        round(hamiltonian.offset, 6),
    )


def lexicographic_permutation_twirl(hamiltonian: Hamiltonian) -> Hamiltonian:
    """Reproduce the ordered twirl used for the n=8,12,14 spin instances."""

    twirled = round_coefficients(permutation_twirl(hamiltonian), 6)
    order = sorted(range(twirled.num_terms), key=lambda index: tuple(twirled.paulis[index]))
    return Hamiltonian(
        twirled.coefficients[order],
        twirled.paulis[order],
        twirled.offset,
    )


def generate_spin_hamiltonian(num_qubits: int, seed: int) -> tuple[np.ndarray, Hamiltonian]:
    """Generate H=sum_(i!=j) J_ij Z_i X_j with the paper's NumPy RNG convention."""

    rng = np.random.RandomState(seed)
    couplings = rng.uniform(-1.0, 1.0, size=(num_qubits, num_qubits))
    np.fill_diagonal(couplings, 0.0)
    paulis: list[np.ndarray] = []
    coefficients: list[float] = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                continue
            row = np.zeros(num_qubits, dtype=int)
            row[i] = 3
            row[j] = 1
            paulis.append(row)
            coefficients.append(float(couplings[i, j]))
    return couplings, Hamiltonian(np.asarray(coefficients), np.asarray(paulis))

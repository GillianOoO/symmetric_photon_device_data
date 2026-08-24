from __future__ import annotations

import math

import numpy as np

from .hamiltonian import Hamiltonian


PAULI_MATRICES = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def pauli_matrix(pauli: np.ndarray) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=complex)
    for label in pauli:
        matrix = np.kron(matrix, PAULI_MATRICES[int(label)])
    return matrix


def expectation(rho: np.ndarray, hamiltonian: Hamiltonian) -> float:
    expected_dimension = 2**hamiltonian.num_qubits
    if rho.shape != (expected_dimension, expected_dimension):
        raise ValueError(
            f"State dimension {rho.shape} does not match {hamiltonian.num_qubits} qubits"
        )
    value = hamiltonian.offset
    for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis):
        value += coefficient * float(np.trace(rho @ pauli_matrix(pauli)).real)
    return float(value)


def ghz_pauli_expectation(pauli: np.ndarray) -> float:
    labels = set(int(value) for value in pauli)
    if labels.issubset({0, 3}):
        num_z = int(np.count_nonzero(pauli == 3))
        return 1.0 if num_z % 2 == 0 else 0.0
    if labels.issubset({1, 2}):
        num_y = int(np.count_nonzero(pauli == 2))
        if num_y % 2:
            return 0.0
        return 1.0 if num_y % 4 == 0 else -1.0
    return 0.0


def ghz_expectation(hamiltonian: Hamiltonian) -> float:
    value = hamiltonian.offset
    for coefficient, pauli in zip(hamiltonian.coefficients, hamiltonian.paulis):
        value += coefficient * ghz_pauli_expectation(pauli)
    return float(value)


def all_outcome_bits(num_qubits: int) -> np.ndarray:
    indices = np.arange(2**num_qubits, dtype=np.uint32)
    shifts = np.arange(num_qubits, dtype=np.uint32)
    return ((indices[:, None] >> shifts) & 1).astype(np.uint8)


def support_masks(paulis: np.ndarray) -> np.ndarray:
    if paulis.size == 0:
        return np.zeros(0, dtype=np.int64)
    bit_values = (1 << np.arange(paulis.shape[1], dtype=np.int64)).reshape(1, -1)
    return np.sum((paulis != 0).astype(np.int64) * bit_values, axis=1)


def fwht(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    width = len(out)
    if width == 0 or width & (width - 1):
        raise ValueError("Walsh-Hadamard input length must be a positive power of two")
    half = 1
    while half < width:
        for start in range(0, width, 2 * half):
            left = out[start : start + half].copy()
            right = out[start + half : start + 2 * half].copy()
            out[start : start + half] = left + right
            out[start + half : start + 2 * half] = left - right
        half *= 2
    return out


def ghz_measurement_probabilities(setting: np.ndarray) -> np.ndarray:
    bits = all_outcome_bits(len(setting))
    amp_zero = np.ones(len(bits), dtype=complex)
    amp_one = np.ones(len(bits), dtype=complex)
    inv_sqrt_two = 1 / math.sqrt(2)
    for qubit, basis in enumerate(setting):
        outcomes = bits[:, qubit]
        if basis == 1:
            amp_zero *= inv_sqrt_two
            amp_one *= np.where(outcomes == 0, inv_sqrt_two, -inv_sqrt_two)
        elif basis == 2:
            amp_zero *= inv_sqrt_two
            amp_one *= np.where(outcomes == 0, -1j * inv_sqrt_two, 1j * inv_sqrt_two)
        elif basis in (0, 3):
            amp_zero *= (outcomes == 0).astype(float)
            amp_one *= (outcomes == 1).astype(float)
        else:
            raise ValueError(f"Invalid measurement basis {basis}")
    probabilities = np.abs((amp_zero + amp_one) * inv_sqrt_two) ** 2
    return probabilities / probabilities.sum()


def density_measurement_probabilities(rho: np.ndarray, setting: np.ndarray) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    s_dagger = np.array([[1, 0], [0, -1j]], dtype=complex)
    unitary = np.ones((1, 1), dtype=complex)
    for basis in setting:
        if basis == 1:
            local = hadamard
        elif basis == 2:
            local = hadamard @ s_dagger
        elif basis in (0, 3):
            local = identity
        else:
            raise ValueError(f"Invalid measurement basis {basis}")
        unitary = np.kron(unitary, local)
    diagonal_big_endian = np.real(np.diag(unitary @ rho @ unitary.conj().T))
    diagonal_big_endian = np.clip(diagonal_big_endian, 0.0, None)

    bits = all_outcome_bits(len(setting))
    big_endian_indices = bits @ (1 << np.arange(len(setting) - 1, -1, -1))
    probabilities = diagonal_big_endian[big_endian_indices]
    return probabilities / probabilities.sum()

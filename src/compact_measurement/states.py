from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def ghz_density(num_qubits: int) -> np.ndarray:
    psi = np.zeros(2**num_qubits, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return np.outer(psi, psi.conj())


def w_density(num_qubits: int) -> np.ndarray:
    psi = np.zeros(2**num_qubits, dtype=complex)
    for qubit in range(num_qubits):
        psi[1 << (num_qubits - 1 - qubit)] = 1 / np.sqrt(num_qubits)
    return np.outer(psi, psi.conj())


def ideal_density(state_name: str, num_qubits: int) -> np.ndarray:
    normalized = state_name.upper()
    if normalized == "GHZ":
        return ghz_density(num_qubits)
    if normalized == "W":
        return w_density(num_qubits)
    raise ValueError(f"Unknown state {state_name!r}")


def load_density_matrix(path: str | Path) -> np.ndarray:
    data = loadmat(Path(path))
    if "rho" not in data:
        raise ValueError(f"MAT file has no 'rho' variable: {path}")
    rho = np.asarray(data["rho"], dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"Density matrix is not square: {path}")
    dimension = rho.shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError(f"Density matrix dimension is not a power of two: {path}")
    if not np.allclose(rho, rho.conj().T, atol=1e-8):
        raise ValueError(f"Density matrix is not Hermitian: {path}")
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=1e-6):
        raise ValueError(f"Density matrix trace is {trace}, expected 1: {path}")
    if np.min(np.linalg.eigvalsh(rho)) < -1e-6:
        raise ValueError(f"Density matrix is not positive semidefinite: {path}")
    return rho / trace

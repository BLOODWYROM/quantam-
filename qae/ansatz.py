"""Exact (noise-free) matrix simulator of the QAE ansatz.

Ansatz (identical to the Qiskit circuit in circuits.py)
-------------------------------------------------------
    U(theta) = R_L . [E . R_{L-1}] ... [E . R_0]       (R_0 acts first)
    R_l = RY(theta_{l,3}) (x) RY(theta_{l,2}) (x) RY(theta_{l,1}) (x) RY(theta_{l,0})
    E   = CX(2,3) . CX(1,2) . CX(0,1)                  (linear entangler)

All gates are real, so U is a real orthogonal 16x16 matrix.  Qubit q is bit q of the
basis-state index (Qiskit little-endian convention), so kron order is q3 (x) q2 (x) q1 (x) q0.
"""
from __future__ import annotations
import numpy as np
from . import N_QUBITS, DIM


def n_params(layers: int, n: int = N_QUBITS) -> int:
    return n * (layers + 1)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def cnot_matrix(control: int, target: int, n: int = N_QUBITS) -> np.ndarray:
    dim = 2 ** n
    P = np.zeros((dim, dim))
    for i in range(dim):
        j = i ^ (1 << target) if (i >> control) & 1 else i
        P[j, i] = 1.0
    return P


def entangler(n: int = N_QUBITS) -> np.ndarray:
    E = np.eye(2 ** n)
    for q in range(n - 1):
        E = cnot_matrix(q, q + 1, n) @ E
    return E


_E = entangler()


def _ry_layers_batch(angles: np.ndarray) -> np.ndarray:
    """angles [B,4] -> batch of 16x16 matrices RY(a3)(x)RY(a2)(x)RY(a1)(x)RY(a0)."""
    c, s = np.cos(angles / 2), np.sin(angles / 2)          # [B,4]
    B = angles.shape[0]
    g = np.empty((B, 4, 2, 2))
    g[:, :, 0, 0], g[:, :, 0, 1], g[:, :, 1, 0], g[:, :, 1, 1] = c, -s, s, c
    # kron(g3, g2, g1, g0) via einsum on the 4 batched 2x2 factors
    full = np.einsum("bap,bcq,bdr,bes->bacdepqrs", g[:, 3], g[:, 2], g[:, 1], g[:, 0])
    return full.reshape(B, DIM, DIM)


def build_unitaries(thetas: np.ndarray, layers: int) -> np.ndarray:
    """Batch builder. thetas [B, (layers+1)*4] -> U [B,16,16]."""
    thetas = np.atleast_2d(thetas)
    B = thetas.shape[0]
    th = thetas.reshape(B, layers + 1, N_QUBITS)
    U = np.broadcast_to(np.eye(DIM), (B, DIM, DIM)).copy()
    for l in range(layers):
        U = np.matmul(_E, np.matmul(_ry_layers_batch(th[:, l]), U))
    return np.matmul(_ry_layers_batch(th[:, layers]), U)


def build_unitary(theta: np.ndarray, layers: int) -> np.ndarray:
    return build_unitaries(np.asarray(theta)[None, :], layers)[0]


def cx_count(layers: int) -> int:
    """CNOTs in encoder (and again in decoder)."""
    return layers * (N_QUBITS - 1)

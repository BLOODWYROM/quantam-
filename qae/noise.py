"""Depolarising gate-noise study with Qiskit Aer (density-matrix simulation)."""
from __future__ import annotations
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .circuits import qae_circuit


def make_noise_model(p1: float, p2: float) -> NoiseModel:
    nm = NoiseModel(basis_gates=["ry", "cx", "reset", "id"])
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry"])
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def noisy_fidelity(psi_batch: np.ndarray, theta: np.ndarray, layers: int, n_latent: int,
                   p1: float, p2: float) -> np.ndarray:
    """Reset-decoder fidelity <psi|rho_rec|psi> under depolarising noise on every RY (p1) and CX (p2)."""
    sim = AerSimulator(method="density_matrix", noise_model=make_noise_model(p1, p2),
                       max_parallel_threads=1, max_parallel_experiments=1)
    # circuits only use ry / cx / reset (+Aer set/save instructions): already native, no transpilation needed
    circs = [qae_circuit(p, theta, layers, n_latent, save_dm=True, ideal_prep=True) for p in psi_batch]
    res = sim.run(circs).result()
    out = np.empty(len(psi_batch))
    for i, p in enumerate(psi_batch):
        rho = np.asarray(res.data(i)["rho"])
        out[i] = float(np.real(p @ rho @ p))
    return out

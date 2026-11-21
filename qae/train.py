"""Training of the QAE.

Cost (Romero et al. 2017):   C(theta) = 1 - < P(trash = 0..0) >_data
Since P(trash=0) = psi^T U^T Pi U psi with Pi = projector onto the first K basis states,
    C(theta) = 1 - Tr[ Pi U Cmat U^T ] ,   Cmat = (1/N) sum_j psi_j psi_j^T
which is *exactly* the data-averaged cost (no approximation), evaluated in O(K*16*16).

Gradient: parameter-shift rule  dC/dtheta_p = [ C(theta_p + pi/2) - C(theta_p - pi/2) ] / 2
(exact for RY gates, each parameter appears in exactly one gate).
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from . import DIM
from .ansatz import build_unitaries, n_params
from .metrics import reconstruct, fidelity_reset


def second_moment(psi: np.ndarray) -> np.ndarray:
    return psi.T @ psi / len(psi)


def trash_cost_batch(thetas: np.ndarray, Cmat: np.ndarray, layers: int, n_latent: int) -> np.ndarray:
    K = 2 ** n_latent
    W = build_unitaries(thetas, layers)[:, :K, :]                  # [B,K,16]
    return 1.0 - np.einsum("bki,ij,bkj->b", W, Cmat, W)


def trash_cost(theta, Cmat, layers, n_latent) -> float:
    return float(trash_cost_batch(np.asarray(theta)[None], Cmat, layers, n_latent)[0])


def param_shift_grad(theta, Cmat, layers, n_latent) -> np.ndarray:
    P = len(theta)
    shifts = (np.pi / 2) * np.eye(P)
    plus = trash_cost_batch(theta[None] + shifts, Cmat, layers, n_latent)
    minus = trash_cost_batch(theta[None] - shifts, Cmat, layers, n_latent)
    return 0.5 * (plus - minus)


def fidelity_cost(theta, psi, layers, n_latent) -> float:
    """End-to-end alternative: 1 - mean reset-decoder fidelity (needs the decoder)."""
    U = build_unitaries(np.asarray(theta)[None], layers)[0]
    return float(1.0 - fidelity_reset(reconstruct(U, psi, n_latent), psi).mean())


def ideal_cost_bound(Cmat: np.ndarray, n_latent: int) -> float:
    """Ky Fan bound: min over ALL unitaries of the trash cost = 1 - sum of the K largest eigenvalues of Cmat."""
    K = 2 ** n_latent
    ev = np.sort(np.linalg.eigvalsh(Cmat))[::-1]
    return float(1.0 - ev[:K].sum())


def train_qae(psi_train: np.ndarray, layers: int, n_latent: int, n_restarts: int = 8, seed: int = 0,
              maxiter: int = 300, cost: str = "trash") -> dict:
    """Multi-start L-BFGS-B. Returns best parameters, history of the best run and timing."""
    rng = np.random.default_rng(seed)
    Cmat = second_moment(psi_train)
    P = n_params(layers)
    best = None
    t0 = time.perf_counter()
    for r in range(n_restarts):
        th0 = rng.uniform(0, 2 * np.pi, P)
        hist = []
        if cost == "trash":
            f = lambda th: trash_cost(th, Cmat, layers, n_latent)
            g = lambda th: param_shift_grad(th, Cmat, layers, n_latent)
        elif cost == "fidelity":
            f = lambda th: fidelity_cost(th, psi_train, layers, n_latent)
            g = None  # finite differences
        else:
            raise ValueError(cost)
        res = minimize(f, th0, jac=g, method="L-BFGS-B", callback=lambda xk: hist.append(f(xk)),
                       options=dict(maxiter=maxiter))
        hist = [f(th0)] + hist
        if best is None or res.fun < best["final_cost"]:
            best = dict(theta=res.x, final_cost=float(res.fun), history=np.array(hist), restart=r)
    best["train_time_s"] = time.perf_counter() - t0
    best.update(layers=layers, n_latent=n_latent, cost_name=cost,
                bound=ideal_cost_bound(Cmat, n_latent), n_restarts=n_restarts)
    return best

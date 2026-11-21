"""Encoding -> trash reset -> decoding, and quality metrics.

Notation: n=4 qubits.  n_latent = k qubits are kept ("latent" = qubits 0..k-1),
the remaining n-k qubits are "trash" (qubits k..n-1).  K = 2^k, T = 2^(n-k).
For phi = U psi we reshape phi[t*K + l] -> M[t,l] (t = trash index, l = latent index).

Two decoders are evaluated
--------------------------
reset        (trace-preserving, Romero et al. 2017): trash is discarded and replaced by |0..0>.
                 rho_lat = M^T M ,  rho_rec = U^T ( |0><0|_trash (x) rho_lat ) U ,  F = <psi|rho_rec|psi>.
post-select  keep only runs where the trash is measured as |0..0> (success prob p0 = ||M[0]||^2);
                 |psi_hat> = U^T (|0> (x) M[0]/||M[0]||) ,  F = p0 .
"""
from __future__ import annotations
import numpy as np
from . import N_QUBITS, DIM


def _split(U: np.ndarray, n_latent: int):
    K = 2 ** n_latent
    return K, DIM // K


def reconstruct(U: np.ndarray, psi: np.ndarray, n_latent: int) -> dict:
    """Run the QAE on a batch of input states psi [N,16] (exact simulation)."""
    K, T = _split(U, n_latent)
    phi = psi @ U.T                                  # [N,16]  = U psi
    M = phi.reshape(-1, T, K)                        # [N,T,K]
    m0 = M[:, 0, :]                                  # trash = |0..0> block
    p0 = np.sum(m0 ** 2, axis=1)                     # P(trash = 0)
    rho_lat = np.einsum("ntk,ntl->nkl", M, M)        # reduced latent state (mixed in general)
    W = U[:K, :]                                     # rows of U that land in trash=0 subspace
    # post-selected pure reconstruction
    safe = np.where(p0 > 1e-15, np.sqrt(p0), 1.0)[:, None]
    psi_ps = (m0 / safe) @ W                         # [N,16]
    return dict(U=U, K=K, T=T, m0=m0, p0=p0, rho_lat=rho_lat, W=W, psi_ps=psi_ps)


def reset_diag(rec: dict) -> np.ndarray:
    """Diagonal of rho_rec (measurement probabilities of the reset-decoder output)."""
    W = rec["W"]
    return np.einsum("li,nlm,mi->ni", W, rec["rho_lat"], W)


def reset_density_matrices(rec: dict) -> np.ndarray:
    W = rec["W"]
    return np.einsum("li,nlm,mj->nij", W, rec["rho_lat"], W)


def fidelity_reset(rec: dict, target: np.ndarray) -> np.ndarray:
    """F = <t|rho_rec|t> for the reset decoder, per sample."""
    mt = (target @ rec["U"].T)[:, : rec["K"]]        # (U t) restricted to trash=0 block
    return np.einsum("nk,nkl,nl->n", mt, rec["rho_lat"], mt)


def fidelity_ps(rec: dict, target: np.ndarray) -> np.ndarray:
    return np.sum(target * rec["psi_ps"], axis=1) ** 2


def readout_reset(rec: dict) -> np.ndarray:
    """Amplitude read-out sqrt(p_i) of the reset-decoder output."""
    return np.sqrt(np.clip(reset_diag(rec), 0, None))


def readout_ps(rec: dict) -> np.ndarray:
    return np.abs(rec["psi_ps"])


def readout_mse(target: np.ndarray, readout: np.ndarray) -> np.ndarray:
    """Per-sample mean squared error between target amplitudes and measured amplitudes."""
    return np.mean((target - readout) ** 2, axis=1)


def trash_cost_samples(U: np.ndarray, psi: np.ndarray, n_latent: int) -> np.ndarray:
    return 1.0 - reconstruct(U, psi, n_latent)["p0"]

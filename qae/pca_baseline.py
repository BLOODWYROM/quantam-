"""Classical PCA baseline operating on the *same* L2-normalised 16-vectors.

Two variants:
  centered=True   standard PCA (mean subtracted; the mean vector is extra stored side-information)
  centered=False  uncentred PCA = truncated SVD. This is the exactly matched classical
                  counterpart of the QAE (no mean is available to the quantum circuit).
"""
from __future__ import annotations
import numpy as np


class PCABaseline:
    def __init__(self, k: int, centered: bool = True):
        self.k, self.centered = k, centered

    def fit(self, X: np.ndarray) -> "PCABaseline":
        self.mean_ = X.mean(axis=0) if self.centered else np.zeros(X.shape[1])
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.k]
        self.explained_ = (S ** 2) / len(X)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean_) @ self.components_.T
        return Z @ self.components_ + self.mean_

    # --- quantum-style metrics -------------------------------------------------
    def fidelity(self, X: np.ndarray, target: np.ndarray | None = None, renorm: bool = True):
        """|<t|x_hat>|^2 with x_hat re-normalised to a unit vector (valid quantum state)."""
        t = X if target is None else target
        xh = self.reconstruct(X)
        n2 = np.sum(xh ** 2, axis=1)
        ov2 = np.sum(t * xh, axis=1) ** 2
        return ov2 / np.where(n2 > 1e-15, n2, 1.0) if renorm else ov2

    def readout(self, X: np.ndarray) -> np.ndarray:
        """Measured amplitudes |x_hat| of the renormalised reconstruction."""
        xh = self.reconstruct(X)
        n = np.linalg.norm(xh, axis=1, keepdims=True)
        return np.abs(xh / np.where(n > 1e-15, n, 1.0))

    def n_stored_numbers(self, d: int = 16) -> int:
        return self.k * d + (d if self.centered else 0)

"""Data pipeline: MNIST -> 4x4 images -> L2-normalised 16-vectors (amplitude-encodable).

Source of MNIST samples
-----------------------
* Original dataset : LeCun, Cortes, Burges - http://yann.lecun.com/exdb/mnist/
* OpenML mirror    : https://www.openml.org/d/554
* Convenience copy : the `mlxtend` package ships the first 5000 training images
  (500 per digit), which is what this project uses so that it runs offline.
A cached copy is written to data/mnist_5000_subset.npz on first use.
"""
from __future__ import annotations
import os
import numpy as np
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "..", "data", "mnist_5000_subset.npz")


def load_mnist_subset() -> tuple[np.ndarray, np.ndarray]:
    """Return (X uint8 [N,784], y int [N]) for the bundled 5000-image MNIST subset."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["X"], d["y"]
    try:
        from mlxtend.data import mnist_data
        X, y = mnist_data()
    except Exception:  # fallback: OpenML (needs internet)
        from sklearn.datasets import fetch_openml
        ds = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = ds.data[:5000], ds.target[:5000].astype(int)
    X = X.astype(np.uint8)
    y = y.astype(int)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y)
    return X, y


def downsample_4x4(X: np.ndarray) -> np.ndarray:
    """28x28 -> 4x4 by exact 7x7 block averaging. Output in [0,1], shape [N,4,4]."""
    imgs = X.reshape(-1, 28, 28).astype(float) / 255.0
    return imgs.reshape(-1, 4, 7, 4, 7).mean(axis=(2, 4))


def to_state(imgs: np.ndarray) -> np.ndarray:
    """Flatten 4x4 images to 16-vectors and L2-normalise: amplitude encoding |psi> = sum_i x_i|i>/||x||."""
    v = imgs.reshape(len(imgs), -1).astype(float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("all-zero image cannot be amplitude-encoded")
    return v / n


def make_dataset(digits=(0, 1), test_size=0.3, seed=0) -> dict:
    """Stratified train/test split of the chosen digit classes."""
    X, y = load_mnist_subset()
    m = np.isin(y, digits)
    imgs = downsample_4x4(X[m])
    lab = y[m]
    idx = np.arange(len(lab))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=lab)
    return dict(
        digits=tuple(digits), seed=seed,
        img_train=imgs[tr], img_test=imgs[te],
        psi_train=to_state(imgs[tr]), psi_test=to_state(imgs[te]),
        y_train=lab[tr], y_test=lab[te],
        raw_test=X[m][te],
    )

"""Validation suite.  Run:  pytest -q"""
import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit import transpile
import qiskit_aer.library  # noqa

from qae.data import make_dataset, downsample_4x4, to_state, load_mnist_subset
from qae.ansatz import build_unitary, n_params
from qae.circuits import ansatz_circuit, qae_circuit, swap_test_p0
from qae.metrics import reconstruct, fidelity_reset, fidelity_ps, reset_density_matrices
from qae.train import (second_moment, trash_cost, param_shift_grad, train_qae, ideal_cost_bound)
from qae.pca_baseline import PCABaseline
from qae.noise import noisy_fidelity

RNG = np.random.default_rng(123)


@pytest.fixture(scope="module")
def ds():
    return make_dataset((0, 1), seed=0)


# ---------- data ------------------------------------------------------------
def test_dataset_shapes_and_norm(ds):
    assert ds["psi_train"].shape[1] == 16
    assert np.allclose(np.linalg.norm(ds["psi_train"], axis=1), 1.0)
    assert np.all(ds["psi_train"] >= 0)
    assert set(np.unique(ds["y_train"])) == {0, 1}


def test_downsample_is_block_mean():
    X = np.zeros((1, 784), dtype=np.uint8)
    X[0, :] = 255
    assert np.allclose(downsample_4x4(X), 1.0)
    img = np.arange(784, dtype=float).reshape(28, 28) % 256
    X2 = img.reshape(1, -1).astype(np.uint8)
    out = downsample_4x4(X2)[0]
    assert np.isclose(out[0, 0], X2.reshape(28, 28)[:7, :7].mean() / 255)


def test_zero_image_rejected():
    with pytest.raises(ValueError):
        to_state(np.zeros((1, 4, 4)))


# ---------- simulator vs Qiskit ----------------------------------------------
@pytest.mark.parametrize("layers", [1, 2, 3, 4])
def test_unitary_matches_qiskit_operator(layers):
    th = RNG.uniform(0, 2 * np.pi, n_params(layers))
    U_np = build_unitary(th, layers)
    U_qk = Operator(ansatz_circuit(th, layers)).data
    assert np.allclose(U_np, U_qk, atol=1e-12)
    assert np.allclose(U_np @ U_np.T, np.eye(16), atol=1e-12)


@pytest.mark.parametrize("layers,k", [(2, 1), (3, 2), (2, 3)])
def test_reset_decoder_density_matrix_matches_aer(ds, layers, k):
    th = RNG.uniform(0, 2 * np.pi, n_params(layers))
    psi = ds["psi_test"][:5]
    rec = reconstruct(build_unitary(th, layers), psi, k)
    rho_np = reset_density_matrices(rec)
    sim = AerSimulator(method="density_matrix")
    for i, p in enumerate(psi):
        qc = qae_circuit(p, th, layers, k, save_dm=True)
        qc = transpile(qc, sim, optimization_level=0)
        rho_aer = np.asarray(sim.run(qc).result().data(0)["rho"])
        assert np.allclose(rho_aer, rho_np[i], atol=1e-8), f"mismatch sample {i}"
        # fidelity too
        F_aer = float(np.real(p @ rho_aer @ p))
        assert np.isclose(F_aer, fidelity_reset(rec, psi)[i], atol=1e-8)


def test_postselected_state_matches_qiskit_projection(ds):
    layers, k = 3, 2
    th = RNG.uniform(0, 2 * np.pi, n_params(layers))
    U = ansatz_circuit(th, layers)
    psi = ds["psi_test"][0]
    phi = Statevector(psi).evolve(U).data                  # Qiskit-evolved state
    K = 2 ** k
    m0 = phi[:K]
    p0 = np.sum(np.abs(m0) ** 2)
    rec = reconstruct(build_unitary(th, layers), psi[None], k)
    assert np.isclose(rec["p0"][0], p0)
    assert np.isclose(fidelity_ps(rec, psi[None])[0], p0)   # F_ps == P(trash = 0)


@pytest.mark.parametrize("layers,k", [(1, 1), (3, 2), (2, 3)])
def test_swap_test_gives_trash_zero_probability(ds, layers, k):
    th = RNG.uniform(0, 2 * np.pi, n_params(layers))
    psi = ds["psi_test"][3]
    rec = reconstruct(build_unitary(th, layers), psi[None], k)
    assert np.isclose(swap_test_p0(psi, th, layers, k), rec["p0"][0], atol=1e-9)


# ---------- training ----------------------------------------------------------
def test_parameter_shift_equals_finite_difference(ds):
    C = second_moment(ds["psi_train"])
    th = RNG.uniform(0, 2 * np.pi, n_params(3))
    g = param_shift_grad(th, C, 3, 2)
    eps = 1e-6
    gf = np.array([(trash_cost(th + eps * e, C, 3, 2) - trash_cost(th - eps * e, C, 3, 2)) / (2 * eps)
                   for e in np.eye(len(th))])
    assert np.allclose(g, gf, atol=1e-7)


def test_cost_equals_sample_average(ds):
    th = RNG.uniform(0, 2 * np.pi, n_params(2))
    C = second_moment(ds["psi_train"])
    rec = reconstruct(build_unitary(th, 2), ds["psi_train"], 2)
    assert np.isclose(trash_cost(th, C, 2, 2), 1 - rec["p0"].mean())


def test_training_reduces_cost_and_respects_kyfan_bound(ds):
    r = train_qae(ds["psi_train"], layers=3, n_latent=2, n_restarts=4, seed=0)
    C = second_moment(ds["psi_train"])
    init = trash_cost(RNG.uniform(0, 2 * np.pi, n_params(3)), C, 3, 2)
    assert r["final_cost"] < init
    assert r["final_cost"] >= r["bound"] - 1e-9           # nothing can beat the Ky Fan bound


def test_qae_postselected_never_beats_uncentred_pca_on_train(ds):
    """mean F_ps = 1 - cost <= sum of top-K eigenvalues = mean fidelity of K-component uncentred PCA."""
    for k in (1, 2, 3):
        r = train_qae(ds["psi_train"], layers=3, n_latent=k, n_restarts=3, seed=1)
        U = build_unitary(r["theta"], 3)
        F_q = fidelity_ps(reconstruct(U, ds["psi_train"], k), ds["psi_train"]).mean()
        F_p = PCABaseline(2 ** k, centered=False).fit(ds["psi_train"]).fidelity(ds["psi_train"]).mean()
        assert F_q <= F_p + 1e-9


def test_full_latent_space_is_lossless(ds):
    th = RNG.uniform(0, 2 * np.pi, n_params(2))
    rec = reconstruct(build_unitary(th, 2), ds["psi_test"], 4)     # no trash qubits
    assert np.allclose(fidelity_reset(rec, ds["psi_test"]), 1.0)
    assert np.allclose(fidelity_ps(rec, ds["psi_test"]), 1.0)


# ---------- PCA ---------------------------------------------------------------
def test_pca_matches_sklearn(ds):
    from sklearn.decomposition import PCA
    X = ds["psi_train"]
    ours = PCABaseline(4, centered=True).fit(X).reconstruct(X)
    sk = PCA(4).fit(X)
    assert np.allclose(ours, sk.inverse_transform(sk.transform(X)), atol=1e-10)


def test_pca_full_rank_is_lossless(ds):
    X = ds["psi_test"]
    p = PCABaseline(16, centered=False).fit(ds["psi_train"])
    assert np.allclose(p.fidelity(X), 1.0)


# ---------- noise ---------------------------------------------------------------
def test_zero_noise_aer_equals_ideal(ds):
    layers, k = 2, 2
    th = RNG.uniform(0, 2 * np.pi, n_params(layers))
    psi = ds["psi_test"][:4]
    ideal = fidelity_reset(reconstruct(build_unitary(th, layers), psi, k), psi)
    assert np.allclose(noisy_fidelity(psi, th, layers, k, 0.0, 0.0), ideal, atol=1e-8)


def test_noise_lowers_fidelity_of_trained_qae(ds):
    r = train_qae(ds["psi_train"], 2, 2, n_restarts=3, seed=0)
    psi = ds["psi_test"][:20]
    f0 = noisy_fidelity(psi, r["theta"], 2, 2, 0.0, 0.0).mean()
    f1 = noisy_fidelity(psi, r["theta"], 2, 2, 0.01, 0.05).mean()
    assert f1 < f0


@pytest.mark.parametrize("layers,k", [(1, 1), (3, 2), (4, 3)])
def test_reset_fidelity_bounds_p0_squared_le_F_le_p0(ds, layers, k):
    """Proposition (report Sec. 6.1):  p0^2 <= F_reset <= F_ps = p0  for every sample."""
    th = RNG.uniform(0, 2 * np.pi, n_params(layers))
    rec = reconstruct(build_unitary(th, layers), ds["psi_test"], k)
    Fr, p0 = fidelity_reset(rec, ds["psi_test"]), rec["p0"]
    assert np.all(Fr <= p0 + 1e-10) and np.all(Fr >= p0 ** 2 - 1e-10)

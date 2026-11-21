"""All numerical experiments. Every function returns plain Python/pandas objects; nothing is plotted here."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import DIM
from .data import make_dataset
from .ansatz import build_unitary, n_params, cx_count
from .metrics import (reconstruct, fidelity_reset, fidelity_ps, readout_reset, readout_ps, readout_mse)
from .train import train_qae, second_moment
from .pca_baseline import PCABaseline
from .noise import noisy_fidelity
from .circuits import resource_counts, qae_circuit

PAIRS = [(0, 1), (3, 8)]
LATENTS = [1, 2, 3]
LAYERS = [1, 2, 3, 4, 6, 8]
DEEP_LAYERS = [12]
CLF_C = 100.0   # weak regularisation: inputs are unit-norm 16-vectors


def pair_name(p):
    return f"{p[0]} vs {p[1]}"


def _clf_acc(ro_train, ro_test, y_train, y_test):
    """Downstream utility: logistic regression trained AND tested on read-outs of the same pipeline."""
    return LogisticRegression(C=CLF_C, max_iter=5000).fit(ro_train, y_train).score(ro_test, y_test)


def evaluate_qae(U, psi_train, psi_test, y_train, y_test, k):
    rec = reconstruct(U, psi_test, k)
    rtr = reconstruct(U, psi_train, k)
    ro_r, ro_p = readout_reset(rec), readout_ps(rec)
    return dict(F_reset=fidelity_reset(rec, psi_test).mean(), F_ps=fidelity_ps(rec, psi_test).mean(),
                mse_reset=readout_mse(psi_test, ro_r).mean(), mse_ps=readout_mse(psi_test, ro_p).mean(),
                acc_reset=_clf_acc(readout_reset(rtr), ro_r, y_train, y_test),
                acc_ps=_clf_acc(readout_ps(rtr), ro_p, y_train, y_test))


def run_sweep(pairs=PAIRS, seeds=(0, 1, 2, 3, 4), layers=LAYERS, latents=LATENTS, restarts=6, verbose=True,
              with_pca=True, maxiter=400):
    """Experiment 1: QAE (depth x latent size) vs PCA, several splits. Returns (qae_df, pca_df, models)."""
    q_rows, p_rows, models = [], [], {}
    for pair in pairs:
        for seed in seeds:
            ds = make_dataset(pair, seed=seed)
            Xtr, Xte, ytr, yte = ds["psi_train"], ds["psi_test"], ds["y_train"], ds["y_test"]
            acc_orig = _clf_acc(Xtr, Xte, ytr, yte)
            for k in latents:
                K = 2 ** k
                for cen in ((False, True) if with_pca else ()):
                    t0 = time.perf_counter(); pca = PCABaseline(K, centered=cen).fit(Xtr); tfit = time.perf_counter() - t0
                    ro, ro_tr = pca.readout(Xte), pca.readout(Xtr)
                    p_rows.append(dict(pair=pair_name(pair), seed=seed, k=k, K=K, centered=cen,
                                       F=pca.fidelity(Xte).mean(), F_raw=pca.fidelity(Xte, renorm=False).mean(),
                                       mse=readout_mse(Xte, ro).mean(), acc=_clf_acc(ro_tr, ro, ytr, yte),
                                       acc_orig=acc_orig, fit_time_s=tfit, n_numbers=pca.n_stored_numbers()))
                for L in layers:
                    r = train_qae(Xtr, L, k, n_restarts=restarts, seed=1000 * seed + 10 * L + k, maxiter=maxiter)
                    U = build_unitary(r["theta"], L)
                    row = evaluate_qae(U, Xtr, Xte, ytr, yte, k)
                    row.update(pair=pair_name(pair), seed=seed, k=k, layers=L, n_params=n_params(L),
                               cx_per_unitary=cx_count(L), train_cost=r["final_cost"], bound=r["bound"],
                               train_time_s=r["train_time_s"], acc_orig=acc_orig)
                    q_rows.append(row)
                    models[(pair_name(pair), seed, k, L)] = dict(theta=r["theta"], history=r["history"])
            if verbose:
                print(f"  sweep done: pair {pair} seed {seed}", flush=True)
    return pd.DataFrame(q_rows), pd.DataFrame(p_rows), models


def pca_curve(pairs=PAIRS, seeds=(0, 1, 2, 3, 4)):
    """Experiment 2: fidelity of PCA as a function of the number of components (1..16)."""
    rows = []
    for pair in pairs:
        for seed in seeds:
            ds = make_dataset(pair, seed=seed)
            for cen in (False, True):
                for k in range(1, 17):
                    pca = PCABaseline(k, centered=cen).fit(ds["psi_train"])
                    rows.append(dict(pair=pair_name(pair), seed=seed, comps=k, centered=cen,
                                     F=pca.fidelity(ds["psi_test"]).mean()))
    return pd.DataFrame(rows)


def cost_ablation(pairs=PAIRS, seeds=(0, 1, 2), layers=3, restarts=4):
    """Experiment 3: train on Romero's trash cost vs directly on end-to-end reconstruction fidelity."""
    rows = []
    for pair in pairs:
        for seed in seeds:
            ds = make_dataset(pair, seed=seed)
            for k in LATENTS:
                for cname in ("trash", "fidelity"):
                    r = train_qae(ds["psi_train"], layers, k, n_restarts=restarts, seed=seed * 7 + k, cost=cname)
                    U = build_unitary(r["theta"], layers)
                    rec = reconstruct(U, ds["psi_test"], k)
                    rows.append(dict(pair=pair_name(pair), seed=seed, k=k, cost=cname,
                                     F_reset=fidelity_reset(rec, ds["psi_test"]).mean(),
                                     F_ps=fidelity_ps(rec, ds["psi_test"]).mean()))
    return pd.DataFrame(rows)


def denoising(models, pair=(0, 1), seed=0, k=2, L=3, sigmas=(0, .05, .1, .2, .3, .4, .5), draws=5):
    """Experiment 4: noisy 4x4 images -> compress/reconstruct -> fidelity to the CLEAN state."""
    ds = make_dataset(pair, seed=seed)
    clean_img, clean = ds["img_test"], ds["psi_test"]
    U = build_unitary(models[(pair_name(pair), seed, k, L)]["theta"], L)
    pu = PCABaseline(2 ** k, False).fit(ds["psi_train"]); pc = PCABaseline(2 ** k, True).fit(ds["psi_train"])
    rows = []
    for s in sigmas:
        for d in range(draws):
            rng = np.random.default_rng(100 * d + int(1000 * s))
            noisy_img = np.clip(clean_img + rng.normal(0, s, clean_img.shape), 0, 1)
            flat = noisy_img.reshape(len(noisy_img), -1)
            flat[np.linalg.norm(flat, axis=1) < 1e-9] = 1e-3   # guard against all-zero image
            noisy = flat / np.linalg.norm(flat, axis=1, keepdims=True)
            rec = reconstruct(U, noisy, k)
            rows.append(dict(sigma=s, draw=d,
                             noisy_input=(np.sum(noisy * clean, 1) ** 2).mean(),
                             QAE_reset=fidelity_reset(rec, clean).mean(), QAE_postselect=fidelity_ps(rec, clean).mean(),
                             PCA_uncentred=pu.fidelity(noisy, target=clean).mean(),
                             PCA_centred=pc.fidelity(noisy, target=clean).mean()))
    return pd.DataFrame(rows)


def gate_noise(models, pair=(0, 1), seed=0, k=2, layers=(1, 2, 3), scales=(0, .25, .5, 1, 2, 4), n_test=60):
    """Experiment 5: depolarising noise on every RY (p1 = p2/10) and CX (p2) -- Aer density-matrix simulation."""
    ds = make_dataset(pair, seed=seed)
    psi = ds["psi_test"][:n_test]
    pu = PCABaseline(2 ** k, False).fit(ds["psi_train"])
    rows = []
    for L in layers:
        th = models[(pair_name(pair), seed, k, L)]["theta"]
        for s in scales:
            p2 = 0.01 * s
            f = noisy_fidelity(psi, th, L, k, p2 / 10, p2)
            rows.append(dict(layers=L, p2=p2, p1=p2 / 10, F=f.mean(), F_std=f.std()))
    return pd.DataFrame(rows), float(pu.fidelity(psi).mean())


def timing(models, pair=(0, 1), seed=0, k=2, L=3, reps=20):
    """Experiment 7: wall-clock comparison (classical machine!)."""
    import qiskit_aer.library  # noqa
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    ds = make_dataset(pair, seed=seed)
    Xtr, Xte = ds["psi_train"], ds["psi_test"]
    th = models[(pair_name(pair), seed, k, L)]["theta"]
    U = build_unitary(th, L)
    t = {}
    t0 = time.perf_counter(); [PCABaseline(2 ** k, False).fit(Xtr) for _ in range(reps)]; t["pca_fit_s"] = (time.perf_counter() - t0) / reps
    pca = PCABaseline(2 ** k, False).fit(Xtr)
    t0 = time.perf_counter(); [pca.reconstruct(Xte) for _ in range(reps)]; t["pca_infer_per_sample_s"] = (time.perf_counter() - t0) / reps / len(Xte)
    t0 = time.perf_counter(); [reconstruct(U, Xte, k) for _ in range(reps)]; t["qae_numpy_infer_per_sample_s"] = (time.perf_counter() - t0) / reps / len(Xte)
    sim = AerSimulator(method="density_matrix")
    circs = [transpile(qae_circuit(p, th, L, k, save_dm=True), sim, optimization_level=0) for p in Xte[:50]]
    t0 = time.perf_counter(); sim.run(circs).result(); t["qae_aer_infer_per_sample_s"] = (time.perf_counter() - t0) / 50
    t["qae_train_s_(8 restarts)"] = train_qae(Xtr, L, k, n_restarts=8, seed=0)["train_time_s"]
    return t


def resources(layers=LAYERS):
    return pd.DataFrame([resource_counts(L) for L in layers])

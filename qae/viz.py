"""All figures (PNG, 200 dpi) written to results/figures/."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .data import make_dataset, load_mnist_subset, downsample_4x4, to_state
from .ansatz import build_unitary
from .metrics import (reconstruct, fidelity_reset, fidelity_ps, readout_reset, readout_ps, reset_diag)
from .pca_baseline import PCABaseline
from .circuits import ansatz_circuit, swap_test_circuit

C = dict(reset="#1f77b4", ps="#ff7f0e", pca_u="#2ca02c", pca_c="#d62728", grey="#7f7f7f", noisy="#9467bd")
LBL = dict(reset="QAE (reset decoder)", ps="QAE (post-selected)", pca_u="PCA uncentred (matched)", pca_c="PCA centred (standard)")


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100, "savefig.dpi": 200,
                         "savefig.bbox": "tight", "axes.titlesize": 11, "legend.frameon": False})


def _save(fig, R, name):
    path = os.path.join(R, "figures", name)
    fig.savefig(path)
    plt.close(fig)


def pn(pair):
    return f"{pair[0]} vs {pair[1]}"


# ------------------------------------------------------------------ pipeline / architecture
def fig_dataset(R):
    X, y = load_mnist_subset()
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 2.6]))
    for r, digit in enumerate((0, 1)):
        i = np.where(y == digit)[0][3]
        img = X[i].reshape(28, 28) / 255.0
        small = downsample_4x4(X[i:i + 1])[0]
        psi = to_state(small[None])[0]
        axes[r, 0].imshow(img, cmap="gray"); axes[r, 0].set_title(f"MNIST digit {digit}  (28×28)"); axes[r, 0].axis("off")
        axes[r, 1].imshow(small, cmap="gray", vmin=0, vmax=small.max()); axes[r, 1].set_title("4×4 block-average"); axes[r, 1].axis("off")
        axes[r, 2].bar(range(16), psi ** 2, color=C["reset"])
        axes[r, 2].set_xticks(range(16)); axes[r, 2].set_xticklabels([format(j, "04b") for j in range(16)], rotation=90, fontsize=7)
        axes[r, 2].set_ylabel(r"$|\langle i|\psi\rangle|^2$"); axes[r, 2].set_title("amplitude-encoded state on 4 qubits")
    fig.tight_layout()
    return fig


def fig_architecture(R):
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#333", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333"))

    box(0.1, 1.5, 1.35, 0.9, "MNIST\n28×28", "#eef3fb")
    box(1.85, 1.5, 1.35, 0.9, "4×4 average\n+ L2-normalise\n(16 numbers)", "#eef3fb")
    box(3.6, 1.5, 1.35, 0.9, "Amplitude\nencoding\n|ψ⟩ (4 qubits)", "#fdf1e3")
    box(5.35, 1.5, 1.2, 0.9, "Encoder\nU(θ)", "#fdf1e3")
    box(6.95, 2.35, 1.35, 0.8, "latent qubits\n(kept)", "#e6f4e6")
    box(6.95, 0.75, 1.35, 0.8, "trash qubits\n→ reset to |0…0⟩", "#fbe5e5")
    box(8.65, 1.5, 1.2, 0.9, "Decoder\nU(θ)†", "#fdf1e3")
    box(10.0, 1.5, 0.95, 0.9, "measure\n→ 4×4", "#eef3fb", fs=8)
    for a, b in [(1.45, 1.85), (3.2, 3.6), (4.95, 5.35)]:
        arrow(a, 1.95, b, 1.95)
    arrow(6.55, 2.2, 6.95, 2.75); arrow(6.55, 1.7, 6.95, 1.15)
    arrow(8.3, 2.75, 8.65, 2.2); arrow(8.3, 1.15, 8.65, 1.7); arrow(9.85, 1.95, 10.0, 1.95)
    ax.text(5.5, 3.45, "Quantum autoencoder (Romero et al., 2017): train U so that the trash qubits end in |0…0⟩",
            ha="center", fontsize=10, weight="bold")
    ax.text(5.5, 0.25, "Classical baseline: PCA on the same 16-vectors with K = 2^(latent qubits) components", ha="center",
            fontsize=9, color=C["pca_u"])
    return fig


def fig_circuits(D, R):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    th = D["models"][("0 vs 1", 0, 2, 2)]["theta"]
    qc = QuantumCircuit(4)
    qc.append(Gate("AmpEnc(ψ)", 4, []), range(4)); qc.barrier()
    U = ansatz_circuit(th, 2)
    qc.compose(U, inplace=True); qc.barrier()
    qc.reset(2); qc.reset(3); qc.barrier()
    qc.compose(U.inverse(), inplace=True)
    f1 = qc.draw("mpl", fold=-1, style={"backgroundcolor": "white"}, scale=0.7)
    _save(f1, R, "fig_circuit_qae.png")
    sw = swap_test_circuit(np.ones(16) / 4, th, 2, 2)
    f2 = sw.draw("mpl", fold=-1, style={"backgroundcolor": "white"}, scale=0.7)
    _save(f2, R, "fig_circuit_swaptest.png")


# ------------------------------------------------------------------ main comparison
def _agg(q, p):
    g = q.groupby(["pair", "k", "layers"]).agg(F_reset=("F_reset", "mean"), F_reset_sd=("F_reset", "std"),
                                               F_ps=("F_ps", "mean"), F_ps_sd=("F_ps", "std"),
                                               cost=("train_cost", "mean"), cost_sd=("train_cost", "std"),
                                               bound=("bound", "mean")).reset_index()
    gp = p.groupby(["pair", "k", "centered"]).agg(F=("F", "mean"), F_sd=("F", "std")).reset_index()
    return g, gp


def fig_fidelity_vs_layers(D, R):
    g, gp = _agg(D["q"], D["p"])
    for pair in g.pair.unique():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
        for ax, k in zip(axes, (1, 2, 3)):
            s = g[(g.pair == pair) & (g.k == k)].sort_values("layers")
            x = np.arange(len(s))
            ax.errorbar(x, s.F_reset, s.F_reset_sd.fillna(0), color=C["reset"], marker="o", capsize=3, label=LBL["reset"])
            ax.errorbar(x, s.F_ps, s.F_ps_sd.fillna(0), color=C["ps"], marker="s", capsize=3, label=LBL["ps"])
            for cen, key in ((False, "pca_u"), (True, "pca_c")):
                v = gp[(gp.pair == pair) & (gp.k == k) & (gp.centered == cen)].F.iloc[0]
                ax.axhline(v, color=C[key], ls="--", label=f"{LBL[key]}: {v:.3f}")
            ax.set_xticks(x); ax.set_xticklabels(s.layers.astype(int))
            ax.set_xlabel("ansatz layers L"); ax.set_title(f"{k} latent qubit{'s' if k > 1 else ''}  (PCA K = {2 ** k})")
        axes[0].set_ylabel("mean test fidelity")
        axes[0].legend(fontsize=7.5, loc="lower right")
        fig.suptitle(f"Digits {pair}: reconstruction fidelity vs circuit depth (mean ± std over splits)", y=1.02)
        fig.tight_layout(); _save(fig, R, f"fig_fidelity_vs_layers_{pair.replace(' ', '')}.png")


def fig_cost_vs_bound(D, R):
    g, _ = _agg(D["q"], D["p"])
    pairs = list(g.pair.unique())
    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 3.2 * len(pairs)), squeeze=False)
    for r, pair in enumerate(pairs):
        for c, k in enumerate((1, 2, 3)):
            s = g[(g.pair == pair) & (g.k == k)].sort_values("layers"); ax = axes[r, c]
            x = np.arange(len(s))
            ax.errorbar(x, s.cost, s.cost_sd.fillna(0), color=C["reset"], marker="o", capsize=3, label="trained QAE cost")
            ax.axhline(s.bound.mean(), color="k", ls=":", label="Ky Fan bound (= PCA ceiling)")
            ax.set_xticks(x); ax.set_xticklabels(s.layers.astype(int)); ax.set_yscale("log")
            ax.set_title(f"{pair}, {k} latent qubit{'s' if k > 1 else ''}")
            ax.set_xlabel("ansatz layers L")
            if c == 0: ax.set_ylabel("training cost  1 − ⟨P(trash=0)⟩")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); _save(fig, R, "fig_cost_vs_bound.png")


def fig_pca_curve(D, R):
    pc = D["pc"]; pairs = list(pc.pair.unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.6 * len(pairs), 3.6), squeeze=False)
    for ax, pair in zip(axes[0], pairs):
        for cen, key in ((False, "pca_u"), (True, "pca_c")):
            s = pc[(pc.pair == pair) & (pc.centered == cen)].groupby("comps").F.mean()
            ax.plot(s.index, s.values, color=C[key], marker="o", ms=3.5, label=LBL[key])
        for K, k in ((2, 1), (4, 2), (8, 3)):
            ax.axvline(K, color=C["grey"], ls=":", lw=1); ax.text(K + 0.1, 0.52, f"{k} qubit{'s' if k > 1 else ''}", rotation=90, fontsize=8, color=C["grey"])
        ax.set_xlabel("number of PCA components"); ax.set_ylabel("mean test fidelity"); ax.set_title(f"Digits {pair}"); ax.set_ylim(0.5, 1.01)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); _save(fig, R, "fig_pca_curve.png")


def fig_training_curves(D, R):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, col in zip((1, 2, 3), ("#1f77b4", "#2ca02c", "#d62728")):
        m = D["models"][("0 vs 1", 0, k, 4)]
        ax.plot(m["history"], color=col, label=f"{k} latent qubit{'s' if k > 1 else ''}")
        bound = D["q"].query("pair=='0 vs 1' and seed==0 and k==@k and layers==4").bound.iloc[0]
        ax.axhline(bound, color=col, ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("L-BFGS iteration"); ax.set_ylabel("training cost  1 − ⟨P(trash=0)⟩")
    ax.set_title("Training curves, L = 4, digits 0 vs 1 (dotted: Ky Fan bound)"); ax.legend()
    fig.tight_layout(); _save(fig, R, "fig_training_curves.png")


# ------------------------------------------------------------------ qualitative
def fig_reconstructions(D, R, pair=(0, 1), k=2, seed=0, n_each=4):
    Ls = sorted(D["q"][(D["q"].pair == pn(pair)) & (D["q"].seed == seed) & (D["q"].k == k)].layers.unique())
    L = 8 if 8 in Ls else Ls[-1]
    ds = make_dataset(pair, seed=seed)
    th = D["models"][(pn(pair), seed, k, L)]["theta"]; U = build_unitary(th, L)
    idx = np.concatenate([np.where(ds["y_test"] == d)[0][:n_each] for d in pair])
    psi = ds["psi_test"][idx]
    rec = reconstruct(U, psi, k)
    pu = PCABaseline(2 ** k, False).fit(ds["psi_train"]); pcn = PCABaseline(2 ** k, True).fit(ds["psi_train"])
    rows = [("Original", psi, None),
            (f"QAE reset\n(L={L})", readout_reset(rec), fidelity_reset(rec, psi)),
            (f"QAE post-sel.\n(L={L})", readout_ps(rec), fidelity_ps(rec, psi)),
            (f"PCA uncentred\n(K={2 ** k})", pu.readout(psi), pu.fidelity(psi)),
            (f"PCA centred\n(K={2 ** k})", pcn.readout(psi), pcn.fidelity(psi))]
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(1.25 * len(idx) + 1.5, 1.3 * len(rows)))
    for r, (name, arr, F) in enumerate(rows):
        for c in range(len(idx)):
            ax = axes[r, c]; ax.imshow(arr[c].reshape(4, 4), cmap="gray", vmin=0, vmax=psi[c].max()); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if F is not None: ax.set_xlabel(f"F={F[c]:.2f}", fontsize=7, labelpad=1)
            if c == 0: ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center", labelpad=34)
            if r == 0: ax.set_title(f"digit {ds['y_test'][idx[c]]}", fontsize=8)
    fig.suptitle(f"Test reconstructions, {k} latent qubits ({pn(pair)})", y=1.0, fontsize=10)
    fig.tight_layout(); _save(fig, R, f"fig_reconstructions_{pn(pair).replace(' ', '')}_k{k}.png")


def fig_bloch(D, R, pair=(0, 1), seed=0, L=6):
    k = 1
    ds = make_dataset(pair, seed=seed); U = build_unitary(D["models"][(pn(pair), seed, k, L)]["theta"], L)
    rec = reconstruct(U, ds["psi_test"], k); rho = rec["rho_lat"]                    # [N,2,2] real symmetric
    bx, bz = 2 * rho[:, 0, 1], rho[:, 0, 0] - rho[:, 1, 1]; by = np.zeros_like(bx)
    fig = plt.figure(figsize=(10, 4.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#cccccc", lw=0.3, alpha=0.6)
    for d, col in zip(pair, ("#1f77b4", "#d62728")):
        m = ds["y_test"] == d
        ax.scatter(bx[m], by[m], bz[m], s=8, color=col, alpha=0.6, label=f"digit {d}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_box_aspect((1, 1, 1)); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Latent qubit: Bloch vectors (test set)")
    ax2 = fig.add_subplot(1, 2, 2)
    t = np.linspace(0, 2 * np.pi, 200); ax2.plot(np.cos(t), np.sin(t), color="#999", lw=1)
    for d, col in zip(pair, ("#1f77b4", "#d62728")):
        m = ds["y_test"] == d
        ax2.scatter(bx[m], bz[m], s=10, color=col, alpha=0.5)
        ax2.scatter(bx[m].mean(), bz[m].mean(), s=120, marker="*", color=col, edgecolor="k", zorder=5, label=f"class mean, digit {d}")
    ax2.set_aspect("equal"); ax2.set_xlabel("Bloch x"); ax2.set_ylabel("Bloch z"); ax2.legend(fontsize=8)
    ax2.set_title("x–z plane (y = 0: all amplitudes real); inside = mixed")
    fig.tight_layout(); _save(fig, R, "fig_bloch_latent.png")


def fig_shots(D, R, pair=(0, 1), seed=0, k=2, L=8, shots=8192):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from .circuits import qae_circuit
    ds = make_dataset(pair, seed=seed); th = D["models"][(pn(pair), seed, k, L)]["theta"]
    psi = ds["psi_test"][0]
    rec = reconstruct(build_unitary(th, L), psi[None], k); exact = reset_diag(rec)[0]
    sim = AerSimulator()
    qc = transpile(qae_circuit(psi, th, L, k, measure=True), sim, optimization_level=1)
    counts = sim.run(qc, shots=shots, seed_simulator=7).result().get_counts()
    est = np.zeros(16)
    for key, v in counts.items(): est[int(key.replace(" ", ""), 2)] = v / shots
    fig, ax = plt.subplots(figsize=(9, 3.6)); w = 0.28; x = np.arange(16)
    ax.bar(x - w, psi ** 2, w, color=C["grey"], label="input |ψ⟩ (probabilities)")
    ax.bar(x, exact, w, color=C["reset"], label="QAE output, exact simulation")
    ax.bar(x + w, est, w, yerr=np.sqrt(est * (1 - est) / shots), color=C["ps"], capsize=1.5, label=f"QAE output, Aer {shots} shots")
    ax.set_xticks(x); ax.set_xticklabels([format(j, "04b") for j in x], rotation=90, fontsize=7)
    ax.set_ylabel("probability"); ax.set_title(f"Measurement statistics of the full QAE circuit (digits {pn(pair)}, test sample 0)"); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, R, "fig_measurement_shots.png")


# ------------------------------------------------------------------ extensions
def fig_denoising(D, R):
    dn = D["dn"].groupby("sigma").agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6.6, 4))
    for col, key, lab, ls in (("noisy_input", "noisy", "noisy input (no processing)", "-"), ("QAE_reset", "reset", LBL["reset"], "-"),
                              ("QAE_postselect", "ps", LBL["ps"], "-"), ("PCA_uncentred", "pca_u", LBL["pca_u"], "--"),
                              ("PCA_centred", "pca_c", LBL["pca_c"], "--")):
        ax.errorbar(dn.index, dn[(col, "mean")], dn[(col, "std")], color=C[key], ls=ls, marker="o", ms=4, capsize=2, label=lab)
    ax.set_xlabel("Gaussian pixel-noise σ (on [0,1] images)"); ax.set_ylabel("fidelity with the CLEAN state")
    ax.set_title("Denoising test: QAE (L=3) and PCA, 2 latent qubits / K=4"); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, R, "fig_denoising.png")


def fig_gate_noise(D, R):
    gn = D["gn"]; fig, ax = plt.subplots(figsize=(6.6, 4))
    for L, col in zip(sorted(gn.layers.unique()), ("#9ecae1", "#4292c6", "#08519c")):
        s = gn[gn.layers == L]; ax.errorbar(s.p2 * 100, s.F, marker="o", capsize=0, color=col, label=f"QAE, L={L} (reset decoder)")
    ax.axhline(D["pca_ref"], color=C["pca_u"], ls="--", label="PCA K=4 (noise-free classical)")
    ax.axhline(1 / 16, color=C["grey"], ls=":", label="maximally mixed output (1/16)")
    ax.set_xlabel("two-qubit (CX) depolarising error  p₂ [%]   (RY error p₁ = p₂/10)"); ax.set_ylabel("mean test fidelity")
    ax.set_title("Robustness to gate noise (Qiskit Aer, 2 latent qubits)"); ax.legend(fontsize=8)
    fig.tight_layout(); _save(fig, R, "fig_gate_noise.png")


def fig_cost_ablation(D, R):
    ab = D["ab"].groupby(["pair", "k", "cost"]).F_reset.agg(["mean", "std"]).reset_index(); pairs = list(ab.pair.unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 3.6), squeeze=False)
    for ax, pair in zip(axes[0], pairs):
        for j, (cname, col) in enumerate((("trash", C["reset"]), ("fidelity", C["ps"]))):
            s = ab[(ab.pair == pair) & (ab.cost == cname)].sort_values("k")
            ax.bar(s.k + (j - 0.5) * 0.36, s["mean"], 0.34, yerr=s["std"], color=col, capsize=3,
                   label={"trash": "trash-qubit cost (Romero)", "fidelity": "end-to-end fidelity cost"}[cname])
        ax.set_xticks([1, 2, 3]); ax.set_xlabel("latent qubits"); ax.set_ylabel("test fidelity (reset decoder)"); ax.set_title(f"Digits {pair}, L=3")
        ax.set_ylim(0.6, 1.0)
    axes[0, 0].legend(fontsize=8, loc="lower right"); fig.tight_layout(); _save(fig, R, "fig_cost_ablation.png")


def fig_classification(D, R, L=8):
    q, p = D["q"], D["p"]; pairs = list(q.pair.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, k in zip(axes, (1, 2, 3)):
        for pi, pair in enumerate(pairs):
            qq = q[(q.pair == pair) & (q.k == k) & (q.layers == L)]; pp = p[(p.pair == pair) & (p.k == k)]
            vals = [("original", pp.acc_orig.mean(), pp.acc_orig.std(), C["grey"]),
                    ("QAE reset", qq.acc_reset.mean(), qq.acc_reset.std(), C["reset"]),
                    ("QAE post-sel.", qq.acc_ps.mean(), qq.acc_ps.std(), C["ps"]),
                    ("PCA unc.", pp[~pp.centered].acc.mean(), pp[~pp.centered].acc.std(), C["pca_u"]),
                    ("PCA cent.", pp[pp.centered].acc.mean(), pp[pp.centered].acc.std(), C["pca_c"])]
            for j, (lab, m, s, col) in enumerate(vals):
                ax.bar(pi * 6 + j, m, 0.85, yerr=s, color=col, capsize=2, label=lab if (pi == 0 and k == 1) else None)
            ax.text(pi * 6 + 2, 0.52, f"digits {pair}", ha="center", fontsize=8)
        ax.set_xticks([]); ax.set_ylim(0.5, 1.02); ax.set_title(f"{k} latent qubit{'s' if k > 1 else ''} (QAE L={L})")
    axes[0].set_ylabel("test accuracy (logistic regression)"); axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout(); _save(fig, R, "fig_classification.png")


def make_all(D, R):
    _style()
    steps = [("dataset", lambda: _save(fig_dataset(R), R, "fig_dataset.png")),
             ("architecture", lambda: _save(fig_architecture(R), R, "fig_architecture.png")),
             ("circuits", lambda: fig_circuits(D, R)),
             ("fidelity_vs_layers", lambda: fig_fidelity_vs_layers(D, R)),
             ("cost_vs_bound", lambda: fig_cost_vs_bound(D, R)),
             ("pca_curve", lambda: fig_pca_curve(D, R)),
             ("training_curves", lambda: fig_training_curves(D, R)),
             ("reconstructions", lambda: [fig_reconstructions(D, R, pair=pr, k=k) for pr, k in
                                          (((0, 1), 1), ((0, 1), 2), ((0, 1), 3), ((3, 8), 2))]),
             ("bloch", lambda: fig_bloch(D, R)),
             ("shots", lambda: fig_shots(D, R)),
             ("denoising", lambda: fig_denoising(D, R)),
             ("gate_noise", lambda: fig_gate_noise(D, R)),
             ("cost_ablation", lambda: fig_cost_ablation(D, R)),
             ("classification", lambda: fig_classification(D, R))]
    for name, fn in steps:
        fn(); print("  figure:", name, flush=True)

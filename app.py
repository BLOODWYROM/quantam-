"""Interactive dashboard:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from qae.data import make_dataset
from qae.ansatz import build_unitary, n_params, cx_count
import base64, json, os, subprocess, sys
from qae.metrics import (reconstruct, fidelity_reset, fidelity_ps, readout_reset, readout_ps, readout_mse)
from qae.pca_baseline import PCABaseline
from qae.train import train_qae

ROOT = os.path.dirname(os.path.abspath(__file__))


def worker(**req):
    """Run Qiskit-heavy work (drawing, transpilation, Aer) in an isolated subprocess."""
    p = subprocess.run([sys.executable, "-m", "qae.worker"], input=json.dumps(req), capture_output=True, text=True, cwd=ROOT, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-500:])
    return json.loads(p.stdout)


@st.cache_data(show_spinner=False)
def circuit_png(theta, layers):
    return base64.b64decode(worker(cmd="circuit", theta=list(theta), layers=layers))


@st.cache_data(show_spinner=False)
def resources(layers):
    return worker(cmd="resources", layers=layers)


st.set_page_config(page_title="Quantum Autoencoder vs PCA", page_icon="⚛️", layout="wide")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"

st.title("⚛️ Quantum Autoencoder vs PCA on 4×4 MNIST digits")
st.caption("Simulator-based study · Qiskit + NumPy · 16 pixels → 4 qubits (amplitude encoding) → compress to 1–3 latent qubits")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Experiment settings")
    c1, c2 = st.columns(2)
    dA = c1.selectbox("Digit A", list(range(10)), index=0)
    dB = c2.selectbox("Digit B", list(range(10)), index=1)
    k = st.radio("Latent qubits (kept)", [1, 2, 3], index=1, horizontal=True,
                 help="The remaining 4−k qubits are the 'trash' qubits that are reset to |0…0⟩.")
    L = st.slider("Ansatz layers L", 1, 12, 4, help="Each layer = RY on every qubit + CX chain.")
    restarts = st.slider("Random restarts", 1, 8, 4)
    cost = st.selectbox("Training cost", ["trash", "fidelity"], format_func=lambda c: {"trash": "Trash-qubit cost (Romero et al.)", "fidelity": "End-to-end fidelity (slower)"}[c])
    seed = st.number_input("Split / optimiser seed", 0, 99, 0)
    st.divider()
    st.markdown("**Theory in one line**  \nWith enough expressivity, the best possible QAE cost equals PCA's captured variance (Ky Fan). "
                "A shallow ansatz can only approach that ceiling.")

if dA == dB:
    st.error("Choose two different digits."); st.stop()
pair = (min(dA, dB), max(dA, dB))


@st.cache_data(show_spinner="Training the quantum autoencoder (exact simulation) …")
def fit(pair, k, L, restarts, seed, cost):
    ds = make_dataset(pair, seed=seed)
    r = train_qae(ds["psi_train"], L, k, n_restarts=restarts, seed=seed, cost=cost, maxiter=400)
    return r


ds = make_dataset(pair, seed=int(seed))
r = fit(pair, k, L, restarts, int(seed), cost)
U = build_unitary(r["theta"], L)
psi_te = ds["psi_test"]
rec = reconstruct(U, psi_te, k)
pu = PCABaseline(2 ** k, False).fit(ds["psi_train"]); pc = PCABaseline(2 ** k, True).fit(ds["psi_train"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "🧩 Circuit & states", "🔊 Noise lab", "📘 Theory"])

# ------------------------------------------------------------------ results
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("QAE fidelity (reset)", f"{fidelity_reset(rec, psi_te).mean():.3f}")
    m2.metric("QAE fidelity (post-selected)", f"{fidelity_ps(rec, psi_te).mean():.3f}")
    m3.metric("PCA uncentred (K=%d)" % 2 ** k, f"{pu.fidelity(psi_te).mean():.3f}")
    m4.metric("PCA centred (K=%d)" % 2 ** k, f"{pc.fidelity(psi_te).mean():.3f}")
    tbl = pd.DataFrame({
        "Method": ["QAE – reset decoder", "QAE – post-selected", "PCA – uncentred", "PCA – centred"],
        "Test fidelity": [fidelity_reset(rec, psi_te).mean(), fidelity_ps(rec, psi_te).mean(), pu.fidelity(psi_te).mean(), pc.fidelity(psi_te).mean()],
        "Pixel-readout MSE": [readout_mse(psi_te, readout_reset(rec)).mean(), readout_mse(psi_te, readout_ps(rec)).mean(),
                              readout_mse(psi_te, pu.readout(psi_te)).mean(), readout_mse(psi_te, pc.readout(psi_te)).mean()],
        "Stored parameters": [n_params(L), n_params(L), pu.n_stored_numbers(), pc.n_stored_numbers()],
    }).round(4)
    st.dataframe(tbl, hide_index=True, width="stretch")
    gap = r["final_cost"] - r["bound"]
    st.info(f"Training cost **{r['final_cost']:.4f}** vs. theoretical optimum (Ky Fan / PCA ceiling) **{r['bound']:.4f}** → gap {gap:.4f}. "
            f"Mean train fidelity of the post-selected QAE = 1 − cost = {1 - r['final_cost']:.4f}. "
            f"A gap of ~0 means this circuit has become as good as PCA – it cannot be better (on the training data).")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        ax.plot(r["history"], color=BLUE); ax.axhline(r["bound"], color="k", ls=":", label="Ky Fan bound")
        ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("cost"); ax.set_title("Training curve (best restart)"); ax.legend()
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        ax.hist(fidelity_ps(rec, psi_te), bins=25, alpha=.6, color=ORANGE, label="QAE post-sel.")
        ax.hist(pu.fidelity(psi_te), bins=25, alpha=.5, color=GREEN, label="PCA uncentred")
        ax.set_xlabel("per-sample test fidelity"); ax.set_title("Fidelity distribution"); ax.legend()
        st.pyplot(fig)
    st.subheader("Reconstructions")
    n_each = 5
    idx = np.concatenate([np.where(ds["y_test"] == d)[0][:n_each] for d in pair]); psi = psi_te[idx]
    rc = reconstruct(U, psi, k)
    rows = [("Original", psi), ("QAE reset", readout_reset(rc)), ("QAE post-sel.", readout_ps(rc)), ("PCA unc.", pu.readout(psi)), ("PCA cent.", pc.readout(psi))]
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(1.1 * len(idx), 1.15 * len(rows)))
    for i, (nm, arr) in enumerate(rows):
        for j in range(len(idx)):
            axes[i, j].imshow(arr[j].reshape(4, 4), cmap="gray", vmin=0, vmax=psi[j].max()); axes[i, j].set_xticks([]); axes[i, j].set_yticks([]); axes[i, j].grid(False)
            if j == 0: axes[i, j].set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
            if i == 0: axes[i, j].set_title(f"digit {ds['y_test'][idx[j]]}", fontsize=7)
    st.pyplot(fig)

# ------------------------------------------------------------------ circuit & states
with tab2:
    st.subheader("Encoder U(θ) – exactly what is simulated")
    st.image(circuit_png(tuple(float(x) for x in r["theta"]), L))
    res = resources(L)
    st.write(f"**Resources** – parameters: {res['n_parameters']}, CX in encoder: {res['encoder_cx']} (same again in the decoder), "
             f"encoder depth: {res['encoder_depth']}.  Amplitude-encoding state preparation for a generic 4-qubit state costs about "
             f"{res['stateprep_cx']} CX (depth {res['stateprep_depth']}) – more than the encoder itself.")
    st.subheader("Single-sample explorer")
    i = st.slider("Test sample", 0, len(psi_te) - 1, 0)
    p = psi_te[i:i + 1]; rs = reconstruct(U, p, k)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(16); w = .2
        ax.bar(x - 1.5 * w, p[0] ** 2, w, color="#7f7f7f", label="input")
        ax.bar(x - .5 * w, readout_reset(rs)[0] ** 2, w, color=BLUE, label="QAE reset")
        ax.bar(x + .5 * w, readout_ps(rs)[0] ** 2, w, color=ORANGE, label="QAE post-sel.")
        ax.bar(x + 1.5 * w, pu.readout(p)[0] ** 2, w, color=GREEN, label="PCA unc.")
        ax.set_xticks(x); ax.set_xticklabels([format(j, "04b") for j in x], rotation=90, fontsize=7); ax.set_ylabel("probability |amplitude|²"); ax.legend(fontsize=7)
        st.pyplot(fig)
    with c2:
        st.metric("P(trash = 0…0)", f"{rs['p0'][0]:.3f}", help="Post-selection success probability; equals the post-selected fidelity.")
        st.metric("Fidelity (reset)", f"{fidelity_reset(rs, p)[0]:.3f}")
        st.metric("Digit", int(ds["y_test"][i]))
        fig, ax = plt.subplots(figsize=(1.8, 1.8)); ax.imshow(ds["img_test"][i], cmap="gray"); ax.axis("off"); st.pyplot(fig)
    if k == 1:
        st.subheader("Latent qubit on the Bloch sphere (x–z plane)")
        rho = reconstruct(U, psi_te, 1)["rho_lat"]; bx, bz = 2 * rho[:, 0, 1], rho[:, 0, 0] - rho[:, 1, 1]
        fig, ax = plt.subplots(figsize=(3.6, 3.6)); t = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(t), np.sin(t), c="#999")
        for d, c in zip(pair, (BLUE, RED)):
            m = ds["y_test"] == d; ax.scatter(bx[m], bz[m], s=8, alpha=.5, color=c, label=f"digit {d}")
        ax.set_aspect("equal"); ax.legend(fontsize=7); st.pyplot(fig)

# ------------------------------------------------------------------ noise lab
with tab3:
    st.write("Depolarising noise on every RY (p₁) and CX (p₂) gate, simulated with **Qiskit Aer (density matrix)**. State preparation is kept ideal so only the autoencoder circuit is stressed.")
    c1, c2, c3 = st.columns(3)
    p2 = c1.slider("CX error p₂ (%)", 0.0, 5.0, 1.0, 0.1) / 100
    p1 = c2.slider("RY error p₁ (%)", 0.0, 1.0, 0.1, 0.05) / 100
    ns = c3.slider("Test samples", 5, 60, 20)
    if st.button("Run noisy simulation"):
        with st.spinner("Aer is simulating …"):
            f_noisy = np.array(worker(cmd="noise", psi=psi_te[:ns].tolist(), theta=r["theta"].tolist(), layers=L, k=k, p1=p1, p2=p2))
        f_ideal = fidelity_reset(reconstruct(U, psi_te[:ns], k), psi_te[:ns])
        a, b = st.columns(2); a.metric("Ideal fidelity", f"{f_ideal.mean():.3f}"); b.metric("Noisy fidelity", f"{f_noisy.mean():.3f}", f"{f_noisy.mean() - f_ideal.mean():+.3f}")
        st.caption("Deeper circuits score higher without noise but lose more of it with noise – try L=2 vs L=8 at p₂ = 3 %.")

# ------------------------------------------------------------------ theory
with tab4:
    st.markdown("""
**Encoding.** A 4×4 image is flattened to $x\\in\\mathbb R^{16}$ and normalised: $|\\psi\\rangle=\\sum_{i=0}^{15}\\frac{x_i}{\\|x\\|}|i\\rangle$ (4 qubits).

**Autoencoder.** A trainable real orthogonal circuit $U(\\theta)$ is applied. Qubits $k,\\dots,3$ are *trash*. The cost is
$C(\\theta)=1-\\mathbb E\\big[\\,P(\\text{trash}=0\\ldots0)\\,\\big]=1-\\mathrm{Tr}\\big[\\Pi\\,U\\,\\Sigma\\,U^{\\top}\\big]$, with $\\Sigma=\\mathbb E[\\psi\\psi^\\top]$ and $\\Pi$ the projector onto the first $K=2^k$ basis states.

**Decoding.** *Reset*: trash replaced by $|0\\ldots0\\rangle$, then $U^\\dagger$ (a valid quantum channel). *Post-selected*: keep only shots with trash = 0…0 → fidelity $=P(\\text{trash}=0)$.

**Ceiling.** By Ky Fan's theorem, $\\min_U C = 1-\\sum_{j\\le K}\\lambda_j(\\Sigma)$, which is exactly the variance captured by uncentred PCA with $K$ components. Hence a post-selected QAE can *match* PCA but not beat it on the training data.

**Gradients.** Parameter-shift: $\\partial_{\\theta_p}C=\\tfrac12\\,[C(\\theta_p+\\pi/2)-C(\\theta_p-\\pi/2)]$.
""")

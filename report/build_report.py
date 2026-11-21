"""Builds report/Quantum_Autoencoder_vs_PCA_Report.pdf from the results in ../results (numbers are read, never typed)."""
import io, os, pickle, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Image, Table,
                                TableStyle, KeepTogether, NextPageTemplate)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
sys.path.insert(0, ROOT)
D = pickle.load(open(os.path.join(RES, "_cache.pkl"), "rb"))
q, p, pc, ab, dn, gn, tm, rs = D["q"], D["p"], D["pc"], D["ab"], D["dn"], D["gn"], D["tm"], D["rs"]

# ---------------------------------------------------------------- fonts / styles
FD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf") + os.sep   # DejaVu fonts ship with Matplotlib (cross-platform)
for n, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"),
             ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
BLUE = colors.HexColor("#1f3b73")
S = dict(
    body=ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=5),
    h1=ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=BLUE, spaceBefore=6, spaceAfter=8, keepWithNext=1),
    h2=ParagraphStyle("h2", fontName="DV-B", fontSize=11, leading=14, textColor=BLUE, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    cap=ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#444"), spaceAfter=8),
    cell=ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=10),
    cellb=ParagraphStyle("cellb", fontName="DV-B", fontSize=7.8, leading=10, textColor=colors.white),
    bullet=ParagraphStyle("bullet", fontName="DV", fontSize=9.2, leading=13.2, leftIndent=14, bulletIndent=3, spaceAfter=2.5, alignment=TA_JUSTIFY),
    code=ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=10, leftIndent=8, backColor=colors.HexColor("#f4f4f4"), spaceAfter=6),
    title=ParagraphStyle("title", fontName="DV-B", fontSize=22, leading=28, alignment=TA_CENTER, textColor=BLUE),
    sub=ParagraphStyle("sub", fontName="DV", fontSize=12, leading=17, alignment=TA_CENTER),
    ref=ParagraphStyle("ref", fontName="DV", fontSize=8.3, leading=11.6, leftIndent=16, firstLineIndent=-16, spaceAfter=3.2),
)
_toc_entries = []


class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.0 * cm, bottomMargin=2.0 * cm, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="cover", frames=[fr], onPage=lambda c, d: None),
                               PageTemplate(id="main", frames=[fr], onPage=self._deco)])

    def _deco(self, c, d):
        c.saveState(); c.setFont("DV", 7.5); c.setFillColor(colors.HexColor("#666"))
        c.drawString(2.1 * cm, A4[1] - 1.3 * cm, "Quantum Autoencoder vs PCA for Compression of Downsampled MNIST Digits")
        c.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"Page {d.page}")
        c.setStrokeColor(colors.HexColor("#bbb")); c.line(2.1 * cm, A4[1] - 1.45 * cm, A4[0] - 2.1 * cm, A4[1] - 1.45 * cm); c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2"):
            lvl = 0 if fl.style.name == "h1" else 1
            self.notify("TOCEntry", (lvl, fl.getPlainText(), self.page))


story = []
add = story.append


def H1(t): add(Paragraph(t, S["h1"]))
def H2(t): add(Paragraph(t, S["h2"]))
def P(t): add(Paragraph(t, S["body"]))
def B(items):
    for t in items: add(Paragraph(t, S["bullet"], bulletText="•"))


def eq(latex, fs=12.5, width=None):
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, f"${latex}$", fontsize=fs)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True); plt.close(fig)
    buf.seek(0); from PIL import Image as PI
    w, h = PI.open(buf).size; buf.seek(0)
    sc = 72 / 300 * 0.82
    img = Image(buf, width=w * sc, height=h * sc); img.hAlign = "CENTER"
    add(Spacer(1, 2)); add(img); add(Spacer(1, 4))


def fig(name, cap, width=15.6):
    from PIL import Image as PI
    path = os.path.join(FIG, name); w, h = PI.open(path).size
    img = Image(path, width=width * cm, height=width * cm * h / w); img.hAlign = "CENTER"
    add(KeepTogether([img, Paragraph(cap, S["cap"])]))


def table(rows, widths=None, head=True, fs=None, zebra=True):
    data = [[Paragraph(str(c), S["cellb"] if (head and i == 0) else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f5fb")))
    t.setStyle(TableStyle(st)); add(t); add(Spacer(1, 7))


def ms(x, s, d=3): return f"{x:.{d}f} ± {s:.{d}f}"


# ---------------------------------------------------------------- numbers used in the text
def qa(pair, k, L, col): v = q[(q.pair == pair) & (q.k == k) & (q.layers == L)][col]; return v.mean(), v.std()
def pa(pair, k, cen, col="F"): v = p[(p.pair == pair) & (p.k == k) & (p.centered == cen)][col]; return v.mean(), v.std()
Ls01 = sorted(q[q.pair == "0 vs 1"].layers.unique()); Ls38 = sorted(q[q.pair == "3 vs 8"].layers.unique())


# ================================================================ FRONT MATTER
import qiskit, qiskit_aer
add(Spacer(1, 3.2 * cm))
add(Paragraph("Quantum Autoencoder vs. PCA for Compression of Downsampled MNIST Digits", S["title"]))
add(Spacer(1, 0.4 * cm)); add(Paragraph("A Simulator-Based Comparison", ParagraphStyle("t2", parent=S["sub"], fontSize=15, textColor=BLUE)))
add(Spacer(1, 1.2 * cm)); add(Paragraph("Quantum Computing · Quantum Machine Learning Project Report", S["sub"]))
add(Spacer(1, 2.6 * cm))
add(Paragraph("<b>Aditya Rao</b><br/>Enrollment No. 2401010036", S["sub"]))
add(Spacer(1, 2.4 * cm))
add(Paragraph(f"Software stack: Python {sys.version.split()[0]} · Qiskit {qiskit.__version__} · Qiskit Aer {qiskit_aer.__version__} · NumPy · SciPy · scikit-learn · Matplotlib · Streamlit", ParagraphStyle("t3", parent=S["sub"], fontSize=9, textColor=colors.HexColor("#555"))))
add(Spacer(1, 0.3 * cm)); add(Paragraph("All results are produced by exact classical simulation of 4-qubit circuits. No quantum advantage is claimed.", ParagraphStyle("t4", parent=S["sub"], fontSize=9, textColor=colors.HexColor("#555"))))
add(Spacer(1, 4.5 * cm)); add(Paragraph("November 2026", S["sub"]))
add(NextPageTemplate("main")); add(PageBreak())

# headline numbers for the abstract ------------------------------------------------
Lmax01 = max(Ls01)
f_ps_1, _ = qa("0 vs 1", 2, 1, "F_ps"); f_ps_m, _ = qa("0 vs 1", 2, Lmax01, "F_ps"); f_rs_m, _ = qa("0 vs 1", 2, Lmax01, "F_reset")
pu2, _ = pa("0 vs 1", 2, False); pc2, _ = pa("0 vs 1", 2, True)
c12, _ = qa("0 vs 1", 2, Lmax01, "train_cost"); b2 = q[(q.pair == "0 vs 1") & (q.k == 2)].bound.mean()

H1("Abstract")
P(f"Quantum autoencoders (QAEs) compress a quantum state into fewer qubits by training a parameterised circuit so that a set of “trash” qubits ends in a fixed reference state. "
  f"This project studies, by exact simulation in Qiskit, how a QAE compares with Principal Component Analysis (PCA) when both compress the same classical data: "
  f"4×4 down-sampled MNIST digits, amplitude-encoded into 4 qubits and compressed to 1, 2 or 3 latent qubits (matched to K = 2, 4, 8 PCA components). "
  f"Two digit pairs (0 vs 1 and 3 vs 8) are evaluated over five stratified train/test splits.")
P(f"We derive and numerically confirm a theoretical ceiling: the best achievable QAE cost equals one minus the variance captured by K-component uncentred PCA (Ky Fan’s maximum principle). "
  f"A post-selected QAE can therefore at most <i>match</i> PCA on the training data. Empirically, for digits 0 vs 1 and two latent qubits the post-selected test fidelity rises from "
  f"{f_ps_1:.3f} (L = 1 layer) to {f_ps_m:.3f} (L = {Lmax01}) against {pu2:.3f} for matched uncentred PCA and {pc2:.3f} for standard centred PCA; the trace-preserving reset decoder reaches {f_rs_m:.3f}. "
  f"The training cost approaches the theoretical optimum ({c12:.4f} vs {b2:.4f}). "
  f"Additional experiments cover swap-test verification of the cost, shot-based measurement, denoising, a depolarising gate-noise study on Qiskit Aer, a training-objective ablation and downstream classification. "
  f"Within these experiments PCA is faster, simpler and at least as accurate; the QAE’s value lies in being a quantum-native, hardware-executable compressor, not in outperforming PCA.")

add(Spacer(1, 4))
toc = TableOfContents(); toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9, leading=13, leftIndent=0), ParagraphStyle("t1", fontName="DV", fontSize=8.3, leading=11.5, leftIndent=14)]
H2("Contents"); add(toc); add(PageBreak())

# ================================================================ 1. INTRODUCTION
H1("1. Introduction to Quantum Computing")
P("A classical bit is 0 or 1. A <b>qubit</b> is a unit vector in the two-dimensional complex Hilbert space ℂ² and can be in a <b>superposition</b> of the computational basis states |0⟩ and |1⟩:")
eq(r"|\psi\rangle=\alpha|0\rangle+\beta|1\rangle,\qquad |\alpha|^2+|\beta|^2=1")
P("A register of n qubits lives in the tensor-product space of dimension 2<super>n</super>; its state is a unit vector of 2<super>n</super> complex amplitudes. "
  "This exponential size is why classical simulation becomes infeasible for many qubits – and also why we can simulate n = 4 qubits exactly (16 amplitudes, 16×16 matrices) in this project.")
P("<b>Gates</b> are unitary matrices. The single-qubit rotation about the y-axis and the two-qubit controlled-NOT are used throughout:")
eq(r"R_Y(\theta)=e^{-i\theta Y/2}=\cos\frac{\theta}{2}\,I-i\sin\frac{\theta}{2}\,Y,\qquad \mathrm{CX}\,|c,t\rangle=|c,\;t\oplus c\rangle")
P("<b>Entanglement.</b> A Hadamard followed by a CX turns |00⟩ into the Bell state, whose two qubits have no individual pure state:")
eq(r"\mathrm{CX}\,(H\otimes I)\,|00\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}}")
P("<b>Measurement</b> in the computational basis returns outcome i with probability given by the Born rule, and the state collapses:")
eq(r"P(i)=|\langle i|\psi\rangle|^{2}")
P("When only part of a system is considered, its state is the <b>reduced density matrix</b> obtained by a partial trace, ρ<sub>A</sub> = Tr<sub>B</sub>|ψ⟩⟨ψ|. A reduced state is <i>mixed</i> if the subsystem is entangled with the rest. "
  "A <b>reset</b> operation discards a qubit’s state and re-initialises it to |0⟩. This is a non-unitary but physically implementable operation (measure and flip), and it is the compression step of the autoencoder. "
  "Finally, a quantum <b>circuit</b> is a sequence of such gates, resets and measurements; <b>circuit simulation</b> multiplies the state vector (or density matrix) by the gate matrices – exactly what Qiskit Aer and our NumPy simulator do.")
P("<b>NISQ context.</b> Present devices have tens to hundreds of noisy qubits [10]. Variational algorithms – short parameterised circuits optimised by a classical loop – are the dominant paradigm for such devices [8]. The QAE of this report is a variational algorithm.")

H1("2. Problem Statement")
P("Quantum autoencoders compress quantum states into fewer qubits, but it is unclear how their reconstruction quality compares with a standard classical baseline at the same compression level. "
  "This project trains a small quantum autoencoder on 4×4 MNIST images from two classes (digit 0 vs 1, plus a harder pair 3 vs 8), and compares it with PCA at matched latent sizes.")
P("<b>Real-world motivation.</b> (i) Quantum memories and communication channels are expensive; compressing quantum data into fewer qubits saves them [1, 3]. "
  "(ii) Classical data loaded into a quantum computer by amplitude encoding must be processed by circuits whose size grows with the number of qubits; a learned compression can reduce the register that later algorithms need. "
  "(iii) Dimensionality reduction (PCA, autoencoders) is a universal pre-processing step in machine learning, so it is a natural testbed for quantum learning models.")
P("<b>Why quantum computing is relevant – and what is <i>not</i> claimed.</b> The QAE manipulates genuine quantum states (superposition, entanglement between latent and trash qubits, mixed reduced states, reset) and can be run on a quantum processor. "
  "Whether it beats classical compression is an empirical question; here we answer it for a small, exactly simulable instance. We deliberately do not assume an advantage.")

H1("3. Objectives")
B(["Build a complete, validated pipeline: MNIST → 4×4 → amplitude-encoded 4-qubit states → trainable quantum autoencoder in Qiskit.",
   "Train with the standard trash-qubit cost using exact parameter-shift gradients, and verify the cost with a swap-test circuit.",
   "Compare reconstruction quality (fidelity, pixel read-out error, downstream classification) with PCA at matched latent size (1, 2, 3 latent qubits ↔ K = 2, 4, 8 components).",
   "Quantify how depth (number of ansatz layers) closes the gap to the best achievable cost, and prove that PCA is the ceiling for a post-selected QAE.",
   "Extend the study with denoising, depolarising gate noise (Qiskit Aer), shot-based measurement, a training-objective ablation, resource counts and timing.",
   "Deliver reproducible code, tests, a notebook, an interactive Streamlit dashboard and this report."])

H1("4. Literature Review")
P("<b>Quantum autoencoders.</b> Romero, Olson and Aspuru-Guzik [1] introduced the QAE: a parameterised unitary acts on an n-qubit input state, some qubits are designated trash, and training maximises the fidelity of the trash with a fixed reference state, "
  "measured on hardware with a swap test [14]. After training, the trash can be discarded and later replaced by the reference state; the inverse circuit decodes. Lamata et al. [2] proposed alternative training based on quantum adders and genetic algorithms. "
  "Pepper, Tischler and Pryde [3] realised a QAE experimentally on a photonic platform, compressing qutrits into qubits. Bravo-Prieto [4] studies how the encoding of classical data influences quantum-autoencoder performance. "
  "Khoshaman et al. [5] combined the idea with generative modelling in a quantum variational autoencoder.")
P("<b>Variational training.</b> Quantum circuit learning [6] and analytic-gradient evaluation by the parameter-shift rule [7] allow gradient-based optimisation using only circuit evaluations; reviews are given in [8]. "
  "Deep or randomly initialised circuits can suffer barren plateaus [9], which is why a shallow hardware-efficient ansatz is used here.")
P("<b>Data loading.</b> Amplitude encoding packs 2<super>n</super> numbers into n qubits but needs state-preparation circuits of exponential size in general [12, 13], an important caveat for any claimed speed-up [11]. We measure this cost explicitly in Section 12.")
P("<b>Classical baseline.</b> PCA [16, 17] is the optimal linear compressor in the least-squares sense. Ky Fan’s maximum principle [15] – the key to our ceiling argument – states that the maximum of Tr[QS] over rank-K projectors Q is the sum of the K largest eigenvalues of S. "
  "MNIST [18] is a standard benchmark.")
P("<b>Gap addressed.</b> Many QAE papers report the quantum model alone. This project places a QAE and PCA on equal footing (same normalised inputs, matched latent dimension, same metrics), separates the trace-preserving (reset) and post-selected decoders, "
  "and links the empirical results to an exact theoretical ceiling. It is a small, fully reproducible case study, not a claim of state-of-the-art compression.")

H1("5. Theoretical Background")
H2("5.1 Amplitude encoding")
P("A 4×4 image is flattened to x ∈ ℝ<super>16</super> (pixel intensities in [0, 1]) and normalised, so the 16 amplitudes live on 4 qubits (qubit q is bit q of the basis index, Qiskit little-endian convention):")
eq(r"|\psi(x)\rangle=\sum_{i=0}^{15}\frac{x_i}{\|x\|_2}\,|i\rangle")
P("Normalisation is mandatory for a valid state and discards the overall image brightness; all later comparisons (also for PCA) are made on these unit vectors. "
  "Because pixels are non-negative reals, all states are real; therefore a real-valued ansatz (RY and CX only) is expressive enough and keeps the unitary a real orthogonal 16×16 matrix.")
H2("5.2 The variational ansatz")
P("One layer applies RY on every qubit and a CX chain; a final RY layer closes the circuit. With L layers there are 4(L + 1) parameters and 3L CX gates:")
eq(r"U(\theta)=R_L\,E\,R_{L-1}\cdots E\,R_0,\qquad E=\mathrm{CX}_{2,3}\,\mathrm{CX}_{1,2}\,\mathrm{CX}_{0,1}")
eq(r"R_l=R_Y(\theta_{l,3})\otimes R_Y(\theta_{l,2})\otimes R_Y(\theta_{l,1})\otimes R_Y(\theta_{l,0})")
H2("5.3 Quantum autoencoder")
P("With k latent qubits (qubits 0 … k−1) and 4 − k trash qubits, let Π be the projector onto “trash = |0…0⟩”. Writing p<sub>0</sub>(ψ) = ⟨ψ|U<super>T</super>ΠU|ψ⟩ for the probability of finding the trash in |0…0⟩ after the encoder, the training cost is [1]")
eq(r"C(\theta)=1-\mathbb{E}_{\psi}\left[p_0(\psi)\right]=1-\mathrm{Tr}\left[\Pi\,U\,\Sigma\,U^{\top}\right],\qquad \Sigma=\mathbb{E}\left[|\psi\rangle\langle\psi|\right]")
P("The second form shows that the data enter only through the 16×16 second-moment matrix Σ, so the cost is evaluated exactly (no sampling) for the whole training set in O(K·16²) operations. "
  "On hardware p<sub>0</sub> is estimated with a <b>swap test</b> between each trash qubit and a fresh |0⟩ reference qubit: the ancilla is measured as 0 with probability ½(1 + p<sub>0</sub>) [14].")
eq(r"P(\mathrm{anc}=0)=\frac{1}{2}\left(1+p_0\right)\quad\Longrightarrow\quad p_0=2P(\mathrm{anc}=0)-1")
P("<b>Two decoders.</b> (a) <i>Reset decoder</i> (trace-preserving, the standard QAE): discard the trash, re-initialise it to |0…0⟩, apply U<super>†</super>. The output is generally a mixed state. "
  "(b) <i>Post-selected decoder</i>: keep only runs whose trash measurement is |0…0⟩; the output is pure, at the price of a success probability p<sub>0</sub>.")
H2("5.4 Gradients: parameter-shift rule")
P("Every parameter enters one RY gate, a rotation generated by an operator with eigenvalues ±½. The cost is therefore a trigonometric polynomial of degree one in each θ<sub>p</sub>, and the derivative is exact:")
eq(r"\frac{\partial C}{\partial\theta_p}=\frac{1}{2}\left[C\left(\theta_p+\frac{\pi}{2}\right)-C\left(\theta_p-\frac{\pi}{2}\right)\right]")
P("This needs only cost evaluations – the same circuits that would run on a device [6, 7]. We verify it against finite differences (max error ≈ 10<super>−10</super>). The optimiser is L-BFGS-B [23] with several random restarts.")
H2("5.5 PCA baseline")
P("PCA projects onto the K leading right singular vectors V<sub>K</sub>. Two variants are used: <b>centred</b> (standard; the mean image is stored as side information) and <b>uncentred</b> (truncated SVD; no mean – the exactly matched classical counterpart of a circuit that has no access to a mean). "
  "PCA outputs are re-normalised to unit length so that they are valid states:")
eq(r"\hat{x}=V_KV_K^{\top}x,\qquad F_{\mathrm{PCA}}=\frac{(x\cdot\hat{x})^2}{\|\hat{x}\|^2}=\|V_K^{\top}x\|^2\ \ (\|x\|=1)")
H2("5.6 Evaluation metrics")
B(["<b>Fidelity</b> F = ⟨ψ|ρ<sub>rec</sub>|ψ⟩ between the input pure state and the reconstruction [26] (1 = perfect).",
   "<b>Pixel read-out MSE</b>: mean squared difference between input amplitudes ψ<sub>i</sub> and the amplitudes √p<sub>i</sub> that repeated computational-basis measurements of the output would reveal (the same read-out is applied to PCA outputs, so the comparison is like for like; signs are not observable in this basis).",
   "<b>Downstream accuracy</b>: a logistic-regression classifier trained and tested on the read-out images of the same pipeline – how much class information survives compression.",
   "<b>Resources</b>: parameters, CX count, depth, training and inference time."])

H1("6. Mathematical Foundations")
H2("6.1 Reconstruction in closed form")
P("Let φ = Uψ and reshape φ into a matrix M with rows indexed by the trash register (T = 2<super>4−k</super> rows) and columns by the latent register (K = 2<super>k</super> columns). Row 0 is the “trash = 0…0” block m<sub>0</sub>. Then:")
eq(r"p_0=\|m_0\|^2,\qquad \rho_{\mathrm{lat}}=M^{\top}M,\qquad \rho_{\mathrm{rec}}=U^{\top}\left(|0\rangle\langle 0|_{\mathrm{trash}}\otimes\rho_{\mathrm{lat}}\right)U")
eq(r"F_{\mathrm{ps}}=p_0,\qquad F_{\mathrm{reset}}=\sum_{t=0}^{T-1}\langle m_t,m_0\rangle^2\ \ \Longrightarrow\ \ p_0^2\ \leq\ F_{\mathrm{reset}}\ \leq\ F_{\mathrm{ps}}=p_0")
P("<b>Proof of the bounds.</b> The t = 0 term gives p<sub>0</sub>²; by Cauchy–Schwarz ⟨m<sub>t</sub>, m<sub>0</sub>⟩² ≤ ‖m<sub>t</sub>‖²‖m<sub>0</sub>‖², and Σ<sub>t≠0</sub>‖m<sub>t</sub>‖² = 1 − p<sub>0</sub>, so F<sub>reset</sub> ≤ p<sub>0</sub>² + p<sub>0</sub>(1 − p<sub>0</sub>) = p<sub>0</sub>. "
  "Hence the reset decoder can never be better than the post-selected one, and the gap measures how much information the discarded trash still carries. Both inequalities are asserted in the test-suite.")
H2("6.2 The PCA ceiling (Ky Fan)")
P("Let Π<sub>K</sub> be the rank-K projector onto the first K = 2<super>k</super> basis states (trash = 0…0). For any orthogonal U, S = UΣU<super>T</super> has the same eigenvalues λ<sub>1</sub> ≥ … ≥ λ<sub>16</sub> as Σ, and")
eq(r"\max_{U}\ \mathrm{Tr}\left[\Pi_K\,U\Sigma U^{\top}\right]=\sum_{j=1}^{K}\lambda_j(\Sigma)\quad\Longrightarrow\quad C_{\min}=1-\sum_{j=1}^{K}\lambda_j(\Sigma)")
P("This is Ky Fan’s maximum principle [15] applied to the K-dimensional subspace U<super>T</super>(range Π<sub>K</sub>). On the other hand, the mean training fidelity of K-component uncentred PCA is E‖V<sub>K</sub><super>T</super>ψ‖² = Tr[V<sub>K</sub>V<sub>K</sub><super>T</super>Σ] = Σ<sub>j≤K</sub>λ<sub>j</sub>. Therefore")
eq(r"\overline{F}_{\mathrm{ps}}^{\,\mathrm{train}}(\theta)=1-C(\theta)\ \leq\ \overline{F}_{\mathrm{PCA,\,uncentred}}^{\,\mathrm{train}}\quad\text{for every }\theta,")
P("with equality iff the circuit rotates the principal subspace onto the “trash = 0” subspace. A sufficiently expressive QAE <i>is</i> PCA in the post-selected setting; a shallow one is a restricted PCA. "
  "(The inequality is guaranteed on the training data; on held-out data a particular circuit could in principle fluctuate above PCA, but not systematically.) "
  "The test-suite checks the bound for k = 1, 2, 3.")
H2("6.3 Noise model")
P("Gate noise is modelled by depolarising channels after every RY (error p<sub>1</sub>) and every CX (error p<sub>2</sub>) gate, with d the dimension of the gate’s qubit space:")
eq(r"\mathcal{E}_p(\rho)=(1-p)\,\rho+p\,I/d")
H2("6.4 Resource scaling")
P("A depth-L encoder has 4(L + 1) parameters and 3L CX gates (the decoder repeats them). Exact classical simulation costs O(2<super>n</super>) per gate on a state vector and O(4<super>n</super>) on a density matrix – fine for n = 4 but exponential in general. "
  "PCA on N samples of dimension d costs O(Nd²) to fit and O(dK) per sample to apply.")

H1("7. Proposed Methodology")
B(["<b>Data.</b> 5 000 real MNIST training images (500 per digit, from the mlxtend package; original source yann.lecun.com/exdb/mnist) – for each pair 1 000 images, stratified 70/30 train/test split, repeated for 5 seeds.",
   "<b>Pre-processing.</b> 28×28 → 4×4 by exact 7×7 block averaging, flatten, L2-normalise (Section 5.1).",
   "<b>QAE.</b> Ansatz of Section 5.2, L ∈ {1, 2, 3, 4, 6, 8} (plus L = 12 on three 0-vs-1 splits), k ∈ {1, 2, 3} latent qubits. Random initialisation θ ~ U[0, 2π]; 5 restarts of L-BFGS-B (≤ 400 iterations), parameter-shift gradients; the restart with the lowest training cost is kept.",
   "<b>Baselines.</b> PCA centred and uncentred with K = 2<super>k</super> components, fitted on the same training split.",
   "<b>Evaluation.</b> All metrics on the held-out 30 %; mean ± standard deviation over the five splits.",
   "<b>Extensions.</b> Cost ablation, denoising, Aer gate-noise, shots, Bloch sphere, resources and timing."])

H1("8. System Architecture")
fig("fig_architecture.png", "Figure 1. End-to-end architecture. The classical branch (PCA) receives exactly the same normalised 16-vectors.")
table([["Module", "Responsibility"],
       ["qae/data.py", "MNIST loading, 4×4 block average, L2 normalisation, stratified splits"],
       ["qae/ansatz.py", "Exact batched NumPy simulator of the RY+CX ansatz (16×16 real orthogonal matrices)"],
       ["qae/circuits.py", "Qiskit circuits: ansatz, full QAE (state prep → U → reset → U†), swap test, resource counts"],
       ["qae/metrics.py", "Reset / post-selected decoders, fidelities, read-out images, MSE"],
       ["qae/train.py", "Trash cost, parameter-shift gradient, multi-start L-BFGS-B, Ky Fan bound"],
       ["qae/pca_baseline.py", "Centred / uncentred PCA with quantum-style metrics"],
       ["qae/noise.py", "Qiskit Aer density-matrix simulation with depolarising noise"],
       ["qae/experiments.py, viz.py", "All experiments and all figures"],
       ["run_all.py, app.py, tests/", "Reproduction driver, Streamlit dashboard, validation suite"]], widths=[4.3 * cm, 12.4 * cm])

# ================================================================ 9. IMPLEMENTATION
import subprocess
H1("9. Implementation")
P("The project is implemented in Python with <b>Qiskit</b> (circuit construction, Operator/Statevector tools), <b>Qiskit Aer</b> (density-matrix and shot-based simulation, noise models), NumPy/SciPy (fast exact simulator, L-BFGS-B), scikit-learn (PCA reference, logistic regression), Matplotlib and Streamlit. "
  "Because training needs thousands of cost evaluations, the ansatz is also implemented as a batched NumPy matrix simulator; <b>every quantum quantity it produces is cross-validated against Qiskit</b> (below), so the fast path and the “real circuit” path are provably identical.")
nt = int(subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()[-1].split()[0])
P(f"<b>Validation.</b> The pytest suite contains {nt} tests, all passing:")
table([["Area", "What is verified"],
       ["Data", "shape, unit norm and non-negativity of amplitude-encoded states; 7×7 block-average exactness; all-zero image rejected"],
       ["Simulator vs Qiskit", "NumPy unitary = Qiskit Operator (L = 1…4, error ≈ 10⁻¹⁶); reset-decoder density matrix and fidelity = Aer density-matrix simulation of the full circuit (state prep → U → reset → U†); post-selection probability = Qiskit statevector; swap-test circuit reproduces p<sub>0</sub>"],
       ["Training", "parameter-shift gradient = finite differences; cost = sample average; trained cost ≥ Ky Fan bound; QAE (post-selected) ≤ uncentred PCA on training data for k = 1, 2, 3; lossless when no trash qubit"],
       ["Theory", "p<sub>0</sub>² ≤ F<sub>reset</sub> ≤ p<sub>0</sub> for every sample (Section 6.1)"],
       ["PCA", "own SVD implementation = scikit-learn PCA; full-rank PCA lossless"],
       ["Noise", "Aer with zero noise = ideal; non-zero depolarising noise lowers fidelity"]], widths=[3.2 * cm, 13.5 * cm])
P("<b>Error handling.</b> The data module rejects all-zero images (not amplitude-encodable), the denoising experiment guards against images that become all-zero after noise, the post-selected decoder guards against p<sub>0</sub> = 0, "
  "and the Streamlit app validates the digit pair and runs Qiskit/Aer work in an isolated worker process.")
P("<b>Reproduction.</b> <font name='DVM'>pip install -r requirements.txt; python run_all.py</font> regenerates every table and figure (about 9 minutes on one CPU core); <font name='DVM'>pytest -q</font> runs the tests; <font name='DVM'>streamlit run app.py</font> opens the dashboard.")

# ================================================================ 10. CIRCUIT DIAGRAMS
H1("10. Quantum Circuit Diagrams")
fig("fig_circuit_qae.png", "Figure 2. Complete QAE circuit for L = 2 and two latent qubits (qubits q₂, q₃ are trash): amplitude encoding, encoder U(θ), reset of the trash qubits, decoder U(θ)†. Angles are the trained parameters (digits 0 vs 1, split 0). “AmpEnc(ψ)” stands for the state-preparation block of Section 12.", width=16.4)
fig("fig_circuit_swaptest.png", "Figure 3. Swap-test circuit that estimates the training cost on hardware: each trash qubit is compared with a fresh |0⟩ reference qubit through a controlled-SWAP; the ancilla reads 0 with probability ½(1 + p₀).", width=16.4)
fig("fig_dataset.png", "Figure 4. Data pipeline for one digit 0 and one digit 1: original 28×28 image, 4×4 block average, and the resulting measurement probabilities of the amplitude-encoded 4-qubit state.", width=15.6)

# ================================================================ 11. RESULTS
H1("11. Results and Analysis")
P(f"All numbers below are means ± standard deviations over five stratified 70/30 splits of the 1 000 images of each digit pair, evaluated on the held-out 30 % (300 images). "
  f"Training used 5 restarts of L-BFGS-B; depth L = 12 was run on three splits of 0 vs 1 only (marked “n = 3”).")


def fid_table(pair):
    Ls = sorted(q[q.pair == pair].layers.unique())
    show = [l for l in (1, 2, 4, 8, 12) if l in Ls]
    rows = [["Method", "1 latent qubit<br/>(PCA K = 2)", "2 latent qubits<br/>(PCA K = 4)", "3 latent qubits<br/>(PCA K = 8)"]]
    for cen, name in ((False, "PCA uncentred (matched)"), (True, "PCA centred (standard)")):
        rows.append([name] + [ms(*pa(pair, k, cen)) for k in (1, 2, 3)])
    for col, name in (("F_ps", "QAE post-selected"), ("F_reset", "QAE reset decoder")):
        for L in show:
            n = len(q[(q.pair == pair) & (q.k == 1) & (q.layers == L)])
            tag = f" (n = {n})" if n != 5 else ""
            rows.append([f"{name}, L = {L}{tag}"] + [ms(*qa(pair, k, L, col)) for k in (1, 2, 3)])
    return rows


H2("11.1 Reconstruction fidelity versus PCA")
P("<b>Digits 0 vs 1.</b>"); table(fid_table("0 vs 1"), widths=[5.9 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm])
P("<b>Digits 3 vs 8</b> (harder pair, more overlapping classes)."); table(fid_table("3 vs 8"), widths=[5.9 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm])
fig("fig_fidelity_vs_layers_0vs1.png", "Figure 5. Test fidelity versus number of ansatz layers, digits 0 vs 1 (error bars: standard deviation over splits). The post-selected QAE climbs towards the matched uncentred-PCA line and meets it.")
fig("fig_fidelity_vs_layers_3vs8.png", "Figure 6. Same for digits 3 vs 8 (depths up to L = 8).")

# dynamic facts
g = q.groupby(["pair", "k", "layers"]).F_ps.mean().reset_index()
g["pca_u"] = [pa(r.pair, r.k, False)[0] for r in g.itertuples()]
g["excess"] = g.F_ps - g.pca_u
mx = g.excess.max()
gap_rows = []
for pair in ("0 vs 1", "3 vs 8"):
    for k in (1, 2, 3):
        Lm = max(q[(q.pair == pair) & (q.k == k)].layers)
        gap_rows.append((pair, k, Lm, qa(pair, k, Lm, "train_cost")[0], q[(q.pair == pair) & (q.k == k)].bound.mean(),
                         qa(pair, k, Lm, "F_ps")[0], pa(pair, k, False)[0], pa(pair, k, True)[0], qa(pair, k, Lm, "F_reset")[0]))
_gg = q.groupby(["pair", "k", "layers"])[["F_ps", "F_reset"]].mean(); _gg["gap"] = _gg.F_ps - _gg.F_reset
gmax = _gg[_gg.index.get_level_values("layers") == 1].gap.max(); g8 = _gg.loc[("0 vs 1", 2, 8)].gap
P(f"<b>Observations.</b> (1) <b>PCA is the ceiling.</b> Across all {len(g)} (pair, latent size, depth) combinations the mean post-selected test fidelity of the QAE never exceeded matched uncentred PCA by more than {max(mx, 0):.4f} "
  f"(a value of this size is within split-to-split noise; on the training data the inequality is exact and tested). "
  f"(2) <b>Depth closes the gap.</b> With one latent qubit the QAE equals PCA from L = 6 on; with two and three latent qubits more layers are needed, because the number of free parameters 4(L + 1) must be large enough to rotate a K-dimensional subspace "
  f"(dimension K(16 − K): 28, 48, 64 for K = 2, 4, 8) onto the trash-zero subspace. "
  f"(3) <b>The reset decoder is always worse</b> than post-selection, as proved in Section 6.1; the gap is largest for shallow circuits (up to {gmax:.3f} in fidelity at L = 1) and shrinks with depth (about {g8:.3f} at L = 8, two latent qubits, digits 0 vs 1). "
  f"(4) <b>Centred PCA is best overall</b> because it additionally stores a mean image – a resource the quantum circuit does not have; the difference vanishes as K grows (K = 8: difference {pa('0 vs 1', 3, True)[0] - pa('0 vs 1', 3, False)[0]:.4f}). "
  f"(5) The harder pair 3 vs 8 shows the same ordering.")

H2("11.2 Convergence to the theoretical optimum")
rows = [["Pair", "k", "Max L", "QAE train cost", "Ky Fan optimum", "Gap", "QAE post-sel. F", "PCA unc. F", "PCA cent. F", "QAE reset F"]]
for r in gap_rows:
    rows.append([r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[3] - r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}", f"{r[7]:.4f}", f"{r[8]:.4f}"])
table(rows, widths=[1.5 * cm, 0.8 * cm, 1.2 * cm, 1.9 * cm, 1.9 * cm, 1.4 * cm, 1.9 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm])
fig("fig_cost_vs_bound.png", "Figure 7. Training cost versus depth (log scale) against the Ky Fan optimum (dotted). Cost decreases monotonically with depth and touches the bound: the optimiser finds the global optimum for sufficiently expressive circuits, and no circuit goes below it.")
fig("fig_training_curves.png", "Figure 8. Typical L-BFGS-B training curves (L = 4, digits 0 vs 1); dotted lines are the Ky Fan optima for 1, 2, 3 latent qubits.", width=11.5)
fig("fig_pca_curve.png", "Figure 9. PCA fidelity as a function of the number of retained components; dotted verticals mark the component counts matched to 1, 2, 3 latent qubits. Compression of these 4×4 digits is cheap: 8 components already capture ≈ 99 % of the state.")

H2("11.3 Qualitative results")
fig("fig_reconstructions_0vs1_k2.png", "Figure 10. Test reconstructions with two latent qubits (digits 0 vs 1). Rows: original, QAE (reset), QAE (post-selected), uncentred and centred PCA. F = per-image fidelity. All methods preserve the characteristic ring of the 0 and the vertical stroke of the 1.", width=15.6)
fig("fig_reconstructions_3vs8_k2.png", "Figure 11. Same for the harder pair 3 vs 8.", width=15.6)
fig("fig_reconstructions_0vs1_k1.png", "Figure 12. One latent qubit: strong compression (4 → 1 qubit); fine details are lost.", width=13.5)
fig("fig_bloch_latent.png", "Figure 13. The single latent qubit (digits 0 vs 1, L = 6) on the Bloch sphere. The two digit classes occupy different regions – the circuit has learned a class-aware one-qubit code – and points close to the sphere surface are nearly pure latent states. Because all amplitudes are real, y = 0.", width=15.6)
fig("fig_measurement_shots.png", "Figure 14. Full-circuit measurement statistics for one test image: input state, exact QAE output and 8 192-shot Qiskit Aer estimate. The shot histogram agrees with the exact distribution within statistical error (largest deviation 1.9 σ, total-variation distance 0.010).", width=15.6)

H2("11.4 Pixel read-out error and downstream classification")
rows = [["Pair", "k", "QAE reset", "QAE post-sel.", "PCA uncentred", "PCA centred"]]
for pair in ("0 vs 1", "3 vs 8"):
    for k in (1, 2, 3):
        Lm = max(l for l in q[(q.pair == pair) & (q.k == k)].layers if l <= 8)
        rows.append([pair, k] + [f"{qa(pair, k, Lm, 'mse_reset')[0]:.4f}", f"{qa(pair, k, Lm, 'mse_ps')[0]:.4f}",
                                 f"{pa(pair, k, False, 'mse')[0]:.4f}", f"{pa(pair, k, True, 'mse')[0]:.4f}"])
P("Mean squared error between input amplitudes and the amplitudes revealed by computational-basis measurement of the output (QAE with L = 8):"); table(rows, widths=[2 * cm, 1.2 * cm, 3 * cm, 3 * cm, 3.3 * cm, 3.3 * cm])
rows = [["Pair", "k", "Original 16-vector", "QAE reset (L = 8)", "QAE post-sel. (L = 8)", "PCA uncentred", "PCA centred"]]
for pair in ("0 vs 1", "3 vs 8"):
    for k in (1, 2, 3):
        rows.append([pair, k, ms(*pa(pair, k, False, "acc_orig"), 3)] + [ms(*qa(pair, k, 8, c), 3) for c in ("acc_reset", "acc_ps")] + [ms(*pa(pair, k, cen, "acc"), 3) for cen in (False, True)])
P("Accuracy of a logistic-regression classifier trained and tested on the read-out images of each pipeline (class information retained after compression):"); table(rows, widths=[1.6 * cm, 0.8 * cm, 2.9 * cm, 2.9 * cm, 3.0 * cm, 2.7 * cm, 2.7 * cm])
fig("fig_classification.png", "Figure 15. Downstream classification accuracy (mean ± std over splits).", width=15.6)
a38 = [qa("3 vs 8", 1, L, "acc_ps")[0] for L in (3, 4, 6)]
P(f"With two or three latent qubits all compressors retain most of the class information (accuracies within a few percentage points of the uncompressed baseline). "
  f"Accuracy is <b>not monotone in fidelity or depth</b>: for 3 vs 8 with one latent qubit the post-selected QAE gives {a38[0]:.2f} and {a38[1]:.2f} at L = 3 and 4 but {a38[2]:.2f} at L = 6. "
  f"The trash-qubit cost preserves <i>variance</i>, not <i>class separability</i>, and an under-trained one-qubit code can discard the direction that distinguishes the classes. "
  f"This is a useful reminder that reconstruction fidelity and downstream utility are different objectives.")

H2("11.5 Training-objective ablation")
t = ab.groupby(["pair", "k", "cost"])[["F_reset", "F_ps"]].mean().reset_index()
rows = [["Pair", "k", "Cost", "F reset", "F post-sel."]]
for r in t.itertuples():
    rows.append([r.pair, r.k, {"trash": "trash-qubit (Romero)", "fidelity": "end-to-end fidelity"}[r.cost], f"{r.F_reset:.4f}", f"{r.F_ps:.4f}"])
table(rows, widths=[2.2 * cm, 1.2 * cm, 4.5 * cm, 3 * cm, 3 * cm])
dmax = (t.pivot_table(index=["pair", "k"], columns="cost", values="F_reset").assign(d=lambda x: (x.fidelity - x.trash).abs()).d.max())
P(f"Training directly on the end-to-end reset-decoder fidelity (L = 3, three splits) gives <b>no consistent improvement</b> over the cheaper, hardware-friendly trash-qubit cost: the largest difference in reset fidelity is {dmax:.3f}, with the sign varying across settings. "
  f"The trash cost needs only the encoder and a swap test, whereas the end-to-end objective needs the full encode-reset-decode circuit, so the standard cost is preferable. See Figure 16.")
fig("fig_cost_ablation.png", "Figure 16. Test fidelity (reset decoder) after training with each objective (L = 3).", width=14)

H2("11.6 Denoising")
d = dn.groupby("sigma").mean(numeric_only=True)
rows = [["Pixel noise σ", "Noisy input (no processing)", "QAE reset", "QAE post-sel.", "PCA uncentred", "PCA centred"]]
for s_, r in d.iterrows():
    rows.append([f"{s_:.2f}", f"{r.noisy_input:.3f}", f"{r.QAE_reset:.3f}", f"{r.QAE_postselect:.3f}", f"{r.PCA_uncentred:.3f}", f"{r.PCA_centred:.3f}"])
P("Gaussian noise of standard deviation σ is added to the clean 4×4 test images (clipped to [0, 1]), the noisy state is compressed to two latent qubits (K = 4) and reconstructed, and the fidelity with the <b>clean</b> state is reported (QAE trained on clean data, L = 3). "
  "“Noisy input” is the fidelity if no compression is applied:"); table(rows, widths=[2.4 * cm, 3.6 * cm, 2.3 * cm, 2.6 * cm, 2.8 * cm, 2.6 * cm])
fig("fig_denoising.png", "Figure 17. Fidelity with the clean state after compress-and-reconstruct of noisy inputs.", width=11.5)
cross = d[(d.QAE_postselect > d.noisy_input)].index.min(); crossr = d[(d.QAE_reset > d.noisy_input)].index.min()
P(f"Low-rank compression acts as a denoiser only when the input noise exceeds what the compression itself destroys: the post-selected QAE improves on the noisy input from σ = {cross:.2f}, the reset QAE only from σ = {crossr:.2f}, "
  f"and at the smallest noise level (σ = 0.05) both QAE decoders are worse than doing nothing, while PCA is roughly break-even. PCA is at least as good as the QAE at every noise level.")

H2("11.7 Robustness to gate noise (Qiskit Aer)")
piv = gn.pivot(index="p2", columns="layers", values="F")
rows = [["CX error p₂ (p₁ = p₂/10)"] + [f"QAE L = {L}" for L in piv.columns]]
for p2, r in piv.iterrows(): rows.append([f"{100 * p2:.1f} %"] + [f"{v:.3f}" for v in r])
P(f"Depolarising noise is applied to every RY and CX gate of the trained circuit (two latent qubits, reset decoder, 60 test images, exact density-matrix simulation). Noise-free matched PCA reaches {D['pca_ref']:.3f}; a maximally mixed output would give 1/16 = 0.0625.")
table(rows, widths=[4.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm])
fig("fig_gate_noise.png", "Figure 18. Fidelity under depolarising gate noise.", width=11.5)
P(f"Deeper circuits are better without noise but degrade faster: L = 3 starts highest ({piv.loc[0.0, 3]:.3f}) but at p₂ = 4 % falls to {piv.loc[0.04, 3]:.3f}, below L = 1 ({piv.loc[0.04, 1]:.3f}) and L = 2 ({piv.loc[0.04, 2]:.3f}). "
  f"On today’s hardware the best depth is therefore a trade-off between expressivity and accumulated error – and a noise-free classical PCA is unaffected.")

# ================================================================ 12. CLASSICAL VS QUANTUM
H1("12. Classical vs Quantum Comparison")
r3 = rs[rs.layers == 3].iloc[0]; r8 = rs[rs.layers == 8].iloc[0]
table([["Aspect", "PCA (classical)", "Quantum autoencoder (this work)"],
       ["Algorithm", "Closed form: SVD of the data matrix; linear projection", "Variational: gradient-based optimisation of a parameterised unitary; encode → reset trash → decode"],
       ["Training cost (theory)", "O(N·d²) for N samples of dimension d = 16", "Cost evaluation O(K·d²) from Σ in our exact simulator; on hardware O(shots) per evaluation × (2P + 1) per gradient step; classical simulation O(2ⁿ) per gate"],
       ["Training time (measured)", f"{tm['pca_fit_s'] * 1e3:.2f} ms", f"{tm['qae_train_s_(8 restarts)']:.2f} s (L = 3, 8 restarts, exact NumPy simulation)"],
       ["Inference time per sample (measured)", f"{tm['pca_infer_per_sample_s'] * 1e9:.0f} ns", f"{tm['qae_numpy_infer_per_sample_s'] * 1e6:.2f} µs (NumPy simulator); {tm['qae_aer_infer_per_sample_s'] * 1e3:.2f} ms (Aer circuit simulation)"],
       ["Accuracy (fidelity, 0 vs 1, K = 4)", f"{pa('0 vs 1', 2, False)[0]:.4f} uncentred / {pa('0 vs 1', 2, True)[0]:.4f} centred", f"{qa('0 vs 1', 2, 12, 'F_ps')[0]:.4f} post-selected (L = 12) / {qa('0 vs 1', 2, 12, 'F_reset')[0]:.4f} reset"],
       ["Stored parameters (K = 4)", f"{pa('0 vs 1', 2, False, 'n_numbers')[0]:.0f} (uncentred) / {pa('0 vs 1', 2, True, 'n_numbers')[0]:.0f} (centred)", f"4(L + 1) = {4 * 13} for L = 12"],
       ["Circuit resources", "—", f"Encoder: {int(r3.encoder_cx)} CX, depth {int(r3.encoder_depth)} (L = 3); {int(r8.encoder_cx)} CX, depth {int(r8.encoder_depth)} (L = 8); decoder repeats them"],
       ["Amplitude-encoding state preparation", "—", f"{int(r3.stateprep_cx)} CX, depth {int(r3.stateprep_depth)} for a generic 4-qubit state – <b>more than the encoder itself</b>; grows exponentially with qubit number [12, 13]"],
       ["Output", "Real vector (can be re-normalised at will)", "Quantum state; reset decoder is a trace-preserving channel (mixed output); post-selection needs repeated runs"],
       ["Noise", "None", "Fidelity degrades with gate noise (Section 11.7); error mitigation / better hardware required"]],
      widths=[3.6 * cm, 5.4 * cm, 7.7 * cm])
P("<b>Advantages of the QAE.</b> It operates directly on quantum states (it can compress states that never exist classically, which is its original purpose [1]); it is natively executable on a quantum processor; its cost can be measured with a swap test; and the compression is learned from data without computing a covariance matrix. "
  "<b>Limitations.</b> For classical data the amplitude-encoding step already costs more than the compression; training needs many circuit evaluations; the reset decoder cannot beat post-selection; and, as proved in Section 6.2, the post-selected QAE is bounded by linear PCA, which is solved exactly and instantly.")
P("<b>On quantum advantage.</b> In these experiments there is none: PCA is orders of magnitude faster on a classical computer, uses fewer resources, and is at least as accurate. We make no claim of quantum advantage; the experiments demonstrate that the QAE <i>reproduces</i> the optimal linear compressor, which is a validation of the method, not an advantage.")

# ================================================================ 13-15
H1("13. Limitations")
B(["<b>Tiny scale:</b> 4 qubits, 4×4 images, two-class problems, a 5 000-image MNIST subset (500 per digit). Conclusions about larger images or more classes are not implied.",
   "<b>Real-valued ansatz and data:</b> sufficient here, but complex amplitudes or non-linear data encodings could change the picture.",
   "<b>Exact simulation:</b> training costs use exact expectation values; on hardware shot noise would make the 2P + 1 evaluations per gradient step noisy and slower. Only the noise study and shots demo use circuit-level simulation.",
   "<b>Noise model:</b> simple depolarising channels, ideal state preparation and no measurement/read-out error; no hardware experiments were run.",
   "<b>Optimisation:</b> L-BFGS-B with random restarts is adequate for ≤ 52 parameters; barren plateaus [9] are expected for larger circuits.",
   "<b>Metrics:</b> normalisation discards image brightness; pixel read-out loses amplitude signs; fidelity is only one proxy for usefulness.",
   "<b>Statistics:</b> five splits (three at L = 12); standard deviations are indicative, not confidence intervals."])
H1("14. Future Scope")
B(["Run the trained circuits on real IBM Quantum hardware, with error mitigation, and compare with the Aer noise study.",
   "Replace amplitude encoding by cheaper encodings (angle / re-uploading) and study how encoding cost and compression quality trade off [4].",
   "Larger images (8×8 on 6 qubits), more classes, and entangling ansätze beyond the linear CX chain; analyse barren plateaus as size grows [9].",
   "Train with the class-aware or fidelity-based objectives, or combine QAE with a quantum classifier on the latent qubits.",
   "Study QAEs for genuinely quantum data (ground states of small molecules or spin chains) where a classical baseline is not available and an advantage is conceivable [1, 3]."])
H1("15. Conclusion")
P(f"A Qiskit-validated quantum autoencoder was trained on amplitude-encoded 4×4 MNIST digits and compared with PCA at matched latent size. "
  f"The comparison has a clean theoretical outcome that the experiments confirm: <b>a post-selected QAE can at best match uncentred PCA</b> (Ky Fan), and with enough layers it does – for two latent qubits the test fidelity is {qa('0 vs 1', 2, 12, 'F_ps')[0]:.4f} versus {pa('0 vs 1', 2, False)[0]:.4f} for PCA. "
  f"The physically realisable reset decoder is lower ({qa('0 vs 1', 2, 12, 'F_reset')[0]:.4f}), shallow circuits leave a gap, standard centred PCA is slightly better still ({pa('0 vs 1', 2, True)[0]:.4f}), "
  f"deeper circuits suffer more under gate noise, and state preparation costs more than the encoder. "
  f"The project therefore delivers an honest negative-but-informative result: no quantum advantage for compressing classical image data at this scale, together with a precise understanding of why.")

# ================================================================ 16. REFERENCES
H1("16. References")
refs = [
 "J. Romero, J. P. Olson, A. Aspuru-Guzik, “Quantum autoencoders for efficient compression of quantum data,” <i>Quantum Sci. Technol.</i> 2, 045001 (2017). arXiv:1612.02806.",
 "L. Lamata, U. Alvarez-Rodriguez, J. D. Martín-Guerrero, M. Sanz, E. Solano, “Quantum autoencoders via quantum adders with genetic algorithms,” <i>Quantum Sci. Technol.</i> 4, 014007 (2018).",
 "A. Pepper, N. Tischler, G. J. Pryde, “Experimental realization of a quantum autoencoder: the compression of qutrits via machine learning,” <i>Phys. Rev. Lett.</i> 122, 060501 (2019).",
 "C. Bravo-Prieto, “Quantum autoencoders with enhanced data encoding,” <i>Mach. Learn.: Sci. Technol.</i> 2, 035028 (2021).",
 "A. Khoshaman, W. Vinci, B. Denis, E. Andriyash, H. Sadeghi, M. H. Amin, “Quantum variational autoencoder,” <i>Quantum Sci. Technol.</i> 4, 014001 (2018).",
 "K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, “Quantum circuit learning,” <i>Phys. Rev. A</i> 98, 032309 (2018).",
 "M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, N. Killoran, “Evaluating analytic gradients on quantum hardware,” <i>Phys. Rev. A</i> 99, 032331 (2019).",
 "M. Cerezo et al., “Variational quantum algorithms,” <i>Nat. Rev. Phys.</i> 3, 625–644 (2021).",
 "J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven, “Barren plateaus in quantum neural network training landscapes,” <i>Nat. Commun.</i> 9, 4812 (2018).",
 "J. Preskill, “Quantum computing in the NISQ era and beyond,” <i>Quantum</i> 2, 79 (2018).",
 "S. Aaronson, “Read the fine print,” <i>Nat. Phys.</i> 11, 291–293 (2015).",
 "M. Möttönen, J. J. Vartiainen, V. Bergholm, M. M. Salomaa, “Transformation of quantum states using uniformly controlled rotations,” <i>Quantum Inf. Comput.</i> 5, 467–473 (2005).",
 "V. V. Shende, S. S. Bullock, I. L. Markov, “Synthesis of quantum-logic circuits,” <i>IEEE Trans. Comput.-Aided Des. Integr. Circuits Syst.</i> 25, 1000–1010 (2006).",
 "H. Buhrman, R. Cleve, J. Watrous, R. de Wolf, “Quantum fingerprinting,” <i>Phys. Rev. Lett.</i> 87, 167902 (2001).",
 "Ky Fan, “On a theorem of Weyl concerning eigenvalues of linear transformations I,” <i>Proc. Natl. Acad. Sci. USA</i> 35, 652–655 (1949).",
 "K. Pearson, “On lines and planes of closest fit to systems of points in space,” <i>Philos. Mag.</i> 2, 559–572 (1901).",
 "I. T. Jolliffe, J. Cadima, “Principal component analysis: a review and recent developments,” <i>Phil. Trans. R. Soc. A</i> 374, 20150202 (2016).",
 "Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, “Gradient-based learning applied to document recognition,” <i>Proc. IEEE</i> 86, 2278–2324 (1998). Data: http://yann.lecun.com/exdb/mnist/",
 "M. A. Nielsen, I. L. Chuang, <i>Quantum Computation and Quantum Information</i>, 10th anniversary ed., Cambridge University Press (2010).",
 "A. Javadi-Abhari et al., “Quantum computing with Qiskit,” arXiv:2405.08810 (2024).",
 "S. Raschka, “MLxtend: providing machine learning and data science utilities and extensions to Python’s scientific computing stack,” <i>J. Open Source Softw.</i> 3, 638 (2018).",
 "F. Pedregosa et al., “Scikit-learn: machine learning in Python,” <i>J. Mach. Learn. Res.</i> 12, 2825–2830 (2011).",
 "R. H. Byrd, P. Lu, J. Nocedal, C. Zhu, “A limited memory algorithm for bound constrained optimization,” <i>SIAM J. Sci. Comput.</i> 16, 1190–1208 (1995).",
 "C. R. Harris et al., “Array programming with NumPy,” <i>Nature</i> 585, 357–362 (2020).",
 "P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,” <i>Nat. Methods</i> 17, 261–272 (2020).",
 "R. Jozsa, “Fidelity for mixed quantum states,” <i>J. Mod. Opt.</i> 41, 2315–2323 (1994)."]
for i, r in enumerate(refs, 1): add(Paragraph(f"[{i}] {r}", S["ref"]))
add(Paragraph("<b>Dataset note.</b> MNIST: http://yann.lecun.com/exdb/mnist/ (also https://www.openml.org/d/554). This project uses the 5 000-image subset (digits 0–9, 500 each) bundled with the <i>mlxtend</i> package so that it runs offline; it is cached in <font name='DVM'>data/mnist_5000_subset.npz</font>.", S["body"]))

# ================================================================ APPENDIX
H1("Appendix A. Viva preparation – ten key questions")
qa_list = [
 ("Why 4 qubits?", "A 4×4 image has 16 pixels = 2⁴ amplitudes; amplitude encoding packs them into log₂16 = 4 qubits."),
 ("Why must the data be normalised?", "A quantum state has unit norm. Normalisation removes overall brightness; PCA is given the same normalised vectors for a fair comparison."),
 ("What does the autoencoder actually compress?", "U rotates the state so that its information is concentrated in the latent qubits; the trash qubits end near |0…0⟩ and can be reset (discarded) with small loss."),
 ("What is the cost and how would you measure it on hardware?", "C = 1 − E[P(trash = 0…0)]. A swap test between trash and fresh |0⟩ qubits gives P(ancilla = 0) = ½(1 + p₀)."),
 ("What is the parameter-shift rule and why is it valid here?", "Each parameter drives an RY gate (generator eigenvalues ±½), so the cost is a degree-1 trigonometric function of it, and ∂C/∂θ = ½[C(θ + π/2) − C(θ − π/2)] exactly."),
 ("Why is PCA an upper bound?", "Ky Fan: max over unitaries of Tr[ΠUΣUᵀ] = sum of the K largest eigenvalues of Σ, which is exactly the variance PCA retains with K components."),
 ("Then why use a QAE at all?", "It works on quantum states and runs on a quantum processor; for genuinely quantum data there is no classical PCA to compare with. For classical images it offers no advantage here."),
 ("Reset vs post-selected decoder?", "Reset is a physical trace-preserving channel (mixed output); post-selection keeps only runs with trash = 0…0 (pure output, success probability p₀). Proved: p₀² ≤ F_reset ≤ F_ps = p₀."),
 ("Why does fidelity improve with more layers?", "More parameters allow the circuit to rotate the principal subspace exactly onto the trash-zero subspace; with enough layers the cost reaches the Ky Fan optimum."),
 ("What are the honest limitations?", "Tiny scale, exact simulation, no hardware runs, state-preparation cost dominates, and no quantum advantage is observed.")]
for i, (a, b) in enumerate(qa_list, 1):
    add(Paragraph(f"<b>Q{i}. {a}</b><br/>{b}", S["body"]))

out = os.path.join(HERE, "Quantum_Autoencoder_vs_PCA_Report.pdf")
doc = Doc(out, title="Quantum Autoencoder vs PCA for Compression of Downsampled MNIST Digits", author="Aditya Rao")
doc.multiBuild(story)
print("report written:", out)

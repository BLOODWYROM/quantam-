# Quantum Autoencoder vs PCA for Compression of Downsampled MNIST Digits
*A simulator-based comparison – Qiskit · Qiskit Aer · NumPy · SciPy · Streamlit*

16 pixels of a 4×4 MNIST digit are amplitude-encoded into **4 qubits**; a trainable quantum autoencoder (Romero et al., 2017)
compresses them to **1, 2 or 3 latent qubits** and is compared with **PCA at matched latent size** (K = 2, 4, 8 components).
Digit pairs: **0 vs 1** (main) and **3 vs 8** (harder). Everything runs locally – no quantum hardware, no paid services.

## Headline results (5 stratified splits, test set)
| 0 vs 1, 2 latent qubits (PCA K = 4) | test fidelity |
|---|---|
| PCA centred (standard) | 0.970 |
| PCA uncentred (exactly matched baseline) | 0.955 |
| QAE post-selected, L = 12 layers | 0.956 |
| QAE reset decoder (trace-preserving), L = 12 | 0.926 |
| QAE post-selected, L = 1 layer | 0.841 |

* **Theory:** by Ky Fan's maximum principle a post-selected QAE can at best *match* uncentred PCA. Experiments confirm it: the training cost converges to the exact bound as depth grows.
* **Honest conclusion:** no quantum advantage – PCA is faster, simpler and at least as accurate. The QAE reproduces the optimal linear compressor, which validates the method.
* Extras: swap-test cost circuit, shot-based measurement, denoising, depolarising gate noise (Aer), cost-function ablation, downstream classification, resource counts, timing.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ (developed on 3.12, Qiskit 2.x, Qiskit Aer 0.18).

## Run
| What | Command | Time (1 CPU core) |
|---|---|---|
| Tests (26) | `pytest -q` | ~7 s |
| All experiments + figures + tables | `python run_all.py` | ~9 min |
| Quick smoke run (overwrites results/) | `python run_all.py --quick` | ~1–2 min |
| Re-draw figures only | `python run_all.py --figures-only` | ~1 min |
| Interactive dashboard | `streamlit run app.py` | instant |
| Notebook (executed copy included) | `jupyter notebook notebooks/QAE_vs_PCA.ipynb` or `python build_notebook.py` | ~30 s |
| PDF report (reads results/) | `python report/build_report.py` | ~15 s |

## Folder structure
```
qae/            data.py · ansatz.py (exact simulator) · circuits.py (Qiskit) · metrics.py · train.py · pca_baseline.py
                noise.py (Aer) · experiments.py · viz.py · worker.py (isolated Qiskit worker for the app)
tests/          test_qae.py – 26 tests incl. cross-validation against Qiskit/Aer
notebooks/      QAE_vs_PCA.ipynb (executed, step-by-step walkthrough)
app.py          Streamlit dashboard (train, compare, circuit, single-sample explorer, noise lab, theory)
run_all.py      reproduces every table and figure
results/        CSV tables, timing.json, trained_parameters.npz, figures/ (19 PNGs)
report/         build_report.py + Quantum_Autoencoder_vs_PCA_Report.pdf (21 pages)
data/           mnist_5000_subset.npz (cached) – see Dataset
docs/           VIVA_QA.md
```

## Dataset
MNIST (LeCun et al.): <http://yann.lecun.com/exdb/mnist/> · OpenML mirror <https://www.openml.org/d/554>.
The project uses 5 000 real MNIST training images (500 per digit) shipped inside the `mlxtend` package, cached in `data/mnist_5000_subset.npz`
(falls back to OpenML if neither is available). 28×28 → 4×4 by exact 7×7 block averaging → L2 normalisation.

## How it works (30-second version)
1. `|ψ⟩ = Σ xᵢ/‖x‖ |i⟩` – 16 amplitudes on 4 qubits.
2. Encoder U(θ): L layers of RY on each qubit + CX chain (real orthogonal 16×16).
3. Qubits k…3 are *trash*; cost `C = 1 − E[P(trash = 0…0)]` (measurable by swap test). Gradients by parameter-shift; L-BFGS-B, 5 restarts.
4. Decoder: reset trash to |0…0⟩ and apply U†. *Post-selected* variant keeps only runs with trash = 0…0 (fidelity = P(trash = 0)).
5. Compare with PCA (centred and uncentred) on the same normalised vectors: fidelity, pixel read-out MSE, downstream accuracy.

## Validation
The fast NumPy simulator is checked against Qiskit: unitary vs `Operator` (1e-16), reset-decoder density matrix vs Aer, swap-test circuit vs formula,
parameter-shift vs finite differences, `QAE ≤ PCA` on training data, `p₀² ≤ F_reset ≤ p₀`, noise-free Aer = ideal.

## Limitations
4 qubits / 4×4 images, exact simulation (no hardware runs), simple depolarising noise, five splits (three at L = 12). No quantum advantage is claimed.

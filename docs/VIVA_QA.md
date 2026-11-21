# Viva preparation – questions and short answers

**1. Why 4 qubits?** 16 pixels = 2⁴ amplitudes; amplitude encoding uses log₂16 = 4 qubits.

**2. Why normalise?** States must have unit norm. It removes brightness; PCA gets the same normalised vectors so the comparison is fair.

**3. What does the autoencoder compress?** U concentrates the information in the latent qubits so the trash qubits end near |0…0⟩ and can be reset with little loss.

**4. Cost, and how to measure it on hardware?** C = 1 − E[P(trash=0…0)]. Swap test between trash and fresh |0⟩ qubits: P(ancilla=0) = ½(1+p₀). Verified in `tests/` and the notebook.

**5. Parameter-shift rule?** Each parameter drives an RY gate (generator eigenvalues ±½) → cost is degree-1 trigonometric in θ → ∂C/∂θ = ½[C(θ+π/2) − C(θ−π/2)] exactly. Checked against finite differences (error ~1e-10).

**6. Why is PCA an upper bound?** P(trash=0) = Tr[ΠUΣUᵀ]; Ky Fan: its maximum over unitaries is the sum of the K largest eigenvalues of Σ = variance retained by K-component uncentred PCA. So mean train fidelity of a post-selected QAE ≤ PCA.

**7. Then why a QAE?** It acts on quantum states and runs on a quantum device; for genuinely quantum data there is no classical PCA. For classical images it gives no advantage here.

**8. Reset vs post-selected?** Reset = trace-preserving channel (mixed output); post-selection = pure output with success probability p₀. Proved p₀² ≤ F_reset ≤ F_ps = p₀.

**9. Why does fidelity rise with layers?** More parameters (4(L+1)) let the circuit rotate the principal subspace exactly onto the trash-zero subspace; the cost reaches the Ky Fan optimum (k=1 at L=6, k=2 at L≈12).

**10. Why is centred PCA better?** It stores a mean image (extra classical information); the circuit has none. The gap shrinks as K grows.

**11. What happens under noise?** Deeper circuits are better noise-free but degrade faster (depolarising noise on every RY/CX); at p₂ = 4 % depth 3 is worse than depth 1 or 2.

**12. Is state preparation a problem?** Yes: generic 4-qubit amplitude encoding costs more CX gates than the encoder, and grows exponentially – the usual data-loading caveat.

**13. Is there quantum advantage?** No. PCA is ~10⁶× faster here and at least as accurate. We never claim advantage.

**14. How do you know the simulator is right?** 26 tests, including comparison with Qiskit Operator/Statevector and Aer density-matrix simulation of the full circuit.

**15. Why a real-valued ansatz?** MNIST amplitudes are real and non-negative; RY+CX gates keep the state real, so a real orthogonal U suffices.

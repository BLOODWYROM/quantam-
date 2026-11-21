"""Qiskit circuits: encoder/decoder ansatz, full QAE circuit, shot-based run, SWAP-test cost circuit."""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from . import N_QUBITS


def ansatz_circuit(theta: np.ndarray, layers: int, n: int = N_QUBITS, name: str = "U(θ)") -> QuantumCircuit:
    th = np.asarray(theta).reshape(layers + 1, n)
    qc = QuantumCircuit(n, name=name)
    for l in range(layers):
        for q in range(n):
            qc.ry(float(th[l, q]), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        qc.barrier()
    for q in range(n):
        qc.ry(float(th[layers, q]), q)
    return qc


def trash_qubits(n_latent: int, n: int = N_QUBITS) -> list[int]:
    return list(range(n_latent, n))


def qae_circuit(psi: np.ndarray, theta: np.ndarray, layers: int, n_latent: int,
                measure: bool = False, save_dm: bool = False, ideal_prep: bool = False) -> QuantumCircuit:
    """|0> -> amplitude-encode psi -> U -> reset trash -> U^dagger [-> measure / save density matrix].

    ideal_prep=True uses Aer's set_density_matrix (noise-free initialisation, used for noise studies);
    otherwise a StatePreparation gate (a real gate sequence) is used.
    """
    n = N_QUBITS
    qc = QuantumCircuit(n)
    if ideal_prep:
        import qiskit_aer.library  # noqa: F401  (adds set_density_matrix)
        from qiskit.quantum_info import DensityMatrix
        qc.set_density_matrix(DensityMatrix(np.asarray(psi, dtype=complex)))
    else:
        qc.append(StatePreparation(np.asarray(psi, dtype=complex)), range(n))
    qc.barrier()
    U = ansatz_circuit(theta, layers)
    qc.compose(U, inplace=True)
    qc.barrier()
    for q in trash_qubits(n_latent):
        qc.reset(q)
    qc.barrier()
    qc.compose(U.inverse(), inplace=True)
    if save_dm:
        import qiskit_aer.library  # noqa: F401
        qc.save_density_matrix(label="rho")
    if measure:
        qc.measure_all()
    return qc


def encoder_only_circuit(psi, theta, layers) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    qc.append(StatePreparation(np.asarray(psi, dtype=complex)), range(N_QUBITS))
    qc.compose(ansatz_circuit(theta, layers), inplace=True)
    return qc


def swap_test_circuit(psi, theta, layers, n_latent) -> QuantumCircuit:
    """Swap test between each trash qubit and a fresh |0> reference qubit.

    P(ancilla = 0) = (1 + Tr[rho_trash |0..0><0..0|]) / 2   =>   p0 = 2 P(anc=0) - 1.
    Layout: qubits 0..3 data, 4..4+nt-1 references, last qubit = ancilla.
    """
    nt = N_QUBITS - n_latent
    data = QuantumRegister(N_QUBITS, "q")
    ref = QuantumRegister(nt, "ref")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(data, ref, anc)
    qc.append(StatePreparation(np.asarray(psi, dtype=complex)), data)
    qc.compose(ansatz_circuit(theta, layers), qubits=list(range(N_QUBITS)), inplace=True)
    qc.barrier()
    qc.h(anc[0])
    for i, tq in enumerate(trash_qubits(n_latent)):
        qc.cswap(anc[0], data[tq], ref[i])
    qc.h(anc[0])
    return qc


def swap_test_p0(psi, theta, layers, n_latent) -> float:
    """Exact (statevector) value of p0 obtained through the swap-test circuit."""
    qc = swap_test_circuit(psi, theta, layers, n_latent)
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc)
    p_anc0 = sv.probabilities([qc.num_qubits - 1])[0]
    return 2 * p_anc0 - 1


def resource_counts(layers: int) -> dict:
    """Gate/depth counts after transpiling to {ry, rz, cx, ...} for encoder+decoder and for state preparation."""
    from qiskit import transpile
    rng = np.random.default_rng(0)
    psi = np.abs(rng.normal(size=16)); psi /= np.linalg.norm(psi)
    th = rng.uniform(0, 2 * np.pi, (layers + 1) * N_QUBITS)
    basis = ["ry", "rz", "sx", "x", "cx"]
    prep = QuantumCircuit(N_QUBITS); prep.append(StatePreparation(psi), range(N_QUBITS))
    prep_t = transpile(prep, basis_gates=basis, optimization_level=1)
    enc = transpile(ansatz_circuit(th, layers), basis_gates=basis, optimization_level=1)
    return dict(layers=layers,
                encoder_cx=enc.count_ops().get("cx", 0), encoder_depth=enc.depth(),
                stateprep_cx=prep_t.count_ops().get("cx", 0), stateprep_depth=prep_t.depth(),
                n_parameters=(layers + 1) * N_QUBITS)

"""Isolated Qiskit worker (used by the Streamlit app so that Qiskit/Aer native code never runs inside
Streamlit's script threads).   echo '<json>' | python -m qae.worker
Commands: circuit -> PNG (base64), resources -> dict, noise -> list of fidelities."""
import base64, io, json, sys
import numpy as np


def main():
    req = json.load(sys.stdin)
    cmd = req["cmd"]
    if cmd == "circuit":
        import matplotlib; matplotlib.use("Agg")
        from qae.circuits import ansatz_circuit
        L = req["layers"]
        fig = ansatz_circuit(np.array(req["theta"]), L).draw("mpl", fold=-1 if L <= 3 else 40, scale=0.6,
                                                             style={"backgroundcolor": "white"})
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        out = base64.b64encode(buf.getvalue()).decode()
    elif cmd == "resources":
        from qae.circuits import resource_counts
        out = {k: int(v) for k, v in resource_counts(req["layers"]).items()}
    elif cmd == "noise":
        from qae.noise import noisy_fidelity
        out = noisy_fidelity(np.array(req["psi"]), np.array(req["theta"]), req["layers"], req["k"], req["p1"], req["p2"]).tolist()
    else:
        raise SystemExit(f"unknown command {cmd}")
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()

"""Run every experiment, save tables + figures into results/.

    python run_all.py                 full run (about 10 min on one CPU core)
    python run_all.py --quick         2 seeds, 3 restarts, no deep circuits (about 1 min)
    python run_all.py --figures-only  re-draw figures from results/_cache.pkl
"""
import argparse, json, os, pickle, time
import numpy as np
import pandas as pd
from qae import experiments as ex

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--figures-only", action="store_true")
args = ap.parse_args()
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(os.path.join(R, "figures"), exist_ok=True)
CACHE = os.path.join(R, "_cache.pkl")
t0 = time.time()

if args.figures_only:
    D = pickle.load(open(CACHE, "rb"))
else:
    seeds = (0, 1) if args.quick else (0, 1, 2, 3, 4)
    restarts = 3 if args.quick else 5
    print("[1/6] depth x latent sweep (QAE vs PCA)", flush=True)
    q, p, models = ex.run_sweep(seeds=seeds, restarts=restarts)
    if not args.quick:   # deep circuits (L=12) on 3 splits of 0-vs-1: convergence to the PCA / Ky-Fan ceiling
        qd, _, md = ex.run_sweep(pairs=[(0, 1)], seeds=(0, 1, 2), layers=ex.DEEP_LAYERS, restarts=5, with_pca=False)
        q = pd.concat([q, qd], ignore_index=True); models.update(md)
    print("[2/6] PCA component curve", flush=True); pc = ex.pca_curve(seeds=seeds)
    print("[3/6] cost-function ablation", flush=True); ab = ex.cost_ablation(seeds=seeds[:3], restarts=max(2, restarts // 2))
    print("[4/6] denoising", flush=True); dn = ex.denoising(models)
    print("[5/6] Aer gate-noise study", flush=True); gn, pca_ref = ex.gate_noise(models, layers=(1, 2, 3))
    print("[6/6] timing & resources", flush=True); tm = ex.timing(models); rs = ex.resources()
    D = dict(q=q, p=p, pc=pc, ab=ab, dn=dn, gn=gn, pca_ref=pca_ref, tm=tm, rs=rs, models=models)
    pickle.dump(D, open(CACHE, "wb"))
    for name in ("q", "p", "pc", "ab", "dn", "gn", "rs"):
        D[name].to_csv(f"{R}/{ {'q':'qae_sweep','p':'pca_baseline','pc':'pca_curve','ab':'cost_ablation','dn':'denoising','gn':'gate_noise','rs':'resources'}[name] }.csv", index=False)
    json.dump({k: float(v) for k, v in tm.items()}, open(f"{R}/timing.json", "w"), indent=2)
    np.savez(f"{R}/trained_parameters.npz", **{"|".join(map(str, k)): v["theta"] for k, v in models.items()})
    print(f"computation finished in {time.time() - t0:.0f}s", flush=True)

from qae import viz
viz.make_all(D, R)
print(f"all done in {time.time() - t0:.0f}s -> {R}")

"""Per-step diagnostics of guided generation: average number of UNIQUE
candidate blocks (out of n_is) and ESS, as a function of denoising progress t
(t in (0,1]; t->0 = near-clean end). Both kernels, several block sizes.

Produces one 2x2 figure: rows = {mask, graph}, cols = {unique count, ESS}.

  python eval_uniq_diag.py -blocks 4,16,64 -n 120
"""
import argparse
import pickle
from os.path import join

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT_SEED = 777

def run_one(kernel, blk, real, A_dev, deg_ratio, n_is, device):
    ck = f"./sets_model/BD_porto_v3_normal_{kernel}_blk{blk}_v2_bd.pth"
    dk = (f"./sets_disc/BDdisc_f0.05_p1_e0_model_blk{blk}.pth" if kernel == "mask"
          else f"./sets_disc/BDdisc_f0.05_p1_e0_model_graph{blk}.pth")
    model = torch.load(ck, map_location=device); model.eval()
    disc = torch.load(dk, map_location=device); disc.eval()
    torch.manual_seed(7); np.random.seed(7)
    diag = []
    o = [p[0] for p in real]; d = [p[-1] for p in real]
    model.plan_guided(o, d, disc, A_dev, deg_ratio, n_is=n_is,
                      diag_log=diag, adj_prop=False)
    t = np.array([e["t"] for e in diag])
    uq = np.array([e["uniq"] for e in diag])
    es = np.array([e["ess"] for e in diag])
    ok = np.isfinite(t) & np.isfinite(uq)
    return t[ok], uq[ok], es[ok]

def binned(t, y, nb=25):
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
    xs, ys = [], []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 1:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            ys.append(float(np.mean(y[m])))
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="4,16,64")
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-out", type=str, default="./sets_res/uniq_diag.png")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    porto, fam = "./porto_data", "0.05"
    A_exc = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    sp_exc = pickle.load(open(join(porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    deg_ratio = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
    A_dev = A_exc.float().to(device)
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:2000] if len(sp_exc[i]) >= 2][:args.n]
    blocks = [int(b) for b in args.blocks.split(",")]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    colors = {4: "tab:blue", 8: "tab:green", 16: "tab:orange", 32: "tab:purple", 64: "tab:red"}
    data = {}
    for r, kernel in enumerate(["mask", "graph"]):
        for blk in blocks:
            t, uq, es = run_one(kernel, blk, real, A_dev, deg_ratio, args.n_is, device)
            xs_u, ys_u = binned(t, uq); xs_e, ys_e = binned(t, es)
            c = colors.get(blk, None)
            ax[r, 0].plot(xs_u, ys_u, "-o", ms=3, color=c, label=f"blk{blk}")
            ax[r, 1].plot(xs_e, ys_e, "-o", ms=3, color=c, label=f"blk{blk}")
            data[f"{kernel}_blk{blk}"] = {"t": t.tolist(), "uniq": uq.tolist(), "ess": es.tolist()}
            # ys_*[0] is the smallest-t bin (t~0 = clean END);
            # ys_*[-1] is the largest-t bin (t~1 = noisy START).
            print(f"[{kernel} blk{blk}] steps={len(t)} "
                  f"uniq@t~1(start)={ys_u[-1]:.1f} uniq@t~0(end)={ys_u[0]:.1f} "
                  f"ess@t~1(start)={ys_e[-1]:.1f} ess@t~0(end)={ys_e[0]:.1f}", flush=True)
        ax[r, 0].set_title(f"{kernel}: avg #unique candidates (of {args.n_is})")
        ax[r, 1].set_title(f"{kernel}: ESS")
        for c in range(2):
            ax[r, c].set_xlabel("denoising progress  (t → 0 = clean end)")
            ax[r, c].invert_xaxis()   # so the clean end (t->0) is on the RIGHT->left? keep t axis
            ax[r, c].grid(alpha=0.3); ax[r, c].legend(fontsize=8)
        ax[r, 0].set_ylim(0, args.n_is + 3)
        ax[r, 1].set_ylim(0, args.n_is + 3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    import json
    json.dump(data, open(args.out.replace(".png", ".json"), "w"))
    print("UNIQ_DIAG_DONE " + args.out, flush=True)

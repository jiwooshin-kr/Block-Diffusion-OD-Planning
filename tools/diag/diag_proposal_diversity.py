"""
Why is the importance-weighting ESS almost equal to n_is? Two candidate causes with
opposite fixes, so measure which one it is.

  python tools/diag/diag_proposal_diversity.py -variant LE -blocks 1,2,4,8,16,64 -n 100

  (a) the proposal has no diversity -- the n_is candidate blocks collapse onto a few
      distinct values, so there is nothing for the discriminator to rank. With
      adjacency masking the reveal position is restricted to the scenario-legal
      successors of the revealed left neighbour, i.e. about deg + 2 = 5 tokens on
      this graph (mean degree 2.88, plus END/PAD), which caps the number of distinct
      candidates far below n_is = 100.
  (b) the candidates are diverse but the discriminator cannot separate them. Its
      training task ("except-scenario data vs unconditional model pool") is easier
      than its deployment task ("rank 100 candidate blocks that the same p_theta
      produced at the same reveal step"), so it may be near-blind locally even with
      a healthy global val accuracy (measured 0.69-0.73, logit_std ~2.0 against a
      bound of 4.0).

Reported per (block, adj_prop):
  uniq     mean number of DISTINCT candidate blocks out of n_is
  ess      mean effective sample size of the importance weights
  legal    mean number of scenario-legal successors at the reveal position
           (the structural cap on uniq when adj_prop is on)
  w_cv     implied coefficient of variation of the weights, from ESS/n = 1/(1+CV^2)

Reading it: uniq << n_is points at (a); uniq large with ess ~ n_is points at (b).
The fix differs -- (a) needs a better/enumerated proposal, (b) needs the
discriminator trained on the deployment task.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import json
import pickle
from os.path import join

import numpy as np
import torch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,64")
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-n", type=int, default=100, help="OD pairs to plan")
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-nis_list", type=str, default="",
                    help="sweep n_is too, e.g. '10,30,50,100,150'. Overrides -n_is. "
                         "This is the structural half of the n_is sweep: it shows the "
                         "distinct-candidate count saturating while n_is grows.")
    ap.add_argument("-adj_prop", type=str, default="0,1",
                    help="comma list of adjacency-masking settings to run")
    ap.add_argument("-out", type=str, default="", help="write the rows to JSON")
    ap.add_argument("-seed", type=int, default=7)
    args = ap.parse_args()

    V = args.variant
    fam = f"{V}1.0-0.05"
    dev = torch.device("cuda:0")
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v6-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_v6-{fam}_normal.ts"), "rb")).bool()
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_v6-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(777).permutation(len(sp))
    real = [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:args.n]
    deg_ratio = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(dev)
    A_dev = A_exc.float().to(dev)
    # structural cap: legal successors per vertex, on the scenario graph
    outdeg = A_exc.float().sum(1)
    print(f"[{V}] scenario graph out-degree: mean {outdeg.mean():.2f}  "
          f"median {outdeg.median():.0f}  max {outdeg.max():.0f}")
    print(f"     -> with adjacency masking the reveal position has about "
          f"{outdeg.mean():.1f} legal vertices (+ END/PAD)\n")

    nis_list = ([int(x) for x in args.nis_list.split(",")] if args.nis_list
                else [args.n_is])
    adjps = [bool(int(x)) for x in args.adj_prop.split(",")]
    out_rows = []

    print(f"{'blk':>4} {'n_is':>6} {'adj_prop':>9} {'uniq':>7} {'/n_is':>7} {'ess':>7} "
          f"{'ess/n':>7} {'w_cv':>7} {'steps':>7}")
    for blk in [int(x) for x in args.blocks.split(",")]:
        ck = f"sets_v6/{V}/model/BD_v6{V}_normal_mask_blk{blk}_bd.pth"
        dsc = f"sets_v6/{V}/disc/BDdisc_f{fam}_p1_e0_model_blk{blk}.pth"
        model = torch.load(ck, map_location=dev)
        model.eval()
        disc = torch.load(dsc, map_location=dev)
        disc.eval()
        for nis in nis_list:
            for adjp in adjps:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                diag = []
                model.plan_guided([p[0] for p in real], [p[-1] for p in real], disc,
                                  A_dev, deg_ratio, n_is=nis, w_gamma=1.0,
                                  order="l2r", adj_prop=adjp, diag_log=diag)
                if not diag:
                    print(f"{blk:>4} {nis:>6} {str(adjp):>9}   (no diag records)")
                    continue
                u = float(np.nanmean([d["uniq"] for d in diag]))
                e = float(np.nanmean([d["ess"] for d in diag]))
                cv = np.sqrt(max(nis / e - 1.0, 0.0)) if e > 0 else float("nan")
                print(f"{blk:>4} {nis:>6} {str(adjp):>9} {u:>7.2f} {u / nis:>7.3f} "
                      f"{e:>7.2f} {e / nis:>7.3f} {cv:>7.3f} {len(diag):>7}", flush=True)
                out_rows.append({"variant": V, "blk": blk, "n_is": nis,
                                 "adj_prop": adjp, "uniq": u, "uniq_frac": u / nis,
                                 "ess": e, "ess_frac": e / nis, "w_cv": float(cv),
                                 "steps": len(diag), "n_pairs": len(real),
                                 "outdeg_mean": float(outdeg.mean())})
        del model, disc
        torch.cuda.empty_cache()

    if args.out:
        json.dump(out_rows, open(args.out, "w"), indent=2)
        print(f"written {args.out}  ({len(out_rows)} rows)")

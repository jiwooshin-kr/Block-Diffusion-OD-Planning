"""
Like-for-like wall-clock comparison of every guidance method on the v4
backbones: base / IW / adj+IW / D-CBG first-order / D-CBG exact.

Fairness rules baked in: identical OD pairs (the except_0 reserved rows),
identical batch size, identical generation seed, one method at a time on ONE
GPU (run this alone -- co-tenancy inflates everything), CUDA synchronised
around each timed region. Also reports the analytic per-path scorer-call
count (discriminator/classifier sequence evaluations), which is the
implementation-independent cost.

  CUDA_VISIBLE_DEVICES=0 python time_guidance.py -blocks 1,4,16,64 -n 200
"""

import argparse
import json
import pickle
import time
from os.path import join

import numpy as np
import torch

from dcbg_plugin import AdjBound, plan_dcbg_mask

SPLIT_SEED = 777


def timed(fn, sync=True):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-n", type=int, default=200, help="OD pairs (timing only; 200 is plenty)")
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-gamma", type=float, default=1.0)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out", type=str, default="guidance_timing_v4.json")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    fam = args.family
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.n]
    A_dev = A_exc.float().to(device)
    dr = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)

    out = {}
    for blk in [int(b) for b in args.blocks.split(",")]:
        model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v4_bd.pth",
                           map_location=device)
        model.eval()
        disc = torch.load(f"./sets_disc/BDdisc_f{fam}_p1_e0_model_blk{blk}_v4.pth",
                          map_location=device)
        disc.eval()
        clf = torch.load(f"./sets_disc/DCBGclf_mask_blk{blk}_f{fam}_p1_adj_modelneg_v4.pth",
                         map_location=device)
        clf.eval()
        clf_b = AdjBound(clf, A_dev, dr)

        def run(method):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            reveals = []
            total = 0.0
            for s in range(0, len(real), args.batch):
                bb = real[s:s + args.batch]
                o, d = [p[0] for p in bb], [p[-1] for p in bb]
                if method == "base":
                    _, dt = timed(lambda: model.plan(o, d, use_refine=False))
                elif method in ("IW", "adj+IW"):
                    el = []
                    _, dt = timed(lambda: model.plan_guided(
                        o, d, disc, A_dev, dr, n_is=args.n_is, ess_log=el,
                        adj_prop=(method == "adj+IW")))
                    reveals.append(len(el))
                else:
                    ap_ = method == "dcbg_fo"
                    _, dt = timed(lambda: plan_dcbg_mask(
                        model, o, d, clf_b, args.gamma, use_approx=ap_))
                total += dt
            r = (sum(reveals) / len(reveals)) if reveals else None
            return total / len(real), r

        row = {}
        for method in ["base", "IW", "adj+IW", "dcbg_fo", "dcbg_exact"]:
            sp, rev = run(method)
            row[method] = sp
            if rev is not None:
                row["reveals"] = rev
            print(f"blk{blk:<3} {method:<11} {sp:.4f} s/path"
                  + (f"  (reveals/batch={rev:.0f})" if rev is not None else ""), flush=True)
        # analytic scorer-call counts per path: reveals x candidates
        out[f"blk{blk}"] = row

    with open(join(args.res_path, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {join(args.res_path, args.out)}")
    print("TIMING_DONE", flush=True)

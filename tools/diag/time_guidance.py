"""
Fair wall-clock comparison of every guidance method on the v4 backbones:
base / IW / adj+IW / D-CBG first-order / D-CBG exact.

WHAT THE TIMER MEASURES (exactly one thing):
    one call of the planner on a batch of `-batch` OD pairs, from handing in
    the (origin, destination) lists to receiving the finished vertex paths.
    INSIDE  : denoiser forward passes, guidance scoring (discriminator /
              classifier), candidate sampling, per-reveal bookkeeping, the
              planner's own CPU sync + stop scan, path assembly.
    OUTSIDE : checkpoint/data loading, post-processing (P1/P3), EM/PC/DTW
              scoring, and anything else after the planner returns.
    Reported as seconds per path = (batch wall-clock) / batch — the amortised
    cost when `-batch` paths are generated concurrently, NOT the latency of
    generating one path on its own.

FAIRNESS CONTROLS:
  - identical OD pairs, identical batch size, identical seed for every method
  - one method at a time, sequentially, on one GPU; the script refuses to run
    if another process is resident on that GPU (-force to override)
  - a warm-up batch per (block, method) is executed and DISCARDED, so CUDA
    context creation and kernel autotuning never land in a reported number
  - every timed region is wrapped in torch.cuda.synchronize()
  - `-repeats` timed batches per cell, each a DIFFERENT slice of the reserved
    pairs (pairs 1-100, 101-200, ...), identical across methods, so the
    reported std reflects batch composition as well as run-to-run jitter
    (the planner runs until the whole batch stops, so the longest path in a
    batch drives its wall-clock)

  CUDA_VISIBLE_DEVICES=0 python tools/diag/time_guidance.py -blocks 1,2,4,8,16,32,64
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse
import json
import pickle
import subprocess
import time
from os.path import join

import numpy as np
import torch

from dcbg_plugin import AdjBound, plan_dcbg_mask

SPLIT_SEED = 777
METHODS = ["base", "IW", "adj+IW", "dcbg_fo", "dcbg_exact"]


def gpu_is_exclusive():
    """True when no compute process is resident on the visible GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
    except Exception:
        return True, "nvidia-smi unavailable"
    lines = [l for l in out.strip().splitlines() if l.strip()]
    return (len(lines) == 0), (out.strip() or "none")


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-repeats", type=int, default=3, help="timed batches per cell (after warm-up)")
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-gamma", type=float, default=1.0)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-force", type=int, default=0, help="1 = run even if the GPU is shared")
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out", type=str, default="guidance_timing_v4.json")
    args = ap.parse_args()

    excl, who = gpu_is_exclusive()
    print(f"[gpu] exclusive={excl} | resident compute apps: {who}", flush=True)
    if not excl and not args.force:
        raise SystemExit("GPU is not exclusive — refusing to produce contaminated timings (-force 1 to override)")

    device = torch.device("cuda:0")
    fam = args.family
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    need = args.batch * args.repeats
    pool = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:need]
    assert len(pool) == need, f"need {need} reserved pairs, have {len(pool)}"
    batches = [pool[k * args.batch:(k + 1) * args.batch] for k in range(args.repeats)]
    real = batches[0]
    A_dev = A_exc.float().to(device)
    dr = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
    OD = [([p[0] for p in b], [p[-1] for p in b]) for b in batches]
    print(f"[setup] batch={args.batch} x {args.repeats} distinct batches "
          f"(+1 discarded warm-up per cell, on batch 0) n_is={args.n_is} gamma={args.gamma}", flush=True)

    out = {"_meta": {"batch": args.batch, "repeats": args.repeats, "n_is": args.n_is,
                     "gamma": args.gamma, "seed": args.seed, "family": fam,
                     "measures": "one planner call over the batch; excludes loading, "
                                 "post-processing and scoring"}}
    for blk in [int(b) for b in args.blocks.split(",")]:
        model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v4_bd.pth", map_location=device)
        model.eval()
        disc = torch.load(f"./sets_disc/BDdisc_f{fam}_p1_e0_model_blk{blk}_v4.pth", map_location=device)
        disc.eval()
        clf = torch.load(f"./sets_disc/DCBGclf_mask_blk{blk}_f{fam}_p1_adj_modelneg_v4.pth", map_location=device)
        clf.eval()
        clf_b = AdjBound(clf, A_dev, dr)

        def call(method, k=0, ess_log=None):
            O, D = OD[k]
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if method == "base":
                return model.plan(O, D, use_refine=False)
            if method in ("IW", "adj+IW"):
                return model.plan_guided(O, D, disc, A_dev, dr, n_is=args.n_is,
                                         ess_log=ess_log, adj_prop=(method == "adj+IW"))
            return plan_dcbg_mask(model, O, D, clf_b, args.gamma,
                                  use_approx=(method == "dcbg_fo"))

        row = {}
        for method in METHODS:
            timed(lambda: call(method, 0))                   # warm-up, discarded
            ts = [timed(lambda: call(method, k)) for k in range(args.repeats)]
            per = np.array(ts) / args.batch
            el = []
            if method in ("IW", "adj+IW"):
                call(method, 0, ess_log=el)
                row[f"{method}_reveals"] = len(el)
            row[method] = float(per.mean())
            row[f"{method}_std"] = float(per.std())
            row[f"{method}_batch_sec"] = [float(x) for x in ts]
            print(f"blk{blk:<3} {method:<11} {per.mean():.4f} ± {per.std():.4f} s/path"
                  + f"   [batches: {', '.join(f'{x:.3f}s' for x in ts)}]"
                  + (f"   (reveals={len(el)})" if el else ""), flush=True)
        out[f"blk{blk}"] = row

    with open(join(args.res_path, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {join(args.res_path, args.out)}")
    print("TIMING_DONE", flush=True)

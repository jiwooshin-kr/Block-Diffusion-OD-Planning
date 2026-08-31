"""
Unconditional-generation distribution match using the PROJECT'S OWN metric
battery (utils/evaluate_new.py :: Evaluator.calculate_divergences and
calculate_rank_correlation), so the numbers are directly comparable with the
paper/baseline convention rather than an ad-hoc reimplementation:

  JSEV  Jensen-Shannon divergence between the real and generated
        directed-edge frequency distributions over the full V x V grid
        (scipy rel_entr -> natural log, i.e. NATS; divide by ln 2 for bits)
  RMSE / MAE / R2   on the same flattened edge distributions
  Spearman rho      rank correlation of edge counts (all cells, and masked to
                    cells with a nonzero real count)

Run per block over the stored v4 unconditional pools; the real reference is
the full NORMAL shortest-path set. Optionally restrict to valid generations
(-valid_only 1) to separate "wrong edges" from "wrong edge frequencies".

  python tools/eval/eval_uncond_jsev.py -pool_pat "./sets_disc/uncond_pool_v4_blk{B}.pth"
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse
import json
import pickle
from os.path import join

import numpy as np
import torch

from utils.evaluate_new import Evaluator

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-pool_pat", type=str, default="./sets_disc/uncond_pool_v4_blk{B}.pth")
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-norm_ver", type=str, default="v3-0.05")
    ap.add_argument("-n_real", type=int, default=0, help="0 = use every real path")
    ap.add_argument("-valid_only", type=int, default=0)
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out", type=str, default="uncond_jsev_v4.json")
    args = ap.parse_args()

    A = pickle.load(open(join(args.porto, f"porto_shrink_A_{args.norm_ver}_normal.ts"), "rb")).bool()
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_{args.norm_ver}_normal.pkl"), "rb"))
    real = [list(map(int, p)) for p in sp if len(p) >= 2]
    if args.n_real:
        real = real[:args.n_real]
    n_vertex = A.shape[0]
    rl = np.array([len(p) for p in real])
    print(f"[real] n={len(real)} len mean={rl.mean():.2f} std={rl.std():.2f} "
          f"median={np.median(rl):.0f} p5={np.percentile(rl,5):.0f} p95={np.percentile(rl,95):.0f}",
          flush=True)

    out = {"real": {"n": len(real), "len_mean": float(rl.mean()), "len_std": float(rl.std()),
                    "len_median": float(np.median(rl)),
                    "len_p5": float(np.percentile(rl, 5)), "len_p95": float(np.percentile(rl, 95))}}
    for B in [int(b) for b in args.blocks.split(",")]:
        gen = [list(map(int, p)) for p in torch.load(args.pool_pat.format(B=B))["paths"] if len(p) >= 2]
        n_all = len(gen)
        if args.valid_only:
            gen = [p for p in gen if all(A[u, v] for u, v in zip(p[:-1], p[1:]))]
        gl = np.array([len(p) for p in gen])
        ev = Evaluator(real_paths=real, gen_paths=gen, model=None, n_vertex=n_vertex,
                       dataset=None, name=f"uncond_blk{B}")
        div = ev.calculate_divergences()
        rank = ev.calculate_rank_correlation()
        row = {"n_generated": n_all, "n_scored": len(gen),
               "len_mean": float(gl.mean()), "len_std": float(gl.std()),
               "len_median": float(np.median(gl)),
               "len_p5": float(np.percentile(gl, 5)), "len_p95": float(np.percentile(gl, 95))}
        row.update({k: float(v) for k, v in div.items()})
        row.update({k: float(v) for k, v in rank.items() if isinstance(v, (int, float))})
        out[f"blk{B}"] = row
        print(f"blk{B:<3} n={len(gen)} len={row['len_mean']:.2f}±{row['len_std']:.2f} "
              f"JSEV={row['JSEV']:.4f} nats ({row['JSEV']/np.log(2):.4f} bits) "
              f"RMSE={row['RMSE']:.2e} MAE={row['MAE']:.2e} R2={row['R2']:.4f}", flush=True)

    sfx = "_validonly" if args.valid_only else ""
    p = join(args.res_path, args.out.replace(".json", f"{sfx}.json"))
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {p}")
    print("UNCOND_JSEV_DONE", flush=True)

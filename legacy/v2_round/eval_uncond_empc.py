"""
EM/PC for UNCONDITIONAL generations: each generated path's own endpoints are
taken as its OD (evaluate_em_pc already works this way), scored against the
NORMAL world. Since training paths are shortest paths, uncond EM = "is the
sample a shortest path between its own endpoints" -- a distribution-fidelity
metric that needs no given OD.

Also joins novelty (verbatim membership in the real set) so the synthetic-data
currency valid&novel and novel&EM are reported. EM/PC are averaged over the
COVERED subset (generated ODs present in the reference SP data); coverage is
reported separately.

  python legacy/v2_round/eval_uncond_empc.py -pool_pat "./sets_disc/uncond_pool_v4_blk{B}.pth" -out_sfx _v4
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

from eval_shortest import evaluate_em_pc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-pool_pat", type=str, default="./sets_disc/uncond_pool_v4_blk{B}.pth")
    ap.add_argument("-n_eval", type=int, default=3000)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-norm_ver", type=str, default="v3-0.05")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out_sfx", type=str, default="_v4")
    args = ap.parse_args()

    A = pickle.load(open(join(args.porto, f"porto_shrink_A_{args.norm_ver}_normal.ts"), "rb")).bool().float()
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_{args.norm_ver}_normal.pkl"), "rb"))
    real_hash = set(hash(tuple(map(int, p))) for p in sp if len(p) >= 2)
    rng = np.random.RandomState(args.seed)

    out = {}
    for B in [int(b) for b in args.blocks.split(",")]:
        P = [list(map(int, p)) for p in torch.load(args.pool_pat.format(B=B))["paths"] if len(p) >= 2]
        idx = rng.choice(len(P), min(args.n_eval, len(P)), replace=False)
        sub = [P[i] for i in idx]
        summ, records, _ = evaluate_em_pc(
            gen_paths=sub, A=A, shortest_paths=sp,
            save_dir=join(args.res_path, "em_pc"), prefix=f"v4uncond_blk{B}")
        novel = np.array([hash(tuple(p)) not in real_hash for p in sub], dtype=float)
        valid = np.array([int(r["valid"]) for r in records], dtype=float)
        em = np.array([int(r["em"]) for r in records], dtype=float)
        pc = np.array([float(r["pc"]) for r in records], dtype=float)
        cov = np.array([r["shortest_len"] is not None for r in records], dtype=bool)
        row = {
            "n": len(sub),
            "valid": float(valid.mean()),
            "novel": float(novel.mean()),
            "valid_novel": float((valid * novel).mean()),
            "od_coverage": float(cov.mean()),
            "em_cov": float(em[cov].mean()) if cov.any() else float("nan"),
            "pc_cov": float(pc[cov].mean()) if cov.any() else float("nan"),
            "novel_em_cov": float((em * novel)[cov].mean()) if cov.any() else float("nan"),
        }
        out[f"blk{B}"] = row
        print(f"blk{B:<3} valid={row['valid']:.3f} novel={row['novel']:.3f} "
              f"valid&novel={row['valid_novel']:.3f} | cov={row['od_coverage']:.3f} "
              f"EM|cov={row['em_cov']:.3f} PC|cov={row['pc_cov']:.3f} "
              f"novel&EM|cov={row['novel_em_cov']:.3f}", flush=True)

    with open(join(args.res_path, f"uncond_empc{args.out_sfx}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("UNCOND_EMPC_DONE", flush=True)

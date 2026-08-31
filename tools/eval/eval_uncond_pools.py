"""
Unconditional-generation quality per block backbone, scored on the NORMAL
world (the training distribution). Reuses the stored seed-1 uncond pools
(sets_disc/uncond_pool_blk{B}.pth, 20k open-ended generations each) -- pure
CPU, no GPU needed.

Metrics per block:
  valid        : fraction of paths with every edge legal in A_normal
  invE         : invalid-edge rate over all transitions
  len mean/std : generated length stats (real reference printed once)
  len KS       : Kolmogorov-Smirnov distance between generated and real
                 length distributions
  cap%         : fraction hitting the canvas cap (len >= cap) = failed to stop
  uniq         : fraction of unique paths within the pool
  novel        : fraction not appearing verbatim in the real normal set
  edge JSD     : Jensen-Shannon divergence between generated and real
                 directed-edge usage distributions

  python tools/eval/eval_uncond_pools.py
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse
import json
import pickle
from collections import Counter
from os.path import join

import numpy as np
import torch


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def jsd(p_cnt, q_cnt):
    keys = set(p_cnt) | set(q_cnt)
    p = np.array([p_cnt.get(k, 0) for k in keys], dtype=float)
    q = np.array([q_cnt.get(k, 0) for k in keys], dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-pool_pat", type=str, default="./sets_disc/uncond_pool_blk{B}.pth")
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-norm_ver", type=str, default="v3-0.05",
                    help="normal-world files version (backbone training distribution)")
    ap.add_argument("-real_edge_sample", type=int, default=200000)
    ap.add_argument("-cap_len", type=int, default=100)
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out_name", type=str, default="uncond_pool_eval.json")
    args = ap.parse_args()

    A = pickle.load(open(join(args.porto, f"porto_shrink_A_{args.norm_ver}_normal.ts"), "rb")).bool()
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_{args.norm_ver}_normal.pkl"), "rb"))
    real = [list(map(int, p)) for p in sp if len(p) >= 2]
    real_lens = np.array([len(p) for p in real])
    real_hash = set(hash(tuple(p)) for p in real)
    rng = np.random.RandomState(1)
    ridx = rng.choice(len(real), min(args.real_edge_sample, len(real)), replace=False)
    real_edges = Counter((u, v) for i in ridx for u, v in zip(real[i][:-1], real[i][1:]))
    print(f"[real normal {args.norm_ver}] n={len(real)} len mean={real_lens.mean():.1f} "
          f"std={real_lens.std():.1f} | edge sample={len(ridx)} paths", flush=True)

    out = {}
    for B in [int(b) for b in args.blocks.split(",")]:
        P = [list(map(int, p)) for p in torch.load(args.pool_pat.format(B=B))["paths"] if len(p) >= 2]
        lens = np.array([len(p) for p in P])
        n_valid, bad_e, tot_e = 0, 0, 0
        gen_edges = Counter()
        for p in P:
            ok = True
            for u, v in zip(p[:-1], p[1:]):
                tot_e += 1
                gen_edges[(u, v)] += 1
                if not A[u, v]:
                    bad_e += 1
                    ok = False
            n_valid += ok
        uniq = len(set(hash(tuple(p)) for p in P)) / len(P)
        novel = float(np.mean([hash(tuple(p)) not in real_hash for p in P]))
        row = {
            "n": len(P),
            "valid": n_valid / len(P),
            "invE": bad_e / max(tot_e, 1),
            "len_mean": float(lens.mean()), "len_std": float(lens.std()),
            "len_ks": ks_stat(lens, real_lens),
            "cap_frac": float((lens >= args.cap_len).mean()),
            "uniq": uniq, "novel": novel,
            "edge_jsd": jsd(gen_edges, real_edges),
            # boundary residue: share of lengths with (L+1) % B == 0 -- the
            # class starved of END supervision under the pre-fix EOS scheme
            "bres_real": float(((real_lens + 1) % B == 0).mean()) if B > 1 else 1.0,
            "bres_gen": float(((lens + 1) % B == 0).mean()) if B > 1 else 1.0,
        }
        out[f"blk{B}"] = row
        print(f"blk{B:<3} valid={row['valid']:.3f} invE={100*row['invE']:.2f}% "
              f"len={row['len_mean']:.1f}±{row['len_std']:.1f} KS={row['len_ks']:.3f} "
              f"cap={100*row['cap_frac']:.1f}% uniq={row['uniq']:.3f} novel={row['novel']:.3f} "
              f"edgeJSD={row['edge_jsd']:.3f} bres={row['bres_gen']:.3f}/{row['bres_real']:.3f}",
              flush=True)

    with open(join(args.res_path, args.out_name), "w") as f:
        json.dump(out, f, indent=2)
    print("UNCOND_EVAL_DONE", flush=True)

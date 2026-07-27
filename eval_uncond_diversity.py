"""
Diversity of UNCONDITIONAL generations, per block, against a matched real
sample. All metrics computed on the VALID subset (diversity of garbage is
noise, not diversity), except distinct-OD which is also reported overall.

  distinct_od : fraction of distinct (start,end) pairs among valid paths
  edge_support: number of distinct directed edges used by valid paths
  edge_entropy: Shannon entropy (bits) of the edge-usage distribution
  pair_jacc   : mean pairwise Jaccard similarity of edge sets over random
                pairs of valid paths (lower = more diverse)
Real reference row: the same metrics on an equal-size sample of real paths.

  python eval_uncond_diversity.py -pool_pat "./sets_disc/uncond_pool_v4_blk{B}.pth"
"""

import argparse
import json
import pickle
from collections import Counter
from os.path import join

import numpy as np
import torch


def metrics(paths, A, rng, n_pairs=4000):
    vp = [p for p in paths if all(A[u, v] for u, v in zip(p[:-1], p[1:]))]
    if len(vp) < 10:
        return None, len(vp)
    ods = set((p[0], p[-1]) for p in vp)
    cnt = Counter((u, v) for p in vp for u, v in zip(p[:-1], p[1:]))
    freq = np.array(list(cnt.values()), dtype=float)
    prob = freq / freq.sum()
    ent = float(-(prob * np.log2(prob)).sum())
    esets = [set(zip(p[:-1], p[1:])) for p in vp]
    sims = []
    for _ in range(n_pairs):
        i, j = rng.integers(len(esets)), rng.integers(len(esets))
        if i == j:
            continue
        a, b = esets[i], esets[j]
        sims.append(len(a & b) / max(len(a | b), 1))
    return {
        "n_valid": len(vp),
        "distinct_od": len(ods) / len(vp),
        "edge_support": len(cnt),
        "edge_entropy": ent,
        "pair_jacc": float(np.mean(sims)),
    }, len(vp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-pool_pat", type=str, default="./sets_disc/uncond_pool_v4_blk{B}.pth")
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-norm_ver", type=str, default="v3-0.05")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    args = ap.parse_args()

    A = pickle.load(open(join(args.porto, f"porto_shrink_A_{args.norm_ver}_normal.ts"), "rb")).bool()
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_{args.norm_ver}_normal.pkl"), "rb"))
    rng = np.random.default_rng(args.seed)

    out = {}
    ridx = rng.choice(len(sp), args.n, replace=False)
    real = [list(map(int, sp[i])) for i in ridx if len(sp[i]) >= 2]
    m, nv = metrics(real, A, rng)
    out["real"] = m
    print(f"real   n_valid={m['n_valid']} distinctOD={m['distinct_od']:.3f} "
          f"edges={m['edge_support']} H={m['edge_entropy']:.2f}b jacc={m['pair_jacc']:.4f}", flush=True)

    for B in [int(b) for b in args.blocks.split(",")]:
        P = [list(map(int, p)) for p in torch.load(args.pool_pat.format(B=B))["paths"]
             if len(p) >= 2][:args.n]
        m, nv = metrics(P, A, rng)
        out[f"blk{B}"] = m
        print(f"blk{B:<3} n_valid={m['n_valid']} distinctOD={m['distinct_od']:.3f} "
              f"edges={m['edge_support']} H={m['edge_entropy']:.2f}b jacc={m['pair_jacc']:.4f}", flush=True)

    with open(join(args.res_path, "uncond_diversity_v4.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("UNCOND_DIVERSITY_DONE", flush=True)

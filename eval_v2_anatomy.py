"""v2 failure anatomy on held-out data: len-corr split, miss endpoint hops,
invalid-edge stats (normal graph). Single fixed generation seed.

Covers the SAME settings as Sec.2.1/2.2: both kernels, all 7 block sizes,
on the shared held-out set (default 1,000 pairs).

  CUDA_VISIBLE_DEVICES=0 python eval_v2_anatomy.py -kernel mask  -blocks 1,2,4,8,16,32,64
  CUDA_VISIBLE_DEVICES=1 python eval_v2_anatomy.py -kernel graph -blocks 1,2,4,8,16,32,64
"""
import argparse
import pickle
import numpy as np, torch, networkx as nx

SPLIT_SEED, EVAL_SEED = 1, 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-kernel", type=str, default="mask", choices=["mask", "graph"])
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-n", type=int, default=1000)          # held-out pairs (== Sec.2)
    ap.add_argument("-seed", type=int, default=7)          # ONE fixed generation seed
    ap.add_argument("-batch", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    sp = pickle.load(open("./porto_data/porto_shrink_SP_v3-0.05_normal.pkl", "rb"))
    A = pickle.load(open("./porto_data/porto_shrink_A_v3-0.05_normal.ts", "rb")).bool()
    G = pickle.load(open("./porto_data/porto_shrink_G_v3-0.05_normal.pkl", "rb"))

    # ---- reconstruct the v2 held-out eval set (identical to Sec.2) ----
    n = len(sp); train_num = int(0.8 * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    test_idx = perm[train_num:]
    np.random.RandomState(EVAL_SEED).shuffle(test_idx)
    real = []
    for i in test_idx:
        p = list(map(int, sp[i]))
        if len(p) >= 2:
            real.append(p)
        if len(real) == args.n:
            break
    rl = np.array([len(g) for g in real])
    print(f"[anat {args.kernel}] held-out n={len(real)} seed={args.seed} "
          f"real len mean={rl.mean():.2f}", flush=True)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")

    for blk in [int(b) for b in args.blocks.split(",")]:
        model = torch.load(f"./sets_model/BD_porto_v3_normal_{args.kernel}_blk{blk}_v2_bd.pth",
                           map_location=device)
        model.eval()
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        planned, hits = [], []
        for s in range(0, len(real), args.batch):
            b = real[s:s + args.batch]
            planned += model.plan([p[0] for p in b], [p[-1] for p in b], use_refine=False)
            hits += model.last_hits
        gl = np.array([len(p) for p in planned])
        h = np.array(hits, dtype=bool)
        dists, bad_e, tot_e, bad_hops, pos_first, pos_rest = [], 0, 0, [], 0, 0
        for p, g in zip(planned, real):
            if len(p) and p[-1] != g[-1]:
                try:
                    dists.append(nx.shortest_path_length(G, p[-1], g[-1]))
                except Exception:
                    pass
            for i, (u, v) in enumerate(zip(p[:-1], p[1:])):
                tot_e += 1
                if not A[u, v]:
                    bad_e += 1
                    cpos = (i + 2) % blk
                    if cpos == 0:
                        pos_first += 1
                    else:
                        pos_rest += 1
                    try:
                        bad_hops.append(nx.shortest_path_length(G, u, v))
                    except Exception:
                        pass
        d = np.array(dists); bh = np.array(bad_hops)
        row = {
            "kernel": args.kernel, "blk": blk, "n": len(real), "seed": args.seed,
            "hit": float(h.mean()),
            "lcorr_all": corr(gl, rl),
            "lcorr_hit": corr(gl[h], rl[h]),
            "lcorr_miss": corr(gl[~h], rl[~h]),
            "n_miss": int((~h).sum()),
            "miss_hop_med": float(np.median(d)) if len(d) else -1.0,
            "miss_hop_mean": float(np.mean(d)) if len(d) else -1.0,
            "miss_le2_pct": float(np.mean(d <= 2) * 100) if len(d) else 0.0,
            "inv_edge_pct": float(100 * bad_e / max(tot_e, 1)),
            "inv_hop_med": float(np.median(bh)) if len(bh) else -1.0,
            "boundary": pos_first, "interior": pos_rest,
        }
        print("ROW " + str(row), flush=True)
        print(f"== {args.kernel} blk{blk}: hit={row['hit']:.3f} | lcorr all={row['lcorr_all']:.3f} "
              f"hit={row['lcorr_hit']:.3f} miss={row['lcorr_miss']:.3f}(n={row['n_miss']}) | "
              f"miss->dst med={row['miss_hop_med']:.0f} mean={row['miss_hop_mean']:.2f} <=2 {row['miss_le2_pct']:.0f}% | "
              f"invE={row['inv_edge_pct']:.2f}% hop med={row['inv_hop_med']:.0f} "
              f"boundary={row['boundary']} interior={row['interior']}", flush=True)
    print(f"ANAT_DONE_{args.kernel}", flush=True)

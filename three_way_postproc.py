"""
Three-way deep dive (base / modelD-guided / adj+modelD) x post-processing
(raw / P1 / P3 / P1+P3), scored on except_0 with EM / PC / invalid edge &
node ratios. Works for both kernels (plan_guided dispatches on model.kernel).

  python three_way_postproc.py -ckpt sets_model/..._bd.pth \
      -disc sets_disc/BDdisc_..._model_blk64.pth -tag mask64
"""

import argparse
import pickle
import numpy as np
import torch
import networkx as nx
from os.path import join

from eval_shortest import evaluate_em_pc

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ckpt", type=str, required=True)
    ap.add_argument("-disc", type=str, required=True)
    ap.add_argument("-tag", type=str, required=True)
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-order", type=str, default="first_hit", choices=["first_hit", "l2r"],
                    help="mask within-block reveal order")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    porto, fam = "./porto_data", args.family

    A_exc = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    G_exc = pickle.load(open(join(porto, f"porto_shrink_G_v4-{fam}_except_0.pkl"), "rb"))
    sp_exc = pickle.load(open(join(porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    removed = A_norm & ~A_exc
    deg_ratio = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
    A_dev = A_exc.float().to(device)

    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.eval_num]

    def splice(p):
        out, added = [p[0]], 0
        for u, v in zip(p[:-1], p[1:]):
            if A_exc[u, v]:
                out.append(v)
                continue
            try:
                seg = nx.shortest_path(G_exc, u, v)[1:]
                added += len(seg) - 1
                out.extend(seg)
            except Exception:
                out.append(v)
        return out, added

    def endpoint(p, dst):
        if len(p) == 0 or p[-1] == dst:
            return list(p), 0
        try:
            seg = nx.shortest_path(G_exc, p[-1], dst)[1:]
            return list(p) + seg, len(seg)
        except Exception:
            return list(p), 0

    def score(tag, paths, patch, ess=None):
        summ, _, _ = evaluate_em_pc(gen_paths=paths, A=A_exc.float(), shortest_paths=sp_exc,
                                    save_dir="./sets_res/em_pc", prefix=tag)
        arr = float(np.mean([len(q) > 0 and q[-1] == g[-1] for q, g in zip(paths, real)]))
        bad_e, tot_e, bad_n, tot_n, rem_e = 0, 0, 0, 0, 0
        for p in paths:
            marked = set()
            for i, (u, v) in enumerate(zip(p[:-1], p[1:])):
                tot_e += 1
                if not A_exc[u, v]:
                    bad_e += 1
                    marked.add(i)
                    marked.add(i + 1)
                    if removed[u, v]:
                        rem_e += 1
            bad_n += len(marked)
            tot_n += len(p)
        es = f" ess={ess:.1f}" if ess is not None else ""
        print(f"{tag:<30} arr={arr:.3f} valid={summ['valid_rate']:.3f} em={summ['em_score']:.3f} "
              f"pc={summ['pc_score']:.3f} invE={100*bad_e/max(tot_e,1):.2f}% invN={100*bad_n/max(tot_n,1):.2f}% "
              f"remE={100*rem_e/max(tot_e,1):.2f}% patch={np.mean(patch):.2f}{es}", flush=True)

    disc = torch.load(args.disc, map_location=device)
    disc.eval()
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    for cfg, kw in [("base", None), ("modelD", dict(adj_prop=False)), ("adj+modelD", dict(adj_prop=True))]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        planned, ess_log = [], []
        for s in range(0, len(real), args.batch):
            b = real[s:s + args.batch]
            o, d = [p[0] for p in b], [p[-1] for p in b]
            if kw is None:
                out = model.plan(o, d, use_refine=False, order=args.order)
            else:
                out = model.plan_guided(o, d, disc, A_dev, deg_ratio, n_is=args.n_is,
                                        ess_log=ess_log, order=args.order, **kw)
            planned += out
        ess = float(np.nanmean(ess_log)) if ess_log else None
        p1 = [splice(p) for p in planned]
        p3 = [endpoint(p, g[-1]) for p, g in zip(planned, real)]
        p13 = []
        for (q, a1), g in zip(p1, real):
            q2, a2 = endpoint(q, g[-1])
            p13.append((q2, a1 + a2))
        score(f"{args.tag}_{cfg}_raw", planned, [0] * len(real), ess)
        score(f"{args.tag}_{cfg}_P1", [q for q, _ in p1], [a for _, a in p1])
        score(f"{args.tag}_{cfg}_P3", [q for q, _ in p3], [a for _, a in p3])
        score(f"{args.tag}_{cfg}_P1P3", [q for q, _ in p13], [a for _, a in p13])
        print(flush=True)
    print("THREEWAY_DONE", flush=True)

"""
Evaluate D-CBG guidance (Schiff et al. plug-in) on except_0 with the same
protocol and metrics as three_way_postproc.py (raw + P1/P3/P1+P3).

  python eval_dcbg.py -kernel mask -blk 4 -gamma 2.0 \
      -clf sets_disc/DCBGclf_mask_blk4_f0.05_p1.pth
"""

import argparse
import pickle
import time
from os.path import join

import networkx as nx
import numpy as np
import torch

from eval_shortest import evaluate_em_pc
from dcbg_plugin import plan_dcbg_mask, plan_dcbg_graph, AdjBound

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-kernel", type=str, required=True, choices=["mask", "graph"])
    ap.add_argument("-blk", type=int, required=True)
    ap.add_argument("-gamma", type=float, required=True)
    ap.add_argument("-clf", type=str, required=True)
    ap.add_argument("-adj", type=int, default=0, help="1 = adjacency-aware classifier")
    ap.add_argument("-ckpt", type=str, default="")
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    fam = args.family
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    G_exc = pickle.load(open(join(args.porto, f"porto_shrink_G_v4-{fam}_except_0.pkl"), "rb"))
    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    removed = A_norm & ~A_exc
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.eval_num]

    ckpt = args.ckpt or (f"./sets_model/BD_porto_v3_normal_mask_blk{args.blk}_base_bd.pth"
                         if args.kernel == "mask" else
                         "./sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth")
    model = torch.load(ckpt, map_location=device)
    model.eval()
    clf = torch.load(args.clf, map_location=device)
    clf.eval()
    if args.adj:
        dr_dev = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
        clf = AdjBound(clf, A_exc.float().to(device), dr_dev)
    plan_fn = plan_dcbg_mask if args.kernel == "mask" else plan_dcbg_graph

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    planned = []
    t0 = time.time()
    for s in range(0, len(real), args.batch):
        bb = real[s:s + args.batch]
        planned += plan_fn(model, [p[0] for p in bb], [p[-1] for p in bb], clf, args.gamma)
        print(f"  {len(planned)}/{len(real)} ({time.time()-t0:.0f}s)", flush=True)

    def splice(p):
        out, add = [p[0]], 0
        for u, v in zip(p[:-1], p[1:]):
            if A_exc[u, v]:
                out.append(v)
                continue
            try:
                seg = nx.shortest_path(G_exc, u, v)[1:]
                add += len(seg) - 1
                out.extend(seg)
            except Exception:
                out.append(v)
        return out, add

    def endpoint(p, dst):
        if len(p) == 0 or p[-1] == dst:
            return list(p), 0
        try:
            seg = nx.shortest_path(G_exc, p[-1], dst)[1:]
            return list(p) + seg, len(seg)
        except Exception:
            return list(p), 0

    tag0 = f"DCBG_{args.kernel}{args.blk}_g{args.gamma}" + ("_adj" if args.adj else "")

    def score(tag, paths, patch):
        summ, _, _ = evaluate_em_pc(gen_paths=paths, A=A_exc.float(), shortest_paths=sp_exc,
                                    save_dir="./sets_res/em_pc", prefix=tag)
        arr = float(np.mean([len(q) > 0 and q[-1] == g[-1] for q, g in zip(paths, real)]))
        bad_e, tot_e, rem_e = 0, 0, 0
        for p in paths:
            for u, v in zip(p[:-1], p[1:]):
                tot_e += 1
                if not A_exc[u, v]:
                    bad_e += 1
                    if removed[u, v]:
                        rem_e += 1
        print(f"{tag:<28} arr={arr:.3f} valid={summ['valid_rate']:.3f} em={summ['em_score']:.3f} "
              f"pc={summ['pc_score']:.3f} invE={100*bad_e/max(tot_e,1):.2f}% "
              f"remE={100*rem_e/max(tot_e,1):.2f}% patch={np.mean(patch):.2f}", flush=True)

    p1 = [splice(p) for p in planned]
    p13 = []
    for (q, a1), g in zip(p1, real):
        q2, a2 = endpoint(q, g[-1])
        p13.append((q2, a1 + a2))
    score(f"{tag0}_raw", planned, [0] * len(real))
    score(f"{tag0}_P1P3", [q for q, _ in p13], [a for _, a in p13])
    print(f"sec_per_path={(time.time()-t0)/len(real):.3f}")
    print("DCBG_EVAL_DONE", flush=True)

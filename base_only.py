"""
Base-only evaluation (no guidance) on except_0, identical protocol/scoring to
three_way_postproc.py. Used for block sizes where guidance is not applied
(e.g. blk 1/2): reports base raw / P1 / P3 / P1P3.

  python base_only.py -ckpt sets_model/BD_porto_v3_normal_mask_blk1_v2_bd.pth -tag v3m1
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
    ap.add_argument("-tag", type=str, required=True)
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    porto, fam = "./porto_data", args.family

    A_exc = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    G_exc = pickle.load(open(join(porto, f"porto_shrink_G_v4-{fam}_except_0.pkl"), "rb"))
    sp_exc = pickle.load(open(join(porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    removed = A_norm & ~A_exc

    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.eval_num]

    def splice(p):
        out, added = [p[0]], 0
        for u, v in zip(p[:-1], p[1:]):
            if A_exc[u, v]:
                out.append(v); continue
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

    def score(tag, paths, patch):
        summ, _, _ = evaluate_em_pc(gen_paths=paths, A=A_exc.float(), shortest_paths=sp_exc,
                                    save_dir="./sets_res/em_pc", prefix=tag)
        arr = float(np.mean([len(q) > 0 and q[-1] == g[-1] for q, g in zip(paths, real)]))
        bad_e, tot_e, bad_n, tot_n, rem_e = 0, 0, 0, 0, 0
        for p in paths:
            marked = set()
            for i, (u, v) in enumerate(zip(p[:-1], p[1:])):
                tot_e += 1
                if not A_exc[u, v]:
                    bad_e += 1; marked.add(i); marked.add(i + 1)
                    if removed[u, v]:
                        rem_e += 1
            bad_n += len(marked); tot_n += len(p)
        print(f"{tag:<30} arr={arr:.3f} valid={summ['valid_rate']:.3f} em={summ['em_score']:.3f} "
              f"pc={summ['pc_score']:.3f} invE={100*bad_e/max(tot_e,1):.2f}% invN={100*bad_n/max(tot_n,1):.2f}% "
              f"remE={100*rem_e/max(tot_e,1):.2f}% patch={np.mean(patch):.2f}", flush=True)

    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    planned = []
    for s in range(0, len(real), args.batch):
        b = real[s:s + args.batch]
        o, d = [p[0] for p in b], [p[-1] for p in b]
        planned += model.plan(o, d, use_refine=False)
    p1 = [splice(p) for p in planned]
    p3 = [endpoint(p, g[-1]) for p, g in zip(planned, real)]
    p13 = []
    for (q, a1), g in zip(p1, real):
        q2, a2 = endpoint(q, g[-1])
        p13.append((q2, a1 + a2))
    score(f"{args.tag}_base_raw", planned, [0] * len(real))
    score(f"{args.tag}_base_P1", [q for q, _ in p1], [a for _, a in p1])
    score(f"{args.tag}_base_P3", [q for q, _ in p3], [a for _, a in p3])
    score(f"{args.tag}_base_P1P3", [q for q, _ in p13], [a for _, a in p13])
    print("BASEONLY_DONE", flush=True)

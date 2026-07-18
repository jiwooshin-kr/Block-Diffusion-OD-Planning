"""
Guidance evaluation on exceptional scenario 0.

Loads a normal-trained BD mask-kernel checkpoint, plans 1,000 held-out OD
pairs drawn from except_0's shortest-path data (the rows RESERVED by
train_bd_disc.py's split, so the discriminator never saw them), and scores
against the EXCEPTIONAL graph: validity/EM/PC use A_except0 + except_0
shortest paths, plus the removed-edge usage rate (the direct guidance target).

  baseline: python eval_bd_guidance.py -blk 4 -family 0.05 -disc none
  guided  : python eval_bd_guidance.py -blk 4 -family 0.05 \
                -disc sets_disc/BDdisc_f0.05_p1_e0_data.pth -n_is 100
"""

import argparse
import pickle
import time
from os.path import join

import numpy as np
import torch

from eval_shortest import evaluate_em_pc, success_arrival_rate

SPLIT_SEED = 777
EVAL_RESERVE = 1000


def load_pickle(path):
    return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blk", type=int, required=True)
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-disc", type=str, default="none")
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-w_gamma", type=float, default=1.0)
    ap.add_argument("-cand_temp", type=float, default=1.0)
    ap.add_argument("-adj_prop", type=int, default=0, help="1 = adjacency-masked proposals (Lemma 3)")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-seed", type=int, default=1)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-res_name", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fam = args.family

    # exceptional scenario data (scoring world)
    A_exc = load_pickle(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts")).bool().float()
    A_norm = load_pickle(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts")).bool().float()
    sp_exc = load_pickle(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"))
    removed = (A_norm.bool() & ~A_exc.bool())
    deg_ratio = (A_exc.sum(1) / A_norm.sum(1).clamp(min=1.0)).float().to(device)
    A_exc_dev = A_exc.to(device)

    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:EVAL_RESERVE] if len(sp_exc[i]) >= 2]
    real = real[:args.eval_num]
    n = len(real)

    model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{args.blk}_base_bd.pth",
                       map_location=device)
    model.eval()

    disc = None
    if args.disc != "none":
        disc = torch.load(args.disc, map_location=device)
        disc.eval()
        disc.requires_grad_(False)

    planned, hits, ess_log = [], [], []
    t0 = time.time()
    for s in range(0, n, args.batch):
        b = real[s:s + args.batch]
        origs = [p[0] for p in b]
        dests = [p[-1] for p in b]
        if disc is None and not args.adj_prop:
            out = model.plan(origs, dests, use_refine=False)
        else:
            out = model.plan_guided(origs, dests, disc, A_exc_dev, deg_ratio,
                                    n_is=args.n_is, w_gamma=args.w_gamma,
                                    cand_temp=args.cand_temp, ess_log=ess_log,
                                    adj_prop=bool(args.adj_prop))
        planned.extend(out)
        hits.extend(model.last_hits)
        print(f"  {s + len(b)}/{n} planned ({time.time()-t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    # ---- scoring against the exceptional world -------------------------
    tag = args.res_name or (f"BDguid_{fam}_blk{args.blk}_" +
                            ("base" if disc is None else args.disc.split('/')[-1].replace('.pth', '')))
    em_summary, em_records, _ = evaluate_em_pc(
        gen_paths=planned, A=A_exc, shortest_paths=sp_exc,
        save_dir=join(args.res_path, "em_pc"), prefix=tag)
    valid_flags = [bool(r["valid"]) for r in em_records]

    rem_paths, rem_edges, tot_edges = 0, 0, 0
    for p in planned:
        bad = sum(1 for u, v in zip(p[:-1], p[1:]) if removed[u, v])
        rem_edges += bad
        tot_edges += max(len(p) - 1, 0)
        rem_paths += bad > 0

    gl = np.array([len(p) for p in planned])
    rl = np.array([len(g) for g in real])
    res = {
        "tag": tag, "blk": args.blk, "family": fam,
        "disc": args.disc, "n_is": args.n_is if disc is not None else 0,
        "adj_prop": int(args.adj_prop),
        "n": n,
        "hit_ratio": float(np.mean(hits)),
        "arrival_success_rate": float(success_arrival_rate(planned, real)),
        "valid_rate(exc)": em_summary["valid_rate"],
        "valid_and_hit_rate": float(np.mean([v and h for v, h in zip(valid_flags, hits)])),
        "removed_edge_path_rate": float(rem_paths / max(n, 1)),
        "removed_edge_per_path": float(rem_edges / max(n, 1)),
        "removed_edge_edge_rate": float(rem_edges / max(tot_edges, 1)),
        "em_score": em_summary["em_score"],
        "pc_score": em_summary["pc_score"],
        "gen_len_mean": float(gl.mean()),
        "real_len_mean": float(rl.mean()),
        "len_corr": float(np.corrcoef(gl, rl)[0, 1]) if gl.std() > 0 and rl.std() > 0 else 0.0,
        "sec_per_path": elapsed / max(n, 1),
    }
    if ess_log:
        e = np.array(ess_log)
        res["ess_mean"] = float(np.nanmean(e))
        res["ess_p10"] = float(np.nanpercentile(e, 10))
        res["ess_p50"] = float(np.nanpercentile(e, 50))
    for k, v in res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    with open(join(args.res_path, f"{tag}.res"), "w") as f:
        f.writelines(str(res))
    print(f"written {join(args.res_path, f'{tag}.res')}")

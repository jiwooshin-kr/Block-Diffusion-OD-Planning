"""
Controlled evaluation of the v2 (seeded-split) block-size sweep, both kernels:
  - reconstructs the EXACT train/test membership of the v2 trainings
    (dataset loaded with shuffle=False + torch random_split seeded with
    args.seed=1), and draws ONE fixed 1,000-pair eval set from the TEST
    split only -> truly held-out, shared by every model
  - 3 generation seeds per model -> mean +/- std
  - raw (no-refine) protocol, Sec.1/Sec.2 metrics incl. length-bin arrivals

  python eval_bd_sweep_controlled.py -kernel mask  -blocks 1,2,4,8,16,32,64
  python eval_bd_sweep_controlled.py -kernel graph -blocks 1,2,4,8,16,32,64
"""

import argparse
import pickle
import time

import numpy as np
import torch

from eval_shortest import evaluate_em_pc
from main_bd import dtw_distance, path_to_coords

SPLIT_SEED = 1        # args.seed default used by the v2 trainings
EVAL_SEED = 777
LEN_BINS = [(2, 15), (16, 25), (26, 35), (36, 100)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-kernel", type=str, default="mask", choices=["mask", "graph"])
    ap.add_argument("-suffix", type=str, default="v2")
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-seeds", type=str, default="1,2,3")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    sp = pickle.load(open("./porto_data/porto_shrink_SP_v3-0.05_normal.pkl", "rb"))
    A = pickle.load(open("./porto_data/porto_shrink_A_v3-0.05_normal.ts", "rb")).bool().float()
    G = pickle.load(open("./porto_data/porto_shrink_G_v3-0.05_normal.pkl", "rb"))

    # ---- reconstruct the v2 test split (mirrors TrainerBD.__init__) -----
    n = len(sp)
    train_num = int(0.8 * n)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=g).tolist()      # random_split order
    test_idx = perm[train_num:]
    rng = np.random.RandomState(EVAL_SEED)
    rng.shuffle(test_idx)
    real = []
    for i in test_idx:
        p = list(map(int, sp[i]))
        if len(p) >= 2:
            real.append(p)
        if len(real) == args.eval_num:
            break
    rl = np.array([len(g_) for g_ in real])
    print(f"[v2-ctrl] held-out eval set: n={len(real)} from {len(test_idx)} test rows, "
          f"real len mean={rl.mean():.2f}", flush=True)

    for blk in [int(b) for b in args.blocks.split(",")]:
        ckpt = f"./sets_model/BD_porto_v3_normal_{args.kernel}_blk{blk}_{args.suffix}_bd.pth"
        model = torch.load(ckpt, map_location=device)
        model.eval()
        rows = []
        for seed in [int(x) for x in args.seeds.split(",")]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            planned, hits = [], []
            t0 = time.time()
            for s in range(0, len(real), args.batch):
                b = real[s:s + args.batch]
                out = model.plan([p[0] for p in b], [p[-1] for p in b], use_refine=False)
                planned += out
                hits += model.last_hits
            summ, recs, _ = evaluate_em_pc(gen_paths=planned, A=A, shortest_paths=sp,
                                           save_dir="./sets_res/em_pc",
                                           prefix=f"V2_{args.kernel}{blk}_s{seed}")
            valid_flags = [bool(r["valid"]) for r in recs]
            gl = np.array([len(p) for p in planned])
            arr_flags = np.array([int(len(p) > 0 and p[-1] == g_[-1]) for p, g_ in zip(planned, real)])
            dtws = [dtw_distance(path_to_coords(p, G), path_to_coords(g_, G))
                    for p, g_ in zip(planned, real) if len(p) >= 1]
            row = {
                "seed": seed,
                "hit": float(np.mean(hits)),
                "arr": float(arr_flags.mean()),
                "valid": summ["valid_rate"],
                "vh": float(np.mean([v and h for v, h in zip(valid_flags, hits)])),
                "em": summ["em_score"], "pc": summ["pc_score"],
                "dtw": float(np.mean(dtws)),
                "lerr": float(np.mean(np.abs(gl - rl))),
                "lcorr": float(np.corrcoef(gl, rl)[0, 1]),
                "sec": (time.time() - t0) / max(len(real), 1),
            }
            for lo, hi in LEN_BINS:
                m = (rl >= lo) & (rl <= hi)
                row[f"arr{lo}-{hi}"] = float(arr_flags[m].mean())
            rows.append(row)
            print(f"[{args.kernel} blk{blk} seed{seed}] " +
                  " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "seed"), flush=True)
        keys = [k for k in rows[0] if k != "seed"]
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
               for k in keys}
        print(f"AGG {args.kernel} blk{blk} " +
              " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in agg.items()), flush=True)
        with open(f"./sets_res/V2_sweep_{args.kernel}_blk{blk}.res", "w") as f:
            f.writelines(str({"kernel": args.kernel, "blk": blk, "rows": rows, "agg": agg}))
    print("V2_SWEEP_DONE", flush=True)

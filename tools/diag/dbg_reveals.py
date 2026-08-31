"""Why is adj+IW U-shaped in block size? Count, per (block, method, batch):
   - reveal steps actually executed (= wall-clock driver, since every reveal is
     one full-canvas forward: no KV cache)
   - how many rows are still un-stopped when the sampler gives up
   - the canvas length the batch ran to
The planner loops until EVERY row has stopped, so one straggler sets the cost.
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------

import argparse
import pickle
from os.path import join

import numpy as np
import torch

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-repeats", type=int, default=3)
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    fam = args.family
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp))
    need = args.batch * args.repeats
    pool = [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:need]
    OD = [([p[0] for p in pool[k * args.batch:(k + 1) * args.batch]],
           [p[-1] for p in pool[k * args.batch:(k + 1) * args.batch]]) for k in range(args.repeats)]
    A_dev = A_exc.float().to(device)
    dr = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)

    print(f"{'blk':>4} {'method':<7} {'cap':>5} | " +
          " | ".join(f"{'rev':>4} {'nostop':>6} {'arr':>5}" for _ in range(args.repeats)), flush=True)
    for blk in [int(b) for b in args.blocks.split(",")]:
        model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v4_bd.pth", map_location=device)
        model.eval()
        disc = torch.load(f"./sets_disc/BDdisc_f{fam}_p1_e0_model_blk{blk}_v4.pth", map_location=device)
        disc.eval()
        cap = model.bd_max_len - 2
        for adj in (False, True):
            cells = []
            for k in range(args.repeats):
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                el = []
                paths = model.plan_guided(OD[k][0], OD[k][1], disc, A_dev, dr,
                                          n_is=args.n_is, ess_log=el, adj_prop=adj)
                hits = np.array(model.last_hits)
                # a row that neither hit dst nor emitted END ran to the canvas end
                nostop = int(sum(1 for p, h in zip(paths, hits)
                                 if not h and len(p) >= model.bd_max_len - 2))
                cells.append(f"{len(el):>4} {nostop:>6} {hits.mean():>5.3f}")
            print(f"{blk:>4} {'adj+IW' if adj else 'IW':<7} {cap:>5} | " + " | ".join(cells), flush=True)
    print("DBG_DONE", flush=True)

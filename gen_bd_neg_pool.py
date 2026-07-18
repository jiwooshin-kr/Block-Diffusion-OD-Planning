"""
Model-sample negative pool for the disc 'mix' ablation.

Two kinds of canvas-form partial sequences [dst, ori, ...], drawn from the BD
model trained on NORMAL data (they represent p_theta, the denominator of the
importance ratio):
  1. raw:        open-ended generations (sequential first-hitting), randomly
                 truncated
  2. mean-field: [j committed blocks || one-shot independent per-position
                 completion of the next block] -- exactly the guidance-time
                 candidate distribution

Uses the blk4 and blk16 checkpoints (the pivot sizes).
"""

import argparse
import pickle
from os.path import join

import numpy as np
import torch

from models_seq.bd_models import get_block_causal_mask

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n_od", type=int, default=3000, help="OD pairs per checkpoint")
    ap.add_argument("-batch", type=int, default=200)
    ap.add_argument("-out", type=str, default="./sets_disc/neg_pool.pth")
    ap.add_argument("-seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import os
    os.makedirs("./sets_disc", exist_ok=True)

    normal_sp = pickle.load(open("./porto_data/porto_shrink_SP_v4-0.05_normal.pkl", "rb"))
    ods = [normal_sp[i] for i in rng.choice(len(normal_sp), args.n_od * 2, replace=False)
           if len(normal_sp[i]) >= 2]

    pool = []
    for blk in [4, 16]:
        model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_base_bd.pth",
                           map_location=device)
        model.eval()
        n_vertex = model.n_vertex

        # ---- 1. raw generations, randomly truncated -------------------
        sub = ods[:args.n_od]
        for s in range(0, len(sub), args.batch):
            bpaths = sub[s:s + args.batch]
            origs = [int(p[0]) for p in bpaths]
            dests = [int(p[-1]) for p in bpaths]
            out = model.plan(origs, dests, use_refine=False)
            for p, d in zip(out, dests):
                if len(p) < 2:
                    continue
                j = int(rng.integers(2, len(p) + 1))
                pool.append([d] + [int(v) for v in p[:j]])

        # ---- 2. mean-field block completions ---------------------------
        with torch.no_grad():
            sub = ods[args.n_od:2 * args.n_od]
            for s in range(0, len(sub), args.batch):
                bpaths = sub[s:s + args.batch]
                b = len(bpaths)
                origs = torch.tensor([int(p[0]) for p in bpaths], device=device)
                dests = torch.tensor([int(p[-1]) for p in bpaths], device=device)
                # commit a random number of blocks with the normal sampler
                max_blocks = model.bd_max_len // model.block_size
                j_stop = int(rng.integers(1, min(8, max_blocks)))
                seq = torch.empty(b, 0, dtype=torch.long, device=device)
                for j in range(j_stop + 1):
                    seq = torch.cat([seq, model._init_block(b)], dim=1)
                    ssz = seq.shape[1]
                    for p_idx, val in [(0, dests), (1, origs)]:
                        if ssz - model.block_size <= p_idx < ssz:
                            seq[:, p_idx] = val
                    if j < j_stop:
                        if ssz > model.pfx:
                            model._denoise_block_mask(seq, origs, dests, 1.0)
                    else:
                        # one-shot mean-field completion of the final block
                        block = model.block_size
                        attn = get_block_causal_mask(ssz, block, device)
                        t_in = model._t_input(b, ssz, 100.0)
                        logits = model._cfg_logits(seq, t_in, origs, dests, attn, 1.0)
                        bl = logits[:, -block:].clone()
                        bl[..., model.MASK] = -1e9
                        p_x0 = bl.softmax(dim=-1)
                        V = p_x0.shape[-1]
                        fill = torch.multinomial(p_x0.reshape(-1, V), 1).view(b, block)
                        m = seq[:, -block:] == model.MASK
                        seq[:, -block:] = torch.where(m, fill, seq[:, -block:])
                seq = seq.cpu()
                for i in range(b):
                    row, d = seq[i].tolist(), int(dests[i])
                    cut = len(row)
                    for k in range(model.pfx, len(row)):
                        if row[k] >= n_vertex:
                            cut = k
                            break
                        if row[k] == d:
                            cut = k + 1
                            break
                    part = [row[0]] + [v for v in row[1:cut] if v < n_vertex]
                    if len(part) >= 3:
                        pool.append(part)

    torch.save({"sequences": pool}, args.out)
    print(f"pool saved: {len(pool)} sequences -> {args.out}")

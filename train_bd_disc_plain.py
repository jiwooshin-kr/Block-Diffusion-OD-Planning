"""
Plain partial-path discriminator (BDDiscriminator MINUS the adjacency
machinery) -- the IW-mechanism x plain-classifier cell of the controlled
2x2 comparison (RESULTS_BD sec. 6.12). Data-negative protocol, same splits.
"""

import argparse
import pickle
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

from dcbg_plugin import BDDiscriminatorPlain
from models_seq.bd_disc import make_partial, pad_batch

SPLIT_SEED = 777
EVAL_RESERVE = 1000

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-frac", type=float, default=1.0)
    ap.add_argument("-steps", type=int, default=4000)
    ap.add_argument("-bs", type=int, default=128)
    ap.add_argument("-lr", type=float, default=1e-3)
    ap.add_argument("-seed", type=int, default=1)
    ap.add_argument("-porto", type=str, default="./porto_data")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fam = args.family

    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    sp_norm = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_normal.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    n_train = max(1, int(np.ceil(args.frac / 100.0 * len(sp_exc))))
    pos_paths = [list(map(int, sp_exc[i])) for i in perm[EVAL_RESERVE:][:n_train] if len(sp_exc[i]) >= 2]
    neg_paths = [list(map(int, p)) for p in sp_norm if len(p) >= 2]
    val_pos = [list(map(int, sp_exc[i])) for i in perm[:500] if len(sp_exc[i]) >= 2]

    disc = BDDiscriminatorPlain(1390, device, pretrain_path="./sets_data/porto_node2vec.pkl")
    opt = torch.optim.Adam(disc.parameters(), lr=args.lr)
    half = args.bs // 2
    dummy_adj = torch.zeros(1, 1, device=device)
    dummy_dr = torch.zeros(1390, device=device)

    t0 = time.time()
    run_loss, run_acc = 0.0, 0.0
    for step in range(1, args.steps + 1):
        pos = [make_partial(pos_paths[int(rng.integers(len(pos_paths)))], rng) for _ in range(half)]
        neg = [make_partial(neg_paths[int(rng.integers(len(neg_paths)))], rng) for _ in range(half)]
        tokens, lengths = pad_batch(pos + neg, disc.PAD, device, max_len=128)
        labels = torch.cat([torch.full((half,), 0.95), torch.full((half,), 0.05)]).to(device)
        z = disc(tokens, lengths, dummy_adj, dummy_dr)
        loss = F.binary_cross_entropy_with_logits(z, labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        opt.step()
        run_loss += float(loss)
        hard = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)
        run_acc += float(((z > 0).float() == hard).float().mean())
        if step % 500 == 0 or step == args.steps:
            disc.eval()
            with torch.no_grad():
                vp = [make_partial(val_pos[int(rng.integers(len(val_pos)))], rng) for _ in range(128)]
                vn = [make_partial(neg_paths[int(rng.integers(len(neg_paths)))], rng) for _ in range(128)]
                vt, vl = pad_batch(vp + vn, disc.PAD, device, max_len=128)
                vz = disc(vt, vl, dummy_adj, dummy_dr)
                vacc = float(((vz > 0).float() == torch.cat(
                    [torch.ones(128), torch.zeros(128)]).to(device)).float().mean())
            disc.train()
            print(f"step {step}: loss={run_loss/500:.4f} acc={run_acc/500:.3f} | "
                  f"val acc={vacc:.3f} | {time.time()-t0:.0f}s", flush=True)
            run_loss, run_acc = 0.0, 0.0

    disc.eval()
    out = f"./sets_disc/BDdisc_plain_f{fam}_p{int(args.frac)}_e0_data.pth"
    torch.save(disc, out)
    print(f"saved {out}")

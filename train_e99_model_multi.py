"""
Train e99 (unseen: except 1..99) MODEL-negative discriminators for several
block sizes in ONE process. The 99-scenario positive pool is loaded once and
reused across every block; only the per-block model-generated negative pool
(uncond_pool_blk{B}.pth) differs. Mirrors train_bd_disc.py exactly (same
make_partial canvas, per-batch 50/50, adjacency conditioning, BCE, bounds).

  python train_e99_model_multi.py -jobs 4:sets_disc/uncond_pool_blk4.pth,...
Each job "B:pool[:outname]" -> saves BDdisc_f{fam}_p{frac}_e99_model_blk{B}{sfx}.pth
"""
import argparse
import pickle
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

from models_seq.bd_disc import BDDiscriminator, make_partial, pad_batch

SPLIT_SEED = 777
EVAL_RESERVE = 1000


def load_sp(porto, fam, tag):
    return pickle.load(open(join(porto, f"porto_shrink_SP_v4-{fam}_{tag}.pkl"), "rb"))


def load_A(porto, fam, tag):
    return pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_{tag}.ts"), "rb")).bool().float()


def except_train_split(sp_len, e, frac):
    rng = np.random.RandomState(SPLIT_SEED + e)
    perm = rng.permutation(sp_len)
    pool = perm[EVAL_RESERVE:] if e == 0 else perm
    n = max(1, int(np.ceil(frac / 100.0 * sp_len)))
    return pool[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-frac", type=float, default=1.0)
    ap.add_argument("-steps", type=int, default=4000)
    ap.add_argument("-bs", type=int, default=128)
    ap.add_argument("-lr", type=float, default=1e-3)
    ap.add_argument("-label_smooth", type=float, default=0.05)
    ap.add_argument("-logit_bound", type=float, default=4.0)
    ap.add_argument("-seed", type=int, default=1)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-out", type=str, default="./sets_disc")
    ap.add_argument("-jobs", type=str, required=True,
                    help="comma-sep 'B:poolpath[:outsuffix]'; e.g. 4:sets_disc/uncond_pool_blk4.pth")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fam = args.family

    # ---- load positives ONCE (99 scenarios) ----
    t0 = time.time()
    A_norm = load_A(args.porto, fam, "normal")
    deg_norm = A_norm.sum(1).clamp(min=1.0)
    n_vertex = A_norm.shape[0]
    exceptions = list(range(1, 100))
    exc_paths, exc_adj, exc_dr = {}, {}, {}
    for e in exceptions:
        sp = load_sp(args.porto, fam, f"except_{e}")
        idx = except_train_split(len(sp), e, args.frac)
        exc_paths[e] = [list(map(int, sp[i])) for i in idx if len(sp[i]) >= 2]
        A_e = load_A(args.porto, fam, f"except_{e}")
        exc_adj[e] = A_e
        exc_dr[e] = (A_e.sum(1) / deg_norm).float()
    tot_pos = sum(len(v) for v in exc_paths.values())
    print(f"[load] 99 scenarios in {time.time()-t0:.0f}s | total positives={tot_pos} "
          f"({tot_pos/len(exceptions):.0f}/scenario)", flush=True)

    # validation: except_0 eval-reserved rows vs normal (generalization target)
    sp0 = load_sp(args.porto, fam, "except_0")
    perm0 = np.random.RandomState(SPLIT_SEED).permutation(len(sp0))
    val_pos = [list(map(int, sp0[i])) for i in perm0[:EVAL_RESERVE][:500] if len(sp0[i]) >= 2]
    A0 = load_A(args.porto, fam, "except_0")
    dr0 = (A0.sum(1) / deg_norm).float()

    jobs = []
    for spec in args.jobs.split(","):
        parts = spec.split(":")
        B, pool = parts[0], parts[1]
        sfx = parts[2] if len(parts) > 2 else ""
        jobs.append((int(B), pool, sfx))

    eps = args.label_smooth
    half = args.bs // 2

    for (B, pool_path, sfx) in jobs:
        pool_paths = torch.load(pool_path)["paths"]
        print(f"\n[blk{B}] neg pool={len(pool_paths)} ({pool_path})", flush=True)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        disc = BDDiscriminator(n_vertex, device, logit_bound=args.logit_bound,
                               pretrain_path="./sets_data/porto_node2vec.pkl")
        opt = torch.optim.Adam(disc.parameters(), lr=args.lr)
        t1 = time.time()
        run_loss, run_acc = 0.0, 0.0
        for step in range(1, args.steps + 1):
            e = exceptions[int(rng.integers(len(exceptions)))]
            pos = [make_partial(exc_paths[e][int(rng.integers(len(exc_paths[e])))], rng) for _ in range(half)]
            negs = [make_partial(pool_paths[int(rng.integers(len(pool_paths)))], rng) for _ in range(half)]
            seqs = pos + negs
            labels = torch.cat([torch.full((half,), 1.0 - eps), torch.full((half,), eps)]).to(device)
            tokens, lengths = pad_batch(seqs, disc.PAD, device, max_len=128)
            logits = disc(tokens, lengths, exc_adj[e].to(device), exc_dr[e].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                pred = (logits > 0).float()
                true = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)
                run_acc += float((pred == true).float().mean()); run_loss += float(loss)
            if step % 500 == 0 or step == args.steps:
                disc.eval()
                with torch.no_grad():
                    vp = [make_partial(val_pos[int(rng.integers(len(val_pos)))], rng) for _ in range(128)]
                    vn = [make_partial(pool_paths[int(rng.integers(len(pool_paths)))], rng) for _ in range(128)]
                    vt, vl = pad_batch(vp + vn, disc.PAD, device, max_len=128)
                    vlog = disc(vt, vl, A0.to(device), dr0.to(device))
                    vacc = float(((vlog > 0).float() == torch.cat(
                        [torch.ones(128), torch.zeros(128)]).to(device)).float().mean())
                    vstd = float(vlog.std())
                disc.train()
                print(f"  blk{B} step {step}: loss={run_loss/500:.4f} acc={run_acc/500:.3f} | "
                      f"val(e0-vs-model) acc={vacc:.3f} logit_std={vstd:.2f} | {time.time()-t1:.0f}s", flush=True)
                run_loss, run_acc = 0.0, 0.0
        disc.eval()
        name = f"BDdisc_f{fam}_p{int(args.frac)}_e99_model_blk{B}{sfx}"
        torch.save(disc, join(args.out, f"{name}.pth"))
        print(f"  saved {name}.pth", flush=True)
    print("MULTI_DONE", flush=True)

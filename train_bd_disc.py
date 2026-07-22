"""
Train the partial-path discriminator for BD importance-weight guidance.

  python train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg data

-exp e0  : positives = except_0 paths   (experiment 1)
-exp e99 : positives = except_1..99     (experiment 2; per-batch scenario)
-neg data: negatives = normal real paths
-neg mix : negatives = 50% normal real + 50% BD model samples (neg pool)

Data protocol (BD_GUIDANCE_FORMULATION.pdf §6):
  - except_e paths: RandomState(777+e).permutation; for e = 0 the first 1000
    indices are RESERVED FOR EVALUATION; training draws the first frac% of the
    REMAINING pool. For e >= 1 the first frac% of the permutation is used.
  - every example is a random-truncation canvas-form partial path
    [dst, v0, ..., v_j]; truncation resampled every batch.
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
EVAL_RESERVE = 1000  # except_0 rows reserved for guidance evaluation


def load_sp(porto, family, tag):
    return pickle.load(open(join(porto, f"porto_shrink_SP_v4-{family}_{tag}.pkl"), "rb"))


def load_A(porto, family, tag):
    A = pickle.load(open(join(porto, f"porto_shrink_A_v4-{family}_{tag}.ts"), "rb"))
    return A.bool().float()


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
    ap.add_argument("-exp", type=str, default="e0", choices=["e0", "e99"])
    ap.add_argument("-neg", type=str, default="data", choices=["data", "mix", "model"])
    ap.add_argument("-steps", type=int, default=4000)
    ap.add_argument("-bs", type=int, default=128)
    ap.add_argument("-lr", type=float, default=1e-3)
    ap.add_argument("-label_smooth", type=float, default=0.05)
    ap.add_argument("-logit_bound", type=float, default=4.0)
    ap.add_argument("-seed", type=int, default=1)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-out", type=str, default="./sets_disc")
    ap.add_argument("-pool", type=str, default="./sets_disc/neg_pool.pth")
    ap.add_argument("-max_pos_total", type=int, default=0,
                    help="cap total positive pool (split evenly across scenarios) to balance vs the negative pool; 0=off")
    ap.add_argument("-outsfx", type=str, default="", help="suffix appended to output filename")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import os
    os.makedirs(args.out, exist_ok=True)

    fam = args.family
    A_norm = load_A(args.porto, fam, "normal")
    deg_norm = A_norm.sum(1).clamp(min=1.0)
    normal_paths = [list(map(int, p)) for p in load_sp(args.porto, fam, "normal") if len(p) >= 2]

    exceptions = [0] if args.exp == "e0" else list(range(1, 100))
    exc_paths, exc_adj, exc_dr = {}, {}, {}
    for e in exceptions:
        sp = load_sp(args.porto, fam, f"except_{e}")
        idx = except_train_split(len(sp), e, args.frac)
        exc_paths[e] = [list(map(int, sp[i])) for i in idx if len(sp[i]) >= 2]
        A_e = load_A(args.porto, fam, f"except_{e}")
        exc_adj[e] = A_e
        exc_dr[e] = (A_e.sum(1) / deg_norm).float()
    if args.max_pos_total > 0:
        per_exc = max(1, args.max_pos_total // len(exceptions))
        for e in exceptions:
            exc_paths[e] = exc_paths[e][:per_exc]
        print(f"[disc] balanced positives: capped to {per_exc}/scenario "
              f"-> total {sum(len(v) for v in exc_paths.values())}")
    n_vertex = A_norm.shape[0]
    print(f"[disc] fam={fam} frac={args.frac}% exp={args.exp} neg={args.neg} | "
          f"exceptions={len(exceptions)}, paths/exc={np.mean([len(v) for v in exc_paths.values()]):.0f}, "
          f"normal pool={len(normal_paths)}")

    pool_seqs = pool_paths = None
    if args.neg == "mix":
        pool_seqs = torch.load(args.pool)["sequences"]
        print(f"[disc] model-sample pool: {len(pool_seqs)}")
    elif args.neg == "model":
        # pure model-distribution negatives (Eq. 2 denominator = p_theta):
        # RAW unconditional paths, partial-cut like the positives
        pool_paths = torch.load(args.pool)["paths"]
        print(f"[disc] unconditional model pool: {len(pool_paths)}")

    # validation: except_0 eval-reserved rows vs normal (generalization target)
    sp0 = load_sp(args.porto, fam, "except_0")
    perm0 = np.random.RandomState(SPLIT_SEED).permutation(len(sp0))
    val_pos = [list(map(int, sp0[i])) for i in perm0[:EVAL_RESERVE][:500] if len(sp0[i]) >= 2]
    A0 = load_A(args.porto, fam, "except_0")
    dr0 = (A0.sum(1) / deg_norm).float()

    disc = BDDiscriminator(n_vertex, device, logit_bound=args.logit_bound,
                           pretrain_path="./sets_data/porto_node2vec.pkl")
    opt = torch.optim.Adam(disc.parameters(), lr=args.lr)
    eps = args.label_smooth
    half = args.bs // 2

    t0 = time.time()
    run_loss, run_acc = 0.0, 0.0
    for step in range(1, args.steps + 1):
        e = exceptions[int(rng.integers(len(exceptions)))]
        pos = [make_partial(exc_paths[e][int(rng.integers(len(exc_paths[e])))], rng) for _ in range(half)]
        negs = []
        for _ in range(half):
            if pool_paths is not None:
                negs.append(make_partial(pool_paths[int(rng.integers(len(pool_paths)))], rng))
            elif pool_seqs is not None and rng.random() < 0.5:
                negs.append(list(pool_seqs[int(rng.integers(len(pool_seqs)))]))
            else:
                negs.append(make_partial(normal_paths[int(rng.integers(len(normal_paths)))], rng))
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
                vn = [make_partial(normal_paths[int(rng.integers(len(normal_paths)))], rng) for _ in range(128)]
                vt, vl = pad_batch(vp + vn, disc.PAD, device, max_len=128)
                vlog = disc(vt, vl, A0.to(device), dr0.to(device))
                vacc = float(((vlog > 0).float() == torch.cat(
                    [torch.ones(128), torch.zeros(128)]).to(device)).float().mean())
                vstd = float(vlog.std())
            disc.train()
            print(f"step {step}: loss={run_loss/500:.4f} acc={run_acc/500:.3f} | "
                  f"val(except_0) acc={vacc:.3f} logit_std={vstd:.2f} | {time.time()-t0:.0f}s")
            run_loss, run_acc = 0.0, 0.0

    disc.eval()
    name = f"BDdisc_f{fam}_p{int(args.frac)}_{args.exp}_{args.neg}"
    if args.neg == "model":
        import re as _re
        m = _re.search(r"blk(\d+)", args.pool)
        g = _re.search(r"graph(\d+)", args.pool)
        if m:
            name += f"_blk{m.group(1)}"
        elif g:
            name += f"_graph{g.group(1)}"
    name += args.outsfx
    torch.save(disc, join(args.out, f"{name}.pth"))
    print(f"saved {join(args.out, f'{name}.pth')}")

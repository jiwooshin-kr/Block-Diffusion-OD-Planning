"""
Discriminator generalization diagnostic: does the unseen (e99) disc learn the
same function as the seen (e0) disc, or does it merely look similar through
the downstream guidance pipeline?

For each block size, two symmetric probes on held-out data:
  scenario 0 : positives = except_0 eval-reserved rows (never in ANY disc's
               training); zero-shot for the e99 disc, in-scenario for e0.
  scenario 1 : positives = except_1 rows OUTSIDE the 1% training slice;
               in-scenario (held-out) for e99, zero-shot for e0.
Negatives are FRESH unconditional generations (seed-99 pool, disjoint from
the seed-1 20k training pools). Metrics per (scenario, disc): AUC and
accuracy@0; plus Pearson/Spearman correlation between the two discs' logits
on identical inputs -- direct evidence of learning the same ratio.

With -neg data, the block-INDEPENDENT data-negative discs (e0_data / e99_data,
negatives = normal real paths) are evaluated instead: positives as above,
negatives = normal real paths (fresh seed-555 sample; training drew negatives
uniformly from the same 1.9M-path normal pool, so no disjoint split exists --
the sample is fresh truncations, and both discs face the identical set).
This removes the per-block negative-pool confound entirely.

  python disc_gen_eval.py -family 0.05
  python disc_gen_eval.py -family 0.1
  python disc_gen_eval.py -family 0.05 -neg data
"""

import argparse
import json
import pickle
from os.path import join

import numpy as np
import torch

from models_seq.bd_disc import make_partial, pad_batch

SPLIT_SEED = 777
EVAL_RESERVE = 1000


def load_sp(porto, fam, tag):
    return pickle.load(open(join(porto, f"porto_shrink_SP_v4-{fam}_{tag}.pkl"), "rb"))


def load_A(porto, fam, tag):
    return pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_{tag}.ts"), "rb")).bool().float()


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    r = np.empty(len(x))
    order = x.argsort()
    r[order] = np.arange(1, len(x) + 1)
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def score(disc, seqs, A_dev, dr_dev, device, bs=256):
    out = []
    for s in range(0, len(seqs), bs):
        tok, lens = pad_batch(seqs[s:s + bs], disc.PAD, device, max_len=128)
        out.append(disc(tok, lens, A_dev, dr_dev).float().cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-neg", type=str, default="model", choices=["model", "data"],
                    help="model: per-block model-negative discs vs fresh uncond pools; "
                         "data: block-independent data-negative discs vs normal real paths")
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-frac", type=float, default=1.0, help="training frac (%) used by the discs")
    ap.add_argument("-n_eval", type=int, default=1000)
    ap.add_argument("-seed", type=int, default=123, help="truncation rng seed")
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-disc_dir", type=str, default="./sets_disc")
    ap.add_argument("-pool_pat", type=str, default="./sets_disc/geneval_pool_blk{B}.pth")
    ap.add_argument("-sfx", type=str, default="", help="disc filename suffix (e.g. _v4)")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fam = args.family
    A_norm = load_A(args.porto, fam, "normal")
    deg_norm = A_norm.sum(1).clamp(min=1.0)

    # ---- held-out positives per probe scenario --------------------------
    probes = {}
    # scenario 0: the eval-reserved rows (identical reconstruction to eval scripts)
    sp0 = load_sp(args.porto, fam, "except_0")
    perm0 = np.random.RandomState(SPLIT_SEED).permutation(len(sp0))
    pos0 = [list(map(int, sp0[i])) for i in perm0[:EVAL_RESERVE] if len(sp0[i]) >= 2][:args.n_eval]
    probes[0] = pos0
    # scenario 1: rows outside the 1% training slice (same split fn as training)
    sp1 = load_sp(args.porto, fam, "except_1")
    perm1 = np.random.RandomState(SPLIT_SEED + 1).permutation(len(sp1))
    n_tr = max(1, int(np.ceil(args.frac / 100.0 * len(sp1))))
    pos1 = [list(map(int, sp1[i])) for i in perm1[n_tr:] if len(sp1[i]) >= 2][:args.n_eval]
    probes[1] = pos1

    A_scn = {e: load_A(args.porto, fam, f"except_{e}") for e in probes}
    dr_scn = {e: (A_scn[e].sum(1) / deg_norm).float() for e in probes}

    # ---- evaluation units: (label, disc paths, negatives) ---------------
    units = []
    if args.neg == "data":
        # block-independent data-negative discs; negatives = normal real paths
        sp_norm = load_sp(args.porto, fam, "normal")
        idxn = np.random.RandomState(555).choice(len(sp_norm), args.n_eval * 2, replace=False)
        neg_norm = [list(map(int, sp_norm[i])) for i in idxn if len(sp_norm[i]) >= 2][:args.n_eval]
        units.append(("data", {
            "seen": join(args.disc_dir, f"BDdisc_f{fam}_p{int(args.frac)}_e0_data{args.sfx}.pth"),
            "unseen": join(args.disc_dir, f"BDdisc_f{fam}_p{int(args.frac)}_e99_data{args.sfx}.pth"),
        }, neg_norm))
    else:
        for blk in [int(b) for b in args.blocks.split(",")]:
            neg_raw = [list(map(int, p)) for p in torch.load(args.pool_pat.format(B=blk))["paths"]
                       if len(p) >= 2][:args.n_eval]
            units.append((f"blk{blk}", {
                "seen": join(args.disc_dir, f"BDdisc_f{fam}_p{int(args.frac)}_e0_model_blk{blk}{args.sfx}.pth"),
                "unseen": join(args.disc_dir, f"BDdisc_f{fam}_p{int(args.frac)}_e99_model_blk{blk}{args.sfx}.pth"),
            }, neg_raw))

    results = {}
    for label, disc_paths, neg_raw in units:
        discs = {}
        for kind, path in disc_paths.items():
            discs[kind] = torch.load(path, map_location=device)
            discs[kind].eval()

        for e, pos_raw in probes.items():
            rng = np.random.default_rng(args.seed)
            pos = [make_partial(p, rng) for p in pos_raw]
            neg = [make_partial(p, rng) for p in neg_raw]
            A_dev, dr_dev = A_scn[e].to(device), dr_scn[e].to(device)
            logits = {k: score(d, pos + neg, A_dev, dr_dev, device) for k, d in discs.items()}
            np_, nn_ = len(pos), len(neg)
            row = {}
            for k in discs:
                lp, ln = logits[k][:np_], logits[k][np_:]
                row[f"auc_{k}"] = auc(lp, ln)
                row[f"acc_{k}"] = float(((lp > 0).sum() + (ln <= 0).sum()) / (np_ + nn_))
            row["pearson"] = float(np.corrcoef(logits["seen"], logits["unseen"])[0, 1])
            row["spearman"] = spearman(logits["seen"], logits["unseen"])
            results[f"{label}_exc{e}"] = row
            print(f"fam={fam} {label:<6} exc={e} | "
                  f"AUC seen={row['auc_seen']:.3f} unseen={row['auc_unseen']:.3f} | "
                  f"acc seen={row['acc_seen']:.3f} unseen={row['acc_unseen']:.3f} | "
                  f"logit corr pearson={row['pearson']:.3f} spearman={row['spearman']:.3f}",
                  flush=True)

    sfx = "" if args.neg == "model" else "_data"
    out = join(args.res_path, f"disc_gen_eval_f{fam.replace('.', '')}{sfx}{args.sfx}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"written {out}")
    print("DISC_GEN_EVAL_DONE", flush=True)

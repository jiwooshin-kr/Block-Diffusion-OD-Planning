"""
Instrument D-CBG (mask kernel, exact enumeration) to measure how much the
classifier tilt actually MOVES the reveal distribution, per reveal step:

  p0 = softmax(log p_theta)              (unguided reveal distribution)
  pg = softmax(log p_theta + g * clf_lp) (guided, for a grid of gammas)

Recorded per reveal step (active samples only):
  - TV(p0, pg) and argmax-flip for each gamma, under BOTH classifier time
    scales: t*100 (what tools/eval/eval_dcbg.py currently feeds -- 100x outside the
    classifier's training range t in (0,1)) and t*1 (the training scale),
    to quantify the train/inference t-scale mismatch.
  - spread of clf_lp vs spread of log p_theta over the top-10 candidates
    (guidance can only act where the model is near-tied; this ratio is the
    "leverage" the classifier has).
  - entropy / top-1 prob of p0.

The trajectory itself is advanced with gamma_run (default 1.0) at t*100,
i.e. exactly the as-evaluated sampler, so the visited states match eval.

  python tools/diag/diag_dcbg_leverage.py -blk 4 -clf sets_disc/DCBGclf_mask_blk4_f0.05_p1_adj_modelneg_v4.pth -adj 1 -n 60
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse
import pickle
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

from dcbg_plugin import AdjBound
from models_seq.bd_models import get_block_causal_mask

SPLIT_SEED = 777
GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]


@torch.no_grad()
def clf_enum(clf, seq, pos, t_cond, V, micro_bs=4096):
    """Exact enumeration at the reveal position: log p_phi(y=1 | x_t^{pos->k})
    for all k. t_cond: (b,) ALREADY at the scale to feed the classifier."""
    b, s = seq.shape
    arange = torch.arange(b, device=seq.device)
    xt_jumps = seq.unsqueeze(1).repeat(1, V, 1)
    kk = torch.arange(V, device=seq.device)[None, :].expand(b, V)
    xt_jumps[arange[:, None], kk, pos[:, None].expand(b, V)] = kk
    flat = xt_jumps.view(b * V, s)
    t_rep = t_cond.repeat_interleave(V)
    out = torch.empty(b * V, device=seq.device)
    for lo in range(0, b * V, micro_bs):
        hi = min(lo + micro_bs, b * V)
        out[lo:hi] = clf.get_log_probs(flat[lo:hi], t_rep[lo:hi])[:, 1]
    return out.view(b, V)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blk", type=int, default=4)
    ap.add_argument("-clf", type=str, required=True)
    ap.add_argument("-ckpt", type=str, default="")
    ap.add_argument("-adj", type=int, default=1)
    ap.add_argument("-gamma_run", type=float, default=1.0)
    ap.add_argument("-order", type=str, default="l2r", choices=["first_hit", "l2r"])
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    fam = args.family
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool()
    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.n]

    ckpt = args.ckpt or f"./sets_model/BD_porto_v3_normal_mask_blk{args.blk}_v4_bd.pth"
    model = torch.load(ckpt, map_location=device)
    model.eval()
    clf = torch.load(args.clf, map_location=device)
    clf.eval()
    if args.adj:
        dr_dev = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
        clf = AdjBound(clf, A_exc.float().to(device), dr_dev)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    origs = torch.tensor([p[0] for p in real]).long().to(device)
    dests = torch.tensor([p[-1] for p in real]).long().to(device)
    b = origs.shape[0]
    block = model.block_size
    pfx = model.pfx
    max_blocks = model.bd_max_len // block
    V = model.backbone.vocab_size
    stop_idx = torch.full((b,), -1, dtype=torch.long)
    end_idx = torch.full((b,), -1, dtype=torch.long)
    seq = torch.empty(b, 0, dtype=torch.long, device=model.device)
    arange = torch.arange(b, device=model.device)

    gam = torch.tensor(GAMMAS, device=device).view(-1, 1, 1)   # (G,1,1)
    REC = {k: [] for k in ["tv", "flip", "clf_std_full", "clf_rng_top10",
                           "logp_rng_top10", "p0_top1", "p0_ent", "t_val",
                           "corr_scale", "step_in_block", "block_j"]}
    example = None

    t0 = time.time()
    with torch.no_grad():
        for j in range(max_blocks):
            seq = torch.cat([seq, model._init_block(b)], dim=1)
            s = seq.shape[1]
            for p_idx, val in [(0, dests), (1, origs)]:
                if s - block <= p_idx < s:
                    seq[:, p_idx] = val
            if s > pfx:
                attn = get_block_causal_mask(s, block, model.device)
                t = torch.ones(b, device=model.device)
                step_in_block = 0
                while True:
                    masked = seq[:, -block:] == model.MASK
                    active = masked.any(dim=1)
                    if not bool(active.any()):
                        break
                    num_masked = masked.sum(dim=1).clamp(min=1)
                    u = torch.rand(b, device=model.device)
                    t = t * u ** (1.0 / num_masked.float())

                    logits = model._cfg_logits(seq, model._t_input(b, s, t * 100.0),
                                               origs, dests, attn, 1.0)
                    block_logits = logits[:, -block:].clone()
                    block_logits[..., model.MASK] = -1e9
                    log_p = block_logits.log_softmax(dim=-1)

                    if args.order == "l2r":
                        idx = masked.float().argmax(dim=1)
                    else:
                        sel = masked.float()
                        sel[~active] = 1.0
                        idx = torch.multinomial(sel, 1).squeeze(1)
                    pos = s - block + idx

                    lp = log_p[arange, idx]                       # (b, V)
                    clf100 = clf_enum(clf, seq, pos, t * 100.0, V)
                    clf001 = clf_enum(clf, seq, pos, t, V)

                    # ---------- instrumentation ----------
                    lp_m = lp.clone(); lp_m[:, model.MASK] = -1e9
                    p0 = lp_m.softmax(-1)                          # (b, V)
                    act = active.cpu().numpy()
                    for si, cl in enumerate([clf100, clf001]):
                        g_lp = lp_m[None] + gam * cl[None]         # (G, b, V)
                        g_lp[..., model.MASK] = -1e9
                        pg = g_lp.softmax(-1)
                        tv = 0.5 * (pg - p0[None]).abs().sum(-1)   # (G, b)
                        flip = (pg.argmax(-1) != p0.argmax(-1)).float()
                        if si == 0:
                            tv_all = torch.empty(2, len(GAMMAS), b, device=device)
                            fl_all = torch.empty_like(tv_all)
                        tv_all[si], fl_all[si] = tv, flip
                    top10 = lp_m.topk(10, dim=-1).indices          # (b, 10)
                    c100_t = clf100.gather(1, top10)
                    c001_t = clf001.gather(1, top10)
                    lp_t = lp_m.gather(1, top10)
                    valid = torch.ones(V, dtype=torch.bool, device=device)
                    valid[model.MASK] = False
                    cf, cf1 = clf100[:, valid], clf001[:, valid]
                    corr = F.cosine_similarity(cf - cf.mean(1, keepdim=True),
                                               cf1 - cf1.mean(1, keepdim=True), dim=1)
                    REC["tv"].append(tv_all.permute(2, 0, 1).cpu().numpy()[act])
                    REC["flip"].append(fl_all.permute(2, 0, 1).cpu().numpy()[act])
                    REC["clf_std_full"].append(torch.stack(
                        [cf.std(1), cf1.std(1)], 1).cpu().numpy()[act])
                    REC["clf_rng_top10"].append(torch.stack(
                        [c100_t.max(1).values - c100_t.min(1).values,
                         c001_t.max(1).values - c001_t.min(1).values], 1).cpu().numpy()[act])
                    REC["logp_rng_top10"].append(
                        (lp_t.max(1).values - lp_t.min(1).values).cpu().numpy()[act])
                    REC["p0_top1"].append(p0.max(-1).values.cpu().numpy()[act])
                    REC["p0_ent"].append((-(p0 * p0.clamp(min=1e-12).log()).sum(-1)).cpu().numpy()[act])
                    REC["t_val"].append(t.cpu().numpy()[act])
                    REC["corr_scale"].append(corr.cpu().numpy()[act])
                    REC["step_in_block"].append(np.full(int(act.sum()), step_in_block))
                    REC["block_j"].append(np.full(int(act.sum()), j))
                    if example is None and j >= 1 and bool(active[0]):
                        k10 = top10[0].cpu().numpy()
                        example = dict(idx=k10, logp=lp_t[0].cpu().numpy(),
                                       clf100=c100_t[0].cpu().numpy(),
                                       clf001=c001_t[0].cpu().numpy(),
                                       t=float(t[0]))
                    # ---------- advance trajectory (as evaluated: gamma_run, t*100) ----------
                    guided = lp + args.gamma_run * clf100
                    guided[:, model.MASK] = -1e9
                    tok = torch.multinomial(guided.softmax(dim=-1), 1).squeeze(1)
                    seq[arange[active], pos[active]] = tok[active]
                    step_in_block += 1

            lo2 = max(pfx, s - block)
            bt = seq[:, lo2:s].cpu()
            for i in range(b):
                if stop_idx[i] >= 0 or end_idx[i] >= 0:
                    continue
                d = int(dests[i].item())
                for k, tok_ in enumerate(bt[i].tolist()):
                    if tok_ == d:
                        stop_idx[i] = lo2 + k
                        break
                    if tok_ in (model.END, model.PAD, model.MASK):
                        end_idx[i] = lo2 + k
                        break
            if bool(((stop_idx >= 0) | (end_idx >= 0)).all()):
                break
            print(f"  block {j}: {int(((stop_idx>=0)|(end_idx>=0)).sum())}/{b} done "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out = {k: np.concatenate(v, axis=0) for k, v in REC.items()}
    out["gammas"] = np.array(GAMMAS)
    for k, v in (example or {}).items():
        out[f"ex_{k}"] = v
    path = args.out or f"./sets_res/dcbg_leverage_blk{args.blk}_g{args.gamma_run}.npz"
    np.savez(path, **out)
    n = out["tv"].shape[0]
    print(f"\nsaved {path}  ({n} reveal-step records, {time.time()-t0:.0f}s)")
    for si, nm in enumerate(["t*100 (as evaluated)", "t*1 (training scale)"]):
        print(f"\n-- classifier t scale: {nm} --")
        print(f"{'gamma':>8} {'medTV':>8} {'meanTV':>8} {'flip%':>7}")
        for gi, g in enumerate(GAMMAS):
            tv = out["tv"][:, si, gi]
            fl = out["flip"][:, si, gi]
            print(f"{g:>8.1f} {np.median(tv):>8.4f} {tv.mean():>8.4f} {100*fl.mean():>6.2f}%")
    print(f"\nclf spread (std over vocab)  t*100: {out['clf_std_full'][:,0].mean():.4f}"
          f"   t*1: {out['clf_std_full'][:,1].mean():.4f}")
    print(f"clf range over top-10 cand   t*100: {out['clf_rng_top10'][:,0].mean():.4f}"
          f"   t*1: {out['clf_rng_top10'][:,1].mean():.4f}")
    print(f"log_p range over top-10 cand       : {out['logp_rng_top10'].mean():.4f}")
    print(f"corr(clf@t*100, clf@t*1) over vocab: {out['corr_scale'].mean():.4f}")
    print(f"p0 top-1 prob: median {np.median(out['p0_top1']):.4f}   "
          f"entropy: median {np.median(out['p0_ent']):.4f}")
    print("DIAG_DONE", flush=True)

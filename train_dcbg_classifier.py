"""
Train the D-CBG noise-conditioned classifier p(y=exceptional | x_t, t)
following Schiff et al.'s protocol: LABELED DATA (exceptional vs normal),
corrupted by the same forward process the diffusion model uses at
generation time. Same 1% data budget / split seeds as train_bd_disc.py.

  python train_dcbg_classifier.py -kernel mask  -blk 4
  python train_dcbg_classifier.py -kernel mask  -blk 64
  python train_dcbg_classifier.py -kernel graph -blk 64 \
      -graph_ckpt sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth
"""

import argparse
import pickle
import time
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

from dcbg_plugin import DCBGClassifier, DCBGClassifierAdj, corrupt_mask, corrupt_graph

SPLIT_SEED = 777
EVAL_RESERVE = 1000

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-kernel", type=str, required=True, choices=["mask", "graph"])
    ap.add_argument("-blk", type=int, required=True)
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-frac", type=float, default=1.0)
    ap.add_argument("-graph_ckpt", type=str, default="")
    ap.add_argument("-adj", type=int, default=0, help="1 = adjacency-aware classifier (matched to BDDiscriminator)")
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

    matrices, max_T = None, 100
    if args.kernel == "graph":
        m = torch.load(args.graph_ckpt, map_location="cpu")
        matrices, max_T = m.matrices.cpu(), m.max_T
        del m

    n_vertex = 1390
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool().float()
    A_nrm = pickle.load(open(join(args.porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool().float()
    adj_dev = A_exc.to(device)
    dr_dev = (A_exc.sum(1) / A_nrm.sum(1).clamp(min=1)).to(device)
    if args.adj:
        clf = DCBGClassifierAdj(n_vertex, device, pretrain_path="./sets_data/porto_node2vec.pkl")
    else:
        clf = DCBGClassifier(n_vertex, device, pretrain_path="./sets_data/porto_node2vec.pkl")
    END, PAD, MASK = clf.END, clf.PAD, clf.MASK
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
    half = args.bs // 2

    def corrupt(paths):
        if args.kernel == "mask":
            return corrupt_mask(paths, args.blk, rng, END, PAD, MASK)
        return corrupt_graph(paths, matrices, max_T, rng, END, PAD, block=args.blk)

    t0 = time.time()
    run_loss, run_acc = 0.0, 0.0
    for step in range(1, args.steps + 1):
        pos = [pos_paths[int(rng.integers(len(pos_paths)))] for _ in range(half)]
        neg = [neg_paths[int(rng.integers(len(neg_paths)))] for _ in range(half)]
        tok, ts = corrupt(pos + neg)
        tok, ts = tok.to(device), ts.to(device)
        eps_ls = 0.05 if args.adj else 0.0
        labels = torch.cat([torch.full((half,), 1.0 - eps_ls), torch.full((half,), eps_ls)]).to(device)
        z = clf(tok, ts, adj_dev, dr_dev) if args.adj else clf(tok, ts)
        loss = F.binary_cross_entropy_with_logits(z, labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        opt.step()
        run_loss += float(loss)
        hard = torch.cat([torch.ones(half), torch.zeros(half)]).to(device)
        run_acc += float(((z > 0).float() == hard).float().mean())

        if step % 500 == 0 or step == args.steps:
            clf.eval()
            with torch.no_grad():
                vp = [val_pos[int(rng.integers(len(val_pos)))] for _ in range(128)]
                vn = [neg_paths[int(rng.integers(len(neg_paths)))] for _ in range(128)]
                vt, vts = corrupt(vp + vn)
                vz = (clf(vt.to(device), vts.to(device), adj_dev, dr_dev) if args.adj else clf(vt.to(device), vts.to(device)))
                vacc = float(((vz > 0).float() == torch.cat(
                    [torch.ones(128), torch.zeros(128)]).to(device)).float().mean())
            clf.train()
            d = 500 if step % 500 == 0 else step % 500
            print(f"step {step}: loss={run_loss/d:.4f} acc={run_acc/d:.3f} | "
                  f"val acc={vacc:.3f} | {time.time()-t0:.0f}s", flush=True)
            run_loss, run_acc = 0.0, 0.0

    clf.eval()
    sfx = "_adj" if args.adj else ""
    out = f"./sets_disc/DCBGclf_{args.kernel}_blk{args.blk}_f{fam}_p{int(args.frac)}{sfx}.pth"
    torch.save(clf, out)
    print(f"saved {out}")

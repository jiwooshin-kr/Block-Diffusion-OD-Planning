"""
Three-way deep dive (base / modelD-guided / adj+modelD) x post-processing
(raw / P1 / P3 / P1+P3), scored on except_0 with EM / PC / invalid edge &
node ratios. Works for both kernels (plan_guided dispatches on model.kernel).

  python tools/eval/three_way_postproc.py -ckpt sets_model/..._bd.pth \
      -disc sets_disc/BDdisc_..._model_blk64.pth -tag mask64
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse
import pickle
import numpy as np
import torch
from os.path import join

from eval_shortest import evaluate_em_pc
from postproc import splice, endpoint

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ckpt", type=str, required=True)
    ap.add_argument("-disc", type=str, required=True)
    ap.add_argument("-tag", type=str, required=True)
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-dver", type=str, default="v4", help="데이터 버전 접두 (v4 | v6)")
    ap.add_argument("-res_path", type=str, default="./sets_res", help="em_pc 레코드 저장 위치")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-n_is", type=int, default=100)
    ap.add_argument("-w_gamma", type=float, default=1.0,
                    help="IW weight exponent: SNIS targets q^g * p^(1-g) (see pdfs/temp_math.pdf)")
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-cand_mode", type=str, default="meanfield",
                    choices=["meanfield", "ancestral"],
                    help="within-block candidate proposal for the guided arms")
    ap.add_argument("-anc_correct", type=int, default=1,
                    help="ancestral only: multiply the weight by prod_j alpha_j "
                         "(the per-candidate proposal normaliser, which does NOT "
                         "cancel under self-normalisation)")
    ap.add_argument("-adjonly", type=int, default=0,
                    help="1 = also run the Lemma-3 adjacency-only control (disc=None, adj_prop=True)")
    ap.add_argument("-order", type=str, default="first_hit", choices=["first_hit", "l2r"],
                    help="mask within-block reveal order")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    porto, fam, dver = args.porto, args.family, args.dver

    A_exc = pickle.load(open(join(porto, f"porto_shrink_A_{dver}-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(porto, f"porto_shrink_A_{dver}-{fam}_normal.ts"), "rb")).bool()
    G_exc = pickle.load(open(join(porto, f"porto_shrink_G_{dver}-{fam}_except_0.pkl"), "rb"))
    sp_exc = pickle.load(open(join(porto, f"porto_shrink_SP_{dver}-{fam}_except_0.pkl"), "rb"))
    removed = A_norm & ~A_exc
    deg_ratio = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
    A_dev = A_exc.float().to(device)

    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.eval_num]

    def score(tag, paths, patch, ess=None):
        summ, _, _ = evaluate_em_pc(gen_paths=paths, A=A_exc.float(), shortest_paths=sp_exc,
                                    save_dir=join(args.res_path, "em_pc"), prefix=tag)
        arr = float(np.mean([len(q) > 0 and q[-1] == g[-1] for q, g in zip(paths, real)]))
        bad_e, tot_e, bad_n, tot_n, rem_e = 0, 0, 0, 0, 0
        for p in paths:
            marked = set()
            for i, (u, v) in enumerate(zip(p[:-1], p[1:])):
                tot_e += 1
                if not A_exc[u, v]:
                    bad_e += 1
                    marked.add(i)
                    marked.add(i + 1)
                    if removed[u, v]:
                        rem_e += 1
            bad_n += len(marked)
            tot_n += len(p)
        es = f" ess={ess:.1f}" if ess is not None else ""
        print(f"{tag:<30} arr={arr:.3f} valid={summ['valid_rate']:.3f} em={summ['em_score']:.3f} "
              f"pc={summ['pc_score']:.3f} invE={100*bad_e/max(tot_e,1):.2f}% invN={100*bad_n/max(tot_n,1):.2f}% "
              f"remE={100*rem_e/max(tot_e,1):.2f}% patch={np.mean(patch):.2f}{es}", flush=True)

    disc = torch.load(args.disc, map_location=device)
    disc.eval()
    model = torch.load(args.ckpt, map_location=device)
    model.eval()

    CFGS = [("base", None), ("modelD", dict(adj_prop=False)), ("adj+modelD", dict(adj_prop=True))]
    if args.adjonly:
        # Lemma-3 unguided control: adjacency-constrained proposals, NO discriminator.
        # Isolates the masking from the reweighting (the third ablation arm).
        CFGS.append(("adjonly", dict(adj_prop=True, _no_disc=True)))
    for cfg, kw in CFGS:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        planned, ess_log = [], []
        for s in range(0, len(real), args.batch):
            b = real[s:s + args.batch]
            o, d = [p[0] for p in b], [p[-1] for p in b]
            if kw is None:
                out = model.plan(o, d, use_refine=False, order=args.order)
            else:
                kw2 = {k: v for k, v in kw.items() if not k.startswith("_")}
                out = model.plan_guided(o, d, None if kw.get("_no_disc") else disc,
                                        A_dev, deg_ratio, n_is=args.n_is,
                                        w_gamma=args.w_gamma,
                                        ess_log=ess_log, order=args.order,
                                        cand_mode=args.cand_mode,
                                        anc_correct=bool(args.anc_correct), **kw2)
            planned += out
        ess = float(np.nanmean(ess_log)) if ess_log else None
        # P1 = validity repair, P3 = arrival repair (postproc.py). P1P3 runs P1 first
        # so that P3 extends the already-legalised tail.
        p1 = [splice(p, A_exc, G_exc) for p in planned]
        p3 = [endpoint(p, g[-1], G_exc) for p, g in zip(planned, real)]
        p13 = []
        for (q, a1), g in zip(p1, real):
            q2, a2 = endpoint(q, g[-1], G_exc)
            p13.append((q2, a1 + a2))
        score(f"{args.tag}_{cfg}_raw", planned, [0] * len(real), ess)
        score(f"{args.tag}_{cfg}_P1", [q for q, _ in p1], [a for _, a in p1])
        score(f"{args.tag}_{cfg}_P3", [q for q, _ in p3], [a for _, a in p3])
        score(f"{args.tag}_{cfg}_P1P3", [q for q, _ in p13], [a for _, a in p13])
        print(flush=True)
    print("THREEWAY_DONE", flush=True)

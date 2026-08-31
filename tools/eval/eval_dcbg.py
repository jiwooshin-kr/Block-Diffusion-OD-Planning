"""
Evaluate D-CBG guidance (Schiff et al. plug-in) on except_0 with the same
protocol and metrics as tools/eval/three_way_postproc.py (raw + P1/P3/P1+P3).

  python tools/eval/eval_dcbg.py -kernel mask -blk 4 -gamma 2.0 \
      -clf sets_disc/DCBGclf_mask_blk4_f0.05_p1.pth
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

from eval_shortest import evaluate_em_pc
from postproc import splice, endpoint
from dcbg_plugin import plan_dcbg_mask, plan_dcbg_graph, AdjBound

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-kernel", type=str, required=True, choices=["mask", "graph"])
    ap.add_argument("-blk", type=int, required=True)
    ap.add_argument("-gamma", type=float, required=True)
    ap.add_argument("-clf", type=str, required=True)
    ap.add_argument("-adj", type=int, default=0, help="1 = adjacency-aware classifier")
    ap.add_argument("-approx", type=int, default=0, help="1 = first-order (Taylor) D-CBG")
    ap.add_argument("-adj_prop", type=int, default=0,
                    help="1 = Lemma-3 adjacency legality mask on the guided reveal (mask kernel), same rule as the IW side's adj_prop")
    ap.add_argument("-order", type=str, default="first_hit", choices=["first_hit", "l2r"],
                    help="within-block reveal order; same rule as the IW side")
    ap.add_argument("-res_suffix", type=str, default="")
    ap.add_argument("-ckpt", type=str, default="")
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=100)
    ap.add_argument("-seed", type=int, default=7)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-dver", type=str, default="v4", help="데이터 버전 접두 (v4 | v6)")
    ap.add_argument("-res_path", type=str, default="./sets_res", help="em_pc 레코드 저장 위치")
    ap.add_argument("-tag", type=str, default="",
                    help="주면 레코드 prefix 를 {tag}_{cfg}_{variant} 로 쓴다(three_way 와 동일한 "
                         "규칙이라 collect_va_table 이 그대로 읽는다). 비우면 구 v4 이름 유지")
    ap.add_argument("-cfg", type=str, default="",
                    help="-tag 와 함께 쓰는 arm 이름. 기본 dcbg / dcbg+adjp")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    fam, dver = args.family, args.dver
    A_exc = pickle.load(open(join(args.porto, f"porto_shrink_A_{dver}-{fam}_except_0.ts"), "rb")).bool()
    A_norm = pickle.load(open(join(args.porto, f"porto_shrink_A_{dver}-{fam}_normal.ts"), "rb")).bool()
    G_exc = pickle.load(open(join(args.porto, f"porto_shrink_G_{dver}-{fam}_except_0.pkl"), "rb"))
    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_{dver}-{fam}_except_0.pkl"), "rb"))
    removed = A_norm & ~A_exc
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.eval_num]

    ckpt = args.ckpt or (f"./sets_model/BD_porto_v3_normal_mask_blk{args.blk}_base_bd.pth"
                         if args.kernel == "mask" else
                         "./sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth")
    model = torch.load(ckpt, map_location=device)
    model.eval()
    clf = torch.load(args.clf, map_location=device)
    clf.eval()
    # 분류기와 백본은 토큰 id 공간을 공유해야 한다(plan_dcbg_* 가 백본 canvas 를
    # 그대로 clf 에 먹인다). v4=1390 / v6=1387 이라 데이터 버전이 섞이면 조용히 틀린다.
    assert clf.vocab == model.backbone.vocab_size, (
        f"vocab mismatch: clf={clf.vocab} vs backbone={model.backbone.vocab_size} "
        f"(clf 학습 때의 -dver/-porto 가 백본과 다름)")
    if args.adj:
        dr_dev = (A_exc.float().sum(1) / A_norm.float().sum(1).clamp(min=1)).to(device)
        clf = AdjBound(clf, A_exc.float().to(device), dr_dev)
    plan_fn = plan_dcbg_mask if args.kernel == "mask" else plan_dcbg_graph

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    planned = []
    t0 = time.time()
    adj_scn = A_exc.float().to(device) if args.adj_prop else None
    for s in range(0, len(real), args.batch):
        bb = real[s:s + args.batch]
        planned += plan_fn(model, [p[0] for p in bb], [p[-1] for p in bb], clf, args.gamma,
                           use_approx=bool(args.approx), order=args.order, adj_scn=adj_scn)
        print(f"  {len(planned)}/{len(real)} ({time.time()-t0:.0f}s)", flush=True)

    if args.tag:
        # three_way_postproc 와 동일한 {tag}_{cfg}_{variant} 규칙 -> collect_va_table 이 그대로 읽는다
        cfg_name = args.cfg or ("dcbg+adjp" if args.adj_prop else "dcbg")
        pfx = f"{args.tag}_{cfg_name}"
    else:
        pfx = (f"DCBG_{args.kernel}{args.blk}_g{args.gamma}" + ("_adj" if args.adj else "")
               + ("_fo" if args.approx else "") + ("_adjp" if args.adj_prop else "") + args.res_suffix)

    def score(tag, paths, patch):
        summ, _, _ = evaluate_em_pc(gen_paths=paths, A=A_exc.float(), shortest_paths=sp_exc,
                                    save_dir=join(args.res_path, "em_pc"), prefix=tag)
        arr = float(np.mean([len(q) > 0 and q[-1] == g[-1] for q, g in zip(paths, real)]))
        bad_e, tot_e, rem_e = 0, 0, 0
        for p in paths:
            for u, v in zip(p[:-1], p[1:]):
                tot_e += 1
                if not A_exc[u, v]:
                    bad_e += 1
                    if removed[u, v]:
                        rem_e += 1
        print(f"{tag:<28} arr={arr:.3f} valid={summ['valid_rate']:.3f} em={summ['em_score']:.3f} "
              f"pc={summ['pc_score']:.3f} invE={100*bad_e/max(tot_e,1):.2f}% "
              f"remE={100*rem_e/max(tot_e,1):.2f}% patch={np.mean(patch):.2f}", flush=True)

    # Store all four variants (raw / P1 / P3 / P1P3), same as three_way_postproc.
    # P1 = validity repair, P3 = arrival repair (postproc.py); P1P3 runs P1 first so
    # that P3 extends the already-legalised tail.
    p1 = [splice(p, A_exc, G_exc) for p in planned]
    p3 = [endpoint(p, g[-1], G_exc) for p, g in zip(planned, real)]
    p13 = []
    for (q, a1), g in zip(p1, real):
        q2, a2 = endpoint(q, g[-1], G_exc)
        p13.append((q2, a1 + a2))
    score(f"{pfx}_raw", planned, [0] * len(real))
    score(f"{pfx}_P1", [q for q, _ in p1], [a for _, a in p1])
    score(f"{pfx}_P3", [q for q, _ in p3], [a for _, a in p3])
    score(f"{pfx}_P1P3", [q for q, _ in p13], [a for _, a in p13])
    print(f"sec_per_path={(time.time()-t0)/len(real):.3f}")
    print("DCBG_EVAL_DONE", flush=True)

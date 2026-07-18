"""
Entry point for block-diffusion (BD3-LM) OD planning.

Methods:
  -method bd_train  : train BlockDiffusion (-kernel mask | graph), then run
                      the planning evaluation on held-out OD pairs.
  -method bd_plan   : load {model_path}/{model_name}_bd.pth, evaluation only.
  -method bd_sample : unconditional generation (stops at <end>).

No length model: generation is open-ended semi-autoregressive (blocks are
appended until the destination or <end> is emitted). "oracle" length mode
only sets the block budget.
"""

import math
import os
import time
from os.path import join

import numpy as np
import torch

from eval_shortest import evaluate_em_pc, success_arrival_rate
from eval_shortest import refine as es_refine
from loader.dataset import TrajFastShortestDataset
from utils.argparser import get_argparser

LEN_BINS = [(2, 15), (16, 25), (26, 35), (36, 100)]


# =====================================================================
# Path-similarity helpers (from the prior project's main_od.py)
# =====================================================================
def path_to_coords(path, G):
    return np.array([[G.nodes[v]["lat"], G.nodes[v]["lng"]] for v in path])


def dtw_distance(a, b):
    """Coordinate DTW (L1 ground distance, in ~km via *100 scaling)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.abs(a[i - 1] - b[j - 1]).sum() * 100
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / max(n, m)


def lcs_length(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[n, m])


# =====================================================================
# Planning evaluation
# =====================================================================
def evaluate_planning_bd(model, dataset, n_samples, batch_size, length_mode,
                         guidance_scale, res_path, tag=""):
    """
    Plans once with NO refinement, then scores three variants of the same
    generated paths:
      raw       : model output as-is (truncated at dst when hit)
      refine_nx : misses patched with a networkx shortest path to dst
      refine_es : eval_shortest.refine() on all paths

    Per variant: arrival, validity, valid-and-hit, EM, PC, DTW, LCS, length
    error. Raw additionally reports len_corr (does the emergent length track
    the per-OD-pair real length?) and arrival broken down by real length.
    """
    real_paths = dataset.get_real_paths(min(n_samples, len(dataset)))
    real_paths = [list(map(int, p)) for p in real_paths if len(p) >= 2]
    n = len(real_paths)
    dests_all = [p[-1] for p in real_paths]

    planned, hits = [], []
    t0 = time.time()
    for s in range(0, n, batch_size):
        batch = real_paths[s: s + batch_size]
        origs = [p[0] for p in batch]
        dests = [p[-1] for p in batch]
        lengths = [len(p) for p in batch] if length_mode == "oracle" else None
        out = model.plan(origs, dests, lengths=lengths,
                         guidance_scale=guidance_scale, use_refine=False)
        planned.extend(out)
        hits.extend(model.last_hits)
    elapsed = time.time() - t0

    # ---- variant construction ----------------------------------------
    nx_paths, nx_patch = [], []
    for p, hit, d in zip(planned, hits, dests_all):
        if hit or len(p) == 0 or model.G is None:
            nx_paths.append(list(p))
            nx_patch.append(0)
        else:
            q = model._refine_to_dest(list(p), d)
            nx_paths.append(q)
            nx_patch.append(len(q) - len(p))

    es_paths = es_refine([list(p) for p in planned], dests_all,
                         dataset.A, dataset.n_vertex)

    variants = [
        ("raw", planned, [0] * n),
        ("refine_nx", nx_paths, nx_patch),
        ("refine_es", es_paths, [len(q) - len(p) for p, q in zip(planned, es_paths)]),
    ]

    # ---- scoring -------------------------------------------------------
    G = dataset.G
    results = []
    for vname, paths, patch in variants:
        dtws, lcss, len_err = [], [], []
        for p, g in zip(paths, real_paths):
            if len(p) < 1:
                continue
            dtws.append(dtw_distance(path_to_coords(p, G), path_to_coords(g, G)))
            lcss.append(lcs_length(p, g))
            len_err.append(abs(len(p) - len(g)))

        em_summary, em_records, _ = evaluate_em_pc(
            gen_paths=paths,
            A=dataset.A,
            shortest_paths=dataset.shortest_path_data,
            save_dir=join(res_path, "em_pc"),
            prefix=f"{tag}_{vname}",
        )
        valid_flags = [bool(r["valid"]) for r in em_records]
        n_hit = int(np.sum(hits))
        valid_and_hit = int(np.sum([v and h for v, h in zip(valid_flags, hits)]))
        res = {
            "tag": f"{tag}_{vname}",
            "variant": vname,
            "n": n,
            "hit_ratio(before_refine)": float(np.mean(hits)),
            "arrival_success_rate": float(success_arrival_rate(paths, real_paths)),
            "avg_patch_len": float(np.mean(patch)),
            "valid_rate": em_summary["valid_rate"],
            "valid_and_hit_rate": float(valid_and_hit / max(n, 1)),
            "valid_given_hit": float(valid_and_hit / max(n_hit, 1)),
            "em_score": em_summary["em_score"],
            "pc_score": em_summary["pc_score"],
            "DTW": float(np.mean(dtws)),
            "LCS": float(np.mean(lcss)),
            "len_abs_err": float(np.mean(len_err)),
            "sec_per_path": elapsed / max(n, 1),
        }

        if vname == "raw":
            # length calibration: does the emergent length track the OD pair?
            gl = np.array([len(p) for p in paths])
            rl = np.array([len(g) for g in real_paths])
            arr = np.array([int(len(p) > 0 and p[-1] == g[-1]) for p, g in zip(paths, real_paths)])
            res["gen_len_mean"] = float(gl.mean())
            res["real_len_mean"] = float(rl.mean())
            res["len_corr"] = float(np.corrcoef(gl, rl)[0, 1]) if gl.std() > 0 and rl.std() > 0 else 0.0
            for lo_b, hi_b in LEN_BINS:
                m = (rl >= lo_b) & (rl <= hi_b)
                if m.sum() > 0:
                    res[f"arrival_len{lo_b}-{hi_b}"] = float(arr[m].mean())
                    res[f"n_len{lo_b}-{hi_b}"] = int(m.sum())

        print("=" * 70)
        print(f"[{tag}] variant={vname}, length_mode={length_mode}, cfg={guidance_scale}")
        for k, v in res.items():
            if k not in ("tag", "variant"):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("=" * 70)
        results.append(res)
    return results


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.device == "default":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)

    date = "20190701" if "dj" in args.d_name else "dj"
    dataset = TrajFastShortestDataset(
        args.d_name, [date], args.path, device, is_pretrain=True,
        index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path,
    )
    n_vertex = dataset.n_vertex
    print(f"vertex: {n_vertex}")

    pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")

    # canvas cap (block multiple); 0 = auto from od_max_len
    bd_max_len = args.bd_max_len if args.bd_max_len > 0 else args.od_max_len + 2
    bd_max_len = int(math.ceil(bd_max_len / args.block_size) * args.block_size)

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.res_path, exist_ok=True)
    with open(join(args.model_path, f"{args.model_name}_bd.info"), "w") as f:
        f.writelines(str(args))

    # ==================================================================
    if args.method == "bd_train":
        from models_seq.bd_models import BDTransformer, BlockDiffusion
        from models_seq.bd_trainer import TrainerBD
        from models_seq.seq_models import Destroyer

        destroyer = None
        if args.kernel == "graph":
            if args.beta_schedule == "uniform":
                betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
            elif args.beta_schedule == "front":
                uuu = torch.linspace(0, 1, args.max_T)
                kkk = 2.0
                sss = (torch.exp(kkk * uuu) - 1) / (torch.exp(kkk * torch.ones_like(uuu)) - 1)
                betas = args.beta_lb + (args.beta_ub - args.beta_lb) * sss
            else:
                raise NotImplementedError
            # <end>-augmented CTMC state space (always on): virtual node of
            # total degree bd_eos_deg, uniformly connected, self-loop kept so
            # <end> -> <end> stays legal in the binarized decode. Symmetric
            # => the uniform limiting distribution is preserved.
            V = dataset.A.shape[0]
            if args.bd_uniform_forward:
                # PURE uniform-state forward kernel (classic D3PM uniform):
                # complete-graph generator => Q_t = e^{-K b} I + (1-e^{-K b}) 11^T/K
                # (alpha_t = e^{-K beta_t}); pass betas scaled by 1/K so the
                # per-step survival profile matches the graph kernel's.
                A_ctmc = torch.ones(V + 1, V + 1)
            else:
                w_eos = args.bd_eos_deg / V
                A_ctmc = torch.zeros(V + 1, V + 1)
                A_ctmc[:V, :V] = dataset.A.cpu().float()
                A_ctmc[V, :V] = w_eos
                A_ctmc[:V, V] = w_eos
                A_ctmc[V, V] = w_eos
            destroyer = Destroyer(A_ctmc, betas, args.max_T, device)

        backbone = BDTransformer(
            n_vertex, device,
            hidden_dim=args.bd_hidden_dim,
            n_layers=args.bd_n_layers,
            n_heads=args.bd_n_heads,
            cond_dim=args.bd_cond_dim,
            dropout=args.bd_dropout,
            max_canvas=bd_max_len,
            x_emb_dim=args.x_emb_dim,
            pretrain_path=pretrain_path,
        )
        model = BlockDiffusion(backbone, destroyer, device, args)
        model.set_graph(dataset.G)

        total_p = sum(p.numel() for p in backbone.parameters())
        print("=" * 60)
        print(f"BDTransformer ({args.kernel} kernel, block={args.block_size}): "
              f"{total_p / 1e6:.2f}M params, canvas={bd_max_len}")
        print("=" * 60)

        trainer = TrainerBD(model, dataset, args.model_path, args.model_name, args=args)

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()
        trainer.train(args.n_epoch, args.bs, args.lr)
        torch.cuda.synchronize() if device.type == "cuda" else None
        print("=" * 60)
        print(f"Training time: {(time.time() - start) / 60:.2f} min")
        print("=" * 60)

        model.eval()
        torch.save(model, join(args.model_path, f"{args.model_name}_bd.pth"))

    elif args.method in ("bd_plan", "bd_sample"):
        ckpt = join(args.model_path, f"{args.model_name}_bd.pth")
        model = torch.load(ckpt, map_location=device)
        model.eval()
        model.set_graph(dataset.G)
    else:
        raise NotImplementedError(f"main_bd.py supports bd_train / bd_plan / bd_sample, got: {args.method}")

    # ==================================================================
    # Unconditional generation (-method bd_sample): plan(None, None)
    # ==================================================================
    if args.method == "bd_sample":
        from eval_shortest import print_path_stats

        t0 = time.time()
        paths = []
        for s in range(0, args.eval_num, args.batch_traj_num):
            nb = min(args.batch_traj_num, args.eval_num - s)
            paths.extend(model.plan(None, None, n_samples=nb))
        elapsed = time.time() - t0

        paths = [p for p in paths if len(p) >= 2]
        print(f"unconditional: {len(paths)} usable paths "
              f"({elapsed / max(args.eval_num, 1):.4f} s/path)")
        print_path_stats(paths, prefix=f"[{args.model_name}_bd_uncond]")

        summary, _, _ = evaluate_em_pc(
            gen_paths=paths,
            A=dataset.A,
            shortest_paths=dataset.shortest_path_data,
            save_dir=join(args.res_path, "em_pc"),
            prefix=f"{args.model_name}_bd_uncond",
        )
        real = dataset.get_real_paths(min(args.eval_num, len(dataset)))
        res = {
            "n": len(paths),
            "avg_len": float(np.mean([len(p) for p in paths])),
            "real_avg_len": float(np.mean([len(p) for p in real])),
            "valid_rate": summary["valid_rate"],
            "em_score": summary["em_score"],
            "pc_score": summary["pc_score"],
            "sec_per_path": elapsed / max(args.eval_num, 1),
        }
        print(res)
        with open(join(args.res_path, f"{args.model_name}_bd_uncond.res"), "w") as f:
            f.writelines(str(res))
        torch.save(paths, join(args.res_path, f"{args.model_name}_bd_uncond_paths.pth"))
        raise SystemExit(0)

    # ==================================================================
    # Planning evaluation (open-ended; "oracle" only sets the block budget)
    # ==================================================================
    results = []
    modes = [args.length_mode] if args.length_mode != "both" else ["oracle", "open"]
    for mode in modes:
        results.extend(evaluate_planning_bd(
            model, dataset,
            n_samples=args.eval_num,
            batch_size=args.batch_traj_num,
            length_mode=mode,
            guidance_scale=args.guidance_scale,
            res_path=args.res_path,
            tag=f"{args.model_name}_bd_{mode}",
        ))

    with open(join(args.res_path, f"{args.model_name}_bd.res"), "w") as f:
        f.writelines(str(results))
    print(f"Results written to {join(args.res_path, f'{args.model_name}_bd.res')}")

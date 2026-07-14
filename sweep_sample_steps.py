"""
Respaced-sampling sweep: quality vs number of reverse steps.

Loads a trained full-canvas OD model and evaluates the cond variant at a
fixed CFG weight for several reverse-step counts (exact composed CTMC
kernels, -sample_steps). Reports arrival / valid / hit / len metrics and
wall-clock, raw and refine_es.

Usage:
  python sweep_sample_steps.py -device cpu -d_name porto \
      -model_name OD_EOS_porto_v3_normal_d2.0_A1A2A3 \
      -shortest_org_idx v3-0.05_normal -shortest_new_idx v3-0.05_normal \
      -shortest_data_path ./porto_data -eval_num 1000 -seed 1 -guidance_scale 1.5
"""
import json
import time
from os.path import join

import numpy as np
import torch

from loader.dataset import TrajFastShortestDataset
from utils.argparser import get_argparser
from eval_shortest import set_seed, normalize_path, evaluate_em_pc, load_shortest_paths_for_em_pc, refine

STEP_GRID = [10, 25, 50, 0]  # 0 = full max_T (=100)

if __name__ == "__main__":
    args = get_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "default" else "cuda:0")
    print(device)

    dataset = TrajFastShortestDataset(args.d_name, ["dj"], args.path, device, is_pretrain=True,
                                      shuffle=False, index=args.shortest_org_idx,
                                      shortest_data_path=args.shortest_data_path)
    model = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)
    model.args = args
    model.device = device
    model.eps_model.device = device
    model = model.to(device)
    model.model_device = device
    model.des_device = device
    for mod in model.modules():
        if hasattr(mod, "device"):
            mod.device = device
    model.destroyer.device = device
    model.destroyer.A = model.destroyer.A.to(device)
    model.destroyer.matrices = model.destroyer.matrices.to(device)
    model.destroyer.betas = model.destroyer.betas.to(device)
    model.Q = model.Q.to(device)
    model.matrices = model.matrices.to(device)
    model.A = model.A.to(device)
    model.eval()
    model.applying_mask_intermediate = False
    model.applying_mask_intermediate_temperature = False

    shortest_paths = load_shortest_paths_for_em_pc(args)
    real_paths = sorted(dataset.get_real_paths(args.eval_num), key=len)
    dests = [normalize_path(p)[-1] for p in real_paths]

    results = {"model": args.model_name, "cfg_w": args.guidance_scale, "eval_num": args.eval_num}
    for n_steps in STEP_GRID:
        args.sample_steps = n_steps
        label = f"steps{n_steps if n_steps > 0 else model.max_T}"
        set_seed(args.seed)
        t0 = time.time()
        gen = model.sample(args.eval_num, args.batch_traj_num, real_paths=real_paths,
                           bool_prefix=True, bool_od=True)
        dt = time.time() - t0

        out = {"sampling_time_sec": dt}
        for stage, paths in (("raw", gen),
                             ("refine_es", refine([list(normalize_path(g)) for g in gen], dests,
                                                  dataset.A, dataset.n_vertex))):
            summary, records, _ = evaluate_em_pc(paths, dataset.A, shortest_paths,
                                                 join(args.res_path, "od_eval"), f"STEPS_{label}_{stage}")
            arr = [int(len(normalize_path(g)) > 0 and normalize_path(g)[-1] == normalize_path(r)[-1])
                   for g, r in zip(paths, real_paths)]
            hit = [int(normalize_path(r)[-1] in set(normalize_path(g))) for g, r in zip(paths, real_paths)]
            gl = np.array([len(normalize_path(g)) for g in paths])
            rl = np.array([len(normalize_path(r)) for r in real_paths])
            out[stage] = {
                "valid": summary["valid_rate"], "em": summary["em_score"], "pc": summary["pc_score"],
                "arrival": float(np.mean(arr)), "hit": float(np.mean(hit)),
                "len_err": float(np.mean(np.abs(gl - rl))),
                "len_corr": float(np.corrcoef(gl, rl)[0, 1]) if gl.std() > 0 else 0.0,
                "gen_len_mean": float(gl.mean()),
            }
        results[label] = out
        print(f"[{label}] time={dt:.1f}s raw_arr={out['raw']['arrival']:.4f} "
              f"ref_arr={out['refine_es']['arrival']:.4f} len_corr={out['raw']['len_corr']:.3f}")

    out_path = join(args.res_path, "od_eval", f"STEPS_SWEEP_{args.model_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

"""
Evaluation for the O/D-conditional diffusion models (EPSM_OD / EPSM_OD_EOS).

Variants:
  cond_w{w}   : O prefix + (O, D) embedding condition, CFG guidance scale w
  prefix_only : O prefix, null OD condition (skipped for dst_token models)
  uncond      : no prefix, null OD condition

Each variant is scored raw and after refine_es (eval_shortest.refine()).

Metrics: valid / arrival / valid_and_arr / valid_given_arr / hit / EM / PC /
len_err / unique_ratio, plus (raw only) len_corr and arrival broken down by
real path length bins.

Usage (server):
  CUDA_VISIBLE_DEVICES=1 python eval_od_conditional.py \
      -d_name porto -model_name OD_EOS_porto_v3_normal_d0.05 \
      -shortest_org_idx v3-0.05_normal -shortest_new_idx v3-0.05_normal \
      -shortest_data_path ./porto_data -eval_num 1000 -seed 1
"""
from os.path import join
import os
import json
import time

import numpy as np
import torch

from loader.dataset import TrajFastShortestDataset
from utils.argparser import get_argparser
from eval_shortest import (
    set_seed,
    normalize_path,
    evaluate_em_pc,
    load_shortest_paths_for_em_pc,
    refine,
)

LEN_BINS = [(2, 15), (16, 25), (26, 35), (36, 100)]


def path_metrics(gen_paths, real_paths, records):
    """Joint metrics on aligned (gen, real) pairs; records from evaluate_em_pc."""
    n = len(gen_paths)
    valid_flags = [r["valid"] for r in records]
    arr_flags, hit_flags, len_errs = [], [], []
    for g, r in zip(gen_paths, real_paths):
        g_n, r_n = normalize_path(g), normalize_path(r)
        arr_flags.append(int(len(g_n) > 0 and g_n[-1] == r_n[-1]))
        hit_flags.append(int(r_n[-1] in set(g_n)))
        len_errs.append(abs(len(g_n) - len(r_n)))

    joint = [v & a for v, a in zip(valid_flags, arr_flags)]
    n_arr = max(sum(arr_flags), 1)
    unique_ratio = len(set(tuple(normalize_path(g)) for g in gen_paths)) / max(n, 1)

    return {
        "valid": float(np.mean(valid_flags)),
        "arrival": float(np.mean(arr_flags)),
        "valid_and_arr": float(np.mean(joint)),
        "valid_given_arr": float(sum(joint) / n_arr),
        "hit": float(np.mean(hit_flags)),
        "len_err": float(np.mean(len_errs)),
        "unique_ratio": float(unique_ratio),
    }


def length_analysis(gen_paths, real_paths):
    """Does the model's emergent length actually track the OD pair?"""
    gl = np.array([len(normalize_path(g)) for g in gen_paths])
    rl = np.array([len(normalize_path(r)) for r in real_paths])
    arr = np.array([int(len(normalize_path(g)) > 0 and normalize_path(g)[-1] == normalize_path(r)[-1])
                    for g, r in zip(gen_paths, real_paths)])
    out = {
        "len_corr": float(np.corrcoef(gl, rl)[0, 1]) if gl.std() > 0 and rl.std() > 0 else 0.0,
        "gen_len_mean": float(gl.mean()),
        "real_len_mean": float(rl.mean()),
    }
    for lo, hi in LEN_BINS:
        m = (rl >= lo) & (rl <= hi)
        if m.sum() > 0:
            out[f"arrival_len{lo}-{hi}"] = float(arr[m].mean())
            out[f"n_len{lo}-{hi}"] = int(m.sum())
    return out


def eval_variant(name, gen_paths, real_paths, dataset, shortest_paths, save_dir, sampling_time):
    dests = [normalize_path(p)[-1] for p in real_paths]
    out = {"sampling_time_sec": sampling_time}

    for stage, paths in (
        ("raw", gen_paths),
        ("refine_es", refine([list(normalize_path(g)) for g in gen_paths], dests, dataset.A, dataset.n_vertex)),
    ):
        summary, records, _ = evaluate_em_pc(
            gen_paths=paths,
            A=dataset.A,
            shortest_paths=shortest_paths,
            save_dir=save_dir,
            prefix=f"{name}_{stage}",
        )
        m = path_metrics(paths, real_paths, records)
        m["em"] = summary["em_score"]
        m["pc"] = summary["pc_score"]
        if stage == "raw":
            m.update(length_analysis(paths, real_paths))
        out[stage] = m
        print(f"[{name}/{stage}] " + ", ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)))
    return out


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "default":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)

    date = "20190701" if "dj" in args.d_name else "dj"
    dataset = TrajFastShortestDataset(
        args.d_name, [date], args.path, device, is_pretrain=True,
        shuffle=False, index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path,
    )
    print(f"vertex: {dataset.n_vertex}")

    model = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)
    model.args = args
    model.device = device
    model.eps_model.device = device
    model = model.to(device)
    # move every device-carrying attribute (submodules create tensors on their
    # own .device; plain-tensor attributes are not covered by nn.Module.to)
    model.model_device = device
    model.des_device = device
    for mod in model.modules():
        if hasattr(mod, "device"):
            mod.device = device
    model.destroyer.device = device
    model.destroyer.A = model.destroyer.A.to(device)
    model.destroyer.matrices = model.destroyer.matrices.to(device)
    model.Q = model.Q.to(device)
    model.matrices = model.matrices.to(device)
    model.A = model.A.to(device)
    model.eval()
    model.applying_mask_intermediate = args.applying_mask_intermediate
    model.applying_mask_intermediate_temperature = args.applying_mask_intermediate_temperature
    # attributes added after some checkpoints were trained
    for attr, dv in (("clean_prefix", False), ("dst_token", False), ("eos_loss_weight", 1.0),
                     ("eos_mode", False), ("eos_canvas_len", 64)):
        if not hasattr(model, attr):
            setattr(model, attr, dv)
    print(f"model: eos_mode={model.eos_mode}, clean_prefix={model.clean_prefix}, dst_token={model.dst_token}")

    shortest_paths = load_shortest_paths_for_em_pc(args)

    save_dir = join(args.res_path, "od_eval")
    os.makedirs(save_dir, exist_ok=True)

    # Same real (O, D, length) triples for every variant, sorted by length so
    # sample()'s internal sort keeps gen/real alignment.
    real_paths = dataset.get_real_paths(args.eval_num)
    real_paths = sorted(real_paths, key=len)
    print(f"# eval OD pairs: {len(real_paths)}, avg len: {np.mean([len(p) for p in real_paths]):.2f}")

    results = {"model_name": args.model_name, "eval_num": args.eval_num, "seed": args.seed}

    variant_list = [(f"cond_w{w:g}", dict(bool_prefix=True, bool_od=True), w) for w in (1.0, 1.5, 2.0, 4.0)]
    if not model.dst_token:
        variant_list.append(("prefix_only", dict(bool_prefix=True, bool_od=False), 1.0))
    variant_list.append(("uncond", dict(bool_prefix=False, bool_od=False), 1.0))

    for name, kw, w in variant_list:
        set_seed(args.seed)
        args.guidance_scale = w
        t0 = time.time()
        gen_paths = model.sample(args.eval_num, args.batch_traj_num, real_paths=real_paths, **kw)
        dt = time.time() - t0
        print(f"\n===== variant: {name} (sampling {dt:.1f}s) =====")
        results[name] = eval_variant(f"{args.model_name}_{name}", gen_paths, real_paths, dataset, shortest_paths, save_dir, dt)
        torch.save(gen_paths, join(save_dir, f"{args.model_name}_{name}_gen_paths.pth"))

    out_path = join(save_dir, f"OD_EVAL_{args.model_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n========== FINAL SUMMARY ==========")
    header = ["variant", "stage", "valid", "arrival", "valid_and_arr", "valid_given_arr", "hit", "em", "pc", "len_err", "unique"]
    print(" | ".join(header))
    for name, _, _ in variant_list:
        for stage in ("raw", "refine_es"):
            m = results[name][stage]
            print(" | ".join([name, stage] + [f"{m[k]:.4f}" for k in
                  ("valid", "arrival", "valid_and_arr", "valid_given_arr", "hit", "em", "pc", "len_err", "unique_ratio")]))

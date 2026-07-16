"""
Post-processing evaluation for the block-diffusion block-size sweep.

Applies three post-processors to the SAME raw generations of each trained
mask-kernel checkpoint and scores all variants side by side:

  raw    : model output as-is (truncated at dst when hit)
  P1     : local splice -- every invalid consecutive pair (u, v) is replaced
           by the shortest path u -> v (validity 1.0 by construction,
           endpoints untouched, hit unchanged)
  P3     : endpoint patch (refine_nx) -- miss paths get a shortest-path
           graft from their last vertex to dst (arrival 1.0 by construction)
  P1+P3  : splice first, then endpoint patch (valid 1.0 AND arrival 1.0)

Usage:
  python eval_bd_postproc.py [-blocks 1,2,4,8,16,32,64] [-eval_num 1000]
"""

import argparse
import time
from os.path import join

import networkx as nx
import numpy as np
import torch

from eval_shortest import evaluate_em_pc, success_arrival_rate
from loader.dataset import TrajFastShortestDataset
from main_bd import dtw_distance, lcs_length, path_to_coords


def splice_invalid(p, A, G):
    """P1: insert shortest-path bridges over invalid edges. Returns
    (path, n_inserted, n_unfixable)."""
    if len(p) < 2:
        return list(p), 0, 0
    out, added, unfix = [p[0]], 0, 0
    for u, v in zip(p[:-1], p[1:]):
        if A[u, v]:
            out.append(v)
            continue
        try:
            seg = nx.shortest_path(G, source=u, target=v)[1:]
            added += len(seg) - 1
            out.extend(seg)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            unfix += 1
            out.append(v)
    return out, added, unfix


def endpoint_patch(p, dst, G):
    """P3: graft a shortest path from the last vertex to dst (no-op when
    already at dst or empty). Returns (path, n_appended)."""
    if len(p) == 0 or p[-1] == dst:
        return list(p), 0
    try:
        patch = nx.shortest_path(G, source=p[-1], target=dst)[1:]
        return list(p) + patch, len(patch)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return list(p), 0


def score(tag, vname, paths, patch, real_paths, hits, dataset, res_path):
    G = dataset.G
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
    n = len(real_paths)
    valid_flags = [bool(r["valid"]) for r in em_records]
    valid_and_hit = int(np.sum([v and h for v, h in zip(valid_flags, hits)]))
    return {
        "tag": f"{tag}_{vname}",
        "variant": vname,
        "n": n,
        "hit_ratio(before_refine)": float(np.mean(hits)),
        "arrival_success_rate": float(success_arrival_rate(paths, real_paths)),
        "avg_patch_len": float(np.mean(patch)),
        "valid_rate": em_summary["valid_rate"],
        "valid_and_hit_rate": float(valid_and_hit / max(n, 1)),
        "em_score": em_summary["em_score"],
        "pc_score": em_summary["pc_score"],
        "DTW": float(np.mean(dtws)),
        "LCS": float(np.mean(lcss)),
        "len_abs_err": float(np.mean(len_err)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-batch", type=int, default=200)
    ap.add_argument("-model_path", type=str, default="./sets_model")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TrajFastShortestDataset("porto", ["dj"], "./sets_data", device, is_pretrain=True,
                                      index="v3-0.05_normal", shortest_data_path="./porto_data")
    A = dataset.A.cpu().numpy() > 0
    G = dataset.G

    all_results = {}
    for blk in [int(b) for b in args.blocks.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        ckpt = join(args.model_path, f"BD_porto_v3_normal_mask_blk{blk}_base_bd.pth")
        model = torch.load(ckpt, map_location=device)
        model.eval()
        model.set_graph(G)

        real = [list(map(int, p)) for p in dataset.get_real_paths(args.eval_num) if len(p) >= 2]
        planned, hits = [], []
        t0 = time.time()
        for s in range(0, len(real), args.batch):
            b = real[s: s + args.batch]
            out = model.plan([p[0] for p in b], [p[-1] for p in b], use_refine=False)
            planned.extend(out)
            hits.extend(model.last_hits)
        gen_sec = time.time() - t0

        # ---- variants from the SAME generations -------------------------
        p1_paths, p1_patch, n_unfix = [], [], 0
        for p in planned:
            q, added, unfix = splice_invalid(p, A, G)
            p1_paths.append(q)
            p1_patch.append(added)
            n_unfix += unfix

        p3_paths, p3_patch = [], []
        for p, g in zip(planned, real):
            q, added = endpoint_patch(p, g[-1], G)
            p3_paths.append(q)
            p3_patch.append(added)

        p13_paths, p13_patch = [], []
        for p, g, a1 in zip(p1_paths, real, p1_patch):
            q, added = endpoint_patch(p, g[-1], G)
            p13_paths.append(q)
            p13_patch.append(a1 + added)

        tag = f"BD_mask_blk{blk}_postproc"
        results = [
            score(tag, "raw", planned, [0] * len(real), real, hits, dataset, args.res_path),
            score(tag, "P1_splice", p1_paths, p1_patch, real, hits, dataset, args.res_path),
            score(tag, "P3_endpoint", p3_paths, p3_patch, real, hits, dataset, args.res_path),
            score(tag, "P1+P3", p13_paths, p13_patch, real, hits, dataset, args.res_path),
        ]
        if n_unfix:
            print(f"[blk{blk}] WARNING: {n_unfix} invalid pairs had no connecting path (left as-is)")
        for r in results:
            r["gen_sec_per_path"] = gen_sec / max(len(real), 1)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        all_results[blk] = results

        with open(join(args.res_path, f"{tag}.res"), "w") as f:
            f.writelines(str(results))
        print(f"[blk{blk}] done, written to {join(args.res_path, f'{tag}.res')}")

    print("ALL POSTPROC EVALS DONE")

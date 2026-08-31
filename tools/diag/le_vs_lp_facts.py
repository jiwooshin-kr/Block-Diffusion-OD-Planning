"""
Dataset-level facts for the LE-vs-LP comparison, in one JSON the report reads.

  python tools/diag/le_vs_lp_facts.py -out sets_v6/le_vs_lp_facts.json

Context. LE and LP are two CHOICE-SET generation methods, not two exception
scenarios:
  LE  link elimination  -- iteratively delete edges and take the shortest path,
      so every choice-set member IS a shortest path on some edge-deleted subgraph.
  LP  link penalization -- iteratively raise edge costs and take the shortest path;
      edges are never deleted, so members are shortest under a modified cost, not
      shortest in hop count on any subgraph.
One member is then sampled per OD pair to become the reference trajectory.

The hypothesis this script tests. The except_e scenario perturbs the graph by
DELETING edges. Under LE, the training distribution already contains "shortest
path on a graph with edges missing" trajectories, so a scenario deletion produces
paths that look like ordinary LE data -- the discriminator's positive class
(except_0 paths) is less distinctive against a model trained on normal-graph LE
data. Under LP nothing in the training distribution looks like a deletion, so the
scenario signal should be cleaner and guidance should help more.

What is measured
  perturbation   edges in normal vs except_e, added/removed, overlap between two
                 scenarios. Tests whether the scenario itself differs by dataset
                 (it must not -- the datasets differ in choice-set construction).
  ref_validity   are the reference paths legal on their own scenario graph? Rules
                 out "LP keeps using the penalized links".
  shortestness   how close the reference paths sit to a hop-minimal BFS path:
                 hop excess, DTW, LCS, and the per-pair detour ratio distribution.
                 This is the direct prediction of the hypothesis -- LE references
                 should be measurably CLOSER to shortest than LP's.
  disc_val       discriminator val accuracy per block, scraped from the training
                 logs, i.e. whether the training task itself is harder for LE.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import json
import pickle
import re
from os.path import join

import numpy as np

from main_bd import dtw_distance, lcs_length, path_to_coords
from tools.eval.shortest_baseline import bfs_path

SPLIT_SEED = 777
BLKS = [1, 2, 4, 8, 16, 32, 64]


def eval_pairs(porto, fam, n):
    sp = pickle.load(open(join(porto,
        f"porto_shrink_SP_v6-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp))
    return [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:n]


def scrape_disc(variant):
    """Last reported val accuracy per block from the e0 discriminator logs."""
    out = {}
    for b in BLKS:
        p = f"sets_v6/{variant}/log/disc_e0_blk{b}.log"
        try:
            lines = [l for l in open(p) if l.startswith("step")]
        except FileNotFoundError:
            continue
        if not lines:
            continue
        last = lines[-1]
        m = re.search(r"acc=([\d.]+) \|.*val\(except_0\) acc=([\d.]+) "
                      r"logit_std=([\d.]+)", last)
        if m:
            out[b] = {"train_acc": float(m.group(1)), "val_acc": float(m.group(2)),
                      "logit_std": float(m.group(3))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-variants", type=str, default="LE,LP")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("-n_scen", type=int, default=10,
                    help="how many except_e scenarios to average the perturbation over")
    ap.add_argument("-out", type=str, default="sets_v6/le_vs_lp_facts.json")
    args = ap.parse_args()

    res = {}
    for V in args.variants.split(","):
        fam = f"{V}1.0-0.05"
        A_p = join(args.porto, f"porto_shrink_A_v6-{fam}_%s.ts")
        N = pickle.load(open(A_p % "normal", "rb")).bool()
        G = pickle.load(open(join(args.porto,
            f"porto_shrink_G_v6-{fam}_except_0.pkl"), "rb"))
        A0 = pickle.load(open(A_p % "except_0", "rb")).bool()
        A0n = A0.numpy()

        # ---- perturbation: same in both datasets? ----
        rem, add, cnt = [], [], []
        for e in range(args.n_scen):
            E = pickle.load(open(A_p % f"except_{e}", "rb")).bool()
            rem.append(int((N & ~E).sum()))
            add.append(int((~N & E).sum()))
            cnt.append(int(E.sum()))
        E1 = pickle.load(open(A_p % "except_1", "rb")).bool()
        r0, r1 = (N & ~A0), (N & ~E1)
        pert = {"normal_edges": int(N.sum()),
                "scenario_edges_mean": float(np.mean(cnt)),
                "removed_mean": float(np.mean(rem)),
                "added_mean": float(np.mean(add)),
                "removed_frac": float(np.mean(rem) / int(N.sum())),
                "overlap_exc0_exc1": int((r0 & r1).sum()),
                "removed_exc0": int(r0.sum()), "n_scen": args.n_scen}

        # ---- reference paths: legal on their own scenario graph? ----
        real = eval_pairs(args.porto, fam, args.n)
        bad_paths = bad_edges = tot_edges = used_removed = 0
        for q in real:
            ok = True
            for u, v in zip(q[:-1], q[1:]):
                tot_edges += 1
                if not A0n[u, v]:
                    bad_edges += 1
                    ok = False
                    if bool(N[u, v]):
                        used_removed += 1
            bad_paths += (not ok)
        valid = {"n_pairs": len(real), "invalid_paths": bad_paths,
                 "illegal_edges": bad_edges, "total_edges": tot_edges,
                 "illegal_that_were_removed": used_removed}

        # ---- shortestness: the hypothesis' direct prediction ----
        dtw, lcs, ratio, excess = [], [], [], []
        n_short = n_equal = n_unreach = 0
        for q in real:
            p = bfs_path(G, q[0], q[-1])
            if p is None:
                n_unreach += 1
                continue
            dtw.append(dtw_distance(path_to_coords(q, G), path_to_coords(p, G),
                                    normalize=None))
            lcs.append(lcs_length(q, p))
            ratio.append(len(q) / len(p))
            excess.append(len(q) - len(p))
            n_short += (len(q) < len(p))
            n_equal += (len(q) == len(p))
        short = {"ref_len": float(np.mean([len(q) for q in real])),
                 "bfs_len": float(np.mean([len(q) - e for q, e in zip(real, excess)]))
                 if excess else float("nan"),
                 "hop_excess_mean": float(np.mean(excess)),
                 "hop_excess_median": float(np.median(excess)),
                 "detour_ratio_mean": float(np.mean(ratio)),
                 "detour_ratio_p90": float(np.percentile(ratio, 90)),
                 "dtw_to_bfs": float(np.mean(dtw)),
                 "lcs_with_bfs": float(np.mean(lcs)),
                 "frac_equal_to_bfs": n_equal / max(len(ratio), 1),
                 "frac_shorter_than_bfs": n_short / max(len(ratio), 1),
                 "unreachable": n_unreach}

        res[V] = {"perturbation": pert, "ref_validity": valid,
                  "shortestness": short, "disc_val": scrape_disc(V)}
        d = res[V]
        print(f"== {V} ==")
        print(f"  perturbation: {pert['normal_edges']} -> "
              f"{pert['scenario_edges_mean']:.0f} edges, removed "
              f"{pert['removed_mean']:.0f} ({100*pert['removed_frac']:.2f}%), "
              f"added {pert['added_mean']:.0f}")
        print(f"  ref validity: {valid['invalid_paths']}/{valid['n_pairs']} invalid, "
              f"{valid['illegal_edges']}/{valid['total_edges']} illegal edges")
        print(f"  shortestness: ref {short['ref_len']:.2f} hops vs BFS "
              f"{short['bfs_len']:.2f} (+{short['hop_excess_mean']:.2f}), "
              f"ratio {short['detour_ratio_mean']:.3f}, "
              f"DTW {short['dtw_to_bfs']:.0f} m, LCS {short['lcs_with_bfs']:.2f}, "
              f"=BFS {100*short['frac_equal_to_bfs']:.1f}%")
        va = [x["val_acc"] for x in d["disc_val"].values()]
        if va:
            print(f"  disc val acc: mean {np.mean(va):.3f}  "
                  f"range {min(va):.3f}-{max(va):.3f}  (n={len(va)} blocks)")

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"written {args.out}")

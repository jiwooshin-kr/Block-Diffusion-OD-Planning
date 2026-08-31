"""
Non-learned reference row for the result tables: BFS shortest path on the
SCENARIO graph, scored with exactly the metrics the model arms are scored with.

  python tools/eval/shortest_baseline.py -variant LE
  python tools/eval/shortest_baseline.py -variant LP -out sets_v6/LP/res/shortest_v6LP.json

Why it belongs directly under `base`: valid & arrival is a reference-free
constraint check, and BFS satisfies both by construction (every consecutive pair
is a scenario edge; the search terminates at dst). So it scores v&a = 1.000 for
free and shows that v&a alone cannot be a headline result -- the interesting
question is whether the model reproduces the observed routing behaviour, which
only the shape metrics (DTW, LCS) measure. BFS is expected to LOSE there,
because the reference trajectories are not shortest paths.

BFS, not Dijkstra: the scenario graph is unweighted for this purpose -- hop count
is the quantity the model's block length also counts, so a hop-minimal path is
the right "as short as possible" comparison. Dijkstra on great-circle edge length
would answer a different question (metric-shortest, not hop-shortest).

The row is a single number set, repeated across the table:
  * identical for every block size -- BFS has no block structure
  * identical for seen and unseen -- the eval OD pairs are the same 1,000 pairs
    (SPLIT_SEED 777) in both regimes; seen/unseen distinguishes which model
    checkpoint saw the scenario, and BFS has no training
  * identical for every post-processing variant -- repair has nothing to fix
    (already valid and arriving) and simplification has nothing to cut (BFS paths
    are simple by construction). Verified, not assumed: the script asserts it.

Ties are broken by the graph's own adjacency order, i.e. deterministically but
arbitrarily. Where several hop-minimal paths exist, a different tie-break would
give different DTW/LCS; the numbers therefore describe "a" shortest path, not the
best-case shortest path. `n_tied` reports how often that ambiguity arises.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import json
import pickle
from collections import deque
from os.path import join

import numpy as np

from main_bd import dtw_distance, lcs_length, path_to_coords
from postproc import simplify_path

SPLIT_SEED = 777


def bfs_path(G, src, dst):
    """Hop-minimal src->dst path, or None if dst is unreachable.

    Plain BFS with a parent map. Predecessors are recorded on first discovery,
    so the returned path is hop-minimal and the tie-break follows G's adjacency
    iteration order.
    """
    if src == dst:
        return [src]
    if src not in G or dst not in G:
        return None
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in G[u]:
            if v in prev:
                continue
            prev[v] = u
            if v == dst:
                out = [v]
                while prev[out[-1]] is not None:
                    out.append(prev[out[-1]])
                return out[::-1]
            q.append(v)
    return None


def n_shortest(G, src, dst, hops):
    """Count hop-minimal src->dst paths, to report how ambiguous the tie-break is.

    Layered DP over BFS levels: paths[v] at level k counts hop-minimal walks
    reaching v in exactly k hops. Capped so pathological fan-out cannot stall.
    """
    if src == dst:
        return 1
    cnt = {src: 1}
    for _ in range(hops):
        nxt = {}
        for u, c in cnt.items():
            for v in G[u]:
                nxt[v] = nxt.get(v, 0) + c
        cnt = nxt
        if len(cnt) > 2_000_000:
            return -1
    return cnt.get(dst, 0)


def path_valid(G, p):
    """Same validity definition the model arms are scored with: every node exists
    and every consecutive pair is an edge of the scenario graph."""
    if len(p) < 1:
        return False
    if any(v not in G for v in p):
        return False
    return all(G.has_edge(p[i], p[i + 1]) for i in range(len(p) - 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-dver", type=str, default="v6")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-count_ties", type=int, default=1,
                    help="also count how many OD pairs have >1 hop-minimal path")
    ap.add_argument("-out", type=str, default="")
    args = ap.parse_args()

    V = args.variant
    fam = f"{V}1.0-0.05"
    out = args.out or f"sets_v6/{V}/res/shortest_v6{V}.json"

    sp = pickle.load(open(join(args.porto,
        f"porto_shrink_SP_{args.dver}-{fam}_except_0.pkl"), "rb"))
    G = pickle.load(open(join(args.porto,
        f"porto_shrink_G_{args.dver}-{fam}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp))
    real = [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:args.eval_num]
    n = len(real)
    print(f"[{V}] eval pairs: {n}   scenario graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    valid, arrival, dtwsum, dtwmax, lcss, lcsn, glen = [], [], [], [], [], [], []
    unreach, tied, ref_shorter, ref_equal = 0, 0, 0, 0
    for q in real:
        src, dst = q[0], q[-1]
        p = bfs_path(G, src, dst)
        if p is None:
            # Unreachable: score it the way an empty generation is scored, so the
            # row stays comparable instead of silently dropping the pair.
            unreach += 1
            valid.append(0.0)
            arrival.append(0.0)
            continue
        assert simplify_path(p) == p, f"BFS path has a repeat: {p[:8]}..."
        v = path_valid(G, p)
        assert v, f"BFS path failed the validity check: {p[:8]}..."
        valid.append(float(v))
        arrival.append(float(p[-1] == dst))
        gc, rc = path_to_coords(p, G), path_to_coords(q, G)
        dtwsum.append(dtw_distance(gc, rc, normalize=None))
        dtwmax.append(dtw_distance(gc, rc, normalize="max"))
        l = lcs_length(p, q)
        lcss.append(l)
        lcsn.append(l / len(q))
        glen.append(len(p))
        if len(q) < len(p):
            ref_shorter += 1
        elif len(q) == len(p):
            ref_equal += 1
        if args.count_ties:
            c = n_shortest(G, src, dst, len(p) - 1)
            tied += int(c != 1)

    va = float(np.mean([v * a for v, a in zip(valid, arrival)]))
    rec = {
        "arm": "shortest",
        "method": "BFS hop-minimal path on the except_0 scenario graph",
        "n_pairs": n,
        "valid": float(np.mean(valid)),
        "arrival": float(np.mean(arrival)),
        "valid_and_arrival": va,
        "dtw_sum": float(np.mean(dtwsum)),
        "dtw": float(np.mean(dtwmax)),
        "lcs": float(np.mean(lcss)),
        "lcs_norm": float(np.mean(lcsn)),
        "gen_len": float(np.mean(glen)),
        "ref_len": float(np.mean([len(q) for q in real])),
        "unreachable": unreach,
        "n_tied": tied if args.count_ties else None,
        "ref_shorter_than_bfs": ref_shorter,
        "ref_equal_to_bfs": ref_equal,
        "invariant_to_postproc": True,
        "invariant_to_block_and_regime": True,
    }
    json.dump(rec, open(out, "w"), indent=2)
    print(f"valid={rec['valid']:.3f}  arrival={rec['arrival']:.3f}  "
          f"v&a={va:.3f}  dtw_sum={rec['dtw_sum']:.1f} m  lcs={rec['lcs']:.2f}  "
          f"gen_len={rec['gen_len']:.2f}  ref_len={rec['ref_len']:.2f}")
    print(f"unreachable={unreach}  tied(>1 shortest)={rec['n_tied']}  "
          f"ref shorter than BFS={ref_shorter}  ref equal={ref_equal}")
    print(f"written {out}")

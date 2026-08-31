"""
Simplification post-processing (loop cut) applied to already-saved records.

  python tools/collect/postproc_loopcut.py -res_path sets_v6/LE/res -porto ./porto_data_v6 \
      -dver v6 -family LE1.0-0.05 -variant P1P3 -out_variant P1P3L \
      -pairs "seen:v6LE{B}seen,unseen:v6LE{B}uns" \
      -cfgs base,modelD,adj+modelD,adjonly,dcbg,dcbg+adjp

Post-processing has two distinct axes and they must not be conflated:
  repair          P1 (illegal-edge splice) and P3 (endpoint patch). Fixes a
                  failure, so it CHANGES validity/arrival (0.38/0.80 -> 1.0/1.0)
                  and adds nodes. Reporting valid&arrival after repair is
                  meaningless -- it saturates to 1.0 for every arm.
  simplification  this script. Fixes nothing, so arrival is preserved exactly and
                  validity is non-decreasing; it only removes nodes. Safe to
                  apply unconditionally before reporting shape metrics.

Why loops are pure waste here: masking forces a legal walk, so a wandering model
produces long paths, and a segment between two visits of the same vertex is a
cycle that contributes nothing. All 1,000/1,000 v6 reference paths are simple
(zero revisits), so the target distribution puts no mass on non-simple paths.

Measured (LE seen blk2, on P1P3 output): 34% of Adj+IW paths had loops, 11.6
nodes (29%) removed on average, gen_len 40.7 -> 29.1, DTW 0.143 -> 0.134,
valid/arrival stay 1.000, LCS -0.003. Side effect: every arm converges to 29-30
hops, which removes the length bias of LCS/|ref| (a recall measure with no
penalty for extra nodes).

A greedy shortcut pass (jump to the farthest graph-adjacent later vertex) was
also tried and gave almost nothing on top of loop cutting (29.10 -> 28.88), so it
is deliberately not included: loops are the whole story.

Output: {tag}_{cfg}_{out_variant}_em_pc_records.csv, same schema as the eval
scripts write, so collect_va_table.py -variant {out_variant} reads it unchanged.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import csv
import json
import pickle
from collections import defaultdict
from os.path import join

import numpy as np

SPLIT_SEED = 777


def loopcut(p):
    """Remove the segment between two visits of the same vertex -> simple path.

    Identical to models_seq.bd_models.simplify_path (kept local so this tool can
    run without importing the model module). See that docstring for the walk-through
    and the invariants: endpoints preserved, validity non-decreasing, length shrinks.
    """
    out, pos = [], {}
    for v in p:
        if v in pos:
            out = out[:pos[v] + 1]
            pos = {u: i for i, u in enumerate(out)}
        else:
            pos[v] = len(out)
            out.append(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-res_path", type=str, required=True)
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-dver", type=str, default="v6")
    ap.add_argument("-family", type=str, required=True)
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-pairs", type=str, required=True)
    ap.add_argument("-cfgs", type=str, required=True)
    ap.add_argument("-variant", type=str, default="P1P3", help="input variant to read")
    ap.add_argument("-out_variant", type=str, default="P1P3L", help="output variant name")
    ap.add_argument("-eval_num", type=int, default=1000)
    args = ap.parse_args()

    A = pickle.load(open(join(args.porto,
        f"porto_shrink_A_{args.dver}-{args.family}_except_0.ts"), "rb")).bool().numpy()
    sp = pickle.load(open(join(args.porto,
        f"porto_shrink_SP_{args.dver}-{args.family}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp))
    real = [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:args.eval_num]
    dests = [p[-1] for p in real]
    # od -> set of reference lengths, only so the 'em' column can be filled with its
    # original definition. The report no longer uses that column (see the metric
    # section of report_src/build_porto_LE_dcbg.py for why it was dropped).
    od_len = defaultdict(set)
    for q in sp:
        q = list(map(int, q))
        if len(q) >= 2:
            od_len[(q[0], q[-1])].add(len(q))

    groups = [tuple(x.split(":", 1)) for x in args.pairs.split(",")]
    blocks = [int(x) for x in args.blocks.split(",")]
    cfgs = args.cfgs.split(",")
    n_done = n_skip = 0
    stats = []

    for regime, pat in groups:
        for blk in blocks:
            tag = pat.format(B=blk)
            for cfg in cfgs:
                src = join(args.res_path, "em_pc",
                           f"{tag}_{cfg}_{args.variant}_em_pc_records.csv")
                try:
                    rows = list(csv.DictReader(open(src)))
                except FileNotFoundError:
                    n_skip += 1
                    continue
                out_rows, before, after = [], [], []
                for r in rows:
                    i = int(r["idx"])
                    gp = json.loads(r["gen_path"])
                    q = loopcut(gp) if len(gp) >= 1 else gp
                    before.append(len(gp))
                    after.append(len(q))
                    ok = all(bool(A[u, v]) for u, v in zip(q[:-1], q[1:])) if len(q) >= 2 else False
                    sl = min(od_len[(q[0], q[-1])]) if len(q) >= 2 and (q[0], q[-1]) in od_len else None
                    out_rows.append({
                        "idx": i, "start": q[0] if q else "", "end": q[-1] if q else "",
                        "gen_path": json.dumps(q), "gen_len": len(q),
                        "valid": int(ok),
                        "invalid_nodes": "[]", "invalid_edges": "[]",
                        "shortest_len": sl if sl is not None else "",
                        "shortest_lengths_same_od": "[]",
                        "em": int(bool(ok) and sl is not None and len(q) == sl),
                        "shorter_feasible_lengths": "[]",
                        "num_shorter_feasible_lengths": 0,
                        "graph_rank": "", "pc": 0.0,
                    })
                dst = join(args.res_path, "em_pc",
                           f"{tag}_{cfg}_{args.out_variant}_em_pc_records.csv")
                with open(dst, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
                    w.writeheader()
                    w.writerows(out_rows)
                cut = np.mean(before) - np.mean(after)
                had = float(np.mean([b > a for b, a in zip(before, after)]))
                stats.append((regime, blk, cfg, np.mean(before), np.mean(after), cut, had))
                n_done += 1

    print(f"{'regime':>7}{'blk':>5} {'cfg':<16}{'len(in)':>9}{'len(out)':>9}"
          f"{'제거':>7}{'루프有':>8}")
    for regime, blk, cfg, b, a, cut, had in stats:
        print(f"{regime:>7}{blk:>5} {cfg:<16}{b:>9.2f}{a:>9.2f}{cut:>7.2f}{had:>8.1%}")
    print(f"\nwritten {n_done} record files ({args.out_variant}), skipped {n_skip}")
    print(f"다음: collect_va_table.py -variant {args.out_variant} 로 표를 뽑는다")

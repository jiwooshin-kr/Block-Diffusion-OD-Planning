"""
Flatten the va_table_* JSONs into one tidy CSV per dataset, ready for plotting.

  python tools/collect/make_plot_csv.py -variant LE
  python tools/collect/make_plot_csv.py -variant LP -out /tmp/lp.csv

Columns: post, regime, blk, arm, family, va, valid, arrival, dtw_sum, lcs,
         dtw_per_pair, lcs_norm, gen_len, ref_len

`post` is the post-processing stage, which is two independent axes:
  raw     model output as returned
  rawL    + simplification (loop cut). Fixes nothing, so arrival is preserved and
          validity is non-decreasing; only removes nodes. Same result the samplers
          now produce by default (simplify=True), since loop cutting is a pure
          function of the returned path.
  P1P3    + repair (P1 illegal-edge splice, P3 endpoint patch). Fixes failures, so
          valid&arrival saturates to 1.000 for every arm and stops being a model
          signal. Only adds nodes.
  P1P3L   repair then simplification.

Reported metrics (2026-08-21 convention):
  va        valid & arrival, needs no reference, immune to length effects
  dtw_sum   haversine DTW accumulated cost in METRES, no normalisation. Lower is
            better. Not normalised on purpose: dividing by max(n, m) makes the
            divisor depend on the generated length, which rewards a path for
            staying near the reference while being long.
  lcs       raw longest-common-subsequence node count. Higher is better. The
            reference set is identical for every arm, so no normalisation is needed.
  gen_len / ref_len  length context; read dtw_sum and lcs against them.
dtw_per_pair and lcs_norm are the normalised forms, kept for cross-checks.

Ceilings (same units) live in sets_v6/{V}/res/ceiling_v6{V}.json: two independent
draws from the same choice-set distribution do not agree, so the reference-based
metrics cannot reach 0 / len(ref).
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import csv
import json
import re
from os.path import dirname, join

ROOT = dirname(dirname(dirname(_pathlib.Path(__file__).resolve().as_posix())))
BLKS = [1, 2, 4, 8, 16, 32, 64]
POSTS = [("raw", ""), ("rawL", "_rawL"), ("P1P3", "_P1P3"), ("P1P3L", "_P1P3L")]
IW = {"base": "none", "adjonly": "iw", "modelD": "iw", "adj+modelD": "iw"}
FIELDS = ["post", "regime", "blk", "arm", "family", "va", "valid", "arrival",
          "dtw_sum", "lcs", "dtw_per_pair", "lcs_norm", "gen_len", "ref_len"]


def family(arm):
    return IW.get(arm, "dcbg" if arm.startswith("dcbg") else "other")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    ap.add_argument("-res_path", type=str, default="")
    ap.add_argument("-out", type=str, default="")
    args = ap.parse_args()
    V = args.variant
    res = args.res_path or join(ROOT, f"sets_v6/{V}/res")
    out = args.out or join(res, "plot_data.csv")

    rows = []
    for post, sfx in POSTS:
        p = join(res, f"va_table_v6{V}_dcbg{sfx}.json")
        try:
            J = json.load(open(p))
        except FileNotFoundError:
            print(f"MISSING {p} -- skipping post={post}")
            continue
        for k, m in J.items():
            g = re.match(r"^(seen|unseen)_blk(\d+)_(.+)$", k)
            if not g:
                continue
            regime, blk, arm = g.group(1), int(g.group(2)), g.group(3)
            rows.append({
                "post": post, "regime": regime, "blk": blk, "arm": arm,
                "family": family(arm),
                "va": round(m.get("valid_and_arrival", float("nan")), 6),
                "valid": round(m.get("valid", float("nan")), 6),
                "arrival": round(m.get("arrival", float("nan")), 6),
                "dtw_sum": round(m["dtw_sum"], 3) if "dtw_sum" in m else "",
                "lcs": round(m["lcs"], 4) if "lcs" in m else "",
                "dtw_per_pair": round(m["dtw"], 4) if "dtw" in m else "",
                "lcs_norm": round(m["lcs_norm"], 6) if "lcs_norm" in m else "",
                "gen_len": round(m["gen_len"], 2) if "gen_len" in m else "",
                "ref_len": round(m["ref_len"], 2) if "ref_len" in m else "",
            })
    # Non-learned reference (tools/eval/shortest_baseline.py). Replicated across
    # every post/regime/blk cell so a plot can draw it as a horizontal line without
    # special-casing: BFS has no block structure, no training, and nothing for
    # repair or simplification to change.
    try:
        S = json.load(open(join(res, f"shortest_v6{V}.json")))
    except FileNotFoundError:
        S = None
        print(f"shortest_v6{V}.json not found -- reference row omitted")
    if S:
        cells = {(r["post"], r["regime"], r["blk"]) for r in rows}
        for post, regime, blk in cells:
            rows.append({
                "post": post, "regime": regime, "blk": blk,
                "arm": "shortest", "family": "ref",
                "va": round(S["valid_and_arrival"], 6),
                "valid": round(S["valid"], 6), "arrival": round(S["arrival"], 6),
                "dtw_sum": round(S["dtw_sum"], 3), "lcs": round(S["lcs"], 4),
                "dtw_per_pair": round(S["dtw"], 4),
                "lcs_norm": round(S["lcs_norm"], 6),
                "gen_len": round(S["gen_len"], 2), "ref_len": round(S["ref_len"], 2),
            })

    rows.sort(key=lambda r: (r["post"], r["regime"], r["blk"], r["arm"]))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"written {out}  ({len(rows)} rows)")

    try:
        c = json.load(open(join(res, f"ceiling_v6{V}.json")))
        keys = ["dtw_sum", "dtw", "lcs", "lcs_norm", "exact_path", "em",
                "ref_len", "n_pairs"]
        print("ceiling: " + "  ".join(
            f"{k}={c[k]:.4f}" if isinstance(c.get(k), float) else f"{k}={c.get(k)}"
            for k in keys if k in c))
    except FileNotFoundError:
        print("ceiling json not found")
    posts = sorted({r["post"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    print(f"posts: {posts}")
    print(f"arms ({len(arms)}): {arms}")

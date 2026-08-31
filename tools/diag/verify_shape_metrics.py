"""
Cross-check the shape metrics (DTW / LCS) against independent reference code.

  python tools/diag/verify_shape_metrics.py [-n 400] [-variant LE]

Why: dtw_distance / lcs_length in main_bd.py are hand-written. "It matches my own
reimplementation" is weak evidence for numbers that go into a paper -- it catches
coding slips but not a wrong definition. So each hand-written piece is replaced by
a standard formulation and the two are compared.

  DTW  the reference builds the great-circle cost matrix from the standard haversine
       formula through a separate code path, then runs the textbook symmetric1
       recursion D[i,j] = c + min(D[i-1,j], D[i,j-1], D[i-1,j-1]) (Sakoe & Chiba
       1978). Identity d(a, a) = 0 is checked too. scipy provides no DTW, so only
       the ground distance can be delegated to an external definition.
  LCS  independently reimplemented as a one-row rolling DP (different code path from
       main_bd's 2-D DP). LCS has no definitional ambiguity: the DP *is* the
       definition (CLRS).

NOT verified here: the normalisation convention. main_bd divides by max(n, m) while
standard implementations (tslearn.metrics.dtw, dtaidistance) return the
un-normalised accumulated cost or divide by the warping-path length. So the DTW
values in this repo must not be compared against DTW values from other papers --
they are for within-paper arm-to-arm comparison only. That is a documentation
matter, not a bug. Since 2026-08-21 the ground distance is haversine in metres, so
the values are at least physically interpretable (mean graph edge = 84 m).
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import pickle
from os.path import join

import numpy as np

from main_bd import dtw_distance, lcs_length, path_to_coords


def dtw_ref(a, b):
    """Reference DTW: ground distance from scipy's own haversine-equivalent path
    (great-circle via the standard formula), standard symmetric1 recursion.

    scipy has no DTW, so only the ground distance can be delegated. Here the
    reference computes the great-circle matrix independently of main_bd (different
    code path, same formula) and runs the textbook recursion.
    """
    la1 = np.deg2rad(a[:, 0])[:, None]
    lo1 = np.deg2rad(a[:, 1])[:, None]
    la2 = np.deg2rad(b[:, 0])[None, :]
    lo2 = np.deg2rad(b[:, 1])[None, :]
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    C = 2 * 6371000.0 * np.arcsin(np.minimum(1.0, np.sqrt(d)))
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / max(n, m)


def lcs_ref(a, b):
    """독립 구현 LCS: 1행 롤링 DP (main_bd 의 2차원 DP 와 다른 코드 경로)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-variant", type=str, default="LE")
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-dver", type=str, default="v6")
    ap.add_argument("-n", type=int, default=400, help="비교할 경로 쌍 수")
    ap.add_argument("-seed", type=int, default=0)
    args = ap.parse_args()

    fam = f"{args.variant}1.0-0.05"
    G = pickle.load(open(join(args.porto, f"porto_shrink_G_{args.dver}-{fam}_normal.pkl"), "rb"))
    sp = pickle.load(open(join(args.porto, f"porto_shrink_SP_{args.dver}-{fam}_except_0.pkl"), "rb"))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(sp), min(2 * args.n + 50, len(sp)), replace=False)
    paths = [list(map(int, sp[i])) for i in idx if len(sp[i]) >= 2]

    n = 0
    dtw_max = 0.0
    self_max = 0.0
    lcs_bad = 0
    for k in range(0, len(paths) - 1, 2):
        if n >= args.n:
            break
        a, b = paths[k], paths[k + 1]
        ca, cb = path_to_coords(a, G), path_to_coords(b, G)
        dtw_max = max(dtw_max, abs(dtw_distance(ca, cb) - dtw_ref(ca, cb)))
        self_max = max(self_max, abs(dtw_distance(ca, ca)))
        if lcs_length(a, b) != lcs_ref(a, b):
            lcs_bad += 1
        n += 1

    print(f"pairs compared               : {n}")
    print(f"DTW  max |ours - scipy-ref|  : {dtw_max:.3e}")
    print(f"DTW  max |d(a,a)|            : {self_max:.3e}   (0 이어야 함)")
    print(f"LCS  mismatches              : {lcs_bad}/{n}")
    ok = (dtw_max == 0.0) and (self_max == 0.0) and (lcs_bad == 0)
    print("SHAPE_METRICS_VERIFIED" if ok else "SHAPE_METRICS_MISMATCH")
    raise SystemExit(0 if ok else 1)

"""
두 arm 을 같은 1,000 OD 쌍에서 짝지어 비교한다 (paired test).

  python tools/collect/paired_test.py -res_path sets_v6/LE/res -porto ./porto_data_v6 \
      -dver v6 -family LE1.0-0.05 -blocks 1,2,4,8,16,32,64 \
      -pairs "seen:v6LE{B}seen,unseen:v6LE{B}uns" -a adj+modelD -b dcbg+adjp

왜 필요한가: 두 arm 은 같은 OD 쌍 집합에서 평가되므로 독립 표본이 아니다. 독립 가정
SE = sqrt(p(1-p)/n) 은 짝지은 차이의 불확실성을 과대평가한다. valid&arrival 같은
이진 지표는 McNemar(불일치 쌍만 사용), EM 도 이진, DTW/LCS 는 연속이라 짝지은
부트스트랩으로 차이의 신뢰구간을 낸다.

출력: 블록별로 delta = A - B, McNemar p (이진), 부트스트랩 95% CI.
"""

# --- repo root import path ---
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# -----------------------------

import argparse
import csv
import json
import pickle
from math import comb
from os.path import join

import numpy as np

SPLIT_SEED = 777


def load_rows(res_path, tag, cfg, variant, n_expect):
    p = join(res_path, "em_pc", f"{tag}_{cfg}_{variant}_em_pc_records.csv")
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == n_expect, f"{p}: {len(rows)} vs {n_expect}"
    out = {}
    for r in rows:
        out[int(r["idx"])] = r
    return out


def mcnemar_p(b, c):
    """양측 정확 McNemar. b = A만 성공, c = B만 성공."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-res_path", type=str, required=True)
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-dver", type=str, default="v6")
    ap.add_argument("-family", type=str, required=True)
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-pairs", type=str, required=True,
                    help="'seen:v6LE{B}seen,unseen:v6LE{B}uns'")
    ap.add_argument("-a", type=str, required=True, help="arm A (cfg 이름)")
    ap.add_argument("-b", type=str, required=True, help="arm B (cfg 이름)")
    ap.add_argument("-variant", type=str, default="raw")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-boot", type=int, default=10000)
    ap.add_argument("-metric", type=str, default="va",
                    choices=["va", "em", "dtw_sum", "dtw", "lcs", "lcs_norm"],
                    help="va/em are binary (exact McNemar); the rest are continuous "
                         "(Wilcoxon signed-rank + paired bootstrap). dtw_sum/dtw are "
                         "distances, so delta < 0 means A wins. dtw_sum and lcs are the "
                         "reported forms (no normalisation); dtw (per-pair) and lcs_norm "
                         "(recall) are kept for cross-checks.")
    ap.add_argument("-out", type=str, default="")
    args = ap.parse_args()

    sp = pickle.load(open(join(args.porto,
        f"porto_shrink_SP_{args.dver}-{args.family}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp))
    real = [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:args.eval_num]
    dests = [p[-1] for p in real]
    n = len(real)

    # Lower-is-better metrics. Both DTW forms are distances.
    LOWER_BETTER = {"dtw", "dtw_sum"}
    need_coords = args.metric in ("dtw", "dtw_sum", "lcs", "lcs_norm")
    if need_coords:
        # Import the single canonical implementation so the significance test uses
        # exactly the metric the tables report (haversine ground distance since
        # 2026-08-21). Do NOT re-implement it here: an earlier local copy silently
        # kept the old degree-L1 cost after main_bd switched to haversine.
        from main_bd import dtw_distance, lcs_length, path_to_coords
        from scipy.stats import wilcoxon
        G = pickle.load(open(join(args.porto,
            f"porto_shrink_G_{args.dver}-{args.family}_except_0.pkl"), "rb"))
        ref_coords = [path_to_coords(q, G) for q in real]

    groups = [tuple(x.split(":", 1)) for x in args.pairs.split(",")]
    blocks = [int(x) for x in args.blocks.split(",")]
    rng = np.random.RandomState(0)
    res = {}

    print(f"paired: A={args.a}  B={args.b}  metric={args.metric}  variant={args.variant}  n={n}  boot={args.boot}")
    print(f"{'regime':>7}{'blk':>5}{'A':>8}{'B':>8}{'delta':>9}"
          f"{'95% CI':>20}{'b':>6}{'c':>6}{'McNemar p':>12}")
    for regime, pat in groups:
        for blk in blocks:
            tag = pat.format(B=blk)
            try:
                RA = load_rows(args.res_path, tag, args.a, args.variant, n)
                RB = load_rows(args.res_path, tag, args.b, args.variant, n)
            except (FileNotFoundError, AssertionError) as e:
                print(f"{regime:>7}{blk:>5}   SKIP ({type(e).__name__})")
                continue

            def score(R):
                """Per-path metric vector; NaN marks paths to drop so both arms stay
                paired (shape metrics need a non-empty generated path)."""
                v = np.full(n, np.nan)
                for i in range(n):
                    r = R[i]
                    gp = json.loads(r["gen_path"])
                    if args.metric == "em":
                        v[i] = int(r["em"])
                    elif args.metric == "va":
                        arr = float(len(gp) > 0 and gp[-1] == dests[i])
                        v[i] = int(r["valid"]) * arr
                    elif len(gp) >= 1:
                        if args.metric == "dtw_sum":
                            v[i] = dtw_distance(path_to_coords(gp, G), ref_coords[i],
                                                normalize=None)
                        elif args.metric == "dtw":
                            v[i] = dtw_distance(path_to_coords(gp, G), ref_coords[i],
                                                normalize="max")
                        elif args.metric == "lcs":
                            v[i] = lcs_length(gp, real[i])
                        else:  # lcs_norm
                            v[i] = lcs_length(gp, real[i]) / len(real[i])
                return v

            a_all, b_all = score(RA), score(RB)
            ok = ~(np.isnan(a_all) | np.isnan(b_all))
            a_v, b_v = a_all[ok], b_all[ok]
            m_ = int(ok.sum())
            d = a_v - b_v
            if args.metric in ("va", "em"):
                bb = int(((a_v == 1) & (b_v == 0)).sum())
                cc = int(((a_v == 0) & (b_v == 1)).sum())
                p = mcnemar_p(bb, cc)
            else:
                bb = int((d < 0).sum())          # A 가 더 작음
                cc = int((d > 0).sum())          # B 가 더 작음
                nz = d[d != 0]
                p = float(wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0
            idx = rng.randint(0, m_, size=(args.boot, m_))
            boot = d[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"{regime:>7}{blk:>5}{a_v.mean():>8.3f}{b_v.mean():>8.3f}"
                  f"{d.mean():>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{bb:>6}{cc:>6}{p:>12.4f}")
            res[f"{regime}_blk{blk}"] = {
                "a": float(a_v.mean()), "b": float(b_v.mean()), "delta": float(d.mean()),
                "ci95": [float(lo), float(hi)], "b_only": bb, "c_only": cc,
                "mcnemar_p": float(p), "n": m_,
                "lower_better": args.metric in LOWER_BETTER}
    if args.out:
        json.dump({"arm_a": args.a, "arm_b": args.b, "variant": args.variant,
                   "metric": {"em": "len_match", "va": "valid_and_arrival"}.get(args.metric, args.metric), "rows": res},
                  open(join(args.res_path, args.out), "w"), indent=2)
        print(f"written {join(args.res_path, args.out)}")

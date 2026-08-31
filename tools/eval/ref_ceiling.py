"""
참조 대비 지표(EM / DTW / LCS)의 천장 측정.

이 데이터의 정답 경로는 choice set C_od 위의 PSL 확률로 **확률적으로 선택된 표본**이다
(LE = Link Elimination / LP = Link Penalization으로 C_od 생성, Beta > 0 이면 PSL).
따라서 같은 분포에서 뽑은 두 표본조차 서로 일치하지 않으며, 그 일치율이 곧
"완벽하게 학습된 모델"이 도달할 수 있는 상한이다. EM = 1.0, LCS = 1.0, DTW = 0 이 아니다.

두 번째 표본을 어디서 구하나: normal 과 except_1..K 는 같은 OD 집합에 대한 각각의
독립 추출이다. 두 시나리오의 제거 edge가 어느 쪽 경로도 건드리지 않은 OD만 고르면
(= 양쪽 그래프에서 두 경로가 모두 적법) 같은 choice set·같은 분포에서의 독립 추출 2개로 볼 수 있다.

  python tools/eval/ref_ceiling.py -family LE1.0-0.05 -dver v6 -porto ./porto_data_v6 \
      -res_path sets_v6/LE/res -out ceiling_v6LE.json
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------

import argparse
import json
import pickle
from os.path import join

import numpy as np

from main_bd import dtw_distance, lcs_length, path_to_coords

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="LE1.0-0.05")
    ap.add_argument("-dver", type=str, default="v6")
    ap.add_argument("-porto", type=str, default="./porto_data_v6")
    ap.add_argument("-eval_num", type=int, default=1000)
    ap.add_argument("-alts", type=str, default="normal,except_1,except_2,except_3,except_4,except_5",
                    help="두 번째 표본을 가져올 시나리오들 (참조는 except_0)")
    ap.add_argument("-res_path", type=str, default="sets_v6/LE/res")
    ap.add_argument("-out", type=str, default="ceiling_v6LE.json")
    args = ap.parse_args()

    L = lambda k, t: pickle.load(open(
        join(args.porto, f"porto_shrink_{k}_{args.dver}-{args.family}_{t}.{'ts' if k == 'A' else 'pkl'}"), "rb"))

    ref_sp = L("SP", "except_0")
    A0 = L("A", "except_0").bool()
    G0 = L("G", "except_0")
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(ref_sp))
    ev = [i for i in perm[:1000] if len(ref_sp[i]) >= 2][:args.eval_num]
    print(f"참조: except_0, 평가 OD {len(ev)}개")

    ident = em = 0
    # Report the shape metrics in the same form the tables use: un-normalised
    # haversine DTW sum (metres) and the raw LCS count. The normalised forms are
    # kept alongside so older numbers stay checkable.
    dtws, dtwsum, lcsn, lcss, dlen, reflen = [], [], [], [], [], []
    n = 0
    per_alt = {}
    for tag in args.alts.split(","):
        sp = L("SP", tag)
        A = L("A", tag).bool()
        c = 0
        for i in ev:
            r = [int(v) for v in ref_sp[i]]          # 참조 추출 (except_0)
            a = [int(v) for v in sp[i]]              # 또 하나의 추출
            if len(a) < 2:
                continue
            # 두 경로가 양쪽 그래프에서 모두 적법할 때만 = 같은 choice set으로 간주
            if not all(A[u, v] for u, v in zip(r[:-1], r[1:])):
                continue
            if not all(A0[u, v] for u, v in zip(a[:-1], a[1:])):
                continue
            n += 1
            c += 1
            ident += (a == r)
            em += (len(a) == len(r))
            dlen.append(abs(len(a) - len(r)))
            ca, cr = path_to_coords(a, G0), path_to_coords(r, G0)
            dtws.append(dtw_distance(ca, cr))
            dtwsum.append(dtw_distance(ca, cr, normalize=None))
            l = lcs_length(a, r)
            lcss.append(l)
            lcsn.append(l / len(r))
            reflen.append(len(r))
        per_alt[tag] = c
        print(f"  {tag:<12} 사용 가능 OD {c}")
        del sp, A

    d = np.array(dlen)
    out = {
        "n_pairs": n, "per_alt": per_alt, "eval_num": len(ev),
        "exact_path": ident / n,
        "em": em / n,
        "em_tol1": float((d <= 1).mean()),
        "em_tol2": float((d <= 2).mean()),
        "dtw_sum": float(np.mean(dtwsum)),
        "dtw": float(np.mean(dtws)),
        "lcs": float(np.mean(lcss)),
        "lcs_norm": float(np.mean(lcsn)),
        "ref_len": float(np.mean(reflen)),
        "abs_len_diff": float(d.mean()),
    }
    print("\n=== 같은 분포에서의 독립 추출 2개 = 지표의 천장 ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:<14} {v:.4f}")
    p = join(args.res_path, args.out)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwritten {p}")

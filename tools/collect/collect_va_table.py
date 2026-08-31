"""
Assemble the 10-column guidance table (base / adj+IW x {valid, arrival,
valid&arrival, EM, PC}) from saved em_pc per-path records of three_way runs.
Arrival is recomputed from each record's gen_path endpoint vs the true
destination of the reserved eval pair (SPLIT_SEED=777).

  python tools/collect/collect_va_table.py -family 0.05 -tag_seen 'v4m{B}seen' -tag_uns 'v4m{B}uns' \
      -out va_table_v4_f005.json
  python tools/collect/collect_va_table.py -family 0.1 -tag_seen 'v4f01m{B}seen' -tag_uns 'v4f01m{B}uns' \
      -out va_table_v4_f01.json
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import argparse
import csv
import json
import pickle
from os.path import join

import numpy as np

SPLIT_SEED = 777
DEFAULT_CFGS = ["base", "modelD", "adj+modelD"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-tag_seen", type=str, default="", help="e.g. 'v4m{B}seen'")
    ap.add_argument("-tag_uns", type=str, default="", help="e.g. 'v4m{B}uns'")
    ap.add_argument("-pairs", type=str, default="",
                    help="explicit label:tagpattern list, e.g. 'fh:abl{B}fh,l2r:abl{B}l2r'; "
                         "overrides -tag_seen/-tag_uns and names the output keys")
    ap.add_argument("-cfgs", type=str, default=",".join(DEFAULT_CFGS),
                    help="comma-separated configs to collect")
    ap.add_argument("-variant", type=str, default="raw",
                    help="후처리 변형: raw | P1(불법 edge splice) | P3(끝점 patch) | P1P3")
    ap.add_argument("-eval_num", type=int, default=1000,
                    help="three_way 실행 때의 -eval_num 과 반드시 동일해야 함")
    ap.add_argument("-out", type=str, required=True)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-dver", type=str, default="v4", help="데이터 버전 접두 (v4 | v6)")
    ap.add_argument("-shape", type=int, default=0,
                    help="1 = DTW/LCS(참조 경로 대비 형상 지표)도 계산. PC가 의미 없는 데이터용")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    args = ap.parse_args()

    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_{args.dver}-{args.family}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:args.eval_num]
    dests = [p[-1] for p in real]

    if args.pairs:
        groups = [tuple(x.split(":", 1)) for x in args.pairs.split(",")]
    else:
        assert args.tag_seen and args.tag_uns, "give -pairs, or both -tag_seen and -tag_uns"
        groups = [("seen", args.tag_seen), ("unseen", args.tag_uns)]
    cfg_list = args.cfgs.split(",")

    # -shape: 참조 경로 대비 형상 지표(DTW/LCS). 데이터의 참조 경로가 최단경로가 아닌
    # 경우(v6) PC는 의미를 잃으므로 이쪽을 쓴다. 구현은 main_bd의 것을 그대로 재사용
    # (기존 보고서의 DTW/LCS 정의와 동일 — L1x100 ground distance, max(n,m) 정규화).
    if args.shape:
        from main_bd import dtw_distance, lcs_length, path_to_coords
        G_exc = pickle.load(open(
            join(args.porto, f"porto_shrink_G_{args.dver}-{args.family}_except_0.pkl"), "rb"))
        ref_coords = [path_to_coords(p, G_exc) for p in real]

    out = {}
    for regime, pat in groups:
        for blk in [int(b) for b in args.blocks.split(",")]:
            for cfg in cfg_list:
                path = join(args.res_path, "em_pc",
                            f"{pat.format(B=blk)}_{cfg}_{args.variant}_em_pc_records.csv")
                try:
                    rows = list(csv.DictReader(open(path)))
                except FileNotFoundError:
                    print(f"MISSING {path}")
                    continue
                assert len(rows) == len(real), f"{path}: {len(rows)} rows vs {len(real)}"
                valid = np.array([int(r["valid"]) for r in rows])
                em = np.array([int(r["em"]) for r in rows])
                pc = np.array([float(r["pc"]) for r in rows])
                arr = np.zeros(len(rows))
                # Shape metrics. Both an un-normalised and a normalised form are
                # stored so the report can choose:
                #   dtw_sum  accumulated DTW cost, no divisor. Report next to
                #            gen_len -- the sum grows with the longer sequence.
                #   dtw      / max(n, m). Its divisor depends on the generated
                #            length, which REWARDS staying near the reference while
                #            being long (this is why loop removal looked like a
                #            regression under it).
                #   lcs      raw longest-common-subsequence count. The reference set
                #            is identical for every arm, so this is already directly
                #            comparable -- same ranking as lcs_norm.
                #   lcs_norm / len(ref), i.e. order-preserving recall in [0, 1].
                # Only empty paths are excluded; non-arriving and illegal paths are
                # included, so raw-variant shape numbers mix in the arrival failure.
                dtws, dtwsum, lcss, lcsn, glen = [], [], [], [], []
                for r in rows:
                    i = int(r["idx"])
                    gp = json.loads(r["gen_path"])
                    arr[i] = float(len(gp) > 0 and gp[-1] == dests[i])
                    if args.shape and len(gp) >= 1:
                        # normalize is passed explicitly on both calls so this table
                        # never depends on dtw_distance's default.
                        gc = path_to_coords(gp, G_exc)
                        dtwsum.append(dtw_distance(gc, ref_coords[i], normalize=None))
                        dtws.append(dtw_distance(gc, ref_coords[i], normalize="max"))
                        l = lcs_length(gp, real[i])
                        lcss.append(l)
                        lcsn.append(l / len(real[i]))
                        glen.append(len(gp))
                key = f"{regime}_blk{blk}_{cfg}"
                out[key] = {
                    "valid": float(valid.mean()), "arrival": float(arr.mean()),
                    "valid_and_arrival": float((valid * arr).mean()),
                    "em": float(em.mean()), "pc": float(pc.mean()),
                }
                if args.shape:
                    out[key].update({
                        "dtw_sum": float(np.mean(dtwsum)),
                        "dtw": float(np.mean(dtws)), "lcs": float(np.mean(lcss)),
                        "lcs_norm": float(np.mean(lcsn)), "gen_len": float(np.mean(glen)),
                        "ref_len": float(np.mean([len(real[int(r["idx"])]) for r in rows])),
                        "n_scored_shape": len(dtws),
                    })
                m = out[key]
                extra = (f" dtwSum={m['dtw_sum']:.2f} dtw={m['dtw']:.3f}"
                         f" lcs={m['lcs']:.2f} lcsN={m['lcs_norm']:.3f} len={m['gen_len']:.1f}"
                         if args.shape else f" pc={m['pc']:.3f}")
                print(f"{key:<30} valid={m['valid']:.3f} arr={m['arrival']:.3f} "
                      f"va={m['valid_and_arrival']:.3f} em={m['em']:.3f}{extra}")

    with open(join(args.res_path, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {join(args.res_path, args.out)}")

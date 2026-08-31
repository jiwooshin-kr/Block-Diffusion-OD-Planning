"""
Aggregate the three-seed replication (seeds 7, 8, 9; fam 0.05, l2r, seen)
into per-cell mean/std over {valid, arrival, valid_and_arrival, em, pc},
reconstructed from the per-path em_pc records exactly as collect_va_table /
collect_dcbg_table do (arrival = generated endpoint == reserved destination).

Seed-7 runs keep their original tags; 8/9 carry seed-suffixed tags:
  base/adjonly/modelD/adj+modelD : abl{B}l2r_{cfg}      | abl{B}l2rs{S}_{cfg}
  Adj+D-CBG exact g1/g4          : DCBG_mask{B}_g{G}_adj_adjp_v4l2r[_s{S}]

  python tools/collect/collect_seed_stats.py            -> sets_res/seed_stats_f005.json
"""

# --- repo root import path (2026-08 정리: 이 파일은 하위 폴더로 이동됨) ---
# 실행은 repo root 기준: python tools/<group>/<file>.py ...
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
# ---------------------------------------------------------------------


import csv
import json
import pickle
from os.path import join

import numpy as np

SPLIT_SEED = 777
BLKS = [1, 2, 4, 8, 16, 32, 64]
MET = ["valid", "arrival", "valid_and_arrival", "em", "pc"]

sp_exc = pickle.load(open("./porto_data/porto_shrink_SP_v4-0.05_except_0.pkl", "rb"))
perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:1000]
dests = [p[-1] for p in real]


def metrics(tag):
    path = f"./sets_res/em_pc/{tag}_raw_em_pc_records.csv"
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == len(real), f"{tag}: {len(rows)} rows"
    valid = np.array([int(r["valid"]) for r in rows])
    em = np.array([int(r["em"]) for r in rows])
    pc = np.array([float(r["pc"]) for r in rows])
    arr = np.zeros(len(rows))
    for r in rows:
        i = int(r["idx"])
        gp = json.loads(r["gen_path"])
        arr[i] = float(len(gp) > 0 and gp[-1] == dests[i])
    return dict(valid=float(valid.mean()), arrival=float(arr.mean()),
                valid_and_arrival=float((valid * arr).mean()),
                em=float(em.mean()), pc=float(pc.mean()))


ARMS = {
    "base":       lambda b, s: f"abl{b}l2r{s}_base",
    "adjonly":    lambda b, s: f"abl{b}l2r{s}_adjonly",
    "modelD":     lambda b, s: f"abl{b}l2r{s}_modelD",
    "adj+modelD": lambda b, s: f"abl{b}l2r{s}_adj+modelD",
    "adjD_exact_g1": lambda b, s: f"DCBG_mask{b}_g1.0_adj_adjp_v4l2r{s}",
    "adjD_exact_g4": lambda b, s: f"DCBG_mask{b}_g4.0_adj_adjp_v4l2r{s}",
}
SEED_SFX = {  # seed -> (abl sfx, dcbg sfx)
    7: ("", ""), 8: ("s8", "_s8"), 9: ("s9", "_s9"),
}

out = {}
for arm, tagf in ARMS.items():
    dcbg = arm.startswith("adjD")
    for b in BLKS:
        vals = {m: [] for m in MET}
        for seed, (sa, sd) in SEED_SFX.items():
            tag = tagf(b, sd if dcbg else sa)
            try:
                m = metrics(tag)
            except FileNotFoundError:
                print(f"MISSING {tag}")
                continue
            for k in MET:
                vals[k].append(m[k])
        if not vals["valid"]:
            continue
        out[f"{arm}_blk{b}"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": len(v), "vals": [round(x, 4) for x in v]}
            for k, v in vals.items()}
        r = out[f"{arm}_blk{b}"]
        print(f"{arm:>14}_blk{b:<3} n={r['valid']['n']}  " + "  ".join(
            f"{k}={r[k]['mean']:.3f}±{r[k]['std']:.3f}" for k in MET))

with open("./sets_res/seed_stats_f005.json", "w") as f:
    json.dump(out, f, indent=2)
print("written ./sets_res/seed_stats_f005.json")

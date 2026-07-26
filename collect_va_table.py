"""
Assemble the 10-column table for RESULTS_BD §4.1/§4.2 from saved em_pc records:
base / adj+IW x {valid, arrival, valid&arrival, EM, PC}, blocks 1-64,
seen (disc=except_0) and unseen (disc=except 1-99) regimes.

Reads sets_res/em_pc/{tag}_{cfg}_raw_em_pc_records.csv written by
three_way_postproc.py runs (v3 series, v2 backbones, seed 7) -- no GPU needed.
Arrival is recomputed from each record's gen_path endpoint vs the true
destination of the same reserved eval pair (SPLIT_SEED=777, identical
reconstruction to three_way_postproc.py).

  python collect_va_table.py
"""

import csv
import json
import pickle
from os.path import join

import numpy as np

SPLIT_SEED = 777
BLOCKS = [1, 2, 4, 8, 16, 32, 64]
TAGS = {
    "seen": {1: "v3mm1seen", 2: "v3mm2seen", 4: "v3m4", 8: "v3m8",
             16: "v3m16", 32: "v3m32", 64: "v3m64"},
    "unseen": {1: "v3mm1unseen", 2: "v3mm2unseen", 4: "v3m4uns", 8: "v3m8uns",
               16: "v3m16uns", 32: "v3m32uns", 64: "v3m64uns"},
}
CFGS = ["base", "modelD", "adj+modelD"]

if __name__ == "__main__":
    sp_exc = pickle.load(open("./porto_data/porto_shrink_SP_v4-0.05_except_0.pkl", "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:1000]
    dests = [p[-1] for p in real]

    out = {}
    for regime, tags in TAGS.items():
        for blk in BLOCKS:
            for cfg in CFGS:
                path = join("./sets_res/em_pc", f"{tags[blk]}_{cfg}_raw_em_pc_records.csv")
                try:
                    rows = list(csv.DictReader(open(path)))
                except FileNotFoundError:
                    print(f"MISSING {path}")
                    continue
                assert len(rows) == len(real), f"{path}: {len(rows)} rows vs {len(real)} pairs"
                valid = np.array([int(r["valid"]) for r in rows])
                em = np.array([int(r["em"]) for r in rows])
                pc = np.array([float(r["pc"]) for r in rows])
                arr = np.zeros(len(rows))
                for r in rows:
                    i = int(r["idx"])
                    gp = json.loads(r["gen_path"])
                    arr[i] = float(len(gp) > 0 and gp[-1] == dests[i])
                key = f"{regime}_blk{blk}_{cfg}"
                out[key] = {
                    "valid": float(valid.mean()), "arrival": float(arr.mean()),
                    "valid_and_arrival": float((valid * arr).mean()),
                    "em": float(em.mean()), "pc": float(pc.mean()),
                }
                m = out[key]
                print(f"{key:<28} valid={m['valid']:.3f} arr={m['arrival']:.3f} "
                      f"va={m['valid_and_arrival']:.3f} em={m['em']:.3f} pc={m['pc']:.3f}")

    with open("./sets_res/va_table.json", "w") as f:
        json.dump(out, f, indent=2)
    print("written ./sets_res/va_table.json")

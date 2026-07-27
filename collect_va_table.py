"""
Assemble the 10-column guidance table (base / adj+IW x {valid, arrival,
valid&arrival, EM, PC}) from saved em_pc per-path records of three_way runs.
Arrival is recomputed from each record's gen_path endpoint vs the true
destination of the reserved eval pair (SPLIT_SEED=777).

  python collect_va_table.py -family 0.05 -tag_seen 'v4m{B}seen' -tag_uns 'v4m{B}uns' \
      -out va_table_v4_f005.json
  python collect_va_table.py -family 0.1 -tag_seen 'v4f01m{B}seen' -tag_uns 'v4f01m{B}uns' \
      -out va_table_v4_f01.json
"""

import argparse
import csv
import json
import pickle
from os.path import join

import numpy as np

SPLIT_SEED = 777
CFGS = ["base", "modelD", "adj+modelD"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-tag_seen", type=str, required=True, help="e.g. 'v4m{B}seen'")
    ap.add_argument("-tag_uns", type=str, required=True, help="e.g. 'v4m{B}uns'")
    ap.add_argument("-out", type=str, required=True)
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    args = ap.parse_args()

    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{args.family}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:1000]
    dests = [p[-1] for p in real]

    out = {}
    for regime, pat in [("seen", args.tag_seen), ("unseen", args.tag_uns)]:
        for blk in [int(b) for b in args.blocks.split(",")]:
            for cfg in CFGS:
                path = join(args.res_path, "em_pc",
                            f"{pat.format(B=blk)}_{cfg}_raw_em_pc_records.csv")
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
                print(f"{key:<30} valid={m['valid']:.3f} arr={m['arrival']:.3f} "
                      f"va={m['valid_and_arrival']:.3f} em={m['em']:.3f} pc={m['pc']:.3f}")

    with open(join(args.res_path, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {join(args.res_path, args.out)}")

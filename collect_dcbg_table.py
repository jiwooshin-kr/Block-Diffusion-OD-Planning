"""
Assemble the §4.1-format 10-column table for D-CBG runs: valid / arrival /
valid&arrival / EM / PC, recomputed from eval_dcbg's per-path em_pc records
(arrival = generated endpoint == the reserved pair's destination, identical
reconstruction to collect_va_table.py).

Tags produced by eval_dcbg: DCBG_mask{B}_g{gamma}_adj[_fo]{res_suffix}_raw
with res_suffix _v4 (seen) / _v4uns (unseen).

  python collect_dcbg_table.py -gamma 1.0 -out dcbg_table_v4_f005.json
"""

import argparse
import csv
import json
import pickle
from os.path import join

import numpy as np

SPLIT_SEED = 777

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-family", type=str, default="0.05")
    ap.add_argument("-gamma", type=str, default="1.0")
    ap.add_argument("-blocks", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("-porto", type=str, default="./porto_data")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out", type=str, default="dcbg_table_v4_f005.json")
    args = ap.parse_args()

    sp_exc = pickle.load(open(join(args.porto, f"porto_shrink_SP_v4-{args.family}_except_0.pkl"), "rb"))
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(sp_exc))
    real = [list(map(int, sp_exc[i])) for i in perm[:1000] if len(sp_exc[i]) >= 2][:1000]
    dests = [p[-1] for p in real]

    out = {}
    for variant, fo in [("exact", ""), ("fo", "_fo")]:
        for regime, sfx in [("seen", "_v4"), ("unseen", "_v4uns")]:
            for blk in [int(b) for b in args.blocks.split(",")]:
                tag = f"DCBG_mask{blk}_g{args.gamma}_adj{fo}{sfx}"
                path = join(args.res_path, "em_pc", f"{tag}_raw_em_pc_records.csv")
                try:
                    rows = list(csv.DictReader(open(path)))
                except FileNotFoundError:
                    print(f"MISSING {tag}")
                    continue
                assert len(rows) == len(real), f"{tag}: {len(rows)} rows vs {len(real)}"
                valid = np.array([int(r["valid"]) for r in rows])
                em = np.array([int(r["em"]) for r in rows])
                pc = np.array([float(r["pc"]) for r in rows])
                arr = np.zeros(len(rows))
                for r in rows:
                    i = int(r["idx"])
                    gp = json.loads(r["gen_path"])
                    arr[i] = float(len(gp) > 0 and gp[-1] == dests[i])
                key = f"{variant}_{regime}_blk{blk}"
                out[key] = {
                    "valid": float(valid.mean()), "arrival": float(arr.mean()),
                    "valid_and_arrival": float((valid * arr).mean()),
                    "em": float(em.mean()), "pc": float(pc.mean()),
                }
                m = out[key]
                print(f"{key:<24} valid={m['valid']:.3f} arr={m['arrival']:.3f} "
                      f"va={m['valid_and_arrival']:.3f} em={m['em']:.3f} pc={m['pc']:.3f}")

    with open(join(args.res_path, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {join(args.res_path, args.out)}")

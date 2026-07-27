"""
Collect the post-processed (P1 / P3 / P1+P3) results for the v4 round by
parsing the evaluation logs, which already carry every stage plus the patch
length (three_way_postproc prints raw/P1/P3/P1P3; eval_dcbg prints raw/P1P3).
No GPU work, no recomputation.

  P1 = splice each illegal edge with a shortest detour on A_except0
  P3 = patch the endpoint to the destination with a shortest path

  python collect_postproc.py -out postproc_v4.json
"""

import argparse
import json
import re
from glob import glob
from os.path import join

LINE = re.compile(
    r"^(?P<tag>\S+)\s+arr=(?P<arr>[\d.]+)\s+valid=(?P<valid>[\d.]+)\s+em=(?P<em>[\d.]+)\s+"
    r"pc=(?P<pc>[\d.]+)\s+invE=(?P<invE>[\d.]+)%.*?remE=(?P<remE>[\d.]+)%\s+patch=(?P<patch>[\d.]+)")

# three_way tags: v4m{B}seen | v4m{B}uns | v4f01m{B}seen | v4f01m{B}uns
TW = re.compile(r"^v4(?P<f01>f01)?m(?P<blk>\d+)(?P<reg>seen|uns)_(?P<cfg>base|modelD|adj\+modelD)_"
                r"(?P<stage>raw|P1|P3|P1P3)$")
# dcbg tags: DCBG_mask{B}_g1.0_adj[_fo]_v4[uns]
DC = re.compile(r"^DCBG_mask(?P<blk>\d+)_g(?P<g>[\d.]+)_adj(?P<fo>_fo)?_v4(?P<uns>uns)?_"
                r"(?P<stage>raw|P1P3)$")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-log_dir", type=str, default="./sets_log")
    ap.add_argument("-res_path", type=str, default="./sets_res")
    ap.add_argument("-out", type=str, default="postproc_v4.json")
    args = ap.parse_args()

    out = {"guided": {}, "dcbg": {}}
    files = sorted(glob(join(args.log_dir, "v4_eval_job*.log"))) + \
        sorted(glob(join(args.log_dir, "v4_dcbgeval_job*.log")))
    for path in files:
        for line in open(path):
            m = LINE.match(line.strip())
            if not m:
                continue
            d = m.groupdict()
            vals = {k: float(d[k]) for k in ("arr", "valid", "em", "pc", "invE", "remE", "patch")}
            t = TW.match(d["tag"])
            if t:
                fam = "0.1" if t["f01"] else "0.05"
                reg = "seen" if t["reg"] == "seen" else "unseen"
                cfg = {"base": "base", "modelD": "IW", "adj+modelD": "adj+IW"}[t["cfg"]]
                out["guided"][f"f{fam}_{reg}_blk{t['blk']}_{cfg}_{t['stage']}"] = vals
                continue
            c = DC.match(d["tag"])
            if c:
                variant = "fo" if c["fo"] else "exact"
                reg = "unseen" if c["uns"] else "seen"
                out["dcbg"][f"{variant}_{reg}_blk{c['blk']}_{c['stage']}"] = vals

    print(f"guided rows: {len(out['guided'])}   dcbg rows: {len(out['dcbg'])}")
    miss = []
    for fam in ("0.05", "0.1"):
        for reg in ("seen", "unseen"):
            for b in (1, 2, 4, 8, 16, 32, 64):
                for cfg in ("base", "IW", "adj+IW"):
                    for st in ("raw", "P1", "P3", "P1P3"):
                        k = f"f{fam}_{reg}_blk{b}_{cfg}_{st}"
                        if k not in out["guided"]:
                            miss.append(k)
    for v in ("exact", "fo"):
        for reg in ("seen", "unseen"):
            for b in (1, 2, 4, 8, 16, 32, 64):
                for st in ("raw", "P1P3"):
                    k = f"{v}_{reg}_blk{b}_{st}"
                    if k not in out["dcbg"]:
                        miss.append(k)
    print(f"missing: {len(miss)}" + (f" -> {miss[:6]}" if miss else ""))

    with open(join(args.res_path, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {join(args.res_path, args.out)}")

    # quick view: P1P3 EM/PC at fam 0.05 seen
    print("\nfam0.05 seen, P1P3 (EM / PC / patch):")
    for b in (1, 2, 4, 8, 16, 32, 64):
        row = []
        for cfg in ("base", "IW", "adj+IW"):
            v = out["guided"].get(f"f0.05_seen_blk{b}_{cfg}_P1P3")
            row.append(f"{cfg} {v['em']:.3f}/{v['pc']:.3f}/{v['patch']:.2f}" if v else f"{cfg} --")
        print(f"  blk{b:<3} " + " | ".join(row))

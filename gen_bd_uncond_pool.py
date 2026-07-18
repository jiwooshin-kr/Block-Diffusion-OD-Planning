"""
Per-block-size unconditional sample pools for the model-distribution
discriminator (negatives = p_theta itself, the exact denominator of Eq. 2).

For each block size B, draws N open-ended unconditional generations
plan(None, None) from the blk-B checkpoint (valid because condition dropout
trains ori+dst jointly nulled) and stores the RAW vertex paths;
train_bd_disc.py -neg model turns them into canvas-form partials via
make_partial (dst slot = the path's own final vertex, mirroring positives).
"""

import argparse

import torch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-blocks", type=str, default="4,8,16,32,64")
    ap.add_argument("-n", type=int, default=20000)
    ap.add_argument("-batch", type=int, default=400)
    ap.add_argument("-seed", type=int, default=1)
    ap.add_argument("-ckpt", type=str, default="", help="explicit checkpoint (overrides -blocks naming)")
    ap.add_argument("-out", type=str, default="", help="explicit output path (with -ckpt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import os
    os.makedirs("./sets_disc", exist_ok=True)

    jobs = ([(args.ckpt, args.out)] if args.ckpt else
            [(f"./sets_model/BD_porto_v3_normal_mask_blk{b}_base_bd.pth",
              f"./sets_disc/uncond_pool_blk{b}.pth") for b in
             [int(x) for x in args.blocks.split(",")]])
    for ckpt, outp in jobs:
        model = torch.load(ckpt, map_location=device)
        model.eval()
        paths = []
        while len(paths) < args.n:
            out = model.plan(None, None, n_samples=args.batch)
            paths.extend([p for p in out if len(p) >= 3])
        paths = paths[:args.n]
        torch.save({"paths": paths}, outp)
        lens = [len(p) for p in paths]
        print(f"[{outp}] {len(paths)} uncond paths saved "
              f"(len mean={sum(lens)/len(lens):.1f})", flush=True)

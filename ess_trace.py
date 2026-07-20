"""ESS traces: mask (per reveal index) and graph (per diffusion step t)."""
import pickle, numpy as np, torch
from os.path import join
device = torch.device("cuda:0")
porto, fam = "./porto_data", "0.05"
A_exc = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_except_0.ts"), "rb")).bool().float()
A_nrm = pickle.load(open(join(porto, f"porto_shrink_A_v4-{fam}_normal.ts"), "rb")).bool().float()
sp = pickle.load(open(join(porto, f"porto_shrink_SP_v4-{fam}_except_0.pkl"), "rb"))
dr = (A_exc.sum(1) / A_nrm.sum(1).clamp(min=1)).to(device)
A_dev = A_exc.to(device)
perm = np.random.RandomState(777).permutation(len(sp))
real = [list(map(int, sp[i])) for i in perm[:1000] if len(sp[i]) >= 2][:200]
out = {}
for kernel, blk in [("mask", 4), ("mask", 16), ("graph", 64)]:
    ck = (f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v2_bd.pth" if kernel == "mask"
          else f"./sets_model/BD_porto_v3_normal_graph_blk{blk}_v2_bd.pth")
    dk = (f"./sets_disc/BDdisc_f0.05_p1_e0_model_blk{blk}.pth" if kernel == "mask"
          else f"./sets_disc/BDdisc_f0.05_p1_e0_model_graph{blk}.pth")
    m = torch.load(ck, map_location=device); m.eval()
    d = torch.load(dk, map_location=device); d.eval()
    torch.manual_seed(7); np.random.seed(7)
    ess = []
    m.plan_guided([p[0] for p in real], [p[-1] for p in real], d, A_dev, dr,
                  n_is=100, ess_log=ess)
    out[f"{kernel}{blk}"] = ess
    print(f"{kernel}{blk}: {len(ess)} steps, first={ess[0]:.1f} last={ess[-1]:.1f} "
          f"min={min(ess):.1f}", flush=True)
torch.save(out, "./sets_res/ess_traces.pth")
print("ESS_TRACE_DONE", flush=True)

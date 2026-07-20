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
for blk in [4, 8, 16, 32, 64]:
    m = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v2_bd.pth", map_location=device); m.eval()
    d = torch.load(f"./sets_disc/BDdisc_f0.05_p1_e0_model_blk{blk}.pth", map_location=device); d.eval()
    torch.manual_seed(7); np.random.seed(7)
    ess = []
    m.plan_guided([p[0] for p in real], [p[-1] for p in real], d, A_dev, dr, n_is=100, ess_log=ess)
    out[f"mask{blk}"] = ess
    print(f"mask{blk}: {len(ess)} reveals, first={ess[0]:.1f} min={min(ess):.1f} last={ess[-1]:.1f}", flush=True)
for blk in [4, 64]:
    m = torch.load(f"./sets_model/BD_porto_v3_normal_graph_blk{blk}_v2_bd.pth", map_location=device); m.eval()
    d = torch.load(f"./sets_disc/BDdisc_f0.05_p1_e0_model_graph{blk}.pth", map_location=device); d.eval()
    torch.manual_seed(7); np.random.seed(7)
    ess = []
    m.plan_guided([p[0] for p in real], [p[-1] for p in real], d, A_dev, dr, n_is=100, ess_log=ess)
    out[f"graph{blk}"] = ess
    print(f"graph{blk}: {len(ess)} steps, first={ess[0]:.1f} min={min(ess):.1f} last={ess[-1]:.1f}", flush=True)
torch.save(out, "./sets_res/ess_traces_all.pth")
print("DONE", flush=True)

"""v2 failure anatomy on held-out data: len-corr split, miss endpoint hops,
invalid-edge stats (normal graph)."""
import pickle
import numpy as np, torch, networkx as nx

SPLIT_SEED, EVAL_SEED = 1, 777
device = torch.device("cuda:0")
sp = pickle.load(open("./porto_data/porto_shrink_SP_v3-0.05_normal.pkl", "rb"))
A = pickle.load(open("./porto_data/porto_shrink_A_v3-0.05_normal.ts", "rb")).bool()
G = pickle.load(open("./porto_data/porto_shrink_G_v3-0.05_normal.pkl", "rb"))
n = len(sp); train_num = int(0.8 * n)
perm = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
test_idx = perm[train_num:]
np.random.RandomState(EVAL_SEED).shuffle(test_idx)
real = []
for i in test_idx:
    p = list(map(int, sp[i]))
    if len(p) >= 2: real.append(p)
    if len(real) == 400: break

for blk in [4, 16, 64]:
    model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v2_bd.pth", map_location=device)
    model.eval()
    torch.manual_seed(7); np.random.seed(7)
    planned, hits = [], []
    for s in range(0, len(real), 200):
        b = real[s:s+200]
        planned += model.plan([p[0] for p in b], [p[-1] for p in b], use_refine=False)
        hits += model.last_hits
    gl = np.array([len(p) for p in planned]); rl = np.array([len(g) for g in real])
    h = np.array(hits)
    def corr(a, b): return float(np.corrcoef(a, b)[0,1]) if len(a) > 2 else float("nan")
    dists, bad_e, tot_e, bad_hops, pos_first, pos_rest = [], 0, 0, [], 0, 0
    for p, g in zip(planned, real):
        if len(p) and p[-1] != g[-1]:
            try: dists.append(nx.shortest_path_length(G, p[-1], g[-1]))
            except Exception: pass
        for i, (u, v) in enumerate(zip(p[:-1], p[1:])):
            tot_e += 1
            if not A[u, v]:
                bad_e += 1
                cpos = (i + 2) % blk
                if cpos == 0: pos_first += 1
                else: pos_rest += 1
                try: bad_hops.append(nx.shortest_path_length(G, u, v))
                except Exception: pass
    d = np.array(dists); bh = np.array(bad_hops)
    print(f"== blk{blk}: hit={h.mean():.3f} | lcorr all={corr(gl,rl):.3f} hit={corr(gl[h],rl[h]):.3f} "
          f"miss={corr(gl[~h],rl[~h]):.3f}(n={int((~h).sum())}) | "
          f"miss->dst hops med={np.median(d) if len(d) else -1:.0f} <=2 {np.mean(d<=2)*100 if len(d) else 0:.0f}% | "
          f"invE={100*bad_e/max(tot_e,1):.2f}% hops med={np.median(bh) if len(bh) else -1:.0f} "
          f"boundary={pos_first} interior={pos_rest}", flush=True)
print("V3ANAT_DONE", flush=True)

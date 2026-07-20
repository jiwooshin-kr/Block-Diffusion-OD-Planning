"""v2 post-processing sweep (P1/P3/P1+P3) on the NORMAL graph, truly
held-out shared eval set (same reconstruction as eval_bd_sweep_controlled)."""
import pickle, time
import numpy as np, torch, networkx as nx
from eval_shortest import evaluate_em_pc

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
    if len(real) == 1000: break

def splice(p):
    out, add = [p[0]], 0
    for u, v in zip(p[:-1], p[1:]):
        if A[u, v]: out.append(v); continue
        try:
            seg = nx.shortest_path(G, u, v)[1:]; add += len(seg) - 1; out.extend(seg)
        except Exception: out.append(v)
    return out, add

def endpoint(p, d):
    if len(p) == 0 or p[-1] == d: return list(p), 0
    try:
        seg = nx.shortest_path(G, p[-1], d)[1:]; return list(p) + seg, len(seg)
    except Exception: return list(p), 0

def score(tag, paths, patch, hits):
    summ, recs, _ = evaluate_em_pc(gen_paths=paths, A=A.float(), shortest_paths=sp,
                                   save_dir="./sets_res/em_pc", prefix=tag)
    arr = float(np.mean([len(q) > 0 and q[-1] == g[-1] for q, g in zip(paths, real)]))
    vf = [bool(r["valid"]) for r in recs]
    print(f"{tag:<22} arr={arr:.3f} valid={summ['valid_rate']:.3f} "
          f"vh={np.mean([v and h for v, h in zip(vf, hits)]):.3f} em={summ['em_score']:.3f} "
          f"pc={summ['pc_score']:.3f} patch={np.mean(patch):.2f}", flush=True)

for blk in [1, 2, 4, 8, 16, 32, 64]:
    model = torch.load(f"./sets_model/BD_porto_v3_normal_mask_blk{blk}_v2_bd.pth", map_location=device)
    model.eval()
    torch.manual_seed(7); np.random.seed(7)
    planned, hits = [], []
    for s in range(0, len(real), 200):
        b = real[s:s+200]
        planned += model.plan([p[0] for p in b], [p[-1] for p in b], use_refine=False)
        hits += model.last_hits
    p1 = [splice(p) for p in planned]
    p3 = [endpoint(p, g[-1]) for p, g in zip(planned, real)]
    p13 = []
    for (q, a1), g in zip(p1, real):
        q2, a2 = endpoint(q, g[-1]); p13.append((q2, a1 + a2))
    score(f"v3pp_blk{blk}_raw", planned, [0]*len(real), hits)
    score(f"v3pp_blk{blk}_P1", [q for q,_ in p1], [a for _,a in p1], hits)
    score(f"v3pp_blk{blk}_P3", [q for q,_ in p3], [a for _,a in p3], hits)
    score(f"v3pp_blk{blk}_P1P3", [q for q,_ in p13], [a for _,a in p13], hits)
print("V3PP_DONE", flush=True)

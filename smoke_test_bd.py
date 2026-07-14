"""
Smoke test for the ported block-diffusion implementation (bd_models.py).

Checks (tiny model, max_T=5, canvas cap 28):
  1. canvas construction matches the project length spec
     (END fills the dst block's tail, later blocks are PAD and out of loss)
  2. mask kernel: training forward/backward + conditional/unconditional plan
  3. graph kernel (<end>-augmented CTMC): forward/backward + plan
"""
import torch
from utils.argparser import get_argparser
from loader.dataset import TrajFastShortestDataset
from models_seq.seq_models import Destroyer
from models_seq.bd_models import BDTransformer, BlockDiffusion


def build(kernel, dataset, device, eos_deg=0.05):
    args = get_argparser().parse_args([
        "-d_name", "porto", "-model_name", "smoke_bd", "-method", "bd_train",
        "-kernel", kernel, "-block_size", "4", "-od_max_len", "24",
        "-max_T", "5", "-drop_cond", "0.2", "-bd_eos_deg", str(eos_deg),
        "-bd_hidden_dim", "64", "-bd_n_layers", "2", "-bd_n_heads", "4", "-bd_cond_dim", "32",
    ])
    destroyer = None
    if kernel == "graph":
        V = dataset.A.shape[0]
        w = args.bd_eos_deg / V
        A = torch.zeros(V + 1, V + 1)
        A[:V, :V] = dataset.A.cpu().float()
        A[V, :V] = w
        A[:V, V] = w
        A[V, V] = w
        destroyer = Destroyer(A, torch.linspace(1e-4, 10, 5), 5, device)
    backbone = BDTransformer(dataset.n_vertex, device, hidden_dim=64, n_layers=2, n_heads=4,
                             cond_dim=32, max_canvas=28, x_emb_dim=50,
                             pretrain_path="./sets_data/porto_node2vec.pkl")
    model = BlockDiffusion(backbone, destroyer, device, args)
    model.set_graph(dataset.G)
    return model


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajFastShortestDataset("porto", ["dj"], "./sets_data", device, is_pretrain=True,
                                      shuffle=False, index="v3-0.05_normal",
                                      shortest_data_path="./porto_data")
    xs = [torch.Tensor(dataset[k]) for k in range(6) if len(dataset[k]) <= 20]
    while len(xs) < 6:
        xs.append(xs[-1])

    # ---- 1. canvas spec check (mask kernel instance, block=4) ----------
    m = build("mask", dataset, device)
    p = torch.arange(10, 15).float()          # synthetic path, L=5
    x0, lm, lengths, ori, dst = m.build_canvas([p.to(device)])
    # layout: [dst, 10,11,12,13,14, END, END] (dst at pos 5, block end = 8)
    assert x0.shape[1] == 8, x0.shape
    assert int(x0[0, 0]) == 14 and int(x0[0, 1]) == 10
    assert x0[0, 6].item() == m.END and x0[0, 7].item() == m.END
    assert lm[0, :2].tolist() == [False, False] and lm[0, 2:8].all()
    p2 = torch.arange(20, 27).float()         # L=7 -> dst at pos 7 (block boundary)
    x0b, lmb, *_ = m.build_canvas([p2.to(device)])
    assert x0b.shape[1] == 8 and int((x0b[0] == m.END).sum()) == 0, "boundary case: no END"
    x0c, lmc, *_ = m.build_canvas([p.to(device), p2.to(device)])  # mixed batch
    assert x0c.shape[1] == 8 and bool((x0c[0, 6:] == m.END).all())
    print("[1] canvas spec ok (END tail in dst block, PAD excluded, boundary case)")

    # ---- 2. mask kernel -------------------------------------------------
    m.train()
    out = m(xs)
    out["loss"].backward()
    print(f"[2] mask train ok: loss={out['loss'].item():.4f}, masked_ce={out['masked_ce'].item():.4f}")
    m.eval()
    real = [list(map(int, dataset[k])) for k in range(8, 12)]
    plans = m.plan([r[0] for r in real], [r[-1] for r in real])
    print(f"[3] mask plan ok: lens={[len(q) for q in plans]}, hits={m.last_hits}")
    up = m.plan(None, None, n_samples=4)
    print(f"[4] mask uncond ok: lens={[len(q) for q in up]}")

    # ---- 3. graph kernel ------------------------------------------------
    g = build("graph", dataset, device)
    assert g.V_states == dataset.n_vertex + 1
    g.train()
    out = g(xs)
    (out["kl"] + out["ce"] + out["con"]).backward()
    print(f"[5] graph train ok: kl={out['kl'].item():.4f}, ce={out['ce'].item():.4f}, con={out['con'].item():.4f}")
    g.eval()
    plans = g.plan([r[0] for r in real], [r[-1] for r in real])
    print(f"[6] graph plan ok: lens={[len(q) for q in plans]}, hits={g.last_hits}")
    up = g.plan(None, None, n_samples=4)
    print(f"[7] graph uncond ok: lens={[len(q) for q in up]}")

    print("ALL BD SMOKE TESTS PASSED")

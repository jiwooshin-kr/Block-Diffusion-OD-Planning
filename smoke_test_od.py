"""
Quick smoke test for O/D-conditional diffusion (EPSM_OD).

Checks, with a small max_T for speed:
  1. training forward/backward pass with od condition + dropout
  2. eval-mode loss (no od dropout)
  3. sample_with_len with od condition (+ O prefix)
  4. unconditional fallback (od=None -> null token)

Run on server:
  python smoke_test_od.py
"""
import torch
from utils.argparser import get_argparser
from loader.dataset import TrajFastShortestDataset
from models_seq.seq_models import Destroyer, Restorer
from models_seq.eps_models import EPSM_OD
from os.path import join

if __name__ == "__main__":
    torch.manual_seed(1)
    args = get_argparser().parse_args([
        "-d_name", "porto",
        "-model_name", "smoke_od",
        "-method", "seq",
        "-path", "./sets_data",
        "-shortest_data_path", "./porto_data",
        "-shortest_org_idx", "v3-0.05_normal",
        "-beta_lb", "0.0001",
        "-beta_ub", "10",
        "-max_T", "5",
        "-dims", "[100, 120, 200]",
        "-hidden_dim", "32",
        "-od_cond",
        "-od_dropout", "0.5",
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = TrajFastShortestDataset(args.d_name, ["dj"], args.path, device, is_pretrain=True,
                                      index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path)
    print(f"n_vertex: {dataset.n_vertex}")

    betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
    destroyer = Destroyer(dataset.A, betas, args.max_T, device)
    pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
    eps_model = EPSM_OD(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=eval(args.dims), device=device,
                        hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
    model = Restorer(eps_model, destroyer, device, args)

    # ------------------------------------------------------------
    # 1. training loss (od condition + dropout active)
    # ------------------------------------------------------------
    xs = [torch.Tensor(dataset[k]).to(device) for k in range(8)]
    model.train()
    kl_loss, ce_loss, con_loss = model(xs)
    loss = kl_loss + ce_loss + con_loss
    loss.backward()
    grad_norm = eps_model.od_mlp[0].weight.grad.norm().item()
    print(f"[1] train loss ok: kl={kl_loss.item():.4f}, ce={ce_loss.item():.4f}, con={con_loss.item():.4f}, od_mlp grad norm={grad_norm:.6f}")
    assert grad_norm > 0, "od_mlp did not receive gradients"

    # ------------------------------------------------------------
    # 2. eval-mode loss (no dropout)
    # ------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        kl_loss, ce_loss, con_loss = model(xs)
    print(f"[2] eval loss ok: kl={kl_loss.item():.4f}, ce={ce_loss.item():.4f}, con={con_loss.item():.4f}")

    # ------------------------------------------------------------
    # 3. conditional sampling: real (O, D, length), O prefixed
    # ------------------------------------------------------------
    real_paths = sorted([dataset[k] for k in range(8, 16)], key=len)
    lengths = torch.tensor([len(p) for p in real_paths]).long().to(device)
    od = torch.tensor([[p[0], p[-1]] for p in real_paths], dtype=torch.long, device=device)
    import numpy as np
    prefix = np.array([p[0] for p in real_paths])
    gen = model.sample_with_len(lengths, od=od, prefix=prefix)
    o_match = sum(1 for g, p in zip(gen, real_paths) if g[0] == p[0])
    d_reach = sum(1 for g, p in zip(gen, real_paths) if g[-1] == p[-1])
    print(f"[3] conditional sampling ok: {len(gen)} paths, O match {o_match}/{len(gen)}, D reach {d_reach}/{len(gen)} (untrained model, low rates expected)")
    assert all(len(g) == l for g, l in zip(gen, lengths.tolist()))

    # ------------------------------------------------------------
    # 4. unconditional sampling (od=None -> null token)
    # ------------------------------------------------------------
    gen_u = model.sample_with_len(lengths)
    print(f"[4] unconditional sampling ok: {len(gen_u)} paths")

    print("ALL SMOKE TESTS PASSED")

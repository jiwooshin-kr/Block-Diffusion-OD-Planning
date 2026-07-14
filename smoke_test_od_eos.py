"""
Smoke test for the <eos> full-canvas O/D-conditional model (EPSM_OD_EOS).

Checks (small max_T for speed):
  1. state space / head dims: destroyer V+1, logits V+1, null token = V+1
  2. train forward/backward with eos-padded canvas + od condition
  3. sample(): fixed canvas, truncation at first <end>, no <end> inside paths
  4. unconditional sampling without gmm
"""
import torch
import numpy as np
from utils.argparser import get_argparser
from loader.dataset import TrajFastShortestDataset
from models_seq.seq_models import Destroyer, Restorer
from models_seq.eps_models import EPSM_OD_EOS
from os.path import join

if __name__ == "__main__":
    torch.manual_seed(1)
    args = get_argparser().parse_args([
        "-d_name", "porto", "-model_name", "smoke_od_eos", "-method", "seq",
        "-path", "./sets_data", "-shortest_data_path", "./porto_data",
        "-shortest_org_idx", "v3-0.05_normal",
        "-max_T", "5", "-dims", "[100, 120, 200]", "-hidden_dim", "32",
        "-od_cond", "-od_dropout", "0.2",
        "-eos_mode", "-eos_deg", "0.05", "-eos_canvas_len", "64",
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajFastShortestDataset(args.d_name, ["dj"], args.path, device, is_pretrain=True,
                                      index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path)
    V = dataset.n_vertex
    w = args.eos_deg / V
    A_aug = torch.zeros(V + 1, V + 1)
    A_aug[:V, :V] = dataset.A.cpu().float()
    A_aug[:V, V] = w
    A_aug[V, :V] = w
    A_aug[V, V] = w

    betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
    destroyer = Destroyer(A_aug, betas, args.max_T, device)
    eps_model = EPSM_OD_EOS(V, x_emb_dim=args.x_emb_dim, dims=eval(args.dims), device=device,
                            hidden_dim=args.hidden_dim, pretrain_path=join(args.path, f"{args.d_name}_node2vec.pkl"))
    model = Restorer(eps_model, destroyer, device, args)

    # 1. dims
    assert model.n_vertex == V + 1, model.n_vertex
    assert model.eos_idx == V and model.null_idx == V + 1
    xs = [torch.Tensor(dataset[k]).to(device) for k in range(8)]
    logits = model.restore(torch.randint(0, V + 1, (2, 64)).to(device),
                           torch.tensor([64, 64]).to(device), torch.tensor([1, 1]).to(device))
    assert logits.shape[-1] == V + 1, logits.shape
    print(f"[1] dims ok: states={model.n_vertex}, eos_idx={model.eos_idx}, null_idx={model.null_idx}, head={logits.shape[-1]}")

    # 2. training step
    model.train()
    kl, ce, con = model(xs)
    (kl + ce + con).backward()
    g = eps_model.od_mlp[0].weight.grad.norm().item()
    g_head = eps_model.final_conv[2].weight.grad.norm().item()
    print(f"[2] train ok: kl={kl.item():.4f}, ce={ce.item():.4f}, con={con.item():.4f}, od_mlp grad={g:.6f}, head grad={g_head:.6f}")
    assert g > 0 and g_head > 0

    # 3. conditional sampling with truncation
    model.eval()
    real_paths = [dataset[k] for k in range(8, 16)]
    gen = model.sample(8, batch_traj_num=8, real_paths=real_paths, bool_prefix=True, bool_od=True)
    lens = [len(g_) for g_ in gen]
    assert all(model.eos_idx not in g_ for g_ in gen), "eos token leaked into a truncated path"
    assert all(l <= 64 for l in lens)
    print(f"[3] cond sampling ok: lens={lens} (canvas 64, truncated at first <end>)")

    # 4. unconditional (no gmm needed)
    if hasattr(model, "gmm"):
        del model.gmm
    gen_u = model.sample(8, batch_traj_num=8)
    print(f"[4] uncond sampling ok (no gmm): lens={[len(g_) for g_ in gen_u]}")

    print("ALL EOS SMOKE TESTS PASSED")

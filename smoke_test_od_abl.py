"""
Smoke test for the conditional-generation ablations:
  A1 clean_prefix, A2 eos_loss_weight, A3 dst_token, and CFG sampling.
"""
import torch
import numpy as np
from utils.argparser import get_argparser
from loader.dataset import TrajFastShortestDataset
from models_seq.seq_models import Destroyer, Restorer
from models_seq.eps_models import EPSM_OD_EOS
from os.path import join

def build(dataset, device, extra_flags):
    args = get_argparser().parse_args([
        "-d_name", "porto", "-model_name", "smoke_abl", "-method", "seq",
        "-path", "./sets_data", "-shortest_data_path", "./porto_data",
        "-shortest_org_idx", "v3-0.05_normal",
        "-max_T", "5", "-dims", "[100, 120, 200]", "-hidden_dim", "32",
        "-od_cond", "-od_dropout", "0.3",
        "-eos_mode", "-eos_deg", "0.05", "-eos_canvas_len", "64",
    ] + extra_flags)
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
                            hidden_dim=args.hidden_dim, pretrain_path=join("./sets_data", "porto_node2vec.pkl"))
    return Restorer(eps_model, destroyer, device, args), args

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajFastShortestDataset("porto", ["dj"], "./sets_data", device, is_pretrain=True,
                                      index="v3-0.05_normal", shortest_data_path="./porto_data")
    xs = [torch.Tensor(dataset[k]).to(device) for k in range(8)]
    real_paths = [dataset[k] for k in range(8, 16)]

    # ---- config 1: A1 + A2 (clean prefix, eos loss weight, no dst token) ----
    m1, args1 = build(dataset, device, ["-clean_prefix", "-eos_loss_weight", "0.1"])
    m1.train()
    kl, ce, con = m1(xs)
    (kl + ce + con).backward()
    print(f"[1] A1+A2 train ok: kl={kl.item():.4f}, ce={ce.item():.4f}, con={con.item():.4f}")
    m1.eval()
    gen = m1.sample(8, 8, real_paths=real_paths, bool_prefix=True, bool_od=True)
    rp = sorted(real_paths, key=len)
    o_match = sum(1 for g, p in zip(gen, rp) if len(g) > 0 and g[0] == p[0])
    print(f"[2] A1 sampling ok: O match {o_match}/8, lens={[len(g) for g in gen]}")
    assert o_match == 8

    # CFG
    args1.guidance_scale = 2.0
    gen_w2 = m1.sample(8, 8, real_paths=real_paths, bool_prefix=True, bool_od=True)
    print(f"[3] CFG w=2 sampling ok: lens={[len(g) for g in gen_w2]}")
    args1.guidance_scale = 1.0

    # ---- config 2: A3 (+A1) dst in-context token ----
    m2, args2 = build(dataset, device, ["-clean_prefix", "-dst_token"])
    m2.train()
    # canvas check: position 0 must be the destination
    raw = xs[0]
    kl, ce, con = m2(xs)
    (kl + ce + con).backward()
    print(f"[4] A3 train ok: kl={kl.item():.4f}, ce={ce.item():.4f}, con={con.item():.4f}")
    m2.eval()
    gen = m2.sample(8, 8, real_paths=real_paths, bool_prefix=True, bool_od=True)
    o_match = sum(1 for g, p in zip(gen, rp) if len(g) > 0 and g[0] == p[0])
    no_dst_leak = all(len(g) == 0 or g[0] != p[-1] or p[0] == p[-1] for g, p in zip(gen, rp))
    print(f"[5] A3 cond sampling ok: O match {o_match}/8 (dst token stripped), lens={[len(g) for g in gen]}")
    assert o_match == 8

    # dst-only conditioning (no O prefix) and uncond
    gen_d = m2.sample(8, 8, real_paths=real_paths, bool_prefix=False, bool_od=True)
    gen_u = m2.sample(8, 8, real_paths=real_paths, bool_prefix=False, bool_od=False)
    print(f"[6] A3 dst-only / uncond ok: lens={[len(g) for g in gen_d]} / {[len(g) for g in gen_u]}")

    # prefix_only must raise on dst_token models
    try:
        m2.sample(8, 8, real_paths=real_paths, bool_prefix=True, bool_od=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[7] A3 prefix-only correctly rejected")

    print("ALL ABLATION SMOKE TESTS PASSED")

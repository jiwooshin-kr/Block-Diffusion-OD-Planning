"""Build a tiny (max_T=5, 30 iters) EPSM_OD checkpoint to smoke-test eval_od_conditional.py."""
import torch
from os.path import join
from utils.argparser import get_argparser
from loader.dataset import TrajFastShortestDataset
from models_seq.seq_models import Destroyer, Restorer
from models_seq.eps_models import EPSM_OD
from models_seq.trainer import Trainer

if __name__ == "__main__":
    args = get_argparser().parse_args([
        "-d_name", "porto", "-model_name", "tiny_od", "-method", "seq",
        "-path", "./sets_data", "-shortest_data_path", "./porto_data",
        "-shortest_org_idx", "v3-0.05_normal",
        "-max_T", "5", "-dims", "[100, 120, 200]", "-hidden_dim", "32",
        "-od_cond", "-od_dropout", "0.1",
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TrajFastShortestDataset(args.d_name, ["dj"], args.path, device, is_pretrain=True,
                                      index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path)
    betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
    destroyer = Destroyer(dataset.A, betas, args.max_T, device)
    eps_model = EPSM_OD(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=eval(args.dims), device=device,
                        hidden_dim=args.hidden_dim, pretrain_path=join(args.path, f"{args.d_name}_node2vec.pkl"))
    model = Restorer(eps_model, destroyer, device, args)

    trainer = Trainer(model, dataset, "./sets_model", "tiny_od", args=args)
    trainer.train_gmm(gmm_samples=2000, n_comp=3)

    opt = torch.optim.Adam(model.parameters(), 5e-4)
    model.train()
    for i in range(30):
        xs = [torch.Tensor(dataset[k]).to(device) for k in range(i * 8, (i + 1) * 8)]
        kl, ce, con = model(xs)
        loss = kl + ce + con
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    torch.save(model, "./sets_model/tiny_od.pth")
    print("saved ./sets_model/tiny_od.pth")

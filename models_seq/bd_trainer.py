"""
Trainer for block-diffusion OD planning (BlockDiffusion).

No length-model stages (GMM / LengthPredictor): block diffusion generates
open-endedly until the destination / <end> token is emitted.

The model returns a loss dict:
  mask kernel  : {"loss", "nll", "masked_ce", "masked_frac"}
  graph kernel : {"kl", "ce", "con"}  (combined here with the same
                 ce-gating as models_seq/trainer.py)
"""

import logging
from itertools import cycle
from os.path import join

import torch
from torch.utils.data import DataLoader, random_split

from models_seq.bd_models import BlockDiffusion


class TrainerBD:
    def __init__(self, model: BlockDiffusion, dataset, model_path, model_name, args):
        self.model = model
        self.device = model.device
        self.dataset = dataset
        self.model_path = model_path
        self.model_name = model_name
        self.args = args

        train_num = int(0.8 * len(dataset))
        self.train_dataset, self.test_dataset = random_split(
            dataset, [train_num, len(dataset) - train_num],
            generator=torch.Generator().manual_seed(args.seed),
        )

    def _combine(self, out):
        """Scalar training loss from the model's loss dict."""
        if "kl" not in out:
            return out["loss"]
        kl_loss, ce_loss, con_loss = out["kl"], out["ce"], out["con"]
        # same gating as models_seq/trainer.py:70-79
        if ce_loss.item() < 60 / 16:
            return kl_loss
        if (self.args.lam_ce == 0.0) and (self.args.lam_con == 0.0):
            return kl_loss
        if self.args.lam_ce == 0.0:
            return kl_loss + self.args.lam_con * con_loss
        if self.args.lam_con == 0.0:
            return kl_loss + self.args.lam_ce * ce_loss
        return kl_loss + self.args.lam_ce * ce_loss + self.args.lam_con * con_loss

    def train(self, n_epoch, batch_size, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        trainloader = DataLoader(self.train_dataset, batch_size, shuffle=True,
                                 collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader = DataLoader(self.test_dataset, batch_size,
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])

        self.model.train()
        iter, loss_avg = 0, 0.0
        part_avg = {}

        logging.basicConfig(
            filename=join(self.model_path, f"{self.model_name}_bd_train_log.txt"),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    out = self.model(xs)
                    loss = self._combine(out)

                    loss_avg += loss.item()
                    for k, v in out.items():
                        part_avg[k] = part_avg.get(k, 0.0) + float(v)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    iter += 1

                    if iter % 100 == 0 or iter == 1:
                        denom = 1 if iter == 1 else 100
                        test_out = next(self.eval_test(testloader))
                        parts = ", ".join(f"{k}: {v / denom: .4f}" for k, v in part_avg.items())
                        test_parts = ", ".join(f"{k}: {v: .4f}" for k, v in test_out.items())
                        msg = (f"e: {epoch}, i: {iter}, train loss: {loss_avg / denom: .4f} "
                               f"({parts}), test ({test_parts})")
                        print(msg)
                        logging.info(msg)
                        loss_avg, part_avg = 0.0, {}

                if self.args.save_step > 0 and (epoch + 1) % self.args.save_step == 0:
                    torch.save(self.model, join(self.model_path, f"{self.model_name}_bd_epoch_{epoch}.pth"))
        except KeyboardInterrupt:
            print("Training interrupted, saving...")
            self.model.eval()
            torch.save(self.model, join(self.model_path, f"{self.model_name}_bd_tmp_iter_{iter}.pth"))

        self.model.eval()

    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                out = self.model(txs)
                yield {k: float(v) for k, v in out.items()}

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from models_seq.eps_models import EPSM
from models_seq.disc_models import Discriminator
from torch.distributions.utils import probs_to_logits, clamp_probs
from collections import defaultdict
import numpy as np
import math

from tqdm import tqdm
import logging

class Destroyer:
    """
    Continuous-time graph diffusion process.
    Given an adjacency matrix A, this class constructs the CTMC generator

        G = A - D
    
    and precomputes transition matrices

        Q_t = exp(G * beta_t)

    for all timesteps.

    Notation:
        Q[t]:
            One-step transition q(x_t | x_{t-1})

        matrices[t]:
            Cumulative transition q(x_t | x_0)
    """
    def __init__(self, A, betas, max_T, device) -> None:
        self.device = device
        self.n_vertex = A.shape[0]
        self.betas = betas.to(device=self.device)
        self.max_T = max_T

        # Build CTMC generator G = A - D
        self.A = A.clone().detach().to(self.device, dtype=torch.float32)
        G = (self.A - torch.diag(self.A.sum(dim=0)).to(self.device))

        # Precompute cumulative kernels: matrices[t] = q(x_t | x_0)
        self.matrices = torch.zeros(self.max_T + 1, self.n_vertex, self.n_vertex).to(self.device)

        # Precompute one-step kernels: Q[t] = q(x_t | x_{t-1})
        self.Q = torch.zeros_like(self.matrices).to(self.device)

        self.matrices[0] = torch.eye(self.n_vertex, device=self.device)
        self.Q[0] = torch.eye(self.n_vertex, device=self.device)

        for i in range(1, self.max_T + 1):
            self.Q[i] = torch.linalg.matrix_exp(G * self.betas[i - 1])
            self.matrices[i] = self.Q[i] @ self.matrices[i - 1]
            
    def get_Q(self):
        Q = self.Q 
        del self.Q 
        return Q
        
    def diffusion(self, xs, ts, ret_distr=False):
        lengths = [x.shape[0] for x in xs]
        batch_size, horizon = len(lengths), max(lengths)

        if type(xs) is torch.Tensor and xs.dim() == 3:
            # Input: distribution -> Output: distribution
            xs = rearrange(xs, "b h c -> c (b h)").to(self.device)
            x_distr = self.matrices[ts[0]] @ xs
            x_distr = rearrange(x_distr, "c (b h) -> b h c", h=horizon)
            return x_distr

        else:
            # Input: sample -> Output: sample
            xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.).to(self.device).long()
            ts_padded = ts.view(-1, 1).repeat(1, horizon).view(-1,)

            x_distr_padded = self.matrices[ts_padded, :, xs_padded.view(-1)]

            if ret_distr:
                return x_distr_padded

            # Sample x_t ~ q(x_t | x_0)
            x_diffused_padded = torch.multinomial(x_distr_padded, 1).view(batch_size, -1)
            x_diffused = [x_diffused_padded[k][:length] for k, length in enumerate(lengths)]
            return x_diffused
  
  
class Restorer(nn.Module):
    """
    Discrete diffusion reverse process.

    Given the forward diffusion process q(x_t | x_0),
    this class parameterizes the reverse process

        p_theta(x_{t-1} | x_t)

    through an x_0-prediction network.

    Main functionalities:
        - Reverse process training
        - Reverse diffusion sampling
        - Variational NLL evaluation
        - Discriminator-guided inference
    """
    def __init__(self, eps_model: EPSM, destroyer: Destroyer, device, args):
        super().__init__()
        self.n_vertex = destroyer.n_vertex
        self.eps_model = eps_model
        self.model_device = self.eps_model.device
        self.device = device
        self.destroyer = destroyer
        self.des_device = destroyer.device
        self.max_T = self.destroyer.max_T
        self.matrices = self.destroyer.matrices
        self.A = destroyer.A
        self.Q = self.destroyer.get_Q()
        self.Q = self.Q.to(self.device)
        self.max_deg = self.A.sum(1).max()

        # [Edit-Jiwoo] ----------------------------------------
        self.matrices_org = None
        self.Q_org = None
        self.new_nll = False
        # [Edit-Jiwoo] ----------------------------------------

        # [Edit-Jiwoo] O/D conditional generation ---------------
        self.od_cond = getattr(args, "od_cond", False)
        self.od_dropout = getattr(args, "od_dropout", 0.1)
        # In eos_mode the destroyer state space is V+1 (last state = <end>),
        # so self.n_vertex is already V+1 and the null token lands on <pad>.
        self.null_idx = self.n_vertex
        # [Edit-Jiwoo] ----------------------------------------

        # [Edit-Jiwoo] <eos> full-canvas mode -------------------
        # Paths are padded to a fixed canvas with the <end> state, which the
        # model predicts like any other state (no external length model).
        self.eos_mode = getattr(args, "eos_mode", False)
        self.eos_canvas_len = getattr(args, "eos_canvas_len", 64)
        self.eos_idx = self.n_vertex - 1 if self.eos_mode else None
        # [Edit-Jiwoo] ----------------------------------------

        # [Edit-Jiwoo] conditional-generation ablations ---------
        # A1: condition tokens stay clean (never noised) when their condition
        #     is active; A2: downweight <end>-target positions in the loss;
        # A3: destination as in-context token at canvas position 0
        self.clean_prefix = getattr(args, "clean_prefix", False)
        self.eos_loss_weight = getattr(args, "eos_loss_weight", 1.0)
        self.dst_token = getattr(args, "dst_token", False)
        # A4: destination-matching losses
        self.dst_loss_weight = getattr(args, "dst_loss_weight", 1.0)
        self.lam_arr = getattr(args, "lam_arr", 0.0)
        # [Edit-Jiwoo] ----------------------------------------

        self.args = args

        self.applying_mask_intermediate = False
        self.applying_mask_intermediate_temperature = False
    

    def forward(self, xs):
        # xs: list of tensors of labels
        batch_size = len(xs)
        if batch_size == 0:
            import pdb
            pdb.set_trace()

        # [eos mode] O/D come from the raw path endpoints, then every path is
        # padded to the fixed canvas with the <end> state so the tail is a
        # normal training target.
        raw_xs = xs
        if self.eos_mode:
            L = self.eos_canvas_len
            if self.dst_token:
                # A3: canvas = [dst, ori, v1, ..., <end>, ...]
                xs = [torch.cat([x[-1:], x]) for x in xs]
            xs = [torch.cat([x, torch.full((L - x.shape[0],), float(self.eos_idx), device=x.device, dtype=x.dtype)])
                  if x.shape[0] < L else x[:L] for x in xs]

        lengths = torch.Tensor([x.shape[0] for x in xs]).long().to(self.device)

        # Sample timestep t
        if self.args.train_timestep_sampling == 'uniform':
            # uniformly choose t
            ts = torch.randint(1, self.max_T + 1, [batch_size]).to(self.device)
        elif self.args.train_timestep_sampling == 'early':
            gamma = 0.7
            t = torch.arange(1, self.max_T + 1, device=self.device)
            probs = t.float().pow(-gamma)
            probs = probs / probs.sum()
            ts = torch.multinomial(probs, batch_size, replacement=True)
        elif self.args.train_timestep_sampling == 'cosine':
            t = torch.arange(1, self.max_T + 1, device=self.device)
            probs = torch.sin(torch.pi * t / self.max_T)
            probs = probs / probs.sum()
            ts = torch.multinomial(probs, batch_size, replacement=True)
        else:
            raise NotImplementedError(f"[seq_models.py] train_timestep_sampling not implemented: {self.args.train_timestep_sampling}")
        
        # O/D condition from each path with independent condition dropout.
        # Dropping O and D independently lets the model learn
        # unconditional / O-only / D-only / O+D generation in one model.
        od = None
        drop = None
        if self.od_cond:
            od = torch.stack([torch.stack([x[0], x[-1]]) for x in raw_xs]).long().to(self.model_device)
            if self.training and self.od_dropout > 0:
                drop = torch.rand(od.shape, device=od.device) < self.od_dropout
                od = od.masked_fill(drop, self.null_idx)

        # x_t ~ q(x_t | x_0)
        x_t = self.destroyer.diffusion(xs, ts, ret_distr=False)

        # [A1/A3] keep condition tokens clean (never noised) when the
        # corresponding condition is active for that sample
        if self.clean_prefix and self.od_cond:
            o_pos = 1 if self.dst_token else 0
            for k in range(batch_size):
                o_kept = drop is None or not bool(drop[k, 0])
                d_kept = drop is None or not bool(drop[k, 1])
                if self.dst_token and d_kept:
                    x_t[k][0] = xs[k][0].long()
                if o_kept:
                    x_t[k][o_pos] = xs[k][o_pos].long()

        xt_padded = pad_sequence(x_t, batch_first=True, padding_value=0).long()
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).long()
        horizon = xt_padded.shape[1]
        ts_padded = ts.view(-1, 1).repeat(1, horizon)
        
        # EtXt = Q_t @ x_t
        EtXt = self.Q[ts_padded.view(-1,).to(self.device), :, xt_padded.view(-1).to(self.device)]
        
        # true_probs_unorm = Q_t @ x_t * \bar{E}_{t-1} @ x_0
        true_probs_unorm = EtXt * self.matrices[ts_padded.view(-1,) - 1, :, xs_padded.view(-1)].to(self.device)
        true_probs = true_probs_unorm / true_probs_unorm.sum(1, keepdim=True)
        true_probs = clamp_probs(true_probs)
        true_probs = rearrange(true_probs, "(b h) c -> b h c", h=horizon)
        
        # p_theta(x_0 | x_t, t)
        x0_pred_logits = self.restore(xt_padded.to(self.model_device), lengths.to(self.model_device), ts.to(self.model_device), od=od)
        x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)

        # Et_minus_one_bar_hat_x0 = \bar{E}_{t-1} @ \hat{x}_0
        Et_minus_one_bar_hat_x0 = (self.matrices[ts - 1] @ x0_pred_probs.transpose(2, 1).to(self.des_device)).to(self.device)
        Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")

        # pred_probs_unorm = E_t @ x_t * \bar{E}_{t-1} @ \hat{x}_0  x_0 is logits while x_t is categorical
        pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
        pred_probs = pred_probs_unorm / torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)

        # logits for p_theta(x_{t-1} | x_t)       
        pred_logits = probs_to_logits(pred_probs)
        pred_logits = rearrange(pred_logits, "(b h) c -> b h c", h=horizon)
        
        eps = 0.000001
        if self.eos_mode and (self.eos_loss_weight != 1.0 or self.dst_loss_weight != 1.0):
            # [A2] downweight positions whose x_0 target is <end> so the eos
            # tail (~2/3 of the canvas) does not dominate the objective
            # [A4a] upweight each sample's true endpoint position (= destination)
            lam = self.eos_loss_weight
            s = 1 if self.dst_token else 0
            kl_loss, ce_loss = 0., 0.
            for k, l in enumerate(lengths):
                tgt = xs[k][:l].long()
                w = torch.where(tgt == self.eos_idx,
                                torch.full((int(l),), lam, device=tgt.device),
                                torch.ones(int(l), device=tgt.device))
                if self.dst_loss_weight != 1.0:
                    end_pos = s + raw_xs[k].shape[0] - 1
                    if end_pos < int(l):
                        w[end_pos] = self.dst_loss_weight
                kl_pos = F.kl_div(pred_logits[k][:l] + eps, true_probs[k][:l], reduction="none").sum(-1)
                kl_loss = kl_loss + (kl_pos * w.to(kl_pos)).sum() / w.sum()
                ce_pos = F.cross_entropy(x0_pred_logits[k][:l].to(xs[k]) + eps, tgt, reduction="none")
                ce_loss = ce_loss + (ce_pos * w.to(ce_pos)).sum() / w.sum()
            kl_loss = kl_loss / batch_size
            ce_loss = ce_loss / batch_size
        else:
            kl_loss = sum([F.kl_div(pred_logits[k][:l] + eps, true_probs[k][:l], reduction="batchmean") for k, l in enumerate(lengths)]) / batch_size
            ce_loss = sum([F.cross_entropy(x0_pred_logits[k][:lengths[k]].to(x) + eps, x[:lengths[k]].long(), reduction="mean") for k, x in enumerate(xs)]) / batch_size

        # [A4b] mean-field arrival loss: maximize the probability that the
        # destination is immediately followed by <end> somewhere on the canvas.
        # Folded into kl_loss so the trainer's loss combination is unchanged.
        if self.eos_mode and self.lam_arr > 0:
            arr_loss = 0.
            for k in range(batch_size):
                p = x0_pred_probs[k]
                dv = int(raw_xs[k][-1].item())
                arr_prob = (p[:-1, dv] * p[1:, self.eos_idx]).sum()
                arr_loss = arr_loss + (-torch.log(arr_prob + eps))
            kl_loss = kl_loss + self.lam_arr * (arr_loss / batch_size)

        # [A3] the dst token at canvas position 0 is not part of the path, so
        # the adjacency (connectivity) loss starts between positions 1 and 2
        s = 1 if self.dst_token else 0
        con_loss = -sum([((self.A @ (x0_pred_probs[k, s+1:l, :] + eps).log().T).T * x0_pred_probs[k, s:l-1, :]).mean() for k, l in enumerate(lengths)]) / batch_size
        con_loss += -sum([((self.A @ (x0_pred_probs[k, s:l-1, :] + eps).log().T).T * x0_pred_probs[k, s+1:l, :]).mean() for k, l in enumerate(lengths)]) / batch_size

        if torch.isnan(kl_loss):
            print('kl_loss nan')
            import pdb
            pdb.set_trace()
        if torch.isnan(ce_loss):
            print('ce_loss nan')
            import pdb
            pdb.set_trace()
        if torch.isnan(con_loss):
            print('con_loss nan')
            import pdb
            pdb.set_trace()
        return kl_loss, ce_loss, con_loss * 100
         

    def restore(self, xt_padded, lengths=None, ts=None, od=None):
        # Predicts logits of p_theta(x_0 | x_t, t)
        batch_size = xt_padded.shape[0]

        if ts is None:
            ts = torch.Tensor([self.max_T]).repeat(batch_size).to(self.device)

        if self.od_cond:
            # od=None falls back to the null token inside EPSM_OD (unconditional)
            if od is not None:
                od = od.to(self.model_device)
            x0_pred_logits = self.eps_model(xt_padded, lengths, ts, od=od)
        else:
            x0_pred_logits = self.eps_model(xt_padded, lengths, ts)
        return x0_pred_logits
    

    def sample(self, n_samples: int, batch_traj_num=200, real_paths=None, bool_prefix=False, ret_org=False, bool_od=False):
        if self.eos_mode:
            # Fixed canvas for every sample; length emerges from where the
            # model places the <end> state (truncated below).
            if real_paths is not None:
                real_paths = sorted(real_paths, key=len)
            lengths = torch.full((n_samples,), self.eos_canvas_len, dtype=torch.long, device=self.device)
        elif real_paths is not None:
            # Sort the paths themselves (not only lengths) so that each sample's
            # length / prefix / od condition stay aligned to the same real path
            assert hasattr(self, "gmm")
            real_paths = sorted(real_paths, key=len)
            lengths = np.array([len(x) for x in real_paths])
            lengths = torch.Tensor(lengths).long().to(self.device)
        else:
            assert hasattr(self, "gmm")
            lengths = self.gmm.sample(n_samples)[0].reshape(-1).astype(int)
            lengths = np.sort(lengths[lengths > 0])
            lengths = torch.Tensor(lengths).long().to(self.device)

        od_all = None
        if bool_od:
            assert self.od_cond, "bool_od requires a model trained with -od_cond"
            assert real_paths is not None, "bool_od requires real_paths to provide (O, D) conditions"
            od_all = torch.tensor([[p[0], p[-1]] for p in real_paths], dtype=torch.long, device=self.device)

        dst_token = getattr(self, "dst_token", False)
        if dst_token and bool_prefix and not bool_od:
            raise ValueError("dst_token models clamp canvas position 0 to the destination; prefix-only sampling is not defined")

        n_batch = n_samples // batch_traj_num
        paths = []
        for b in range(n_batch):
            left, right = b * batch_traj_num, min((b + 1) * batch_traj_num, n_samples)
            od = od_all[left: right] if od_all is not None else None
            if dst_token and bool_od:
                # canvas prefix = [dst] or [dst, ori]
                if bool_prefix:
                    prefix = np.array([[x[-1], x[0]] for x in real_paths])
                else:
                    prefix = np.array([[x[-1]] for x in real_paths])
                paths.extend(self.sample_with_len(lengths[left: right], prefix=prefix[left: right], ret_org=ret_org, od=od))
            elif bool_prefix:
                prefix = np.array([x[0] for x in real_paths])
                paths.extend(self.sample_with_len(lengths[left: right], prefix=prefix[left: right], ret_org=ret_org, od=od))
            else:
                paths.extend(self.sample_with_len(lengths[left: right], ret_org=ret_org, od=od))

        if self.eos_mode and not ret_org:
            if dst_token:
                # canvas position 0 is the dst token, not part of the path
                paths = [p[1:] for p in paths]
            paths = [self._truncate_eos(p) for p in paths]
        return paths

    def _truncate_eos(self, path):
        # Cut at the first <end> state (exclusive)
        for i, v in enumerate(path):
            if v == self.eos_idx:
                return path[:i]
        return path

    def _strided_schedule(self, n_steps):
        """
        Respaced sampling schedule. Returns [(t_hi, t_lo, Q_step), ...] from
        t = max_T down to 0 in n_steps jumps, where Q_step = exp(G * sum(beta))
        over (t_lo, t_hi] is the EXACT composed forward kernel (CTMC property
        C_{a+b} = C_a C_b), indexed like self.Q: Q_step[:, x_t] = q(x_hi | .).
        """
        A = self.destroyer.A
        G = (A - torch.diag(A.sum(dim=0))).to(A.device)
        betas = self.destroyer.betas.to(A.device)
        B = torch.cat([torch.zeros(1, device=A.device), betas.cumsum(0)])  # B[t] = sum_{s<=t} beta_s
        ts = torch.linspace(self.max_T, 0, n_steps + 1).round().long().tolist()
        # de-duplicate while keeping order (rounding can repeat values)
        ts = [ts[0]] + [t for i, t in enumerate(ts[1:]) if t < ts[i]]
        sched = []
        for hi, lo in zip(ts[:-1], ts[1:]):
            Q_step = torch.linalg.matrix_exp(G * (B[hi] - B[lo]).item())
            sched.append((hi, lo, Q_step.to(self.device)))
        return sched

    def sample_with_len(self, lengths, ret_distr=False, xt=None, T=None, ret_trace=False, prefix=None, ret_org=False, od=None):
        # ============================================================
        # Setup
        # ============================================================
        applying_mask_intermediate = self.applying_mask_intermediate
        applying_mask_intermediate_temperature = self.applying_mask_intermediate_temperature

        if ret_trace:
            reverse_trace = defaultdict(list) # t -> [path1, path2,...]

        if T is None:
            T = self.max_T

        # Respaced (strided) sampling: n_steps < max_T composed exact kernels
        n_steps = int(getattr(self.args, "sample_steps", 0) or 0)
        strided = None
        if 0 < n_steps < T:
            strided = self._strided_schedule(n_steps)

        n_samples = lengths.shape[0]
        horizon = max(lengths)
        
        # Initialize x_T if not provided
        if xt is None:
            xt = torch.randint(0, self.n_vertex, [n_samples, horizon]).to(self.device)
        else:
            xt = xt.to(self.device)

        # Convert prefix to tensor if prefix conditioning is used
        # # Org
        # if prefix is not None:
        #     prefix = torch.as_tensor(prefix, device=self.device, dtype=xt.dtype).unsqueeze(-1)

        # New
        prefix_len = 0
        
        if prefix is not None:
            prefix = torch.as_tensor(prefix, device=self.device, dtype=xt.dtype)
            if prefix.ndim == 1:
                prefix = prefix.unsqueeze(-1)
            prefix_len = prefix.shape[1]

        with torch.no_grad():
            # ============================================================
            # Reverse diffusion sampling: x_T -> x_0
            # ============================================================
            step_list = strided if strided is not None else [(t, t - 1, None) for t in range(T, 0, -1)]
            for t, t_prev, Q_step in step_list:
                ts = torch.Tensor([t]).long().to(self.device).repeat(n_samples)

                # Apply prefix conditioning at the current timestep
                if prefix is not None:
                    if getattr(self, "clean_prefix", False):
                        # [A1/A3] condition tokens are held clean, matching training
                        xt[:, :prefix_len] = prefix[:, :prefix_len]
                    else:
                        prefix_t = self.destroyer.diffusion(prefix, ts, ret_distr=False)
                        prefix_t = pad_sequence(prefix_t, batch_first=True, padding_value=0).long()
                        # xt[:, 0:1] = prefix_t                 # org
                        xt[:, :prefix_len] = prefix_t[:, :prefix_len]    # new

                # Predict p_theta(x_0 | x_t)
                x0_pred_logits = self.restore(xt, lengths, ts, od=od)

                # Classifier-free guidance: mix conditional / null-condition logits
                w_cfg = getattr(self.args, "guidance_scale", 1.0)
                if self.od_cond and od is not None and w_cfg != 1.0:
                    x0_uncond_logits = self.restore(xt, lengths, ts, od=None)
                    x0_pred_logits = x0_uncond_logits + w_cfg * (x0_pred_logits - x0_uncond_logits)

                x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)

                # Approximate reverse posterior p(x_{t_prev} | x_t); with
                # respacing, Q_step is the exact composed kernel over (t_prev, t]
                EtXt = (Q_step if Q_step is not None else self.Q[t])[:, xt.view(-1)].T
                x0_pred_probs_rearrange = rearrange(x0_pred_probs, "b h c -> (b h) c", b=n_samples)

                # Monte-Carlo estimate over predicted x_0 samples
                num_mc_samples = 10
                x0_sample = torch.multinomial(x0_pred_probs_rearrange, num_samples=num_mc_samples, replacement=True)
                x0_sample = rearrange(x0_sample, "(b h) n -> b h n", b=n_samples, n=num_mc_samples)

                Et_minus_one_bar_hat_x0 = self.matrices[t_prev, x0_sample.view(-1)]
                Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "(b h n) d -> (b h) n d", b=n_samples, n=num_mc_samples)
                Et_minus_one_bar_hat_x0 = Et_minus_one_bar_hat_x0.mean(dim=1)

                # Normalize reverse transition probabilities
                pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
                sum_probs = torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)
                pred_probs = pred_probs_unorm / sum_probs

                # Uniform distribution for degenerate cases
                mask = (sum_probs == 1e-8)[:, 0]
                pred_probs[mask] = 1.0 / pred_probs.shape[1]

                # ========================================================
                # Sample x_{t-1}
                # ========================================================
                if applying_mask_intermediate:
                    pred_prob_ = rearrange(pred_probs, "(b h) c -> b h c", b=n_samples)
                    xt = torch.zeros([n_samples, horizon]).long().to(self.device)
                    x_mask = pred_prob_[:, 0].clone()
                    if (self.A.sum(dim=1) == 0).sum() != 0:
                        x_mask[:, self.A.sum(dim=1) == 0] = 0.
                    xt[:, 0] = torch.multinomial(x_mask, 1).view(-1)

                    for k in range(1, horizon):
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k]) * ((self.max_T - t) / self.max_T) + pred_prob_[:, k] * ( t / self.max_T)
                        else: ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k])  # b * v
                        random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                        x_next_masked_prob[random] = 1.
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob * ((self.max_T - t) / self.max_T) + x_next_masked_prob * (t / self.max_T)
                        else:  ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob  # b * v
                        xt[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)

                else:
                    # Independent sampling w/o topology constraint
                    xt = torch.multinomial(pred_probs, num_samples=1, replacement=True)
                    xt = rearrange(xt, "(b h) 1 -> b h", b=n_samples) #torch.Size([n_samples, horizon])

                if ret_trace:
                    reverse_trace[t] = [xt[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]

            # ============================================================
            # Final trajectory decoding
            # ============================================================
            if ret_org:
                x = xt.long().to(self.device)

            else:
                x = torch.zeros_like(xt).long().to(self.device)

                if prefix is not None:
                    # x[:, 0:1] = prefix        # org
                    x[:, :prefix_len] = prefix  # new
                    filled = prefix_len
                else:
                    x_mask = x0_pred_probs[:, 0].clone()
                    if (self.A.sum(dim=1)==0).sum() != 0:
                        x_mask[:, self.A.sum(dim=1)==0] = 0.
                    x[:, 0] = torch.multinomial(x_mask, 1).view(-1)
                    filled = 1

                # [A3] canvas position 0 is the dst token: the adjacency chain
                # starts between positions 1 and 2; unfilled pre-chain
                # positions are sampled freely from p(x_0)
                start_k = max(filled, 2) if getattr(self, "dst_token", False) else filled
                for k in range(filled, start_k):
                    x[:, k] = torch.multinomial(x0_pred_probs[:, k], 1).view(-1)

                # Generate topologically valid trajectory
                for k in range(start_k, horizon):
                    x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * (x0_pred_probs[:, k])
                    
                    # Handle cases with no valid next node
                    random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                    x_next_masked_prob[random] = 1.
                    x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * x_next_masked_prob

                    try:
                        x[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                    except:
                        bad_mask = x_next_masked_prob.sum(-1) <= 0  # shape: (batch,)
                        good_mask = ~bad_mask
                        if good_mask.any():
                            x_next_masked_prob_good = x_next_masked_prob[good_mask]  # (good_batch, V)
                            sampled_good = torch.multinomial(x_next_masked_prob_good, 1).view(-1)
                            x[good_mask, k] = sampled_good
                        if bad_mask.any():
                            batch_size, vocab_size = x_next_masked_prob.shape
                            random_idx = torch.randint(0, vocab_size, (bad_mask.sum(),), device=x.device)
                            x[bad_mask, k] = random_idx
                            lengths[bad_mask] = k - 1

            # ============================================================
            # Return results
            # ============================================================
            x_list = [x[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]

            if ret_trace:
                reverse_trace[0] = x_list
                return reverse_trace

            if ret_distr:
                return x_list, x0_pred_probs

            return x_list    
    

    def eval_nll_fix(self, real_paths, disc=None, adj_matrix=None):
        total = len(real_paths)
        batch_traj_num = 200
        n_batch = (total + batch_traj_num - 1) // batch_traj_num

        kl_all = []
        kl_before_all = []

        file_name = self.args.disc_name
        logging.basicConfig(
            filename=f'./sets_log/nll_IW_{file_name}.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
        )


        if disc is not None:
            disc.eval()
            disc.requires_grad_(False)

        with torch.no_grad():
            for k in range(n_batch):
                left = k * batch_traj_num
                right = min((k + 1) * batch_traj_num, total)
                batch_size = right - left

                xs = [torch.tensor(path).to(self.device) for path in real_paths[left: right]]

                for t in tqdm(range(1, self.max_T + 1)):
                    # ====================================================
                    # Prepare x_0, x_t, and timestep t
                    # ====================================================
                    lengths = torch.Tensor([x.shape[0] for x in xs]).long().to(self.device)

                    ts = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    x_t = self.destroyer.diffusion(xs, ts, ret_distr=False)
 
                    xt_padded = pad_sequence(x_t, batch_first=True, padding_value=0).long()
                    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0).long()

                    horizon = xt_padded.shape[1]
                    ts_padded = ts.view(-1, 1).repeat(1, horizon)

                    # ====================================================
                    # True posterior q(x_{t-1} | x_t, x_0)
                    # ====================================================
                    EtXt = self.Q[ts_padded.view(-1, ).to(self.device), :, xt_padded.view(-1).to(self.device)]
                    true_probs_unorm = EtXt * self.matrices[ts_padded.view(-1, ) - 1, :, xs_padded.view(-1)].to(self.device)

                    true_probs = true_probs_unorm / true_probs_unorm.sum(1, keepdim=True)
                    true_probs = clamp_probs(true_probs)
                    true_probs = rearrange(true_probs, "(b h) c -> b h c", h=horizon)
                    
                    # ====================================================
                    # Predict p_theta(x_0 | x_t)
                    # ====================================================
                    x0_pred_logits = self.restore(xt_padded.to(self.model_device), lengths.to(self.model_device), ts.to(self.model_device))
                    x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)

                    # Shape Information - b: batch size, h: horizon, c: n_vertex
                    b, h, c = x0_pred_probs.shape
                    n = 1000        # Number of importance samples

                    x0_pred_probs_flat = rearrange(x0_pred_probs, "b h c -> (b h) c") 
                    x0_sample_flat = torch.multinomial(x0_pred_probs_flat, num_samples=n, replacement=True)

                    # ====================================================
                    # Original (no guidance) posterior p_theta(x_{t-1} | x_t)
                    # ====================================================
                    counts = torch.zeros((b * h, c), device=x0_sample_flat.device, dtype=torch.float32) 
                    ones = torch.ones_like(x0_sample_flat, dtype=torch.float32) 
                    counts.scatter_add_(dim=1, index=x0_sample_flat, src=ones)

                    x0_sample_probs = (counts / float(n)).view(b, h, c)
                    
                    # Quesion. Why do we use MC here? We can directly marginalize from model distribution.
                    # Et_minus_one_bar_hat_x0 = (self.matrices[ts - 1] @ x0_pred_probs.transpose(2, 1).to(self.des_device)).to(self.device)
                    Et_minus_one_bar_hat_x0 = (self.matrices[ts - 1] @ x0_sample_probs.transpose(2, 1).to(self.des_device)).to(self.device)
                    Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")


                    pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
                    pred_probs = pred_probs_unorm / torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)

                    pred_logits = probs_to_logits(pred_probs)
                    pred_logits = rearrange(pred_logits, "(b h) c -> b h c", h=horizon)

                    # Original KL / NLL
                    if t == 1:
                        kl_before = torch.stack([F.nll_loss(pred_logits[u][:l], xs[u][:l].long(), reduction="mean") for u, l in enumerate(lengths)])
                    elif t == self.max_T:
                        kl_before += torch.stack([F.kl_div(pred_logits[u][:l], true_probs[u][:l], reduction="batchmean") for u, l in enumerate(lengths)])

                        # prior term
                        x_distr_padded = self.destroyer.diffusion(xs, ts, ret_distr=True)
                        x_distr_padded = probs_to_logits(x_distr_padded)
                        x_distr_padded = rearrange(x_distr_padded, "(b h) c -> b h c", h=horizon)
                        true_prior = torch.ones_like(x_distr_padded) / x_distr_padded.shape[-1]     # true prior is uniform distribution
                        kl_before += torch.stack([F.kl_div(x_distr_padded[u][:l], true_prior[u][:l], reduction="batchmean") for u, l in enumerate(lengths)])
                    else:
                        kl_before += torch.stack([F.kl_div(pred_logits[u][:l], true_probs[u][:l], reduction="batchmean") for u, l in enumerate(lengths)])

                    # ====================================================
                    # Importance-weighted posterior
                    # ====================================================
                    if disc is not None and n > 0 and t < 10:
                        # Shape information
                        V = disc.n_vertex + 2  # disc embedding vocab
                        bn = b * n

                        x0_sample = x0_sample_flat.view(b, h, n)  # [b, h, n]
                        x0_seq_all = x0_sample.permute(0, 2, 1).reshape(bn, h)  # [b*n, h]
                        
                        # For Micro-batch iteration
                        lengths_rep_all = lengths.repeat_interleave(n)  # [b*n]

                        if ts.ndim == 1:
                            ts_rep_all = ts.repeat_interleave(n)  # [b*n]
                        else:
                            ts_rep_all = ts.repeat_interleave(n, dim=0)  # [b*n, ...]

                        # Micro-batch loop (Reduce VRAM usage)
                        micro_b = b * 10
                        disc_logits_flat = torch.empty((bn,), device=x0_sample.device, dtype=torch.float32)

                        for s in range(0, bn, micro_b):
                            e = min(s + micro_b, bn)

                            x0_seq = x0_seq_all[s:e]  # [mb, h]
                            lengths_rep = lengths_rep_all[s:e]
                            ts_rep = ts_rep_all[s:e]
                            
                            pos = torch.arange(h, device=x0_seq.device).unsqueeze(0)
                            mask = pos >= lengths_rep.unsqueeze(1)
                            x0_seq = x0_seq.masked_fill(mask, 0)                           

                            x_in = F.one_hot(x0_seq, num_classes=V).to(torch.float32)  # [mb, h, V]

                            logits_mb = disc.discriminate(x_in, lengths_rep, torch.ones_like(ts_rep), adj_matrix=adj_matrix)  # [mb]

                            disc_logits_flat[s:e] = logits_mb

                        disc_logits = disc_logits_flat.view(b, n)  # [b, n]

                        # [Optional for IW log] - can remove this part ==========================================================================================
                        clip_val = 3
                        disc_logits_clipped = torch.clamp(disc_logits, min=-1 * clip_val, max=clip_val)

                        num_upper = (disc_logits > clip_val).float().mean().item()
                        num_lower = (disc_logits < -clip_val).float().mean().item()

                        logging.info(
                            f"Batch: {k}, t={t}, "
                            f"disc mean={torch.exp(disc_logits).mean().item():.4f}, disc std={torch.exp(disc_logits).std().item():.4f}, "
                            f"disc_clipped mean={torch.exp(disc_logits_clipped).mean().item():.4f}, disc_clipped std={torch.exp(disc_logits_clipped).std().item():.4f}, "
                            f"upper clipped ratio: {num_upper:.4f}, lower clipped ratio: {num_lower:.4f}, total num: {b*n} "
                        )
                        # [Optional for IW log] - can remove this part ==========================================================================================

                        weights = torch.exp(disc_logits)   # [b, n]
                        weights_flat = weights.unsqueeze(1).expand(b, h, n).reshape(b * h, n)  # (b*h, n)
                        weighted_counts = torch.zeros((b * h, c), device=x0_sample_flat.device, dtype=torch.float32)

                        weighted_counts.scatter_add_(
                            dim=1,
                            index=x0_sample_flat,       # Shape: (b*h, n)
                            src=weights_flat            # Shape: (b*h, n)
                        )

                        x0_sample_probs_weighted = weighted_counts.view(b, h, c)

                        Et_minus_one_bar_hat_x0 = (self.matrices[ts - 1] @ x0_sample_probs_weighted.transpose(2, 1).to(self.des_device)).to(self.device)
                        Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")

                        pred_probs_unorm = EtXt * Et_minus_one_bar_hat_x0
                    
                    # ====================================================
                    # Importance Sampling KL / NLL
                    # ====================================================
                    pred_probs = pred_probs_unorm / torch.clamp(pred_probs_unorm.sum(1, keepdim=True), min=1e-8)

                    pred_logits = probs_to_logits(pred_probs)
                    pred_logits = rearrange(pred_logits, "(b h) c -> b h c", h=horizon)

                    eps = 0.000001
                    if t == 1:
                        kl = torch.stack([F.nll_loss(pred_logits[u][:l], xs[u][:l].long(), reduction="mean") for u, l in enumerate(lengths)])
                    elif t == self.max_T:
                        kl += torch.stack([F.kl_div(pred_logits[u][:l], true_probs[u][:l], reduction="batchmean") for u, l in enumerate(lengths)])

                        # prior term
                        x_distr_padded = self.destroyer.diffusion(xs, ts, ret_distr=True)
                        x_distr_padded = probs_to_logits(x_distr_padded)
                        x_distr_padded = rearrange(x_distr_padded, "(b h) c -> b h c", h=horizon)
                        true_prior = torch.ones_like(x_distr_padded) / x_distr_padded.shape[-1]
                        kl += torch.stack([F.kl_div(x_distr_padded[u][:l], true_prior[u][:l], reduction="batchmean") for u, l in enumerate(lengths)])
                    else:
                        kl += torch.stack([F.kl_div(pred_logits[u][:l], true_probs[u][:l], reduction="batchmean") for u, l in enumerate(lengths)])

                # Compare Original KL and Importance Sampling KL     
                kl = kl / math.log(2)
                kl_before = kl_before / math.log(2)
                print("="*50)
                print(f"Original KL: {kl_before.mean()}")
                print(f"Importance Sampling KL: {kl.mean()}")

                print((kl - kl_before).mean())
                kl_all += kl.detach().to("cpu").tolist()
                kl_before_all += kl_before.detach().to("cpu").tolist()
                torch.cuda.empty_cache()
        
        print(f"Avg. of Original KL: {np.array(kl_before_all).mean()}")
        print(f"Org KL - min: {np.array(kl_before_all).min()}, max: {np.array(kl_before_all).max()}, avg: {np.array(kl_before_all).mean()}")

        return np.array(kl_all)


    def sample_with_disc(self, n_samples: int, batch_traj_num=200, real_paths=None, bool_prefix=False, disc=None, adj_matrix=None):
        assert hasattr(self, "gmm")
        if real_paths is not None:
            lengths = np.array([len(x) for x in real_paths])
        else:
            lengths = self.gmm.sample(n_samples)[0].reshape(-1).astype(int)

        lengths = np.sort(lengths[lengths > 0])
        lengths = torch.Tensor(lengths).long().to(self.device)
        
        n_batch = n_samples // batch_traj_num
        paths = []
        for b in range(n_batch):
            if bool_prefix:
                prefix = np.array([x[0] for x in real_paths])
                paths.extend(
                    self.sample_with_len_disc(lengths[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)], 
                                prefix=prefix[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)],
                                disc=disc, batch_num = b, adj_matrix=adj_matrix
                                )
                            )
            else:
                paths.extend(self.sample_with_len_disc(lengths[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)], disc=disc, batch_num=b, adj_matrix=adj_matrix))
        return paths

    def sample_with_disc_mn(self, n_samples: int, batch_traj_num=200, real_paths=None, bool_prefix=False, disc=None, adj_matrix=None):
        assert hasattr(self, "gmm")
        if real_paths is not None:
            lengths = np.array([len(x) for x in real_paths])
        else:
            lengths = self.gmm.sample(n_samples)[0].reshape(-1).astype(int)

        lengths = np.sort(lengths[lengths > 0])
        lengths = torch.Tensor(lengths).long().to(self.device)
        
        n_batch = n_samples // batch_traj_num
        paths = []
        for b in range(n_batch):
            if bool_prefix:
                prefix = np.array([x[0] for x in real_paths])
                paths.extend(
                    self.sample_with_len_disc_mn(lengths[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)], 
                                prefix=prefix[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)],
                                disc=disc, batch_num = b, adj_matrix=adj_matrix
                                )
                            )
            else:
                paths.extend(self.sample_with_len_disc_mn(lengths[b * batch_traj_num: min((b + 1) * batch_traj_num, n_samples)], disc=disc, batch_num=b, adj_matrix=adj_matrix))
        return paths

    def sample_with_len_disc_mn(self, lengths, ret_distr=False, xt=None, T=None, ret_trace=False, destroyer_new=None, disc=None, prefix=None, batch_num=None, adj_matrix=None):
        # ============================================================
        # Setup
        # ============================================================
        print("Sampling with len disc (t<10)!!!!!")

        applying_mask_intermediate = self.applying_mask_intermediate
        applying_mask_intermediate_temperature = self.applying_mask_intermediate_temperature

        V = disc.n_vertex + 2 # disc embedding vocab

        if ret_trace:
            reverse_trace = defaultdict(list) # t -> [path1, path2,...]
        
        if T is None:
            T = self.max_T

        n_samples = lengths.shape[0]
        horizon = max(lengths)

        # Initialize x_T if not provided
        if xt is None:
            xt = torch.randint(0, self.n_vertex, [n_samples, horizon]).to(self.device)
        else:
            xt = xt.to(self.device)

        # Convert prefix to tensor if prefix conditioning is used
        if prefix is not None:
            prefix = torch.as_tensor(prefix, device=self.device, dtype=xt.dtype).unsqueeze(-1)

        with torch.no_grad():
            # ============================================================
            # Reverse diffusion sampling: x_T -> x_0
            # ============================================================
            for t in range(T, 0, -1):
                ts = torch.Tensor([t]).long().to(self.device).repeat(n_samples)

                # Apply prefix conditioning at the current timestep
                if prefix is not None:
                    prefix_t = self.destroyer.diffusion(prefix, ts, ret_distr=False)
                    prefix_t = pad_sequence(prefix_t, batch_first=True, padding_value=0).long()
                    # [note] 첫 토큰은 prefix로 고정
                    xt[:, 0:1] = prefix_t

                # Predict p_theta(x_0 | x_t)
                x0_pred_logits = self.restore(xt, lengths, ts)
                x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
                x0_pred_probs_flat = rearrange(x0_pred_probs, "b h c -> (b h) c")  # [(b*h), c]
                b, h, c = x0_pred_probs.shape

                DISC_START_T = 10   # t < 10에서만 disc guidance 적용

                if t < DISC_START_T:
                    # Importance samples
                    n = self.args.n_importance_samples
                    bn = b * n
                    x0_sample_flat = torch.multinomial(x0_pred_probs_flat, num_samples=n, replacement=True)  # [(b*h), n]
                    x0_sample = x0_sample_flat.view(b, h, n)  # [b, h, n]
                    x0_seq_all = x0_sample.permute(0, 2, 1).reshape(bn, h)

                    # Discriminator importance weights with micro batch
                    lengths_rep_all = lengths.repeat_interleave(n)
                    if ts.ndim == 1:
                        ts_rep_all = ts.repeat_interleave(n)
                    else:
                        ts_rep_all = ts.repeat_interleave(n, dim=0)

                    micro_b = b * 10
                    disc_logits_flat = torch.empty((bn,), device=x0_sample.device, dtype=torch.float32)
                    for s in range(0, bn, micro_b):
                        e = min(s + micro_b, bn)
                        x0_seq = x0_seq_all[s:e]
                        lengths_rep = lengths_rep_all[s:e]
                        ts_rep = ts_rep_all[s:e]

                        pos = torch.arange(h, device=x0_seq.device).unsqueeze(0)
                        mask = pos >= lengths_rep.unsqueeze(1)
                        x0_seq = x0_seq.masked_fill(mask, 0)

                        x_in = F.one_hot(x0_seq, num_classes=V).to(torch.float32)  # [mb, h, V]
                        logits_mb = disc.discriminate(x_in, lengths_rep, torch.ones_like(ts_rep), adj_matrix=adj_matrix)  # [mb]
                        disc_logits_flat[s:e] = logits_mb

                    disc_logits = disc_logits_flat.view(b, n)  # [b, n]
                    weights = torch.exp(disc_logits)  # [b, n]

                    weights_flat = weights.unsqueeze(1).expand(b, h, n).reshape(b * h, n)  # (b*h, n)
                    weighted_counts = torch.zeros((b * h, c), device=x0_sample_flat.device, dtype=torch.float32)
                    weighted_counts.scatter_add_(dim=1, index=x0_sample_flat, src=weights_flat)

                    x0_sample_probs_weighted = weighted_counts.view(b, h, c)
                else:
                    # guidance 미적용: 예측된 x0 분포를 그대로 사용
                    x0_sample_probs_weighted = x0_pred_probs

                Et_minus_one_bar_hat_x0 = (self.matrices[t - 1] @ x0_sample_probs_weighted.transpose(2, 1)).to(self.device)
                Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")
                
                # ========================================================
                # Construct p(x_{t-1} | x_t)
                # ========================================================
                EtXt = self.Q[t, :, xt.view(-1)].T
                pred_probs = EtXt * Et_minus_one_bar_hat_x0
                
                sum_probs = torch.clamp(pred_probs.sum(1, keepdim=True), min=1e-8)
                pred_probs = pred_probs / sum_probs
                
                mask = (sum_probs == 1e-8)[:, 0]
                pred_probs[mask] = 1.0 / pred_probs.shape[1]

                # ========================================================
                # Sample x_{t-1}
                # ========================================================
                if applying_mask_intermediate:
                    pred_prob_ = rearrange(pred_probs, "(b h) c -> b h c", b=n_samples)
                    xt = torch.zeros([n_samples, horizon]).long().to(self.device)

                    x_mask = pred_prob_[:, 0].clone()
                    if (self.A.sum(dim=1) == 0).sum() != 0:
                        x_mask[:, self.A.sum(dim=1) == 0] = 0.
                    xt[:, 0] = torch.multinomial(x_mask, 1).view(-1)

                    for k in range(1, horizon):
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k]) * ((self.max_T - t) / self.max_T) + pred_prob_[:, k] * ( t / self.max_T)
                        else: ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k])  # b * v
                        random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                        x_next_masked_prob[random] = 1.
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob * ((self.max_T - t) / self.max_T) + x_next_masked_prob * (t / self.max_T)
                        else:  ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob  # b * v
                        xt[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                    
                
                else:
                    xt = torch.multinomial(pred_probs, num_samples=1, replacement=True)
                    xt = rearrange(xt, "(b h) 1 -> b h", b=n_samples) #torch.Size([n_samples, horizon])

                if ret_trace:
                    reverse_trace[t] = [xt[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]

            # ============================================================
            # Decode final trajectory from p_theta(x_0 | x_t)
            # ============================================================
            x = torch.zeros_like(xt).long().to(self.device)

            # Sample start node
            if prefix is not None:
                x[:, 0:1] = prefix
            else:
                x_mask = x0_pred_probs[:, 0].clone()            

                if (self.A.sum(dim=1) == 0).sum() != 0:
                    x_mask[:, self.A.sum(dim=1) == 0] = 0.

                x[:, 0] = torch.multinomial(x_mask, 1).view(-1)

            # Generate topology-valid trajectory
            for k in range(1, horizon):
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * (x0_pred_probs[:, k])

                random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                x_next_masked_prob[random] = 1.
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * x_next_masked_prob

                try:
                    x[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                except:
                    bad_mask = x_next_masked_prob.sum(-1) <= 0 
                    good_mask = ~bad_mask
                    if good_mask.any():
                        x_next_masked_prob_good = x_next_masked_prob[good_mask] 
                        sampled_good = torch.multinomial(x_next_masked_prob_good, 1).view(-1)
                        x[good_mask, k] = sampled_good
                    if bad_mask.any():
                        batch_size, vocab_size = x_next_masked_prob.shape
                        random_idx = torch.randint(0, vocab_size, (bad_mask.sum(),), device=x.device)
                        x[bad_mask, k] = random_idx
                        lengths[bad_mask] = k - 1

            x_list = [x[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]
            
            if ret_trace:
                reverse_trace[0] = x_list
                return reverse_trace
            if ret_distr:
                return x_list, x0_pred_probs
            return x_list

    def sample_with_len_disc(self, lengths, ret_distr=False, xt=None, T=None, ret_trace=False, destroyer_new=None, disc=None, prefix=None, batch_num=None, adj_matrix=None):
        # ============================================================
        # Setup
        # ============================================================
        print("Sampling with len disc!!!!!")

        applying_mask_intermediate = self.applying_mask_intermediate
        applying_mask_intermediate_temperature = self.applying_mask_intermediate_temperature

        V = disc.n_vertex + 2 # disc embedding vocab

        if ret_trace:
            reverse_trace = defaultdict(list) # t -> [path1, path2,...]
        
        if T is None:
            T = self.max_T

        n_samples = lengths.shape[0]
        horizon = max(lengths)

        # Initialize x_T if not provided
        if xt is None:
            xt = torch.randint(0, self.n_vertex, [n_samples, horizon]).to(self.device)
        else:
            xt = xt.to(self.device)

        # # Convert prefix to tensor if prefix conditioning is used
        # if prefix is not None:
        #     prefix = torch.as_tensor(prefix, device=self.device, dtype=xt.dtype).unsqueeze(-1)

        # New
        prefix_len = 0
        
        if prefix is not None:
            prefix = torch.as_tensor(prefix, device=self.device, dtype=xt.dtype)
            if prefix.ndim == 1:
                prefix = prefix.unsqueeze(-1)
            prefix_len = prefix.shape[1]

        with torch.no_grad():
            # ============================================================
            # Reverse diffusion sampling: x_T -> x_0
            # ============================================================
            for t in range(T, 0, -1):
                ts = torch.Tensor([t]).long().to(self.device).repeat(n_samples)

                # Apply prefix conditioning at the current timestep
                if prefix is not None:
                    prefix_t = self.destroyer.diffusion(prefix, ts, ret_distr=False)
                    prefix_t = pad_sequence(prefix_t, batch_first=True, padding_value=0).long()
                    # [note] 첫 토큰은 prefix로 고정
                    # xt[:, 0:1] = prefix_t                          # org
                    xt[:, :prefix_len] = prefix_t[:, :prefix_len]    # new

                # Predict p_theta(x_0 | x_t)
                x0_pred_logits = self.restore(xt, lengths, ts)
                x0_pred_probs = F.softmax(x0_pred_logits, dim=-1)
                x0_pred_probs_flat = rearrange(x0_pred_probs, "b h c -> (b h) c")  # [(b*h), c]
                b, h, c = x0_pred_probs.shape

                # Importance samples
                n = self.args.n_importance_samples      # number of importance samples
                bn = b * n
                x0_sample_flat = torch.multinomial(x0_pred_probs_flat, num_samples=n, replacement=True)  # [(b*h), n]
                x0_sample = x0_sample_flat.view(b, h, n)  # [b, h, n]
                x0_seq_all = x0_sample.permute(0, 2, 1).reshape(bn, h)

                # ========================================================
                # Compute discriminator importance weights with Micro batch
                # ========================================================
                lengths_rep_all = lengths.repeat_interleave(n)

                if ts.ndim == 1:
                    ts_rep_all = ts.repeat_interleave(n)
                else:  
                    ts_rep_all = ts.repeat_interleave(n, dim=0)

                micro_b = b * 10
                disc_logits_flat = torch.empty((bn,), device=x0_sample.device, dtype=torch.float32)

                for s in range(0, bn, micro_b):
                    e = min(s + micro_b, bn)

                    x0_seq = x0_seq_all[s:e]
                    lengths_rep = lengths_rep_all[s:e]
                    ts_rep = ts_rep_all[s:e]

                    pos = torch.arange(h, device=x0_seq.device).unsqueeze(0)
                    mask = pos >= lengths_rep.unsqueeze(1)
                    x0_seq = x0_seq.masked_fill(mask, 0)
                    
                    x_in = F.one_hot(x0_seq, num_classes=V).to(torch.float32) # [mb, h, V]
                    logits_mb = disc.discriminate(x_in, lengths_rep, torch.ones_like(ts_rep), adj_matrix=adj_matrix)  # [mb]
                    
                    disc_logits_flat[s:e] = logits_mb
                
                disc_logits = disc_logits_flat.view(b, n)  # [b, n]
                weights = torch.exp(disc_logits)  # [b, n]

                weights_flat = weights.unsqueeze(1).expand(b, h, n).reshape(b * h, n)  # (b*h, n)
                weighted_counts = torch.zeros((b * h, c), device=x0_sample_flat.device, dtype=torch.float32)
                weighted_counts.scatter_add_(
                    dim=1,
                    index=x0_sample_flat,  # (b, h, n)
                    src=weights_flat  # (b, h, n)
                )

                x0_sample_probs_weighted = weighted_counts.view(b, h, c)

                Et_minus_one_bar_hat_x0 = (self.matrices[t - 1] @ x0_sample_probs_weighted.transpose(2, 1)).to(self.device)
                Et_minus_one_bar_hat_x0 = rearrange(Et_minus_one_bar_hat_x0, "b c h -> (b h) c")
                
                # ========================================================
                # Construct p(x_{t-1} | x_t)
                # ========================================================
                EtXt = self.Q[t, :, xt.view(-1)].T
                pred_probs = EtXt * Et_minus_one_bar_hat_x0
                
                sum_probs = torch.clamp(pred_probs.sum(1, keepdim=True), min=1e-8)
                pred_probs = pred_probs / sum_probs
                
                mask = (sum_probs == 1e-8)[:, 0]
                pred_probs[mask] = 1.0 / pred_probs.shape[1]

                # ========================================================
                # Sample x_{t-1}
                # ========================================================
                if applying_mask_intermediate:
                    pred_prob_ = rearrange(pred_probs, "(b h) c -> b h c", b=n_samples)
                    xt = torch.zeros([n_samples, horizon]).long().to(self.device)

                    x_mask = pred_prob_[:, 0].clone()
                    if (self.A.sum(dim=1) == 0).sum() != 0:
                        x_mask[:, self.A.sum(dim=1) == 0] = 0.
                    xt[:, 0] = torch.multinomial(x_mask, 1).view(-1)

                    for k in range(1, horizon):
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k]) * ((self.max_T - t) / self.max_T) + pred_prob_[:, k] * ( t / self.max_T)
                        else: ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * (pred_prob_[:, k])  # b * v
                        random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                        x_next_masked_prob[random] = 1.
                        if applying_mask_intermediate_temperature:
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob * ((self.max_T - t) / self.max_T) + x_next_masked_prob * (t / self.max_T)
                        else:  ## Hard topology on every xt
                            x_next_masked_prob = self.A[xt[:, k - 1].view(-1)] * x_next_masked_prob  # b * v
                        xt[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                    
                
                else:
                    xt = torch.multinomial(pred_probs, num_samples=1, replacement=True)
                    xt = rearrange(xt, "(b h) 1 -> b h", b=n_samples) #torch.Size([n_samples, horizon])

                if ret_trace:
                    reverse_trace[t] = [xt[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]

            # ============================================================
            # Decode final trajectory from p_theta(x_0 | x_t)
            # ============================================================
            x = torch.zeros_like(xt).long().to(self.device)

            # Sample start node
            if prefix is not None:
                # x[:, 0:1] = prefix        # org
                x[:, :prefix_len] = prefix  # new
                start_k = prefix_len
            else:
                x_mask = x0_pred_probs[:, 0].clone()            

                if (self.A.sum(dim=1) == 0).sum() != 0:
                    x_mask[:, self.A.sum(dim=1) == 0] = 0.

                x[:, 0] = torch.multinomial(x_mask, 1).view(-1)
                start_k = 1

            # Generate topology-valid trajectory
            for k in range(start_k, horizon):
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * (x0_pred_probs[:, k])

                random = x_next_masked_prob.sum(-1, keepdim=False) < 0.000001
                x_next_masked_prob[random] = 1.
                x_next_masked_prob = self.A[x[:, k - 1].view(-1)] * x_next_masked_prob

                try:
                    x[:, k] = torch.multinomial(x_next_masked_prob, 1).view(-1)
                except:
                    bad_mask = x_next_masked_prob.sum(-1) <= 0 
                    good_mask = ~bad_mask
                    if good_mask.any():
                        x_next_masked_prob_good = x_next_masked_prob[good_mask] 
                        sampled_good = torch.multinomial(x_next_masked_prob_good, 1).view(-1)
                        x[good_mask, k] = sampled_good
                    if bad_mask.any():
                        batch_size, vocab_size = x_next_masked_prob.shape
                        random_idx = torch.randint(0, vocab_size, (bad_mask.sum(),), device=x.device)
                        x[bad_mask, k] = random_idx
                        lengths[bad_mask] = k - 1

            x_list = [x[k][:lengths[k]].cpu().tolist() for k in range(n_samples)]
            
            if ret_trace:
                reverse_trace[0] = x_list
                return reverse_trace
            if ret_distr:
                return x_list, x0_pred_probs
            return x_list


class Discriminator_module(nn.Module):
    def __init__(self, disc_model: Discriminator, destroyer_org: Destroyer, destroyer_new: Destroyer, device, use_logit_reg=False):
        super().__init__()
        self.n_vertex = destroyer_org.n_vertex
        self.disc_model = disc_model
        self.model_device = self.disc_model.device
        self.device = device
        self.destroyer_org = destroyer_org
        self.destroyer_new = destroyer_new
        self.des_device = destroyer_org.device
        self.max_T = self.destroyer_org.max_T

        self.matrices = self.destroyer_org.matrices
        self.A = destroyer_org.A
        self.Q = self.destroyer_org.get_Q()
        self.Q = self.Q.to(self.device)
        self.max_deg = self.A.sum(1).max()

        self.matrices_new = self.destroyer_new.matrices
        self.A_new = destroyer_new.A
        self.Q_new = self.destroyer_new.get_Q()
        self.Q_new = self.Q_new.to(self.device)
        self.max_deg_new = self.A_new.sum(1).max()

        self.applying_mask_intermediate = False
        self.applying_mask_intermediate_temperature = False

        self.criterion = nn.BCEWithLogitsLoss()
        self.use_logit_reg = use_logit_reg
        self.reg_lambda = 1e-3


    def forward(self, orgxs, newxs, orgA, newA):
        # xs: list of tensors of labels
        batch_size_A = len(orgxs)
        batch_size_new = len(newxs)
        batch_size = batch_size_A + batch_size_new
        if batch_size_A == 0:
            import pdb
            pdb.set_trace()

        # ========================================================
        # Prepare trajectories and discriminator labels
        # ========================================================
        lengths = torch.Tensor([x.shape[0] for x in orgxs+newxs]).long().to(self.device)

        labels_orgxs = torch.zeros(batch_size_A, dtype=torch.float, device=self.device)
        labels_newxs = torch.ones(batch_size_new, dtype=torch.float, device=self.device)
        labels = torch.cat([labels_orgxs, labels_newxs], dim=0)  # shape: (9,)

        # timestep - ts = [1, 1, ..., 1]
        ts = torch.randint(1, 2, [batch_size]).to(self.device)

        # =============================================================================
        # Diffusion forward process - "Same forward kernel" for both original and new samples
        # =============================================================================
        orgx_t = self.destroyer_org.diffusion(orgxs, ts[:batch_size_A], ret_distr=False)
        newx_t = self.destroyer_org.diffusion(newxs, ts[batch_size_A:], ret_distr=False)
        # merge and padding
        xt_padded = pad_sequence(orgx_t+newx_t, batch_first=True, padding_value=0).long()

        # =============================================================================
        # Adjacency Matrix 
        # =============================================================================
        # A_expanded = newA.unsqueeze(0).repeat(batch_size_A, 1, 1).float()
        # A_new_expanded = newA.unsqueeze(0).repeat(batch_size_new, 1, 1).float()
        # adj_matrix = torch.cat((A_expanded, A_new_expanded), dim=0)
        adj_matrix = newA

        # =============================================================================
        # Discriminator
        # =============================================================================
        disc_logits = self.discriminate(xt_padded.to(self.model_device), lengths.to(self.model_device),
                                        ts.to(self.model_device), adj_matrix.to(self.model_device))
        org_logits = disc_logits[labels == 0]
        new_logits = disc_logits[labels == 1]

        # =============================================================================
        # Loss calculation
        # =============================================================================
        bce_loss = self.criterion(disc_logits, labels)

        if self.use_logit_reg:
            # Regularization in logits
            logit_penalty = torch.mean(disc_logits ** 2)
            loss = bce_loss + self.reg_lambda * logit_penalty
        else:
            loss = bce_loss

        with torch.no_grad():
            probs = torch.sigmoid(disc_logits)
            preds = (probs >= 0.5).float()
            acc = (preds == labels).float().mean()

        return loss, org_logits, new_logits, acc


    def discriminate(self, xt_padded, lengths=None, ts=None, adj_matrix=None):
        # xt_padded: b, h value is vertex number
        # ts: b value is time for each
        batch_size = xt_padded.shape[0]
        if ts is None:
            ts = torch.Tensor([self.max_T]).repeat(batch_size).to(self.device)
        x0_pred_logits = self.disc_model(xt_padded, lengths, ts, adj_matrix)
        
        return x0_pred_logits
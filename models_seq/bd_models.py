"""
Block diffusion (BD3-LM) for OD trajectory planning.

Port of the prior project's bd_models.py (itself a port of "Block Diffusion:
Interpolating Between Autoregressive and Diffusion Language Models",
Arriola et al., ICLR 2025, github.com/kuleshov-group/bd3lms), cleaned up and
adapted to the new length specification:

  -kernel mask  : faithful BD3-LM. Absorbing ([MASK]) noising, SUBS weighted
                  cross-entropy on masked positions, first-hitting semi-AR
                  sampling (~1 forward per revealed token).
  -kernel graph : this repo's uniform-state graph CTMC kernel (Destroyer,
                  Q_t = exp((A'-D') beta_t)) applied per block with the D3PM
                  ELBO losses of Restorer.forward (seq_models.py). The state
                  space is ALWAYS augmented with the <end> state (index
                  n_vertex, total degree -bd_eos_deg) -- <end> is a first-
                  class training and generation target.

Length handling (project spec, CLAUDE.md):
  canvas = [dst, ori=v0, v1, ..., v_{L-1}=dst, END, ..., END | PAD ... PAD]
  - within the block that contains the path's final (dst) token, every
    position after it is an END token and IS a training target;
  - all subsequent blocks are PAD and are EXCLUDED from the loss;
  - generation stops when dst (hit) or END (miss) is emitted.

OD conditioning (both channels, per the full-canvas ablation results):
  1. dst as in-context prefix token (canvas position 0, held clean when
     conditioned) + ori at position 1;
  2. OD embedding od_mlp([emb(ori); emb(dst)]) added to the per-token adaLN
     conditioning vector, with a learned null condition and drop_cond
     dropout (enables unconditional generation and CFG; guidance_scale=1.0
     disables CFG).

Removed relative to the prior project (deliberately, per project decision):
  discriminator guidance (reweight / ratio / marginal), the no-<end> graph
  variant, embedding-only conditioning (bd_no_dst_prefix), the fixed-step
  mask sampler, antithetic t sampling, and the optional mask-kernel
  adjacency-filtered decode.

torch 1.12 compatible: manual matmul attention, no SDPA / FlexAttention.
"""

import math
import pickle

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import clamp_probs, probs_to_logits

from models_seq.blocks import SinusoidalPosEmb


# =====================================================================
# Block-diffusion attention masks
# =====================================================================
_MASK_CACHE = {}


def get_block_diff_mask(n, block_size, device):
    """
    Training mask over the concatenated (xt || x0) sequence of length 2n.
    Boolean (2n, 2n), True = may attend. Rows/cols [0, n) are the noised
    half, [n, 2n) the clean half. OR of three sub-masks
    (bd3lms/models/hf/modeling_bd3lm.py:28-72):
      block_diagonal      : within-block self-attention, same half
      offset_block_causal : noised query -> clean keys of strictly earlier blocks
      block_causal        : clean query  -> clean keys of earlier-or-equal blocks
    """
    key = ("train", n, block_size)
    if key not in _MASK_CACHE:
        q = torch.arange(2 * n)[:, None]
        kv = torch.arange(2 * n)[None, :]
        x0_q, x0_kv = q >= n, kv >= n
        bq = torch.div(torch.where(x0_q, q - n, q), block_size, rounding_mode="floor")
        bkv = torch.div(torch.where(x0_kv, kv - n, kv), block_size, rounding_mode="floor")
        block_diagonal = (bq == bkv) & (x0_q == x0_kv)
        offset_block_causal = (bq > bkv) & x0_kv & ~x0_q
        block_causal = (bq >= bkv) & x0_kv & x0_q
        mask = block_diagonal | offset_block_causal | block_causal
        assert bool(mask.any(dim=-1).all()), "attention mask has an empty row"
        _MASK_CACHE[key] = mask
    return _MASK_CACHE[key].to(device)


def get_block_causal_mask(s, block_size, device):
    """
    Sampling mask over the current (committed + noised-last-block) sequence:
    tokens attend within their own block and to all earlier blocks.
    """
    key = ("sample", s, block_size)
    if key not in _MASK_CACHE:
        blk = torch.div(torch.arange(s), block_size, rounding_mode="floor")
        _MASK_CACHE[key] = blk[:, None] >= blk[None, :]
    return _MASK_CACHE[key].to(device)


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


# =====================================================================
# Transformer denoiser
# =====================================================================
class BDBlock(nn.Module):
    """
    Pre-LN transformer block with per-token adaLN conditioning (DDiTBlock,
    bd3lms/models/hf/modeling_bd3lm.py:298-437; the conditioning vector c is
    (b, S, cond_dim) because each block carries its own noise level).
    """

    def __init__(self, hidden_dim, n_heads, cond_dim, dropout):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # zero-init so every block starts as identity (standard DiT)
        self.adaLN_modulation = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def _attention(self, x, attn_mask):
        b, s, _ = x.shape
        qkv = self.attn_qkv(x).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (b, s, h, d)
        q = q.transpose(1, 2)  # (b, h, s, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask[None, None], -1e9)
        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.matmul(attn, v)  # (b, h, s, d)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.attn_out(out)

    def forward(self, x, c, attn_mask):
        (shift_msa, scale_msa, gate_msa,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self._attention(_modulate(self.norm1(x), shift_msa, scale_msa), attn_mask)
        x = x + gate_mlp * self.dropout(self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class BDTransformer(nn.Module):
    """
    DiT-style x0-prediction denoiser for block diffusion.

    Vocabulary: 0..n_vertex-1 road vertices, END = n_vertex,
    PAD = n_vertex + 1 (repo convention), MASK = n_vertex + 2.

    Conditioning c (per token) = time_mlp(t_tok) + od_mlp([emb(ori); emb(dst)]),
    injected through adaLN in every block. The OD pathway has a learned
    null_cond used when the condition is dropped or absent.
    """

    def __init__(self, n_vertex, device, hidden_dim=256, n_layers=6, n_heads=8,
                 cond_dim=128, dropout=0.1, max_canvas=104, x_emb_dim=50,
                 pretrain_path=None):
        super().__init__()
        self.device = device
        self.n_vertex = n_vertex
        self.vocab_size = n_vertex + 3
        self.END = n_vertex
        self.PAD = n_vertex + 1
        self.MASK = n_vertex + 2
        self.max_canvas = max_canvas

        # vertex embedding, node2vec-initialized (pattern of eps_models.py)
        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert n_vertex == len(node2vec)
            if x_emb_dim != node2vec[0].shape[0]:
                print("Use pretrained embed dims")
            x_emb_dim = node2vec[0].shape[0]
            # small random init for END/PAD/MASK (MASK carries the whole
            # signal in the absorbing kernel -- do not zero-init it)
            nodeemb = torch.randn(self.vocab_size, x_emb_dim) * 0.02
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False)
        else:
            self.x_embedding = nn.Embedding(self.vocab_size, x_emb_dim)
        self.x_emb_dim = x_emb_dim

        self.in_proj = nn.Linear(x_emb_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_canvas, hidden_dim)

        # per-token time conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim, device),
            nn.Linear(cond_dim, 4 * cond_dim),
            nn.Mish(),
            nn.Linear(4 * cond_dim, cond_dim),
        )

        # OD condition pathway (same design validated by the full-canvas
        # ablations: shared vertex embedding -> concat -> MLP)
        self.od_mlp = nn.Sequential(
            nn.Linear(2 * x_emb_dim, 4 * cond_dim),
            nn.Mish(),
            nn.Linear(4 * cond_dim, cond_dim),
        )
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        self.blocks = nn.ModuleList([
            BDBlock(hidden_dim, n_heads, cond_dim, dropout) for _ in range(n_layers)
        ])

        # final layer: adaLN-modulated LayerNorm + zero-init head
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)
        self.head = nn.Linear(hidden_dim, self.vocab_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.to(device)

    def cond_embed(self, ori, dst, cond_mask=None, batch_size=None):
        """
        OD condition vector (b, cond_dim). ori/dst None -> null condition
        (unconditional / CFG branch); cond_mask True = keep condition
        (training-time condition dropout).
        """
        if ori is None or dst is None:
            assert batch_size is not None
            return self.null_cond.unsqueeze(0).expand(batch_size, -1)

        ori_emb = self.x_embedding(ori.long())
        dst_emb = self.x_embedding(dst.long())
        c = self.od_mlp(torch.cat([ori_emb, dst_emb], dim=-1))
        if cond_mask is not None:
            null = self.null_cond.unsqueeze(0).expand_as(c)
            c = torch.where(cond_mask.unsqueeze(-1), c, null)
        return c

    def forward(self, tokens, t_tok, ori=None, dst=None, cond_mask=None,
                attn_mask=None, canvas_len=None):
        """
        tokens     : (b, S) long. Training: S = 2n (cat(xt, x0)); sampling: S = s.
        t_tok      : (b, S) float, per-token noise level (0 for the clean half /
                     committed blocks).
        attn_mask  : (S, S) bool, True = attend.
        canvas_len : n. Both halves share positions: pos = arange(S) % n.
        returns    : (b, S, vocab_size) x0-prediction logits.
        """
        b, S = tokens.shape
        if canvas_len is None:
            canvas_len = S
        pos = torch.arange(S, device=tokens.device) % canvas_len

        x = self.in_proj(self.x_embedding(tokens)) + self.pos_emb(pos)[None]
        t_emb = self.time_mlp(t_tok.reshape(-1).float()).view(b, S, -1)
        c = t_emb + self.cond_embed(ori, dst, cond_mask, batch_size=b)[:, None, :]

        for blk in self.blocks:
            x = blk(x, c, attn_mask)

        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        return self.head(_modulate(self.final_norm(x), shift, scale))


# =====================================================================
# Block diffusion model (switchable kernel)
# =====================================================================
class BlockDiffusion(nn.Module):
    """Training + open-ended semi-AR planning. See module docstring."""

    def __init__(self, backbone: BDTransformer, destroyer, device, args):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.args = args

        self.kernel = args.kernel
        assert self.kernel in ("mask", "graph"), self.kernel
        self.block_size = args.block_size
        self.n_vertex = backbone.n_vertex
        self.END = backbone.END
        self.PAD = backbone.PAD
        self.MASK = backbone.MASK
        self.pfx = 2  # clean prefix positions: [dst, ori]

        # canvas cap, always a block multiple
        bd_max_len = getattr(args, "bd_max_len", 0)
        if bd_max_len <= 0:
            bd_max_len = getattr(args, "od_max_len", 100) + 2
        self.bd_max_len = int(math.ceil(bd_max_len / self.block_size) * self.block_size)
        assert self.bd_max_len <= backbone.max_canvas

        # mask kernel: time conditioning optional (BD3-LM default: off --
        # masked tokens self-identify the noise level). graph kernel: x_t
        # carries no visible noise indicator, so time conditioning is forced.
        self.time_cond = getattr(args, "bd_time_cond", False) or self.kernel == "graph"
        self.drop_cond = getattr(args, "drop_cond", 0.1)

        if self.kernel == "graph":
            assert destroyer is not None, "graph kernel needs a Destroyer"
            # <end> is ALWAYS a CTMC state: V_states = n_vertex + 1, with the
            # <end> STATE index equal to the END TOKEN id (= n_vertex).
            assert destroyer.n_vertex == self.n_vertex + 1, \
                "graph kernel requires the <end>-augmented state space (V+1)"
            self.destroyer = destroyer
            self.max_T = destroyer.max_T
            self.matrices = destroyer.matrices
            self.Q = destroyer.get_Q().to(device)
            self.V_states = destroyer.n_vertex
            # binarized adjacency for the block decode (the virtual <end>
            # edges are weak in the CTMC but fully allowed as moves)
            self.A = (destroyer.A > 0).float()
        else:
            self.destroyer = None
            self.V_states = self.n_vertex
            self.A = None

        self.G = None
        self._trunc_warned = False
        self.last_hits, self.last_patch_lens = [], []

    def set_graph(self, G: nx.Graph):
        self.G = G

    # -----------------------------------------------------------------
    # Canvas construction (project length spec)
    # -----------------------------------------------------------------
    def build_canvas(self, xs):
        """
        xs: list of 1-D float/long tensors (paths of vertex ids).
        Returns x0 (b, n) long, loss_mask (b, n) bool, lengths (b,) long,
        ori (b,), dst (b,). Layout (path p = [v0=ori, ..., v_{L-1}=dst],
        block size B):

          pos:   0     1    2    ...  L      L+1 ... end_blk-1 | end_blk ... n-1
          x0:   [dst,  v0,  v1,  ..., v_{L-1}, END ... END     | PAD ... PAD]

        where end_blk = end of the block containing position L (the dst
        token). END fills only the remainder of that block (possibly zero
        tokens when dst lands on a block boundary); every later block is PAD.

        loss_mask: positions [pfx, end_blk) -- the path body, dst, and the
        END tail are training targets; PAD blocks are NOT. Positions 0-1
        ([dst, ori] prefix) are clean and excluded (they are given at
        sampling); condition-dropped samples get them noised + in-loss so
        unconditional generation is in-distribution.
        """
        block = self.block_size
        cap = self.bd_max_len - 2
        paths = []
        n_trunc = 0
        for x in xs:
            p = x.long().to(self.device)
            assert p.shape[0] >= 2, "paths must have length >= 2"
            if p.shape[0] > cap:
                p = p[:cap]
                n_trunc += 1
            paths.append(p)
        if n_trunc and not self._trunc_warned:
            print(f"[BlockDiffusion] {n_trunc} path(s) in batch truncated to {cap} tokens")
            self._trunc_warned = True

        b = len(paths)
        lengths = torch.tensor([p.shape[0] for p in paths], dtype=torch.long, device=self.device)
        ori = torch.stack([p[0] for p in paths])
        dst = torch.stack([p[-1] for p in paths])
        pfx = self.pfx

        # dst token sits at position L = pfx - 1 + (len - 1) + 1 - 1 = len;
        # end of its block (exclusive):
        dst_pos = pfx - 1 + lengths - 1                          # (b,)
        end_blk = (torch.div(dst_pos, block, rounding_mode="floor") + 1) * block
        end_blk = torch.clamp(end_blk, max=self.bd_max_len)

        n = int(end_blk.max().item())
        x0 = torch.full((b, n), self.PAD, dtype=torch.long, device=self.device)
        x0[:, 0] = dst
        for i, p in enumerate(paths):
            L = p.shape[0]
            x0[i, pfx - 1:pfx - 1 + L] = p
            x0[i, pfx - 1 + L:end_blk[i]] = self.END             # END tail (may be empty)

        pos_idx = torch.arange(n, device=self.device)[None]
        loss_mask = (pos_idx >= pfx) & (pos_idx < end_blk[:, None])
        return x0, loss_mask, lengths, ori, dst

    # -----------------------------------------------------------------
    def _sample_t_blocks(self, b, nb):
        """Continuous t in [1e-3, 1], one per (sample, block)."""
        return torch.rand(b, nb, device=self.device).clamp(1e-3, 1.0)

    def _cfg_cond_mask(self, b):
        keep_p = 1.0 - self.drop_cond
        return torch.rand(b, device=self.device) < keep_p

    # -----------------------------------------------------------------
    # Training objectives
    # -----------------------------------------------------------------
    def forward(self, xs):
        if self.kernel == "mask":
            return self._forward_mask(xs)
        return self._forward_graph(xs)

    def _forward_mask(self, xs):
        """Absorbing kernel: per-block t, SUBS loss -(1/t) log p(x0) on
        masked positions (bd3lms/diffusion.py:819-891). Only loss-mask
        positions (path + END tail; prefix for dropped samples) are noised --
        PAD blocks stay clean and untrained."""
        x0, loss_mask, lengths, ori, dst = self.build_canvas(xs)
        b, n = x0.shape
        block = self.block_size
        nb = n // block

        t_b = self._sample_t_blocks(b, nb)                       # (b, nb)
        t_tok = t_b.repeat_interleave(block, dim=1)              # (b, n)

        # condition dropout: dropped samples are trained fully
        # unconditionally (no OD embedding AND the [dst, ori] prefix is
        # noised + in-loss), so plan(None, None) is in-distribution.
        cond_mask = self._cfg_cond_mask(b)
        pfx = self.pfx
        loss_mask = loss_mask.clone()
        loss_mask[~cond_mask, :pfx] = True

        move = (torch.rand(b, n, device=self.device) < t_tok) & loss_mask
        xt = torch.where(move, torch.full_like(x0, self.MASK), x0)

        x_input = torch.cat([xt, x0], dim=1)                     # (b, 2n)
        t_in = torch.cat([t_tok, torch.zeros_like(t_tok)], dim=1)
        if self.time_cond:
            t_in = t_in * 100.0  # match the graph kernel's 1..max_T scale
        else:
            t_in = torch.zeros_like(t_in)
        attn = get_block_diff_mask(n, block, self.device)
        logits = self.backbone(x_input, t_in, ori, dst, cond_mask, attn, canvas_len=n)[:, :n]

        # SUBS: the MASK state has zero probability under p(x0)
        logits = logits.clone()
        logits[..., self.MASK] = -1e9
        log_probs = logits.log_softmax(dim=-1)
        log_p_x0 = log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # (b, n)

        in_loss = move
        nll = -(1.0 / t_tok) * log_p_x0
        denom = loss_mask.float().sum().clamp(min=1.0)
        loss = (nll * in_loss.float()).sum() / denom

        with torch.no_grad():
            acc_denom = in_loss.float().sum().clamp(min=1.0)
            masked_ce = (-log_p_x0 * in_loss.float()).sum() / acc_denom
        return {"loss": loss, "nll": loss.detach(), "masked_ce": masked_ce,
                "masked_frac": move.float().mean().detach()}

    def _forward_graph(self, xs):
        """Graph CTMC kernel: Restorer.forward math (seq_models.py) with
        per-block integer t and block-diffusion attention. Diffusable states:
        real vertices + <end> (V_states = V + 1); PAD is not diffusable and
        not trained."""
        x0, loss_mask, lengths, ori, dst = self.build_canvas(xs)
        b, n = x0.shape
        block = self.block_size
        nb = n // block
        V0 = self.V_states

        # per-block integer timestep, same sampling schemes as Restorer
        sampling = self.args.train_timestep_sampling
        if sampling == "uniform":
            t_b = torch.randint(1, self.max_T + 1, (b, nb), device=self.device)
        elif sampling == "early":
            gamma = 0.7
            t = torch.arange(1, self.max_T + 1, device=self.device)
            probs = t.float().pow(-gamma)
            t_b = torch.multinomial(probs / probs.sum(), b * nb, replacement=True).view(b, nb) + 1
        elif sampling == "cosine":
            t = torch.arange(1, self.max_T + 1, device=self.device)
            probs = torch.sin(torch.pi * t / self.max_T)
            t_b = torch.multinomial(probs / probs.sum(), b * nb, replacement=True).view(b, nb) + 1
        else:
            raise NotImplementedError(sampling)
        t_tok = t_b.repeat_interleave(block, dim=1)              # (b, n)

        # condition dropout (dropped samples also diffuse the prefix and
        # learn it, enabling unconditional generation)
        cond_mask = self._cfg_cond_mask(b)
        loss_mask = loss_mask.clone()
        loss_mask[~cond_mask, :self.pfx] = True

        # x_t ~ q(x_t | x_0) on diffusable positions (PAD stays clean; the
        # prefix stays clean only for conditioned samples)
        noise_mask = loss_mask
        x0_safe = torch.where(x0 < V0, x0, torch.zeros_like(x0))
        distr = self.matrices[t_tok.reshape(-1), :, x0_safe.reshape(-1)]  # (b n, V0)
        xt_noised = torch.multinomial(distr, 1).view(b, n)
        xt = torch.where(noise_mask, xt_noised, x0)

        x_input = torch.cat([xt, x0], dim=1)
        t_in = torch.cat([t_tok.float(), torch.zeros_like(t_tok, dtype=torch.float)], dim=1)
        attn = get_block_diff_mask(n, block, self.device)
        logits = self.backbone(x_input, t_in, ori, dst, cond_mask, attn, canvas_len=n)[:, :n]
        logits_v = logits[..., :V0]
        x0_pred_probs = F.softmax(logits_v, dim=-1)              # (b, n, V0)

        # true posterior q(x_{t-1} | x_t, x_0), per token
        xt_safe = torch.where(xt < V0, xt, torch.zeros_like(xt))
        EtXt = self.Q[t_tok.reshape(-1), :, xt_safe.reshape(-1)]  # (b n, V0)
        true_unorm = EtXt * self.matrices[t_tok.reshape(-1) - 1, :, x0_safe.reshape(-1)]
        true_probs = clamp_probs(true_unorm / true_unorm.sum(1, keepdim=True))

        # predicted posterior: \bar{E}_{t-1} @ \hat{x}_0, grouped by timestep
        probs_flat = x0_pred_probs.reshape(b * n, V0)
        t_flat = t_tok.reshape(-1)
        Em1_hat = torch.empty(b * n, V0, device=self.device)
        for tv in torch.unique(t_flat).tolist():
            sel = t_flat == tv
            Em1_hat[sel] = probs_flat[sel] @ self.matrices[tv - 1].t()

        pred_unorm = EtXt * Em1_hat
        pred_probs = pred_unorm / torch.clamp(pred_unorm.sum(1, keepdim=True), min=1e-8)
        pred_logits = probs_to_logits(pred_probs)

        eps = 1e-6
        lm = loss_mask.reshape(-1).float()
        denom = lm.sum().clamp(min=1.0)

        kl_all = F.kl_div(pred_logits + eps, true_probs, reduction="none").sum(-1)
        kl_loss = (kl_all * lm).sum() / denom
        ce_all = F.cross_entropy(logits_v.reshape(-1, V0) + eps, x0_safe.reshape(-1), reduction="none")
        ce_loss = (ce_all * lm).sum() / denom

        # connectivity regularizer across consecutive real-vertex positions
        # (path body only, positions 1..L; same bilinear form as
        # seq_models.py's con_loss)
        pos_idx = torch.arange(n, device=self.device)[None]
        real = (pos_idx >= 1) & (pos_idx <= lengths[:, None])
        pair = (real[:, :-1] & real[:, 1:]).float()               # (b, n-1)
        pair_denom = (pair.sum() * V0).clamp(min=1.0)
        log_p = (x0_pred_probs + eps).log()
        t1 = torch.matmul(log_p[:, 1:], self.A.t()) * x0_pred_probs[:, :-1]
        t2 = torch.matmul(log_p[:, :-1], self.A.t()) * x0_pred_probs[:, 1:]
        con_loss = -((t1 * pair[..., None]).sum() + (t2 * pair[..., None]).sum()) / pair_denom

        return {"kl": kl_loss, "ce": ce_loss, "con": con_loss * 100}

    # -----------------------------------------------------------------
    # Planning (open-ended semi-AR generation)
    # -----------------------------------------------------------------
    def _cfg_logits(self, tokens, t_tok, ori, dst, attn_mask, w):
        """cond/uncond forward + CFG mix. w == 1.0 -> single (conditional)
        forward; the dst-prefix token stays in the sequence for both branches
        (CFG modulates only the embedding-channel condition)."""
        cond = self.backbone(tokens, t_tok, ori, dst, None, attn_mask, canvas_len=tokens.shape[1])
        if w == 1.0:
            return cond
        uncond = self.backbone(tokens, t_tok, None, None, None, attn_mask, canvas_len=tokens.shape[1])
        return uncond + w * (cond - uncond)

    @torch.no_grad()
    def plan(self, origs, dests, lengths=None, use_refine=True,
             guidance_scale=None, num_mc_samples=10, n_samples=None, **kwargs):
        """
        Plan paths for OD pairs, block by block, until the destination (hit)
        or END/PAD (miss) is emitted. `lengths` (oracle mode) only sets the
        block budget -- the model still decides where to stop.

        Unconditional generation: plan(None, None, n_samples=N) samples N
        trajectories with no OD, stopping at <end>. Needs a model trained
        with drop_cond > 0.

        Returns list of vertex-id paths; sets last_hits / last_patch_lens.
        """
        if guidance_scale is None:
            guidance_scale = getattr(self.args, "guidance_scale", 1.0)
        w = float(guidance_scale)
        order = kwargs.get("order", "first_hit")  # mask reveal order: first_hit | l2r

        uncond = origs is None or dests is None
        if uncond:
            assert n_samples is not None and n_samples > 0, \
                "unconditional plan() needs n_samples"
            b = int(n_samples)
            w = 1.0
        else:
            if type(origs) is list:
                origs = torch.Tensor(origs)
            if type(dests) is list:
                dests = torch.Tensor(dests)
            origs = origs.long().to(self.device)
            dests = dests.long().to(self.device)
            b = origs.shape[0]
        block = self.block_size

        max_blocks = self.bd_max_len // block
        if lengths is not None:
            if type(lengths) is list:
                lengths = torch.Tensor(lengths)
            budget = int(math.ceil((int(lengths.max().item()) + 2) / block))
            max_blocks = min(max(budget, 1), max_blocks)

        stop_idx = torch.full((b,), -1, dtype=torch.long)   # inclusive dst index (hit)
        end_idx = torch.full((b,), -1, dtype=torch.long)    # first END/PAD index (miss)

        pfx = self.pfx
        prefix_vals = [(0, dests), (1, origs)]
        seq = torch.empty(b, 0, dtype=torch.long, device=self.device)
        for j in range(max_blocks):
            seq = torch.cat([seq, self._init_block(b)], dim=1)
            s = seq.shape[1]
            lo = s - block
            if not uncond:
                for p_idx, val in prefix_vals:
                    if lo <= p_idx < s:
                        seq[:, p_idx] = val

            if uncond or s > pfx:  # the block has generative positions
                if self.kernel == "mask":
                    self._denoise_block_mask(seq, origs, dests, w, order=order)
                else:
                    self._denoise_block_graph(seq, origs, dests, w, num_mc_samples)

            # scan the committed block; the pfx clean prefix positions never stop
            lo = max(pfx, s - block)
            block_tokens = seq[:, lo:s].cpu()
            for i in range(b):
                if stop_idx[i] >= 0 or end_idx[i] >= 0:
                    continue
                d = -1 if uncond else int(dests[i].item())
                for k, tok in enumerate(block_tokens[i].tolist()):
                    if tok == d:
                        stop_idx[i] = lo + k
                        break
                    if tok in (self.END, self.PAD, self.MASK):
                        end_idx[i] = lo + k
                        break
            if bool(((stop_idx >= 0) | (end_idx >= 0)).all()):
                break

        # assemble paths (the returned path starts at ori = position 1;
        # the dst prefix at position 0 is dropped)
        seq = seq.cpu()
        o = pfx - 1
        paths, self.last_hits, self.last_patch_lens = [], [], []
        for i in range(b):
            if stop_idx[i] >= 0:
                p = seq[i, o:stop_idx[i] + 1].tolist()
                self.last_hits.append(True)
                self.last_patch_lens.append(0)
                paths.append(p)
                continue
            hi = int(end_idx[i].item()) if end_idx[i] >= 0 else seq.shape[1]
            p = [tok for tok in seq[i, o:hi].tolist() if tok < self.n_vertex]
            self.last_hits.append(False)
            if not uncond and use_refine and self.G is not None and len(p) > 0:
                before = len(p)
                p = self._refine_to_dest(p, int(dests[i].item()))
                self.last_patch_lens.append(len(p) - before)
            else:
                self.last_patch_lens.append(0)
            paths.append(p)
        return paths

    def _init_block(self, b):
        if self.kernel == "mask":
            return torch.full((b, self.block_size), self.MASK, dtype=torch.long, device=self.device)
        return torch.randint(0, self.V_states, (b, self.block_size), device=self.device)

    def _t_input(self, b, s, t_last):
        """Per-token t over the current sequence: committed blocks are clean
        (t=0), the last block carries t_last ((b,) or scalar)."""
        t_tok = torch.zeros(b, s, device=self.device)
        if self.time_cond:
            if torch.is_tensor(t_last):
                t_tok[:, -self.block_size:] = t_last.view(-1, 1).float()
            else:
                t_tok[:, -self.block_size:] = float(t_last)
        return t_tok

    # ---- mask kernel: first-hitting sampler ---------------------------
    def _denoise_block_mask(self, seq, origs, dests, w, temp=1.0, order="first_hit"):
        """order="first_hit": reveal one uniformly-chosen masked position per
        step (default, stochastic reveal order). order="l2r": reveal the
        left-most still-masked position each step (deterministic within-block
        left-to-right order). The first-hitting time schedule is identical in
        both, so this isolates the effect of reveal ORDER only."""
        b, s = seq.shape
        block = self.block_size
        attn = get_block_causal_mask(s, block, self.device)
        arange = torch.arange(b, device=self.device)

        t = torch.ones(b, device=self.device)
        while True:
            masked = seq[:, -block:] == self.MASK            # (b, block)
            active = masked.any(dim=1)
            if not bool(active.any()):
                break
            num_masked = masked.sum(dim=1).clamp(min=1)
            u = torch.rand(b, device=self.device)
            t = t * u ** (1.0 / num_masked.float())           # first-hitting time

            logits = self._cfg_logits(seq, self._t_input(b, s, t * 100.0), origs, dests, attn, w)
            block_logits = logits[:, -block:].clone() / temp
            block_logits[..., self.MASK] = -1e9
            p_x0 = block_logits.softmax(dim=-1)               # (b, block, V)

            if order == "l2r":
                # deterministic left-to-right: leftmost still-masked position
                idx = masked.float().argmax(dim=1)            # (b,) first True per row
            else:
                # reveal exactly one uniformly-chosen masked position per sample
                sel_probs = masked.float()
                sel_probs[~active] = 1.0                      # dummy rows
                idx = torch.multinomial(sel_probs, 1).squeeze(1)  # (b,)
            p_sel = p_x0[arange, idx]                         # (b, V)
            tok = torch.multinomial(p_sel, 1).squeeze(1)
            pos = s - block + idx
            upd = active
            seq[arange[upd], pos[upd]] = tok[upd]

    # ---- graph kernel: full T-step reverse per block ------------------
    def _denoise_block_graph(self, seq, origs, dests, w, num_mc_samples, temp=1.0):
        b, s = seq.shape
        block = self.block_size
        V0 = self.V_states
        attn = get_block_causal_mask(s, block, self.device)
        lo = s - block  # first position of the current block

        pfx = self.pfx
        cond = origs is not None and dests is not None
        clamp_prefix = lo == 0 and cond   # this block holds the clean prefix
        arange = torch.arange(b, device=seq.device)

        def _pin_prefix_tokens():
            seq[:, 0] = dests
            seq[:, 1] = origs

        def _pin_prefix_probs(probs):
            probs[:, 0] = 0.0
            probs[arange, 0, dests] = 1.0
            probs[:, 1] = 0.0
            probs[arange, 1, origs] = 1.0

        x0_probs = None
        for t in range(self.max_T, 0, -1):
            if clamp_prefix:  # block contains the clean prefix (block >= pfx)
                _pin_prefix_tokens()
            logits = self._cfg_logits(seq, self._t_input(b, s, float(t)), origs, dests, attn, w)
            logits_v = logits[:, -block:, :V0] / temp
            x0_probs = F.softmax(logits_v, dim=-1)                # (b, block, V0)
            if clamp_prefix:
                _pin_prefix_probs(x0_probs)

            # MC reverse posterior (width = block)
            xb = seq[:, -block:]
            EtXt = self.Q[t, :, xb.reshape(-1)].T                 # (b*block, V0)
            x0_flat = x0_probs.reshape(-1, V0)
            x0_sample = torch.multinomial(x0_flat, num_samples=num_mc_samples, replacement=True)
            Em1 = self.matrices[t - 1, x0_sample.reshape(-1)]
            Em1 = Em1.view(-1, num_mc_samples, V0).mean(dim=1)

            pred_unorm = EtXt * Em1
            sum_probs = torch.clamp(pred_unorm.sum(1, keepdim=True), min=1e-8)
            pred_probs = pred_unorm / sum_probs
            degenerate = (sum_probs == 1e-8)[:, 0]
            pred_probs[degenerate] = 1.0 / V0

            seq[:, -block:] = torch.multinomial(pred_probs, 1).view(b, block)

        # final adjacency-constrained decode of the block, continuing from
        # the last committed vertex. Clean-prefix positions are pinned;
        # unconditional mode free-samples the first position.
        if clamp_prefix:
            _pin_prefix_tokens()
            start = pfx                       # walk begins after the prefix
        elif lo == 0 and not cond:            # unconditional: free-sample pos 0
            seq[:, 0] = torch.multinomial(torch.clamp(x0_probs[:, 0], min=1e-12), 1).view(-1)
            start = 1
        else:
            start = 0
        if start >= block:
            return
        prev = seq[:, lo + start - 1]  # ori (pos 1) or previous block's last vertex
        for k in range(start, block):
            probs_k = x0_probs[:, k]
            masked_prob = self.A[prev] * probs_k
            bad = masked_prob.sum(-1) < 1e-6
            masked_prob[bad] = 1.0
            masked_prob = self.A[prev] * masked_prob
            still_bad = masked_prob.sum(-1) <= 0
            if bool(still_bad.any()):
                masked_prob[still_bad] = 1.0
            tok = torch.multinomial(masked_prob, 1).view(-1)
            seq[:, s - block + k] = tok
            prev = tok

    # -----------------------------------------------------------------
    # Importance-weight guidance (mask kernel; BD_GUIDANCE_FORMULATION.pdf)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def plan_guided(self, origs, dests, disc, adj_scn, deg_ratio, n_is=100,
                    w_gamma=1.0, cand_temp=1.0, guidance_scale=None,
                    disc_micro_bs=4096, ess_log=None, adj_prop=False,
                    diag_log=None, **kwargs):
        """plan() with discriminator importance-weight guidance at every
        first-hitting reveal. Mask kernel only. disc: BDDiscriminator or None
        (None + adj_prop=True -> pure adjacency-constrained sampling, the
        unguided control of BD_GUIDANCE_FORMULATION.pdf Lemma 3).
        adj_prop: mask candidate proposals to scenario-legal transitions
        w.r.t. REVEALED neighbours (Lemma 3: exact under the ideal D --
        the excluded candidates carry zero target weight).
        adj_scn (V, V) float, deg_ratio (V,) float on self.device.
        ess_log: optional list collecting per-reveal mean ESS.
        Kernel dispatch: mask -> guided first-hitting reveal; graph -> guided
        T-step CTMC reverse (the GDP formulation: the MC-posterior x0-hat is
        replaced by the D/(1-D)-weighted candidate average x0-bar; with
        adj_prop the final adjacency-constrained decode runs on the SCENARIO
        adjacency instead of the training graph)."""
        if guidance_scale is None:
            guidance_scale = getattr(self.args, "guidance_scale", 1.0)
        w = float(guidance_scale)

        if type(origs) is list:
            origs = torch.Tensor(origs)
        if type(dests) is list:
            dests = torch.Tensor(dests)
        origs = origs.long().to(self.device)
        dests = dests.long().to(self.device)
        b = origs.shape[0]
        block = self.block_size
        max_blocks = self.bd_max_len // block

        stop_idx = torch.full((b,), -1, dtype=torch.long)
        end_idx = torch.full((b,), -1, dtype=torch.long)
        pfx = self.pfx
        prefix_vals = [(0, dests), (1, origs)]
        seq = torch.empty(b, 0, dtype=torch.long, device=self.device)
        for j in range(max_blocks):
            seq = torch.cat([seq, self._init_block(b)], dim=1)
            s = seq.shape[1]
            lo = s - block
            for p_idx, val in prefix_vals:
                if lo <= p_idx < s:
                    seq[:, p_idx] = val
            if s > pfx:
                if self.kernel == "mask":
                    self._denoise_block_mask_guided(
                        seq, origs, dests, w, disc, adj_scn, deg_ratio,
                        n_is, w_gamma, cand_temp, disc_micro_bs, ess_log,
                        adj_prop=adj_prop, diag_log=diag_log)
                else:
                    self._denoise_block_graph_guided(
                        seq, origs, dests, w, disc, adj_scn, deg_ratio,
                        n_is, w_gamma, cand_temp, disc_micro_bs, ess_log,
                        adj_prop=adj_prop, diag_log=diag_log)

            lo = max(pfx, s - block)
            block_tokens = seq[:, lo:s].cpu()
            for i in range(b):
                if stop_idx[i] >= 0 or end_idx[i] >= 0:
                    continue
                d = int(dests[i].item())
                for k, tok in enumerate(block_tokens[i].tolist()):
                    if tok == d:
                        stop_idx[i] = lo + k
                        break
                    if tok in (self.END, self.PAD, self.MASK):
                        end_idx[i] = lo + k
                        break
            if bool(((stop_idx >= 0) | (end_idx >= 0)).all()):
                break

        seq = seq.cpu()
        o = pfx - 1
        paths, self.last_hits, self.last_patch_lens = [], [], []
        for i in range(b):
            if stop_idx[i] >= 0:
                paths.append(seq[i, o:stop_idx[i] + 1].tolist())
                self.last_hits.append(True)
                self.last_patch_lens.append(0)
                continue
            hi = int(end_idx[i].item()) if end_idx[i] >= 0 else seq.shape[1]
            paths.append([t for t in seq[i, o:hi].tolist() if t < self.n_vertex])
            self.last_hits.append(False)
            self.last_patch_lens.append(0)
        return paths

    def _disc_lengths(self, x, dests):
        """Truncation lengths for disc input: cut before END/PAD/MASK, cut
        after dst (inclusive), scanning positions >= pfx. x: (m, s), dests (m,)."""
        m, s = x.shape
        pos = torch.arange(s, device=x.device)[None]
        term = (x >= self.n_vertex) & (pos >= self.pfx)
        hit = (x == dests[:, None]) & (pos >= self.pfx)
        ev = term | hit
        any_ev = ev.any(dim=1)
        idx = torch.argmax(ev.float(), dim=1)
        ar = torch.arange(m, device=x.device)
        length = torch.where(any_ev, idx + hit[ar, idx].long(),
                             torch.full_like(idx, s))
        return length.clamp(min=2)

    def _denoise_block_mask_guided(self, seq, origs, dests, w, disc, adj_scn,
                                   deg_ratio, n_is, w_gamma, cand_temp,
                                   micro_bs, ess_log, adj_prop=False,
                                   diag_log=None):
        """First-hitting reveal where the reveal-target x0-bar is the
        D/(1-D)-reweighted candidate average (Eqs. 3+5 of the guidance doc)."""
        b, s = seq.shape
        block = self.block_size
        V = self.backbone.vocab_size
        attn = get_block_causal_mask(s, block, self.device)
        arange = torch.arange(b, device=self.device)
        PAD_D = disc.PAD if disc is not None else 0

        t = torch.ones(b, device=self.device)
        while True:
            masked = seq[:, -block:] == self.MASK
            active = masked.any(dim=1)
            if not bool(active.any()):
                break
            num_masked = masked.sum(dim=1).clamp(min=1)
            u = torch.rand(b, device=self.device)
            t = t * u ** (1.0 / num_masked.float())

            logits = self._cfg_logits(seq, self._t_input(b, s, t * 100.0), origs, dests, attn, w)
            block_logits = logits[:, -block:].clone()
            block_logits[..., self.MASK] = -1e9
            p_x0 = block_logits.softmax(dim=-1)                       # (b, block, V)
            p_cand = (block_logits / cand_temp).softmax(dim=-1) if cand_temp != 1.0 else p_x0

            if adj_prop:
                # Lemma 3 legality mask from REVEALED neighbours only.
                # left neighbour of block position j is canvas position lo+j-1
                # (committed block or already-revealed token); right neighbour
                # is lo+j+1 when inside the current sequence.
                lo = s - block
                m = torch.ones_like(p_cand)                       # (b, block, V)
                nv = self.n_vertex
                left = seq[:, lo - 1:s - 1] if lo >= 1 else torch.cat(
                    [torch.full((b, 1), self.MASK, dtype=seq.dtype, device=seq.device),
                     seq[:, :s - 1]], dim=1)                      # (b, block)
                right = torch.cat([seq[:, lo + 1:s],
                                   torch.full((b, 1), self.MASK, dtype=seq.dtype,
                                              device=seq.device)], dim=1)
                lv = (left < nv)                                  # real revealed vertex
                if lo == 0:  # canvas pos 0 (dst prefix) is not a traversal
                    lv[:, :2] = False
                    lv[:, 2:] &= True
                rv = (right < nv)
                lsafe = torch.where(lv, left, torch.zeros_like(left))
                rsafe = torch.where(rv, right, torch.zeros_like(right))
                m_left = adj_scn[lsafe.reshape(-1)].view(b, block, -1)
                m_right = adj_scn[rsafe.reshape(-1)].view(b, block, -1)
                m[..., :nv] = torch.where(lv.unsqueeze(-1), m_left, torch.ones_like(m_left)) \
                    * torch.where(rv.unsqueeze(-1), m_right, torch.ones_like(m_right))
                pm = p_cand * m
                z = pm.sum(-1, keepdim=True)
                p_cand = torch.where(z > 1e-9, pm / z.clamp(min=1e-9), p_cand)

            if disc is None:
                # unguided adjacency-constrained control: reveal directly from
                # the (masked) marginal at one uniformly-chosen masked position
                sel_probs = masked.float()
                sel_probs[~active] = 1.0
                idx = torch.multinomial(sel_probs, 1).squeeze(1)
                p_sel = p_cand[arange, idx].clamp(min=1e-12)
                tok = torch.multinomial(p_sel, 1).squeeze(1)
                pos = s - block + idx
                seq[arange[active], pos[active]] = tok[active]
                continue

            # ---- mean-field candidates: (b, n, block) ----------------
            cand = torch.multinomial(
                p_cand.reshape(b * block, V), n_is, replacement=True
            ).view(b, block, n_is).permute(0, 2, 1).contiguous()
            cur = seq[:, -block:].unsqueeze(1).expand(b, n_is, block)
            cand = torch.where(masked.unsqueeze(1), cand, cur)

            # ---- disc scoring of [prefix || candidate block] ----------
            full = seq.unsqueeze(1).expand(b, n_is, s).clone()
            full[:, :, -block:] = cand
            full = full.view(b * n_is, s)
            d_rep = dests.repeat_interleave(n_is)
            lens = self._disc_lengths(full, d_rep)
            x_disc = torch.where(
                torch.arange(s, device=full.device)[None] < lens[:, None],
                full, torch.full_like(full, PAD_D))
            x_disc = torch.where(x_disc >= self.n_vertex,
                                 torch.full_like(x_disc, PAD_D), x_disc)

            dlogits = torch.empty(b * n_is, device=self.device)
            for lo_i in range(0, b * n_is, micro_bs):
                hi_i = min(lo_i + micro_bs, b * n_is)
                dlogits[lo_i:hi_i] = disc(x_disc[lo_i:hi_i], lens[lo_i:hi_i],
                                          adj_scn, deg_ratio)
            dlogits = dlogits.view(b, n_is)

            # self-normalized weights (row-max shift for stability) + ESS
            g = w_gamma * dlogits
            wgt = torch.exp(g - g.max(dim=1, keepdim=True).values)     # (b, n)
            if ess_log is not None:
                ess = (wgt.sum(1) ** 2 / (wgt ** 2).sum(1).clamp(min=1e-12))
                ess_log.append(float(ess[active].mean().item()) if bool(active.any()) else float("nan"))
            if diag_log is not None:
                ess_d = (wgt.sum(1) ** 2 / (wgt ** 2).sum(1).clamp(min=1e-12))
                uus = [torch.unique(cand[bi], dim=0).shape[0]
                       for bi in range(b) if bool(active[bi])]
                diag_log.append({
                    "t": float(t[active].mean().item()) if bool(active.any()) else float("nan"),
                    "uniq": (sum(uus) / len(uus)) if uus else float("nan"),
                    "ess": float(ess_d[active].mean().item()) if bool(active.any()) else float("nan"),
                    "n_is": n_is,
                })

            # ---- reweighted reveal target x0-bar ----------------------
            idx_flat = cand.permute(0, 2, 1).reshape(b * block, n_is)   # (b*block, n)
            w_flat = wgt.unsqueeze(1).expand(b, block, n_is).reshape(b * block, n_is)
            xbar = torch.zeros(b * block, V, device=self.device)
            xbar.scatter_add_(1, idx_flat, w_flat)
            xbar_sum = xbar.sum(1, keepdim=True)
            xbar = torch.where(xbar_sum > 1e-12, xbar / xbar_sum.clamp(min=1e-12),
                               p_cand.reshape(b * block, V))
            xbar = xbar.view(b, block, V)

            # ---- reveal ONE uniformly-chosen masked position ----------
            sel_probs = masked.float()
            sel_probs[~active] = 1.0
            idx = torch.multinomial(sel_probs, 1).squeeze(1)
            p_sel = xbar[arange, idx].clamp(min=0)
            p_sel = torch.where(p_sel.sum(1, keepdim=True) > 1e-12, p_sel,
                                p_cand[arange, idx])
            tok = torch.multinomial(p_sel.clamp(min=1e-12), 1).squeeze(1)
            pos = s - block + idx
            upd = active
            seq[arange[upd], pos[upd]] = tok[upd]


    def _denoise_block_graph_guided(self, seq, origs, dests, w, disc, adj_scn,
                                    deg_ratio, n_is, w_gamma, cand_temp,
                                    micro_bs, ess_log, adj_prop=False,
                                    diag_log=None):
        """Guided T-step CTMC reverse for the uniform (graph) kernel.
        Per step: mean-field candidates from x0-hat, D/(1-D) weights on
        [prefix || candidate block], reweighted x0-bar drives the posterior
        (Eqs. 1-3 of the guidance doc with the CTMC posterior). disc=None ->
        unguided x0-hat (adj-only control). The final adjacency-constrained
        walk uses the scenario adjacency when adj_prop=True."""
        b, s = seq.shape
        block = self.block_size
        V0 = self.V_states
        attn = get_block_causal_mask(s, block, self.device)
        lo = s - block
        pfx = self.pfx
        clamp_prefix = lo == 0
        arange = torch.arange(b, device=seq.device)
        PAD_D = disc.PAD if disc is not None else 0

        def _pin_tokens():
            seq[:, 0] = dests
            seq[:, 1] = origs

        def _pin_probs(probs):
            probs[:, 0] = 0.0
            probs[arange, 0, dests] = 1.0
            probs[:, 1] = 0.0
            probs[arange, 1, origs] = 1.0

        xbar = None
        for t in range(self.max_T, 0, -1):
            if clamp_prefix:
                _pin_tokens()
            logits = self._cfg_logits(seq, self._t_input(b, s, float(t)), origs, dests, attn, w)
            logits_v = logits[:, -block:, :V0] / cand_temp
            x0_probs = F.softmax(logits_v, dim=-1)                 # (b, block, V0)
            if clamp_prefix:
                _pin_probs(x0_probs)

            if disc is None:
                xbar = x0_probs
            else:
                cand = torch.multinomial(
                    x0_probs.reshape(b * block, V0).clamp(min=1e-12), n_is, replacement=True
                ).view(b, block, n_is).permute(0, 2, 1).contiguous()   # (b, n, block)
                full = seq.unsqueeze(1).expand(b, n_is, s).clone()
                full[:, :, -block:] = cand
                full = full.view(b * n_is, s)
                d_rep = dests.repeat_interleave(n_is)
                lens = self._disc_lengths(full, d_rep)
                x_disc = torch.where(
                    torch.arange(s, device=full.device)[None] < lens[:, None],
                    full, torch.full_like(full, PAD_D))
                x_disc = torch.where(x_disc >= self.n_vertex,
                                     torch.full_like(x_disc, PAD_D), x_disc)
                dlogits = torch.empty(b * n_is, device=self.device)
                for lo_i in range(0, b * n_is, micro_bs):
                    hi_i = min(lo_i + micro_bs, b * n_is)
                    dlogits[lo_i:hi_i] = disc(x_disc[lo_i:hi_i], lens[lo_i:hi_i],
                                              adj_scn, deg_ratio)
                g = w_gamma * dlogits.view(b, n_is)
                wgt = torch.exp(g - g.max(dim=1, keepdim=True).values)
                if ess_log is not None:
                    ess = (wgt.sum(1) ** 2 / (wgt ** 2).sum(1).clamp(min=1e-12))
                    ess_log.append(float(ess.mean().item()))
                if diag_log is not None:
                    ess_d = (wgt.sum(1) ** 2 / (wgt ** 2).sum(1).clamp(min=1e-12))
                    uus = [torch.unique(cand[bi], dim=0).shape[0] for bi in range(b)]
                    diag_log.append({
                        "t": float(t) / float(self.max_T),
                        "uniq": sum(uus) / len(uus),
                        "ess": float(ess_d.mean().item()),
                        "n_is": n_is,
                    })

                idx_flat = cand.permute(0, 2, 1).reshape(b * block, n_is)
                w_flat = wgt.unsqueeze(1).expand(b, block, n_is).reshape(b * block, n_is)
                xbar = torch.zeros(b * block, V0, device=self.device)
                xbar.scatter_add_(1, idx_flat, w_flat)
                xs = xbar.sum(1, keepdim=True)
                xbar = torch.where(xs > 1e-12, xbar / xs.clamp(min=1e-12),
                                   x0_probs.reshape(b * block, V0))
                xbar = xbar.view(b, block, V0)
                if clamp_prefix:
                    _pin_probs(xbar)

            # CTMC posterior with the (reweighted) x0-bar
            xb = seq[:, -block:]
            EtXt = self.Q[t, :, xb.reshape(-1)].T                   # (b*block, V0)
            Em1 = torch.matmul(xbar, self.matrices[t - 1])          # (b, block, V0)
            pred_unorm = EtXt * Em1.reshape(b * block, V0)
            sum_p = torch.clamp(pred_unorm.sum(1, keepdim=True), min=1e-8)
            pred = pred_unorm / sum_p
            pred[(sum_p == 1e-8)[:, 0]] = 1.0 / V0
            seq[:, -block:] = torch.multinomial(pred, 1).view(b, block)

        # final adjacency-constrained decode (scenario adjacency if adj_prop)
        if adj_prop:
            A_dec = torch.zeros(V0, V0, device=self.device)
            V = self.n_vertex
            A_dec[:V, :V] = adj_scn
            A_dec[V, :] = 1.0
            A_dec[:, V] = 1.0
        else:
            A_dec = self.A
        if clamp_prefix:
            _pin_tokens()
            start = pfx
        else:
            start = 0
        if start >= block:
            return
        prev = seq[:, lo + start - 1]
        for k in range(start, block):
            probs_k = xbar[:, k]
            masked_prob = A_dec[prev] * probs_k
            bad = masked_prob.sum(-1) < 1e-6
            masked_prob[bad] = 1.0
            masked_prob = A_dec[prev] * masked_prob
            still_bad = masked_prob.sum(-1) <= 0
            if bool(still_bad.any()):
                masked_prob[still_bad] = 1.0
            tok = torch.multinomial(masked_prob, 1).view(-1)
            seq[:, s - block + k] = tok
            prev = tok

    # -----------------------------------------------------------------
    def _refine_to_dest(self, seq, dst, max_patch=None):
        """Append a shortest-path patch from the last vertex to dst."""
        try:
            patch = nx.shortest_path(self.G, source=seq[-1], target=dst)
            if max_patch is None or len(patch) - 1 <= max_patch:
                return seq + patch[1:]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return seq

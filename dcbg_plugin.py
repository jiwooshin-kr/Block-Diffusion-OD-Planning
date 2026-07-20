"""
D-CBG plug-in: classifier-based guidance of Schiff et al.,
"Simple Guidance Mechanisms for Discrete Diffusion Models"
(github.com/kuleshov-group/discrete-diffusion-guidance), applied to this
repo's BD models WITHOUT modifying any existing file.

Ported pieces (kept as close to their diffusion.py::_cbg_denoise as the
sampler mismatch allows; their-code sections are marked):
  - exact D-CBG: enumerate token substitutions, tilt the posterior by
    gamma * log p(y | x_t^{l->k}, t). Under first-hitting only the revealed
    position's distribution is consumed, so restricting the enumeration to
    that position is EXACT, not an approximation.
  - first-order D-CBG (use_approx=True): Taylor expansion via the gradient
    of the classifier w.r.t. the one-hot input (their lines 1367-1381),
    used for the graph/uniform kernel where full enumeration x 100 steps
    is intractable (their own choice on large-vocab tasks).
  - noise-conditioned classifier p(y | x_t, t) trained on LABELED DATA
    (exceptional vs normal) corrupted by the model's own forward process --
    their classifier.py protocol (NOT this repo's partial-path/model-negative
    discriminator).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_seq.bd_disc import _EncBlock, _SAGELayer, BDDiscriminator
from models_seq.blocks import SinusoidalPosEmb


# =====================================================================
# Noise-conditioned classifier (their classifier.py protocol)
# =====================================================================
class DCBGClassifier(nn.Module):
    """p(y=exceptional | x_t, t). Input: canvas-form noisy sequences
    (vocab = vertices + END + PAD + MASK) as indices or one-hot floats
    (one-hot needed for the first-order approximation's input gradient)."""

    def __init__(self, n_vertex, device, dim=128, n_layers=2, n_heads=4,
                 max_len=128, dropout=0.1, pretrain_path=None):
        super().__init__()
        self.n_vertex = n_vertex
        self.END = n_vertex
        self.PAD = n_vertex + 1
        self.MASK = n_vertex + 2
        self.vocab = n_vertex + 3
        self.device = device

        x_dim = 100
        emb = torch.randn(self.vocab, x_dim) * 0.02
        if pretrain_path is not None:
            import pickle
            n2v = pickle.load(open(pretrain_path, "rb"))
            x_dim = n2v[0].shape[0]
            emb = torch.randn(self.vocab, x_dim) * 0.02
            for k in n2v:
                emb[k] = torch.from_numpy(n2v[k])
        self.x_embedding = nn.Embedding.from_pretrained(emb, freeze=False)
        self.in_proj = nn.Linear(x_dim, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim, device),
                                      nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_EncBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.to(device)

    def forward(self, x, t):
        """x: (b, s) long indices OR (b, s, vocab) one-hot float. t: (b,) float."""
        if x.dtype in (torch.float16, torch.float32, torch.float64):
            h = torch.matmul(x, self.x_embedding.weight)
            pad_mask = x.argmax(-1) == self.PAD
        else:
            h = self.x_embedding(x)
            pad_mask = x == self.PAD
        b, s = h.shape[0], h.shape[1]
        h = self.in_proj(h) + self.pos_emb(torch.arange(s, device=h.device))[None]
        h = h + self.time_mlp(t.float())[:, None, :]
        for blk in self.blocks:
            h = blk(h, pad_mask)
        h = self.final_norm(h)
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)          # logit of p(y=1 | x_t, t)

    def get_log_probs(self, x, t):
        """(b, 2) log-probs over {normal, exceptional} -- mirrors their API."""
        z = self.forward(x, t)
        return torch.stack([F.logsigmoid(-z), F.logsigmoid(z)], dim=-1)


# =====================================================================
# Generation-time-matched corruption for classifier training
# =====================================================================
def make_canvas(path, block, max_len=128):
    """[dst, ori, v1, ..., v_L=dst, END...END] up to the dst block's end."""
    L = len(path)
    dst_pos = 1 + L - 1
    end_blk = min(((dst_pos // block) + 1) * block, max_len)
    c = [path[-1]] + list(path[:min(L, end_blk - 1)])
    c = c + [None] * (end_blk - len(c))
    return c  # None marks END-tail slots


def corrupt_mask(paths, block, rng, END, PAD, MASK, max_len=128):
    """Mask-kernel generation-time state: j committed clean blocks + the
    current block masked at rate t (first-hitting marginal); future absent.
    Returns tokens (b, n), t (b,)."""
    outs, ts = [], []
    for p in paths:
        c = make_canvas(p, block, max_len)
        c = [END if v is None else v for v in c]
        nb = len(c) // block
        j = int(rng.integers(0, nb))                  # committed blocks
        t = float(rng.uniform(1e-3, 1.0))
        cur_lo, cur_hi = j * block, (j + 1) * block
        row = list(c[:cur_hi])
        for i in range(max(cur_lo, 2), cur_hi):
            if rng.random() < t:
                row[i] = MASK
        outs.append(row)
        ts.append(t)
    n = max(len(r) for r in outs)
    tok = torch.full((len(outs), n), PAD, dtype=torch.long)
    for i, r in enumerate(outs):
        tok[i, :len(r)] = torch.tensor(r)
    return tok, torch.tensor(ts)


def corrupt_graph(paths, matrices, max_T, rng, END, PAD, block=64, max_len=128):
    """Graph/uniform-kernel state: whole current block noised by
    q(x_t | x_0) at integer t; prefix (dst, ori) clean."""
    V0 = matrices.shape[1]
    outs, ts = [], []
    for p in paths:
        c = make_canvas(p, block, max_len)
        c = [END if v is None else v for v in c]
        t = int(rng.integers(1, max_T + 1))
        row = torch.tensor(c, dtype=torch.long)
        x0 = row.clamp(max=V0 - 1)
        distr = matrices[t, :, x0].T.clamp(min=0)     # (n, V0)
        noised = torch.multinomial(distr, 1).squeeze(1)
        row[2:] = noised[2:]
        outs.append(row)
        ts.append(float(t))
    n = max(len(r) for r in outs)
    tok = torch.full((len(outs), n), PAD, dtype=torch.long)
    for i, r in enumerate(outs):
        tok[i, :len(r)] = r
    return tok, torch.tensor(ts)


# =====================================================================
# D-CBG samplers (plug-in around a loaded BlockDiffusion model)
# =====================================================================
from models_seq.bd_models import get_block_causal_mask


@torch.no_grad()
def plan_dcbg_mask(model, origs, dests, clf, gamma, micro_bs=4096, use_approx=False, **kw):
    """Mask kernel + exact D-CBG at the revealed position: the reveal target
    softmax(log p_theta + gamma * log p_phi(y | x_t^{l->k})) over the vocab
    (their _cbg_denoise absorbing branch, restricted -- exactly -- to the one
    position first-hitting consumes)."""
    origs = torch.as_tensor(origs).long().to(model.device)
    dests = torch.as_tensor(dests).long().to(model.device)
    b = origs.shape[0]
    block = model.block_size
    pfx = model.pfx
    max_blocks = model.bd_max_len // block
    stop_idx = torch.full((b,), -1, dtype=torch.long)
    end_idx = torch.full((b,), -1, dtype=torch.long)
    seq = torch.empty(b, 0, dtype=torch.long, device=model.device)
    arange = torch.arange(b, device=model.device)

    for j in range(max_blocks):
        seq = torch.cat([seq, model._init_block(b)], dim=1)
        s = seq.shape[1]
        for p_idx, val in [(0, dests), (1, origs)]:
            if s - block <= p_idx < s:
                seq[:, p_idx] = val
        if s > pfx:
            attn = get_block_causal_mask(s, block, model.device)
            t = torch.ones(b, device=model.device)
            while True:
                masked = seq[:, -block:] == model.MASK
                active = masked.any(dim=1)
                if not bool(active.any()):
                    break
                num_masked = masked.sum(dim=1).clamp(min=1)
                u = torch.rand(b, device=model.device)
                t = t * u ** (1.0 / num_masked.float())

                logits = model._cfg_logits(seq, model._t_input(b, s, t * 100.0),
                                           origs, dests, attn, 1.0)
                block_logits = logits[:, -block:].clone()
                block_logits[..., model.MASK] = -1e9
                log_p = block_logits.log_softmax(dim=-1)          # (b, block, V)

                sel = masked.float()
                sel[~active] = 1.0
                idx = torch.multinomial(sel, 1).squeeze(1)        # reveal position
                pos = s - block + idx

                V = model.backbone.vocab_size
                if use_approx:
                    # ---- their first-order (Taylor) approximation:
                    # ONE classifier forward+backward per reveal step ----
                    xt_one_hot = F.one_hot(seq, clf.vocab).to(torch.float)
                    with torch.enable_grad():
                        xt_one_hot.requires_grad_(True)
                        lp_xt = clf.get_log_probs(xt_one_hot, t * 100.0)
                        lp_xt[..., 1].sum().backward()
                        grad = xt_one_hot.grad
                    ratio = (grad - (xt_one_hot * grad).sum(dim=-1, keepdim=True)).detach()
                    full_lp = ratio + lp_xt[..., 1].detach()[:, None, None]   # (b, s, vocab)
                    clf_lp = full_lp[arange, pos][:, :V]
                else:
                    # ---- their exact enumeration, at the reveal position ----
                    xt_jumps = seq.unsqueeze(1).repeat(1, V, 1)        # (b, V, s)
                    xt_jumps[arange[:, None], torch.arange(V, device=seq.device)[None, :].expand(b, V),
                             pos[:, None].expand(b, V)] = torch.arange(V, device=seq.device)[None, :].expand(b, V)
                    flat = xt_jumps.view(b * V, s)
                    t_rep = (t * 100.0).repeat_interleave(V)
                    clf_lp = torch.empty(b * V, device=model.device)
                    for lo in range(0, b * V, micro_bs):
                        hi = min(lo + micro_bs, b * V)
                        clf_lp[lo:hi] = clf.get_log_probs(flat[lo:hi], t_rep[lo:hi])[:, 1]
                    clf_lp = clf_lp.view(b, V)

                guided = log_p[arange, idx] + gamma * clf_lp       # (b, V)
                guided[:, model.MASK] = -1e9
                tok = torch.multinomial(guided.softmax(dim=-1), 1).squeeze(1)
                seq[arange[active], pos[active]] = tok[active]

        lo2 = max(pfx, s - block)
        bt = seq[:, lo2:s].cpu()
        for i in range(b):
            if stop_idx[i] >= 0 or end_idx[i] >= 0:
                continue
            d = int(dests[i].item())
            for k, tok_ in enumerate(bt[i].tolist()):
                if tok_ == d:
                    stop_idx[i] = lo2 + k
                    break
                if tok_ in (model.END, model.PAD, model.MASK):
                    end_idx[i] = lo2 + k
                    break
        if bool(((stop_idx >= 0) | (end_idx >= 0)).all()):
            break

    seq = seq.cpu()
    o = pfx - 1
    paths, hits = [], []
    for i in range(b):
        if stop_idx[i] >= 0:
            paths.append(seq[i, o:stop_idx[i] + 1].tolist())
            hits.append(True)
            continue
        hi = int(end_idx[i].item()) if end_idx[i] >= 0 else seq.shape[1]
        paths.append([v for v in seq[i, o:hi].tolist() if v < model.n_vertex])
        hits.append(False)
    model.last_hits = hits
    return paths


def plan_dcbg_graph(model, origs, dests, clf, gamma, **kw):
    """Graph/uniform kernel + first-order D-CBG (their use_approx branch,
    copied): per reverse step one classifier forward+backward on the one-hot
    x_t gives approx log p(y | x_t^{l->k}) for every (l, k); the CTMC
    posterior is tilted by gamma * that. Final decode = the model's own
    (unguided) adjacency walk, as in base."""
    origs = torch.as_tensor(origs).long().to(model.device)
    dests = torch.as_tensor(dests).long().to(model.device)
    b = origs.shape[0]
    block = model.block_size
    pfx = model.pfx
    V0 = model.V_states
    max_blocks = model.bd_max_len // block
    stop_idx = torch.full((b,), -1, dtype=torch.long)
    end_idx = torch.full((b,), -1, dtype=torch.long)
    seq = torch.empty(b, 0, dtype=torch.long, device=model.device)
    arange = torch.arange(b, device=model.device)

    for j in range(max_blocks):
        seq = torch.cat([seq, model._init_block(b)], dim=1)
        s = seq.shape[1]
        for p_idx, val in [(0, dests), (1, origs)]:
            if s - block <= p_idx < s:
                seq[:, p_idx] = val
        if s > pfx:
            attn = get_block_causal_mask(s, block, model.device)
            lo = s - block
            clamp_prefix = lo == 0
            x0_probs = None
            for t in range(model.max_T, 0, -1):
                if clamp_prefix:
                    seq[:, 0] = dests
                    seq[:, 1] = origs
                with torch.no_grad():
                    logits = model._cfg_logits(seq, model._t_input(b, s, float(t)),
                                               origs, dests, attn, 1.0)
                    logits_v = logits[:, -block:, :V0]
                    x0_probs = F.softmax(logits_v, dim=-1)
                    if clamp_prefix:
                        x0_probs[:, 0] = 0.0
                        x0_probs[arange, 0, dests] = 1.0
                        x0_probs[:, 1] = 0.0
                        x0_probs[arange, 1, origs] = 1.0
                    xb = seq[:, -block:]
                    EtXt = model.Q[t, :, xb.reshape(-1)].T
                    Em1 = torch.matmul(x0_probs, model.matrices[t - 1])
                    post = EtXt * Em1.reshape(b * block, V0)
                    post = post / post.sum(1, keepdim=True).clamp(min=1e-8)
                    diffusion_log_probs = post.clamp(min=1e-12).log().view(b, block, V0)

                # ---- their first-order approximation (lines 1367-1381) ----
                xt_one_hot = F.one_hot(seq[:, -block:], clf.vocab).to(torch.float)
                with torch.enable_grad():
                    xt_one_hot.requires_grad_(True)
                    lp_xt = clf.get_log_probs(xt_one_hot, torch.full((b,), float(t),
                                                                     device=model.device))
                    lp_xt[..., 1].sum().backward()
                    grad = xt_one_hot.grad
                ratio = (grad - (xt_one_hot * grad).sum(dim=-1, keepdim=True)).detach()
                clf_lp = (ratio + lp_xt[..., 1].detach()[:, None, None])[..., :V0]

                with torch.no_grad():
                    guided = (gamma * clf_lp) + diffusion_log_probs
                    probs = guided.softmax(dim=-1)
                    seq[:, -block:] = torch.multinomial(
                        probs.reshape(b * block, V0).clamp(min=1e-12), 1).view(b, block)

            # final adjacency-constrained walk (model's own, unguided)
            with torch.no_grad():
                if clamp_prefix:
                    seq[:, 0] = dests
                    seq[:, 1] = origs
                    start = pfx
                else:
                    start = 0
                if start < block:
                    prev = seq[:, lo + start - 1]
                    for k in range(start, block):
                        mp = model.A[prev] * x0_probs[:, k]
                        bad = mp.sum(-1) < 1e-6
                        mp[bad] = 1.0
                        mp = model.A[prev] * mp
                        sb = mp.sum(-1) <= 0
                        if bool(sb.any()):
                            mp[sb] = 1.0
                        tok = torch.multinomial(mp, 1).view(-1)
                        seq[:, s - block + k] = tok
                        prev = tok

        lo2 = max(pfx, s - block)
        bt = seq[:, lo2:s].cpu()
        for i in range(b):
            if stop_idx[i] >= 0 or end_idx[i] >= 0:
                continue
            d = int(dests[i].item())
            for k, tok_ in enumerate(bt[i].tolist()):
                if tok_ == d:
                    stop_idx[i] = lo2 + k
                    break
                if tok_ in (model.END, model.PAD, model.MASK):
                    end_idx[i] = lo2 + k
                    break
        if bool(((stop_idx >= 0) | (end_idx >= 0)).all()):
            break

    seq = seq.cpu()
    o = pfx - 1
    paths, hits = [], []
    for i in range(b):
        if stop_idx[i] >= 0:
            paths.append(seq[i, o:stop_idx[i] + 1].tolist())
            hits.append(True)
            continue
        hi = int(end_idx[i].item()) if end_idx[i] >= 0 else seq.shape[1]
        paths.append([v for v in seq[i, o:hi].tolist() if v < model.n_vertex])
        hits.append(False)
    model.last_hits = hits
    return paths


# =====================================================================
# Matched-architecture variants for the controlled 2x2 comparison
# (mechanism x classifier-features), see RESULTS_BD sec. 6.12
# =====================================================================
class DCBGClassifierAdj(DCBGClassifier):
    """DCBGClassifier + the SAME adjacency machinery as BDDiscriminator:
    GraphSAGE-updated vertex embeddings under A_scn, per-transition
    [edge_exists, is_transition, deg_ratio] features, and the bounded logit
    head. Only the input type (noisy x_t + t) differs from our
    discriminator -- that difference is intrinsic to D-CBG."""

    def __init__(self, n_vertex, device, gnn_layers=2, logit_bound=4.0, **kw):
        super().__init__(n_vertex, device, **kw)
        x_dim = self.x_embedding.weight.shape[1]
        self.gnn = nn.ModuleList([_SAGELayer(x_dim) for _ in range(gnn_layers)])
        self.feat_proj = nn.Linear(3, 128)
        self.logit_bound = logit_bound
        self.to(device)

    def forward(self, x, t, adj=None, deg_ratio=None):
        one_hot = x.dtype in (torch.float16, torch.float32, torch.float64)
        tok = x.argmax(-1) if one_hot else x
        E = self.x_embedding.weight
        if adj is not None:
            h_v = E[:self.n_vertex]
            for layer in self.gnn:
                h_v = F.relu(layer(h_v, adj))
            E = torch.cat([E[:self.n_vertex] + h_v, E[self.n_vertex:]], dim=0)
        if one_hot:
            h = torch.matmul(x, E)
        else:
            h = E[tok]
        pad_mask = tok == self.PAD
        b, s = tok.shape
        h = self.in_proj(h) + self.pos_emb(torch.arange(s, device=tok.device))[None]
        h = h + self.time_mlp(t.float())[:, None, :]
        if adj is not None:
            safe = torch.where(tok < self.n_vertex, tok, torch.zeros_like(tok))
            prev = torch.roll(safe, 1, dims=1)
            real = tok < self.n_vertex
            prev_real = torch.roll(real, 1, dims=1)
            pos_i = torch.arange(s, device=tok.device)[None].expand(b, s)
            is_tr = ((pos_i >= 2) & real & prev_real).float()
            e_ex = adj[prev.reshape(-1), safe.reshape(-1)].view(b, s) * is_tr
            dr = (deg_ratio[safe.reshape(-1)].view(b, s) * real.float()
                  if deg_ratio is not None else torch.zeros_like(e_ex))
            h = h + self.feat_proj(torch.stack([e_ex, is_tr, dr], dim=-1))
        for blk in self.blocks:
            h = blk(h, pad_mask)
        h = self.final_norm(h)
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        z = self.head(pooled).squeeze(-1)
        lb = self.logit_bound
        return lb * torch.tanh(z / lb)

    def get_log_probs(self, x, t, adj=None, deg_ratio=None):
        z = self.forward(x, t, adj, deg_ratio)
        return torch.stack([F.logsigmoid(-z), F.logsigmoid(z)], dim=-1)


class AdjBound:
    """Binds (adj, deg_ratio) so plan_dcbg_* can keep calling
    get_log_probs(x, t)."""

    def __init__(self, clf, adj, deg_ratio):
        self.clf, self.adj, self.deg = clf, adj, deg_ratio
        self.vocab = clf.vocab

    def get_log_probs(self, x, t):
        return self.clf.get_log_probs(x, t, self.adj, self.deg)


class BDDiscriminatorPlain(BDDiscriminator):
    """BDDiscriminator with the adjacency machinery DISABLED (no GNN update,
    no transition features) -- the IW-mechanism x plain-classifier cell.
    Keeps the same call signature (adj/deg args accepted and ignored)."""

    def forward(self, tokens, lengths, adj, deg_ratio):
        b, s = tokens.shape
        pad_mask = torch.arange(s, device=tokens.device)[None] >= lengths[:, None]
        tok = torch.where(pad_mask, torch.full_like(tokens, self.PAD), tokens)
        x = self.in_proj(self.x_embedding(tok))
        x = x + self.pos_emb(torch.arange(s, device=tokens.device))[None]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.final_norm(x)
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        z = self.head(pooled).squeeze(-1)
        lb = self.logit_bound
        return lb * torch.tanh(z / lb)


class PlaceboDisc(nn.Module):
    """Cheating-audit control: outputs a constant logit -> uniform importance
    weights. plan_guided with this disc isolates everything EXCEPT the
    discriminator signal (candidate averaging, masking, sampling plumbing)."""

    def __init__(self, n_vertex=1390):
        super().__init__()
        self.PAD = n_vertex
        self.n_vertex = n_vertex
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, lengths, adj, deg_ratio):
        return torch.zeros(tokens.shape[0], device=tokens.device)

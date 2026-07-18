"""
Partial-path discriminator for importance-weight guidance in block diffusion
(BD_GUIDANCE_FORMULATION.pdf).

Estimates D(x) = Pr[exceptional | partial path x, scenario adjacency A_scn],
so that exp(logit) = D/(1-D) approximates the density ratio
q_exc(x_partial) / p_ref(x_partial) where p_ref is defined by the negative
class (normal data, or normal data + model samples).

Design (fixes to the prior Discriminator, per project discussion):
  - input = canvas-form partial path [dst, v0=ori, v1, ..., v_j] (variable j),
    matching the guidance-time input [committed prefix || candidate completion]
  - dedicated PAD token (= n_vertex) + masked mean pooling: a path receives the
    same logit regardless of batch composition (no vertex-0 padding pollution)
  - edge-existence channel: for every consecutive transition (v_{i-1}, v_i),
    A_scn[v_{i-1}, v_i] in {0,1} is injected as a per-position feature -- the
    removed-edge signal is exposed directly and generalizes to unseen
    scenarios (experiment 2), plus a degree-ratio channel deg_scn/deg_norm
  - scenario-conditioned vertex embeddings via a small GraphSAGE over A_scn
  - bounded logit head lam * tanh(z / lam): |logit| <= lam, so the importance
    weight is capped at e^lam by construction
  - BCE with label smoothing (saturation control)

torch 1.12 compatible (manual attention).
"""

import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SAGELayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(2 * dim, dim)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        deg = adj.sum(1, keepdim=True).clamp(min=1.0)
        neigh = (adj / deg) @ x
        return self.lin(torch.cat([x, neigh], dim=-1))


class _EncBlock(nn.Module):
    """Pre-LN transformer encoder block, manual attention with key-padding mask."""

    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        # pad_mask: (b, s) True = PAD (do not attend to)
        b, s, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(pad_mask[:, None, None, :], -1e9)
        att = self.drop(scores.softmax(dim=-1))
        h = torch.matmul(att, v).transpose(1, 2).reshape(b, s, d)
        x = x + self.out(h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class BDDiscriminator(nn.Module):
    def __init__(self, n_vertex, device, dim=128, n_layers=2, n_heads=4,
                 max_len=128, gnn_layers=2, logit_bound=4.0, dropout=0.1,
                 pretrain_path=None):
        super().__init__()
        self.n_vertex = n_vertex
        self.PAD = n_vertex          # dedicated pad token
        self.device = device
        self.logit_bound = logit_bound

        # vertex embedding (node2vec init when available)
        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert n_vertex == len(node2vec)
            x_dim = node2vec[0].shape[0]
            emb = torch.zeros(n_vertex + 1, x_dim)
            for k in node2vec:
                emb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=self.PAD)
        else:
            x_dim = 100
            self.x_embedding = nn.Embedding(n_vertex + 1, x_dim, padding_idx=self.PAD)
        self.x_dim = x_dim

        self.gnn = nn.ModuleList([_SAGELayer(x_dim) for _ in range(gnn_layers)])
        self.in_proj = nn.Linear(x_dim, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        # per-position transition features: [edge_exists, is_transition, deg_ratio]
        self.feat_proj = nn.Linear(3, dim)
        self.blocks = nn.ModuleList([_EncBlock(dim, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.to(device)

    def _scenario_embedding(self, adj):
        """Base embedding table augmented by GraphSAGE over the scenario graph."""
        E = self.x_embedding.weight                       # (V+1, x_dim)
        h = E[:-1]                                        # real vertices only
        for layer in self.gnn:
            h = F.relu(layer(h, adj))
        E_scn = torch.cat([E[:-1] + h, E[-1:]], dim=0)    # residual; PAD row untouched
        return E_scn

    def forward(self, tokens, lengths, adj, deg_ratio):
        """
        tokens    : (b, s) long, canvas-form partial paths [dst, ori, v1, ...],
                    PAD (= n_vertex) after each row's length.
        lengths   : (b,) long, valid length per row.
        adj       : (V, V) float 0/1 scenario adjacency.
        deg_ratio : (V,) float, deg_scn / deg_normal per vertex.
        returns   : (b,) bounded logits.
        """
        b, s = tokens.shape
        pad_mask = torch.arange(s, device=tokens.device)[None] >= lengths[:, None]
        tok = torch.where(pad_mask, torch.full_like(tokens, self.PAD), tokens)

        E_scn = self._scenario_embedding(adj)
        x = self.in_proj(E_scn[tok])
        x = x + self.pos_emb(torch.arange(s, device=tokens.device))[None]

        # transition features: position i>=2 describes edge (tok[i-1] -> tok[i]);
        # positions 0 (dst) and 1 (ori) are not traversals.
        safe = torch.where(pad_mask, torch.zeros_like(tok), tok)
        prev = torch.roll(safe, 1, dims=1)
        edge_exists = adj[prev.reshape(-1), safe.reshape(-1)].view(b, s)
        pos_idx = torch.arange(s, device=tokens.device)[None].expand(b, s)
        is_trans = ((pos_idx >= 2) & ~pad_mask).float()
        edge_exists = edge_exists * is_trans
        dratio = deg_ratio[safe.reshape(-1)].view(b, s) * (~pad_mask).float()
        feats = torch.stack([edge_exists, is_trans, dratio], dim=-1)
        x = x + self.feat_proj(feats)

        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.final_norm(x)

        # masked mean pooling
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        z = self.head(pooled).squeeze(-1)
        lb = self.logit_bound
        return lb * torch.tanh(z / lb)


# =====================================================================
# Training-example construction
# =====================================================================
def make_partial(path, rng, min_len=2):
    """path: list [v0=ori, ..., v_{L-1}=dst] -> canvas-form random prefix
    [dst, v0, ..., v_j], j ~ U{min_len-1, ..., L-1} (may be the full path)."""
    L = len(path)
    j = int(rng.integers(min_len, L + 1)) if L > min_len else L
    return [path[-1]] + list(path[:j])


def pad_batch(seqs, pad_id, device, max_len=None):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long, device=device)
    m = int(lengths.max().item()) if max_len is None else max_len
    out = torch.full((len(seqs), m), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s[:m], dtype=torch.long, device=device)
    return out, lengths.clamp(max=m)

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange
from einops.layers.torch import Rearrange
import pickle

from models_seq.blocks import (
    SinusoidalPosEmb, 
    Conv1dBlock, 
    Residual, 
    LinearAttention, 
)


class XTResBlock(nn.Module):

    def __init__(self, x_in_dim, t_in_dim, out_dim, device, kernel_size=5):
        super().__init__()
        self.device = device
        self.block1 = Conv1dBlock(x_in_dim, out_dim, kernel_size).to(device)
        self.block2 = Conv1dBlock(out_dim, out_dim, kernel_size).to(device)

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(t_in_dim, out_dim, device=device),
            Rearrange("b c -> b 1 c") 
        )
        self.residual_conv = nn.Sequential(
            Rearrange("b h c -> b c h"), 
            nn.Conv1d(x_in_dim, out_dim, 1, device=device) if x_in_dim != out_dim else nn.Identity(), 
            Rearrange("b c h -> b h c")
        )

    def forward(self, x, t):
        '''
            x : b h c
            t : b h d
            returns:
            out : b h e
        '''
        out = self.block1(x) + self.time_mlp(t)
        out = self.block2(out)
        return out + self.residual_conv(x)
    
    
class UnetBlock(nn.Module):
    def __init__(self, x_dim, time_dim, out_dim, device, down_up, last):
        super().__init__()
        self.device = device
        self.xtblock1 = XTResBlock(x_dim, time_dim, out_dim, device, kernel_size=5)
        self.xtblock2 = XTResBlock(out_dim, time_dim, out_dim, device, kernel_size=5)
        self.attn = Residual(LinearAttention(out_dim, device))
        self.down = down_up == "down"
        self.sample = nn.Identity()
        
    def forward(self, xs, lengths, ts):
        x = self.xtblock1(xs, ts)
        x = self.xtblock2(x, ts)
        h = self.attn(x, lengths)
        x = self.sample(h)
        return x, h


class EPSM(nn.Module):
    
    def __init__(self, n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=None):
        super().__init__()
        time_dim = hidden_dim
        self.device = device
        # temporal embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim, device), 
            nn.Linear(time_dim, 4 * time_dim, device=device), 
            nn.Mish(), 
            nn.Linear(4 * time_dim, time_dim, device=device)
        )
        # n_vertex denotes <end>,  n_vertex + 1 denotes <padding>
        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert n_vertex == len(node2vec)
            if x_emb_dim != node2vec[0].shape[0]:
                print("Use pretrained embed dims")
            x_emb_dim = node2vec[0].shape[0]
            nodeemb = torch.zeros(n_vertex + 2, x_emb_dim)
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False).to(device)
        else:
            self.x_embedding = nn.Embedding(n_vertex + 2, x_emb_dim, padding_idx=n_vertex, device=device)
        
        in_out_dim = [(a, b) for a, b in zip(dims, dims[1:])]
        print(in_out_dim)
        # down blocks
        self.down_blocks = []
        n_reso = len(in_out_dim)
        for k, (in_dim, out_dim) in enumerate(in_out_dim):
            self.down_blocks.append(UnetBlock(
                in_dim, time_dim, out_dim, device, 
                down_up="down", last=(k == n_reso - 1)))
        
        # middle parts
        mid_dim = dims[-1]
        self.mid_block1 = XTResBlock(mid_dim, time_dim, mid_dim, device)
        self.mid_attn = Residual(LinearAttention(mid_dim, device))
        self.mid_block2 = XTResBlock(mid_dim, time_dim, mid_dim, device)

        # up blocks
        self.up_blocks = []
        for k, (out_dim, in_dim) in enumerate(reversed(in_out_dim[1:])):
            self.up_blocks.append(UnetBlock(
                in_dim * 2, time_dim, out_dim, device, 
                down_up="up", last=(k == n_reso - 1)))
        
        # final parts
        self.final_conv = nn.Sequential(
            Conv1dBlock(in_out_dim[1][0], dims[0], kernel_size=5),
            Rearrange("b h c -> b c h"), 
            nn.Conv1d(dims[0], n_vertex, 1, device=device),
            Rearrange("b c h -> b h c")
        ).to(device)
        
        
    def forward(self, xt_padded, lengths, t):
        # xt_padded: shape b, h, each is a xt label
        # t: shape b
        t = self.time_mlp(t)
        x = self.x_embedding(xt_padded)
        hiddens = []
        for k, down_block in enumerate(self.down_blocks):
            x, h = down_block(x, lengths if k == 0 else None, t)
            hiddens.append(h)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, None)
        x = self.mid_block2(x, t)
        
        for up_block in self.up_blocks:
            x = torch.cat((x, hiddens.pop()), dim=-1)
            x, _ = up_block(x, None, t)
        x = self.final_conv(x)
        return x

class EPSM_OD(EPSM):
    """
    O/D-conditional EPSM.

    Global conditioning (same pattern as EPSM_SimTime): origin/destination
    node embeddings pass through an MLP and are added to the time embedding.
    The null condition token (index n_vertex) is used for condition dropout
    during training, which also enables unconditional generation.
    """

    def __init__(self, n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=None):
        super().__init__(n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=pretrain_path)
        # Register UNet blocks as submodules so the optimizer trains them.
        # Base EPSM keeps them in plain lists (unregistered = frozen at init);
        # left untouched there to preserve baseline behavior.
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        time_dim = hidden_dim
        self.null_idx = n_vertex
        emb_dim = self.x_embedding.weight.shape[1]  # x_emb_dim may be overridden by pretrained node2vec
        self.od_mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, 4 * time_dim, device=device),
            nn.Mish(),
            nn.Linear(4 * time_dim, time_dim, device=device),
        )

    def forward(self, xt_padded, lengths, t, od=None):
        # xt_padded: shape b, h, each is a xt label
        # t: shape b
        # od: shape b, 2 = (origin, destination) node labels, null token = n_vertex
        t = self.time_mlp(t)

        if od is None:
            od = torch.full((xt_padded.shape[0], 2), self.null_idx,
                            dtype=torch.long, device=xt_padded.device)
        od_emb = rearrange(self.x_embedding(od), "b n e -> b (n e)")
        t = t + self.od_mlp(od_emb)

        x = self.x_embedding(xt_padded)
        hiddens = []
        for k, down_block in enumerate(self.down_blocks):
            x, h = down_block(x, lengths if k == 0 else None, t)
            hiddens.append(h)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, None)
        x = self.mid_block2(x, t)

        for up_block in self.up_blocks:
            x = torch.cat((x, hiddens.pop()), dim=-1)
            x, _ = up_block(x, None, t)
        x = self.final_conv(x)
        return x


class EPSM_OD_EOS(EPSM_OD):
    """
    EPSM_OD with the <end> state in the output vocabulary (LLaDA-style
    full-canvas length handling): the head predicts n_vertex + 1 states
    (real vertices + <end> = index n_vertex), so the model learns where
    paths terminate and no external length model is needed.

    The null condition token moves to n_vertex + 1 (<pad>) because
    n_vertex now denotes a real diffusion state.
    """

    def __init__(self, n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=None):
        super().__init__(n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=pretrain_path)
        self.null_idx = n_vertex + 1
        in_out_dim = [(a, b) for a, b in zip(dims, dims[1:])]
        self.final_conv = nn.Sequential(
            Conv1dBlock(in_out_dim[1][0], dims[0], kernel_size=5),
            Rearrange("b h c -> b c h"),
            nn.Conv1d(dims[0], n_vertex + 1, 1, device=device),
            Rearrange("b c h -> b h c")
        ).to(device)


class EPSM_SimTime(nn.Module):

    def __init__(self, n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=None):
        super().__init__()
        time_dim = hidden_dim
        self.device = device
        # temporal embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim, device),
            nn.Linear(time_dim, 4 * time_dim, device=device),
            nn.Mish(),
            nn.Linear(4 * time_dim, time_dim, device=device)
        )

        self.sim_time_mlp = nn.Sequential(
            nn.Linear(1, time_dim, device=device),
            nn.Mish(),
            nn.Linear(time_dim, 4 * time_dim, device=device),
            nn.Mish(),
            nn.Linear(4 * time_dim, time_dim, device=device)
        )
        # n_vertex denotes <end>,  n_vertex + 1 denotes <padding>
        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert n_vertex == len(node2vec)
            if x_emb_dim != node2vec[0].shape[0]:
                print("Use pretrained embed dims")
            x_emb_dim = node2vec[0].shape[0]
            nodeemb = torch.zeros(n_vertex + 2, x_emb_dim)
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False).to(device)
        else:
            self.x_embedding = nn.Embedding(n_vertex + 2, x_emb_dim, padding_idx=n_vertex, device=device)

        in_out_dim = [(a, b) for a, b in zip(dims, dims[1:])]
        print(in_out_dim)
        # down blocks
        self.down_blocks = []
        n_reso = len(in_out_dim)
        for k, (in_dim, out_dim) in enumerate(in_out_dim):
            self.down_blocks.append(UnetBlock(
                in_dim, time_dim, out_dim, device,
                down_up="down", last=(k == n_reso - 1)))

        # middle parts
        mid_dim = dims[-1]
        self.mid_block1 = XTResBlock(mid_dim, time_dim, mid_dim, device)
        self.mid_attn = Residual(LinearAttention(mid_dim, device))
        self.mid_block2 = XTResBlock(mid_dim, time_dim, mid_dim, device)

        # up blocks
        self.up_blocks = []
        for k, (out_dim, in_dim) in enumerate(reversed(in_out_dim[1:])):
            self.up_blocks.append(UnetBlock(
                in_dim * 2, time_dim, out_dim, device,
                down_up="up", last=(k == n_reso - 1)))

        # final parts
        self.final_conv = nn.Sequential(
            Conv1dBlock(in_out_dim[1][0], dims[0], kernel_size=5),
            Rearrange("b h c -> b c h"),
            nn.Conv1d(dims[0], n_vertex, 1, device=device),
            Rearrange("b c h -> b h c")
        ).to(device)


    def forward(self, xt_padded, lengths, t, sim_time):
        # xt_padded: shape b, h, each is a xt label
        # t: shape b
        # sim_time: shape b
        t = self.time_mlp(t)
        sim_time = self.sim_time_mlp(sim_time)
        t = t + sim_time
        x = self.x_embedding(xt_padded)
        hiddens = []
        for k, down_block in enumerate(self.down_blocks):
            x, h = down_block(x, lengths if k == 0 else None, t)
            hiddens.append(h)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, None)
        x = self.mid_block2(x, t)

        for up_block in self.up_blocks:
            x = torch.cat((x, hiddens.pop()), dim=-1)
            x, _ = up_block(x, None, t)
        x = self.final_conv(x)
        return x
"""Multi-organ, multi-task shape network.

A shared PointNet encoder embeds each organ's (size-normalized) point cloud; a
masked attention pool fuses the present organs into one subject embedding; three
linear heads predict sex, age, and pathology. Per-task label masks make it
trivial to mix datasets with different available labels.
"""
from __future__ import annotations
import torch
import torch.nn as nn

TASKS = ("sex", "age", "pathology")


class PointNetEncoder(nn.Module):
    """Per-point Conv1d + GroupNorm (per-sample, robust to zero-filled organs) +
    symmetric max-pool -> permutation-invariant organ embedding."""
    def __init__(self, emb=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.GroupNorm(8, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(256, emb), nn.ReLU())

    def forward(self, x):                      # x: [M, N, 3]
        h = self.conv(x.transpose(1, 2))       # [M, 256, N]
        h = h.max(dim=2).values                # [M, 256] (permutation-invariant)
        return self.head(h)                    # [M, emb]


def _knn_idx(x, k):                              # x: [M, N, 3] -> [M, N, k]
    with torch.no_grad():
        inner = -2 * torch.bmm(x, x.transpose(1, 2))
        xx = (x ** 2).sum(-1, keepdim=True)
        dist = -(xx + inner + xx.transpose(1, 2))
        return dist.topk(k, dim=-1).indices


def _edge_feature(feat, idx):                    # feat: [M, N, C], idx: [M, N, k]
    M, N, C = feat.shape
    k = idx.shape[-1]
    base = (torch.arange(M, device=feat.device).view(-1, 1, 1) * N)
    flat = feat.reshape(M * N, C)
    nb = flat[(idx + base).reshape(-1)].view(M, N, k, C)
    center = feat.unsqueeze(2).expand(-1, -1, k, -1)
    return torch.cat([center, nb - center], dim=-1)  # [M, N, k, 2C]


class DGCNNEncoder(nn.Module):
    """Two EdgeConv layers (static k-NN graph on input coords) + global max-pool."""
    def __init__(self, emb=128, k=20):
        super().__init__()
        self.k = k
        self.ec1 = nn.Sequential(nn.Conv2d(6, 64, 1), nn.GroupNorm(8, 64), nn.ReLU())
        self.ec2 = nn.Sequential(nn.Conv2d(128, 128, 1), nn.GroupNorm(8, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(192, 256), nn.ReLU(), nn.Linear(256, emb), nn.ReLU())

    def forward(self, x):                         # x: [M, N, 3]
        idx = _knn_idx(x, self.k)
        h1 = self.ec1(_edge_feature(x, idx).permute(0, 3, 1, 2)).max(dim=-1).values  # [M,64,N]
        h1 = h1.transpose(1, 2)                                                       # [M,N,64]
        h2 = self.ec2(_edge_feature(h1, idx).permute(0, 3, 1, 2)).max(dim=-1).values  # [M,128,N]
        g = torch.cat([h1.max(dim=1).values, h2.max(dim=2).values], dim=-1)           # [M,192]
        return self.head(g)


class MultiOrganMultiTask(nn.Module):
    def __init__(self, n_organs, emb=128, use_size=True, encoder="pointnet"):
        super().__init__()
        self.n_organs, self.emb, self.use_size = n_organs, emb, use_size
        self.encoder = DGCNNEncoder(emb) if encoder == "dgcnn" else PointNetEncoder(emb)
        in_dim = emb + (1 if use_size else 0)
        self.organ_proj = nn.Linear(in_dim, emb)
        self.attn = nn.Linear(emb, 1)          # masked attention over organs
        self.heads = nn.ModuleDict({t: nn.Linear(emb, 1) for t in TASKS})

    def forward(self, points, size, organ_mask):
        B, O, N, _ = points.shape
        e = self.encoder(points.reshape(B * O, N, 3)).reshape(B, O, self.emb)
        if self.use_size:
            e = torch.cat([e, size.unsqueeze(-1)], dim=-1)
        e = torch.relu(self.organ_proj(e))     # [B, O, emb]
        a = self.attn(e).squeeze(-1)           # [B, O]
        a = a.masked_fill(organ_mask < 0.5, float("-inf"))
        a = torch.softmax(a, dim=1)
        a = torch.nan_to_num(a)                # subjects with 0 organs -> 0 weights
        z = (e * a.unsqueeze(-1)).sum(dim=1)   # [B, emb]
        return {t: self.heads[t](z).squeeze(-1) for t in TASKS}

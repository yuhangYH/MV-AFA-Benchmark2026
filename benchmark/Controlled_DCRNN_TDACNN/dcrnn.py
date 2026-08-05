"""
dcrnn.py
========
Dynamic-graph baseline: DCRNN (Diffusion-Convolutional Recurrent Neural Network)
after Tang et al., "Self-Supervised Graph Neural Networks for Improved
Electroencephalographic Seizure Analysis", ICLR 2022 (arXiv:2104.08336).

This is the *supervised detection* variant (no self-supervised pre-training),
reimplemented to run under the MV-AFA unified protocol so the comparison
isolates the architecture, not the preprocessing.

Faithful design choices kept from the paper:
  * electrodes = graph nodes; a DYNAMIC functional-connectivity graph is built
    per window from inter-channel correlation (Tang's "individual"/correlation
    graph), then sparsified with top-k neighbours.
  * node features per time-step = log-magnitude FFT of that channel's frame
    (Tang uses per-second FFT features).
  * temporal modelling via stacked DCGRU cells (diffusion convolution inside a
    GRU), then last hidden state -> node mean-pool -> 2-class head.

Adapted for the 2 s MV-AFA window: the window is split into `n_frames` equal
frames (default 4 -> 0.5 s each); per-frame per-channel FFT features form the
input sequence to the DCGRU.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Graph construction (dynamic, per batch of windows)
# ----------------------------------------------------------------------------
def correlation_supports(x_bct: torch.Tensor, top_k: int = 3,
                         max_diffusion_step: int = 2):
    """Build random-walk diffusion supports from per-window channel correlation.

    x_bct : [B, C, T] raw window (used only to estimate functional connectivity)
    returns list of support matrices, each [B, C, C], for both walk directions,
    powered up to `max_diffusion_step`.
    """
    B, C, T = x_bct.shape
    x = x_bct - x_bct.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + 1e-6
    xn = x / std
    corr = torch.matmul(xn, xn.transpose(1, 2)) / T        # [B, C, C] Pearson
    corr = corr.abs()
    eye = torch.eye(C, device=x.device).unsqueeze(0)
    corr = corr * (1 - eye)                                # drop self-loops first
    # top-k sparsification per node
    if top_k < C - 1:
        kth = torch.topk(corr, k=top_k, dim=2).values[:, :, -1:].detach()
        corr = torch.where(corr >= kth, corr, torch.zeros_like(corr))
    adj = corr + eye                                       # re-add self loops
    # random-walk normalization  P = D^-1 A  and its transpose
    deg = adj.sum(dim=2, keepdim=True) + 1e-6
    P = adj / deg
    Pt = P.transpose(1, 2)
    supports = []
    for base in (P, Pt):
        Tk = torch.eye(C, device=x.device).unsqueeze(0).expand(B, C, C)
        cur = base
        for _ in range(max_diffusion_step):
            supports.append(cur)
            cur = torch.matmul(base, cur)
    return supports                                        # 2 * K matrices


# ----------------------------------------------------------------------------
# Diffusion graph convolution
# ----------------------------------------------------------------------------
class DiffusionGraphConv(nn.Module):
    """Y = sum_s support_s @ X @ W_s  (+ bias), concatenated over supports."""

    def __init__(self, num_supports: int, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        # +1 for the identity (k=0) term
        self.weight = nn.Parameter(torch.empty((num_supports + 1) * in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, supports, x):
        # x: [B, C, in_dim]
        B, C, _ = x.shape
        out = [x]
        for S in supports:
            out.append(torch.matmul(S, x))                 # [B, C, in_dim]
        h = torch.cat(out, dim=2)                          # [B, C, (S+1)*in_dim]
        h = torch.matmul(h, self.weight) + self.bias       # [B, C, out_dim]
        return h


class DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell operating on graph node states."""

    def __init__(self, num_supports: int, in_dim: int, hid_dim: int):
        super().__init__()
        self.hid_dim = hid_dim
        self.gate = DiffusionGraphConv(num_supports, in_dim + hid_dim, 2 * hid_dim)
        self.cand = DiffusionGraphConv(num_supports, in_dim + hid_dim, hid_dim)

    def forward(self, supports, x, h):
        # x: [B, C, in_dim]   h: [B, C, hid_dim]
        xh = torch.cat([x, h], dim=2)
        ru = torch.sigmoid(self.gate(supports, xh))
        r, u = torch.split(ru, self.hid_dim, dim=2)
        xrh = torch.cat([x, r * h], dim=2)
        c = torch.tanh(self.cand(supports, xrh))
        return u * h + (1 - u) * c


# ----------------------------------------------------------------------------
# Per-frame FFT node features
# ----------------------------------------------------------------------------
def fft_node_features(x_bct: torch.Tensor, n_frames: int = 4):
    """[B, C, T] -> sequence of [B, C, F] log-FFT magnitudes, one per frame."""
    B, C, T = x_bct.shape
    L = T // n_frames
    feats = []
    for f in range(n_frames):
        seg = x_bct[:, :, f * L:(f + 1) * L]               # [B, C, L]
        mag = torch.fft.rfft(seg, dim=2).abs()             # [B, C, L//2+1]
        feats.append(torch.log1p(mag))
    return feats, feats[0].shape[-1]                       # list[T_steps], feat_dim


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------
class DCRNN(nn.Module):
    def __init__(self, n_channels: int = 18, n_frames: int = 4,
                 hid_dim: int = 64, n_layers: int = 2,
                 max_diffusion_step: int = 2, top_k: int = 3,
                 feat_dim: int | None = None):
        super().__init__()
        self.n_frames = n_frames
        self.max_diffusion_step = max_diffusion_step
        self.top_k = top_k
        self.hid_dim = hid_dim
        num_supports = 2 * max_diffusion_step
        # feat_dim is the rfft size of one frame; resolved lazily if None
        self._feat_dim = feat_dim
        self._num_supports = num_supports
        self.cells = None
        self.head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hid_dim, 2))
        self._n_layers = n_layers

    def _build(self, feat_dim, device):
        cells = []
        in_dim = feat_dim
        for _ in range(self._n_layers):
            cells.append(DCGRUCell(self._num_supports, in_dim, self.hid_dim))
            in_dim = self.hid_dim
        self.cells = nn.ModuleList(cells).to(device)
        self._feat_dim = feat_dim

    def forward(self, x_bct):
        supports = correlation_supports(x_bct, self.top_k, self.max_diffusion_step)
        feats, feat_dim = fft_node_features(x_bct, self.n_frames)
        if self.cells is None:
            self._build(feat_dim, x_bct.device)
        B, C, _ = x_bct.shape
        h = [torch.zeros(B, C, self.hid_dim, device=x_bct.device)
             for _ in self.cells]
        for x_t in feats:                                  # over time frames
            inp = x_t
            for li, cell in enumerate(self.cells):
                h[li] = cell(supports, inp, h[li])
                inp = h[li]
        z = h[-1].mean(dim=1)                              # node mean-pool [B, hid]
        return self.head(z)

"""
tda_cnn.py
==========
Topological baseline: Vietoris-Rips persistence -> CNN, after Wang et al.,
"Automatic epileptic seizure detection based on persistent homology",
Frontiers in Physiology 2023 (PMC10773586).

In that work multi-channel EEG is summarized by Vietoris-Rips barcodes and the
topological representation is classified by a CNN. We reimplement it under the
MV-AFA unified protocol. To feed barcodes to a 2-D CNN we rasterize the H0 and
H1 persistence diagrams into fixed-size PERSISTENCE IMAGES (two input channels),
which is the standard vectorization for CNN-on-persistence pipelines.

KEY POINT FOR THE PAPER:
  This baseline uses a *single* topological representation in isolation -- it is
  the control that isolates the benefit of MV-AFA's multi-view fusion over
  TDA-alone. It deliberately does NOT use the folded-point-cloud + 11-other
  features; it classifies the persistence image directly.

Dependencies: `ripser` (pip install ripser). If unavailable, a NumPy fallback
distance-matrix H0 computation is used (H1 is then empty) so the file still
imports, but install ripser for a faithful H0+H1 baseline.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

try:
    from ripser import ripser
    _HAS_RIPSER = True
except Exception:                                          # pragma: no cover
    _HAS_RIPSER = False


# ----------------------------------------------------------------------------
# Persistence computation
# ----------------------------------------------------------------------------
def _channel_distance_matrix(window: np.ndarray) -> np.ndarray:
    """[C, T] -> [C, C] Euclidean distance between (z-scored) channel series.

    Mirrors VR persistence on the multi-channel metric space: channels are the
    points, distance encodes dis-similarity of their waveforms in the window.
    """
    x = window - window.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-6)
    # pairwise Euclidean distance
    sq = np.sum(x ** 2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * (x @ x.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float64)


def vr_diagrams(window: np.ndarray, maxdim: int = 1):
    """Return [dgm0, dgm1] Vietoris-Rips persistence diagrams for one window."""
    D = _channel_distance_matrix(window)
    if _HAS_RIPSER:
        res = ripser(D, distance_matrix=True, maxdim=maxdim)
        return res["dgms"]
    # ---- fallback: H0 from single-linkage MST merge distances; H1 empty ----
    C = D.shape[0]
    import heapq
    # Prim's MST -> H0 death times (births all 0)
    visited = [False] * C
    heap = [(0.0, 0)]
    deaths = []
    while heap:
        w, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        if u != 0:
            deaths.append(w)
        for v in range(C):
            if not visited[v]:
                heapq.heappush(heap, (D[u, v], v))
    dgm0 = np.array([[0.0, d] for d in deaths] + [[0.0, np.inf]])
    dgm1 = np.empty((0, 2))
    return [dgm0, dgm1]


# ----------------------------------------------------------------------------
# Persistence image rasterization
# ----------------------------------------------------------------------------
def persistence_image(dgm: np.ndarray, res: int = 32, sigma: float = 0.1,
                      birth_max: float = None, pers_max: float = None):
    """Rasterize one diagram to a [res, res] image in (birth, persistence) space.

    Weighting = persistence (linear ramp), kernel = isotropic Gaussian.
    """
    img = np.zeros((res, res), dtype=np.float32)
    if dgm is None or len(dgm) == 0:
        return img
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return img
    births = finite[:, 0]
    pers = finite[:, 1] - finite[:, 0]
    bmax = birth_max if birth_max else max(births.max(), 1e-3)
    pmax = pers_max if pers_max else max(pers.max(), 1e-3)
    bx = np.linspace(0, bmax, res)
    py = np.linspace(0, pmax, res)
    BX, PY = np.meshgrid(bx, py)
    for b, p in zip(births, pers):
        w = p                                              # persistence weight
        img += w * np.exp(-(((BX - b) ** 2 + (PY - p) ** 2) / (2 * sigma ** 2)))
    m = img.max()
    if m > 0:
        img /= m
    return img


def windows_to_images(X: np.ndarray, res: int = 32, sigma: float = 0.1):
    """[N, C, T] -> [N, 2, res, res] persistence images (channels = H0, H1).

    NOTE: VR persistence is the expensive offline step. Per the project's
    standing guidance, parallelize this across CPU cores (multiprocessing) and
    CACHE the result; ripser does not benefit from GPU/MPS.
    """
    out = np.zeros((len(X), 2, res, res), dtype=np.float32)
    for i in range(len(X)):
        dgms = vr_diagrams(X[i], maxdim=1)
        out[i, 0] = persistence_image(dgms[0], res, sigma)
        out[i, 1] = persistence_image(dgms[1] if len(dgms) > 1 else None, res, sigma)
    return out


def windows_to_images_parallel(X: np.ndarray, res: int = 32, sigma: float = 0.1,
                               n_jobs: int = None):
    """Multiprocessing wrapper -- use this for the full CHB-MIT cache."""
    import multiprocessing as mp
    n_jobs = n_jobs or mp.cpu_count()
    chunks = np.array_split(np.arange(len(X)), n_jobs)
    args = [(X[c], res, sigma) for c in chunks if len(c)]
    with mp.Pool(n_jobs) as pool:
        parts = pool.starmap(windows_to_images, args)
    return np.concatenate(parts, axis=0)


# ----------------------------------------------------------------------------
# CNN classifier on persistence images
# ----------------------------------------------------------------------------
class TDACNN(nn.Module):
    def __init__(self, res: int = 32, in_ch: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, img):
        return self.head(self.features(img))

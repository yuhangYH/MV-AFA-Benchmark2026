"""
protocol.py
===========
Shared evaluation protocol for the MV-AFA controlled baselines.

This file reproduces the *unified protocol* described in the MV-AFA paper so that
the DCRNN and TDA+CNN baselines are evaluated under IDENTICAL conditions to
MV-AFA:

    * 2 s windows / 4 s step, 18 bipolar channels (already done upstream -> cache)
    * subject-disjoint split  55% train / 15% val / 30% test
    * majority-class downsampling applied ONLY inside the training pool
    * decision threshold selected on the validation pool
    * metrics: Acc, Sens, Spec, F1, AUROC, AUPRC, event detection rate, mean delay
      (+ false alarms / hour, which the paper should add for natural-prevalence runs)

------------------------------------------------------------------------------
INTEGRATION NOTE
------------------------------------------------------------------------------
Three functions below are marked `# >>> SWAP <<<`. They are faithful
re-implementations of the protocol as written in the paper, but if your real
benchmark harness (BioCAS2026-Benchmark) already defines them, replace these
with imports from your harness so every baseline shares byte-identical logic:

    from your_harness import subject_disjoint_split, select_threshold, compute_metrics

Everything else (data container, event metrics) can stay as-is.
"""
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------------------------------------------------------
# Windowed-cache container
# ----------------------------------------------------------------------------
@dataclass
class WindowCache:
    """A precomputed windowed dataset (the same cache MV-AFA consumes).

    X        : float32 [N, C, T]   e.g. CHB-MIT -> [N, 18, 512] at 256 Hz, 2 s
    y        : int     [N]         0 = non-seizure, 1 = seizure
    subject  : object  [N]         subject id per window (for subject-disjoint split)
    rec_id   : object  [N]         recording id per window (for event grouping)
    t_start  : float32 [N]         window start time (s) within its recording
    win_sec  : float               window length in seconds (for FA/h, delay)
    step_sec : float               window step in seconds
    """
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    rec_id: np.ndarray
    t_start: np.ndarray
    win_sec: float = 2.0
    step_sec: float = 4.0

    @classmethod
    def load_npz(cls, path: str) -> "WindowCache":
        d = np.load(path, allow_pickle=True)
        return cls(
            X=d["X"].astype(np.float32),
            y=d["y"].astype(np.int64),
            subject=d["subject"],
            rec_id=d["rec_id"],
            t_start=d["t_start"].astype(np.float32),
            win_sec=float(d["win_sec"]) if "win_sec" in d else 2.0,
            step_sec=float(d["step_sec"]) if "step_sec" in d else 4.0,
        )

    def index(self, idx) -> "WindowCache":
        return WindowCache(self.X[idx], self.y[idx], self.subject[idx],
                           self.rec_id[idx], self.t_start[idx],
                           self.win_sec, self.step_sec)


# ----------------------------------------------------------------------------
# Subject-disjoint split                                            # >>> SWAP <<<
# ----------------------------------------------------------------------------
def subject_disjoint_split(cache: WindowCache, seed: int = 42,
                           frac=(0.55, 0.15, 0.30)):
    """Assign whole subjects to train/val/test (no subject in two pools)."""
    subs = np.array(sorted(set(cache.subject.tolist())))
    rng = np.random.RandomState(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    tr, va, te = subs[:n_tr], subs[n_tr:n_tr + n_va], subs[n_tr + n_va:]
    m = lambda s: np.isin(cache.subject, s)
    return cache.index(m(tr)), cache.index(m(va)), cache.index(m(te))


def loso_folds(cache: WindowCache):
    """Yield (train_val_cache, test_cache, test_subject) for each held-out subject."""
    subs = np.array(sorted(set(cache.subject.tolist())))
    for s in subs:
        te = cache.index(cache.subject == s)
        tr = cache.index(cache.subject != s)
        yield tr, te, s


def balance_training(cache: WindowCache, seed: int = 42,
                     neg_pos_ratio: float = 1.0) -> WindowCache:
    """Majority-class downsampling. TRAIN POOL ONLY. Never call on val/test.

    neg_pos_ratio = 1.0 -> fully balanced (paper main setting).
    Set 5.0 to reproduce the `neg-pos-ratio 5.0` LOSO optimization runs.
    """
    rng = np.random.RandomState(seed)
    pos = np.where(cache.y == 1)[0]
    neg = np.where(cache.y == 0)[0]
    keep_neg = int(min(len(neg), round(neg_pos_ratio * len(pos))))
    neg = rng.permutation(neg)[:keep_neg]
    idx = rng.permutation(np.concatenate([pos, neg]))
    return cache.index(idx)


# ----------------------------------------------------------------------------
# Threshold selection on the validation pool                       # >>> SWAP <<<
# ----------------------------------------------------------------------------
def select_threshold(y_val: np.ndarray, p_val: np.ndarray,
                     strategy: str = "sens_floor_spec",
                     min_sensitivity: float = 0.70) -> float:
    """Pick the decision threshold on VALIDATION outputs only.

    Strategies mirror the ones used in the LOSO optimization runs:
      'f1'             : threshold maximizing validation F1
      'sens_floor_spec': among thresholds with Sens >= min_sensitivity,
                         pick the one maximizing Spec (the run that pushed
                         CHB-MIT Acc/AUROC > 80).
      'youden'         : maximize Sens + Spec - 1.
    """
    ths = np.unique(np.concatenate([[0.0], np.sort(p_val), [1.0]]))
    best_t, best_obj = 0.5, -1.0
    for t in ths:
        pred = (p_val >= t).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1 = 2 * prec * sens / (prec + sens + 1e-9)
        if strategy == "f1":
            obj = f1
        elif strategy == "youden":
            obj = sens + spec - 1.0
        elif strategy == "sens_floor_spec":
            obj = spec if sens >= min_sensitivity else -1.0 + spec * 1e-3
        else:
            raise ValueError(strategy)
        if obj > best_obj:
            best_obj, best_t = obj, float(t)
    return best_t


# ----------------------------------------------------------------------------
# Metrics                                                          # >>> SWAP <<<
# ----------------------------------------------------------------------------
def _window_metrics(y, pred, p):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    f1 = 2 * prec * sens / (prec + sens + 1e-9)
    out = dict(acc=acc, sens=sens, spec=spec, f1=f1)
    # AUROC / AUPRC need both classes present
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def _event_metrics(cache: WindowCache, pred: np.ndarray):
    """Event-level detection rate, mean delay, and false alarms / hour.

    Ground-truth events = maximal runs of consecutive positive windows within a
    recording (ordered by t_start). An event is DETECTED if any predicted-
    positive window overlaps it; delay = (first positive window start) - (event
    onset), clipped to >= 0. FA/h counts predicted-positive windows that do not
    overlap any true event, normalized by total recorded hours.
    """
    n_events = 0
    n_detected = 0
    delays = []
    fp_windows = 0
    total_seconds = 0.0
    for rid in np.unique(cache.rec_id):
        m = cache.rec_id == rid
        order = np.argsort(cache.t_start[m])
        ts = cache.t_start[m][order]
        yy = cache.y[m][order]
        pp = pred[m][order]
        total_seconds += len(ts) * cache.step_sec
        # group true events
        i = 0
        in_event_window = np.zeros(len(yy), dtype=bool)
        while i < len(yy):
            if yy[i] == 1:
                j = i
                while j < len(yy) and yy[j] == 1:
                    j += 1
                in_event_window[i:j] = True
                n_events += 1
                onset = ts[i]
                hit = np.where(pp[i:j] == 1)[0]
                if len(hit):
                    n_detected += 1
                    delays.append(max(0.0, float(ts[i + hit[0]] - onset)))
                i = j
            else:
                i += 1
        fp_windows += int(((pp == 1) & (~in_event_window)).sum())
    det_rate = n_detected / (n_events + 1e-9)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")
    fa_per_hour = fp_windows / (total_seconds / 3600.0 + 1e-9)
    return dict(det_rate=det_rate, mean_delay=mean_delay,
                median_delay=median_delay, fa_per_hour=fa_per_hour,
                n_events=n_events, n_detected=n_detected)


def compute_metrics(cache_test: WindowCache, p_test: np.ndarray, threshold: float):
    """Full MV-AFA metric set on a held-out test pool at a fixed threshold."""
    pred = (p_test >= threshold).astype(int)
    out = _window_metrics(cache_test.y, pred, p_test)
    out.update(_event_metrics(cache_test, pred))
    out["threshold"] = float(threshold)
    return out


def aggregate_folds(per_fold: list[dict]) -> dict:
    """mean +/- std across LOSO folds for every metric (the LOSO reporting format)."""
    keys = [k for k in per_fold[0] if isinstance(per_fold[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([f[k] for f in per_fold if not np.isnan(f[k])], dtype=float)
        agg[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                      n=int(len(vals)))
    return agg


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

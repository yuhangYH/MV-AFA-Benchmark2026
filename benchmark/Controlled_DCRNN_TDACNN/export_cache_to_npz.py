#!/usr/bin/env python
"""Export an MV-AFA window cache to the .npz the standalone baselines consume.

The DCRNN / TDA-CNN baselines in this folder are standalone (they do NOT import
the mvafa data layer), so they read a precomputed window cache as a single .npz.
This one-time helper builds the MV-AFA cache via the SAME `build_or_load_cache`
used by `engine.py` / `loocv.py`, then dumps the fields the baselines need —
guaranteeing the baselines see byte-identical windows / labels / subject ids as
MV-AFA.

It writes, using only known public interfaces (`build_or_load_cache`, the
dataset's `.meta()`, and `cache["label"]/["subject"]`):

    X        = cache["raw"]            # [N, C, temporal_samples] (the same
                                       #   down-sampled raw window MV-AFA feeds
                                       #   its temporal + frequency branches)
    y        = cache["label"]
    subject  = cache["subject"]
    rec_id   = meta["recording"]
    t_start  = meta["onset"]
    win_sec, step_sec = dataset_cfg.window_sec / .step_sec

Usage (on the machine that has the raw corpus + data layer):
    python baselines/export_cache_to_npz.py \
        --dataset chb_mit \
        --raw-root D:/.../1-CHB-MIT-scalp-eeg-database-1.0.0 \
        --cache-dir .../cache/chbmit_loso \
        --out chbmit_windows.npz
"""
import argparse
import os
import sys

import numpy as np

def _load_mvafa_api(code_root: str | None):
    """Import the private MV-AFA data layer used to build the shared cache."""
    root = code_root or os.environ.get("MVAFA_CODE_ROOT")
    if root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "code",
                                            "mvafa-eeg-seizure-code"))
    if not os.path.isdir(root):
        raise FileNotFoundError(
            "Could not find the MV-AFA code root. Pass --mvafa-code-root or set "
            "MVAFA_CODE_ROOT to the directory that contains the mvafa package."
        )
    sys.path.insert(0, root)
    from mvafa.data.dataset import EEGWindowDataset, build_or_load_cache
    from mvafa.presets import get_dataset_config
    from mvafa.utils import get_logger
    return EEGWindowDataset, build_or_load_cache, get_dataset_config, get_logger


def main():
    p = argparse.ArgumentParser(description="Export MV-AFA cache -> baselines .npz")
    p.add_argument("--dataset", required=True, choices=["chb_mit", "siena", "tusz"])
    p.add_argument("--raw-root", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--mvafa-code-root", default=None,
                   help="directory containing the private mvafa package")
    p.add_argument("--limit-recordings", type=int, default=None)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--raw-key", default="raw",
                   help="cache key holding the raw window tensor (default 'raw').")
    args = p.parse_args()

    EEGWindowDataset, build_or_load_cache, get_dataset_config, get_logger = (
        _load_mvafa_api(args.mvafa_code_root)
    )
    logger = get_logger("export")
    cfg = get_dataset_config(args.dataset, args.raw_root, args.cache_dir)
    cache = build_or_load_cache(cfg, limit_recordings=args.limit_recordings,
                                force_rebuild=args.force_rebuild, logger=logger)

    if args.raw_key not in cache:
        raise KeyError(f"cache has no '{args.raw_key}'. Available keys: "
                       f"{list(cache.keys())}. Re-run with --raw-key <name>.")

    idx = np.arange(len(cache["label"]))
    meta = EEGWindowDataset(cache, idx).meta()  # {'recording','onset','label'}

    X = np.asarray(cache[args.raw_key], dtype=np.float32)
    np.savez_compressed(
        args.out,
        X=X,
        y=np.asarray(cache["label"]).astype(np.int64),
        subject=np.asarray(cache["subject"], dtype=object),
        rec_id=np.asarray(meta["recording"], dtype=object),
        t_start=np.asarray(meta["onset"], dtype=np.float32),
        win_sec=float(cfg.window_sec),
        step_sec=float(cfg.step_sec),
    )
    pos = int(np.sum(cache["label"]))
    print(f"Wrote {args.out}: X={X.shape}, N={len(idx)}, "
          f"pos={pos} ({100*pos/len(idx):.3f}%), "
          f"subjects={len(set(np.asarray(cache['subject']).tolist()))}")


if __name__ == "__main__":
    main()

"""
Two-phase training:
  Phase 1: Pre-train on GOES XRS historical data (large corpus, many flares)
  Phase 2: Fine-tune on Aditya-L1 SoLEXS+HEL1OS (freeze Conv layers)

Usage:
    # Pre-train
    python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30

    # Fine-tune on Aditya-L1 data
    python model/train.py --phase finetune --sxr-dir data/processed --epochs 10
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as `python model/train.py`
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
import pipeline._compat  # noqa: F401 — installs parquet fallback if pyarrow missing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.cnn_lstm import SolarFlareForecaster, build_model, load_checkpoint, save_checkpoint
from model.features import FEATURE_NAMES, FLARE_CLASS_ORDER, N_FEATURES, build_feature_series, build_windows, normalize_features


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_goes_training_data(goes_dir: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load GOES-16 XRS NetCDF4 files and build (X, y, y_cls) training arrays.

    Processes files in CHUNK_DAYS-file batches to keep peak RAM ~400 MB;
    intermediate results are concatenated in-place and returned as a single
    pair of arrays covering all input files.
    """
    import pandas as pd
    from pipeline.readers import read_goes_nc
    from pipeline.preprocessor import preprocess_goes
    from pipeline.detector import detect_goes_flares

    fc = cfg["forecasting"]
    window_min = fc["input_window_min"]
    lead_min   = fc["lead_time_min"]

    nc_files = sorted(Path(goes_dir).glob("sci_xrsf-l2-avg1m_g16_d*.nc"))
    # Also pick up NRT files (dn_ prefix)
    nc_files += sorted(Path(goes_dir).glob("dn_xrsf-l2-avg1m_g16_d*.nc"))
    nc_files = sorted(set(nc_files))

    if not nc_files:
        raise FileNotFoundError(
            f"No GOES NetCDF4 files found in {goes_dir}.\n"
            f"Run: python scripts/download_goes.py --start 2020-01-01 --end 2023-12-31"
        )

    n_total_flares = 0
    X_all, y_all, yc_all = [], [], []
    # ponytail: 90-day chunks keep peak RAM ~400 MB; increase if RAM allows
    CHUNK = 90

    for f in tqdm(nc_files, desc="GOES NetCDF files"):
        try:
            raw  = read_goes_nc(f)
            proc = preprocess_goes(raw, cfg)
            feat = build_feature_series(proc, df_hxr=None)

            detected = detect_goes_flares(proc, cfg)
            if detected:
                local_flares = pd.DatetimeIndex([ev.peak_time for ev in detected])
                local_classes = [ev.flare_class for ev in detected]
                n_total_flares += len(detected)
            else:
                local_flares, local_classes = None, []

            # Always request class labels (flare_classes not None) so we get the
            # 3-tuple; y_cls is all -1 when there are no flares this day.
            X, y, yc = build_windows(
                feat, local_flares, window_min=window_min, lead_time_min=lead_min,
                flare_classes=local_classes,
            )
            if len(X) > 0:
                X_all.append(X); y_all.append(y); yc_all.append(yc)

            # Flush chunk to avoid OOM
            if len(X_all) >= CHUNK:
                X_all = [np.concatenate(X_all)]
                y_all = [np.concatenate(y_all)]
                yc_all = [np.concatenate(yc_all)]

        except Exception as e:
            print(f"  Skip {Path(f).name}: {e}")

    if not X_all:
        raise RuntimeError("No training windows built. Check data and labels.")

    print(f"  Self-labeled {n_total_flares} flare peaks across {len(nc_files)} files.")
    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(yc_all)


def load_aditya_training_data(processed_dir: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load Aditya-L1 processed Parquet files for fine-tuning."""
    import pandas as pd
    from pipeline.catalog import load_master_catalog

    fc = cfg["forecasting"]
    window_min = fc["input_window_min"]
    lead_min   = fc["lead_time_min"]

    catalog = load_master_catalog(ROOT / cfg["paths"]["master_catalog"])
    empty_cat = catalog.empty
    flare_times = pd.DatetimeIndex(catalog["peak_time"]) if not empty_cat else None
    # Catalog stores class in 'flare_class' (SoLEXS+HEL1OS merge). Align to flare_times.
    all_classes = catalog["flare_class"].to_numpy() if not empty_cat and "flare_class" in catalog else None

    sxr_files = sorted(Path(processed_dir).glob("*_solexs.parquet"))
    hxr_files = sorted(Path(processed_dir).glob("*_helios.parquet"))

    # Build a dict by date for quick lookup
    hxr_by_date = {f.name[:8]: f for f in hxr_files}

    X_all, y_all, yc_all = [], [], []

    for sf in tqdm(sxr_files, desc="Aditya-L1 files"):
        try:
            sxr = pd.read_parquet(sf)
            date_key = sf.name[:8]
            hxr = pd.read_parquet(hxr_by_date[date_key]) if date_key in hxr_by_date else None
            feat = build_feature_series(sxr, hxr)

            local_flares, local_classes = None, []
            if flare_times is not None:
                mask = (flare_times >= feat.index[0]) & (flare_times <= feat.index[-1])
                local_flares = pd.DatetimeIndex(flare_times[mask])
                if all_classes is not None:
                    local_classes = list(all_classes[np.asarray(mask)])

            X, y, yc = build_windows(
                feat, local_flares, window_min=window_min, lead_time_min=lead_min,
                flare_classes=local_classes,
            )
            if len(X) > 0:
                X_all.append(X); y_all.append(y); yc_all.append(yc)
        except Exception as e:
            print(f"  Skip {sf.name}: {e}")

    if not X_all:
        raise RuntimeError("No fine-tuning windows built.")

    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(yc_all)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def make_loaders(X: np.ndarray, y: np.ndarray, y_cls: np.ndarray, val_frac: float = 0.1, batch: int = 256):
    """Split into train/val and wrap in DataLoaders. y_cls carried for multi-class."""
    n = len(X)
    n_val = max(1, int(n * val_frac))
    idx = np.random.default_rng(0).permutation(n)
    tr, vl = idx[n_val:], idx[:n_val]
    X_tr, y_tr, c_tr = X[tr], y[tr], y_cls[tr]
    X_vl, y_vl, c_vl = X[vl], y[vl], y_cls[vl]

    X_tr, X_vl, mean, std = normalize_features(X_tr, X_vl)

    def _loader(Xa, ya, ca, shuffle):
        ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya), torch.from_numpy(ca))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)

    return _loader(X_tr, y_tr, c_tr, True), _loader(X_vl, y_vl, c_vl, False), mean, std


def run_epoch(model, loader, optimizer, criterion, device, train: bool, ce_criterion=None) -> tuple[float, float]:
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for Xb, yb, yb_cls in loader:
            Xb, yb, yb_cls = Xb.to(device), yb.to(device), yb_cls.to(device)
            out = model(Xb)
            if isinstance(out, tuple):
                logits, class_logits = out          # multi-class model
                loss = criterion(logits, yb)
                # Add class loss only when the batch has ≥1 labelled (positive) window;
                # CrossEntropyLoss(ignore_index=-1) would return NaN if all are ignored.
                if ce_criterion is not None and (yb_cls != -1).any():
                    loss = loss + ce_criterion(class_logits, yb_cls)
            else:
                logits = out                          # binary-only model
                loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)

    return total_loss / total, correct / total


def train(
    model: SolarFlareForecaster,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    ckpt_path: str,
    device: str,
    freeze_conv: bool = False,
    y_cls: np.ndarray | None = None,
) -> None:
    if freeze_conv:
        for p in model.conv.parameters():
            p.requires_grad = False

    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Multi-class loss only when the model actually has an intensity head.
    multiclass = getattr(model, "class_head", None) is not None
    ce_criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device) if multiclass else None
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    if y_cls is None:  # binary-only caller; -1 everywhere is harmless (all ignored)
        y_cls = np.full(len(y), -1, dtype=np.int64)
    train_loader, val_loader, mean, std = make_loaders(X, y, y_cls)

    best_val_loss = float("inf")
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True, ce_criterion=ce_criterion)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False, ce_criterion=ce_criterion)
        scheduler.step(vl_loss)
        print(f"Ep {ep:3d} | tr_loss={tr_loss:.4f} acc={tr_acc:.3f} | vl_loss={vl_loss:.4f} acc={vl_acc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            # Persist normalization stats so inference (dashboard/evaluate) can
            # reproduce the z-scoring the model was trained with.
            save_checkpoint(model, ckpt_path, meta={
                "epoch": ep,
                "val_loss": vl_loss,
                "norm_mean": np.asarray(mean).tolist(),
                "norm_std": np.asarray(std).tolist(),
                "feature_names": FEATURE_NAMES,
                "flare_class_order": FLARE_CLASS_ORDER if multiclass else None,
            })

    print(f"Best val_loss={best_val_loss:.4f} -> saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--goes-dir", default="data/raw/goes")
    parser.add_argument("--sxr-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--unfreeze-conv", action="store_true",
                        help="Finetune with conv layers unfrozen (use very low LR, e.g. 5e-6)")
    args = parser.parse_args()

    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt_path = args.ckpt or cfg["paths"]["model_checkpoint"]

    def _class_dist(y_cls: np.ndarray) -> str:
        pos = y_cls[y_cls >= 0]
        if len(pos) == 0:
            return "no labelled flares"
        counts = {FLARE_CLASS_ORDER[i]: int((pos == i).sum()) for i in range(len(FLARE_CLASS_ORDER))}
        return " ".join(f"{k}:{v}" for k, v in counts.items() if v)

    if args.phase == "pretrain":
        print("Phase 1: Pre-training on GOES XRS data")
        X, y, y_cls = load_goes_training_data(args.goes_dir, cfg)
        print(f"  Dataset: {len(X)} windows  |  pos={y.mean():.2%}  |  classes: {_class_dist(y_cls)}")
        model = build_model(cfg, device)
        lr = args.lr or 1e-3
        train(model, X, y, args.epochs, lr, ckpt_path, device, freeze_conv=False, y_cls=y_cls)

    elif args.phase == "finetune":
        print("Phase 2: Fine-tuning on Aditya-L1 data")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run pretrain first.")
        model = load_checkpoint(ckpt_path, device, cfg)
        X, y, y_cls = load_aditya_training_data(args.sxr_dir, cfg)
        print(f"  Dataset: {len(X)} windows  |  pos={y.mean():.2%}  |  classes: {_class_dist(y_cls)}")
        lr = args.lr or 1e-4
        freeze = not args.unfreeze_conv
        print(f"  Conv layers: {'FROZEN' if freeze else 'UNFROZEN'}")
        train(model, X, y, args.epochs, lr, ckpt_path, device, freeze_conv=freeze, y_cls=y_cls)

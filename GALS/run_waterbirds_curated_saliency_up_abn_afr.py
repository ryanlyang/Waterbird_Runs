#!/usr/bin/env python3
"""
Train fixed AFR (WB95/WB100) and generate curated validation saliency maps for:
- upweight
- abn
- afr

No CLIP ZS/LR in this script.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import waterbirds100_guided_vanilla_saliency as wbsv
import waterbirds_pointing_game_eval as wbe


WB100_CURATED: Dict[str, List[str]] = {
    "Waterbird_Best_100": [
        "060_Glaucous_winged_Gull__Glaucous_Winged_Gull_0012_44264_jpg",
        "084_Red_legged_Kittiwake__Red_Legged_Kittiwake_0068_795430_jpg",
        "100_Brown_Pelican__Brown_Pelican_0077_93464_jpg",
        "005_Crested_Auklet__Crested_Auklet_0071_785255_jpg",
        "087_Mallard__Mallard_0052_76946_jpg",
        "106_Horned_Puffin__Horned_Puffin_0056_101030_jpg",
        "072_Pomarine_Jaeger__Pomarine_Jaeger_0078_795758_jpg",
        "046_Gadwall__Gadwall_0035_30985_jpg",
    ],
    "Landbird_Best_100": [
        "097_Orchard_Oriole__Orchard_Oriole_0006_91724_jpg",
        "057_Rose_breasted_Grosbeak__Rose_Breasted_Grosbeak_0114_39770_jpg",
        "009_Brewer_Blackbird__Brewer_Blackbird_0140_2586_jpg",
        "018_Spotted_Catbird__Spotted_Catbird_0010_19436_jpg",
        "136_Barn_Swallow__Barn_Swallow_0045_130244_jpg",
        "080_Green_Kingfisher__Green_Kingfisher_0004_71076_jpg",
        "165_Chestnut_sided_Warbler__Chestnut_Sided_Warbler_0014_163801_jpg",
        "178_Swainson_Warbler__Swainson_Warbler_0011_174680_jpg",
    ],
}


WB95_CURATED: Dict[str, List[str]] = {
    "wb95_water_Best_Picks": [
        "106_Horned_Puffin__Horned_Puffin_0024_100620_jpg",
        "144_Common_Tern__Common_Tern_0117_148944_jpg",
        "060_Glaucous_winged_Gull__Glaucous_Winged_Gull_0110_44377_jpg",
        "146_Forsters_Tern__Forsters_Tern_0127_150418_jpg",
        "021_Eastern_Towhee__Eastern_Towhee_0101_22559_jpg",
        "084_Red_legged_Kittiwake__Red_Legged_Kittiwake_0036_73814_jpg",
        "147_Least_Tern__Least_Tern_0082_154396_jpg",
        "101_White_Pelican__White_Pelican_0010_96876_jpg",
    ],
    "wb95_guided_vanilla_gals_saliency_21065978_New_Best": [
        "082_Ringed_Kingfisher__Ringed_Kingfisher_0050_73002_jpg",
        "011_Rusty_Blackbird__Rusty_Blackbird_0113_6664_jpg",
        "038_Great_Crested_Flycatcher__Great_Crested_Flycatcher_0009_29831_jpg",
        "160_Black_throated_Blue_Warbler__Black_Throated_Blue_Warbler_0081_161427_jpg",
        "171_Myrtle_Warbler__Myrtle_Warbler_0037_166690_jpg",
        "069_Rufous_Hummingbird__Rufous_Hummingbird_0095_60360_jpg",
        "019_Gray_Catbird__Gray_Catbird_0094_21303_jpg",
        "198_Rock_Wren__Rock_Wren_0019_188968_jpg",
    ],
}


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    data_path: Path
    gt_root: Path
    upweight_ckpt: Path
    abn_ckpt: Path
    curated: Dict[str, List[str]]
    afr_gamma: float
    afr_reg: float
    afr_seed: int


def _normalize_token_text(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("\\", "/")
    t = t.replace(".jpg", "_jpg").replace(".jpeg", "_jpeg").replace(".png", "_png")
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _token_variants_from_relpath(rel_path: str) -> List[str]:
    p = Path(rel_path)
    parent = p.parent.name
    stem = p.stem
    parent_us = parent.replace(".", "_")
    parent_wo_prefix = re.sub(r"^\d+[._]", "", parent)
    parent_us_wo_prefix = re.sub(r"^\d+_", "", parent_us)
    variants = [
        f"{parent}__{stem}_jpg",
        f"{parent_us}__{stem}_jpg",
        f"{parent_us}_{stem}_jpg",
        f"{parent_wo_prefix}__{stem}_jpg",
        f"{parent_us_wo_prefix}__{stem}_jpg",
        f"{stem}_jpg",
        rel_path,
    ]
    out: List[str] = []
    seen = set()
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _resolve_rows_from_tokens(
    metadata_df: pd.DataFrame,
    curated: Dict[str, List[str]],
    split_code: int = 1,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], Dict[str, object]]:
    val_df = metadata_df[metadata_df["split"].astype(int) == int(split_code)].copy()
    if val_df.empty:
        raise RuntimeError(f"No rows for split code={split_code} in metadata.")

    tok_to_rows: Dict[str, List[pd.Series]] = {}
    preview: List[str] = []
    for _, row in val_df.iterrows():
        rel = str(row["img_filename"])
        if len(preview) < 16:
            preview.append(f"{Path(rel).parent.name}__{Path(rel).stem}_jpg")
        for tok in _token_variants_from_relpath(rel):
            tok_to_rows.setdefault(_normalize_token_text(tok), []).append(row)

    selected: List[Dict[str, object]] = []
    missing: List[Dict[str, str]] = []
    for category, toks in curated.items():
        for tok in toks:
            key = _normalize_token_text(tok)
            matches = tok_to_rows.get(key, [])
            if not matches:
                missing.append({"category": category, "token": tok})
                continue
            row = matches[0]
            selected.append(
                {
                    "category": category,
                    "token": tok,
                    "img_filename": str(row["img_filename"]),
                    "y": int(row["y"]),
                    "place": int(row["place"]),
                    "group": int(int(row["y"]) * 2 + int(row["place"])),
                }
            )

    dbg = {
        "val_rows": int(len(val_df)),
        "lookup_keys": int(len(tok_to_rows)),
        "preview_tokens": preview,
    }
    return selected, missing, dbg


def _run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("[CMD] " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(line, end="", flush=True)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


def _train_fixed_afr(
    *,
    repo_root: Path,
    afr_root: Path,
    data_dir: Path,
    output_root: Path,
    logs_root: Path,
    gamma: float,
    reg: float,
    seed: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage2_lr: float,
    force_stage1: bool,
    force_stage2: bool,
) -> Tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str((repo_root / "run_afr_waterbirds_repro.py").resolve()),
        "--afr-root",
        str(afr_root),
        "--data-dir",
        str(data_dir),
        "--output-root",
        str(output_root),
        "--logs-root",
        str(logs_root),
        "--python-exe",
        sys.executable,
        "--seeds",
        str(seed),
        "--stage1-epochs",
        str(stage1_epochs),
        "--stage1-eval-freq",
        "10",
        "--stage1-save-freq",
        "10",
        "--stage1-scheduler",
        "constant_lr_scheduler",
        "--stage2-epochs",
        str(stage2_epochs),
        "--stage2-lr",
        str(stage2_lr),
        "--gammas",
        str(gamma),
        "--reg-coeffs",
        str(reg),
    ]
    if force_stage1:
        cmd.append("--force-stage1")
    if force_stage2:
        cmd.append("--force-stage2")

    log_path = logs_root / f"afr_fixed_seed{seed}_g{gamma}_r{reg}.log"
    _run_cmd(cmd=cmd, cwd=repo_root, log_path=log_path)

    best_csv = output_root / "afr_waterbirds_best_by_seed.csv"
    if not best_csv.is_file():
        raise RuntimeError(f"AFR best CSV missing: {best_csv}")
    df = pd.read_csv(best_csv)
    if df.empty:
        raise RuntimeError(f"AFR best CSV is empty: {best_csv}")
    row = df.iloc[0]
    stage1_dir = Path(str(row["stage1_dir"])).expanduser().resolve()
    stage2_dir = Path(str(row["stage2_dir"])).expanduser().resolve()
    stage1_ckpt = stage1_dir / "final_checkpoint.pt"
    stage2_last = stage2_dir / "final_checkpoint.pt"
    if not stage1_ckpt.is_file():
        raise RuntimeError(f"AFR stage1 checkpoint missing: {stage1_ckpt}")
    if not stage2_last.is_file():
        raise RuntimeError(f"AFR stage2 last-layer checkpoint missing: {stage2_last}")
    return stage1_ckpt, stage2_last, best_csv


def _save_method_saliency(
    method_name: str,
    saliency: np.ndarray,
    image_rgb: np.ndarray,
    sample_dir: Path,
) -> Dict[str, np.ndarray]:
    return wbsv.save_saliency_variants(method_name, saliency, image_rgb, sample_dir)


def _run_dataset(
    spec: DatasetSpec,
    *,
    repo_root: Path,
    afr_root: Path,
    out_root: Path,
    target_mode: str,
    rise_num_masks: int,
    rise_grid_size: int,
    rise_p1: float,
    rise_gpu_batch: int,
    rise_seed: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage2_lr: float,
    force_stage1: bool,
    force_stage2: bool,
    device: torch.device,
) -> None:
    ds_out = out_root / f"waterbirds_{spec.tag}"
    ds_out.mkdir(parents=True, exist_ok=True)
    samples_dir = ds_out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    afr_out = ds_out / "afr_training"
    afr_logs = ds_out / "logs"
    afr_logs.mkdir(parents=True, exist_ok=True)

    print(f"\n[DATASET {spec.tag}] AFR training (fixed): gamma={spec.afr_gamma} reg={spec.afr_reg} seed={spec.afr_seed}")
    afr_stage1_ckpt, afr_stage2_ckpt, afr_best_csv = _train_fixed_afr(
        repo_root=repo_root,
        afr_root=afr_root,
        data_dir=spec.data_path,
        output_root=afr_out,
        logs_root=afr_logs,
        gamma=float(spec.afr_gamma),
        reg=float(spec.afr_reg),
        seed=int(spec.afr_seed),
        stage1_epochs=int(stage1_epochs),
        stage2_epochs=int(stage2_epochs),
        stage2_lr=float(stage2_lr),
        force_stage1=bool(force_stage1),
        force_stage2=bool(force_stage2),
    )
    print(f"[DATASET {spec.tag}] AFR stage1: {afr_stage1_ckpt}", flush=True)
    print(f"[DATASET {spec.tag}] AFR stage2 last-layer: {afr_stage2_ckpt}", flush=True)

    metadata_path = spec.data_path / "metadata.csv"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing metadata: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    num_classes = int(metadata_df["y"].nunique())

    selected, missing, dbg = _resolve_rows_from_tokens(metadata_df, spec.curated, split_code=1)
    print(
        f"[DATASET {spec.tag}] selected={len(selected)} missing={len(missing)} (validation split only)",
        flush=True,
    )
    if not selected:
        raise RuntimeError(
            f"No curated rows resolved for dataset {spec.tag}. "
            f"Debug: val_rows={dbg['val_rows']} lookup={dbg['lookup_keys']} preview={dbg['preview_tokens']}"
        )

    preprocess = wbe.build_preprocess()
    up_runner = wbe.UpweightRunner(spec.upweight_ckpt, num_classes=num_classes, device=device)
    abn_runner = wbe.ABNRunner(spec.abn_ckpt, num_classes=num_classes, device=device)
    afr_runner = wbe.AFRRunner(
        afr_root=afr_root,
        stage1_checkpoint=afr_stage1_ckpt,
        stage2_last_layer_checkpoint=afr_stage2_ckpt,
        num_classes=num_classes,
        device=device,
    )
    runners = {
        "upweight": up_runner,
        "abn": abn_runner,
        "afr": afr_runner,
    }

    rows_out: List[Dict[str, object]] = []
    try:
        for idx, item in enumerate(selected):
            rel = str(item["img_filename"])
            img_path = spec.data_path / rel
            if not img_path.is_file():
                continue
            image_pil = wbe.open_pil_with_retry(img_path, mode="RGB")
            image_rgb = np.array(image_pil, dtype=np.uint8)
            h, w = image_rgb.shape[:2]
            x = preprocess(image_pil).unsqueeze(0).to(device)
            label = int(item["y"])

            token = str(item["token"])
            category = str(item["category"])
            sample_name = f"{idx:03d}_{wbsv.safe_token(category)}__{wbsv.safe_token(token)}"
            sample_dir = samples_dir / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)
            wbsv.save_rgb(sample_dir / "image_rgb.png", image_rgb)

            vis_by_model: Dict[str, Dict[str, np.ndarray]] = {}
            preds: Dict[str, int] = {}
            targets: Dict[str, int] = {}
            for method_name, runner in runners.items():
                pred, target, sal = runner.predict_and_saliency(
                    image_tensor=x,
                    label=label,
                    target_mode=target_mode,
                    pil_image=image_pil,
                )
                sal = wbe.upsample_saliency(sal, h, w)
                vis_by_model[method_name] = _save_method_saliency(method_name, sal, image_rgb, sample_dir)
                preds[method_name] = int(pred)
                targets[method_name] = int(target)

            wbsv.write_comparison_panels(sample_dir, vis_by_model)

            mask_path, mask_default = wbe.resolve_mask_path(spec.gt_root, img_path, spec.data_path)
            has_mask = wbsv.save_gt_mask_variants(mask_path, image_rgb, sample_dir) if mask_path else False

            info = {
                "dataset": spec.tag,
                "category": category,
                "token": token,
                "img_filename": rel,
                "label": label,
                "group": int(item["group"]),
                "upweight_pred": preds["upweight"],
                "abn_pred": preds["abn"],
                "afr_pred": preds["afr"],
                "upweight_target_class": targets["upweight"],
                "abn_target_class": targets["abn"],
                "afr_target_class": targets["afr"],
                "gt_mask_path": str(mask_path) if has_mask and mask_path is not None else None,
                "gt_mask_default_path": str(mask_default),
            }
            with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            rows_out.append(info)
    finally:
        for runner in runners.values():
            runner.close()

    summary = {
        "dataset": spec.tag,
        "data_path": str(spec.data_path),
        "gt_root": str(spec.gt_root),
        "target_mode": target_mode,
        "num_requested": int(sum(len(v) for v in spec.curated.values())),
        "num_resolved": int(len(selected)),
        "num_generated": int(len(rows_out)),
        "num_missing": int(len(missing)),
        "missing_tokens": missing,
        "match_debug": dbg,
        "upweight_checkpoint": str(spec.upweight_ckpt),
        "abn_checkpoint": str(spec.abn_ckpt),
        "afr": {
            "gamma": float(spec.afr_gamma),
            "reg_coeff": float(spec.afr_reg),
            "seed": int(spec.afr_seed),
            "best_csv": str(afr_best_csv),
            "stage1_checkpoint": str(afr_stage1_ckpt),
            "stage2_last_layer_checkpoint": str(afr_stage2_ckpt),
        },
    }
    with open(ds_out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows_out).to_csv(ds_out / "sample_index.csv", index=False)
    if missing:
        pd.DataFrame(missing).to_csv(ds_out / "missing_tokens.csv", index=False)

    print(f"[DATASET {spec.tag}] wrote {len(rows_out)} curated saliency samples to {samples_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed AFR + curated saliency (upweight/abn/afr) on Waterbirds 95/100.")
    p.add_argument("--run-wb95", action="store_true", default=True)
    p.add_argument("--run-wb100", action="store_true", default=True)
    p.add_argument("--target-mode", choices=["label", "pred"], default="label")
    p.add_argument("--device", default="", help="e.g. cuda:0; default auto")

    p.add_argument("--afr-root", default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/afr")
    p.add_argument("--stage1-epochs", type=int, default=50)
    p.add_argument("--stage2-epochs", type=int, default=500)
    p.add_argument("--stage2-lr", type=float, default=0.01)
    p.add_argument("--force-stage1", action="store_true")
    p.add_argument("--force-stage2", action="store_true")

    p.add_argument("--wb95-afr-seed", "--wb95-afar-seed", dest="wb95_afr_seed", type=int, default=2)
    p.add_argument("--wb95-afr-gamma", type=float, default=11.0)
    p.add_argument("--wb95-afr-reg", type=float, default=0.0)
    p.add_argument("--wb100-afr-seed", "--wb100-afar-seed", dest="wb100_afr_seed", type=int, default=2)
    p.add_argument("--wb100-afr-gamma", type=float, default=4.0)
    p.add_argument("--wb100-afr-reg", type=float, default=0.0)

    p.add_argument("--wb95-data-path", default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2")
    p.add_argument("--wb100-data-path", default="/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2")
    p.add_argument(
        "--wb95-gt-root",
        default="/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    )
    p.add_argument(
        "--wb100-gt-root",
        default="/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    )

    p.add_argument(
        "--upweight95-ckpt",
        default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_upweight_wb95_fixed_21073587_trial_000/best_balanced_valacc_0.86_epoch_169.ckpt",
    )
    p.add_argument(
        "--upweight100-ckpt",
        default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_upweight_wb100_fixed_21073587_trial_000/best_balanced_valacc_0.69_epoch_1.ckpt",
    )
    p.add_argument(
        "--abn95-ckpt",
        default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_abn_wb95_fixed_21073587_trial_000/best_balanced_valacc_0.87_epoch_7.ckpt",
    )
    p.add_argument(
        "--abn100-ckpt",
        default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/retrain_abn_wb100_fixed_21073587_trial_000/best_balanced_valacc_0.7_epoch_43.ckpt",
    )

    p.add_argument(
        "--output-dir",
        default="",
        help="If empty: /home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_up_abn_afr_<timestamp>",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    afr_root = Path(args.afr_root).expanduser().resolve()
    if not afr_root.is_dir():
        raise RuntimeError(f"AFR root missing: {afr_root}")
    if not (afr_root / "train_supervised.py").is_file():
        raise RuntimeError(f"AFR script missing: {afr_root / 'train_supervised.py'}")

    if args.output_dir:
        out_root = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/home/ryreu/guided_cnn/logsWaterbird") / f"waterbirds_curated_saliency_up_abn_afr_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] repo_root={repo_root}", flush=True)
    print(f"[INFO] afr_root={afr_root}", flush=True)
    print(f"[INFO] output_dir={out_root}", flush=True)
    print(f"[INFO] device={device}", flush=True)
    print("[INFO] methods=upweight,abn,afr (no CLIP methods)", flush=True)

    specs: List[DatasetSpec] = []
    if args.run_wb95:
        specs.append(
            DatasetSpec(
                tag="95",
                data_path=Path(args.wb95_data_path).expanduser().resolve(),
                gt_root=Path(args.wb95_gt_root).expanduser().resolve(),
                upweight_ckpt=Path(args.upweight95_ckpt).expanduser().resolve(),
                abn_ckpt=Path(args.abn95_ckpt).expanduser().resolve(),
                curated=WB95_CURATED,
                afr_gamma=float(args.wb95_afr_gamma),
                afr_reg=float(args.wb95_afr_reg),
                afr_seed=int(args.wb95_afr_seed),
            )
        )
    if args.run_wb100:
        specs.append(
            DatasetSpec(
                tag="100",
                data_path=Path(args.wb100_data_path).expanduser().resolve(),
                gt_root=Path(args.wb100_gt_root).expanduser().resolve(),
                upweight_ckpt=Path(args.upweight100_ckpt).expanduser().resolve(),
                abn_ckpt=Path(args.abn100_ckpt).expanduser().resolve(),
                curated=WB100_CURATED,
                afr_gamma=float(args.wb100_afr_gamma),
                afr_reg=float(args.wb100_afr_reg),
                afr_seed=int(args.wb100_afr_seed),
            )
        )
    if not specs:
        raise RuntimeError("No dataset selected.")

    for spec in specs:
        for p in [spec.data_path, spec.gt_root]:
            if not p.exists():
                raise RuntimeError(f"Missing path: {p}")
        for ck in [spec.upweight_ckpt, spec.abn_ckpt]:
            if not ck.is_file():
                raise RuntimeError(f"Missing checkpoint: {ck}")

    for spec in specs:
        _run_dataset(
            spec=spec,
            repo_root=repo_root,
            afr_root=afr_root,
            out_root=out_root,
            target_mode=str(args.target_mode),
            rise_num_masks=2000,
            rise_grid_size=8,
            rise_p1=0.1,
            rise_gpu_batch=16,
            rise_seed=0,
            stage1_epochs=int(args.stage1_epochs),
            stage2_epochs=int(args.stage2_epochs),
            stage2_lr=float(args.stage2_lr),
            force_stage1=bool(args.force_stage1),
            force_stage2=bool(args.force_stage2),
            device=device,
        )

    print("\n[DONE] AFR training + curated saliency complete.", flush=True)
    print(f"[DONE] output_root={out_root}", flush=True)


if __name__ == "__main__":
    main()

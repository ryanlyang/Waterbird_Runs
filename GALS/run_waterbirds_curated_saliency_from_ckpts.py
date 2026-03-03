#!/usr/bin/env python3
"""
Generate curated Waterbirds saliency maps from existing checkpoints (no retraining).

Models used per dataset:
- guided
- vanilla
- gals (generic CNN run trained with clip_vit attention maps)

This script evaluates only validation images, matching explicit token lists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import waterbirds100_guided_vanilla_saliency as wbsv


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
    guided_ckpt: Path
    vanilla_ckpt: Path
    gals_ckpt: Path
    curated: Dict[str, List[str]]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_token(rel_path: str) -> str:
    p = Path(rel_path)
    return f"{p.parent.name}__{p.stem}_jpg"


def normalize_token_text(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("\\", "/")
    t = t.replace(".jpg", "_jpg").replace(".jpeg", "_jpeg").replace(".png", "_png")
    # Normalize all separators/punctuation to underscores.
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def token_variants_from_relpath(rel_path: str) -> List[str]:
    p = Path(rel_path)
    parent = p.parent.name
    stem = p.stem
    parent_us = parent.replace(".", "_")
    parent_wo_prefix = re.sub(r"^\d+[._]", "", parent)
    parent_us_wo_prefix = re.sub(r"^\d+_", "", parent_us)

    variants = [
        f"{parent}__{stem}_jpg",
        f"{parent_us}__{stem}_jpg",
        f"{parent_us}_{stem}_jpg",  # mask-like single join
        f"{parent_wo_prefix}__{stem}_jpg",
        f"{parent_us_wo_prefix}__{stem}_jpg",
        f"{stem}_jpg",
        rel_path,
    ]
    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
    return state_dict


def load_gals_checkpoint_compatible(gals_ckpt: Path, device: torch.device) -> wbsv.GALSBinaryCAMModel:
    model = wbsv.GALSBinaryCAMModel().to(device)
    ckpt = wbsv.torch.load(gals_ckpt, map_location=device)
    state = wbsv.extract_state_dict(ckpt)
    state = _strip_module_prefix(state)

    model_keys = list(model.state_dict().keys())
    if not model_keys:
        raise RuntimeError("Empty model state_dict for GALS model.")
    model_uses_net_prefix = model_keys[0].startswith("net.")

    if state:
        first_key = next(iter(state.keys()))
        ckpt_uses_net_prefix = first_key.startswith("net.")
        if model_uses_net_prefix and not ckpt_uses_net_prefix:
            state = {f"net.{k}": v for k, v in state.items()}
        elif (not model_uses_net_prefix) and ckpt_uses_net_prefix:
            state = {k[4:] if k.startswith("net.") else k: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(model.state_dict()) - len(missing)
    frac = float(loaded) / float(max(1, len(model.state_dict())))
    if frac < 0.80:
        raise RuntimeError(
            f"GALS checkpoint load coverage too low: loaded={loaded}/{len(model.state_dict())} "
            f"({100.0 * frac:.1f}%). Checkpoint likely mismatched: {gals_ckpt}"
        )
    if missing:
        print(f"[WARN] GALS missing keys ({len(missing)}).", flush=True)
    if unexpected:
        print(f"[WARN] GALS unexpected keys ({len(unexpected)}).", flush=True)
    model.eval()
    return model


def resolve_rows_from_tokens(
    metadata_df: pd.DataFrame,
    split_code: int,
    curated: Dict[str, List[str]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], Dict[str, object]]:
    val_df = metadata_df[metadata_df["split"].astype(int) == int(split_code)].copy()
    if val_df.empty:
        raise RuntimeError(f"No rows for split code {split_code} in metadata.csv")

    tok_to_rows: Dict[str, List[pd.Series]] = {}
    preview_tokens: List[str] = []
    for _, row in val_df.iterrows():
        rel = str(row["img_filename"])
        # Keep a small preview for debug.
        if len(preview_tokens) < 16:
            preview_tokens.append(canonical_token(rel))
        for tok in token_variants_from_relpath(rel):
            key = normalize_token_text(tok)
            tok_to_rows.setdefault(key, []).append(row)

    selected: List[Dict[str, object]] = []
    missing: List[Dict[str, str]] = []
    for category, tokens in curated.items():
        for tok in tokens:
            key = normalize_token_text(tok)
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
    debug = {
        "val_rows": int(len(val_df)),
        "lookup_keys": int(len(tok_to_rows)),
        "preview_canonical_tokens": preview_tokens,
    }
    return selected, missing, debug


def run_dataset(
    spec: DatasetSpec,
    out_dir: Path,
    target_class: str,
    rise_num_masks: int,
    rise_grid_size: int,
    rise_p1: float,
    rise_gpu_batch: int,
    rise_seed: int,
    device: torch.device,
) -> None:
    print(f"\n[DATASET {spec.tag}] data_path={spec.data_path}", flush=True)
    print(f"[DATASET {spec.tag}] gt_root={spec.gt_root}", flush=True)
    print(f"[DATASET {spec.tag}] guided_ckpt={spec.guided_ckpt}", flush=True)
    print(f"[DATASET {spec.tag}] vanilla_ckpt={spec.vanilla_ckpt}", flush=True)
    print(f"[DATASET {spec.tag}] gals_ckpt={spec.gals_ckpt}", flush=True)

    metadata_path = spec.data_path / "metadata.csv"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing metadata: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)

    # Validation only (user-specified)
    selected, missing, debug = resolve_rows_from_tokens(metadata_df, split_code=1, curated=spec.curated)
    print(
        f"[DATASET {spec.tag}] selected={len(selected)} missing={len(missing)} "
        f"(from val split only)",
        flush=True,
    )

    if not selected:
        raise RuntimeError(
            f"No curated images resolved for dataset {spec.tag}. "
            f"Debug: val_rows={debug['val_rows']} lookup_keys={debug['lookup_keys']} "
            f"preview={debug['preview_canonical_tokens']}"
        )

    ds_out = out_dir / f"waterbirds_{spec.tag}"
    samples_dir = ds_out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(metadata_df["y"].nunique())
    guided_model = wbsv.load_guided_model(spec.guided_ckpt, num_classes=num_classes, device=device)
    vanilla_model = wbsv.load_vanilla_model(spec.vanilla_ckpt, num_classes=num_classes, device=device)
    gals_model = load_gals_checkpoint_compatible(spec.gals_ckpt, device=device)

    rise_input_size = (224, 224)
    rise_masks_path = ds_out / (
        f"rise_masks_n{int(rise_num_masks)}_s{int(rise_grid_size)}_"
        f"p{str(rise_p1).replace('.', 'p')}_seed{int(rise_seed)}.npy"
    )
    shared_masks = wbsv.load_or_create_rise_masks(
        mask_path=rise_masks_path,
        num_masks=int(rise_num_masks),
        input_size=rise_input_size,
        grid_size=int(rise_grid_size),
        p1=float(rise_p1),
        seed=int(rise_seed),
        device=device,
    )

    guided_rise = wbsv.build_rise_explainer(
        prob_model=wbsv.GuidedProbModel(guided_model).to(device).eval(),
        masks=shared_masks,
        input_size=rise_input_size,
        num_classes=num_classes,
        gpu_batch=int(rise_gpu_batch),
        p1=float(rise_p1),
    )
    vanilla_rise = wbsv.build_rise_explainer(
        prob_model=wbsv.VanillaProbModel(vanilla_model).to(device).eval(),
        masks=shared_masks,
        input_size=rise_input_size,
        num_classes=num_classes,
        gpu_batch=int(rise_gpu_batch),
        p1=float(rise_p1),
    )
    gals_rise = wbsv.build_rise_explainer(
        prob_model=wbsv.GALSProbModel(gals_model).to(device).eval(),
        masks=shared_masks,
        input_size=rise_input_size,
        num_classes=2,
        gpu_batch=int(rise_gpu_batch),
        p1=float(rise_p1),
    )

    preprocess = wbsv.build_preprocess()
    use_label_target = str(target_class) == "label"

    rows_out: List[Dict[str, object]] = []
    for idx, item in enumerate(selected):
        rel = str(item["img_filename"])
        img_path = spec.data_path / rel
        if not img_path.is_file():
            continue

        token = str(item["token"])
        category = str(item["category"])
        label = int(item["y"])
        sample_name = f"{idx:03d}_{wbsv.safe_token(category)}__{wbsv.safe_token(token)}"
        sample_dir = samples_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_pil = wbsv.open_rgb_with_retry(img_path)
        image_rgb = np.array(image_pil, dtype=np.uint8)
        h, w = image_rgb.shape[:2]
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)

        g_pred, g_tgt, g_sal = wbsv._guided_predict_and_rise_saliency(
            input_tensor=input_tensor,
            label=label,
            use_label_target=use_label_target,
            guided_model=guided_model,
            guided_rise=guided_rise,
        )
        v_pred, v_tgt, v_sal = wbsv._vanilla_predict_and_rise_saliency(
            input_tensor=input_tensor,
            label=label,
            use_label_target=use_label_target,
            vanilla_model=vanilla_model,
            vanilla_rise=vanilla_rise,
        )
        ga_pred, ga_tgt, ga_sal = wbsv._gals_predict_and_rise_saliency(
            input_tensor=input_tensor,
            label=label,
            use_label_target=use_label_target,
            gals_model=gals_model,
            gals_rise=gals_rise,
        )

        wbsv.save_rgb(sample_dir / "image_rgb.png", image_rgb)
        g_vis = wbsv.save_saliency_variants("guided", g_sal, image_rgb, sample_dir)
        v_vis = wbsv.save_saliency_variants("vanilla", v_sal, image_rgb, sample_dir)
        ga_vis = wbsv.save_saliency_variants("gals_vit", ga_sal, image_rgb, sample_dir)
        wbsv.write_comparison_panels(
            sample_dir,
            vis_by_model={
                "guided": g_vis,
                "vanilla": v_vis,
                "gals_vit": ga_vis,
            },
        )

        gt_mask_path, gt_mask_default = wbsv._resolve_gt_mask_path(spec.gt_root, img_path, spec.data_path)
        has_mask = wbsv.save_gt_mask_variants(gt_mask_path, image_rgb, sample_dir) if gt_mask_path else False

        info = {
            "dataset": spec.tag,
            "category": category,
            "token": token,
            "img_filename": rel,
            "label": label,
            "group": int(item["group"]),
            "guided_pred": int(g_pred),
            "guided_target_class": int(g_tgt),
            "vanilla_pred": int(v_pred),
            "vanilla_target_class": int(v_tgt),
            "gals_vit_pred": int(ga_pred),
            "gals_vit_target_class": int(ga_tgt),
            "gt_mask_path": str(gt_mask_path) if has_mask and gt_mask_path is not None else None,
            "gt_mask_default_path": str(gt_mask_default),
        }
        with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        rows_out.append(info)

    summary = {
        "dataset": spec.tag,
        "data_path": str(spec.data_path),
        "gt_root": str(spec.gt_root),
        "target_class_mode": str(target_class),
        "num_requested": int(sum(len(v) for v in spec.curated.values())),
        "num_resolved": int(len(selected)),
        "num_generated": int(len(rows_out)),
        "num_missing": int(len(missing)),
        "missing_tokens": missing,
        "match_debug": debug,
        "guided_checkpoint": str(spec.guided_ckpt),
        "vanilla_checkpoint": str(spec.vanilla_ckpt),
        "gals_vit_checkpoint": str(spec.gals_ckpt),
        "rise": {
            "num_masks": int(rise_num_masks),
            "grid_size": int(rise_grid_size),
            "p1": float(rise_p1),
            "gpu_batch": int(rise_gpu_batch),
            "seed": int(rise_seed),
            "masks_path": str(rise_masks_path),
        },
    }

    with open(ds_out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows_out).to_csv(ds_out / "sample_index.csv", index=False)
    if missing:
        pd.DataFrame(missing).to_csv(ds_out / "missing_tokens.csv", index=False)

    print(f"[DATASET {spec.tag}] wrote: {ds_out}", flush=True)
    print(f"[DATASET {spec.tag}] generated {len(rows_out)} samples", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curated Waterbirds saliency from fixed checkpoints (validation-only).")
    p.add_argument("--run-wb95", action="store_true", default=True)
    p.add_argument("--run-wb100", action="store_true", default=True)
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="", help="e.g., cuda:0. If empty, auto-select.")

    p.add_argument(
        "--wb95-data-path",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
    )
    p.add_argument(
        "--wb100-data-path",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2",
    )
    p.add_argument(
        "--wb95-gt-root",
        default="/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    )
    p.add_argument(
        "--wb100-gt-root",
        default="/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    )

    p.add_argument(
        "--guided95-ckpt",
        default="/home/ryreu/guided_cnn/logsWaterbird/wb95_guided_vanilla_gals_saliency_21065978/checkpoints/guided/resnet50_final_kl295_attn109_20260226_174302.pth",
    )
    p.add_argument(
        "--vanilla95-ckpt",
        default="/home/ryreu/guided_cnn/logsWaterbird/wb95_guided_vanilla_gals_saliency_21065978/checkpoints/vanilla/vanilla_resnet50_seed0_20260226_190845.pth",
    )
    p.add_argument(
        "--gals95-ckpt",
        default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/gals_vit_fixed_waterbird_complete95_forest2water2_20260302_221759/best_balanced_valacc_0.88_epoch_135.ckpt",
    )

    p.add_argument(
        "--guided100-ckpt",
        default="/home/ryreu/guided_cnn/logsWaterbird/wb100_guided_vanilla_gals_saliency_21065977/checkpoints/guided/resnet50_final_kl495_attn73_20260226_091559.pth",
    )
    p.add_argument(
        "--vanilla100-ckpt",
        default="/home/ryreu/guided_cnn/logsWaterbird/wb100_guided_vanilla_gals_saliency_21065977/checkpoints/vanilla/vanilla_resnet50_seed0_20260226_104014.pth",
    )
    p.add_argument(
        "--gals100-ckpt",
        default="/home/ryreu/guided_cnn/waterbirds/Waterbird_Runs/GALS/trained_weights/waterbirds/gals_vit_fixed_waterbird_1_0_forest2water2_20260302_163408/best_balanced_valacc_0.73_epoch_2.ckpt",
    )

    p.add_argument("--rise-num-masks", type=int, default=2000)
    p.add_argument("--rise-grid-size", type=int, default=8)
    p.add_argument("--rise-p1", type=float, default=0.1)
    p.add_argument("--rise-gpu-batch", type=int, default=16)
    p.add_argument("--rise-seed", type=int, default=0)

    p.add_argument(
        "--output-dir",
        default="",
        help="Output root. If empty: /home/ryreu/guided_cnn/logsWaterbird/waterbirds_curated_saliency_<timestamp>",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("/home/ryreu/guided_cnn/logsWaterbird") / f"waterbirds_curated_saliency_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(str(args.device))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] output_dir={out_dir}", flush=True)
    print(f"[INFO] device={device}", flush=True)
    print("[INFO] methods=guided,vanilla,gals (CLIP ZS/LR intentionally excluded)", flush=True)

    specs: List[DatasetSpec] = []
    if args.run_wb95:
        specs.append(
            DatasetSpec(
                tag="95",
                data_path=Path(args.wb95_data_path).expanduser().resolve(),
                gt_root=Path(args.wb95_gt_root).expanduser().resolve(),
                guided_ckpt=Path(args.guided95_ckpt).expanduser().resolve(),
                vanilla_ckpt=Path(args.vanilla95_ckpt).expanduser().resolve(),
                gals_ckpt=Path(args.gals95_ckpt).expanduser().resolve(),
                curated=WB95_CURATED,
            )
        )
    if args.run_wb100:
        specs.append(
            DatasetSpec(
                tag="100",
                data_path=Path(args.wb100_data_path).expanduser().resolve(),
                gt_root=Path(args.wb100_gt_root).expanduser().resolve(),
                guided_ckpt=Path(args.guided100_ckpt).expanduser().resolve(),
                vanilla_ckpt=Path(args.vanilla100_ckpt).expanduser().resolve(),
                gals_ckpt=Path(args.gals100_ckpt).expanduser().resolve(),
                curated=WB100_CURATED,
            )
        )
    if not specs:
        raise RuntimeError("No datasets selected. Use --run-wb95 and/or --run-wb100.")

    for s in specs:
        for p in [s.data_path, s.gt_root]:
            if not p.exists():
                raise RuntimeError(f"Missing path: {p}")
        for p in [s.guided_ckpt, s.vanilla_ckpt, s.gals_ckpt]:
            if not p.is_file():
                raise RuntimeError(f"Missing checkpoint: {p}")

    for spec in specs:
        run_dataset(
            spec=spec,
            out_dir=out_dir,
            target_class=args.target_class,
            rise_num_masks=int(args.rise_num_masks),
            rise_grid_size=int(args.rise_grid_size),
            rise_p1=float(args.rise_p1),
            rise_gpu_batch=int(args.rise_gpu_batch),
            rise_seed=int(args.rise_seed),
            device=device,
        )

    print("\n[DONE] Curated saliency generation complete.", flush=True)
    print(f"[DONE] Root output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

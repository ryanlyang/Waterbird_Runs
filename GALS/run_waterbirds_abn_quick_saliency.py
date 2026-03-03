#!/usr/bin/env python3
"""Quick ABN saliency dump for Waterbirds validation images."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import waterbirds_pointing_game_eval as wbe
import waterbirds100_guided_vanilla_saliency as wbsv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ABN saliency maps for a random subset of val images.")
    p.add_argument("--data-path", required=True, help="Waterbirds dataset dir with metadata.csv")
    p.add_argument("--abn-ckpt", required=True, help="ABN checkpoint path")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--target-mode", choices=["label", "pred"], default="label")
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    ckpt = Path(args.abn_ckpt).expanduser().resolve()
    if not data_path.is_dir():
        raise RuntimeError(f"Missing data-path: {data_path}")
    if not ckpt.is_file():
        raise RuntimeError(f"Missing abn-ckpt: {ckpt}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("/home/ryreu/guided_cnn/logsWaterbird") / f"abn_quick_saliency_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    md = pd.read_csv(data_path / "metadata.csv")
    split_id = wbe.SPLIT_MAP[str(args.split)]
    md = md[md["split"].astype(int) == int(split_id)].copy()
    if md.empty:
        raise RuntimeError(f"No rows in split={args.split}")
    n = min(int(args.num_samples), len(md))
    md = md.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = int(pd.read_csv(data_path / "metadata.csv")["y"].nunique())
    runner = wbe.ABNRunner(checkpoint=ckpt, num_classes=num_classes, device=device)
    preprocess = wbe.build_preprocess()

    rows: List[Dict[str, object]] = []
    try:
        for i, row in md.iterrows():
            rel = str(row["img_filename"])
            img_path = data_path / rel
            if not img_path.is_file():
                continue

            pil = wbe.open_pil_with_retry(img_path, mode="RGB")
            rgb = np.array(pil, dtype=np.uint8)
            h, w = rgb.shape[:2]
            x = preprocess(pil).unsqueeze(0).to(device)
            label = int(row["y"])

            pred, target, sal = runner.predict_and_saliency(
                image_tensor=x,
                label=label,
                target_mode=str(args.target_mode),
                pil_image=pil,
            )
            sal = wbe.upsample_saliency(sal, h, w)

            sample_dir = samples_dir / f"{i:03d}_{wbsv.safe_token(Path(rel).parent.name)}__{wbsv.safe_token(Path(rel).stem)}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            wbsv.save_rgb(sample_dir / "image_rgb.png", rgb)
            wbsv.save_saliency_variants("abn", sal, rgb, sample_dir)

            info = {
                "img_filename": rel,
                "label": label,
                "place": int(row["place"]),
                "group": int(label * 2 + int(row["place"])),
                "pred": int(pred),
                "target_for_saliency": int(target),
            }
            with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            rows.append(info)
    finally:
        runner.close()

    pd.DataFrame(rows).to_csv(out_dir / "sample_index.csv", index=False)
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(data_path),
                "abn_checkpoint": str(ckpt),
                "split": str(args.split),
                "target_mode": str(args.target_mode),
                "num_requested": int(args.num_samples),
                "num_generated": int(len(rows)),
            },
            f,
            indent=2,
        )
    print(f"[DONE] Generated {len(rows)} ABN saliency samples at: {out_dir}")


if __name__ == "__main__":
    main()


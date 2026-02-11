#!/usr/bin/env python3
"""
Prepare Food-101 Red Meat subset metadata expected by GALS.

This script creates:
  1) <food_root>/meta/all_images.csv with columns:
       - abs_file_path  (relative to <food_root>, e.g. images/steak/123.jpg)
       - label          (class string)
       - split          (train|val|test)
  2) Optional symlink <food_root>/train -> <food_root>/images
     (needed because datasets/food.py initializes ImageFolder on food-101/train).

The split matches the paper setup:
  - Use official Food-101 test split as test (250/class).
  - Split official Food-101 train split (750/class) into:
      train: 500/class
      val:   250/class
"""

import argparse
import csv
import os
import random
from pathlib import Path


DEFAULT_CLASSES = [
    "prime_rib",
    "pork_chop",
    "steak",
    "baby_back_ribs",
    "filet_mignon",
]


def read_split_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line)
    return out


def to_rel_image_path(item: str):
    # Food-101 split txt entries are usually "class_name/image_id" (without .jpg).
    # Keep compatible if .jpg is already present.
    return f"images/{item}.jpg" if not item.endswith(".jpg") else f"images/{item}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--food-root",
        required=True,
        help="Path to food-101 directory (contains images/ and meta/).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/val split.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Subset classes to keep.",
    )
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--val-per-class", type=int, default=250)
    parser.add_argument(
        "--ensure-train-link",
        action="store_true",
        default=True,
        help="Ensure <food-root>/train points to <food-root>/images for ImageFolder init.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    food_root = Path(args.food_root).expanduser().resolve()
    meta_dir = food_root / "meta"
    images_dir = food_root / "images"
    train_txt = meta_dir / "train.txt"
    test_txt = meta_dir / "test.txt"
    out_csv = meta_dir / "all_images.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    train_items = read_split_file(train_txt)
    test_items = read_split_file(test_txt)

    classes = sorted(args.classes)
    records = []

    for cls in classes:
        cls_train = [x for x in train_items if x.startswith(f"{cls}/")]
        cls_test = [x for x in test_items if x.startswith(f"{cls}/")]

        needed_trainval = args.train_per_class + args.val_per_class
        if len(cls_train) < needed_trainval:
            raise ValueError(
                f"Class '{cls}' has only {len(cls_train)} train items; "
                f"need at least {needed_trainval}."
            )
        if len(cls_test) == 0:
            raise ValueError(f"Class '{cls}' has no test items in {test_txt}.")

        rng.shuffle(cls_train)
        train_split = cls_train[: args.train_per_class]
        val_split = cls_train[args.train_per_class : args.train_per_class + args.val_per_class]

        for item in train_split:
            records.append(
                {
                    "abs_file_path": to_rel_image_path(item),
                    "label": cls,
                    "split": "train",
                }
            )
        for item in val_split:
            records.append(
                {
                    "abs_file_path": to_rel_image_path(item),
                    "label": cls,
                    "split": "val",
                }
            )
        for item in cls_test:
            records.append(
                {
                    "abs_file_path": to_rel_image_path(item),
                    "label": cls,
                    "split": "test",
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["abs_file_path", "label", "split"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote: {out_csv}")
    print(f"Num rows: {len(records)}")
    for split in ["train", "val", "test"]:
        n = sum(1 for r in records if r["split"] == split)
        print(f"{split}: {n}")

    if args.ensure_train_link:
        train_link = food_root / "train"
        if train_link.exists() or train_link.is_symlink():
            if train_link.is_symlink():
                target = os.readlink(train_link)
                print(f"Existing symlink kept: {train_link} -> {target}")
            else:
                print(f"Path exists (not symlink), kept as-is: {train_link}")
        else:
            train_link.symlink_to(images_dir, target_is_directory=True)
            print(f"Created symlink: {train_link} -> {images_dir}")


if __name__ == "__main__":
    main()

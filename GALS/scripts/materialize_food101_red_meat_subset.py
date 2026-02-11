#!/usr/bin/env python3
"""
Materialize a clean Red Meat subset from a full Food-101 download.

Creates a new dataset tree containing only 5 classes, plus metadata expected by GALS:
  <output_food_root>/
    images/<class>/<image>.jpg
    meta/all_images.csv
    train -> images         (symlink used by GALS/datasets/food.py init)
    split_images/           (optional convenience layout)
      train/<class>/<image>.jpg
      val/<class>/<image>.jpg
      test/<class>/<image>.jpg

Split policy (paper-style for Red Meat subset):
  - For each class:
      Food-101 train.txt: 750 images -> 500 train, 250 val
      Food-101 test.txt : 250 images -> 250 test
"""

import argparse
import csv
import random
import shutil
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
            if line:
                out.append(line)
    return out


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def transfer_file(src: Path, dst: Path, mode: str):
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def maybe_make_train_link(output_food_root: Path):
    train_link = output_food_root / "train"
    images_dir = output_food_root / "images"
    if train_link.exists() or train_link.is_symlink():
        if train_link.is_symlink():
            return
        raise FileExistsError(
            f"Path exists and is not a symlink: {train_link}. "
            "Remove it or use a different output path."
        )
    train_link.symlink_to(images_dir, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-food-root",
        required=True,
        help="Path to full food-101 root (contains images/ and meta/train.txt/test.txt).",
    )
    parser.add_argument(
        "--output-food-root",
        required=True,
        help="Path to output food-101 root to create (subset only).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--val-per-class", type=int, default=250)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Subset classes to keep.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to materialize image files.",
    )
    parser.add_argument(
        "--create-split-images",
        action="store_true",
        default=True,
        help="Also create split_images/{train,val,test}/<class>/... for convenience.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src_root = Path(args.source_food_root).expanduser().resolve()
    out_root = Path(args.output_food_root).expanduser().resolve()

    src_images = src_root / "images"
    src_meta = src_root / "meta"
    train_txt = src_meta / "train.txt"
    test_txt = src_meta / "test.txt"

    if not src_images.exists():
        raise FileNotFoundError(f"Missing source images dir: {src_images}")
    train_items = read_split_file(train_txt)
    test_items = read_split_file(test_txt)

    out_images = out_root / "images"
    out_meta = out_root / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)

    classes = sorted(args.classes)
    records = []

    for cls in classes:
        cls_train = [x for x in train_items if x.startswith(f"{cls}/")]
        cls_test = [x for x in test_items if x.startswith(f"{cls}/")]

        needed_trainval = args.train_per_class + args.val_per_class
        if len(cls_train) < needed_trainval:
            raise ValueError(
                f"Class '{cls}' has {len(cls_train)} train images; "
                f"need >= {needed_trainval}."
            )
        if len(cls_test) == 0:
            raise ValueError(f"Class '{cls}' has 0 test images in {test_txt}.")

        rng.shuffle(cls_train)
        split_train = cls_train[: args.train_per_class]
        split_val = cls_train[args.train_per_class : args.train_per_class + args.val_per_class]

        split_map = {
            "train": split_train,
            "val": split_val,
            "test": cls_test,
        }

        for split_name, items in split_map.items():
            for item in items:
                rel_image = Path("images") / f"{item}.jpg"
                src_img = src_root / rel_image
                dst_img = out_root / rel_image
                if not src_img.exists():
                    raise FileNotFoundError(f"Missing source image: {src_img}")

                transfer_file(src_img, dst_img, args.mode)

                if args.create_split_images:
                    split_dst = out_root / "split_images" / split_name / f"{item}.jpg"
                    transfer_file(src_img, split_dst, args.mode)

                records.append(
                    {
                        "abs_file_path": rel_image.as_posix(),
                        "label": cls,
                        "split": split_name,
                    }
                )

    # Write csv expected by GALS datasets/food.py
    out_csv = out_meta / "all_images.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["abs_file_path", "label", "split"])
        writer.writeheader()
        writer.writerows(records)

    # Helpful labels file (not strictly required if CLASSES is set in config).
    labels_txt = out_meta / "labels.txt"
    with open(labels_txt, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(f"{c}\n")

    # GALS/datasets/food.py currently initializes ImageFolder at food-101/train.
    maybe_make_train_link(out_root)

    counts = {k: 0 for k in ["train", "val", "test"]}
    for r in records:
        counts[r["split"]] += 1

    print(f"Created subset at: {out_root}")
    print(f"Mode: {args.mode}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {labels_txt}")
    print(f"train rows: {counts['train']}")
    print(f"val rows:   {counts['val']}")
    print(f"test rows:  {counts['test']}")
    print(f"total rows: {len(records)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import copy
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Import shared guided-CNN training/model utilities from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import run_guided_waterbird as base  # noqa: E402


batch_size = 96
num_epochs = 150
base_lr = 0.01
classifier_lr = 0.01
lr2_mult = 1.0
momentum = 0.9
weight_decay = 1e-5

checkpoint_dir = "RedMeat_Guided_Checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0


def _resolve_img_path(data_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    rel = str(rel_or_abs).lstrip("/")
    return os.path.join(data_root, rel)


def _sanitize_label(label: str) -> str:
    return str(label).strip().replace(" ", "_").replace("/", "_").lower()


def _mask_candidates(mask_root: str, label_name: str, image_path: str):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    parent_name = os.path.basename(os.path.dirname(image_path)).replace(".", "_")
    label_key = _sanitize_label(label_name)

    direct = [
        f"{label_key}_{base_name}.png",
        f"{label_key}_{base_name}.jpg",
        f"{label_key}_{base_name}.jpeg",
        f"{parent_name}_{base_name}.png",
        f"{parent_name}_{base_name}.jpg",
        f"{parent_name}_{base_name}.jpeg",
        f"{base_name}.png",
    ]
    nested = [
        os.path.join(label_key, f"{base_name}.png"),
        os.path.join(label_key, f"{base_name}.jpg"),
        os.path.join(label_key, f"{base_name}.jpeg"),
        os.path.join(parent_name, f"{base_name}.png"),
        os.path.join(parent_name, f"{base_name}.jpg"),
        os.path.join(parent_name, f"{base_name}.jpeg"),
    ]

    for rel in direct + nested:
        yield os.path.join(mask_root, rel)


class RedMeatMetadataDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_transform=None,
        mask_root: str = None,
        mask_transform=None,
        return_mask: bool = False,
        return_path: bool = True,
        classes=None,
        split_col: str = "split",
        label_col: str = "label",
        path_col: str = "abs_file_path",
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        self.return_mask = return_mask
        self.return_path = return_path

        meta_path = os.path.join(self.data_root, "all_images.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing all_images.csv at: {meta_path}")

        df = pd.read_csv(meta_path)
        for c in (split_col, label_col, path_col):
            if c not in df.columns:
                raise KeyError(f"Missing column '{c}' in {meta_path}. Columns={list(df.columns)}")

        if classes is None:
            classes = sorted(df[label_col].astype(str).unique().tolist())
        self.classes = [str(c) for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        sdf = df[df[split_col].astype(str) == str(split)]
        if len(sdf) == 0:
            raise ValueError(f"Split '{split}' is empty in {meta_path}")

        label_names = sdf[label_col].astype(str).tolist()
        unknown = sorted(set(label_names) - set(self.class_to_idx.keys()))
        if unknown:
            raise ValueError(f"Labels not in class list for split '{split}': {unknown}")

        self.label_names = label_names
        self.labels = np.array([self.class_to_idx[x] for x in label_names], dtype=np.int64)
        self.paths = [_resolve_img_path(self.data_root, p) for p in sdf[path_col].astype(str).tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])

        img = Image.open(path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        out = [img, label]

        if self.return_mask:
            if self.mask_root is None:
                raise ValueError("mask_root is required when return_mask=True")

            label_name = self.label_names[idx]
            mask_path = None
            tried = []
            for cand in _mask_candidates(self.mask_root, label_name, path):
                tried.append(cand)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                preview = "\n  - ".join(tried[:6])
                raise FileNotFoundError(
                    f"No mask found for image '{path}'. Tried examples:\n  - {preview}"
                )

            mask = Image.open(mask_path).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            out.append(mask)

        if self.return_path:
            out.append(path)

        return tuple(out)


def evaluate_test(model, test_loader, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    total_loss = 0.0
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels, _paths in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            labels_cpu = labels.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            for cls in range(num_classes):
                cls_mask = labels_cpu == cls
                if np.any(cls_mask):
                    class_correct[cls] += np.sum(preds_cpu[cls_mask] == labels_cpu[cls_mask])
                    class_total[cls] += np.sum(cls_mask)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)

    cls_acc = class_correct / np.maximum(class_total, 1)
    per_group = 100.0 * float(np.mean(cls_acc))
    worst_group = 100.0 * float(np.min(cls_acc))

    return avg_loss, acc, cls_acc * 100.0, per_group, worst_group


def run_single(args, attn_epoch, kl_value, kl_increment=None):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_attention = attn_epoch < num_epochs and kl_value > 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }
    mask_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                base.Brighten(8.0),
            ]
        )
    }

    base.seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = RedMeatMetadataDataset(
        data_root=args.data_path,
        split="train",
        image_transform=data_transforms["train"],
        mask_root=args.gt_path,
        mask_transform=mask_transforms["train"],
        return_mask=use_attention,
        return_path=True,
        classes=args.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )
    val_dataset = RedMeatMetadataDataset(
        data_root=args.data_path,
        split="val",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )
    test_dataset = RedMeatMetadataDataset(
        data_root=args.data_path,
        split="test",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        classes=train_dataset.classes,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
    )

    num_classes = len(train_dataset.classes)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=base.seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=base.seed_worker,
            generator=g,
        ),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
    }

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=base.seed_worker,
        generator=g,
    )

    model = base.make_cam_model(num_classes, model_name="resnet50", pretrained=True).to(device)

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n=== RUN: kl_lambda={kl_value}, attention_epoch={attn_epoch} ===", flush=True)
    if kl_increment is None:
        kl_increment = kl_value / 10.0

    best_model, best_score, best_epoch = base.train_model(
        model,
        dataloaders,
        dataset_sizes,
        attn_epoch,
        kl_value,
        num_epochs,
        base_lr=base_lr,
        classifier_lr=classifier_lr,
        lr2_mult=lr2_mult,
        kl_incr=kl_increment,
        use_attention=use_attention,
        num_classes=num_classes,
    )
    print(f"\n[VAL] Best Balanced Acc: {best_score:.4f} at epoch {best_epoch}")

    test_loss, test_acc, class_acc, per_group, worst_group = evaluate_test(best_model, test_loader, num_classes)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")
    for cls_name, acc in zip(train_dataset.classes, class_acc):
        print(f"[TEST] {cls_name}: {acc:.2f}%")
    print(f"[TEST] Per-class mean: {per_group:.2f}%  Worst-class: {worst_group:.2f}%")

    if save_checkpoints:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"resnet50_redmeat_final_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
        save_path = os.path.join(checkpoint_dir, save_name)
        torch.save(best_model.state_dict(), save_path)
    else:
        save_path = "NONE"
        print("[RUN DONE] Checkpoint saving disabled via SAVE_CHECKPOINTS=0", flush=True)

    print(
        f"[RUN DONE] kl={kl_value} attn={attn_epoch} lr2_mult={lr2_mult} kl_incr={kl_increment} "
        f"| best_balanced_val_acc={best_score:.4f} | test_acc={test_acc:.2f}% | saved: {save_path}",
        flush=True,
    )

    return float(best_score), float(test_acc), float(per_group), float(worst_group), save_path


def main():
    global SEED, base_lr, classifier_lr, lr2_mult, num_epochs

    p = argparse.ArgumentParser(description="Guided RedMeat runner (ResNet50 + KL guidance to GT masks).")
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("gt_path", help="GT mask root (expects class_image flat naming by default)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--attention-epoch", type=int, default=num_epochs)
    p.add_argument("--kl-lambda", type=float, default=0.0)
    p.add_argument("--kl-increment", type=float, default=None)
    p.add_argument("--base_lr", type=float, default=base_lr)
    p.add_argument("--classifier_lr", type=float, default=classifier_lr)
    p.add_argument("--lr2-mult", type=float, default=lr2_mult)
    p.add_argument("--num-epochs", type=int, default=num_epochs)

    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated class list. Empty string = infer from metadata.",
    )

    args = p.parse_args()

    SEED = int(args.seed)
    base_lr = float(args.base_lr)
    classifier_lr = float(args.classifier_lr)
    lr2_mult = float(args.lr2_mult)
    num_epochs = int(args.num_epochs)

    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()] if args.classes else None

    run_args = argparse.Namespace(
        data_path=args.data_path,
        gt_path=args.gt_path,
        split_col=args.split_col,
        label_col=args.label_col,
        path_col=args.path_col,
        classes=classes,
    )
    run_single(run_args, int(args.attention_epoch), float(args.kl_lambda), args.kl_increment)


if __name__ == "__main__":
    main()

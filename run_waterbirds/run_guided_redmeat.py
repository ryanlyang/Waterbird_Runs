import os
import argparse
from datetime import datetime
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import run_guided_waterbird as base


# Fixed setup requested for Red Meat guided runs.
batch_size = 96
num_epochs = 150
base_lr = 0.01
classifier_lr = 0.01
lr2_mult = 1.0
momentum = 0.9
weight_decay = 1e-5

checkpoint_dir = "ResNet_Checkpoints_RedMeat"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Brighten(object):
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.clamp(mask * self.factor, 0.0, 1.0)


def mask_name_from_path(image_path: str):
    # WeCLIPPlus prediction_cmap flat naming convention.
    # images/<class_name>/<orig>.jpg -> <class_name>_<orig>.png
    cls = os.path.basename(os.path.dirname(image_path)).replace(".", "_").replace(" ", "_")
    base = os.path.splitext(os.path.basename(image_path))[0]
    return f"{cls}_{base}.png"


def resolve_mask_path(image_path: str, mask_root: str):
    cls = os.path.basename(os.path.dirname(image_path))
    cls_norm = cls.replace(".", "_").replace(" ", "_")
    base = os.path.splitext(os.path.basename(image_path))[0]
    stem = f"{cls_norm}_{base}"

    candidates = [
        os.path.join(mask_root, f"{stem}.png"),
        os.path.join(mask_root, f"{stem}.jpg"),
        os.path.join(mask_root, f"{stem}.jpeg"),
        os.path.join(mask_root, f"{base}.png"),
        os.path.join(mask_root, cls_norm, f"{base}.png"),
        os.path.join(mask_root, cls, f"{base}.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def _select_split(df: pd.DataFrame, split_col: str, split: str):
    vals = df[split_col]
    if pd.api.types.is_numeric_dtype(vals):
        split_map = {"train": 0, "val": 1, "test": 2}
        if split not in split_map:
            raise ValueError(f"Unsupported split '{split}' for numeric split column '{split_col}'")
        return df[vals.astype(int) == split_map[split]]
    return df[vals.astype(str).str.lower() == str(split).lower()]


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
        split_col: str = "split",
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.mask_root = mask_root
        self.mask_transform = mask_transform
        self.return_mask = return_mask
        self.return_path = return_path
        self.split_col = split_col

        csv_path = os.path.join(self.data_root, "meta", "all_images.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing metadata csv: {csv_path}")

        all_df = pd.read_csv(csv_path)
        required = {"abs_file_path", "label", split_col}
        missing_cols = required - set(all_df.columns.tolist())
        if missing_cols:
            raise KeyError(f"Missing required columns in {csv_path}: {sorted(missing_cols)}")

        df = _select_split(all_df, split_col=split_col, split=split)
        if len(df) == 0:
            raise ValueError(f"No rows for split='{split}' in {csv_path} using split_col='{split_col}'")

        self.class_names = sorted(all_df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.rel_paths = df["abs_file_path"].tolist()
        self.paths = [os.path.join(self.data_root, p) for p in self.rel_paths]
        self.labels = np.array([self.class_to_idx[l] for l in df["label"].tolist()], dtype=np.int64)

        self.mask_paths = None
        if self.return_mask:
            if not self.mask_root:
                raise ValueError("mask_root must be provided when return_mask=True")
            self.mask_paths = [resolve_mask_path(p, self.mask_root) for p in self.paths]
            missing = [m for m in self.mask_paths if not os.path.exists(m)]
            if missing:
                sample_img = self.paths[self.mask_paths.index(missing[0])]
                sample_expected = missing[0]
                expected_name = mask_name_from_path(sample_img)
                raise FileNotFoundError(
                    f"Missing {len(missing)} masks under {self.mask_root}. "
                    f"Example image: {sample_img}\n"
                    f"Expected flat mask name: {expected_name}\n"
                    f"Expected path (first candidate): {sample_expected}"
                )

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
            mask = Image.open(self.mask_paths[idx]).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            out.append(mask)

        if self.return_path:
            out.append(path)
        return tuple(out)


@torch.no_grad()
def evaluate_test_classwise(model, test_loader, class_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0

    num_classes = len(class_names)
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    for batch in test_loader:
        images, labels, _paths = batch
        images = images.to(device)
        labels = labels.to(device).long()

        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        labels_cpu = labels.detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()
        for cls in range(num_classes):
            mask = labels_cpu == cls
            if np.any(mask):
                class_correct[cls] += np.sum(preds_cpu[mask] == labels_cpu[mask])
                class_total[cls] += np.sum(mask)

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    class_acc = 100.0 * (class_correct / np.maximum(class_total, 1))
    balanced = float(np.mean(class_acc))
    worst = float(np.min(class_acc))
    return avg_loss, acc, class_acc, balanced, worst


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
                Brighten(float(getattr(args, "mask_brighten", 8.0))),
            ]
        )
    }

    seed_everything(SEED)
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
        split_col=getattr(args, "split_col", "split"),
    )
    val_dataset = RedMeatMetadataDataset(
        data_root=args.data_path,
        split="val",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        split_col=getattr(args, "split_col", "split"),
    )
    test_dataset = RedMeatMetadataDataset(
        data_root=args.data_path,
        split="test",
        image_transform=data_transforms["eval"],
        return_mask=False,
        return_path=True,
        split_col=getattr(args, "split_col", "split"),
    )

    num_classes = int(len(train_dataset.class_names))
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = base.make_cam_model(num_classes, model_name="resnet50", pretrained=True).to(device)

    # base.train_model uses these globals from run_guided_waterbird.py.
    base.momentum = momentum
    base.weight_decay = weight_decay

    save_checkpoints = os.environ.get("SAVE_CHECKPOINTS", "1").lower() not in ("0", "false", "no", "n")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n=== RUN (Guided RedMeat): kl_lambda={kl_value}, attention_epoch={attn_epoch} ===", flush=True)
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

    test_loss, test_acc, class_acc, balanced_test, worst_class = evaluate_test_classwise(
        best_model, test_loader, train_dataset.class_names
    )
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")
    for i, name in enumerate(train_dataset.class_names):
        print(f"[TEST] {name}: {class_acc[i]:.2f}%")
    print(f"[TEST] Balanced(test): {balanced_test:.2f}%  Worst-class: {worst_class:.2f}%")

    if save_checkpoints:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"resnet50_redmeat_guided_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
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
    # Keep same style as waterbird scripts: (val score, test acc, extra metric 1, extra metric 2, ckpt)
    return float(best_score), float(test_acc), float(balanced_test), float(worst_class), save_path


def main():
    global SEED, base_lr, classifier_lr, lr2_mult

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="RedMeat root (expects meta/all_images.csv and images/)")
    parser.add_argument("gt_path", help="Mask root (prediction_cmap). Supports flat naming class_image.png")
    parser.add_argument("--split-col", default="split", help="Column name for split in all_images.csv")
    parser.add_argument("--mask-brighten", type=float, default=8.0, help="Mask intensity multiplier before KL")

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--attention_epoch", type=int, default=num_epochs, help=">= num_epochs disables attention")
    parser.add_argument("--kl_lambda", type=float, default=0.0, help="Weight for attention KL loss")
    parser.add_argument("--kl_increment", type=float, default=None, help="Increment added to KL each epoch after attention_epoch")
    parser.add_argument("--base_lr", type=float, default=base_lr, help="Base learning rate")
    parser.add_argument("--classifier_lr", type=float, default=classifier_lr, help="Classifier learning rate")
    parser.add_argument("--lr2_mult", type=float, default=lr2_mult, help="LR multiplier after attention_epoch restart")
    args = parser.parse_args()

    SEED = args.seed
    base_lr = args.base_lr
    classifier_lr = args.classifier_lr
    lr2_mult = args.lr2_mult

    run_args = SimpleNamespace(
        data_path=args.data_path,
        gt_path=args.gt_path,
        split_col=args.split_col,
        mask_brighten=args.mask_brighten,
    )
    run_single(run_args, args.attention_epoch, args.kl_lambda, args.kl_increment)


if __name__ == "__main__":
    main()

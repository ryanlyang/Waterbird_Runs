#!/usr/bin/env python3
"""Fixed-hyperparameter DecoyMNIST LeNet with GALS-style RRR loss.

This is a decoy-style LeNet run that keeps the standard Decoy optimizer setup
(Adam, lr=0.001, weight_decay=1e-4) and adds an input-gradient suppression
term outside an external attention mask.
"""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor
from tqdm.auto import tqdm


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _load_attention_map(path: Path) -> torch.Tensor:
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True), which
    # rejects non-tensor payloads used by these saved attention maps.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        payload = torch.load(path, map_location="cpu")
    arr = None
    if isinstance(payload, dict):
        for key in ("attentions", "unnormalized_attentions", "attention", "cam", "saliency"):
            if key in payload:
                arr = payload[key]
                break
    else:
        arr = payload
    if arr is None:
        raise ValueError(f"Could not parse attention payload at: {path}")

    att = torch.as_tensor(arr, dtype=torch.float32)
    while att.ndim > 2:
        if att.ndim == 3:
            att = att.max(dim=0).values
        else:
            att = att.squeeze(0)
    if att.ndim != 2:
        raise ValueError(f"Expected 2D attention after reduction, got shape {tuple(att.shape)} at {path}")

    mn = float(att.min())
    mx = float(att.max())
    if mx > mn:
        att = (att - mn) / (mx - mn)
    else:
        att = torch.zeros_like(att)
    return att.unsqueeze(0)  # 1 x H x W


class GuidedImageFolder(utils.Dataset):
    def __init__(self, png_root: str, mask_root: str, split: str, image_transform=None) -> None:
        self.split = split
        self.split_img_root = Path(png_root) / split
        self.split_mask_root = Path(mask_root) / split
        self.images = ImageFolder(str(self.split_img_root), transform=image_transform)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img, label = self.images[idx]
        img_path, _ = self.images.samples[idx]
        rel = Path(img_path).resolve().relative_to(self.split_img_root.resolve())
        mask_path = (self.split_mask_root / rel).with_suffix(".pth")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path}: {mask_path}")
        mask = _load_attention_map(mask_path)
        return img, label, mask


@torch.no_grad()
def evaluate(model: nn.Module, loader: utils.DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        loss_sum += F.nll_loss(out, target, reduction="sum").item()
        correct += out.argmax(dim=1).eq(target).sum().item()
        total += data.size(0)
    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def train_one_seed(args, seed: int, full_train: GuidedImageFolder, test_dataset: ImageFolder, device, loader_kwargs):
    set_seed(seed)

    # Keep split fixed across seeds (matches cdepstyle behavior used in Decoy runners).
    split_g = torch.Generator().manual_seed(0)
    n_total = len(full_train)
    n_val = int(args.val_frac * n_total)
    n_train = n_total - n_val
    train_subset, val_subset = utils.random_split(full_train, [n_train, n_val], generator=split_g)

    train_loader = utils.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = utils.DataLoader(val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.grad_criterion == "L1":
        grad_criterion = nn.L1Loss()
    else:
        grad_criterion = nn.MSELoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_iter = train_loader
        if not args.no_progress_bar:
            epoch_iter = tqdm(
                train_loader,
                desc=f"seed={seed} epoch={epoch}/{args.epochs}",
                leave=False,
                dynamic_ncols=True,
            )

        for data, target, gt_mask in epoch_iter:
            data = data.to(device)
            target = target.to(device)
            gt_mask = gt_mask.to(device)
            gt_mask = F.interpolate(gt_mask, size=data.shape[-2:], mode="nearest")

            data.requires_grad_(True)
            optimizer.zero_grad()

            out = model(data)
            cls_loss = F.nll_loss(out, target)
            dy_dx = torch.autograd.grad(cls_loss, data, create_graph=True)[0]
            rrr_loss = grad_criterion(dy_dx, dy_dx * gt_mask)
            loss = cls_loss + args.grad_weight * rrr_loss

            loss.backward()
            optimizer.step()

            if not args.no_progress_bar:
                epoch_iter.set_postfix(loss=f"{float(loss.item()):.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())

        if args.print_every > 0 and (epoch % args.print_every == 0 or epoch == args.epochs):
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
            )

    model.load_state_dict(best_weights)
    _, test_acc = evaluate(model, test_loader, device)
    return best_val_acc, test_acc, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="DecoyMNIST LeNet with fixed GALS-style RRR loss")
    parser.add_argument(
        "--png-root",
        type=str,
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png",
    )
    parser.add_argument(
        "--mask-root",
        type=str,
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam",
    )
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-weight", type=float, default=72503.48035960984)
    parser.add_argument("--grad-criterion", choices=["L1", "L2"], default="L1")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--no-progress-bar", action="store_true", default=False)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    full_train = GuidedImageFolder(
        png_root=args.png_root,
        mask_root=args.mask_root,
        split="train",
        image_transform=transform,
    )
    test_dataset = ImageFolder(os.path.join(args.png_root, "test"), transform=transform)

    print("Running DecoyMNIST fixed GALS-RRR")
    print(f"device={device}")
    print(f"png_root={args.png_root}")
    print(f"mask_root={args.mask_root}")
    print(f"train={len(full_train)} test={len(test_dataset)} split={1.0 - args.val_frac:.2f}/{args.val_frac:.2f}")
    print(
        f"optimizer=Adam lr={args.lr} weight_decay={args.weight_decay} "
        f"grad_weight={args.grad_weight} grad_criterion={args.grad_criterion}"
    )

    rows = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        best_val_acc, test_acc, best_epoch = train_one_seed(
            args=args,
            seed=seed,
            full_train=full_train,
            test_dataset=test_dataset,
            device=device,
            loader_kwargs=loader_kwargs,
        )
        rows.append((seed, best_val_acc, test_acc, best_epoch))
        print(
            f"seed={seed} best_val_acc={best_val_acc:.2f}% "
            f"best_epoch={best_epoch} test_acc={test_acc:.2f}%"
        )

    vals = np.asarray([r[1] for r in rows], dtype=np.float64)
    tests = np.asarray([r[2] for r in rows], dtype=np.float64)
    print("\nSummary over seeds")
    print(f"val_acc  mean={vals.mean():.2f}% std={vals.std():.2f}%")
    print(f"test_acc mean={tests.mean():.2f}% std={tests.std():.2f}%")


if __name__ == "__main__":
    main()

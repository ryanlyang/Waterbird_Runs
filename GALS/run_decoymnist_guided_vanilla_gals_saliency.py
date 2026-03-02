#!/usr/bin/env python3
"""Train DecoyMNIST guided/vanilla/gals LeNet models and export RISE saliency.

Behavior:
- Fixed 90/10 train/val split from train folder (split seed configurable, default 0).
- Train three variants on same split:
  - guided: LeNet + RRR-style input gradient suppression with external masks.
  - vanilla: LeNet baseline (no mask loss).
  - gals: LeNet + RRR-style loss with a separate hyperparameter set.
- Select best checkpoint per variant by val accuracy (tie-break by val loss).
- Generate saliency examples from val split with RISE.
- Save 15 examples per digit by default (configurable).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor

from utils.rise import RISE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_token(text: str) -> str:
    token = str(text).replace("\\", "__").replace("/", "__").replace(".", "_")
    token = token.replace(" ", "_").replace(":", "_")
    return token[:180]


def normalize_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    out -= out.min()
    mx = out.max()
    if mx > 1e-8:
        out /= mx
    return out


def map_to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def heatmap_rgb(norm_map: np.ndarray) -> np.ndarray:
    u8 = map_to_u8(norm_map)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_rgb(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip((1.0 - alpha) * base_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)


def contour_overlay(base_rgb: np.ndarray, norm_map: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    canvas = base_rgb.copy()
    binary = (norm_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 1)
    return canvas


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


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


def _load_attention_map(path: Path) -> torch.Tensor:
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
    return att.unsqueeze(0)


class GuidedImageFolder(utils.Dataset):
    def __init__(self, png_root: str, mask_root: str, split: str, image_transform=None) -> None:
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


def make_fixed_split_indices(n_total: int, val_frac: float, split_seed: int) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    n_val = int(val_frac * n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def train_variant(
    name: str,
    model: Net,
    train_loader: utils.DataLoader,
    val_loader: utils.DataLoader,
    test_loader: utils.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
    use_guidance: bool = False,
    grad_weight: float = 0.0,
    grad_criterion_name: str = "L1",
) -> Dict[str, object]:
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    grad_criterion: Optional[nn.Module]
    if use_guidance:
        grad_criterion = nn.L1Loss() if grad_criterion_name == "L1" else nn.MSELoss()
    else:
        grad_criterion = None

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if use_guidance:
                data, target, gt_mask = batch
                gt_mask = gt_mask.to(device)
                gt_mask = F.interpolate(gt_mask, size=data.shape[-2:], mode="nearest")
            else:
                data, target = batch
                gt_mask = None

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            if use_guidance:
                data.requires_grad_(True)
            out = model(data)
            cls_loss = F.nll_loss(out, target)
            if use_guidance and grad_criterion is not None and gt_mask is not None:
                dy_dx = torch.autograd.grad(cls_loss, data, create_graph=True)[0]
                rrr_loss = grad_criterion(dy_dx, dy_dx * gt_mask)
                loss = cls_loss + float(grad_weight) * rrr_loss
            else:
                loss = cls_loss

            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if print_every > 0 and (epoch % print_every == 0 or epoch == epochs):
            print(
                f"[{name}] epoch={epoch}/{epochs} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, device)

    return {
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }


class ProbModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = self.model(x)
        return torch.exp(log_probs)


def save_saliency_variants(prefix: str, saliency_28: np.ndarray, image_gray_u8: np.ndarray, sample_dir: Path) -> Dict[str, np.ndarray]:
    sal = normalize_map(saliency_28)
    sal_u8 = map_to_u8(sal)
    heat = heatmap_rgb(sal)
    base_rgb = np.repeat(image_gray_u8[:, :, None], 3, axis=2)
    overlay = overlay_rgb(base_rgb, heat, alpha=0.45)
    contour = contour_overlay(base_rgb, sal, threshold=0.75)
    binary = ((sal >= 0.75).astype(np.uint8) * 255)

    save_rgb(sample_dir / f"{prefix}_saliency_overlay_blue_red.png", overlay)
    save_rgb(sample_dir / f"{prefix}_saliency_heatmap_blue_red.png", heat)
    save_gray(sample_dir / f"{prefix}_saliency_grayscale_white_black.png", sal_u8)
    save_gray(sample_dir / f"{prefix}_saliency_binary_white_black.png", binary)
    save_rgb(sample_dir / f"{prefix}_saliency_contours_on_image.png", contour)

    return {
        "overlay": overlay,
        "heatmap": heat,
        "gray": np.repeat(sal_u8[:, :, None], 3, axis=2),
        "contour": contour,
    }


def write_comparison_panels(sample_dir: Path, vis_by_model: Dict[str, Dict[str, np.ndarray]]) -> None:
    model_names = list(vis_by_model.keys())
    if len(model_names) < 2:
        return

    viz_keys = ["overlay", "heatmap", "gray", "contour"]
    pairs = [("guided", "vanilla"), ("guided", "gals"), ("vanilla", "gals")]
    for key in viz_keys:
        for a, b in pairs:
            if a in vis_by_model and b in vis_by_model:
                pair = np.concatenate([vis_by_model[a][key], vis_by_model[b][key]], axis=1)
                save_rgb(sample_dir / f"pair_{a}_vs_{b}_{key}.png", pair)

        strip = np.concatenate([vis_by_model[m][key] for m in model_names], axis=1)
        save_rgb(sample_dir / f"all_models_{'_'.join(model_names)}_{key}.png", strip)


def select_val_examples_per_class(
    plain_train_ds: ImageFolder,
    val_indices: Sequence[int],
    per_digit: int,
    seed: int,
) -> List[int]:
    by_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx in val_indices:
        _, y = plain_train_ds.samples[idx]
        by_class[int(y)].append(int(idx))

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls in range(10):
        cls_indices = by_class[cls]
        if not cls_indices:
            continue
        cls_indices = list(cls_indices)
        rng.shuffle(cls_indices)
        take = min(per_digit, len(cls_indices))
        selected.extend(cls_indices[:take])
    return selected


def maybe_save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DecoyMNIST guided/vanilla/gals training + saliency export")
    p.add_argument(
        "--png-root",
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png",
        help="DecoyMNIST_png root containing train/ and test/ ImageFolder layout.",
    )
    p.add_argument(
        "--mask-root",
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/DecoyMNIST_png/clip_rn50_attention_gradcam",
        help="Mask root containing train/ split with .pth maps mirroring class/image paths.",
    )
    p.add_argument("--output-dir", default="", help="Output dir. Auto-generated if empty.")
    p.add_argument("--epochs", type=int, default=19)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-batch-size", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--print-every", type=int, default=1)

    # Vanilla hyperparams
    p.add_argument("--vanilla-lr", type=float, default=0.001)
    p.add_argument("--vanilla-weight-decay", type=float, default=1e-4)

    # Guided hyperparams (locked)
    p.add_argument("--guided-lr", type=float, default=0.001)
    p.add_argument("--guided-weight-decay", type=float, default=1e-4)
    p.add_argument("--guided-grad-weight", type=float, default=72503.48035960984)
    p.add_argument("--guided-grad-criterion", choices=["L1", "L2"], default="L1")

    # GALS hyperparams (from your Decoy setup)
    p.add_argument("--gals-lr", type=float, default=0.001)
    p.add_argument("--gals-weight-decay", type=float, default=1e-4)
    p.add_argument("--gals-grad-weight", type=float, default=72503.48035960984)
    p.add_argument("--gals-grad-criterion", choices=["L1", "L2"], default="L1")

    # Saliency
    p.add_argument("--target-class", choices=["label", "pred"], default="label")
    p.add_argument("--samples-per-digit", type=int, default=15)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--rise-num-masks", type=int, default=2000)
    p.add_argument("--rise-grid-size", type=int, default=8)
    p.add_argument("--rise-p1", type=float, default=0.1)
    p.add_argument("--rise-gpu-batch", type=int, default=16)
    p.add_argument("--rise-seed", type=int, default=0)
    p.add_argument("--rise-masks-path", default="", help="Optional .npy path for reusable RISE masks.")

    p.add_argument("--no-cuda", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    loader_kwargs = {"num_workers": int(args.num_workers), "pin_memory": use_cuda}

    png_root = Path(args.png_root).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve()
    if not png_root.is_dir():
        raise RuntimeError(f"Missing png-root: {png_root}")
    if not mask_root.is_dir():
        raise RuntimeError(f"Missing mask-root: {mask_root}")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        tag = safe_token(png_root.name)
        ts = Path(os.path.abspath(__file__)).stat().st_mtime_ns
        out_dir = png_root.parent / f"decoy_guided_vanilla_gals_saliency_{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda x: x * 2.0 - 1.0)])
    plain_train_ds = ImageFolder(str(png_root / "train"), transform=transform)
    plain_test_ds = ImageFolder(str(png_root / "test"), transform=transform)
    guided_train_ds = GuidedImageFolder(png_root=str(png_root), mask_root=str(mask_root), split="train", image_transform=transform)
    if len(plain_train_ds) != len(guided_train_ds):
        raise RuntimeError("Mismatch between plain and guided train dataset lengths.")

    n_total = len(plain_train_ds)
    train_idx, val_idx = make_fixed_split_indices(n_total=n_total, val_frac=float(args.val_frac), split_seed=int(args.split_seed))

    plain_train_subset = utils.Subset(plain_train_ds, train_idx)
    plain_val_subset = utils.Subset(plain_train_ds, val_idx)
    guided_train_subset = utils.Subset(guided_train_ds, train_idx)

    train_loader_vanilla = utils.DataLoader(plain_train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    train_loader_guided = utils.DataLoader(guided_train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    train_loader_gals = utils.DataLoader(guided_train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = utils.DataLoader(plain_val_subset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    test_loader = utils.DataLoader(plain_test_ds, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    print("DecoyMNIST guided/vanilla/gals saliency run")
    print(f"device={device}")
    print(f"png_root={png_root}")
    print(f"mask_root={mask_root}")
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(plain_test_ds)} split_seed={args.split_seed}")
    print(
        "Hyperparams: "
        f"vanilla(lr={args.vanilla_lr},wd={args.vanilla_weight_decay}) "
        f"guided(lr={args.guided_lr},wd={args.guided_weight_decay},gw={args.guided_grad_weight},gc={args.guided_grad_criterion}) "
        f"gals(lr={args.gals_lr},wd={args.gals_weight_decay},gw={args.gals_grad_weight},gc={args.gals_grad_criterion})"
    )

    # Train all three variants
    vanilla_model = Net().to(device)
    guided_model = Net().to(device)
    gals_model = Net().to(device)

    vanilla_metrics = train_variant(
        name="vanilla",
        model=vanilla_model,
        train_loader=train_loader_vanilla,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.vanilla_lr),
        weight_decay=float(args.vanilla_weight_decay),
        print_every=int(args.print_every),
        use_guidance=False,
    )
    guided_metrics = train_variant(
        name="guided",
        model=guided_model,
        train_loader=train_loader_guided,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.guided_lr),
        weight_decay=float(args.guided_weight_decay),
        print_every=int(args.print_every),
        use_guidance=True,
        grad_weight=float(args.guided_grad_weight),
        grad_criterion_name=str(args.guided_grad_criterion),
    )
    gals_metrics = train_variant(
        name="gals",
        model=gals_model,
        train_loader=train_loader_gals,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.gals_lr),
        weight_decay=float(args.gals_weight_decay),
        print_every=int(args.print_every),
        use_guidance=True,
        grad_weight=float(args.gals_grad_weight),
        grad_criterion_name=str(args.gals_grad_criterion),
    )

    ckpt_dir = out_dir / "checkpoints"
    vanilla_ckpt = ckpt_dir / "vanilla" / f"decoy_vanilla_seed{args.seed}_ep{args.epochs}.pth"
    guided_ckpt = ckpt_dir / "guided" / f"decoy_guided_seed{args.seed}_ep{args.epochs}.pth"
    gals_ckpt = ckpt_dir / "gals" / f"decoy_gals_seed{args.seed}_ep{args.epochs}.pth"
    maybe_save_checkpoint(vanilla_model, vanilla_ckpt)
    maybe_save_checkpoint(guided_model, guided_ckpt)
    maybe_save_checkpoint(gals_model, gals_ckpt)

    # Build shared RISE masks
    rise_input_size = (28, 28)
    if args.rise_masks_path:
        masks_path = Path(args.rise_masks_path).expanduser().resolve()
    else:
        p1_token = str(args.rise_p1).replace(".", "p")
        masks_path = out_dir / f"rise_masks_n{args.rise_num_masks}_s{args.rise_grid_size}_p{p1_token}_seed{args.rise_seed}.npy"

    guided_prob = ProbModel(guided_model).to(device).eval()
    vanilla_prob = ProbModel(vanilla_model).to(device).eval()
    gals_prob = ProbModel(gals_model).to(device).eval()

    rise_guided = RISE(guided_prob, rise_input_size, num_classes=10, gpu_batch=int(args.rise_gpu_batch), p1=float(args.rise_p1))
    rise_vanilla = RISE(vanilla_prob, rise_input_size, num_classes=10, gpu_batch=int(args.rise_gpu_batch), p1=float(args.rise_p1))
    rise_gals = RISE(gals_prob, rise_input_size, num_classes=10, gpu_batch=int(args.rise_gpu_batch), p1=float(args.rise_p1))

    if masks_path.is_file():
        rise_guided.load_masks(str(masks_path), device=device)
    else:
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.random.seed(int(args.rise_seed))
        rise_guided.generate_masks(
            N=int(args.rise_num_masks),
            s=int(args.rise_grid_size),
            device=device,
            savepath=str(masks_path),
        )
    rise_vanilla.load_masks(str(masks_path), device=device)
    rise_gals.load_masks(str(masks_path), device=device)

    # Select val examples: 15 per digit (default)
    selected_indices = select_val_examples_per_class(
        plain_train_ds=plain_train_ds,
        val_indices=val_idx,
        per_digit=int(args.samples_per_digit),
        seed=int(args.sample_seed),
    )
    print(
        f"Selected {len(selected_indices)} val samples "
        f"({args.samples_per_digit} per digit requested).",
        flush=True,
    )

    sample_rows: List[Dict[str, object]] = []
    use_label_target = args.target_class == "label"
    for i, global_idx in enumerate(selected_indices):
        img_path_str, label = plain_train_ds.samples[global_idx]
        image_t, _ = plain_train_ds[global_idx]  # transformed [-1,1], 1x28x28
        input_tensor = image_t.unsqueeze(0).to(device)
        label_int = int(label)

        with torch.no_grad():
            logp_guided = guided_model(input_tensor)
            pred_guided = int(logp_guided.argmax(dim=1).item())
            targ_guided = label_int if use_label_target else pred_guided
            sal_guided = rise_guided(input_tensor)[targ_guided].detach().cpu().numpy().astype(np.float32)

            logp_vanilla = vanilla_model(input_tensor)
            pred_vanilla = int(logp_vanilla.argmax(dim=1).item())
            targ_vanilla = label_int if use_label_target else pred_vanilla
            sal_vanilla = rise_vanilla(input_tensor)[targ_vanilla].detach().cpu().numpy().astype(np.float32)

            logp_gals = gals_model(input_tensor)
            pred_gals = int(logp_gals.argmax(dim=1).item())
            targ_gals = label_int if use_label_target else pred_gals
            sal_gals = rise_gals(input_tensor)[targ_gals].detach().cpu().numpy().astype(np.float32)

        gray = image_t[0].detach().cpu().numpy().astype(np.float32)
        gray = np.clip((gray + 1.0) * 0.5, 0.0, 1.0)
        gray_u8 = map_to_u8(gray)
        base_rgb = np.repeat(gray_u8[:, :, None], 3, axis=2)

        img_name = Path(img_path_str).name
        sample_dir = out_dir / "samples" / f"{i:03d}_y{label_int}_{safe_token(img_name)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_rgb(sample_dir / "original_image.png", base_rgb)

        vis_by_model: Dict[str, Dict[str, np.ndarray]] = {}
        vis_by_model["guided"] = save_saliency_variants("guided", sal_guided, gray_u8, sample_dir)
        vis_by_model["vanilla"] = save_saliency_variants("vanilla", sal_vanilla, gray_u8, sample_dir)
        vis_by_model["gals"] = save_saliency_variants("gals", sal_gals, gray_u8, sample_dir)
        write_comparison_panels(sample_dir, vis_by_model)

        row = {
            "index": i,
            "global_index": int(global_idx),
            "img_path": str(img_path_str),
            "label": int(label_int),
            "guided_pred": pred_guided,
            "guided_target_class": int(targ_guided),
            "vanilla_pred": pred_vanilla,
            "vanilla_target_class": int(targ_vanilla),
            "gals_pred": pred_gals,
            "gals_target_class": int(targ_gals),
        }
        with open(sample_dir / "sample_info.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        sample_rows.append(row)

    # Save outputs
    csv_path = out_dir / "sample_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "global_index",
            "img_path",
            "label",
            "guided_pred",
            "guided_target_class",
            "vanilla_pred",
            "vanilla_target_class",
            "gals_pred",
            "gals_target_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    summary = {
        "png_root": str(png_root),
        "mask_root": str(mask_root),
        "output_dir": str(out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "epochs": int(args.epochs),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(plain_test_ds)),
        "samples_per_digit_requested": int(args.samples_per_digit),
        "num_val_samples_generated": int(len(sample_rows)),
        "target_class_mode": args.target_class,
        "rise": {
            "num_masks": int(args.rise_num_masks),
            "grid_size": int(args.rise_grid_size),
            "p1": float(args.rise_p1),
            "gpu_batch": int(args.rise_gpu_batch),
            "seed": int(args.rise_seed),
            "masks_path": str(masks_path),
        },
        "vanilla": {
            "metrics": vanilla_metrics,
            "checkpoint": str(vanilla_ckpt),
            "lr": float(args.vanilla_lr),
            "weight_decay": float(args.vanilla_weight_decay),
        },
        "guided": {
            "metrics": guided_metrics,
            "checkpoint": str(guided_ckpt),
            "lr": float(args.guided_lr),
            "weight_decay": float(args.guided_weight_decay),
            "grad_weight": float(args.guided_grad_weight),
            "grad_criterion": str(args.guided_grad_criterion),
        },
        "gals": {
            "metrics": gals_metrics,
            "checkpoint": str(gals_ckpt),
            "lr": float(args.gals_lr),
            "weight_decay": float(args.gals_weight_decay),
            "grad_weight": float(args.gals_grad_weight),
            "grad_criterion": str(args.gals_grad_criterion),
        },
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE] DecoyMNIST guided/vanilla/gals saliency run complete.", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Sample index: {csv_path}", flush=True)


if __name__ == "__main__":
    main()


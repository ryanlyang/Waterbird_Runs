#!/usr/bin/env python3
"""
Compute Pointing Game accuracy on Waterbirds for multiple trained methods.

Supported methods:
- guided
- vanilla
- gals (binary single-logit GALS model from this repo)
- afr (AFR stage-1 model, optional stage-2 last-layer override)
- upweight (generic CNN checkpoint from main.py training)
- abn (ABN checkpoint from main.py training)
- clip_zs (zero-shot CLIP with prompt templates)
- clip_lr (CLIP features + LogisticRegression head)

For each method and dataset, this script reports:
- overall Pointing Game accuracy
- per-group Pointing Game accuracy over Waterbirds groups (0_0, 1_0, 2_0, 3_0)
- bookkeeping counts (evaluated, missing masks, missing images)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models.resnet import resnet50 as gals_resnet50  # noqa: E402
from models.resnet_abn import resnet50 as resnet50_abn  # noqa: E402


SPLIT_MAP = {"train": 0, "val": 1, "test": 2}
GROUP_NAMES = ["0_0", "1_0", "2_0", "3_0"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_token(text: str) -> str:
    token = text.replace("\\", "__").replace("/", "__").replace(".", "_")
    token = token.replace(" ", "_").replace(":", "_")
    return token[:180]


def open_pil_with_retry(path: Path, mode: str, retries: int = 5, sleep_s: float = 0.2) -> Image.Image:
    last_exc = None
    for attempt in range(retries):
        try:
            with Image.open(path) as im:
                im.load()
                return im.convert(mode)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            last_exc = exc
            if attempt + 1 < retries:
                import time as _time

                _time.sleep(sleep_s)
                continue
            break
    raise last_exc


def mask_name_from_path(image_path: str) -> str:
    folder = os.path.basename(os.path.dirname(image_path))
    folder_prefix = folder.replace(".", "_")
    base = os.path.splitext(os.path.basename(image_path))[0]
    return f"{folder_prefix}_{base}.png"


def candidate_mask_paths(mask_root: Path, image_path: Path, data_path: Path) -> List[Path]:
    """
    Try both WeCLIPPlus flat naming and CUB-like nested naming.
    """
    try:
        rel = image_path.relative_to(data_path)
    except Exception:
        rel = Path(image_path.name)

    rel_no_ext = rel.with_suffix("")
    parent = rel.parent.name
    parent_underscored = parent.replace(".", "_")
    base = rel.stem

    return [
        # WeCLIPPlus flat naming: <parent_with_dots_replaced>_<basename>.png
        mask_root / f"{parent_underscored}_{base}.png",
        # Nested alternatives
        mask_root / parent_underscored / f"{base}.png",
        mask_root / parent / f"{base}.png",
        # CUB segmentation structure mirroring image subpath
        mask_root / rel_no_ext.with_suffix(".png"),
        mask_root / rel_no_ext.with_suffix(".jpg"),
        mask_root / rel_no_ext.with_suffix(".jpeg"),
    ]


def resolve_mask_path(mask_root: Path, image_path: Path, data_path: Path) -> Tuple[Optional[Path], Path]:
    cands = candidate_mask_paths(mask_root, image_path, data_path)
    for cand in cands:
        if cand.is_file():
            return cand, cands[0]
    return None, cands[0]


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = ckpt_obj.get(key)
            if isinstance(value, dict):
                return value
        if ckpt_obj:
            first_key = next(iter(ckpt_obj.keys()))
            if isinstance(first_key, str):
                return ckpt_obj  # already state_dict
    raise RuntimeError("Could not extract model state_dict from checkpoint.")


def align_state_dict_keys(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        return state_dict

    model_has_module = model_keys[0].startswith("module.")
    ckpt_has_module = ckpt_keys[0].startswith("module.")

    if ckpt_has_module and not model_has_module:
        return {k[7:]: v for k, v in state_dict.items() if k.startswith("module.")}
    if model_has_module and not ckpt_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def infer_output_dim_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    candidate_suffixes: List[str],
    default: int,
) -> int:
    for suffix in candidate_suffixes:
        for key, tensor in state_dict.items():
            if key.endswith(suffix) and isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
                out_dim = int(tensor.shape[0])
                if out_dim > 0:
                    return out_dim
    return int(default)


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def normalize_map(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    out -= out.min()
    mx = out.max()
    if mx > 1e-8:
        out /= mx
    return out


def torch_load_compat(path: Path, device: torch.device):
    """
    torch.load compatibility across torch versions:
    - torch<=2.5 default weights_only=False
    - torch>=2.6 default weights_only=True (can fail on older checkpoints)
    """
    try:
        return torch.load(path, map_location=device)
    except Exception as exc:
        msg = str(exc)
        if "Weights only load failed" in msg:
            try:
                return torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                # Older torch that doesn't support weights_only kwarg.
                return torch.load(path, map_location=device)
        raise


def compute_cam(features: torch.Tensor, class_weights: torch.Tensor) -> np.ndarray:
    cam = torch.einsum("c,chw->hw", class_weights, features)
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy().astype(np.float32)


def upsample_saliency(saliency: np.ndarray, h: int, w: int) -> np.ndarray:
    ten = torch.from_numpy(saliency).float().unsqueeze(0).unsqueeze(0)
    up = F.interpolate(ten, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return normalize_map(up)


def pointing_hit(saliency: np.ndarray, mask: np.ndarray) -> bool:
    if saliency.size == 0:
        return False
    sal_max = float(np.max(saliency))
    if sal_max <= 0:
        return False
    max_positions = np.argwhere(np.isclose(saliency, sal_max))
    if max_positions.size == 0:
        return False
    for x, y in max_positions:
        if mask[x, y] > 0:
            return True
    return False


class FeatureHook:
    def __init__(self, module: nn.Module):
        self.features: Optional[torch.Tensor] = None
        self.handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _inputs, output):
        self.features = output.detach()

    def close(self) -> None:
        self.handle.remove()


class GALSBinaryCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = gals_resnet50(pretrained=False, return_fmaps=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, fmaps = self.net(x)
        return logits, fmaps


class GuidedResNetCAM(nn.Module):
    """
    Replica of run_guided_waterbird.ResNetCAM state_dict structure:
    - nested torchvision model under self.base
    - classifier alias at self.classifier
    - layer4 feature hook
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.base = models.resnet50(pretrained=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        self.classifier = self.base.fc
        self.features: Optional[torch.Tensor] = None
        self.base.layer4.register_forward_hook(self._hook_fn)

    def _hook_fn(self, _module, _inputs, output):
        self.features = output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.base(x)
        if self.features is None:
            raise RuntimeError("GuidedResNetCAM did not capture features.")
        return logits, self.features


def _import_afr_models(afr_root: Path):
    init_file = afr_root / "models" / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"AFR models package not found: {init_file}")

    spec = importlib.util.spec_from_file_location(
        "afr_models",
        str(init_file),
        submodule_search_locations=[str(init_file.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for AFR models from {init_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["afr_models"] = module
    spec.loader.exec_module(module)
    return module


def _try_import_clip():
    try:
        import clip  # type: ignore

        return clip
    except Exception:
        from CLIP.clip import clip  # type: ignore

        return clip


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _default_clip_templates() -> List[str]:
    return [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a bright photo of a {}.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}.",
        "a low resolution photo of a {}.",
        "a good photo of a {}.",
        "a photo of the {}.",
    ]


def _split_root_and_dir(dataset_path: str) -> Tuple[str, str]:
    p = Path(dataset_path).expanduser().resolve()
    if (p / "metadata.csv").exists():
        return str(p.parent), p.name
    raise FileNotFoundError(f"Expected Waterbirds dataset dir containing metadata.csv, got: {p}")


@dataclass(frozen=True)
class _ClipCfgData:
    WATERBIRDS_DIR: str
    SIZE: int = 224
    REMOVE_BACKGROUND: bool = False
    ATTENTION_DIR: str = "NONE"


@dataclass(frozen=True)
class _ClipCfg:
    DATA: _ClipCfgData


def _load_waterbirds_for_clip(dataset_path: str):
    from datasets.waterbirds import Waterbirds  # type: ignore

    root, waterbirds_dir = _split_root_and_dir(dataset_path)
    cfg = _ClipCfg(DATA=_ClipCfgData(WATERBIRDS_DIR=waterbirds_dir))
    return root, cfg, Waterbirds


def _extract_clip_features(
    dataset,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            out = model.encode_image(images).float()
            out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    X = np.concatenate(feats, axis=0).astype(np.float64, copy=False)
    y = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    return X, y


def _build_clip_text_features(
    clip_module,
    model,
    device: torch.device,
    class_names: List[str],
    templates: List[str],
) -> torch.Tensor:
    text_features: List[torch.Tensor] = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [tmpl.format(cls) for tmpl in templates]
            tokens = clip_module.tokenize(prompts).to(device)
            feats = model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            cls_feat = feats.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features.append(cls_feat)
    return torch.stack(text_features, dim=0)


class MethodRunnerBase:
    name: str

    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class GuidedRunner(MethodRunnerBase):
    name = "guided"

    def __init__(self, checkpoint: Path, num_classes: int, device: torch.device):
        self.device = device
        self.model = GuidedResNetCAM(num_classes).to(device)
        state = torch_load_compat(checkpoint, device)
        state_dict = extract_state_dict(state) if isinstance(state, dict) else state
        state_dict = align_state_dict_keys(state_dict, self.model)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        logits, feats = self.model(image_tensor)
        pred = int(logits.argmax(dim=1).item())
        target = int(label if target_mode == "label" else pred)
        weights = self.model.classifier.weight[target]  # type: ignore[attr-defined]
        cam = compute_cam(feats[0], weights)
        return pred, target, cam


class VanillaRunner(MethodRunnerBase):
    name = "vanilla"

    def __init__(self, checkpoint: Path, num_classes: int, device: torch.device):
        self.device = device
        state = torch_load_compat(checkpoint, device)
        state_dict = extract_state_dict(state) if isinstance(state, dict) else state
        classifier_classes = infer_output_dim_from_state_dict(
            state_dict=state_dict,
            candidate_suffixes=["fc.weight", "classifier.weight", "base.fc.weight"],
            default=num_classes,
        )

        self.model = models.resnet50(pretrained=False).to(device)
        self.model.fc = nn.Linear(self.model.fc.in_features, int(classifier_classes)).to(device)
        state_dict = align_state_dict_keys(state_dict, self.model)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.hook = FeatureHook(self.model.layer4)  # type: ignore[attr-defined]
        self.classifier_classes = int(classifier_classes)

    @torch.no_grad()
    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        logits = self.model(image_tensor)
        if self.hook.features is None:
            raise RuntimeError("Vanilla feature hook did not capture layer4 features.")

        if logits.ndim == 2 and logits.shape[1] == 1:
            score = logits[:, 0]
            pred = int((score >= 0).long().item())
            target = int(label if target_mode == "label" else pred)
            w_pos = self.model.fc.weight[0]  # type: ignore[attr-defined]
            class_weight = w_pos if target == 1 else (-w_pos)
            cam = compute_cam(self.hook.features[0], class_weight)
        else:
            pred = int(logits.argmax(dim=1).item())
            target = int(label if target_mode == "label" else pred)
            class_weight = self.model.fc.weight[target]  # type: ignore[attr-defined]
            cam = compute_cam(self.hook.features[0], class_weight)
        return pred, target, cam

    def close(self) -> None:
        self.hook.close()


class GALSRunner(MethodRunnerBase):
    name = "gals"

    def __init__(self, checkpoint: Path, device: torch.device):
        self.device = device
        self.model = GALSBinaryCAMModel().to(device)
        ckpt = torch_load_compat(checkpoint, device)
        state = extract_state_dict(ckpt)
        state = align_state_dict_keys(state, self.model)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        logits, feats = self.model(image_tensor)
        prob_1 = torch.sigmoid(logits[:, 0])
        pred = int((prob_1 >= 0.5).long().item())
        target = int(label if target_mode == "label" else pred)

        w_pos = self.model.net.fc.weight[0]  # type: ignore[attr-defined]
        class_weight = w_pos if target == 1 else (-w_pos)
        cam = compute_cam(feats[0], class_weight)
        return pred, target, cam


class AFRRunner(MethodRunnerBase):
    name = "afr"

    def __init__(
        self,
        afr_root: Path,
        stage1_checkpoint: Path,
        stage2_last_layer_checkpoint: Optional[Path],
        num_classes: int,
        device: torch.device,
    ):
        self.device = device
        afr_models = _import_afr_models(afr_root)
        self.model = getattr(afr_models, "imagenet_resnet50_pretrained")(num_classes).to(device)

        stage1 = torch_load_compat(stage1_checkpoint, device)
        state_dict = extract_state_dict(stage1) if isinstance(stage1, dict) else stage1
        state_dict = align_state_dict_keys(state_dict, self.model)
        self.model.load_state_dict(state_dict, strict=False)

        if stage2_last_layer_checkpoint is not None:
            ll_obj = torch_load_compat(stage2_last_layer_checkpoint, device)
            ll_state = extract_state_dict(ll_obj) if isinstance(ll_obj, dict) else ll_obj
            if "weight" in ll_state and "bias" in ll_state:
                with torch.no_grad():
                    self.model.fc.weight.copy_(ll_state["weight"])
                    self.model.fc.bias.copy_(ll_state["bias"])
            else:
                raise RuntimeError(
                    f"AFR stage-2 last-layer checkpoint must contain weight/bias: {stage2_last_layer_checkpoint}"
                )

        self.model.eval()
        self.hook = FeatureHook(self.model.layer4)  # type: ignore[attr-defined]

    @torch.no_grad()
    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        logits = self.model(image_tensor)
        if self.hook.features is None:
            raise RuntimeError("AFR feature hook did not capture layer4 features.")
        pred = int(logits.argmax(dim=1).item())
        target = int(label if target_mode == "label" else pred)
        weights = self.model.fc.weight[target]  # type: ignore[attr-defined]
        cam = compute_cam(self.hook.features[0], weights)
        return pred, target, cam

    def close(self) -> None:
        self.hook.close()


class UpweightRunner(VanillaRunner):
    name = "upweight"


class ABNRunner(MethodRunnerBase):
    name = "abn"

    def __init__(self, checkpoint: Path, num_classes: int, device: torch.device):
        self.device = device
        state = torch_load_compat(checkpoint, device)
        state_dict = extract_state_dict(state) if isinstance(state, dict) else state
        classifier_classes = infer_output_dim_from_state_dict(
            state_dict=state_dict,
            candidate_suffixes=["fc.weight", "att_conv.weight", "bn_att2.weight", "att_conv2.weight"],
            default=num_classes,
        )
        self.model = resnet50_abn(
            pretrained=False,
            num_classes=int(classifier_classes),
            add_after_attention=True,
        ).to(device)
        state_dict = align_state_dict_keys(state_dict, self.model)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.classifier_classes = int(classifier_classes)

    @torch.no_grad()
    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        _att_logits, logits, extra = self.model(image_tensor, provided_att=None)
        if logits.ndim == 2 and logits.shape[1] == 1:
            pred = int((logits[:, 0] >= 0).long().item())
        else:
            pred = int(logits.argmax(dim=1).item())
        target = int(label if target_mode == "label" else pred)
        att = extra[0]  # Bx1xhxw
        sal = att[0, 0].detach().cpu().numpy().astype(np.float32)
        sal = normalize_map(sal)
        return pred, target, sal


class ClipZeroShotRunner(MethodRunnerBase):
    name = "clip_zs"

    def __init__(
        self,
        clip_model_name: str,
        class_names: List[str],
        templates: List[str],
        device: torch.device,
    ):
        self.device = device
        self.clip = _try_import_clip()
        try:
            self.model, self.preprocess = self.clip.load(clip_model_name, device=str(device), jit=False)
        except TypeError:
            self.model, self.preprocess = self.clip.load(clip_model_name, device=str(device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.text_features = _build_clip_text_features(
            self.clip,
            self.model,
            device=device,
            class_names=class_names,
            templates=templates,
        )  # [2, d]

    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        if pil_image is None:
            raise RuntimeError("ClipZeroShotRunner requires pil_image input.")
        x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        x.requires_grad_(True)

        feats = self.model.encode_image(x).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = feats @ self.text_features.T

        pred = int(torch.argmax(logits, dim=1).item())
        target = int(label if target_mode == "label" else pred)
        score = logits[0, target]

        if x.grad is not None:
            x.grad.zero_()
        self.model.zero_grad()
        score.backward()
        sal = x.grad.detach().abs().mean(dim=1)[0].cpu().numpy().astype(np.float32)
        sal = normalize_map(sal)
        return pred, target, sal


class ClipLRRunner(MethodRunnerBase):
    name = "clip_lr"

    def __init__(
        self,
        dataset_path: Path,
        clip_model_name: str,
        C: float,
        penalty: str,
        solver: str,
        fit_intercept: bool,
        tol: float,
        max_iter: int,
        class_weight: Optional[str],
        device: torch.device,
        batch_size: int,
        num_workers: int,
    ):
        from sklearn.linear_model import LogisticRegression

        self.device = device
        self.clip = _try_import_clip()
        try:
            self.model, self.preprocess = self.clip.load(clip_model_name, device=str(device), jit=False)
        except TypeError:
            self.model, self.preprocess = self.clip.load(clip_model_name, device=str(device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        root, cfg, Waterbirds = _load_waterbirds_for_clip(str(dataset_path))
        train_ds = Waterbirds(root=root, cfg=cfg, split="train", transform=self.preprocess)
        X_train, y_train = _extract_clip_features(
            dataset=train_ds,
            model=self.model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        clf = LogisticRegression(
            C=float(C),
            penalty=str(penalty),
            solver=str(solver),
            fit_intercept=bool(fit_intercept),
            class_weight=class_weight,
            tol=float(tol),
            max_iter=int(max_iter),
            random_state=0,
        )
        clf.fit(X_train, y_train)

        # Binary LR in sklearn stores one weight vector for positive class.
        w = clf.coef_[0].astype(np.float32)
        b = float(clf.intercept_[0]) if clf.fit_intercept else 0.0
        self.w = torch.from_numpy(w).to(device)  # [d]
        self.b = torch.tensor(b, device=device, dtype=torch.float32)

    def predict_and_saliency(
        self,
        image_tensor: torch.Tensor,
        label: int,
        target_mode: str,
        pil_image: Optional[Image.Image] = None,
    ) -> Tuple[int, int, np.ndarray]:
        if pil_image is None:
            raise RuntimeError("ClipLRRunner requires pil_image input.")
        x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        x.requires_grad_(True)

        feats = self.model.encode_image(x).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)

        # Binary decision score for class 1.
        s = (feats[0] * self.w).sum() + self.b
        pred = int((s >= 0).long().item())
        target = int(label if target_mode == "label" else pred)
        score = s if target == 1 else -s

        if x.grad is not None:
            x.grad.zero_()
        self.model.zero_grad()
        score.backward()
        sal = x.grad.detach().abs().mean(dim=1)[0].cpu().numpy().astype(np.float32)
        sal = normalize_map(sal)
        return pred, target, sal


@dataclass
class MethodStats:
    hits: int = 0
    total: int = 0
    missing_masks: int = 0
    missing_images: int = 0
    errors: int = 0
    group_hits: Dict[int, int] = None
    group_total: Dict[int, int] = None

    def __post_init__(self):
        if self.group_hits is None:
            self.group_hits = {i: 0 for i in range(4)}
        if self.group_total is None:
            self.group_total = {i: 0 for i in range(4)}


def _pg_acc(hits: int, total: int) -> float:
    return float(hits) / float(total) if total > 0 else float("nan")


def _pick_rows(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _resolve_ckpt_from_summary(summary_path: str, key: str) -> str:
    if not summary_path:
        return ""
    p = Path(summary_path).expanduser().resolve()
    if not p.is_file():
        return ""
    obj = load_json(p)
    v = obj.get(key)
    return str(v) if isinstance(v, str) else ""


def evaluate_dataset(
    *,
    dataset_name: str,
    data_path: Path,
    mask_root: Path,
    split: str,
    preprocess: transforms.Compose,
    runners: Dict[str, MethodRunnerBase],
    target_mode: str,
    max_samples: int,
    sample_seed: int,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    metadata_path = data_path / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"[{dataset_name}] Missing metadata.csv: {metadata_path}")

    split_id = SPLIT_MAP[split]
    df = pd.read_csv(metadata_path)
    df = df[df["split"] == split_id].copy()
    if df.empty:
        raise RuntimeError(f"[{dataset_name}] No rows for split '{split}' in metadata.csv")
    df = _pick_rows(df, max_samples=max_samples, seed=sample_seed)

    stats = {name: MethodStats() for name in runners.keys()}
    per_image_rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        rel_path = str(row["img_filename"])
        image_path = data_path / rel_path
        label = int(row["y"])
        place = int(row["place"])
        group = int(label * 2 + place)
        mask_path, mask_fallback = resolve_mask_path(mask_root, image_path, data_path)

        image_missing = not image_path.is_file()
        mask_missing = mask_path is None

        image_tensor: Optional[torch.Tensor] = None
        pil_image: Optional[Image.Image] = None
        mask_bin: Optional[np.ndarray] = None

        if not image_missing:
            try:
                pil_image = open_pil_with_retry(image_path, mode="RGB")
                image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
            except Exception:
                image_missing = True

        if not mask_missing:
            try:
                assert mask_path is not None
                mask = open_pil_with_retry(mask_path, mode="L").resize((224, 224), Image.NEAREST)
                mask_arr = np.array(mask, dtype=np.uint8)
                mask_bin = (mask_arr > 0).astype(np.uint8)
            except Exception:
                mask_missing = True

        for method_name, runner in runners.items():
            st = stats[method_name]

            if image_missing:
                st.missing_images += 1
                continue
            if mask_missing:
                st.missing_masks += 1
                continue
            assert image_tensor is not None
            assert pil_image is not None
            assert mask_bin is not None

            try:
                pred, target, saliency = runner.predict_and_saliency(
                    image_tensor,
                    label,
                    target_mode,
                    pil_image=pil_image,
                )
                saliency = upsample_saliency(saliency, 224, 224)
                hit = pointing_hit(saliency, mask_bin)
            except Exception:
                st.errors += 1
                continue

            st.total += 1
            st.hits += int(hit)
            st.group_total[group] += 1
            st.group_hits[group] += int(hit)

            per_image_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "img_filename": rel_path,
                    "label": label,
                    "place": place,
                    "group": group,
                    "group_name": GROUP_NAMES[group],
                    "pred": pred,
                    "target_for_saliency": target,
                    "hit": int(hit),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )

    summary_rows: List[Dict[str, object]] = []
    for method_name, st in stats.items():
        row = {
            "dataset": dataset_name,
            "method": method_name,
            "split": split,
            "target_mode": target_mode,
            "pg_hits": st.hits,
            "pg_total": st.total,
            "pg_acc": _pg_acc(st.hits, st.total),
            "missing_images": st.missing_images,
            "missing_masks": st.missing_masks,
            "errors": st.errors,
            "mask_lookup_note": "",
        }
        for g in range(4):
            row[f"group_{GROUP_NAMES[g]}_hits"] = st.group_hits[g]
            row[f"group_{GROUP_NAMES[g]}_total"] = st.group_total[g]
            row[f"group_{GROUP_NAMES[g]}_pg_acc"] = _pg_acc(st.group_hits[g], st.group_total[g])
        summary_rows.append(row)

        if st.missing_masks > 0:
            row["mask_lookup_note"] = (
                "Some masks missing; supports WeCLIP flat and CUB nested naming. "
                f"Example first candidate path: {mask_fallback}"
            )

    return summary_rows, per_image_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pointing Game evaluator for Waterbirds 95/100 across guided/vanilla/gals/afr/"
            "upweight/abn/clip_zs/clip_lr."
        )
    )
    parser.add_argument(
        "--datasets",
        default="95,100",
        help="Comma-separated dataset IDs to evaluate. Any of: 95,100",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--target-mode", choices=["label", "pred"], default="label")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows in split.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wb95-data-path", default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2")
    parser.add_argument("--wb100-data-path", default="/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2")
    parser.add_argument(
        "--wb95-mask-root",
        default="/home/ryreu/guided_cnn/waterbirds/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    )
    parser.add_argument(
        "--wb100-mask-root",
        default="/home/ryreu/guided_cnn/waterbirds/L100/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    )

    parser.add_argument("--wb95-saliency-summary", default="", help="Optional run_summary.json to auto-fill guided/vanilla/gals WB95 checkpoints.")
    parser.add_argument("--wb100-saliency-summary", default="", help="Optional run_summary.json to auto-fill guided/vanilla/gals WB100 checkpoints.")

    parser.add_argument("--guided95-ckpt", default="")
    parser.add_argument("--guided100-ckpt", default="")
    parser.add_argument("--vanilla95-ckpt", default="")
    parser.add_argument("--vanilla100-ckpt", default="")
    parser.add_argument("--gals95-ckpt", default="")
    parser.add_argument("--gals100-ckpt", default="")
    parser.add_argument("--upweight95-ckpt", default="")
    parser.add_argument("--upweight100-ckpt", default="")
    parser.add_argument("--abn95-ckpt", default="")
    parser.add_argument("--abn100-ckpt", default="")

    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--clip-num-workers", type=int, default=4)
    parser.add_argument("--clip-class-names", default="landbird,waterbird")
    parser.add_argument(
        "--clip-template",
        action="append",
        default=[],
        help="Prompt template with '{}' placeholder (can be passed multiple times).",
    )

    # CLIP-LR fixed hyperparameters by dataset (WB95/WB100)
    parser.add_argument("--clip-lr95-C", type=float, default=30.481669053249504)
    parser.add_argument("--clip-lr95-penalty", default="l2")
    parser.add_argument("--clip-lr95-solver", default="lbfgs")
    parser.add_argument("--clip-lr95-fit-intercept", type=int, default=1)
    parser.add_argument("--clip-lr95-tol", type=float, default=1e-4)
    parser.add_argument("--clip-lr95-max-iter", type=int, default=5000)
    parser.add_argument("--clip-lr95-class-weight", default="none", choices=["none", "balanced"])

    parser.add_argument("--clip-lr100-C", type=float, default=0.2515000498909345)
    parser.add_argument("--clip-lr100-penalty", default="l2")
    parser.add_argument("--clip-lr100-solver", default="lbfgs")
    parser.add_argument("--clip-lr100-fit-intercept", type=int, default=1)
    parser.add_argument("--clip-lr100-tol", type=float, default=1e-4)
    parser.add_argument("--clip-lr100-max-iter", type=int, default=5000)
    parser.add_argument("--clip-lr100-class-weight", default="none", choices=["none", "balanced"])

    parser.add_argument("--afr-root", default=str((PARENT_DIR / "afr").resolve()))
    parser.add_argument("--afr95-stage1-ckpt", default="")
    parser.add_argument("--afr100-stage1-ckpt", default="")
    parser.add_argument("--afr95-last-layer-ckpt", default="")
    parser.add_argument("--afr100-last-layer-ckpt", default="")

    parser.add_argument("--methods", default="guided,vanilla,gals,afr")
    parser.add_argument("--output-dir", default="", help="If empty, writes under <repo_parent>/logsWaterbird/")
    return parser.parse_args()


def _resolve_ckpt(path: str) -> Optional[Path]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    return p if p.is_file() else None


def _build_runners_for_dataset(
    *,
    dataset_tag: str,
    dataset_path: Path,
    num_classes: int,
    methods: Iterable[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, MethodRunnerBase]:
    runners: Dict[str, MethodRunnerBase] = {}

    if dataset_tag == "95":
        guided_ckpt = args.guided95_ckpt
        vanilla_ckpt = args.vanilla95_ckpt
        gals_ckpt = args.gals95_ckpt
        upweight_ckpt = args.upweight95_ckpt
        abn_ckpt = args.abn95_ckpt
        afr_stage1_ckpt = args.afr95_stage1_ckpt
        afr_last_layer_ckpt = args.afr95_last_layer_ckpt
        clip_lr_C = args.clip_lr95_C
        clip_lr_penalty = args.clip_lr95_penalty
        clip_lr_solver = args.clip_lr95_solver
        clip_lr_fit_intercept = bool(args.clip_lr95_fit_intercept)
        clip_lr_tol = args.clip_lr95_tol
        clip_lr_max_iter = args.clip_lr95_max_iter
        clip_lr_class_weight = None if args.clip_lr95_class_weight == "none" else args.clip_lr95_class_weight
    else:
        guided_ckpt = args.guided100_ckpt
        vanilla_ckpt = args.vanilla100_ckpt
        gals_ckpt = args.gals100_ckpt
        upweight_ckpt = args.upweight100_ckpt
        abn_ckpt = args.abn100_ckpt
        afr_stage1_ckpt = args.afr100_stage1_ckpt
        afr_last_layer_ckpt = args.afr100_last_layer_ckpt
        clip_lr_C = args.clip_lr100_C
        clip_lr_penalty = args.clip_lr100_penalty
        clip_lr_solver = args.clip_lr100_solver
        clip_lr_fit_intercept = bool(args.clip_lr100_fit_intercept)
        clip_lr_tol = args.clip_lr100_tol
        clip_lr_max_iter = args.clip_lr100_max_iter
        clip_lr_class_weight = None if args.clip_lr100_class_weight == "none" else args.clip_lr100_class_weight

    if "guided" in methods:
        p = _resolve_ckpt(guided_ckpt)
        if p is not None:
            runners["guided"] = GuidedRunner(p, num_classes=num_classes, device=device)
        else:
            print(f"[WARN] Missing guided checkpoint for dataset {dataset_tag}; skipping guided.", flush=True)

    if "vanilla" in methods:
        p = _resolve_ckpt(vanilla_ckpt)
        if p is not None:
            runners["vanilla"] = VanillaRunner(p, num_classes=num_classes, device=device)
        else:
            print(f"[WARN] Missing vanilla checkpoint for dataset {dataset_tag}; skipping vanilla.", flush=True)

    if "gals" in methods:
        p = _resolve_ckpt(gals_ckpt)
        if p is not None:
            runners["gals"] = GALSRunner(p, device=device)
        else:
            print(f"[WARN] Missing GALS checkpoint for dataset {dataset_tag}; skipping gals.", flush=True)

    if "upweight" in methods:
        p = _resolve_ckpt(upweight_ckpt)
        if p is not None:
            runners["upweight"] = UpweightRunner(p, num_classes=num_classes, device=device)
        else:
            print(f"[WARN] Missing upweight checkpoint for dataset {dataset_tag}; skipping upweight.", flush=True)

    if "abn" in methods:
        p = _resolve_ckpt(abn_ckpt)
        if p is not None:
            runners["abn"] = ABNRunner(p, num_classes=num_classes, device=device)
        else:
            print(f"[WARN] Missing ABN checkpoint for dataset {dataset_tag}; skipping abn.", flush=True)

    if "afr" in methods:
        p1 = _resolve_ckpt(afr_stage1_ckpt)
        p2 = _resolve_ckpt(afr_last_layer_ckpt)
        if p1 is not None:
            afr_root = Path(args.afr_root).expanduser().resolve()
            runners["afr"] = AFRRunner(
                afr_root=afr_root,
                stage1_checkpoint=p1,
                stage2_last_layer_checkpoint=p2,
                num_classes=num_classes,
                device=device,
            )
        else:
            print(
                f"[WARN] Missing AFR stage1 checkpoint for dataset {dataset_tag}; skipping afr.",
                flush=True,
            )

    if "clip_zs" in methods:
        class_names = _parse_csv_list(args.clip_class_names)
        if len(class_names) != 2:
            raise ValueError("--clip-class-names must contain exactly 2 class names for Waterbirds.")
        templates = args.clip_template if args.clip_template else _default_clip_templates()
        for t in templates:
            if "{}" not in t:
                raise ValueError(f"Template missing '{{}}' placeholder: {t}")
        runners["clip_zs"] = ClipZeroShotRunner(
            clip_model_name=args.clip_model,
            class_names=class_names,
            templates=templates,
            device=device,
        )

    if "clip_lr" in methods:
        runners["clip_lr"] = ClipLRRunner(
            dataset_path=dataset_path,
            clip_model_name=args.clip_model,
            C=clip_lr_C,
            penalty=clip_lr_penalty,
            solver=clip_lr_solver,
            fit_intercept=clip_lr_fit_intercept,
            tol=clip_lr_tol,
            max_iter=clip_lr_max_iter,
            class_weight=clip_lr_class_weight,
            device=device,
            batch_size=int(args.clip_batch_size),
            num_workers=int(args.clip_num_workers),
        )

    return runners


def _fill_ckpts_from_summaries(args: argparse.Namespace) -> None:
    if args.wb95_saliency_summary:
        args.guided95_ckpt = args.guided95_ckpt or _resolve_ckpt_from_summary(args.wb95_saliency_summary, "guided_checkpoint")
        args.vanilla95_ckpt = args.vanilla95_ckpt or _resolve_ckpt_from_summary(args.wb95_saliency_summary, "vanilla_checkpoint")
        args.gals95_ckpt = args.gals95_ckpt or _resolve_ckpt_from_summary(args.wb95_saliency_summary, "gals_vit_checkpoint")
    if args.wb100_saliency_summary:
        args.guided100_ckpt = args.guided100_ckpt or _resolve_ckpt_from_summary(args.wb100_saliency_summary, "guided_checkpoint")
        args.vanilla100_ckpt = args.vanilla100_ckpt or _resolve_ckpt_from_summary(args.wb100_saliency_summary, "vanilla_checkpoint")
        args.gals100_ckpt = args.gals100_ckpt or _resolve_ckpt_from_summary(args.wb100_saliency_summary, "gals_vit_checkpoint")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    _fill_ckpts_from_summaries(args)

    methods = [m.strip().lower() for m in str(args.methods).split(",") if m.strip()]
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (PARENT_DIR / "logsWaterbird" / f"waterbirds_pointing_game_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] output_dir={out_dir}", flush=True)
    print(f"[INFO] methods={methods}", flush=True)
    print(f"[INFO] datasets={datasets} split={args.split} target_mode={args.target_mode}", flush=True)

    preprocess = build_preprocess()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_summary_rows: List[Dict[str, object]] = []
    all_per_image_rows: List[Dict[str, object]] = []

    for d in datasets:
        if d not in {"95", "100"}:
            print(f"[WARN] Unknown dataset id '{d}', skipping.", flush=True)
            continue

        data_path = Path(args.wb95_data_path if d == "95" else args.wb100_data_path).expanduser().resolve()
        mask_root = Path(args.wb95_mask_root if d == "95" else args.wb100_mask_root).expanduser().resolve()
        if not data_path.is_dir():
            print(f"[WARN] Dataset path missing for {d}: {data_path}; skipping.", flush=True)
            continue
        if not mask_root.is_dir():
            print(f"[WARN] Mask root missing for {d}: {mask_root}; skipping.", flush=True)
            continue

        md = pd.read_csv(data_path / "metadata.csv")
        num_classes = int(md["y"].nunique())
        runners = _build_runners_for_dataset(
            dataset_tag=d,
            dataset_path=data_path,
            num_classes=num_classes,
            methods=methods,
            args=args,
            device=device,
        )
        if not runners:
            print(f"[WARN] No runnable methods for dataset {d}; skipping.", flush=True)
            continue

        print(f"[INFO] Running dataset {d}: methods={list(runners.keys())}", flush=True)
        try:
            summary_rows, per_image_rows = evaluate_dataset(
                dataset_name=f"waterbirds_{d}",
                data_path=data_path,
                mask_root=mask_root,
                split=args.split,
                preprocess=preprocess,
                runners=runners,
                target_mode=args.target_mode,
                max_samples=args.max_samples,
                sample_seed=args.sample_seed,
                device=device,
            )
            all_summary_rows.extend(summary_rows)
            all_per_image_rows.extend(per_image_rows)
        finally:
            for runner in runners.values():
                runner.close()

    summary_csv = out_dir / "pointing_game_summary.csv"
    per_image_csv = out_dir / "pointing_game_per_image.csv"
    _write_csv(summary_csv, all_summary_rows)
    _write_csv(per_image_csv, all_per_image_rows)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\n[DONE] Pointing Game evaluation complete.", flush=True)
    print(f"[DONE] Summary CSV: {summary_csv}", flush=True)
    print(f"[DONE] Per-image CSV: {per_image_csv}", flush=True)


if __name__ == "__main__":
    main()

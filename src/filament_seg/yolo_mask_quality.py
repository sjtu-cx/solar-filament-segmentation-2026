"""Mask-IoU quality head and loss for the vendored YOLO26 segmentation model."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Segment, Segment26
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


def fuse_class_and_mask_quality_scores(
    class_scores: torch.Tensor,
    quality_logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Fuse class confidence with predicted mask quality using a non-negative exponent."""
    if alpha < 0:
        raise ValueError("mask quality alpha must be non-negative")
    return class_scores * quality_logits.sigmoid().pow(alpha)


def binary_mask_iou_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return detached binary mask IoU targets for equally shaped N,H,W tensors."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target masks must have identical shapes")
    predicted = prediction.sigmoid() > 0.5
    truth = target > 0.5
    intersection = (predicted & truth).flatten(1).sum(1).float()
    union = (predicted | truth).flatten(1).sum(1).float()
    return (intersection / union.clamp_min(1.0)).detach()


def scoped_binary_mask_iou_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scope: str,
    boxes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a detached binary MaskIoU target in a cropped or full-image scope."""
    if scope == "cropped":
        if boxes is None:
            raise ValueError("boxes are required for cropped MaskIoU targets")
        prediction = crop_mask(prediction.clone(), boxes)
        target = crop_mask(target.float().clone(), boxes)
    elif scope != "full":
        raise ValueError("mask quality target scope must be 'cropped' or 'full'")
    return binary_mask_iou_target(prediction, target.float())


class QualitySegment26(Segment26):
    """Segment26 with one scalar mask-quality prediction per anchor."""

    def __init__(self, nc=80, nm=32, npr=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        quality_channels = max(ch[0] // 4, 16)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(value, quality_channels, 3),
                Conv(quality_channels, quality_channels, 3),
                nn.Conv2d(quality_channels, 1, 1),
            )
            for value in ch
        )
        if self.end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)
        self.mask_quality_alpha = float(os.getenv("YOLO_MASK_QUALITY_ALPHA", "1.0"))
        self.export_quality_scores = False
        if self.mask_quality_alpha < 0:
            raise ValueError("YOLO_MASK_QUALITY_ALPHA must be non-negative")

    @property
    def one2many(self):
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            mask_head=self.cv4,
            quality_head=self.cv5,
        )

    @property
    def one2one(self):
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv4,
            quality_head=self.one2one_cv5,
        )

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None, quality_head=None):
        predictions = Segment.forward_head(self, x, box_head, cls_head, mask_head)
        if quality_head is not None:
            batch_size = x[0].shape[0]
            predictions["mask_quality"] = torch.cat(
                [quality_head[i](x[i]).view(batch_size, 1, -1) for i in range(self.nl)], dim=2
            )
        return predictions

    def _inference(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        decoded = Detect._inference(self, predictions)
        raw_class_scores = decoded[:, 4 : 4 + self.nc]
        quality_scores = predictions["mask_quality"].sigmoid()
        if getattr(self, "export_quality_scores", False):
            return torch.cat(
                [
                    decoded[:, :4],
                    raw_class_scores,
                    predictions["mask_coefficient"],
                    quality_scores,
                ],
                dim=1,
            )
        class_scores = fuse_class_and_mask_quality_scores(
            raw_class_scores,
            predictions["mask_quality"],
            self.mask_quality_alpha,
        )
        return torch.cat(
            [decoded[:, :4], class_scores, predictions["mask_coefficient"]], dim=1
        )

    def fuse(self) -> None:
        super().fuse()
        self.cv5 = None


class MaskQualitySegmentationLoss(v8SegmentationLoss):
    """Standard segmentation loss plus a detached binary MaskIoU target."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.quality_gain = float(os.getenv("YOLO_MASK_QUALITY_GAIN", "0.5"))
        self.target_scope = os.getenv("YOLO_MASK_QUALITY_TARGET_SCOPE", "cropped")
        if self.quality_gain < 0:
            raise ValueError("YOLO_MASK_QUALITY_GAIN must be non-negative")
        if self.target_scope not in {"cropped", "full"}:
            raise ValueError("YOLO_MASK_QUALITY_TARGET_SCOPE must be 'cropped' or 'full'")

    def _quality_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        foreground: torch.Tensor,
        target_indices: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        quality_logits = predictions["mask_quality"].permute(0, 2, 1).contiguous()
        mask_coefficients = predictions["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = predictions["proto"]
        if isinstance(proto, tuple):
            proto = proto[0]
        masks = batch["masks"].to(self.device).float()
        if tuple(proto.shape[-2:]) != tuple(masks.shape[-2:]):
            proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

        image_size = (
            torch.tensor(
                predictions["feats"][0].shape[2:],
                device=self.device,
                dtype=mask_coefficients.dtype,
            )
            * self.stride[0]
        )
        mask_h, mask_w = proto.shape[-2:]
        normalized_boxes = target_boxes / image_size[[1, 0, 1, 0]]
        mask_boxes = normalized_boxes * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=self.device
        )
        batch_indices = batch["batch_idx"].view(-1).to(self.device)
        losses = []
        for batch_index in range(foreground.shape[0]):
            selected = foreground[batch_index]
            if not selected.any():
                continue
            matched = target_indices[batch_index][selected]
            if self.overlap:
                ground_truth = masks[batch_index] == (matched + 1).view(-1, 1, 1)
            else:
                ground_truth = masks[batch_indices == batch_index][matched] > 0.5
            predicted_logits = torch.einsum(
                "in,nhw->ihw", mask_coefficients[batch_index][selected], proto[batch_index]
            )
            boxes = mask_boxes[batch_index][selected]
            targets = scoped_binary_mask_iou_target(
                predicted_logits,
                ground_truth.float(),
                scope=self.target_scope,
                boxes=boxes,
            )
            losses.append(
                F.binary_cross_entropy_with_logits(
                    quality_logits[batch_index][selected, 0], targets, reduction="mean"
                )
            )
        if not losses:
            return quality_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def loss(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        base_total, base_items = super().loss(predictions, batch)
        # The base loss does not expose its assignment result. Recompute only the
        # detached matching metadata required to construct MaskIoU targets.
        with torch.no_grad():
            (foreground, target_indices, target_boxes, _, _), _, _ = (
                self.get_assigned_targets_and_loss(predictions, batch)
            )
        quality_loss = self._quality_loss(
            predictions, batch, foreground, target_indices, target_boxes
        )
        batch_size = predictions["boxes"].shape[0]
        weighted_quality = quality_loss * self.quality_gain
        total = torch.cat([base_total, (weighted_quality * batch_size).view(1)])
        items = torch.cat([base_items, weighted_quality.detach().view(1)])
        return total, items


class QualitySegmentationModel(SegmentationModel):
    """SegmentationModel using MaskQualitySegmentationLoss in both E2E branches."""

    def init_criterion(self):
        if getattr(self, "end2end", False):
            return E2ELoss(self, MaskQualitySegmentationLoss)
        return MaskQualitySegmentationLoss(self)


def register_mask_quality_yolo_module() -> None:
    """Make model YAML parsing resolve Segment26 to the quality-aware subclass."""
    import ultralytics.nn.tasks as tasks

    tasks.Segment26 = QualitySegment26


def set_mask_quality_alpha(model: Any, alpha: float) -> int:
    """Override the inference exponent on every quality-aware head in a loaded model."""
    value = float(alpha)
    if value < 0:
        raise ValueError("mask quality alpha must be non-negative")
    module = getattr(model, "model", model)
    if not isinstance(module, nn.Module):
        raise TypeError("model must contain a torch.nn.Module")
    heads = [item for item in module.modules() if isinstance(item, QualitySegment26)]
    if not heads:
        raise RuntimeError("No QualitySegment26 head found in the loaded model")
    for head in heads:
        head.mask_quality_alpha = value
    return len(heads)


def set_mask_quality_component_export(model: Any, enabled: bool = True) -> int:
    """Make quality-aware heads append raw quality to NMS extras for diagnostics."""
    module = getattr(model, "model", model)
    if not isinstance(module, nn.Module):
        raise TypeError("model must contain a torch.nn.Module")
    heads = [item for item in module.modules() if isinstance(item, QualitySegment26)]
    if not heads:
        raise RuntimeError("No QualitySegment26 head found in the loaded model")
    for head in heads:
        head.export_quality_scores = bool(enabled)
    return len(heads)


class QualityComponentSegmentationPredictor(SegmentationPredictor):
    """Segmentation predictor that exposes raw class and quality scores after class-score NMS."""

    def construct_result(self, pred, img, orig_img, img_path, proto):
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops

        quality_scores = pred[:, -1]
        if pred.shape[0] == 0:
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(
                proto, pred[:, 6:-1], pred[:, :4], orig_img.shape[:2]
            )
        else:
            masks = ops.process_mask(
                proto, pred[:, 6:-1], pred[:, :4], img.shape[2:], upsample=True
            )
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0
            if not bool(keep.all()):
                pred, masks, quality_scores = pred[keep], masks[keep], quality_scores[keep]
        result = Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            masks=masks,
        )
        result.quality_scores = quality_scores.detach().cpu()
        return result


class QualitySegmentationTrainer(SegmentationTrainer):
    """Trainer that builds the quality-aware segmentation model."""

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        register_mask_quality_yolo_module()
        model = QualitySegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (
            "box_loss",
            "seg_loss",
            "cls_loss",
            "dfl_loss",
            "sem_loss",
            "quality_loss",
        )
        return validator

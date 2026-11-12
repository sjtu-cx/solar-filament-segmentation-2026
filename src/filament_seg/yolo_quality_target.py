"""Quality-aware classification targets for YOLO26 instance segmentation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.tal import make_anchors

from filament_seg.yolo_mask_quality import scoped_binary_mask_iou_target


def blend_quality_target_scores(
    target_scores: torch.Tensor,
    foreground: torch.Tensor,
    mask_quality: torch.Tensor,
    mix: float,
    floor: float = 0.0,
) -> torch.Tensor:
    """Blend task-aligned positive scores with detached mask-IoU targets."""
    if target_scores.shape[:-1] != foreground.shape or target_scores.shape[-1] != 1:
        raise ValueError("quality-target training currently requires one class")
    if mask_quality.shape != (int(foreground.sum()),):
        raise ValueError("mask_quality must contain one value per foreground anchor")
    if not 0.0 <= mix <= 1.0 or not 0.0 <= floor <= 1.0:
        raise ValueError("mix and floor must be between zero and one")
    output = target_scores.clone()
    bounded_quality = floor + (1.0 - floor) * mask_quality.detach().clamp(0.0, 1.0)
    output[foreground] = (
        (1.0 - mix) * output[foreground]
        + mix * bounded_quality.to(output.dtype).unsqueeze(-1)
    )
    return output


class QualityTargetSegmentationLoss(v8SegmentationLoss):
    """Use predicted mask IoU as part of the original classification target."""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.quality_mix = float(os.getenv("YOLO_QUALITY_TARGET_MIX", "0.5"))
        self.quality_floor = float(os.getenv("YOLO_QUALITY_TARGET_FLOOR", "0.0"))
        self.target_scope = os.getenv("YOLO_QUALITY_TARGET_SCOPE", "cropped")
        if not 0.0 <= self.quality_mix <= 1.0:
            raise ValueError("YOLO_QUALITY_TARGET_MIX must be between zero and one")
        if not 0.0 <= self.quality_floor <= 1.0:
            raise ValueError("YOLO_QUALITY_TARGET_FLOOR must be between zero and one")
        if self.target_scope not in {"cropped", "full"}:
            raise ValueError("YOLO_QUALITY_TARGET_SCOPE must be 'cropped' or 'full'")
        if self.nc != 1:
            raise ValueError("Quality-target training is implemented only for the single-class task")

    def _mask_quality_targets(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        foreground: torch.Tensor,
        target_indices: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = predictions["mask_coefficient"].permute(0, 2, 1).contiguous()
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
                dtype=coefficients.dtype,
            )
            * self.stride[0]
        )
        mask_h, mask_w = proto.shape[-2:]
        normalized_boxes = target_boxes / image_size[[1, 0, 1, 0]]
        mask_boxes = normalized_boxes * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h], device=self.device
        )
        batch_indices = batch["batch_idx"].view(-1).to(self.device)
        targets = []
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
                "in,nhw->ihw", coefficients[batch_index][selected], proto[batch_index]
            )
            targets.append(
                scoped_binary_mask_iou_target(
                    predicted_logits,
                    ground_truth.float(),
                    scope=self.target_scope,
                    boxes=mask_boxes[batch_index][selected],
                )
            )
        if not targets:
            return coefficients.new_zeros((0,))
        return torch.cat(targets)

    def get_assigned_targets_and_loss(
        self, predictions: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ):
        """Preserve stock assignment and box losses, changing classification targets only."""
        loss = torch.zeros(3, device=self.device)
        pred_distri = predictions["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = predictions["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(predictions["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = (
            torch.tensor(predictions["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=image_size[[1, 0, 1, 0]]
        )
        gt_labels, gt_boxes = targets.split((1, 4), 2)
        valid_gt = gt_boxes.sum(2, keepdim=True).gt_(0.0)
        pred_boxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_boxes, target_scores, foreground, target_indices = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_boxes.detach() * stride_tensor).type(gt_boxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_boxes,
            valid_gt,
        )

        original_scores = target_scores
        original_sum = max(original_scores.sum(), 1)
        if foreground.any() and self.quality_mix > 0.0:
            with torch.no_grad():
                quality = self._mask_quality_targets(
                    predictions, batch, foreground, target_indices, target_boxes
                )
            classification_targets = blend_quality_target_scores(
                original_scores,
                foreground,
                quality,
                mix=self.quality_mix,
                floor=self.quality_floor,
            )
        else:
            classification_targets = original_scores
        classification_sum = max(classification_targets.sum(), 1)
        loss[1] = self.bce(pred_scores, classification_targets.to(dtype)).sum() / classification_sum

        if foreground.any():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_boxes,
                anchor_points,
                target_boxes / stride_tensor,
                original_scores,
                original_sum,
                foreground,
                image_size,
                stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (foreground, target_indices, target_boxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class QualityTargetSegmentationModel(SegmentationModel):
    """YOLO segmentation model with quality-aware class targets."""

    def init_criterion(self):
        if getattr(self, "end2end", False):
            return E2ELoss(self, QualityTargetSegmentationLoss)
        return QualityTargetSegmentationLoss(self)


class QualityTargetSegmentationTrainer(SegmentationTrainer):
    """Trainer that keeps the stock model head and changes only its class target."""

    def get_model(
        self,
        cfg: dict | str | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ):
        model = QualityTargetSegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

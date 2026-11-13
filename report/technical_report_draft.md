# High-Resolution Solar Filament Instance Segmentation with Consensus Labels and Quality-Aware Calibration

> Earlier editorial draft retained for reference. The authoritative, expanded report is the official
> Overleaf source in `main_draft.tex`, `preamble.tex`, and `main.bib`.

**Team:** Cisdi Info Machine Vision Team  
**Authors:** CaoXin  
**Institution:** CISDI Information Technology CO., LTD.  
**Contact:** Xin.A.Cao@cisdi.com.cn

## Abstract

We present a reproducible high-resolution instance-segmentation pipeline for solar filaments in
GONG H-alpha observations. The method uses YOLO26m-seg at 2048-pixel resolution, one physical image
per observation, majority-consensus aggregation of repeated annotations, and conservative
flip-based test-time augmentation. We select QT01TTAH as the single final submission and retain
Y17-V as a robust reference. Y17-V uses the standard training objective with vertical-flip TTA and
achieves two-fold OOF PQ 0.4575. QT01TTAH mixes detached
predicted mask IoU into positive classification targets and uses horizontal-flip TTA, achieving
two-fold OOF PQ 0.4568. Both variants score 0.39 on the public leaderboard. Extensive ablations show
that larger models, unconditional mask refinement, aggressive recall expansion, static topology
repair and simple cross-model candidate unions do not generalize reliably. The primary remaining
bottleneck is selecting high-quality candidates without increasing false positives.

## 1. Problem and data

The task is single-class instance segmentation of thin, low-contrast solar filaments. The released
training data contain 707 physical observations. Some observations have multiple independent
annotation passes, producing substantial disagreement in weak structures and boundaries. All
annotation versions of a physical image are kept in the same validation fold to prevent leakage.

For training, repeated passes are grouped into instance clusters using mask overlap. An instance is
retained when supported by a majority of available passes. The resulting representation uses each
physical image exactly once, preventing images with more annotation passes from receiving excess
training weight. Under the official IoU-above-0.5 matching rule, the 145 observations with exactly
two annotation passes have pairwise global PQ 0.3596 (559 TP, 476 FP and 393 FN), demonstrating
substantial label ambiguity. Union-expanded majority boundaries lowered Fold0/Fold1 PQ by
0.0207/0.0382 relative to strict majority labels.

## 2. Method

### 2.1 Base segmentation model

The base model is YOLO26m-seg initialized from the official segmentation checkpoint. Images are
resized to 2048 pixels. With architecture and majority labels fixed, moving from 1536 to 2048
improved Fold0 PQ from 0.4257 to 0.4342 and Fold1 from 0.4356 to 0.4620; the corresponding public
score rose from 0.37 to 0.38. We disable mosaic, mixup and copy-paste because these augmentations damaged
the continuity and morphology of thin filament structures. Mild rotation, translation, scale,
brightness variation and horizontal/vertical flips are retained. Training uses seed 42, batch size
4, mixed precision and deterministic execution where supported.

### 2.2 Y17-V

Y17-V is trained on all 707 majority-consensus images for 22 epochs using the standard YOLO26
detection and instance-mask objectives. At inference, the model processes the original image and a
vertical flip. Flipped masks are restored to the original coordinate system. Predictions with mask
IoU above 0.50 are deduplicated, retaining the higher-confidence mask. The final filter uses
confidence 0.28, minimum area 320 pixels and at most 100 instances per image.

### 2.3 QT01 quality-target training

YOLO classification confidence mainly represents target existence and does not necessarily rank
mask shape quality. QT01 computes a soft IoU between each assigned positive's predicted prototype
mask and its matched ground-truth mask after both are cropped to the target box. The IoU target is
detached and mixed equally with the original task-aligned classification target. This changes score
calibration and candidate ordering, not decoded mask geometry; assignment, box and mask losses remain
unchanged. The full model is trained
for 26 epochs. Final inference combines the original and horizontal-flip views, followed by the same
best-mask deduplication. We use confidence 0.34 and minimum area 256 pixels.

### 2.4 Parameter calibration and deterministic post-processing

Raw candidates are exported once with model confidence 0.001, YOLO NMS IoU 0.70 and at most 100
instances per image. All parameter sweeps reuse these cached predictions. Flip masks are restored
to the original 2048-by-2048 coordinates. Candidates are sorted by confidence; masks with IoU at
least 0.50 are treated as duplicates and the higher-confidence mask is retained.

Y17 jointly evaluates horizontal, vertical and horizontal-plus-vertical TTA; confidence values
0.28, 0.32, 0.36 and 0.40; and minimum areas 256 and 320. The selected vertical-flip,
confidence-0.28, area-320 configuration maximizes mean official PQ over Fold0 and Fold1. QT01TTAH
searches 12 confidence values from 0.20 to 0.40 and areas 128, 256, 320 and 384 on Fold0. Promotion
requires a PQ gain of at least 0.0108, SQ regression no larger than 0.003 and at most 50 additional
false positives. Confidence 0.34 and area 256 pass this gate and are then frozen for Fold1 and the
full-data model; public scores are not used for calibration.

Final masks are processed in descending score order. Each candidate is clipped against pixels
already occupied by higher-scoring instances, its area is recomputed after clipping, and empty or
sub-threshold masks are discarded. Accepted masks update the occupied map and are encoded as COCO
RLE. This deterministic rule gives zero output overlap while preserving score priority.

## 3. Validation protocol

We use grouped cross-validation so that repeated annotations of a physical observation and images
from the same observation group do not cross training and validation boundaries. Y17 uses a small
predefined grid and cross-fold consistency selection: a setting is retained only when it improves
both Fold0 and Fold1. QT01 instead selects its operating point on Fold0 and evaluates it unchanged on
Fold1, making Fold1 a frozen confirmation set for QT01 but not for Y17. We report Panoptic Quality (PQ), Segmentation Quality
(SQ), Recognition Quality (RQ), and TP/FP/FN counts. A method is promoted to full-data training only
when improvements are consistent across folds; leaderboard score is not used as the sole selection
criterion.

This matches the organizer's broader rubric: quantitative evidence contributes 70% and includes
PQ, overlap distributions and one-to-many/many-to-one behavior, while pipeline clarity, predicted
morphology and code quality contribute 30%. Public PQ is therefore treated as one validation signal,
not as the only final-selection criterion.

| Method | Fold0 PQ | Fold1 PQ | Mean PQ | Public PQ |
|---|---:|---:|---:|---:|
| Y10/Y13 base | 0.4342 | 0.4620 | 0.4481 | 0.38 |
| Y17-V | 0.4430 | 0.4720 | **0.4575** | **0.39** |
| QT01TTAH | 0.4485 | 0.4651 | 0.4568 | **0.39** |

Y17-V improves both folds over their matched base models. QT01TTAH is close in mean PQ but changes
the score formation mechanism. It is selected as the single final submission, while Y17-V remains
the reproducible reference baseline.

## 4. Ablation findings

Increasing capacity to YOLO26l reduced Fold0 PQ, showing that model size is not the dominant
bottleneck for 707 images. RF-DETR Seg, Mask2Former, EdgeCrafter and a ConvNeXt-Tiny backbone did not
outperform the YOLO26m baseline as standalone systems. Unconditional U-Net/SegFormer ROI refinement
reduced PQ despite a high oracle mask-replacement ceiling. Static topology repair, tiled inference,
difficult-image oversampling, disagreement-region loss weighting and instance SoftIoU either
regressed or failed cross-fold validation.

Stored post-NMS YOLO candidates at confidence 0.001, NMS IoU 0.70 and max_det 100 cover 91.86% of
ground-truth instances at IoU above 0.5, compared with 62.09% after final filtering. A ground-truth
oracle with fixed output count reaches five-fold PQ 0.5737 versus approximately 0.4332; this is a
diagnostic upper bound and is never used for test inference. Learned global, pairwise and visual rerankers produced only noise-level PQ
improvements. This indicates that candidate competition near the output cutoff, rather than raw
candidate availability, is the main unresolved error source.

The latest structured-selection controls reinforce this conclusion. BD01 global calibration lowers
the uniform two-fold mean from 0.4481 to 0.4441; XM01R cross-model matched-pair selection changes
mean PQ by -0.00064; and CS01 constrained subset selection is negative on all five folds (mean
-0.01944). These results reject the corresponding learned selectors, not the diagnostic finding
that a large candidate oracle gap exists.

## 5. Qualitative behavior and limitations

The selected systems preserve elongated filament bodies and suppress many compact sunspot-like
false positives. TTA improves recall for faint structures while conservative thresholds prevent the
large false-positive increase observed with permissive settings. Remaining failures include weak
filaments with no sufficiently overlapping candidate, fragmentation near low-contrast boundaries,
and ambiguity caused by inconsistent annotations.

The limited dataset and annotation disagreement make single-fold gains unreliable. Public and OOF
PQ also differ, so final selection considers cross-fold consistency, morphology, error counts and
reproducibility in addition to leaderboard score.

## 6. Reproducibility

Training and validation used one NVIDIA RTX A6000 GPU with 46 GB memory and the repository's
Ultralytics 8.4.3 checkout. The clean release environment pins Ultralytics 8.4.15 and reproduces the
final CSV hashes from archived candidates. The standard 22-epoch
full-data run took 1.119 hours; QT01's 26 epochs took 1 hour 33 minutes 30 seconds, with approximately
33.7 GB peak memory recorded in both logs. The Y17 validation log reports 52.5 ms/image model-only
inference. QT01 base prediction over 180 test images took 7 minutes 7 seconds, while one additional
horizontal TTA view including full-resolution mask export and fusion took 14 minutes 20 seconds.
The public repository contains exact configurations, dependency versions, data preparation code,
training and inference entry points, a complete Notebook, final checkpoints, and SHA256 manifests.
The inference pipeline generates COCO RLE masks, verifies the required columns and test identifiers,
checks mask dimensions and decodability, and rejects overlapping output instances. Both selected
CSV files pass this audit.

Only official competition H-alpha images and annotations are used. Specifically, public MAGFiLO
annotations corresponding to competition test observations were neither accessed nor used. No
external data, test-set hand labeling, or private test metadata enters training, calibration, or
inference.

## References

1. Kirillov et al., “Panoptic Segmentation,” CVPR, 2019.
2. Ahmadzadeh et al., “MAGFiLO: a manually annotated dataset of solar filaments,” *Scientific Data*, 2024, doi:10.1038/s41597-024-03876-y.
3. Ultralytics, “Ultralytics YOLO,” software release 8.4.15, 2026.
4. Cheng et al., “Masked-attention Mask Transformer for Universal Image Segmentation,” CVPR, 2022.
5. Xie et al., “SegFormer: Simple and Efficient Design for Semantic Segmentation,” NeurIPS, 2021.
6. Liu et al., “A ConvNet for the 2020s,” CVPR, 2022.

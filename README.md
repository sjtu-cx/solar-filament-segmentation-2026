# Solar Filament Segmentation Challenge 2026 - Final Submission Package

**Team:** Cisdi Info Machine Vision Team  
**Author:** CaoXin  
**Contact:** Xin.A.Cao@cisdi.com.cn

This repository reproduces the two selected final submissions with one codebase and one executable
Notebook. Both obtained public Panoptic Quality (PQ) `0.39`, but they are retained for different
technical reasons:

| Final slot | File | Role | Grouped Fold0/Fold1 PQ | Mean PQ |
|---|---|---|---:|---:|
| 1 | `Y17_V_full_submission.csv` | Robust primary: standard objective plus vertical-flip TTA | 0.443035 / 0.471995 | **0.457515** |
| 2 | `QT01TTAH_full_submission.csv` | Method-diverse alternative: quality-aware target plus horizontal-flip TTA | 0.448509 / 0.465089 | **0.456799** |

Y17-V is the highest cross-fold mean and the simpler system. QT01TTAH changes how classification
scores are trained by mixing detached mask IoU into positive targets; it therefore provides a
meaningful calibration alternative rather than a duplicate checkpoint. A simple union of the two
systems was rejected because it added false positives and reduced validation PQ.

The validation roles are intentionally distinguished: Y17-V was chosen by consistency over a small
predefined Fold0/Fold1 grid, whereas QT01 selected its operating point on Fold0 and evaluated it
unchanged on Fold1. Fold1 is therefore a frozen confirmation set for QT01, but not for Y17-V.

## Organizer requirements addressed here

The competition requires a public Git repository, versioned `requirements.txt`, an end-to-end
Jupyter Notebook, and a four-page PDF report. The official evaluation combines quantitative
evidence (70%, including PQ and overlap/matching distributions) with qualitative methodology,
morphology, and code quality (30%). The final report/solution deadline is **November 15, 2026**;
the exact closing time shown by Kaggle remains authoritative.

Official page: <https://www.kaggle.com/competitions/filament-segmentation-2026/overview>

The report uses the competition's [official Overleaf template](https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe).
Upload `report/Official_Overleaf_Source.zip` to Overleaf for an immediately compilable project; the
compiled four-page report is
`report/Solar_Filament_Final_Technical_Report.pdf`.

## License

Original source code in this repository is licensed under `AGPL-3.0-only`; see `LICENSE` and
`NOTICE`. Copyright (C) 2026 CISDI Information Technology CO., LTD. Competition data are not
redistributed or relicensed, and third-party components retain their original terms.

One public repository is sufficient for both final submissions because the `--variant` argument
selects the complete training, inference, and post-processing path for each system.

## Data provenance

Only the official competition H-alpha images and annotations are used. Public MAGFiLO annotations
corresponding to competition test observations were neither accessed nor used. No external training
data, test-set labels, hand labeling, or private test metadata are used. Competition data are not
redistributed. Extract the official download to:

```text
data/MAGFiLO_1.0_Kaggle_2026/
|-- train/
|   |-- MAGFiLO_1.0_Annotations_kaggle2026_train.json
|   `-- train_images/
`-- test/test_images/
```

The released training set contains 707 physical observations and 1,154 annotation records. All
annotation passes for the same physical observation stay in one fold. Majority-consensus
aggregation produces one training example per observation and prevents repeat annotations from
receiving excess optimization weight.

## Repository layout

```text
artifacts/                  archived raw predictions and weights
configs/final/              exact training, TTA, and filtering settings
notebooks/final_pipeline.ipynb
  report/                     official-Overleaf LaTeX source, final PDF, and figures
scripts/                    preparation, training, inference, audit, and verification entry points
src/filament_seg/           minimal custom implementation used by the selected systems
submissions/                final CSV files and independent format audits
  FINAL_SUBMISSION_MANIFEST.json
requirements.txt
```

## Environment

The original selected checkpoints were trained with the repository's Ultralytics 8.4.3 checkout.
The clean release environment is Python 3.8.13, PyTorch 1.10.2+cu113, TorchVision 0.11.3+cu113, and
installed Ultralytics 8.4.15; package verification and exact CPU CSV reconstruction pass there. Full
training used one NVIDIA RTX A6000 (46 GB), 2048-pixel inputs, automatic mixed precision, and batch
size 4. Exact augmentation and optimizer-facing arguments are recorded in `configs/final/*.yaml`
rather than hidden in a launcher.

Measured logs report 1.119 hours for the 22-epoch standard full-data run and 1 hour 33 minutes
30 seconds for QT01's 26 epochs, with approximately 33.7 GB peak GPU memory. Model-only validation
inference was 52.5 ms/image. Full-resolution mask serialization dominates end-to-end TTA time: the
additional horizontal view over 180 test images took 14 minutes 20 seconds including export/fusion.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Artifact download and integrity

Large weights may be distributed as public release assets instead of Git objects. After the public
URLs are filled in `FINAL_SUBMISSION_MANIFEST.json`, download and hash-check them with:

```bash
python scripts/download_artifacts.py
python scripts/verify_package.py
```

`verify_package.py` checks both final CSVs, all three checkpoints, both raw prediction files,
audits, hashes, Notebook structure, report files, and required configurations. Before publishing,
run the stricter check:

```bash
python scripts/verify_package.py --publication-ready
```

It fails if release URLs or author metadata are pending, a license is missing, or public-facing
materials contain private IP addresses or workstation/server paths.

## Fast exact CSV reconstruction (CPU)

The archived JSONL candidates make the final filtering, overlap resolution, COCO RLE conversion,
and CSV audit reproducible without rerunning the GPU model:

```bash
python scripts/run_final_solution.py --variant y17_v --stage rebuild-csv
python scripts/run_final_solution.py --variant qt01_ttah --stage rebuild-csv
```

Outputs are written to `work/submissions/`. Their hashes must match the manifest.

## Inference from final weights

```bash
python scripts/run_final_solution.py --variant y17_v --stage predict --device 0
python scripts/run_final_solution.py --variant qt01_ttah --stage predict --device 0
```

Frozen inference settings are:

| Variant | Views | Duplicate fusion | Confidence | Minimum area | Maximum instances |
|---|---|---:|---:|---:|---:|
| Y17-V | original + vertical flip | best mask, IoU 0.50 | 0.28 | 320 px | 100 |
| QT01TTAH | original + horizontal flip | best mask, IoU 0.50 | 0.34 | 256 px | 100 |

Raw model candidates use confidence `0.001` and NMS IoU `0.70`. After TTA restoration and
deduplication, candidates are sorted by score. Pixels already assigned to a stronger instance are
removed, area is recomputed, and empty or sub-threshold masks are discarded. This deterministic
policy guarantees non-overlapping output instances.

## Training reproduction

Prepare grouped folds and the 707-image majority-consensus dataset:

```bash
python scripts/run_final_solution.py --variant y17_v --stage prepare
```

Train either model, or run training followed by its complete inference path:

```bash
python scripts/run_final_solution.py --variant y17_v --stage train --device 0
python scripts/run_final_solution.py --variant qt01_ttah --stage train --device 0
python scripts/run_final_solution.py --variant qt01_ttah --stage all --device 0
```

Y17-V trains for 22 epochs with the standard objective. QT01 trains for 26 epochs and uses
`quality_target_mix=0.5`; predicted soft mask IoU is detached before it is mixed into the positive
classification target, while box and mask losses are unchanged. CUDA kernels can introduce small
numerical differences across GPU architectures even with deterministic execution requested.

## Notebook

Open `notebooks/final_pipeline.ipynb` and choose **Run All**. By default it reconstructs and verifies
both final CSVs. Set `RUN_PREPARE`, `RUN_TRAINING`, or `RUN_GPU_INFERENCE` to `True` to execute those
long stages. `SOLAR_DATA_ROOT`, `SOLAR_WORK_DIR`, and `SOLAR_DEVICE` can override local paths/device
without editing package code.

## Final submission audit

Both selected CSV files contain exactly the required `filament_id` and `segmentation_rle` columns,
cover the official 180 test identifiers, use decodable 2048 x 2048 COCO RLE masks, have unique and
contiguous per-image instance IDs, and contain zero overlapping output pixels. The manifest records
the exact SHA256 of every submission, final checkpoint, pretrained checkpoint, and cached raw
prediction file.

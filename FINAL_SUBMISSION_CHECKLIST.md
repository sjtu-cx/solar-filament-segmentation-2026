# Final Submission Checklist

## Kaggle final selections

- [ ] Select `Y17_V_full_submission.csv` as Final Submission 1.
- [ ] Select `QT01TTAH_full_submission.csv` as Final Submission 2.
- [ ] Confirm the Kaggle submission descriptions and public scores are `0.39`.
- [ ] Submit the final form, repository URL and four-page PDF by November 15, 2026 (use Kaggle's exact closing time).

## Reproducibility package

- [x] Final CSV files archived.
- [x] CSV format audits archived.
- [x] Final training checkpoints archived locally.
- [x] Official pretrained checkpoint archived locally.
- [x] Raw final TTA predictions archived for exact CSV reconstruction.
- [x] Environment versions recorded.
- [x] Portable data preparation, training, inference and submission entry point added.
- [x] One executable Notebook covers both variants and defaults to exact CSV reconstruction.
- [x] Raw predictions are included in the manifest and protected by SHA256.
- [x] Publication-mode validation rejects private paths, pending links and report placeholders.
- [ ] Re-run exact CSV reconstruction in a clean environment.
- [ ] Re-run inference from both final checkpoints and compare hashes/audits.
- [ ] Upload large checkpoints to a public release and fill in the manifest URLs.

## Public repository

- [x] Add the OSI-approved AGPL-3.0 license aligned with Ultralytics.
- [ ] Initialize the final package as a Git repository.
- [ ] Confirm no competition data, credentials, private IPs, logs or unrelated experiments are included.
- [ ] Publish the repository without access approval.
- [ ] Confirm the repository remains public through the winner announcement.
- [ ] Test all documented commands from a fresh clone.
- [ ] Run `python scripts/verify_package.py --publication-ready` and save the successful output.

## Four-page report

- [x] Initial technical content drafted.
- [x] Insert team name, authors and contact information.
- [x] Adapt the report to the official Overleaf `acmart` template and retain its fixed text.
- [x] Add final qualitative examples and OOF metric plots.
- [x] Add measured efficiency, validation-protocol clarification, references and data-compliance statement.
- [x] Keep the report within the four-page limit (current export: exactly four fully populated pages).
- [x] Export and inspect the PDF layout.

## Organizer form

- [ ] Complete the official Google Form before the stated deadline.
- [ ] Add the public repository URL.
- [ ] Upload the four-page PDF.
- [ ] Verify team and contact details.
- [ ] Save a copy or screenshot of the submitted form response.

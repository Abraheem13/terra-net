# ID6 – HEAL-GTN revised manuscript (IEEE ICCD-2026)

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Revised manuscript. IEEE conference template (IEEEtran, 6/27/2024 version), US Letter, 10 pt. |
| `IEEEtran.cls` | IEEE class file, unmodified. |
| `figures/*.pdf` | Vector figures, Times 8 pt, sized to print at 100%. |
| `figures/src/*.tex` | TikZ/pgfplots sources for the figures. Compile each one with `pdflatex`. |
| `ID6-HEAL-GTN.pdf` | Compiled PDF. Pages 1–8 are the paper; pages 9–10 are the Response to Reviewers the editor asked for. |
| `CITATION_AUDIT.md` | Reference-by-reference audit. |

## Compiling

Run `pdflatex main.tex` twice. The bibliography is inline (`thebibliography`), so no BibTeX run is needed.

## Submission notes

- **Title in the submission form:** enter `ID6-HEAL-GTN: Mining Longitudinal Electronic Health Records with a Hierarchical Graph-Temporal Network for Early Sepsis, Acute Kidney Injury and Mortality Prediction`, as the editor requested.
- **Printed title:** the paper itself keeps the title without the ID prefix.

## Items to confirm with the first author before submitting

1. **Baseline citations.** CT-BERT and SANet need their correct citations (see `CITATION_AUDIT.md`, section C).
2. **Repository.** `https://github.com/sathishv/heal-gtn` must be public.
3. **Title.** The title still says "Early Sepsis". The revised text treats the results as admission-level, retrospective classification. Consider whether to drop "Early".
4. **Code vs paper.** The released repository does not match parts of the paper:
   - `scripts/train.py` passes empty graph inputs, and `HealGTN.forward` feeds `batch["encounter_seq"]` straight to the temporal encoder. As released, the fusion and graph outputs therefore never reach the heads.
   - The counterfactual loss is defined but never computed during training (`cf_consistency` is never set).
   - The Cox weight in `total_loss` is 1.0, not 0.5.
   - `scripts/evaluate.py` builds the model with no edge types, so it cannot load a trained checkpoint.
   - The ablation configs used by `table4_ablation.sh` do not exist.
   - `report_calibration`, which `figure6_calibration.sh` calls, does not exist.
   - There is no script that builds the real `HEALDataset` from MIMIC-IV.
   - The sepsis code list includes R65.10 (non-infectious SIRS).
   - The AKI baseline is the admission minimum, which includes values measured after the event.
   - The paper now describes the code as released and states the leakage limitations openly.
   - Re-running with time-restricted inputs, with the label codes removed from the graph, is the single change most likely to satisfy Reviewer 2.
5. **Material removed because it could not be supported:**
   - Self-supervised pretraining: the claimed 1.2 million admissions exceeds the size of MIMIC-IV and eICU-CRD combined.
   - The "SepsisWatch curation" cohort-sensitivity study.
   - The data-poisoning result, which relied on median aggregation the code does not implement.
   - The model-inversion bound.
   - The INT8 result.
   - PhysioNet 2019, which has no reported results.

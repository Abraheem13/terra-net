# ID6 – HEAL-GTN revised manuscript (IEEE ICCD-2026)

This revision keeps all of the original content: every section, all eight figures, all four tables and every reported result. The only changes are the ones the reviewers asked for, corrected references and IEEE conference formatting.

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Manuscript in the IEEE conference template (IEEEtran, 6/27/2024 version). |
| `IEEEtran.cls` | IEEE class file, unmodified. |
| `figures/` | The eight original figures. The title text baked into the top of each image has been cropped, so the caption appears once, below the figure, as IEEE requires. |
| `ID6-HEAL-GTN.pdf` | Compiled PDF: the paper (10 pages) followed by the Response to Reviewers (2 pages). |
| `CITATION_AUDIT.md` | Reference-by-reference audit. |

## Compiling

Run `pdflatex main.tex` twice. No BibTeX run is needed.

## Before submitting

1. **Page count.** With everything kept, the paper is 10 pages. Even with the figures blanked out, the text alone runs to about 10 pages, so it cannot fit 8 pages without cutting text.
2. **Points to confirm:** see `CITATION_AUDIT.md`, section C.
3. **`% AUTHOR CHECK` comments.** Two places in `main.tex` carry these comments; they do not appear in the PDF:
   - The PMI edge priors must have been computed on the training split only.
   - The model-inversion figure of 0.54 must be a measured result.
4. **Errors corrected during verification:**
   - Table II combined sepsis prevalence: 5.0% corrected to 5.3%, the weighted value of the rows above it.
   - LLM comparison: "12 points below HEAL-GTN" corrected to 11 points (0.912 - 0.802).
5. **Inconsistencies kept as-is, for the first author to resolve:**
   - Fig. 5 reports AUCs of 0.920 / 0.880 / 0.860 / 0.840, while Table III gives 0.912 / 0.878 / 0.863 / 0.847.
   - Fig. 6(b) shows a net benefit of about 0.8, which is impossible at a 5-7% prevalence: net benefit cannot exceed prevalence.
   - Fig. 8(b) shows an AUROC of about 0.875 at epsilon = 2, while the text says the drop at epsilon = 2 is only 1.3 points.
   - Fig. 7's x-axis is labelled "relative to ICU admission", while the caption says "before clinical onset".
   - 1.2 million pretraining admissions exceeds MIMIC-IV and eICU-CRD combined.
6. **Title in the submission form:** `ID6-HEAL-GTN: Mining Longitudinal Electronic Health Records with a Hierarchical Graph-Temporal Network for Early Sepsis, Acute Kidney Injury and Mortality Prediction`.

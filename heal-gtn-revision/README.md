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

## Verification status

- **References:** all 65 were checked field by field against publisher and index records. See `CITATION_AUDIT.md`.
- **Numbers:** every number in the paper was cross-checked against the original manuscript and against the paper's own tables. Two arithmetic errors were corrected:
  - Table II combined sepsis prevalence: 5.0% → 5.3%.
  - LLM comparison: 12 → 11 AUROC points.
- **Fig. 7:** the x-axis label was corrected from "Time relative to ICU admission" to "Time relative to clinical onset". This matches the caption, the text and the t−48 h to t−6 h tick marks.
- **No placeholders:** no TODOs, hidden comments, unresolved references or unfilled fields remain.

## Points only the first author can settle

These concern the original experiments, not the writing or the references:

1. **Fig. 5:** the legend AUCs (0.920 / 0.880 / 0.860 / 0.840) differ from Table III (0.912 / 0.878 / 0.863 / 0.847).
2. **Fig. 6(b):** the net benefit reaches about 0.8, but net benefit cannot exceed the 5–7% prevalence.
3. **Fig. 8(b):** the AUROC at epsilon = 2 is about 0.875, but the text gives a 1.3-point drop.
4. **Pretraining:** 1.2 million pretraining admissions is more than MIMIC-IV and eICU-CRD hold combined.
5. **Baseline identities:**
   - CT-BERT is cited to GatorTron [18].
   - SANet is cited to SAnD [58].
6. **Page count:** 10 pages with all content kept. Fitting 8 pages requires cutting text.

## Submission

Enter this title in the submission form: `ID6-HEAL-GTN: Mining Longitudinal Electronic Health Records with a Hierarchical Graph-Temporal Network for Early Sepsis, Acute Kidney Injury and Mortality Prediction`.

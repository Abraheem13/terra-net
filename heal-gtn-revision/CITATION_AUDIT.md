# Citation audit: HEAL-GTN revision (ID6)

## How the references were checked

- **What was checked:** every reference was checked for:
  - authors;
  - title;
  - venue;
  - year;
  - volume, issue and pages or article number;
  - whether it supports the sentence that cites it.
- **Where:** against the publisher or index record, such as PubMed/PMC, ACL Anthology, PMLR, AAAI OJS, NeurIPS proceedings, IEEE Xplore, ACM, JAIR, BMJ, Nature and arXiv.
- **Google Scholar:** this environment's network policy blocks it, so it was not queried directly. The records above are the same ones Google Scholar indexes.
- **Unconfirmed fields:** where a page range could not be confirmed (RETAIN, Yeom et al., DeepHit, the NeurIPS papers), it was left out rather than guessed.
- **Format:** all entries use IEEE style and are numbered in order of first citation, 65 in total.

## A. Original 40 references

| Old | Reference | Finding | Action (new no.) |
|---|---|---|---|
| [1] | Jensen et al. 2012 | Correct | [1] |
| [2] | Miotto et al. 2018 | Correct | [2] |
| [3] | Singer et al. 2016 | Correct | [3] |
| [4] | KDIGO guideline | Author is the KDIGO AKI Work Group, not "J. A. Kellum et al." | Corrected [4] |
| [5] | Johnson et al., MLHC 2017 | Correct | [5] |
| [6] | RETAIN, NeurIPS 2016 | Correct. Sources disagree on pages, so they are left out. | [8] |
| [7] | Shickel et al. 2018 | Correct | [9] |
| [8] | Alsentzer et al. 2019 | Correct | [11] |
| [9] | BioGPT 2022 | Correct; article no. added | [12] |
| [10] | Harutyunyan et al. 2019 | Correct, but the text said it benchmarked CNN-LSTM; it used LSTM and channel-wise LSTM | Text and Table I corrected [13] |
| [11] | Vaswani et al. 2017 | Correct | [14] |
| [12] | Informer 2021 | Correct | [15] |
| [13] | BEHRT 2020 | Correct. Table I said it uses text; it uses coded records. | Table I corrected [16] |
| [14] | Med-BERT 2021 | Correct. Table I said it uses text; it uses structured codes. | Table I corrected [17] |
| [15] | "Z. Yang et al.", cited for CT-BERT | Real paper, but the first author is **X.** Yang and the model is **GatorTron**, not CT-BERT | Bibliographic entry corrected [18]; **CT-BERT identity must be confirmed** |
| [16] | GRAM 2017 | Correct | [22] |
| [17] | "C. Shang et al.", cited for SANet and mortality | First author is **J.** Shang; the paper is G-BERT (medication recommendation) | Corrected and now cited for medication recommendation [36]. SANet cited to Song et al. [58]; mortality estimation to GCT [23]. |
| [18] | MIMIC-IV 2023 | Correct | [27] |
| [19] | eICU-CRD 2018 | Title truncated | Full title [28] |
| [20] | Reyna et al. 2020 | Correct | [29] |
| [21] | DeepSurv 2018 | Correct | [30] |
| [22] | Dipole 2017 | Correct | [31] |
| [23] | "Y. Tatonetti et al." 2012 | Wrong initials (N. P.); not a graph model, but cited for graph-based drug-interaction models | Replaced by Zitnik et al., Decagon [34] |
| [24] | GAMENet 2019 | Correct | [35] |
| [25] | "Wang, Wang, Wei, Rare disease prediction ..., IJCAI 2022" | **Does not exist** in any index | Replaced by SHEPHERD, Alsentzer et al., npj Digit. Med. 2025 [37] |
| [26] | HeteroMed, CIKM 2018 | **Wrong co-authors** | Corrected to Hosseini, Chen, Wu, Sun, Sarrafzadeh [38] |
| [27] | McMahan et al. 2017 | Correct | [39] |
| [28] | Abadi et al. 2016 | Correct | [40] |
| [29] | FedHealth | Verified on IEEE Xplore (IEEE Intell. Syst. 35(4), 2020). The text said "medical imaging"; the paper uses wearable data. | Text corrected [41] |
| [30] | "L. Zhang et al., FLoP, MICCAI 2021" | **Wrong.** The real FLOP is by Q. Yang et al., KDD 2021, pp. 3845–3853 | Corrected [42] |
| [31] | "H. Liu et al., FedECG, IEEE JBHI 2023" | **Wrong.** The real FedECG is by Z. Ying et al., J. King Saud Univ. CIS 35(6):101568, 2023 | Corrected [43] |
| [32] | Mironov 2017 | Correct | [48] |
| [33] | Bonawitz et al. 2017 | Correct | [50] |
| [34] | "J. D. Dernoncourt" | Wrong initials (F.) | Corrected [51] |
| [35] | TRIPOD 2015 | Correct | [52] |
| [36] | Sendak et al., "JAMA Netw. Open 2020" | **Wrong title and venue.** The real paper is in JMIR Med. Inform. 8(7):e15182, 2020. | Corrected [54]; see C.3 |
| [37] | Guo et al. 2017 | Correct | [46] |
| [38] | McDermott et al. 2021 | Real, but cited for "Clinical-BERT gaps of 2.4–3.8 points", which it does not report | Fairness sentence now cites Zhang et al. [59] without the unsupported numbers; McDermott cited for cohort sensitivity [63] |
| [39] | DeepHit 2018 | Correct; pages not confirmed | [65] |
| [40] | Rajkomar et al. 2019 | Real, but cited for median aggregation and cohort sensitivity, which it does not cover | Now cited for deep learning in medicine [10]; median aggregation → Yin et al. [60]; cohort sensitivity → [5], [63], [64] |

## B. Added references (all checked)

**Published from 2023 onwards (Reviewer 1, comment 6):**

| No. | Reference |
|---|---|
| [7] | Kamran et al., NEJM AI 2024 |
| [19] | Hi-BEHRT, IEEE JBHI 2023 |
| [20] | TransformEHR, Nat. Commun. 2023 |
| [21] | NYUTron, Nature 2023 |
| [24] | GraphCare, ICLR 2024 |
| [26] | Ponomareva et al., JAIR 2023 |
| [32] | Wornow et al., npj Digit. Med. 2023 |
| [33] | Zhang et al., ICML 2023 |
| [37] | SHEPHERD, 2025 |
| [43] | FedECG, 2023 |
| [53] | TRIPOD+AI, BMJ 2024 |
| [55] | Kapoor and Narayanan, Patterns 2023 |
| [62] | Moor et al., eClinicalMedicine 2023 |
| [64] | YAIB, ICLR 2024 |

**Other additions:**

| No. | Reference |
|---|---|
| [6] | Wong et al., 2021 |
| [23] | GCT, AAAI 2020 |
| [25] | Van Calster et al., 2019 |
| [34] | Zitnik et al., 2018 |
| [44] | Focal loss, ICCV 2017 |
| [45] | Bergstra et al., 2011 |
| [47] | Vickers and Elkin, 2006 |
| [49] | Opacus, 2021 |
| [56] | XGBoost, 2016 |
| [57] | TFT, 2021 |
| [58] | Song et al., SAnD, AAAI 2018 |
| [59] | Zhang et al., CHIL 2020 |
| [60] | Yin et al., ICML 2018 |
| [61] | Yeom et al., CSF 2018 |

## C. Points the first author must confirm

1. **CT-BERT [18].** No published EHR model by this name was found. Reference [18] is GatorTron, which encodes clinical text only. Replace [18] with the correct source if CT-BERT is a different model.
2. **SANet [58].** No clinical model with this exact name was found. It is cited to SAnD, the closest match. Confirm or replace.
3. **"SepsisWatch curation" [54].** Sendak et al. describe Duke's Sepsis Watch deployment, not a MIMIC-IV cohort definition. Confirm what curation was applied.
4. **Repository URL.** Make sure `https://github.com/sathishv/heal-gtn` is public.

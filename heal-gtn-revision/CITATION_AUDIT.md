# Citation audit: HEAL-GTN revision (ID6)

## How the references were checked

- **What was checked:** every reference was checked for:
  - author list (first author and initials);
  - title;
  - venue;
  - year;
  - volume, issue and pages or article number;
  - whether the cited work actually supports the sentence it is cited for.
- **Sources:** web searches against the publisher or index record, such as PubMed/PMC, ACL Anthology, PMLR, AAAI OJS, NeurIPS proceedings, IEEE Xplore, JAIR, BMJ, Nature, and arXiv for preprints.
- **Google Scholar:** this environment's network policy blocks it, so it was not queried directly. All the records below are the same records Google Scholar indexes.
- **Unconfirmed fields:** where a page range could not be confirmed (RETAIN, Yeom et al., DeepHit, the NeurIPS papers, Kamran et al.), it was left out rather than guessed. IEEE style allows this.
- **Final numbering:** the final list has 59 references, numbered in order of first citation.

## A. Original reference list (40 entries): findings

| Old | Reference | Finding | Action (new number) |
|---|---|---|---|
| [1] | Jensen et al., Nat. Rev. Genet. 2012 | Correct | Kept [1] |
| [2] | Miotto et al., Brief. Bioinform. 2018 | Correct | Kept [2] |
| [3] | Singer et al., JAMA 2016 (Sepsis-3) | Correct | Kept [3] |
| [4] | "J. A. Kellum et al.", KDIGO AKI guideline | The guideline is authored by the KDIGO AKI Work Group | Author corrected [4] |
| [5] | Johnson, Pollard, Mark, MLHC 2017 | Correct | Kept; PMLR vol. 68 added [5] |
| [6] | Choi et al., RETAIN, NeurIPS 2016 | Correct; sources disagree on the page range (3504–3512 vs 3512–3520) | Kept; pages omitted [24] |
| [7] | Shickel et al., IEEE JBHI 2018 | Correct | Removed to save space |
| [8] | Alsentzer et al., Clinical NLP Workshop 2019 | Correct | Kept [15] |
| [9] | Luo et al., BioGPT, Brief. Bioinform. 2022 | Correct | Kept; article no. bbac409 added [26] |
| [10] | Harutyunyan et al., Sci. Data 2019 | Correct, but the paper's Table I said it benchmarked CNN-LSTM, which it did not | Kept [8]; Table I row corrected to LSTM |
| [11] | Vaswani et al., NeurIPS 2017 | Correct | Kept [48] |
| [12] | Zhou et al., Informer, AAAI 2021 | Correct | Kept [49] |
| [13] | Li et al., BEHRT, Sci. Rep. 2020 | Correct | Kept [11] |
| [14] | Rasmy et al., Med-BERT, npj Digit. Med. 2021 | Correct | Kept [12] |
| [15] | "Z. Yang et al.", cited for **CT-BERT** | **Wrong.** This is GatorTron by **X.** Yang et al. It describes no model called CT-BERT. | Now cited as GatorTron [16]; CT-BERT has no citation (see section C) |
| [16] | Choi et al., GRAM, KDD 2017 | Correct | Kept [18] |
| [17] | "C. Shang et al.", IJCAI 2019, cited for **SANet** and mortality | **Wrong first author** (J. Shang). The paper is G-BERT for medication recommendation and does not describe SANet or mortality estimation. | Removed. SANet now cited to Song et al., AAAI 2018 [10] (see section C) |
| [18] | Johnson et al., MIMIC-IV, Sci. Data 2023 | Correct | Kept [45] |
| [19] | Pollard et al., eICU-CRD, Sci. Data 2018 | Title truncated | Full title restored [46] |
| [20] | Reyna et al., Crit. Care Med. 2020 | Correct | Kept [57] |
| [21] | Katzman et al., DeepSurv, 2018 | Correct | Kept [25] |
| [22] | Ma et al., Dipole, KDD 2017 | Correct | Kept [9] |
| [23] | "Y. Tatonetti et al.", Sci. Transl. Med. 2012 | **Wrong initials** (N. P. Tatonetti). Not a graph-learning paper, but it was cited for graph-based drug-interaction models. | Removed |
| [24] | Shang et al., GAMENet, AAAI 2019 | Correct | Removed to save space |
| [25] | "Y. Wang, F. Wang, Y. Wei, Rare disease prediction ..., IJCAI 2022" | **Could not be found in any index. Appears not to exist.** | Removed; replaced by Choi et al., GCT, AAAI 2020 [19] |
| [26] | "A. Hosseini, R. Chen, A. Bhatia, A. Tellez, W. Hsu", HeteroMed, CIKM 2018 | **Wrong co-authors.** The correct list is A. Hosseini, T. Chen, W. Wu, Y. Sun, M. Sarrafzadeh. | Corrected [28] |
| [27] | McMahan et al., AISTATS 2017 | Correct | Kept [31] |
| [28] | Abadi et al., CCS 2016 | Correct | Kept [32] |
| [29] | Chen et al., FedHealth, "IEEE Intell. Syst. 35(4)" | The journal version could not be confirmed (arXiv 2019 confirmed). The work uses wearable activity data, but the paper said medical imaging. | Removed |
| [30] | "L. Zhang et al., FLoP ..., MICCAI 2021" | **Wrong.** FLOP is by Q. Yang, J. Zhang, W. Hao, G. Spell, L. Carin (KDD 2021) and has a different title. | Removed; replaced by Rieke et al., npj Digit. Med. 2020 [33] |
| [31] | "H. Liu, X. Wang, Z. Lin, FedECG ..., IEEE JBHI 2023" | **Wrong.** FedECG is by Ying et al. in J. King Saud Univ. CIS (2023); no such JBHI paper exists. | Removed; replaced by Dayan et al., Nat. Med. 2021 [34] |
| [32] | Mironov, Rényi DP, CSF 2017 | Correct | Kept [43] |
| [33] | Bonawitz et al., CCS 2017 | Correct | Kept [44] |
| [34] | "J. D. Dernoncourt et al.", JAMIA 2017 | **Wrong initials** (F. Dernoncourt) | Corrected [54] |
| [35] | Collins et al., TRIPOD, Ann. Intern. Med. 2015 | Correct | Kept [55] |
| [36] | "Sendak et al., ... evaluation of workflow and acceptance, JAMA Netw. Open 3(3), 2020" | **Wrong title and venue.** The real paper is in JMIR Med. Inform. 8(7), e15182, 2020. The "SepsisWatch curation" of MIMIC-IV it was cited for does not exist. | Removed; cohort now defined by explicit criteria |
| [37] | Guo et al., ICML 2017 | Correct | Kept [41] |
| [38] | McDermott et al., Sci. Transl. Med. 2021 | Real, but cited for "Clinical-BERT subgroup gaps of 2.4–3.8 points", which it does not report | Now cited for reproducibility [38]; fairness sentence cites Zhang et al., CHIL 2020 [53] with no invented numbers |
| [39] | Lee et al., DeepHit, AAAI 2018 | Correct; page range not confirmed | Kept; pages omitted [59] |
| [40] | Rajkomar et al., NEJM 2019 | Real, but cited for median aggregation and cohort sensitivity, which it does not support | Removed |

## B. References added in the revision, all checked

**Published from 2023 onwards (13, for Reviewer 1, comment 6):**

| New | Reference |
|---|---|
| [7] | Kamran et al., NEJM AI 2024 (volume and issue not confirmed, so omitted) |
| [13] | Li et al., Hi-BEHRT, IEEE JBHI 27(2):1106–1117, 2023 |
| [14] | Yang et al., TransformEHR, Nat. Commun. 14:7857, 2023 |
| [17] | Jiang et al., NYUTron, Nature 619:357–362, 2023 |
| [20] | Jiang et al., GraphCare, ICLR 2024 |
| [21] | Zhang et al., ICML 2023, PMLR 202:41300–41313 |
| [23] | Ponomareva et al., JAIR 77:1113–1201, 2023 |
| [27] | Wornow et al., npj Digit. Med. 6:135, 2023 |
| [35] | Kapoor and Narayanan, Patterns 4(9):100804, 2023 |
| [36] | van de Water et al., YAIB, ICLR 2024 |
| [37] | Moor et al., eClinicalMedicine 62:102124, 2023 |
| [45] | MIMIC-IV, 2023 (kept from the original list) |
| [56] | Collins et al., TRIPOD+AI, BMJ 385:e078378, 2024 |

**Other additions:**

| New | Reference |
|---|---|
| [6] | Wong et al., JAMA Intern. Med. 2021 |
| [10] | Song et al., AAAI 2018, pp. 4091–4098 |
| [16] | Yang et al., GatorTron, 2022 |
| [19] | Choi et al., GCT, AAAI 2020, pp. 606–613 |
| [22] | Van Calster et al., BMC Med. 2019 |
| [29] | Brody et al., GATv2, ICLR 2022 |
| [30] | Wang et al., HAN, WWW 2019 |
| [33] | Rieke et al., 2020 |
| [34] | Dayan et al., 2021 |
| [39] | Lin et al., focal loss, ICCV 2017 |
| [40] | Bergstra et al., NeurIPS 2011 |
| [42] | Yousefpour et al., Opacus, arXiv:2109.12298 |
| [47] | Chen and Guestrin, XGBoost, KDD 2016 |
| [50] | Lim et al., TFT, Int. J. Forecast. 2021 |
| [51] | Hamilton et al., GraphSAGE, NeurIPS 2017 |
| [52] | Yeom et al., CSF 2018 |
| [53] | Zhang et al., Hurtful Words, CHIL 2020 |
| [58] | Vickers and Elkin, Med. Decis. Making 2006 |

## C. Items the authors must confirm

1. **CT-BERT.** No published EHR model with this name could be found; the original citation pointed to GatorTron. The revised paper calls it "a clinical BERT-style transformer" with no citation.
   - Before submission, add the correct citation and a one-line description of exactly what was run.
2. **SANet.** No clinical model with this exact name was found. It is now described as "a self-attention model for clinical time series" and cited to SAnD (Song et al., AAAI 2018).
   - Confirm that this is the model that was run.
3. **Repository URL** (`https://github.com/sathishv/heal-gtn`). Make sure it is public before submission.

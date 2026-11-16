# Team contributions — Phase 2

Roles were assigned in **random order** (Python `random.shuffle` on the five names) and each role has an
equal share of: one experiment notebook, one section of the shared library notebook (`00_common.ipynb`),
one section of the IEEE report, slides in the presentation, and a personal viva guide in `docs/viva/`.

| Member | Role | Notebook(s) | Section of `00_common.ipynb` | Report sections | Slides |
|---|---|---|---|---|---|
| **Saumya** | Data & DL dataset engineer | `01_data_pipeline.ipynb` | B — panel loading, window sampler, scaling, loss masks | III (Data & DL dataset), II-D (lit: M5, intermittent demand) | 3–4 |
| **Harshvardhan** | Local state estimation (Kalman) & Phase-1 baselines | `02_baselines_kalman.ipynb` | C — Kalman filter, MLE of q/r, h-step forecast | IV-A/B (Baselines, Kalman theory), II-C (lit: state-space) | 5–6 |
| **Shivansh** | Global deep model (DeepAR) | `03_deepar_global.ipynb` | D — DeepAR network, NB likelihood, training loop, ancestral sampling | IV-C/E (DeepAR, regularisation), II-A/B (lit: RNN/LSTM, DeepAR family) | 7–8 |
| **Krishiv** | Hybrid model (KF-DeepAR) & cold start | `04_hybrid_kalman_deepar.ipynb` | E — causal Kalman → DeepAR input channels | IV-D (Hybrid), VI-G (cold start), II-C (lit: hybrids) | 9–10 |
| **Harsha** | Ablations, evaluation protocol & metrics | `05_ablation_studies.ipynb`, `06_final_evaluation.ipynb` | A — config/seeds, F — metrics | V (Setup & metrics), VI (Results, ablations), II-E (lit: scoring rules, rolling origin) | 11–13 |

Shared by all: Introduction, Discussion, Conclusion, slides 1–2 and 14–15, README.

## How to see who did what in Git
Each notebook starts with an **Owner** line, and every section of `00_common.ipynb` is labelled with its owner.
When you push this repo, commit each person's files from their own GitHub account (see the README, *Git workflow*).

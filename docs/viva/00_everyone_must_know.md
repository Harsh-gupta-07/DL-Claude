# Viva — what EVERY member must be able to explain

The viva is individual: you will be asked about your own part **and** about the project as a whole.
Read this page plus your own guide.

## The project in 5 sentences
1. We forecast 28 days of daily sales for 300 Walmart SKUs (M5, store CA_1, 3 departments) and must give **quantiles** (q10, q50, q90).
2. Phase 1 (ML): seasonal naive, LightGBM with lag features, and a Kalman filter per SKU.
3. Phase 2 (DL): DeepAR — one LSTM for all SKUs with a Negative Binomial output — written from scratch in PyTorch.
4. Hybrid: KF-DeepAR — the Kalman filter's causal prediction and its std are two extra inputs to DeepAR.
5. Everything is evaluated with the same rolling-origin backtest (3 origins × 28 days), MAE/RMSE/WAPE/pinball/coverage, 3 seeds, bootstrap tests,
   a cold-start test and 7 ablations.

## Headline numbers (memorise)
| | WAPE | pinball |
|---|---|---|
| seasonal naive | 0.757 | 0.514 |
| LightGBM | 0.655 | 0.402 |
| Kalman | 0.636 | 0.416 |
| DeepAR | 0.641 | 0.395 |
| 50/50 average | **0.633** | 0.400 |
| **KF-DeepAR** | 0.641 | **0.394** |

- DL improves pinball: −1.8% vs LightGBM, −5.1% vs Kalman. Biggest gain on **lumpy** SKUs (−9.6% vs Kalman).
- Hybrid vs DeepAR: tie overall (not significant), lower seed variance, **best for cold start**.
- Ablations: the **Negative Binomial likelihood** is the one component that clearly matters (Gaussian: +4.8% pinball, worse in 9/9 runs);
  everything else is within seed noise.
- Hybrid is significantly better than LightGBM, Kalman, seasonal naive and the 50/50 average.

## Words you must be able to define
| term | one-line definition |
|---|---|
| local model | one model per series (Kalman, seasonal naive) |
| global model | one model trained on all series (LightGBM, DeepAR) |
| rolling origin | evaluate at several forecast dates; train only on data before each |
| WAPE | Σ\|error\| / Σ actual — like MAPE but works with zeros |
| pinball loss | quantile loss max(τ(y−q), (τ−1)(y−q)); minimised by the true τ-quantile |
| coverage | share of actuals inside the 80% interval (target 0.80) |
| proper scoring rule | a score whose expected value is best when you report your true belief — can't be gamed |
| stockout | zero sales because the shelf was empty, not because nobody wanted it (censored demand) |
| teacher forcing | in training, feed the true previous value as input |
| ancestral sampling | in forecasting, sample a value, feed it back, repeat → sample paths |
| embedding | learned vector per category (SKU, department) |
| early stopping | stop training when validation loss stops improving; keep the best weights |
| ablation | remove one component, keep everything else fixed, measure the change |

## Syllabus map (where each course topic appears)
| syllabus week | where in the project |
|---|---|
| 1 Linear algebra, probability, PCA/SVD | Kalman filter (Gaussian conditioning), NB likelihood, PCA of item embeddings (nb 03) |
| 2–3 FFNN, backprop, optimisers, dropout, LR decay, generalisation | AdamW, ReduceLROnPlateau, dropout, weight decay, early stopping, train/val curves (nb 03, 05) |
| 5 RNN/LSTM/GRU, vanishing/exploding gradients | DeepAR LSTM, GRU ablation (A5), gradient clipping |
| 6 Representation learning | item/department embeddings, unknown-item embedding for cold start |
| 7–8 Attention / Transformers | discussed as alternative (TFT) — future work |
| SSMs (course objective: SSM ↔ linear attention) | Kalman filter = linear state-space model; Mamba is a learned selective SSM |

## Questions anyone can get
1. *Why is a probabilistic forecast needed for inventory?* Stock levels are set from a high quantile (service level), not from the mean.
2. *Why rolling origins and not one test split?* One period can be lucky; three origins show stability. Random splits would leak the future.
3. *How did you avoid leakage?* Causal stockout flags (tested), lags ≥ 28 in LightGBM, input y_{t−1} only, Kalman params fitted before the first
   validation window, early stopping on days before the origin, known-future covariates only.
4. *Is the hybrid "meaningful" or just a combination?* It is trained end-to-end with the Kalman channels as inputs, it beats the naive 50/50
   combination significantly, and it helps where theory says it should (cold start). We report honestly that it ties DeepAR overall.
5. *What would you do with more time?* Deep state-space model (network outputs q, r), TFT/Mamba encoder, more stores, tune the stockout threshold.
6. *How did you split the work?* Randomly assigned roles; each member owns one notebook, one section of `00_common.ipynb`, report section, slides.

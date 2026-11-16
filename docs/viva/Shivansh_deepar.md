# Viva guide — Shivansh (Global DeepAR model)

**Your files:** `notebooks/03_deepar_global.ipynb`, Section D of `notebooks/00_common.ipynb`
(`nb_log_prob`, `DeepAR`, `masked_nll`, `train_deepar`, `forecast_deepar`, `samples_to_frame`).
Report Sections IV-C and IV-E. Slides 7–8.

## 1. What you built, in 60 seconds
- One **LSTM** network (2 layers × 40 units, **27,578 parameters**) trained on **all 300 SKUs together** (global model).
- Input per day (29 numbers): previous sales / ν, "was observed", "was stockout", 14 calendar/price covariates,
  a 10-d **item embedding**, a 2-d **department embedding**.
- Output per day: parameters (μ, α) of a **Negative Binomial** distribution → full predictive distribution.
- Training: masked NLL, teacher forcing, AdamW, dropout, item-ID dropout, early stopping, LR decay, gradient clipping.
- Forecast: **200 sampled paths** per SKU (ancestral sampling) → mean + q10/q50/q90.
- Rolling-origin backtest: 3 origins × 3 seeds = 9 trained models.

## 2. Results (3 origins pooled, mean ± std over 3 seeds)
| | MAE | RMSE | WAPE | pinball | 80% cov. |
|---|---|---|---|---|---|
| DeepAR | 1.336 ± 0.010 | 2.579 ± 0.027 | 0.641 ± 0.005 | **0.395 ± 0.004** | 0.91 |
| LightGBM | 1.366 | 2.666 | 0.655 | 0.402 | 0.80 |
| Kalman | 1.325 | 2.571 | 0.636 | 0.416 | 0.87 |

DeepAR has the **best probabilistic accuracy (pinball)** of all non-hybrid models, but its point forecast (WAPE) is slightly worse than
the Kalman filter's. Its intervals over-cover (0.91 vs 0.80 target) — partly because for many intermittent SKUs q10 = 0, so every zero
counts as covered (discrete data). Training takes ~40–60 s per model on a laptop CPU; early stopping chose 15–37 epochs.

## 3. Things to explain on the board
**Autoregressive factorisation:** p(y_{o+1:o+H} | past) = Π_t p(y_t | θ(h_t)), h_t = LSTM(h_{t−1}, [y_{t−1}, x_t, e_item]).

**LSTM gates:** f (forget), i (input), o (output), candidate c̃. c_t = f ⊙ c_{t−1} + i ⊙ c̃ — the **additive** update means
∂c_t/∂c_{t−1} = f_t, so the gradient does not have to pass through repeated matrix products and tanh derivatives → no vanishing
gradient over the 84-day window when f ≈ 1. (Vanilla RNN: product of k Jacobians → vanishes/explodes.)

**Why Negative Binomial:** counts ≥ 0, many zeros, variance > mean. NB: mean μ, variance μ + αμ². α → 0 gives Poisson.
Gaussian would assign probability to negative sales (we test this in ablation A4).
We verified our hand-written log-likelihood against `torch.distributions.NegativeBinomial` (identical to 5 decimals)
and the sample mean/variance against μ and μ + αμ².

**Scale ν:** inputs divided by ν = 1 + context mean; μ = ν·softplus(·), α = softplus(·)/√ν. The √ν keeps the *relative* noise sensible
across volumes. Evidence it works: the learned item embeddings are **not** correlated with sales volume (corr ≈ 0) — volume is handled by ν,
embeddings store the demand *shape*.

**Teacher forcing vs ancestral sampling:** in training we feed the true y_{t−1} (fast, stable, parallel). In forecasting the true value
is unknown, so we feed back our own sample. Sampling whole paths keeps the dependence between days (a high day 1 makes day 2 higher),
which matters for e.g. total demand over the lead time.

**Softplus:** log(1 + e^x) — smooth, always positive, so μ and α are valid.

## 4. Regularisation (and why each one)
| technique | value | reason |
|---|---|---|
| dropout | 0.1 | between LSTM layers + before head; prevents co-adaptation |
| AdamW weight decay | 1e-4 | L2 penalty = Gaussian prior on weights (MAP); "decoupled" from Adam's adaptive step |
| item-ID dropout | 0.1 | replace item embedding by an "unknown" vector → the net also learns from shared features; enables cold start |
| early stopping | patience 5 on validation NLL | stop when the 28 days before the origin stop improving; restore best weights |
| LR decay | ×0.5 on plateau (patience 2) | large steps first, fine steps near the minimum |
| gradient clipping | norm 10 | exploding gradients in RNNs |
| random windows | 16 per SKU per epoch | data augmentation |

## 5. Likely viva questions
1. *Why not a Transformer?* With 56–84 steps and 300 short, noisy series an LSTM is enough and cheaper; DeepAR is the method named in
   the brief. A Transformer (TFT) is listed as future work. Attention is O(L²), the LSTM O(L).
2. *Why is the train NLL higher than validation NLL?* Different windows: training loss covers all 84 steps including the start of
   the window where the LSTM has little context; dropout is also active in training. It is not a bug.
3. *How do you get quantiles?* Empirical quantiles over the 200 sampled values for each day.
4. *What does early stopping use?* NLL on the 28 days *before* the origin — never the test days.
5. *Why 3 seeds?* Neural nets depend on initialisation and window sampling; std over seeds (pinball ±0.004) tells us which differences are real.
6. *What is the item embedding?* A learned lookup table (300+1 rows × 10) — like word embeddings; row 301 is the "unknown item".

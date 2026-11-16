# Viva guide — Krishiv (KF-DeepAR hybrid & cold start)

**Your files:** `notebooks/04_hybrid_kalman_deepar.ipynb`, Section E of `notebooks/00_common.ipynb`
(`load_kalman_states`, `kalman_window_features`). Report Sections IV-D and VI-E. Slides 9–10.

## 1. What you built, in 60 seconds
- **KF-DeepAR:** the DeepAR network from notebook 03 plus **two extra input channels per day**: what the Kalman filter predicts for that
  day (mean) and how unsure it is (std), both divided by the scale ν.
- Everything else (data, windows, hyper-parameters, seeds) is identical to DeepAR → any difference is caused by the hybrid design.
- Two controls: plain DeepAR (no Kalman) and a **50/50 average** of the Kalman and DeepAR forecasts (the naive ML+DL combination).
- A **cold-start experiment**: 30 SKUs hidden from training, then forecast with only 28 days of history.

## 2. The design — say this clearly
**Why combine them?** (draw this table)
| | Kalman (local) | DeepAR (global) |
|---|---|---|
| memory | whole history in 2 numbers (x̂, P) | 56-day window |
| level shift | adapts immediately | only if visible in context |
| shares patterns across SKUs | no | yes |
| new SKU | adapts from day 1, but knows nothing else | knows department patterns, not this SKU's level |

**What exactly are the channels?** For a window with split point t₀:
- context days t ≤ t₀: one-step-ahead prediction s_dow·x̂_{t−1|t−1}, std s·√(P_{t−1}+q+r)
- forecast days t > t₀: h-step forecast s_dow·x̂_{t₀|t₀}, std s·√(P_{t₀}+(t−t₀)q+r)

These are exactly what the filter could have said **at that moment** → causal, **no leakage**. The filter's parameters were fitted on data
before the first validation window.

**Why inputs and not averaging?** A fixed average gives the Kalman filter the same weight for every SKU and day. As an input, the LSTM can
*learn* when to trust it (e.g. after a level shift, for a new item) and when to override it (event days, SNAP, price cuts the filter knows nothing about).

**Where it comes from in the literature:** ES-RNN (Smyl 2020, won M4): exponential-smoothing states + global LSTM.
Deep State Space Models (Rangapuram et al. 2018): RNN outputs the parameters of a Kalman model. Ours is the simplest version: fit the
state-space model first, feed its outputs to the network.

## 3. Results (3 origins, 300 SKUs, 3 seeds)
| | WAPE | pinball | pinball seed-std |
|---|---|---|---|
| Kalman | 0.636 | 0.416 | – |
| DeepAR | 0.641 | 0.395 | 0.0036 |
| 50/50 average | **0.633** | 0.400 | 0.0015 |
| **KF-DeepAR** | 0.641 | **0.394** | **0.0011** |

Paired bootstrap over SKUs (pinball): hybrid is **significantly better** than LightGBM, Kalman, seasonal naive and the 50/50 average;
**not significantly different from DeepAR** (95% CI of the difference [−0.0024, +0.0016]).

**Cold start (30 unseen SKUs, 28 days of history, origin O3, seed 0):**
| | WAPE | pinball |
|---|---|---|
| Kalman (fitted on 28 days) | 0.615 | 0.571 |
| DeepAR (unknown-item embedding) | 0.629 | 0.556 |
| **KF-DeepAR** | **0.612** | **0.552** |

## 4. The honest summary (examiners like this)
1. Overall the hybrid **ties** DeepAR — we do not claim a significant overall improvement.
2. It **varied less across seeds** (pinball std 0.0011 vs 0.0036) — but with only 3 seeds this is indicative, not proof.
3. It beats the naive 50/50 combination on uncertainty (pinball) — so "learning how to combine" matters.
4. It is **best for cold-start SKUs**, which is one of the three problems named in the brief.
5. Why not more? Most filters chose a very smooth level (median ρ ≈ 0.0006), which for long histories is close to the average DeepAR already
   gets through ν. The Kalman information is most valuable where history is short or changes fast.

## 5. How the cold-start test works (be ready to walk through the code)
- Pick 10 random established SKUs per department (seed 7); delete their history before O3−27 (also listed/age/price features).
- Train DeepAR and KF-DeepAR **without** these SKUs (`exclude=cold`).
- Forecast them with the **unknown-item embedding** (index 300) — this row was trained by item-ID dropout (10% of the time every SKU
  is shown as "unknown").
- Re-fit the Kalman filter for these SKUs on their 28 days only (profile, ρ, σ²) and recompute their channels.
- Reference rows "(full history)" show the same 30 SKUs forecast by the normal models.

## 6. Likely viva questions
1. *Is it leakage to use Kalman states in training?* No — at every position the channel uses only data up to t−1 (context) or t₀ (forecast);
   parameters come from data before the first validation window.
2. *Why give the std and not only the mean?* So the network knows when the local estimate is reliable (e.g. a new item has a large P).
3. *Why does the 50/50 average have better WAPE?* Averaging two partly independent mean forecasts reduces error variance (Bates & Granger 1969);
   but the averaged quantiles of a Gaussian and an NB distribution are not a good distribution, so the pinball loss is worse.
4. *What would you try next?* Let the network output the Kalman noise parameters (q, r) per SKU and day — a Deep State Space Model — so the
   filter can adapt faster where needed; or use the Kalman level as the scale ν (ES-RNN style).
5. *What is the effect in the ablation?* Removing the Kalman channels (A1) changes the pinball loss by only +0.1% (worse in 5 of 9 paired runs)
   — within seed noise for the full panel. The value shows up for **cold-start** SKUs, not on average.

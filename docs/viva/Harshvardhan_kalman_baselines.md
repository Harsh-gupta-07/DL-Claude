# Viva guide — Harshvardhan (Kalman filter & Phase-1 baselines)

**Your files:** `notebooks/02_baselines_kalman.ipynb`, Section C of `notebooks/00_common.ipynb`
(`kalman_local_level`, `concentrated_loglik`, `dow_profile`, `fit_kalman_series`, `kalman_forecast`).
Report Sections IV-A/B. Slides 5–6.

## 1. What you built, in 60 seconds
- Re-ran the three Phase-1 models under the **Phase-2 protocol** (same 300 SKUs, same 3 origins, same metrics) so the
  comparison with deep learning is fair: **seasonal naive**, **LightGBM** (lag ≥ 28 features, Tweedie mean + 3 quantile models),
  and a **Kalman filter per SKU**.
- Wrote the Kalman filter from scratch in NumPy (~25 lines), estimated its noise ratio by **maximum likelihood**, ran it
  causally over the full panel and saved the states (level, variance) for Krishiv's hybrid model.
- Showed the **gap**: local filters cannot borrow seasonal information from other SKUs when history is short.

## 2. The Kalman filter — what to say on the board
Model (after dividing sales by the weekday profile s_dow):
- state: x_t = x_{t−1} + w_t,  w ~ N(0, q)   (the level does a random walk)
- observation: z_t = x_t + v_t,  v ~ N(0, r)   (daily noise)

Predict: x̂_{t|t−1} = x̂_{t−1|t−1}, P_{t|t−1} = P_{t−1|t−1} + q
Update: K = P_{t|t−1}/(P_{t|t−1} + r), x̂_{t|t} = x̂_{t|t−1} + K (z_t − x̂_{t|t−1}), P_{t|t} = (1 − K) P_{t|t−1}

**Derivation of K (1 line):** error variance after update = (1−K)² P⁻ + K² r. d/dK = 0 ⇒ K = P⁻/(P⁻ + r). This is the
minimum-variance linear estimator (Kalman 1960).

**Intuition:** K is a trust dial. Big q/r ⇒ K→1 ⇒ follow the data (fast, noisy). Small q/r ⇒ K→0 ⇒ long average (smooth, slow).
Steady state of the local level model = **simple exponential smoothing** with α = K.

**Parameter estimation:** innovations v_t = z_t − x̂_{t|t−1} are independent N(0, F_t) ⇒ log-likelihood
= −½ Σ (log 2πF_t + v_t²/F_t) (prediction-error decomposition). Write q = ρ r; for fixed ρ the best r has a closed form
(r̂ = mean of v²/F*), so only ρ needs a grid search (17 values from 1e−4 to 1). Fitted on data before the first validation window only.
Result: median ρ ≈ 0.0006 → filters are smooth (long memory) for most SKUs.

**Missing data / stockouts:** just skip the update (predict only) — the variance P grows, so the next real observation gets more weight.
This is exactly "mark, don't impute".

**Forecast h days ahead:** mean = s_dow · x̂_{o|o}; variance = s_dow² (P_{o|o} + h q + r) → intervals widen with h.

**Diagnostic:** standardised innovations should have std ≈ 1; we get a median of 1.25 → the Gaussian is a bit too thin-tailed
for count data (zeros and spikes). That's one reason the DL model uses a Negative Binomial.

## 3. Baseline results (all 3 origins, 300 SKUs, 28-day horizon)
| model | WAPE | pinball | 80% coverage |
|---|---|---|---|
| seasonal naive | 0.757 | 0.514 | 0.84 |
| LightGBM | 0.655 | 0.402 | 0.80 |
| Kalman | 0.636 | 0.416 | 0.87 |

Kalman has the best point accuracy (WAPE) of the three, LightGBM the best quantiles (pinball).
(The final numbers including the DL models are in `results/tables/06_main_results.csv`.)

**Why LightGBM uses only lags ≥ 28:** one model forecasts all 28 days; any lag < 28 would need values after the origin
for some horizons (peeking) or recursive prediction. Cost: it ignores the most recent 27 days.

## 4. The gap (your key slide)
Experiment: weekday profile from the SKU alone vs pooled over its department, using the full history or only the last 8 weeks.
- full history: pooling changes WAPE by < ±1.1% (own profile already good, shrinkage helps sparse items);
- **last 8 weeks: pooling improves every segment** (smooth +1.8%, intermittent +1.0%, lumpy +0.9%, erratic +0.3%).
⇒ With sparse/short histories the useful information is in *other* SKUs, and a local filter cannot use it. The Kalman filter also
has no inputs for events, SNAP or price. ⇒ motivation for a **global** DL model and for the hybrid.

## 5. Likely viva questions
1. *Why a local level and not a local linear trend?* Daily retail sales over 28 days have little persistent trend; the level model has
   one parameter (ρ) and is easy to estimate per SKU. A trend model often over-extrapolates.
2. *Why multiplicative weekday profile?* The weekend effect scales with the level (a 40/day item gets +10 on Saturday, a 1/day item +0.3).
3. *What is κ in the profile?* 10 "pseudo-days" of shrinkage toward a flat profile; sparse SKUs end up near flat.
4. *Kalman vs exponential smoothing?* Same point forecast in steady state, but the Kalman filter also gives a variance, handles
   missing days exactly and has a likelihood for estimating parameters.
5. *Relation to the syllabus?* The Kalman model is a linear **state-space model**; modern SSMs (S4/Mamba) are learned, input-dependent
   versions of the same recurrence x_t = A x_{t−1} + B u_t, y_t = C x_t.

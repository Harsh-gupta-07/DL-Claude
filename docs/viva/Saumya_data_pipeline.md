# Viva guide — Saumya (Data pipeline & DL dataset)

**Your files:** `notebooks/01_data_pipeline.ipynb`, Section B of `notebooks/00_common.ipynb`
(`load_panel`, `window_index`, `make_batch`, `sample_train_windows`). Report Section III. Slides 3–4.

## 1. What you built, in 60 seconds
- Took **M5** (Walmart daily unit sales) → store **CA_1**, 3 departments (FOODS_3, HOUSEHOLD_1, HOBBIES_1),
  **100 random SKUs each = 300 series**, last **1000 days** (Aug 2013 – May 2016).
- Built a **complete (SKU × day) index** and asserted no row is missing (300 × 1000 = 300,000 rows).
- **Availability:** no price in a week ⇒ item not on the shelf (e.g. not launched yet) ⇒ sales set to **NaN**, not 0.
  The store is closed on Christmas (3 days in the window) ⇒ also NaN. 52 SKUs were launched inside the window.
- **Causal stockout detector** (mark, don't impute): 8.6% of observed days flagged, 272 of 300 SKUs have ≥1 flag.
- **14 known-future covariates:** weekday one-hot (7), month sin/cos (2), SNAP, event, relative price, listed, age.
- **Syntetos–Boylan segments** (on training data only): most SKUs are *intermittent*.
- **Rolling-origin split:** 3 origins; for each: train ≤ o−28, validation (o−28, o], test (o, o+28].
- **DL windows:** 56-day context + 28-day prediction range, scaled by ν, with a loss mask.

## 2. Key ideas you must be able to explain
**Stockout rule.** Assume demand is Poisson with rate λ = mean sales of the 56 days before a zero-run starts.
P(k zeros in a row) = e^{−λk}. If this is < 0.001 (i.e. λk > ln 1000 ≈ 6.91) the day is flagged.
Example: an item selling 3/day → flagged from the 3rd zero day (3×3 = 9 > 6.91).
It is **causal**: the flag at day t uses only sales up to t. We *proved* it with a test: recomputing on a series cut at
a random day gives identical flags. Flags are used to (1) remove those days from the training loss, (2) skip the Kalman update,
(3) report metrics with and without stockout days. The sales value itself is never changed.

**Why NaN instead of 0 for "not on sale".** A 0 before launch is not "zero demand"; training on it would teach the
model that demand is lower than it is. NaN → the loss mask and the Kalman filter both ignore it.

**Why sin/cos for month.** December (12) and January (1) are neighbours; a single number 1..12 puts them far apart.
sin/cos puts the months on a circle.

**Why scale by ν = 1 + mean(context).** A HOBBIES item selling 0.2/day and a FOODS item selling 40/day must share one
network. Dividing inputs by ν makes them comparable; the network re-multiplies by ν at the output. The "+1" avoids
dividing by zero for all-zero windows.

**No look-ahead.** The input at position t is y_{t−1}; we assert in the notebook that input[t] == target[t−1].
Training windows must end at or before o−28 so they never touch validation or test.

**ADI / CV².** ADI = average number of days between non-zero sales; CV² = (std/mean)² of non-zero sizes.
Thresholds 1.32 and 0.49 (Syntetos, Boylan & Croston 2005) → smooth / erratic / intermittent / lumpy.

**Rolling origin vs random split.** Time series are ordered; a random split would put future days in training (leakage).
Rolling origins (Tashman 2000) evaluate several realistic "forecast days" and show whether results are stable over time.

## 3. Likely viva questions
1. *Why only 300 SKUs?* Course-project compute budget (CPU); stratified by department so we keep high-volume and sparse items.
2. *Could the stockout detector be wrong?* Yes — it is a heuristic; a genuinely unpopular week can be flagged. That is why we
   **mark** (and report both views) instead of imputing values.
3. *Is using future prices leakage?* No — M5 provides the planned price for future weeks; retailers know their own prices in advance.
4. *What is the loss mask?* A 0/1 tensor; the log-likelihood is multiplied by it, so masked days contribute no gradient.
5. *What is data augmentation here?* Each epoch samples new random (SKU, start day) windows → the network rarely sees the
   exact same input twice.
6. *How big is one batch?* 128 windows × 84 steps × (3 + 14) dynamic features (+2 Kalman channels in the hybrid).

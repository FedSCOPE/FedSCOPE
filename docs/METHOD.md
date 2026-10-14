# FedSCOPE method specification

## 1. Measurement preprocessing

CoSSMic channels are cumulative energy meters. For each 15-minute interval, energy is obtained by differencing consecutive cumulative readings. In the primary strict analysis, a source-marked interpolated endpoint invalidates that interval. Negative increments and non-consecutive timestamps are treated as unavailable rather than clipped or silently imputed.

The native-load balance is

`grid import + PV + storage discharge = native load + storage charge + grid export`.

The essential-load proxy is the sum of configured measured end-use channels. The uncovered essential-energy deficit is

`max(essential load - local PV - measured storage discharge, 0)`.

The phrase *essential-load proxy* is used deliberately: the source dataset does not certify those channels as safety-critical loads.

## 2. Expanding causal backtest

Every complete non-overlapping 96-step focal segment is a candidate 24-hour episode. For an episode beginning at `T`, a direct-horizon training origin `t` is eligible only if its furthest target is observed before the episode:

`t + H <= T - 15 min`.

Thus every label used for every forecast horizon is historical at decision time. Early candidate episodes are skipped only when the focal regression design is mathematically non-identifiable.

No random train/test split, random client sampling, random episode selection, or random estimator initialization is used.

## 3. Direct multi-horizon forecasting

For target `y`, client `i`, and direct horizon `h`,

`y_i(t+h) = a_i,h + x_i(t)^T beta_i,h + eps_i,h`,

where `x` contains lags at 15 min, 1 h, 1 day, and 1 week. Client intercepts remain client-specific.

### Local baseline

Independent client ordinary least squares with a Moore-Penrose solution.

### Fixed-effects pooled baseline

A common lag vector is estimated after eliminating each client intercept by within-client centering. Pooling therefore does not force a common level.

### Federated empirical-Bayes random effects

For each horizon,

`b_i,h | theta_i,h ~ N(theta_i,h, V_i,h)`

and

`theta_i,h ~ N(mu_h, tau_h^2 I)`.

`b_i,h` is the local lag estimate and `V_i,h` its estimated covariance. The server receives client sufficient statistics rather than raw rows. The between-client variance `tau_h^2` is estimated by one-dimensional REML. Posterior means therefore shrink toward the population mean only to the degree supported by local uncertainty and observed cross-client heterogeneity; no hand-set personalization coefficient is used.

## 4. Model-based forecast uncertainty

Predictive standard deviation combines local residual variation with coefficient uncertainty. The primary reporting level is 95%; 90% and 99% are prespecified sensitivity settings. These are Gaussian model-based intervals, not distribution-free conformal guarantees. Empirical coverage is reported directly.

The executed-opacity action is driven by the causal point-estimated anticipated-distress state; the interval level is retained as a forecast diagnostic and sensitivity setting.

## 5. Genuine scarcity game

A state is *game-active* exactly when all three conditions hold:

1. transferable peer supply is strictly positive;
2. at least two measured-essential-proxy claimants have positive deficits;
3. supply is strictly less than aggregate claimant deficit.

Outside these states, strategic priority is not interpreted because either nothing can be allocated or no allocation conflict exists.

## 6. Weighted proportional fairness

At a game-active state, allocation solves

`max_a sum_i s_i log(a_i)`

subject to `0 <= a_i <= d_i` and `sum_i a_i <= S`.

For positive weights, the KKT solution has the form

`a_i = min(d_i, s_i/lambda)`.

A zero signal is an actual opt-out. There is no equal-priority fallback.

## 7. Current viability and forward-looking distress

The current truthful priority is the measured viability-shortfall ratio

`v_i,t = d_i,t / L_i,t^ess`

when the essential-load proxy is positive.

The focal private anticipated state is an integrated energy fraction over the causal forecast horizon:

`D_t = sum_h max(d_hat(t+h),0) / sum_h max(L_hat_ess(t+h),0)`.

This definition avoids the saturation caused by a max-over-horizon indicator. The realized-future audit uses the same ratio computed ex post from measured future quantities and is never supplied to the controller.

## 8. Exact service-preserving floor

At each game-active time `t`, define `ell_t` as the smallest focal signal that achieves the focal allocation attainable at full priority `s=1` under the current market state. The focal allocation is monotone in its own weight under weighted proportional fairness, so the lower edge is found by deterministic one-dimensional bisection.

Any signal in the full-priority plateau preserves the same focal service. FedSCOPE therefore constrains the executed strategic action to remain at or above `ell_t`.

## 9. Causal executed-history opacity

The opacity objective is defined on the sequence that is actually released. Under receding-horizon operation, future planned signals may be revised before publication, so they are excluded from the information objective. The controller optimizes only the current message together with signals that have already been released.

At each game-active time, the controller constructs two protocol-native candidate actions:

- **service floor:** `ell_t`;
- **truth-bounded upper action:** `u_t = max(ell_t, s_truth,t)`.

The upper action never exaggerates urgency beyond the truthful current signal unless the service floor itself must exceed that signal to preserve allocation. This truth-bounded envelope prevents the trivial constant-full-priority policy without introducing a tunable coefficient.

Let the executed game-active history before the current decision be

`H_{t-1} = {(s_k, D_k)}_{k < t}`.

For each candidate `c in {ell_t, u_t}`, the controller computes the biased sample distance correlation of

`H_{t-1} union {(c, D_t)}`.

The candidate with the smaller cumulative distance correlation is published. Ties choose `ell_t`. The selected `(s_t,D_t)` pair is then appended to the executed history.

No reward weight, rolling-window length, randomization, manually selected leakage threshold, or future realized quantity is used.

The implementation maintains exact summaries of the biased sample distance-correlation statistic, so each candidate update is O(n) in the number of previously executed game-active states rather than recomputing all pairwise matrices from scratch.

## 10. Common-target leakage audit

All policies are evaluated against the **same** anticipated-distress value at the same timestamp. On game-active states the code reports:

- absolute Pearson correlation;
- absolute Spearman correlation;
- distance correlation;
- a causal prequential linear attacker.

A separate realized-future target is reported only as an ex-post audit.

## 11. Statistical reporting

Forecast-error differences use chronological Newey-West HAC intervals on common origins. Policy outcomes are also stored episode-wise for paired diagnostics. The main opacity claim is based on the pooled chronological executed-history dependence statistic, which is the quantity directly optimized by the controller. Focal rotations and horizon/data-treatment sensitivities are reported to characterize heterogeneity rather than to imply universal benefit.

## 12. Non-claims

FedSCOPE does not provide differential privacy, cryptographic secrecy, or information-theoretic independence. Sufficient-statistic federation does not itself constitute a privacy guarantee. The CoSSMic experiment also does not include a distribution-network topology, physical power flow, certified critical-load labels, or recorded outage-market outcomes.

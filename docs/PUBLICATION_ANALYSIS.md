# Publication analysis layer

The core FedSCOPE controller is evaluated with Pearson, Spearman, and distance
correlation on a common causal anticipated-distress target. Because distance
correlation is also the online executed-action selection criterion, the paper
adds post-hoc metrics that are not available to the controller.

## Independent dependence audits

`fedscope.publication_audit` reports:

- normalized HSIC using one-dimensional RBF kernels with a deterministic median
  bandwidth heuristic;
- k-nearest-neighbor mutual-information estimates at the prespecified values
  `k = 3, 5, 10, 20`.

The multiple `k` values are reported as a sensitivity analysis; no value is
selected after observing the policy outcome.

## Focal opacity headroom

For every game-active state, the exact minimum-service floor is `ell_t` and the
truth-bounded upper candidate is `u_t = max(ell_t, s_truth,t)`. The publication
audit records:

- instantaneous service-preserving headroom `u_t - ell_t`;
- executed lift `|s_t - ell_t|`;
- the resulting dCor reduction for each focal rotation.

This is used as a mechanism diagnostic for client-dependent opacity gain, not as
a population-level significance claim.

## Federated systems evidence

The publication audit also summarizes:

- REML `tau^2` distributions;
- the number of client–horizon groups in which the federated estimator is best
  among learned estimators or no worse than local OLS;
- the number of sufficient-statistic scalars sent to the server versus the
  scalar count of the corresponding raw centralized local design/target rows;
- whether the protocol audit recorded any server-side raw rows.

The scalar ratio quantifies avoided raw-row centralization. It is not a network
latency, bandwidth, or wall-clock speedup measurement.

## Reproduction

After `RUN_COMPLETE.ps1`, these analyses and the paper plots are already
present. To rerun them without forecast retraining:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

# FedSCOPE

**Causal service-preserving strategic opacity for federated peer-to-peer emergency energy sharing in networked microgrids.**

FedSCOPE studies a specific information-exposure problem in networked microgrids. During scarcity, a participant may need to publish a priority signal to obtain peer energy, but the sequence of published priorities can also reveal how vulnerable that participant expects to become. FedSCOPE combines causal federated state estimation, a proportional-fair scarcity allocation model, and an executed-history signaling policy that reduces statistical dependence between the released priority sequence and the participant's anticipated energy distress while preserving the focal allocation.

The implementation is designed for chronological, reproducible evaluation on measured OPSD CoSSMic data. Raw client time-series rows remain local to the federated estimation layer; only sufficient statistics are exchanged.

---

## Repository structure

```text
FedSCOPE/
├── .github/          # CI workflow
├── analysis/         # post-processing and figure-generation code
├── configs/          # experiment configuration
├── data/             # downloaded input data; raw files are not committed
├── docs/             # method and experiment documentation
├── evaluation/       # post-hoc publication-facing audits
├── experiments/      # sensitivity and focal-rotation runs
├── figures/          # generated publication figures
├── results_final/    # primary experiment outputs
├── scripts/          # reproducibility entry points
├── src/              # FedSCOPE package
├── tests/            # unit and scientific-invariant tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements-lock.txt
```

Generated outputs are intentionally separated from source code. For a public repository, large raw data files and transient environments such as `.venv/` and `.pytest_cache/` should remain untracked.

---

## Scientific scope

### Measured inputs

Experiments use the Open Power System Data (OPSD) CoSSMic household-data package. Cumulative meters are converted to 15-minute interval energies by endpoint differencing. The primary analysis excludes source-marked interpolated endpoints and non-consecutive meter intervals.

### Counterfactual emergency-sharing layer

The underlying load, photovoltaic, storage, and device-level measurements are real. The emergency peer-sharing market is counterfactual: CoSSMic does not label the evaluated periods as historical multi-microgrid islanding events or peer-energy trades.

### Essential-load terminology

Configured device-level channels are treated as **measured essential-load proxies**. They are not assumed to be certified safety-critical loads.

---

## Method overview

FedSCOPE is evaluated through the following sequence.

1. **Chronological expanding backtest**  
   Every complete 24-hour focal segment is evaluated once. Forecast models use only information available before the corresponding episode.

2. **Genuine scarcity detection**  
   Priority signaling is evaluated only when transferable peer supply is positive, at least two measured-proxy claimants have positive deficits, and available supply is insufficient to satisfy their aggregate deficit.

3. **Weighted proportional-fair allocation**  
   Energy allocation solves a weighted proportional-fair resource-allocation problem under supply and deficit constraints. A zero priority is an actual opt-out.

4. **Federated empirical-Bayes state estimation**  
   Client intercepts remain local. Autoregressive lag coefficients are partially pooled through a Gaussian random-effects model, with between-client variance estimated by REML. The server receives sufficient statistics rather than raw time-series rows.

5. **Integrated anticipated distress**  
   The focal private state is the forecast fraction of essential energy expected to remain locally uncovered over the anticipation horizon.

6. **Exact service floor**  
   At each game-active state, FedSCOPE computes the minimum focal priority that yields the same focal allocation as full priority.

7. **Executed-history opacity control**  
   The controller compares the service-floor action with a truth-bounded upper action and selects the one that gives lower cumulative distance correlation between the **released** priority sequence and the common causal anticipated-distress sequence.

8. **Independent audits**  
   Final signal sequences are evaluated using Pearson and Spearman correlation, distance correlation, normalized HSIC, and kNN mutual information. HSIC and mutual information are not used by the controller.

No random chronological split, client sampling, episode selection, optimizer initialization, or random hyperparameter search is used in the primary experiment.

---

## Data integrity

The downloader verifies the exact source files used in the study:

```text
household_data_15min_singleindex.csv
SHA-256: eca7dee5ac6e53373cd0209bc05ba3383cd860bfa5d5e995f74fa78db8857c58

time_series_60min_singleindex.csv
SHA-256: 56b96e807e8e4e5b86d8ebe8097c1f9a21c289b145df5bd8bddc655f568a8257
```

A file with the expected name but a different hash aborts the run rather than silently changing the experiment.

---

# Reproducing the experiments on Windows

## Requirements

- Windows 10 or 11
- 64-bit Python 3.11
- internet access for the first data download
- enough disk space and memory for the expanding-window sensitivity suite

Extract the repository, for example to:

```text
D:\Projects\FedSCOPE
```

Open PowerShell and run:

```powershell
cd D:\Projects\FedSCOPE
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

This is the recommended end-to-end reproduction command. The execution procedure itself is unchanged: the script prepares the environment, installs the locked dependencies, verifies the source data, runs the primary experiment, executes the prespecified sensitivity suite, computes the publication-facing audits, and generates the result figures.

The full run is computationally expensive because forecasting models are refitted chronologically over an expanding history and the complete sensitivity suite is evaluated.

---

## Primary experiment only

For a faster check of the main experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PRIMARY.ps1
```

Primary outputs are written to:

```text
results_final/
```

The main files include:

```text
results_final/
├── data_audit.csv
├── episodes.csv
├── game_activity.csv
├── federated_protocol.csv
├── eb_sharing.csv
├── forecast_metrics.csv
├── forecast_effects_hac.csv
├── policy_step_results.csv
├── episode_metrics.csv
├── leakage_global.csv
├── attacker_prequential.csv
├── policy_effects.csv
├── primary_summary.csv
└── run_manifest.json
```

---

## Sensitivity experiments

After the primary environment and data are available:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PAPER_SUITE.ps1
```

Sensitivity experiments cover the anticipation horizons, reporting levels, interpolation treatment, and focal-client rotations used in the study. Their archived outputs are organized under:

```text
experiments/
```

---

## Evaluation and figure generation

The publication-facing post-hoc analyses can be reproduced without retraining the forecasting models:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

The resulting analyses and plots are organized under:

```text
evaluation/
figures/
```

The evaluation layer includes:

- normalized HSIC with deterministic median-heuristic RBF kernels;
- kNN mutual-information estimates for `k = 3, 5, 10, 20`;
- focal service-preserving signaling headroom;
- client-horizon federated win counts;
- REML between-client heterogeneity summaries;
- sufficient-statistic payload diagnostics.

These analyses are post-hoc and do not alter the controller outputs.

---

## Main empirical result

In the primary experiment, the expanding backtest contains:

- **451** complete 24-hour episodes;
- **1,639** genuine game-active 15-minute states.

Minimum-service signaling and FedSCOPE produce the same focal allocation:

```text
served energy: 14.149587 kWh
```

For the common anticipated-distress target, distance correlation changes from:

```text
0.107813  ->  0.047258
```

corresponding to a **56.2% reduction** with no focal service loss.

The same policy is also evaluated with dependence measures that are not part of the controller objective, including normalized HSIC and kNN mutual information. Horizon and focal-client sensitivity analyses are reported separately because the achievable opacity reduction is not uniform across clients.

---

## Tests

After installation, tests can be run directly with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers interval construction, proportional-fair allocation, integrated distress, empirical-Bayes estimation, federated protocol boundaries, dependence metrics, and executed-history opacity.

---

## Interpretation and limitations

The implementation supports empirical claims about **service-preserving reduction of measured statistical dependence** between a public priority signal and a causal anticipated-distress state under the stated allocation mechanism.

It does **not** establish:

- differential privacy;
- cryptographic secrecy;
- information-theoretic independence;
- physical power-flow feasibility for a distribution network not represented in CoSSMic;
- certified critical-load operation;
- universal benefit for every focal client;
- observed historical emergency-market outcomes.

The federated sufficient-statistic protocol also does not by itself constitute a formal privacy mechanism.

---

## Documentation

Additional details are available in:

```text
docs/
├── METHOD.md
├── EXPERIMENT_PROTOCOL.md
├── PUBLICATION_ANALYSIS.md
├── SCIENTIFIC_SCOPE.md
└── REFERENCES.md
```

`METHOD.md` describes the mathematical formulation and implementation. `EXPERIMENT_PROTOCOL.md` records the chronological evaluation design. `PUBLICATION_ANALYSIS.md` documents the result-facing audits, and `SCIENTIFIC_SCOPE.md` defines the intended claims and non-claims.

---

## Citation

Please cite the accompanying FedSCOPE manuscript for the method and OPSD/CoSSMic separately for the data source. Full references are listed in `docs/REFERENCES.md`.

---

## License

MIT. See `LICENSE`.

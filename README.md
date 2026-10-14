# FedSCOPE

**Causal service-preserving strategic opacity for federated peer-to-peer emergency energy sharing in networked microgrids.**

FedSCOPE is a research codebase for studying public-signal information exposure during scarce peer-to-peer energy sharing. The framework combines chronological federated state estimation, weighted proportional-fair allocation, and a service-preserving signaling policy that reduces statistical dependence between a released priority sequence and a participant's anticipated energy-distress state.

The implementation accompanies the FedSCOPE manuscript and is intended to support end-to-end reproduction of the reported experiments on OPSD CoSSMic data.

## Overview

During a scarcity event, several participants may compete for a limited amount of transferable peer energy. Priority signals are operationally useful because they determine allocation, but their temporal pattern may also reveal private information about a participant's expected energy condition.

FedSCOPE addresses this disclosure channel without reducing the focal participant's allocation. At each game-active state, the method:

1. estimates the focal participant's future energy condition from a causal federated model;
2. computes an integrated anticipated-distress state over the forecast horizon;
3. identifies the minimum priority that preserves the focal allocation obtained under full priority;
4. restricts the signaling decision to service-equivalent current actions; and
5. selects the action that reduces dependence between the **executed** priority history and the corresponding anticipated-distress history.

The controller operates only on information available at the time of execution. Future realized energy states are used only for ex-post evaluation.

## Main components

- **Chronological expanding-window evaluation**  
  Forecast models are fitted using only observations available before each evaluation episode.

- **Genuine scarcity detection**  
  Signaling is evaluated only when peer supply is positive, at least two claimants have positive deficits, and total demand exceeds available supply.

- **Weighted proportional-fair allocation**  
  Priority signals determine allocation under explicit supply and deficit constraints.

- **Federated empirical-Bayes forecasting**  
  Client-specific models exchange sufficient statistics rather than raw time-series rows. Cross-client shrinkage is controlled by REML-estimated between-client variance.

- **Integrated anticipated distress**  
  Private state is defined as the forecast fraction of essential energy expected to remain locally uncovered over the anticipation horizon.

- **Exact service floor**  
  The minimum focal priority preserving the full-priority allocation is computed at every game-active state.

- **Executed-history opacity control**  
  The signaling policy acts on the public sequence that has actually been released, rather than on a future action plan that may later be revised.

- **Independent dependence audits**  
  Final signal sequences are evaluated with Pearson correlation, Spearman correlation, distance correlation, normalized HSIC, and kNN mutual information.

## Repository layout

```text
FedSCOPE/
├── .github/          GitHub Actions workflow
├── analysis/         Post-processing and publication analysis
├── configs/          Experiment configuration
├── docs/             Method and experiment documentation
├── scripts/          Reproducibility entry points
├── src/              FedSCOPE Python package
├── tests/            Unit and scientific-invariant tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements-lock.txt
```

Raw data, virtual environments, cached files, and generated experiment outputs are intentionally not tracked by Git.

## Requirements

- Windows 10 or 11
- 64-bit Python 3.11
- PowerShell
- Internet access for the first OPSD data download
- Sufficient local disk space and memory for the full expanding-window sensitivity suite

The repository uses a dependency lock file for reproducible reruns.

## Reproduce the full study

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/FedSCOPE/FedSCOPE.git
cd FedSCOPE
```

Run the complete experiment suite:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

This is the recommended end-to-end reproduction command. It:

1. creates a local Python environment;
2. installs the locked dependencies and the package;
3. runs the test suite;
4. downloads and verifies the OPSD source data;
5. executes the primary expanding-window experiment;
6. runs the scientific consistency checks;
7. evaluates the prespecified sensitivity experiments;
8. computes the post-hoc dependence and federated diagnostics; and
9. regenerates the result figures from the stored outputs.

The full run is computationally expensive because the forecasting models are repeatedly refitted over an expanding chronological history and the sensitivity experiments are evaluated independently.

## Partial runs

### Primary experiment

For a faster check of the main experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PRIMARY.ps1
```

### Sensitivity experiments

After the primary environment and source data are available:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PAPER_SUITE.ps1
```

### Post-hoc analysis

After a completed experiment run, the dependence audits and result figures can be regenerated without retraining the forecasting models:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

## Data

Experiments use the **CoSSMic household-data package** distributed by Open Power System Data (OPSD).

The pipeline converts cumulative meter readings to 15-minute interval energies by endpoint differencing. The primary analysis excludes source-marked interpolated endpoints, non-consecutive meter intervals, and invalid cumulative increments.

The downloader verifies the exact source files used in the study:

```text
household_data_15min_singleindex.csv
SHA-256: eca7dee5ac6e53373cd0209bc05ba3383cd860bfa5d5e995f74fa78db8857c58

time_series_60min_singleindex.csv
SHA-256: 56b96e807e8e4e5b86d8ebe8097c1f9a21c289b145df5bd8bddc655f568a8257
```

A file with the expected name but a different hash causes the run to stop instead of silently changing the experiment.

## Reproducibility design

The primary pipeline is deterministic with respect to the input data and configuration:

- no random chronological split;
- no random episode selection;
- no random client sampling;
- no random optimizer initialization;
- no outcome-driven hyperparameter search.

All signaling policies are evaluated at the same game-active timestamps and against the same anticipated-distress sequence.

The code also checks structural invariants including service preservation, federated raw-row boundaries, valid game-active states, and non-degenerate distress targets.

## Reported reference experiment

The primary experiment reported in the accompanying manuscript contains:

- 451 complete 24-hour episodes;
- 1,639 genuine game-active 15-minute states;
- identical focal service under minimum-service signaling and FedSCOPE;
- anticipated-state distance correlation reduced from `0.107813` to `0.047258` (56.2%);
- independent reductions in normalized HSIC and kNN mutual-information estimates.

These values describe the primary focal participant. Focal-rotation experiments show that the available opacity reduction is client-dependent because the width of the service-preserving signaling region varies across participants.

## Tests

After installation, the test suite can be run with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Tests cover:

- interval construction and data preprocessing;
- proportional-fair allocation;
- service-floor computation;
- integrated distress;
- empirical-Bayes estimation;
- federated protocol boundaries;
- dependence metrics; and
- executed-history opacity.

## Scientific scope

FedSCOPE supports empirical study of **service-preserving reduction of measured statistical dependence** between a public priority signal and a causal anticipated-distress state under the stated allocation model.

The current implementation does not claim:

- differential privacy;
- cryptographic secrecy;
- information-theoretic independence;
- physical power-flow feasibility for a distribution network not represented in CoSSMic;
- certified critical-load operation;
- universal improvement for every focal participant; or
- reconstruction of historical emergency-market events.

The emergency peer-sharing setting is counterfactual: the underlying load, photovoltaic, storage, and device-level measurements are real, but CoSSMic does not label the evaluated intervals as historical multi-microgrid islanding events.

## Documentation

Additional implementation and experiment details are available in `docs/`:

- `METHOD.md` — mathematical formulation and algorithmic details
- `EXPERIMENT_PROTOCOL.md` — chronological evaluation protocol
- `PUBLICATION_ANALYSIS.md` — post-hoc dependence and robustness analyses
- `SCIENTIFIC_SCOPE.md` — intended claims and limitations
- `REFERENCES.md` — references used by the accompanying study

## Citation

If you use this repository, please cite the accompanying FedSCOPE manuscript and the OPSD/CoSSMic data source. Full bibliographic information is provided in `docs/REFERENCES.md`.

## License

MIT. See `LICENSE`.

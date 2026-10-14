# FedSCOPE

**Causal service-preserving strategic opacity for federated peer-to-peer emergency energy sharing in networked microgrids.**

FedSCOPE is a reproducible research implementation for federated peer-to-peer energy-sharing experiments under scarcity. The repository provides the complete computational pipeline used to prepare the data, estimate federated forecasting models, execute the allocation and signaling mechanisms, evaluate the resulting policies, and reproduce the reported experimental analyses.

The codebase is organized as a Python package with configuration-driven experiments, automated tests, analysis utilities, and PowerShell entry points for end-to-end reproduction.

## Repository structure

```text
FedSCOPE/
├── .github/              GitHub Actions workflow
├── analysis/             Result analysis and figure-generation utilities
├── configs/              Experiment configuration files
├── scripts/              Experiment execution entry points
├── src/                  FedSCOPE Python package
├── tests/                Automated tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements-lock.txt
```

## Requirements

- Windows 10 or 11
- Python 3.11 (64-bit)
- PowerShell
- Internet access for the initial data download

The Python environment used for the experiments is specified in `requirements-lock.txt`.

## Quick start

Clone the repository:

```powershell
git clone https://github.com/FedSCOPE/FedSCOPE.git
cd FedSCOPE
```

Run the complete pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

`RUN_COMPLETE.ps1` is the main reproduction entry point. It prepares the Python environment, installs the required dependencies, runs the automated tests, prepares the dataset, executes the primary and sensitivity experiments, performs the analysis, and generates the associated figures.

The complete experiment suite can take substantial time because the forecasting models are evaluated chronologically using expanding training histories.

## Execution

The individual stages of the pipeline can also be run separately.

### Primary experiment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PRIMARY.ps1
```

Runs the main FedSCOPE experiment.

### Experiment suite

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PAPER_SUITE.ps1
```

Runs the additional experiment configurations used for sensitivity and focal-client analyses.

### Analysis

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
```

Runs the post-processing and evaluation procedures on completed experiment results.

### Figure generation

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

Generates the figures from the experiment and analysis outputs.

## Configuration

Experiment settings are defined in `configs/`.

The configuration used for the main experiment is:

```text
configs/paper.yaml
```

The configuration layer controls the experimental setup independently of the source implementation, allowing experiments to be modified without changing the core package.

## Source package

The main implementation is located in:

```text
src/fedscope/
```

The package contains the components required by the experiment pipeline, including:

- data preparation and episode construction;
- chronological forecasting;
- federated empirical-Bayes estimation;
- scarcity-state construction;
- proportional-fair energy allocation;
- priority-signal computation;
- service-preserving signaling;
- dependence and performance evaluation;
- experiment orchestration.

The scripts under `scripts/` provide the top-level execution interface to these components.

## Analysis

Post-processing and result-analysis utilities are located in:

```text
analysis/
```

These utilities operate on completed experiment runs to compute summary statistics, dependence diagnostics, robustness analyses, and publication figures.

## Data

The experiments use the CoSSMic dataset distributed by Open Power System Data (OPSD).

The required source data are retrieved during the experiment setup. The pipeline then performs the preprocessing required by the FedSCOPE experiments before model estimation and evaluation.

## Testing

After the environment has been created, run the complete test suite with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The tests cover the main computational components of the pipeline, including data processing, forecasting, allocation, federated estimation, signaling, dependence metrics, and experiment-level invariants.

## Continuous integration

The GitHub Actions workflow is defined in `.github/`.

It installs the package and runs the automated test suite for repository updates.

## Citation

If you use FedSCOPE in academic work, please cite the associated manuscript and the OPSD CoSSMic data source.

## License

FedSCOPE is released under the MIT License. See `LICENSE`.

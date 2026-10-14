# FedSCOPE

**Causal service-preserving strategic opacity for federated peer-to-peer emergency energy sharing in networked microgrids.**

This repository contains the implementation used for the FedSCOPE experiments. It provides the source package, experiment configuration, execution scripts, analysis utilities, tests, and supporting documentation required to reproduce the computational results reported in the accompanying manuscript.

## Repository structure

```text
FedSCOPE/
├── .github/          GitHub Actions workflow
├── analysis/         Post-processing and result-analysis utilities
├── configs/          Experiment configuration files
├── docs/             Method, protocol, and analysis documentation
├── scripts/          Windows entry points for running experiments
├── src/              FedSCOPE Python package
├── tests/            Unit and scientific-invariant tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements-lock.txt
```

Generated data, caches, virtual environments, and experiment outputs are not tracked by Git.

## Requirements

- Windows 10 or 11
- 64-bit Python 3.11
- PowerShell
- Internet access for the initial data download

Dependencies are pinned in `requirements-lock.txt`.

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

The full pipeline prepares the environment, installs the package and dependencies, runs the test suite, retrieves the required data, executes the main experiment and sensitivity analyses, runs the post-processing checks, and generates the associated figures.

The complete reproduction can take substantial time because the forecasting stage is evaluated chronologically over expanding training histories.

## Execution scripts

The main entry points are located in `scripts/`.

### Complete reproduction

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

Runs the full workflow from environment setup through final analysis.

### Primary experiment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PRIMARY.ps1
```

Runs the main experiment only.

### Sensitivity suite

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PAPER_SUITE.ps1
```

Runs the prespecified sensitivity and focal-rotation experiments used in the manuscript.

### Post-processing and figures

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

Recomputes the publication-facing diagnostics and figures from previously generated experiment outputs.

## Configuration

Experiment settings are stored under:

```text
configs/
```

The paper configuration is:

```text
configs/paper.yaml
```

Configuration files control the experiment setup without requiring changes to the source package.

## Source code

The Python implementation is located in:

```text
src/
```

The package contains the components used by the execution scripts, including data preparation, forecasting, allocation, signaling, evaluation, and experiment orchestration.

For the mathematical definition of the method and the mapping between the implementation and the manuscript, see:

```text
docs/METHOD.md
docs/EXPERIMENT_PROTOCOL.md
```

## Analysis

Post-processing code is kept separate from the core experiment implementation:

```text
analysis/
```

These utilities operate on generated experiment outputs and are used for result aggregation, diagnostic analysis, and figure generation.

## Data

The experiments use the CoSSMic data distributed by Open Power System Data (OPSD).

Raw source data are not committed to this repository. The run scripts retrieve the required files locally when needed, so a fresh clone does not require manually downloading the dataset in advance.

Data preparation is performed by the pipeline before the experiment code is executed.

## Tests

After the environment has been created, run:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers the main computational components and key experiment invariants.

Tests are also executed automatically by the GitHub Actions workflow under `.github/`.

## Documentation

Detailed documentation is available in `docs/`:

- `METHOD.md` — mathematical formulation and implementation details
- `EXPERIMENT_PROTOCOL.md` — experimental procedure and chronological evaluation design
- `PUBLICATION_ANALYSIS.md` — post-processing and reported diagnostics
- `SCIENTIFIC_SCOPE.md` — interpretation and scope of the computational study
- `REFERENCES.md` — references used by the project

## Reproducibility

The repository is structured so that a fresh clone can reproduce the experiment pipeline from the provided code, configuration, dependency lock file, and run scripts.

Generated outputs are intentionally excluded from version control. They are recreated locally by the execution scripts.

## Citation

If you use this repository, please cite the accompanying FedSCOPE manuscript and the OPSD/CoSSMic data source. See `docs/REFERENCES.md` for bibliographic details.

## License

MIT. See `LICENSE`.

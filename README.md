# Project 2: TCAD ML Surrogate Model

This project builds a surrogate regression model from TCAD simulation runs to accelerate design-space exploration.

## What it does
- Loads Project 1 simulation runs from `csv`/`parquet` files.
- Trains 3 regressors:
  - Linear Regression (baseline)
  - Random Forest
  - MLP (small neural network via scikit-learn)
- Evaluates with train/test split and metrics:
  - RMSE
  - MAE
  - R2
- Generates trust-focused diagnostics:
  - Predicted vs true plots
  - Error distribution
  - Feature importance (permutation)
  - Error vs parameter range (binned MAE)
  - Runtime comparison (simulation vs ML inference)

## Expected data shape
Each row should be one simulation run with:
- Input parameter columns (numeric): e.g. `voltage`, `temperature`, `gate_length_nm`
- One target column: e.g. `drain_current`
- Optional simulation runtime column: `simulation_time_s`

You can point to either:
- A single `.csv` / `.parquet` file
- A directory containing multiple `.csv` / `.parquet` files (they are concatenated)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run with real Project 1 data
```bash
python surrogate_pipeline.py \
  --data-path /path/to/project1_outputs \
  --target-col drain_current \
  --sim-time-col simulation_time_s \
  --output-dir outputs
```

## Run in demo mode (no data required)
```bash
python surrogate_pipeline.py --demo --target-col drain_current --output-dir outputs_demo
```

## Key outputs
All artifacts are written under the folder passed to `--output-dir` (for example: `outputs` or `outputs_demo`):
- `<output-dir>/metrics.csv`
- `<output-dir>/runtime_comparison.csv`
- `<output-dir>/feature_importance.csv`
- `<output-dir>/breakdown_mae_by_feature.csv`
- Plots in `<output-dir>/plots/`
- `<output-dir>/summary.md`

## Notes for engineering credibility
Use your real simulation dataset and interpret:
- Which parameters drive model behavior (importance)
- Where errors spike (range-based breakdown)
- Whether speedup is enough to justify surrogate usage
- Which regions still require full TCAD simulation

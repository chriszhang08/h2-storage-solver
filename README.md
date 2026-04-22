# MADS Final Thesis: Hydrogen Storage Trading with RL

This repository contains a research codebase for **hydrogen storage trading and valuation** under uncertain commodity prices. It combines:

- **Curve modeling** for natural gas and power forwards.
- **Hydrogen cost index construction** (e.g., blue/green pathways).
- A **Gymnasium trading environment** for storage operations.
- **Maskable PPO** training, hyperparameter search, and evaluation workflows.

---

## Repository Structure

- `trader/` – RL environment, agent training/evaluation entrypoint, reward logic, callbacks, and utilities.
- `curve_factory/` – Forward-curve regressors, ARMA/GARCH components, basis construction, and ETL helpers.
- `analysis/` – Visualizers and analysis helpers for curves, rewards, and episode logs.
- `configs/` – YAML configs for training, simulation, and Optuna hyperparameter search.
- `configs/storage_types/` – Per-technology constant overlays (`lh2`, `lnh3`, `meoh`, `lohc`, `baseline`).
- `tests/` – Unit tests for storage valuation, hydrogen calculations, and regressors.

---

## Config Loading and the `--type` Flag

All entrypoints share a two-layer config system implemented in `trader/utils/config_loader.py`:

1. **Base config** (`configs/{mode}.yaml`) — sets data paths, agent hyperparameters, logging settings, and simulation parameters.
2. **Storage-type overlay** (`configs/storage_types/{type}.yaml`) — overrides the `constants:` section with technology-specific physical and cost parameters (injection/withdrawal costs, boil-off rate, discount rate, etc.).

The `--type` flag selects which overlay to apply. Valid types are:

| `--type`   | Technology             | File                                  |
| ---------- | ---------------------- | ------------------------------------- |
| `lh2`      | Liquid hydrogen        | `configs/storage_types/lh2.yaml`      |
| `lnh3`     | Liquid ammonia         | `configs/storage_types/lnh3.yaml`     |
| `meoh`     | Methanol               | `configs/storage_types/meoh.yaml`     |
| `lohc`     | Liquid organic carrier | `configs/storage_types/lohc.yaml`     |
| `baseline` | Baseline (default)     | `configs/storage_types/baseline.yaml` |

When a matching Optuna study exists at `optuna_studies/{type}/optuna.db`, the loader automatically injects the best-trial hyperparameters into `cfg["agent"]` (pass `load_optuna=False` to suppress this).

Constants are patched onto the `constants` module via `apply_config_overrides(cfg)` so that all downstream modules (`trader.action`, `trader.state`, `trader.utils.reward_calc_utils`) see the correct values at import time.

---

## Quick Start

### 1) Create environment

**Conda (recommended for reproducibility):**

```bash
conda env create -f environment.yml
conda activate mads-final-thesis
```

**pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training (`trader.train`)

### Local training

```bash
python -m trader.train --type lh2 --mode train
```

The `--type` flag loads `configs/train.yaml` as the base and merges `configs/storage_types/lh2.yaml` constants on top. If an Optuna study for `lh2` exists, its best hyperparameters are injected automatically.

```bash
# Specify output directory explicitly
python -m trader.train --type meoh --mode train --output-dir ./runs/meoh_test
```

### Hyperparameter search (local single trial)

```bash
python -m trader.train --type lh2 --mode hparam --trial-id 0
```

Uses `configs/hparam.yaml` as the base config, with the `lh2` constants overlay.

### Supported modes

| Mode       | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| `train`    | Single PPO training session; saves model + metadata to disk.    |
| `hparam`   | Optuna hyperparameter search trial (supports Slurm job arrays). |
| `evaluate` | Post-training evaluation against a saved policy checkpoint.     |

```bash
python -m trader.train --help
```

---

## Simulation (`main`)

`main.py` loads one or more pre-trained PPO models and evaluates them against generated H2 price scenarios. Models are **not** retrained; weights are loaded from disk.

### Local simulation

```bash
python -m main
```

By default this reads `configs/simulate.yaml`, which specifies which `.zip` model files to load under `data.ppo_models`. The `--type` flag is not used by `main`; storage-type constants are instead determined per-model from the `ppo_models` map in the config (each entry carries its own `storage_type`).

```bash
# Override seed start
python -m main --seed-start 100
```

### Configuring models in `configs/simulate.yaml`

Two formats are supported:

**Flat** (model name is the storage type — recommended):

```yaml
data:
  ppo_models:
    meoh: models/ppo_meoh.zip
    lh2: models/ppo_lh2.zip
```

**Explicit** (model name differs from storage type):

```yaml
data:
  ppo_models:
    meoh_v2:
      path: models/ppo_meoh_v2.zip
      storage_type: meoh
```

Each model runs in its own worker process. `_worker_init()` applies the correct storage-type constants for that model by calling `load_config("simulate", storage_type=...)` and `apply_config_overrides(cfg)` before loading the `.zip`.

### Simulation output

Per-run metrics are written as Parquet files to `logging.parquet_dir`. Ingest them into the long-lived DuckDB with:

```bash
python scripts/aggregate_results.py
```

---

## Configuration Reference

### `configs/train.yaml`

| Section         | Key fields                                                                 |
| --------------- | -------------------------------------------------------------------------- |
| `data`          | `ttf_model` — path to TTF forward-curve regressor                          |
| `agent`         | PPO hyperparameters (`n_steps`, `batch_size`, `gamma`, `learning_rate`, …) |
| `normalization` | `norm_obs`, `norm_reward`, `clip_obs`, `clip_reward`                       |
| `logging`       | `tensorboard`, `scratch_dir`                                               |
| `eval`          | `episodes` — episode indices to visualise after training                   |

### `configs/simulate.yaml`

| Section    | Key fields                                      |
| ---------- | ----------------------------------------------- |
| `data`     | `ttf_model`, `carbon_price`, `ppo_models`       |
| `simulate` | `num_seeds`, `seed_start`, `t_sim_years`        |
| `logging`  | `verbosity` (0/2), `scratch_dir`, `parquet_dir` |

### `configs/storage_types/{type}.yaml`

Each file contains a `constants:` block that overrides values in `constants.py`:

| Constant                   | Description                               |
| -------------------------- | ----------------------------------------- |
| `INJECTION_COST_PER_UNIT`  | Levelized cost per kg H2 injected ($/kg)  |
| `WITHDRAWAL_COST_PER_UNIT` | Levelized cost per kg H2 withdrawn ($/kg) |
| `BOIL_OFF`                 | Daily inventory loss rate                 |
| `RISK_FREE_INT`            | Risk-free interest rate for discounting   |
| `INJ_ALPHA`                | Injection reward weighting factor         |
| `PV_LCOE`                  | PV levelised cost of electricity ($/MWh)  |
| `GREEN_H2_CONTRIB`         | Green H2 weight in blue-green LCOH index  |
| `BLUE_H2_CONTRIB`          | Blue H2 weight in blue-green LCOH index   |
| `CCUS_COST_PER_TON`        | Carbon capture and storage cost ($/tCO2)  |

---

## Testing

```bash
pytest
```

Individual tests live in `tests/`.

---

## Outputs and Artifacts

Typical outputs include:

- Trained PPO checkpoints (`ppo_<session_id>.zip`)
- Run metadata JSON (`ppo_<session_id>_meta.json`)
- TensorBoard logs
- Debug episode logs and evaluation plots
- Per-task Parquet result files (simulation) → ingested into DuckDB via `scripts/aggregate_results.py`

For cluster jobs, write long-lived artifacts to your configured scratch directory.

---

## Notes

- `main.py` contains the simulation workflow for multi-model, multi-seed evaluation.
- Physical parameter notes and citations:
  - `pv_e_green_h2_params.md`
  - `smr_blue_h2_params.md`

---

## License

No explicit license file is currently included. Add one before external redistribution.

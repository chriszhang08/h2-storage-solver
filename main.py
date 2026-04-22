"""
main
====
Entry-point for simulating trained PPO agents against generated H2 price scenarios.

Unlike ``trader.train``, this module loads pre-trained models from disk and
evaluates them over multiple randomly generated price paths.  It does **not**
train or update the model weights.

Multiple models are evaluated against each price path in parallel: one
persistent ``multiprocessing.Pool`` worker is created per model at startup,
loading the model from disk once.  Per seed, all workers receive the same
``h2_np_matrix`` and run their episodes concurrently.  The LP-optimal baseline
is computed once per seed in the main process and shared across all model rows.

Per-run metrics are written as Parquet files to ``logging.parquet_dir``.  Use
``scripts/aggregate_results.py`` to ingest these files into the long-lived
DuckDB and delete the source Parquet files.

Usage examples
--------------
**Local simulation**::

    python -m main --type lh2

**Override seed start**::

    python -m main --type lh2 --seed-start 100

**Great Lakes Slurm job array**::

    sbatch scripts/simulate.sh          # SLURM_ARRAY_TASK_ID drives seed_start
"""

import argparse
import importlib
import json
import multiprocessing
import os
import random
import uuid

import numpy as np
import pandas as pd
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from trader.utils.config_loader import load_config
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import constants
from curve_factory import CurveRegressorFactory, HydrogenLCOHCalculator
from trader.utils.reward_calc_utils import storage_valuation
from trader.utils.config_loader import apply_config_overrides

# ─── Data loading ────────────────────────────────────────────────────────────

def load_ttf_regressor(cfg: dict) -> CurveRegressorFactory:
    """Load and fit the TTF forward-curve regressor from the path in config."""
    data_cfg = cfg.get("data", {})
    ttf_path = data_cfg.get("ttf_model", "models/ttf_regressor_v5.joblib")
    ttf_regressor = CurveRegressorFactory.load(ttf_path)
    ttf_regressor.fit_arma_garch_regressors()
    return ttf_regressor


def _resolve_model_paths(data_cfg: dict) -> dict:
    """Return a {name: {path, storage_type}} dict from config.

    Supports two YAML formats:

    **Flat** (model name IS the storage type -- recommended)::

        ppo_models:
          meoh: models/ppo_meoh.zip
          lh2:  models/ppo_lh2.zip

    **Explicit** (model name differs from storage type)::

        ppo_models:
          meoh_v2:
            path: models/ppo_meoh_v2.zip
            storage_type: meoh

    Falls back to data.ppo_model using data.storage_type as the type.
    """
    ppo_models = data_cfg.get("ppo_models")
    if ppo_models:
        resolved = {}
        for name, val in ppo_models.items():
            if isinstance(val, dict):
                resolved[name] = {"path": val["path"], "storage_type": val["storage_type"]}
            else:
                # Flat format: model name is the storage type.
                resolved[name] = {"path": val, "storage_type": name}
        return resolved
    single       = data_cfg.get("ppo_model", "models/ppo_final.zip")
    storage_type = data_cfg.get("storage_type", "baseline")
    return {"default": {"path": single, "storage_type": storage_type}}


# ─── Multiprocessing worker ───────────────────────────────────────────────────
# Functions must be at module level for pickle compatibility.
# Each worker process loads exactly one model via _worker_init(), then handles
# one seed at a time via _worker_run_episode().
#
# On Linux (Slurm), multiprocessing defaults to fork — child processes inherit
# the parent's already-patched constants module automatically.

_worker_model: MaskablePPO = None  # type: ignore[assignment]
_worker_model_name: str   = ""
_worker_model_path: str   = ""
_worker_storage_type: str = ""
_worker_constants: dict   = {}


def _worker_init(model_name: str, model_path: str, storage_type: str) -> None:
    """Apply storage-type constants and load the model once per worker process.

    Steps
    -----
    1. Load the storage-type config to obtain the correct physical constants.
    2. Patch the constants module in this worker process.
    3. Reload every module that already bound constants via from constants import X
       -- patching the module alone does not update those in-place bindings.
    4. Load the PPO model from disk (after reloads so model internals see patched values).
    """
    global _worker_model, _worker_model_name, _worker_model_path
    global _worker_storage_type, _worker_constants

    _worker_model_name   = model_name
    _worker_model_path   = model_path
    _worker_storage_type = storage_type

    # 1+2. Load and apply this model's storage-type constants.
    type_cfg = load_config("simulate", storage_type=storage_type, load_optuna=False)
    apply_config_overrides(type_cfg)

    # 3. Reload modules that bound constants at import time (from constants import X).
    #    Order: leaf dependencies first so each reload sees correct upstream values.
    import trader.action
    import trader.state
    import trader.utils.reward_calc_utils
    importlib.reload(trader.action)
    importlib.reload(trader.state)
    importlib.reload(trader.utils.reward_calc_utils)

    # Snapshot effective constants for this worker (written per-row to Parquet).
    _worker_constants = type_cfg.get("constants", {})

    # 4. Load model after reloads.
    _worker_model = MaskablePPO.load(model_path)


def _worker_run_episode(args: tuple) -> dict:
    """Run one episode for this worker's model against the given price path.

    Returns a partial row dict (model identity + episode metrics).
    Seed-level metrics (spot stats, LP baseline) are added by the main process.
    """
    from trader.environment import TradingEnv

    h2_np_matrix, env_verbosity, log_dir = args

    lp_result = storage_valuation(
        h2_np_matrix[:,0],
        inventory=0,
        end_inventory=0,
        is_daily=True,
        max_inject_rate=constants.MAX_INJECTION_RATE,
        max_withdraw_rate=constants.MAX_WITHDRAW_RATE,
    )
    optimal_npv = float(lp_result["npv"])
    optimal_withdrawal_cashflow = float(
        np.sum(lp_result["discounted_cashflows"], where=lp_result["discounted_cashflows"] > 0))
    optimal_withdrawal_units = float(np.sum(lp_result["withdraw_schedule"]))

    test_env = DummyVecEnv([lambda m=h2_np_matrix: Monitor(TradingEnv(
        h2_np_matrix=m,
        verbosity=env_verbosity,
        log_dir=log_dir,
    ))])
    _worker_model.set_env(test_env)

    obs = test_env.reset()
    episode_reward = 0.0
    summary: dict = {}

    while True:
        action_masks = get_action_masks(test_env)
        action, _ = _worker_model.predict(
            obs, deterministic=True, action_masks=action_masks
        )
        obs, rewards, dones, infos = test_env.step(action)
        episode_reward += float(rewards[0])
        if dones[0]:
            summary = infos[0].get("episode_summary", {})
            break

    return {
        "model_name":                   _worker_model_name,
        "ppo_model":                    _worker_model_path,
        "storage_type":                 _worker_storage_type,
        "constants_json":               json.dumps(_worker_constants),
        "optimal_npv":                  optimal_npv,
        "optimal_withdrawal_cashflow":  optimal_withdrawal_cashflow,
        "optimal_withdrawal_units":     optimal_withdrawal_units,
        "episode_reward":               episode_reward,
        **summary,
    }


# ─── Simulation ──────────────────────────────────────────────────────────────

def simulate(cfg: dict, output_dir: str) -> None:
    """Load trained PPO models and run them against generated H2 price scenarios.

    For each seed in ``[seed_start, seed_start + num_seeds)``, this function:

    1. Generates a simulated TTF forward-curve feature matrix.
    2. Derives the H2 LCOH forward-curve matrix.
    3. Pre-computes the LP-optimal baseline on a downsampled price path (main process).
    4. Dispatches each model to its persistent worker process via ``apply_async``.
    5. Collects episode summaries from all workers for this seed.
    6. After all seeds, writes a single Parquet file to ``parquet_dir``.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    output_dir : str
        Root directory for run-specific debug logs and the config snapshot.
    """
    data_cfg    = cfg.get("data", {})
    sim_cfg     = cfg.get("simulate", {})
    logging_cfg = cfg.get("logging", {})

    ttf_path      = data_cfg.get("ttf_model", "models/ttf_regressor_v5.joblib")
    carbon_price  = data_cfg.get("carbon_price", constants.CCUS_COST_PER_TON)

    num_seeds     = sim_cfg.get("num_seeds", 10)
    seed_start    = sim_cfg.get("seed_start", 0)
    t_sim_years   = sim_cfg.get("t_sim_years", 2)
    env_verbosity = logging_cfg.get("verbosity", 0)

    t_sim = int(constants.TRADING_DAYS_PER_YEAR * t_sim_years)

    model_paths = _resolve_model_paths(data_cfg)

    print(f"Models      : { {n: v['storage_type'] for n, v in model_paths.items()} }")
    print(f"Seeds       : {seed_start} .. {seed_start + num_seeds - 1}")
    print(f"T_SIM       : {t_sim} days ({t_sim_years} years)")
    print(f"Output      : {output_dir}")

    run_id  = str(uuid.uuid4())[:8]
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # ── Create one persistent worker pool per model ───────────────────────────
    # Each pool has exactly one worker that loads its model via _worker_init().
    # Workers are created here (before the seed loop) so models are loaded once.
    # Each worker applies its own storage-type constants in _worker_init(), so
    # models for different technologies run with physically correct parameters.
    pools: dict = {}
    for model_name, model_info in model_paths.items():
        pools[model_name] = multiprocessing.Pool(
            processes=1,
            initializer=_worker_init,
            initargs=(model_name, model_info["path"], model_info["storage_type"]),
        )
        print(f"Worker ready : '{model_name}' [{model_info['storage_type']}] ← {model_info['path']}")

    ttf_regressor = load_ttf_regressor(cfg)
    h2_calculator = HydrogenLCOHCalculator()

    parquet_rows: list = []

    try:
        for i in range(num_seeds):
            seed = seed_start + i
            print(f"\n── Seed {seed} ({i + 1}/{num_seeds}) ──")

            sim_feature_df = ttf_regressor.generate_feature_df(T=t_sim, seed=seed)
            raw_predicted  = ttf_regressor.predict_curves_given_features(sim_feature_df)

            seasonality_alpha = random.uniform(0.5, 2)
            h2_np_matrix = h2_calculator.compute_blue_green_h2_index(
                ng_price=raw_predicted.values,
                carbon_price=carbon_price,
                seasonality_alpha=seasonality_alpha,
            )

            # ── LP baseline — computed once per seed, shared across all models ──
            spot_prices = h2_np_matrix[:, 0]
            h2_spot_mean                = float(np.mean(spot_prices))
            h2_spot_variance            = float(np.var(spot_prices))

            # ── Dispatch all models concurrently ──────────────────────────────
            futures = {}
            for model_name, pool in pools.items():
                log_dir = (
                    os.path.join(output_dir, "debug", f"sim_{seed}_{model_name}")
                    if env_verbosity >= 2 else None
                )
                futures[model_name] = pool.apply_async(
                    _worker_run_episode,
                    [(h2_np_matrix, env_verbosity, log_dir)],
                )

            # ── Collect results (blocks until all models finish this seed) ────
            for model_name, future in futures.items():
                worker_result = future.get()
                parquet_rows.append({
                    "run_id":          run_id,
                    "seed":            seed,
                    "ttf_model":       ttf_path,
                    "t_sim_years":     t_sim_years,
                    "seasonality_alpha": seasonality_alpha,
                    "carbon_price":    carbon_price,
                    "h2_spot_mean":    h2_spot_mean,
                    "h2_spot_variance": h2_spot_variance,
                    # storage_type, constants_json, model_name, ppo_model,
                    # optimal_*, episode_reward, and episode_summary fields all come from worker.
                    **worker_result,
                })
                print(f"  [{model_name}/{worker_result.get('storage_type')}] "
                      f"reward={worker_result.get('episode_reward', float('nan')):.4f}  "
                      f"cashflow={worker_result.get('total_cashflow', float('nan')):.2f}  "
                      f"opt_npv={worker_result.get('optimal_npv', float('nan')):.2f}")

    finally:
        # Always clean up worker processes, even on exception.
        for pool in pools.values():
            pool.terminate()
            pool.join()

    # ── Write Parquet to static parquet_dir ───────────────────────────────────
    parquet_dir = logging_cfg.get(
        "parquet_dir",
        os.path.join(logging_cfg.get("scratch_dir", "runs"), "parquet_outputs"),
    )
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(
        parquet_dir, f"results_task{task_id:05d}_{run_id}.parquet"
    )
    pd.DataFrame(parquet_rows).to_parquet(parquet_path, index=False, compression="snappy")
    print(f"\nWrote {len(parquet_rows)} rows → {parquet_path}")
    print("Run scripts/aggregate_results.py to ingest into DuckDB.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate trained PPO agents against generated H2 price scenarios.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for debug logs and config snapshot. "
             "Defaults to ./runs/simulate_<seed_start> locally, "
             "or uses $SLURM_JOB_NAME / $SLURM_JOBID on Great Lakes.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="First seed index (overrides config and Slurm auto-computation).",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, cfg: dict, seed_start: int) -> str:
    """Determine the output directory, respecting Slurm environment variables."""
    if args.output_dir:
        out = args.output_dir
    elif os.environ.get("SLURM_JOBID"):
        job_name = os.environ.get("SLURM_JOB_NAME", "simulate")
        job_id = os.environ["SLURM_JOBID"]
        scratch = cfg.get("logging", {}).get(
            "scratch_dir", "/scratch/chrzhang_root/chrzhang0"
        )
        out = os.path.join(scratch, "sim", f"{job_name}_{job_id}")
    else:
        out = os.path.join("runs", f"simulate_{seed_start}")
    os.makedirs(out, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()

    cfg = load_config("simulate", storage_type="unset", load_optuna=False)  # "unset" is a placeholder; worker processes load the correct storage type in _worker_init()

    sim_cfg   = cfg.setdefault("simulate", {})
    num_seeds = sim_cfg.get("num_seeds", 10)

    if args.seed_start is not None:
        seed_start = args.seed_start
    else:
        seed_start = sim_cfg.get("seed_start", 0)

    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        seed_start += task_id * num_seeds

    sim_cfg["seed_start"] = seed_start

    output_dir = resolve_output_dir(args, cfg, seed_start)

    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    simulate(cfg, output_dir)


if __name__ == "__main__":
    main()

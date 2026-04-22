import datetime as dt
import os
import random

import numpy as np

from analysis import (
    BasisVisualizer,
    CurveVisualizer,
)
from analysis.utils import evaluate_regressor, print_diagnostics
from analysis import SimulationVisualizer
from constants import NUM_M, TRADING_DAYS_PER_YEAR
from curve_factory import CurveRegressorFactory
from curve_factory.hydrogen_curve_factory import HydrogenLCOHCalculator

# %%
VERSION = 6
ttf_regressor = CurveRegressorFactory.load(f"models/ttf_regressor_v{VERSION}.joblib")

# %%

COMMODITY = "ttf"
diagnostics = evaluate_regressor(COMMODITY, ttf_regressor, ttf_regressor.feature_df)
print_diagnostics(diagnostics)

visualizer = BasisVisualizer(ttf_regressor)
visualizer.plot_macro_predicted_decomposition(
    feature_df=ttf_regressor.feature_df,
    n_pick=6,
    save_path=f"results/{COMMODITY}/macro_coef_basis_decomposition.png",
)
raw_predicted = ttf_regressor.predict_curves_given_features(ttf_regressor.feature_df)

T = min(len(raw_predicted), len(ttf_regressor.curve_data))

predicted_curves = {
    "matrix": raw_predicted,
    "dates": raw_predicted.index[-T:],
}

# %% visualize macro-predicted curves vs observed curves for a few dates
ttf_viz = CurveVisualizer(commodity_name="TTF", commodity_type="energy")
ttf_viz.load_curve_data(
    matrix=ttf_regressor.curve_data[-TRADING_DAYS_PER_YEAR * 8 + 50:-TRADING_DAYS_PER_YEAR * 5 - 50],
    maturities=np.arange(NUM_M + 1),
    dates=ttf_regressor.unique_dates[-TRADING_DAYS_PER_YEAR * 8+50:-TRADING_DAYS_PER_YEAR * 5 - 50],
    label="Macro-Predicted TTF",
)
# ttf_viz.generate_diagnostic_suite(output_dir="./results/ttf/")
ttf_viz.plot_surface_3d_static(
    title="TTF Forward Curve Surface",
    z_label="Price ($/MWh)",
    save_path="./results/ttf.png",
)

# %% compute Blue-Green H2 LCOH from TTF prices and visualize
h2_calc = HydrogenLCOHCalculator()
h2_lcoh_matrix = h2_calc.compute_blue_green_h2_index(ng_price=ttf_regressor.curve_data)
h2_viz = CurveVisualizer(commodity_name="Blue-Green H2 LCOH", commodity_type="energy")

#%%
h2_viz.load_curve_data(
    matrix=h2_lcoh_matrix[-TRADING_DAYS_PER_YEAR * 8 + 50:-TRADING_DAYS_PER_YEAR * 5 - 50],
    maturities=np.arange(NUM_M + 1),
    dates=ttf_regressor.unique_dates[-TRADING_DAYS_PER_YEAR * 8+50:-TRADING_DAYS_PER_YEAR * 5 - 50],
    label="Blue-Green H2 LCOH",
)
h2_viz.plot_surface_3d_static(
    title="Blue-Green H2 LCOH Surface (derived from TTF)",
    z_label="LCOH ($/MWh)",
    cmap="plasma",
    save_path="./results/h2.png",
)

# %%
res = ttf_regressor.reconstruct_maturity_for_date(dt.datetime(2023, 11, 8))
ukraine = ttf_regressor.reconstruct_maturity_for_date(dt.datetime(2022, 6, 20))

# %% Fit ARMA-GARCH feature simulators and generate scenario curves
T_SIM = int(TRADING_DAYS_PER_YEAR) * 3-100  # 2-year horizon of business days

ttf_regressor.fit_arma_garch_regressors()

# %%
seed=42
sim_feature_df = ttf_regressor.generate_feature_df(T=T_SIM, seed=seed)
raw_predicted = ttf_regressor.predict_curves_given_features(sim_feature_df)

predicted_curves = {
    "matrix": raw_predicted,
    "dates": raw_predicted.index,
}

viz = CurveVisualizer(commodity_name="ttf", commodity_type="energy")
viz.load_curve_data(
    matrix=predicted_curves["matrix"].to_numpy(),
    maturities=np.arange(NUM_M + 1),
    dates=predicted_curves["dates"],
    label=f"Simulated ttf seed {seed}",
)
viz.plot_surface_3d_static(
    title=f"Generated TTF Forward Curve Surface (seed={seed})",
    z_label="Price ($/MWh)",
)

#%%
h2_lcoh_matrix_pred = h2_calc.compute_blue_green_h2_index(ng_price=predicted_curves["matrix"].to_numpy())

h2_viz.plot_surface_3d_static(
    matrix=h2_lcoh_matrix_pred,
    dates=predicted_curves["dates"],
    title="Generated Blue-Green H2 Surface (seed=42)",
    z_label="LCOH ($/MWh)",
    cmap="plasma",
    save_path="./results/h242.png",
)

#%%
# h2 = 0.043 * TTF + 2.69513

N_SEEDS = 1_000_000
H2_STORE_PATH = "results/h2_simulations.npy"

os.makedirs(os.path.dirname(H2_STORE_PATH), exist_ok=True)

# Pre-allocate a contiguous memory-mapped array on disk.
# Shape: (N_SEEDS, TRADING_DAYS_PER_YEAR), float32 (~1 GB for 1M seeds × 252 days).
# Writing is O(1) per row; reading back uses np.load(..., mmap_mode='r') without
# loading the full array into RAM.
sim_store = np.lib.format.open_memmap(
    H2_STORE_PATH,
    mode="w+",
    dtype=np.float32,
    shape=(N_SEEDS, TRADING_DAYS_PER_YEAR),
)

for seed in range(N_SEEDS):
    sim_feature_df = ttf_regressor.generate_feature_df(T=TRADING_DAYS_PER_YEAR, seed=seed)
    raw_predicted = ttf_regressor.predict_curves_given_features(sim_feature_df).values[:, 0]

    # compress ttf randomly
    mean = np.mean(raw_predicted)
    compressed = mean + random.random() * (raw_predicted - mean)

    h2_raw = raw_predicted * 0.043 + 2.69513

    sim_store[seed] = h2_raw.astype(np.float32)

# Flush all pending writes to disk.
del sim_store
print(f"Saved {N_SEEDS} simulations to {H2_STORE_PATH}")

#%%
# load for visualization
# e.g. percentile fan chart across all seeds per day:
viz = SimulationVisualizer("results/h2_simulations.npy")
viz.plot(output_path="results/h2_distribution_diagnostic.png")

# %%
VERSION = 6

# heavy backwardation, high var but no LP revenue
seed = 115981
alpha = 1.9204460866782043
model = "lnh3"

# best runs
# lnh3, seed=182283, alpha=0.9631077624339696
# seed=57081, alpha=0.918026, model=baseline
# seed=57390, alpha=0.957557, model=lnh3
# 57746, 0.8732045063595729, meoh
# 57746, 0.8732045063595729, meoh
# 57746, 0.8732045063595729, meoh


ttf_regressor = CurveRegressorFactory.load(f"models/ttf_regressor_v{VERSION}.joblib")
ttf_regressor.fit_arma_garch_regressors()

sim_feature_df = ttf_regressor.generate_feature_df(T=TRADING_DAYS_PER_YEAR, seed=seed)
raw_predicted = ttf_regressor.predict_curves_given_features(sim_feature_df)

h2_calc = HydrogenLCOHCalculator()

h2_lcoh_matrix_pred = h2_calc.compute_blue_green_h2_index(ng_price=raw_predicted.to_numpy(), seasonality_alpha=alpha)

h2_viz = CurveVisualizer(commodity_name="Blue-Green H2 LCOH", commodity_type="energy")
h2_viz.load_curve_data(
    matrix=h2_lcoh_matrix_pred,
    maturities=np.arange(NUM_M + 1),
    dates=raw_predicted.index,
    label="Blue-Green H2 LCOH",
)
h2_viz.plot_spot_price(
    save_path=f"./results/{seed}.png",
)

#%%
from trader.utils.config_loader import load_config, apply_config_overrides
import importlib

# first load the correct constants
type_cfg = load_config("simulate", storage_type=model, load_optuna=False)
apply_config_overrides(type_cfg)

# 3. Reload modules that bound constants at import time (from constants import X).
#    Order: leaf dependencies first so each reload sees correct upstream values.
import trader.action
import trader.state
import trader.utils.reward_calc_utils

importlib.reload(trader.action)
importlib.reload(trader.state)
importlib.reload(trader.utils.reward_calc_utils)

from constants import INJECTION_COST_PER_UNIT, WITHDRAWAL_COST_PER_UNIT, BOIL_OFF
# validate constants have been overridden
print(INJECTION_COST_PER_UNIT, WITHDRAWAL_COST_PER_UNIT, BOIL_OFF)

#%%
# now load the correct model and run a full forward pass for that data with verbose = 2
def simulate_given_model_price(model: str, h2_np_matrix: np.ndarray):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from trader.environment import TradingEnv

    model_path = f"models/ppo_{model}.zip"
    loaded_model = MaskablePPO.load(model_path)

    test_env = DummyVecEnv([lambda m=h2_np_matrix: Monitor(TradingEnv(
        h2_np_matrix=m,
        verbosity=2,
        log_dir='./logs/manual',
    ))])
    obs = test_env.reset()
    episode_reward = 0.0
    summary: dict = {}

    while True:
        action_masks = get_action_masks(test_env)
        action, _ = loaded_model.predict(
            obs, deterministic=True, action_masks=action_masks
        )
        obs, rewards, dones, infos = test_env.step(action)
        episode_reward += float(rewards[0])
        if dones[0]:
            summary = infos[0].get("episode_summary", {})
            break

simulate_given_model_price(model=model, h2_np_matrix=h2_lcoh_matrix_pred)

#%%
from analysis import IntraEpisodeVisualizer
viz = IntraEpisodeVisualizer()
viz.load("logs/manual/")

# plot the inventory path and the h2 curve
fig = viz.plot_spot_and_inventory(episode=1)


#%%
from analysis import RunDBVisualizer
viz = RunDBVisualizer("simulation.duckdb")

techs = ["lnh3", "lh2", "meoh", "lohc"]
for t in techs:
    viz.save_plot("plot_2d_heatmap", f"out/heatmap_{t}.png", tech=t, figsize=(9, 7))

viz.save_grid(
    [("plot_2d_heatmap", t) for t in ["lnh3", "lh2", "meoh", "lohc"]],
    "out/heatmaps_all.png",
    ncols=4,
    panel_size=(9, 7),
)
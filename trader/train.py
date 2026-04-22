"""
trader.train
============
Unified CLI entry-point for PPO training, Optuna hyperparameter search,
and post-training evaluation on Great Lakes HPC or local machines.

Usage examples
--------------
**Local training**::

    python -m trader.train --type {lh2|lnh3|meoh|lohc|baseline} \
        --mode train \
        --output-dir ./runs/ppo_test

**Hyperparameter search (single trial)**::

    python -m trader.train --mode hparam \
        --type {lh2|lnh3|meoh|lohc|baseline} \
        --trial-id 0

**Great Lakes Slurm**::

    sbatch scripts/train_ppo.sh          # wraps --mode train
    sbatch scripts/hparam_search.sh      # wraps --mode hparam with --array
"""

import argparse
import json
import os
import uuid
from pathlib import Path

import torch
import yaml
from optuna import Study

from constants import Z_SCORE_CLIP
from curve_factory.hydrogen_curve_factory import load_theoretical_hydrogen_price
from trader.utils.config_loader import load_config, apply_config_overrides


# ─── Training ────────────────────────────────────────────────────────────────

def train(cfg: dict, output_dir: str) -> str:
    """Run a single PPO training session.

    Returns the session_id (used to locate saved artifacts).
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from trader.environment import TradingEnv
    from trader.utils.callbacks import TensorboardCallback

    session_id = cfg.get("session_id") or str(uuid.uuid4())[:8]
    seed = cfg.get("seed", 42)

    h2_np_matrix = load_theoretical_hydrogen_price(cfg)

    agent_cfg = cfg.get("agent", {})
    norm_cfg = cfg.get("normalization", {})

    trading_env = TradingEnv(
        h2_np_matrix=h2_np_matrix,
        verbosity=agent_cfg.get("verbosity", 1),
        log_dir=os.path.join(output_dir, "debug", f"ppo_{session_id}"),
    )

    # Wrap in DummyVecEnv → VecNormalize for observation and reward normalization.
    # action_masks() is transparently proxied through VecEnvWrapper.env_method().
    vec_env = DummyVecEnv([lambda: trading_env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=norm_cfg.get("norm_obs", False),
        norm_reward=norm_cfg.get("norm_reward", True),
        clip_obs=norm_cfg.get("clip_obs", Z_SCORE_CLIP),
        clip_reward=norm_cfg.get("clip_reward", Z_SCORE_CLIP),
        gamma=agent_cfg.get("gamma", 0.9883),
    )

    activation_fn_name = agent_cfg.get("activation_fn", "relu")
    activation_fn = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[activation_fn_name]

    # Asymmetric pi/vf architecture: pi_size and vf_size define separate widths.
    # Falls back to net_arch (symmetric) for backward compat with older configs.
    if "pi_size" in agent_cfg or "vf_size" in agent_cfg:
        pi_size = agent_cfg.get("pi_size", 64)
        vf_size = agent_cfg.get("vf_size", 128)
        net_arch = dict(pi=[pi_size, pi_size], vf=[vf_size, vf_size])
    else:
        net_arch_name = agent_cfg.get("net_arch", "large")
        net_arch = {"small": [64, 64], "medium": [128, 128], "large": [256, 256]}[net_arch_name]

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=activation_fn,
    )

    # Build PPO kwargs, filtering out None values (null in YAML).
    ppo_kwargs = dict(
        n_steps=agent_cfg.get("n_steps", 1024),
        batch_size=agent_cfg.get("batch_size", 256),
        gamma=agent_cfg.get("gamma", 0.9883),
        gae_lambda=agent_cfg.get("gae_lambda", 0.9695),
        learning_rate=agent_cfg.get("learning_rate", 2.39e-4),
        ent_coef=agent_cfg.get("ent_coef", 5.52e-6),
        clip_range=agent_cfg.get("clip_range", 0.2224),
        n_epochs=agent_cfg.get("n_epochs", 13),
        vf_coef=agent_cfg.get("vf_coef", 0.6908),
        max_grad_norm=agent_cfg.get("max_grad_norm", 0.6226),
        normalize_advantage=agent_cfg.get("normalize_advantage", True),
    )

    # clip_range_vf and target_kl are optional — only pass if not None/null.
    clip_range_vf = agent_cfg.get("clip_range_vf")
    if clip_range_vf is not None:
        ppo_kwargs["clip_range_vf"] = clip_range_vf

    target_kl = agent_cfg.get("target_kl")
    if target_kl is not None:
        ppo_kwargs["target_kl"] = target_kl

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=agent_cfg.get("verbosity", 1),
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
        policy_kwargs=policy_kwargs,
        device="cpu",
        **ppo_kwargs,
    )

    total_timesteps = agent_cfg.get("total_timesteps", 500_000)

    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback(verbose=agent_cfg.get("verbosity", 1)),
    )

    # Save model, VecNormalize stats, and config.
    model_path = os.path.join(output_dir, f"ppo_{session_id}.zip")
    vecnorm_path = os.path.join(output_dir, f"ppo_{session_id}_vecnormalize.pkl")
    model.save(model_path)
    vec_env.save(vecnorm_path)

    meta = {"session_id": session_id, "seed": seed, "config": cfg}
    meta_path = os.path.join(output_dir, f"ppo_{session_id}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Model saved       : {model_path}")
    print(f"VecNormalize saved: {vecnorm_path}")
    print(f"Metadata saved    : {meta_path}")
    return model_path


# ─── Hyperparameter search ───────────────────────────────────────────────────

def sample_hparams(trial) -> dict:
    """Sample a hyperparameter configuration from Optuna.

    Search space:
      clip_range         : {0.1, 0.2, 0.3}
      ent_coef           : {0, 0.0001, 0.001, 0.01}
      gae_lambda         : U(0.9, 1.0)
      gamma              : U(0.8, 0.9997)
      learning_rate      : {3e-6, 3e-5, 3e-4, 3e-3}
      log_std_init       : {-1, 0, 1, 2, 3}  (policy_kwargs)
      n_epochs           : {3,6,9,12,15,18,21,24,27,30}
      n_steps            : {512, 1024, 2048, 4096}
      normalize_advantage: {True, False}
      target_kl          : U(0.003, 0.03)
      vf_coef            : {0.5, 1.0}
      batch_size         : {64, 128, 256}
    """
    pi_size = trial.suggest_categorical("pi_size", [64, 128, 256])
    vf_size = trial.suggest_categorical("vf_size", [128, 256, 512])

    net_arch = dict(pi=[pi_size, pi_size], vf=[vf_size, vf_size, vf_size])

    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[activation_fn_name]

    return dict(
        net_arch=net_arch,
        activation_fn=activation_fn,
        n_steps=trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        gamma=trial.suggest_float("gamma", 0.8, 0.9997),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 1.0),
        learning_rate=trial.suggest_categorical("learning_rate", [3e-6, 3e-5, 3e-4, 3e-3]),
        ent_coef=trial.suggest_categorical("ent_coef", [0.0, 1e-4, 1e-3, 1e-2]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        n_epochs=trial.suggest_categorical("n_epochs", [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
        target_kl=trial.suggest_float("target_kl", 0.003, 0.03),
        vf_coef=trial.suggest_categorical("vf_coef", [0.5, 1.0]),
        normalize_advantage=trial.suggest_categorical("normalize_advantage", [True, False]),
    )


def hparam_search(cfg: dict, output_dir: str, trial_id: int) -> Study:
    """Run a single Optuna trial (designed for Slurm job arrays).

    Each trial gets its own output subdirectory and uses a shared Optuna
    storage backend (SQLite by default) so parallel trials can coordinate.
    """
    import optuna
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from trader.environment import TradingEnv
    from trader.utils.callbacks import TensorboardCallback

    search_cfg = cfg.get("search", {})
    norm_cfg = cfg.get("normalization", {})
    n_trials = search_cfg.get("n_trials_per_worker", 1)
    total_timesteps = search_cfg.get("total_timesteps", 200_000)
    seed = cfg.get("seed", 42) + trial_id
    study_name = search_cfg.get("study_name", "h2_ppo_hparam")
    storage = search_cfg.get(
        "storage", f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"
    )

    h2_np_matrix = load_theoretical_hydrogen_price(cfg)

    def objective(trial):
        hparams = sample_hparams(trial)

        env = TradingEnv(
            h2_np_matrix=h2_np_matrix,
            verbosity=0,
        )
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=norm_cfg.get("norm_obs", False),
            norm_reward=norm_cfg.get("norm_reward", True),
            clip_obs=norm_cfg.get("clip_obs", Z_SCORE_CLIP),
            clip_reward=norm_cfg.get("clip_reward", Z_SCORE_CLIP),
            gamma=hparams.get("gamma", 0.99),
        )

        # normalize_advantage is a PPO arg, not a policy_kwargs field.
        normalize_advantage = hparams.pop("normalize_advantage")
        policy_kwargs = {
            "net_arch": hparams.pop("net_arch"),
            "activation_fn": hparams.pop("activation_fn"),
        }

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            verbose=0,
            tensorboard_log=os.path.join(output_dir, "tensorboard", f"trial_{trial_id}"),
            policy_kwargs=policy_kwargs,
            normalize_advantage=normalize_advantage,
            device="cpu",
            **hparams,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=TensorboardCallback(verbose=0),
        )

        # Evaluate: mean episode reward over a handful of rollouts.
        # Set VecNormalize to eval mode (freeze running stats, no reward norm).
        vec_env.training = False
        vec_env.norm_reward = False

        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=5)

        # Save trial checkpoint.
        trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        model.save(os.path.join(trial_dir, "model.zip"))
        vec_env.save(os.path.join(trial_dir, "vecnormalize.pkl"))
        with open(os.path.join(trial_dir, "params.json"), "w") as f:
            json.dump(trial.params, f, indent=2)

        return mean_reward

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"Trial {trial_id} complete.  Best value so far: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


def evaluate(cfg: dict, model_path: str, output_dir: str) -> None:
    """Load a trained model and run full-horizon + episode-level evaluation.

    The debug log directory is resolved in this order:
      1. ``eval.log_dir`` in the YAML config (explicit override)
      2. Sibling ``debug/ppo_<session_id>`` directory next to the model zip
         (standard layout produced by ``train()``)
      3. ``./logs/debug/ppo_<session_id>`` (legacy local fallback)
    """
    from analysis.visualizers.episode_log_visualizer import (
        evaluate_complete_horizon,
        evaluate_specific_episode,
    )

    h2_np_matrix = load_theoretical_hydrogen_price(cfg)

    # Extract session_id from the model filename (ppo_<session_id>.zip).
    session_id = Path(model_path).stem.replace("ppo_", "")

    eval_cfg = cfg.get("eval", {})

    # Resolve log_dir: config override → model sibling → local fallback.
    log_dir = eval_cfg.get("log_dir")
    if log_dir is None:
        # The training script writes logs to <run_dir>/debug/ppo_<session_id>/
        candidate = Path(model_path).parent / "debug" / f"ppo_{session_id}"
        if candidate.is_dir():
            log_dir = str(candidate)

    print(f"Session  : {session_id}")
    print(f"Log dir  : {log_dir}")
    print(f"Output   : {output_dir}")

    evaluate_complete_horizon(
        h2_np_matrix, session_id, log_dir=log_dir, output_dir=output_dir,
    )

    for ep in eval_cfg.get("episodes", []):
        evaluate_specific_episode(
            session_id, ep, log_dir=log_dir, output_dir=output_dir,
        )

    print(f"Evaluation complete for session {session_id}")
    print(f"Outputs in: {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO training, hyperparameter search, and evaluation for H2 trading.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "hparam", "evaluate"],
        default="train",
        help="Execution mode (default: train).",
    )
    parser.add_argument(
        "--type",
        dest="storage_type",
        type=str,
        required=True,
        help="Storage technology type (lh2, lnh3, meoh, lohc, baseline)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models, logs, and plots.  "
             "Defaults to ./runs/<mode>_<session_id> locally, "
             "or uses $SLURM_JOB_NAME / $SLURM_JOBID on Great Lakes.",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=None,
        help="Trial ID for hparam search (set automatically from "
             "$SLURM_ARRAY_TASK_ID on Great Lakes).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model .zip file (required for --mode evaluate).",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, cfg: dict) -> str:
    """Determine the output directory, respecting Slurm env vars."""
    if args.output_dir:
        out = args.output_dir
    elif os.environ.get("SLURM_JOBID"):
        job_name = os.environ.get("SLURM_JOB_NAME", args.mode)
        job_id = os.environ["SLURM_JOBID"]
        scratch = cfg.get("logging", {}).get("scratch_dir", "/scratch/siads699w26_class_root/siads699w26_class/chrzhang")
        out = os.path.join(scratch, "runs", f"{job_name}_{job_id}")
    else:
        session = cfg.get("session_id") or str(uuid.uuid4())[:8]
        out = os.path.join("runs", f"{args.mode}_{session}")
    os.makedirs(out, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()

    cfg = load_config(args.mode, storage_type=args.storage_type)

    # Patch constants module BEFORE any trader.* imports happen
    apply_config_overrides(cfg)

    output_dir = resolve_output_dir(args, cfg)

    # Save a copy of the config used for this run.
    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    if args.mode == "train":
        trained_model_path = train(cfg, output_dir)

        # eval_output_dir = output_dir + "/eval"
        # os.makedirs(eval_output_dir, exist_ok=True)
        # evaluate(cfg, trained_model_path, eval_output_dir)

    elif args.mode == "hparam":
        trial_id = args.trial_id
        if trial_id is None:
            trial_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        hparam_search(cfg, output_dir, trial_id)


if __name__ == "__main__":
    main()

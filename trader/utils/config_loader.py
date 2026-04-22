"""
trader.utils.config_loader
==========================
Shared config loading with optional storage-type overlay and automatic
best-hyperparameter injection from a completed Optuna study.

Usage::

    from trader.utils.config_loader import load_config

    cfg = load_config("train", storage_type="lh2")
    # cfg["constants"] now contains the lh2-specific values
    # cfg["storage_type"] == "lh2"
    # cfg["agent"] is overwritten with best Optuna params if a study DB exists
"""

from pathlib import Path

import yaml

import constants

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STORAGE_TYPES_DIR = _PROJECT_ROOT / "configs" / "storage_types"
_OPTUNA_STUDIES_DIR = _PROJECT_ROOT / "optuna_studies"
_VALID_TYPES = {p.stem for p in _STORAGE_TYPES_DIR.glob("*.yaml")} if _STORAGE_TYPES_DIR.is_dir() else set()

# Optuna trial param names that belong in policy_kwargs (agent config fields)
# rather than being passed directly as PPO constructor args.
_POLICY_KWARGS_PARAMS = {"pi_size", "vf_size", "activation_fn"}


def _load_best_optuna_params(storage_type: str, study_name: str) -> dict | None:
    """Load best hyperparameters from an Optuna SQLite study, if one exists.

    Looks for::

        {project_root}/optuna_studies/{storage_type}/optuna.db

    Returns ``None`` if the DB or study does not exist, or if optuna is not
    installed.  On success returns the raw ``study.best_params`` dict (the
    same names used by ``trial.suggest_*`` in ``sample_hparams``).
    """
    db_path = _OPTUNA_STUDIES_DIR / storage_type / "optuna.db"
    if not db_path.exists():
        return None

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage_url = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        if not study.trials:
            return None
        return study.best_params
    except Exception as exc:  # noqa: BLE001 — optuna not installed or study missing
        print(f"[config_loader] Could not load Optuna study '{study_name}' "
              f"from {db_path}: {exc}")
        return None


def _apply_optuna_params(cfg: dict, best_params: dict) -> None:
    """Merge Optuna best_params into cfg['agent'].

    Optuna stores raw trial-suggestion names.  The mapping to agent config
    fields is 1-to-1 for all PPO constructor args.  Architecture params
    (``pi_size``, ``vf_size``) and ``activation_fn`` (stored as a string)
    are kept as-is — ``train.py`` already reads them with ``agent_cfg.get()``.

    Parameters modified in-place on ``cfg['agent']``:
      pi_size, vf_size, activation_fn (str), n_steps, batch_size, gamma,
      gae_lambda, learning_rate, ent_coef, clip_range, n_epochs, target_kl,
      vf_coef, normalize_advantage
    """
    agent = cfg.setdefault("agent", {})
    for key, value in best_params.items():
        agent[key] = value


def load_config(mode: str, storage_type: str, load_optuna: bool = True) -> dict:
    """Load a base YAML config, merge storage-type constants, and inject
    best Optuna hyperparameters when a completed study exists.

    Parameters
    ----------
    mode : str
        Mode {train,evaluate,hparam,simulate}. Looks up ``configs/{mode}.yaml``.
    storage_type : str
        Storage technology identifier (``lh2``, ``lnh3``, ``meoh``, ``lohc``).
        Controls both the constants overlay and the Optuna study path.
    load_optuna : bool
        When ``False``, skip the Optuna best-params injection entirely.
        Useful for simulation / evaluation where agent hparams are baked into
        the saved model and do not need to be re-applied to the config.

    Returns
    -------
    dict
        Fully-resolved config dict.  ``cfg["storage_type"]`` is always set.
        If a completed Optuna study is found at
        ``optuna_studies/{storage_type}/optuna.db``, the ``agent`` section is
        overwritten with the best trial's hyperparameters (unless
        ``load_optuna=False``).
    """
    config_path = _PROJECT_ROOT / "configs" / f"{mode}.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if storage_type not in _VALID_TYPES:
        return cfg

    type_path = _STORAGE_TYPES_DIR / f"{storage_type}.yaml"
    with open(type_path, "r") as f:
        type_cfg = yaml.safe_load(f) or {}

    # Flat merge: storage-type constants override base constants.
    base_constants = cfg.get("constants", {}) or {}
    base_constants.update(type_cfg.get("constants", {}))
    cfg["constants"] = base_constants

    # Record which type was used (persisted in config_used.yaml).
    cfg["storage_type"] = storage_type

    # Inject best Optuna hyperparameters if a completed study exists.
    if load_optuna:
        study_name = cfg.get("search", {}).get("study_name", "h2_ppo_hparam")
        best_params = _load_best_optuna_params(storage_type, study_name)
        if best_params is not None:
            _apply_optuna_params(cfg, best_params)
            print(f"[config_loader] Loaded best Optuna params for '{storage_type}' "
                  f"(study='{study_name}'): {best_params}")
        else:
            print(f"[config_loader] No Optuna study found for '{storage_type}' — "
                  f"using config defaults.")

    return cfg


def apply_config_overrides(cfg: dict) -> None:
    """Patch the ``constants`` module from the YAML ``constants:`` section.

    Must be called **before** any ``trader.*`` modules are imported so that
    module-level bindings (dataclass field defaults in ``state.py``, function
    parameter defaults in ``reward_calc_utils.py``) pick up the overridden
    values.

    Derived constants (``DISCOUNT_RATE``, ``TRADING_DAYS_PER_MONTH``) are
    automatically recomputed after all overrides are applied, unless they
    were themselves explicitly overridden.

    Example YAML::

        constants:
          STORAGE_CAPACITY: 2000.0
          MAX_INJECTION_RATE: 60.0
          BOIL_OFF: 0.02            # DISCOUNT_RATE is recomputed automatically
    """
    overrides = cfg.get("constants", {})
    if not overrides:
        return

    for name, value in overrides.items():
        if not hasattr(constants, name):
            raise ValueError(
                f"Unknown constant '{name}'. "
                f"Check spelling against constants.py."
            )
        setattr(constants, name, value)

    # Log what was changed.
    patched = ", ".join(f"{k}={v}" for k, v in overrides.items())
    print(f"Constants overridden: {patched}")

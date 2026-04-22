import optuna
from optuna import Study


def visualize_hparam_study(study):
    # Built-in plots
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
    )
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    plot_parallel_coordinate(study).show()


def load_optuna_study(study_name="h2_ppo_hparam", db="sqlite:///optuna.db") -> Study:
    study = optuna.load_study(
        study_name=study_name,
        storage=db,
    )

    # Best trial
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # All trials ranked
    df = study.trials_dataframe().sort_values("value", ascending=False)

    return study

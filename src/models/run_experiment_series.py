"""Run a comparable series of MLflow experiments with different hyperparameters."""

from __future__ import annotations

import argparse

from src.models.train_experiment import ExperimentConfig, run_experiment


EXPERIMENT_GRID = (
    {"max_depth": 4, "n_estimators": 100},
    {"max_depth": 5, "n_estimators": 200},
    {"max_depth": 7, "n_estimators": 300},
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a series of MLflow experiments for model comparison.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/patient_features.csv",
        help="Path to a processed CSV dataset. A demo dataset is used if absent.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cvd-risk-prediction",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for stratified cross-validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    print("Starting MLflow experiment series.")
    for config_item in EXPERIMENT_GRID:
        config = ExperimentConfig(
            data_path=arguments.data_path,
            experiment_name=arguments.experiment_name,
            max_depth=config_item["max_depth"],
            n_estimators=config_item["n_estimators"],
            cv_folds=arguments.cv_folds,
            random_state=arguments.random_state,
        )
        run_id = run_experiment(config)
        print(
            "Completed run:",
            f"run_id={run_id},",
            f"max_depth={config.max_depth},",
            f"n_estimators={config.n_estimators}",
        )

    print("MLflow experiment series finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Select the best MLflow run by ROC-AUC with support from precision and recall."""

from __future__ import annotations

import argparse

import mlflow

from src.utils.mlflow_config import get_local_tracking_uri


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best MLflow run for CVD risk prediction.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cvd-risk-prediction",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def get_best_run(experiment_name: str):
    """Return the best MLflow run ordered by ROC-AUC, Precision, then Recall."""
    mlflow.set_tracking_uri(get_local_tracking_uri())
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' was not found in MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[
            "metrics.roc_auc DESC",
            "metrics.precision DESC",
            "metrics.recall DESC",
        ],
    )
    if runs.empty:
        raise ValueError(f"Experiment '{experiment_name}' does not contain runs yet.")

    return runs.iloc[0]


def main() -> int:
    arguments = parse_arguments()

    try:
        best_run = get_best_run(arguments.experiment_name)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print("Best MLflow run selected.")
    print(f"Experiment: {arguments.experiment_name}")
    print(f"Run ID: {best_run['run_id']}")
    print(f"ROC-AUC: {best_run['metrics.roc_auc']:.4f}")
    print(f"Precision: {best_run['metrics.precision']:.4f}")
    print(f"Recall: {best_run['metrics.recall']:.4f}")
    print(f"max_depth: {best_run['params.max_depth']}")
    print(f"n_estimators: {best_run['params.n_estimators']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

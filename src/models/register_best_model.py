"""Register the best MLflow model version and move it to Staging."""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.models.select_best_run import get_best_run
from src.utils.mlflow_config import REGISTERED_MODEL_NAME, get_local_tracking_uri


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register the best MLflow model version in Model Registry.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cvd-risk-prediction",
        help="MLflow experiment name used for model selection.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=REGISTERED_MODEL_NAME,
        help="Registered model name in MLflow Model Registry.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    mlflow.set_tracking_uri(get_local_tracking_uri())
    client = MlflowClient()

    try:
        best_run = get_best_run(arguments.experiment_name)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    model_uri = f"runs:/{best_run['run_id']}/mlflow_model"

    try:
        client.create_registered_model(arguments.model_name)
    except Exception:
        # The model may already exist, which is acceptable for repeated runs.
        pass

    created_version = mlflow.register_model(
        model_uri=model_uri,
        name=arguments.model_name,
    )
    client.transition_model_version_stage(
        name=arguments.model_name,
        version=created_version.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    print("Model registered successfully.")
    print(f"Experiment: {arguments.experiment_name}")
    print(f"Selected run ID: {best_run['run_id']}")
    print(f"Registered model: {arguments.model_name}")
    print(f"Version: {created_version.version}")
    print("Stage: Staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

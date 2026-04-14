"""Simulate a Champion/Challenger evaluation flow with MLflow Registry."""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.mlflow_config import REGISTERED_MODEL_NAME, get_local_tracking_uri


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate Champion/Challenger validation using MLflow stages.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=REGISTERED_MODEL_NAME,
        help="Registered model name in MLflow Model Registry.",
    )
    parser.add_argument(
        "--evaluation-experiment",
        type=str,
        default="cvd-risk-model-validation",
        help="Experiment name for Champion/Challenger evaluation runs.",
    )
    parser.add_argument(
        "--min-roc-auc-gain",
        type=float,
        default=0.01,
        help="Minimum ROC-AUC improvement required for promotion.",
    )
    return parser.parse_args()


def _get_stage_version(client: MlflowClient, model_name: str, stage: str):
    versions = client.get_latest_versions(model_name, stages=[stage])
    return versions[0] if versions else None


def _get_run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    run = client.get_run(run_id)
    return run.data.metrics


def main() -> int:
    arguments = parse_arguments()

    mlflow.set_tracking_uri(get_local_tracking_uri())
    client = MlflowClient()

    champion_version = _get_stage_version(client, arguments.model_name, "Production")
    challenger_version = _get_stage_version(client, arguments.model_name, "Staging")

    if challenger_version is None:
        raise SystemExit(
            f"Model '{arguments.model_name}' does not have a Staging version yet."
        )

    challenger_metrics = _get_run_metrics(client, challenger_version.run_id)
    champion_metrics = (
        _get_run_metrics(client, champion_version.run_id)
        if champion_version is not None
        else {"roc_auc": 0.0, "precision": 0.0, "recall": 0.0}
    )

    roc_auc_gain = challenger_metrics.get("roc_auc", 0.0) - champion_metrics.get(
        "roc_auc", 0.0
    )
    precision_gain = challenger_metrics.get(
        "precision", 0.0
    ) - champion_metrics.get("precision", 0.0)
    recall_gain = challenger_metrics.get("recall", 0.0) - champion_metrics.get(
        "recall", 0.0
    )

    promote_to_production = (
        champion_version is None
        or roc_auc_gain >= arguments.min_roc_auc_gain
        or (
            roc_auc_gain >= 0
            and precision_gain >= 0
            and recall_gain >= 0
        )
    )

    mlflow.set_experiment(arguments.evaluation_experiment)
    with mlflow.start_run(run_name="champion-challenger-evaluation"):
        mlflow.log_param("model_name", arguments.model_name)
        mlflow.log_param("champion_version", champion_version.version if champion_version else "none")
        mlflow.log_param("challenger_version", challenger_version.version)
        mlflow.log_param("min_roc_auc_gain", arguments.min_roc_auc_gain)
        mlflow.log_metric("champion_roc_auc", champion_metrics.get("roc_auc", 0.0))
        mlflow.log_metric("champion_precision", champion_metrics.get("precision", 0.0))
        mlflow.log_metric("champion_recall", champion_metrics.get("recall", 0.0))
        mlflow.log_metric("challenger_roc_auc", challenger_metrics.get("roc_auc", 0.0))
        mlflow.log_metric(
            "challenger_precision", challenger_metrics.get("precision", 0.0)
        )
        mlflow.log_metric("challenger_recall", challenger_metrics.get("recall", 0.0))
        mlflow.log_metric("roc_auc_gain", roc_auc_gain)
        mlflow.log_metric("precision_gain", precision_gain)
        mlflow.log_metric("recall_gain", recall_gain)
        mlflow.log_param("promotion_decision", "promote" if promote_to_production else "hold")

    if promote_to_production:
        client.transition_model_version_stage(
            name=arguments.model_name,
            version=challenger_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

    print("Champion/Challenger evaluation completed.")
    print(f"Model: {arguments.model_name}")
    print(
        f"Champion version: {champion_version.version if champion_version else 'none'}"
    )
    print(f"Challenger version: {challenger_version.version}")
    print(f"ROC-AUC gain: {roc_auc_gain:.4f}")
    print(f"Precision gain: {precision_gain:.4f}")
    print(f"Recall gain: {recall_gain:.4f}")
    print("Decision:", "promote to Production" if promote_to_production else "keep in Staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a tracked MLflow experiment for CVD complication risk prediction."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from src.data.demo_dataset import TARGET_COLUMN, load_training_data
from src.models.artifacts import save_confusion_matrix_figure, save_model_pickle
from src.models.evaluation import evaluate_classifier_cv
from src.utils.mlflow_config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_BACKEND_DIR,
    get_local_artifact_root,
    get_local_tracking_uri,
)


@dataclass
class ExperimentConfig:
    data_path: str
    experiment_name: str
    max_depth: int
    n_estimators: int
    cv_folds: int
    random_state: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tracked MLflow experiment for CVD risk prediction.",
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
        "--max-depth",
        type=int,
        default=5,
        help="RandomForest max_depth hyperparameter.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="RandomForest n_estimators hyperparameter.",
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


def run_experiment(config: ExperimentConfig) -> str:
    MLFLOW_BACKEND_DIR.mkdir(exist_ok=True)
    MLFLOW_ARTIFACTS_DIR.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(get_local_tracking_uri())
    mlflow.set_experiment(config.experiment_name)

    dataset = load_training_data(
        data_path=config.data_path,
        random_state=config.random_state,
    )
    features = dataset.drop(columns=[TARGET_COLUMN])
    target = dataset[TARGET_COLUMN]

    model = RandomForestClassifier(
        max_depth=config.max_depth,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1,
    )

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("selection_metric", "roc_auc")
        mlflow.log_param("max_depth", config.max_depth)
        mlflow.log_param("n_estimators", config.n_estimators)
        mlflow.log_param("cv_folds", config.cv_folds)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("artifact_root", get_local_artifact_root())
        mlflow.log_param(
            "data_source",
            config.data_path
            if Path(config.data_path).exists()
            else "generated_demo_medical_dataset",
        )

        evaluation_result = evaluate_classifier_cv(
            estimator=model,
            features=features,
            target=target,
            cv_folds=config.cv_folds,
            random_state=config.random_state,
        )
        mlflow.log_metric("roc_auc", evaluation_result.roc_auc)
        mlflow.log_metric("precision", evaluation_result.precision)
        mlflow.log_metric("recall", evaluation_result.recall)

        model.fit(features, target)

        with tempfile.TemporaryDirectory() as temp_directory:
            temp_dir = Path(temp_directory)
            confusion_matrix_path = temp_dir / "confusion_matrix.png"
            model_path = temp_dir / "random_forest_model.pkl"

            save_confusion_matrix_figure(
                matrix=evaluation_result.confusion_matrix,
                output_path=confusion_matrix_path,
            )
            save_model_pickle(model=model, output_path=model_path)

            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.sklearn.log_model(model, artifact_path="mlflow_model")

        run_id = mlflow.active_run().info.run_id

    return run_id


def main() -> int:
    arguments = parse_arguments()
    config = ExperimentConfig(
        data_path=arguments.data_path,
        experiment_name=arguments.experiment_name,
        max_depth=arguments.max_depth,
        n_estimators=arguments.n_estimators,
        cv_folds=arguments.cv_folds,
        random_state=arguments.random_state,
    )
    run_id = run_experiment(config)

    print("MLflow experiment finished successfully.")
    print(f"Tracking URI: {get_local_tracking_uri()}")
    print(f"Experiment: {arguments.experiment_name}")
    print(f"Run ID: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

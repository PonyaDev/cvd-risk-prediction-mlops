"""Local MLflow configuration for the project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_BACKEND_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
REGISTERED_MODEL_NAME = "cvd-risk-prediction-random-forest"


def get_local_backend_store_uri() -> str:
    """Return a local SQLite backend URI for MLflow tracking and registry."""
    return f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"


def get_local_tracking_uri() -> str:
    """Return a local tracking URI compatible with the MLflow Model Registry."""
    return get_local_backend_store_uri()


def get_local_artifact_root() -> str:
    """Return a local file-based artifact root for MLflow."""
    return MLFLOW_ARTIFACTS_DIR.resolve().as_uri()

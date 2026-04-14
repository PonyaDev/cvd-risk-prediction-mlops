"""Utilities for loading project data or generating a demo medical dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "cvd_complication_risk"


def create_demo_dataset(
    sample_size: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset with medically meaningful feature names."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(35, 86, size=sample_size)
    systolic_bp = rng.normal(138, 18, size=sample_size).clip(90, 220)
    cholesterol = rng.normal(5.8, 1.1, size=sample_size).clip(2.5, 10.0)
    bmi = rng.normal(29, 5.2, size=sample_size).clip(17, 48)
    glucose = rng.normal(6.1, 1.8, size=sample_size).clip(3.5, 16.0)
    smoker = rng.binomial(1, 0.28, size=sample_size)
    hypertension = rng.binomial(1, 0.42, size=sample_size)
    prior_cvd = rng.binomial(1, 0.18, size=sample_size)

    linear_score = (
        0.045 * (age - 50)
        + 0.03 * (systolic_bp - 130)
        + 0.42 * (cholesterol - 5.2)
        + 0.06 * (bmi - 27)
        + 0.22 * (glucose - 5.5)
        + 0.75 * smoker
        + 0.95 * hypertension
        + 1.15 * prior_cvd
        - 2.8
    )
    probability = 1 / (1 + np.exp(-linear_score))
    target = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age,
            "systolic_bp": np.round(systolic_bp, 1),
            "cholesterol": np.round(cholesterol, 2),
            "bmi": np.round(bmi, 1),
            "glucose": np.round(glucose, 2),
            "smoker": smoker,
            "hypertension": hypertension,
            "prior_cvd": prior_cvd,
            TARGET_COLUMN: target,
        }
    )


def load_training_data(
    data_path: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a CSV dataset if it exists or generate a demo dataset otherwise."""
    if data_path:
        path = Path(data_path)
        if path.exists():
            return pd.read_csv(path)

    return create_demo_dataset(random_state=random_state)

"""Evaluation utilities for MLflow experiment tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


@dataclass
class EvaluationResult:
    roc_auc: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray


def evaluate_classifier_cv(
    estimator,
    features,
    target,
    cv_folds: int = 5,
    random_state: int = 42,
) -> EvaluationResult:
    """Evaluate a classifier with stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    probability_predictions = cross_val_predict(
        clone(estimator),
        features,
        target,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    class_predictions = (probability_predictions >= 0.5).astype(int)

    return EvaluationResult(
        roc_auc=roc_auc_score(target, probability_predictions),
        precision=precision_score(target, class_predictions, zero_division=0),
        recall=recall_score(target, class_predictions, zero_division=0),
        confusion_matrix=confusion_matrix(target, class_predictions),
    )

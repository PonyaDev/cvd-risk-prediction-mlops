"""Artifact creation helpers for ML experiments."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_figure(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix plot as an image file."""
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_model_pickle(model, output_path: Path) -> None:
    """Serialize a trained model as a pickle file."""
    with output_path.open("wb") as file_pointer:
        pickle.dump(model, file_pointer)

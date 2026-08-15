from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def feature_bounds(data: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Return min/max bounds for each selected numeric feature."""
    bounds: dict[str, tuple[float, float]] = {}
    for column in feature_columns:
        minimum = float(data[column].min())
        maximum = float(data[column].max())
        if minimum == maximum:
            maximum = minimum + 1.0
        bounds[column] = (minimum, maximum)
    return bounds


def scale_features(values: Iterable[float], feature_columns: Sequence[str], bounds: Mapping[str, tuple[float, float]]) -> np.ndarray:
    """Apply min-max scaling to a feature vector using supplied dataset bounds."""
    values = list(values)
    scaled = []
    for value, column in zip(values, feature_columns):
        minimum, maximum = bounds[column]
        scaled.append((float(value) - minimum) / (maximum - minimum))
    return np.asarray(scaled, dtype=float)


def calculate_distance(
    student1: Iterable[float],
    student2: Iterable[float],
    feature_columns: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
) -> float:
    """Calculate Euclidean distance after min-max scaling each feature."""
    first = scale_features(student1, feature_columns, bounds)
    second = scale_features(student2, feature_columns, bounds)
    return float(np.linalg.norm(first - second))


def get_nearest_neighbors(
    data: pd.DataFrame,
    new_student: Iterable[float],
    k: int,
    feature_columns: Sequence[str],
    label_column: str,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """Return the K training examples with the smallest scaled distance."""
    if k < 1:
        raise ValueError("K must be at least 1.")
    if k > len(data):
        raise ValueError("K cannot be larger than the number of training examples.")
    if len(feature_columns) < 2:
        raise ValueError("At least two feature columns are required.")

    bounds = bounds or feature_bounds(data, feature_columns)
    new_student = list(new_student)
    distances = []

    for index, row in data.reset_index(drop=True).iterrows():
        row_features = [row[column] for column in feature_columns]
        distance = calculate_distance(row_features, new_student, feature_columns, bounds)
        record = {
            "Student": f"Student {index + 1:02d}",
            "Index": index,
            "Result": str(row[label_column]),
            "Distance": round(distance, 4),
        }
        record.update({column: float(row[column]) for column in feature_columns})
        distances.append(record)

    return sorted(distances, key=lambda item: item["Distance"])[:k]


def vote_scores(
    neighbors: list[dict], weighting: str = "uniform", gaussian_sigma: float = 0.15
) -> dict[str, float]:
    """Return class vote scores using uniform, inverse-distance, or Gaussian weighting."""
    if not neighbors:
        raise ValueError("At least one neighbor is required to make a prediction.")
    if weighting not in {"uniform", "distance", "gaussian"}:
        raise ValueError("weighting must be 'uniform', 'distance', or 'gaussian'.")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive.")

    scores: dict[str, float] = defaultdict(float)
    for neighbor in neighbors:
        distance = float(neighbor["Distance"])
        if weighting == "uniform":
            weight = 1.0
        elif weighting == "distance":
            weight = 1.0 / max(distance, 1e-12)
        else:
            weight = float(np.exp(-(distance**2) / (2.0 * gaussian_sigma**2)))
        scores[neighbor["Result"]] += weight
    return dict(scores)


def predict_result(
    neighbors: list[dict], weighting: str = "uniform", gaussian_sigma: float = 0.15
) -> str:
    """Predict a class using the selected voting method."""
    scores = vote_scores(neighbors, weighting, gaussian_sigma)
    highest_score = max(scores.values())
    tied_labels = {label for label, score in scores.items() if np.isclose(score, highest_score)}
    for neighbor in neighbors:
        if neighbor["Result"] in tied_labels:
            return neighbor["Result"]
    raise RuntimeError("Unable to resolve the vote.")


def cross_validate_k(
    data: pd.DataFrame,
    k_values: Sequence[int],
    feature_columns: Sequence[str],
    label_column: str,
    weighting: str = "uniform",
    gaussian_sigma: float = 0.15,
) -> pd.DataFrame:
    """Evaluate candidate K values with leave-one-out cross-validation."""
    if len(data) < 3:
        raise ValueError("At least three rows are required for cross-validation.")
    if len(feature_columns) < 2:
        raise ValueError("At least two feature columns are required.")

    rows = []
    reset_data = data.reset_index(drop=True)
    for k in sorted(set(int(value) for value in k_values)):
        if k < 1 or k >= len(reset_data):
            continue
        correct = 0
        for held_out_index in range(len(reset_data)):
            training_data = reset_data.drop(index=held_out_index).reset_index(drop=True)
            held_out = reset_data.iloc[held_out_index]
            fold_bounds = feature_bounds(training_data, feature_columns)
            neighbors = get_nearest_neighbors(
                training_data,
                [held_out[column] for column in feature_columns],
                k,
                feature_columns,
                label_column,
                fold_bounds,
            )
            prediction = predict_result(neighbors, weighting, gaussian_sigma)
            correct += int(prediction == str(held_out[label_column]))
        accuracy = correct / len(reset_data)
        rows.append({"K": k, "Correct": correct, "Accuracy": accuracy})

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No valid K values were provided for cross-validation.")
    return results

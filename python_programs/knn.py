"""
K-Nearest Neighbors classifier (from scratch).
"""
from collections import Counter
import numpy as np


def run_knn(training_points, training_labels, query_point, k):
    if len(training_points) != len(training_labels):
        raise ValueError("Every training point needs exactly one label.")
    if len(training_points) == 0:
        raise ValueError("Please provide at least one training point.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if k > len(training_points):
        raise ValueError("k cannot exceed the number of training points.")

    dim = len(training_points[0])
    if any(len(p) != dim for p in training_points):
        raise ValueError("All training points must have the same number of features.")
    if len(query_point) != dim:
        raise ValueError(f"The query point must also have {dim} feature(s).")

    pts = np.array(training_points, dtype=float)
    q = np.array(query_point, dtype=float)

    distances = np.linalg.norm(pts - q, axis=1)
    order = np.argsort(distances)
    nearest_idx = order[:k]

    neighbors = [
        {
            "point": [round(float(v), 3) for v in training_points[i]],
            "label": training_labels[i],
            "distance": round(float(distances[i]), 4),
        }
        for i in nearest_idx
    ]

    votes = Counter(n["label"] for n in neighbors)
    predicted_label, top_count = votes.most_common(1)[0]

    return {
        "query_point": [round(float(v), 3) for v in query_point],
        "k": k,
        "neighbors": neighbors,
        "vote_counts": dict(votes),
        "predicted_label": predicted_label,
    }

def explain_knn(result):
    vote_summary = ", ".join(f"{count} voted {label}" for label, count in result["vote_counts"].items())
    return (
        f"To classify your new point, the program measured the straight-line distance from it to "
        f"every training point, then looked only at the {result['k']} closest ones (its 'neighbors'). "
        f"Among those neighbors: {vote_summary}. Since most neighbors belonged to "
        f"'{result['predicted_label']}', the program predicted that label for your new point — "
        f"like asking your {result['k']} nearest acquaintances what they think and going with the majority."
    )
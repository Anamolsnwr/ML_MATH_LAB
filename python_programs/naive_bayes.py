"""
Gaussian Naive Bayes Classifier (from scratch).

Assumes numeric features are normally distributed within each class.
Computes per-class mean/variance, then classifies a query point by
picking the class with the highest posterior probability.
"""
import math

import numpy as np


def _gaussian_pdf(x, mean, var):
    var = max(var, 1e-6)  # avoid division by zero for constant features
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1.0 / math.sqrt(2 * math.pi * var)) * exponent


def run_naive_bayes(X, y, query_point):
    if len(X) != len(y):
        raise ValueError("Every training row needs exactly one label.")
    if len(X) == 0:
        raise ValueError("Please provide at least one training row.")

    n_features = len(X[0])
    if any(len(row) != n_features for row in X):
        raise ValueError("All training rows must have the same number of features.")
    if len(query_point) != n_features:
        raise ValueError(f"The query point must have {n_features} feature(s).")

    X_arr = np.array(X, dtype=float)
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("Please provide training rows from at least 2 classes.")

    class_stats = {}
    for c in classes:
        rows = X_arr[[i for i, label in enumerate(y) if label == c]]
        class_stats[c] = {
            "prior": len(rows) / len(X_arr),
            "mean": rows.mean(axis=0),
            "var": rows.var(axis=0),
            "count": len(rows),
        }

    log_posteriors = {}
    for c in classes:
        stats = class_stats[c]
        log_prob = math.log(stats["prior"])
        for i, x in enumerate(query_point):
            log_prob += math.log(_gaussian_pdf(x, stats["mean"][i], stats["var"][i]))
        log_posteriors[c] = log_prob

    # Normalize log-probabilities into readable relative probabilities
    max_log = max(log_posteriors.values())
    exp_scores = {c: math.exp(v - max_log) for c, v in log_posteriors.items()}
    total = sum(exp_scores.values())
    probabilities = {c: round(v / total, 4) for c, v in exp_scores.items()}

    predicted_label = max(probabilities, key=probabilities.get)

    return {
        "classes": [
            {
                "label": c,
                "prior": round(class_stats[c]["prior"], 4),
                "mean": [round(float(m), 3) for m in class_stats[c]["mean"]],
                "variance": [round(float(v), 3) for v in class_stats[c]["var"]],
                "count": class_stats[c]["count"],
                "probability": probabilities[c],
            }
            for c in classes
        ],
        "query_point": [round(float(v), 3) for v in query_point],
        "predicted_label": predicted_label,
    }

def explain_naive_bayes(result):
    class_summaries = ", ".join(
        f"{c['label']}: {round(c['probability']*100,1)}% likely" for c in result["classes"]
    )
    return (
        f"For each class in your data, the program learned the typical average and spread of each "
        f"feature (assuming a bell-curve shape). It then checked how 'typical' your new point looks "
        f"for each class — a point very close to a class's average scores higher. "
        f"Result: {class_summaries}. Since '{result['predicted_label']}' scored highest, "
        f"that's the predicted class."
    )
"""
Logistic Regression (from scratch).
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_logistic_regression(X, y, query_point=None, learning_rate=0.1, epochs=1000):
    if len(X) != len(y):
        raise ValueError("Every training row needs exactly one label.")
    if len(X) < 2:
        raise ValueError("Please provide at least 2 training rows.")
    if any(label not in (0, 1) for label in y):
        raise ValueError("Labels must be 0 or 1 for binary classification.")

    n_features = len(X[0])
    if any(len(row) != n_features for row in X):
        raise ValueError("All training rows must have the same number of features.")
    if query_point is not None and len(query_point) != n_features:
        raise ValueError(f"The query point must have {n_features} feature(s).")

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    n_samples = X_arr.shape[0]

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        linear_output = X_arr.dot(weights) + bias
        predictions = _sigmoid(linear_output)
        error = predictions - y_arr
        dw = (1 / n_samples) * X_arr.T.dot(error)
        db = (1 / n_samples) * np.sum(error)
        weights -= learning_rate * dw
        bias -= learning_rate * db

    final_probs = _sigmoid(X_arr.dot(weights) + bias)
    final_preds = (final_probs >= 0.5).astype(int)
    accuracy = float(np.mean(final_preds == y_arr))

    result = {
        "weights": [round(float(w), 4) for w in weights],
        "bias": round(float(bias), 4),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_accuracy": round(accuracy, 4),
        "training_rows": [
            {
                "features": [round(float(v), 3) for v in X_arr[i]],
                "actual": int(y_arr[i]),
                "predicted": int(final_preds[i]),
                "probability": round(float(final_probs[i]), 4),
            }
            for i in range(n_samples)
        ],
    }

    if query_point is not None:
        q = np.array(query_point, dtype=float)
        prob = _sigmoid(q.dot(weights) + bias)
        result["query_point"] = [round(float(v), 3) for v in query_point]
        result["query_probability"] = round(float(prob), 4)
        result["query_predicted_label"] = int(prob >= 0.5)

    return result
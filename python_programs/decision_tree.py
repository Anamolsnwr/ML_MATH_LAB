"""
Decision Tree Classifier (from scratch, CART-style with Gini impurity).

Recursively splits the data on the numeric feature/threshold that most
reduces Gini impurity, until a stopping condition is reached.
"""
import numpy as np


class _Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label  # set only on leaf nodes

    @property
    def is_leaf(self):
        return self.label is not None


def _gini(labels):
    if len(labels) == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    impurity = 1.0
    n = len(labels)
    for c in counts.values():
        impurity -= (c / n) ** 2
    return impurity


def _best_split(X, y):
    n_samples, n_features = X.shape
    best_gini = float("inf")
    best_feature, best_threshold = None, None

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_labels = [y[i] for i in range(n_samples) if left_mask[i]]
            right_labels = [y[i] for i in range(n_samples) if right_mask[i]]

            weighted_gini = (
                len(left_labels) * _gini(left_labels) + len(right_labels) * _gini(right_labels)
            ) / n_samples

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gini


def _majority_label(labels):
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts, key=counts.get)


def _build_tree(X, y, depth, max_depth, min_samples_split):
    if len(set(y)) == 1:
        return _Node(label=y[0])
    if depth >= max_depth or len(y) < min_samples_split:
        return _Node(label=_majority_label(y))

    feature_index, threshold, _ = _best_split(X, y)
    if feature_index is None:
        return _Node(label=_majority_label(y))

    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask

    left_X, left_y = X[left_mask], [y[i] for i in range(len(y)) if left_mask[i]]
    right_X, right_y = X[right_mask], [y[i] for i in range(len(y)) if right_mask[i]]

    left_node = _build_tree(left_X, left_y, depth + 1, max_depth, min_samples_split)
    right_node = _build_tree(right_X, right_y, depth + 1, max_depth, min_samples_split)

    return _Node(feature_index=feature_index, threshold=threshold, left=left_node, right=right_node)


def _serialize(node):
    if node.is_leaf:
        return {"leaf": True, "label": node.label}
    return {
        "leaf": False,
        "feature_index": node.feature_index,
        "feature_name": f"x{node.feature_index}",
        "threshold": round(float(node.threshold), 3),
        "left": _serialize(node.left),
        "right": _serialize(node.right),
    }


def _predict_one(node, x):
    if node.is_leaf:
        return node.label
    if x[node.feature_index] <= node.threshold:
        return _predict_one(node.left, x)
    return _predict_one(node.right, x)


def run_decision_tree(X, y, query_point, max_depth=3, min_samples_split=2):
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
    tree = _build_tree(X_arr, list(y), depth=0, max_depth=max_depth, min_samples_split=min_samples_split)

    predicted_label = _predict_one(tree, query_point)

    predictions = [_predict_one(tree, list(row)) for row in X_arr]
    accuracy = sum(1 for p, actual in zip(predictions, y) if p == actual) / len(y)

    return {
        "tree": _serialize(tree),
        "query_point": [round(float(v), 3) for v in query_point],
        "predicted_label": predicted_label,
        "training_accuracy": round(accuracy, 4),
        "max_depth": max_depth,
    }

def explain_decision_tree(result):
    acc_pct = round(result["training_accuracy"] * 100, 1)
    return (
        f"The program built a flowchart of yes/no questions about your feature values (like "
        f"'is x0 less than 5?'). At each step, it picked the question that best separated your "
        f"labels into clean, single-class groups (measured using something called Gini impurity — "
        f"lower means purer groups). Starting at the top and following the matching answers down to "
        f"a leaf gives the prediction '{result['predicted_label']}' for your query point. "
        f"This tree correctly classifies {acc_pct}% of your training rows."
    )
"""
Linear Regression (from scratch).
Fits a straight line y = slope*x + intercept using least squares.
"""
import numpy as np


def run_linear_regression(x_values, y_values, predict_x=None):
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same number of values.")
    if len(x_values) < 2:
        raise ValueError("Please enter at least 2 data points.")

    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        raise ValueError("All x values are identical — a line cannot be fit.")

    slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    result = {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "equation": f"y = {round(float(slope), 4)}x + {round(float(intercept), 4)}",
        "r_squared": round(float(r_squared), 4),
        "points": [
            {"x": float(xi), "y": float(yi), "y_pred": round(float(pi), 3)}
            for xi, yi, pi in zip(x, y, y_pred)
        ],
    }

    if predict_x is not None:
        prediction = slope * predict_x + intercept
        result["predict_x"] = float(predict_x)
        result["predicted_y"] = round(float(prediction), 4)

    return result
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

def explain_linear_regression(result):
    """
    Turns the raw numbers from run_linear_regression() into a plain-English
    explanation a non-technical reader can understand.
    """
    slope = result["slope"]
    r2 = result["r_squared"]

    # Step 1: describe the direction and strength of the relationship
    if slope > 0:
        direction = f"as X goes up by 1, Y tends to go UP by about {abs(slope)}"
    elif slope < 0:
        direction = f"as X goes up by 1, Y tends to go DOWN by about {abs(slope)}"
    else:
        direction = "X and Y don't appear to move together at all"

    # Step 2: translate R² (0 to 1) into a plain-language quality label
    if r2 >= 0.8:
        fit_quality = "a very strong, reliable pattern"
    elif r2 >= 0.5:
        fit_quality = "a moderate pattern, with some scatter around the line"
    else:
        fit_quality = "a weak pattern — the line doesn't explain the data very well"

    explanation = (
        f"The program looked at how your X and Y numbers move together and drew "
        f"the single straight line that stays closest to all the points. It found that "
        f"{direction}. "
        f"This line explains {round(r2 * 100, 1)}% of the variation in your data, "
        f"which means {fit_quality}."
    )

    if "predicted_y" in result:
        explanation += (
            f" Using that line, it predicted y = {result['predicted_y']} "
            f"when x = {result['predict_x']}."
        )

    return explanation
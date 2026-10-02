from flask import Blueprint, render_template, request, jsonify
from python_programs.ml_algorithms import (
    run_linear_regression,
    run_kmeans,
    run_knn,
    run_logistic_regression,
)

ml_bp = Blueprint('ml', __name__)


# =========================================================================
# MACHINE LEARNING ALGORITHM PAGE ROUTES
# =========================================================================

@ml_bp.route("/linear-regression")
def linear_regression_page():
    return render_template("linear_regression.html")


@ml_bp.route("/kmeans")
def kmeans_page():
    return render_template("kmeans.html")


@ml_bp.route("/knn")
def knn_page():
    return render_template("knn.html")


@ml_bp.route("/logistic-regression")
def logistic_regression_page():
    return render_template("logistic_regression.html")


# =========================================================================
# MACHINE LEARNING ALGORITHM API ENDPOINTS
# =========================================================================

@ml_bp.route("/api/linear-regression", methods=["POST"])
def api_linear_regression():
    try:
        payload = request.get_json(force=True)
        x_values = [float(v) for v in payload["x_values"].split()]
        y_values = [float(v) for v in payload["y_values"].split()]

        predict_x = payload.get("predict_x")
        predict_x = float(predict_x) if predict_x not in (None, "") else None

        result = run_linear_regression(x_values, y_values, predict_x)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@ml_bp.route("/api/kmeans", methods=["POST"])
def api_kmeans():
    try:
        payload = request.get_json(force=True)
        k = int(payload["k"])

        points = []
        for line in payload["points"].strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [float(v) for v in line.replace(",", " ").split()]
            if len(parts) != 2:
                raise ValueError("Each point needs exactly 2 numbers (x y).")
            points.append(parts)

        if len(points) == 0:
            raise ValueError("Please enter at least one point.")

        result = run_kmeans(points, k)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@ml_bp.route("/api/knn", methods=["POST"])
def api_knn():
    try:
        payload = request.get_json(force=True)
        k = int(payload["k"])

        training_points = []
        training_labels = []
        for line in payload["training_data"].strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(
                    "Each training row needs one or more numbers followed by a label."
                )
            *features, label = parts
            training_points.append([float(v) for v in features])
            training_labels.append(label)

        if len(training_points) == 0:
            raise ValueError("Please enter at least one training row.")

        query_point = [float(v) for v in payload["query_point"].replace(",", " ").split()]

        result = run_knn(training_points, training_labels, query_point, k)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@ml_bp.route("/api/logistic-regression", methods=["POST"])
def api_logistic_regression():
    try:
        payload = request.get_json(force=True)

        X = []
        y = []
        for line in payload["training_data"].strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(
                    "Each training row needs one or more numbers followed by a 0/1 label."
                )
            *features, label = parts
            X.append([float(v) for v in features])
            y.append(int(float(label)))

        if len(X) == 0:
            raise ValueError("Please enter at least one training row.")

        query_raw = payload.get("query_point", "").strip()
        query_point = (
            [float(v) for v in query_raw.replace(",", " ").split()] if query_raw else None
        )

        result = run_logistic_regression(X, y, query_point)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

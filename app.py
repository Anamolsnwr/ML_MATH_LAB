from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import statistics
from python_programs.identity_matrix import is_identity_matrix, explain_identity_matrix
from python_programs.vector import compute_vectors, explain_vector_operations
from python_programs.probability import calculate_probability, explain_probability
from python_programs.random import generate_random_password, explain_random_password
from python_programs.statistics import calculate_statistics, explain_statistics
from python_programs.linear_regression import run_linear_regression
from python_programs.kmeans import run_kmeans
from python_programs.knn import run_knn
from python_programs.logistic_regression import run_logistic_regression
from python_programs.naive_bayes import run_naive_bayes
from python_programs.pca import run_pca
from python_programs.decision_tree import run_decision_tree
app = Flask(__name__)

# --- PAGE ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/identity')
def identity_page():
    return render_template('identity.html')

@app.route('/probability')
def probability_page():
    return render_template('probability.html')

@app.route('/random')
def random_page():
    return render_template('random.html')

@app.route('/vector')
def vector_page():
    return render_template('vector.html')

@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')

@app.route("/linear-regression")
def linear_regression_page():
    return render_template("linear_regression.html")


@app.route("/kmeans")
def kmeans_page():
    return render_template("kmeans.html")


@app.route("/knn")
def knn_page():
    return render_template("knn.html")


@app.route("/logistic-regression")
def logistic_regression_page():
    return render_template("logistic_regression.html")

@app.route("/naive-bayes")
def naive_bayes_page():
    return render_template("naive_bayes.html")


@app.route("/pca")
def pca_page():
    return render_template("pca.html")


@app.route("/decision-tree")
def decision_tree_page():
    return render_template("decision_tree.html")
# --- API EXECUTION ENDPOINTS ---

@app.route('/api/identity', methods=['POST'])
def api_identity():
    data = request.get_json()
    size = data.get('size', 3)
    if size < 1 or size > 15:
        return jsonify({'status': 'error', 'message': 'Size must be between 1 and 15.'})
    
    matrix = np.eye(size, dtype=int)
    formatted_matrix = "\n".join(["[" + " ".join(f"{num:3d}" for num in row) + "]" for row in matrix])
    output = f"[SYS_EXEC] Identity Matrix ({size}x{size}):\n\n{formatted_matrix}"

    # <-- ADDED: plain-English explanation
    explanation = (
        f"An identity matrix has 1s running diagonally from the top-left to the "
        f"bottom-right corner and 0s everywhere else. It is the 'do nothing' matrix: "
        f"multiplying any matrix by it leaves that matrix unchanged, just like "
        f"multiplying a number by 1. The program built a {size}x{size} one, so it "
        f"has {size} ones on the diagonal and {size * size - size} zeros elsewhere."
    )
    return jsonify({'status': 'success', 'output': output, 'explanation': explanation})  # <-- CHANGED: added explanation


@app.route('/api/probability', methods=['POST'])
def api_probability():
    data = request.get_json()
    favorable = data.get('favorable', 0)
    total = data.get('total', 1)
    
    if total <= 0:
        return jsonify({'status': 'error', 'message': 'Total outcomes must be greater than 0.'})
    if favorable > total:
        return jsonify({'status': 'error', 'message': 'Favorable outcomes cannot exceed total outcomes.'})
    
    prob = favorable / total
    percentage = prob * 100
    odds = f"{favorable} : {total - favorable}"
    
    output = (f"[BAYESIAN_ENGINE] Probability Output:\n"
              f"  • Probability P(A) : {prob:.4f}\n"
              f"  • Likelihood Ratio : {percentage:.2f}%\n"
              f"  • Odds Ratio (A:A'): {odds}")

    # <-- ADDED: plain-English explanation
    if percentage >= 75:
        odds_feel = "very likely to happen"
    elif percentage >= 50:
        odds_feel = "more likely than not to happen"
    elif percentage >= 25:
        odds_feel = "possible, but more likely not to happen"
    else:
        odds_feel = "unlikely to happen"

    explanation = (
        f"Probability is (ways the thing you want can happen) divided by "
        f"(all the things that could possibly happen). Out of {total} possible "
        f"outcomes, {favorable} count as a 'win', so the probability is "
        f"{prob:.4f}, or {percentage:.2f}%. This event is {odds_feel}. "
        f"The odds {odds} compare wins to non-wins directly: for every "
        f"{favorable} win(s) there are {total - favorable} non-win(s)."
    )
    return jsonify({'status': 'success', 'output': output, 'explanation': explanation})  # <-- CHANGED: added explanation


@app.route('/api/random', methods=['POST'])
def api_random():
    data = request.get_json()
    count = data.get('count', 10)
    min_val = data.get('min', 1)
    max_val = data.get('max', 100)
    
    if min_val >= max_val:
        return jsonify({'status': 'error', 'message': 'Min boundary must be strictly less than Max boundary.'})
    
    samples = [round(random.uniform(min_val, max_val), 2) for _ in range(count)]
    output = f"[STOCHASTIC_ENGINE] Generated {count} Random Samples:\n\n" + str(samples)

    # <-- ADDED: plain-English explanation
    explanation = (
        f"The program picked {count} numbers between {min_val} and {max_val}. "
        f"Each number is chosen independently, and every value in that range is "
        f"equally likely (this is called a uniform distribution). "
    )
    if samples:
        avg = round(sum(samples) / len(samples), 2)
        middle = (min_val + max_val) / 2
        explanation += (
            f"The average of this batch is {avg}, while the exact middle of your "
            f"range is {middle}. With only a few samples the average can drift "
            f"away from the middle; with many samples it gets closer and closer."
        )
    return jsonify({'status': 'success', 'output': output, 'explanation': explanation})  # <-- CHANGED: added explanation


@app.route('/api/vector', methods=['POST'])
def api_vector():
    data = request.get_json()
    try:
        vec_a = np.array([float(x.strip()) for x in data.get('vector_a', '').split(',')])
        vec_b = np.array([float(x.strip()) for x in data.get('vector_b', '').split(',')])
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid vector format. Use numbers separated by commas.'})
    
    if vec_a.shape != vec_b.shape:
        return jsonify({'status': 'error', 'message': f'Dimension mismatch! Vector A has {len(vec_a)} dims, Vector B has {len(vec_b)} dims.'})
    
    addition = vec_a + vec_b
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    cosine_sim = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    
    output = (f"[VECTOR_SPACE_NODE] Calculations:\n"
              f"  • Vector A + B : {addition.tolist()}\n"
              f"  • Dot Product  : {dot_product}\n"
              f"  • Magnitude ||A|| : {norm_a:.4f}\n"
              f"  • Magnitude ||B|| : {norm_b:.4f}\n"
              f"  • Cosine Similarity: {cosine_sim:.4f}")

    # <-- ADDED: plain-English explanation
    if cosine_sim > 0.9:
        direction_feel = "pointing in almost exactly the same direction"
    elif cosine_sim > 0.1:
        direction_feel = "pointing in broadly similar directions"
    elif cosine_sim >= -0.1:
        direction_feel = "pointing at right angles (perpendicular) to each other"
    elif cosine_sim >= -0.9:
        direction_feel = "pointing in broadly opposite directions"
    else:
        direction_feel = "pointing in almost exactly opposite directions"

    explanation = (
        f"Adding vectors combines them coordinate by coordinate. The dot product "
        f"({round(float(dot_product), 3)}) multiplies matching coordinates and adds "
        f"them up. The magnitude is a vector's length "
        f"({round(float(norm_a), 3)} for A, {round(float(norm_b), 3)} for B). "
        f"Cosine similarity ({round(float(cosine_sim), 3)}) divides the dot product by "
        f"both lengths, so it ignores size and only measures direction: 1 means the "
        f"same direction, 0 means perpendicular, -1 means opposite. "
        f"Here the vectors are {direction_feel}."
    )
    return jsonify({'status': 'success', 'output': output, 'explanation': explanation})  # <-- CHANGED: added explanation


@app.route('/api/statistics', methods=['POST'])
def api_statistics():
    data = request.get_json()
    try:
        raw_dataset = [float(x.strip()) for x in data.get('dataset', '').split(',')]
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid dataset format. Enter numbers separated by commas.'})
    
    if len(raw_dataset) == 0:
        return jsonify({'status': 'error', 'message': 'Dataset cannot be empty.'})
    
    mean_val = np.mean(raw_dataset)
    median_val = np.median(raw_dataset)
    variance_val = np.var(raw_dataset, ddof=1) if len(raw_dataset) > 1 else 0
    std_val = np.std(raw_dataset, ddof=1) if len(raw_dataset) > 1 else 0
    
    output = (f"[STATISTICAL_METRICS] Feature Dataset Analysis:\n"
              f"  • Sample Count (N): {len(raw_dataset)}\n"
              f"  • Mean (μ)        : {mean_val:.4f}\n"
              f"  • Median          : {median_val:.4f}\n"
              f"  • Sample Variance : {variance_val:.4f}\n"
              f"  • Std Deviation (σ): {std_val:.4f}")

    # <-- ADDED: plain-English explanation
    if len(raw_dataset) > 1:
        spread_hint = (
            f"The standard deviation ({std_val:.4f}) tells you how far, on average, "
            f"each number strays from the mean: small means tightly clustered, "
            f"large means spread out. The variance ({variance_val:.4f}) is just the "
            f"standard deviation squared."
        )
    else:
        spread_hint = "There is only one number, so spread cannot be measured."

    if abs(mean_val - median_val) < 0.01:
        shape_hint = "The mean and median are nearly equal, so your data looks fairly symmetric."
    elif mean_val > median_val:
        shape_hint = "The mean is higher than the median, so a few unusually large values are pulling the average up."
    else:
        shape_hint = "The mean is lower than the median, so a few unusually small values are pulling the average down."

    explanation = (
        f"The mean ({mean_val:.4f}) is the regular average: add everything up and "
        f"divide by how many numbers there are ({len(raw_dataset)}). The median "
        f"({median_val:.4f}) is the middle value once the numbers are sorted, and "
        f"it is less affected by extreme values. {spread_hint} {shape_hint}"
    )
    return jsonify({'status': 'success', 'output': output, 'explanation': explanation})  # <-- CHANGED: added explanation


from python_programs.linear_regression import run_linear_regression, explain_linear_regression

@app.route("/api/linear-regression", methods=["POST"])
def api_linear_regression():
    try:
        payload = request.get_json(force=True)
        x_values = [float(v) for v in payload["x_values"].split()]
        y_values = [float(v) for v in payload["y_values"].split()]

        predict_x = payload.get("predict_x")
        predict_x = float(predict_x) if predict_x not in (None, "") else None

        result = run_linear_regression(x_values, y_values, predict_x)
        result["explanation"] = explain_linear_regression(result)   # <-- new line
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

from python_programs.kmeans import run_kmeans, explain_kmeans
@app.route("/api/kmeans", methods=["POST"])
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
        result["explanation"] = explain_kmeans(result) 
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

from python_programs.knn import run_knn, explain_knn
@app.route("/api/knn", methods=["POST"])
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
        result["explanation"] = explain_knn(result)  
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

from python_programs.logistic_regression import run_logistic_regression, explain_logistic_regression
@app.route("/api/logistic-regression", methods=["POST"])
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
        result["explanation"] = explain_logistic_regression(result) 
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

from python_programs.naive_bayes import run_naive_bayes, explain_naive_bayes
@app.route("/api/naive-bayes", methods=["POST"])
def api_naive_bayes():
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
                    "Each training row needs one or more numbers followed by a label."
                )
            *features, label = parts
            X.append([float(v) for v in features])
            y.append(label)

        if len(X) == 0:
            raise ValueError("Please enter at least one training row.")

        query_point = [float(v) for v in payload["query_point"].replace(",", " ").split()]

        result = run_naive_bayes(X, y, query_point)
        result["explanation"] = explain_naive_bayes(result)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

from python_programs.pca import run_pca, explain_pca
@app.route("/api/pca", methods=["POST"])
def api_pca():
    try:
        payload = request.get_json(force=True)
        n_components = int(payload.get("n_components", 2))

        points = []
        for line in payload["points"].strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [float(v) for v in line.replace(",", " ").split()]
            points.append(parts)

        if len(points) == 0:
            raise ValueError("Please enter at least one point.")

        result = run_pca(points, n_components)
        result["explanation"] = explain_pca(result)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

from python_programs.decision_tree import run_decision_tree, explain_decision_tree
@app.route("/api/decision-tree", methods=["POST"])
def api_decision_tree():
    try:
        payload = request.get_json(force=True)
        max_depth = int(payload.get("max_depth", 3))

        X = []
        y = []
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
            X.append([float(v) for v in features])
            y.append(label)

        if len(X) == 0:
            raise ValueError("Please enter at least one training row.")

        query_point = [float(v) for v in payload["query_point"].replace(",", " ").split()]

        result = run_decision_tree(X, y, query_point, max_depth=max_depth)
        result["explanation"] = explain_decision_tree(result)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)   # <-- FIXED: removed the duplicate second app.run line
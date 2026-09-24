from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import statistics

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
    return jsonify({'status': 'success', 'output': output})


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
    return jsonify({'status': 'success', 'output': output})


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
    return jsonify({'status': 'success', 'output': output})


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
    return jsonify({'status': 'success', 'output': output})


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
    return jsonify({'status': 'success', 'output': output})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
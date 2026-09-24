import numpy as np

def compute_vectors(vector_a_str, vector_b_str):
    clean_a = str(vector_a_str).replace(',', ' ').split()
    clean_b = str(vector_b_str).replace(',', ' ').split()
    
    if not clean_a or not clean_b:
        raise ValueError("Please provide non-empty values for both vectors.")

    try:
        A = np.array([float(x) for x in clean_a])
        B = np.array([float(x) for x in clean_b])
    except ValueError:
        raise ValueError("Vectors can only contain numbers.")

    if len(A) != len(B):
        raise ValueError(f"Dimensions mismatch: Vector A has {len(A)} elements, Vector B has {len(B)}.")

    norm_a = float(np.linalg.norm(A))
    norm_b = float(np.linalg.norm(B))

    results = {
        "vector_a": A.tolist(),
        "vector_b": B.tolist(),
        "addition": (A + B).tolist(),
        "subtraction": (A - B).tolist(),
        "dot_product": float(np.dot(A, B)),
        "magnitude_a": round(norm_a, 4),
        "magnitude_b": round(norm_b, 4),
        "unit_vector_a": (A / norm_a).round(4).tolist() if norm_a != 0 else "Undefined",
        "unit_vector_b": (B / norm_b).round(4).tolist() if norm_b != 0 else "Undefined"
    }

    if len(A) == 3:
        results["cross_product"] = np.cross(A, B).tolist()

    return results
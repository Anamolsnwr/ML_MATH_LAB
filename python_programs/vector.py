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

def explain_vector_operations(result):
    dot = result["dot_product"]
    if dot > 0:
        angle_hint = "pointing in a broadly similar direction (angle under 90°)"
    elif dot < 0:
        angle_hint = "pointing in broadly opposite directions (angle over 90°)"
    else:
        angle_hint = "pointing exactly perpendicular to each other (90° apart)"

    explanation = (
        f"Addition and subtraction just combine the vectors coordinate by coordinate. "
        f"The dot product ({round(dot, 3)}) multiplies matching coordinates and adds "
        f"them up — its sign tells you the vectors are {angle_hint}. "
        f"The magnitude is each vector's length, found with the Pythagorean theorem "
        f"extended to more dimensions ({round(result['magnitude_a'], 3)} for A, "
        f"{round(result['magnitude_b'], 3)} for B). "
        f"The unit vectors are the same directions scaled down to exactly length 1."
    )
    if result.get("cross_product") is not None:
        explanation += (
            " The cross product gives a new vector perpendicular to both A and B — "
            "useful for finding a direction 'sideways' to a flat plane."
        )
    return explanation
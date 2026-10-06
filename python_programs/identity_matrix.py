import numpy as np

def generate_identity_matrix(size):
    try:
        size = int(size)
    except (ValueError, TypeError):
        raise ValueError("Matrix size must be a valid integer.")

    if size < 1 or size > 15:
        raise ValueError("Matrix order size must be between 1 and 15.")
    
    matrix = np.eye(size, dtype=int).tolist()
    return matrix

def is_identity_matrix(matrix, n):
    for i in range(n):
        for j in range(n):
            if i == j and matrix[i][j] != 1:
                return False
            if i != j and matrix[i][j] != 0:
                return False
    return True

def explain_identity_matrix(result):
    n = result["order"]
    if result["is_identity"]:
        explanation = (
            f"An identity matrix has 1s running diagonally from the top-left to the "
            f"bottom-right corner, and 0s everywhere else — think of it as the "
            f"'do nothing' matrix, the number 1 of matrix math. "
            f"The program checked every one of the {n * n} positions in your {n}×{n} "
            f"matrix and confirmed that pattern holds exactly, so it is an identity matrix."
        )
    else:
        explanation = (
            f"An identity matrix needs 1s on the diagonal (top-left to bottom-right) "
            f"and 0s everywhere else. The program checked all {n * n} positions in your "
            f"{n}×{n} matrix and found at least one position that breaks that rule, "
            f"so it is not an identity matrix."
        )
    return explanation
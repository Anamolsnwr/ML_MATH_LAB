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
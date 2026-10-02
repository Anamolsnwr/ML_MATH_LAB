"""
Mathematical Calculation Modules Package
"""
from .identity_matrix import generate_identity_matrix, is_identity_matrix
from .probability import calculate_probability
from .random_gen import generate_random_password
from .statistics_calc import calculate_statistics
from .vector import compute_vectors

__all__ = [
    "generate_identity_matrix",
    "is_identity_matrix",
    "calculate_probability",
    "generate_random_password",
    "calculate_statistics",
    "compute_vectors",
]

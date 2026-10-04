"""
Principal Component Analysis (from scratch).

Centers the data, computes the covariance matrix, finds its eigenvectors
(the principal components), and projects the data onto the top components.
"""
import numpy as np


def run_pca(points, n_components=2):
    pts = np.array(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("Points must be a 2D list of numbers.")

    n_samples, n_features = pts.shape
    if n_samples < 2:
        raise ValueError("Please provide at least 2 data points.")
    if n_components < 1 or n_components > n_features:
        raise ValueError(f"Number of components must be between 1 and {n_features}.")

    mean = pts.mean(axis=0)
    centered = pts - mean

    cov_matrix = np.atleast_2d(np.cov(centered, rowvar=False))

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # eigh returns ascending order — flip to descending (largest variance first)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = (
        eigenvalues / total_variance if total_variance != 0 else np.zeros_like(eigenvalues)
    )

    top_components = eigenvectors[:, :n_components]
    projected = centered.dot(top_components)

    return {
        "mean": [round(float(m), 3) for m in mean],
        "n_components": n_components,
        "eigenvalues": [round(float(v), 4) for v in eigenvalues],
        "explained_variance_ratio": [round(float(v), 4) for v in explained_variance_ratio],
        "components": [[round(float(v), 4) for v in comp] for comp in top_components.T],
        "original_points": [[round(float(v), 3) for v in p] for p in pts],
        "projected_points": [[round(float(v), 3) for v in p] for p in projected],
    }
"""
K-Means Clustering (from scratch).
"""
import numpy as np


def run_kmeans(points, k, max_iters=100, seed=42):
    pts = np.array(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Each point must have exactly 2 coordinates (x, y).")
    if k < 1:
        raise ValueError("Number of clusters (k) must be at least 1.")
    if k > len(pts):
        raise ValueError("Number of clusters (k) cannot exceed the number of points.")

    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(len(pts), size=k, replace=False)
    centroids = pts[initial_idx].copy()

    labels = np.zeros(len(pts), dtype=int)
    iterations_used = max_iters

    for iteration in range(1, max_iters + 1):
        distances = np.linalg.norm(pts[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centroids = centroids.copy()
        for i in range(k):
            members = pts[labels == i]
            if len(members) > 0:
                new_centroids[i] = members.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            iterations_used = iteration
            break
        centroids = new_centroids
    else:
        iterations_used = max_iters

    clusters = []
    for i in range(k):
        members = pts[labels == i].tolist()
        clusters.append({
            "cluster_id": i,
            "centroid": [round(float(c), 3) for c in centroids[i]],
            "points": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in members],
            "size": int(len(members)),
        })

    return {
        "k": k,
        "iterations": int(iterations_used),
        "assignments": [int(l) for l in labels],
        "points": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts],
        "clusters": clusters,
    }
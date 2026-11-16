# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Small linear-algebra helpers shared by the GMAT-UQ engines."""

from __future__ import annotations

import numpy as np

__all__ = [
    "symmetrize",
    "check_covariance",
    "covariance_factor",
    "perpendicular_basis",
    "velocity_input_matrix",
    "whitening_matrix",
]


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def check_covariance(matrix, size: int, name: str = "covariance") -> np.ndarray:
    """Validate a covariance matrix and return a symmetric float copy.

    The matrix must be ``size`` x ``size``, finite, symmetric to rounding and
    positive semi-definite to rounding.
    """
    cov = np.array(matrix, dtype=float)
    if cov.shape != (size, size):
        raise ValueError(f"{name} must be a {size}x{size} matrix, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} contains non-finite values")
    scale = max(float(np.abs(cov).max()), np.finfo(float).tiny)
    if np.abs(cov - cov.T).max() > 1e-9 * scale:
        raise ValueError(f"{name} is not symmetric")
    cov = symmetrize(cov)
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues.min() < -1e-9 * max(eigenvalues.max(), np.finfo(float).tiny):
        raise ValueError(f"{name} is not positive semi-definite")
    return cov


def covariance_factor(cov: np.ndarray) -> np.ndarray:
    """Return a square factor ``L`` with ``L @ L.T == cov``.

    A Cholesky factor is used when the matrix is positive definite, because it
    is unique and therefore reproducible across platforms.  Semi-definite
    matrices fall back to an eigen-decomposition whose eigenvector signs are
    normalised so that the factor is still deterministic.
    """
    cov = symmetrize(cov)
    if not cov.any():
        return np.zeros_like(cov)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        pass
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # Deterministic sign: the largest-magnitude component of each eigenvector is positive.
    pivot = np.abs(eigenvectors).argmax(axis=0)
    signs = np.sign(eigenvectors[pivot, np.arange(eigenvectors.shape[1])])
    signs[signs == 0.0] = 1.0
    return (eigenvectors * signs) * np.sqrt(eigenvalues)


def perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors that complete ``direction`` to a right-handed triad.

    ``direction`` holds unit vectors in its last axis (shape ``(..., 3)``).  The
    construction is deterministic so that sampled pointing errors are
    reproducible.
    """
    direction = np.asarray(direction, dtype=float)
    helper = np.zeros_like(direction)
    use_x = np.abs(direction[..., 0]) < 0.9
    helper[..., 0] = np.where(use_x, 1.0, 0.0)
    helper[..., 1] = np.where(use_x, 0.0, 1.0)
    first = np.cross(direction, helper)
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    second = np.cross(direction, first)
    return first, second


def velocity_input_matrix() -> np.ndarray:
    """Return the 6x3 matrix that maps a velocity change into the orbit state."""
    matrix = np.zeros((6, 3))
    matrix[3:, :] = np.eye(3)
    return matrix


def whitening_matrix(cov: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """Return ``W`` such that ``W @ cov @ W.T`` is the identity on the range of ``cov``.

    Directions whose variance is below ``rtol`` times the largest variance are
    dropped, so ``W`` has one row per significant direction.
    """
    cov = symmetrize(cov)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    largest = eigenvalues.max() if eigenvalues.size else 0.0
    if largest <= 0.0:
        return np.zeros((0, cov.shape[0]))
    keep = eigenvalues > rtol * largest
    return (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])).T

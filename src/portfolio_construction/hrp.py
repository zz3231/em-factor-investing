"""Hierarchical Risk Parity (HRP) following Lopez de Prado (2016)."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))."""
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diagonalise(link: np.ndarray) -> list[int]:
    """Reorder assets so that similar assets are adjacent (quasi-diagonalisation).

    Returns the ordered list of original asset indices.
    """
    return list(leaves_list(link))


def _recursive_bisection(
    cov: np.ndarray,
    sorted_indices: list[int],
) -> np.ndarray:
    """Allocate weights by recursive bisection on the quasi-diagonalised order."""
    n = len(sorted_indices)
    weights = np.ones(n)
    cluster_items = [list(range(n))]

    while cluster_items:
        next_round = []
        for subset in cluster_items:
            if len(subset) <= 1:
                continue
            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            left_idx = [sorted_indices[i] for i in left]
            right_idx = [sorted_indices[i] for i in right]

            cov_left = cov[np.ix_(left_idx, left_idx)]
            cov_right = cov[np.ix_(right_idx, right_idx)]

            inv_var_left = 1.0 / np.diag(cov_left)
            w_left = inv_var_left / inv_var_left.sum()
            var_left = w_left @ cov_left @ w_left

            inv_var_right = 1.0 / np.diag(cov_right)
            w_right = inv_var_right / inv_var_right.sum()
            var_right = w_right @ cov_right @ w_right

            alpha = 1.0 - var_left / (var_left + var_right)

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1.0 - alpha)

            if len(left) > 1:
                next_round.append(left)
            if len(right) > 1:
                next_round.append(right)

        cluster_items = next_round

    return weights


def compute_weights(
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Steps:
        1. Compute distance matrix from correlation.
        2. Single-linkage agglomerative clustering.
        3. Quasi-diagonalisation of the covariance matrix.
        4. Recursive bisection to allocate weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix. If None, estimated from *returns_df*.
    mean_returns : np.ndarray, optional
        N-vector of expected returns (unused).
    bounds : tuple[float, float], optional
        Weight bounds (unused -- HRP is unconstrained).

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    if cov_matrix is None:
        cov_matrix = returns_df.cov().values

    corr = returns_df.corr().values
    dist = _correlation_distance(corr)

    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method='single')

    sorted_indices = _quasi_diagonalise(link)

    raw_weights = _recursive_bisection(cov_matrix, sorted_indices)

    # Map weights back to original asset order
    weights = np.zeros(len(sorted_indices))
    for position, original_idx in enumerate(sorted_indices):
        weights[original_idx] = raw_weights[position]

    weights /= weights.sum()
    return weights

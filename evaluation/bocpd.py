"""BARO's onset detector (multivariate Bayesian online change-point detection), for the full BARO pipeline.

BARO [Pham et al., FSE 2024] detects the anomaly onset with multivariate BOCPD and then ranks metrics with its
RobustScorer on the data before and after the first change point. This module re-implements the detector from the
BARO code (``baro/anomaly_detection.py: bocpd`` and ``baro/_bocpd.py``):

* input: the case's latency and error columns (names containing ``latency`` or ``_error``, except
  ``queue-master`` and ``rabbitmq_``; all columns if there are none), constant columns dropped, missing values
  forward-filled then set to 0, each column min-max normalised to [0, 1];
* model: multivariate Student-t posterior predictive (Gaussian likelihood with a Normal-Wishart prior: mean 0,
  identity scale, kappa 1, dof = dimensions + 1), constant hazard 1/50;
* run-length recursion in probability space with renormalisation at every step, ``maxes[t]`` = the most probable
  run length of column t;
* change points: every i at which ``maxes`` jumps by more than 1; the first one splits the data.

Two implementation details differ from the BARO code and do not change the model: the predictive density is
evaluated for all run lengths at once, and the Wishart scale of every run length is updated with the
Sherman-Morrison identity (and its log-determinant with the matrix-determinant lemma) instead of two matrix
inversions per step. :func:`reference_bocpd` is a direct transcription of the BARO recursion (explicit inverses,
``scipy.stats.multivariate_t``) used by the tests to check this implementation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

HAZARD_LAMBDA = 50


def select_columns(columns):
    """BARO's selection of latency and error metrics (all columns if none qualify)."""
    sel = [j for j, c in enumerate(columns)
           if not ("queue-master" in c or "rabbitmq_" in c) and ("latency" in c or "latency-50" in c or "_error" in c)]
    return sel if sel else list(range(len(columns)))


def prepare(columns, X: np.ndarray) -> np.ndarray:
    """Selected columns, constant columns dropped, ffill/0, min-max normalised (BARO ``bocpd`` preprocessing)."""
    Y = X[:, select_columns(columns)].astype(float)
    if Y.shape[0] == 0 or Y.shape[1] == 0:
        return Y
    Y = Y.copy()
    for j in range(Y.shape[1]):          # forward fill, then 0
        col = Y[:, j]
        mask = np.isnan(col)
        if mask.any():
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
            col[np.isnan(col)] = 0.0
            Y[:, j] = col
    keep = np.any(Y != Y[0:1, :], axis=0)
    Y = Y[:, keep]
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    rng = hi - lo
    Y = np.where(rng > 0, (Y - lo) / np.where(rng > 0, rng, 1.0), 0.0)
    return Y


def online_maxes(Y: np.ndarray, lam: float = HAZARD_LAMBDA) -> np.ndarray:
    """``maxes`` of BARO's ``online_changepoint_detection`` with ``MultivariateT(dims)`` and constant hazard."""
    T, D = Y.shape
    maxes = np.zeros(T + 1)
    if T == 0 or D == 0:
        return maxes
    H = 1.0 / lam
    R = np.zeros(T + 1)
    R[0] = 1.0
    mu = np.zeros((1, D))
    W = np.eye(D)[None, :, :]
    logdet = np.zeros(1)
    kappa = np.ones(1)
    dof = np.full(1, float(D + 1))
    half_d = D / 2.0
    for t in range(T):
        x = Y[t]
        nu = dof - D + 1.0
        c = kappa * nu / (kappa + 1.0)
        u = x[None, :] - mu
        Wu = np.einsum("rij,rj->ri", W, u)
        q = np.einsum("ri,ri->r", u, Wu)
        logp = (gammaln((nu + D) / 2.0) - gammaln(nu / 2.0) - half_d * np.log(nu * math.pi)
                + 0.5 * (D * np.log(c) + logdet) - ((nu + D) / 2.0) * np.log1p(c * q / nu))
        pred = np.exp(logp)
        new = np.zeros(T + 1)
        new[1:t + 2] = R[0:t + 1] * pred * (1 - H)
        new[0] = np.sum(R[0:t + 1] * pred * H)
        with np.errstate(invalid="ignore", divide="ignore"):
            new = new / np.sum(new)
        # parameter update for every run length, the prior prepended
        b = kappa / (kappa + 1.0)
        denom = 1.0 + b * q
        W_new = W - (b / denom)[:, None, None] * np.einsum("ri,rj->rij", Wu, Wu)
        logdet_new = logdet - np.log(denom)
        mu_new = (kappa[:, None] * mu + x[None, :]) / (kappa[:, None] + 1.0)
        W = np.concatenate([np.eye(D)[None, :, :], W_new])
        logdet = np.concatenate([[0.0], logdet_new])
        mu = np.concatenate([np.zeros((1, D)), mu_new])
        kappa = np.concatenate([[1.0], kappa + 1.0])
        dof = np.concatenate([[float(D + 1)], dof + 1.0])
        maxes[t] = np.argmax(R)
        R = new
    return maxes


def find_cps(maxes) -> list:
    return [(i, abs(maxes[i] - maxes[i - 1])) for i in range(1, len(maxes)) if abs(maxes[i] - maxes[i - 1]) > 1]


def first_change_point(columns, X: np.ndarray):
    """Index of BARO's first change point in the rows of ``X`` (None if there is none)."""
    Y = prepare(columns, X)
    if Y.shape[1] == 0:
        return None
    cps = find_cps(online_maxes(Y))
    return int(cps[0][0]) if cps else None


def reference_bocpd(Y: np.ndarray, lam: float = HAZARD_LAMBDA) -> np.ndarray:
    """Direct transcription of BARO's ``online_changepoint_detection`` with ``MultivariateT`` (slow; tests only)."""
    import scipy.stats as ss
    from numpy.linalg import inv
    T, D = Y.shape
    dof, kappa = np.array([D + 1]), np.array([1])
    mu, scale = np.array([[0.0] * D]), np.array([np.identity(D)])
    maxes = np.zeros(T + 1)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1
    for t, x in enumerate(Y):
        t_dof = dof - D + 1
        expanded = np.expand_dims((kappa * t_dof) / (kappa + 1), (1, 2))
        pred = np.empty(t + 1)
        for i, (df, loc, shape) in enumerate(zip(t_dof, mu, inv(expanded * scale))):
            pred[i] = ss.multivariate_t.pdf(x=x, df=df, loc=loc, shape=shape)
        H = 1 / lam * np.ones(t + 1)
        R[1:t + 2, t + 1] = R[0:t + 1, t] * pred * (1 - H)
        R[0, t + 1] = np.sum(R[0:t + 1, t] * pred * H)
        R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])
        centered = x - mu
        scale = np.concatenate([scale[:1], inv(inv(scale) + np.expand_dims(kappa / (kappa + 1), (1, 2))
                                               * (np.expand_dims(centered, 2) @ np.expand_dims(centered, 1)))])
        mu = np.concatenate([mu[:1], (np.expand_dims(kappa, 1) * mu + x) / np.expand_dims(kappa + 1, 1)])
        dof = np.concatenate([dof[:1], dof + 1])
        kappa = np.concatenate([kappa[:1], kappa + 1])
        maxes[t] = R[:, t].argmax()
    return maxes

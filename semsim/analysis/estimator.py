"""
semsim.analysis.estimator
=========================
The fitting engine: it takes a dataset plus the :class:`ModelSpec` that
describes the model and returns maximum-likelihood (or ULS / DWLS) estimates of
every free parameter, with standard errors, Wald tests and a fully
standardised solution.

Identification
--------------
The **marker-variable** scheme is used throughout, because it is the most
robust one for an automatically generated model:

* the *first* loading of every factor is fixed to ``1.0``;
* every latent variance and every disturbance variance is free;
* every other loading, cross-loading, method-factor loading and structural
  path is free;
* for a second-order construct the second-order loading of its *first*
  dimension is fixed to ``1.0`` in the same way.

The solution is then standardised *post hoc* for reporting::

    Phi    = (I - B)^-1 Psi (I - B)^-T
    Sigma  = Lambda Phi Lambda' + Theta
    lambda*_jk = lambda_jk * sqrt(Phi_kk) / sqrt(Sigma_jj)
    beta*_ij   = beta_ij   * sqrt(Phi_jj) / sqrt(Phi_ii)

Both metrics are reported: ``est`` carries the unstandardised estimate and
``std`` / :attr:`EstimationResult.standardised` the standardised one.

Latent interactions cannot be expressed in a covariance structure, so they are
omitted from the fitted model and named in
:attr:`EstimationResult.message`.

References
----------
Joreskog, K. G. (1969). A general approach to confirmatory maximum likelihood
    factor analysis. *Psychometrika*, 34(2), 183-202.
Joreskog, K. G. (1970). A general method for analysis of covariance
    structures. *Biometrika*, 57(2), 239-251.
Browne, M. W. (1984). Asymptotically distribution-free methods for the
    analysis of covariance structures. *British Journal of Mathematical and
    Statistical Psychology*, 37(1), 62-83.
Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
    Applications in Statistics and Econometrics* (rev. ed.). Wiley.
Rosseel, Y. (2012). lavaan: An R package for structural equation modeling.
    *Journal of Statistical Software*, 48(2), 1-36.
"""

from __future__ import annotations

import warnings as _warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Dict, Iterator, List, Optional,
                    Sequence, Tuple)

import numpy as np
import pandas as pd
from scipy import linalg as _sla
from scipy import optimize as _opt
from scipy.stats import norm as _norm

from semsim.core.compile import compile_model
from semsim.core.errors import (ConvergenceError, IdentificationError,
                                SpecificationError)
from semsim.core.spec import EstimatorKind, ModelSpec, ScaleKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semsim.core.compile import CompiledModel


#: Lower bound imposed on every variance parameter, so the optimiser cannot
#: wander into an improper (negative-variance) solution.
VARIANCE_FLOOR = 1e-6

#: Relative finite-difference step for the Jacobian ``d vech(Sigma)/d theta``.
JACOBIAN_STEP = 1e-6

#: Relative finite-difference step for the numerical Hessian of the
#: discrepancy function.
HESSIAN_STEP = 1e-4

#: Largest ``p`` for which the duplication-matrix form of the expected
#: information is built explicitly; above it the algebraically identical trace
#: form is used instead.
MAX_P_FOR_KRONECKER = 30

#: Parameter kinds, used as the ``kind`` column of the reported tables.
KIND_LOADING = "loading"
KIND_SECOND_ORDER = "second_order"
KIND_PATH = "structural"
KIND_LATENT_VARIANCE = "latent_variance"
KIND_LATENT_COVARIANCE = "latent_covariance"
KIND_RESIDUAL_VARIANCE = "residual_variance"
KIND_RESIDUAL_COVARIANCE = "residual_covariance"

_VARIANCE_KINDS = (KIND_LATENT_VARIANCE, KIND_RESIDUAL_VARIANCE)

__all__ = [
    "VARIANCE_FLOOR", "JACOBIAN_STEP", "HESSIAN_STEP",
    "KIND_LOADING", "KIND_SECOND_ORDER", "KIND_PATH",
    "KIND_LATENT_VARIANCE", "KIND_LATENT_COVARIANCE",
    "KIND_RESIDUAL_VARIANCE", "KIND_RESIDUAL_COVARIANCE",
    "EstimationResult", "estimate",
    "vech", "unvech", "duplication_matrix",
    "ml_discrepancy", "uls_discrepancy", "dwls_discrepancy",
    "expected_information_acov", "observed_information_acov",
]


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def _bilingual(en: str, ar: str) -> str:
    """One note string carrying both interface languages."""
    return f"{en} | {ar}"


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence numpy floating-point and RuntimeWarning noise.

    The optimiser deliberately probes inadmissible parameter values, where a
    Cholesky factorisation or a logarithm legitimately overflows; those are
    handled by returning ``+inf`` rather than by surfacing a warning.
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield


def _symmetrise(a: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (a + a.T)`` as a fresh float array."""
    m = np.asarray(a, dtype=float)
    return 0.5 * (m + m.T)


def vech(a: np.ndarray) -> np.ndarray:
    """Half-vectorisation of a symmetric matrix, lower triangle column-wise.

    Parameters
    ----------
    a : numpy.ndarray
        ``(p, p)`` symmetric matrix.

    Returns
    -------
    numpy.ndarray
        Vector of length ``p (p + 1) / 2`` holding
        ``a[0,0], a[1,0], ..., a[p-1,0], a[1,1], a[2,1], ...``.

    References
    ----------
    Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
        Applications in Statistics and Econometrics*, chapter 3. Wiley.
    """
    m = np.asarray(a, dtype=float)
    p = m.shape[0]
    rows, cols = _vech_indices(p)
    return m[rows, cols]


def unvech(v: np.ndarray, p: Optional[int] = None) -> np.ndarray:
    """Inverse of :func:`vech`: rebuild the symmetric matrix.

    Parameters
    ----------
    v : numpy.ndarray
        Vector of length ``p (p + 1) / 2``.
    p : int, optional
        Side length; inferred from ``len(v)`` when omitted.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` symmetric matrix.

    References
    ----------
    Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
        Applications in Statistics and Econometrics*, chapter 3. Wiley.
    """
    vec = np.asarray(v, dtype=float).ravel()
    if p is None:
        p = int(round((np.sqrt(8.0 * vec.size + 1.0) - 1.0) / 2.0))
    rows, cols = _vech_indices(p)
    out = np.zeros((p, p), dtype=float)
    out[rows, cols] = vec
    out[cols, rows] = vec
    return out


def _vech_indices(p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row and column indices of ``vech`` in lower-triangle, column-wise order."""
    rows: List[int] = []
    cols: List[int] = []
    for j in range(p):
        for i in range(j, p):
            rows.append(i)
            cols.append(j)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def duplication_matrix(p: int) -> np.ndarray:
    """Duplication matrix ``D`` with ``vec(A) = D vech(A)``.

    Parameters
    ----------
    p : int
        Side length of the symmetric matrices involved.

    Returns
    -------
    numpy.ndarray
        ``(p**2, p (p + 1) / 2)`` matrix of zeros and ones.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1.0, 2.0], [2.0, 3.0]])
    >>> d = duplication_matrix(2)
    >>> bool(np.allclose(d @ vech(a), a.reshape(-1, order="F")))
    True

    References
    ----------
    Magnus, J. R., & Neudecker, H. (1980). The elimination matrix: some lemmas
        and applications. *SIAM Journal on Algebraic and Discrete Methods*,
        1(4), 422-449.
    """
    p = int(p)
    rows, cols = _vech_indices(p)
    d = np.zeros((p * p, rows.size), dtype=float)
    for k, (i, j) in enumerate(zip(rows, cols)):
        d[i + j * p, k] = 1.0
        d[j + i * p, k] = 1.0
    return d


# --------------------------------------------------------------------------- #
# Discrepancy functions
# --------------------------------------------------------------------------- #


def _logdet_cholesky(a: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """``(log|a|, cholesky_factor)``, or ``(-inf, None)`` when ``a`` is not PD."""
    try:
        chol = np.linalg.cholesky(_symmetrise(a))
    except np.linalg.LinAlgError:
        return float("-inf"), None
    diag = np.diag(chol)
    if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
        return float("-inf"), None
    return 2.0 * float(np.sum(np.log(diag))), chol


def ml_discrepancy(sample_cov: np.ndarray, sigma: np.ndarray, *,
                   logdet_sample: Optional[float] = None) -> float:
    """Wishart maximum-likelihood discrepancy ``F_ML``.

    Evaluates ``F = log|Sigma| + tr(S Sigma^-1) - log|S| - p``, which is zero
    if and only if ``Sigma == S`` and positive otherwise.

    Parameters
    ----------
    sample_cov : numpy.ndarray
        ``(p, p)`` sample covariance (or correlation) matrix ``S``.
    sigma : numpy.ndarray
        ``(p, p)`` model-implied matrix at the current parameter values.
    logdet_sample : float, optional
        Pre-computed ``log|S|``, so the constant is not recomputed on every
        call of the optimiser.

    Returns
    -------
    float
        The discrepancy, or ``+inf`` when ``sigma`` is not positive definite
        (so the optimiser simply backs off instead of raising).

    References
    ----------
    Joreskog, K. G. (1969). A general approach to confirmatory maximum
        likelihood factor analysis. *Psychometrika*, 34(2), 183-202.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        section 4.3. Wiley.
    """
    S = np.asarray(sample_cov, dtype=float)
    p = S.shape[0]
    with _quiet():
        logdet_sigma, chol = _logdet_cholesky(sigma)
        if chol is None:
            return float("inf")
        if logdet_sample is None:
            logdet_sample, _ = _logdet_cholesky(S)
            if not np.isfinite(logdet_sample):
                raise SpecificationError(
                    "The sample covariance matrix is singular, so the maximum "
                    "likelihood discrepancy is undefined. Remove constant or "
                    "duplicated items and estimate again.",
                    "مصفوفة تغاير العينة مفردة (غير قابلة للعكس)، لذلك لا يمكن "
                    "حساب دالة الإمكان الأقصى. احذف البنود الثابتة أو المكررة "
                    "تمامًا ثم أعد التقدير.",
                )
        trace = float(np.trace(_sla.cho_solve((chol, True), S,
                                              check_finite=False)))
        value = logdet_sigma + trace - float(logdet_sample) - p
    return float(value) if np.isfinite(value) else float("inf")


def uls_discrepancy(sample_cov: np.ndarray, sigma: np.ndarray) -> float:
    """Unweighted least squares discrepancy ``F_ULS``.

    Evaluates ``F = 0.5 * tr((S - Sigma)^2)``, which for symmetric matrices is
    half the sum of the squared residuals of the whole matrix.

    Parameters
    ----------
    sample_cov : numpy.ndarray
        ``(p, p)`` sample covariance matrix.
    sigma : numpy.ndarray
        ``(p, p)`` model-implied matrix.

    Returns
    -------
    float
        The discrepancy, or ``+inf`` when ``sigma`` is not positive definite.

    References
    ----------
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    """
    with _quiet():
        _, chol = _logdet_cholesky(sigma)
        if chol is None:
            return float("inf")
        residual = np.asarray(sample_cov, dtype=float) - np.asarray(sigma,
                                                                    dtype=float)
        value = 0.5 * float(np.sum(residual * residual))
    return float(value) if np.isfinite(value) else float("inf")


def dwls_discrepancy(sample_cov: np.ndarray, sigma: np.ndarray,
                     weights: np.ndarray) -> float:
    """Diagonally weighted least squares discrepancy ``F_DWLS``.

    Evaluates ``F = (s - sigma)' diag(W)^-1 (s - sigma)`` with
    ``s = vech(S)`` and ``sigma = vech(Sigma)``.  ``weights`` holds the
    *diagonal* of the weight matrix, i.e. the normal-theory asymptotic
    variances of the sample (polychoric) moments, so the division by them is
    the ``diag(W)^-1`` of the formula.

    Parameters
    ----------
    sample_cov : numpy.ndarray
        ``(p, p)`` sample covariance / polychoric correlation matrix.
    sigma : numpy.ndarray
        ``(p, p)`` model-implied matrix.
    weights : numpy.ndarray
        Length ``p (p + 1) / 2`` vector of positive weights, in ``vech`` order.

    Returns
    -------
    float
        The discrepancy, or ``+inf`` when ``sigma`` is not positive definite.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    Browne, M. W. (1984). Asymptotically distribution-free methods for the
        analysis of covariance structures. *British Journal of Mathematical
        and Statistical Psychology*, 37(1), 62-83.
    """
    with _quiet():
        _, chol = _logdet_cholesky(sigma)
        if chol is None:
            return float("inf")
        residual = vech(sample_cov) - vech(sigma)
        value = float(np.sum(residual * residual / np.asarray(weights,
                                                              dtype=float)))
    return float(value) if np.isfinite(value) else float("inf")


def _normal_theory_weights(sample_cov: np.ndarray) -> np.ndarray:
    """Diagonal of the normal-theory weight matrix, in ``vech`` order.

    The asymptotic variance of the sample moment ``s_ij`` under normality is
    ``s_ii s_jj + s_ij**2`` (Browne, 1984, eq. 2.4), which is what DWLS uses as
    ``diag(W)``.
    """
    S = np.asarray(sample_cov, dtype=float)
    rows, cols = _vech_indices(S.shape[0])
    w = S[rows, rows] * S[cols, cols] + S[rows, cols] ** 2
    return np.clip(w, 1e-12, None)


# --------------------------------------------------------------------------- #
# Parameterisation
# --------------------------------------------------------------------------- #


@dataclass
class _FreeParameter:
    """One free entry of one model matrix."""

    kind: str
    label: str
    i: int
    j: int
    start: float = 0.0
    #: Human-readable left/right hand side used by the reported tables.
    lhs: str = ""
    rhs: str = ""
    lhs_id: str = ""
    rhs_id: str = ""


@dataclass
class _Parameterisation:
    """The marker-variable parameterisation of a compiled model.

    Holds the free-parameter list, the fixed marker entries, and the mapping
    from a parameter vector back to ``Lambda``, ``B``, ``Psi`` and ``Theta``.
    """

    cm: "CompiledModel"
    parameters: List[_FreeParameter]
    lambda_fixed: Dict[Tuple[int, int], float]
    b_fixed: Dict[Tuple[int, int], float]
    factor_items: Dict[int, List[int]]
    second_order_children: Dict[int, List[int]]
    endogenous: np.ndarray
    p: int
    m: int

    # -- sizes ------------------------------------------------------------- #

    @property
    def q(self) -> int:
        """Number of free parameters."""
        return len(self.parameters)

    @property
    def labels(self) -> List[str]:
        """Parameter labels, in vector order."""
        return [prm.label for prm in self.parameters]

    # -- vectors ----------------------------------------------------------- #

    def start_vector(self) -> np.ndarray:
        """The compiled population values expressed in the marker metric."""
        x = np.asarray([prm.start for prm in self.parameters], dtype=float)
        return self.clip(x)

    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Box constraints: every variance bounded below by ``VARIANCE_FLOOR``."""
        out: List[Tuple[Optional[float], Optional[float]]] = []
        for prm in self.parameters:
            if prm.kind in _VARIANCE_KINDS:
                out.append((VARIANCE_FLOOR, None))
            else:
                out.append((None, None))
        return out

    def clip(self, theta: np.ndarray) -> np.ndarray:
        """Push a parameter vector inside the box constraints."""
        x = np.array(theta, dtype=float, copy=True)
        for k, prm in enumerate(self.parameters):
            if prm.kind in _VARIANCE_KINDS and x[k] < VARIANCE_FLOOR:
                x[k] = VARIANCE_FLOOR
        return x

    # -- matrices ---------------------------------------------------------- #

    def matrices(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
        """Rebuild ``(Lambda, B, Psi, Theta)`` from a parameter vector."""
        Lambda = np.zeros((self.p, self.m), dtype=float)
        B = np.zeros((self.m, self.m), dtype=float)
        Psi = np.zeros((self.m, self.m), dtype=float)
        Theta = np.zeros((self.p, self.p), dtype=float)
        for (j, f), value in self.lambda_fixed.items():
            Lambda[j, f] = value
        for (i, k), value in self.b_fixed.items():
            B[i, k] = value
        for idx, prm in enumerate(self.parameters):
            value = float(theta[idx])
            if prm.kind == KIND_LOADING:
                Lambda[prm.i, prm.j] = value
            elif prm.kind in (KIND_PATH, KIND_SECOND_ORDER):
                B[prm.i, prm.j] = value
            elif prm.kind == KIND_LATENT_VARIANCE:
                Psi[prm.i, prm.i] = value
            elif prm.kind == KIND_LATENT_COVARIANCE:
                Psi[prm.i, prm.j] = Psi[prm.j, prm.i] = value
            elif prm.kind == KIND_RESIDUAL_VARIANCE:
                Theta[prm.i, prm.i] = value
            elif prm.kind == KIND_RESIDUAL_COVARIANCE:
                Theta[prm.i, prm.j] = Theta[prm.j, prm.i] = value
        return Lambda, B, Psi, Theta

    def phi(self, B: np.ndarray, Psi: np.ndarray) -> Optional[np.ndarray]:
        """``Phi = (I - B)^-1 Psi (I - B)^-T``, or ``None`` if ``I - B`` is singular."""
        try:
            inv = np.linalg.inv(np.eye(self.m) - B)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(inv)):
            return None
        return _symmetrise(inv @ Psi @ inv.T)

    def sigma(self, theta: np.ndarray) -> Optional[np.ndarray]:
        """Model-implied covariance matrix, or ``None`` on a singular system."""
        Lambda, B, Psi, Theta = self.matrices(theta)
        with _quiet():
            Phi = self.phi(B, Psi)
            if Phi is None:
                return None
            sigma = _symmetrise(Lambda @ Phi @ Lambda.T + Theta)
        if not np.all(np.isfinite(sigma)):
            return None
        return sigma


def _factor_scales(cm: "CompiledModel",
                   item_markers: Dict[int, int],
                   so_markers: Dict[int, int],
                   notes: List[str]) -> np.ndarray:
    """Scale factor ``s_f`` taking the standardised metric to the marker metric.

    Writing ``eta_new = s_f eta_std`` leaves ``Sigma`` untouched provided
    ``Lambda_new = Lambda_std S^-1``, ``B_new = S B_std S^-1`` and
    ``Psi_new = S Psi_std S``.  Choosing ``s_f`` as the standardised loading of
    the factor's marker makes that marker loading exactly one, which is the
    identification constraint this module imposes.
    """
    m = cm.n_factors
    Lambda = np.asarray(cm.Lambda, dtype=float)
    B = np.asarray(cm.B, dtype=float)
    scales = np.ones(m, dtype=float)
    for f, j in item_markers.items():
        value = float(Lambda[j, f])
        if abs(value) < 1e-8:
            notes.append(_bilingual(
                f"MARKER_NEAR_ZERO: the population loading of the marker item "
                f"for '{cm.factor_names[f]}' is ~0, so the compiled start "
                "values could not be rescaled exactly; the two-step start was "
                "used for that factor.",
                f"تحميل البند المرجعي للعامل «{cm.factor_names[f]}» يساوي صفرًا "
                "تقريبًا، لذلك تعذّر تحويل قيم البداية المُشتقة من النموذج بدقة، "
                "واستُخدمت بداية العاملين لهذا العامل.",
            ))
            scales[f] = 1.0
        else:
            scales[f] = value
    for c, d1 in so_markers.items():
        gamma = float(B[d1, c])
        if abs(gamma) < 1e-8:
            scales[c] = 1.0
        else:
            scales[c] = gamma * scales[d1]
    return scales


def _build_parameterisation(cm: "CompiledModel",
                            notes: List[str]) -> _Parameterisation:
    """Derive the marker-variable parameterisation of a compiled model.

    Parameters
    ----------
    cm : CompiledModel
        Compiled model supplying the matrix patterns, the orderings and the
        population values used as starting points.
    notes : list of str
        Appended to in place with any reinterpretation the caller should see.

    Returns
    -------
    _Parameterisation
        Free-parameter list plus the fixed marker entries.

    Raises
    ------
    semsim.core.errors.IdentificationError
        When a latent variable has nothing that can fix its scale.
    """
    spec = cm.spec
    p, m = cm.n_items, cm.n_factors
    Lambda_std = np.asarray(cm.Lambda, dtype=float)
    B_std = np.asarray(cm.B, dtype=float)
    Psi_std = np.asarray(cm.Psi, dtype=float)
    Theta_std = np.asarray(cm.Theta, dtype=float)

    # -- which items belong to which factor ------------------------------- #
    factor_items: Dict[int, List[int]] = {f: [] for f in range(m)}
    for j, f in enumerate(cm.home_factor):
        factor_items[int(f)].append(j)

    second_order_children: Dict[int, List[int]] = {}
    for construct in spec.constructs:
        if not construct.dimensions:
            continue
        ci = cm.factor_index[construct.id]
        second_order_children[ci] = [cm.factor_index[d.id]
                                     for d in construct.dimensions]

    method_index = cm.method_factor_index

    # -- the Lambda pattern ------------------------------------------------ #
    pattern: set[Tuple[int, int]] = {
        (int(j), int(f)) for j, f in zip(*np.nonzero(Lambda_std))
    }
    for j, f in enumerate(cm.home_factor):
        pattern.add((j, int(f)))

    # -- markers ----------------------------------------------------------- #
    item_markers: Dict[int, int] = {}
    for f in range(m):
        if factor_items[f]:
            item_markers[f] = factor_items[f][0]
    if method_index is not None and method_index not in item_markers:
        method_rows = sorted(j for (j, f) in pattern if f == method_index)
        if method_rows:
            item_markers[method_index] = method_rows[0]

    so_markers: Dict[int, int] = {}
    for ci, children in second_order_children.items():
        if ci in item_markers or not children:
            continue
        so_markers[ci] = children[0]

    for f in range(m):
        if f not in item_markers and f not in so_markers:
            raise IdentificationError(
                f"The latent variable '{cm.factor_names[f]}' has no indicator "
                "and no dimension that can fix its scale, so the model is not "
                "identified. Give it at least one item.",
                f"العامل الكامن «{cm.factor_names[f]}» لا يملك أي بند أو بُعد "
                "يُثبّت وحدة قياسه، لذلك النموذج غير محدَّد. أضف له بندًا واحدًا "
                "على الأقل.",
                factor=cm.factor_names[f],
            )

    scales = _factor_scales(cm, item_markers, so_markers, notes)
    inv_scales = 1.0 / scales
    Lambda_m = Lambda_std * inv_scales[None, :]
    B_m = (scales[:, None] * B_std) * inv_scales[None, :]
    Psi_m = (scales[:, None] * Psi_std) * scales[None, :]
    Theta_m = Theta_std

    lambda_fixed: Dict[Tuple[int, int], float] = {
        (j, f): 1.0 for f, j in item_markers.items()
    }
    b_fixed: Dict[Tuple[int, int], float] = {
        (d1, c): 1.0 for c, d1 in so_markers.items()
    }

    parameters: List[_FreeParameter] = []

    # -- 1. loadings ------------------------------------------------------- #
    for j, f in sorted(pattern):
        if (j, f) in lambda_fixed:
            continue
        parameters.append(_FreeParameter(
            kind=KIND_LOADING,
            label=f"{cm.factor_names[f]}=~{cm.item_names[j]}",
            i=j, j=f, start=float(Lambda_m[j, f]),
            lhs=cm.item_names[j], rhs=cm.factor_names[f],
            lhs_id=cm.item_ids[j], rhs_id=cm.factor_ids[f],
        ))

    # -- 2. structural matrix B -------------------------------------------- #
    b_pattern: Dict[Tuple[int, int], str] = {}
    for ci, children in second_order_children.items():
        for di in children:
            b_pattern[(di, ci)] = KIND_SECOND_ORDER
    for path in spec.paths:
        if path.source == path.target:
            continue
        si = cm.factor_index.get(path.source)
        ti = cm.factor_index.get(path.target)
        if si is None or ti is None:
            continue
        b_pattern[(ti, si)] = KIND_PATH
    for (i, k), kind in sorted(b_pattern.items()):
        if (i, k) in b_fixed:
            continue
        parameters.append(_FreeParameter(
            kind=kind,
            label=f"{cm.factor_names[i]}~{cm.factor_names[k]}",
            i=i, j=k, start=float(B_m[i, k]),
            lhs=cm.factor_names[i], rhs=cm.factor_names[k],
            lhs_id=cm.factor_ids[i], rhs_id=cm.factor_ids[k],
        ))

    endogenous = np.zeros(m, dtype=bool)
    for (i, _k) in list(b_pattern) + list(b_fixed):
        endogenous[i] = True

    # -- 3. latent variances ----------------------------------------------- #
    for f in range(m):
        start = float(Psi_m[f, f])
        parameters.append(_FreeParameter(
            kind=KIND_LATENT_VARIANCE,
            label=f"{cm.factor_names[f]}~~{cm.factor_names[f]}",
            i=f, j=f, start=max(start, VARIANCE_FLOOR),
            lhs=cm.factor_names[f], rhs=cm.factor_names[f],
            lhs_id=cm.factor_ids[f], rhs_id=cm.factor_ids[f],
        ))

    # -- 4. latent covariances --------------------------------------------- #
    latent_pairs: set[Tuple[int, int]] = set()
    for lc in spec.latent_covariances:
        a = cm.factor_index.get(lc.construct_a)
        b = cm.factor_index.get(lc.construct_b)
        if a is None or b is None or a == b:
            continue
        latent_pairs.add((min(a, b), max(a, b)))
    for a, b in sorted(latent_pairs):
        parameters.append(_FreeParameter(
            kind=KIND_LATENT_COVARIANCE,
            label=f"{cm.factor_names[a]}~~{cm.factor_names[b]}",
            i=a, j=b, start=float(Psi_m[a, b]),
            lhs=cm.factor_names[a], rhs=cm.factor_names[b],
            lhs_id=cm.factor_ids[a], rhs_id=cm.factor_ids[b],
        ))

    # -- 5. residual variances --------------------------------------------- #
    for j in range(p):
        parameters.append(_FreeParameter(
            kind=KIND_RESIDUAL_VARIANCE,
            label=f"{cm.item_names[j]}~~{cm.item_names[j]}",
            i=j, j=j, start=max(float(Theta_m[j, j]), VARIANCE_FLOOR),
            lhs=cm.item_names[j], rhs=cm.item_names[j],
            lhs_id=cm.item_ids[j], rhs_id=cm.item_ids[j],
        ))

    # -- 6. residual covariances ------------------------------------------- #
    residual_pairs: set[Tuple[int, int]] = set()
    for ec in spec.error_covariances:
        a = cm.item_index.get(ec.item_a)
        b = cm.item_index.get(ec.item_b)
        if a is None or b is None or a == b:
            continue
        residual_pairs.add((min(a, b), max(a, b)))
    for a, b in sorted(residual_pairs):
        parameters.append(_FreeParameter(
            kind=KIND_RESIDUAL_COVARIANCE,
            label=f"{cm.item_names[a]}~~{cm.item_names[b]}",
            i=a, j=b, start=float(Theta_m[a, b]),
            lhs=cm.item_names[a], rhs=cm.item_names[b],
            lhs_id=cm.item_ids[a], rhs_id=cm.item_ids[b],
        ))

    return _Parameterisation(
        cm=cm, parameters=parameters,
        lambda_fixed=lambda_fixed, b_fixed=b_fixed,
        factor_items=factor_items,
        second_order_children=second_order_children,
        endogenous=endogenous, p=p, m=m,
    )


# --------------------------------------------------------------------------- #
# Two-step (factor-analytic) starting values
# --------------------------------------------------------------------------- #


def _principal_axis_loadings(block: np.ndarray) -> np.ndarray:
    """Standardised loadings of a one-factor principal-axis solution.

    One communality-refinement pass: the diagonal of the item correlation
    block is replaced by the largest absolute off-diagonal correlation of each
    row, the leading eigenpair of the reduced matrix is taken, and the implied
    loadings are read off as ``v * sqrt(lambda_1)``.

    Parameters
    ----------
    block : numpy.ndarray
        ``(k, k)`` correlation matrix of one factor's items.

    Returns
    -------
    numpy.ndarray
        Length-``k`` vector of standardised loadings in ``[0.15, 0.95]``.

    References
    ----------
    Harman, H. H. (1976). *Modern Factor Analysis* (3rd ed.), chapter 8.
        University of Chicago Press.
    """
    k = block.shape[0]
    if k == 1:
        return np.full(1, 0.70)
    reduced = np.array(block, dtype=float, copy=True)
    off = np.abs(reduced - np.diag(np.diag(reduced)))
    np.fill_diagonal(reduced, np.clip(off.max(axis=1), 0.05, 0.95))
    with _quiet():
        values, vectors = np.linalg.eigh(_symmetrise(reduced))
    lead = int(np.argmax(values))
    v = vectors[:, lead]
    if float(np.sum(v)) < 0.0:
        v = -v
    loadings = v * float(np.sqrt(max(values[lead], 1e-6)))
    return np.clip(loadings, 0.15, 0.95)


def _two_step_start(par: _Parameterisation, sample_cov: np.ndarray,
                    data: Optional[np.ndarray]) -> np.ndarray:
    """Factor-analytic starting values, independent of the compiled model.

    Step one estimates the measurement model factor by factor with a
    principal-axis solution and sets each residual variance to
    ``s_jj (1 - h_j)``.  Step two forms unit-weight factor-score proxies from
    the standardised items and reads the structural coefficients off an
    ordinary least squares regression of each endogenous proxy on its
    predictors, the disturbance variance following from the regression
    ``R**2``.  Everything is then converted into the marker metric.

    Parameters
    ----------
    par : _Parameterisation
        The marker-variable parameterisation whose vector layout is filled.
    sample_cov : numpy.ndarray
        ``(p, p)`` sample covariance matrix.
    data : numpy.ndarray or None
        ``(n, p)`` observed data, used only for the factor-score proxies.  The
        latent correlations fall back on the measurement model when it is
        ``None``.

    Returns
    -------
    numpy.ndarray
        A parameter vector inside the box constraints.

    References
    ----------
    Joreskog, K. G. (1983). Factor analysis as an error-in-variables model. In
        H. Wainer & S. Messick (Eds.), *Principals of Modern Psychological
        Measurement* (pp. 185-196). Erlbaum.
    Harman, H. H. (1976). *Modern Factor Analysis* (3rd ed.).
        University of Chicago Press.
    """
    cm = par.cm
    p, m = par.p, par.m
    S = np.asarray(sample_cov, dtype=float)
    sd = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    R = S / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)

    std_loadings = np.full(p, 0.70)
    for f, items in par.factor_items.items():
        if not items:
            continue
        idx = np.asarray(items, dtype=int)
        std_loadings[idx] = _principal_axis_loadings(R[np.ix_(idx, idx)])

    # Latent standard deviations in the marker metric: sqrt(Phi_ff) is the
    # unstandardised loading of the marker, i.e. lambda*_marker * sd_marker.
    latent_sd = np.ones(m, dtype=float)
    for f, j in ((f, items[0]) for f, items in par.factor_items.items() if items):
        latent_sd[f] = max(float(std_loadings[j] * sd[j]), 1e-3)
    if cm.method_factor_index is not None:
        mi = cm.method_factor_index
        if not par.factor_items.get(mi):
            rows = [j for (j, f) in par.lambda_fixed if f == mi]
            anchor = rows[0] if rows else 0
            latent_sd[mi] = max(0.25 * float(sd[anchor]), 1e-3)
    for ci, children in par.second_order_children.items():
        if par.factor_items.get(ci) or not children:
            continue
        # The marker dimension's second-order loading is fixed to one, so the
        # construct explains part of that dimension's variance; half is a
        # neutral, always-admissible split.
        latent_sd[ci] = max(float(latent_sd[children[0]]) * np.sqrt(0.5), 1e-3)

    # Factor-score proxies (standardised), in topological factor order.
    scores: Optional[np.ndarray] = None
    if data is not None and data.shape[0] > max(3, m):
        x = np.asarray(data, dtype=float)
        z = (x - x.mean(axis=0)) / sd
        scores = np.zeros((x.shape[0], m), dtype=float)
        for f in range(m):
            items = par.factor_items.get(f) or []
            if items:
                scores[:, f] = z[:, items].mean(axis=1)
            elif f in par.second_order_children:
                children = par.second_order_children[f]
                scores[:, f] = scores[:, children].mean(axis=1)
        with _quiet():
            keep = scores.std(axis=0) > 1e-9
            scores[:, keep] = ((scores[:, keep] - scores[:, keep].mean(axis=0))
                               / scores[:, keep].std(axis=0))
            scores[:, ~keep] = 0.0

    # Standardised structural coefficients and residual variance ratios.
    std_beta = np.zeros((m, m), dtype=float)
    unexplained = np.ones(m, dtype=float)
    predictors: Dict[int, List[int]] = {}
    for (i, k) in list(par.b_fixed) + [(prm.i, prm.j) for prm in par.parameters
                                       if prm.kind in (KIND_PATH,
                                                       KIND_SECOND_ORDER)]:
        predictors.setdefault(i, []).append(k)
    for i, preds in predictors.items():
        cols = sorted(set(preds))
        if scores is None:
            std_beta[i, cols] = 0.30
            unexplained[i] = 0.70
            continue
        X = scores[:, cols]
        y = scores[:, i]
        with _quiet():
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                beta = np.full(len(cols), 0.30)
            residual = y - X @ beta
            r2 = 1.0 - float(residual.var()) / max(float(y.var()), 1e-12)
        beta = np.clip(np.nan_to_num(beta, nan=0.3), -0.95, 0.95)
        std_beta[i, cols] = beta
        unexplained[i] = float(np.clip(1.0 - r2, 0.02, 1.0))

    theta = np.zeros(par.q, dtype=float)
    for idx, prm in enumerate(par.parameters):
        if prm.kind == KIND_LOADING:
            j, f = prm.i, prm.j
            if f == cm.home_factor[j]:
                theta[idx] = float(std_loadings[j] * sd[j] / latent_sd[f])
            else:
                theta[idx] = float(0.10 * sd[j] / latent_sd[f])
        elif prm.kind in (KIND_PATH, KIND_SECOND_ORDER):
            i, k = prm.i, prm.j
            theta[idx] = float(std_beta[i, k] * latent_sd[i] / latent_sd[k])
        elif prm.kind == KIND_LATENT_VARIANCE:
            f = prm.i
            total = latent_sd[f] ** 2
            theta[idx] = float(max(total * unexplained[f], VARIANCE_FLOOR))
        elif prm.kind == KIND_LATENT_COVARIANCE:
            a, b = prm.i, prm.j
            r = 0.20
            if scores is not None:
                with _quiet():
                    r = float(np.corrcoef(scores[:, a], scores[:, b])[0, 1])
                if not np.isfinite(r):
                    r = 0.20
            r = float(np.clip(r, -0.9, 0.9))
            theta[idx] = r * float(np.sqrt(theta_of(par, theta, a)
                                           * theta_of(par, theta, b)))
        elif prm.kind == KIND_RESIDUAL_VARIANCE:
            j = prm.i
            theta[idx] = float(max(S[j, j] * (1.0 - std_loadings[j] ** 2),
                                   VARIANCE_FLOOR))
        elif prm.kind == KIND_RESIDUAL_COVARIANCE:
            a, b = prm.i, prm.j
            va = S[a, a] * (1.0 - std_loadings[a] ** 2)
            vb = S[b, b] * (1.0 - std_loadings[b] ** 2)
            theta[idx] = float(0.10 * np.sqrt(max(va, 0.0) * max(vb, 0.0)))
    return par.clip(theta)


def theta_of(par: _Parameterisation, theta: np.ndarray, factor: int) -> float:
    """Current disturbance variance of ``factor`` in a partially filled vector.

    Used while assembling the two-step start, where the latent covariances are
    written after the latent variances.

    Parameters
    ----------
    par : _Parameterisation
        The parameterisation being filled.
    theta : numpy.ndarray
        The vector under construction.
    factor : int
        Index of the latent variable.

    Returns
    -------
    float
        The variance already written for that latent variable, or
        ``VARIANCE_FLOOR`` when none has been.

    References
    ----------
    Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*.
        Scientific Software International.
    """
    for idx, prm in enumerate(par.parameters):
        if prm.kind == KIND_LATENT_VARIANCE and prm.i == factor:
            return max(float(theta[idx]), VARIANCE_FLOOR)
    return VARIANCE_FLOOR


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #


@dataclass
class _Problem:
    """Everything the discrepancy function needs besides ``theta``."""

    par: _Parameterisation
    kind: EstimatorKind
    sample_cov: np.ndarray
    logdet_sample: float
    weights: np.ndarray

    def value(self, theta: np.ndarray) -> float:
        """Discrepancy at ``theta``; ``+inf`` for an inadmissible solution."""
        sigma = self.par.sigma(theta)
        if sigma is None:
            return float("inf")
        if self.kind is EstimatorKind.ULS:
            return uls_discrepancy(self.sample_cov, sigma)
        if self.kind is EstimatorKind.DWLS:
            return dwls_discrepancy(self.sample_cov, sigma, self.weights)
        return ml_discrepancy(self.sample_cov, sigma,
                              logdet_sample=self.logdet_sample)


_OPTIMISER_CHAIN: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("L-BFGS-B", {"maxiter": 4000, "maxfun": 200000,
                  "ftol": 1e-14, "gtol": 1e-10}),
    ("Nelder-Mead", {"maxiter": 40000, "maxfev": 80000,
                     "xatol": 1e-10, "fatol": 1e-14, "adaptive": True}),
    ("Powell", {"maxiter": 40000, "maxfev": 160000,
                "xtol": 1e-10, "ftol": 1e-14}),
)


def _run_optimiser(problem: _Problem, x0: np.ndarray
                   ) -> Tuple[np.ndarray, float, bool, int, str]:
    """Minimise the discrepancy, escalating through three optimisers.

    L-BFGS-B is tried first because the discrepancy is smooth and the box
    constraints keep every variance positive; a failure to converge escalates
    to Nelder-Mead and then to Powell, both bounded, before the fit is
    reported as non-converged.

    Parameters
    ----------
    problem : _Problem
        Discrepancy function, sample moments and parameterisation.
    x0 : numpy.ndarray
        Starting values, already inside the box constraints.

    Returns
    -------
    (numpy.ndarray, float, bool, int, str)
        Estimates, discrepancy at the optimum, convergence flag, iteration
        count and the name of the optimiser that produced the solution.

    References
    ----------
    Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory
        algorithm for bound constrained optimization. *SIAM Journal on
        Scientific Computing*, 16(5), 1190-1208.
    Nelder, J. A., & Mead, R. (1965). A simplex method for function
        minimization. *The Computer Journal*, 7(4), 308-313.
    Powell, M. J. D. (1964). An efficient method for finding the minimum of a
        function of several variables without calculating derivatives.
        *The Computer Journal*, 7(2), 155-162.
    """
    par = problem.par
    bounds = par.bounds()
    best_x = par.clip(x0)
    best_f = problem.value(best_x)
    best_iter = 0
    best_method = "start"
    current = best_x

    with _quiet():
        for method, options in _OPTIMISER_CHAIN:
            try:
                res = _opt.minimize(problem.value, current, method=method,
                                    bounds=bounds, options=dict(options))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            x = par.clip(np.asarray(res.x, dtype=float))
            f = problem.value(x)
            if np.isfinite(f) and f < best_f:
                best_f, best_x = f, x
                best_iter = int(getattr(res, "nit", 0) or getattr(res, "nfev", 0))
                best_method = method
            if bool(res.success) and np.isfinite(f):
                return best_x, best_f, True, max(best_iter, 1), best_method
            if np.isfinite(f):
                current = x

    return best_x, best_f, False, max(best_iter, 1), best_method


# --------------------------------------------------------------------------- #
# Derivatives, information matrices and standard errors
# --------------------------------------------------------------------------- #


def _sigma_jacobian(par: _Parameterisation, theta: np.ndarray) -> np.ndarray:
    """``Delta = d vech(Sigma) / d theta'`` by central differences.

    Parameters
    ----------
    par : _Parameterisation
        The parameterisation.
    theta : numpy.ndarray
        Point at which the Jacobian is taken, normally the optimum.

    Returns
    -------
    numpy.ndarray
        ``(p (p + 1) / 2, q)`` matrix; a column is all-``nan`` when the
        implied matrix could not be formed on one side of the step.

    References
    ----------
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    """
    q = par.q
    rows = par.p * (par.p + 1) // 2
    delta = np.zeros((rows, q), dtype=float)
    with _quiet():
        for k in range(q):
            step = JACOBIAN_STEP * max(1.0, abs(float(theta[k])))
            up = np.array(theta, dtype=float, copy=True)
            down = np.array(theta, dtype=float, copy=True)
            up[k] += step
            down[k] -= step
            s_up = par.sigma(up)
            s_down = par.sigma(down)
            if s_up is None or s_down is None:
                delta[:, k] = np.nan
                continue
            delta[:, k] = (vech(s_up) - vech(s_down)) / (2.0 * step)
    return delta


def _numerical_hessian(problem: _Problem, theta: np.ndarray) -> np.ndarray:
    """Hessian of the discrepancy function by central differences."""
    q = par_q = problem.par.q
    steps = np.asarray([HESSIAN_STEP * max(1.0, abs(float(t))) for t in theta],
                       dtype=float)
    hessian = np.zeros((q, q), dtype=float)
    f0 = problem.value(theta)
    with _quiet():
        for a in range(par_q):
            ea = np.zeros(q)
            ea[a] = steps[a]
            f_plus = problem.value(theta + ea)
            f_minus = problem.value(theta - ea)
            hessian[a, a] = (f_plus - 2.0 * f0 + f_minus) / (steps[a] ** 2)
            for b in range(a + 1, q):
                eb = np.zeros(q)
                eb[b] = steps[b]
                f_pp = problem.value(theta + ea + eb)
                f_pm = problem.value(theta + ea - eb)
                f_mp = problem.value(theta - ea + eb)
                f_mm = problem.value(theta - ea - eb)
                value = ((f_pp - f_pm - f_mp + f_mm)
                         / (4.0 * steps[a] * steps[b]))
                hessian[a, b] = hessian[b, a] = value
    return _symmetrise(hessian)


def _information_from_delta(delta: np.ndarray, sigma: np.ndarray,
                            *, use_kronecker: Optional[bool] = None
                            ) -> np.ndarray:
    """``Delta' (D' (Sigma^-1 (x) Sigma^-1) D) Delta``, the ML information.

    Parameters
    ----------
    delta : numpy.ndarray
        ``(P, q)`` Jacobian ``d vech(Sigma) / d theta'``.
    sigma : numpy.ndarray
        ``(p, p)`` model-implied matrix at the optimum.
    use_kronecker : bool, optional
        Force the explicit duplication-matrix / Kronecker form (``True``) or
        the algebraically identical trace form
        ``tr(dSigma_a Sigma^-1 dSigma_b Sigma^-1)`` (``False``).  By default
        the explicit form is used while ``p <= MAX_P_FOR_KRONECKER``.

    Returns
    -------
    numpy.ndarray
        ``(q, q)`` symmetric information matrix (before the ``2 / (N - 1)``
        scaling).

    References
    ----------
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
        Applications in Statistics and Econometrics*, chapter 10. Wiley.
    """
    Sigma = _symmetrise(sigma)
    p = Sigma.shape[0]
    with _quiet():
        inv = np.linalg.inv(Sigma)
        if use_kronecker is None:
            use_kronecker = p <= MAX_P_FOR_KRONECKER
        if use_kronecker:
            D = duplication_matrix(p)
            middle = D.T @ np.kron(inv, inv) @ D
            info = delta.T @ middle @ delta
        else:
            q = delta.shape[1]
            blocks = [inv @ unvech(delta[:, k], p) @ inv for k in range(q)]
            info = np.empty((q, q), dtype=float)
            for a in range(q):
                da = unvech(delta[:, a], p)
                for b in range(a, q):
                    value = float(np.sum(da * blocks[b]))
                    info[a, b] = info[b, a] = value
    return _symmetrise(info)


def expected_information_acov(delta: np.ndarray, sigma: np.ndarray, n: int,
                              *, use_kronecker: Optional[bool] = None
                              ) -> Tuple[np.ndarray, bool]:
    """Asymptotic covariance of ``theta_hat`` from the expected information.

    Evaluates
    ``ACOV = (2 / (N - 1)) [Delta' (D' (Sigma^-1 (x) Sigma^-1) D) Delta]^-1``.

    Parameters
    ----------
    delta : numpy.ndarray
        ``(P, q)`` Jacobian ``d vech(Sigma) / d theta'`` at the optimum.
    sigma : numpy.ndarray
        ``(p, p)`` model-implied matrix at the optimum.
    n : int
        Number of complete cases; the sample covariance is assumed to use
        ``N - 1`` in its denominator, hence the ``N - 1`` here.
    use_kronecker : bool, optional
        Passed through to the information matrix construction.

    Returns
    -------
    (numpy.ndarray, bool)
        The ``(q, q)`` asymptotic covariance matrix and a flag that is
        ``True`` when a Moore-Penrose pseudo-inverse had to be used because
        the information matrix was singular.

    References
    ----------
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        section 4.4. Wiley.
    """
    info = _information_from_delta(delta, sigma, use_kronecker=use_kronecker)
    return _scaled_inverse(info, n)


def observed_information_acov(hessian: np.ndarray, n: int
                              ) -> Tuple[np.ndarray, bool]:
    """Asymptotic covariance of ``theta_hat`` from the observed information.

    Evaluates ``ACOV = (2 / (N - 1)) H^-1`` with ``H`` the Hessian of the
    discrepancy function at the optimum, which is the finite-sample
    counterpart of :func:`expected_information_acov` and agrees with it to
    order ``1 / N``.

    Parameters
    ----------
    hessian : numpy.ndarray
        ``(q, q)`` numerical Hessian of ``F`` at the optimum.
    n : int
        Number of complete cases.

    Returns
    -------
    (numpy.ndarray, bool)
        The asymptotic covariance matrix and a pseudo-inverse flag.

    References
    ----------
    Efron, B., & Hinkley, D. V. (1978). Assessing the accuracy of the maximum
        likelihood estimator: observed versus expected Fisher information.
        *Biometrika*, 65(3), 457-483.
    """
    return _scaled_inverse(_symmetrise(hessian), n)


def _scaled_inverse(info: np.ndarray, n: int) -> Tuple[np.ndarray, bool]:
    """``(2 / (n - 1)) * info^-1``, falling back on a pseudo-inverse."""
    scale = 2.0 / max(float(n) - 1.0, 1.0)
    with _quiet():
        if not np.all(np.isfinite(info)):
            return np.full(info.shape, np.nan), True
        try:
            cond = float(np.linalg.cond(info))
        except np.linalg.LinAlgError:
            cond = float("inf")
        if np.isfinite(cond) and cond < 1e12:
            try:
                return scale * np.linalg.inv(info), False
            except np.linalg.LinAlgError:
                pass
        return scale * np.linalg.pinv(info, rcond=1e-12), True


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #


@dataclass
class EstimationResult:
    """Everything one model fit produced.

    Attributes
    ----------
    converged : bool
        Whether an optimiser reported success.
    n_iter : int
        Iterations used by the optimiser that produced the solution.
    estimator : EstimatorKind
        The discrepancy function that was minimised.
    n, n_free, df : int
        Complete cases, free parameters and degrees of freedom
        ``p (p + 1) / 2 - n_free``.
    fmin : float
        Discrepancy at the optimum.
    sigma_hat, sample_cov : numpy.ndarray
        ``(p, p)`` model-implied and observed matrices.
    parameters, standard_errors, z_values, p_values, standardised : dict
        Free-parameter estimates keyed by label, all five sharing one key set.
    loadings, path_estimates : pandas.DataFrame
        Reporting tables carrying both metrics; fixed marker rows are included
        and flagged.
    r_squared : dict of str to float
        ``1 - Psi_ii / Phi_ii`` per endogenous latent variable, keyed by name.
    message : str
        Bilingual notes: the optimiser used, pseudo-inverse fallbacks, omitted
        latent interactions, listwise deletion, and so on.
    """

    converged: bool
    n_iter: int
    estimator: EstimatorKind
    n: int
    n_free: int
    df: int
    fmin: float
    sigma_hat: np.ndarray
    sample_cov: np.ndarray
    parameters: Dict[str, float]
    standard_errors: Dict[str, float]
    z_values: Dict[str, float]
    p_values: Dict[str, float]
    standardised: Dict[str, float]
    loadings: pd.DataFrame
    path_estimates: pd.DataFrame
    r_squared: Dict[str, float]
    message: str
    # -- additions (never positional for the published contract) ----------- #
    chi_square: float = 0.0
    item_names: List[str] = field(default_factory=list)
    factor_names: List[str] = field(default_factory=list)
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    parameter_labels: List[str] = field(default_factory=list)
    parameter_kinds: Dict[str, str] = field(default_factory=dict)
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    standardised_all: Dict[str, float] = field(default_factory=dict)
    standard_errors_observed: Dict[str, float] = field(default_factory=dict)
    acov: Optional[np.ndarray] = None
    acov_observed: Optional[np.ndarray] = None
    se_method: str = "expected_information"
    optimiser: str = ""
    start_used: str = ""
    r_squared_by_id: Dict[str, float] = field(default_factory=dict)
    second_order_loadings: Optional[pd.DataFrame] = None
    latent_covariance: Optional[np.ndarray] = None
    warnings: List[str] = field(default_factory=list)

    def parameter_table(self) -> pd.DataFrame:
        """Every reported parameter in one long table.

        Returns
        -------
        pandas.DataFrame
            Columns ``label, kind, lhs, rhs, est, se, z, p, std, fixed``.

        References
        ----------
        Rosseel, Y. (2012). lavaan: An R package for structural equation
            modeling. *Journal of Statistical Software*, 48(2), 1-36.
        """
        frames = [self.loadings, self.path_estimates, self._variance_table]
        return pd.concat([f for f in frames if f is not None],
                         ignore_index=True, sort=False)

    _variance_table: Optional[pd.DataFrame] = None


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #


def _standardise(par: _Parameterisation, theta: np.ndarray
                 ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray,
                            np.ndarray]:
    """Standardised values for every free and fixed parameter.

    Applies the completely standardised transformation of Bollen (1989,
    section 8.1)::

        lambda*_jk = lambda_jk sqrt(Phi_kk) / sqrt(Sigma_jj)
        beta*_ij   = beta_ij   sqrt(Phi_jj) / sqrt(Phi_ii)

    with variances expressed as proportions of the total variance of the
    variable they belong to and covariances as correlations.

    Parameters
    ----------
    par : _Parameterisation
        The parameterisation.
    theta : numpy.ndarray
        Unstandardised estimates.

    Returns
    -------
    (dict, dict, numpy.ndarray, numpy.ndarray)
        Standardised values of the free parameters, of the fixed marker
        entries, and the implied ``Phi`` and ``Sigma``.

    References
    ----------
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        section 8.1. Wiley.
    """
    cm = par.cm
    Lambda, B, Psi, Theta = par.matrices(theta)
    with _quiet():
        Phi = par.phi(B, Psi)
        if Phi is None:
            Phi = np.eye(par.m)
        Sigma = _symmetrise(Lambda @ Phi @ Lambda.T + Theta)
        sd_f = np.sqrt(np.clip(np.diag(Phi), 1e-300, None))
        sd_y = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))

    free: Dict[str, float] = {}
    for idx, prm in enumerate(par.parameters):
        value = float(theta[idx])
        if prm.kind == KIND_LOADING:
            free[prm.label] = value * sd_f[prm.j] / sd_y[prm.i]
        elif prm.kind in (KIND_PATH, KIND_SECOND_ORDER):
            free[prm.label] = value * sd_f[prm.j] / sd_f[prm.i]
        elif prm.kind == KIND_LATENT_VARIANCE:
            free[prm.label] = (1.0 if not par.endogenous[prm.i]
                               else value / (sd_f[prm.i] ** 2))
        elif prm.kind == KIND_LATENT_COVARIANCE:
            free[prm.label] = value / (sd_f[prm.i] * sd_f[prm.j])
        elif prm.kind == KIND_RESIDUAL_VARIANCE:
            free[prm.label] = value / (sd_y[prm.i] ** 2)
        elif prm.kind == KIND_RESIDUAL_COVARIANCE:
            free[prm.label] = value / (sd_y[prm.i] * sd_y[prm.j])

    fixed: Dict[str, float] = {}
    for (j, f), value in par.lambda_fixed.items():
        label = f"{cm.factor_names[f]}=~{cm.item_names[j]}"
        fixed[label] = float(value) * sd_f[f] / sd_y[j]
    for (i, k), value in par.b_fixed.items():
        label = f"{cm.factor_names[i]}~{cm.factor_names[k]}"
        fixed[label] = float(value) * sd_f[k] / sd_f[i]
    return free, fixed, Phi, Sigma


# --------------------------------------------------------------------------- #
# Sample moments
# --------------------------------------------------------------------------- #


def _extract_matrix(data: pd.DataFrame, cm: "CompiledModel",
                    notes: List[str]) -> Tuple[np.ndarray, int]:
    """Pull the model's items out of a data frame as a complete-case array."""
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(np.asarray(data, dtype=float),
                                columns=list(cm.item_names))
        except Exception as exc:  # pragma: no cover - defensive
            raise SpecificationError(
                "The data could not be read as a table of item scores.",
                "تعذّر قراءة البيانات كجدول من درجات البنود.",
                detail=str(exc),
            ) from exc

    columns: List[Any] = []
    missing: List[str] = []
    for name, iid in zip(cm.item_names, cm.item_ids):
        if name in data.columns:
            columns.append(name)
        elif iid in data.columns:
            columns.append(iid)
        else:
            missing.append(name)
    if missing:
        raise SpecificationError(
            "The data does not contain a column for the following item(s): "
            + ", ".join(missing)
            + ". Make sure the column names match the item names in the model.",
            "لا تحتوي البيانات على أعمدة للبنود التالية: "
            + "، ".join(missing)
            + ". تأكد من تطابق أسماء الأعمدة مع أسماء البنود في النموذج.",
            missing=missing,
        )

    frame = data.loc[:, columns]
    try:
        values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise SpecificationError(
            "At least one item column contains non-numeric values, so a "
            "covariance matrix cannot be computed from it.",
            "أحد أعمدة البنود على الأقل يحتوي على قيم غير رقمية، لذلك لا يمكن "
            "حساب مصفوفة التغاير منه.",
            detail=str(exc),
        ) from exc

    n_rows = values.shape[0]
    complete = np.all(np.isfinite(values), axis=1)
    values = values[complete]
    dropped = int(n_rows - values.shape[0])
    if dropped:
        notes.append(_bilingual(
            f"LISTWISE_DELETION: {dropped} case(s) with missing values were "
            "excluded, leaving "
            f"{values.shape[0]} complete case(s).",
            f"الحذف بالقائمة: تم استبعاد {dropped} حالة تحتوي على قيم مفقودة، "
            f"وبقيت {values.shape[0]} حالة كاملة.",
        ))

    n = int(values.shape[0])
    p = cm.n_items
    if n < 3:
        raise SpecificationError(
            f"Only {n} complete case(s) are available; a covariance matrix "
            "cannot be estimated.",
            f"لا تتوفر سوى {n} حالة كاملة، ولا يمكن حساب مصفوفة التغاير منها.",
            n=n,
        )
    if n <= p:
        raise IdentificationError(
            f"The number of complete cases ({n}) does not exceed the number of "
            f"items ({p}), so the sample covariance matrix is singular. "
            "Increase the sample size or reduce the number of items.",
            f"عدد الحالات الكاملة ({n}) لا يزيد على عدد البنود ({p})، لذلك تكون "
            "مصفوفة تغاير العينة مفردة. ارفع حجم العينة أو قلّل عدد البنود.",
            n=n, p=p,
        )
    return values, n


def _resolve_polychoric(spec: ModelSpec, estimator: EstimatorKind,
                        use_polychoric: Optional[bool]) -> bool:
    """Decide whether the fit should run on a polychoric correlation matrix."""
    if use_polychoric is not None:
        return bool(use_polychoric)
    if estimator is not EstimatorKind.DWLS:
        return False
    if not spec.analysis.use_polychoric_for_ordinal:
        return False
    items = spec.all_items()
    return bool(items) and all(it.scale.kind is ScaleKind.LIKERT
                               for it in items)


def _sample_moments(values: np.ndarray, cm: "CompiledModel",
                    want_polychoric: bool, notes: List[str]) -> np.ndarray:
    """Sample covariance matrix, or the polychoric correlation matrix.

    The polychoric estimator lives in :mod:`semsim.analysis.polychoric`, which
    is imported lazily and duck-typed so this module keeps working while that
    one is still being written; a Pearson covariance matrix is then used and
    the substitution is recorded in the notes.
    """
    if want_polychoric:
        try:  # pragma: no cover - depends on a sibling module
            from semsim.analysis import polychoric as _poly  # type: ignore
            frame = pd.DataFrame(values, columns=list(cm.item_names))
            matrix = np.asarray(
                getattr(_poly, "polychoric_matrix")(frame), dtype=float)
            if matrix.shape == (cm.n_items, cm.n_items):
                notes.append(_bilingual(
                    "POLYCHORIC: the model was fitted to the polychoric "
                    "correlation matrix of the ordinal items.",
                    "ارتباط متعدد التصنيف: تم تقدير النموذج على مصفوفة الارتباط "
                    "متعدد التصنيف للبنود الرتبية.",
                ))
                return _symmetrise(matrix)
        except Exception as exc:
            notes.append(_bilingual(
                "POLYCHORIC_UNAVAILABLE: the polychoric correlation matrix "
                "could not be computed, so the Pearson covariance matrix was "
                f"used instead ({type(exc).__name__}).",
                "تعذّر حساب مصفوفة الارتباط متعدد التصنيف، فتم استخدام مصفوفة "
                f"تغاير بيرسون بدلًا منها ({type(exc).__name__}).",
            ))
    with _quiet():
        cov = np.cov(values, rowvar=False, ddof=1)
    return _symmetrise(np.atleast_2d(np.asarray(cov, dtype=float)))


# --------------------------------------------------------------------------- #
# Reporting tables
# --------------------------------------------------------------------------- #

_TABLE_COLUMNS = ["est", "se", "z", "p", "std"]


def _row(label: str, kind: str, lhs: str, rhs: str, lhs_id: str, rhs_id: str,
         est: float, se: float, z: float, pval: float, std: float,
         fixed: bool) -> Dict[str, Any]:
    """One row of a reporting table."""
    return {
        "label": label, "kind": kind,
        "lhs": lhs, "rhs": rhs, "lhs_id": lhs_id, "rhs_id": rhs_id,
        "est": float(est), "se": float(se), "z": float(z), "p": float(pval),
        "std": float(std), "fixed": bool(fixed),
    }


def _build_tables(par: _Parameterisation, theta: np.ndarray,
                  se: Dict[str, float], z: Dict[str, float],
                  pv: Dict[str, float], std_free: Dict[str, float],
                  std_fixed: Dict[str, float]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                             pd.DataFrame]:
    """Assemble the loading, structural and variance tables."""
    cm = par.cm
    nan = float("nan")

    loading_rows: List[Dict[str, Any]] = []
    for (j, f), value in sorted(par.lambda_fixed.items()):
        label = f"{cm.factor_names[f]}=~{cm.item_names[j]}"
        loading_rows.append(_row(
            label, KIND_LOADING, cm.item_names[j], cm.factor_names[f],
            cm.item_ids[j], cm.factor_ids[f], value, nan, nan, nan,
            std_fixed.get(label, nan), True))
    for idx, prm in enumerate(par.parameters):
        if prm.kind != KIND_LOADING:
            continue
        loading_rows.append(_row(
            prm.label, KIND_LOADING, prm.lhs, prm.rhs, prm.lhs_id, prm.rhs_id,
            float(theta[idx]), se.get(prm.label, nan), z.get(prm.label, nan),
            pv.get(prm.label, nan), std_free.get(prm.label, nan), False))
    loadings = pd.DataFrame(loading_rows)
    if not loadings.empty:
        loadings = loadings.sort_values(
            by=["rhs", "lhs"], kind="stable").reset_index(drop=True)
        loadings = loadings.rename(columns={"lhs": "item", "rhs": "factor"})
        loadings = loadings[["item", "factor"] + _TABLE_COLUMNS
                            + ["label", "kind", "lhs_id", "rhs_id", "fixed"]]

    path_rows: List[Dict[str, Any]] = []
    for (i, k), value in sorted(par.b_fixed.items()):
        label = f"{cm.factor_names[i]}~{cm.factor_names[k]}"
        path_rows.append(_row(
            label, KIND_SECOND_ORDER, cm.factor_names[i], cm.factor_names[k],
            cm.factor_ids[i], cm.factor_ids[k], value, nan, nan, nan,
            std_fixed.get(label, nan), True))
    for idx, prm in enumerate(par.parameters):
        if prm.kind not in (KIND_PATH, KIND_SECOND_ORDER):
            continue
        path_rows.append(_row(
            prm.label, prm.kind, prm.lhs, prm.rhs, prm.lhs_id, prm.rhs_id,
            float(theta[idx]), se.get(prm.label, nan), z.get(prm.label, nan),
            pv.get(prm.label, nan), std_free.get(prm.label, nan), False))
    paths = pd.DataFrame(path_rows)
    if not paths.empty:
        paths = paths.rename(columns={"rhs": "source", "lhs": "target"})
        paths = paths.sort_values(by=["kind", "source", "target"],
                                  kind="stable").reset_index(drop=True)
        paths = paths[["source", "target"] + _TABLE_COLUMNS
                      + ["label", "kind", "lhs_id", "rhs_id", "fixed"]]
        second_order = paths[paths["kind"] == KIND_SECOND_ORDER].reset_index(
            drop=True)
    else:
        second_order = paths

    variance_rows: List[Dict[str, Any]] = []
    for idx, prm in enumerate(par.parameters):
        if prm.kind in (KIND_LOADING, KIND_PATH, KIND_SECOND_ORDER):
            continue
        variance_rows.append(_row(
            prm.label, prm.kind, prm.lhs, prm.rhs, prm.lhs_id, prm.rhs_id,
            float(theta[idx]), se.get(prm.label, nan), z.get(prm.label, nan),
            pv.get(prm.label, nan), std_free.get(prm.label, nan), False))
    variances = pd.DataFrame(variance_rows)
    return loadings, paths, second_order, variances


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #


def estimate(data: pd.DataFrame, spec: ModelSpec, *,
             estimator: EstimatorKind = EstimatorKind.ML,
             use_polychoric: Optional[bool] = None,
             start_from: Optional["CompiledModel"] = None,
             observed_se: bool = True) -> EstimationResult:
    """Fit ``spec`` to ``data`` and return the full parameter solution.

    The model is identified by the marker-variable scheme (the first loading
    of every factor, and the second-order loading of every second-order
    construct's first dimension, fixed to one; every latent and disturbance
    variance free) and the solution is standardised afterwards for reporting.
    Latent interactions are omitted, because a product of two latent variables
    has no representation in a covariance structure; the omission is named in
    :attr:`EstimationResult.message`.

    Optimisation starts from the compiled population values, rescaled into the
    marker metric, and falls back on a two-step factor-analytic start.  Within
    each start L-BFGS-B is tried first, then Nelder-Mead, then Powell.
    Standard errors come from the expected (Fisher) information,
    ``ACOV = (2 / (N - 1)) [Delta' (D' (Sigma^-1 (x) Sigma^-1) D) Delta]^-1``;
    the observed-information counterpart ``(2 / (N - 1)) H^-1`` is computed
    alongside it and reported in
    :attr:`EstimationResult.standard_errors_observed`.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed item scores.  Columns are matched on item name first and item
        id second; cases with missing values are dropped listwise.
    spec : ModelSpec
        The model to fit.
    estimator : EstimatorKind, default ``EstimatorKind.ML``
        ``ML`` and ``MLR`` minimise ``F_ML``, ``ULS`` minimises
        ``0.5 tr((S - Sigma)^2)`` and ``DWLS`` the diagonally weighted
        residual sum of squares.
    use_polychoric : bool, optional
        Fit the polychoric correlation matrix instead of the Pearson
        covariance matrix.  ``None`` enables it for ``DWLS`` on fully ordinal
        data when ``spec.analysis.use_polychoric_for_ordinal`` is set.
    start_from : CompiledModel, optional
        A pre-compiled model supplying both the matrix patterns and the
        starting values.  Compiled from ``spec`` when omitted.
    observed_se : bool, default True
        Also compute the observed-information standard errors.  Skipped
        automatically for models with more than 150 free parameters.

    Returns
    -------
    EstimationResult
        Estimates, standard errors, Wald tests, the standardised solution,
        per-construct ``R**2`` and the reporting tables.

    Raises
    ------
    semsim.core.errors.SpecificationError
        The data lacks an item column, holds non-numeric values, or leaves too
        few complete cases.
    semsim.core.errors.IdentificationError
        A latent variable has no scale-fixing indicator, the sample is smaller
        than the number of items, or the degrees of freedom are negative.
    semsim.core.errors.ConvergenceError
        The discrepancy function could not be evaluated anywhere, so no
        solution exists to report.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from semsim.core import examples
    >>> from semsim.core.compile import compile_model
    >>> from semsim.core.implied import implied_covariance
    >>> spec = examples.minimal_cfa()
    >>> cm = compile_model(spec)
    >>> rng = np.random.default_rng(7)
    >>> x = rng.multivariate_normal(np.zeros(cm.n_items),
    ...                             implied_covariance(cm), size=800)
    >>> result = estimate(pd.DataFrame(x, columns=cm.item_names), spec)
    >>> result.df
    8

    References
    ----------
    Joreskog, K. G. (1969). A general approach to confirmatory maximum
        likelihood factor analysis. *Psychometrika*, 34(2), 183-202.
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        chapters 4 and 8. Wiley.
    Rosseel, Y. (2012). lavaan: An R package for structural equation modeling.
        *Journal of Statistical Software*, 48(2), 1-36.
    """
    notes: List[str] = []
    cm = start_from if start_from is not None else compile_model(spec)
    par = _build_parameterisation(cm, notes)

    if spec.interactions:
        notes.append(_bilingual(
            f"INTERACTION_NOT_ESTIMATED: {len(spec.interactions)} latent "
            "interaction term(s) cannot be fitted from a covariance matrix and "
            "were omitted from the estimated model; only the main effects were "
            "estimated.",
            f"حد التفاعل غير مُقدَّر: لا يمكن تقدير {len(spec.interactions)} من "
            "حدود التفاعل الكامنة انطلاقًا من مصفوفة التغاير، لذلك استُثنيت من "
            "النموذج المُقدَّر، ولم تُقدَّر إلا الآثار الرئيسة.",
        ))

    values, n = _extract_matrix(data, cm, notes)
    want_poly = _resolve_polychoric(spec, estimator, use_polychoric)
    S = _sample_moments(values, cm, want_poly, notes)

    logdet_sample, chol_sample = _logdet_cholesky(S)
    if chol_sample is None:
        raise SpecificationError(
            "The sample covariance matrix is singular, usually because an item "
            "is constant or two items are identical. Remove the constant or "
            "duplicated items and estimate again.",
            "مصفوفة تغاير العينة مفردة (غير قابلة للعكس)، وغالبًا لأن أحد البنود "
            "ثابت أو لأن بندين متطابقان تمامًا. احذف البنود الثابتة أو المكررة ثم "
            "أعد التقدير.",
        )

    n_moments = par.p * (par.p + 1) // 2
    n_free = par.q
    df = n_moments - n_free
    if df < 0:
        raise IdentificationError(
            f"The model has {n_free} free parameters but the covariance matrix "
            f"supplies only {n_moments} moments, so the degrees of freedom are "
            f"{df}. Fix some parameters or add items.",
            f"النموذج يحتوي على {n_free} معلمة حرة بينما لا توفر مصفوفة التغاير "
            f"سوى {n_moments} معلومة، فدرجات الحرية تساوي {df}. ثبّت بعض المعالم "
            "أو أضف بنودًا.",
            n_free=n_free, n_moments=n_moments, df=df,
        )

    problem = _Problem(par=par, kind=estimator, sample_cov=S,
                       logdet_sample=float(logdet_sample),
                       weights=_normal_theory_weights(S))

    candidates: List[Tuple[str, np.ndarray]] = [
        ("compiled", par.start_vector()),
        ("two_step", _two_step_start(par, S, values)),
    ]
    best: Optional[Tuple[np.ndarray, float, bool, int, str, str]] = None
    for name, x0 in candidates:
        theta, fmin, converged, n_iter, method = _run_optimiser(problem, x0)
        candidate = (theta, fmin, converged, n_iter, method, name)
        if best is None:
            best = candidate
        else:
            better = ((converged and not best[2])
                      or (converged == best[2] and np.isfinite(fmin)
                          and fmin < best[1]))
            if better:
                best = candidate
        if converged:
            break
    assert best is not None  # one candidate is always produced
    theta_hat, fmin, converged, n_iter, method, start_used = best

    if not np.isfinite(fmin):
        raise ConvergenceError(
            "The discrepancy function could not be evaluated at any parameter "
            "values, because the model-implied covariance matrix never became "
            "positive definite. Review the model specification and the "
            "starting values.",
            "تعذّر حساب دالة التناقض عند أي قيم للمعالم، لأن مصفوفة التغاير "
            "المُشتقة من النموذج لم تكن موجبة التحديد في أي مرة. راجع مواصفات "
            "النموذج وقيم البداية.",
            estimator=str(estimator.value),
        )
    fmin = max(float(fmin), 0.0)
    if not converged:
        notes.append(_bilingual(
            "NOT_CONVERGED: none of L-BFGS-B, Nelder-Mead or Powell reported a "
            "successful termination; the reported solution is the best one "
            "found and should be treated with caution.",
            "لم يتحقق التقارب: لم يُبلّغ أي من خوارزميات L-BFGS-B أو نيلدر-ميد أو "
            "باول عن انتهاء ناجح؛ الحل المعروض هو أفضل ما تم الوصول إليه ويجب "
            "التعامل معه بحذر.",
        ))
    notes.append(_bilingual(
        f"OPTIMISER: {method} from the {start_used} start, F = {fmin:.6g}.",
        f"خوارزمية التحسين: {method} انطلاقًا من قيم البداية "
        f"({start_used})، وقيمة دالة التناقض {fmin:.6g}.",
    ))

    sigma_hat = par.sigma(theta_hat)
    if sigma_hat is None:  # pragma: no cover - guarded by the finite fmin above
        raise ConvergenceError(
            "The model-implied covariance matrix at the optimum is singular.",
            "مصفوفة التغاير المُشتقة عند الحل الأمثل مفردة (غير قابلة للعكس).",
        )

    # -- standard errors ---------------------------------------------------- #
    labels = par.labels
    delta = _sigma_jacobian(par, theta_hat)
    acov, used_pinv = expected_information_acov(delta, sigma_hat, n)
    with _quiet():
        variances = np.diag(acov).astype(float)
        se_values = np.sqrt(np.where(variances > 0.0, variances, np.nan))
    if used_pinv:
        notes.append(_bilingual(
            "SINGULAR_INFORMATION: the expected information matrix was "
            "singular, so the standard errors were obtained from a "
            "Moore-Penrose pseudo-inverse; one or more parameters may be "
            "empirically under-identified.",
            "مصفوفة المعلومات المتوقعة مفردة، لذلك تم حساب الأخطاء المعيارية "
            "باستخدام المعكوس المعمَّم (مور-بنروز)؛ وقد يكون أحد المعالم أو أكثر "
            "غير محدَّد تجريبيًا.",
        ))

    se = {lab: float(v) for lab, v in zip(labels, se_values)}
    with _quiet():
        z_values = {lab: float(theta_hat[k] / se_values[k])
                    for k, lab in enumerate(labels)}
        p_values = {lab: float(2.0 * _norm.sf(abs(z_values[lab])))
                    for lab in labels}

    se_observed: Dict[str, float] = {}
    acov_observed: Optional[np.ndarray] = None
    if observed_se and n_free <= 150 and estimator in (EstimatorKind.ML,
                                                       EstimatorKind.MLR,
                                                       EstimatorKind.ULS,
                                                       EstimatorKind.DWLS):
        hessian = _numerical_hessian(problem, theta_hat)
        acov_observed, obs_pinv = observed_information_acov(hessian, n)
        with _quiet():
            obs_var = np.diag(acov_observed).astype(float)
            obs_se = np.sqrt(np.where(obs_var > 0.0, obs_var, np.nan))
        se_observed = {lab: float(v) for lab, v in zip(labels, obs_se)}
        if obs_pinv:
            notes.append(_bilingual(
                "SINGULAR_OBSERVED_INFORMATION: the numerical Hessian was "
                "singular, so the observed-information standard errors used a "
                "pseudo-inverse.",
                "مصفوفة هسه العددية مفردة، لذلك استُخدم المعكوس المعمَّم في حساب "
                "الأخطاء المعيارية المبنية على المعلومات المشاهدة.",
            ))
    elif observed_se:
        notes.append(_bilingual(
            f"OBSERVED_SE_SKIPPED: the model has {n_free} free parameters, so "
            "the numerical Hessian was not computed.",
            f"تم تخطي الأخطاء المعيارية المشاهدة: النموذج يحتوي على {n_free} "
            "معلمة حرة، لذلك لم تُحسب مصفوفة هسه العددية.",
        ))

    if estimator in (EstimatorKind.ULS, EstimatorKind.DWLS):
        notes.append(_bilingual(
            "NORMAL_THEORY_SE: the standard errors use the normal-theory "
            "(maximum likelihood) information matrix, which is an "
            "approximation for a least-squares fit.",
            "الأخطاء المعيارية محسوبة بمصفوفة المعلومات وفق نظرية التوزيع "
            "الطبيعي (الإمكان الأقصى)، وهي تقريب في حالة التقدير بالمربعات "
            "الصغرى.",
        ))
    if estimator is EstimatorKind.MLR:
        notes.append(_bilingual(
            "MLR_NOT_ROBUST: the point estimates are maximum likelihood, but "
            "the Satorra-Bentler robust corrections are not applied by this "
            "module.",
            "مقدّرات MLR: القيم المقدَّرة هي مقدّرات الإمكان الأقصى، لكن تصحيحات "
            "ساتورا-بنتلر القوية غير مطبَّقة في هذه الوحدة.",
        ))

    # -- standardisation, R-squared and tables ------------------------------ #
    std_free, std_fixed, Phi_hat, _ = _standardise(par, theta_hat)
    _, _, Psi_hat, _ = par.matrices(theta_hat)
    r_squared: Dict[str, float] = {}
    r_squared_by_id: Dict[str, float] = {}
    with _quiet():
        for f in range(par.m):
            if not par.endogenous[f]:
                continue
            total = float(Phi_hat[f, f])
            value = (1.0 - float(Psi_hat[f, f]) / total
                     if total > 0.0 else float("nan"))
            r_squared[cm.factor_names[f]] = value
            r_squared_by_id[cm.factor_ids[f]] = value

    loadings, paths, second_order, variance_table = _build_tables(
        par, theta_hat, se, z_values, p_values, std_free, std_fixed)

    standardised_all = dict(std_free)
    standardised_all.update(std_fixed)
    fixed_parameters: Dict[str, float] = {}
    for (j, f), value in par.lambda_fixed.items():
        fixed_parameters[f"{cm.factor_names[f]}=~{cm.item_names[j]}"] = float(value)
    for (i, k), value in par.b_fixed.items():
        fixed_parameters[f"{cm.factor_names[i]}~{cm.factor_names[k]}"] = float(value)

    return EstimationResult(
        converged=bool(converged),
        n_iter=int(n_iter),
        estimator=estimator,
        n=int(n),
        n_free=int(n_free),
        df=int(df),
        fmin=float(fmin),
        sigma_hat=sigma_hat,
        sample_cov=S,
        parameters={lab: float(theta_hat[k]) for k, lab in enumerate(labels)},
        standard_errors=se,
        z_values=z_values,
        p_values=p_values,
        standardised=std_free,
        loadings=loadings,
        path_estimates=paths,
        r_squared=r_squared,
        message=" ;; ".join(notes),
        chi_square=float(max(n - 1, 1) * fmin),
        item_names=list(cm.item_names),
        factor_names=list(cm.factor_names),
        theta=np.asarray(theta_hat, dtype=float),
        parameter_labels=list(labels),
        parameter_kinds={prm.label: prm.kind for prm in par.parameters},
        fixed_parameters=fixed_parameters,
        standardised_all=standardised_all,
        standard_errors_observed=se_observed,
        acov=acov,
        acov_observed=acov_observed,
        se_method=("expected_information_pinv" if used_pinv
                   else "expected_information"),
        optimiser=method,
        start_used=start_used,
        r_squared_by_id=r_squared_by_id,
        second_order_loadings=second_order,
        latent_covariance=Phi_hat,
        warnings=list(cm.warnings),
        _variance_table=variance_table,
    )

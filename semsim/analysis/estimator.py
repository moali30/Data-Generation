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

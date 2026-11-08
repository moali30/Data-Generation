"""
semsim.targeting.cudeck_browne
==============================
Inject *model error* into a population covariance matrix so that a simulated
dataset reproduces a researcher-chosen degree of misfit instead of the perfect
fit that naive simulation produces.

The problem
-----------
If data are generated from ``Sigma(theta)`` and then the very model that
produced ``Sigma(theta)`` is fitted back to them, the only misfit left is
sampling error.  Chi-square is then a pure sampling fluctuation and RMSEA
drifts towards zero as ``N`` grows, which is not what any real questionnaire
study looks like.  The researcher wants a population in which the model is a
*useful approximation* rather than the literal truth -- "good fit", not perfect
fit.

The principled solution
-----------------------
Cudeck & Browne (1992) construct a generating matrix

.. math::

    \\Sigma_0 = \\Sigma(\\theta) + k E

that satisfies **two** conditions simultaneously:

1. the population discrepancy is exactly the requested amount,
   ``F_ML(Sigma_0, Sigma(theta)) = F_0 = df * target_rmsea**2`` (Browne &
   Cudeck, 1993, define the population RMSEA as ``sqrt(F_0 / df)``);
2. ``theta`` is still the **exact minimiser** of
   ``F_ML(Sigma_0, Sigma(.))`` -- the fitted parameters of a study run on this
   population are the parameters the researcher asked for, not something
   shifted by the injected error.

The second condition is what separates this from a fudge, and it is a pure
first-order condition.  Writing ``Delta = d vech(Sigma) / d theta'`` and
``A = Sigma^-1``, the gradient of the discrepancy with respect to ``theta`` at
``theta`` itself is

.. math::

    \\partial F / \\partial \\theta_i
      = \\mathrm{tr}\\!\\left(A (\\Sigma - \\Sigma_0) A\\,
        \\partial \\Sigma / \\partial \\theta_i\\right)
      = -2 k\\, e' W \\Delta_i ,
    \\qquad W = \\tfrac12 D' (A \\otimes A) D ,

with ``e = vech(E)`` and ``D`` the duplication matrix.  So ``theta`` stays a
stationary point for **every** ``k`` as soon as ``Delta' W e = 0``: the
perturbation has to be orthogonal to the model's tangent space in the ML
metric.  That is achieved by removing the oblique projection onto the tangent
space from a random symmetric direction,

.. math::

    P = \\Delta (\\Delta' W \\Delta)^{-1} \\Delta' W , \\qquad
    e = (I - P)\\, \\mathrm{vech}(E_{\\text{random}}) .

Because ``F_ML`` is strictly convex along the ray ``k \\mapsto \\Sigma + k E``
and vanishes with zero derivative at ``k = 0``, the stationary point is a
minimum and ``F(k)`` is strictly increasing in ``k > 0``; the scalar ``k`` is
therefore found by a one-dimensional root solve (Brent's method), with the
upper bracket shrunk whenever ``Sigma + k E`` loses positive definiteness.

Because every residual variance is a free parameter, the tangent space spans
all ``p`` diagonal directions of moment space; the projection therefore strips
the diagonal out of ``E`` and ``Sigma_0`` keeps the unit diagonal that the rest
of the engine relies on.

What this module does *not* do
------------------------------
It works entirely in the population.  The realised sample RMSEA of a finite
dataset drawn from ``Sigma_0`` still carries sampling error on top of ``F_0``;
closing that loop is :mod:`semsim.targeting.calibration`'s job.

References
----------
Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
    yields a specified minimizer and a specified minimum discrepancy function
    value. *Psychometrika*, 57(3), 357-369.
Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model fit.
    In K. A. Bollen & J. S. Long (Eds.), *Testing Structural Equation Models*
    (pp. 136-162). Sage.
Steiger, J. H., & Lind, J. C. (1980). Statistically based tests for the number
    of common factors. Paper presented at the annual meeting of the
    Psychometric Society, Iowa City, IA.
Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
    Applications in Statistics and Econometrics*. Wiley.
"""

from __future__ import annotations

import warnings as _warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Callable, Dict, Iterator, List, Optional,
                    Tuple)

import numpy as np
from scipy import optimize as _opt

from semsim.core.errors import (ConvergenceError, InfeasibleTargetError,
                                SpecificationError)
from semsim.core.implied import implied_covariance, is_positive_definite

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from semsim.core.compile import CompiledModel


__all__ = [
    "DEFAULT_MIN_EIGENVALUE", "JACOBIAN_STEP",
    "METHOD_CUDECK_BROWNE", "METHOD_RESIDUAL", "METHOD_NONE", "METHODS",
    "ModelErrorResult", "FreeParameterMap",
    "inject_model_error",
    "population_discrepancy", "rmsea_from_discrepancy",
    "discrepancy_for_rmsea",
    "vech", "unvech", "duplication_matrix", "ml_weight_matrix",
    "free_parameter_map", "model_jacobian", "tangent_projector",
    "random_symmetric_direction", "perturbation_direction",
]


#: Smallest eigenvalue a generating matrix may have, relative to the mean
#: variance of ``Sigma(theta)``.  A matrix closer to singular than this is
#: useless for data generation (Cholesky factorisation and the Vale-Maurelli
#: intermediate-correlation solve both fail on it), so targets that would need
#: one are reported as infeasible rather than silently returned.
DEFAULT_MIN_EIGENVALUE = 1e-6

#: Relative central-difference step for ``d vech(Sigma) / d theta``.
JACOBIAN_STEP = 1e-6

METHOD_CUDECK_BROWNE = "cudeck_browne"
METHOD_RESIDUAL = "residual"
METHOD_NONE = "none"
METHODS = (METHOD_CUDECK_BROWNE, METHOD_RESIDUAL, METHOD_NONE)

#: ``FitTargetSpec.method`` is a free-text field in the spec; accept the
#: obvious spellings rather than failing on a hyphen.
_METHOD_ALIASES: Dict[str, str] = {
    "cudeck_browne": METHOD_CUDECK_BROWNE,
    "cudeck-browne": METHOD_CUDECK_BROWNE,
    "cudeckbrowne": METHOD_CUDECK_BROWNE,
    "cb": METHOD_CUDECK_BROWNE,
    "residual": METHOD_RESIDUAL,
    "residuals": METHOD_RESIDUAL,
    "none": METHOD_NONE,
    "off": METHOD_NONE,
    "": METHOD_NONE,
}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def _bilingual(en: str, ar: str) -> str:
    """One warning string carrying both interface languages."""
    return f"{en} | {ar}"


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence numpy warnings while deliberately probing inadmissible values.

    The bracket search evaluates ``Sigma + k E`` at values of ``k`` that are
    known to be too large; a failed Cholesky factorisation there is an answer,
    not an error, so the noise is suppressed and ``+inf`` returned instead.
    """
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            yield


def _symmetrise(a: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (a + a.T)`` as a fresh float array."""
    m = np.asarray(a, dtype=float)
    return 0.5 * (m + m.T)


def _vech_indices(p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Row/column indices of ``vech`` in lower-triangle, column-wise order."""
    p = int(p)
    rows: List[int] = []
    cols: List[int] = []
    for j in range(p):
        for i in range(j, p):
            rows.append(i)
            cols.append(j)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


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
        ``a[0,0], a[1,0], ..., a[p-1,0], a[1,1], ...``.  The ordering is the
        same one :mod:`semsim.analysis.estimator` uses, so Jacobians and
        weight matrices are interchangeable between the two modules.

    References
    ----------
    Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
        Applications in Statistics and Econometrics*, chapter 3. Wiley.
    """
    m = np.asarray(a, dtype=float)
    rows, cols = _vech_indices(m.shape[0])
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
    if rows.size != vec.size:
        raise SpecificationError(
            f"A vech vector of length {vec.size} does not describe a "
            f"{p} x {p} symmetric matrix.",
            f"متجه نصف التشكيل بطول {vec.size} لا يصف مصفوفة متماثلة "
            f"بأبعاد {p} × {p}.",
            length=int(vec.size), p=int(p),
        )
    out = np.zeros((p, p), dtype=float)
    out[rows, cols] = vec
    out[cols, rows] = vec
    return out


def duplication_matrix(p: int) -> np.ndarray:
    """Duplication matrix ``D`` satisfying ``vec(A) = D vech(A)``.

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
    >>> bool(np.allclose(duplication_matrix(2) @ vech(a),
    ...                  a.reshape(-1, order="F")))
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


def ml_weight_matrix(sigma: np.ndarray) -> np.ndarray:
    """Maximum-likelihood weight matrix ``W = 0.5 D' (Sigma^-1 (x) Sigma^-1) D``.

    This is the metric in which the ML discrepancy is locally a weighted sum of
    squared moment residuals, and therefore the metric in which the Cudeck-Browne
    perturbation must be orthogonal to the model's tangent space.

    The Kronecker product is never formed.  With ``A = Sigma^-1`` the entry of
    ``D' (A (x) A) D`` indexed by the ``vech`` positions ``(i, j)`` and
    ``(k, l)`` is ``(2 - delta_ij)(2 - delta_kl) / 2`` times
    ``A_ik A_jl + A_il A_jk``, which is evaluated here by fancy indexing in
    ``O(p**4)`` time and ``O(p**4 / 4)`` memory instead of ``O(p**6)``.

    Parameters
    ----------
    sigma : numpy.ndarray
        ``(p, p)`` symmetric positive-definite matrix.

    Returns
    -------
    numpy.ndarray
        ``(p*, p*)`` symmetric positive-definite weight matrix, with
        ``p* = p (p + 1) / 2``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``sigma`` is singular, so no ML metric exists.

    References
    ----------
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    Magnus, J. R., & Neudecker, H. (1999). *Matrix Differential Calculus with
        Applications in Statistics and Econometrics*, chapter 3. Wiley.
    """
    s = _symmetrise(sigma)
    a = _safe_inverse(
        s,
        "The model-implied covariance matrix is singular, so the maximum "
        "likelihood weight matrix does not exist. Reduce the loadings or the "
        "residual correlations that make two items collinear.",
        "مصفوفة التغاير المُشتقة من النموذج مفردة (غير قابلة للعكس)، لذلك لا "
        "يمكن تكوين مصفوفة أوزان الإمكان الأقصى. خفّض التشبعات أو ارتباطات "
        "البواقي التي تجعل بندين متطابقين خطيًا.",
    )
    rows, cols = _vech_indices(s.shape[0])
    i = rows[:, None]
    j = cols[:, None]
    k = rows[None, :]
    l = cols[None, :]
    term = a[i, k] * a[j, l] + a[i, l] * a[j, k]
    factor = 2.0 - (rows == cols).astype(float)
    w = 0.25 * factor[:, None] * factor[None, :] * term
    return _symmetrise(w)


def _safe_inverse(a: np.ndarray, message_en: str, message_ar: str) -> np.ndarray:
    """``inv(a)`` with a user-facing error instead of a raw ``LinAlgError``."""
    try:
        out = np.linalg.inv(_symmetrise(a))
    except np.linalg.LinAlgError as exc:
        raise SpecificationError(message_en, message_ar, detail=str(exc)) from exc
    if not np.all(np.isfinite(out)):
        raise SpecificationError(message_en, message_ar,
                                 detail="non-finite inverse")
    return _symmetrise(out)


def _logdet(a: np.ndarray) -> float:
    """``log|a|`` by Cholesky, or ``-inf`` when ``a`` is not positive definite."""
    with _quiet():
        try:
            chol = np.linalg.cholesky(_symmetrise(a))
        except np.linalg.LinAlgError:
            return float("-inf")
        diag = np.diag(chol)
        if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
            return float("-inf")
        return 2.0 * float(np.sum(np.log(diag)))


# --------------------------------------------------------------------------- #
# The population discrepancy
# --------------------------------------------------------------------------- #


def population_discrepancy(sigma_0: np.ndarray, sigma: np.ndarray) -> float:
    """Maximum-likelihood discrepancy between two **population** matrices.

    Evaluates

    .. math::

        F(\\Sigma_0, \\Sigma)
          = \\log|\\Sigma| - \\log|\\Sigma_0|
            + \\mathrm{tr}(\\Sigma_0 \\Sigma^{-1}) - p ,

    which is the ordinary Wishart ML fit function with the "sample" matrix
    replaced by the generating matrix ``Sigma_0``.  It is zero if and only if
    ``Sigma_0 == Sigma`` and strictly positive otherwise, and it is exactly the
    quantity whose minimum Browne & Cudeck (1993) turn into the population
    RMSEA.

    Parameters
    ----------
    sigma_0 : numpy.ndarray
        ``(p, p)`` generating (perturbed) covariance matrix.
    sigma : numpy.ndarray
        ``(p, p)`` model-implied covariance matrix ``Sigma(theta)``.

    Returns
    -------
    float
        The discrepancy, or ``+inf`` when either matrix is not positive
        definite, so a bracket search can simply back off.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.eye(3)
    >>> float(population_discrepancy(s, s))
    0.0

    References
    ----------
    Joreskog, K. G. (1969). A general approach to confirmatory maximum
        likelihood factor analysis. *Psychometrika*, 34(2), 183-202.
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
        yields a specified minimizer and a specified minimum discrepancy
        function value. *Psychometrika*, 57(3), 357-369.
    """
    s0 = _symmetrise(sigma_0)
    s = _symmetrise(sigma)
    if s0.shape != s.shape:
        raise SpecificationError(
            "The two matrices passed to the discrepancy function have "
            f"different shapes ({s0.shape} and {s.shape}).",
            "المصفوفتان الممرّرتان إلى دالة التباعد مختلفتان في الأبعاد "
            f"({s0.shape} و{s.shape}).",
        )
    p = s.shape[0]
    with _quiet():
        logdet_sigma = _logdet(s)
        logdet_s0 = _logdet(s0)
        if not np.isfinite(logdet_sigma) or not np.isfinite(logdet_s0):
            return float("inf")
        try:
            solved = np.linalg.solve(s, s0)
        except np.linalg.LinAlgError:
            return float("inf")
        value = logdet_sigma - logdet_s0 + float(np.trace(solved)) - p
    return float(value) if np.isfinite(value) else float("inf")


def rmsea_from_discrepancy(fmin: float, df: int) -> float:
    """Population RMSEA implied by a discrepancy value.

    ``RMSEA = sqrt(F_0 / df)``, the population counterpart of Steiger & Lind's
    (1980) index as reparameterised by Browne & Cudeck (1993).

    Parameters
    ----------
    fmin : float
        Population minimum of the discrepancy function, ``F_0``.
    df : int
        Degrees of freedom of the model.

    Returns
    -------
    float
        The non-negative root; ``0.0`` when ``df <= 0`` or ``fmin <= 0``.

    References
    ----------
    Steiger, J. H., & Lind, J. C. (1980). Statistically based tests for the
        number of common factors. Psychometric Society, Iowa City, IA.
    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model
        fit. In *Testing Structural Equation Models* (pp. 136-162). Sage.
    """
    if df is None or int(df) <= 0:
        return 0.0
    value = float(fmin)
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(np.sqrt(value / float(int(df))))


def discrepancy_for_rmsea(target_rmsea: float, df: int) -> float:
    """Discrepancy value a target population RMSEA corresponds to.

    ``F_0 = df * target_rmsea**2``, the inverse of
    :func:`rmsea_from_discrepancy`.

    Parameters
    ----------
    target_rmsea : float
        Desired population RMSEA.
    df : int
        Degrees of freedom of the model.

    Returns
    -------
    float
        ``F_0``; ``0.0`` when either argument is non-positive.

    References
    ----------
    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model
        fit. In *Testing Structural Equation Models* (pp. 136-162). Sage.
    """
    if df is None or int(df) <= 0:
        return 0.0
    r = float(target_rmsea)
    if not np.isfinite(r) or r <= 0.0:
        return 0.0
    return float(int(df)) * r * r


# --------------------------------------------------------------------------- #
# The free-parameter mapping (theta -> Sigma) and its Jacobian
# --------------------------------------------------------------------------- #


@dataclass
class FreeParameterMap:
    """A minimal, identified parameterisation of a compiled model.

    Only one thing is asked of this object: that ``sigma_of(theta0)`` reproduce
    ``Sigma(theta)`` exactly and that the columns of
    ``d vech(sigma_of) / d theta`` span the model's tangent space at that
    point.  Any reparameterisation with those two properties yields the *same*
    tangent space and therefore the same Cudeck-Browne perturbation, which is
    why this module is happy to use whichever mapping is available.

    Attributes
    ----------
    labels : list of str
        One label per free parameter, in vector order.
    theta0 : numpy.ndarray
        The population parameter vector, i.e. the point whose tangent space is
        projected out.
    sigma_of : callable
        ``theta -> Sigma`` returning ``None`` for an inadmissible vector.
    bounds : list of (float or None, float or None)
        Box constraints, so a caller that wants to re-fit the model can do so
        without wandering into negative variances.
    source : {'estimator', 'builtin'}
        Where the mapping came from: :mod:`semsim.analysis.estimator`'s
        marker-variable parameterisation, or this module's own fallback.
    q : int
        Number of free parameters.
    """

    labels: List[str]
    theta0: np.ndarray
    sigma_of: Callable[[np.ndarray], Optional[np.ndarray]]
    bounds: List[Tuple[Optional[float], Optional[float]]]
    source: str
    q: int


def _estimator_parameter_map(cm: "CompiledModel") -> Optional[FreeParameterMap]:
    """Borrow :mod:`semsim.analysis.estimator`'s parameterisation, or give up.

    Imported lazily and defensively: the estimator is written by a different
    author on the same schedule as this module, so a missing module, a missing
    helper or a mapping that does not reproduce ``Sigma(theta)`` exactly all
    fall back on :func:`_builtin_parameter_map` instead of raising.
    """
    try:  # pragma: no cover - exercised only when the estimator exists
        from semsim.analysis import estimator as _est
    except Exception:
        return None

    builder = getattr(_est, "_build_parameterisation", None)
    if builder is None:
        builder = getattr(_est, "build_parameterisation", None)
    if builder is None:
        return None

    try:
        notes: List[str] = []
        par = builder(cm, notes)
        theta0 = np.asarray(par.start_vector(), dtype=float)
        sigma = par.sigma(theta0)
        if sigma is None:
            return None
        target = implied_covariance(cm)
        if not np.allclose(sigma, target, atol=1e-9, rtol=0.0):
            return None
        labels = [str(x) for x in par.labels]
        bounds = list(par.bounds())
    except Exception:
        return None

    return FreeParameterMap(
        labels=labels,
        theta0=theta0,
        sigma_of=par.sigma,
        bounds=bounds,
        source="estimator",
        q=int(theta0.size),
    )


def _second_order_children(cm: "CompiledModel") -> Dict[int, List[int]]:
    """Dimension indices grouped by the index of their parent construct."""
    out: Dict[int, List[int]] = {}
    spec = getattr(cm, "spec", None)
    if spec is None:
        return out
    index = dict(getattr(cm, "factor_index", {}) or {})
    for c in getattr(spec, "constructs", []) or []:
        dims = getattr(c, "dimensions", None) or []
        parent = index.get(getattr(c, "id", None))
        if parent is None or not dims:
            continue
        kids = [int(index[d.id]) for d in dims if getattr(d, "id", None) in index]
        if kids:
            out[int(parent)] = kids
    return out


def _builtin_parameter_map(cm: "CompiledModel") -> FreeParameterMap:
    """This module's own minimal parameterisation of a compiled model.

    Free: every non-zero loading in ``Lambda`` except one marker per factor,
    every non-zero structural coefficient in ``B`` except the first
    second-order loading of each second-order construct, every latent
    disturbance variance and specified latent covariance in ``Psi``, and every
    residual variance and specified residual covariance in ``Theta``.

    Scales are fixed the same way LISREL does it -- one indicator per latent --
    but at the *compiled* value rather than at one, so ``theta0`` is literally
    the compiled population solution and ``sigma_of(theta0)`` reproduces
    ``Sigma(theta)`` to machine precision.  Fixing a marker at ``c`` instead of
    at ``1`` is a rescaling of that latent variable, which changes neither the
    number of free parameters nor the tangent space.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.

    Returns
    -------
    FreeParameterMap
        With ``source == 'builtin'``.

    References
    ----------
    Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*,
        chapter 1 (scale fixing by a reference indicator). Scientific Software
        International.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        section 7.2. Wiley.
    """
    Lambda = np.asarray(cm.Lambda, dtype=float)
    B = np.asarray(cm.B, dtype=float)
    Psi = _symmetrise(cm.Psi)
    Theta = _symmetrise(cm.Theta)
    p, m = Lambda.shape

    factor_items: Dict[int, List[int]] = {}
    for j, f in enumerate(list(getattr(cm, "home_factor", []) or [])):
        factor_items.setdefault(int(f), []).append(int(j))

    children = _second_order_children(cm)

    lambda_fixed: Dict[Tuple[int, int], float] = {}
    b_fixed: Dict[Tuple[int, int], float] = {}
    psi_fixed: Dict[int, float] = {}

    for f in range(m):
        rows = factor_items.get(f, [])
        if rows:
            marker = max(rows, key=lambda r: abs(Lambda[r, f]))
            if abs(Lambda[marker, f]) > 1e-10:
                lambda_fixed[(marker, f)] = float(Lambda[marker, f])
                continue
        kids = children.get(f, [])
        picked = False
        for kid in kids:
            if abs(B[kid, f]) > 1e-10:
                b_fixed[(kid, f)] = float(B[kid, f])
                picked = True
                break
        if picked:
            continue
        # A latent with no usable indicator (a common-method factor, say):
        # fix its own variance instead.
        psi_fixed[f] = float(Psi[f, f]) if Psi[f, f] > 0.0 else 1.0

    kinds: List[str] = []
    labels: List[str] = []
    slots: List[Tuple[str, int, int]] = []
    start: List[float] = []
    factor_names = list(getattr(cm, "factor_names", []) or
                        [f"F{i}" for i in range(m)])
    item_names = list(getattr(cm, "item_names", []) or
                      [f"y{i}" for i in range(p)])

    for j in range(p):
        for f in range(m):
            if Lambda[j, f] == 0.0 or (j, f) in lambda_fixed:
                continue
            slots.append(("lambda", j, f))
            kinds.append("loading")
            labels.append(f"{factor_names[f]}=~{item_names[j]}")
            start.append(float(Lambda[j, f]))

    for i in range(m):
        for k in range(m):
            if i == k or B[i, k] == 0.0 or (i, k) in b_fixed:
                continue
            slots.append(("B", i, k))
            kinds.append("structural")
            labels.append(f"{factor_names[i]}~{factor_names[k]}")
            start.append(float(B[i, k]))

    for f in range(m):
        if f in psi_fixed:
            continue
        slots.append(("psi_var", f, f))
        kinds.append("latent_variance")
        labels.append(f"{factor_names[f]}~~{factor_names[f]}")
        start.append(float(Psi[f, f]))

    for a in range(m):
        for b in range(a):
            if Psi[a, b] == 0.0:
                continue
            slots.append(("psi_cov", a, b))
            kinds.append("latent_covariance")
            labels.append(f"{factor_names[a]}~~{factor_names[b]}")
            start.append(float(Psi[a, b]))

    for j in range(p):
        slots.append(("theta_var", j, j))
        kinds.append("residual_variance")
        labels.append(f"{item_names[j]}~~{item_names[j]}")
        start.append(float(Theta[j, j]))

    for a in range(p):
        for b in range(a):
            if Theta[a, b] == 0.0:
                continue
            slots.append(("theta_cov", a, b))
            kinds.append("residual_covariance")
            labels.append(f"{item_names[a]}~~{item_names[b]}")
            start.append(float(Theta[a, b]))

    theta0 = np.asarray(start, dtype=float)
    variance_kinds = {"latent_variance", "residual_variance"}
    bounds: List[Tuple[Optional[float], Optional[float]]] = [
        (1e-8, None) if kind in variance_kinds else (None, None)
        for kind in kinds
    ]

    eye_m = np.eye(m)

    def sigma_of(theta: np.ndarray) -> Optional[np.ndarray]:
        x = np.asarray(theta, dtype=float).ravel()
        if x.size != len(slots):
            raise SpecificationError(
                f"Expected a parameter vector of length {len(slots)}, "
                f"received {x.size}.",
                f"المتوقع متجه معالم بطول {len(slots)}، والوارد فعليًا "
                f"{x.size}.",
            )
        lam = np.zeros((p, m), dtype=float)
        bb = np.zeros((m, m), dtype=float)
        psi = np.zeros((m, m), dtype=float)
        th = np.zeros((p, p), dtype=float)
        for (j, f), value in lambda_fixed.items():
            lam[j, f] = value
        for (i, k), value in b_fixed.items():
            bb[i, k] = value
        for f, value in psi_fixed.items():
            psi[f, f] = value
        for idx, (where, i, j) in enumerate(slots):
            value = float(x[idx])
            if where == "lambda":
                lam[i, j] = value
            elif where == "B":
                bb[i, j] = value
            elif where == "psi_var":
                psi[i, i] = value
            elif where == "psi_cov":
                psi[i, j] = psi[j, i] = value
            elif where == "theta_var":
                th[i, i] = value
            else:
                th[i, j] = th[j, i] = value
        with _quiet():
            try:
                inv = np.linalg.inv(eye_m - bb)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(inv)):
                return None
            phi = _symmetrise(inv @ psi @ inv.T)
            out = _symmetrise(lam @ phi @ lam.T + th)
        if not np.all(np.isfinite(out)):
            return None
        return out

    return FreeParameterMap(
        labels=labels,
        theta0=theta0,
        sigma_of=sigma_of,
        bounds=bounds,
        source="builtin",
        q=int(theta0.size),
    )


def free_parameter_map(cm: "CompiledModel",
                       *, prefer_estimator: bool = True) -> FreeParameterMap:
    """Obtain an identified ``theta -> Sigma`` mapping for a compiled model.

    Tries :mod:`semsim.analysis.estimator`'s marker-variable parameterisation
    first -- so the tangent space projected out here is the tangent space of
    the model that will actually be fitted -- and falls back on this module's
    own equivalent mapping when the estimator is absent, exposes no
    parameterisation, or returns one that does not reproduce ``Sigma(theta)``.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.
    prefer_estimator : bool, default True
        Set ``False`` to force the built-in mapping, which is useful for
        testing that the two agree.

    Returns
    -------
    FreeParameterMap
        Check ``.source`` to see which one was used.

    References
    ----------
    Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*.
        Scientific Software International.
    """
    if prefer_estimator:
        borrowed = _estimator_parameter_map(cm)
        if borrowed is not None:
            return borrowed
    return _builtin_parameter_map(cm)


def model_jacobian(pmap: FreeParameterMap, *,
                   theta: Optional[np.ndarray] = None,
                   step: float = JACOBIAN_STEP) -> np.ndarray:
    """``Delta = d vech(Sigma) / d theta'`` by central differences.

    Parameters
    ----------
    pmap : FreeParameterMap
        The parameterisation whose tangent space is wanted.
    theta : numpy.ndarray, optional
        Point at which to differentiate; ``pmap.theta0`` by default.
    step : float, default :data:`JACOBIAN_STEP`
        Relative finite-difference step, scaled by ``max(1, |theta_k|)``.

    Returns
    -------
    numpy.ndarray
        ``(p*, q)`` Jacobian with ``p* = p (p + 1) / 2``.

    Raises
    ------
    semsim.core.errors.ConvergenceError
        If the implied matrix could not be formed on both sides of the step for
        some parameter, so that column of the tangent space is unknown.

    References
    ----------
    Browne, M. W. (1974). Generalized least squares estimators in the analysis
        of covariance structures. *South African Statistical Journal*, 8, 1-24.
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
        yields a specified minimizer and a specified minimum discrepancy
        function value. *Psychometrika*, 57(3), 357-369.
    """
    base = np.asarray(pmap.theta0 if theta is None else theta,
                      dtype=float).ravel()
    sigma0 = pmap.sigma_of(base)
    if sigma0 is None:
        raise ConvergenceError(
            "The model-implied covariance matrix could not be evaluated at the "
            "population parameter values, so the model error cannot be "
            "projected out of the model's own directions.",
            "تعذّر حساب مصفوفة التغاير المُشتقة من النموذج عند قيم المعالم "
            "السكانية، لذلك لا يمكن إسقاط خطأ النموذج خارج اتجاهات النموذج "
            "نفسه.",
        )
    p_star = vech(sigma0).size
    delta = np.zeros((p_star, base.size), dtype=float)
    with _quiet():
        for k in range(base.size):
            h = float(step) * max(1.0, abs(float(base[k])))
            up = base.copy()
            down = base.copy()
            up[k] += h
            down[k] -= h
            s_up = pmap.sigma_of(up)
            s_down = pmap.sigma_of(down)
            if s_up is None or s_down is None:
                raise ConvergenceError(
                    "The derivative of the implied covariance matrix with "
                    f"respect to parameter {k + 1} "
                    f"({pmap.labels[k] if k < len(pmap.labels) else '?'}) "
                    "could not be computed, so the model's tangent space is "
                    "incomplete.",
                    "تعذّر حساب مشتقة مصفوفة التغاير المُشتقة من النموذج بالنسبة "
                    f"إلى المعلمة رقم {k + 1} "
                    f"({pmap.labels[k] if k < len(pmap.labels) else '؟'})، "
                    "لذلك فإن الفضاء المماسي للنموذج غير مكتمل.",
                    parameter=int(k),
                )
            delta[:, k] = (vech(s_up) - vech(s_down)) / (2.0 * h)
    if not np.all(np.isfinite(delta)):
        raise ConvergenceError(
            "The numerical Jacobian of the implied covariance matrix contains "
            "non-finite entries, so model error cannot be injected.",
            "المشتقة العددية لمصفوفة التغاير المُشتقة من النموذج تحتوي على قيم "
            "غير منتهية، لذلك لا يمكن إدخال خطأ النموذج.",
        )
    return delta


# --------------------------------------------------------------------------- #
# The perturbation direction
# --------------------------------------------------------------------------- #


def tangent_projector(delta: np.ndarray, w: np.ndarray,
                      *, warnings_out: Optional[List[str]] = None
                      ) -> np.ndarray:
    """Oblique projector onto the model's tangent space in the ML metric.

    ``P = Delta (Delta' W Delta)^-1 Delta' W`` projects a moment-space vector
    onto the span of the model's own derivative directions, measuring angles
    with ``W`` rather than with the identity.  ``P`` is idempotent and
    self-adjoint with respect to ``W`` (so ``W P`` is symmetric), which is
    exactly what makes ``(I - P) v`` orthogonal to every column of ``Delta``
    in that metric and hence leaves ``theta`` a stationary point.

    Parameters
    ----------
    delta : numpy.ndarray
        ``(p*, q)`` Jacobian ``d vech(Sigma) / d theta'``.
    w : numpy.ndarray
        ``(p*, p*)`` ML weight matrix from :func:`ml_weight_matrix`.
    warnings_out : list of str, optional
        Appended to in place when the information matrix ``Delta' W Delta`` is
        singular -- an empirically under-identified model -- and the
        Moore-Penrose pseudo-inverse has to be used instead.  The projector
        onto the column space of ``Delta`` is still well defined in that case,
        so the construction remains valid.

    Returns
    -------
    numpy.ndarray
        ``(p*, p*)`` projector.

    References
    ----------
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
        yields a specified minimizer and a specified minimum discrepancy
        function value. *Psychometrika*, 57(3), 357-369.
    Rao, C. R., & Mitra, S. K. (1971). *Generalized Inverse of Matrices and
        its Applications*, chapter 5. Wiley.
    """
    d = np.asarray(delta, dtype=float)
    weight = _symmetrise(w)
    dtw = d.T @ weight                      # (q, p*)
    information = _symmetrise(dtw @ d)      # (q, q)
    try:
        middle = np.linalg.solve(information, dtw)
        if not np.all(np.isfinite(middle)):
            raise np.linalg.LinAlgError("non-finite solution")
    except np.linalg.LinAlgError:
        middle = np.linalg.pinv(information, rcond=1e-12) @ dtw
        if warnings_out is not None:
            warnings_out.append(_bilingual(
                "SINGULAR_INFORMATION: the model's information matrix "
                "(Delta' W Delta) is singular, which means the model is "
                "empirically under-identified at these population values. The "
                "pseudo-inverse was used, so the injected error is still "
                "orthogonal to every identified direction, but check the "
                "identification diagnostics before trusting the parameters.",
                "مصفوفة المعلومات للنموذج (Delta' W Delta) مفردة، وهذا يعني أن "
                "النموذج غير مُحدَّد تجريبيًا عند هذه القيم السكانية. استُخدم "
                "المعكوس المعمَّم، فما زال الخطأ المُدخَل متعامدًا على كل "
                "الاتجاهات المُحدَّدة، لكن راجع تشخيصات التحديد قبل الاعتماد على "
                "المعالم.",
            ))
    return d @ middle


def random_symmetric_direction(p: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random symmetric ``(p, p)`` matrix from a Generator.

    Each entry of a ``(p, p)`` standard-normal matrix ``G`` is drawn from
    ``rng`` and the result is symmetrised, ``E = (G + G') / 2``.  This is the
    Gaussian orthogonal ensemble up to scale, so the direction is isotropic in
    the space of symmetric matrices and no pattern of misfit is privileged.

    **Reproducibility is the caller's to control**: the same ``rng`` state
    gives the same matrix, and two different seeds give two different -- but
    equally valid -- model-error directions with the *same* achieved RMSEA.
    That is the point of the construction; the degree of misfit is targeted,
    its shape is not.

    Parameters
    ----------
    p : int
        Side length.
    rng : numpy.random.Generator
        The explicit generator; never the global ``numpy.random`` state.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` symmetric matrix.

    References
    ----------
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
        yields a specified minimizer and a specified minimum discrepancy
        function value. *Psychometrika*, 57(3), 357-369.
    Mehta, M. L. (2004). *Random Matrices* (3rd ed.), chapter 3. Elsevier.
    """
    if not hasattr(rng, "standard_normal"):
        raise SpecificationError(
            "A numpy.random.Generator is required to draw the model-error "
            "direction; global numpy.random state is never used because the "
            "whole simulation has to be reproducible from a seed.",
            "يلزم تمرير مُولِّد أرقام عشوائية من نوع numpy.random.Generator "
            "لسحب اتجاه خطأ النموذج؛ فالمُولِّد العام لا يُستخدم مطلقًا لأن "
            "المحاكاة بأكملها يجب أن تكون قابلة للتكرار من البذرة نفسها.",
        )
    g = np.asarray(rng.standard_normal((int(p), int(p))), dtype=float)
    return _symmetrise(g)


def perturbation_direction(cm: "CompiledModel", rng: np.random.Generator, *,
                           method: str = METHOD_CUDECK_BROWNE,
                           pmap: Optional[FreeParameterMap] = None,
                           sigma: Optional[np.ndarray] = None,
                           warnings_out: Optional[List[str]] = None
                           ) -> np.ndarray:
    """The matrix ``E`` along which model error is injected, normalised.

    For ``method='cudeck_browne'`` a random symmetric direction is projected
    out of the model's tangent space in the ML metric, which is what keeps
    ``theta`` the exact minimiser.  For ``method='residual'`` the random
    direction is used as drawn: cheaper, because no Jacobian and no weight
    matrix are needed, but the fitted parameters of the resulting population
    are then biased away from ``theta``.

    The returned matrix has unit Frobenius norm, so the scalar ``k`` found by
    :func:`inject_model_error` is directly comparable across models.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.
    rng : numpy.random.Generator
        Explicit generator for the random direction.
    method : {'cudeck_browne', 'residual'}, default 'cudeck_browne'
        Whether to project.
    pmap : FreeParameterMap, optional
        Re-use a previously built parameterisation.
    sigma : numpy.ndarray, optional
        Re-use a previously computed ``Sigma(theta)``.
    warnings_out : list of str, optional
        Appended to in place with bilingual notes.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` symmetric matrix of unit Frobenius norm.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        When the model is saturated (``df == 0``), so the tangent space fills
        the whole moment space and no orthogonal direction exists.

    References
    ----------
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
        yields a specified minimizer and a specified minimum discrepancy
        function value. *Psychometrika*, 57(3), 357-369.
    """
    s = implied_covariance(cm) if sigma is None else _symmetrise(sigma)
    p = s.shape[0]
    e_random = random_symmetric_direction(p, rng)

    if method == METHOD_RESIDUAL:
        residual = e_random
    else:
        if pmap is None:
            pmap = free_parameter_map(cm)
        delta = model_jacobian(pmap)
        weight = ml_weight_matrix(s)
        projector = tangent_projector(delta, weight, warnings_out=warnings_out)
        v = vech(e_random)
        residual = unvech(v - projector @ v, p)

    residual = _symmetrise(residual)
    norm = float(np.linalg.norm(residual, ord="fro"))
    reference = float(np.linalg.norm(e_random, ord="fro"))
    if not np.isfinite(norm) or norm <= 1e-10 * max(reference, 1.0):
        raise InfeasibleTargetError(
            "This model is saturated (zero degrees of freedom): its parameters "
            "can reproduce any covariance matrix exactly, so there is no "
            "direction left in which model error could be injected. Fit "
            "indices are degenerate for such a model. Remove a free parameter "
            "-- a residual covariance or a cross-loading -- or add items.",
            "هذا النموذج مُشبَع (درجات حريته صفر): معالمه قادرة على إعادة إنتاج "
            "أي مصفوفة تغاير إنتاجًا تامًا، لذلك لا يتبقى أي اتجاه يمكن إدخال "
            "خطأ النموذج فيه، ومؤشرات حسن المطابقة تفقد معناها في هذه الحالة. "
            "احذف معلمة حرة — تغايرَ بواقٍ أو تشبعًا متقاطعًا — أو أضف بنودًا "
            "جديدة.",
            projected_norm=norm, raw_norm=reference,
        )
    return residual / norm


# --------------------------------------------------------------------------- #
# The public result
# --------------------------------------------------------------------------- #


@dataclass
class ModelErrorResult:
    """Outcome of one model-error injection.

    Attributes
    ----------
    sigma_0 : numpy.ndarray
        The generating matrix ``Sigma_0 = Sigma(theta) + k E``, symmetric and
        positive definite.  This is what the data generator should sample from.
    achieved_rmsea : float
        ``sqrt(achieved_fmin / df)``: the population RMSEA a researcher fitting
        the model to an infinitely large sample from ``sigma_0`` would obtain.
    achieved_fmin : float
        ``F_ML(sigma_0, Sigma(theta))``, the population minimum of the
        discrepancy function.
    target_rmsea : float
        What was asked for, echoed back for reporting and calibration.
    scale : float
        The scalar ``k``: how far along the unit-norm perturbation direction
        the generating matrix sits.
    converged : bool
        Whether the root solve hit ``|F - F_0| <= tol * max(1, F_0)``.
    warnings : list of str
        Bilingual ``"English | Arabic"`` notes.
    df : int
        Degrees of freedom used to convert between ``F_0`` and RMSEA.
    n_free : int
        Number of free parameters in the mapping whose tangent space was
        projected out.
    method : str
        The resolved method: ``'cudeck_browne'``, ``'residual'`` or ``'none'``.
    parameterisation : str
        ``'estimator'``, ``'builtin'`` or ``''`` when no mapping was needed.
    direction : numpy.ndarray or None
        The unit-norm perturbation matrix ``E``, kept so a calibration loop can
        re-scale along the same direction without re-drawing it.
    min_eigenvalue : float
        Smallest eigenvalue of ``sigma_0``, a cheap conditioning report.
    iterations : int
        Function evaluations spent bracketing and solving for ``k``.
    """

    sigma_0: np.ndarray
    achieved_rmsea: float
    achieved_fmin: float
    target_rmsea: float
    scale: float
    converged: bool
    warnings: List[str] = field(default_factory=list)
    df: int = 0
    n_free: int = 0
    method: str = METHOD_CUDECK_BROWNE
    parameterisation: str = ""
    direction: Optional[np.ndarray] = None
    min_eigenvalue: float = float("nan")
    iterations: int = 0

    @property
    def rmsea_error(self) -> float:
        """``achieved_rmsea - target_rmsea``."""
        return float(self.achieved_rmsea - self.target_rmsea)

    @property
    def perturbed(self) -> bool:
        """Whether anything was actually injected."""
        return bool(self.scale != 0.0)


# --------------------------------------------------------------------------- #
# Internals of the scalar solve
# --------------------------------------------------------------------------- #


def _resolve_method(method: Optional[str]) -> str:
    """Normalise the free-text ``FitTargetSpec.method`` to a known constant."""
    key = str(method or "").strip().lower().replace(" ", "_")
    resolved = _METHOD_ALIASES.get(key)
    if resolved is None:
        raise SpecificationError(
            f"Unknown model-error method {method!r}. Choose "
            "'cudeck_browne' (keeps the specified parameters as the exact "
            "minimiser), 'residual' (a plain perturbation, slightly biased) "
            "or 'none'.",
            f"طريقة إدخال خطأ النموذج «{method}» غير معروفة. اختر "
            "«cudeck_browne» وهي التي تُبقي المعالم المحددة هي المُصغِّر التام "
            "لدالة التباعد، أو «residual» وهي اضطراب بسيط متحيّز قليلًا، أو "
            "«none» لتعطيل إدخال الخطأ.",
            method=method, allowed=list(METHODS),
        )
    return resolved


def _resolve_rng(cm: "CompiledModel",
                 rng: Optional[np.random.Generator],
                 warnings_out: List[str]) -> np.random.Generator:
    """Return the caller's Generator, or build one from ``spec.seed``.

    Reproducibility is a product feature, so the generator is always explicit:
    when the caller passes none, the seed recorded in the model specification
    is used, and only a specification with no seed at all falls back on fresh
    operating-system entropy (with a warning, because that run cannot be
    repeated).
    """
    if rng is not None:
        return rng
    seed = getattr(getattr(cm, "spec", None), "seed", None)
    if seed is None:
        warnings_out.append(_bilingual(
            "NO_SEED: neither a random generator nor a seed was supplied, so "
            "the shape of the injected model error cannot be reproduced. Set "
            "ModelSpec.seed to make this run repeatable.",
            "لم يُمرَّر مُولِّد أرقام عشوائية ولا بذرة، لذلك لا يمكن تكرار شكل "
            "خطأ النموذج المُدخَل. اضبط الحقل ModelSpec.seed لتصبح هذه النتيجة "
            "قابلة للتكرار.",
        ))
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _resolve_df(cm: "CompiledModel", df: Optional[int], n_free: int,
                warnings_out: List[str]) -> int:
    """Degrees of freedom used to translate ``target_rmsea`` into ``F_0``.

    The default is ``p (p + 1) / 2 - q`` for the very parameterisation whose
    tangent space is projected out, which is both the dimension of the space
    the perturbation lives in and the ``df`` the estimator will report.  When
    ``ModelSpec.degrees_of_freedom`` disagrees -- different identification
    bookkeeping -- the discrepancy is reported rather than silently resolved.
    """
    p = int(getattr(cm, "n_items", 0) or np.asarray(cm.Lambda).shape[0])
    p_star = p * (p + 1) // 2
    if df is not None:
        return int(df)
    derived = p_star - int(n_free)
    spec_df: Optional[int] = None
    spec = getattr(cm, "spec", None)
    try:
        if spec is not None:
            spec_df = int(spec.degrees_of_freedom())
    except Exception:
        spec_df = None
    if spec_df is not None and spec_df != derived:
        warnings_out.append(_bilingual(
            f"DF_MISMATCH: the model specification reports df = {spec_df} "
            f"while the fitted parameterisation has df = {derived} "
            f"({p_star} moments minus {int(n_free)} free parameters). The "
            "fitted value was used, because that is the df the estimator will "
            "divide the chi-square by.",
            f"تقرير مواصفات النموذج يفيد بأن درجات الحرية {spec_df}، في حين أن "
            f"التشكيل المعلمي المُقدَّر لديه {derived} درجة حرية "
            f"({p_star} عزمًا مطروحًا منها {int(n_free)} معلمة حرة). استُخدمت "
            "القيمة المُقدَّرة لأنها القيمة التي سيقسم عليها المُقدِّر مربع كاي.",
        ))
    return derived


def _unperturbed(sigma: np.ndarray, target_rmsea: float, df: int,
                 method: str, warnings_out: List[str],
                 n_free: int = 0, source: str = "") -> ModelErrorResult:
    """The no-op outcome: ``Sigma(theta)`` returned bit-identically."""
    with _quiet():
        try:
            min_eig = float(np.linalg.eigvalsh(sigma).min())
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            min_eig = float("nan")
    return ModelErrorResult(
        sigma_0=sigma,
        achieved_rmsea=0.0,
        achieved_fmin=0.0,
        target_rmsea=float(target_rmsea),
        scale=0.0,
        converged=True,
        warnings=warnings_out,
        df=int(df),
        n_free=int(n_free),
        method=method,
        parameterisation=source,
        direction=None,
        min_eigenvalue=min_eig,
        iterations=0,
    )


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #


def inject_model_error(cm: "CompiledModel", target_rmsea: float, *,
                       df: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None,
                       method: str = METHOD_CUDECK_BROWNE,
                       tol: float = 1e-8,
                       max_iter: int = 200,
                       min_eigenvalue: float = DEFAULT_MIN_EIGENVALUE
                       ) -> ModelErrorResult:
    """Build a generating matrix with a specified population RMSEA.

    Implements Cudeck & Browne (1992).  Returns
    ``Sigma_0 = Sigma(theta) + k E`` where

    * ``F_ML(Sigma_0, Sigma(theta)) = df * target_rmsea**2`` exactly, so the
      population RMSEA of the model in this population is ``target_rmsea``;
    * ``theta`` is still the **exact minimiser** of
      ``F_ML(Sigma_0, Sigma(.))``, because ``E`` is orthogonal to the model's
      tangent space in the ML metric ``W = 0.5 D'(Sigma^-1 (x) Sigma^-1) D``.

    The direction ``E`` is a random symmetric matrix drawn from ``rng`` and
    then projected; the caller therefore controls the *shape* of the misfit
    through the generator and its *amount* through ``target_rmsea``.  Two
    different seeds give two different generating matrices with the same
    achieved RMSEA, which is exactly the right behaviour: "RMSEA = 0.05" does
    not pin down a unique population.

    A useful side effect of the projection: because every residual variance is
    a free parameter, ``dSigma / dTheta_jj = E_jj`` and the tangent space
    contains all ``p`` diagonal directions, so the projection removes the
    whole diagonal component of the random direction.  ``diag(E)`` is zero to
    the accuracy of the numerical Jacobian (around ``1e-11``) and ``Sigma_0``
    therefore stays a correlation matrix with a unit diagonal, which is what
    the downstream Vale-Maurelli generator assumes.  The unprojected
    ``'residual'`` method has no such property and does move the variances.

    Parameters
    ----------
    cm : CompiledModel
        The compiled model supplying ``Sigma(theta)`` and the parameter
        pattern.
    target_rmsea : float
        Desired population RMSEA.  ``<= 0`` returns ``Sigma(theta)``
        untouched.
    df : int, optional
        Degrees of freedom used for ``F_0 = df * target_rmsea**2``.  Defaults
        to ``p (p + 1) / 2`` minus the number of free parameters of the fitted
        parameterisation.  ``<= 0`` returns ``Sigma(theta)`` untouched, because
        a saturated model has no fit to speak of.
    rng : numpy.random.Generator, optional
        Explicit generator for the perturbation direction.  Defaults to
        ``numpy.random.default_rng(cm.spec.seed)``; the global
        ``numpy.random`` state is never touched.
    method : {'cudeck_browne', 'residual', 'none'}, default 'cudeck_browne'
        ``'cudeck_browne'`` projects the random direction out of the model's
        tangent space and so preserves ``theta`` exactly.  ``'residual'``
        skips the projection: cheaper, but the parameters that minimise the
        discrepancy in the resulting population are no longer ``theta``.
        ``'none'`` disables injection.
    tol : float, default 1e-8
        Accepted relative error on the discrepancy,
        ``|F(k) - F_0| <= tol * max(1, F_0)``.  The root solve is driven to
        machine precision and then polished by Newton steps; failing the check
        sets ``converged`` to ``False`` rather than raising.
    max_iter : int, default 200
        Cap on the iterations of the bracket search and of Brent's method.
    min_eigenvalue : float, default :data:`DEFAULT_MIN_EIGENVALUE`
        Smallest eigenvalue ``Sigma_0`` may have, as a fraction of the mean
        variance of ``Sigma(theta)``.  A target that could only be reached
        with a matrix closer to singular than this raises
        :class:`~semsim.core.errors.InfeasibleTargetError` instead of
        returning an unusable matrix.

    Returns
    -------
    ModelErrorResult
        Carrying ``sigma_0``, the achieved RMSEA and discrepancy, the scalar
        ``k``, the perturbation direction and bilingual warnings.

    Raises
    ------
    semsim.core.errors.SpecificationError
        Unknown ``method``, a non-finite ``target_rmsea``, or a singular
        ``Sigma(theta)``.
    semsim.core.errors.InfeasibleTargetError
        The model is saturated, so no perturbation direction exists; or the
        requested RMSEA is so large that no positive-definite ``Sigma_0`` on
        this ray attains it.  The error context carries
        ``max_achievable_rmsea``.
    semsim.core.errors.ConvergenceError
        The Jacobian of ``Sigma(theta)`` could not be formed.

    Examples
    --------
    >>> import numpy as np
    >>> from semsim.core import examples
    >>> from semsim.core.compile import compile_model
    >>> cm = compile_model(examples.minimal_cfa())
    >>> out = inject_model_error(cm, 0.05, rng=np.random.default_rng(1))
    >>> out.df
    8
    >>> bool(abs(out.achieved_rmsea - 0.05) < 1e-9)
    True

    References
    ----------
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
        yields a specified minimizer and a specified minimum discrepancy
        function value. *Psychometrika*, 57(3), 357-369.
    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model
        fit. In K. A. Bollen & J. S. Long (Eds.), *Testing Structural Equation
        Models* (pp. 136-162). Sage.
    Steiger, J. H., & Lind, J. C. (1980). Statistically based tests for the
        number of common factors. Psychometric Society, Iowa City, IA.
    """
    notes: List[str] = []
    resolved_method = _resolve_method(method)
    sigma = implied_covariance(cm)

    target = float(target_rmsea)
    if not np.isfinite(target):
        raise SpecificationError(
            "The target RMSEA must be a finite number.",
            "القيمة المستهدفة لمؤشر RMSEA يجب أن تكون عددًا منتهيًا.",
            target_rmsea=target_rmsea,
        )

    if resolved_method == METHOD_NONE or target <= 0.0:
        return _unperturbed(sigma, max(target, 0.0),
                            int(df) if df is not None else 0,
                            resolved_method, notes)

    if not is_positive_definite(sigma, tol=1e-12):
        raise SpecificationError(
            "The model-implied covariance matrix is not positive definite, so "
            "no amount of model error can be injected into it. Check the "
            "Heywood-case and residual-covariance warnings from the compiler "
            "first.",
            "مصفوفة التغاير المُشتقة من النموذج ليست معرّفة موجبة، لذلك لا يمكن "
            "إدخال أي خطأ نموذج إليها. راجع أولًا تحذيرات حالات هايوود وتغايرات "
            "البواقي الصادرة من مُصرِّف النموذج.",
        )

    pmap: Optional[FreeParameterMap] = None
    source = ""
    n_free = 0
    if resolved_method == METHOD_CUDECK_BROWNE:
        pmap = free_parameter_map(cm)
        source = pmap.source
        n_free = pmap.q
        if source == "builtin":
            notes.append(_bilingual(
                "OWN_PARAMETERISATION: semsim.analysis.estimator exposed no "
                "usable free-parameter mapping, so this module's own "
                "marker-variable parameterisation was used to build the "
                "tangent space (free loadings, structural paths, latent "
                "(co)variances, residual variances and covariances; one marker "
                "loading fixed per latent variable).",
                "لم تُوفّر الوحدة semsim.analysis.estimator تعيينًا صالحًا "
                "للمعالم الحرة، لذلك استُخدم التشكيل المعلمي الخاص بهذه الوحدة "
                "لبناء الفضاء المماسي (تشبعات ومسارات وتباينات وتغايرات كامنة "
                "وتباينات وتغايرات بواقٍ حرة، مع تثبيت تشبع مرجعي واحد لكل "
                "متغير كامن).",
            ))
    else:
        notes.append(_bilingual(
            "UNPROJECTED_ERROR: the 'residual' method perturbs the implied "
            "matrix without projecting the perturbation out of the model's "
            "tangent space, so the parameters that minimise the discrepancy "
            "in the generated population are no longer exactly the ones you "
            "specified. Use 'cudeck_browne' when the recovered parameters "
            "matter.",
            "طريقة «residual» تُضطرب مصفوفة النموذج دون إسقاط الاضطراب خارج "
            "الفضاء المماسي للنموذج، لذلك لن تكون المعالم المُصغِّرة لدالة "
            "التباعد في المجتمع المُولَّد مطابقة تمامًا للمعالم التي حددتها. "
            "استخدم «cudeck_browne» إذا كانت المعالم المُستردة مهمة لك.",
        ))

    resolved_df = _resolve_df(cm, df, n_free, notes)
    if resolved_df <= 0:
        notes.append(_bilingual(
            "ZERO_DF: the model has no degrees of freedom, so RMSEA is "
            "undefined and no model error was injected.",
            "النموذج لا يملك درجات حرية، لذلك يفقد مؤشر RMSEA معناه ولم يُدخل "
            "أي خطأ نموذج.",
        ))
        return _unperturbed(sigma, target, resolved_df, resolved_method,
                            notes, n_free, source)

    f_target = discrepancy_for_rmsea(target, resolved_df)

    generator = _resolve_rng(cm, rng, notes)
    direction = perturbation_direction(
        cm, generator, method=resolved_method, pmap=pmap, sigma=sigma,
        warnings_out=notes)

    k, achieved, converged, iterations = _solve_scale(
        sigma, direction, f_target, tol=tol, max_iter=max_iter,
        min_eigenvalue=min_eigenvalue, df=resolved_df, target_rmsea=target)

    sigma_0 = _symmetrise(sigma + k * direction)
    with _quiet():
        min_eig = float(np.linalg.eigvalsh(sigma_0).min())

    if not is_positive_definite(sigma_0, tol=0.0):  # pragma: no cover
        raise InfeasibleTargetError(
            f"A population RMSEA of {target:.4f} cannot be reached on this "
            "model without making the generating covariance matrix "
            "non-positive-definite. Ask for less misfit, or add items so the "
            "model has more degrees of freedom to absorb it.",
            f"لا يمكن الوصول إلى قيمة RMSEA سكانية مقدارها {target:.4f} في هذا "
            "النموذج دون أن تصبح مصفوفة التغاير المُولِّدة غير معرّفة موجبة. "
            "اطلب قدرًا أقل من سوء المطابقة، أو أضف بنودًا حتى تزيد درجات حرية "
            "النموذج القادرة على استيعابه.",
            target_rmsea=target, df=resolved_df,
        )

    achieved_rmsea = rmsea_from_discrepancy(achieved, resolved_df)
    if not converged:
        notes.append(_bilingual(
            f"RMSEA_TOLERANCE: the injected model error reached RMSEA "
            f"{achieved_rmsea:.6f} instead of the requested {target:.6f}; the "
            "scalar solve stopped at the limits of numerical precision.",
            f"بلغ خطأ النموذج المُدخَل قيمة RMSEA مقدارها {achieved_rmsea:.6f} "
            f"بدلًا من القيمة المطلوبة {target:.6f}؛ إذ توقّف حل المعادلة "
            "العددية عند حدود الدقة الحسابية.",
        ))

    return ModelErrorResult(
        sigma_0=sigma_0,
        achieved_rmsea=achieved_rmsea,
        achieved_fmin=float(achieved),
        target_rmsea=target,
        scale=float(k),
        converged=bool(converged),
        warnings=notes,
        df=int(resolved_df),
        n_free=int(n_free),
        method=resolved_method,
        parameterisation=source,
        direction=direction,
        min_eigenvalue=min_eig,
        iterations=int(iterations),
    )


def _solve_scale(sigma: np.ndarray, direction: np.ndarray, f_target: float, *,
                 tol: float, max_iter: int, min_eigenvalue: float,
                 df: int, target_rmsea: float
                 ) -> Tuple[float, float, bool, int]:
    """Solve ``F_ML(Sigma + k E, Sigma) = F_0`` for ``k > 0``.

    ``F`` vanishes with zero derivative at ``k = 0`` and is strictly convex
    along the ray, so it increases monotonically and the root is unique.  The
    upper bracket is doubled while ``Sigma + k E`` stays admissible and
    bisected back towards the admissibility boundary once it does not, exactly
    as Cudeck & Browne (1992, p. 362) describe.  Brent's method then locates
    the root, and two Newton steps using
    ``F'(k) = tr(E Sigma^-1) - tr(E Sigma_0^-1)`` polish it.

    Returns
    -------
    (float, float, bool, int)
        ``(k, F(k), converged, n_evaluations)``.
    """
    s = _symmetrise(sigma)
    e = _symmetrise(direction)
    floor = float(min_eigenvalue) * float(np.mean(np.diag(s)))
    budget = {"n": 0}

    sigma_inv = _safe_inverse(
        s,
        "The model-implied covariance matrix is singular, so the model-error "
        "scale cannot be solved for.",
        "مصفوفة التغاير المُشتقة من النموذج مفردة، لذلك تعذّر حل مقدار خطأ "
        "النموذج.",
    )
    trace_const = float(np.trace(e @ sigma_inv))

    def admissible(k: float) -> bool:
        with _quiet():
            candidate = _symmetrise(s + float(k) * e)
            if not np.all(np.isfinite(candidate)):
                return False
            try:
                return bool(np.linalg.eigvalsh(candidate).min() > floor)
            except np.linalg.LinAlgError:
                return False

    def f_of(k: float) -> float:
        budget["n"] += 1
        if not admissible(k):
            return float("inf")
        return population_discrepancy(s + float(k) * e, s)

    # -- an admissible starting point ------------------------------------- #
    hi = 1.0
    for _ in range(int(max_iter)):
        if admissible(hi) and np.isfinite(f_of(hi)):
            break
        hi *= 0.5
        if hi < 1e-14:
            raise InfeasibleTargetError(
                "Even an infinitesimal amount of model error makes the "
                "generating covariance matrix singular, so the requested fit "
                "cannot be produced. The implied matrix is almost certainly "
                "already near-singular: look for two items with a residual "
                "correlation close to one, or a loading at the Heywood bound.",
                "حتى أقل قدر من خطأ النموذج يجعل مصفوفة التغاير المُولِّدة مفردة، "
                "لذلك لا يمكن إنتاج حسن المطابقة المطلوب. من المرجّح جدًا أن "
                "مصفوفة النموذج شبه مفردة بالأصل: ابحث عن بندين بينهما ارتباط "
                "بواقٍ يقارب الواحد، أو عن تشبع عند حدّ هايوود.",
                target_rmsea=target_rmsea,
            )

    # -- bracket the root ------------------------------------------------- #
    for _ in range(int(max_iter)):
        if f_of(hi) >= f_target:
            break
        nxt = 2.0 * hi
        if admissible(nxt):
            hi = nxt
            continue
        low, up = hi, nxt
        for _ in range(200):
            mid = 0.5 * (low + up)
            if admissible(mid):
                low = mid
            else:
                up = mid
            if up - low <= 1e-14 * max(1.0, up):
                break
        hi = low
        break

    reachable = f_of(hi)
    if not np.isfinite(reachable) or reachable < f_target:
        max_rmsea = rmsea_from_discrepancy(reachable, df)
        raise InfeasibleTargetError(
            f"A population RMSEA of {target_rmsea:.4f} is out of reach for "
            f"this model: with {df} degrees of freedom the largest misfit that "
            "still leaves a usable (positive definite, well conditioned) "
            f"generating matrix is about {max_rmsea:.4f}. Ask for a smaller "
            "RMSEA, or specify more items so the model has more degrees of "
            "freedom to carry the error.",
            f"قيمة RMSEA السكانية المطلوبة {target_rmsea:.4f} بعيدة عن متناول "
            f"هذا النموذج: فبدرجات حرية مقدارها {df}، أقصى سوء مطابقة يُبقي "
            "مصفوفة التوليد صالحة للاستخدام (معرّفة موجبة وجيدة الشرطية) هو "
            f"نحو {max_rmsea:.4f}. اطلب قيمة RMSEA أصغر، أو حدِّد عددًا أكبر من "
            "البنود حتى تزيد درجات الحرية القادرة على حمل الخطأ.",
            target_rmsea=target_rmsea, df=int(df),
            max_achievable_rmsea=max_rmsea,
            max_achievable_fmin=float(reachable),
        )

    try:
        k = float(_opt.brentq(lambda x: f_of(x) - f_target, 0.0, hi,
                              xtol=1e-16, rtol=8.9e-16,
                              maxiter=max(int(max_iter), 20)))
    except (ValueError, RuntimeError) as exc:
        raise ConvergenceError(
            "Brent's method failed to solve for the amount of model error "
            "needed to reach the requested RMSEA.",
            "فشلت طريقة برنت في حل مقدار خطأ النموذج اللازم للوصول إلى قيمة "
            "RMSEA المطلوبة.",
            detail=str(exc),
        ) from exc

    # -- Newton polish ----------------------------------------------------- #
    value = f_of(k)
    for _ in range(12):
        gap = value - f_target
        if abs(gap) <= tol * max(1.0, abs(f_target)) * 1e-3:
            break
        with _quiet():
            inv_0 = np.linalg.inv(_symmetrise(s + k * e))
            slope = trace_const - float(np.trace(e @ inv_0))
        if not np.isfinite(slope) or abs(slope) < 1e-300:
            break
        step = gap / slope
        candidate = k - step
        if not (candidate > 0.0) or not admissible(candidate):
            break
        new_value = f_of(candidate)
        if not np.isfinite(new_value) or abs(new_value - f_target) >= abs(gap):
            break
        k, value = candidate, new_value

    converged = bool(abs(value - f_target) <= tol * max(1.0, abs(f_target)))
    return k, float(value), converged, int(budget["n"])

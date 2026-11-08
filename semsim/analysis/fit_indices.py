"""
semsim.analysis.fit_indices
===========================
Global goodness-of-fit indices for a fitted covariance-structure model.

Everything in this module is a pure function of four numbers-and-matrices:
the observed covariance matrix ``S``, the model-implied covariance matrix
``Sigma(theta_hat)``, the sample size ``N`` and the number of free parameters
``q``.  Nothing here estimates anything; it only scores a solution that
:mod:`semsim.analysis.estimator` has already produced (or, in the population
case, a ``Sigma`` straight out of :mod:`semsim.core.implied`).

The formulas are the conventional ones, because these numbers go verbatim into
a researcher's results table::

    F_ML   = log|Sigma| + tr(S Sigma^-1) - log|S| - p
    chi2   = (N - 1) F_ML
    df     = p (p + 1) / 2 - q
    RMSEA  = sqrt( max(chi2 - df, 0) / (df (N - 1)) )
    CFI    = 1 - max(chi2 - df, 0) / max(chi2_b - df_b, chi2 - df, 0)
    TLI    = (chi2_b/df_b - chi2/df) / (chi2_b/df_b - 1)
    NFI    = (chi2_b - chi2) / chi2_b
    IFI    = (chi2_b - chi2) / (chi2_b - df)
    RFI    = 1 - (chi2/df) / (chi2_b/df_b)
    SRMR   = sqrt( sum_{i<=j} ((s_ij - sigma_ij)/sqrt(s_ii s_jj))^2 / (p(p+1)/2) )
    GFI    = 1 - tr((Sigma^-1 S - I)^2) / tr((Sigma^-1 S)^2)
    AGFI   = 1 - (p(p+1) / (2 df)) (1 - GFI)
    AIC    = chi2 + 2 q
    BIC    = chi2 + q log(N)
    SABIC  = chi2 + q log((N + 2)/24)

The baseline (independence) model is ``diag(S)``; its discrepancy is
``sum_i log(s_ii) - log|S|`` with ``df_b = p(p - 1)/2``.

Numerical notes
---------------
*   ``F_ML`` is evaluated as ``sum_k (e_k - log e_k - 1)`` over the eigenvalues
    ``e_k`` of ``Sigma^-1 S`` (obtained from a Cholesky-whitened symmetric
    problem).  Each summand is non-negative with equality exactly at
    ``e_k == 1``, so the discrepancy can never come out negative through
    round-off, and ``S == Sigma`` gives a value at machine level that is
    snapped to an exact zero (:data:`PERFECT_FIT_TOLERANCE`).
*   The incremental family (CFI, TLI, NFI, IFI, RFI) is truncated to
    ``[0, 1]``: Bentler (1990) recommends it for CFI and reporting practice
    applies it to the whole family, so a saturated or perfectly reproducing
    solution reports exactly ``1.000`` instead of ``1.004``.
*   A singular ``S`` or ``Sigma`` raises
    :class:`semsim.core.errors.EstimationError` with a bilingual message
    rather than letting a ``LinAlgError`` escape.

References
----------
Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model fit.
    In K. A. Bollen & J. S. Long (Eds.), *Testing Structural Equation Models*
    (pp. 136-162). Sage.
Bentler, P. M. (1990). Comparative fit indexes in structural models.
    *Psychological Bulletin*, 107(2), 238-246.
Tucker, L. R., & Lewis, C. (1973). A reliability coefficient for maximum
    likelihood factor analysis. *Psychometrika*, 38(1), 1-10.
Bentler, P. M., & Bonett, D. G. (1980). Significance tests and goodness of fit
    in the analysis of covariance structures. *Psychological Bulletin*, 88(3),
    588-606.
Bollen, K. A. (1989). A new incremental fit index for general structural
    equation models. *Sociological Methods & Research*, 17(3), 303-316.
Joreskog, K. G., & Sorbom, D. (1981). *LISREL V: Analysis of Linear Structural
    Relationships by Maximum Likelihood and Least Squares Methods*. National
    Educational Resources.
Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance
    structure analysis. *Structural Equation Modeling*, 6(1), 1-55.
Steiger, J. H., & Lind, J. C. (1980). Statistically based tests for the number
    of common factors. Paper presented at the Psychometric Society, Iowa City.
Sclove, S. L. (1987). Application of model-selection criteria to some problems
    in multivariate analysis. *Psychometrika*, 52(3), 333-343.
Kline, R. B. (2016). *Principles and Practice of Structural Equation
    Modeling* (4th ed.). Guilford Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg as _sla
from scipy import optimize as _opt
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import ncx2 as _ncx2_dist

from semsim.core.errors import ConvergenceError, EstimationError, SpecificationError
from semsim.core.spec import FitLevel

#: Discrepancies below this are treated as an exact zero, so that a model whose
#: implied matrix equals the observed one reports ``chi2 == 0.0`` exactly rather
#: than the ``1e-30`` that floating-point eigenvalues leave behind.
PERFECT_FIT_TOLERANCE = 1e-12

#: Largest non-centrality parameter the RMSEA interval search will report.
MAX_NONCENTRALITY = 1e12

#: RMSEA value defining "close fit" for the PCLOSE test (Browne & Cudeck, 1993).
CLOSE_FIT_RMSEA = 0.05

__all__ = [
    "PERFECT_FIT_TOLERANCE", "MAX_NONCENTRALITY", "CLOSE_FIT_RMSEA",
    "FitIndices", "compute", "baseline_fit", "ml_discrepancy",
    "rmsea_confidence_interval", "thresholds_table",
]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _validate_matrix(m: object, what_en: str, what_ar: str) -> np.ndarray:
    """Return ``m`` as a finite, square, symmetrised float array."""
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0:
        raise SpecificationError(
            f"{what_en} must be a non-empty square matrix; got shape "
            f"{getattr(a, 'shape', None)}.",
            f"يجب أن تكون {what_ar} مصفوفة مربعة غير فارغة، والشكل المُمرَّر هو "
            f"{getattr(a, 'shape', None)}.",
            shape=getattr(a, "shape", None),
        )
    if not np.all(np.isfinite(a)):
        raise SpecificationError(
            f"{what_en} contains missing or infinite entries, so no fit index "
            "can be computed.",
            f"تحتوي {what_ar} على قيم مفقودة أو غير منتهية، ولذلك لا يمكن حساب "
            "مؤشرات المطابقة.",
        )
    return 0.5 * (a + a.T)


def _check_sample_size(n: int) -> int:
    n_int = int(n)
    if n_int < 2:
        raise SpecificationError(
            f"The sample size must be at least 2 to compute a chi-square "
            f"statistic; got N = {n_int}.",
            f"يجب أن يكون حجم العينة 2 على الأقل لحساب إحصاء مربع كاي، والقيمة "
            f"المُمرَّرة هي N = {n_int}.",
            n=n_int,
        )
    return n_int


def _cholesky(m: np.ndarray, what_en: str, what_ar: str) -> np.ndarray:
    """Lower Cholesky factor, or a bilingual :class:`EstimationError`."""
    try:
        return _sla.cholesky(m, lower=True)
    except (np.linalg.LinAlgError, _sla.LinAlgError, ValueError) as exc:
        raise EstimationError(
            f"{what_en} is singular or not positive definite, so the "
            "maximum-likelihood discrepancy is undefined. This usually means "
            "two items are (nearly) identical or an item has no variance.",
            f"{what_ar} شبه منفردة (غير قابلة للعكس) أو غير معرَّفة موجبة، ولذلك "
            "لا يمكن حساب دالة التباعد بطريقة الإمكان الأعظم. يحدث هذا عادةً عند "
            "وجود بندين متطابقين تقريبًا أو بند بلا تباين.",
            detail=str(exc),
        ) from exc


def _snap(value: float) -> float:
    """Clamp a discrepancy to zero below :data:`PERFECT_FIT_TOLERANCE`."""
    v = float(value)
    if not math.isfinite(v):
        return v
    if v < PERFECT_FIT_TOLERANCE:
        return 0.0
    return v


# --------------------------------------------------------------------------- #
# Discrepancy functions
# --------------------------------------------------------------------------- #


def ml_discrepancy(sample_cov: np.ndarray, sigma_hat: np.ndarray) -> float:
    """Maximum-likelihood discrepancy ``F_ML`` between two covariance matrices.

    Evaluates ``F = log|Sigma| + tr(S Sigma^-1) - log|S| - p`` in the
    algebraically equivalent but numerically safer eigenvalue form
    ``F = sum_k (e_k - log e_k - 1)``, where ``e_k`` are the eigenvalues of
    ``Sigma^-1 S``.  Every summand is non-negative, so the result can never be
    a small negative number, and an exact match returns exactly ``0.0``.

    Parameters
    ----------
    sample_cov : numpy.ndarray
        ``(p, p)`` observed (or target) covariance matrix ``S``.
    sigma_hat : numpy.ndarray
        ``(p, p)`` model-implied covariance matrix ``Sigma(theta_hat)``.

    Returns
    -------
    float
        The discrepancy, ``>= 0``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the matrices are not conformable, square or finite.
    semsim.core.errors.EstimationError
        If either matrix is singular / not positive definite.

    References
    ----------
    Joreskog, K. G. (1969). A general approach to confirmatory maximum
        likelihood factor analysis. *Psychometrika*, 34(2), 183-202.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        chapter 4. Wiley.
    """
    S = _validate_matrix(sample_cov, "The observed covariance matrix",
                         "مصفوفة التغايرات المشاهدة")
    Sigma = _validate_matrix(sigma_hat, "The model-implied covariance matrix",
                             "مصفوفة التغايرات المُقدَّرة من النموذج")
    if S.shape != Sigma.shape:
        raise SpecificationError(
            f"The observed matrix is {S.shape} but the implied matrix is "
            f"{Sigma.shape}; they must have the same number of items.",
            f"مصفوفة التغايرات المشاهدة بالشكل {S.shape} بينما المصفوفة المُقدَّرة "
            f"بالشكل {Sigma.shape}؛ يجب أن يتساوى عدد البنود في الاثنتين.",
            observed=S.shape, implied=Sigma.shape,
        )
    if np.array_equal(S, Sigma):
        # Short-circuit the common population / saturated case exactly.
        _cholesky(Sigma, "The model-implied covariance matrix",
                  "مصفوفة التغايرات المُقدَّرة من النموذج")
        return 0.0

    chol = _cholesky(Sigma, "The model-implied covariance matrix",
                     "مصفوفة التغايرات المُقدَّرة من النموذج")
    # A = L^-1 S L^-T has the same eigenvalues as Sigma^-1 S and is symmetric.
    half = _sla.solve_triangular(chol, S, lower=True)
    a = _sla.solve_triangular(chol, half.T, lower=True)
    a = 0.5 * (a + a.T)
    eigenvalues = np.linalg.eigvalsh(a)
    if eigenvalues.min() <= 0.0:
        raise EstimationError(
            "The observed covariance matrix is singular or not positive "
            "definite, so the maximum-likelihood discrepancy is undefined. "
            "Check for collinear items, a zero-variance item, or fewer cases "
            "than items.",
            "مصفوفة التغايرات المشاهدة شبه منفردة أو غير معرَّفة موجبة، ولذلك لا "
            "يمكن حساب دالة التباعد بطريقة الإمكان الأعظم. راجع البنود المتداخلة "
            "خطيًا، أو بندًا بلا تباين، أو كون عدد الحالات أقل من عدد البنود.",
            min_eigenvalue=float(eigenvalues.min()),
        )
    return _snap(float(np.sum(eigenvalues - np.log(eigenvalues) - 1.0)))


def baseline_fit(sample_cov: np.ndarray, n: int) -> Tuple[float, int]:
    """Chi-square and degrees of freedom of the independence (null) model.

    The baseline against which every incremental index is scaled is the model
    that fixes every covariance to zero and leaves the variances free, i.e.
    ``Sigma_b = diag(S)``.  Its maximum-likelihood discrepancy collapses to
    ``sum_i log(s_ii) - log|S| = -log|R|`` with ``R`` the sample correlation
    matrix, and it uses ``p`` of the ``p(p+1)/2`` moments, leaving
    ``df_b = p(p-1)/2``.

    Parameters
    ----------
    sample_cov : numpy.ndarray
        ``(p, p)`` observed covariance matrix.
    n : int
        Sample size.

    Returns
    -------
    (float, int)
        ``(baseline_chi_square, baseline_df)``.

    Raises
    ------
    semsim.core.errors.EstimationError
        If ``S`` is singular or has a non-positive variance on its diagonal.

    References
    ----------
    Bentler, P. M., & Bonett, D. G. (1980). Significance tests and goodness of
        fit in the analysis of covariance structures. *Psychological
        Bulletin*, 88(3), 588-606.
    """
    S = _validate_matrix(sample_cov, "The observed covariance matrix",
                         "مصفوفة التغايرات المشاهدة")
    n_int = _check_sample_size(n)
    p = S.shape[0]
    variances = np.diag(S)
    if np.any(variances <= 0.0):
        bad = [int(i) for i in np.flatnonzero(variances <= 0.0)]
        raise EstimationError(
            f"Items at positions {bad} have zero or negative variance, so the "
            "independence baseline model cannot be evaluated.",
            f"البنود في المواضع {bad} تباينها صفر أو سالب، ولذلك لا يمكن تقدير "
            "نموذج الاستقلال المرجعي.",
            positions=bad,
        )
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0 or not math.isfinite(logdet):
        raise EstimationError(
            "The observed covariance matrix is singular, so the independence "
            "baseline model cannot be evaluated. Two or more items are "
            "linearly dependent.",
            "مصفوفة التغايرات المشاهدة منفردة (محدِّدها صفر)، ولذلك لا يمكن تقدير "
            "نموذج الاستقلال المرجعي. هناك بندان أو أكثر مرتبطان خطيًا تمامًا.",
            sign=float(sign),
        )
    f_baseline = _snap(max(float(np.sum(np.log(variances)) - logdet), 0.0))
    return float((n_int - 1) * f_baseline), int(p * (p - 1) // 2)


# --------------------------------------------------------------------------- #
# RMSEA confidence interval
# --------------------------------------------------------------------------- #


def _ncx2_cdf(x: float, df: int, ncp: float) -> float:
    """``P(chi2_df(ncp) <= x)``, using the central law when ``ncp <= 0``."""
    if ncp <= 0.0:
        return float(_chi2_dist.cdf(x, df))
    return float(_ncx2_dist.cdf(x, df, ncp))


def _ncp_at_tail(chi_square: float, df: int, target_cdf: float) -> float:
    """Smallest ``lambda >= 0`` with ``ncx2.cdf(chi2, df, lambda) == target``.

    ``lambda -> ncx2.cdf(x, df, lambda)`` is strictly decreasing, so the root
    is unique when it exists.  ``0.0`` is returned when the curve already lies
    below the target at ``lambda = 0`` (the usual situation for the upper tail
    of a very well fitting model).
    """
    def excess(ncp: float) -> float:
        return _ncx2_cdf(chi_square, df, ncp) - target_cdf

    if excess(0.0) <= 0.0:
        return 0.0

    upper = max(float(chi_square), float(df), 1.0)
    bracketed = False
    for _ in range(200):
        if excess(upper) < 0.0:
            bracketed = True
            break
        upper *= 2.0
        if upper > MAX_NONCENTRALITY:
            break
    if not bracketed:
        return MAX_NONCENTRALITY

    try:
        return float(_opt.brentq(excess, 0.0, upper, xtol=1e-10, rtol=1e-12,
                                 maxiter=300))
    except (ValueError, RuntimeError) as exc:  # pragma: no cover - guarded above
        raise ConvergenceError(
            "The search for the RMSEA confidence limit did not converge, so "
            "the interval cannot be reported for this solution.",
            "لم يتقارب البحث عن حدود فترة الثقة لمؤشر RMSEA، ولذلك لا يمكن عرض "
            "الفترة لهذا الحل.",
            chi_square=float(chi_square), df=int(df), target=float(target_cdf),
            detail=str(exc),
        ) from exc


def rmsea_confidence_interval(chi2: float, df: int, n: int,
                              level: float = 0.90) -> Tuple[float, float]:
    """Confidence interval for RMSEA by inverting the non-central chi-square.

    RMSEA is a monotone transform of the non-centrality parameter,
    ``rmsea = sqrt(lambda / (df (N - 1)))``, so a confidence interval for
    ``lambda`` maps straight onto one for RMSEA.  The limits for ``lambda``
    solve

    * lower: ``ncx2.cdf(chi2, df, lambda_L) = 1 - (1 - level)/2``  (``0.95``),
    * upper: ``ncx2.cdf(chi2, df, lambda_U) = (1 - level)/2``      (``0.05``),

    found with Brent's method on a bracket that is doubled until it contains
    the root.  When the central chi-square already falls below the required
    cumulative probability at ``lambda = 0`` no positive root exists and the
    limit is reported as ``0.0`` — the familiar ``[0.000, ...]`` lower bound of
    a well fitting model.

    Parameters
    ----------
    chi2 : float
        The model chi-square statistic.
    df : int
        Model degrees of freedom; must be positive for an interval to exist.
    n : int
        Sample size.  The scaling uses ``N - 1``, matching
        ``chi2 = (N - 1) F_ML``.
    level : float, default 0.90
        Two-sided confidence level.  ``0.90`` is the convention for RMSEA.

    Returns
    -------
    (float, float)
        ``(lower, upper)`` RMSEA limits.  ``(0.0, 0.0)`` when ``df <= 0``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``level`` is not strictly inside ``(0, 1)``, ``chi2`` is negative or
        non-finite, or ``n < 2``.

    References
    ----------
    Steiger, J. H., & Lind, J. C. (1980). Statistically based tests for the
        number of common factors. Paper presented at the annual meeting of the
        Psychometric Society, Iowa City, IA.
    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model
        fit. In K. A. Bollen & J. S. Long (Eds.), *Testing Structural Equation
        Models* (pp. 136-162). Sage.
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149.
    """
    if not (0.0 < float(level) < 1.0):
        raise SpecificationError(
            f"The confidence level must lie strictly between 0 and 1; got "
            f"{level}.",
            f"يجب أن يكون مستوى الثقة بين صفر وواحد تمامًا، والقيمة المُمرَّرة هي "
            f"{level}.",
            level=float(level),
        )
    value = float(chi2)
    if not math.isfinite(value) or value < 0.0:
        raise SpecificationError(
            f"The chi-square statistic must be a finite non-negative number; "
            f"got {chi2}.",
            f"يجب أن يكون إحصاء مربع كاي رقمًا منتهيًا غير سالب، والقيمة المُمرَّرة "
            f"هي {chi2}.",
            chi_square=chi2,
        )
    n_int = _check_sample_size(n)
    df_int = int(df)
    if df_int <= 0:
        return (0.0, 0.0)

    tail = 0.5 * (1.0 - float(level))
    ncp_lower = _ncp_at_tail(value, df_int, 1.0 - tail)
    ncp_upper = _ncp_at_tail(value, df_int, tail)
    denominator = float(df_int) * float(n_int - 1)
    lower = math.sqrt(max(ncp_lower, 0.0) / denominator)
    upper = math.sqrt(max(ncp_upper, 0.0) / denominator)
    if upper < lower:  # pragma: no cover - monotonicity makes this impossible
        lower, upper = upper, lower
    return (float(lower), float(upper))


# --------------------------------------------------------------------------- #
# Reporting metadata: thresholds and bilingual interpretations
# --------------------------------------------------------------------------- #

#: ``code -> (label_en, label_ar, threshold_text, direction, good, acceptable)``
#: ``direction`` is ``"ge"`` when larger is better, ``"le"`` when smaller is
#: better and ``None`` for purely descriptive rows.  ``good`` is the Hu &
#: Bentler (1999) cutoff, ``acceptable`` the widely used tolerant cutoff.
_ROWS: Tuple[Tuple[str, str, str, str, Optional[str], Optional[float],
                   Optional[float]], ...] = (
    ("chi_square", "Chi-square", "مربع كاي", "—", None, None, None),
    ("df", "Degrees of freedom", "درجات الحرية", "—", None, None, None),
    ("p_value", "p (chi-square)", "الدلالة الاحتمالية لمربع كاي",
     "> .05", "ge", 0.05, 0.01),
    ("chi_square_df_ratio", "Chi-square / df", "مربع كاي ÷ درجات الحرية",
     "< 3", "le", 3.0, 5.0),
    ("rmsea", "RMSEA", "جذر متوسط مربع خطأ التقريب", "<= .06", "le", 0.06, 0.08),
    ("rmsea_ci_lower", "RMSEA 90% CI lower",
     "الحد الأدنى لفترة ثقة 90% لـ RMSEA", "—", None, None, None),
    ("rmsea_ci_upper", "RMSEA 90% CI upper",
     "الحد الأعلى لفترة ثقة 90% لـ RMSEA", "<= .08", "le", 0.08, 0.10),
    ("rmsea_pclose", "PCLOSE", "احتمال المطابقة القريبة", "> .05", "ge", 0.05, 0.01),
    ("cfi", "CFI", "مؤشر المطابقة المقارن", ">= .95", "ge", 0.95, 0.90),
    ("tli", "TLI", "مؤشر تاكر-لويس", ">= .95", "ge", 0.95, 0.90),
    ("nfi", "NFI", "مؤشر المطابقة المعياري", ">= .95", "ge", 0.95, 0.90),
    ("ifi", "IFI", "مؤشر المطابقة التزايدي", ">= .95", "ge", 0.95, 0.90),
    ("rfi", "RFI", "مؤشر المطابقة النسبي", ">= .95", "ge", 0.95, 0.90),
    ("srmr", "SRMR", "جذر متوسط مربعات البواقي المعيارية",
     "<= .08", "le", 0.08, 0.10),
    ("gfi", "GFI", "مؤشر جودة المطابقة", ">= .95", "ge", 0.95, 0.90),
    ("agfi", "AGFI", "مؤشر جودة المطابقة المُعدَّل", ">= .90", "ge", 0.90, 0.85),
    ("aic", "AIC", "معيار أكايكي للمعلومات", "lower is better", None, None, None),
    ("bic", "BIC", "معيار بايز للمعلومات", "lower is better", None, None, None),
    ("sabic", "SABIC", "معيار بايز المُعدَّل بحجم العينة",
     "lower is better", None, None, None),
)

#: ``code -> {band: (interpretation_en, interpretation_ar)}``
_INTERPRETATION: Dict[str, Dict[str, Tuple[str, str]]] = {
    "chi_square": {"info": (
        "The likelihood-ratio statistic; it grows with sample size, so read it "
        "beside its degrees of freedom rather than on its own.",
        "إحصاء نسبة الإمكان؛ يزداد بزيادة حجم العينة، ولذلك يُقرأ مقترنًا بدرجات "
        "الحرية لا بمعزل عنها.")},
    "df": {"info": (
        "Overidentifying restrictions: the number of moments the model is not "
        "allowed to reproduce freely.",
        "عدد القيود الزائدة عن التحديد، أي عدد العزوم التي لا يُسمح للنموذج "
        "بإعادة إنتاجها بحرية.")},
    "p_value": {
        "pass": ("The hypothesis of exact fit is not rejected at alpha = .05.",
                 "لا يُرفض فرض المطابقة التامة عند مستوى دلالة 0.05."),
        "borderline": (
            "Exact fit is rejected at .05 but not at .01; treat the global "
            "test as inconclusive and lean on RMSEA, CFI and SRMR.",
            "يُرفض فرض المطابقة التامة عند 0.05 دون 0.01؛ يُعدّ الاختبار الكلي غير "
            "قاطع، ويُعتمد على RMSEA و CFI و SRMR."),
        "fail": (
            "Exact fit is rejected; this is expected in large samples and is "
            "not by itself a reason to discard the model.",
            "يُرفض فرض المطابقة التامة؛ وهذا متوقَّع في العينات الكبيرة ولا يكفي "
            "وحده لرفض النموذج.")},
    "chi_square_df_ratio": {
        "pass": ("Misfit per degree of freedom is small (below 3).",
                 "مقدار سوء المطابقة لكل درجة حرية صغير (أقل من 3)."),
        "borderline": (
            "Misfit per degree of freedom is elevated (3 to 5); inspect the "
            "residual covariances.",
            "مقدار سوء المطابقة لكل درجة حرية مرتفع (بين 3 و5)؛ راجع مصفوفة "
            "البواقي."),
        "fail": ("Misfit per degree of freedom exceeds 5; the structure is "
                 "misspecified.",
                 "مقدار سوء المطابقة لكل درجة حرية يتجاوز 5؛ البنية المفترضة "
                 "خاطئة التحديد.")},
    "rmsea": {
        "pass": ("Close approximate fit: the model reproduces the covariances "
                 "with little error per degree of freedom.",
                 "مطابقة تقريبية جيدة: يعيد النموذج إنتاج التغايرات بخطأ ضئيل "
                 "لكل درجة حرية."),
        "borderline": ("Mediocre but tolerable approximation; report the "
                       "confidence interval beside the point estimate.",
                       "مطابقة تقريبية متواضعة لكنها مقبولة؛ يُستحسن عرض فترة "
                       "الثقة إلى جانب التقدير النقطي."),
        "fail": ("Poor approximation: the model misses systematic covariation "
                 "and should be respecified.",
                 "مطابقة ضعيفة: النموذج يُغفل تغايرًا منظَّمًا في البيانات ويحتاج "
                 "إلى إعادة صياغة.")},
    "rmsea_ci_lower": {"info": (
        "Lower 90% confidence limit for RMSEA; a value of 0.000 means exact "
        "fit cannot be ruled out.",
        "الحد الأدنى لفترة ثقة 90% لمؤشر RMSEA؛ القيمة صفر تعني أن المطابقة "
        "التامة تبقى احتمالًا قائمًا.")},
    "rmsea_ci_upper": {
        "pass": ("Even the pessimistic end of the interval stays within the "
                 "acceptable range.",
                 "حتى الطرف المتشائم من فترة الثقة يبقى داخل المدى المقبول."),
        "borderline": ("The interval reaches into the mediocre range; a larger "
                       "sample would narrow it.",
                       "تمتد فترة الثقة إلى المدى المتواضع؛ وزيادة حجم العينة "
                       "تضيّقها."),
        "fail": ("The interval admits poor fit, so the point estimate is not "
                 "trustworthy on its own.",
                 "تسمح فترة الثقة باحتمال مطابقة ضعيفة، ولذلك لا يُعتمد على "
                 "التقدير النقطي منفردًا.")},
    "rmsea_pclose": {
        "pass": ("The hypothesis of close fit (RMSEA <= .05) is not rejected.",
                 "لا يُرفض فرض المطابقة القريبة (RMSEA ≤ 0.05)."),
        "borderline": ("Close fit is rejected at .05 but not at .01.",
                       "يُرفض فرض المطابقة القريبة عند 0.05 دون 0.01."),
        "fail": ("The hypothesis of close fit is rejected.",
                 "يُرفض فرض المطابقة القريبة للنموذج.")},
    "cfi": {
        "pass": ("Incremental fit over the independence model is strong.",
                 "التحسُّن النسبي في المطابقة مقارنةً بنموذج الاستقلال مرتفع."),
        "borderline": ("Incremental fit is acceptable but below the .95 "
                       "convention.",
                       "التحسُّن النسبي مقبول لكنه دون الحد المُتعارف عليه (0.95)."),
        "fail": ("Incremental fit is weak: the model does not improve enough "
                 "on the independence model.",
                 "التحسُّن النسبي ضعيف: النموذج لا يتجاوز نموذج الاستقلال بفارق "
                 "كافٍ.")},
    "tli": {
        "pass": ("Tucker-Lewis index is high after charging the model for its "
                 "free parameters.",
                 "مؤشر تاكر-لويس مرتفع بعد خصم ثمن المعالم الحرة في النموذج."),
        "borderline": ("Tucker-Lewis index is borderline; the model pays for "
                       "its complexity.",
                       "مؤشر تاكر-لويس في المنطقة الحدّية؛ النموذج يدفع ثمن "
                       "تعقيده."),
        "fail": ("Tucker-Lewis index is low: the free parameters do not earn "
                 "the fit they buy.",
                 "مؤشر تاكر-لويس منخفض: المعالم الحرة لا تُبرِّر التحسُّن الذي "
                 "تحقِّقه.")},
    "nfi": {
        "pass": ("Normed fit index is high, though it is known to penalise "
                 "small samples.",
                 "مؤشر المطابقة المعياري مرتفع، مع التنبيه إلى تأثره بصغر حجم "
                 "العينة."),
        "borderline": ("Normed fit index is acceptable but below .95.",
                       "مؤشر المطابقة المعياري مقبول لكنه دون 0.95."),
        "fail": ("Normed fit index is low.",
                 "مؤشر المطابقة المعياري منخفض.")},
    "ifi": {
        "pass": ("Bollen's incremental index is high.",
                 "مؤشر بولن التزايدي مرتفع."),
        "borderline": ("Bollen's incremental index is acceptable but below .95.",
                       "مؤشر بولن التزايدي مقبول لكنه دون 0.95."),
        "fail": ("Bollen's incremental index is low.",
                 "مؤشر بولن التزايدي منخفض.")},
    "rfi": {
        "pass": ("Relative fit index is high.",
                 "مؤشر المطابقة النسبي مرتفع."),
        "borderline": ("Relative fit index is acceptable but below .95.",
                       "مؤشر المطابقة النسبي مقبول لكنه دون 0.95."),
        "fail": ("Relative fit index is low.",
                 "مؤشر المطابقة النسبي منخفض.")},
    "srmr": {
        "pass": ("The average standardised residual is small: fitted and "
                 "observed correlations are close.",
                 "متوسط البواقي المعيارية صغير: الارتباطات المُقدَّرة قريبة من "
                 "المشاهدة."),
        "borderline": ("The average standardised residual is borderline; read "
                       "the residual matrix item by item.",
                       "متوسط البواقي المعيارية في المنطقة الحدّية؛ راجع مصفوفة "
                       "البواقي بندًا ببند."),
        "fail": ("The average standardised residual is large: some item pairs "
                 "are not explained by the model.",
                 "متوسط البواقي المعيارية كبير: هناك أزواج بنود لا يُفسِّر "
                 "النموذج علاقتها.")},
    "gfi": {
        "pass": ("A high share of the observed variances and covariances is "
                 "accounted for.",
                 "النموذج يُفسِّر نسبة مرتفعة من التباينات والتغايرات المشاهدة."),
        "borderline": ("An acceptable share is accounted for; GFI is sensitive "
                       "to the number of items.",
                       "النسبة المُفسَّرة مقبولة، مع التنبيه إلى تأثر المؤشر بعدد "
                       "البنود."),
        "fail": ("Too little of the observed covariance structure is "
                 "accounted for.",
                 "النموذج يُفسِّر قدرًا ضئيلًا من بنية التغايرات المشاهدة.")},
    "agfi": {
        "pass": ("Still high once the index is adjusted for model complexity.",
                 "يبقى المؤشر مرتفعًا بعد تعديله بدرجات الحرية وتعقيد النموذج."),
        "borderline": ("Borderline once adjusted for model complexity.",
                       "في المنطقة الحدّية بعد التعديل بتعقيد النموذج."),
        "fail": ("Low once adjusted for model complexity.",
                 "منخفض بعد التعديل بتعقيد النموذج.")},
    "aic": {"info": (
        "Comparative only: the smaller value wins among models fitted to the "
        "same data; it has no absolute cutoff.",
        "مؤشر مقارنة فقط: الأصغر أفضل بين نماذج بديلة على البيانات نفسها، وليس "
        "له حد فاصل مطلق.")},
    "bic": {"info": (
        "Comparative only, with a heavier penalty on free parameters than AIC.",
        "مؤشر مقارنة فقط، وعقوبته على عدد المعالم الحرة أشد من AIC.")},
    "sabic": {"info": (
        "Comparative only; the sample-size adjustment makes it the fairest of "
        "the three in small samples.",
        "مؤشر مقارنة فقط؛ وتعديله بحجم العينة يجعله أنصف الثلاثة في العينات "
        "الصغيرة.")},
}

_VERDICT_WORDS: Dict[str, Tuple[str, str]] = {
    "pass": ("pass", "مُستوفى"),
    "borderline": ("borderline", "حدّي"),
    "fail": ("fail", "غير مُستوفى"),
    "info": ("descriptive", "وصفي"),
}


def _band(value: float, direction: Optional[str], good: Optional[float],
          acceptable: Optional[float]) -> str:
    """Classify one index value as ``pass`` / ``borderline`` / ``fail``."""
    if direction is None or good is None or acceptable is None:
        return "info"
    if not math.isfinite(value):
        return "info"
    if direction == "ge":
        if value >= good:
            return "pass"
        if value >= acceptable:
            return "borderline"
        return "fail"
    if value <= good:
        return "pass"
    if value <= acceptable:
        return "borderline"
    return "fail"

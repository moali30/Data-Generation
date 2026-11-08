"""
semsim.analysis.reliability
===========================
The reliability and construct-validity table that every measurement paper
reports: Cronbach's alpha with a confidence interval, Joreskog's composite
reliability, McDonald's omega, the average variance extracted, the
Fornell-Larcker criterion and the HTMT ratio.

The coefficients, exactly as implemented
----------------------------------------
Write ``S`` for the item covariance matrix of one factor's ``k`` items, ``R``
for the corresponding correlation matrix, ``lambda_i`` for completely
standardised loadings and ``theta_i = 1 - lambda_i**2`` for the standardised
unique variances.

**Cronbach's alpha** (Cronbach, 1951), the covariance form SPSS prints::

    alpha = k / (k - 1) * (1 - sum_i s_ii / sum_ij s_ij)

the second term being the ratio of summed item variances to the variance of
the total score.  ``sum_ij s_ij`` *is* the variance of the unweighted sum, so
no separate total column is needed.

**Confidence interval for alpha** (Feldt, Woodruff & Salih, 1987).  Under
compound symmetry ``(1 - alpha_hat) / (1 - alpha)`` follows an ``F``
distribution with ``n - 1`` and ``(n - 1)(k - 1)`` degrees of freedom, which
inverts to::

    lower = 1 - (1 - alpha_hat) / F_{gamma/2}(n-1, (n-1)(k-1))
    upper = 1 - (1 - alpha_hat) / F_{1-gamma/2}(n-1, (n-1)(k-1))

The interval is asymmetric about ``alpha_hat``, as it must be: alpha is
bounded above by one.

**Composite reliability**, Joreskog's (1971) ``rho_c``::

    CR = (sum_i lambda_i)^2 / ((sum_i lambda_i)^2 + sum_i theta_i)

**Omega total** (McDonald, 1999, eq. 6.20a)::

    omega_total = (sum_i lambda_i)^2 / (1' R 1)

The numerators are identical; the denominators are not.  ``CR`` rebuilds the
total variance *from the model*, as ``(sum lambda)^2 + sum theta``, and so
assumes strictly uncorrelated residuals.  ``omega_total`` divides by the total
variance actually observed, ``1' R 1``, which carries whatever residual
covariance the items happen to share.  For a perfectly congeneric model the
two coincide; where items share method or wording variance the observed total
variance is the larger of the two denominators and ``omega_total < CR``.  The
gap between them is therefore a diagnostic, not a rounding difference, and
both are reported.

**Average variance extracted** (Fornell & Larcker, 1981)::

    AVE = mean_i(lambda_i^2)

**Fornell-Larcker criterion**: ``sqrt(AVE)`` of each factor on the diagonal
against the factor correlations off it.  A factor passes when its
``sqrt(AVE)`` exceeds its largest absolute correlation with any other factor;
the report passes when every factor does.

**HTMT** (Henseler, Ringle & Sarstedt, 2015)::

    HTMT_ij = mean(r_between) / sqrt(mean(r_within_i) * mean(r_within_j))

with both within-factor means taken over off-diagonal correlations only.  For
a congeneric population this estimates the *disattenuated* factor
correlation, so with loadings of ``0.8`` and a true factor correlation of
``0.5`` the between-item correlations are ``0.5 * 0.64 = 0.32``, the
within-item correlations ``0.64``, and ``HTMT = 0.32 / 0.64 = 0.50`` -- the
factor correlation itself, recovered from item correlations alone.

**Corrected item-total correlation**: the item against the sum of the *other*
items of its factor, computed from the covariance matrix as::

    r_i(rest) = sum_{j != i} s_ij / sqrt(s_ii * sum_{j,k != i} s_jk)

which is algebraically identical to correlating the two scores and costs
nothing beyond the matrix.  ``alpha_if_item_dropped`` applies the alpha
formula to the ``k - 1`` remaining rows and columns.

Where the loadings come from
----------------------------
When an :class:`~semsim.analysis.estimator.EstimationResult` is supplied,
:func:`analyse` reads its ``loadings`` frame (columns ``item``, ``factor``,
``std``) and uses those standardised estimates -- the right thing to do, since
they come from the full model including every cross-loading and residual
covariance.  The module never imports the estimator: the object is duck-typed,
so this file is usable before that module exists and keeps working if its
internals change.

Without an estimation result the loadings are re-derived per factor by
**iterated principal axis factoring** on the factor's own correlation matrix
(Harman, 1976): communalities are initialised at the squared multiple
correlations, the largest eigenvalue-eigenvector pair of the communality-
substituted matrix gives ``lambda = v * sqrt(theta_1)``, and the communalities
are updated until they move by less than ``1e-7``.  If that fails to converge
the first principal component of the untouched correlation matrix is used
instead, which always exists because the matrix is positive semi-definite.

Ordinal items
-------------
With ``use_polychoric=True`` the correlation matrix is estimated by
:func:`semsim.analysis.polychoric.polychoric_matrix`, so the loadings, AVE,
CR, omega, HTMT and factor correlations are all free of the attenuation that
categorising a continuous response causes; alpha then becomes the *ordinal*
alpha of Zumbo, Gadermann and Zeisser (2007), computed from the polychoric
matrix instead of the raw covariances.

References
----------
Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.
    *Psychometrika*, 16(3), 297-334.
Feldt, L. S., Woodruff, D. J., & Salih, F. A. (1987). Statistical inference
    for coefficient alpha. *Applied Psychological Measurement*, 11(1), 93-103.
Joreskog, K. G. (1971). Statistical analysis of sets of congeneric tests.
    *Psychometrika*, 36(2), 109-133.
McDonald, R. P. (1999). *Test Theory: A Unified Treatment*. Erlbaum.
Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models
    with unobservable variables and measurement error. *Journal of Marketing
    Research*, 18(1), 39-50.
Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
    assessing discriminant validity in variance-based structural equation
    modeling. *Journal of the Academy of Marketing Science*, 43(1), 115-135.
Harman, H. H. (1976). *Modern Factor Analysis* (3rd ed.). University of
    Chicago Press.
Zumbo, B. D., Gadermann, A. M., & Zeisser, C. (2007). Ordinal versions of
    coefficients alpha and theta for Likert rating scales. *Journal of Modern
    Applied Statistical Methods*, 6(1), 21-29.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple,
)

import numpy as np
import pandas as pd
from scipy import stats

from semsim.core.errors import SpecificationError
from semsim.core.spec import ModelSpec

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from semsim.analysis.estimator import EstimationResult

__all__ = [
    "ALPHA_FLOOR",
    "AVE_FLOOR",
    "CR_FLOOR",
    "CROSS_LOADING_MARGIN",
    "FactorReliability",
    "ValidityReport",
    "alpha_confidence_interval",
    "alpha_from_matrix",
    "analyse",
    "average_variance_extracted",
    "composite_reliability",
    "cronbach_alpha",
    "heywood_items",
    "htmt_matrix",
    "omega_total",
    "principal_axis_loadings",
]

#: Conventional cut-offs (Nunnally, 1978; Fornell & Larcker, 1981).
ALPHA_FLOOR = 0.70
CR_FLOOR = 0.70
AVE_FLOOR = 0.50

#: An item is flagged as cross-loading when it correlates with a foreign
#: factor within this margin of its own factor.
CROSS_LOADING_MARGIN = 0.10

#: Loadings are kept strictly inside the unit circle so ``theta`` stays
#: positive and no reliability coefficient divides by zero.
_MAX_ABS_LOADING = 0.999


def _bilingual(en: str, ar: str) -> str:
    """One flag string carrying both interface languages."""
    return f"{en} | {ar}"


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class FactorReliability:
    """Reliability and convergent validity of one factor's item set."""

    factor_id: str
    factor_name: str
    n_items: int
    cronbach_alpha: float
    alpha_ci: Tuple[float, float]
    composite_reliability: float        # Joreskog's rho_c
    omega_total: float
    ave: float
    sqrt_ave: float
    mean_inter_item_r: float
    item_total_correlations: Dict[str, float] = field(default_factory=dict)
    alpha_if_item_dropped: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidityReport:
    """The full measurement-quality section of the results."""

    per_factor: List[FactorReliability] = field(default_factory=list)
    factor_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    fornell_larcker: pd.DataFrame = field(default_factory=pd.DataFrame)
    fornell_larcker_passed: bool = True
    htmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    htmt_passed: bool = True
    htmt_threshold: float = 0.85
    cross_loading_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    flags: List[str] = field(default_factory=list)

    def factor(self, factor_id: str) -> Optional[FactorReliability]:
        """Look one factor's reliability up by id, or ``None``."""
        for entry in self.per_factor:
            if entry.factor_id == factor_id:
                return entry
        return None

    def to_table(self, lang: str = "en") -> pd.DataFrame:
        """One row per factor, in the order a results table prints them.

        Parameters
        ----------
        lang : {"en", "ar"}, default "en"
            Reserved for symmetry with the other report objects; the columns
            are machine codes in both languages, so the value is not used
            beyond selecting the factor label already stored.

        Returns
        -------
        pandas.DataFrame
            Columns ``factor``, ``n_items``, ``alpha``, ``alpha_lo``,
            ``alpha_hi``, ``cr``, ``omega_total``, ``ave``, ``sqrt_ave``,
            ``mean_inter_item_r``.
        """
        rows = [{
            "factor": entry.factor_name or entry.factor_id,
            "n_items": entry.n_items,
            "alpha": entry.cronbach_alpha,
            "alpha_lo": entry.alpha_ci[0],
            "alpha_hi": entry.alpha_ci[1],
            "cr": entry.composite_reliability,
            "omega_total": entry.omega_total,
            "ave": entry.ave,
            "sqrt_ave": entry.sqrt_ave,
            "mean_inter_item_r": entry.mean_inter_item_r,
        } for entry in self.per_factor]
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Alpha
# --------------------------------------------------------------------------- #


def alpha_from_matrix(matrix: np.ndarray) -> float:
    """Cronbach's alpha from an item covariance (or correlation) matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric ``(k, k)`` covariance matrix of the items of one scale.
        Passing a *correlation* matrix instead yields the standardised
        (ordinal, when the matrix is polychoric) alpha.

    Returns
    -------
    float
        ``k / (k - 1) * (1 - sum(diag) / sum(all))``.  May be negative when
        the items covary negatively, which is informative and is therefore
        not clipped.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the matrix is not square, has fewer than two items, or implies a
        zero-variance total score.

    Examples
    --------
    Compound symmetry with ``r = 0.5`` and ``k = 4`` gives the Spearman-Brown
    value ``k r / (1 + (k - 1) r) = 0.8``:

    >>> import numpy as np
    >>> R = np.full((4, 4), 0.5); np.fill_diagonal(R, 1.0)
    >>> round(alpha_from_matrix(R), 10)
    0.8

    References
    ----------
    Cronbach, L. J. (1951). Coefficient alpha and the internal structure of
        tests. *Psychometrika*, 16(3), 297-334.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise SpecificationError(
            "Cronbach's alpha needs a square item covariance matrix.",
            "حساب معامل ألفا لكرونباخ يتطلّب مصفوفة تغايُر مربّعة للبنود.",
            shape=tuple(m.shape),
        )
    k = m.shape[0]
    if k < 2:
        raise SpecificationError(
            f"Cronbach's alpha is undefined for a scale of {k} item(s); at "
            f"least two are needed.",
            f"معامل ألفا لكرونباخ غير معرّف لمقياس مكوّن من {k} بند؛ ويلزم "
            f"وجود بندين على الأقل.",
            n_items=int(k),
        )
    if not np.isfinite(m).all():
        raise SpecificationError(
            "The item covariance matrix contains non-finite values.",
            "مصفوفة تغايُر البنود تحتوي على قيم غير منتهية.",
        )
    total_variance = float(m.sum())
    if abs(total_variance) < 1e-12:
        raise SpecificationError(
            "The total score has zero variance, so alpha cannot be computed; "
            "the items are constant or perfectly cancel one another.",
            "تباين الدرجة الكلية يساوي صفرًا، ولذلك يتعذّر حساب معامل ألفا؛ "
            "فالبنود ثابتة أو يلغي بعضها أثر بعض تمامًا.",
            n_items=int(k),
        )
    item_variance = float(np.trace(m))
    return float(k / (k - 1.0) * (1.0 - item_variance / total_variance))




def cronbach_alpha(x: np.ndarray) -> float:
    """Cronbach's alpha of a set of items from the raw case-by-item data.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n, k)`` array of item scores, cases in rows.  A one-dimensional
        array is rejected: a single item has no internal consistency.  Rows
        with any missing value are deleted listwise, as SPSS does by default.

    Returns
    -------
    float
        ``k / (k - 1) * (1 - sum(item variances) / var(total score))`` with
        the unbiased ``n - 1`` divisor throughout.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the array is not two-dimensional, carries fewer than two items or
        two complete cases, or the total score has no variance.

    Examples
    --------
    For items drawn from a compound-symmetric population with ``r = 0.5`` and
    ``k = 4`` the population value is ``4(0.5) / (1 + 3(0.5)) = 0.80``, which
    a sample of 20 000 recovers to within about ``0.005``.

    References
    ----------
    Cronbach, L. J. (1951). Coefficient alpha and the internal structure of
        tests. *Psychometrika*, 16(3), 297-334.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise SpecificationError(
            f"Alpha needs a two-dimensional cases-by-items array; got an "
            f"array with {arr.ndim} dimension(s).",
            f"حساب معامل ألفا يتطلّب مصفوفة ثنائية الأبعاد (حالات × بنود)، "
            f"وقد ورد مصفوفة ذات {arr.ndim} بُعد.",
            ndim=int(arr.ndim),
        )
    complete = arr[np.isfinite(arr).all(axis=1)]
    if complete.shape[0] < 2:
        raise SpecificationError(
            f"Alpha needs at least two complete cases; only "
            f"{complete.shape[0]} of {arr.shape[0]} rows are complete.",
            f"حساب معامل ألفا يتطلّب حالتين مكتملتين على الأقل، ولا توجد سوى "
            f"{complete.shape[0]} حالة مكتملة من أصل {arr.shape[0]}.",
            n_complete=int(complete.shape[0]),
        )
    return alpha_from_matrix(np.cov(complete, rowvar=False, ddof=1))


def alpha_confidence_interval(alpha: float, n: int, k: int,
                              level: float = 0.95) -> Tuple[float, float]:
    """Feldt's ``F``-based confidence interval for coefficient alpha.

    Parameters
    ----------
    alpha : float
        The sample alpha.
    n : int
        Number of complete cases it was computed on.
    k : int
        Number of items.
    level : float, default 0.95
        Confidence level, strictly between 0 and 1.

    Returns
    -------
    (float, float)
        Lower and upper limits.  The interval is asymmetric about ``alpha``
        and its upper limit never exceeds one.  ``(nan, nan)`` is returned
        when there are too few cases or items for the ``F`` reference
        distribution to exist (``n < 3`` or ``k < 2``).

    Notes
    -----
    Under compound symmetry ``(1 - alpha_hat) / (1 - alpha)`` is distributed
    as ``F(n - 1, (n - 1)(k - 1))``, so inverting the two tails gives the
    limits quoted in the module docstring.  Feldt's interval is the one SPSS
    and ``psych::alpha`` report; it assumes equal item variances and
    covariances, and is mildly liberal when that assumption fails
    (Duhachek & Iacobucci, 2004).

    Examples
    --------
    ``alpha = 0.80``, ``n = 300``, ``k = 4`` gives
    ``(0.758, 0.833)`` -- wider below the point estimate than above it.

    References
    ----------
    Feldt, L. S., Woodruff, D. J., & Salih, F. A. (1987). Statistical
        inference for coefficient alpha. *Applied Psychological Measurement*,
        11(1), 93-103.
    Duhachek, A., & Iacobucci, D. (2004). Alpha's standard error (ASE): An
        accurate and precise confidence interval estimate. *Journal of
        Applied Psychology*, 89(5), 792-808.
    """
    if not 0.0 < float(level) < 1.0:
        raise SpecificationError(
            f"A confidence level must lie strictly between 0 and 1; got "
            f"{level:g}.",
            f"يجب أن تكون درجة الثقة بين 0 و 1 تمامًا، والقيمة المُمرَّرة هي "
            f"{level:g}.",
            level=float(level),
        )
    n_i = int(n)
    k_i = int(k)
    if n_i < 3 or k_i < 2 or not np.isfinite(alpha):
        return (float("nan"), float("nan"))

    gamma = 1.0 - float(level)
    df1 = n_i - 1
    df2 = (n_i - 1) * (k_i - 1)
    f_lo = float(stats.f.ppf(gamma / 2.0, df1, df2))
    f_hi = float(stats.f.ppf(1.0 - gamma / 2.0, df1, df2))
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo <= 0:
        return (float("nan"), float("nan"))
    lower = 1.0 - (1.0 - float(alpha)) / f_lo
    upper = 1.0 - (1.0 - float(alpha)) / f_hi
    return (float(lower), float(upper))


# --------------------------------------------------------------------------- #
# Loading-based coefficients
# --------------------------------------------------------------------------- #


def _clean_loadings(loadings: Sequence[float] | np.ndarray,
                    what: str) -> np.ndarray:
    """Validate a loading vector and keep it strictly inside the unit circle."""
    arr = np.asarray(loadings, dtype=float).ravel()
    if arr.size == 0:
        raise SpecificationError(
            f"{what} needs at least one standardised loading.",
            f"حساب {what} يتطلّب تشبُّعًا معياريًا واحدًا على الأقل.",
        )
    if not np.isfinite(arr).all():
        raise SpecificationError(
            f"{what} received a non-finite standardised loading.",
            f"تمّ تمرير تشبُّع معياري غير منتهٍ إلى حساب {what}.",
        )
    return np.clip(arr, -_MAX_ABS_LOADING, _MAX_ABS_LOADING)


def composite_reliability(
    loadings: Sequence[float] | np.ndarray,
    error_variances: Optional[Sequence[float] | np.ndarray] = None,
) -> float:
    """Joreskog's composite reliability ``rho_c`` of a congeneric factor.

    Parameters
    ----------
    loadings : sequence of float
        Completely standardised loadings of the factor's items.
    error_variances : sequence of float, optional
        Standardised unique variances, the same length as ``loadings``.
        ``None`` (the default) substitutes ``theta_i = 1 - lambda_i**2``,
        which is what "completely standardised" implies.

    Returns
    -------
    float
        ``(sum lambda)^2 / ((sum lambda)^2 + sum theta)``, in ``(0, 1]``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the loadings are empty or non-finite, the two vectors have
        different lengths, or the implied total variance is not positive.

    Examples
    --------
    >>> round(composite_reliability([0.8, 0.8, 0.8, 0.8]), 6)
    0.876712

    Notes
    -----
    Unlike alpha, ``rho_c`` does not assume equal loadings, so it is the
    coefficient to report for a congeneric measurement model; unlike
    :func:`omega_total` it reconstructs the total variance from the model and
    therefore ignores correlated residuals.

    A negative ``error_variances`` entry -- a Heywood case -- is clipped to
    zero so the denominator stays usable, which flatters the coefficient and,
    when every entry is negative, returns exactly ``1.0``.  The clip is
    therefore never the whole story: pair the call with
    :func:`heywood_items`, which reports the offending positions, and say so
    in the output.  :func:`analyse` does this and raises a flag naming the
    factor and the item.

    References
    ----------
    Joreskog, K. G. (1971). Statistical analysis of sets of congeneric tests.
        *Psychometrika*, 36(2), 109-133.
    Raykov, T. (1997). Estimation of composite reliability for congeneric
        measures. *Applied Psychological Measurement*, 21(2), 173-184.
    """
    lam = _clean_loadings(loadings, "composite reliability")
    if error_variances is None:
        theta = 1.0 - lam ** 2
    else:
        theta = np.asarray(error_variances, dtype=float).ravel()
        if theta.size != lam.size:
            raise SpecificationError(
                f"There must be one error variance per loading; got "
                f"{lam.size} loadings and {theta.size} error variances.",
                f"يجب أن يقابل كل تشبُّع تباينُ خطأٍ واحد، وقد ورد "
                f"{lam.size} تشبُّعًا مقابل {theta.size} تباينًا للخطأ.",
            )
        if not np.isfinite(theta).all():
            raise SpecificationError(
                "Composite reliability received a non-finite error variance.",
                "تمّ تمرير تباين خطأ غير منتهٍ إلى حساب الثبات المركّب.",
            )
        # A negative standardised unique variance is a Heywood case: the model
        # claims to explain more than the whole of an item's variance, which
        # cannot happen in a population and always means the solution is
        # improper.  The clip is kept, because without it ``(sum lambda)^2 +
        # sum theta`` can reach zero or turn negative and the ratio becomes
        # meaningless -- but clipping *silently* was the defect.  With every
        # theta negative the clip returns exactly 1.0, i.e. the scale is
        # reported as perfectly reliable precisely when the estimates are
        # inadmissible, which is the opposite of what the researcher needs to
        # hear.  The clipped value is still returned; :func:`heywood_items`
        # exposes which items were clipped so a caller -- :func:`analyse`
        # does it -- can name them in the report instead of hiding them.
        theta = np.clip(theta, 0.0, None)
    numerator = float(lam.sum()) ** 2
    denominator = numerator + float(theta.sum())
    if denominator <= 1e-12:
        raise SpecificationError(
            "The implied total variance of the scale is zero, so composite "
            "reliability is undefined.",
            "التباين الكلي المُستنتج للمقياس يساوي صفرًا، ولذلك يتعذّر حساب "
            "الثبات المركّب.",
        )
    return float(numerator / denominator)


def heywood_items(
    loadings: Sequence[float] | np.ndarray,
    error_variances: Optional[Sequence[float] | np.ndarray] = None,
    *, tol: float = 1e-10,
) -> List[int]:
    """Positions of the items whose standardised unique variance is negative.

    Parameters
    ----------
    loadings : sequence of float
        Completely standardised loadings, exactly as handed to
        :func:`composite_reliability` -- *not* clipped, since the clipping is
        what this function is meant to see through.
    error_variances : sequence of float, optional
        Standardised unique variances.  ``None`` (the default) substitutes
        ``theta_i = 1 - lambda_i**2``, so an item is flagged whenever
        ``|lambda_i| > 1``.
    tol : float, default 1e-10
        Only variances below ``-tol`` count, so that the exact zero a
        single-indicator factor produces is not mistaken for a Heywood case.

    Returns
    -------
    list of int
        The offending positions, in order; empty when the solution is proper.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the two vectors have different lengths.

    Notes
    -----
    :func:`composite_reliability` and :func:`omega_total` have to clip a
    negative ``theta`` to keep their denominators usable, which makes an
    improper solution look like a flawless one.  This function is the
    companion that recovers the fact, so a report can state it.  A Heywood
    case is the single most diagnostic thing an estimator can report: it
    usually means an empirically under-identified factor, too small a sample,
    a misspecified model, or an item that is a near-duplicate of another
    (Bollen, 1989, pp. 281-284).

    Examples
    --------
    >>> heywood_items([1.04, 0.8, 0.8])
    [0]
    >>> heywood_items([0.8, 0.8], [-0.2, 0.36])
    [0]
    >>> heywood_items([1.0])
    []

    References
    ----------
    Heywood, H. B. (1931). On finite sequences of real numbers. *Proceedings
        of the Royal Society A*, 134, 486-501.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
    """
    lam = np.asarray(loadings, dtype=float).ravel()
    if error_variances is None:
        theta = 1.0 - lam ** 2
    else:
        theta = np.asarray(error_variances, dtype=float).ravel()
        if theta.size != lam.size:
            raise SpecificationError(
                f"There must be one error variance per loading; got "
                f"{lam.size} loadings and {theta.size} error variances.",
                f"يجب أن يقابل كل تشبُّع تباينُ خطأٍ واحد، وقد ورد "
                f"{lam.size} تشبُّعًا مقابل {theta.size} تباينًا للخطأ.",
            )
    limit = -abs(float(tol))
    flagged = np.isfinite(theta) & (theta < limit)
    return [int(position) for position in np.flatnonzero(flagged)]


def omega_total(loadings: Sequence[float] | np.ndarray,
                correlation_matrix: np.ndarray) -> float:
    """McDonald's omega total for one factor.

    Parameters
    ----------
    loadings : sequence of float
        Completely standardised loadings of the factor's ``k`` items.
    correlation_matrix : numpy.ndarray
        The ``(k, k)`` observed correlation matrix of those same items, in
        the same order.

    Returns
    -------
    float
        ``(sum lambda)^2 / (1' R 1)``, clipped to ``[0, 1]``: the share of
        total-score variance attributable to the common factor, where the
        total-score variance is the one the data actually show.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the shapes disagree, or the total score has no variance.

    Notes
    -----
    Identical to :func:`composite_reliability` when the residuals really are
    uncorrelated, because ``1' R 1`` then equals
    ``(sum lambda)^2 + sum theta``.  When residuals covary, ``1' R 1`` is
    larger and ``omega_total`` is correspondingly smaller -- it charges the
    shared residual variance against the scale instead of crediting it to the
    construct, which is why it is the more conservative of the two.

    References
    ----------
    McDonald, R. P. (1999). *Test Theory: A Unified Treatment*, eq. 6.20a.
        Erlbaum.
    Revelle, W., & Zinbarg, R. E. (2009). Coefficients alpha, beta, omega and
        the glb: Comments on Sijtsma. *Psychometrika*, 74(1), 145-154.
    """
    lam = _clean_loadings(loadings, "omega total")
    R = np.asarray(correlation_matrix, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] != lam.size:
        raise SpecificationError(
            f"Omega total needs a square ({lam.size}, {lam.size}) correlation "
            f"matrix matching the loading vector; got shape "
            f"{tuple(R.shape)}.",
            f"حساب أوميغا الكلي يتطلّب مصفوفة ارتباط مربّعة بأبعاد "
            f"({lam.size}, {lam.size}) مطابقة لمتجه التشبُّعات، وقد وردت "
            f"مصفوفة بأبعاد {tuple(R.shape)}.",
            shape=tuple(R.shape),
        )
    total = float(R.sum())
    if abs(total) < 1e-12:
        raise SpecificationError(
            "The total score has zero variance, so omega total is undefined.",
            "تباين الدرجة الكلية يساوي صفرًا، ولذلك يتعذّر حساب أوميغا الكلي.",
        )
    return float(np.clip(float(lam.sum()) ** 2 / total, 0.0, 1.0))


def average_variance_extracted(loadings: Sequence[float] | np.ndarray) -> float:
    """Average variance extracted: the mean squared standardised loading.

    Parameters
    ----------
    loadings : sequence of float
        Completely standardised loadings of one factor's items.

    Returns
    -------
    float
        ``mean(lambda_i**2)``, in ``[0, 1)``.  Values of ``0.50`` and above
        mean the construct explains more of its items' variance than the
        residuals do.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the loading vector is empty or non-finite.

    Examples
    --------
    >>> round(average_variance_extracted([0.8, 0.8, 0.8, 0.8]), 6)
    0.64

    References
    ----------
    Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation
        models with unobservable variables and measurement error. *Journal of
        Marketing Research*, 18(1), 39-50.
    """
    lam = _clean_loadings(loadings, "average variance extracted")
    return float(np.mean(lam ** 2))


def principal_axis_loadings(correlation_matrix: np.ndarray, *,
                            max_iter: int = 50,
                            tol: float = 1e-7) -> np.ndarray:
    """One-factor loadings by iterated principal axis factoring.

    Parameters
    ----------
    correlation_matrix : numpy.ndarray
        Symmetric ``(k, k)`` correlation matrix of one factor's items.
    max_iter : int, default 50
        Maximum communality updates.
    tol : float, default 1e-7
        Convergence tolerance on the largest communality change.

    Returns
    -------
    numpy.ndarray
        ``k`` standardised loadings, signed so that their sum is positive and
        clipped to ``+/- 0.999``.  A one-item matrix returns ``[1.0]``, there
        being no way to separate common from unique variance.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the matrix is not square or contains non-finite values.

    Notes
    -----
    Communalities start at the squared multiple correlations
    ``1 - 1 / diag(R^-1)`` (the standard Guttman lower bound), falling back on
    each row's largest absolute off-diagonal correlation when ``R`` is
    singular.  Each pass puts the current communalities on the diagonal, takes
    the leading eigenpair ``(theta_1, v_1)`` and sets
    ``lambda = v_1 sqrt(max(theta_1, 0))``.  Should the iteration diverge or
    produce a non-positive leading eigenvalue, the first principal component
    of the untouched correlation matrix is returned instead -- always defined,
    since a correlation matrix is positive semi-definite.

    References
    ----------
    Harman, H. H. (1976). *Modern Factor Analysis* (3rd ed.), chapter 8.
        University of Chicago Press.
    Guttman, L. (1956). "Best possible" systematic estimates of communalities.
        *Psychometrika*, 21(3), 273-285.
    """
    R = np.asarray(correlation_matrix, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise SpecificationError(
            "Principal axis factoring needs a square correlation matrix.",
            "التحليل العاملي بطريقة المحاور الأساسية يتطلّب مصفوفة ارتباط "
            "مربّعة.",
            shape=tuple(R.shape),
        )
    if not np.isfinite(R).all():
        raise SpecificationError(
            "The correlation matrix contains non-finite values; check for "
            "constant items or items with no overlapping cases.",
            "مصفوفة الارتباط تحتوي على قيم غير منتهية؛ يُرجى التحقّق من البنود "
            "الثابتة أو البنود التي لا تشترك في أي حالات.",
        )
    k = R.shape[0]
    if k == 1:
        return np.array([1.0])

    R = 0.5 * (R + R.T)
    off = np.abs(R - np.diag(np.diag(R)))
    fallback_h = np.clip(off.max(axis=1), 1e-3, 0.998)
    try:
        inv_diag = np.diag(np.linalg.inv(R))
        smc = 1.0 - 1.0 / np.where(np.abs(inv_diag) < 1e-12, np.nan, inv_diag)
        communalities = np.where(np.isfinite(smc), smc, fallback_h)
    except np.linalg.LinAlgError:
        communalities = fallback_h
    communalities = np.clip(communalities, 1e-3, 0.998)

    loadings = np.sqrt(communalities)
    for _ in range(int(max_iter)):
        reduced = R.copy()
        np.fill_diagonal(reduced, communalities)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        except np.linalg.LinAlgError:          # pragma: no cover - defensive
            return _first_principal_component(R)
        theta = float(eigenvalues[-1])
        if not np.isfinite(theta) or theta <= 0:
            return _first_principal_component(R)
        candidate = eigenvectors[:, -1] * np.sqrt(theta)
        if not np.isfinite(candidate).all():   # pragma: no cover - defensive
            return _first_principal_component(R)
        new_h = np.clip(candidate ** 2, 1e-6, 0.998)
        shift = float(np.max(np.abs(new_h - communalities)))
        communalities = new_h
        loadings = candidate
        if shift < float(tol):
            break

    if float(loadings.sum()) < 0:
        loadings = -loadings
    return np.clip(loadings, -_MAX_ABS_LOADING, _MAX_ABS_LOADING)


def _first_principal_component(R: np.ndarray) -> np.ndarray:
    """Loadings from the leading eigenpair of an unreduced correlation matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (R + R.T))
    theta = max(float(eigenvalues[-1]), 0.0)
    loadings = eigenvectors[:, -1] * np.sqrt(theta)
    if float(loadings.sum()) < 0:
        loadings = -loadings
    return np.clip(loadings, -_MAX_ABS_LOADING, _MAX_ABS_LOADING)


# --------------------------------------------------------------------------- #
# Discriminant validity
# --------------------------------------------------------------------------- #


def _mean_offdiagonal(block: np.ndarray) -> float:
    """Mean of a square block's off-diagonal entries (``nan`` if 1 x 1)."""
    k = block.shape[0]
    if k < 2:
        return float("nan")
    total = float(block.sum()) - float(np.trace(block))
    return total / (k * (k - 1))


def htmt_matrix(data: pd.DataFrame, factor_items: Mapping[str, Sequence[str]],
                *, correlations: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Heterotrait-monotrait ratio of correlations for every pair of factors.

    Parameters
    ----------
    data : pandas.DataFrame
        Items in columns, cases in rows.  Ignored when ``correlations`` is
        supplied, but still required so the published signature is honoured.
    factor_items : mapping of str to sequence of str
        Factor key to the column names of its items; the keys become the
        index and columns of the result.  Key by factor **id**: display
        labels are free text and two factors may share one, in which case the
        mapping collapses and the call fails.  A factor with fewer than two
        items has no monotrait average and its row and column are ``nan``.
    correlations : pandas.DataFrame, optional
        A pre-computed item correlation matrix to use instead of Pearson's
        ``r`` on ``data`` -- a polychoric matrix, typically, so the ratio is
        free of categorisation attenuation.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix indexed by factor key with ``1.0`` on the diagonal
        and ``HTMT_ij = mean(r_between) / sqrt(mean(r_within_i) *
        mean(r_within_j))`` off it, the within-factor means excluding the
        diagonal.  The entries are **signed**, so the direction of the
        relation stays readable; a threshold comparison must therefore be
        made on ``abs()``, which is what :func:`analyse` does.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If fewer than two factors are supplied, a named item is absent from
        the correlation matrix, or the data cannot be read as a table.

    Notes
    -----
    For a congeneric population HTMT estimates the disattenuated correlation
    between the two constructs, which is the construct correlation itself;
    *absolute* values at or below ``0.85`` (``0.90`` for conceptually close
    constructs) are taken as evidence of discriminant validity -- the
    criterion is about how strongly two constructs are related, not about the
    sign, so a ratio of ``-0.94`` fails it exactly as ``+0.94`` does.  Note that the
    denominator is the *geometric* mean of the two monotrait averages, so with
    equal loadings ``lambda`` and factor correlation ``phi`` the ratio returns
    ``phi`` exactly and not ``phi / lambda^2``.

    References
    ----------
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
        assessing discriminant validity in variance-based structural equation
        modeling. *Journal of the Academy of Marketing Science*, 43(1),
        115-135.
    """
    keys = [str(key) for key in factor_items.keys()]
    if len(keys) < 2:
        raise SpecificationError(
            f"HTMT compares factors in pairs, so at least two factors are "
            f"needed; got {len(keys)}.",
            f"يقارن مؤشّر HTMT بين العوامل مثنى مثنى، ولذلك يلزم وجود عاملين "
            f"على الأقل، وقد ورد {len(keys)} عامل.",
            n_factors=len(keys),
        )

    if correlations is None:
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as exc:
                raise SpecificationError(
                    "The data could not be read as a table of items by cases.",
                    "لم يتمكّن البرنامج من قراءة البيانات كجدول من البنود "
                    "والحالات.",
                    detail=str(exc),
                ) from exc
        corr = data.apply(pd.to_numeric, errors="coerce").corr()
    else:
        # ``.copy()`` is not defensive book-keeping: the two lines below
        # stringify the axis labels, and doing that to the caller's own object
        # silently rewrites the index of a correlation matrix it may still be
        # using (and which it will then no longer be able to index with the
        # original, non-string labels).  A function that is handed a matrix to
        # read must not rewrite it.
        corr = (correlations.copy() if isinstance(correlations, pd.DataFrame)
                else pd.DataFrame(correlations))

    corr.index = [str(i) for i in corr.index]
    corr.columns = [str(c) for c in corr.columns]
    positions: Dict[str, List[int]] = {}
    lookup = {name: idx for idx, name in enumerate(corr.columns)}
    for key, items in zip(keys, factor_items.values()):
        rows: List[int] = []
        for item in items:
            name = str(item)
            if name not in lookup:
                raise SpecificationError(
                    f"Item {name!r} of factor {key!r} is not among the "
                    f"correlation matrix columns.",
                    f"البند {name!r} التابع للعامل {key!r} غير موجود ضمن أعمدة "
                    f"مصفوفة الارتباط.",
                    item=name, factor=key,
                )
            rows.append(lookup[name])
        positions[key] = rows

    values = corr.to_numpy(dtype=float)
    monotrait = {key: _mean_offdiagonal(values[np.ix_(rows, rows)])
                 for key, rows in positions.items()}

    f = len(keys)
    out = np.full((f, f), np.nan, dtype=float)
    for a in range(f):
        out[a, a] = 1.0
    for a in range(f):
        for b in range(a + 1, f):
            rows_a = positions[keys[a]]
            rows_b = positions[keys[b]]
            mono = monotrait[keys[a]] * monotrait[keys[b]]
            if not np.isfinite(mono) or mono <= 0 or not rows_a or not rows_b:
                continue
            hetero = float(np.mean(values[np.ix_(rows_a, rows_b)]))
            out[a, b] = out[b, a] = hetero / float(np.sqrt(mono))
    return pd.DataFrame(out, index=keys, columns=keys)


def _corrected_item_total(S: np.ndarray, i: int,
                          rows: Sequence[int]) -> float:
    """Correlation of item ``i`` with the sum of the other items in ``rows``."""
    others = [j for j in rows if j != i]
    if not others:
        return float("nan")
    covariance = float(S[i, others].sum())
    rest_variance = float(S[np.ix_(others, others)].sum())
    own_variance = float(S[i, i])
    if rest_variance <= 1e-12 or own_variance <= 1e-12:
        return float("nan")
    return covariance / float(np.sqrt(own_variance * rest_variance))


def _factor_correlation(R: np.ndarray, rows_a: Sequence[int],
                        rows_b: Sequence[int], lam_a: np.ndarray,
                        lam_b: np.ndarray) -> float:
    """Least-squares estimate of the correlation between two factors.

    For a congeneric model ``R[a, b] = lambda_a lambda_b phi``, so the value
    of ``phi`` minimising ``sum (R_ab - lambda_a lambda_b phi)^2`` over the
    between-factor block is ``sum(lambda_a lambda_b R_ab) / sum((lambda_a
    lambda_b)^2)``.  Disattenuating this way, rather than correlating
    unit-weighted scale scores, keeps the factor correlations on the latent
    metric that the Fornell-Larcker comparison assumes.
    """
    outer = np.outer(lam_a, lam_b)
    block = R[np.ix_(rows_a, rows_b)]
    denominator = float((outer ** 2).sum())
    if denominator <= 1e-12:
        return float("nan")
    return float(np.clip(float((outer * block).sum()) / denominator,
                         -1.0, 1.0))


def _structure_coefficient(S: np.ndarray, i: int,
                           rows: Sequence[int]) -> float:
    """Correlation of item ``i`` with the summed score of ``rows``.

    The item itself is excluded from the sum whenever it belongs to ``rows``,
    so a home-factor coefficient is the corrected one and is therefore
    directly comparable with the foreign-factor coefficients beside it.
    """
    return _corrected_item_total(S, i, rows)


def _resolve_factor_items(
    spec: ModelSpec, columns: Sequence[str],
) -> Tuple[List[Tuple[str, str, List[str]]],
           List[Tuple[str, str, List[str]]],
           List[str]]:
    """Map the spec's factors onto the columns actually present in the data.

    Returns
    -------
    (primary, second_order, missing)
        ``primary`` holds ``(factor_id, factor_name, column_names)`` for every
        factor items attach to directly -- the construct for a first-order
        construct, each dimension for a second-order one.  ``second_order``
        holds the same triples for second-order constructs taken as whole
        scales.  ``missing`` lists item names the data does not carry.
    """
    available = {str(c) for c in columns}
    missing: List[str] = []

    def column_for(item) -> Optional[str]:
        if item.name and str(item.name) in available:
            return str(item.name)
        if str(item.id) in available:
            return str(item.id)
        missing.append(str(item.name or item.id))
        return None

    primary: List[Tuple[str, str, List[str]]] = []
    second_order: List[Tuple[str, str, List[str]]] = []
    for construct in spec.constructs:
        if construct.is_second_order:
            whole: List[str] = []
            for dimension in construct.dimensions:
                cols = [c for c in (column_for(it) for it in dimension.items)
                        if c is not None]
                whole.extend(cols)
                if cols:
                    primary.append((dimension.id,
                                    spec.factor_label(dimension.id, "en"),
                                    cols))
            if whole:
                second_order.append((construct.id,
                                     spec.factor_label(construct.id, "en"),
                                     whole))
        else:
            cols = [c for c in (column_for(it) for it in construct.items)
                    if c is not None]
            if cols:
                primary.append((construct.id,
                                spec.factor_label(construct.id, "en"),
                                cols))
    return primary, second_order, missing


def _factor_aliases(spec: ModelSpec, factor_id: str,
                    factor_name: str) -> set:
    """Every label an estimation table might use for one factor.

    Estimators label their ``loadings`` rows with whatever they find most
    readable -- the factor id, the short ``name`` from the spec, or the
    bilingual display label -- so all of them are accepted.
    """
    aliases = {str(factor_id), str(factor_name)}
    construct = spec.construct(factor_id)
    dimension = spec.dimension(factor_id)
    for owner in (construct, dimension):
        if owner is not None and getattr(owner, "name", ""):
            aliases.add(str(owner.name))
    for lang in ("en", "ar"):
        label = spec.factor_label(factor_id, lang)
        if label:
            aliases.add(str(label))
    return {alias for alias in aliases if alias}


def _loadings_from_estimation(
    estimation: object, factor_id: str, factor_name: str,
    columns: Sequence[str], spec: ModelSpec,
) -> Optional[np.ndarray]:
    """Pull standardised loadings for one factor out of an estimation result.

    The estimator module is never imported: the argument is duck-typed.  It
    must expose a ``loadings`` table with the columns ``item``, ``factor`` and
    ``std``; ``item`` may carry item names or ids and ``factor`` factor names
    or ids.  ``None`` is returned whenever the table is absent, malformed, or
    does not cover every item of this factor, which makes the caller fall
    back on principal axis factoring for that factor alone.

    The loadings come back **unclipped**.  An estimate with ``|lambda| > 1``
    is a Heywood case, and clipping it here would destroy the only evidence
    of it; the caller inspects the raw vector with :func:`heywood_items`,
    reports what it finds, and clips afterwards.
    """
    table = getattr(estimation, "loadings", None)
    if table is None:
        return None
    try:
        frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    except Exception:
        return None
    if not {"item", "factor", "std"}.issubset(set(map(str, frame.columns))):
        return None

    wanted_factor = _factor_aliases(spec, factor_id, factor_name)
    name_to_id = {str(it.name): str(it.id) for it in spec.all_items()
                  if it.name}
    try:
        subset = frame[frame["factor"].astype(str).isin(wanted_factor)]
    except Exception:
        return None
    if subset.empty:
        return None

    by_key: Dict[str, float] = {}
    for item_key, value in zip(subset["item"].astype(str),
                               subset["std"]):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            by_key[item_key] = number

    out: List[float] = []
    for column in columns:
        if column in by_key:
            out.append(by_key[column])
        elif name_to_id.get(column, "") in by_key:
            out.append(by_key[name_to_id[column]])
        else:
            return None
    return np.asarray(out, dtype=float)


def analyse(data: pd.DataFrame, spec: ModelSpec,
            estimation: Optional["EstimationResult"] = None,
            *, use_polychoric: bool = False,
            htmt_threshold: float = 0.85) -> ValidityReport:
    """Full reliability and construct-validity report for a generated dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed items in columns, cases in rows.  Columns are matched to the
        spec's items by ``Item.name`` first and ``Item.id`` second; rows with
        a missing value on any analysed item are deleted listwise and the fact
        is recorded in ``flags``.
    spec : ModelSpec
        The model whose factors define which items belong together.  Items
        attach to the construct for a first-order construct and to each
        dimension for a second-order one; a second-order construct also gets
        one extra :class:`FactorReliability` row covering all of its items,
        because that is the total-scale reliability researchers quote.  Those
        whole-construct rows are excluded from the discriminant-validity
        blocks, whose item sets must not overlap.
    estimation : EstimationResult, optional
        A fitted model.  When given, its ``loadings`` table (columns ``item``,
        ``factor``, ``std``) supplies the standardised loadings, which is
        preferable because they come from the full model.  Duck-typed: the
        estimator module is never imported, and a table that does not cover a
        factor's items simply sends that factor back to the fallback.
    use_polychoric : bool, default False
        Estimate the item correlation matrix with
        :func:`semsim.analysis.polychoric.polychoric_matrix` instead of
        Pearson's ``r``.  Alpha then becomes the ordinal alpha of Zumbo,
        Gadermann and Zeisser (2007), and every loading-based coefficient is
        computed on the disattenuated matrix.
    htmt_threshold : float, default 0.85
        The *absolute* HTMT value above which a pair of factors is flagged.
        The reported matrix keeps the signed ratio; the comparison is made on
        its magnitude, because discriminant validity concerns the strength of
        the relation between two constructs and not its direction.

    Returns
    -------
    ValidityReport
        Per-factor reliability, the factor correlation matrix, the
        Fornell-Larcker and HTMT tables with their verdicts, a cross-loading
        table and a list of bilingual flags.

        ``cross_loading_table`` carries one row per item: its correlation
        with every factor's remaining items, ``home_correlation``,
        ``max_cross_correlation`` (a magnitude), ``flagged`` -- true when
        that magnitude comes within ``CROSS_LOADING_MARGIN`` of the magnitude
        of the home correlation -- and ``negative_home_correlation``.  The
        last is deliberately a separate verdict: an item that correlates
        *negatively* with its own factor is a serious problem, almost always
        a reverse-worded item nobody recoded, but it is not a cross-loading
        and must not be reported as one.  Both conditions raise their own
        flag.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the data cannot be read as a table, the spec names no factor whose
        items are present in the data, or fewer than two complete cases
        remain.

    Notes
    -----
    Everything is computed from one covariance matrix and one correlation
    matrix, so the cost is a single pass over the data plus ``O(p^2)`` work
    per factor; a 500 x 60 dataset is reported in a few milliseconds unless
    ``use_polychoric`` is set, which adds one bounded optimisation per item
    pair.

    Single-item factors are reported with ``lambda = 1``, ``AVE = 1``,
    ``CR = 1``, ``omega = 1`` and ``alpha = nan`` -- there is no information
    in one indicator with which to separate common from unique variance, so
    these are the values of the standard single-indicator specification
    (loading fixed at one, error variance fixed at zero) rather than
    estimates -- and are flagged.

    A standardised loading above unity implies a negative error variance: a
    Heywood case.  Such a loading is clipped to ``+/- 0.999`` so the
    coefficients stay computable, but every clipped item raises a flag naming
    the factor and the item first, because an improper solution is the most
    important thing in the report and the clip alone would make it look like
    an unusually good one.

    References
    ----------
    Cronbach, L. J. (1951). Coefficient alpha and the internal structure of
        tests. *Psychometrika*, 16(3), 297-334.
    Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation
        models with unobservable variables and measurement error. *Journal of
        Marketing Research*, 18(1), 39-50.
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
        assessing discriminant validity in variance-based structural equation
        modeling. *Journal of the Academy of Marketing Science*, 43(1),
        115-135.
    Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2017). *A
        Primer on Partial Least Squares Structural Equation Modeling* (2nd
        ed.). Sage.
    """
    if isinstance(data, pd.DataFrame):
        frame = data
    else:
        try:
            frame = pd.DataFrame(data)
        except Exception as exc:
            raise SpecificationError(
                "The data could not be read as a table of items by cases.",
                "لم يتمكّن البرنامج من قراءة البيانات كجدول من البنود "
                "والحالات.",
                detail=str(exc),
            ) from exc
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]

    flags: List[str] = []
    primary, second_order, missing = _resolve_factor_items(
        spec, list(frame.columns))
    if missing:
        flags.append(_bilingual(
            f"{len(missing)} item(s) in the model are absent from the data "
            f"and were skipped: {', '.join(missing[:8])}.",
            f"عدد {len(missing)} من بنود النموذج غير موجودة في البيانات وقد "
            f"تم تجاهلها: {'، '.join(missing[:8])}."))
    if not primary:
        raise SpecificationError(
            "None of the model's items could be found among the data "
            "columns, so no reliability can be computed.",
            "لم يُعثر على أي من بنود النموذج بين أعمدة البيانات، ولذلك يتعذّر "
            "حساب الثبات.",
            n_columns=int(frame.shape[1]),
        )

    used_columns: List[str] = []
    for _, _, columns in primary:
        for column in columns:
            if column not in used_columns:
                used_columns.append(column)
    numeric = frame[used_columns].apply(pd.to_numeric, errors="coerce")
    complete_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    n_dropped = int(numeric.shape[0] - complete_mask.sum())
    numeric = numeric.loc[complete_mask]
    n_cases = int(numeric.shape[0])
    if n_cases < 2:
        raise SpecificationError(
            f"Only {n_cases} case(s) are complete on every analysed item, so "
            f"no covariance matrix can be formed.",
            f"لا توجد سوى {n_cases} حالة مكتملة في جميع البنود المُحلَّلة، "
            f"ولذلك يتعذّر تكوين مصفوفة التغايُر.",
            n_complete=n_cases,
        )
    if n_dropped:
        flags.append(_bilingual(
            f"Listwise deletion removed {n_dropped} case(s) with missing "
            f"values; {n_cases} remain.",
            f"أدّى الحذف بالقائمة إلى استبعاد {n_dropped} حالة تحتوي على قيم "
            f"مفقودة، وبقيت {n_cases} حالة."))

    index = {column: position for position, column in enumerate(used_columns)}
    values = numeric.to_numpy(dtype=float)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    sd = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    pearson = covariance / np.outer(sd, sd)
    np.fill_diagonal(pearson, 1.0)

    if use_polychoric:
        from semsim.analysis.polychoric import polychoric_matrix
        correlation = polychoric_matrix(
            numeric, repair_pd=True).to_numpy(dtype=float)
        #: Ordinal alpha and the corrected item-total correlations are read off
        #: the polychoric matrix, whose diagonal is one, so it doubles as the
        #: "covariance" matrix for those two quantities.
        alpha_source = correlation
    else:
        correlation = pearson
        alpha_source = covariance

    # ------------------------------------------------------------------ #
    # Loadings, per factor
    # ------------------------------------------------------------------ #
    all_groups = list(primary) + list(second_order)
    rows_of: Dict[str, List[int]] = {}
    loadings_of: Dict[str, np.ndarray] = {}
    for factor_id, factor_name, columns in all_groups:
        rows = [index[column] for column in columns]
        rows_of[factor_id] = rows
        lam: Optional[np.ndarray] = None
        if estimation is not None:
            lam = _loadings_from_estimation(
                estimation, factor_id, factor_name, columns, spec)
        if lam is None:
            block = correlation[np.ix_(rows, rows)]
            lam = principal_axis_loadings(block)
        lam = np.asarray(lam, dtype=float)

        # The clip below keeps every downstream coefficient defined, but it
        # also turns an inadmissible estimate into a merely excellent-looking
        # one: a standardised loading above unity implies a negative unique
        # variance, and after clipping it reads as 0.999 with nothing to show
        # that anything was wrong.  Heywood cases are the most informative
        # thing a measurement model can report, so they are recorded here --
        # naming the factor and the item -- *before* the clip is applied.
        heywood_label = factor_name or factor_id
        for position in heywood_items(lam):
            raw = float(lam[position])
            flags.append(_bilingual(
                f"Heywood case: item '{columns[position]}' on factor "
                f"'{heywood_label}' has a standardised loading of {raw:.3f}, "
                f"which implies a negative error variance of "
                f"{1.0 - raw ** 2:.3f}. The solution is inadmissible and the "
                f"reliability of this factor is overstated; the loading was "
                f"clipped to {_MAX_ABS_LOADING:.3f} only so the coefficients "
                f"below remain computable.",
                f"حالة هيوود: البند '{columns[position]}' على العامل "
                f"'{heywood_label}' يحمل تشبُّعًا معياريًا مقداره {raw:.3f}، "
                f"وهو ما يستلزم تباينًا سالبًا لخطأ القياس مقداره "
                f"{1.0 - raw ** 2:.3f}. والحل بهذه الصورة غير مقبول إحصائيًا، "
                f"وثبات هذا العامل مُبالَغ فيه؛ ولم يُقصَر التشبُّع عند "
                f"{_MAX_ABS_LOADING:.3f} إلا لكي تبقى المعاملات التالية "
                f"قابلة للحساب."))

        loadings_of[factor_id] = np.clip(
            lam, -_MAX_ABS_LOADING, _MAX_ABS_LOADING)

    # ------------------------------------------------------------------ #
    # Per-factor reliability
    # ------------------------------------------------------------------ #
    per_factor: List[FactorReliability] = []
    for factor_id, factor_name, columns in all_groups:
        rows = rows_of[factor_id]
        lam = loadings_of[factor_id]
        k = len(rows)
        block_r = correlation[np.ix_(rows, rows)]
        block_alpha = alpha_source[np.ix_(rows, rows)]

        if k >= 2:
            alpha = alpha_from_matrix(block_alpha)
            ci = alpha_confidence_interval(alpha, n_cases, k)
            mean_r = _mean_offdiagonal(block_r)
            omega = omega_total(lam, block_r)
        else:
            alpha = float("nan")
            ci = (float("nan"), float("nan"))
            mean_r = float("nan")
            omega = 1.0
            flags.append(_bilingual(
                f"Factor '{factor_name or factor_id}' has a single item, so "
                f"its reliability cannot be estimated from the data.",
                f"العامل '{factor_name or factor_id}' يتكوّن من بند واحد، "
                f"ولذلك يتعذّر تقدير ثباته من البيانات."))

        if k >= 2:
            cr = composite_reliability(lam)
            ave = average_variance_extracted(lam)
        else:
            # A single indicator is not *estimated* at all: the standard
            # single-indicator specification fixes lambda = 1 and theta = 0,
            # which makes AVE, CR and omega exactly one by definition -- and
            # that is what this function's docstring promises and what the
            # omega branch above already reports.  Running the general
            # coefficients instead returned 0.998001, because
            # ``_clean_loadings`` clips every loading to +/- 0.999 to stop the
            # *estimated* case from dividing by zero.  Here nothing can divide
            # by zero (the denominator is exactly one), so the guard bought
            # nothing and merely produced a number that matched neither the
            # documentation, the convention, nor the omega printed beside it.
            cr = 1.0
            ave = 1.0
        item_total = {columns[position]:
                      _corrected_item_total(block_alpha, position,
                                            range(k))
                      for position in range(k)}
        dropped: Dict[str, float] = {}
        for position, column in enumerate(columns):
            keep = [other for other in range(k) if other != position]
            if len(keep) >= 2:
                dropped[column] = alpha_from_matrix(
                    block_alpha[np.ix_(keep, keep)])
            else:
                dropped[column] = float("nan")

        per_factor.append(FactorReliability(
            factor_id=factor_id,
            factor_name=factor_name,
            n_items=k,
            cronbach_alpha=alpha,
            alpha_ci=ci,
            composite_reliability=cr,
            omega_total=omega,
            ave=ave,
            sqrt_ave=float(np.sqrt(max(ave, 0.0))),
            mean_inter_item_r=mean_r,
            item_total_correlations=item_total,
            alpha_if_item_dropped=dropped,
        ))

        label = factor_name or factor_id
        if k < 3:
            flags.append(_bilingual(
                f"Factor '{label}' is measured by {k} item(s); three are the "
                f"conventional minimum for an identified factor.",
                f"العامل '{label}' يُقاس بعدد {k} من البنود، والحد الأدنى "
                f"المتعارف عليه لتحديد العامل هو ثلاثة بنود."))
        if np.isfinite(alpha) and alpha < ALPHA_FLOOR:
            flags.append(_bilingual(
                f"Cronbach's alpha for '{label}' is {alpha:.3f}, below the "
                f"conventional {ALPHA_FLOOR:.2f} floor.",
                f"معامل ألفا لكرونباخ للعامل '{label}' يساوي {alpha:.3f}، "
                f"وهو أقل من الحد المتعارف عليه ({ALPHA_FLOOR:.2f})."))
        if cr < CR_FLOOR:
            flags.append(_bilingual(
                f"Composite reliability for '{label}' is {cr:.3f}, below the "
                f"conventional {CR_FLOOR:.2f} floor.",
                f"الثبات المركّب للعامل '{label}' يساوي {cr:.3f}، وهو أقل من "
                f"الحد المتعارف عليه ({CR_FLOOR:.2f})."))
        if ave < AVE_FLOOR:
            flags.append(_bilingual(
                f"AVE for '{label}' is {ave:.3f}: the residuals explain more "
                f"of the items than the construct does.",
                f"متوسط التباين المُستخلَص للعامل '{label}' يساوي {ave:.3f}؛ "
                f"أي أن البواقي تفسّر من تباين البنود أكثر مما يفسّره "
                f"المفهوم نفسه."))

    # ------------------------------------------------------------------ #
    # Discriminant validity, on the non-overlapping primary factors only
    # ------------------------------------------------------------------ #
    primary_ids = [factor_id for factor_id, _, _ in primary]
    primary_labels = [factor_name or factor_id
                      for factor_id, factor_name, _ in primary]
    f = len(primary_ids)

    phi = np.eye(f, dtype=float)
    for a in range(f):
        for b in range(a + 1, f):
            value = _factor_correlation(
                correlation, rows_of[primary_ids[a]], rows_of[primary_ids[b]],
                loadings_of[primary_ids[a]], loadings_of[primary_ids[b]])
            phi[a, b] = phi[b, a] = value
    factor_correlations = pd.DataFrame(phi, index=primary_labels,
                                       columns=primary_labels)

    ave_by_primary = {entry.factor_id: entry for entry in per_factor
                      if entry.factor_id in set(primary_ids)}
    fl = phi.copy()
    for a, factor_id in enumerate(primary_ids):
        fl[a, a] = ave_by_primary[factor_id].sqrt_ave
    fornell_larcker = pd.DataFrame(fl, index=primary_labels,
                                   columns=primary_labels)

    fl_passed = True
    for a, factor_id in enumerate(primary_ids):
        if f < 2:
            break
        others = [abs(phi[a, b]) for b in range(f) if b != a
                  and np.isfinite(phi[a, b])]
        if not others:
            continue
        largest = max(others)
        if not (ave_by_primary[factor_id].sqrt_ave > largest):
            fl_passed = False
            partner = primary_labels[int(np.argmax(
                [abs(phi[a, b]) if b != a and np.isfinite(phi[a, b]) else -1.0
                 for b in range(f)]))]
            flags.append(_bilingual(
                f"Fornell-Larcker fails for '{primary_labels[a]}': "
                f"sqrt(AVE) = {ave_by_primary[factor_id].sqrt_ave:.3f} does "
                f"not exceed its correlation with '{partner}' "
                f"({largest:.3f}).",
                f"لم يتحقّق معيار فورنيل-لاركر للعامل '{primary_labels[a]}'؛ "
                f"إذ إن الجذر التربيعي لمتوسط التباين المُستخلَص "
                f"({ave_by_primary[factor_id].sqrt_ave:.3f}) لا يتجاوز "
                f"ارتباطه بالعامل '{partner}' ({largest:.3f})."))

    if f >= 2:
        # Keyed by factor *id*, never by display label.  Labels are free text
        # the user types, so two constructs can perfectly well share one; the
        # dict then collapsed to a single entry and HTMT refused to run at
        # all ("at least two factors are needed").  Ids are unique by
        # construction, and the labels are put back on the axes afterwards so
        # the printed table still reads the way the rest of the report does.
        htmt = htmt_matrix(
            numeric,
            {primary_ids[a]: primary[a][2] for a in range(f)},
            correlations=pd.DataFrame(correlation, index=used_columns,
                                      columns=used_columns),
        )
        htmt.index = list(primary_labels)
        htmt.columns = list(primary_labels)
        htmt_values = htmt.to_numpy(dtype=float)
        offdiag = ~np.eye(f, dtype=bool)
        finite = np.isfinite(htmt_values) & offdiag
        # Discriminant validity is a statement about the *strength* of the
        # relation between two constructs, not its direction -- exactly as
        # Fornell-Larcker above compares ``abs(phi)``.  Testing the signed
        # ratio let the worst possible case through: two constructs
        # correlated -0.95 give HTMT = -0.944, which is not "below 0.85" in
        # any meaningful sense, yet ``-0.944 > 0.85`` is False and the pair
        # passed.  The matrix keeps the signed value so the direction stays
        # visible to the reader; only the comparison is made on the magnitude.
        htmt_passed = bool(
            not np.any(np.abs(htmt_values[finite]) > float(htmt_threshold)))
        for a in range(f):
            for b in range(a + 1, f):
                value = htmt_values[a, b]
                if (np.isfinite(value)
                        and abs(value) > float(htmt_threshold)):
                    flags.append(_bilingual(
                        f"HTMT between '{primary_labels[a]}' and "
                        f"'{primary_labels[b]}' is {value:.3f} (magnitude "
                        f"{abs(value):.3f}), which exceeds the "
                        f"{float(htmt_threshold):.2f} threshold in absolute "
                        f"value: the two constructs are not empirically "
                        f"distinct.",
                        f"قيمة مؤشّر HTMT بين '{primary_labels[a]}' و "
                        f"'{primary_labels[b]}' تساوي {value:.3f} (وقيمتها "
                        f"المطلقة {abs(value):.3f})، وهي تتجاوز الحد الفاصل "
                        f"({float(htmt_threshold):.2f}) بالقيمة المطلقة؛ أي أن "
                        f"المفهومين غير متمايزين تجريبيًا."))
    else:
        htmt = pd.DataFrame(np.ones((f, f)), index=primary_labels,
                            columns=primary_labels)
        htmt_passed = True
        flags.append(_bilingual(
            "Discriminant validity needs at least two factors; HTMT and "
            "Fornell-Larcker were not evaluated.",
            "يتطلّب فحص الصدق التمييزي وجود عاملين على الأقل، ولذلك لم يتم "
            "حساب مؤشّر HTMT ولا معيار فورنيل-لاركر."))

    # ------------------------------------------------------------------ #
    # Cross-loading table
    # ------------------------------------------------------------------ #
    cross_rows: List[Dict[str, object]] = []
    for a, (factor_id, factor_name, columns) in enumerate(primary):
        for position, column in enumerate(columns):
            row: Dict[str, object] = {"item": column,
                                      "home_factor": primary_labels[a]}
            item_row = index[column]
            for b, other_id in enumerate(primary_ids):
                row[primary_labels[b]] = _structure_coefficient(
                    alpha_source, item_row, rows_of[other_id])
            home = row[primary_labels[a]]
            home_value = float(home)
            foreign = [float(row[primary_labels[b]]) for b in range(f)
                       if b != a and np.isfinite(float(row[primary_labels[b]]))]
            largest = max((abs(v) for v in foreign), default=float("nan"))
            row["home_correlation"] = home
            row["max_cross_correlation"] = largest

            # ``largest`` is a magnitude (it is a max over absolute values),
            # so the home coefficient must be compared as a magnitude too.
            # Testing it signed mixed two scales: any reverse-keyed item,
            # whose home correlation is negative until the item is recoded,
            # satisfied ``largest > home - 0.10`` automatically and was
            # reported as cross-loading on a factor it may hardly touch.  A
            # negative home correlation is a real and serious problem, but it
            # is a different one, so it gets its own column and its own flag
            # below instead of being disguised as poor discriminant validity.
            negative_home = bool(np.isfinite(home_value) and home_value < 0.0)
            flagged = bool(np.isfinite(largest) and np.isfinite(home_value)
                           and largest > abs(home_value)
                           - CROSS_LOADING_MARGIN)
            row["negative_home_correlation"] = negative_home
            row["flagged"] = flagged
            if flagged:
                flags.append(_bilingual(
                    f"Item '{column}' correlates with a factor other than "
                    f"'{primary_labels[a]}' almost as strongly as with its "
                    f"own ({largest:.3f} against a magnitude of "
                    f"{abs(home_value):.3f}).",
                    f"البند '{column}' يرتبط بعامل آخر غير "
                    f"'{primary_labels[a]}' بقوة تكاد تماثل ارتباطه بعامله "
                    f"الأصلي ({largest:.3f} مقابل قيمة مطلقة مقدارها "
                    f"{abs(home_value):.3f})."))
            if negative_home:
                flags.append(_bilingual(
                    f"Item '{column}' correlates negatively with its own "
                    f"factor '{primary_labels[a]}' ({home_value:.3f}): the "
                    f"item runs against the construct it is meant to measure, "
                    f"which is what a reverse-worded item that has not been "
                    f"recoded looks like. It depresses alpha and the AVE of "
                    f"the whole scale until it is recoded.",
                    f"البند '{column}' يرتبط ارتباطًا سالبًا بعامله الأصلي "
                    f"'{primary_labels[a]}' ({home_value:.3f})؛ أي أن اتجاه "
                    f"البند معاكس للمفهوم الذي يُفترض أن يقيسه، وهذا هو ما "
                    f"يحدث عادةً مع البند المعكوس الصياغة الذي لم يُعَد "
                    f"ترميزه. وهو يخفض معامل ألفا ومتوسط التباين المُستخلَص "
                    f"للمقياس بأكمله إلى أن يُعاد ترميزه."))
            cross_rows.append(row)
    cross_loading_table = pd.DataFrame(cross_rows)
    if not cross_loading_table.empty:
        ordered = (["item", "home_factor"] + primary_labels
                   + ["home_correlation", "max_cross_correlation",
                      "negative_home_correlation", "flagged"])
        seen: List[str] = []
        for name in ordered:
            if name in cross_loading_table.columns and name not in seen:
                seen.append(name)
        cross_loading_table = cross_loading_table[seen]

    return ValidityReport(
        per_factor=per_factor,
        factor_correlations=factor_correlations,
        fornell_larcker=fornell_larcker,
        fornell_larcker_passed=bool(fl_passed),
        htmt=htmt,
        htmt_passed=bool(htmt_passed),
        htmt_threshold=float(htmt_threshold),
        cross_loading_table=cross_loading_table,
        flags=flags,
    )

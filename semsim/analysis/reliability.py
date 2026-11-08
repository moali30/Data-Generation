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
    ``alpha = 0.80``, ``n = 300``, ``k = 4`` gives roughly
    ``(0.766, 0.830)``.

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
        Factor id (or name) to the column names of its items.  A factor with
        fewer than two items has no monotrait average and its row and column
        are ``nan``.
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
        diagonal.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If fewer than two factors are supplied, a named item is absent from
        the correlation matrix, or the data cannot be read as a table.

    Notes
    -----
    For a congeneric population HTMT estimates the disattenuated correlation
    between the two constructs, which is the construct correlation itself;
    values at or below ``0.85`` (``0.90`` for conceptually close constructs)
    are taken as evidence of discriminant validity.  Note that the
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
        corr = correlations

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

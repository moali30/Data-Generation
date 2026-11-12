"""
semsim.power.analysis
=====================
Sample-size determination for the simulated study.

Three answers to "how many respondents do I need?", none of which is a
substitute for the others:

1.  **Test of close fit** -- MacCallum, Browne & Sugawara (1996).  A *global*
    question about the whole model: with ``df`` degrees of freedom, how large
    must ``N`` be before a chi-square test can tell an RMSEA of 0.08 apart
    from an RMSEA of 0.05?  The null hypothesis here is *not* exact fit, so
    the null distribution is a **non-central** chi-square with
    ``lambda_0 = (N - 1) df epsilon_0^2``.  Using a central ``chi2`` for the
    critical value is the classic bug in power code; it inflates the apparent
    power badly, and this module deliberately never calls ``chi2.ppf`` for the
    close-fit critical value.

2.  **Per-parameter power** -- Satorra & Saris (1985).  A *local* question:
    with this population model, how large must ``N`` be before the test of one
    particular path is significant?  The population noncentrality is obtained
    by fitting the model with that path **fixed to zero** to the population
    covariance matrix; the minimum of the discrepancy function times
    ``(N - 1)`` is the noncentrality parameter of a one-degree-of-freedom
    chi-square.

3.  **Monte Carlo** -- Muthen & Muthen (2002).  The empirical answer: generate
    many datasets from the population matrix, estimate each one, and count how
    often each parameter clears ``alpha``.  Reported together with parameter
    bias and the coverage of the nominal 95 % intervals, because a sample size
    that gives 80 % power but 30 % bias is not an adequate sample size.

Alongside those, the rules of thumb researchers and reviewers actually quote:
``N`` per free parameter, ``N`` per item, Kline's minimum of 200, and the
absolute 5:1 floor below which a covariance-structure model should not be
fitted at all.

References
----------
MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis and
    determination of sample size for covariance structure modeling.
    *Psychological Methods*, 1(2), 130-149.
Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
    covariance structure analysis. *Psychometrika*, 50(1), 83-90.
Muthen, L. K., & Muthen, B. O. (2002). How to use a Monte Carlo study to
    decide on sample size and determine power. *Structural Equation
    Modeling*, 9(4), 599-620.
Kline, R. B. (2016). *Principles and Practice of Structural Equation
    Modeling* (4th ed.). Guilford Press.
Bentler, P. M., & Chou, C.-P. (1987). Practical issues in structural
    modeling. *Sociological Methods & Research*, 16(1), 78-117.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, ncx2

from semsim.core.errors import (
    ConvergenceError,
    InfeasibleTargetError,
    SemSimError,
    SpecificationError,
)
from semsim.core.implied import implied_covariance, is_positive_definite
from semsim.core.spec import ModelSpec, Path, PowerMethod

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semsim.core.compile import CompiledModel


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Conventional null RMSEA for the test of close fit (MacCallum et al., 1996).
DEFAULT_RMSEA_NULL: float = 0.05
#: Conventional alternative RMSEA for the test of close fit.
DEFAULT_RMSEA_ALT: float = 0.08
#: Conventional alternative RMSEA for the test of not-close fit.
DEFAULT_RMSEA_ALT_NOT_CLOSE: float = 0.01
#: Kline (2016) rules of thumb.
KLINE_MINIMUM_N: int = 200
#: Bentler & Chou (1987): never fit a model on fewer than 5 cases per
#: free parameter.
ABSOLUTE_FLOOR_RATIO: int = 5
#: Widest sample size the bisection routines will consider.
MAX_SEARCH_N: int = 1_000_000

_Z_95: float = 1.959963984540054

_METHOD_SATORRA_SARIS = "satorra_saris"
_METHOD_WALD_SE = "wald_from_standard_error"
_METHOD_WALD_ANALYTIC = "wald_analytic"

#: Columns of the frame returned by :func:`monte_carlo_power`.
MONTE_CARLO_COLUMNS: Tuple[str, ...] = (
    "parameter", "kind", "source", "target", "population",
    "mean_estimate", "mean_bias", "relative_bias", "empirical_sd",
    "mean_se", "power", "coverage_95", "coverage_metric",
    "n_converged", "replications", "n", "alpha",
)

#: Columns of :attr:`PowerResult.per_path_power`.
PER_PATH_COLUMNS: Tuple[str, ...] = (
    "source", "target", "coefficient", "n_required", "power",
    "path_id", "ncp_per_case", "ncp_at_n", "method", "note",
)


# --------------------------------------------------------------------------- #
# Small bilingual helpers
# --------------------------------------------------------------------------- #


def _note(notes_en: List[str], notes_ar: List[str], en: str, ar: str) -> None:
    """Append one bilingual note to the two parallel lists."""
    notes_en.append(en)
    notes_ar.append(ar)


def _check_df(df: int) -> int:
    """Validate a degrees-of-freedom argument."""
    try:
        value = int(df)
    except (TypeError, ValueError):
        raise SpecificationError(
            "Degrees of freedom must be a whole number.",
            "درجات الحرية يجب أن تكون عددًا صحيحًا.",
            df=df,
        ) from None
    if value < 1:
        raise SpecificationError(
            f"The test of close fit needs at least one degree of freedom, but "
            f"this model has df = {value}. A saturated or under-identified "
            f"model has no global fit to have power against.",
            f"اختبار المطابقة القريبة يحتاج درجة حرية واحدة على الأقل، لكن هذا "
            f"النموذج درجات حريته {value}. النموذج المشبَع أو غير المحدَّد لا "
            f"توجد له مطابقة كلية يمكن اختبار قوتها.",
            df=value,
        )
    return value


def _check_n(n: int, *, minimum: int = 2) -> int:
    """Validate a sample-size argument."""
    try:
        value = int(n)
    except (TypeError, ValueError):
        raise SpecificationError(
            "The sample size must be a whole number.",
            "حجم العينة يجب أن يكون عددًا صحيحًا.",
            n=n,
        ) from None
    if value < minimum:
        raise SpecificationError(
            f"The sample size must be at least {minimum}; {value} was given.",
            f"حجم العينة يجب أن يكون {minimum} على الأقل، والقيمة المُمرَّرة "
            f"هي {value}.",
            n=value,
        )
    return value


def _check_alpha(alpha: float) -> float:
    """Validate a significance level."""
    value = float(alpha)
    if not (0.0 < value < 1.0):
        raise SpecificationError(
            f"The significance level must lie strictly between 0 and 1; "
            f"alpha = {value} was given.",
            f"مستوى الدلالة يجب أن يقع بين صفر وواحد حصرًا، والقيمة المُمرَّرة "
            f"هي {value}.",
            alpha=value,
        )
    return value


def _check_rmsea(value: float, name: str) -> float:
    """Validate one RMSEA argument."""
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise SpecificationError(
            f"{name} must be a finite non-negative number; {value} was given.",
            f"قيمة {name} يجب أن تكون عددًا منتهيًا غير سالب، والقيمة "
            f"المُمرَّرة هي {value}.",
            **{name: value},
        )
    return out


def _check_power(target_power: float) -> float:
    """Validate a target-power argument."""
    value = float(target_power)
    if not (0.0 < value < 1.0):
        raise SpecificationError(
            f"The target power must lie strictly between 0 and 1; "
            f"{value} was given.",
            f"القوة الإحصائية المستهدفة يجب أن تقع بين صفر وواحد حصرًا، "
            f"والقيمة المُمرَّرة هي {value}.",
            target_power=value,
        )
    return value


# --------------------------------------------------------------------------- #
# The published result container
# --------------------------------------------------------------------------- #


@dataclass
class PowerResult:
    """Everything the power stage has to say about one model.

    Attributes
    ----------
    method : PowerMethod
        The procedure that produced :attr:`n_recommended`.
    n_recommended : int
        Suggested number of cases, already passed through the guard rails in
        :class:`~semsim.core.spec.PowerSpec` (``min_n`` / ``max_n``).
    power_at_recommended : float
        Power of the test of close fit at :attr:`n_recommended`, or ``nan``
        when the model has no degrees of freedom to test.
    alpha : float
        Significance level the analysis used.
    df : int
        Degrees of freedom of the model as specified.
    n_free : int
        Free parameters of the model as specified.
    curve : pandas.DataFrame
        Columns ``n``, ``power`` (test of close fit) and
        ``power_not_close_fit``; ready to plot.
    rule_of_thumb : dict of str to int
        ``n_per_parameter``, ``n_per_item``, ``minimum_200`` and
        ``absolute_floor_5_to_1``.
    per_path_power : pandas.DataFrame or None
        One row per structural path, with the Satorra-Saris noncentrality, the
        sample size that path needs on its own, and its power at
        :attr:`n_recommended`.  ``None`` when the model has no paths.
    notes_en, notes_ar : list of str
        Parallel bilingual explanations, concrete about this model's numbers.
    n_items : int
        Observed items in the model.
    rmsea_null, rmsea_alt : float
        Hypotheses of the test of close fit.
    target_power : float
        Power the recommendation aimed at.
    n_from_method : int
        What the formal method alone asked for, before the rules of thumb and
        the guard rails were applied.
    binding_constraint : str
        Which consideration ended up setting :attr:`n_recommended`.

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149.
    """

    method: PowerMethod
    n_recommended: int
    power_at_recommended: float
    alpha: float
    df: int
    n_free: int
    curve: pd.DataFrame
    rule_of_thumb: Dict[str, int]
    per_path_power: Optional[pd.DataFrame]
    notes_en: List[str]
    notes_ar: List[str]
    # -- additions; never positional, so the published contract is intact --- #
    n_items: int = 0
    rmsea_null: float = DEFAULT_RMSEA_NULL
    rmsea_alt: float = DEFAULT_RMSEA_ALT
    target_power: float = 0.80
    n_from_method: int = 0
    binding_constraint: str = ""

    def to_table(self, lang: str = "en") -> pd.DataFrame:
        """One tidy frame summarising the recommendation.

        Parameters
        ----------
        lang : {'en', 'ar'}, default 'en'
            Language of the ``criterion`` labels.

        Returns
        -------
        pandas.DataFrame
            Columns ``criterion`` and ``n``.

        References
        ----------
        Kline, R. B. (2016). *Principles and Practice of Structural Equation
            Modeling* (4th ed.). Guilford Press.
        """
        labels_en = {
            "n_per_parameter": "N per free parameter",
            "n_per_item": "N per item",
            "minimum_200": "Kline minimum (200)",
            "absolute_floor_5_to_1": "Absolute floor (5:1)",
        }
        labels_ar = {
            "n_per_parameter": "حجم العينة لكل معلمة حرة",
            "n_per_item": "حجم العينة لكل بند",
            "minimum_200": "الحد الأدنى عند كلاين (200)",
            "absolute_floor_5_to_1": "الحد المطلق (5 حالات لكل معلمة)",
        }
        labels = labels_ar if lang == "ar" else labels_en
        rows = [
            {"criterion": ("الطريقة الرسمية" if lang == "ar"
                           else f"Formal method ({self.method.value})"),
             "n": self.n_from_method},
        ]
        rows += [{"criterion": labels[k], "n": v}
                 for k, v in self.rule_of_thumb.items()]
        rows.append({"criterion": ("الحجم الموصى به" if lang == "ar"
                                   else "Recommended N"),
                     "n": self.n_recommended})
        return pd.DataFrame(rows, columns=["criterion", "n"])


@dataclass
class PathPowerDetail:
    """Population noncentrality of the test of a single structural path.

    The noncentrality parameter of a Satorra-Saris test is linear in the
    sample size, ``lambda = (N - 1) F_restricted``, so the expensive part --
    the restricted fit to the population matrix -- is done once and reused for
    every ``N`` on a power curve.  :attr:`ncp_per_case` is that ``F``.

    Attributes
    ----------
    path_id : str
        Id of the path in :attr:`semsim.core.spec.ModelSpec.paths`.
    source_id, target_id, source_name, target_name : str
        Endpoints of the path.
    coefficient : float
        Completely standardised coefficient actually compiled into ``B``
        (which may differ from the requested value if the compiler had to
        shrink it to avoid a latent Heywood case).
    ncp_per_case : float
        Noncentrality per ``(N - 1)``: the minimum of the discrepancy function
        for the model with this path fixed to zero, fitted to the population
        covariance matrix.
    method : str
        ``"satorra_saris"``, ``"wald_from_standard_error"`` or
        ``"wald_analytic"`` -- see :func:`satorra_saris_power`.
    df : int
        Degrees of freedom of the test; always 1 for a single fixed path.
    restricted_df_change : int
        Degrees of freedom the restricted model actually gained.  Anything
        other than 1 means fixing the path imposed additional restrictions,
        and the test is no longer a clean one-parameter test.
    converged : bool
        Whether the restricted fit converged.
    notes_en, notes_ar : list of str
        Parallel bilingual notes, including every fallback that was taken.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """

    path_id: str
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    coefficient: float
    ncp_per_case: float
    method: str
    df: int = 1
    restricted_df_change: int = 1
    converged: bool = True
    notes_en: List[str] = field(default_factory=list)
    notes_ar: List[str] = field(default_factory=list)

    def ncp(self, n: int) -> float:
        """Noncentrality parameter at sample size ``n``.

        Parameters
        ----------
        n : int
            Sample size.

        Returns
        -------
        float
            ``(n - 1) * ncp_per_case``.

        References
        ----------
        Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test
            in covariance structure analysis. *Psychometrika*, 50(1), 83-90.
        """
        return float(max(int(n) - 1, 0) * self.ncp_per_case)


# --------------------------------------------------------------------------- #
# 1. MacCallum, Browne & Sugawara (1996): tests of close / not-close fit
# --------------------------------------------------------------------------- #


def noncentrality(df: int, n: int, rmsea: float) -> float:
    """Noncentrality parameter implied by an RMSEA value.

    ``lambda = (N - 1) * df * epsilon^2``, the standard reparameterisation of
    the population discrepancy ``F_0 = df * epsilon^2`` into the scale of the
    chi-square statistic.

    Parameters
    ----------
    df : int
        Model degrees of freedom.
    n : int
        Sample size.
    rmsea : float
        Population RMSEA.

    Returns
    -------
    float
        The noncentrality parameter.

    References
    ----------
    Browne, M. W., & Cudeck, R. (1993). Alternative ways of assessing model
        fit. In K. A. Bollen & J. S. Long (Eds.), *Testing Structural
        Equation Models* (pp. 136-162). Sage.
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149.
    """
    return float((int(n) - 1) * int(df) * float(rmsea) ** 2)


def power_close_fit(df: int, n: int, *,
                    rmsea_null: float = DEFAULT_RMSEA_NULL,
                    rmsea_alt: float = DEFAULT_RMSEA_ALT,
                    alpha: float = 0.05) -> float:
    """Power of the test of close fit (MacCallum, Browne & Sugawara, 1996).

    The null hypothesis is *close* fit, ``H0: epsilon <= epsilon_0`` with
    ``epsilon_0`` conventionally 0.05 -- **not** exact fit.  The null
    distribution of the chi-square statistic is therefore non-central with
    ``lambda_0 = (N - 1) df epsilon_0^2``, and the critical value must come
    from :func:`scipy.stats.ncx2.ppf`, not from a central ``chi2``.  Using the
    central distribution is the single most common error in power code and
    overstates power substantially at small ``df``.

    .. math::

        \\lambda_0 &= (N-1)\\,df\\,\\epsilon_0^2 \\\\
        \\lambda_a &= (N-1)\\,df\\,\\epsilon_a^2 \\\\
        c          &= F^{-1}_{\\chi^2(df,\\lambda_0)}(1-\\alpha) \\\\
        power      &= 1 - F_{\\chi^2(df,\\lambda_a)}(c)

    Parameters
    ----------
    df : int
        Degrees of freedom of the model being tested.  Must be at least 1.
    n : int
        Sample size.  Must be at least 2.
    rmsea_null : float, default 0.05
        RMSEA under the null hypothesis of close fit.
    rmsea_alt : float, default 0.08
        RMSEA under the alternative hypothesis.
    alpha : float, default 0.05
        Significance level of the one-sided test.

    Returns
    -------
    float
        Probability of rejecting close fit when the population RMSEA equals
        ``rmsea_alt``.  Equals ``alpha`` when the two hypotheses coincide.

    Raises
    ------
    semsim.core.errors.SpecificationError
        ``df < 1``, ``n < 2``, a negative RMSEA, or ``alpha`` outside
        ``(0, 1)``.

    Examples
    --------
    >>> round(power_close_fit(100, 132), 3)
    0.802
    >>> round(power_close_fit(8, 100), 3)
    0.153

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149. (Table 4.)
    """
    df = _check_df(df)
    n = _check_n(n)
    alpha = _check_alpha(alpha)
    e0 = _check_rmsea(rmsea_null, "rmsea_null")
    ea = _check_rmsea(rmsea_alt, "rmsea_alt")

    lambda_null = noncentrality(df, n, e0)
    lambda_alt = noncentrality(df, n, ea)
    # The critical value lives under the NON-CENTRAL null.
    crit = float(ncx2.ppf(1.0 - alpha, df, lambda_null))
    if not math.isfinite(crit):
        return float("nan")
    power = 1.0 - float(ncx2.cdf(crit, df, lambda_alt))
    return float(min(max(power, 0.0), 1.0))


def power_not_close_fit(df: int, n: int, *,
                        rmsea_null: float = DEFAULT_RMSEA_NULL,
                        rmsea_alt: float = DEFAULT_RMSEA_ALT_NOT_CLOSE,
                        alpha: float = 0.05) -> float:
    """Power of the test of not-close fit (MacCallum et al., 1996).

    The roles of the two hypotheses are swapped: the null is *poor* fit,
    ``H0: epsilon >= epsilon_0``, and rejecting it is positive evidence that
    the model fits.  The rejection region is therefore the **lower** tail,

    .. math::

        c     &= F^{-1}_{\\chi^2(df,\\lambda_0)}(\\alpha) \\\\
        power &= F_{\\chi^2(df,\\lambda_a)}(c)

    Parameters
    ----------
    df : int
        Degrees of freedom of the model being tested.
    n : int
        Sample size.
    rmsea_null : float, default 0.05
        RMSEA under the null hypothesis of not-close fit.
    rmsea_alt : float, default 0.01
        RMSEA under the alternative hypothesis of (very) close fit.
    alpha : float, default 0.05
        Significance level of the one-sided test.

    Returns
    -------
    float
        Probability of rejecting not-close fit.

    Raises
    ------
    semsim.core.errors.SpecificationError
        Same conditions as :func:`power_close_fit`.

    Examples
    --------
    >>> round(power_not_close_fit(100, 178), 3)
    0.802

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149. (Table 4, right-hand panel.)
    """
    df = _check_df(df)
    n = _check_n(n)
    alpha = _check_alpha(alpha)
    e0 = _check_rmsea(rmsea_null, "rmsea_null")
    ea = _check_rmsea(rmsea_alt, "rmsea_alt")

    lambda_null = noncentrality(df, n, e0)
    lambda_alt = noncentrality(df, n, ea)
    crit = float(ncx2.ppf(alpha, df, lambda_null))
    if not math.isfinite(crit):
        return float("nan")
    power = float(ncx2.cdf(crit, df, lambda_alt))
    return float(min(max(power, 0.0), 1.0))


def _bisect_n(power_fn: Callable[[int], float], target_power: float,
              min_n: int, max_n: int, *, label_en: str, label_ar: str,
              context: Dict[str, Any]) -> int:
    """Smallest integer ``n`` in ``[min_n, max_n]`` with ``power_fn(n) >= target``.

    ``power_fn`` must be monotone non-decreasing in ``n``, which both RMSEA
    tests and the Satorra-Saris test are.
    """
    lo, hi = int(min_n), int(max_n)
    if hi < lo:
        raise SpecificationError(
            f"max_n ({hi}) must not be smaller than min_n ({lo}).",
            f"الحد الأعلى لحجم العينة ({hi}) لا يصح أن يكون أصغر من الحد "
            f"الأدنى ({lo}).",
            min_n=lo, max_n=hi,
        )
    top = power_fn(hi)
    if not (top >= target_power):
        raise InfeasibleTargetError(
            f"A power of {target_power:.2f} for {label_en} cannot be reached "
            f"at any sample size up to {hi}; the best available is "
            f"{top:.3f}. Widen the gap between the null and alternative "
            f"values, lower the target power, or raise max_n.",
            f"لا يمكن الوصول إلى قوة إحصائية قدرها {target_power:.2f} "
            f"لـ{label_ar} عند أي حجم عينة حتى {hi}؛ أقصى قوة متاحة هي "
            f"{top:.3f}. وسِّع الفرق بين الفرض الصفري والفرض البديل، أو خفِّض "
            f"القوة المستهدفة، أو ارفع الحد الأعلى لحجم العينة.",
            target_power=target_power, best_power=top, max_n=hi, **context,
        )
    if power_fn(lo) >= target_power:
        return lo
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fn(mid) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def n_for_close_fit_power(df: int, target_power: float = 0.80, **kw: Any) -> int:
    """Smallest ``N`` whose test of close fit reaches ``target_power``.

    Bisection on ``N``, which is legitimate because
    :func:`power_close_fit` is monotone increasing in ``N`` (the two
    noncentralities grow at rates proportional to ``epsilon_0^2`` and
    ``epsilon_a^2``, and the alternative grows faster).

    Parameters
    ----------
    df : int
        Degrees of freedom of the model.
    target_power : float, default 0.80
        Power to reach.
    **kw
        ``rmsea_null``, ``rmsea_alt`` and ``alpha`` are forwarded to
        :func:`power_close_fit`; ``min_n`` (default 2) and ``max_n``
        (default 1 000 000) bound the search.

    Returns
    -------
    int
        The smallest ``N`` with ``power >= target_power``.  By construction
        ``power_close_fit(df, N - 1) < target_power`` whenever
        ``N > min_n``.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        ``target_power`` is out of reach at ``max_n``.
    semsim.core.errors.SpecificationError
        An invalid argument, as in :func:`power_close_fit`.

    Examples
    --------
    MacCallum et al. (1996), Table 4: a model with 100 degrees of freedom
    needs 132 cases for 80 % power in the test of close fit.

    >>> n_for_close_fit_power(100)
    132
    >>> n_for_close_fit_power(30)
    314

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149. (Table 4.)
    """
    df = _check_df(df)
    target_power = _check_power(target_power)
    min_n = _check_n(kw.pop("min_n", 2))
    max_n = int(kw.pop("max_n", MAX_SEARCH_N))
    unexpected = set(kw) - {"rmsea_null", "rmsea_alt", "alpha"}
    if unexpected:
        raise SpecificationError(
            f"Unknown argument(s) for n_for_close_fit_power: "
            f"{', '.join(sorted(unexpected))}.",
            f"مُعامِلات غير معروفة في دالة تحديد حجم العينة: "
            f"{', '.join(sorted(unexpected))}.",
        )
    return _bisect_n(
        lambda n: power_close_fit(df, n, **kw),
        target_power, min_n, max_n,
        label_en="the test of close fit",
        label_ar="اختبار المطابقة القريبة",
        context={"df": df},
    )


def n_for_not_close_fit_power(df: int, target_power: float = 0.80,
                              **kw: Any) -> int:
    """Smallest ``N`` whose test of not-close fit reaches ``target_power``.

    Parameters
    ----------
    df : int
        Degrees of freedom of the model.
    target_power : float, default 0.80
        Power to reach.
    **kw
        ``rmsea_null``, ``rmsea_alt`` and ``alpha`` are forwarded to
        :func:`power_not_close_fit`; ``min_n`` and ``max_n`` bound the search.

    Returns
    -------
    int
        The smallest ``N`` with ``power >= target_power``.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        ``target_power`` is out of reach at ``max_n``.

    Examples
    --------
    >>> n_for_not_close_fit_power(100)
    178

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149. (Table 4.)
    """
    df = _check_df(df)
    target_power = _check_power(target_power)
    min_n = _check_n(kw.pop("min_n", 2))
    max_n = int(kw.pop("max_n", MAX_SEARCH_N))
    unexpected = set(kw) - {"rmsea_null", "rmsea_alt", "alpha"}
    if unexpected:
        raise SpecificationError(
            f"Unknown argument(s) for n_for_not_close_fit_power: "
            f"{', '.join(sorted(unexpected))}.",
            f"مُعامِلات غير معروفة في دالة تحديد حجم العينة: "
            f"{', '.join(sorted(unexpected))}.",
        )
    return _bisect_n(
        lambda n: power_not_close_fit(df, n, **kw),
        target_power, min_n, max_n,
        label_en="the test of not-close fit",
        label_ar="اختبار عدم المطابقة القريبة",
        context={"df": df},
    )


def power_curve(df: int, *, n_min: int = 50, n_max: int = 1000,
                n_points: int = 40,
                rmsea_null: float = DEFAULT_RMSEA_NULL,
                rmsea_alt: float = DEFAULT_RMSEA_ALT,
                rmsea_alt_not_close: float = DEFAULT_RMSEA_ALT_NOT_CLOSE,
                alpha: float = 0.05,
                include: Sequence[int] = ()) -> pd.DataFrame:
    """Tabulate power against sample size, ready for plotting.

    Parameters
    ----------
    df : int
        Degrees of freedom of the model.
    n_min, n_max : int
        Ends of the sample-size grid.
    n_points : int, default 40
        Number of grid points, spaced geometrically so the interesting small
        samples are not swamped by the large ones.
    rmsea_null, rmsea_alt, alpha : float
        Hypotheses of the test of close fit.
    rmsea_alt_not_close : float, default 0.01
        Alternative for the companion test of not-close fit.
    include : sequence of int, optional
        Extra sample sizes to force onto the grid (typically the recommended
        ``N``).

    Returns
    -------
    pandas.DataFrame
        Columns ``n``, ``power`` and ``power_not_close_fit``, sorted by ``n``.

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149.
    """
    df = _check_df(df)
    lo = max(int(n_min), 2)
    hi = max(int(n_max), lo + 1)
    grid = np.unique(np.round(np.geomspace(lo, hi, max(int(n_points), 2)))
                     ).astype(int)
    extra = [int(v) for v in include if int(v) >= 2]
    grid = np.unique(np.concatenate([grid, np.asarray(extra, dtype=int)])
                     ) if extra else grid
    rows = []
    for n in grid.tolist():
        rows.append({
            "n": int(n),
            "power": power_close_fit(df, int(n), rmsea_null=rmsea_null,
                                     rmsea_alt=rmsea_alt, alpha=alpha),
            "power_not_close_fit": power_not_close_fit(
                df, int(n), rmsea_null=rmsea_null,
                rmsea_alt=rmsea_alt_not_close, alpha=alpha),
        })
    return pd.DataFrame(rows, columns=["n", "power", "power_not_close_fit"])


# --------------------------------------------------------------------------- #
# 2. Satorra & Saris (1985): power for one parameter
# --------------------------------------------------------------------------- #


# Cache of restricted fits.  ``CompiledModel`` is an ordinary (unhashable)
# dataclass, so the cache is keyed on ``id`` and a weak-reference finaliser
# drops the bucket when the model is collected -- a discarded model never
# pins its fits in memory, and a recycled ``id`` never returns a stale one.
_SS_CACHE: Dict[int, Dict[str, "PathPowerDetail"]] = {}


def _ss_bucket(cm: "CompiledModel") -> Dict[str, "PathPowerDetail"]:
    """Per-model cache of Satorra-Saris restricted fits."""
    key = id(cm)
    bucket = _SS_CACHE.get(key)
    if bucket is None:
        bucket = {}
        _SS_CACHE[key] = bucket
        try:
            weakref.finalize(cm, _SS_CACHE.pop, key, None)
        except TypeError:  # pragma: no cover - non-weakrefable model
            _SS_CACHE.pop(key, None)
            return {}
    return bucket


def _resolve_path(spec: ModelSpec, path: Any) -> Path:
    """Accept a :class:`~semsim.core.spec.Path`, a path id, or a label."""
    if hasattr(path, "source") and hasattr(path, "target"):
        return path  # type: ignore[return-value]
    ref = str(path)
    for p in spec.paths:
        if p.id == ref or (p.label and p.label == ref):
            return p
    raise SpecificationError(
        f"No structural path matching {ref!r} exists in this model.",
        f"لا يوجد مسار بنائي يطابق {ref!r} في هذا النموذج.",
        path=ref,
    )


def _exact_moment_sample(sigma: np.ndarray, n: int) -> np.ndarray:
    """A dataset whose unbiased sample covariance is exactly ``sigma``.

    Satorra & Saris (1985) want the discrepancy function evaluated at the
    *population* matrix.  The estimator in :mod:`semsim.analysis.estimator`
    consumes raw data, so a pseudo-sample with the population matrix as its
    sample covariance is manufactured: centre a fixed pseudo-random matrix,
    whiten it with the Cholesky factor of its own covariance, then colour it
    with the Cholesky factor of ``sigma``.  The resulting
    ``numpy.cov(x, ddof=1)`` reproduces ``sigma`` to machine precision, so no
    sampling error contaminates the noncentrality.

    Parameters
    ----------
    sigma : numpy.ndarray
        ``(p, p)`` population covariance matrix, positive definite.
    n : int
        Rows to produce; must exceed ``p``.

    Returns
    -------
    numpy.ndarray
        ``(n, p)`` array with mean zero and sample covariance ``sigma``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        ``sigma`` is singular or not positive definite.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """
    sigma = np.asarray(sigma, dtype=float)
    p = sigma.shape[0]
    n = int(max(n, p + 2))
    try:
        colour = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        raise SpecificationError(
            "The model-implied covariance matrix is singular, so the "
            "population discrepancy of the restricted model cannot be "
            "computed. Check the loadings and path coefficients for a "
            "Heywood case.",
            "المصفوفة المضمَّنة للنموذج منفردة (لا يمكن تحليلها)، ولذلك تعذّر "
            "حساب دالة التناقض للنموذج المقيَّد. راجع التحميلات ومعاملات "
            "المسارات بحثًا عن حالة هيوود.",
        ) from None
    # Deterministic: the pseudo-sample carries no information, only moments.
    rng = np.random.default_rng(0x5E3515)
    a = rng.standard_normal((n, p))
    a -= a.mean(axis=0, keepdims=True)
    s = (a.T @ a) / (n - 1)
    try:
        whiten = np.linalg.inv(np.linalg.cholesky(s))
    except np.linalg.LinAlgError:  # pragma: no cover - needs n <= p
        raise SpecificationError(
            "Could not build a pseudo-sample with the requested population "
            "covariance matrix.",
            "تعذّر بناء عينة صورية تحمل مصفوفة التغاير المطلوبة للمجتمع.",
        ) from None
    return (a @ whiten.T) @ colour.T


def _analytic_wald_ncp(cm: "CompiledModel", path: Path) -> Tuple[float, float]:
    """Latent-regression Wald noncentrality per case, ignoring measurement error.

    Last-resort fallback when no estimator is available at all.  For an
    endogenous latent ``i`` with predictor set ``P`` and standardised
    coefficients ``b``,

    .. math::

        \\psi_{ii} &= 1 - b' \\Phi_{PP} b \\\\
        avar(b_k)  &= \\psi_{ii}\\,[\\Phi_{PP}^{-1}]_{kk} \\\\
        \\lambda   &= (N-1)\\,b_k^2 / avar(b_k)

    This is the asymptotic variance of a regression coefficient among
    *perfectly measured* latent variables, so it is optimistic: real
    indicators attenuate the information and need a larger sample.

    Parameters
    ----------
    cm : CompiledModel
        Compiled population model.
    path : Path
        The path to test.

    Returns
    -------
    (float, float)
        Noncentrality per ``(N - 1)`` and the standardised coefficient.

    Raises
    ------
    semsim.core.errors.SpecificationError
        The path's endpoints are not in the model, or the predictor
        covariance block is singular.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
        Sciences* (2nd ed.). Erlbaum.
    """
    ti = cm.factor_index.get(path.target)
    si = cm.factor_index.get(path.source)
    if ti is None or si is None:
        raise SpecificationError(
            "The path refers to a construct that is not part of the compiled "
            "model.",
            "يشير المسار إلى متغير كامن غير موجود في النموذج المُجمَّع.",
            source=path.source, target=path.target,
        )
    predictors = sorted({j for j in range(cm.n_factors)
                         if abs(float(cm.B[ti, j])) > 0.0} | {si})
    phi_pp = np.asarray(cm.Phi, dtype=float)[np.ix_(predictors, predictors)]
    beta = np.asarray(cm.B, dtype=float)[ti, predictors]
    try:
        inv_phi = np.linalg.inv(phi_pp)
    except np.linalg.LinAlgError:
        raise SpecificationError(
            "The predictors of this path are collinear in the population, so "
            "its asymptotic variance is undefined.",
            "المتغيرات المستقلة لهذا المسار مترابطة ترابطًا تامًا في المجتمع، "
            "ولذلك لا يمكن تعريف تباينه المقارب.",
            source=path.source, target=path.target,
        ) from None
    psi = float(1.0 - beta @ phi_pp @ beta)
    psi = max(psi, 1e-6)
    k = predictors.index(si)
    avar = psi * float(inv_phi[k, k])
    coefficient = float(cm.B[ti, si])
    if avar <= 0.0:
        return 0.0, coefficient
    return coefficient ** 2 / avar, coefficient


def _restricted_spec(spec: ModelSpec, path: Path) -> ModelSpec:
    """A deep copy of ``spec`` with ``path`` removed (i.e. fixed to zero)."""
    import copy as _copy

    restricted = _copy.deepcopy(spec)
    restricted.paths = [p for p in restricted.paths if p.id != path.id]
    # An indirect effect that ran through the deleted path is meaningless now.
    restricted.indirect_effects = [
        ie for ie in restricted.indirect_effects
        if not _chain_uses(ie.chain, path.source, path.target)
    ]
    return restricted


def _chain_uses(chain: Sequence[str], source: str, target: str) -> bool:
    """Whether a mediation chain traverses ``source -> target``."""
    return any(a == source and b == target
               for a, b in zip(list(chain)[:-1], list(chain)[1:]))


def path_power_detail(spec: ModelSpec, cm: "CompiledModel", path: Any, *,
                      use_cache: bool = True) -> PathPowerDetail:
    """Population noncentrality of the Satorra-Saris test for one path.

    Fits the model with ``path`` fixed to zero to the population covariance
    matrix implied by ``cm``.  The minimum of the ML discrepancy function is
    the population discrepancy ``F``, and ``lambda = (N - 1) F`` is the
    noncentrality parameter of a one-degree-of-freedom chi-square.  Because
    ``F`` does not depend on ``N``, one restricted fit serves an entire power
    curve.

    Three routes, tried in order, with every fallback recorded in
    :attr:`PathPowerDetail.notes_en` / ``notes_ar``:

    1. the restricted ML refit (the real Satorra-Saris procedure);
    2. the Wald approximation ``lambda = (N - 1) beta^2 / avar(beta)`` with
       ``avar`` read off the standard error of an unrestricted fit;
    3. a closed-form latent-regression Wald value, used when the estimator
       cannot be imported at all.

    Parameters
    ----------
    spec : ModelSpec
        The population model.
    cm : CompiledModel
        Its compiled form; supplies the population ``Sigma``.
    path : Path or str
        The path to test, or its id.
    use_cache : bool, default True
        Reuse an earlier restricted fit for the same ``cm`` and path.

    Returns
    -------
    PathPowerDetail
        The noncentrality per case and how it was obtained.

    Raises
    ------
    semsim.core.errors.SpecificationError
        ``path`` is not part of ``spec``, or the population matrix is
        singular.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    Saris, W. E., & Satorra, A. (1993). Power evaluations in structural
        equation models. In K. A. Bollen & J. S. Long (Eds.), *Testing
        Structural Equation Models* (pp. 181-204). Sage.
    """
    resolved = _resolve_path(spec, path)
    if use_cache:
        cached = _ss_bucket(cm).get(resolved.id)
        if cached is not None:
            return cached

    notes_en: List[str] = []
    notes_ar: List[str] = []
    ti = cm.factor_index.get(resolved.target)
    si = cm.factor_index.get(resolved.source)
    coefficient = (float(cm.B[ti, si]) if ti is not None and si is not None
                   else float(resolved.resolved_coefficient()))
    source_name = (cm.factor_names[si] if si is not None
                   else spec.factor_label(resolved.source))
    target_name = (cm.factor_names[ti] if ti is not None
                   else spec.factor_label(resolved.target))

    sigma = implied_covariance(cm)
    if not is_positive_definite(sigma):
        raise SpecificationError(
            "The model-implied covariance matrix is not positive definite, so "
            "no population discrepancy can be computed for this path.",
            "المصفوفة المضمَّنة للنموذج ليست معرَّفة إيجابًا، ولذلك لا يمكن "
            "حساب دالة التناقض في المجتمع لهذا المسار.",
            source=resolved.source, target=resolved.target,
        )

    estimate_fn = None
    try:  # lazy, so a circular import can never occur
        from semsim.analysis import estimator as _estimator  # type: ignore
        estimate_fn = getattr(_estimator, "estimate", None)
    except Exception as exc:  # pragma: no cover - module exists in-tree
        _note(notes_en, notes_ar,
              f"FALLBACK: semsim.analysis.estimator could not be imported "
              f"({type(exc).__name__}), so the analytic Wald approximation "
              f"was used instead of a restricted refit.",
              f"بديل: تعذّر استيراد وحدة التقدير ({type(exc).__name__})، "
              f"فاستُخدم تقريب والد التحليلي بدلًا من إعادة التقدير المقيَّدة.")

    detail: Optional[PathPowerDetail] = None
    if callable(estimate_fn):
        n_pseudo = int(max(2 * cm.n_items + 10, 200))
        try:
            data = pd.DataFrame(_exact_moment_sample(sigma, n_pseudo),
                                columns=list(cm.item_names))
            restricted = _restricted_spec(spec, resolved)
            fit = estimate_fn(data, restricted, use_polychoric=False,
                              observed_se=False)
            fmin = float(fit.fmin)
            df_change = int(fit.df) - int(spec.degrees_of_freedom())
            if not math.isfinite(fmin) or fmin < 0.0:
                raise ValueError(f"non-finite restricted minimum: {fmin!r}")
            if not bool(fit.converged):
                _note(notes_en, notes_ar,
                      "The restricted fit did not report convergence; the "
                      "noncentrality below is the best minimum the optimiser "
                      "reached and may understate the true value.",
                      "لم يُبلِّغ التقدير المقيَّد عن تقارب، والقيمة أدناه هي "
                      "أفضل حد أدنى وصلت إليه الخوارزمية وقد تقلّل من القيمة "
                      "الحقيقية.")
            if df_change != 1:
                _note(notes_en, notes_ar,
                      f"Fixing {source_name} -> {target_name} to zero changed "
                      f"the degrees of freedom by {df_change}, not 1, so the "
                      f"test is no longer a clean single-parameter test; the "
                      f"reported power is a one-degree-of-freedom "
                      f"approximation.",
                      f"تثبيت المسار {source_name} ← {target_name} عند الصفر "
                      f"غيَّر درجات الحرية بمقدار {df_change} لا بمقدار واحد، "
                      f"فلم يعد الاختبار اختبارًا نظيفًا لمعلمة واحدة، "
                      f"والقوة المذكورة تقريب بدرجة حرية واحدة.")
            detail = PathPowerDetail(
                path_id=resolved.id, source_id=resolved.source,
                target_id=resolved.target, source_name=source_name,
                target_name=target_name, coefficient=coefficient,
                ncp_per_case=fmin, method=_METHOD_SATORRA_SARIS,
                restricted_df_change=df_change,
                converged=bool(fit.converged),
                notes_en=notes_en, notes_ar=notes_ar,
            )
        except (SemSimError, ValueError, np.linalg.LinAlgError,
                KeyError, TypeError) as exc:
            _note(notes_en, notes_ar,
                  f"FALLBACK: the restricted refit for "
                  f"{source_name} -> {target_name} failed "
                  f"({type(exc).__name__}: {exc}), so the Wald approximation "
                  f"lambda = (N - 1) beta^2 / avar(beta) was used instead.",
                  f"بديل: فشلت إعادة التقدير المقيَّدة للمسار "
                  f"{source_name} ← {target_name} ({type(exc).__name__})، "
                  f"فاستُخدم تقريب والد "
                  f"lambda = (N - 1) beta^2 / avar(beta) بدلًا منها.")
            detail = _wald_from_standard_error(
                spec, cm, resolved, sigma, estimate_fn, source_name,
                target_name, coefficient, notes_en, notes_ar)

    if detail is None:
        ncp_per_case, coefficient = _analytic_wald_ncp(cm, resolved)
        _note(notes_en, notes_ar,
              "The analytic Wald value ignores measurement error, so it is "
              "optimistic: a model with fallible indicators needs a larger "
              "sample than this.",
              "قيمة والد التحليلية تتجاهل خطأ القياس، ولذلك فهي متفائلة: "
              "النموذج ذو المؤشرات غير التامة يحتاج عينة أكبر من ذلك.")
        detail = PathPowerDetail(
            path_id=resolved.id, source_id=resolved.source,
            target_id=resolved.target, source_name=source_name,
            target_name=target_name, coefficient=coefficient,
            ncp_per_case=float(ncp_per_case), method=_METHOD_WALD_ANALYTIC,
            restricted_df_change=1, converged=True,
            notes_en=notes_en, notes_ar=notes_ar,
        )

    if use_cache:
        _ss_bucket(cm)[resolved.id] = detail
    return detail


def _wald_from_standard_error(spec: ModelSpec, cm: "CompiledModel",
                              path: Path, sigma: np.ndarray,
                              estimate_fn: Callable[..., Any],
                              source_name: str, target_name: str,
                              coefficient: float,
                              notes_en: List[str],
                              notes_ar: List[str]) -> PathPowerDetail:
    """Wald noncentrality from the standard error of an unrestricted fit.

    Fits the *full* model to the population matrix at a reference sample size
    ``n_ref`` and converts the reported standard error into an asymptotic
    variance, ``avar = se^2 (n_ref - 1)``, which is free of ``n``.  The
    noncentrality is then ``(N - 1) beta^2 / avar``.

    Parameters
    ----------
    spec : ModelSpec
        The population model.
    cm : CompiledModel
        Its compiled form.
    path : Path
        The path being tested.
    sigma : numpy.ndarray
        Population covariance matrix.
    estimate_fn : callable
        ``semsim.analysis.estimator.estimate``.
    source_name, target_name : str
        Endpoint labels, for the notes.
    coefficient : float
        Standardised coefficient of the path.
    notes_en, notes_ar : list of str
        Note sinks, appended to in place.

    Returns
    -------
    PathPowerDetail
        With ``method`` either ``"wald_from_standard_error"`` or, if the
        unrestricted fit also failed, ``"wald_analytic"``.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """
    n_ref = int(max(2 * cm.n_items + 10, 200))
    try:
        data = pd.DataFrame(_exact_moment_sample(sigma, n_ref),
                            columns=list(cm.item_names))
        fit = estimate_fn(data, spec, use_polychoric=False, observed_se=False)
        table = fit.path_estimates
        row = None
        if table is not None and not table.empty:
            mask = ((table["rhs_id"] == path.source)
                    & (table["lhs_id"] == path.target))
            hits = table[mask]
            if not hits.empty:
                row = hits.iloc[0]
        if row is None:
            raise ValueError("the unrestricted fit reported no such path")
        se = float(row["se"])
        est = float(row["est"])
        if not math.isfinite(se) or se <= 0.0:
            raise ValueError(f"unusable standard error: {se!r}")
        avar = se ** 2 * (n_ref - 1)
        ncp_per_case = est ** 2 / avar if avar > 0.0 else 0.0
        return PathPowerDetail(
            path_id=path.id, source_id=path.source, target_id=path.target,
            source_name=source_name, target_name=target_name,
            coefficient=coefficient, ncp_per_case=float(ncp_per_case),
            method=_METHOD_WALD_SE, restricted_df_change=1,
            converged=bool(fit.converged),
            notes_en=notes_en, notes_ar=notes_ar,
        )
    except (SemSimError, ValueError, KeyError, TypeError,
            np.linalg.LinAlgError) as exc:
        _note(notes_en, notes_ar,
              f"FALLBACK: the unrestricted fit used for the Wald "
              f"approximation also failed ({type(exc).__name__}), so a "
              f"closed-form latent-regression value was used.",
              f"بديل: فشل كذلك التقدير غير المقيَّد المستخدَم في تقريب والد "
              f"({type(exc).__name__})، فاستُخدمت صيغة مغلقة من انحدار "
              f"المتغيرات الكامنة.")
        ncp_per_case, coefficient = _analytic_wald_ncp(cm, path)
        return PathPowerDetail(
            path_id=path.id, source_id=path.source, target_id=path.target,
            source_name=source_name, target_name=target_name,
            coefficient=coefficient, ncp_per_case=float(ncp_per_case),
            method=_METHOD_WALD_ANALYTIC, restricted_df_change=1,
            converged=True, notes_en=notes_en, notes_ar=notes_ar,
        )


def satorra_saris_power(spec: ModelSpec, cm: "CompiledModel", path: Path,
                        n: int, alpha: float = 0.05, *,
                        notes: Optional[List[str]] = None,
                        notes_ar: Optional[List[str]] = None) -> float:
    """Power of the likelihood-ratio test for one structural path.

    Satorra & Saris (1985): refit the model to the **population** covariance
    matrix with the target path **fixed to zero**.  The minimum of the
    discrepancy function, ``F``, times ``(N - 1)`` is the noncentrality
    parameter of a chi-square with one degree of freedom, and

    .. math::

        power = 1 - F_{\\chi^2(1,\\lambda)}\\bigl(
                F^{-1}_{\\chi^2(1)}(1-\\alpha)\\bigr),
        \\qquad \\lambda = (N-1)F.

    Here the null *is* exact (the path really is zero under ``H0``), so the
    critical value is a central ``chi2`` quantile -- the opposite of the
    close-fit test, where the null is itself non-central.

    Parameters
    ----------
    spec : ModelSpec
        The population model.
    cm : CompiledModel
        Its compiled form, supplying the population ``Sigma``.
    path : Path or str
        The path whose power is wanted, or its id.
    n : int
        Sample size.
    alpha : float, default 0.05
        Two-sided significance level of the Wald / LR test on the path.
    notes, notes_ar : list of str, optional
        Sinks for the bilingual notes.  Any fallback taken (a failed
        restricted fit, an unavailable estimator) is appended here; the
        published signature returns only a float, so this is how the caller
        learns that the number is an approximation.

    Returns
    -------
    float
        Probability that the path is significant at ``alpha`` in a sample of
        ``n``.  Equals ``alpha`` in the limit of a zero coefficient.

    Raises
    ------
    semsim.core.errors.SpecificationError
        ``path`` is not in ``spec``, ``n < 2``, ``alpha`` outside ``(0, 1)``,
        or the population matrix is singular.

    Examples
    --------
    >>> import numpy as np
    >>> from semsim.core.compile import compile_model
    >>> from semsim.core.examples import simple_mediation
    >>> spec = simple_mediation()
    >>> cm = compile_model(spec)
    >>> p = satorra_saris_power(spec, cm, spec.paths[0], 300)
    >>> p > 0.99
    True

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """
    n = _check_n(n)
    alpha = _check_alpha(alpha)
    detail = path_power_detail(spec, cm, path)
    if notes is not None:
        notes.extend(detail.notes_en)
    if notes_ar is not None:
        notes_ar.extend(detail.notes_ar)
    crit = float(chi2.ppf(1.0 - alpha, detail.df))
    lam = detail.ncp(n)
    power = 1.0 - float(ncx2.cdf(crit, detail.df, lam))
    return float(min(max(power, 0.0), 1.0))


def n_for_path_power(spec: ModelSpec, cm: "CompiledModel", path: Path,
                     target_power: float = 0.80, alpha: float = 0.05, *,
                     min_n: int = 5, max_n: int = MAX_SEARCH_N) -> int:
    """Smallest ``N`` at which one path reaches ``target_power``.

    Bisection on the Satorra-Saris power, which is monotone increasing in
    ``N`` because the noncentrality ``(N - 1) F`` is linear in ``N``.  The
    restricted fit is performed once and cached.

    Parameters
    ----------
    spec : ModelSpec
        The population model.
    cm : CompiledModel
        Its compiled form.
    path : Path or str
        The path to power.
    target_power : float, default 0.80
        Power to reach.
    alpha : float, default 0.05
        Significance level.
    min_n, max_n : int
        Bounds of the search.

    Returns
    -------
    int
        Smallest ``N`` with ``satorra_saris_power(...) >= target_power``.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        The coefficient is (numerically) zero, so no sample size can detect
        it, or ``target_power`` is out of reach at ``max_n``.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """
    target_power = _check_power(target_power)
    alpha = _check_alpha(alpha)
    detail = path_power_detail(spec, cm, path)
    if detail.ncp_per_case <= 0.0:
        raise InfeasibleTargetError(
            f"The path {detail.source_name} -> {detail.target_name} has a "
            f"population coefficient of {detail.coefficient:.4f}, which is "
            f"indistinguishable from zero, so no sample size gives it power "
            f"above alpha. Give it a non-zero effect size, or leave it out of "
            f"the sample-size calculation.",
            f"المسار {detail.source_name} ← {detail.target_name} معامله في "
            f"المجتمع {detail.coefficient:.4f}، وهو لا يختلف عن الصفر، ولذلك "
            f"لا يوجد حجم عينة يمنحه قوة تزيد على مستوى الدلالة. اضبط حجم "
            f"الأثر على قيمة غير صفرية، أو استبعد هذا المسار من حساب حجم "
            f"العينة.",
            path_id=detail.path_id, coefficient=detail.coefficient,
        )
    crit = float(chi2.ppf(1.0 - alpha, detail.df))

    def _power(n: int) -> float:
        return 1.0 - float(ncx2.cdf(crit, detail.df, detail.ncp(n)))

    return _bisect_n(
        _power, target_power, max(int(min_n), 2), int(max_n),
        label_en=f"the path {detail.source_name} -> {detail.target_name}",
        label_ar=f"المسار {detail.source_name} ← {detail.target_name}",
        context={"path_id": detail.path_id},
    )


# --------------------------------------------------------------------------- #
# 3. Muthen & Muthen (2002): Monte Carlo power
# --------------------------------------------------------------------------- #


@dataclass
class _MCTarget:
    """One parameter watched across the Monte Carlo replications."""

    parameter: str
    kind: str
    source: str
    target: str
    lhs_id: str
    rhs_id: str
    population: float


def _monte_carlo_targets(spec: ModelSpec, cm: "CompiledModel"
                         ) -> List[_MCTarget]:
    """Structural parameters worth monitoring in a Monte Carlo study.

    Every structural path, every second-order loading and every free latent
    covariance.  Population values are read from the *compiled* matrices
    rather than from the spec, so any Heywood shrinkage the compiler applied
    is reflected in the bias column.  Items' loadings are deliberately left
    out: the frame is about the structural claims a reviewer will ask about.
    """
    out: List[_MCTarget] = []
    for p in spec.paths:
        ti = cm.factor_index.get(p.target)
        si = cm.factor_index.get(p.source)
        if ti is None or si is None:
            continue
        out.append(_MCTarget(
            parameter=f"{cm.factor_names[ti]}~{cm.factor_names[si]}",
            kind="structural",
            source=cm.factor_names[si], target=cm.factor_names[ti],
            lhs_id=p.target, rhs_id=p.source,
            population=float(cm.B[ti, si]),
        ))
    for c in spec.constructs:
        if not c.is_second_order:
            continue
        ci = cm.factor_index.get(c.id)
        if ci is None:
            continue
        for d in c.dimensions:
            di = cm.factor_index.get(d.id)
            if di is None:
                continue
            out.append(_MCTarget(
                parameter=f"{cm.factor_names[di]}~{cm.factor_names[ci]}",
                kind="second_order",
                source=cm.factor_names[ci], target=cm.factor_names[di],
                lhs_id=d.id, rhs_id=c.id,
                population=float(cm.B[di, ci]),
            ))
    seen = set()
    for lc in spec.latent_covariances:
        a = cm.factor_index.get(lc.construct_a)
        b = cm.factor_index.get(lc.construct_b)
        if a is None or b is None or a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        lo, hi = key
        out.append(_MCTarget(
            parameter=f"{cm.factor_names[lo]}~~{cm.factor_names[hi]}",
            kind="latent_covariance",
            source=cm.factor_names[lo], target=cm.factor_names[hi],
            lhs_id=cm.factor_ids[lo], rhs_id=cm.factor_ids[hi],
            population=float(cm.Phi[lo, hi]),
        ))
    return out


def _match_row(table: pd.DataFrame, item: _MCTarget) -> Optional[pd.Series]:
    """Locate ``item`` in an estimator parameter table."""
    if table is None or table.empty:
        return None
    kinds = table["kind"].astype(str)
    mask = kinds == item.kind
    if item.kind == "latent_covariance":
        pair = {item.lhs_id, item.rhs_id}
        ids = [set(v) for v in zip(table["lhs_id"].astype(str),
                                   table["rhs_id"].astype(str))]
        mask = mask & np.array([s == pair for s in ids])
    else:
        mask = (mask
                & (table["lhs_id"].astype(str) == item.lhs_id)
                & (table["rhs_id"].astype(str) == item.rhs_id))
    hits = table[mask]
    if hits.empty:
        return None
    return hits.iloc[0]


def monte_carlo_power(spec: ModelSpec, *, n: int, replications: int = 200,
                      alpha: float = 0.05, seed: Optional[int] = None,
                      progress: Optional[Callable[[str, float], Any]] = None,
                      cm: Optional["CompiledModel"] = None
                      ) -> pd.DataFrame:
    """Empirical power, bias and interval coverage by simulation.

    The procedure Muthen & Muthen (2002) recommend: draw ``replications``
    samples of size ``n`` from the population covariance matrix with
    :class:`scipy.stats.multivariate_normal`, estimate the model in each, and
    report for every structural parameter

    * **power** -- the proportion of replications with ``p < alpha``;
    * **bias** -- mean estimate minus population value, raw and relative;
    * **coverage** -- the proportion of nominal 95 % intervals that contain
      the population value.

    Muthen & Muthen's own criteria are power at or above 0.80, relative bias
    under 10 % and coverage between 0.91 and 0.98; a sample size that buys
    power while wrecking coverage has not solved the problem.

    Bias is reported on the **completely standardised** scale, because that is
    the metric the spec is written in: ``population`` is read straight off the
    compiled ``B`` / ``Phi``.  Coverage, however, needs the parameter and its
    standard error in the *same* metric, so it is computed in the estimator's
    own marker metric against the population value obtained by fitting the
    model to a pseudo-sample whose covariance matrix is exactly
    ``Sigma(theta)`` (see :func:`_exact_moment_sample`); at that point the
    discrepancy is zero, so the estimates *are* the population values.  The
    column ``coverage_metric`` records which metric was used.  If that
    reference fit is unavailable, coverage falls back on the first-order
    proportional approximation ``se_std = se |std / est|`` and
    ``coverage_metric`` says ``"standardised_approx"``; that approximation
    holds the standardising factors fixed and is conservative.  ``mean_se``
    is always the standardised proportional approximation, so it can be read
    next to ``empirical_sd``.

    Parameters
    ----------
    spec : ModelSpec
        The population model to simulate from.
    n : int
        Sample size of each replication.
    replications : int, default 200
        Number of datasets.
    alpha : float, default 0.05
        Significance level for the power column.
    seed : int, optional
        Seed of the :class:`numpy.random.Generator` driving the draws.  No
        global ``numpy.random`` state is ever touched.
    progress : callable, optional
        Called as ``progress(message, fraction)`` after each replication.
        Returning ``False`` stops the study early and the frame reports the
        replications completed so far, which is what makes a long run
        interruptible from a UI thread.  ``KeyboardInterrupt`` has the same
        effect.
    cm : CompiledModel, optional
        Pre-compiled model; compiled from ``spec`` when omitted.

    Returns
    -------
    pandas.DataFrame
        One row per monitored structural parameter -- every path, every
        second-order loading and every free latent covariance -- with the
        columns listed in :data:`MONTE_CARLO_COLUMNS`.  A model with no
        structural parameters at all yields an empty frame with those
        columns.  A parameter the estimator holds fixed for identification --
        the first second-order loading of each second-order construct -- has
        no standard error and therefore ``NaN`` in ``power`` and
        ``coverage_95``; its bias columns are still meaningful.

    Raises
    ------
    semsim.core.errors.SpecificationError
        ``n`` or ``replications`` is too small, or the population matrix is
        not positive definite.
    semsim.core.errors.ConvergenceError
        Not one replication could be estimated.

    References
    ----------
    Muthen, L. K., & Muthen, B. O. (2002). How to use a Monte Carlo study to
        decide on sample size and determine power. *Structural Equation
        Modeling*, 9(4), 599-620.
    Boomsma, A. (2013). Reporting Monte Carlo studies in structural equation
        modeling. *Structural Equation Modeling*, 20(3), 518-540.
    """
    from scipy.stats import multivariate_normal

    n = _check_n(n, minimum=3)
    alpha = _check_alpha(alpha)
    reps = int(replications)
    if reps < 1:
        raise SpecificationError(
            f"A Monte Carlo study needs at least one replication; "
            f"{reps} was requested.",
            f"محاكاة مونت كارلو تحتاج تكرارًا واحدًا على الأقل، والعدد "
            f"المطلوب هو {reps}.",
            replications=reps,
        )
    if cm is None:
        from semsim.core.compile import compile_model
        cm = compile_model(spec)
    sigma = implied_covariance(cm)
    if not is_positive_definite(sigma):
        raise SpecificationError(
            "The model-implied covariance matrix is not positive definite, so "
            "no data can be generated from it. Repair the specification "
            "first (the compiler's warnings say where).",
            "المصفوفة المضمَّنة للنموذج ليست معرَّفة إيجابًا، ولذلك لا يمكن "
            "توليد بيانات منها. أصلح مواصفات النموذج أولًا (تحذيرات المُجمِّع "
            "تبيّن الموضع).",
        )

    targets = _monte_carlo_targets(spec, cm)
    if not targets:
        return pd.DataFrame(columns=list(MONTE_CARLO_COLUMNS))

    try:  # lazy, so a circular import can never occur
        from semsim.analysis import estimator as _estimator  # type: ignore
    except Exception as exc:  # pragma: no cover - module exists in-tree
        raise ConvergenceError(
            f"A Monte Carlo power study needs the SEM estimator, which could "
            f"not be loaded ({type(exc).__name__}). Use the analytic power "
            f"methods instead.",
            f"تحتاج محاكاة مونت كارلو إلى وحدة تقدير النماذج، وقد تعذّر "
            f"تحميلها ({type(exc).__name__}). استخدم طرق القوة التحليلية "
            f"بدلًا منها.",
        ) from exc

    rng = np.random.default_rng(seed)
    distribution = multivariate_normal(mean=np.zeros(cm.n_items), cov=sigma,
                                       allow_singular=True)
    columns = list(cm.item_names)

    estimates: Dict[str, List[float]] = {t.parameter: [] for t in targets}
    ses: Dict[str, List[float]] = {t.parameter: [] for t in targets}
    hits: Dict[str, int] = {t.parameter: 0 for t in targets}
    covered: Dict[str, int] = {t.parameter: 0 for t in targets}
    tested: Dict[str, int] = {t.parameter: 0 for t in targets}
    n_interval: Dict[str, int] = {t.parameter: 0 for t in targets}
    completed = 0
    failures = 0

    # Population values in the estimator's own metric, for honest coverage.
    population_unstd: Dict[str, float] = {}
    coverage_metric = "standardised_approx"
    try:
        reference = _estimator.estimate(
            pd.DataFrame(_exact_moment_sample(sigma,
                                              max(2 * cm.n_items + 10, 200)),
                         columns=columns),
            spec, use_polychoric=False, observed_se=False)
        if float(reference.fmin) < 1e-6:
            table = reference.parameter_table()
            for t in targets:
                row = _match_row(table, t)
                if row is not None and math.isfinite(float(row["est"])):
                    population_unstd[t.parameter] = float(row["est"])
            if len(population_unstd) == len(targets):
                coverage_metric = "unstandardised"
            else:
                population_unstd.clear()
    except Exception:
        population_unstd.clear()

    try:
        for rep in range(reps):
            sample = np.atleast_2d(distribution.rvs(size=n, random_state=rng))
            frame = pd.DataFrame(sample, columns=columns)
            try:
                fit = _estimator.estimate(frame, spec, use_polychoric=False,
                                          observed_se=False)
                table = fit.parameter_table()
            except Exception:
                failures += 1
                if progress is not None:
                    if progress("monte_carlo", (rep + 1) / reps) is False:
                        break
                continue
            completed += 1
            for t in targets:
                row = _match_row(table, t)
                if row is None:
                    continue
                est = float(row["est"])
                std = float(row["std"])
                se = float(row["se"])
                p_value = float(row["p"])
                if math.isfinite(std):
                    estimates[t.parameter].append(std)
                if math.isfinite(p_value):
                    tested[t.parameter] += 1
                    if p_value < alpha:
                        hits[t.parameter] += 1
                if (math.isfinite(se) and se > 0.0 and math.isfinite(std)
                        and abs(est) > 1e-12):
                    ses[t.parameter].append(se * abs(std / est))
                if math.isfinite(se) and se > 0.0:
                    if coverage_metric == "unstandardised":
                        centre = est
                        truth = population_unstd[t.parameter]
                        half = _Z_95 * se
                    elif math.isfinite(std) and abs(est) > 1e-12:
                        centre = std
                        truth = t.population
                        half = _Z_95 * se * abs(std / est)
                    else:
                        continue
                    n_interval[t.parameter] += 1
                    if centre - half <= truth <= centre + half:
                        covered[t.parameter] += 1
            if progress is not None:
                if progress("monte_carlo", (rep + 1) / reps) is False:
                    break
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass

    if completed == 0:
        raise ConvergenceError(
            f"None of the {reps} Monte Carlo replications could be "
            f"estimated, so no empirical power is available. The model may be "
            f"under-identified at N = {n}, or the sample size may be too "
            f"small for the number of free parameters "
            f"({spec.count_free_parameters()}).",
            f"تعذّر تقدير أي تكرار من تكرارات مونت كارلو الـ{reps}، ولذلك لا "
            f"تتوفر قوة إحصائية تجريبية. قد يكون النموذج غير محدَّد عند حجم "
            f"عينة {n}، أو قد يكون حجم العينة أصغر من أن يكفي لعدد المعلمات "
            f"الحرة ({spec.count_free_parameters()}).",
            n=n, replications=reps,
        )

    rows: List[Dict[str, Any]] = []
    for t in targets:
        vals = np.asarray(estimates[t.parameter], dtype=float)
        se_vals = np.asarray(ses[t.parameter], dtype=float)
        k_tested = tested[t.parameter]
        k_cover = n_interval[t.parameter]
        mean_est = float(vals.mean()) if vals.size else float("nan")
        bias = mean_est - t.population if vals.size else float("nan")
        rel = (bias / t.population
               if vals.size and abs(t.population) > 1e-12 else float("nan"))
        rows.append({
            "parameter": t.parameter,
            "kind": t.kind,
            "source": t.source,
            "target": t.target,
            "population": t.population,
            "mean_estimate": mean_est,
            "mean_bias": bias,
            "relative_bias": rel,
            "empirical_sd": (float(vals.std(ddof=1)) if vals.size > 1
                             else float("nan")),
            "mean_se": float(se_vals.mean()) if se_vals.size else float("nan"),
            "power": (hits[t.parameter] / k_tested if k_tested
                      else float("nan")),
            "coverage_95": (covered[t.parameter] / k_cover if k_cover
                            else float("nan")),
            "coverage_metric": coverage_metric,
            "n_converged": int(completed),
            "replications": int(reps),
            "n": int(n),
            "alpha": float(alpha),
        })
    return pd.DataFrame(rows, columns=list(MONTE_CARLO_COLUMNS))


# --------------------------------------------------------------------------- #
# 4. Rules of thumb
# --------------------------------------------------------------------------- #


def rules_of_thumb(spec: ModelSpec, *, per_parameter: int = 10,
                   per_item: int = 10) -> Dict[str, int]:
    """The sample-size heuristics reviewers quote, in cases.

    Parameters
    ----------
    spec : ModelSpec
        The model to size.
    per_parameter : int, default 10
        Cases per free parameter (Bentler & Chou, 1987, call 10:1 adequate
        for normal data and 5:1 the absolute floor).
    per_item : int, default 10
        Cases per observed item.

    Returns
    -------
    dict of str to int
        ``n_per_parameter``, ``n_per_item``, ``minimum_200`` (Kline's rule of
        thumb for a model of average complexity) and
        ``absolute_floor_5_to_1``.

    References
    ----------
    Bentler, P. M., & Chou, C.-P. (1987). Practical issues in structural
        modeling. *Sociological Methods & Research*, 16(1), 78-117.
    Kline, R. B. (2016). *Principles and Practice of Structural Equation
        Modeling* (4th ed.). Guilford Press.
    """
    n_free = int(spec.count_free_parameters())
    n_items = int(spec.n_items)
    return {
        "n_per_parameter": int(max(per_parameter, 1) * n_free),
        "n_per_item": int(max(per_item, 1) * n_items),
        "minimum_200": KLINE_MINIMUM_N,
        "absolute_floor_5_to_1": int(ABSOLUTE_FLOOR_RATIO * n_free),
    }


# --------------------------------------------------------------------------- #
# 5. The assembled analysis
# --------------------------------------------------------------------------- #


def _per_path_frame(spec: ModelSpec, cm: "CompiledModel", n: int,
                    alpha: float, target_power: float,
                    notes_en: List[str], notes_ar: List[str],
                    max_n: int,
                    progress: Optional[Callable[[str, float], Any]] = None
                    ) -> Optional[pd.DataFrame]:
    """Satorra-Saris table for every structural path, or ``None``."""
    if not spec.paths:
        return None
    rows: List[Dict[str, Any]] = []
    for index, p in enumerate(spec.paths):
        if progress is not None:
            # Each path costs a model fit or two: on a large model this loop
            # is a minute long, and a bar frozen for that minute looks hung.
            progress("status.power", index / len(spec.paths))
        try:
            detail = path_power_detail(spec, cm, p)
        except SemSimError as exc:
            _note(notes_en, notes_ar,
                  f"The per-path power of {spec.factor_label(p.source)} -> "
                  f"{spec.factor_label(p.target)} could not be computed: "
                  f"{exc.message_en}",
                  f"تعذّر حساب القوة الإحصائية للمسار "
                  f"{spec.factor_label(p.source, 'ar')} ← "
                  f"{spec.factor_label(p.target, 'ar')}: {exc.message_ar}")
            continue
        crit = float(chi2.ppf(1.0 - alpha, detail.df))
        power_at_n = 1.0 - float(ncx2.cdf(crit, detail.df, detail.ncp(n)))
        try:
            needed: Any = n_for_path_power(spec, cm, p,
                                           target_power=target_power,
                                           alpha=alpha, max_n=max_n)
            note = ""
        except InfeasibleTargetError as exc:
            needed = pd.NA
            note = exc.message_en
        rows.append({
            "source": detail.source_name,
            "target": detail.target_name,
            "coefficient": detail.coefficient,
            "n_required": needed,
            "power": float(min(max(power_at_n, 0.0), 1.0)),
            "path_id": detail.path_id,
            "ncp_per_case": detail.ncp_per_case,
            "ncp_at_n": detail.ncp(n),
            "method": detail.method,
            "note": note,
        })
        for en, ar in zip(detail.notes_en, detail.notes_ar):
            _note(notes_en, notes_ar, en, ar)
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=list(PER_PATH_COLUMNS))
    frame["n_required"] = frame["n_required"].astype("Int64")
    return frame


def _refresh_per_path_power(frame: pd.DataFrame, n: int,
                            alpha: float) -> pd.DataFrame:
    """Re-evaluate the ``power`` / ``ncp_at_n`` columns at a new ``N``.

    The expensive part of a Satorra-Saris table -- the restricted fit -- gives
    a noncentrality *per case*, so moving the table to a different sample size
    is pure arithmetic.

    Parameters
    ----------
    frame : pandas.DataFrame
        A per-path table carrying ``ncp_per_case``.
    n : int
        The sample size to report at.
    alpha : float
        Significance level.

    Returns
    -------
    pandas.DataFrame
        A copy with ``ncp_at_n`` and ``power`` recomputed.

    References
    ----------
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """
    crit = float(chi2.ppf(1.0 - alpha, 1))
    out = frame.copy()
    out["ncp_at_n"] = (int(n) - 1) * out["ncp_per_case"].astype(float)
    out["power"] = [
        float(min(max(1.0 - float(ncx2.cdf(crit, 1, lam)), 0.0), 1.0))
        for lam in out["ncp_at_n"].tolist()
    ]
    return out


def analyse(spec: ModelSpec, cm: Optional["CompiledModel"] = None,
            progress: Optional[Callable[[str, float], Any]] = None
            ) -> PowerResult:
    """Assemble the whole power stage for one model.

    Runs the formal method named in :attr:`spec.power.method
    <semsim.core.spec.PowerSpec.method>`, the rules of thumb, and -- when the
    model has structural paths -- a Satorra-Saris table, then reconciles them
    into a single recommendation.

    The recommendation is the largest of

    * what the formal method asks for,
    * Kline's minimum of 200 cases, and
    * the absolute 5:1 floor of ``5 * free parameters``,

    clipped to ``[spec.power.min_n, spec.power.max_n]``.  The 10:1 ratios are
    reported in :attr:`PowerResult.rule_of_thumb` and quoted in the notes but
    are not forced on the recommendation, because a well-powered global test
    on a large, highly over-identified model genuinely needs fewer cases than
    10 per parameter.  When ``spec.power.mode == "fixed"`` the user's ``N`` is
    honoured and every criterion it misses is listed in the notes instead.

    Parameters
    ----------
    spec : ModelSpec
        The model to size.
    cm : CompiledModel, optional
        Its compiled form.  Compiled on demand; needed only for the
        Satorra-Saris table, which is skipped with a note if compilation
        fails.

    Returns
    -------
    PowerResult
        The recommendation, the power curve, the heuristics, the per-path
        table and bilingual notes naming this model's actual numbers.

    Raises
    ------
    semsim.core.errors.SpecificationError
        ``spec.power`` carries an impossible alpha or target power.

    Examples
    --------
    >>> from semsim.core.examples import simple_mediation
    >>> result = analyse(simple_mediation())
    >>> result.n_recommended >= 200
    True

    References
    ----------
    MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
        and determination of sample size for covariance structure modeling.
        *Psychological Methods*, 1(2), 130-149.
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    Kline, R. B. (2016). *Principles and Practice of Structural Equation
        Modeling* (4th ed.). Guilford Press.
    """
    ps = spec.power
    alpha = _check_alpha(ps.alpha)
    target_power = _check_power(ps.target_power)
    rmsea_null = _check_rmsea(ps.rmsea_null, "rmsea_null")
    rmsea_alt = _check_rmsea(ps.rmsea_alt, "rmsea_alt")
    df = int(spec.degrees_of_freedom())
    n_free = int(spec.count_free_parameters())
    n_items = int(spec.n_items)
    min_n = max(int(ps.min_n), 2)
    max_n = max(int(ps.max_n), min_n)

    notes_en: List[str] = []
    notes_ar: List[str] = []
    rot = rules_of_thumb(spec, per_parameter=ps.ratio_per_parameter,
                         per_item=ps.ratio_per_item)

    _note(notes_en, notes_ar,
          f"Your model has {n_free} free parameters and {n_items} items, with "
          f"{n_items * (n_items + 1) // 2} observed moments, leaving "
          f"df = {df}.",
          f"نموذجك يحتوي {n_free} معلمة حرة و{n_items} بندًا، أي "
          f"{n_items * (n_items + 1) // 2} عزمًا مشاهدًا، فتبقى درجات حرية "
          f"قدرها {df}.")
    _note(notes_en, notes_ar,
          f"Your model has {n_free} free parameters, so the "
          f"{ps.ratio_per_parameter}:1 rule asks for "
          f"{rot['n_per_parameter']} cases; the 5:1 absolute floor asks for "
          f"{rot['absolute_floor_5_to_1']}.",
          f"نموذجك يحتوي {n_free} معلمة حرة، ولذلك تطلب قاعدة "
          f"{ps.ratio_per_parameter}:1 عددًا قدره "
          f"{rot['n_per_parameter']} حالة، ويطلب الحد المطلق (5 حالات لكل "
          f"معلمة) عددًا قدره {rot['absolute_floor_5_to_1']} حالة.")
    _note(notes_en, notes_ar,
          f"Your model has {n_items} items, so the {ps.ratio_per_item}:1 "
          f"rule asks for {rot['n_per_item']} cases, and Kline's rule of "
          f"thumb asks for at least {KLINE_MINIMUM_N}.",
          f"نموذجك يحتوي {n_items} بندًا، ولذلك تطلب قاعدة "
          f"{ps.ratio_per_item}:1 عددًا قدره {rot['n_per_item']} حالة، "
          f"ويطلب الحد الأدنى عند كلاين {KLINE_MINIMUM_N} حالة على الأقل.")

    method = ps.method
    n_from_method = 0
    curve = pd.DataFrame(columns=["n", "power", "power_not_close_fit"])

    # -- the formal method -------------------------------------------------- #
    if df < 1:
        _note(notes_en, notes_ar,
              f"With df = {df} the model is saturated or under-identified: "
              f"there is no global fit to have power against, so the "
              f"recommendation falls back on the rules of thumb. Free fewer "
              f"parameters or add items if you want a testable model.",
              f"عند درجات حرية قدرها {df} يكون النموذج مشبَعًا أو غير "
              f"محدَّد: لا توجد مطابقة كلية يمكن اختبار قوتها، ولذلك يعتمد "
              f"الحجم الموصى به على القواعد الاسترشادية. قلِّل عدد المعلمات "
              f"الحرة أو أضف بنودًا إذا أردت نموذجًا قابلًا للاختبار.")
        method = PowerMethod.RULE_OF_THUMB
        n_from_method = max(rot.values())
    elif ps.method is PowerMethod.RULE_OF_THUMB:
        n_from_method = max(rot.values())
        _note(notes_en, notes_ar,
              "The rules of thumb were asked for explicitly, so the "
              "recommendation is the largest of them.",
              "طُلبت القواعد الاسترشادية صراحةً، ولذلك فإن الحجم الموصى به هو "
              "أكبر قيمة بينها.")
    else:
        try:
            n_from_method = n_for_close_fit_power(
                df, target_power, rmsea_null=rmsea_null,
                rmsea_alt=rmsea_alt, alpha=alpha, min_n=2, max_n=MAX_SEARCH_N)
            _note(notes_en, notes_ar,
                  f"MacCallum, Browne & Sugawara's test of close fit "
                  f"(H0: RMSEA = {rmsea_null:g} against RMSEA = "
                  f"{rmsea_alt:g}, alpha = {alpha:g}) reaches power "
                  f"{target_power:.2f} at N = {n_from_method} for df = {df}.",
                  f"اختبار المطابقة القريبة عند ماكالوم وبراون وسوجاوارا "
                  f"(الفرض الصفري RMSEA = {rmsea_null:g} مقابل "
                  f"{rmsea_alt:g}، ومستوى دلالة {alpha:g}) يبلغ قوة "
                  f"{target_power:.2f} عند حجم عينة {n_from_method} بدرجات "
                  f"حرية {df}.")
        except SemSimError as exc:
            method = PowerMethod.RULE_OF_THUMB
            n_from_method = max(rot.values())
            _note(notes_en, notes_ar,
                  f"The test of close fit could not be sized: "
                  f"{exc.message_en} The rules of thumb were used instead.",
                  f"تعذّر تحديد حجم العينة من اختبار المطابقة القريبة: "
                  f"{exc.message_ar} واستُخدمت القواعد الاسترشادية بدلًا منه.")

    # -- Satorra-Saris, which may raise the bar ----------------------------- #
    # Control paths are structural paths and need power like any other.
    spec = spec.expanded()
    per_path: Optional[pd.DataFrame] = None
    n_paths_required = 0
    if cm is None:
        try:
            from semsim.core.compile import compile_model
            cm = compile_model(spec)
        except SemSimError as exc:
            _note(notes_en, notes_ar,
                  f"The model could not be compiled, so no per-path power was "
                  f"computed: {exc.message_en}",
                  f"تعذّر تجميع النموذج، ولذلك لم تُحسب القوة الإحصائية لكل "
                  f"مسار: {exc.message_ar}")
            cm = None

    # -- reconcile ---------------------------------------------------------- #
    floor = max(rot["minimum_200"], rot["absolute_floor_5_to_1"])
    if str(ps.mode).lower() == "fixed":
        n_recommended = _check_n(ps.n, minimum=3)
        binding = "fixed_by_user"
        _note(notes_en, notes_ar,
              f"The sample size was fixed by hand at N = {n_recommended}; the "
              f"criteria above are reported for comparison, not applied.",
              f"حُدِّد حجم العينة يدويًا عند {n_recommended}؛ والمعايير "
              f"السابقة مذكورة للمقارنة فقط دون تطبيقها.")
        for key, value in sorted(rot.items(), key=lambda kv: -kv[1]):
            if value > n_recommended:
                _note(notes_en, notes_ar,
                      f"Warning: N = {n_recommended} falls short of the "
                      f"'{key}' criterion, which asks for {value}.",
                      f"تنبيه: حجم العينة {n_recommended} أقل من معيار "
                      f"'{key}' الذي يطلب {value} حالة.")
        if df >= 1 and n_from_method > n_recommended:
            _note(notes_en, notes_ar,
                  f"Warning: at N = {n_recommended} the test of close fit has "
                  f"power "
                  f"{power_close_fit(df, n_recommended, rmsea_null=rmsea_null, rmsea_alt=rmsea_alt, alpha=alpha):.3f}"
                  f", below the target of {target_power:.2f}; "
                  f"{n_from_method} cases would be needed.",
                  f"تنبيه: عند حجم عينة {n_recommended} تبلغ قوة اختبار "
                  f"المطابقة القريبة "
                  f"{power_close_fit(df, n_recommended, rmsea_null=rmsea_null, rmsea_alt=rmsea_alt, alpha=alpha):.3f}"
                  f"، وهي أقل من الهدف {target_power:.2f}؛ وتلزم "
                  f"{n_from_method} حالة لبلوغه.")
    else:
        n_recommended = max(n_from_method, floor)
        if n_recommended == n_from_method:
            binding = ("formal_method" if method is not PowerMethod.RULE_OF_THUMB
                       else "rule_of_thumb")
        elif n_recommended == rot["minimum_200"]:
            binding = "minimum_200"
        else:
            binding = "absolute_floor_5_to_1"
        if n_recommended > n_from_method:
            _note(notes_en, notes_ar,
                  f"The formal analysis asked for {n_from_method} cases, but "
                  f"the binding constraint is '{binding}' at "
                  f"{n_recommended}, so that is the recommendation.",
                  f"طلب التحليل الرسمي {n_from_method} حالة، لكن القيد "
                  f"الملزم هو '{binding}' عند {n_recommended} حالة، ولذلك "
                  f"فهو الحجم الموصى به.")
        if n_recommended < min_n:
            _note(notes_en, notes_ar,
                  f"The recommendation of {n_recommended} was raised to the "
                  f"floor of {min_n} set in the power settings.",
                  f"رُفع الحجم الموصى به من {n_recommended} إلى الحد الأدنى "
                  f"{min_n} المضبوط في إعدادات القوة.")
            n_recommended = min_n
            binding = "min_n_guard_rail"
        if n_recommended > max_n:
            _note(notes_en, notes_ar,
                  f"The recommendation of {n_recommended} exceeds the ceiling "
                  f"of {max_n} set in the power settings and was capped; the "
                  f"study will be under-powered against the stated target.",
                  f"يتجاوز الحجم الموصى به ({n_recommended}) الحدَّ الأعلى "
                  f"{max_n} المضبوط في إعدادات القوة فجرى تقييده، وستكون "
                  f"الدراسة أقل قوة من الهدف المعلَن.")
            n_recommended = max_n
            binding = "max_n_guard_rail"

    if cm is not None and spec.paths:
        per_path = _per_path_frame(spec, cm, n_recommended, alpha,
                                   target_power, notes_en, notes_ar, max_n,
                                   progress=progress)
        if per_path is not None and not per_path.empty:
            needed = per_path["n_required"].dropna()
            if not needed.empty:
                n_paths_required = int(needed.max())
            if (str(ps.mode).lower() != "fixed"
                    and n_paths_required > n_recommended
                    and n_paths_required <= max_n):
                _note(notes_en, notes_ar,
                      f"Detecting every structural path with power "
                      f"{target_power:.2f} needs {n_paths_required} cases, "
                      f"more than the global test asks for, so the "
                      f"recommendation was raised to it.",
                      f"يحتاج الكشف عن كل مسار بنائي بقوة "
                      f"{target_power:.2f} إلى {n_paths_required} حالة، وهو "
                      f"أكبر مما يطلبه الاختبار الكلي، ولذلك رُفع الحجم "
                      f"الموصى به إليه.")
                n_recommended = n_paths_required
                binding = "satorra_saris"
                if method is PowerMethod.SATORRA_SARIS:
                    n_from_method = n_paths_required
                per_path = _refresh_per_path_power(per_path, n_recommended,
                                                   alpha)
            elif n_paths_required > max_n:
                _note(notes_en, notes_ar,
                      f"Detecting every structural path with power "
                      f"{target_power:.2f} would need {n_paths_required} "
                      f"cases, beyond the ceiling of {max_n}; the smallest "
                      f"paths will stay under-powered.",
                      f"يحتاج الكشف عن كل مسار بنائي بقوة "
                      f"{target_power:.2f} إلى {n_paths_required} حالة، وهو "
                      f"يتجاوز الحدَّ الأعلى {max_n}؛ وستبقى أصغر المسارات "
                      f"أقل قوة من المطلوب.")
            weakest = per_path.loc[per_path["power"].idxmin()]
            _note(notes_en, notes_ar,
                  f"Satorra-Saris: at N = {n_recommended} the weakest path "
                  f"({weakest['source']} -> {weakest['target']}, "
                  f"beta = {weakest['coefficient']:.3f}) has power "
                  f"{weakest['power']:.3f}.",
                  f"ساتورا وساريس: عند حجم عينة {n_recommended} يبلغ أضعف "
                  f"المسارات ({weakest['source']} ← {weakest['target']}، "
                  f"بمعامل {weakest['coefficient']:.3f}) قوة قدرها "
                  f"{weakest['power']:.3f}.")

    # -- curve and the headline power -------------------------------------- #
    if df >= 1:
        power_at = power_close_fit(df, n_recommended, rmsea_null=rmsea_null,
                                   rmsea_alt=rmsea_alt, alpha=alpha)
        curve = power_curve(
            df, n_min=max(min(min_n, n_recommended), 20),
            n_max=max(int(2.5 * n_recommended), n_recommended + 100),
            rmsea_null=rmsea_null, rmsea_alt=rmsea_alt, alpha=alpha,
            include=(n_recommended,))
        _note(notes_en, notes_ar,
              f"At the recommended N = {n_recommended} the test of close fit "
              f"has power {power_at:.3f}, and the companion test of not-close "
              f"fit has power "
              f"{power_not_close_fit(df, n_recommended, rmsea_null=rmsea_null, alpha=alpha):.3f}.",
              f"عند الحجم الموصى به {n_recommended} تبلغ قوة اختبار المطابقة "
              f"القريبة {power_at:.3f}، وتبلغ قوة اختبار عدم المطابقة "
              f"القريبة المرافق "
              f"{power_not_close_fit(df, n_recommended, rmsea_null=rmsea_null, alpha=alpha):.3f}.")
    else:
        power_at = float("nan")

    return PowerResult(
        method=method,
        n_recommended=int(n_recommended),
        power_at_recommended=float(power_at),
        alpha=float(alpha),
        df=df,
        n_free=n_free,
        curve=curve,
        rule_of_thumb=rot,
        per_path_power=per_path,
        notes_en=notes_en,
        notes_ar=notes_ar,
        n_items=n_items,
        rmsea_null=rmsea_null,
        rmsea_alt=rmsea_alt,
        target_power=target_power,
        n_from_method=int(n_from_method),
        binding_constraint=binding,
    )


__all__ = [
    "DEFAULT_RMSEA_NULL", "DEFAULT_RMSEA_ALT", "DEFAULT_RMSEA_ALT_NOT_CLOSE",
    "KLINE_MINIMUM_N", "ABSOLUTE_FLOOR_RATIO", "MAX_SEARCH_N",
    "MONTE_CARLO_COLUMNS", "PER_PATH_COLUMNS",
    "PowerResult", "PathPowerDetail",
    "noncentrality", "power_close_fit", "power_not_close_fit",
    "n_for_close_fit_power", "n_for_not_close_fit_power", "power_curve",
    "path_power_detail", "satorra_saris_power", "n_for_path_power",
    "monte_carlo_power", "rules_of_thumb", "analyse",
]

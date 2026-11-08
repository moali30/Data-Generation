"""
semsim.generation.thresholds
============================
The *threshold model*: how one continuous latent response ``y*`` becomes the
integer a respondent actually ticks on a Likert scale.

The model is the classical one of Pearson (1900) as used throughout ordinal
factor analysis (Olsson, 1979; Muthen, 1984).  A standardised continuous
response ``y* ~ N(0, 1)`` is observed only through ``K`` ordered categories
delimited by ``K - 1`` cut points ``tau_1 < ... < tau_{K-1}``::

    x = k      iff      tau_{k-1} < y* <= tau_k        (tau_0 = -inf, tau_K = +inf)

so that ``P(x = k) = Phi(tau_k) - Phi(tau_{k-1})``.

Everything in this module is built the honest way round: a *response style* is
first expressed as a vector of target category **proportions**, and the cut
points are then recovered as the inverse normal CDF of the cumulative
proportions.  That is what makes "mass on the low categories" mean a genuine
floor effect in the generated data rather than a rounded normal.

Two consequences of categorisation are quantified here as well:

* :func:`achieved_proportions` -- what the sample actually looks like;
* :func:`polychoric_attenuation` -- how much a Pearson correlation shrinks
  because the continuum was coarsened.  The calibration loop uses it to
  pre-compensate loadings so the *fitted* model lands on the requested values.

References
----------
Pearson, K. (1900). Mathematical contributions to the theory of evolution,
    VII. *Philosophical Transactions of the Royal Society A*, 195, 1-47.
Olsson, U. (1979). Maximum likelihood estimation of the polychoric correlation
    coefficient. *Psychometrika*, 44(4), 443-460.
Muthen, B. (1984). A general structural equation model with dichotomous,
    ordered categorical, and continuous latent variable indicators.
    *Psychometrika*, 49(1), 115-132.
Bollen, K. A., & Barb, K. H. (1981). Pearson's r and coarsely categorized
    measures. *American Sociological Review*, 46(2), 232-239.
Olsson, U., Drasgow, F., & Dorans, N. J. (1982). The polyserial correlation
    coefficient. *Psychometrika*, 47(3), 337-347.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np
from scipy import stats

from semsim.core.errors import SpecificationError
from semsim.core.spec import ResponseStyle, ScaleKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semsim.core.spec import ScaleSpec

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Smallest category proportion tolerated.  Anything at or below this would map
#: to an infinite cut point, which breaks every downstream matrix.
PROPORTION_FLOOR: float = 1e-6

#: Cumulative proportions are clipped into this open interval before the
#: inverse normal CDF is applied, so a cut point can never be +/- inf.
_CUM_EPS: float = 1e-9

#: Hard bounds on the number of Likert categories (see the
#: ``LIKERT_CATEGORY_RANGE`` diagnostic in :mod:`semsim.core.validation`).
MIN_CATEGORIES: int = 2
MAX_CATEGORIES: int = 11

#: ``(alpha, beta)`` of the Beta density whose equal-width discretisation gives
#: each response style its shape.  ``POSITIVE_SKEW`` piles mass on the *low*
#: categories (a floor effect), ``NEGATIVE_SKEW`` on the high ones (a ceiling).
STYLE_BETA_PARAMS: Dict[ResponseStyle, Tuple[float, float]] = {
    ResponseStyle.SYMMETRIC: (3.5, 3.5),
    ResponseStyle.POSITIVE_SKEW: (2.0, 5.0),
    ResponseStyle.NEGATIVE_SKEW: (5.0, 2.0),
    ResponseStyle.EXTREME_POSITIVE: (1.4, 6.0),
    ResponseStyle.EXTREME_NEGATIVE: (6.0, 1.4),
}

#: ``BIMODAL`` is an equal-weight mixture of these two mirrored Beta densities:
#: polarised opinion, with a trough in the middle of the scale.
BIMODAL_COMPONENTS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (1.5, 8.0),
    (8.0, 1.5),
)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _coerce_style(style: "ResponseStyle | str") -> ResponseStyle:
    """Accept either the enum or its bare string value."""
    if isinstance(style, ResponseStyle):
        return style
    try:
        return ResponseStyle(str(style))
    except ValueError as exc:  # pragma: no cover - defensive
        raise SpecificationError(
            f"Unknown response style {style!r}.",
            f"نمط استجابة غير معروف: {style!r}.",
            style=str(style),
        ) from exc


def _check_n_categories(n_categories: int) -> int:
    """Validate a Likert category count and return it as an ``int``.

    Why this is strict about integrality.  The previous implementation was a
    bare ``int(n_categories)``, and ``int`` *truncates* rather than rejects:
    ``int(5.9)`` is 5 and ``int(2.999)`` is 2.  So a caller who asked for 5.9
    categories silently received a five-point scale, and one who asked for
    2.999 received a dichotomy -- while the function's own error message
    promised that the count "must be a whole number".  A non-integral count is
    always an arithmetic slip upstream (a proportion mistaken for a count, an
    unrounded mean, a slider value), and quietly rounding it down hands back a
    dataset whose response scale is *not* the one that was requested, with no
    trace in the output that the request was altered.  The function therefore
    now keeps the promise its message already made.

    Integral values of any numeric type are still accepted -- ``5``, ``5.0``
    and ``numpy.int64(5)`` all mean five categories, and refusing ``5.0``
    would break every caller that arrives through a float-typed form field or
    a numpy array element.  Strings are rejected: ``int("5")`` used to succeed,
    which let an unparsed text field through as a category count and so turned
    a parsing bug into a silently mis-scaled dataset.
    """
    # Gate on the type first.  ``float(...)`` alone would accept ``"5"``.
    if isinstance(n_categories, (str, bytes, bytearray)) or not isinstance(
        n_categories, (int, float, np.integer, np.floating)
    ):
        raise SpecificationError(
            f"Number of categories must be a whole number, got {n_categories!r}.",
            f"عدد الفئات يجب أن يكون عدداً صحيحاً، والقيمة المُدخلة هي {n_categories!r}.",
            n_categories=n_categories,
        )
    value = float(n_categories)
    if not np.isfinite(value) or value != np.floor(value):
        raise SpecificationError(
            f"Number of categories must be a whole number, got {n_categories!r}; "
            "a fractional count would be silently truncated and the generated "
            "scale would not be the one requested.",
            f"عدد الفئات يجب أن يكون عدداً صحيحاً، والقيمة المُدخلة هي "
            f"{n_categories!r}؛ إذ تُقتطع الكسور دون إشعار، فلا يكون المقياس "
            "المُولَّد مطابقاً للمقياس المطلوب.",
            n_categories=n_categories,
        )
    n = int(value)
    if n < MIN_CATEGORIES or n > MAX_CATEGORIES:
        raise SpecificationError(
            f"A Likert scale needs between {MIN_CATEGORIES} and {MAX_CATEGORIES} "
            f"categories; got {n}.",
            f"مقياس ليكرت يجب أن يتضمّن عدداً من الفئات بين {MIN_CATEGORIES} "
            f"و{MAX_CATEGORIES}، والعدد المُدخل هو {n}.",
            n_categories=n,
        )
    return n


def _beta_bin_probabilities(alpha: float, beta: float, n_categories: int) -> np.ndarray:
    """Probability of each of ``n_categories`` equal-width bins of (0, 1)
    under a ``Beta(alpha, beta)`` density."""
    edges = np.linspace(0.0, 1.0, n_categories + 1)
    cdf = stats.beta.cdf(edges, alpha, beta)
    cdf[0], cdf[-1] = 0.0, 1.0
    return np.diff(cdf)


def _normalise_proportions(p: np.ndarray) -> np.ndarray:
    """Floor every proportion at :data:`PROPORTION_FLOOR` and renormalise.

    Flooring is what keeps a cut point finite: a category with probability
    exactly zero would need ``tau = Phi^-1(0) = -inf``.
    """
    p = np.asarray(p, dtype=float).copy()
    p[~np.isfinite(p)] = PROPORTION_FLOOR
    p = np.maximum(p, PROPORTION_FLOOR)
    total = float(p.sum())
    if total <= 0.0:  # pragma: no cover - unreachable after flooring
        raise SpecificationError(
            "Category proportions sum to zero.",
            "مجموع نسب الفئات يساوي صفراً.",
        )
    return p / total


# --------------------------------------------------------------------------- #
# Proportions and thresholds
# --------------------------------------------------------------------------- #


def proportions_for_style(
    n_categories: int, style: "ResponseStyle | str"
) -> np.ndarray:
    """Target marginal category proportions for a response style.

    Parameters
    ----------
    n_categories : int
        Number of ordered response options, ``2 <= n_categories <= 11``.
    style : ResponseStyle or str
        Shape of the response distribution.  :attr:`ResponseStyle.CUSTOM` is
        rejected because it carries no shape of its own -- the caller must
        supply explicit proportions or thresholds on the
        :class:`~semsim.core.spec.ScaleSpec`.

    Returns
    -------
    numpy.ndarray
        Proportions of length ``n_categories``, strictly positive, summing to
        one.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``n_categories`` is outside ``2..11`` or ``style`` is ``CUSTOM``.

    Notes
    -----
    ``UNIFORM`` is flat by construction.  Every other named style is the
    equal-width discretisation of a Beta density (see
    :data:`STYLE_BETA_PARAMS`), which is the standard device for producing
    realistically lopsided Likert marginals: ``Beta(2, 5)`` leans low (a floor
    effect), ``Beta(5, 2)`` leans high (a ceiling effect), and the symmetric
    ``Beta(3.5, 3.5)`` gives the familiar centred bell.  ``BIMODAL`` mixes the
    two mirrored extremes in equal weight.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous
        Univariate Distributions*, Vol. 2, ch. 25 (the Beta distribution).
    """
    n = _check_n_categories(n_categories)
    st = _coerce_style(style)

    if st is ResponseStyle.CUSTOM:
        raise SpecificationError(
            "ResponseStyle.CUSTOM carries no shape of its own: supply explicit "
            "category proportions or thresholds on the scale.",
            "نمط الاستجابة المخصّص لا يحمل شكلاً محدّداً بذاته، لذا يجب تحديد نسب "
            "الفئات أو نقاط القطع صراحةً في مواصفات المقياس.",
            n_categories=n,
        )

    if st is ResponseStyle.UNIFORM:
        return np.full(n, 1.0 / n, dtype=float)

    if st is ResponseStyle.BIMODAL:
        (a1, b1), (a2, b2) = BIMODAL_COMPONENTS
        p = 0.5 * (
            _beta_bin_probabilities(a1, b1, n)
            + _beta_bin_probabilities(a2, b2, n)
        )
        return _normalise_proportions(p)

    alpha, beta = STYLE_BETA_PARAMS[st]
    return _normalise_proportions(_beta_bin_probabilities(alpha, beta, n))


def thresholds_from_proportions(proportions: Sequence[float]) -> np.ndarray:
    """Convert target category proportions into z-scale cut points.

    ``tau_k = Phi^-1(p_1 + ... + p_k)`` for ``k = 1 .. K-1``; the final
    cumulative proportion is 1.0 and is dropped, since ``Phi^-1(1) = +inf``.

    Parameters
    ----------
    proportions : sequence of float
        Non-negative target proportions of length ``K >= 2``, summing to one
        (a deviation up to 1e-3 is renormalised silently).

    Returns
    -------
    numpy.ndarray
        Strictly increasing cut points of length ``K - 1`` on the standard
        normal scale.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If fewer than two proportions are given, any value is negative or
        non-finite, or the proportions do not sum to one.

    Notes
    -----
    Zero proportions are floored at :data:`PROPORTION_FLOOR` before the inverse
    CDF is taken, which is what prevents an infinite threshold from reaching
    the generator.  This is the sample-free version of the first step of
    Olsson's (1979) two-step polychoric estimator, where thresholds are read
    off the observed marginals.

    References
    ----------
    Olsson, U. (1979). Maximum likelihood estimation of the polychoric
        correlation coefficient. *Psychometrika*, 44(4), 443-460.
    Joreskog, K. G., & Sorbom, D. (1996). *PRELIS 2 User's Reference Guide*,
        section on threshold estimation.
    """
    p = np.asarray(list(proportions), dtype=float)
    if p.ndim != 1 or p.size < MIN_CATEGORIES:
        raise SpecificationError(
            f"Need at least {MIN_CATEGORIES} category proportions, got {p.size}.",
            f"يجب تحديد {MIN_CATEGORIES} نسبة على الأقل للفئات، وعدد النسب "
            f"المُدخلة هو {p.size}.",
            n_proportions=int(p.size),
        )
    if not np.all(np.isfinite(p)):
        raise SpecificationError(
            "Category proportions must all be finite numbers.",
            "نسب الفئات يجب أن تكون جميعها أعداداً منتهية.",
        )
    if np.any(p < 0.0):
        raise SpecificationError(
            "Category proportions cannot be negative.",
            "نسب الفئات لا يمكن أن تكون سالبة.",
        )
    total = float(p.sum())
    if total <= 0.0 or abs(total - 1.0) > 1e-3:
        raise SpecificationError(
            f"Category proportions must sum to 1.0, they sum to {total:.6g}.",
            f"مجموع نسب الفئات يجب أن يساوي 1، والمجموع الحالي هو {total:.6g}.",
            total=total,
        )

    cumulative = np.cumsum(_normalise_proportions(p))[:-1]
    cumulative = np.clip(cumulative, _CUM_EPS, 1.0 - _CUM_EPS)
    tau = np.asarray(stats.norm.ppf(cumulative), dtype=float)

    if tau.size and not np.all(np.diff(tau) > 0.0):  # pragma: no cover
        raise SpecificationError(
            "The supplied proportions do not produce strictly increasing "
            "thresholds; at least two categories are indistinguishable.",
            "النسب المُدخلة لا تنتج نقاط قطع متزايدة تماماً، ما يعني أن فئتين "
            "على الأقل غير قابلتين للتمييز.",
        )
    return tau


def thresholds_for_style(
    n_categories: int, style: "ResponseStyle | str"
) -> np.ndarray:
    """z-scale cut points realising a named response style.

    Parameters
    ----------
    n_categories : int
        Number of ordered response options, ``2 <= n_categories <= 11``.
    style : ResponseStyle or str
        Shape of the response distribution.

    Returns
    -------
    numpy.ndarray
        Strictly increasing cut points of length ``n_categories - 1``.  Feeding
        them to :func:`categorise` on standard normal input reproduces the
        proportions returned by :func:`proportions_for_style`.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``n_categories`` is outside ``2..11`` or ``style`` is ``CUSTOM``.

    Examples
    --------
    >>> import numpy as np
    >>> from semsim.core.spec import ResponseStyle
    >>> tau = thresholds_for_style(5, ResponseStyle.UNIFORM)
    >>> np.round(tau, 4)
    array([-0.8416, -0.2533,  0.2533,  0.8416])

    Notes
    -----
    The style is realised through its target proportions rather than by
    spacing the cut points directly, so "mass on the low categories" really
    produces a floor effect in the generated data.

    Every named style except ``UNIFORM`` is the equal-width discretisation of
    a Beta density, with the ``(alpha, beta)`` pairs listed in
    :data:`STYLE_BETA_PARAMS` (and :data:`BIMODAL_COMPONENTS` for
    ``BIMODAL``).  ``SYMMETRIC`` in particular is ``Beta(3.5, 3.5)``: a
    centred, bell-shaped response distribution with most respondents near the
    middle of the scale and few at either end -- *not* equal-probability
    normal cuts.  Equal-probability cuts are flat marginals, and that is
    precisely what ``UNIFORM`` is (``tau_k = Phi^-1(k / K)``), so the two
    styles would otherwise be the same thing under two names.  At ``K = 5``
    the two differ plainly: ``SYMMETRIC`` gives proportions of about
    ``0.044, 0.259, 0.394, 0.259, 0.044`` while ``UNIFORM`` gives ``0.20``
    five times over.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    Rhemtulla, M., Brosseau-Liard, P. E., & Savalei, V. (2012). When can
        categorical variables be treated as continuous? *Psychological
        Methods*, 17(3), 354-373.
    """
    return thresholds_from_proportions(proportions_for_style(n_categories, style))


def category_probabilities(thresholds: Sequence[float]) -> np.ndarray:
    """Normal-CDF category probabilities implied by a set of cut points.

    Parameters
    ----------
    thresholds : sequence of float
        Strictly increasing z-scale cut points, length ``K - 1``.

    Returns
    -------
    numpy.ndarray
        ``P(x = k) = Phi(tau_k) - Phi(tau_{k-1})``, length ``K``, summing to
        one.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the cut points are not strictly increasing or not finite.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    """
    tau = _validated_thresholds(thresholds)
    edges = np.concatenate(([-np.inf], tau, [np.inf]))
    return np.diff(stats.norm.cdf(edges))


def _validated_thresholds(thresholds: Sequence[float]) -> np.ndarray:
    tau = np.asarray(list(thresholds), dtype=float).ravel()
    if tau.size == 0:
        raise SpecificationError(
            "At least one threshold is required (a scale needs two categories).",
            "يجب تحديد نقطة قطع واحدة على الأقل، لأن أي مقياس يحتاج إلى فئتين.",
        )
    if not np.all(np.isfinite(tau)):
        raise SpecificationError(
            "Thresholds must be finite; an infinite cut point means a category "
            "with zero probability.",
            "نقاط القطع يجب أن تكون أعداداً منتهية، فنقطة القطع اللانهائية تعني "
            "وجود فئة احتمالها صفر.",
        )
    if tau.size > 1 and not np.all(np.diff(tau) > 0.0):
        raise SpecificationError(
            "Thresholds must be strictly increasing.",
            "نقاط القطع يجب أن تكون متزايدة تماماً ومرتّبة تصاعدياً.",
        )
    return tau


# --------------------------------------------------------------------------- #
# Applying the model to data
# --------------------------------------------------------------------------- #


def categorise(
    y_star: np.ndarray, thresholds: np.ndarray, *, base: int = 1
) -> np.ndarray:
    """Discretise a continuous latent response into ordered categories.

    Parameters
    ----------
    y_star : numpy.ndarray
        Continuous latent responses, any shape.  Expected to be on the
        standard normal scale, which is the metric the cut points live in.
    thresholds : numpy.ndarray
        Strictly increasing z-scale cut points, length ``K - 1``.
    base : int, optional
        Code of the lowest category.  ``1`` (the default) yields values in
        ``1..K`` as a questionnaire would be coded; ``0`` yields ``0..K-1``.

    Returns
    -------
    numpy.ndarray
        Integer codes of the same shape as ``y_star``, in
        ``base .. base + K - 1``.  If ``y_star`` contains ``NaN`` the result is
        a float array with those cells left as ``NaN``, so missing data survives
        the round trip.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the cut points are not finite and strictly increasing.

    Notes
    -----
    Implements ``x = k`` iff ``tau_{k-1} < y* <= tau_k``, vectorised with
    :func:`numpy.searchsorted`, so a 500x60 matrix is categorised in
    microseconds.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    """
    tau = _validated_thresholds(thresholds)
    y = np.asarray(y_star, dtype=float)
    base_int = int(base)

    flat = y.ravel()
    missing = np.isnan(flat)
    safe = np.where(missing, 0.0, flat)
    codes = np.searchsorted(tau, safe, side="left") + base_int

    if missing.any():
        out = codes.astype(float)
        out[missing] = np.nan
        return out.reshape(y.shape)
    return codes.astype(np.int64).reshape(y.shape)


def scale_thresholds(scale: "ScaleSpec") -> np.ndarray:
    """Cut points for one item, honouring the precedence rules of ``ScaleSpec``.

    Explicit ``scale.thresholds`` win; then ``scale.category_proportions``;
    otherwise the ``scale.response_style`` shape is used.

    Parameters
    ----------
    scale : ScaleSpec
        The item's scale description.  Duck-typed: any object exposing
        ``n_categories``, ``thresholds``, ``category_proportions`` and
        ``response_style`` will do.

    Returns
    -------
    numpy.ndarray
        Strictly increasing cut points of length ``n_categories - 1``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If explicit thresholds or proportions have the wrong length, or the
        category count is outside ``2..11``.

    References
    ----------
    Olsson, U. (1979). Maximum likelihood estimation of the polychoric
        correlation coefficient. *Psychometrika*, 44(4), 443-460.
    """
    n = _check_n_categories(getattr(scale, "n_categories", 5))

    explicit = getattr(scale, "thresholds", None)
    if explicit is not None and len(explicit) > 0:
        tau = _validated_thresholds(explicit)
        if tau.size != n - 1:
            raise SpecificationError(
                f"A {n}-category scale needs exactly {n - 1} thresholds, "
                f"got {tau.size}.",
                f"المقياس المكوّن من {n} فئات يحتاج إلى {n - 1} نقطة قطع بالضبط، "
                f"وعدد النقاط المُدخلة هو {tau.size}.",
                n_categories=n, n_thresholds=int(tau.size),
            )
        return tau

    proportions = getattr(scale, "category_proportions", None)
    if proportions is not None and len(proportions) > 0:
        if len(proportions) != n:
            raise SpecificationError(
                f"A {n}-category scale needs exactly {n} category proportions, "
                f"got {len(proportions)}.",
                f"المقياس المكوّن من {n} فئات يحتاج إلى {n} نسبة بالضبط، وعدد "
                f"النسب المُدخلة هو {len(proportions)}.",
                n_categories=n, n_proportions=len(proportions),
            )
        return thresholds_from_proportions(proportions)

    style = getattr(scale, "response_style", ResponseStyle.SYMMETRIC)
    return thresholds_for_style(n, style)


def apply_scale(y_star: np.ndarray, scale: "ScaleSpec") -> np.ndarray:
    """Turn continuous latent responses into the item's reported metric.

    Parameters
    ----------
    y_star : numpy.ndarray
        Standardised continuous latent responses, any shape.
    scale : ScaleSpec
        The item's scale.  Ordinal scales are categorised into
        ``1..n_categories``; continuous scales are placed on the reported
        metric as ``mean + sd * y_star``.

    Returns
    -------
    numpy.ndarray
        Integer Likert codes, or floats on the reported continuous metric.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the scale's thresholds, proportions or category count are invalid.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    """
    y = np.asarray(y_star, dtype=float)
    kind = getattr(scale, "kind", ScaleKind.LIKERT)
    if kind == ScaleKind.CONTINUOUS or str(kind) == str(ScaleKind.CONTINUOUS.value):
        mean = float(getattr(scale, "mean", 0.0))
        sd = float(getattr(scale, "sd", 1.0))
        return mean + sd * y
    return categorise(y, scale_thresholds(scale), base=1)


def reverse_code(values: np.ndarray, scale: "ScaleSpec") -> np.ndarray:
    """Flip a reverse-worded item onto the direction of the other items.

    Parameters
    ----------
    values : numpy.ndarray
        Scored responses, as produced by :func:`apply_scale`.
    scale : ScaleSpec
        The item's scale.  Likert codes in ``1..K`` are mapped to
        ``K + 1 - value``; continuous scores are reflected about
        ``scale.mean``.

    Returns
    -------
    numpy.ndarray
        Reversed scores, same shape and dtype family as ``values``.  ``NaN``
        cells stay ``NaN``.

    Notes
    -----
    The mapping is an involution -- applying it twice is the identity -- which
    is what makes it safe for the exporter to flip a column without tracking
    how many times it has been flipped.  Likert reversal assumes the base-1
    coding that :func:`apply_scale` produces.

    References
    ----------
    DeVellis, R. F. (2017). *Scale Development: Theory and Applications*
        (4th ed.), ch. 5 on reverse-worded items.
    """
    v = np.asarray(values)
    kind = getattr(scale, "kind", ScaleKind.LIKERT)
    if kind == ScaleKind.CONTINUOUS or str(kind) == str(ScaleKind.CONTINUOUS.value):
        mean = float(getattr(scale, "mean", 0.0))
        return 2.0 * mean - v.astype(float)
    n = _check_n_categories(getattr(scale, "n_categories", 5))
    return (n + 1) - v


def achieved_proportions(
    values: np.ndarray, n_categories: int, *, base: "int | None" = None
) -> np.ndarray:
    """Observed marginal proportions of a categorised item.

    Parameters
    ----------
    values : numpy.ndarray
        Integer category codes, either base-1 (``1..K``) or base-0
        (``0..K-1``).  ``NaN`` cells are ignored.
    n_categories : int
        Number of categories ``K`` the proportions are reported over.
    base : {1, 0} or None, keyword-only
        Code of the lowest category.  Pass it whenever you know it -- it is
        the only way to be certain, see the Notes.  ``None`` (the default)
        keeps the historical auto-detection: base-0 when a code below 1 is
        observed, base-1 otherwise.

    Returns
    -------
    numpy.ndarray
        Proportions of length ``n_categories``, summing to one (all zeros if
        every value is missing).

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``n_categories`` is outside ``2..11``, ``base`` is neither 0 nor 1,
        or the values fall outside the range implied by the base in force.

    Notes
    -----
    **The coding of the input is genuinely ambiguous, and auto-detection
    cannot resolve it.**  Base-0 codes of a ``K``-category item whose lowest
    category happens to be unobserved span ``1..K-1``, which is *also* a legal
    base-1 sample.  There is no information in the data that separates the two
    readings, so the old unconditional auto-detection -- which tested
    ``1 <= min`` before ``0 <= min`` and therefore always answered "base-1" --
    silently shifted every proportion by one category for such a sample: a
    base-0 vector of ``1, 2, 3, 4`` on a five-point item was reported as mass
    on categories 1-4 instead of 2-5.

    Two things follow.  First, the caller who knows the coding should say so
    through ``base``; that removes the guess entirely.  Second, the default
    tie-break is base-1 *by design*, not by accident, because base-1 is what
    this engine's own data carries: :func:`apply_scale` scores every
    :class:`~semsim.core.spec.ScaleSpec` into ``1..n_categories``, so for any
    column produced from a ``ScaleSpec`` the ambiguous case resolves the right
    way.  The guess is wrong only for base-0 data from elsewhere, and that is
    exactly the case ``base=0`` is for.

    References
    ----------
    Bollen, K. A., & Barb, K. H. (1981). Pearson's r and coarsely categorized
        measures. *American Sociological Review*, 46(2), 232-239.
    """
    n = _check_n_categories(n_categories)
    if base is not None and int(base) not in (0, 1):
        raise SpecificationError(
            f"The category base must be 0 or 1, got {base!r}.",
            f"يجب أن يكون أساس ترميز الفئات 0 أو 1، والقيمة المُدخلة هي {base!r}.",
            base=base,
        )
    v = np.asarray(values, dtype=float).ravel()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.zeros(n, dtype=float)

    lo, hi = float(v.min()), float(v.max())
    if base is None:
        # Auto-detection.  A code below 1 settles the question outright; when
        # the sample fits both codings the tie goes to base-1, which is the
        # coding every ``ScaleSpec`` implies (see the Notes).
        detected = 1 if lo >= 1.0 else 0
    else:
        detected = int(base)

    if lo < detected or hi > detected + n - 1:
        raise SpecificationError(
            f"Observed codes span [{lo:g}, {hi:g}], which does not fit the "
            f"{detected}..{detected + n - 1} coding of a {n}-category item.",
            f"القيم المرصودة تمتد من {lo:g} إلى {hi:g}، وهي لا تتوافق مع ترميز "
            f"الفئات من {detected} إلى {detected + n - 1} لعنصر من {n} فئات.",
            n_categories=n, base=detected, observed_min=lo, observed_max=hi,
        )

    counts = np.bincount(
        np.rint(v).astype(np.int64) - detected, minlength=n
    )[:n].astype(float)
    return counts / counts.sum()


# --------------------------------------------------------------------------- #
# Attenuation
# --------------------------------------------------------------------------- #


def polychoric_attenuation(thresholds: np.ndarray) -> float:
    """Correlation attenuation caused by categorising at these cut points.

    This is the point-polyserial correlation between the categorised variable
    ``x`` and the standard normal latent ``y*`` it came from.  Coarsening a
    continuum destroys information, so the Pearson correlation between two
    categorised variables is *smaller* than the correlation between their
    latents; to first order

    .. math::
        r(x_i, x_j) \\approx \\rho(y^*_i, y^*_j) \\; \\kappa_i \\, \\kappa_j

    where ``kappa`` is the factor this function returns for each variable.
    The calibration loop divides a requested loading by ``kappa`` to
    pre-compensate.  **That product rule is a first-order approximation, and
    how good it is depends strongly on the response style** -- see "Accuracy
    of the product rule" below before relying on it.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Strictly increasing z-scale cut points, length ``K - 1``.

    Returns
    -------
    float
        The attenuation factor ``kappa`` in ``(0, 1]``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the cut points are not finite and strictly increasing, or they imply
        a degenerate (zero-variance) categorised variable.

    Notes
    -----
    With unit-spaced category codes ``c_k = k``,

    .. math::
        \\operatorname{Cov}(x, y^*) =
            \\sum_{k=1}^{K-1} \\phi(\\tau_k)\\,(c_{k+1} - c_k)
          = \\sum_{k=1}^{K-1} \\phi(\\tau_k),

    which follows from ``\\int_{\\tau_{k-1}}^{\\tau_k} y\\,\\phi(y)\\,dy =
    \\phi(\\tau_{k-1}) - \\phi(\\tau_k)`` and summation by parts.  Since
    ``Var(y*) = 1``,

    .. math::
        \\kappa = \\frac{\\sum_k \\phi(\\tau_k)}{\\operatorname{sd}(x)} .

    The result is invariant to the category codes' origin and scale, so
    ``base`` is irrelevant.  For the familiar five-point scale with
    equal-probability cuts ``kappa = 0.9423``; for a median split of a normal
    it is ``sqrt(2/pi) = 0.7979``, the classic result behind the warning that
    dichotomisation costs about 20% of a correlation.

    **Accuracy of the product rule.**
    ``kappa`` itself is exact: it is the point-polyserial correlation of the
    categorised variable with its own latent, computed in closed form above.
    What is only *approximate* is the two-variable product rule
    ``r(x_i, x_j) ~= rho kappa_i kappa_j``.  It is a first-order
    (small-``rho``) expansion: it treats the categorised variable as a linear
    function of its latent, which it is not, and the discrepancy grows with
    ``|rho|``, grows as ``K`` falls, and grows sharply as the thresholds move
    away from the centre of the distribution.

    The table below is the **exact** relative error of the product rule,
    obtained from bivariate-normal orthant probabilities (the cell
    probabilities ``P(x_i = a, x_j = b)`` assembled from
    ``scipy.stats.multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf`` at the
    cut points, and summed into the exact Pearson ``r``) -- not from
    simulation, so there is no Monte-Carlo noise in it.  Both items carry the
    same style and the same ``K``:

    ======================  ===================  ===================
    style                   K = 5 or 7,          K = 3,
                            ``|rho| <= 0.7``     ``|rho| <= 0.8``
    ======================  ===================  ===================
    ``SYMMETRIC``           <= 0.3%              <= 2.7%
    ``UNIFORM``             <= 3.8%              <= 8.0%
    ``POSITIVE_SKEW``       <= 4.3%              <= 15.6%
    ``NEGATIVE_SKEW``       <= 4.3%              <= 15.6%
    ``BIMODAL``             <= 8.2%              <= 13.4%
    ``EXTREME_POSITIVE``    <= 14.2%             <= 37.1%
    ``EXTREME_NEGATIVE``    <= 14.2%             <= 37.1%
    ======================  ===================  ===================

    So the often-quoted "a fraction of a percent" envelope holds for
    ``SYMMETRIC`` **only**.  ``UNIFORM`` is already an order of magnitude
    worse than that, and the shipped ``EXTREME_POSITIVE`` /
    ``EXTREME_NEGATIVE`` / ``BIMODAL`` styles are two orders of magnitude
    worse: at ``K = 5`` and ``rho = 0.7`` the extreme styles are out by
    0.081 in ``r`` (exact 0.5704 against a predicted 0.4896, 14.2%), and a
    dichotomy (``K = 2``) of an extreme style is out by 0.214 in ``r``
    (exact 0.3258 against 0.1116, 65.7%).

    **The sign of the correlation matters, and the error is not one-signed.**
    For a style with *symmetric* marginals -- ``SYMMETRIC``, ``UNIFORM``,
    ``BIMODAL`` -- the error really is one-signed and the envelope is the same
    either way, because the categorisation is invariant under reflection.  For
    an *asymmetric* style it is not: the two marginals are skewed the same
    way, so a negative correlation puts most of the joint mass in the sparse
    corners of the table, and there the product rule *over*-predicts ``|r|``.
    Measured exactly over a signed grid in steps of 0.05 (worst relative error
    over ``K`` in {3, 5, 7}, ``|rho| <= 0.7`` at ``K >= 5`` and
    ``|rho| <= 0.8`` at ``K = 3``):

    =================  =============  =============  ==================
    style              worst rho > 0  worst rho < 0  over-predicts |r|?
    =================  =============  =============  ==================
    SYMMETRIC                  2.68%          2.68%  no
    UNIFORM                    7.98%          7.98%  no
    BIMODAL                   13.33%         13.33%  no
    POSITIVE_SKEW             15.51%         13.04%  yes, when rho < 0
    NEGATIVE_SKEW             15.51%         13.04%  yes, when rho < 0
    EXTREME_POSITIVE          37.06%        100.47%  yes, when rho < 0
    EXTREME_NEGATIVE          37.06%        100.47%  yes, when rho < 0
    =================  =============  =============  ==================

    The worst cell in that sweep is ``EXTREME_NEGATIVE`` at ``K = 3`` and
    ``rho = -0.7``: the exact correlation is ``-0.1704`` against a predicted
    ``-0.3033``, nearly double in magnitude and in the wrong direction for a
    correction.  Treat the product rule as usable for the symmetric-marginal
    styles and as an order-of-magnitude guide elsewhere.

    **Consequence for the calibration loop.**  Pre-dividing a loading by
    ``kappa`` assumes the product rule, so for ``EXTREME_POSITIVE``,
    ``EXTREME_NEGATIVE`` and ``BIMODAL`` -- and for any style at ``K <= 3`` or
    with large target correlations -- the one-shot correction overshoots by
    roughly the percentages above, in the direction of loadings that are too
    high.  The loop must not lean on it as a closed-form answer for those
    styles: it should treat ``kappa`` as a starting point and iterate on the
    *realised* fit, or compute the exact ordinal correlation from the orthant
    probabilities.  For ``SYMMETRIC`` at ``K >= 5`` the single analytic
    correction is accurate enough to use as-is.

    The figures above are reproduced exactly, without simulation, by the
    ``test_product_rule_envelope_*`` tests in ``tests/test_thresholds.py``,
    so they stay honest if the style parameters are ever retuned.

    References
    ----------
    Olsson, U., Drasgow, F., & Dorans, N. J. (1982). The polyserial
        correlation coefficient. *Psychometrika*, 47(3), 337-347.
    Cox, N. R. (1974). Estimation of the correlation between a continuous and a
        discrete variable. *Biometrics*, 30(1), 171-178.
    Bollen, K. A., & Barb, K. H. (1981). Pearson's r and coarsely categorized
        measures. *American Sociological Review*, 46(2), 232-239.
    Pearson, K. (1900). Mathematical contributions to the theory of evolution,
        VII. *Philosophical Transactions of the Royal Society A*, 195, 1-47.
        (The polychoric series whose leading term is the product rule.)
    Rhemtulla, M., Brosseau-Liard, P. E., & Savalei, V. (2012). When can
        categorical variables be treated as continuous? *Psychological
        Methods*, 17(3), 354-373. (Why the asymmetric and few-category cases
        are the ones that misbehave.)
    """
    tau = _validated_thresholds(thresholds)
    probs = category_probabilities(tau)
    codes = np.arange(1.0, probs.size + 1.0)

    mean = float(probs @ codes)
    variance = float(probs @ (codes - mean) ** 2)
    if variance <= 0.0:
        raise SpecificationError(
            "These thresholds leave every respondent in one category, so the "
            "item has no variance and no correlation can be attenuated.",
            "نقاط القطع هذه تضع جميع المستجيبين في فئة واحدة، فلا يبقى للعنصر "
            "أي تباين ولا يمكن حساب معامل التضييق.",
        )

    covariance = float(stats.norm.pdf(tau).sum())
    kappa = covariance / float(np.sqrt(variance))
    return float(min(kappa, 1.0))


__all__ = [
    "PROPORTION_FLOOR", "MIN_CATEGORIES", "MAX_CATEGORIES",
    "STYLE_BETA_PARAMS", "BIMODAL_COMPONENTS",
    "proportions_for_style", "thresholds_from_proportions",
    "thresholds_for_style", "category_probabilities",
    "categorise", "scale_thresholds", "apply_scale", "reverse_code",
    "achieved_proportions", "polychoric_attenuation",
]

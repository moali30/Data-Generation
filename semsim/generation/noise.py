"""
semsim.generation.noise
=======================
The realism layer: what turns a matrix of model-implied scores into something
that looks like it came off a questionnaire filled in by human beings.

Three contaminations are applied, always in this order:

1. **Careless responders** -- a share of respondents who answer without
   reading.  Two archetypes are implemented: the *straight-liner*, who ticks
   one category all the way down the page, and the *random responder*, who
   answers uniformly over each item's response scale (Meade & Craig, 2012;
   Huang, Curran, Keeney, Poposki & DeShon, 2012).
2. **Outliers** -- single cells pushed ``magnitude_sd`` standard deviations away
   from their column mean (Aguinis, Gottfredson & Joo, 2013).  On an ordinal
   item the value is then clipped back into the legal response range, which is
   a real and frequently missed constraint: a 3.5-SD outlier on a 1-5 Likert
   item can only ever show up as a 1 or a 5.
3. **Missing data** -- MCAR, or MAR through a logistic link on an observed
   driver score whose intercept is calibrated so that the *realised* missing
   rate matches what the researcher asked for (Rubin, 1976; Schouten, Lugtig &
   Vink, 2018).

Every stochastic function takes an explicit :class:`numpy.random.Generator`.
Nothing in this module touches the global ``numpy.random`` state, so a seed
reproduces a dataset exactly.

References
----------
Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey
    data. *Psychological Methods*, 17(3), 437-455.
Huang, J. L., Curran, P. G., Keeney, J., Poposki, E. M., & DeShon, R. P.
    (2012). Detecting and deterring insufficient effort responding to surveys.
    *Journal of Business and Psychology*, 27(1), 99-114.
Aguinis, H., Gottfredson, R. K., & Joo, H. (2013). Best-practice
    recommendations for defining, identifying, and handling outliers.
    *Organizational Research Methods*, 16(2), 270-301.
Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581-592.
Schouten, R. M., Lugtig, P., & Vink, G. (2018). Generating missing values for
    simulation purposes: A multivariate amputation procedure. *Journal of
    Statistical Computation and Simulation*, 88(15), 2909-2930.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit, ndtri

from semsim.core.errors import ConvergenceError, SpecificationError
from semsim.core.spec import MissingMechanism, ModelSpec, ScaleKind, ScaleSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semsim.core.compile import CompiledModel

__all__ = [
    "NoiseReport",
    "CARELESS_STYLES",
    "MAR_LOGIT_SLOPE",
    "inject_careless",
    "inject_outliers",
    "inject_missing",
    "apply_noise",
]

#: The careless-responding archetypes understood by :func:`inject_careless`.
CARELESS_STYLES: Tuple[str, ...] = ("random", "straightline", "mixed")

#: Default steepness of the MAR logistic link, in standardised driver units.
#: 1.5 gives a clearly detectable but not degenerate dependence.
MAR_LOGIT_SLOPE: float = 1.5

#: Fallback SD used when a column is constant, so outlier injection cannot
#: silently become a no-op through a zero spread.
_SD_FLOOR: float = 1e-8


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class NoiseReport:
    """What the realism layer actually did to a dataset.

    Attributes
    ----------
    n_careless : int
        Number of rows replaced by careless response patterns.
    careless_rows : list of int
        Zero-based row indices of those respondents, ascending and unique.
    n_outlier_cells : int
        Number of individual cells replaced by an outlying value.
    n_missing_cells : int
        Number of cells set to ``numpy.nan``.
    applied : list of str
        Human-readable trace of the steps that ran, in execution order.
    """

    n_careless: int = 0
    careless_rows: List[int] = field(default_factory=list)
    n_outlier_cells: int = 0
    n_missing_cells: int = 0
    applied: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _as_matrix(data: Any) -> np.ndarray:
    """Validate that ``data`` is a respondents-by-items matrix."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise SpecificationError(
            "The dataset must be a two-dimensional array "
            "(respondents x items); got "
            f"{arr.ndim} dimension(s).",
            "يجب أن تكون مصفوفة البيانات ثنائية الأبعاد (مستجيبون × فقرات)، "
            f"لكن عدد الأبعاد المُمرَّر هو {arr.ndim}.",
            ndim=arr.ndim,
        )
    if arr.size and not np.issubdtype(arr.dtype, np.number):
        raise SpecificationError(
            "The dataset must be numeric.",
            "يجب أن تكون قيم مصفوفة البيانات رقمية.",
            dtype=str(arr.dtype),
        )
    return arr


def _check_proportion(value: float, label_en: str, label_ar: str) -> float:
    """Reject a proportion outside ``[0, 1]`` with a bilingual message."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = float("nan")
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        raise SpecificationError(
            f"{label_en} must be a proportion between 0 and 1; got {value!r}.",
            f"يجب أن تكون {label_ar} نسبة بين 0 و 1، والقيمة المُدخلة هي "
            f"{value!r}.",
            value=value,
        )
    return p


def _ordinal_flags_and_bounds(
    scales: Optional[Sequence[Optional[ScaleSpec]]], p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column ordinal flag and legal response range.

    Columns without usable metadata are treated as continuous and unbounded,
    which is the safe default: nothing gets clipped that should not be.

    Returns
    -------
    tuple of numpy.ndarray
        ``(is_ordinal, lower, upper)``, each of length ``p``.  ``lower`` and
        ``upper`` hold ``nan`` for continuous columns.
    """
    is_ordinal = np.zeros(p, dtype=bool)
    lower = np.full(p, np.nan)
    upper = np.full(p, np.nan)
    if not scales:
        return is_ordinal, lower, upper
    for j in range(min(p, len(scales))):
        sc = scales[j]
        if sc is None:
            continue
        kind = getattr(sc, "kind", None)
        ordinal = bool(getattr(sc, "is_ordinal", False)) or (
            kind is not None
            and str(getattr(kind, "value", kind)) == ScaleKind.LIKERT.value
        )
        if not ordinal:
            continue
        try:
            k = int(getattr(sc, "n_categories", 5))
        except (TypeError, ValueError):
            continue
        if k < 2:
            raise SpecificationError(
                f"Item {j + 1} declares {k} response categories; an ordinal "
                "item needs at least 2.",
                f"الفقرة رقم {j + 1} مُعرَّفة بـ {k} من فئات الاستجابة، "
                "والفقرة الترتيبية تحتاج فئتين على الأقل.",
                column=j,
                n_categories=k,
            )
        is_ordinal[j] = True
        lower[j] = 1.0
        upper[j] = float(k)
    return is_ordinal, lower, upper


def _column_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware column means and (population) SDs, with a floor on the SD."""
    x = np.asarray(arr, dtype=float)
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    safe = np.where(finite, x, 0.0)
    denom = np.where(counts > 0, counts, 1)
    mean = safe.sum(axis=0) / denom
    var = (np.where(finite, (x - mean) ** 2, 0.0)).sum(axis=0) / denom
    sd = np.sqrt(np.maximum(var, 0.0))
    mean = np.where(counts > 0, mean, 0.0)
    sd = np.where((counts > 0) & (sd > _SD_FLOOR), sd, 1.0)
    return mean, sd


def _column_range(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware column minima and maxima, used as a continuous item's scale."""
    x = np.asarray(arr, dtype=float)
    finite = np.isfinite(x)
    any_finite = finite.any(axis=0)
    big = np.where(finite, x, np.inf)
    small = np.where(finite, x, -np.inf)
    lo = np.where(any_finite, big.min(axis=0), 0.0)
    hi = np.where(any_finite, small.max(axis=0), 1.0)
    return lo.astype(float), hi.astype(float)


def _assign_block(
    out: np.ndarray, rows: np.ndarray, block: np.ndarray
) -> np.ndarray:
    """Write a float block into ``out``, promoting the dtype only if needed."""
    if rows.size == 0:
        return out
    if np.issubdtype(out.dtype, np.integer):
        if np.allclose(block, np.rint(block)):
            out[rows, :] = np.rint(block).astype(out.dtype)
            return out
        out = out.astype(float)
    out[rows, :] = block.astype(out.dtype, copy=False)
    return out


# --------------------------------------------------------------------------- #
# 1. Careless responders
# --------------------------------------------------------------------------- #


def inject_careless(
    data: np.ndarray,
    scales: List[ScaleSpec],
    proportion: float,
    style: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int]]:
    """Replace a share of respondents with careless response patterns.

    The number of careless respondents is ``round(proportion * n)`` and the
    rows are drawn **without replacement**, so no respondent is contaminated
    twice.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n, p)`` matrix of scores; integer Likert codes or continuous floats.
        Never modified in place.
    scales : list of ScaleSpec
        One scale per column, in dataset column order.  Shorter lists are
        padded with "continuous, unbounded"; extra entries are ignored.
    proportion : float
        Share of respondents who answer carelessly, in ``[0, 1]``.
    style : {"random", "straightline", "mixed"}
        ``"straightline"`` draws **one** category for the whole row and repeats
        it across every item -- the classic flat-liner.  ``"random"`` draws
        each cell uniformly over that item's own response scale, honouring a
        per-item ``n_categories`` that may differ across items.  ``"mixed"``
        splits the careless rows between the two archetypes, roughly half each.
    rng : numpy.random.Generator
        The only source of randomness.

    Returns
    -------
    tuple
        ``(contaminated, rows)`` where ``contaminated`` is a new array and
        ``rows`` is the ascending list of careless row indices.

    Notes
    -----
    A straight-liner picks a category that exists on *every* item, so the
    repeated code is drawn from ``1..min(n_categories)``.  Continuous columns
    receive the matching quantile of their own reported metric, which keeps the
    row flat in standardised position rather than in raw units.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``proportion`` is outside ``[0, 1]`` or ``style`` is unknown.

    References
    ----------
    Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in
        survey data. *Psychological Methods*, 17(3), 437-455.
    Huang, J. L., Curran, P. G., Keeney, J., Poposki, E. M., & DeShon, R. P.
        (2012). Detecting and deterring insufficient effort responding to
        surveys. *Journal of Business and Psychology*, 27(1), 99-114.
    """
    arr = _as_matrix(data)
    out = arr.copy()
    n, p = out.shape
    prop = _check_proportion(
        proportion, "The careless-responder proportion", "نسبة المستجيبين غير المبالين"
    )
    key = str(style or "").strip().lower()
    if key not in CARELESS_STYLES:
        raise SpecificationError(
            f"Unknown careless-responding style {style!r}; expected one of "
            f"{', '.join(CARELESS_STYLES)}.",
            f"أسلوب الاستجابة غير المبالية غير معروف: {style!r}. "
            f"الأساليب المتاحة هي: {'، '.join(CARELESS_STYLES)}.",
            style=style,
        )
    n_careless = int(np.rint(prop * n))
    n_careless = int(min(max(n_careless, 0), n))
    if n_careless == 0 or p == 0:
        return out, []

    rows = rng.choice(n, size=n_careless, replace=False)
    is_ordinal, _lower, upper = _ordinal_flags_and_bounds(scales, p)
    col_lo, col_hi = _column_range(arr)

    if key == "straightline":
        straight_rows, random_rows = rows, rows[:0]
    elif key == "random":
        straight_rows, random_rows = rows[:0], rows
    else:  # "mixed" -- rows are already in random order, so a split is random
        cut = n_careless // 2
        straight_rows, random_rows = rows[:cut], rows[cut:]

    # -- straight-liners: one category, repeated ------------------------- #
    if straight_rows.size:
        ordinal_upper = upper[is_ordinal]
        k_ref = int(ordinal_upper.min()) if ordinal_upper.size else 5
        k_ref = max(k_ref, 2)
        picked = rng.integers(1, k_ref + 1, size=straight_rows.size)
        block = np.empty((straight_rows.size, p), dtype=float)
        # Ordinal columns: literally the same code in every column.
        if is_ordinal.any():
            block[:, is_ordinal] = picked[:, None].astype(float)
        # Continuous columns: the same quantile position on the metric.
        cont = ~is_ordinal
        if cont.any():
            z = ndtri((picked - 0.5) / k_ref)
            mean_c, sd_c = _column_stats(arr)
            block[:, cont] = mean_c[cont][None, :] + sd_c[cont][None, :] * z[:, None]
        out = _assign_block(out, straight_rows, block)

    # -- random responders: uniform over each item's own scale ----------- #
    if random_rows.size:
        block = np.empty((random_rows.size, p), dtype=float)
        if is_ordinal.any():
            k_vec = upper[is_ordinal].astype(int)
            u = rng.random((random_rows.size, int(is_ordinal.sum())))
            block[:, is_ordinal] = np.floor(u * k_vec[None, :]) + 1.0
        cont = ~is_ordinal
        if cont.any():
            u = rng.random((random_rows.size, int(cont.sum())))
            lo = col_lo[cont][None, :]
            hi = col_hi[cont][None, :]
            block[:, cont] = lo + u * (hi - lo)
        out = _assign_block(out, random_rows, block)

    return out, sorted(int(r) for r in rows)


# --------------------------------------------------------------------------- #
# 2. Outliers
# --------------------------------------------------------------------------- #


def inject_outliers(
    data: np.ndarray,
    proportion: float,
    magnitude_sd: float,
    rng: np.random.Generator,
    scales: Optional[List[ScaleSpec]] = None,
) -> Tuple[np.ndarray, int]:
    """Replace a proportion of cells with values far from the column mean.

    Target cells are drawn without replacement over the whole ``n * p`` grid.
    Each one becomes ``mean_j +/- magnitude_sd * sd_j`` with the sign chosen at
    random, where ``mean_j`` and ``sd_j`` are that column's own moments.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n, p)`` matrix of scores.  Never modified in place.
    proportion : float
        Share of **cells** (not rows) to contaminate, in ``[0, 1]``.
    magnitude_sd : float
        How many standard deviations away the replacement value sits.  Must be
        positive; 3.0-4.0 is the usual range in the outlier literature.
    rng : numpy.random.Generator
        The only source of randomness.
    scales : list of ScaleSpec, optional
        Column metadata.  When a column is ordinal the outlier is rounded and
        **clipped back into the legal response range** ``1..n_categories``.
        This is not a cosmetic detail: an ordinal item cannot express an
        out-of-range value, so a 3.5-SD outlier on a 1-5 Likert scale can only
        ever appear as a 1 or a 5, and the realised contamination is therefore
        bounded by the scale itself.  Omit (or pass ``None``) to treat every
        column as continuous and unbounded.

    Returns
    -------
    tuple
        ``(contaminated, n_cells)``: a new array and the number of cells that
        were replaced.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``proportion`` is outside ``[0, 1]`` or ``magnitude_sd <= 0``.

    References
    ----------
    Aguinis, H., Gottfredson, R. K., & Joo, H. (2013). Best-practice
        recommendations for defining, identifying, and handling outliers.
        *Organizational Research Methods*, 16(2), 270-301.
    """
    arr = _as_matrix(data)
    out = arr.copy()
    n, p = out.shape
    prop = _check_proportion(
        proportion, "The outlier proportion", "نسبة الخلايا الشاذة"
    )
    try:
        mag = float(magnitude_sd)
    except (TypeError, ValueError):
        mag = float("nan")
    if not np.isfinite(mag) or mag <= 0.0:
        raise SpecificationError(
            "The outlier magnitude must be a positive number of standard "
            f"deviations; got {magnitude_sd!r}.",
            "يجب أن يكون مقدار الشذوذ عددًا موجبًا من الانحرافات المعيارية، "
            f"والقيمة المُدخلة هي {magnitude_sd!r}.",
            magnitude_sd=magnitude_sd,
        )

    n_cells = int(np.rint(prop * n * p))
    n_cells = int(min(max(n_cells, 0), n * p))
    if n_cells == 0:
        return out, 0

    flat = rng.choice(n * p, size=n_cells, replace=False)
    rows, cols = np.unravel_index(flat, (n, p))
    mean, sd = _column_stats(arr)
    sign = np.where(rng.random(n_cells) < 0.5, -1.0, 1.0)
    values = mean[cols] + sign * mag * sd[cols]

    is_ordinal, lower, upper = _ordinal_flags_and_bounds(scales, p)
    ord_mask = is_ordinal[cols]
    if ord_mask.any():
        clipped = np.clip(
            np.rint(values[ord_mask]), lower[cols][ord_mask], upper[cols][ord_mask]
        )
        values[ord_mask] = clipped

    if np.issubdtype(out.dtype, np.integer) and not np.allclose(
        values, np.rint(values)
    ):
        out = out.astype(float)
    out[rows, cols] = values.astype(out.dtype, copy=False)
    return out, n_cells


# --------------------------------------------------------------------------- #
# 3. Missing data
# --------------------------------------------------------------------------- #


def _resolve_driver(
    arr: np.ndarray, driver: Any
) -> np.ndarray:
    """Standardised MAR driver score, one value per respondent."""
    n, p = arr.shape
    if driver is None:
        col = 0 if p else None
        if col is None:
            raise SpecificationError(
                "MAR missingness needs at least one observed column to drive "
                "the mechanism.",
                "تحتاج آلية الفقد من نوع MAR إلى عمود مُشاهد واحد على الأقل "
                "لتوليد الفقد اعتمادًا عليه.",
            )
        values = np.asarray(arr[:, col], dtype=float)
    elif np.isscalar(driver) and not isinstance(driver, (str, bytes)):
        idx = int(driver)
        if idx < 0 or idx >= p:
            raise SpecificationError(
                f"The MAR driver column index {idx} is outside the dataset's "
                f"{p} column(s).",
                f"رقم العمود المُحدَّد لتوليد الفقد من نوع MAR ({idx}) خارج "
                f"نطاق أعمدة البيانات وعددها {p}.",
                driver=idx,
                n_columns=p,
            )
        values = np.asarray(arr[:, idx], dtype=float)
    else:
        values = np.asarray(driver, dtype=float).ravel()
        if values.size != n:
            raise SpecificationError(
                "The MAR driver must hold one value per respondent; got "
                f"{values.size} value(s) for {n} respondent(s).",
                "يجب أن يحتوي متغيّر توليد الفقد من نوع MAR على قيمة واحدة "
                f"لكل مستجيب، لكنه يحتوي {values.size} قيمة مقابل {n} مستجيبًا.",
                driver_size=int(values.size),
                n=n,
            )
    finite = np.isfinite(values)
    if not finite.any():
        raise SpecificationError(
            "The MAR driver column holds no usable values.",
            "لا يحتوي متغيّر توليد الفقد من نوع MAR على أي قيم صالحة للاستخدام.",
        )
    mean = float(values[finite].mean())
    sd = float(values[finite].std())
    if not np.isfinite(sd) or sd <= _SD_FLOOR:
        raise SpecificationError(
            "The MAR driver is constant, so missingness cannot depend on it.",
            "متغيّر توليد الفقد من نوع MAR ثابت القيمة، لذا لا يمكن أن يعتمد "
            "الفقد عليه.",
        )
    z = np.zeros(n, dtype=float)
    z[finite] = (values[finite] - mean) / sd
    return z


def _calibrate_intercept(z: np.ndarray, slope: float, target: float) -> float:
    """Bisect for the logistic intercept that realises ``target`` on average.

    ``mean(expit(a + slope * z))`` is continuous and strictly increasing in
    ``a``, from 0 to 1, so a plain bisection on ``[-50, 50]`` is both safe and
    fast.
    """
    lo, hi = -50.0, 50.0
    if expit(lo + slope * z).mean() > target or expit(hi + slope * z).mean() < target:
        raise ConvergenceError(
            f"Could not calibrate the MAR logistic link to a missing rate of "
            f"{target:.4f}.",
            "تعذّر معايرة دالة الفقد اللوجستية للوصول إلى نسبة فقد مقدارها "
            f"{target:.4f}.",
            target=target,
        )
    mid = 0.5 * (lo + hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expit(mid + slope * z).mean() < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    achieved = float(expit(mid + slope * z).mean())
    if abs(achieved - target) > 1e-6:
        raise ConvergenceError(
            "The MAR logistic link settled on a missing rate of "
            f"{achieved:.4f} instead of the requested {target:.4f}.",
            f"استقرّت دالة الفقد اللوجستية على نسبة فقد {achieved:.4f} بدلًا "
            f"من النسبة المطلوبة {target:.4f}.",
            achieved=achieved,
            target=target,
        )
    return mid


def inject_missing(
    data: np.ndarray,
    mechanism: "MissingMechanism | str",
    proportion: float,
    rng: np.random.Generator,
    driver: Any = None,
    *,
    slope: float = MAR_LOGIT_SLOPE,
) -> Tuple[np.ndarray, int]:
    """Blank out cells under MCAR or MAR and report how many were lost.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n, p)`` matrix of scores.  Never modified in place.
    mechanism : MissingMechanism or str
        ``NONE`` leaves the data untouched (but still returns a copy).
        ``MCAR`` makes every cell missing independently with probability
        ``proportion``.  ``MAR`` makes a whole respondent more or less likely
        to skip items according to an observed driver score.
    proportion : float
        Requested overall share of missing cells, in ``[0, 1]``.
    rng : numpy.random.Generator
        The only source of randomness.
    driver : int or array_like, optional
        MAR only.  Either a column index into ``data`` or a length-``n`` vector
        of observed scores (a composite score, for instance).  Defaults to the
        first column.
    slope : float, keyword-only
        Steepness of the logistic link in standardised driver units.  Positive
        values mean high scorers skip more items.

    Returns
    -------
    tuple
        ``(data_with_nan, n_missing)``.  Whenever a cell is blanked the result
        is a float array carrying ``numpy.nan``; the dtype is preserved when
        nothing goes missing.

    Notes
    -----
    Under MAR the per-respondent probability is
    ``p_i = expit(a + slope * z_i)`` with ``z`` the standardised driver.  The
    intercept ``a`` is solved by bisection so that ``mean(p_i)`` equals the
    requested proportion exactly; the *realised* rate is then that proportion
    plus ordinary binomial noise, rather than the far-off rate you get from an
    uncalibrated link.

    **Every column is eligible for missingness, the driver's own items
    included.**  Passing ``driver`` as an explicit vector only decides where
    the driver *values* are read from; it does not shield any column from
    being blanked.  A caller that needs the driver to stay fully observed in
    the returned matrix must hold those columns out itself -- restore them
    from a pre-call copy, or blank only the sub-matrix of the remaining
    columns.  :func:`apply_noise` does exactly that for the MAR driver
    construct it chooses.

    Raises
    ------
    semsim.core.errors.SpecificationError
        Unknown mechanism, out-of-range proportion, or an unusable driver.
    semsim.core.errors.ConvergenceError
        If the logistic link cannot be calibrated to the requested rate.

    References
    ----------
    Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3),
        581-592.
    Schouten, R. M., Lugtig, P., & Vink, G. (2018). Generating missing values
        for simulation purposes: A multivariate amputation procedure. *Journal
        of Statistical Computation and Simulation*, 88(15), 2909-2930.
    Enders, C. K. (2010). *Applied Missing Data Analysis*. Guilford Press.
    """
    arr = _as_matrix(data)
    out = arr.copy()
    n, p = out.shape
    prop = _check_proportion(
        proportion, "The missing-data proportion", "نسبة البيانات المفقودة"
    )

    raw = getattr(mechanism, "value", mechanism)
    key = str(raw or "").strip().lower()
    if key in ("", "none"):
        return out, 0
    if key not in (MissingMechanism.MCAR.value, MissingMechanism.MAR.value):
        raise SpecificationError(
            f"Unknown missing-data mechanism {mechanism!r}; expected one of "
            "none, mcar, mar.",
            f"آلية البيانات المفقودة غير معروفة: {mechanism!r}. "
            "الآليات المتاحة هي: none أو mcar أو mar.",
            mechanism=mechanism,
        )
    if prop <= 0.0 or n == 0 or p == 0:
        return out, 0

    if key == MissingMechanism.MCAR.value:
        mask = rng.random((n, p)) < prop
    else:
        if prop >= 1.0:
            mask = np.ones((n, p), dtype=bool)
        else:
            z = _resolve_driver(arr, driver)
            intercept = _calibrate_intercept(z, float(slope), prop)
            row_prob = expit(intercept + float(slope) * z)
            mask = rng.random((n, p)) < row_prob[:, None]

    n_missing = int(mask.sum())
    if n_missing == 0:
        return out, 0
    out = out.astype(float, copy=False) if out.dtype.kind == "f" else out.astype(float)
    out[mask] = np.nan
    return out, n_missing


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _scales_for(spec: ModelSpec, cm: Optional["CompiledModel"], p: int
                ) -> List[Optional[ScaleSpec]]:
    """Column scales in dataset order, preferring the compiled column order.

    The fallback has to trigger on an *unusable* lookup, not merely an empty
    one.  ``spec.item(id)`` returns ``None`` for an id it does not know, so
    when ``cm.item_ids`` cannot be resolved against the spec -- a compiled
    model built from a different spec object, ids renamed or regenerated after
    compilation, a duck-typed stand-in -- the comprehension yields a list of
    the right *length* made entirely of ``None``.  The old guard was
    ``if not items:``, which a list of ``None`` passes happily, so the
    fallback never ran and every column came back with no scale.  Downstream
    that is not a mild loss of metadata: :func:`_ordinal_flags_and_bounds`
    treats a ``None`` scale as "continuous, unbounded", so
    :func:`inject_outliers` skips the Likert clip and writes values such as
    -1.9 or 7.9 into a 1-5 item -- illegal response codes in the delivered
    dataset, from a function whose documented contract is that ordinal
    outliers are clipped back into range.  Testing that *at least one* item
    actually resolved is what makes the fallback do its job.
    """
    scales: List[Optional[ScaleSpec]] = []
    items: List[Any] = []
    if cm is not None:
        ids = list(getattr(cm, "item_ids", []) or [])
        getter = getattr(spec, "item", None)
        if ids and callable(getter):
            items = [getter(i) for i in ids]
    if not any(it is not None for it in items):
        try:
            items = list(spec.all_items())
        except Exception:  # pragma: no cover - duck-typed spec
            items = []
    for it in items[:p]:
        scales.append(getattr(it, "scale", None) if it is not None else None)
    while len(scales) < p:
        scales.append(None)
    return scales


def _mar_driver_columns(spec: ModelSpec, cm: Optional["CompiledModel"],
                        driver_id: str, p: int) -> List[int]:
    """Column indices of the items belonging to the MAR driver construct."""
    if not driver_id:
        return []
    construct = None
    getter = getattr(spec, "construct", None)
    if callable(getter):
        construct = getter(driver_id)
    if construct is None:
        by_name = getattr(spec, "construct_by_name", None)
        if callable(by_name):
            construct = by_name(driver_id)
    if construct is None:
        return []
    try:
        item_ids = [it.id for it in construct.all_items()]
    except Exception:  # pragma: no cover - duck-typed spec
        return []
    index = dict(getattr(cm, "item_index", {}) or {}) if cm is not None else {}
    if not index:
        try:
            index = {it.id: k for k, it in enumerate(spec.all_items())}
        except Exception:  # pragma: no cover - duck-typed spec
            return []
    cols = [index[i] for i in item_ids if i in index and index[i] < p]
    return cols


def apply_noise(
    data: np.ndarray,
    spec: ModelSpec,
    cm: Optional["CompiledModel"],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, NoiseReport]:
    """Apply every realism contamination requested by ``spec.noise``.

    The three steps run in a fixed order -- careless responders, then outliers,
    then missingness -- because that is the order in which they happen to a
    real questionnaire: a respondent answers (carelessly or not), some answers
    are extreme, and only then do items get skipped or lost in data entry.
    Running missingness last also means the preceding steps never have to
    reason about ``nan``.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n, p)`` matrix of scores in dataset column order.  Integer Likert
        codes stay integer unless missingness forces ``nan``; the input array
        is never modified in place.
    spec : ModelSpec
        The simulation request.  Only ``spec.noise`` is consulted here.
    cm : CompiledModel or None
        Compiled model, used for column metadata (item order and index).  May
        be ``None``, in which case the order of ``spec.all_items()`` is used.
    rng : numpy.random.Generator
        The only source of randomness.

    Returns
    -------
    tuple
        ``(contaminated, report)`` with a populated :class:`NoiseReport`.

    Notes
    -----
    MAR missingness is driven by the respondent's observed composite score on
    ``spec.noise.mar_driver`` (the row mean of that construct's items).  When
    the construct cannot be resolved, the first exogenous construct's composite
    is used, and failing that the first column.

    The composite is read before any cell is blanked, **and the driver's own
    columns are then held out of the missingness step** -- whichever of the
    three resolutions above supplied them -- so the driver is a genuinely
    observed quantity in the returned matrix: an analyst can recompute it
    from the delivered data for every respondent.

    Reading it pre-blanking is not on its own enough for that: missingness
    used to be applied to all columns, so a driver item was itself blanked for
    a large share of rows (around 40% of respondents at a 20% missing rate on
    a six-item, three-item-driver dataset), and the quantity the mechanism
    depended on was not recoverable from the output.  That is a real loss and
    not a cosmetic one, because MAR is *defined* as missingness that depends
    only on observed data (Rubin, 1976): a partly-missing driver makes the
    mechanism MNAR for the rows where the driver is gone, and silently voids
    the assumption that every MAR-valid method -- FIML, multiple imputation --
    is tested against.

    Holding the driver columns out would drive the realised rate below the
    request, so the per-cell rate on the remaining columns is scaled up by
    ``p / (p - p_driver)``; the *overall* share of missing cells therefore
    still matches ``spec.noise.missing_proportion``, while the driver columns
    stay complete.  Two edge cases: if the driver spans every column there is
    nothing to hold out and the old all-columns behaviour applies, and if the
    scaled-up rate would exceed 1 it is clamped, which leaves the realised
    overall rate below the request.  MCAR needs none of this -- it has no
    driver -- and is applied to all columns as before.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If any requested proportion is outside ``[0, 1]``, or the careless
        style / missing mechanism is unknown.
    semsim.core.errors.ConvergenceError
        If the MAR link cannot be calibrated to the requested rate.

    References
    ----------
    Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in
        survey data. *Psychological Methods*, 17(3), 437-455.
    Aguinis, H., Gottfredson, R. K., & Joo, H. (2013). Best-practice
        recommendations for defining, identifying, and handling outliers.
        *Organizational Research Methods*, 16(2), 270-301.
    Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3),
        581-592.
    """
    arr = _as_matrix(data)
    out = arr.copy()
    report = NoiseReport()
    noise = getattr(spec, "noise", None)
    if noise is None or not bool(getattr(noise, "enabled", False)):
        return out, report

    n, p = out.shape
    scales = _scales_for(spec, cm, p)

    # Validate every requested proportion up front.  The step guards below are
    # ``> 0.0``, and a negative value fails that test exactly as 0.0 does, so
    # a negative proportion used to be silently treated as "no contamination"
    # -- while a value above 1 raised, because it reached the inject_* call
    # that checks it.  The docstring promises a SpecificationError for any
    # proportion outside [0, 1], and the asymmetry hid the far more likely
    # mistake of the two: a sign slip or an off-by-one on a percentage
    # (``-5`` for 5%) turned the whole realism layer off without a word, so
    # the researcher got a pristine dataset and believed it was contaminated.
    # Checking here rather than inside the guards also means the error is
    # raised for a bad request even when that step would have been skipped.
    careless_prop = _check_proportion(
        getattr(noise, "careless_proportion", 0.0) or 0.0,
        "The careless-responder proportion", "نسبة المستجيبين غير المبالين",
    )
    outlier_prop = _check_proportion(
        getattr(noise, "outlier_proportion", 0.0) or 0.0,
        "The outlier proportion", "نسبة الخلايا الشاذة",
    )
    missing_prop = _check_proportion(
        getattr(noise, "missing_proportion", 0.0) or 0.0,
        "The missing-data proportion", "نسبة البيانات المفقودة",
    )

    # -- 1. careless responders ------------------------------------------ #
    if careless_prop > 0.0:
        style = str(getattr(noise, "careless_style", "random") or "random")
        out, rows = inject_careless(out, scales, careless_prop, style, rng)
        report.n_careless = len(rows)
        report.careless_rows = rows
        if rows:
            report.applied.append(f"careless[{style}]={len(rows)} rows")

    # -- 2. outliers ------------------------------------------------------ #
    if outlier_prop > 0.0:
        mag = float(getattr(noise, "outlier_magnitude_sd", 3.5) or 3.5)
        out, n_out = inject_outliers(out, outlier_prop, mag, rng, scales=scales)
        report.n_outlier_cells = n_out
        if n_out:
            report.applied.append(f"outliers[{mag:g}sd]={n_out} cells")

    # -- 3. missingness --------------------------------------------------- #
    mechanism = getattr(noise, "missing_mechanism", MissingMechanism.NONE)
    mech_key = str(getattr(mechanism, "value", mechanism) or "none").lower()
    if mech_key not in ("", "none") and missing_prop > 0.0:
        driver: Any = None
        cols: List[int] = []
        if mech_key == MissingMechanism.MAR.value:
            cols = _mar_driver_columns(
                spec, cm, str(getattr(noise, "mar_driver", "") or ""), p
            )
            if not cols:
                exo = getattr(spec, "exogenous_construct_ids", None)
                if callable(exo):
                    for cid in exo() or []:
                        cols = _mar_driver_columns(spec, cm, cid, p)
                        if cols:
                            break
            if cols:
                driver = np.nanmean(np.asarray(out, dtype=float)[:, cols], axis=1)
            elif p:
                # Last-resort driver: the first column.  Name it explicitly
                # rather than leaving ``driver=None`` for ``inject_missing``
                # to resolve, so that the hold-out below protects it too.
                # Standardising the raw column is exactly what
                # ``_resolve_driver`` would have done with ``None``, so the
                # mechanism is unchanged -- but the guarantee that the driver
                # stays observed must not depend on which fallback fired.
                cols = [0]
                driver = np.asarray(out, dtype=float)[:, 0]

        # Hold the MAR driver's own items out of the blanking.  Reading the
        # composite from the pre-missingness matrix (above) is necessary but
        # not sufficient: if the driver items are then blanked like any other
        # column, the quantity the mechanism depends on is unrecoverable from
        # the delivered data, and "missingness depends on observed data" --
        # the definition of MAR (Rubin, 1976) -- no longer holds for those
        # rows.  The rate on the remaining columns is scaled up by
        # ``p / (p - len(cols))`` so the *overall* share of missing cells
        # still equals what was requested; see the Notes for the two edge
        # cases this cannot cover.
        hold_out = bool(cols) and len(cols) < p
        effective_prop = missing_prop
        if hold_out:
            effective_prop = min(missing_prop * p / (p - len(cols)), 1.0)

        pre_missing = out
        out, n_missing = inject_missing(
            out, mechanism, effective_prop, rng, driver
        )
        if hold_out:
            before_nan = np.isnan(np.asarray(pre_missing, dtype=float))
            out[:, cols] = np.asarray(pre_missing, dtype=out.dtype)[:, cols]
            # Recount: the cells restored above are no longer missing.
            n_missing = int(
                (np.isnan(np.asarray(out, dtype=float)) & ~before_nan).sum()
            )

        report.n_missing_cells = n_missing
        if n_missing:
            report.applied.append(f"missing[{mech_key}]={n_missing} cells")

    return out, report

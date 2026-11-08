"""
semsim.analysis.descriptives
============================
Univariate and multivariate distributional description of a generated (or
imported) dataset: the table that goes at the top of every empirical paper,
plus the normality evidence a reviewer asks for before accepting maximum
likelihood estimation.

What is computed, and why in this particular form
-------------------------------------------------
*   **Mean, SD, median, minimum, maximum** -- ``SD`` uses the unbiased
    ``n - 1`` divisor, as SPSS, R and Excel all do.
*   **Skewness and excess (Fisher) kurtosis**, bias-corrected exactly the way
    SPSS reports them, i.e. the ``G1`` and ``G2`` estimators::

        G1 = sqrt(n(n-1)) / (n-2)                 * m3 / m2**1.5
        G2 = (n-1) / ((n-2)(n-3)) * ((n+1) * g2 + 6),   g2 = m4/m2**2 - 3

    so a normal variable has ``G1 = G2 = 0`` in expectation.  These are what
    :func:`scipy.stats.skew` / :func:`scipy.stats.kurtosis` return with
    ``bias=False``.
*   **Their standard errors**, from the exact sampling variances of ``G1`` and
    ``G2`` under normality (Cramer, 1946; the formulas SPSS prints)::

        SE_skew = sqrt( 6 n (n-1) / ((n-2)(n+1)(n+3)) )
        SE_kurt = 2 SE_skew sqrt( (n^2 - 1) / ((n-3)(n+5)) )

    At ``n = 100`` this gives ``SE_skew = 0.2414``, the number on an SPSS
    printout.  The ``z`` ratios are the estimate over its standard error;
    ``|z| > 3.29`` (alpha = .001) is the usual flag at moderate ``n``
    (Tabachnick & Fidell, 2013).
*   **Shapiro-Wilk**, skipped above ``n = 5000`` (where the test rejects any
    substantive departure and the implementation loses accuracy) and reported
    as ``None`` in that case.
*   **Category proportions** for ordinal items, and **floor / ceiling
    percentages**, which is how a researcher diagnoses a response-style
    artefact rather than a distributional one.
*   **Mardia's multivariate skewness and kurtosis**, the standard screen for
    the joint normality that ML estimation assumes::

        b1p = (1/n^2) sum_i sum_j d_ij^3,   n b1p / 6 ~ chi2, p(p+1)(p+2)/6 df
        b2p = (1/n)   sum_i d_ii^2,         (b2p - p(p+2)) / sqrt(8p(p+2)/n) ~ N(0,1)

    with ``d_ij = (x_i - xbar)' S^-1 (x_j - xbar)`` and ``S`` the
    maximum-likelihood (divisor ``n``) covariance matrix, as in Mardia's
    original derivation.

Computational note on Mardia's skewness
---------------------------------------
``b1p`` appears to need the whole ``n x n`` matrix of Mahalanobis
cross-products, which is impossible at ``n = 20000``.  It does not: writing
``z_i = L^-1 (x_i - xbar)`` for the Cholesky factor ``S = L L'`` gives
``d_ij = z_i' z_j`` and hence the exact identity

.. math::

    \\sum_i \\sum_j (z_i' z_j)^3 = \\sum_{a,b,c} T_{abc}^2,
    \\qquad T_{abc} = \\sum_i z_{ia} z_{ib} z_{ic}

which costs ``O(n p^3)`` instead of ``O(n^2 p)``.  The module uses the tensor
identity when ``n > p^2`` and a blocked Gram-matrix sum otherwise; the two are
algebraically identical and are cross-checked against each other in the tests.

References
----------
Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with
    applications. *Biometrika*, 57(3), 519-530.
Mardia, K. V. (1974). Applications of some measures of multivariate skewness
    and kurtosis in testing normality and robustness studies. *Sankhya B*,
    36(2), 115-128.
Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for
    normality (complete samples). *Biometrika*, 52(3/4), 591-611.
Cramer, H. (1946). *Mathematical Methods of Statistics*. Princeton University
    Press.
Joanes, D. N., & Gill, C. A. (1998). Comparing measures of sample skewness and
    kurtosis. *Journal of the Royal Statistical Society D*, 47(1), 183-189.
Tabachnick, B. G., & Fidell, L. S. (2013). *Using Multivariate Statistics*
    (6th ed.). Pearson.
Curran, P. J., West, S. G., & Finch, J. F. (1996). The robustness of test
    statistics to nonnormality and specification error in confirmatory factor
    analysis. *Psychological Methods*, 1(1), 16-29.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg as _sla
from scipy import stats as _stats

from semsim.core.errors import EstimationError, SpecificationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semsim.core.spec import ModelSpec

#: Shapiro-Wilk is not reported above this sample size.
SHAPIRO_MAX_N = 5000

#: Largest number of distinct values an unlabelled column may take before it
#: stops being treated as an ordinal (Likert) item.
MAX_INFERRED_CATEGORIES = 11

#: ``|z|`` above this flags a non-normal marginal (alpha = .001, two-tailed).
Z_NONNORMALITY_FLAG = 3.29

#: Target number of elements per block when the Gram-matrix route is used.
_GRAM_BLOCK_ELEMENTS = 8_000_000

__all__ = [
    "SHAPIRO_MAX_N", "MAX_INFERRED_CATEGORIES", "Z_NONNORMALITY_FLAG",
    "ItemDescriptives", "DescriptivesReport",
    "describe", "mardia_test", "se_skewness", "se_kurtosis",
]


# --------------------------------------------------------------------------- #
# Standard errors of the shape statistics
# --------------------------------------------------------------------------- #


def se_skewness(n: int) -> float:
    """Standard error of the bias-corrected sample skewness ``G1``.

    ``SE = sqrt( 6 n (n - 1) / ((n - 2)(n + 1)(n + 3)) )`` -- the exact
    sampling standard deviation of ``G1`` under normality, and the value SPSS
    prints next to "Skewness".  At ``n = 100`` it is ``0.2414``.

    Parameters
    ----------
    n : int
        Number of non-missing observations; must be at least 3.

    Returns
    -------
    float
        The standard error, or ``nan`` when ``n < 3``.

    References
    ----------
    Cramer, H. (1946). *Mathematical Methods of Statistics*. Princeton
        University Press.
    Joanes, D. N., & Gill, C. A. (1998). Comparing measures of sample skewness
        and kurtosis. *Journal of the Royal Statistical Society D*, 47(1),
        183-189.
    """
    k = int(n)
    if k < 3:
        return float("nan")
    return math.sqrt(6.0 * k * (k - 1.0) / ((k - 2.0) * (k + 1.0) * (k + 3.0)))


def se_kurtosis(n: int) -> float:
    """Standard error of the bias-corrected sample excess kurtosis ``G2``.

    ``SE = 2 SE_skew sqrt( (n^2 - 1) / ((n - 3)(n + 5)) )``, the companion of
    :func:`se_skewness` and the value SPSS prints next to "Kurtosis".

    Parameters
    ----------
    n : int
        Number of non-missing observations; must be at least 4.

    Returns
    -------
    float
        The standard error, or ``nan`` when ``n < 4``.

    References
    ----------
    Cramer, H. (1946). *Mathematical Methods of Statistics*. Princeton
        University Press.
    Joanes, D. N., & Gill, C. A. (1998). Comparing measures of sample skewness
        and kurtosis. *Journal of the Royal Statistical Society D*, 47(1),
        183-189.
    """
    k = int(n)
    if k < 4:
        return float("nan")
    return 2.0 * se_skewness(k) * math.sqrt(
        (k * k - 1.0) / ((k - 3.0) * (k + 5.0)))


# --------------------------------------------------------------------------- #
# Mardia's multivariate normality screen
# --------------------------------------------------------------------------- #


def _whiten(x: np.ndarray) -> np.ndarray:
    """Centre ``x`` and multiply by ``L^-1`` for the ML covariance ``S = L L'``."""
    n = x.shape[0]
    centred = x - x.mean(axis=0, keepdims=True)
    cov_ml = (centred.T @ centred) / float(n)
    try:
        chol = _sla.cholesky(cov_ml, lower=True)
    except (np.linalg.LinAlgError, _sla.LinAlgError, ValueError) as exc:
        raise EstimationError(
            "The covariance matrix of these variables is singular, so "
            "Mardia's multivariate normality test cannot be computed. Two or "
            "more items are (nearly) identical, or one has no variance.",
            "مصفوفة التغاير لهذه المتغيرات منفردة (غير قابلة للعكس)، ولذلك لا "
            "يمكن حساب اختبار مارديا للاعتدالية المتعددة. هناك بندان أو أكثر "
            "متطابقان تقريبًا، أو بند بلا تباين.",
            detail=str(exc),
        ) from exc
    # z_i = L^-1 (x_i - xbar), returned row-wise so that Z @ Z.T == D.
    return _sla.solve_triangular(chol, centred.T, lower=True).T


def _sum_cubed_cross_products(z: np.ndarray) -> float:
    """``sum_i sum_j (z_i . z_j)^3``, by whichever route is cheaper.

    The tensor identity ``sum_ij (z_i'z_j)^3 = sum_abc T_abc^2`` with
    ``T_abc = sum_i z_ia z_ib z_ic`` costs ``O(n p^3)``; the direct blocked sum
    over the Gram matrix costs ``O(n^2 p)``.  Both are exact.
    """
    n, p = z.shape
    if n > p * p:
        third = np.einsum("ia,ib,ic->abc", z, z, z, optimize=True)
        return float(np.sum(third * third))
    block = max(1, int(_GRAM_BLOCK_ELEMENTS // max(n, 1)))
    total = 0.0
    for start in range(0, n, block):
        gram = z[start:start + block] @ z.T
        total += float(np.sum(gram * gram * gram))
    return total


def mardia_test(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Mardia's test of multivariate normality.

    Both statistics are built from the Mahalanobis cross-products
    ``d_ij = (x_i - xbar)' S^-1 (x_j - xbar)``, with ``S`` the
    maximum-likelihood (divisor ``n``) covariance matrix:

    * multivariate skewness ``b1p = (1/n^2) sum_ij d_ij^3``, tested as
      ``n b1p / 6`` against a chi-square with ``p(p+1)(p+2)/6`` degrees of
      freedom;
    * multivariate kurtosis ``b2p = (1/n) sum_i d_ii^2``, tested as
      ``(b2p - p(p+2)) / sqrt(8 p (p+2) / n)`` against the standard normal,
      two-tailed.

    Rows containing missing values are dropped (listwise), which is what
    every SEM package does before a normality screen.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n, p)`` numeric data matrix.

    Returns
    -------
    (float, float, float, float)
        ``(skewness_statistic, skewness_p, kurtosis_z, kurtosis_p)`` -- i.e.
        the chi-square statistic ``n b1p / 6`` and its ``p`` value, then the
        standard-normal statistic for kurtosis and its two-tailed ``p`` value.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``x`` is not a two-dimensional numeric array, or fewer than
        ``p + 2`` complete cases remain.
    semsim.core.errors.EstimationError
        If the covariance matrix of the complete cases is singular.

    Notes
    -----
    The kurtosis statistic converges slowly; with ``n`` below roughly ``50``
    it is better read as descriptive than as a formal test (Mardia, 1974).

    References
    ----------
    Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with
        applications. *Biometrika*, 57(3), 519-530.
    Mardia, K. V. (1974). Applications of some measures of multivariate
        skewness and kurtosis in testing normality and robustness studies.
        *Sankhya B*, 36(2), 115-128.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2 or data.size == 0:
        raise SpecificationError(
            "Mardia's test needs a two-dimensional numeric data matrix.",
            "يحتاج اختبار مارديا إلى مصفوفة بيانات رقمية ثنائية الأبعاد.",
            shape=getattr(data, "shape", None),
        )
    complete = data[np.all(np.isfinite(data), axis=1)]
    n, p = complete.shape
    if n < p + 2:
        raise SpecificationError(
            f"Mardia's test needs more complete cases than variables; {n} "
            f"complete rows were found for {p} variables.",
            f"يحتاج اختبار مارديا إلى عدد حالات كاملة أكبر من عدد المتغيرات؛ "
            f"وُجد {n} صفًا كاملًا مقابل {p} متغيرًا.",
            n_complete=int(n), p=int(p),
        )

    z = _whiten(complete)
    d_ii = np.einsum("ij,ij->i", z, z)

    b1p = _sum_cubed_cross_products(z) / float(n) ** 2
    b2p = float(np.mean(d_ii ** 2))

    skew_stat = float(n) * b1p / 6.0
    skew_df = p * (p + 1) * (p + 2) / 6.0
    skew_p = float(_stats.chi2.sf(skew_stat, skew_df))

    variance = 8.0 * p * (p + 2.0) / float(n)
    kurt_z = (b2p - p * (p + 2.0)) / math.sqrt(variance)
    kurt_p = float(2.0 * _stats.norm.sf(abs(kurt_z)))

    return (float(skew_stat), skew_p, float(kurt_z), kurt_p)


# --------------------------------------------------------------------------- #
# Per-item description
# --------------------------------------------------------------------------- #


@dataclass
class ItemDescriptives:
    """Univariate description of one observed item.

    Attributes
    ----------
    name : str
        Column name in the dataset.
    n : int
        Number of non-missing observations.
    n_missing : int
        Number of missing (``NaN``) observations.
    mean, sd, median, minimum, maximum : float
        Location and spread; ``sd`` uses the ``n - 1`` divisor.
    skewness, kurtosis : float
        Bias-corrected ``G1`` and *excess* (Fisher) ``G2``, as SPSS reports
        them.
    se_skewness, se_kurtosis : float
        Standard errors under normality.
    z_skewness, z_kurtosis : float
        The estimates divided by their standard errors.
    shapiro_w, shapiro_p : float or None
        Shapiro-Wilk statistic and ``p`` value; ``None`` when ``n`` exceeds
        :data:`SHAPIRO_MAX_N` or falls below 3.
    category_proportions : dict or None
        ``{category: proportion}`` for ordinal items, ``None`` otherwise.
    floor_pct, ceiling_pct : float
        Percentage of cases at the bottom / top of the response scale (at the
        observed extreme when the scale bounds are unknown).
    is_ordinal : bool
        Whether the item was treated as ordinal (added field).
    n_categories : int or None
        Number of response categories when known (added field).
    label_en, label_ar : str
        Item labels for reporting (added fields).
    """

    name: str
    n: int
    n_missing: int
    mean: float
    sd: float
    median: float
    minimum: float
    maximum: float
    skewness: float
    kurtosis: float
    se_skewness: float
    se_kurtosis: float
    z_skewness: float
    z_kurtosis: float
    shapiro_w: Optional[float]
    shapiro_p: Optional[float]
    category_proportions: Optional[Dict[int, float]]
    floor_pct: float
    ceiling_pct: float
    # ---- additions (appended, so positional construction stays valid) ----- #
    is_ordinal: bool = False
    n_categories: Optional[int] = None
    label_en: str = ""
    label_ar: str = ""

    @property
    def normal_marginal(self) -> bool:
        """Whether both shape ``z`` ratios stay inside the .001 flag."""
        zs = (self.z_skewness, self.z_kurtosis)
        if any(not math.isfinite(v) for v in zs):
            return False
        return all(abs(v) <= Z_NONNORMALITY_FLAG for v in zs)


@dataclass
class DescriptivesReport:
    """Per-item descriptives plus the multivariate normality screen.

    Attributes
    ----------
    items : list of ItemDescriptives
        One entry per analysed column, in dataset order.
    mardia_skewness : float
        ``n b1p / 6``, Mardia's chi-square statistic for multivariate
        skewness.
    mardia_skewness_p : float
        Its ``p`` value.
    mardia_kurtosis : float
        The standard-normal statistic for multivariate kurtosis.
    mardia_kurtosis_p : float
        Its two-tailed ``p`` value.
    multivariate_normal : bool
        ``True`` when neither Mardia test rejects at alpha = .05.
    n : int
        Number of rows in the dataset.
    n_complete : int
        Number of rows with no missing value (added field).
    mardia_b1p, mardia_b2p : float
        The raw Mardia coefficients (added fields).
    warnings : list of str
        Bilingual notes, e.g. that Mardia's test had to be skipped (added
        field).
    """

    items: List[ItemDescriptives]
    mardia_skewness: float
    mardia_skewness_p: float
    mardia_kurtosis: float
    mardia_kurtosis_p: float
    multivariate_normal: bool
    n: int
    # ---- additions (appended, so positional construction stays valid) ----- #
    n_complete: int = 0
    mardia_b1p: float = float("nan")
    mardia_b2p: float = float("nan")
    warnings: List[str] = field(default_factory=list)

    def item(self, name: str) -> Optional[ItemDescriptives]:
        """Look one item up by column name (added convenience)."""
        for entry in self.items:
            if entry.name == name:
                return entry
        return None

    def to_table(self, lang: str = "en") -> pd.DataFrame:
        """Render the per-item descriptives as the table the report prints.

        Parameters
        ----------
        lang : {"en", "ar"}, default "en"
            Language of the ``label`` and ``normality`` columns.  Column keys
            stay in English so the report and the UI can address them
            programmatically.

        Returns
        -------
        pandas.DataFrame
            One row per item with the columns ``item``, ``label``, ``n``,
            ``n_missing``, ``mean``, ``sd``, ``median``, ``min``, ``max``,
            ``skewness``, ``se_skewness``, ``z_skewness``, ``kurtosis``,
            ``se_kurtosis``, ``z_kurtosis``, ``shapiro_w``, ``shapiro_p``,
            ``floor_pct``, ``ceiling_pct`` and ``normality``.

        References
        ----------
        Tabachnick, B. G., & Fidell, L. S. (2013). *Using Multivariate
            Statistics* (6th ed.). Pearson.
        """
        arabic = str(lang).lower().startswith("ar")
        verdicts = ("normal", "non-normal") if not arabic else (
            "اعتدالي", "غير اعتدالي")
        records: List[Dict[str, object]] = []
        for entry in self.items:
            label = (entry.label_ar or entry.name) if arabic else (
                entry.label_en or entry.name)
            records.append({
                "item": entry.name,
                "label": label,
                "n": entry.n,
                "n_missing": entry.n_missing,
                "mean": entry.mean,
                "sd": entry.sd,
                "median": entry.median,
                "min": entry.minimum,
                "max": entry.maximum,
                "skewness": entry.skewness,
                "se_skewness": entry.se_skewness,
                "z_skewness": entry.z_skewness,
                "kurtosis": entry.kurtosis,
                "se_kurtosis": entry.se_kurtosis,
                "z_kurtosis": entry.z_kurtosis,
                "shapiro_w": (float("nan") if entry.shapiro_w is None
                              else entry.shapiro_w),
                "shapiro_p": (float("nan") if entry.shapiro_p is None
                              else entry.shapiro_p),
                "floor_pct": entry.floor_pct,
                "ceiling_pct": entry.ceiling_pct,
                "normality": verdicts[0] if entry.normal_marginal else verdicts[1],
            })
        return pd.DataFrame.from_records(records, columns=[
            "item", "label", "n", "n_missing", "mean", "sd", "median", "min",
            "max", "skewness", "se_skewness", "z_skewness", "kurtosis",
            "se_kurtosis", "z_kurtosis", "shapiro_w", "shapiro_p",
            "floor_pct", "ceiling_pct", "normality"])


# --------------------------------------------------------------------------- #
# Spec-driven metadata (duck-typed, so the spec argument stays optional)
# --------------------------------------------------------------------------- #


def _item_metadata(spec: Optional["ModelSpec"]
                   ) -> Dict[str, Tuple[bool, Optional[int], str, str]]:
    """Map item name to ``(is_ordinal, n_categories, label_en, label_ar)``.

    Reads the specification defensively: any attribute that is absent simply
    falls back on inference from the data, so a half-built spec never breaks
    the descriptives.
    """
    out: Dict[str, Tuple[bool, Optional[int], str, str]] = {}
    if spec is None:
        return out
    items = getattr(spec, "all_items", None)
    if not callable(items):
        return out
    try:
        collection = list(items())
    except Exception:  # pragma: no cover - defensive against a partial spec
        return out
    for item in collection:
        name = str(getattr(item, "name", "") or "")
        if not name:
            continue
        scale = getattr(item, "scale", None)
        ordinal = bool(getattr(scale, "is_ordinal", False))
        categories = getattr(scale, "n_categories", None)
        n_categories = (int(categories)
                        if ordinal and isinstance(categories, (int, np.integer))
                        else None)
        label = getattr(item, "label", None)
        label_en = str(getattr(label, "en", "") or "")
        label_ar = str(getattr(label, "ar", "") or "")
        out[name] = (ordinal, n_categories, label_en, label_ar)
    return out


def _infer_ordinal(values: np.ndarray) -> Tuple[bool, Optional[int]]:
    """Guess whether an unlabelled column is an ordinal (Likert) item."""
    if values.size == 0:
        return (False, None)
    if not np.all(np.isclose(values, np.rint(values))):
        return (False, None)
    distinct = np.unique(np.rint(values))
    spread = float(distinct[-1] - distinct[0]) + 1.0
    if distinct.size <= MAX_INFERRED_CATEGORIES and spread <= MAX_INFERRED_CATEGORIES:
        return (True, int(round(spread)))
    return (False, None)


def _numeric_column(frame: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    """Return a column as a float array, or ``None`` when it is not numeric."""
    series = frame[name]
    if isinstance(series, pd.DataFrame):  # duplicated column label
        series = series.iloc[:, 0]
    try:
        converted = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError):
        return None
    return np.asarray(converted, dtype=float)


def _describe_column(name: str, column: np.ndarray, ordinal: bool,
                     n_categories: Optional[int], label_en: str,
                     label_ar: str) -> ItemDescriptives:
    """Build one :class:`ItemDescriptives` from a float column."""
    nan = float("nan")
    finite = np.isfinite(column)
    values = column[finite]
    n_obs = int(values.size)
    n_missing = int(column.size - n_obs)

    if n_obs == 0:
        return ItemDescriptives(
            name=name, n=0, n_missing=n_missing, mean=nan, sd=nan, median=nan,
            minimum=nan, maximum=nan, skewness=nan, kurtosis=nan,
            se_skewness=nan, se_kurtosis=nan, z_skewness=nan, z_kurtosis=nan,
            shapiro_w=None, shapiro_p=None, category_proportions=None,
            floor_pct=nan, ceiling_pct=nan, is_ordinal=ordinal,
            n_categories=n_categories, label_en=label_en, label_ar=label_ar)

    if not ordinal and n_categories is None:
        ordinal, n_categories = _infer_ordinal(values)

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n_obs > 1 else nan
    median = float(np.median(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    constant = maximum <= minimum

    if n_obs >= 3 and not constant:
        skewness = float(_stats.skew(values, bias=False))
    else:
        skewness = nan
    if n_obs >= 4 and not constant:
        kurtosis = float(_stats.kurtosis(values, fisher=True, bias=False))
    else:
        kurtosis = nan

    se_skew = se_skewness(n_obs)
    se_kurt = se_kurtosis(n_obs)
    z_skew = (skewness / se_skew
              if math.isfinite(skewness) and math.isfinite(se_skew) and se_skew > 0
              else nan)
    z_kurt = (kurtosis / se_kurt
              if math.isfinite(kurtosis) and math.isfinite(se_kurt) and se_kurt > 0
              else nan)

    shapiro_w: Optional[float] = None
    shapiro_p: Optional[float] = None
    if 3 <= n_obs <= SHAPIRO_MAX_N and not constant:
        result = _stats.shapiro(values)
        shapiro_w = float(result.statistic)
        shapiro_p = float(result.pvalue)

    proportions: Optional[Dict[int, float]] = None
    if ordinal:
        rounded = np.rint(values).astype(int)
        observed, counts = np.unique(rounded, return_counts=True)
        tally = {int(k): int(v) for k, v in zip(observed, counts)}
        if n_categories is not None and n_categories > 0:
            keys = list(range(1, int(n_categories) + 1))
            for extra in sorted(tally):
                if extra not in keys:
                    keys.append(extra)
            keys.sort()
        else:
            keys = sorted(tally)
        proportions = {k: float(tally.get(k, 0)) / float(n_obs) for k in keys}

    if ordinal and n_categories is not None and n_categories > 0:
        low, high = 1.0, float(int(n_categories))
    else:
        low, high = minimum, maximum
    floor_pct = 100.0 * float(np.count_nonzero(values <= low)) / float(n_obs)
    ceiling_pct = 100.0 * float(np.count_nonzero(values >= high)) / float(n_obs)

    return ItemDescriptives(
        name=name, n=n_obs, n_missing=n_missing, mean=mean, sd=sd,
        median=median, minimum=minimum, maximum=maximum, skewness=skewness,
        kurtosis=kurtosis, se_skewness=se_skew, se_kurtosis=se_kurt,
        z_skewness=z_skew, z_kurtosis=z_kurt, shapiro_w=shapiro_w,
        shapiro_p=shapiro_p, category_proportions=proportions,
        floor_pct=floor_pct, ceiling_pct=ceiling_pct, is_ordinal=bool(ordinal),
        n_categories=n_categories, label_en=label_en, label_ar=label_ar)


# --------------------------------------------------------------------------- #
# The main entry point
# --------------------------------------------------------------------------- #


def describe(data: pd.DataFrame,
             spec: Optional["ModelSpec"] = None) -> DescriptivesReport:
    """Describe every numeric column and screen the set for joint normality.

    Parameters
    ----------
    data : pandas.DataFrame
        Observed items, one column per item.  A two-dimensional array-like is
        accepted and converted.  Missing values (``NaN``) are handled per item
        for the univariate statistics and listwise for Mardia's test.
    spec : ModelSpec, optional
        When supplied, item labels and the declared number of response
        categories are taken from it instead of being inferred from the data,
        which matters for the floor / ceiling percentages: the floor of a
        five-point scale is category 1 whether or not anybody chose it.

    Returns
    -------
    DescriptivesReport
        Per-item statistics, Mardia's two tests and a joint-normality verdict.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``data`` cannot be read as a table, or contains no numeric column.

    Notes
    -----
    Mardia's test is skipped (statistics ``nan``, a note appended to
    ``warnings``, ``multivariate_normal`` left ``False``) when there are too
    few complete cases or the item covariance matrix is singular; the
    univariate part of the report is still returned in full, so a dataset with
    missing values never costs the researcher the whole table.

    References
    ----------
    Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with
        applications. *Biometrika*, 57(3), 519-530.
    Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for
        normality (complete samples). *Biometrika*, 52(3/4), 591-611.
    Curran, P. J., West, S. G., & Finch, J. F. (1996). The robustness of test
        statistics to nonnormality and specification error in confirmatory
        factor analysis. *Psychological Methods*, 1(1), 16-29.
    """
    if isinstance(data, pd.DataFrame):
        frame = data
    else:
        try:
            frame = pd.DataFrame(data)
        except Exception as exc:
            raise SpecificationError(
                "The data could not be read as a table of items by cases.",
                "لم يتمكّن البرنامج من قراءة البيانات كجدول من البنود والحالات.",
                detail=str(exc),
            ) from exc

    warnings: List[str] = []
    metadata = _item_metadata(spec)

    entries: List[ItemDescriptives] = []
    numeric: List[np.ndarray] = []
    for name in list(frame.columns):
        column = _numeric_column(frame, str(name))
        if column is None:
            warnings.append(
                f"Column '{name}' is not numeric and was skipped. / "
                f"العمود «{name}» غير رقمي وتم استثناؤه من الوصف الإحصائي.")
            continue
        ordinal, categories, label_en, label_ar = metadata.get(
            str(name), (False, None, "", ""))
        entries.append(_describe_column(str(name), column, ordinal, categories,
                                        label_en, label_ar))
        numeric.append(column)

    if not entries:
        raise SpecificationError(
            "The dataset contains no numeric item to describe.",
            "لا تحتوي البيانات على أي بند رقمي يمكن وصفه إحصائيًا.",
            columns=[str(c) for c in frame.columns],
        )

    n_rows = int(frame.shape[0])
    matrix = np.column_stack(numeric) if numeric else np.empty((n_rows, 0))
    complete_mask = (np.all(np.isfinite(matrix), axis=1) if matrix.size
                     else np.zeros(n_rows, dtype=bool))
    n_complete = int(np.count_nonzero(complete_mask))

    nan = float("nan")
    skew_stat = skew_p = kurt_z = kurt_p = nan
    b1p = b2p = nan
    multivariate_normal = False
    p = matrix.shape[1]
    if p < 2:
        warnings.append(
            "Mardia's multivariate test needs at least two items. / "
            "يحتاج اختبار مارديا المتعدد إلى بندين على الأقل.")
    else:
        try:
            skew_stat, skew_p, kurt_z, kurt_p = mardia_test(matrix)
        except (EstimationError, SpecificationError) as exc:
            warnings.append(f"{exc.message_en} / {exc.message_ar}")
        else:
            b1p = 6.0 * skew_stat / float(n_complete)
            b2p = (kurt_z * math.sqrt(8.0 * p * (p + 2.0) / float(n_complete))
                   + p * (p + 2.0))
            multivariate_normal = bool(skew_p > 0.05 and kurt_p > 0.05)

    if any(entry.n_missing for entry in entries):
        total_missing = sum(entry.n_missing for entry in entries)
        warnings.append(
            f"{total_missing} missing cells were excluded; univariate "
            f"statistics use the available cases per item and Mardia's test "
            f"uses the {n_complete} complete cases. / "
            f"تم استثناء {total_missing} خلية مفقودة؛ الإحصاءات الأحادية محسوبة "
            f"على الحالات المتاحة لكل بند، واختبار مارديا محسوب على "
            f"{n_complete} حالة كاملة.")

    return DescriptivesReport(
        items=entries,
        mardia_skewness=float(skew_stat),
        mardia_skewness_p=float(skew_p),
        mardia_kurtosis=float(kurt_z),
        mardia_kurtosis_p=float(kurt_p),
        multivariate_normal=multivariate_normal,
        n=n_rows,
        n_complete=n_complete,
        mardia_b1p=float(b1p),
        mardia_b2p=float(b2p),
        warnings=warnings,
    )

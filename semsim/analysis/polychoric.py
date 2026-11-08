"""
semsim.analysis.polychoric
==========================
Correlations between *ordinal* variables that estimate the association of the
continuous variables assumed to underlie them.

Why a researcher needs this
---------------------------
A five-point Likert item is a coarsened view of a latent continuous response
``y*``.  Coarsening attenuates Pearson's ``r``: two normal variables
correlated ``0.60`` and then cut into five symmetric categories correlate only
about ``0.56`` as integers (the attenuation factor is
:func:`semsim.generation.thresholds.polychoric_attenuation` squared).  A
factor model fitted to the attenuated matrix therefore under-estimates every
loading, which is exactly the bias Muthen (1984) introduced the polychoric
correlation to remove: it reconstructs the correlation of the *underlying*
normals and so returns a matrix that an ordinary (DWLS / ULS) SEM estimator
may treat as if the data had been continuous all along.

The estimator implemented here
------------------------------
Olsson's (1979) **two-step** estimator, which is what ``lavaan``,
``polycor`` and Mplus use by default:

1.  **Thresholds from the univariate marginals.**  For a variable with
    categories ``1 .. I`` and observed proportions ``p_1 .. p_I``, the
    thresholds are the inverse standard normal CDF of the cumulative
    proportions::

        tau_i = Phi^-1(p_1 + ... + p_i),      i = 1 .. I - 1

    These are the maximum-likelihood estimates of the thresholds from the
    marginal alone, and they do not depend on ``rho`` -- which is what makes
    the method "two-step" rather than full information.

2.  **One-dimensional maximum likelihood for ``rho``.**  With the thresholds
    held fixed, the multinomial log-likelihood of the ``I x J`` contingency
    table is::

        logL(rho) = sum_ij  n_ij * log pi_ij(rho)

        pi_ij(rho) = Phi2(a_i, b_j; rho) - Phi2(a_{i-1}, b_j; rho)
                   - Phi2(a_i, b_{j-1}; rho) + Phi2(a_{i-1}, b_{j-1}; rho)

    with ``a_0 = b_0 = -inf``, ``a_I = b_J = +inf`` and ``Phi2`` the
    standard bivariate normal CDF.  ``logL`` is maximised over the bounded
    interval ``(-0.999, 0.999)`` with Brent's method
    (:func:`scipy.optimize.minimize_scalar`, ``method="bounded"``).

The bivariate normal CDF
------------------------
:func:`bivariate_normal_cdf` prefers
:func:`scipy.stats.multivariate_normal.cdf` and falls back on the
Drezner-Wesolowsky (1990) one-dimensional integral

.. math::

    \\Phi_2(h, k; \\rho) = \\Phi(h)\\Phi(k)
        + \\frac{1}{2\\pi} \\int_0^{\\rho}
          \\frac{1}{\\sqrt{1 - r^2}}
          \\exp\\!\\left(-\\frac{h^2 - 2 r h k + k^2}{2(1 - r^2)}\\right) dr ,

whose integrand is the bivariate normal density, so the identity is exact.
The fallback is evaluated with 48-node Gauss-Legendre quadrature, split into
two panels at ``|r| = 0.95`` where the integrand becomes peaked; against
SciPy on a grid of ``h, k`` in ``[-4, 4]`` the agreement is ``3e-15`` even at
``rho = 0.999``.  The likelihood above needs ``Phi2`` on a whole
``(I+1) x (J+1)`` grid of threshold pairs at one value of ``rho``, so the
optimiser uses the *vectorised* quadrature (one call per iteration, about
0.1 ms) rather than SciPy's scalar routine; the two agree to machine
precision, which :mod:`tests.test_reliability` asserts.

Degenerate data
---------------
*   A category nobody chose (an interior zero frequency) produces two
    coincident thresholds and hence a zero-probability row or column of the
    table.  Because its observed count is also zero it contributes nothing to
    the likelihood, and the estimate is simply the one obtained from the
    categories that *were* used.  Integer-coded variables are deliberately
    expanded over their full contiguous range ``min .. max`` so that such a
    gap is visible rather than silently closed up.
*   A variable with no variance at all cannot be correlated with anything and
    raises :class:`~semsim.core.errors.SpecificationError`.
*   ``polychoric_correlation(x, x)`` short-circuits to exactly ``1.0``.

References
----------
Olsson, U. (1979). Maximum likelihood estimation of the polychoric
    correlation coefficient. *Psychometrika*, 44(4), 443-460.
Olsson, U., Drasgow, F., & Dorans, N. J. (1982). The polyserial correlation
    coefficient. *Psychometrika*, 47(3), 337-347.
Muthen, B. (1984). A general structural equation model with dichotomous,
    ordered categorical, and continuous latent variable indicators.
    *Psychometrika*, 49(1), 115-132.
Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the bivariate
    normal integral. *Journal of Statistical Computation and Simulation*,
    35(1-2), 101-107.
Pearson, K. (1900). Mathematical contributions to the theory of evolution.
    VII. On the correlation of characters not quantitatively measurable.
    *Philosophical Transactions of the Royal Society A*, 195, 1-47.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from semsim.core.errors import SpecificationError

__all__ = [
    "MAX_ABS_RHO",
    "THRESHOLD_CLAMP",
    "bivariate_normal_cdf",
    "polychoric_correlation",
    "polyserial_correlation",
    "polychoric_matrix",
    "thresholds_from_counts",
]

#: The optimiser never reports a correlation outside this bound; at ``|rho| =
#: 1`` the bivariate normal is singular and the likelihood is unbounded.
MAX_ABS_RHO = 0.999

#: Stand-in for an infinite threshold.  ``Phi(-8.5) = 9.5e-18``, so replacing
#: ``-inf`` by this value changes a cell probability by less than the double
#: precision floor while keeping every expression finite.
THRESHOLD_CLAMP = 8.5

#: Gauss-Legendre nodes for the Drezner-Wesolowsky integral.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)

#: Above this ``|rho|`` the integrand is peaked near the upper limit, so the
#: quadrature is applied to two panels instead of one.
_PANEL_SPLIT = 0.95

_LOG_FLOOR = 1e-15

#: An ordinal variable fed to the polyserial estimator may carry many more
#: categories than a Likert item (a summed scale, for instance).
_POLYSERIAL_MAX_CATEGORIES = 64


# --------------------------------------------------------------------------- #
# The bivariate normal CDF
# --------------------------------------------------------------------------- #


def _dw_cdf(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """Drezner-Wesolowsky bivariate normal CDF, vectorised over ``h`` and ``k``.

    Parameters
    ----------
    h, k : numpy.ndarray
        Upper limits, broadcast against one another.  Finite values only;
        callers substitute :data:`THRESHOLD_CLAMP` for infinities.
    rho : float
        Correlation, ``|rho| <= 1``.

    Returns
    -------
    numpy.ndarray
        ``P(X <= h, Y <= k)`` with the broadcast shape of ``h`` and ``k``.

    References
    ----------
    Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the
        bivariate normal integral. *Journal of Statistical Computation and
        Simulation*, 35(1-2), 101-107.
    """
    h_arr = np.asarray(h, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    base = stats.norm.cdf(h_arr) * stats.norm.cdf(k_arr)
    rho = float(np.clip(rho, -1.0, 1.0))
    if rho == 0.0:
        return base

    if abs(rho) <= _PANEL_SPLIT:
        panels: List[Tuple[float, float]] = [(0.0, rho)]
    else:
        edge = _PANEL_SPLIT * (1.0 if rho > 0 else -1.0)
        panels = [(0.0, edge), (edge, rho)]

    integral = np.zeros(base.shape, dtype=float)
    hh = h_arr[..., None]
    kk = k_arr[..., None]
    for lo, hi in panels:
        half = 0.5 * (hi - lo)
        r = half * (_GL_NODES + 1.0) + lo
        weights = half * _GL_WEIGHTS
        one_minus = 1.0 - r * r
        exponent = -(hh * hh - 2.0 * r * hh * kk + kk * kk) / (2.0 * one_minus)
        density = np.exp(exponent) / np.sqrt(one_minus)
        integral = integral + np.einsum("...i,i->...", density, weights)
    return base + integral / (2.0 * np.pi)


def bivariate_normal_cdf(h: float, k: float, rho: float) -> float:
    """Standard bivariate normal distribution function ``P(X <= h, Y <= k)``.

    Both margins are standard normal and their correlation is ``rho``.
    :func:`scipy.stats.multivariate_normal.cdf` is tried first; if it fails or
    returns something non-finite the Drezner-Wesolowsky quadrature is used
    instead.  Infinite limits are handled analytically.

    Parameters
    ----------
    h : float
        Upper limit for the first variable; ``-inf`` and ``+inf`` allowed.
    k : float
        Upper limit for the second variable; ``-inf`` and ``+inf`` allowed.
    rho : float
        Correlation between the two variables, in ``[-1, 1]``.

    Returns
    -------
    float
        The orthant probability, in ``[0, 1]``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``rho`` is not a real number in ``[-1, 1]``, or a limit is NaN.

    Examples
    --------
    >>> round(bivariate_normal_cdf(0.0, 0.0, 0.0), 6)
    0.25
    >>> round(bivariate_normal_cdf(0.0, 0.0, 1.0), 6)
    0.5

    Notes
    -----
    ``Phi2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi)`` exactly (Sheppard's
    theorem), which is the closed form the test module checks against.

    References
    ----------
    Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the
        bivariate normal integral. *Journal of Statistical Computation and
        Simulation*, 35(1-2), 101-107.
    Sheppard, W. F. (1899). On the application of the theory of error to cases
        of normal distribution and normal correlation. *Philosophical
        Transactions of the Royal Society A*, 192, 101-167.
    """
    h_f = float(h)
    k_f = float(k)
    rho_f = float(rho)
    if np.isnan(h_f) or np.isnan(k_f) or np.isnan(rho_f):
        raise SpecificationError(
            "The bivariate normal CDF needs real limits and a real correlation; "
            "a NaN was supplied.",
            "دالة التوزيع الطبيعي الثنائي تتطلّب حدودًا ومعامل ارتباط بقيم "
            "حقيقية، وقد مُرِّرت قيمة غير معرّفة (NaN).",
            h=h_f, k=k_f, rho=rho_f,
        )
    if not -1.0 <= rho_f <= 1.0:
        raise SpecificationError(
            f"A correlation must lie between -1 and 1; got {rho_f:g}.",
            f"يجب أن تكون قيمة معامل الارتباط بين -1 و 1، والقيمة المُمرَّرة "
            f"هي {rho_f:g}.",
            rho=rho_f,
        )

    # Analytic edges, which also keep the quadrature away from infinities.
    if h_f == -np.inf or k_f == -np.inf:
        return 0.0
    if h_f == np.inf and k_f == np.inf:
        return 1.0
    if h_f == np.inf:
        return float(stats.norm.cdf(k_f))
    if k_f == np.inf:
        return float(stats.norm.cdf(h_f))
    if rho_f == 1.0:
        return float(stats.norm.cdf(min(h_f, k_f)))
    if rho_f == -1.0:
        return float(max(0.0, stats.norm.cdf(h_f) - stats.norm.cdf(-k_f)))

    try:
        value = float(stats.multivariate_normal.cdf(
            np.array([h_f, k_f]),
            mean=np.zeros(2),
            cov=np.array([[1.0, rho_f], [rho_f, 1.0]]),
        ))
    except Exception:          # pragma: no cover - SciPy fallback path
        value = float("nan")
    if not np.isfinite(value):  # pragma: no cover - SciPy fallback path
        value = float(_dw_cdf(np.array(h_f), np.array(k_f), rho_f))
    return float(min(max(value, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# Categorisation and thresholds
# --------------------------------------------------------------------------- #


def _codes_and_counts(values: np.ndarray, max_categories: int,
                      label: str) -> Tuple[np.ndarray, np.ndarray]:
    """Map observed scores onto ``0 .. I-1`` and count each category.

    Integer-like scores keep their *contiguous* range, so a response option
    nobody used shows up as a zero count instead of being closed up.  Any
    other coding (arbitrary floats, strings already converted to codes) is
    ranked by sorted unique value.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SpecificationError(
            f"Variable {label!r} has no usable (non-missing) values.",
            f"المتغيّر {label!r} لا يحتوي على أي قيمة صالحة بعد استبعاد القيم "
            f"المفقودة.",
            variable=label,
        )

    integer_like = bool(np.all(np.isclose(finite, np.round(finite))))
    if integer_like:
        lo = int(round(float(finite.min())))
        hi = int(round(float(finite.max())))
        span = hi - lo + 1
        if 1 <= span <= max_categories:
            codes = np.round(values).astype(np.int64) - lo
            counts = np.bincount(codes, minlength=span).astype(float)
            return codes, counts

    uniques, codes = np.unique(values, return_inverse=True)
    counts = np.bincount(codes, minlength=uniques.size).astype(float)
    if uniques.size > max_categories:
        raise SpecificationError(
            f"Variable {label!r} has {uniques.size} distinct values, more than "
            f"the {max_categories} allowed for an ordinal item; it looks "
            f"continuous rather than categorical.",
            f"المتغيّر {label!r} يحتوي على {uniques.size} قيمة مختلفة، وهو عدد "
            f"يتجاوز الحد المسموح به ({max_categories}) للبند الرتبي؛ يبدو أن "
            f"المتغيّر متصل وليس مصنّفًا.",
            variable=label, n_categories=int(uniques.size),
        )
    return codes, counts


def thresholds_from_counts(counts: Sequence[float] | np.ndarray) -> np.ndarray:
    """Maximum-likelihood thresholds of an ordinal variable from its margins.

    Parameters
    ----------
    counts : sequence of float
        Observed frequency of each category, in category order.  Length
        ``I``; zero entries are allowed.

    Returns
    -------
    numpy.ndarray
        The ``I - 1`` interior thresholds ``Phi^-1`` of the cumulative
        proportions, clipped to ``+/- THRESHOLD_CLAMP`` so a zero or unit
        cumulative proportion stays finite.  Non-decreasing by construction.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the counts are empty, negative, or sum to zero.

    Examples
    --------
    >>> thresholds_from_counts([25, 25, 25, 25]).round(6)
    array([-0.67449 ,  0.      ,  0.67449 ])

    References
    ----------
    Olsson, U. (1979). Maximum likelihood estimation of the polychoric
        correlation coefficient. *Psychometrika*, 44(4), 443-460.
    """
    arr = np.asarray(counts, dtype=float).ravel()
    if arr.size == 0 or np.any(arr < 0) or not np.isfinite(arr).all():
        raise SpecificationError(
            "Category counts must be a non-empty vector of non-negative "
            "finite numbers.",
            "يجب أن تكون تكرارات الفئات متجهًا غير فارغ من أعداد غير سالبة "
            "ومنتهية.",
        )
    total = float(arr.sum())
    if total <= 0:
        raise SpecificationError(
            "Category counts sum to zero, so no thresholds can be estimated.",
            "مجموع تكرارات الفئات يساوي صفرًا، ولا يمكن تقدير الحدود الفاصلة.",
        )
    cumulative = np.cumsum(arr[:-1]) / total
    cumulative = np.clip(cumulative, 1e-12, 1.0 - 1e-12)
    return np.clip(stats.norm.ppf(cumulative), -THRESHOLD_CLAMP, THRESHOLD_CLAMP)


def _cell_probabilities(tau_x: np.ndarray, tau_y: np.ndarray,
                        rho: float) -> np.ndarray:
    """Bivariate normal probability of every cell of the contingency table."""
    a = np.concatenate(([-THRESHOLD_CLAMP], tau_x, [THRESHOLD_CLAMP]))
    b = np.concatenate(([-THRESHOLD_CLAMP], tau_y, [THRESHOLD_CLAMP]))
    grid = _dw_cdf(a[:, None], b[None, :], rho)
    # Second difference of the joint CDF over the threshold grid.
    probs = np.diff(np.diff(grid, axis=0), axis=1)
    return np.maximum(probs, 0.0)


def _negative_log_likelihood(rho: float, table: np.ndarray,
                             tau_x: np.ndarray, tau_y: np.ndarray) -> float:
    """Multinomial ``-logL`` of the table at one value of ``rho``."""
    probs = _cell_probabilities(tau_x, tau_y, float(rho))
    used = table > 0
    if not np.any(used):
        return 0.0
    logs = np.log(np.maximum(probs[used], _LOG_FLOOR))
    return float(-np.sum(table[used] * logs))


# --------------------------------------------------------------------------- #
# Public estimators
# --------------------------------------------------------------------------- #


def polychoric_correlation(x: np.ndarray, y: np.ndarray, *,
                           max_categories: int = 11) -> float:
    """Two-step maximum-likelihood polychoric correlation of two ordinal items.

    The correlation of the two *latent* normal variables assumed to underlie
    the observed categories.  Thresholds come from the univariate marginals
    (step one) and ``rho`` from a one-dimensional maximisation of the
    multinomial likelihood of the contingency table (step two).

    Parameters
    ----------
    x : numpy.ndarray
        First ordinal variable, one entry per case.  ``NaN`` marks a missing
        value; pairs with a missing value in either variable are dropped.
    y : numpy.ndarray
        Second ordinal variable, the same length as ``x``.
    max_categories : int, default 11
        Reject a variable with more distinct non-contiguous values than this,
        on the grounds that it is continuous rather than ordinal.

    Returns
    -------
    float
        The polychoric correlation, in ``[-0.999, 0.999]``.  Exactly ``1.0``
        when ``x`` and ``y`` are element-wise identical.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the two variables have different lengths, share no complete case,
        have no variance, or have implausibly many categories.

    Notes
    -----
    The estimator is consistent but, unlike full-information maximum
    likelihood, it does not re-estimate the thresholds jointly with ``rho``;
    Olsson (1979) shows the efficiency loss is negligible, which is why every
    production package ships the two-step version.

    Two dichotomous variables make this the **tetrachoric** correlation, and
    the same code path handles it.

    Examples
    --------
    With 100 000 bivariate normals correlated ``0.60`` and both cut into five
    symmetric categories, this returns ``0.600`` while Pearson's ``r`` on the
    integers returns about ``0.56``.

    References
    ----------
    Olsson, U. (1979). Maximum likelihood estimation of the polychoric
        correlation coefficient. *Psychometrika*, 44(4), 443-460.
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.size != y_arr.size:
        raise SpecificationError(
            f"The two variables must have the same number of cases; got "
            f"{x_arr.size} and {y_arr.size}.",
            f"يجب أن يكون للمتغيّرين العدد نفسه من الحالات، وقد ورد "
            f"{x_arr.size} مقابل {y_arr.size}.",
            n_x=int(x_arr.size), n_y=int(y_arr.size),
        )
    if x_arr.size and np.array_equal(x_arr, y_arr, equal_nan=True):
        return 1.0

    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 2:
        raise SpecificationError(
            "Fewer than two cases are complete on both variables, so no "
            "correlation can be estimated.",
            "عدد الحالات المكتملة في المتغيّرين أقل من حالتين، ولا يمكن تقدير "
            "معامل الارتباط.",
            n_complete=int(x_arr.size),
        )

    codes_x, counts_x = _codes_and_counts(x_arr, max_categories, "x")
    codes_y, counts_y = _codes_and_counts(y_arr, max_categories, "y")
    n_x = counts_x.size
    n_y = counts_y.size
    for name, counts in (("x", counts_x), ("y", counts_y)):
        if int(np.count_nonzero(counts)) < 2:
            raise SpecificationError(
                f"Variable {name!r} is constant: every case falls in one "
                f"category, so it has no variance to share.",
                f"المتغيّر {name!r} ثابت؛ إذ تتركّز جميع الحالات في فئة واحدة، "
                f"ولا يوجد لديه تباين يمكن أن يرتبط بغيره.",
                variable=name,
            )

    table = np.bincount(codes_x * n_y + codes_y,
                        minlength=n_x * n_y).astype(float).reshape(n_x, n_y)
    tau_x = thresholds_from_counts(counts_x)
    tau_y = thresholds_from_counts(counts_y)

    try:
        result = optimize.minimize_scalar(
            _negative_log_likelihood,
            args=(table, tau_x, tau_y),
            bounds=(-MAX_ABS_RHO, MAX_ABS_RHO),
            method="bounded",
            options={"xatol": 1e-8},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise SpecificationError(
            "The polychoric likelihood could not be maximised for this pair "
            "of items; check for empty or degenerate response categories.",
            "لم يتمكّن البرنامج من تعظيم دالة الإمكان لحساب الارتباط متعدد "
            "التصنيفات لهذين البندين؛ يُرجى التحقّق من الفئات الفارغة أو "
            "المتدهورة.",
            detail=str(exc),
        ) from exc

    rho = float(np.clip(result.x, -MAX_ABS_RHO, MAX_ABS_RHO))
    if not np.isfinite(rho):   # pragma: no cover - defensive
        raise SpecificationError(
            "The polychoric estimate is not a finite number.",
            "قيمة الارتباط متعدد التصنيفات المقدّرة ليست عددًا منتهيًا.",
        )
    return rho


def polyserial_correlation(ordinal: np.ndarray,
                           continuous: np.ndarray) -> float:
    """Two-step polyserial correlation between an ordinal and a continuous item.

    Estimates the correlation between the continuous variable and the latent
    normal variable underlying the ordinal one.  The two-step ("ad hoc")
    estimator of Olsson, Drasgow and Dorans (1982) is used::

        r_ps = r_xy * s_x / sum_i phi(tau_i)

    where ``r_xy`` is Pearson's correlation between the integer scores and the
    continuous variable, ``s_x`` the standard deviation of those scores
    (divisor ``n``), ``tau_i`` the thresholds from the ordinal marginal and
    ``phi`` the standard normal density.  The denominator is exactly the
    attenuation that categorising a normal variable causes, so dividing it out
    restores the underlying correlation.

    Parameters
    ----------
    ordinal : numpy.ndarray
        Ordinal scores, one per case; ``NaN`` marks missing.
    continuous : numpy.ndarray
        Continuous variable, the same length as ``ordinal``.

    Returns
    -------
    float
        The polyserial correlation, clipped to ``[-0.999, 0.999]``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the lengths differ, fewer than two cases are complete, or either
        variable has no variance.

    Notes
    -----
    Because the estimator divides a Pearson correlation by a factor smaller
    than one, ``|r_ps| >= |r_xy|`` always: the polyserial correlation is never
    closer to zero than the Pearson correlation it corrects.

    References
    ----------
    Olsson, U., Drasgow, F., & Dorans, N. J. (1982). The polyserial
        correlation coefficient. *Psychometrika*, 47(3), 337-347.
    Cox, N. R. (1974). Estimation of the correlation between a continuous and
        a discrete variable. *Biometrics*, 30(1), 171-178.
    """
    ord_arr = np.asarray(ordinal, dtype=float).ravel()
    con_arr = np.asarray(continuous, dtype=float).ravel()
    if ord_arr.size != con_arr.size:
        raise SpecificationError(
            f"The two variables must have the same number of cases; got "
            f"{ord_arr.size} and {con_arr.size}.",
            f"يجب أن يكون للمتغيّرين العدد نفسه من الحالات، وقد ورد "
            f"{ord_arr.size} مقابل {con_arr.size}.",
        )
    keep = np.isfinite(ord_arr) & np.isfinite(con_arr)
    ord_arr = ord_arr[keep]
    con_arr = con_arr[keep]
    if ord_arr.size < 2:
        raise SpecificationError(
            "Fewer than two cases are complete on both variables, so no "
            "correlation can be estimated.",
            "عدد الحالات المكتملة في المتغيّرين أقل من حالتين، ولا يمكن تقدير "
            "معامل الارتباط.",
        )

    sd_ord = float(np.std(ord_arr))
    sd_con = float(np.std(con_arr))
    if sd_ord <= 0 or sd_con <= 0:
        raise SpecificationError(
            "One of the two variables is constant, so it cannot correlate "
            "with anything.",
            "أحد المتغيّرين ثابت لا يتغيّر، ولذلك لا يمكن أن يرتبط بأي متغيّر "
            "آخر.",
        )

    pearson = float(np.corrcoef(ord_arr, con_arr)[0, 1])
    _, counts = _codes_and_counts(ord_arr, _POLYSERIAL_MAX_CATEGORIES,
                                  "ordinal")
    tau = thresholds_from_counts(counts)
    density_sum = float(np.sum(stats.norm.pdf(tau)))
    if density_sum <= 1e-12:
        raise SpecificationError(
            "The ordinal variable's thresholds are degenerate, so the "
            "polyserial correction is undefined.",
            "الحدود الفاصلة للمتغيّر الرتبي متدهورة، ولذلك يتعذّر حساب تصحيح "
            "الارتباط متعدد الرتب.",
        )
    value = pearson * sd_ord / density_sum
    return float(np.clip(value, -MAX_ABS_RHO, MAX_ABS_RHO))


def polychoric_matrix(data: pd.DataFrame,
                      ordinal_columns: Optional[Iterable[str]] = None,
                      *, repair_pd: bool = True) -> pd.DataFrame:
    """Pairwise polychoric / polyserial / Pearson correlation matrix.

    Every pair of ordinal columns is correlated with
    :func:`polychoric_correlation`, an ordinal column against a continuous one
    with :func:`polyserial_correlation`, and two continuous columns with
    Pearson's ``r`` -- the mixed-measurement matrix that Muthen's (1984)
    categorical-variable methodology asks for.

    Parameters
    ----------
    data : pandas.DataFrame
        Items in columns, cases in rows.  A two-dimensional array-like is
        accepted and converted.
    ordinal_columns : iterable of str, optional
        Which columns to treat as ordinal.  ``None`` (the default) means
        *all* of them.
    repair_pd : bool, default True
        A pairwise-estimated matrix need not be positive definite.  When
        ``True`` a non-PD result is replaced by the nearest positive definite
        matrix (Higham, 2002, via
        :func:`semsim.core.implied.nearest_positive_definite`) and rescaled
        back to a unit diagonal.

    Returns
    -------
    pandas.DataFrame
        Symmetric correlation matrix with a unit diagonal, indexed and
        columned by the input column names.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the data cannot be read as a table, has fewer than two columns, or
        a column is unusable (constant, or too many categories to be ordinal).

    Notes
    -----
    Cost is one bounded scalar optimisation per pair, so ``p`` items need
    ``p(p-1)/2`` of them: about 1.5 s for 60 items, which is why the caller
    should estimate the matrix once and pass it around.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with dichotomous,
        ordered categorical, and continuous latent variable indicators.
        *Psychometrika*, 49(1), 115-132.
    Higham, N. J. (2002). Computing the nearest correlation matrix -- a
        problem from finance. *IMA Journal of Numerical Analysis*, 22(3),
        329-343.
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

    frame = frame.apply(pd.to_numeric, errors="coerce")
    names = [str(c) for c in frame.columns]
    p = len(names)
    if p < 2:
        raise SpecificationError(
            f"A correlation matrix needs at least two items; got {p}.",
            f"مصفوفة الارتباط تتطلّب بندين على الأقل، وقد ورد {p} بند.",
            n_items=p,
        )

    if ordinal_columns is None:
        is_ordinal = {name: True for name in names}
    else:
        wanted = {str(c) for c in ordinal_columns}
        is_ordinal = {name: name in wanted for name in names}

    values = frame.to_numpy(dtype=float)
    out = np.eye(p, dtype=float)
    for i in range(p):
        for j in range(i + 1, p):
            xi = values[:, i]
            xj = values[:, j]
            if is_ordinal[names[i]] and is_ordinal[names[j]]:
                r = polychoric_correlation(xi, xj)
            elif is_ordinal[names[i]]:
                r = polyserial_correlation(xi, xj)
            elif is_ordinal[names[j]]:
                r = polyserial_correlation(xj, xi)
            else:
                keep = np.isfinite(xi) & np.isfinite(xj)
                if keep.sum() < 2:
                    raise SpecificationError(
                        f"Items {names[i]!r} and {names[j]!r} share fewer "
                        f"than two complete cases.",
                        f"البندان {names[i]!r} و {names[j]!r} لا يشتركان في "
                        f"حالتين مكتملتين على الأقل.",
                    )
                r = float(np.corrcoef(xi[keep], xj[keep])[0, 1])
            out[i, j] = out[j, i] = float(np.clip(r, -MAX_ABS_RHO, MAX_ABS_RHO))

    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 1.0)
    if repair_pd:
        out = _as_positive_definite_correlation(out)
    return pd.DataFrame(out, index=names, columns=names)


def _as_positive_definite_correlation(matrix: np.ndarray) -> np.ndarray:
    """Repair a pairwise correlation matrix to be PD with a unit diagonal."""
    from semsim.core.implied import is_positive_definite, nearest_positive_definite

    if is_positive_definite(matrix):
        return matrix
    repaired = nearest_positive_definite(matrix)
    scale = np.sqrt(np.clip(np.diag(repaired), 1e-12, None))
    repaired = repaired / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired

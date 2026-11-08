"""
semsim.generation.vale_maurelli
===============================
Multivariate non-normal data with exact population means, standard deviations
and correlations, and prescribed marginal skewness / excess kurtosis.

The method is Vale & Maurelli's (1983) extension of Fleishman's (1978)
univariate power method:

1.  Solve Fleishman's three moment equations once per variable, giving
    coefficients ``(a, b, c, d)`` with ``E[Y] = 0`` and ``Var(Y) = 1``.
2.  For every pair ``(i, j)`` invert the cubic that links the correlation of
    the *normal* stage to the correlation after transformation, giving an
    intermediate correlation matrix ``R_z``.
3.  Draw correlated standard normals through the Cholesky factor of ``R_z``.
4.  Apply each variable's polynomial, then scale by the target standard
    deviations and add the target means.

Because step 2 is a pairwise inversion, ``R_z`` can lose positive
definiteness even when the target correlation matrix is perfectly fine.  That
is a known property of the method, not a user error, so the matrix is repaired
with Higham's (2002) nearest-positive-definite projection and a warning is
issued rather than an exception raised.

References
----------
Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
    distributions. *Psychometrika*, 48(3), 465-471.
Fleishman, A. I. (1978). A method for simulating non-normal distributions.
    *Psychometrika*, 43(4), 521-532.
Higham, N. J. (2002). Computing the nearest correlation matrix -- a problem
    from finance. *IMA Journal of Numerical Analysis*, 22(3), 329-343.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from semsim.core.errors import InfeasibleTargetError, SpecificationError
from semsim.generation.fleishman import (
    FleishmanCoefficients,
    intermediate_correlation,
    solve as solve_fleishman,
    transform,
)

__all__ = [
    "generate",
    "generate_normal",
    "achieved_moments",
    "fleishman_coefficients",
    "intermediate_correlation_matrix",
]


# --------------------------------------------------------------------------- #
# Linear-algebra helpers (self-contained; core.implied is used when present)
# --------------------------------------------------------------------------- #


def _repair_to_pd(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Nearest positive-definite matrix, preferring ``core.implied``'s Higham
    implementation and falling back on eigenvalue clipping."""
    try:  # the core module owns the canonical implementation
        from semsim.core.implied import nearest_positive_definite
    except Exception:  # pragma: no cover - defensive, core.implied ships with it
        nearest_positive_definite = None  # type: ignore[assignment]
    if nearest_positive_definite is not None:
        try:
            return np.asarray(nearest_positive_definite(m, eps=eps), dtype=float)
        except Exception:  # pragma: no cover - fall through to the local repair
            pass
    a = 0.5 * (np.asarray(m, dtype=float) + np.asarray(m, dtype=float).T)
    vals, vecs = np.linalg.eigh(a)
    a = (vecs * np.clip(vals, eps, None)) @ vecs.T
    a = 0.5 * (a + a.T)
    scale = np.sqrt(np.clip(np.diag(a), eps, None))
    a = a / np.outer(scale, scale)
    np.fill_diagonal(a, 1.0)
    return a


def _is_pd(m: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(np.asarray(m, dtype=float))
    except np.linalg.LinAlgError:
        return False
    return True


def _cholesky_lower(m: np.ndarray, *, what: str, what_ar: str) -> np.ndarray:
    """Cholesky factor, repairing once and raising a user-facing error after."""
    try:
        return np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        pass
    repaired = _repair_to_pd(m)
    try:
        return np.linalg.cholesky(repaired)
    except np.linalg.LinAlgError as exc:
        raise SpecificationError(
            f"The {what} is not positive definite and could not be repaired, so "
            "no data can be generated from it. Some of the requested "
            "correlations are mutually inconsistent.",
            f"{what_ar} ليست معرّفة موجبة وتعذّر إصلاحها، ولذلك لا يمكن توليد "
            "بيانات منها. بعض الارتباطات المطلوبة غير متّسقة مع بعضها؛ "
            "راجع قيم الارتباطات أو خفّض أكبرها.",
        ) from exc


def _validate_sigma(sigma: np.ndarray) -> np.ndarray:
    """Return ``sigma`` as a validated, symmetrised float array."""
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise SpecificationError(
            "The covariance matrix must be square and non-empty; got shape "
            f"{tuple(arr.shape)}.",
            "يجب أن تكون مصفوفة التغاير مربّعة وغير فارغة؛ "
            f"أبعاد المصفوفة الممرَّرة هي {tuple(arr.shape)}.",
            shape=tuple(arr.shape),
        )
    if not np.all(np.isfinite(arr)):
        raise SpecificationError(
            "The covariance matrix contains non-finite entries (NaN or inf).",
            "مصفوفة التغاير تحتوي على قيم غير منتهية (NaN أو ما لا نهاية).",
        )
    asym = float(np.max(np.abs(arr - arr.T))) if arr.shape[0] > 1 else 0.0
    if asym > 1e-8 * max(1.0, float(np.max(np.abs(arr)))):
        raise SpecificationError(
            f"The covariance matrix is not symmetric (largest asymmetry "
            f"{asym:.3e}).",
            f"مصفوفة التغاير غير متماثلة (أكبر فرق بين عنصر ونظيره "
            f"{asym:.3e}). يجب أن تكون المصفوفة متماثلة حول قطرها.",
            asymmetry=asym,
        )
    diag = np.diag(arr)
    if np.any(diag <= 0.0):
        bad = int(np.argmin(diag))
        raise SpecificationError(
            f"Every variance on the diagonal of the covariance matrix must be "
            f"positive; entry {bad} is {diag[bad]:.6g}.",
            "يجب أن يكون تباين كل متغيّر على قطر مصفوفة التغاير أكبر من الصفر؛ "
            f"العنصر رقم {bad} يساوي {diag[bad]:.6g}.",
            index=bad, value=float(diag[bad]),
        )
    return 0.5 * (arr + arr.T)


def _validate_n(n: int) -> int:
    try:
        n_int = int(n)
    except (TypeError, ValueError) as exc:
        raise SpecificationError(
            f"The sample size must be an integer; got {n!r}.",
            f"يجب أن يكون حجم العيّنة عددًا صحيحًا؛ القيمة الممرَّرة هي {n!r}.",
        ) from exc
    if n_int <= 0:
        raise SpecificationError(
            f"The sample size must be greater than zero; got {n_int}.",
            f"يجب أن يكون حجم العيّنة أكبر من الصفر؛ القيمة الممرَّرة هي {n_int}.",
            n=n_int,
        )
    return n_int


def _validate_rng(rng: np.random.Generator) -> np.random.Generator:
    if not hasattr(rng, "standard_normal"):
        raise SpecificationError(
            "A numpy.random.Generator is required; reproducibility depends on "
            f"an explicit generator, but got {type(rng).__name__}.",
            "يجب تمرير مولّد أرقام عشوائية من النوع numpy.random.Generator، "
            "لأن إمكانية إعادة إنتاج البيانات تعتمد على مولّد صريح؛ "
            f"النوع الممرَّر هو {type(rng).__name__}.",
        )
    return rng


def _validate_moment_vector(values: Sequence[float], p: int, name_en: str,
                            name_ar: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size != p:
        raise SpecificationError(
            f"The number of {name_en} values ({arr.size}) does not match the "
            f"number of variables in the covariance matrix ({p}).",
            f"عدد قيم {name_ar} ({arr.size}) لا يساوي عدد المتغيّرات في مصفوفة "
            f"التغاير ({p}).",
            n_values=int(arr.size), n_variables=int(p),
        )
    if not np.all(np.isfinite(arr)):
        raise SpecificationError(
            f"The {name_en} values must all be finite.",
            f"يجب أن تكون جميع قيم {name_ar} أعدادًا منتهية.",
        )
    return arr


def _validate_mean(mean: Optional[Sequence[float]], p: int) -> np.ndarray:
    if mean is None:
        return np.zeros(p, dtype=float)
    arr = np.asarray(list(np.atleast_1d(mean)), dtype=float).ravel()
    if arr.size == 1 and p > 1:
        arr = np.repeat(arr, p)
    if arr.size != p:
        raise SpecificationError(
            f"The mean vector has {arr.size} entries but the covariance matrix "
            f"describes {p} variables.",
            f"متّجه المتوسطات يحتوي على {arr.size} قيمة بينما تصف مصفوفة "
            f"التغاير {p} متغيّرًا.",
            n_values=int(arr.size), n_variables=int(p),
        )
    if not np.all(np.isfinite(arr)):
        raise SpecificationError(
            "The mean vector contains non-finite entries.",
            "متّجه المتوسطات يحتوي على قيم غير منتهية.",
        )
    return arr


# --------------------------------------------------------------------------- #
# Fleishman coefficients per variable
# --------------------------------------------------------------------------- #


def fleishman_coefficients(skewness: Sequence[float], kurtosis: Sequence[float],
                           *, clamp: bool = False
                           ) -> List[FleishmanCoefficients]:
    """Solve Fleishman's system once per variable, caching repeated targets.

    Parameters
    ----------
    skewness : sequence of float
        Target skewness per variable.
    kurtosis : sequence of float
        Target *excess* kurtosis per variable.
    clamp : bool, default False
        Pass ``True`` to pull infeasible targets just inside the feasible
        region instead of raising.

    Returns
    -------
    list of FleishmanCoefficients
        One entry per variable, in the order given.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        Propagated from :func:`semsim.generation.fleishman.solve` when a
        target is outside the feasible region.

    Notes
    -----
    Questionnaire items very often share the same non-normality targets, so the
    solutions are memoised on the exact ``(skewness, kurtosis)`` pair; a 60-item
    dataset with three distinct shapes costs three Newton solves.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    skew = np.asarray(list(skewness), dtype=float).ravel()
    kurt = np.asarray(list(kurtosis), dtype=float).ravel()
    if skew.size != kurt.size:
        raise SpecificationError(
            f"Got {skew.size} skewness values but {kurt.size} kurtosis values.",
            f"عدد قيم الالتواء ({skew.size}) لا يساوي عدد قيم التفلطح "
            f"({kurt.size}).",
        )
    cache: Dict[Tuple[float, float], FleishmanCoefficients] = {}
    out: List[FleishmanCoefficients] = []
    for s, k in zip(skew, kurt):
        key = (float(s), float(k))
        coef = cache.get(key)
        if coef is None:
            coef = solve_fleishman(key[0], key[1], clamp=clamp)
            cache[key] = coef
        out.append(coef)
    return out


def intermediate_correlation_matrix(target_correlation: np.ndarray,
                                    coefficients: Sequence[FleishmanCoefficients],
                                    ) -> np.ndarray:
    """Pairwise intermediate (normal-stage) correlation matrix.

    Parameters
    ----------
    target_correlation : numpy.ndarray
        ``(p, p)`` target correlation matrix of the *transformed* variables,
        with a unit diagonal.
    coefficients : sequence of FleishmanCoefficients
        Length-``p`` Fleishman solutions, aligned with the matrix.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` symmetric matrix with a unit diagonal whose off-diagonal
        entries solve Vale & Maurelli's cubic.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        If some requested correlation is unattainable for the corresponding
        pair of marginal distributions.

    References
    ----------
    Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
        distributions. *Psychometrika*, 48(3), 465-471.
    """
    r = np.asarray(target_correlation, dtype=float)
    p = r.shape[0]
    if len(coefficients) != p:
        raise SpecificationError(
            f"Got {len(coefficients)} Fleishman solutions for a {p}x{p} "
            "correlation matrix.",
            f"عدد حلول فليشمان ({len(coefficients)}) لا يساوي حجم مصفوفة "
            f"الارتباط ({p}×{p}).",
        )
    rz = np.eye(p, dtype=float)
    for i in range(p):
        for j in range(i + 1, p):
            rho = float(r[i, j])
            try:
                value = intermediate_correlation(rho, coefficients[i],
                                                 coefficients[j])
            except InfeasibleTargetError as exc:
                raise InfeasibleTargetError(
                    f"Variables {i} and {j}: {exc.message_en}",
                    f"المتغيّران {i} و {j}: {exc.message_ar}",
                    i=i, j=j, target_rho=rho,
                ) from exc
            rz[i, j] = rz[j, i] = value
    return rz


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate(n: int, sigma: np.ndarray, skewness: Sequence[float],
             kurtosis: Sequence[float], rng: np.random.Generator,
             *, repair_pd: bool = True,
             mean: Optional[Sequence[float]] = None,
             clamp_to_feasible: bool = False) -> np.ndarray:
    """Draw ``n`` observations with the given covariance and marginal shapes.

    Parameters
    ----------
    n : int
        Number of observations (rows).
    sigma : numpy.ndarray
        ``(p, p)`` target covariance matrix.  Its diagonal supplies the target
        variances; the implied correlation matrix supplies the associations.
    skewness : sequence of float
        Target skewness per variable, length ``p``.
    kurtosis : sequence of float
        Target *excess* kurtosis per variable, length ``p``.
    rng : numpy.random.Generator
        The only source of randomness; the same generator state always
        reproduces the same dataset.
    repair_pd : bool, default True
        When the pairwise intermediate matrix is not positive definite, repair
        it with Higham's projection and emit a :class:`RuntimeWarning`.  With
        ``False`` the situation raises instead.
    mean : sequence of float, optional
        Target means; defaults to zeros.
    clamp_to_feasible : bool, default False
        Pull infeasible ``(skewness, kurtosis)`` pairs just inside the
        Fleishman region instead of raising.

    Returns
    -------
    numpy.ndarray
        ``(n, p)`` array of observations.

    Raises
    ------
    semsim.core.errors.SpecificationError
        Malformed inputs (non-square or asymmetric ``sigma``, non-positive
        variances, length mismatches, non-positive ``n``), or an intermediate
        matrix that cannot be factorised even after repair.
    semsim.core.errors.InfeasibleTargetError
        A marginal target outside Fleishman's region, or a correlation
        unattainable for a pair of marginals.

    Notes
    -----
    Everything after the Fleishman solves is vectorised: one
    ``(n, p)`` normal draw, one matrix product with the Cholesky factor and one
    Horner evaluation per column.  A 500x60 dataset costs milliseconds.

    The population moments are exact in expectation, not in the sample: the
    columns are *not* re-standardised empirically, because doing so would
    destroy the sampling variability that makes the dataset behave like real
    data in a subsequent SEM fit.

    References
    ----------
    Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
        distributions. *Psychometrika*, 48(3), 465-471.
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    n_rows = _validate_n(n)
    cov = _validate_sigma(sigma)
    _validate_rng(rng)
    p = cov.shape[0]
    skew = _validate_moment_vector(skewness, p, "skewness", "الالتواء")
    kurt = _validate_moment_vector(kurtosis, p, "kurtosis", "التفلطح")
    mu = _validate_mean(mean, p)

    sd = np.sqrt(np.diag(cov))
    target_r = cov / np.outer(sd, sd)
    np.fill_diagonal(target_r, 1.0)
    target_r = np.clip(0.5 * (target_r + target_r.T), -1.0, 1.0)
    np.fill_diagonal(target_r, 1.0)

    coefs = fleishman_coefficients(skew, kurt, clamp=clamp_to_feasible)
    all_normal = all(c.is_identity for c in coefs)

    if all_normal:
        # Nothing to invert: the cubic is the identity for normal marginals.
        rz = target_r.copy()
    else:
        rz = intermediate_correlation_matrix(target_r, coefs)

    if not _is_pd(rz):
        if not repair_pd:
            raise SpecificationError(
                "The intermediate correlation matrix required by Vale & "
                "Maurelli's method is not positive definite, and repair was "
                "disabled (repair_pd=False).",
                "مصفوفة الارتباط الوسيطة التي تتطلّبها طريقة فيل وموريلي ليست "
                "معرّفة موجبة، وقد تم تعطيل الإصلاح (repair_pd=False). "
                "فعّل الإصلاح أو خفّض الارتباطات المطلوبة.",
            )
        before = rz
        rz = _repair_to_pd(rz)
        shift = float(np.max(np.abs(rz - before)))
        warnings.warn(
            "The intermediate correlation matrix was not positive definite and "
            f"has been repaired (largest entry moved by {shift:.4g}); the "
            "achieved correlations will deviate slightly from the targets. "
            "[AR] تم إصلاح مصفوفة الارتباط الوسيطة لأنها لم تكن معرّفة موجبة "
            f"(أكبر تغيّر في عنصر {shift:.4g})؛ ستختلف الارتباطات المتحقّقة "
            "اختلافًا طفيفًا عن المستهدفة.",
            RuntimeWarning,
            stacklevel=2,
        )

    chol = _cholesky_lower(
        rz,
        what="intermediate correlation matrix",
        what_ar="مصفوفة الارتباط الوسيطة",
    )

    z = rng.standard_normal((n_rows, p)) @ chol.T

    out = np.empty((n_rows, p), dtype=float)
    for j, coef in enumerate(coefs):
        out[:, j] = transform(z[:, j], coef)

    return out * sd + mu


def generate_normal(n: int, sigma: np.ndarray, rng: np.random.Generator,
                    mean: Optional[Sequence[float]] = None) -> np.ndarray:
    """Draw ``n`` multivariate normal observations with covariance ``sigma``.

    The fast path used whenever every marginal is normal, and the reference
    implementation the non-normal engine is checked against.

    Parameters
    ----------
    n : int
        Number of observations.
    sigma : numpy.ndarray
        ``(p, p)`` target covariance matrix; repaired to positive definite if
        necessary (with a warning).
    rng : numpy.random.Generator
        The only source of randomness.
    mean : sequence of float, optional
        Target means; defaults to zeros.

    Returns
    -------
    numpy.ndarray
        ``(n, p)`` array of observations.

    Raises
    ------
    semsim.core.errors.SpecificationError
        Malformed ``sigma`` / ``mean`` / ``n``, or a matrix that cannot be
        factorised even after repair.

    References
    ----------
    Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*.
        Scientific Software International.
    """
    n_rows = _validate_n(n)
    cov = _validate_sigma(sigma)
    _validate_rng(rng)
    p = cov.shape[0]
    mu = _validate_mean(mean, p)

    if not _is_pd(cov):
        repaired = _repair_to_pd(cov)
        shift = float(np.max(np.abs(repaired - cov)))
        warnings.warn(
            "The covariance matrix was not positive definite and has been "
            f"repaired (largest entry moved by {shift:.4g}). "
            "[AR] تم إصلاح مصفوفة التغاير لأنها لم تكن معرّفة موجبة "
            f"(أكبر تغيّر في عنصر {shift:.4g}).",
            RuntimeWarning,
            stacklevel=2,
        )
        cov = repaired

    chol = _cholesky_lower(cov, what="covariance matrix",
                           what_ar="مصفوفة التغاير")
    return rng.standard_normal((n_rows, p)) @ chol.T + mu


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def achieved_moments(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Column-wise mean, standard deviation, skewness and excess kurtosis.

    Uses the biased (method-of-moments) estimators ``g1 = m3 / m2**1.5`` and
    ``g2 = m4 / m2**2 - 3``, which are the quantities Fleishman's equations
    target, so the returned numbers are directly comparable with the
    requested ones.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n, p)`` data matrix; a 1-D array is treated as a single column.

    Returns
    -------
    dict of str to numpy.ndarray
        Keys ``"mean"``, ``"sd"``, ``"skewness"`` and ``"kurtosis"`` (excess),
        plus the aliases ``"skew"`` and ``"excess_kurtosis"``.  ``"sd"`` is the
        population (``ddof=0``) standard deviation.

        A column that is exactly constant has no shape to report, so its
        skewness and excess kurtosis are ``0.0``.  A column containing
        ``nan`` -- what :func:`semsim.generation.noise.inject_missing` leaves
        behind -- yields ``nan`` for all four statistics: these are
        complete-data moments, and the function will not pretend to have
        measured a column it could not read.  Drop or impute the missing cells
        first (``x[~np.isnan(x).any(axis=1)]``) if you want moments from the
        observed part.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``x`` is empty or not two-dimensional after promotion.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    Joanes, D. N., & Gill, C. A. (1998). Comparing measures of sample skewness
        and kurtosis. *Journal of the Royal Statistical Society: Series D*,
        47(1), 183-189.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.size == 0:
        raise SpecificationError(
            "The data matrix must be a non-empty 2-D array of observations by "
            f"variables; got shape {tuple(arr.shape)}.",
            "يجب أن تكون مصفوفة البيانات ثنائية الأبعاد وغير فارغة (مشاهدات × "
            f"متغيّرات)؛ أبعاد المصفوفة الممرَّرة هي {tuple(arr.shape)}.",
            shape=tuple(arr.shape),
        )

    mean = arr.mean(axis=0)
    centred = arr - mean
    m2 = np.mean(centred ** 2, axis=0)
    m3 = np.mean(centred ** 3, axis=0)
    m4 = np.mean(centred ** 4, axis=0)
    sd = np.sqrt(m2)

    # A column with no variance has no shape, so its skewness and excess
    # kurtosis are reported as 0.0 rather than 0/0.  The test has to be
    # ``m2 == 0.0`` and not the old ``m2 > 0.0``: ``np.where(m2 > 0.0, ..., 0.0)``
    # takes the *else* branch for a NaN ``m2`` as well, because every
    # comparison with NaN is False.  A column holding a single ``nan`` -- which
    # is exactly what :func:`semsim.generation.noise.inject_missing` produces,
    # and therefore what the engine's own output looks like whenever
    # missingness is switched on -- then came back as "skewness 0.0, excess
    # kurtosis 0.0", i.e. indistinguishable from a perfectly normal column,
    # while ``mean`` and ``sd`` for the same column correctly read ``nan``.
    # That is the worst kind of wrong: a diagnostic silently reporting a clean
    # bill of health for data it could not actually measure.  Writing the mask
    # the other way round lets NaN fall through the arithmetic and propagate,
    # so a caller sees ``nan`` and knows to drop or impute first.
    degenerate = m2 == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(degenerate, 0.0, m3 / np.power(m2, 1.5))
        kurt = np.where(degenerate, 0.0, m4 / (m2 * m2) - 3.0)

    return {
        "mean": mean,
        "sd": sd,
        "skewness": skew,
        "kurtosis": kurt,
        "skew": skew,
        "excess_kurtosis": kurt,
    }

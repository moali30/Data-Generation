"""
Numerical tests for :mod:`semsim.analysis.fit_indices` and
:mod:`semsim.analysis.descriptives`.

Nothing here checks "the code ran".  Every assertion is against a closed form,
an independent recomputation, or a number a researcher can look up:

*   a perfectly reproducing solution must give ``chi2 == 0``, ``RMSEA == 0``,
    ``CFI == 1``, ``TLI == 1`` and ``SRMR == 0`` **exactly**;
*   for two compound-symmetric matrices the eigenvalues of ``Sigma^-1 S`` are
    known in closed form, so ``F_ML``, ``chi2``, RMSEA, CFI, TLI, GFI, SRMR,
    AIC, BIC and SABIC can all be written out by hand -- the arithmetic is
    spelled out in the comments beside each assertion;
*   the RMSEA confidence interval must bracket the point estimate and must get
    narrower as ``N`` grows;
*   Mardia's test must not reject 20000 draws from a multivariate normal and
    must reject a strongly skewed sample;
*   ``SE_skew`` at ``n = 100`` must equal the 0.2414 that SPSS prints;
*   ``G1`` / ``G2`` must match the SPSS estimators recomputed by hand;
*   data with ``NaN`` must be described without crashing, with ``n_missing``
    reported.

Run with::

    python3 -m pytest tests/test_fit_indices.py -q

The file also works as a plain script (``python3 tests/test_fit_indices.py``)
and then prints PASS/FAIL per check and exits non-zero on failure.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semsim.analysis import descriptives as desc
from semsim.analysis import fit_indices as fit
from semsim.core.compile import compile_model
from semsim.core.errors import EstimationError, SpecificationError
from semsim.core.examples import minimal_cfa, second_order_hierarchy, simple_mediation
from semsim.core.implied import implied_covariance
from semsim.core.spec import FitLevel

TOL = 1e-10

#: Lines printed by the numerical checks, echoed by the script runner and
#: shown by pytest with ``-s``.
OBSERVED: list[str] = []


def _note(message: str) -> None:
    OBSERVED.append(message)
    print(message)


# --------------------------------------------------------------------------- #
# Closed-form helpers used *instead of* the module under test
# --------------------------------------------------------------------------- #


def compound_symmetry(p: int, r: float) -> np.ndarray:
    """``(1 - r) I + r J``: unit diagonal, every off-diagonal equal to ``r``."""
    return (1.0 - r) * np.eye(p) + r * np.ones((p, p))


def cs_eigenvalues(p: int, r: float) -> list[float]:
    """Eigenvalues of :func:`compound_symmetry`: ``1 + (p-1)r`` once, ``1-r``
    ``p-1`` times."""
    return [1.0 + (p - 1) * r] + [1.0 - r] * (p - 1)


def cs_ml_discrepancy(p: int, r_s: float, r_m: float) -> float:
    """``F_ML`` between two compound-symmetric correlation matrices.

    Both matrices share eigenvectors (the constant vector and its orthogonal
    complement), so the eigenvalues of ``Sigma^-1 S`` are the ratios of the
    corresponding eigenvalues and ``F = sum_k (e_k - log e_k - 1)``.
    """
    ratios = [a / b for a, b in zip(cs_eigenvalues(p, r_s),
                                    cs_eigenvalues(p, r_m))]
    return sum(e - math.log(e) - 1.0 for e in ratios)


def cs_logdet(p: int, r: float) -> float:
    """``log|CS(p, r)| = (p-1) log(1-r) + log(1 + (p-1) r)``."""
    return (p - 1) * math.log(1.0 - r) + math.log(1.0 + (p - 1) * r)


def spss_skewness(values: np.ndarray) -> float:
    """``G1 = sqrt(n(n-1))/(n-2) * m3 / m2**1.5`` (the SPSS estimator)."""
    n = values.size
    d = values - values.mean()
    m2 = float(np.sum(d ** 2) / n)
    m3 = float(np.sum(d ** 3) / n)
    return math.sqrt(n * (n - 1.0)) / (n - 2.0) * m3 / m2 ** 1.5


def spss_excess_kurtosis(values: np.ndarray) -> float:
    """``G2 = (n-1)/((n-2)(n-3)) * ((n+1) g2 + 6)``, ``g2 = m4/m2^2 - 3``."""
    n = values.size
    d = values - values.mean()
    m2 = float(np.sum(d ** 2) / n)
    m4 = float(np.sum(d ** 4) / n)
    g2 = m4 / m2 ** 2 - 3.0
    return (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0)


def mardia_b1p_bruteforce(x: np.ndarray) -> float:
    """``(1/n^2) sum_ij d_ij^3`` built from the explicit ``n x n`` matrix.

    Deliberately the naive O(n^2 p) route, so it is an independent check on
    the tensor identity the module uses.
    """
    n = x.shape[0]
    centred = x - x.mean(axis=0)
    cov_ml = centred.T @ centred / n
    d = centred @ np.linalg.inv(cov_ml) @ centred.T
    return float(np.sum(d ** 3) / n ** 2)


# --------------------------------------------------------------------------- #
# 1. Perfect fit must be exact
# --------------------------------------------------------------------------- #


def test_perfect_fit_is_exactly_perfect() -> None:
    """``sample_cov == sigma_hat`` leaves no misfit to round off."""
    sigma = compound_symmetry(6, 0.45)
    result = fit.compute(sigma, sigma.copy(), n=300, n_free=13)

    assert result.chi_square == 0.0
    assert result.rmsea == 0.0
    assert result.cfi == 1.0
    assert result.tli == 1.0
    assert result.srmr == 0.0
    assert result.rmsea_ci_lower == 0.0
    assert result.gfi == 1.0
    assert result.agfi == 1.0
    assert result.nfi == 1.0
    assert result.df == 21 - 13 == 8
    # AIC = chi2 + 2q = 0 + 26; BIC = 0 + 13 log(300); SABIC = 0 + 13 log(302/24)
    assert abs(result.aic - 26.0) < TOL
    assert abs(result.bic - 13.0 * math.log(300)) < TOL
    assert abs(result.sabic - 13.0 * math.log(302.0 / 24.0)) < TOL
    assert result.verdict() is FitLevel.EXCELLENT
    _note(f"[perfect fit] chi2={result.chi_square!r} rmsea={result.rmsea!r} "
          f"cfi={result.cfi!r} tli={result.tli!r} srmr={result.srmr!r} "
          f"pclose={result.rmsea_pclose:.6f}")


def test_perfect_fit_of_every_compiled_example() -> None:
    """Scoring ``Sigma(theta)`` against itself is perfect for real models too."""
    for builder in (minimal_cfa, simple_mediation, second_order_hierarchy):
        spec = builder()
        cm = compile_model(spec, rng=np.random.default_rng(0))
        sigma = implied_covariance(cm)
        n_free = spec.count_free_parameters()
        result = fit.compute(sigma, sigma.copy(), n=500, n_free=n_free)
        assert result.chi_square == 0.0
        assert result.rmsea == 0.0
        assert result.cfi == 1.0
        assert result.srmr == 0.0
        assert result.df == spec.degrees_of_freedom()
        _note(f"[example {builder.__name__}] p={result.n_items} "
              f"df={result.df} chi2={result.chi_square!r}")
    # minimal_cfa is documented as df = 21 - 13 = 8
    assert minimal_cfa().degrees_of_freedom() == 8


# --------------------------------------------------------------------------- #
# 2. Hand-computed chi2, RMSEA, CFI, TLI
# --------------------------------------------------------------------------- #


def test_hand_computed_chi_square_rmsea_cfi_tli() -> None:
    """Three items, compound symmetry, N = 201, q = 2, so df = 6 - 2 = 4.

    S = CS(3, 0.5) and Sigma = CS(3, 0.3) commute, so the eigenvalues of
    Sigma^-1 S are the eigenvalue ratios:

        e1 = (1 + 2(0.5)) / (1 + 2(0.3)) = 2.0 / 1.6       = 1.25
        e2 = e3 = (1 - 0.5) / (1 - 0.3)  = 0.5 / 0.7       = 0.714285714...

        F_ML = (1.25 - ln 1.25 - 1) + 2 (5/7 - ln(5/7) - 1)
             = (0.25 - 0.223143551) + 2 (-0.285714286 + 0.336472237)
             = 0.026856449 + 0.101515902
             = 0.128372350
        chi2 = (N - 1) F = 200 (0.128372350) = 25.674470100
        df   = 3(4)/2 - 2 = 4
        RMSEA = sqrt((25.674470100 - 4) / (4 x 200))
              = sqrt(21.674470100 / 800) = sqrt(0.0270930876) = 0.1645997801

    Baseline: diag(S) = I, so F_b = sum log(s_ii) - log|S| = -log(0.5^2 x 2)
    = -log 0.5 = ln 2, chi2_b = 200 ln 2 = 138.629436112, df_b = 3(2)/2 = 3.

        CFI = 1 - (chi2 - df) / max(chi2_b - df_b, chi2 - df)
            = 1 - 21.674470100 / 135.629436112 = 0.840192...
        TLI = (chi2_b/df_b - chi2/df) / (chi2_b/df_b - 1)
            = (46.209812037 - 6.418617525) / 45.209812037 = 0.880147...
    """
    p, n, q = 3, 201, 2
    S = compound_symmetry(p, 0.5)
    Sigma = compound_symmetry(p, 0.3)

    f_expected = cs_ml_discrepancy(p, 0.5, 0.3)
    chi2_expected = (n - 1) * f_expected
    df_expected = p * (p + 1) // 2 - q
    rmsea_expected = math.sqrt((chi2_expected - df_expected)
                               / (df_expected * (n - 1)))
    chi2_b_expected = (n - 1) * (0.0 - cs_logdet(p, 0.5))
    df_b_expected = p * (p - 1) // 2
    cfi_expected = 1.0 - (chi2_expected - df_expected) / max(
        chi2_b_expected - df_b_expected, chi2_expected - df_expected, 0.0)
    ratio_b = chi2_b_expected / df_b_expected
    tli_expected = ((ratio_b - chi2_expected / df_expected) / (ratio_b - 1.0))

    result = fit.compute(S, Sigma, n=n, n_free=q)

    assert result.df == df_expected == 4
    assert abs(result.fmin - f_expected) < TOL
    assert abs(result.chi_square - chi2_expected) < TOL
    assert abs(result.chi_square - 25.674470099928) < 1e-9
    assert abs(result.rmsea - rmsea_expected) < TOL
    assert abs(result.rmsea - 0.1645997801) < 1e-9
    assert abs(result.baseline_chi_square - chi2_b_expected) < TOL
    assert abs(result.baseline_chi_square - 200.0 * math.log(2.0)) < TOL
    assert result.baseline_df == df_b_expected == 3
    assert abs(result.cfi - cfi_expected) < TOL
    assert abs(result.tli - tli_expected) < TOL
    assert abs(result.chi_square_df_ratio - chi2_expected / 4.0) < TOL

    # NFI, IFI, RFI from the same four numbers.
    assert abs(result.nfi - (chi2_b_expected - chi2_expected) / chi2_b_expected) < TOL
    assert abs(result.ifi - (chi2_b_expected - chi2_expected)
               / (chi2_b_expected - df_expected)) < TOL
    assert abs(result.rfi - (1.0 - (chi2_expected / df_expected) / ratio_b)) < TOL

    # SRMR: the standardised residual is 0 on the diagonal and 0.2 off it, and
    # the lower triangle including the diagonal holds 3 zeros and 3 twenties:
    #   sqrt(3 (0.2^2) / 6) = sqrt(0.02) = 0.141421356...
    assert abs(result.srmr - math.sqrt(0.02)) < TOL

    # GFI from the eigenvalues: 1 - sum (e-1)^2 / sum e^2
    eigs = [1.25, 0.5 / 0.7, 0.5 / 0.7]
    gfi_expected = 1.0 - sum((e - 1.0) ** 2 for e in eigs) / sum(
        e ** 2 for e in eigs)
    assert abs(result.gfi - gfi_expected) < TOL
    assert abs(result.agfi - (1.0 - (6.0 / 4.0) * (1.0 - gfi_expected))) < TOL

    # Information criteria.
    assert abs(result.aic - (chi2_expected + 2 * q)) < TOL
    assert abs(result.bic - (chi2_expected + q * math.log(n))) < TOL
    assert abs(result.sabic - (chi2_expected + q * math.log((n + 2) / 24.0))) < TOL

    _note(f"[hand check] chi2={result.chi_square:.9f} (expected "
          f"{chi2_expected:.9f}), df={result.df}, "
          f"rmsea={result.rmsea:.9f} (expected {rmsea_expected:.9f}), "
          f"cfi={result.cfi:.9f} (expected {cfi_expected:.9f}), "
          f"tli={result.tli:.9f} (expected {tli_expected:.9f})")


def test_baseline_fit_closed_form() -> None:
    """``chi2_b = (N-1)(sum log s_ii - log|S|)``, ``df_b = p(p-1)/2``."""
    p, n, r = 5, 401, 0.35
    S = 4.0 * compound_symmetry(p, r)          # variances of 4, correlations r
    chi2_b, df_b = fit.baseline_fit(S, n)
    # log|4 R| = p log 4 + log|R|; sum log s_ii = p log 4; so F_b = -log|R|.
    expected = (n - 1) * (-cs_logdet(p, r))
    assert df_b == p * (p - 1) // 2 == 10
    assert abs(chi2_b - expected) < 1e-9
    _note(f"[baseline] chi2_b={chi2_b:.9f} expected={expected:.9f} df_b={df_b}")


def test_chi_square_is_scale_free() -> None:
    """``F_ML`` is invariant to a common rescaling of both matrices."""
    S = compound_symmetry(4, 0.5)
    Sigma = compound_symmetry(4, 0.35)
    d = np.diag([1.0, 2.0, 3.0, 0.5])
    plain = fit.ml_discrepancy(S, Sigma)
    scaled = fit.ml_discrepancy(d @ S @ d, d @ Sigma @ d)
    assert abs(plain - scaled) < 1e-12


# --------------------------------------------------------------------------- #
# 3. RMSEA confidence interval
# --------------------------------------------------------------------------- #


def test_rmsea_interval_brackets_and_narrows_with_n() -> None:
    """lower <= rmsea <= upper, and the interval widens as N falls."""
    p, r_s, r_m, q = 6, 0.50, 0.40, 13
    S = compound_symmetry(p, r_s)
    Sigma = compound_symmetry(p, r_m)

    widths = {}
    for n in (200, 2000):
        result = fit.compute(S, Sigma, n=n, n_free=q)
        assert result.df == 8
        assert result.rmsea_ci_lower <= result.rmsea <= result.rmsea_ci_upper
        widths[n] = result.rmsea_ci_upper - result.rmsea_ci_lower
        # The interval limits invert the non-central chi-square exactly.
        from scipy.stats import ncx2
        for limit, target in ((result.rmsea_ci_lower, 0.95),
                              (result.rmsea_ci_upper, 0.05)):
            ncp = limit ** 2 * result.df * (n - 1)
            if ncp > 0:
                assert abs(ncx2.cdf(result.chi_square, result.df, ncp)
                           - target) < 1e-8
        _note(f"[rmsea ci] N={n}: rmsea={result.rmsea:.6f} "
              f"CI=[{result.rmsea_ci_lower:.6f}, {result.rmsea_ci_upper:.6f}] "
              f"width={widths[n]:.6f} pclose={result.rmsea_pclose:.6g}")

    assert widths[200] > widths[2000]


def test_rmsea_interval_lower_bound_is_zero_for_good_fit() -> None:
    """A chi-square below the 95th central percentile has no positive root."""
    from scipy.stats import chi2 as chi2_dist
    df, n = 8, 300
    chi2_value = float(chi2_dist.ppf(0.50, df))      # well below the .95 point
    lower, upper = fit.rmsea_confidence_interval(chi2_value, df, n)
    assert lower == 0.0
    assert upper > 0.0
    # And a chi-square below the 5th percentile loses the upper bound too.
    tiny = float(chi2_dist.ppf(0.01, df))
    assert fit.rmsea_confidence_interval(tiny, df, n) == (0.0, 0.0)
    _note(f"[rmsea ci] df=8 N=300 chi2={chi2_value:.4f} -> "
          f"[{lower:.6f}, {upper:.6f}]")


def test_pclose_closed_form() -> None:
    """PCLOSE = 1 - ncx2.cdf(chi2, df, (N-1) df 0.05^2)."""
    from scipy.stats import ncx2
    S = compound_symmetry(6, 0.50)
    Sigma = compound_symmetry(6, 0.40)
    result = fit.compute(S, Sigma, n=500, n_free=13)
    expected = 1.0 - float(ncx2.cdf(result.chi_square, result.df,
                                    499 * result.df * 0.05 ** 2))
    assert abs(result.rmsea_pclose - expected) < 1e-12


# --------------------------------------------------------------------------- #
# 4. verdict(), to_table() and the error contract
# --------------------------------------------------------------------------- #


def _blank_indices(**overrides: float) -> fit.FitIndices:
    base = dict(
        chi_square=10.0, df=8, p_value=0.3, chi_square_df_ratio=1.25,
        rmsea=0.02, rmsea_ci_lower=0.0, rmsea_ci_upper=0.05, rmsea_pclose=0.9,
        cfi=0.99, tli=0.99, nfi=0.98, ifi=0.99, rfi=0.97, srmr=0.02, gfi=0.98,
        agfi=0.96, aic=36.0, bic=60.0, sabic=40.0, baseline_chi_square=400.0,
        baseline_df=15, n=300, n_free=13)
    base.update(overrides)
    return fit.FitIndices(**base)


def test_verdict_bands_follow_hu_and_bentler() -> None:
    assert _blank_indices().verdict() is FitLevel.EXCELLENT
    assert _blank_indices(rmsea=0.05, cfi=0.96, srmr=0.06).verdict() is FitLevel.GOOD
    assert _blank_indices(rmsea=0.075, cfi=0.92, srmr=0.09).verdict() \
        is FitLevel.ACCEPTABLE
    assert _blank_indices(rmsea=0.095, cfi=0.88, srmr=0.11).verdict() \
        is FitLevel.MARGINAL
    assert _blank_indices(rmsea=0.14, cfi=0.70, srmr=0.14).verdict() is FitLevel.POOR
    # EXCELLENT needs all three conditions, not just RMSEA.
    assert _blank_indices(cfi=0.955).verdict() is FitLevel.GOOD
    assert _blank_indices(srmr=0.05).verdict() is FitLevel.GOOD


def test_published_field_order_is_unchanged() -> None:
    """The dataclass contract other modules are written against."""
    import dataclasses
    published = [
        "chi_square", "df", "p_value", "chi_square_df_ratio",
        "rmsea", "rmsea_ci_lower", "rmsea_ci_upper", "rmsea_pclose",
        "cfi", "tli", "nfi", "ifi", "rfi", "srmr", "gfi", "agfi",
        "aic", "bic", "sabic", "baseline_chi_square", "baseline_df",
        "n", "n_free",
    ]
    names = [f.name for f in dataclasses.fields(fit.FitIndices)]
    assert names[:len(published)] == published


def test_to_table_is_bilingual_and_complete() -> None:
    result = fit.compute(compound_symmetry(6, 0.5), compound_symmetry(6, 0.4),
                         n=400, n_free=13)
    for lang in ("en", "ar"):
        table = result.to_table(lang)
        assert list(table.columns) == ["index", "label", "value", "threshold",
                                       "verdict", "interpretation_en",
                                       "interpretation_ar"]
        assert len(table) == len(fit._ROWS) == 19
        assert table["label"].map(bool).all()
        assert table["interpretation_ar"].map(bool).all()
        assert table["interpretation_en"].map(bool).all()
        # Arabic text must really be Arabic, not a transliteration.
        assert all(any("؀" <= ch <= "ۿ" for ch in text)
                   for text in table["interpretation_ar"])
    ar = result.to_table("ar")
    rmsea_row = ar.loc[ar["index"] == "rmsea"].iloc[0]
    assert abs(float(rmsea_row["value"]) - result.rmsea) < TOL
    assert rmsea_row["threshold"] == "<= .06"
    thresholds = fit.thresholds_table("ar")
    assert len(thresholds) == 19
    assert "source" in thresholds.columns
    _note(f"[to_table] rmsea row verdict (ar) = {rmsea_row['verdict']}")


def test_user_errors_are_semsim_errors() -> None:
    """A singular matrix or a negative df never reaches the UI as a traceback."""
    singular = np.ones((3, 3))
    good = compound_symmetry(3, 0.2)
    for bad_call in (
        lambda: fit.compute(singular, good, n=100, n_free=3),
        lambda: fit.compute(good, singular, n=100, n_free=3),
        lambda: fit.baseline_fit(singular, 100),
    ):
        try:
            bad_call()
        except EstimationError as exc:
            assert exc.message_ar and exc.message_ar != exc.message_en
            assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
        else:  # pragma: no cover
            raise AssertionError("a singular matrix must raise EstimationError")

    for bad_call in (
        lambda: fit.compute(good, good, n=100, n_free=99),      # df < 0
        lambda: fit.compute(good, good, n=1, n_free=3),         # N < 2
        lambda: fit.compute(good, compound_symmetry(4, 0.2), n=100, n_free=3),
        lambda: fit.compute(good, good, n=100, n_free=3, n_items=7),
        lambda: fit.rmsea_confidence_interval(10.0, 5, 200, level=1.5),
    ):
        try:
            bad_call()
        except SpecificationError as exc:
            assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
        else:  # pragma: no cover
            raise AssertionError("a bad argument must raise SpecificationError")


def test_just_identified_model_is_degenerate_but_safe() -> None:
    sigma = compound_symmetry(3, 0.3)
    result = fit.compute(sigma, sigma.copy(), n=200, n_free=6)
    assert result.df == 0
    assert result.chi_square == 0.0
    assert result.rmsea == 0.0
    assert math.isnan(result.p_value)
    assert math.isnan(result.chi_square_df_ratio)
    assert math.isnan(result.agfi)


# --------------------------------------------------------------------------- #
# 5. Descriptives: the shape statistics and their standard errors
# --------------------------------------------------------------------------- #


def test_se_skewness_matches_spss_printout() -> None:
    """SE_skew = sqrt(6 n (n-1) / ((n-2)(n+1)(n+3))).

    At n = 100: 6 x 100 x 99 = 59400; 98 x 101 x 103 = 1019494;
    59400 / 1019494 = 0.0582535...; sqrt = 0.2413797790... -> 0.2414.
    """
    value = desc.se_skewness(100)
    assert round(value, 4) == 0.2414
    assert abs(value - math.sqrt(59400.0 / 1019494.0)) < 1e-15

    # The companion kurtosis standard error SPSS prints beside it:
    #   2 x 0.2413797790 x sqrt(9999 / (97 x 105))
    #     = 0.4827595581 x 0.9908268498 = 0.4783311330
    kurt = desc.se_kurtosis(100)
    assert abs(kurt - 2.0 * value * math.sqrt(9999.0 / (97.0 * 105.0))) < 1e-15
    assert round(kurt, 4) == 0.4783
    assert math.isnan(desc.se_skewness(2))
    assert math.isnan(desc.se_kurtosis(3))
    _note(f"[se] n=100: SE_skew={value:.10f} SE_kurt={kurt:.10f}")


def test_skewness_and_kurtosis_use_the_spss_estimators() -> None:
    """G1 and G2 recomputed from raw moments, independently of scipy."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    frame = pd.DataFrame({"X": values, "Y": values[::-1]})
    report = desc.describe(frame)
    item = report.item("X")
    assert item is not None
    assert abs(item.skewness - spss_skewness(values)) < 1e-12
    assert abs(item.kurtosis - spss_excess_kurtosis(values)) < 1e-12
    assert abs(item.mean - 4.0) < TOL
    assert abs(item.sd - float(np.std(values, ddof=1))) < TOL
    assert item.median == 3.0 and item.minimum == 1.0 and item.maximum == 10.0
    assert abs(item.z_skewness - item.skewness / desc.se_skewness(5)) < 1e-12
    _note(f"[G1/G2] X: skew={item.skewness:.9f} (hand "
          f"{spss_skewness(values):.9f}) kurt={item.kurtosis:.9f} (hand "
          f"{spss_excess_kurtosis(values):.9f})")


def test_normal_sample_has_shape_statistics_near_zero() -> None:
    rng = np.random.default_rng(2024)
    frame = pd.DataFrame({"A": rng.normal(size=4000)})
    item = desc.describe(frame).items[0]
    assert abs(item.skewness) < 4.0 * item.se_skewness
    assert abs(item.kurtosis) < 4.0 * item.se_kurtosis
    assert item.shapiro_w is not None and item.shapiro_p is not None
    assert 0.0 <= item.shapiro_w <= 1.0


def test_shapiro_is_skipped_above_five_thousand() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"A": rng.normal(size=desc.SHAPIRO_MAX_N + 1),
                          "B": rng.normal(size=desc.SHAPIRO_MAX_N + 1)})
    report = desc.describe(frame)
    assert all(item.shapiro_w is None and item.shapiro_p is None
               for item in report.items)
    table = report.to_table()
    assert table["shapiro_w"].isna().all()


# --------------------------------------------------------------------------- #
# 6. Mardia's multivariate tests
# --------------------------------------------------------------------------- #


def test_mardia_does_not_reject_multivariate_normal() -> None:
    """20000 draws from a correlated multivariate normal must survive."""
    rng = np.random.default_rng(7)
    p, n = 4, 20000
    chol = np.linalg.cholesky(compound_symmetry(p, 0.4))
    x = rng.normal(size=(n, p)) @ chol.T

    skew_stat, skew_p, kurt_z, kurt_p = desc.mardia_test(x)
    _note(f"[mardia normal] n={n} p={p}: skewness chi2={skew_stat:.4f} "
          f"p={skew_p:.4f}; kurtosis z={kurt_z:.4f} p={kurt_p:.4f}")
    assert skew_p > 0.01
    assert kurt_p > 0.01

    # Under normality b2p -> p(p+2) = 24 and b1p -> 0.
    b2p = kurt_z * math.sqrt(8.0 * p * (p + 2.0) / n) + p * (p + 2.0)
    assert abs(b2p - p * (p + 2.0)) < 0.5
    report = desc.describe(pd.DataFrame(x, columns=["a", "b", "c", "d"]))
    assert report.multivariate_normal is True
    assert abs(report.mardia_b2p - b2p) < 1e-9
    assert report.mardia_b1p >= 0.0


def test_mardia_rejects_a_strongly_skewed_sample() -> None:
    rng = np.random.default_rng(99)
    p, n = 4, 1000
    chol = np.linalg.cholesky(compound_symmetry(p, 0.4))
    x = rng.standard_exponential(size=(n, p)) @ chol.T

    skew_stat, skew_p, kurt_z, kurt_p = desc.mardia_test(x)
    _note(f"[mardia skewed] n={n} p={p}: skewness chi2={skew_stat:.2f} "
          f"p={skew_p:.3e}; kurtosis z={kurt_z:.2f} p={kurt_p:.3e}")
    assert skew_p < 0.001
    assert kurt_p < 0.001
    report = desc.describe(pd.DataFrame(x, columns=["a", "b", "c", "d"]))
    assert report.multivariate_normal is False


def test_mardia_b1p_matches_the_brute_force_definition() -> None:
    """The O(n p^3) tensor identity must equal the explicit n x n sum."""
    rng = np.random.default_rng(5)
    for n, p in ((300, 4), (40, 7)):        # n > p^2 and n < p^2: both routes
        x = rng.normal(size=(n, p)) + rng.standard_exponential(size=(n, p))
        skew_stat, _, _, _ = desc.mardia_test(x)
        b1p = 6.0 * skew_stat / n
        brute = mardia_b1p_bruteforce(x)
        assert abs(b1p - brute) < 1e-8 * max(1.0, abs(brute))
        _note(f"[mardia b1p] n={n} p={p}: identity={b1p:.10f} "
              f"brute force={brute:.10f}")


def test_mardia_rejects_degenerate_input() -> None:
    try:
        desc.mardia_test(np.zeros((3, 5)))
    except SpecificationError as exc:
        assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
    else:  # pragma: no cover
        raise AssertionError("too few complete cases must raise")

    rng = np.random.default_rng(3)
    base = rng.normal(size=(50, 1))
    collinear = np.hstack([base, 2.0 * base, rng.normal(size=(50, 1))])
    try:
        desc.mardia_test(collinear)
    except EstimationError as exc:
        assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
    else:  # pragma: no cover
        raise AssertionError("a singular covariance must raise EstimationError")


# --------------------------------------------------------------------------- #
# 7. Descriptives on realistic (ordinal, missing) data
# --------------------------------------------------------------------------- #


def test_describe_handles_missing_values() -> None:
    """NaN must not crash anything, and must be counted."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.integers(1, 6, size=(300, 4)).astype(float),
        columns=["Q1", "Q2", "Q3", "Q4"])
    frame.loc[0:9, "Q1"] = np.nan          # 10 missing
    frame.loc[5:7, "Q3"] = np.nan          # 3 missing, overlapping rows

    report = desc.describe(frame)
    by_name = {item.name: item for item in report.items}
    assert by_name["Q1"].n_missing == 10 and by_name["Q1"].n == 290
    assert by_name["Q3"].n_missing == 3 and by_name["Q3"].n == 297
    assert by_name["Q2"].n_missing == 0 and by_name["Q2"].n == 300
    assert report.n == 300
    assert report.n_complete == 300 - 10 - 3 + 3   # Q3's gaps sit inside Q1's
    assert math.isfinite(report.mardia_skewness)
    assert any("missing" in w for w in report.warnings)
    table = report.to_table("ar")
    assert int(table.loc[table["item"] == "Q1", "n_missing"].iloc[0]) == 10
    _note(f"[missing] n={report.n} complete={report.n_complete} "
          f"Q1 n_missing={by_name['Q1'].n_missing}")

    # An item that is entirely missing must still produce a row.
    frame["Q5"] = np.nan
    all_missing = desc.describe(frame).item("Q5")
    assert all_missing is not None
    assert all_missing.n == 0 and all_missing.n_missing == 300
    assert math.isnan(all_missing.mean)
    assert all_missing.shapiro_w is None


def test_category_proportions_and_floor_ceiling() -> None:
    """Closed-form proportions for a hand-built five-point item."""
    # 10 ones, 20 threes, 70 fives on a 1..5 scale.
    values = np.array([1.0] * 10 + [3.0] * 20 + [5.0] * 70)
    frame = pd.DataFrame({"V1": values, "V2": values[::-1]})
    item = desc.describe(frame).item("V1")
    assert item is not None
    assert item.is_ordinal is True
    assert item.category_proportions == {1: 0.10, 2: 0.0, 3: 0.20, 4: 0.0,
                                          5: 0.70}
    assert abs(sum(item.category_proportions.values()) - 1.0) < TOL
    assert abs(item.floor_pct - 10.0) < TOL
    assert abs(item.ceiling_pct - 70.0) < TOL
    # mean = (10 + 60 + 350)/100 = 4.2
    assert abs(item.mean - 4.2) < TOL
    _note(f"[floor/ceiling] floor={item.floor_pct:.1f}% "
          f"ceiling={item.ceiling_pct:.1f}%")


def test_describe_uses_the_spec_for_labels_and_scale_bounds() -> None:
    """With a spec, the floor is category 1 even if nobody chose it."""
    spec = simple_mediation()
    names = spec.all_item_names()
    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        rng.integers(3, 6, size=(200, len(names))).astype(float),
        columns=names)
    report = desc.describe(data, spec)
    first = report.items[0]
    assert first.is_ordinal is True
    assert first.n_categories == 5
    assert first.floor_pct == 0.0              # category 1 never chosen
    assert set(first.category_proportions) == {1, 2, 3, 4, 5}
    assert first.label_ar and first.label_en
    table = report.to_table("ar")
    assert table["label"].iloc[0] == first.label_ar
    assert len(report.items) == len(names) == 14


def test_describe_rejects_a_table_with_no_numeric_item() -> None:
    frame = pd.DataFrame({"name": ["a", "b", "c"]})
    try:
        desc.describe(frame)
    except SpecificationError as exc:
        assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
    else:  # pragma: no cover
        raise AssertionError("a non-numeric table must raise")


def test_descriptives_on_a_compiled_model_population() -> None:
    """Draw from Sigma(theta) and check the description is sane."""
    spec = minimal_cfa()
    cm = compile_model(spec, rng=np.random.default_rng(0))
    sigma = implied_covariance(cm)
    rng = np.random.default_rng(314)
    x = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=3000)
    frame = pd.DataFrame(x, columns=spec.all_item_names())
    report = desc.describe(frame, spec)
    assert len(report.items) == 6
    assert all(item.category_proportions is None for item in report.items)
    for item in report.items:
        assert abs(item.mean) < 0.1
        assert abs(item.sd - 1.0) < 0.1
    assert report.multivariate_normal is True
    _note(f"[population] mardia skew p={report.mardia_skewness_p:.4f} "
          f"kurt p={report.mardia_kurtosis_p:.4f}")


# --------------------------------------------------------------------------- #
# Script runner
# --------------------------------------------------------------------------- #


def _all_tests() -> list:
    return [obj for name, obj in sorted(globals().items())
            if name.startswith("test_") and callable(obj)]


def main() -> int:
    failures = 0
    for test in _all_tests():
        try:
            test()
        except Exception as exc:                     # noqa: BLE001
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(_all_tests()) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

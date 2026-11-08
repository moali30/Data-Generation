"""Numerical tests for :mod:`semsim.analysis.reliability` and
:mod:`semsim.analysis.polychoric`.

Every assertion is against a closed-form value, a published identity, or a
large-sample simulation with a fixed ``numpy.random.Generator`` -- never
merely "it ran".

Closed-form anchors used here
-----------------------------
* **Alpha under compound symmetry.**  With ``k`` items all correlated ``r``,
  the standardised alpha is the Spearman-Brown value ``k r / (1 + (k-1) r)``
  exactly.  ``k = 4``, ``r = 0.5`` gives ``0.80``; ``k = 4``, ``r = 0.64``
  (loadings of ``0.8``) gives ``0.876712...``.
* **Composite reliability and AVE.**  ``lambda = 0.8``, ``k = 4``:
  ``AVE = 0.64`` and ``CR = 3.2^2 / (3.2^2 + 4(0.36)) = 10.24 / 11.68 =
  0.876712...``  (Equal loadings make CR and alpha coincide, which is itself
  a published identity: alpha is CR under tau-equivalence.)
* **Omega total vs CR.**  Equal on a congeneric population matrix, and
  ``omega_total < CR`` once a residual covariance is added, because the
  observed total variance ``1' R 1`` then exceeds the model-implied one.
* **HTMT.**  Loadings ``0.8`` and factor correlation ``0.5`` give
  between-item correlations ``0.5 * 0.64 = 0.32`` and within-item
  correlations ``0.64``, so Henseler's ratio returns
  ``0.32 / sqrt(0.64 * 0.64) = 0.50`` -- the disattenuated (i.e. true) factor
  correlation.  The variant without the square root would return
  ``0.32 / 0.4096 = 0.78125``; both numbers are reported by the test so the
  distinction is on the record.
* **Feldt's alpha interval.**  Reproduced by inverting
  ``scipy.stats.f`` by hand.
* **Sheppard's theorem.**  ``Phi2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi)``.
* **Polychoric.**  Two normals correlated ``0.60``, both cut into five
  equal-probability categories: the polychoric estimate returns the
  underlying ``0.60`` while Pearson's ``r`` on the integers is attenuated to
  about ``0.55`` (the first-order prediction is
  ``0.942297^2 * 0.6 = 0.533``).
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from semsim.core.errors import SpecificationError
from semsim.core.spec import ModelSpec
from semsim.analysis.polychoric import (
    MAX_ABS_RHO,
    bivariate_normal_cdf,
    polychoric_correlation,
    polychoric_matrix,
    polyserial_correlation,
    thresholds_from_counts,
)
from semsim.analysis.polychoric import _dw_cdf
from semsim.analysis.reliability import (
    FactorReliability,
    ValidityReport,
    alpha_confidence_interval,
    alpha_from_matrix,
    analyse,
    average_variance_extracted,
    composite_reliability,
    cronbach_alpha,
    heywood_items,
    htmt_matrix,
    omega_total,
    principal_axis_loadings,
)

#: Collected so the stand-alone runner can print them next to the checks.
OBSERVED: Dict[str, float] = {}


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def compound_symmetric(k: int, r: float) -> np.ndarray:
    """``(k, k)`` correlation matrix with every off-diagonal equal to ``r``."""
    R = np.full((k, k), float(r))
    np.fill_diagonal(R, 1.0)
    return R


def spearman_brown(k: int, r: float) -> float:
    """Population alpha of ``k`` items all correlated ``r``."""
    return k * r / (1.0 + (k - 1) * r)


def two_factor_population(loading: float, phi: float,
                          k: int = 4) -> np.ndarray:
    """Implied correlation matrix of two congeneric factors.

    ``Lambda Phi Lambda' + Theta`` with a unit diagonal, which is the
    completely standardised metric the whole package works in.
    """
    L = np.zeros((2 * k, 2))
    L[:k, 0] = loading
    L[k:, 1] = loading
    Phi = np.array([[1.0, phi], [phi, 1.0]])
    R = L @ Phi @ L.T
    np.fill_diagonal(R, 1.0)
    return R


def two_factor_spec(k: int = 4, loading: float = 0.8) -> ModelSpec:
    spec = ModelSpec()
    spec.add_construct("FA", n_items=k, loading=loading)
    spec.add_construct("FB", n_items=k, loading=loading)
    return spec


# --------------------------------------------------------------------------- #
# Cronbach's alpha
# --------------------------------------------------------------------------- #


def test_alpha_matches_spearman_brown_in_closed_form() -> None:
    """Alpha of a compound-symmetric matrix is ``k r / (1 + (k-1) r)``."""
    for k, r in ((4, 0.5), (6, 0.3), (3, 0.64), (10, 0.1)):
        expected = spearman_brown(k, r)
        obtained = alpha_from_matrix(compound_symmetric(k, r))
        assert abs(obtained - expected) < 1e-12, (k, r, obtained, expected)
    assert abs(alpha_from_matrix(compound_symmetric(4, 0.5)) - 0.80) < 1e-12


def test_alpha_from_simulated_data_recovers_population_080() -> None:
    """A 4-item, r = 0.5 population has alpha = 0.80; n = 20000 must find it."""
    rng = np.random.default_rng(20240101)
    R = compound_symmetric(4, 0.5)
    x = rng.multivariate_normal(np.zeros(4), R, size=20000)
    alpha = cronbach_alpha(x)
    OBSERVED["alpha_n20000"] = alpha
    assert abs(alpha - 0.80) < 0.03, alpha
    # The covariance form and the matrix form must agree on the same data.
    direct = alpha_from_matrix(np.cov(x, rowvar=False, ddof=1))
    assert abs(alpha - direct) < 1e-12


def test_alpha_equals_the_long_hand_formula() -> None:
    """``k/(k-1) (1 - sum(var_i)/var(total))``, computed item by item."""
    rng = np.random.default_rng(7)
    x = rng.multivariate_normal(np.zeros(5), compound_symmetric(5, 0.4),
                                size=400)
    k = x.shape[1]
    item_variances = sum(float(np.var(x[:, j], ddof=1)) for j in range(k))
    total_variance = float(np.var(x.sum(axis=1), ddof=1))
    expected = k / (k - 1.0) * (1.0 - item_variances / total_variance)
    assert abs(cronbach_alpha(x) - expected) < 1e-12


def test_alpha_rejects_degenerate_input_with_arabic_messages() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(SpecificationError) as one_item:
        alpha_from_matrix(np.array([[1.0]]))
    with pytest.raises(SpecificationError):
        cronbach_alpha(rng.normal(size=50))              # one-dimensional
    with pytest.raises(SpecificationError):
        alpha_from_matrix(np.zeros((3, 3)))              # zero total variance
    message = one_item.value.message_ar
    assert message and any(ord(ch) > 1500 for ch in message), message
    assert one_item.value.message_en != message


# --------------------------------------------------------------------------- #
# Feldt's confidence interval
# --------------------------------------------------------------------------- #


def test_feldt_interval_matches_hand_inversion_of_the_f_distribution() -> None:
    alpha, n, k, level = 0.84, 250, 6, 0.95
    lower, upper = alpha_confidence_interval(alpha, n, k, level)
    gamma = 1.0 - level
    df1, df2 = n - 1, (n - 1) * (k - 1)
    expected_lo = 1.0 - (1.0 - alpha) / stats.f.ppf(gamma / 2.0, df1, df2)
    expected_hi = 1.0 - (1.0 - alpha) / stats.f.ppf(1.0 - gamma / 2.0,
                                                    df1, df2)
    assert abs(lower - expected_lo) < 1e-12
    assert abs(upper - expected_hi) < 1e-12
    assert lower < alpha < upper
    # Asymmetric, and tighter above than below, because alpha <= 1.
    assert (upper - alpha) < (alpha - lower)
    OBSERVED["feldt_lo"] = lower
    OBSERVED["feldt_hi"] = upper


def test_feldt_interval_narrows_with_n_and_degrades_gracefully() -> None:
    narrow = alpha_confidence_interval(0.80, 5000, 4)
    wide = alpha_confidence_interval(0.80, 50, 4)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    assert all(np.isnan(v) for v in alpha_confidence_interval(0.8, 2, 4))
    assert all(np.isnan(v) for v in alpha_confidence_interval(0.8, 200, 1))
    with pytest.raises(SpecificationError):
        alpha_confidence_interval(0.8, 200, 4, level=1.5)


def test_feldt_interval_covers_the_population_value() -> None:
    """Empirical coverage of the nominal 95% interval, 400 replications."""
    rng = np.random.default_rng(99)
    R = compound_symmetric(4, 0.5)
    population = 0.80
    covered = 0
    reps = 400
    for _ in range(reps):
        x = rng.multivariate_normal(np.zeros(4), R, size=200)
        a = cronbach_alpha(x)
        lo, hi = alpha_confidence_interval(a, 200, 4)
        covered += int(lo <= population <= hi)
    rate = covered / reps
    OBSERVED["feldt_coverage"] = rate
    assert 0.90 <= rate <= 0.99, rate


# --------------------------------------------------------------------------- #
# CR, omega, AVE
# --------------------------------------------------------------------------- #


def test_cr_and_ave_closed_form_for_four_loadings_of_08() -> None:
    loadings = [0.8] * 4
    ave = average_variance_extracted(loadings)
    cr = composite_reliability(loadings)
    OBSERVED["ave"] = ave
    OBSERVED["cr"] = cr
    assert abs(ave - 0.64) < 1e-12, ave
    assert abs(cr - 10.24 / 11.68) < 1e-12, cr
    assert abs(cr - 0.8767123287671233) < 1e-12, cr
    # Explicit error variances must give the same answer as theta = 1 - l^2.
    assert abs(composite_reliability(loadings, [0.36] * 4) - cr) < 1e-12


def test_cr_equals_alpha_under_tau_equivalence() -> None:
    """Equal loadings make alpha and CR the same number (Novick & Lewis)."""
    for loading in (0.5, 0.7, 0.8, 0.9):
        k = 5
        r = loading ** 2
        assert abs(composite_reliability([loading] * k)
                   - spearman_brown(k, r)) < 1e-12


def test_cr_exceeds_alpha_when_loadings_are_unequal() -> None:
    loadings = np.array([0.9, 0.8, 0.6, 0.4])
    R = np.outer(loadings, loadings)
    np.fill_diagonal(R, 1.0)
    cr = composite_reliability(loadings)
    alpha = alpha_from_matrix(R)
    OBSERVED["cr_congeneric"] = cr
    OBSERVED["alpha_congeneric"] = alpha
    assert cr > alpha, (cr, alpha)


def test_omega_total_equals_cr_on_a_congeneric_population() -> None:
    loadings = np.array([0.9, 0.8, 0.7, 0.6])
    R = np.outer(loadings, loadings)
    np.fill_diagonal(R, 1.0)
    cr = composite_reliability(loadings)
    omega = omega_total(loadings, R)
    OBSERVED["omega_congeneric"] = omega
    assert abs(omega - cr) < 1e-12, (omega, cr)


def test_omega_total_falls_below_cr_with_a_correlated_residual() -> None:
    loadings = np.array([0.8, 0.8, 0.8, 0.8])
    R = np.outer(loadings, loadings)
    np.fill_diagonal(R, 1.0)
    R[0, 1] = R[1, 0] = R[0, 1] + 0.20      # shared wording variance
    cr = composite_reliability(loadings)
    omega = omega_total(loadings, R)
    OBSERVED["omega_with_residual_cov"] = omega
    assert omega < cr - 0.01, (omega, cr)
    # The whole difference is the extra 2 * 0.20 in the observed total.
    assert abs(omega - 10.24 / (11.68 + 0.40)) < 1e-12


def test_loading_coefficients_reject_bad_input() -> None:
    with pytest.raises(SpecificationError):
        average_variance_extracted([])
    with pytest.raises(SpecificationError):
        composite_reliability([0.8, np.nan])
    with pytest.raises(SpecificationError):
        composite_reliability([0.8, 0.8], [0.3])
    with pytest.raises(SpecificationError):
        omega_total([0.8, 0.8], np.eye(3))


# --------------------------------------------------------------------------- #
# Principal axis fallback
# --------------------------------------------------------------------------- #


def test_principal_axis_recovers_the_generating_loadings() -> None:
    for loading in (0.5, 0.7, 0.8, 0.9):
        R = np.full((5, 5), loading ** 2)
        np.fill_diagonal(R, 1.0)
        lam = principal_axis_loadings(R)
        assert np.allclose(lam, loading, atol=2e-3), (loading, lam)


def test_principal_axis_recovers_unequal_loadings() -> None:
    loadings = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    R = np.outer(loadings, loadings)
    np.fill_diagonal(R, 1.0)
    lam = principal_axis_loadings(R)
    OBSERVED["paf_max_error"] = float(np.max(np.abs(lam - loadings)))
    assert np.allclose(lam, loadings, atol=5e-3), lam
    assert principal_axis_loadings(np.array([[1.0]]))[0] == 1.0


# --------------------------------------------------------------------------- #
# HTMT
# --------------------------------------------------------------------------- #


def test_htmt_recovers_the_disattenuated_factor_correlation() -> None:
    """Loadings 0.8, phi 0.5: Henseler's ratio must return 0.50.

    ``mean(r_between) = 0.32``, ``mean(r_within) = 0.64`` and the denominator
    is the *geometric* mean of the two within averages, so the ratio is
    ``0.32 / 0.64 = 0.50``.  Dividing by the *product* instead -- which is not
    Henseler's definition -- would give ``0.32 / 0.4096 = 0.78125``; both are
    reported here.
    """
    rng = np.random.default_rng(424242)
    R = two_factor_population(0.8, 0.5)
    k = 4
    names = [f"FA{i}" for i in range(1, k + 1)] + \
            [f"FB{i}" for i in range(1, k + 1)]
    x = rng.multivariate_normal(np.zeros(2 * k), R, size=6000)
    frame = pd.DataFrame(x, columns=names)
    groups = {"FA": names[:k], "FB": names[k:]}

    table = htmt_matrix(frame, groups)
    observed = float(table.loc["FA", "FB"])
    OBSERVED["htmt_geometric_mean"] = observed

    corr = frame.corr().to_numpy()
    between = float(corr[:k, k:].mean())
    within_a = (corr[:k, :k].sum() - k) / (k * (k - 1))
    within_b = (corr[k:, k:].sum() - k) / (k * (k - 1))
    no_sqrt = between / (within_a * within_b)
    OBSERVED["htmt_product_variant"] = float(no_sqrt)
    OBSERVED["htmt_mean_r_between"] = between
    OBSERVED["htmt_mean_r_within"] = float(0.5 * (within_a + within_b))

    assert abs(observed - 0.50) < 0.05, observed
    assert abs(no_sqrt - 0.78125) < 0.05, no_sqrt
    assert abs(table.loc["FA", "FA"] - 1.0) < 1e-12
    assert abs(table.loc["FA", "FB"] - table.loc["FB", "FA"]) < 1e-12


def test_htmt_equals_phi_exactly_on_the_population_matrix() -> None:
    """On the implied matrix itself the ratio is the factor correlation."""
    k = 4
    names = [f"A{i}" for i in range(k)] + [f"B{i}" for i in range(k)]
    for phi in (0.1, 0.3, 0.5, 0.75, 0.9):
        R = pd.DataFrame(two_factor_population(0.8, phi), index=names,
                         columns=names)
        table = htmt_matrix(pd.DataFrame(), {"A": names[:k], "B": names[k:]},
                            correlations=R)
        assert abs(float(table.loc["A", "B"]) - phi) < 1e-12, (phi, table)


def test_htmt_rejects_a_single_factor_and_unknown_items() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 3.0, 2.0]})
    with pytest.raises(SpecificationError):
        htmt_matrix(frame, {"only": ["a", "b"]})
    with pytest.raises(SpecificationError):
        htmt_matrix(frame, {"one": ["a"], "two": ["zzz"]})


# --------------------------------------------------------------------------- #
# analyse(): the assembled report
# --------------------------------------------------------------------------- #


def test_analyse_reproduces_every_closed_form_on_population_data() -> None:
    rng = np.random.default_rng(20240202)
    spec = two_factor_spec(k=4, loading=0.8)
    names = spec.all_item_names()
    R = two_factor_population(0.8, 0.5)
    x = rng.multivariate_normal(np.zeros(8), R, size=8000)
    report = analyse(pd.DataFrame(x, columns=names), spec)

    assert isinstance(report, ValidityReport)
    assert [entry.n_items for entry in report.per_factor] == [4, 4]
    for entry in report.per_factor:
        assert isinstance(entry, FactorReliability)
        # alpha for r = 0.64, k = 4 is 0.876712...
        assert abs(entry.cronbach_alpha - spearman_brown(4, 0.64)) < 0.02
        assert abs(entry.ave - 0.64) < 0.03, entry.ave
        assert abs(entry.composite_reliability - 0.8767) < 0.02
        assert abs(entry.mean_inter_item_r - 0.64) < 0.03
        assert abs(entry.sqrt_ave - np.sqrt(entry.ave)) < 1e-12
        assert entry.alpha_ci[0] < entry.cronbach_alpha < entry.alpha_ci[1]
        assert len(entry.item_total_correlations) == 4
        assert len(entry.alpha_if_item_dropped) == 4

    OBSERVED["analyse_alpha"] = report.per_factor[0].cronbach_alpha
    OBSERVED["analyse_ave"] = report.per_factor[0].ave
    OBSERVED["analyse_cr"] = report.per_factor[0].composite_reliability

    htmt = float(report.htmt.iloc[0, 1])
    phi = float(report.factor_correlations.iloc[0, 1])
    OBSERVED["analyse_htmt"] = htmt
    OBSERVED["analyse_phi"] = phi
    assert abs(htmt - 0.50) < 0.05, htmt
    assert abs(phi - 0.50) < 0.05, phi
    assert report.htmt_passed is True
    assert report.fornell_larcker_passed is True
    assert abs(float(report.fornell_larcker.iloc[0, 0])
               - report.per_factor[0].sqrt_ave) < 1e-12
    assert report.cross_loading_table.shape[0] == 8
    assert not report.cross_loading_table["flagged"].any()


def test_item_total_and_drop_one_alpha_match_direct_computation() -> None:
    rng = np.random.default_rng(11)
    spec = ModelSpec()
    spec.add_construct("F", n_items=5, loading=0.75)
    names = spec.all_item_names()
    loadings = np.full(5, 0.75)
    R = np.outer(loadings, loadings)
    np.fill_diagonal(R, 1.0)
    x = rng.multivariate_normal(np.zeros(5), R, size=900)
    report = analyse(pd.DataFrame(x, columns=names), spec)
    entry = report.per_factor[0]

    for j, name in enumerate(names):
        rest = np.delete(x, j, axis=1).sum(axis=1)
        expected = float(np.corrcoef(x[:, j], rest)[0, 1])
        assert abs(entry.item_total_correlations[name] - expected) < 1e-10, name
        expected_alpha = cronbach_alpha(np.delete(x, j, axis=1))
        assert abs(entry.alpha_if_item_dropped[name]
                   - expected_alpha) < 1e-10, name
    OBSERVED["item_total_min"] = min(entry.item_total_correlations.values())


def test_fornell_larcker_fails_when_two_factors_nearly_coincide() -> None:
    rng = np.random.default_rng(606)
    spec = two_factor_spec(k=4, loading=0.7)
    names = spec.all_item_names()
    R = two_factor_population(0.7, 0.95)
    x = rng.multivariate_normal(np.zeros(8), R, size=3000)
    report = analyse(pd.DataFrame(x, columns=names), spec,
                     htmt_threshold=0.85)
    OBSERVED["fl_fail_htmt"] = float(report.htmt.iloc[0, 1])
    OBSERVED["fl_fail_sqrt_ave"] = report.per_factor[0].sqrt_ave
    assert report.fornell_larcker_passed is False
    assert report.htmt_passed is False
    assert any("Fornell-Larcker" in flag for flag in report.flags)
    assert any("HTMT" in flag for flag in report.flags)
    # Every flag carries both languages.
    for flag in report.flags:
        assert " | " in flag and any(ord(ch) > 1500 for ch in flag), flag


def test_analyse_prefers_the_estimations_standardised_loadings() -> None:
    """A duck-typed estimation object drives CR and AVE, not the data."""
    rng = np.random.default_rng(13)
    spec = two_factor_spec(k=4, loading=0.8)
    names = spec.all_item_names()
    x = rng.normal(size=(300, 8))          # uncorrelated: PAF would find ~0

    class FakeEstimation:
        loadings = pd.DataFrame({
            "item": names,
            "factor": ["FA"] * 4 + ["FB"] * 4,
            "std": [0.8] * 8,
            "est": [0.8] * 8,
        })

    report = analyse(pd.DataFrame(x, columns=names), spec, FakeEstimation())
    for entry in report.per_factor:
        assert abs(entry.ave - 0.64) < 1e-12, entry.ave
        assert abs(entry.composite_reliability - 10.24 / 11.68) < 1e-12

    class Useless:
        loadings = pd.DataFrame({"nothing": [1, 2, 3]})

    fallback = analyse(pd.DataFrame(x, columns=names), spec, Useless())
    assert fallback.per_factor[0].ave < 0.3      # back to the data


def test_analyse_handles_second_order_constructs_and_missing_values() -> None:
    rng = np.random.default_rng(17)
    spec = ModelSpec()
    spec.add_second_order_construct("SO", n_dimensions=2,
                                    items_per_dimension=3, loading=0.8)
    spec.add_construct("OTHER", n_items=3, loading=0.8)
    names = spec.all_item_names()
    x = rng.normal(size=(400, len(names)))
    frame = pd.DataFrame(x, columns=names)
    frame.iloc[0, 0] = np.nan
    frame.iloc[5, 2] = np.nan
    report = analyse(frame, spec)

    ids = [entry.factor_id for entry in report.per_factor]
    assert len(ids) == 4                      # 2 dimensions + OTHER + SO total
    whole = report.per_factor[-1]
    assert whole.n_items == 6                 # the second-order total scale
    # The discriminant blocks only hold the three non-overlapping factors.
    assert report.htmt.shape == (3, 3)
    assert report.factor_correlations.shape == (3, 3)
    assert any("Listwise" in flag for flag in report.flags)


def test_analyse_rejects_data_without_any_model_item() -> None:
    spec = two_factor_spec()
    with pytest.raises(SpecificationError):
        analyse(pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]}), spec)


def test_analyse_is_fast_on_a_500_by_60_dataset() -> None:
    rng = np.random.default_rng(31)
    spec = ModelSpec()
    for f in range(10):
        spec.add_construct(f"F{f}", n_items=6, loading=0.8)
    names = spec.all_item_names()
    assert len(names) == 60
    x = rng.normal(size=(500, 60))
    frame = pd.DataFrame(x, columns=names)
    start = time.perf_counter()
    report = analyse(frame, spec)
    elapsed = time.perf_counter() - start
    OBSERVED["analyse_500x60_seconds"] = elapsed
    assert len(report.per_factor) == 10
    assert elapsed < 1.0, elapsed


# --------------------------------------------------------------------------- #
# Bivariate normal CDF
# --------------------------------------------------------------------------- #


def test_bivariate_normal_cdf_matches_sheppards_closed_form() -> None:
    for rho in (-0.9, -0.5, 0.0, 0.25, 0.6, 0.95):
        expected = 0.25 + np.arcsin(rho) / (2.0 * np.pi)
        obtained = bivariate_normal_cdf(0.0, 0.0, rho)
        assert abs(obtained - expected) < 1e-10, (rho, obtained, expected)
    assert abs(bivariate_normal_cdf(0.0, 0.0, 0.0) - 0.25) < 1e-12
    assert abs(bivariate_normal_cdf(0.0, 0.0, 1.0) - 0.5) < 1e-12
    assert bivariate_normal_cdf(-np.inf, 0.3, 0.5) == 0.0
    assert abs(bivariate_normal_cdf(np.inf, 0.3, 0.5)
               - stats.norm.cdf(0.3)) < 1e-12
    with pytest.raises(SpecificationError):
        bivariate_normal_cdf(0.0, 0.0, 1.5)


def test_drezner_wesolowsky_agrees_with_scipy_to_machine_precision() -> None:
    """The vectorised quadrature the likelihood uses is not an approximation."""
    worst = 0.0
    for rho in (-0.999, -0.7, -0.2, 0.2, 0.7, 0.9, 0.99, 0.999):
        for h in (-3.0, -1.0, -0.25, 0.0, 0.5, 2.0, 4.0):
            for k in (-3.0, -0.5, 0.0, 1.0, 3.0):
                reference = float(stats.multivariate_normal.cdf(
                    np.array([h, k]), mean=np.zeros(2),
                    cov=np.array([[1.0, rho], [rho, 1.0]])))
                obtained = float(_dw_cdf(np.array(h), np.array(k), rho))
                worst = max(worst, abs(obtained - reference))
    OBSERVED["dw_vs_scipy_max_abs_error"] = worst
    assert worst < 1e-9, worst


def test_thresholds_from_counts_are_the_inverse_normal_cumulants() -> None:
    tau = thresholds_from_counts([25, 25, 25, 25])
    assert np.allclose(tau, stats.norm.ppf([0.25, 0.50, 0.75]), atol=1e-12)
    tau = thresholds_from_counts([10, 20, 30, 40])
    assert np.allclose(tau, stats.norm.ppf([0.1, 0.3, 0.6]), atol=1e-12)
    # An empty middle category makes two thresholds coincide, not explode.
    tau = thresholds_from_counts([50, 0, 50])
    assert np.isfinite(tau).all() and tau[0] == tau[1]
    with pytest.raises(SpecificationError):
        thresholds_from_counts([0, 0, 0])


# --------------------------------------------------------------------------- #
# Polychoric correlation
# --------------------------------------------------------------------------- #


def _categorise(z: np.ndarray, n_categories: int) -> np.ndarray:
    """Equal-probability normal cut points, scores ``1 .. n_categories``."""
    cuts = stats.norm.ppf(np.arange(1, n_categories) / n_categories)
    return np.digitize(z, cuts) + 1


def test_polychoric_beats_pearson_on_five_point_items() -> None:
    """rho = 0.60, 100 000 cases, five symmetric categories.

    The polychoric estimate must land on the *underlying* 0.60; Pearson's r
    on the integers must be noticeably lower.  That gap is the reason the
    function exists.
    """
    rng = np.random.default_rng(31337)
    n = 100_000
    rho = 0.60
    z = rng.multivariate_normal(np.zeros(2),
                                np.array([[1.0, rho], [rho, 1.0]]), size=n)
    x = _categorise(z[:, 0], 5)
    y = _categorise(z[:, 1], 5)

    start = time.perf_counter()
    poly = polychoric_correlation(x, y)
    elapsed = time.perf_counter() - start
    pearson = float(np.corrcoef(x, y)[0, 1])
    OBSERVED["polychoric_rho"] = poly
    OBSERVED["pearson_of_categories"] = pearson
    OBSERVED["polychoric_seconds"] = elapsed

    assert abs(poly - 0.60) < 0.02, poly
    assert pearson < poly - 0.03, (pearson, poly)
    assert pearson < 0.58, pearson
    # The latent correlation of the uncategorised normals is the benchmark.
    assert abs(float(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) - 0.60) < 0.01
    assert elapsed < 1.0, elapsed


def test_polychoric_recovers_several_rho_values_and_both_signs() -> None:
    rng = np.random.default_rng(2718)
    errors: List[float] = []
    for rho in (-0.7, -0.3, 0.0, 0.45, 0.8):
        z = rng.multivariate_normal(np.zeros(2),
                                    np.array([[1.0, rho], [rho, 1.0]]),
                                    size=40_000)
        poly = polychoric_correlation(_categorise(z[:, 0], 5),
                                      _categorise(z[:, 1], 5))
        errors.append(abs(poly - rho))
        assert abs(poly - rho) < 0.025, (rho, poly)
    OBSERVED["polychoric_max_abs_error"] = max(errors)


def test_tetrachoric_is_the_two_category_special_case() -> None:
    rng = np.random.default_rng(555)
    rho = 0.5
    z = rng.multivariate_normal(np.zeros(2),
                                np.array([[1.0, rho], [rho, 1.0]]),
                                size=60_000)
    x = (z[:, 0] > 0).astype(int)
    y = (z[:, 1] > 0).astype(int)
    tetra = polychoric_correlation(x, y)
    OBSERVED["tetrachoric"] = tetra
    assert abs(tetra - 0.5) < 0.03, tetra
    # Pearson (phi) on dichotomies is attenuated to (2/pi) arcsin(rho).
    phi = float(np.corrcoef(x, y)[0, 1])
    assert abs(phi - 2.0 / np.pi * np.arcsin(rho)) < 0.02, phi
    assert tetra > phi + 0.1


def test_polychoric_handles_an_empty_category_and_self_correlation() -> None:
    rng = np.random.default_rng(1234)
    rho = 0.5
    z = rng.multivariate_normal(np.zeros(2),
                                np.array([[1.0, rho], [rho, 1.0]]),
                                size=20_000)
    x = _categorise(z[:, 0], 5)
    y = _categorise(z[:, 1], 5)
    gapped = np.where(x == 3, 2, x)            # nobody chose category 3
    assert set(np.unique(gapped)) == {1, 2, 4, 5}
    estimate = polychoric_correlation(gapped, y)
    OBSERVED["polychoric_with_empty_category"] = estimate
    assert np.isfinite(estimate) and abs(estimate - 0.5) < 0.05, estimate

    assert polychoric_correlation(x, x) == 1.0
    assert polychoric_correlation(x.astype(float), x.astype(float)) == 1.0


def test_polychoric_rejects_degenerate_pairs_in_arabic() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(1, 6, size=200)
    with pytest.raises(SpecificationError) as constant:
        polychoric_correlation(np.full(200, 3), y)
    with pytest.raises(SpecificationError):
        polychoric_correlation(y[:100], y)                 # length mismatch
    with pytest.raises(SpecificationError):
        polychoric_correlation(rng.normal(size=200), y)    # continuous x
    message = constant.value.message_ar
    assert message and any(ord(ch) > 1500 for ch in message), message


def test_polychoric_handles_missing_values_pairwise() -> None:
    rng = np.random.default_rng(808)
    rho = 0.6
    z = rng.multivariate_normal(np.zeros(2),
                                np.array([[1.0, rho], [rho, 1.0]]),
                                size=30_000)
    x = _categorise(z[:, 0], 5).astype(float)
    y = _categorise(z[:, 1], 5).astype(float)
    x[::17] = np.nan
    y[::23] = np.nan
    estimate = polychoric_correlation(x, y)
    assert abs(estimate - 0.6) < 0.03, estimate


def test_polyserial_corrects_the_attenuation_of_categorising() -> None:
    rng = np.random.default_rng(2024)
    rho = 0.6
    z = rng.multivariate_normal(np.zeros(2),
                                np.array([[1.0, rho], [rho, 1.0]]),
                                size=60_000)
    ordinal = _categorise(z[:, 0], 5)
    continuous = z[:, 1]
    polyserial = polyserial_correlation(ordinal, continuous)
    pearson = float(np.corrcoef(ordinal, continuous)[0, 1])
    OBSERVED["polyserial"] = polyserial
    OBSERVED["pearson_ordinal_continuous"] = pearson
    assert abs(polyserial - 0.6) < 0.02, polyserial
    assert pearson < polyserial - 0.02, (pearson, polyserial)
    with pytest.raises(SpecificationError):
        polyserial_correlation(np.full(100, 2), rng.normal(size=100))


# --------------------------------------------------------------------------- #
# Polychoric matrix
# --------------------------------------------------------------------------- #


def test_polychoric_matrix_is_symmetric_unit_diagonal_and_pd() -> None:
    rng = np.random.default_rng(97)
    R = two_factor_population(0.8, 0.5)
    names = [f"A{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
    z = rng.multivariate_normal(np.zeros(8), R, size=1500)
    frame = pd.DataFrame({name: _categorise(z[:, j], 5)
                          for j, name in enumerate(names)})

    start = time.perf_counter()
    matrix = polychoric_matrix(frame)
    elapsed = time.perf_counter() - start
    values = matrix.to_numpy(dtype=float)

    assert list(matrix.columns) == names and list(matrix.index) == names
    assert np.allclose(values, values.T, atol=1e-12)
    assert np.allclose(np.diag(values), 1.0, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(values)
    OBSERVED["polychoric_matrix_min_eigenvalue"] = float(eigenvalues.min())
    OBSERVED["polychoric_matrix_seconds"] = elapsed
    assert eigenvalues.min() > 0.0, eigenvalues.min()
    off = ~np.eye(8, dtype=bool)
    assert np.all(np.abs(values[off]) <= MAX_ABS_RHO + 1e-12)

    # It must be closer to the generating matrix than Pearson is.
    pearson = frame.corr().to_numpy(dtype=float)
    poly_error = float(np.abs(values[off] - R[off]).mean())
    pearson_error = float(np.abs(pearson[off] - R[off]).mean())
    OBSERVED["poly_matrix_mean_abs_error"] = poly_error
    OBSERVED["pearson_matrix_mean_abs_error"] = pearson_error
    assert poly_error < pearson_error, (poly_error, pearson_error)


def test_polychoric_matrix_mixes_ordinal_and_continuous_columns() -> None:
    rng = np.random.default_rng(64)
    rho = 0.5
    z = rng.multivariate_normal(np.zeros(3),
                                np.full((3, 3), rho) + np.eye(3) * (1 - rho),
                                size=4000)
    frame = pd.DataFrame({
        "ord1": _categorise(z[:, 0], 5),
        "ord2": _categorise(z[:, 1], 5),
        "cont": z[:, 2],
    })
    matrix = polychoric_matrix(frame, ordinal_columns=["ord1", "ord2"])
    assert abs(float(matrix.loc["ord1", "ord2"]) - rho) < 0.04
    assert abs(float(matrix.loc["ord1", "cont"]) - rho) < 0.04
    assert abs(float(matrix.loc["cont", "cont"]) - 1.0) < 1e-12
    with pytest.raises(SpecificationError):
        polychoric_matrix(frame[["ord1"]])


def test_analyse_with_polychoric_recovers_loadings_from_likert_data() -> None:
    """The point of the option: categorisation no longer biases CR and AVE."""
    rng = np.random.default_rng(80808)
    spec = two_factor_spec(k=4, loading=0.8)
    names = spec.all_item_names()
    R = two_factor_population(0.8, 0.5)
    z = rng.multivariate_normal(np.zeros(8), R, size=2000)
    frame = pd.DataFrame({name: _categorise(z[:, j], 5)
                          for j, name in enumerate(names)})

    ordinal = analyse(frame, spec, use_polychoric=True)
    naive = analyse(frame, spec, use_polychoric=False)
    OBSERVED["ave_polychoric"] = ordinal.per_factor[0].ave
    OBSERVED["ave_pearson_on_likert"] = naive.per_factor[0].ave
    assert abs(ordinal.per_factor[0].ave - 0.64) < 0.05, \
        ordinal.per_factor[0].ave
    assert naive.per_factor[0].ave < ordinal.per_factor[0].ave
    assert abs(float(ordinal.htmt.iloc[0, 1]) - 0.5) < 0.06


def test_analyse_consumes_a_real_estimation_result() -> None:
    """End-to-end with the real estimator, skipped while it does not exist.

    :func:`analyse` must read an actual ``EstimationResult.loadings`` frame,
    not just the duck-typed stand-in above.  The estimator is imported inside
    the test so this module stays importable without it.
    """
    try:
        from semsim.analysis.estimator import estimate
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"estimator unavailable: {type(exc).__name__}: {exc}")

    rng = np.random.default_rng(2)
    spec = two_factor_spec(k=4, loading=0.8)
    spec.add_path("FA", "FB", coefficient=0.5)
    names = spec.all_item_names()
    R = two_factor_population(0.8, 0.5)
    frame = pd.DataFrame(rng.multivariate_normal(np.zeros(8), R, size=800),
                         columns=names)
    result = estimate(frame, spec)
    report = analyse(frame, spec, result)

    standardised = {str(row.item): float(row.std)
                    for row in result.loadings.itertuples()}
    for entry in report.per_factor:
        lam = np.array([standardised[name]
                        for name in entry.item_total_correlations])
        assert abs(entry.ave - float(np.mean(lam ** 2))) < 1e-9, entry.factor_id
        assert abs(entry.composite_reliability
                   - composite_reliability(lam)) < 1e-9
    OBSERVED["estimator_ave"] = report.per_factor[0].ave


# --------------------------------------------------------------------------- #
# Regression tests: the seven defects repaired in this module
# --------------------------------------------------------------------------- #


def _reverse_coded_population(loading: float = 0.8, phi: float = 0.3,
                              k: int = 4) -> np.ndarray:
    """Two congeneric factors whose first item of FA is reverse-worded.

    The compiler models a reverse-worded item as a negated loading (see
    ``REVERSE_CODED`` in :mod:`semsim.core.compile`), so until the item is
    recoded its raw responses run *against* its own factor.  That is the
    ordinary simulated data on which the sign defects below fire.
    """
    L = np.zeros((2 * k, 2))
    L[:k, 0] = loading
    L[0, 0] = -loading
    L[k:, 1] = loading
    Phi = np.array([[1.0, phi], [phi, 1.0]])
    R = L @ Phi @ L.T
    np.fill_diagonal(R, 1.0)
    return R


def test_htmt_verdict_is_taken_on_the_magnitude_not_the_signed_ratio() -> None:
    """Defect 1: two constructs correlated -0.95 must fail, not pass.

    Discriminant validity is a claim about how *strongly* two constructs are
    related.  HTMT = -0.944 is as fatal as +0.944; comparing the signed ratio
    against 0.85 let the worst case through untouched.  The reported matrix
    must still carry the sign, so the direction stays visible.
    """
    rng = np.random.default_rng(909)
    spec = two_factor_spec(k=4, loading=0.7)
    names = spec.all_item_names()
    L = np.zeros((8, 2))
    L[:4, 0] = 0.7
    L[4:, 1] = 0.7
    Phi = np.array([[1.0, -0.95], [-0.95, 1.0]])
    R = L @ Phi @ L.T
    np.fill_diagonal(R, 1.0)
    x = rng.multivariate_normal(np.zeros(8), R, size=3000)
    report = analyse(pd.DataFrame(x, columns=names), spec,
                     htmt_threshold=0.85)

    value = float(report.htmt.iloc[0, 1])
    OBSERVED["htmt_negative_pair"] = value
    assert value < -0.85, value                 # the sign is preserved
    assert report.htmt_passed is False, value
    assert any("HTMT" in flag for flag in report.flags)
    # And a genuinely distinct pair still passes, so the fix is not a blanket
    # failure of everything.
    clean = analyse(pd.DataFrame(
        rng.multivariate_normal(np.zeros(8), two_factor_population(0.7, 0.3),
                                size=3000), columns=names), spec)
    assert clean.htmt_passed is True


def test_htmt_block_survives_two_factors_sharing_a_display_label() -> None:
    """Defect 2: the HTMT item mapping must be keyed by factor id.

    Labels are free text, so two constructs can carry the same one.  Keying
    the mapping by label collapsed the two entries into one and the call died
    with "at least two factors are needed".
    """
    rng = np.random.default_rng(4242)
    spec = ModelSpec()
    spec.add_construct("FA", label_en="Trust", n_items=4, loading=0.8)
    spec.add_construct("FB", label_en="Trust", n_items=4, loading=0.8)
    names = spec.all_item_names()
    x = rng.multivariate_normal(np.zeros(8),
                                two_factor_population(0.8, 0.3), size=1500)
    report = analyse(pd.DataFrame(x, columns=names), spec)

    assert report.htmt.shape == (2, 2)
    observed = float(report.htmt.to_numpy(dtype=float)[0, 1])
    OBSERVED["htmt_shared_label"] = observed
    assert abs(observed - 0.3) < 0.05, observed
    # The axes still read in the display labels the rest of the report uses.
    assert list(report.htmt.index) == ["Trust", "Trust"]


def test_cross_loading_flag_compares_magnitude_with_magnitude() -> None:
    """Defect 3: a reverse-keyed item is not a cross-loading.

    ``max_cross_correlation`` is a maximum over *absolute* foreign
    correlations, so it has to be compared against the *absolute* home
    correlation.  Compared against the signed one, every item with a negative
    home correlation was flagged however weakly it touched any other factor.
    The negative home correlation gets its own column and its own flag.
    """
    rng = np.random.default_rng(1515)
    spec = two_factor_spec(k=4, loading=0.8)
    names = spec.all_item_names()
    R = _reverse_coded_population(0.8, 0.3)
    x = rng.multivariate_normal(np.zeros(8), R, size=2500)
    report = analyse(pd.DataFrame(x, columns=names), spec)

    table = report.cross_loading_table.set_index("item")
    row = table.loc[names[0]]
    home = float(row["home_correlation"])
    cross = float(row["max_cross_correlation"])
    OBSERVED["reverse_item_home_r"] = home
    OBSERVED["reverse_item_max_cross_r"] = cross
    assert home < -0.3, home                    # it really is reverse-keyed
    assert cross < abs(home) - 0.10, (cross, home)
    assert bool(row["flagged"]) is False        # no spurious cross-loading
    assert bool(row["negative_home_correlation"]) is True
    assert not table["flagged"].any()
    assert int(table["negative_home_correlation"].sum()) == 1
    assert any("negatively" in flag for flag in report.flags)
    for flag in report.flags:
        assert " | " in flag and any(ord(ch) > 1500 for ch in flag), flag

    # A real cross-loading is still caught: an item built half from each
    # factor correlates with the foreign one about as strongly as with its own.
    mixed = x.copy()
    mixed[:, 0] = x[:, 1] * 0.5 + x[:, 4] * 0.5
    flagged = analyse(pd.DataFrame(mixed, columns=names), spec)
    first = flagged.cross_loading_table.set_index("item").loc[names[0]]
    assert bool(first["flagged"]) is True, dict(first)


def test_heywood_case_is_reported_rather_than_clipped_away() -> None:
    """Defect 4: a negative unique variance must reach the report.

    ``composite_reliability`` clips a negative ``theta`` to keep its
    denominator usable -- with every ``theta`` negative that returns exactly
    1.0, i.e. "perfect reliability" precisely when the solution is
    inadmissible.  The clipped value is still returned, but the fact is now
    recoverable with :func:`heywood_items` and :func:`analyse` names the
    factor and the item.
    """
    assert composite_reliability([0.8] * 4, [-0.1] * 4) == 1.0
    assert heywood_items([0.8] * 4, [-0.1] * 4) == [0, 1, 2, 3]
    assert heywood_items([0.8, 0.8], [-0.2, 0.36]) == [0]
    assert heywood_items([1.04, 0.8, 0.8]) == [0]
    assert heywood_items([0.8, 0.8, 0.8]) == []
    assert heywood_items([1.0]) == []           # a fixed single indicator
    with pytest.raises(SpecificationError):
        heywood_items([0.8, 0.8], [0.3])

    rng = np.random.default_rng(72)
    spec = two_factor_spec(k=4, loading=0.8)
    names = spec.all_item_names()

    class HeywoodEstimation:
        loadings = pd.DataFrame({
            "item": names,
            "factor": ["FA"] * 4 + ["FB"] * 4,
            "std": [1.04, 0.8, 0.8, 0.8] + [0.8] * 4,
        })

    x = rng.multivariate_normal(np.zeros(8), two_factor_population(0.8, 0.3),
                                size=600)
    report = analyse(pd.DataFrame(x, columns=names), spec, HeywoodEstimation())
    heywood_flags = [flag for flag in report.flags if "Heywood" in flag]
    OBSERVED["heywood_flags"] = float(len(heywood_flags))
    assert len(heywood_flags) == 1, report.flags
    assert names[0] in heywood_flags[0] and "FA" in heywood_flags[0]
    assert "هيوود" in heywood_flags[0]
    # The coefficients stay computable: the loading is still clipped.
    assert 0.0 < report.per_factor[0].composite_reliability <= 1.0
    # A proper solution raises no such flag.
    clean = analyse(pd.DataFrame(x, columns=names), spec)
    assert not any("Heywood" in flag for flag in clean.flags)


def test_htmt_matrix_leaves_the_callers_correlation_frame_alone() -> None:
    """Defect 5: ``correlations`` is read, never rewritten.

    The function stringifies its axis labels; doing that to the caller's own
    object left a correlation frame whose integer labels had silently become
    strings, so the caller could no longer index it.
    """
    names = [1, 2, 3, 4]
    corr = pd.DataFrame(np.array([[1.0, 0.6, 0.2, 0.2],
                                  [0.6, 1.0, 0.2, 0.2],
                                  [0.2, 0.2, 1.0, 0.6],
                                  [0.2, 0.2, 0.6, 1.0]]),
                        index=names, columns=names)
    snapshot = corr.copy(deep=True)
    table = htmt_matrix(pd.DataFrame(), {"A": [1, 2], "B": [3, 4]},
                        correlations=corr)

    assert list(corr.index) == names, list(corr.index)
    assert list(corr.columns) == names, list(corr.columns)
    assert corr.equals(snapshot)
    assert float(corr.loc[1, 2]) == 0.6          # still indexable as before
    # The result itself is unchanged by the copy.
    assert abs(float(table.loc["A", "B"]) - 0.2 / 0.6) < 1e-12


def test_polychoric_validates_before_short_circuiting_on_identity() -> None:
    """Defect 6: a degenerate variable is rejected even against itself.

    The identity short-circuit used to run before every check, so a constant
    column correlated with itself returned 1.0 -- a correlation that is
    undefined, there being no variance -- and an all-NaN column returned 1.0
    on the strength of no observations at all.
    """
    with pytest.raises(SpecificationError) as constant:
        polychoric_correlation(np.full(200, 3.0), np.full(200, 3.0))
    assert any(ord(ch) > 1500 for ch in constant.value.message_ar)
    with pytest.raises(SpecificationError):
        polychoric_correlation(np.full(200, np.nan), np.full(200, np.nan))
    with pytest.raises(SpecificationError):      # one complete pair only
        x = np.full(200, np.nan)
        x[0] = 2.0
        polychoric_correlation(x, x)

    # A well-behaved ordinal item against itself is still exactly 1.0.
    rng = np.random.default_rng(66)
    good = rng.integers(1, 6, size=400).astype(float)
    assert polychoric_correlation(good, good) == 1.0
    good[::13] = np.nan
    assert polychoric_correlation(good, good) == 1.0


def test_single_item_factor_matches_its_documented_values() -> None:
    """Defect 7: AVE = CR = omega = 1 exactly, as the docstring states.

    A single indicator is not estimated: the standard specification fixes
    ``lambda = 1`` and ``theta = 0``, so these coefficients are one by
    definition.  The generic loading clip (a guard against dividing by zero
    in the *estimated* case, which cannot arise here) dragged them to
    0.998001, a number matching neither the documentation nor the ``omega =
    1`` printed beside it.
    """
    rng = np.random.default_rng(1001)
    spec = ModelSpec()
    spec.add_construct("SINGLE", n_items=1, loading=0.8)
    spec.add_construct("FB", n_items=4, loading=0.8)
    names = spec.all_item_names()
    x = rng.normal(size=(300, len(names)))
    report = analyse(pd.DataFrame(x, columns=names), spec)

    entry = report.per_factor[0]
    assert entry.n_items == 1
    assert entry.ave == 1.0
    assert entry.composite_reliability == 1.0
    assert entry.omega_total == 1.0
    assert entry.sqrt_ave == 1.0
    assert np.isnan(entry.cronbach_alpha)
    assert float(report.fornell_larcker.iloc[0, 0]) == 1.0
    assert any("single item" in flag for flag in report.flags)
    # The multi-item factor beside it is untouched by the special case.
    assert report.per_factor[1].composite_reliability < 1.0


# --------------------------------------------------------------------------- #
# Stand-alone runner (used when pytest is unavailable)
# --------------------------------------------------------------------------- #


def _main() -> int:
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print("\nObserved values")
    for key in sorted(OBSERVED):
        print(f"  {key} = {OBSERVED[key]!r}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

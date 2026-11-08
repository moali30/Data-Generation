"""Numerical tests for :mod:`semsim.generation.thresholds`.

Every assertion is against a closed-form value, a published result, or a
large-sample simulation -- never merely "it ran".  Checks of the simulation
kind use an explicit ``numpy.random.Generator`` with a fixed seed, so a failure
is reproducible.

Closed-form anchors used here
-----------------------------
* ``UNIFORM`` cut points must equal ``Phi^-1(k / K)`` exactly.
* Proportions ``(.1, .2, .3, .4)`` must give ``Phi^-1`` of ``(.1, .3, .6)``.
* Median-split attenuation ``kappa = sqrt(2 / pi) = 0.797885`` (Cox, 1974).
* Five-point equal-probability attenuation ``kappa = 0.942297``.
* Pearson r between two median-split normals is ``(2 / pi) arcsin(rho)``
  (the tetrachoric relation; Pearson, 1900), which bounds how good the
  first-order ``kappa_i * kappa_j`` approximation can be.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

from semsim.core.errors import SpecificationError
from semsim.core.spec import (
    Bilingual, ResponseStyle, ScaleKind, ScaleSpec,
)
from semsim.generation.thresholds import (
    MAX_CATEGORIES,
    MIN_CATEGORIES,
    STYLE_BETA_PARAMS,
    achieved_proportions,
    apply_scale,
    categorise,
    category_probabilities,
    polychoric_attenuation,
    proportions_for_style,
    reverse_code,
    scale_thresholds,
    thresholds_for_style,
    thresholds_from_proportions,
)

SEED = 20240101
N_BIG = 200_000
STYLES = [s for s in ResponseStyle if s is not ResponseStyle.CUSTOM]
CATEGORY_COUNTS = list(range(MIN_CATEGORIES, MAX_CATEGORIES + 1))


@pytest.fixture(scope="module")
def big_normal() -> np.ndarray:
    """One shared sample of 200 000 standard normal draws."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(N_BIG)


# --------------------------------------------------------------------------- #
# Shape of the threshold vector
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.value)
@pytest.mark.parametrize("k", CATEGORY_COUNTS)
def test_threshold_vector_shape_and_monotonicity(style: ResponseStyle, k: int) -> None:
    tau = thresholds_for_style(k, style)
    assert tau.shape == (k - 1,), f"{style.value} K={k}: got {tau.shape}"
    assert np.all(np.isfinite(tau)), f"{style.value} K={k}: non-finite {tau}"
    if tau.size > 1:
        gaps = np.diff(tau)
        assert np.all(gaps > 0.0), f"{style.value} K={k}: gaps {gaps}"


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.value)
@pytest.mark.parametrize("k", CATEGORY_COUNTS)
def test_proportions_are_a_distribution(style: ResponseStyle, k: int) -> None:
    p = proportions_for_style(k, style)
    assert p.shape == (k,)
    assert np.all(p > 0.0)
    assert abs(p.sum() - 1.0) < 1e-12
    # Thresholds must reproduce the proportions they were built from.
    implied = category_probabilities(thresholds_for_style(k, style))
    assert np.max(np.abs(implied - p)) < 1e-9


# --------------------------------------------------------------------------- #
# Closed-form anchors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", CATEGORY_COUNTS)
def test_uniform_cuts_are_exactly_inverse_phi_of_k_over_K(k: int) -> None:
    tau = thresholds_for_style(k, ResponseStyle.UNIFORM)
    expected = stats.norm.ppf(np.arange(1, k) / k)
    assert np.allclose(tau, expected, atol=1e-12), f"K={k}: {tau} vs {expected}"


def test_thresholds_from_proportions_worked_example() -> None:
    tau = thresholds_from_proportions([0.1, 0.2, 0.3, 0.4])
    expected = stats.norm.ppf([0.1, 0.3, 0.6])
    assert np.allclose(tau, expected, atol=1e-12)
    assert np.allclose(np.round(tau, 4), [-1.2816, -0.5244, 0.2533], atol=1e-4)


def test_zero_proportions_do_not_produce_infinite_thresholds() -> None:
    tau = thresholds_from_proportions([0.0, 0.5, 0.5, 0.0])
    assert np.all(np.isfinite(tau)), tau
    assert np.all(np.diff(tau) > 0.0), tau
    assert abs(tau[1]) < 1e-6, tau            # the median cut stays at zero
    assert tau[0] < -4.0 and tau[2] > 4.0, tau


def test_symmetric_is_centred_and_bell_shaped() -> None:
    for k in (5, 7, 9):
        p = proportions_for_style(k, ResponseStyle.SYMMETRIC)
        assert np.allclose(p, p[::-1], atol=1e-12), f"K={k} not symmetric: {p}"
        mid = k // 2
        assert p[mid] == p.max(), f"K={k} not unimodal at the centre: {p}"
        assert np.all(np.diff(p[: mid + 1]) > 0.0), f"K={k} not bell-shaped: {p}"
        tau = thresholds_for_style(k, ResponseStyle.SYMMETRIC)
        assert np.allclose(tau, -tau[::-1], atol=1e-12)


def test_skew_styles_put_mass_on_the_right_end() -> None:
    for k in (5, 7):
        low = proportions_for_style(k, ResponseStyle.POSITIVE_SKEW)
        high = proportions_for_style(k, ResponseStyle.NEGATIVE_SKEW)
        assert np.allclose(low, high[::-1], atol=1e-12)
        half = k // 2
        assert low[:half].sum() > 0.6, f"K={k} floor effect too weak: {low}"
        assert high[-half:].sum() > 0.6, f"K={k} ceiling effect too weak: {high}"
        xlow = proportions_for_style(k, ResponseStyle.EXTREME_POSITIVE)
        assert xlow[0] > low[0], f"K={k} EXTREME_POSITIVE not more extreme"


def test_bimodal_has_a_trough_in_the_middle() -> None:
    for k in (5, 7, 9, 11):
        p = proportions_for_style(k, ResponseStyle.BIMODAL)
        mid = k // 2
        assert p[mid] == p.min(), f"K={k} no central trough: {p}"
        assert p[0] > 3.0 * p[mid], f"K={k} ends not dominant: {p}"


# --------------------------------------------------------------------------- #
# Large-sample recovery of the intended proportions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.value)
@pytest.mark.parametrize("k", CATEGORY_COUNTS)
def test_simulation_reproduces_intended_proportions(
    style: ResponseStyle, k: int, big_normal: np.ndarray
) -> None:
    tau = thresholds_for_style(k, style)
    values = categorise(big_normal, tau, base=1)
    observed = achieved_proportions(values, k)
    target = proportions_for_style(k, style)
    worst = float(np.max(np.abs(observed - target)))
    assert worst < 0.01, (
        f"{style.value} K={k}: max |observed - target| = {worst:.5f}\n"
        f"  target   = {np.round(target, 4)}\n"
        f"  observed = {np.round(observed, 4)}"
    )


def test_observed_shape_statistics(big_normal: np.ndarray) -> None:
    """Skewness / excess kurtosis of the categorised variable, K = 5."""
    report = {}
    for style in STYLES:
        values = categorise(big_normal, thresholds_for_style(5, style), base=1)
        report[style.value] = (
            float(stats.skew(values)),
            float(stats.kurtosis(values, fisher=True)),
        )
    print("\n  observed moments at K=5 (skew, excess kurtosis):")
    for name, (sk, ku) in report.items():
        print(f"    {name:18s} skew={sk:+.4f}  excess kurtosis={ku:+.4f}")

    assert report["positive_skew"][0] > 0.3, report["positive_skew"]
    assert report["negative_skew"][0] < -0.3, report["negative_skew"]
    assert report["extreme_positive"][0] > 0.3, report["extreme_positive"]
    assert report["extreme_negative"][0] < -0.3, report["extreme_negative"]
    assert report["bimodal"][1] < 0.0, report["bimodal"]
    assert abs(report["symmetric"][0]) < 0.02, report["symmetric"]
    assert abs(report["uniform"][0]) < 0.02, report["uniform"]


# --------------------------------------------------------------------------- #
# categorise
# --------------------------------------------------------------------------- #


def test_categorise_respects_base() -> None:
    tau = thresholds_for_style(5, ResponseStyle.SYMMETRIC)
    rng = np.random.default_rng(7)
    y = rng.standard_normal(5000)

    one = categorise(y, tau, base=1)
    zero = categorise(y, tau, base=0)

    assert one.min() == 1 and one.max() == 5
    assert zero.min() == 0 and zero.max() == 4
    assert np.array_equal(one, zero + 1)
    assert np.issubdtype(one.dtype, np.integer)


def test_categorise_boundaries_are_exact() -> None:
    tau = np.array([-1.0, 0.0, 1.0])
    y = np.array([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0])
    # x = k iff tau_{k-1} < y <= tau_k
    assert np.array_equal(categorise(y, tau, base=1), [1, 1, 2, 2, 3, 3, 4])


def test_categorise_is_shape_preserving_and_fast() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal((500, 60))
    tau = thresholds_for_style(7, ResponseStyle.NEGATIVE_SKEW)
    t0 = time.perf_counter()
    out = categorise(y, tau)
    elapsed = time.perf_counter() - t0
    print(f"\n  500x60 categorise: {elapsed * 1e3:.3f} ms")
    assert out.shape == (500, 60)
    assert elapsed < 0.2


def test_categorise_preserves_missing_cells() -> None:
    tau = thresholds_for_style(5, ResponseStyle.SYMMETRIC)
    y = np.array([-2.0, np.nan, 2.0])
    out = categorise(y, tau)
    assert out[0] == 1 and out[2] == 5
    assert np.isnan(out[1])


# --------------------------------------------------------------------------- #
# apply_scale / reverse_code / achieved_proportions
# --------------------------------------------------------------------------- #


def test_apply_scale_continuous_is_exactly_mean_plus_sd_times_y() -> None:
    rng = np.random.default_rng(3)
    y = rng.standard_normal(1000)
    scale = ScaleSpec(kind=ScaleKind.CONTINUOUS, mean=12.5, sd=2.75)
    out = apply_scale(y, scale)
    assert np.array_equal(out, 12.5 + 2.75 * y)      # bit-for-bit, not approx


def test_apply_scale_likert_uses_style_and_stays_in_range() -> None:
    rng = np.random.default_rng(4)
    y = rng.standard_normal(20_000)
    scale = ScaleSpec(kind=ScaleKind.LIKERT, n_categories=7,
                      response_style=ResponseStyle.POSITIVE_SKEW)
    out = apply_scale(y, scale)
    assert out.min() >= 1 and out.max() <= 7
    assert np.array_equal(out, categorise(y, thresholds_for_style(
        7, ResponseStyle.POSITIVE_SKEW), base=1))
    assert float(stats.skew(out)) > 0.3


def test_scale_spec_precedence_explicit_then_proportions_then_style() -> None:
    explicit = [-1.0, 0.0, 0.5, 1.5]
    s = ScaleSpec(n_categories=5, thresholds=list(explicit),
                  category_proportions=[0.2] * 5,
                  response_style=ResponseStyle.BIMODAL)
    assert np.allclose(scale_thresholds(s), explicit)

    s2 = ScaleSpec(n_categories=4, category_proportions=[0.1, 0.2, 0.3, 0.4],
                   response_style=ResponseStyle.BIMODAL)
    assert np.allclose(scale_thresholds(s2), stats.norm.ppf([0.1, 0.3, 0.6]))

    s3 = ScaleSpec(n_categories=5, response_style=ResponseStyle.UNIFORM)
    assert np.allclose(scale_thresholds(s3), stats.norm.ppf([0.2, 0.4, 0.6, 0.8]))


def test_reverse_code_is_an_involution_likert() -> None:
    rng = np.random.default_rng(5)
    for k in CATEGORY_COUNTS:
        scale = ScaleSpec(n_categories=k)
        values = rng.integers(1, k + 1, size=500)
        once = reverse_code(values, scale)
        twice = reverse_code(once, scale)
        assert once.min() >= 1 and once.max() <= k
        assert np.array_equal(twice, values), f"K={k} not an involution"
    # the mapping is the expected reflection
    scale = ScaleSpec(n_categories=5)
    assert np.array_equal(reverse_code(np.array([1, 2, 3, 4, 5]), scale),
                          [5, 4, 3, 2, 1])


def test_reverse_code_is_an_involution_continuous() -> None:
    rng = np.random.default_rng(6)
    scale = ScaleSpec(kind=ScaleKind.CONTINUOUS, mean=4.0, sd=1.5)
    values = 4.0 + 1.5 * rng.standard_normal(500)
    once = reverse_code(values, scale)
    assert np.allclose(reverse_code(once, scale), values, atol=1e-12)
    assert np.allclose(once, 8.0 - values, atol=1e-12)


def test_achieved_proportions_both_bases_and_missing() -> None:
    values = np.array([1, 1, 2, 3, 3, 3])
    p = achieved_proportions(values, 3)
    assert np.allclose(p, [2 / 6, 1 / 6, 3 / 6])
    assert np.allclose(achieved_proportions(values - 1, 3), [2 / 6, 1 / 6, 3 / 6])
    with_nan = np.array([1.0, np.nan, 3.0, 3.0])
    assert np.allclose(achieved_proportions(with_nan, 3), [1 / 3, 0.0, 2 / 3])


# --------------------------------------------------------------------------- #
# polychoric_attenuation -- the honest check
# --------------------------------------------------------------------------- #


def test_attenuation_closed_form_anchors() -> None:
    # Median split of a normal: kappa = E|z| / sd = sqrt(2 / pi).
    assert polychoric_attenuation(np.array([0.0])) == pytest.approx(
        np.sqrt(2.0 / np.pi), abs=1e-12)
    # Five equal-probability categories.
    k5 = polychoric_attenuation(thresholds_for_style(5, ResponseStyle.UNIFORM))
    assert k5 == pytest.approx(0.9422967954, abs=1e-9)
    print(f"\n  kappa (2-cat median split) = {np.sqrt(2 / np.pi):.6f}"
          f"\n  kappa (5-cat equal prob)   = {k5:.6f}")


def test_attenuation_increases_with_more_categories() -> None:
    kappas = [polychoric_attenuation(
        thresholds_for_style(k, ResponseStyle.UNIFORM)) for k in CATEGORY_COUNTS]
    print("\n  kappa by K (uniform): " + ", ".join(
        f"K={k}:{v:.4f}" for k, v in zip(CATEGORY_COUNTS, kappas)))
    assert all(0.0 < v <= 1.0 for v in kappas)
    assert np.all(np.diff(kappas) > 0.0), kappas
    assert kappas[-1] > 0.96


@pytest.mark.parametrize("style", STYLES, ids=lambda s: s.value)
@pytest.mark.parametrize("k", [2, 3, 5, 7, 11])
def test_attenuation_matches_simulated_latent_correlation(
    style: ResponseStyle, k: int, big_normal: np.ndarray
) -> None:
    """corr(y*, categorised y*) must equal the predicted kappa."""
    tau = thresholds_for_style(k, style)
    predicted = polychoric_attenuation(tau)
    values = categorise(big_normal, tau, base=1)
    observed = float(np.corrcoef(big_normal, values.astype(float))[0, 1])
    assert abs(observed - predicted) < 0.01, (
        f"{style.value} K={k}: predicted {predicted:.5f}, "
        f"observed {observed:.5f}, diff {observed - predicted:+.5f}"
    )


def test_attenuation_product_predicts_pearson_between_two_ordinals() -> None:
    """kappa_i * kappa_j * rho is the first-order Pearson r of two ordinals.

    The exact answer for a median split of both variables is the tetrachoric
    relation ``(2 / pi) arcsin(rho)``; the first-order product is
    ``(2 / pi) rho``.  Both are checked against simulation, which is what
    establishes that ``kappa`` is doing real work.
    """
    rng = np.random.default_rng(99)
    rho = 0.5
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((400_000, 2)) @ chol.T

    rows = []
    for k in (2, 5, 7):
        tau = thresholds_for_style(k, ResponseStyle.SYMMETRIC)
        x1 = categorise(z[:, 0], tau).astype(float)
        x2 = categorise(z[:, 1], tau).astype(float)
        observed = float(np.corrcoef(x1, x2)[0, 1])
        kappa = polychoric_attenuation(tau)
        predicted = rho * kappa * kappa
        rows.append((k, kappa, predicted, observed))
        assert observed < rho, f"K={k}: categorisation did not attenuate"
        assert abs(observed - predicted) < 0.02, (k, predicted, observed)

    print("\n  two-ordinal attenuation, rho=0.5 (K, kappa, rho*kappa^2, observed):")
    for k, kappa, predicted, observed in rows:
        print(f"    K={k:2d}  kappa={kappa:.4f}  predicted={predicted:.4f}  "
              f"observed={observed:.4f}")

    # Exact dichotomous result, for reference: (2/pi) arcsin(rho).
    tau2 = np.array([0.0])
    d1 = categorise(z[:, 0], tau2).astype(float)
    d2 = categorise(z[:, 1], tau2).astype(float)
    exact = (2.0 / np.pi) * np.arcsin(rho)
    observed_2 = float(np.corrcoef(d1, d2)[0, 1])
    print(f"    median split: exact (2/pi)arcsin(rho)={exact:.4f}  "
          f"observed={observed_2:.4f}")
    assert abs(observed_2 - exact) < 0.005


# --------------------------------------------------------------------------- #
# User errors must surface as SemSimError with a real Arabic message
# --------------------------------------------------------------------------- #


def _arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


@pytest.mark.parametrize("bad_k", [-1, 0, 1, 12, 50])
def test_category_count_out_of_range_raises(bad_k: int) -> None:
    with pytest.raises(SpecificationError) as exc:
        thresholds_for_style(bad_k, ResponseStyle.SYMMETRIC)
    assert _arabic(exc.value.message_ar), exc.value.message_ar


def test_custom_style_without_proportions_raises() -> None:
    with pytest.raises(SpecificationError) as exc:
        thresholds_for_style(5, ResponseStyle.CUSTOM)
    assert _arabic(exc.value.message_ar)


@pytest.mark.parametrize("bad", [
    [0.5],                       # too few
    [0.5, 0.2],                  # does not sum to one
    [0.5, 0.6],                  # does not sum to one
    [-0.1, 0.6, 0.5],            # negative
    [np.nan, 0.5, 0.5],          # non-finite
])
def test_bad_proportions_raise(bad: list) -> None:
    with pytest.raises(SpecificationError) as exc:
        thresholds_from_proportions(bad)
    assert _arabic(exc.value.message_ar)


def test_bad_thresholds_raise() -> None:
    for bad in ([], [0.0, -1.0], [0.0, 0.0], [np.inf], [0.0, np.nan]):
        with pytest.raises(SpecificationError) as exc:
            categorise(np.zeros(3), np.asarray(bad, dtype=float))
        assert _arabic(exc.value.message_ar)


def test_scale_with_wrong_threshold_count_raises() -> None:
    s = ScaleSpec(n_categories=5, thresholds=[-1.0, 0.0])
    with pytest.raises(SpecificationError) as exc:
        scale_thresholds(s)
    assert _arabic(exc.value.message_ar)
    s2 = ScaleSpec(n_categories=5, category_proportions=[0.5, 0.5])
    with pytest.raises(SpecificationError):
        scale_thresholds(s2)


def test_achieved_proportions_out_of_range_raises() -> None:
    with pytest.raises(SpecificationError) as exc:
        achieved_proportions(np.array([1, 2, 99]), 5)
    assert _arabic(exc.value.message_ar)


def test_degenerate_item_cannot_be_attenuated() -> None:
    # A cut so far out that essentially nobody crosses it still has variance,
    # so the degenerate case is reached only through zeroed proportions.
    tau = thresholds_from_proportions([1.0 - 2e-6, 1e-6, 1e-6])
    kappa = polychoric_attenuation(tau)
    assert 0.0 < kappa <= 1.0
    print(f"\n  kappa (near-degenerate 3-cat) = {kappa:.6f}")


def test_scale_spec_labels_are_untouched() -> None:
    """apply_scale must not mutate the spec it is handed."""
    scale = ScaleSpec(n_categories=5, min_label=Bilingual(ar="أرفض تماماً"),
                      max_label=Bilingual(ar="أوافق تماماً"))
    before = (scale.n_categories, scale.thresholds, scale.category_proportions)
    apply_scale(np.linspace(-2, 2, 50), scale)
    assert (scale.n_categories, scale.thresholds,
            scale.category_proportions) == before


# --------------------------------------------------------------------------- #
# Regression tests for the repaired defects
# --------------------------------------------------------------------------- #


def _exact_ordinal_correlation(tau: np.ndarray, rho: float) -> float:
    """Exact Pearson r between two identically categorised standard normals.

    Built from bivariate-normal orthant probabilities rather than simulation,
    so there is no Monte-Carlo noise to tolerate: the cell probabilities come
    from the joint CDF evaluated on the grid of cut points,

        P(x1 = a, x2 = b) = F(t_a, t_b) - F(t_{a-1}, t_b)
                            - F(t_a, t_{b-1}) + F(t_{a-1}, t_{b-1}),

    which is the quantity the product rule ``rho * kappa_i * kappa_j``
    approximates.
    """
    from scipy.stats import multivariate_normal

    edges = np.concatenate(([-np.inf], np.asarray(tau, dtype=float), [np.inf]))
    k = edges.size - 1
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])

    cdf = np.empty((k + 1, k + 1))
    for a in range(k + 1):
        for b in range(k + 1):
            ha, hb = edges[a], edges[b]
            if ha == -np.inf or hb == -np.inf:
                cdf[a, b] = 0.0
            elif ha == np.inf and hb == np.inf:
                cdf[a, b] = 1.0
            elif ha == np.inf:
                cdf[a, b] = float(stats.norm.cdf(hb))
            elif hb == np.inf:
                cdf[a, b] = float(stats.norm.cdf(ha))
            else:
                cdf[a, b] = float(mvn.cdf([ha, hb]))

    joint = (cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1])
    codes = np.arange(1.0, k + 1.0)
    marginal = joint.sum(axis=1)
    mu = float(marginal @ codes)
    var = float(marginal @ (codes - mu) ** 2)
    return (float(codes @ joint @ codes) - mu * mu) / var


def test_product_rule_envelope_matches_the_documented_figures() -> None:
    """Pin the accuracy envelope that ``polychoric_attenuation`` documents.

    The docstring used to present ``r ~= rho * kappa_i * kappa_j`` as simply
    "the" first-order relation, with no statement of how far off it can be.
    It is accurate to a fraction of a percent for ``SYMMETRIC`` only; for the
    shipped extreme styles it is out by more than an order of magnitude more
    than that, which matters because the calibration loop pre-divides loadings
    by ``kappa``.  These bounds are the ones the docstring now quotes, so a
    future edit that re-asserts a uniformly tight envelope fails here.
    """
    # style -> (worst relative error over K in {5, 7} and rho in {.3,.5,.7})
    documented = {
        # style -> worst relative error over K in {5, 7} and rho in
        # {-.7, -.5, -.3, .3, .5, .7}. These are BOTH signs: the earlier
        # figures were measured on positive rho only, which understated the
        # asymmetric styles -- extreme_* is 14.2% at positive rho and 18.6%
        # once negative rho is included. The docstring quotes these.
        ResponseStyle.SYMMETRIC: 0.3,
        ResponseStyle.UNIFORM: 3.8,
        ResponseStyle.POSITIVE_SKEW: 4.3,
        ResponseStyle.NEGATIVE_SKEW: 4.3,
        ResponseStyle.BIMODAL: 8.2,
        ResponseStyle.EXTREME_POSITIVE: 18.6,
        ResponseStyle.EXTREME_NEGATIVE: 18.6,
    }
    print("\n  exact vs first-order product rule (worst over K=5,7 and "
          "rho = +/-.3, +/-.5, +/-.7):")
    worst_by_style = {}
    for style, bound in documented.items():
        worst = 0.0
        for k in (5, 7):
            tau = thresholds_for_style(k, style)
            kappa = polychoric_attenuation(tau)
            # Both signs. The old sweep tested positive rho only, which is
            # what let the docstring claim the error was one-signed
            # "throughout": for a style with asymmetric marginals a negative
            # correlation puts the joint mass in the sparse corners of the
            # table, and there the product rule OVER-predicts |r|.
            symmetric_marginals = style in (ResponseStyle.SYMMETRIC,
                                            ResponseStyle.UNIFORM,
                                            ResponseStyle.BIMODAL)
            for rho in (-0.7, -0.5, -0.3, 0.3, 0.5, 0.7):
                exact = _exact_ordinal_correlation(tau, rho)
                product = rho * kappa * kappa
                if rho > 0 or symmetric_marginals:
                    # Here the product rule drops the positive higher-order
                    # terms of the polychoric series and so under-predicts.
                    assert abs(product) <= abs(exact) + 1e-12, (
                        f"{style.value} K={k} rho={rho}: |product| "
                        f"{abs(product):.6f} exceeded |exact| {abs(exact):.6f}"
                    )
                # abs() on the denominator too: with a negative rho the exact
                # correlation is negative, so dividing by it gave a
                # NEGATIVE percentage that max() quietly discarded -
                # which is why the one-signedness claim survived.
                worst = max(worst,
                            100.0 * abs(exact - product) / abs(exact))
        worst_by_style[style] = worst
        print(f"    {style.value:18s} worst {worst:6.2f}%  "
              f"(documented bound {bound}%)")
        assert worst <= bound, (
            f"{style.value}: worst relative error {worst:.2f}% exceeds the "
            f"documented {bound}%"
        )

    # The whole point of the repaired documentation: SYMMETRIC is tight and
    # the extreme styles are nowhere near it.  A 30x gap, not a rounding one.
    assert worst_by_style[ResponseStyle.SYMMETRIC] < 0.5
    assert worst_by_style[ResponseStyle.EXTREME_POSITIVE] > 5.0
    assert worst_by_style[ResponseStyle.EXTREME_NEGATIVE] > 5.0
    assert worst_by_style[ResponseStyle.BIMODAL] > 5.0
    assert (worst_by_style[ResponseStyle.EXTREME_POSITIVE]
            > 30.0 * worst_by_style[ResponseStyle.SYMMETRIC])


def test_product_rule_envelope_at_three_categories() -> None:
    """The second column of the documented table: K = 3, |rho| <= 0.8."""
    documented = {
        ResponseStyle.SYMMETRIC: 2.7,
        ResponseStyle.UNIFORM: 8.0,
        ResponseStyle.POSITIVE_SKEW: 15.6,
        ResponseStyle.NEGATIVE_SKEW: 15.6,
        ResponseStyle.BIMODAL: 13.4,
        ResponseStyle.EXTREME_POSITIVE: 37.1,
        ResponseStyle.EXTREME_NEGATIVE: 37.1,
    }
    print("\n  exact vs product rule at K=3 (worst over rho=.1,.3,.5,.7,.8):")
    for style, bound in documented.items():
        tau = thresholds_for_style(3, style)
        kappa = polychoric_attenuation(tau)
        worst = 0.0
        for rho in (0.1, 0.3, 0.5, 0.7, 0.8):
            exact = _exact_ordinal_correlation(tau, rho)
            worst = max(worst, 100.0 * abs(exact - rho * kappa * kappa) / exact)
        print(f"    {style.value:18s} worst {worst:6.2f}%  "
              f"(documented bound {bound}%)")
        assert worst <= bound, f"{style.value}: {worst:.2f}% > {bound}%"
        # The bound must also be tight, or the documentation is not informative.
        assert worst > 0.6 * bound, (
            f"{style.value}: documented {bound}% is far looser than the "
            f"observed {worst:.2f}%"
        )


def test_product_rule_is_worst_for_a_dichotomised_extreme_style() -> None:
    """K = 2 on an extreme style is the documented worst case: ~66% out."""
    tau = thresholds_for_style(2, ResponseStyle.EXTREME_POSITIVE)
    kappa = polychoric_attenuation(tau)
    exact = _exact_ordinal_correlation(tau, 0.7)
    product = 0.7 * kappa * kappa
    rel = 100.0 * abs(exact - product) / exact
    print(f"\n  K=2 extreme_positive, rho=0.7: kappa={kappa:.4f} "
          f"exact={exact:.5f} product={product:.5f} rel={rel:.1f}%")
    assert abs(exact - 0.32580) < 1e-4
    assert abs(product - 0.11162) < 1e-4
    assert 60.0 < rel < 70.0


def test_attenuation_docstring_states_the_real_envelope() -> None:
    """The envelope must stay *written down*, not just true.

    The defect was documentary: the product rule was presented without its
    accuracy envelope, so the calibration loop had no warning that the
    correction is unreliable for the extreme styles.
    """
    doc = polychoric_attenuation.__doc__ or ""
    assert "first-order" in doc
    assert "Accuracy of the product rule" in doc   # the section heading
    for name in ("EXTREME_POSITIVE", "EXTREME_NEGATIVE", "BIMODAL",
                 "SYMMETRIC", "UNIFORM"):
        assert name in doc, name
    assert "14.2%" in doc and "0.3%" in doc
    assert "calibration loop" in doc


@pytest.mark.parametrize("bad", [5.9, 2.999, 4.5, 5.5, "5", "five", None,
                                 object()])
def test_non_integral_category_count_raises(bad: object) -> None:
    """int(5.9) == 5 used to pass silently; a whole number is now required.

    The error message always claimed the count "must be a whole number", but
    the implementation was a truncating ``int(...)``, so asking for 5.9
    categories produced a five-point scale with no warning.
    """
    with pytest.raises(SpecificationError) as exc:
        thresholds_for_style(bad, ResponseStyle.SYMMETRIC)  # type: ignore[arg-type]
    assert _arabic(exc.value.message_ar), exc.value.message_ar
    print(f"\n  n_categories={bad!r} -> {exc.value.message_en[:70]}")


@pytest.mark.parametrize("good", [5, 5.0, np.int64(7), np.float64(3.0)])
def test_integral_category_count_of_any_numeric_type_is_accepted(
    good: object,
) -> None:
    """Rejecting 5.0 would break float-typed form fields and numpy scalars."""
    tau = thresholds_for_style(good, ResponseStyle.SYMMETRIC)  # type: ignore[arg-type]
    assert tau.size == int(float(good)) - 1  # type: ignore[arg-type]


def test_non_integral_category_count_raises_everywhere_it_is_read() -> None:
    """Every entry point that reads ``n_categories`` must reject a fraction."""
    scale = ScaleSpec(n_categories=5.9)  # type: ignore[arg-type]
    for label, call in (
        ("scale_thresholds", lambda: scale_thresholds(scale)),
        ("apply_scale", lambda: apply_scale(np.zeros(4), scale)),
        ("reverse_code", lambda: reverse_code(np.array([3]), scale)),
        ("achieved_proportions",
         lambda: achieved_proportions(np.array([1, 2, 3]), 5.9)),  # type: ignore
    ):
        with pytest.raises(SpecificationError) as exc:
            call()
        assert _arabic(exc.value.message_ar), label
        print(f"  {label} rejected a fractional category count")


def test_achieved_proportions_takes_an_explicit_base() -> None:
    """Base-0 codes spanning 1..K-1 are ambiguous; the caller can now say.

    ``[1, 2, 3, 4]`` on a five-category item is a legal base-1 sample *and* a
    legal base-0 sample whose lowest category went unobserved.  Nothing in the
    data separates the two, so auto-detection cannot be made correct -- it can
    only be made explicit.  The old API offered no way to say which it was and
    always answered base-1, shifting every proportion by one category for
    base-0 input.
    """
    v = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    base1 = achieved_proportions(v, 5, base=1)
    base0 = achieved_proportions(v, 5, base=0)
    print(f"\n  codes {v.tolist()} on a 5-category item:"
          f"\n    base=1 -> {np.round(base1, 4)}"
          f"\n    base=0 -> {np.round(base0, 4)}")
    assert np.allclose(base1, [0.25, 0.25, 0.25, 0.25, 0.0])
    assert np.allclose(base0, [0.0, 0.25, 0.25, 0.25, 0.25])
    # The default keeps the historical behaviour, which is the base-1
    # tie-break -- and base-1 is what apply_scale emits, so for this engine's
    # own columns the default is the right answer.
    assert np.allclose(achieved_proportions(v, 5), base1)
    scored = apply_scale(np.linspace(-3, 3, 4000), ScaleSpec(n_categories=5))
    assert scored.min() >= 1 and scored.max() <= 5
    assert np.allclose(achieved_proportions(scored, 5),
                       achieved_proportions(scored, 5, base=1))


def test_achieved_proportions_base_is_validated_and_enforced() -> None:
    v = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    with pytest.raises(SpecificationError) as exc:
        achieved_proportions(v, 5, base=2)
    assert _arabic(exc.value.message_ar)
    print(f"\n  base=2 -> {exc.value.message_en}")
    # An explicit base that the data contradict must raise, not be silently
    # overridden by the detector.
    with pytest.raises(SpecificationError) as exc2:
        achieved_proportions(np.array([0, 1, 2]), 3, base=1)
    assert _arabic(exc2.value.message_ar)
    print(f"  base=1 on 0..2 codes -> {exc2.value.message_en}")
    # Unambiguous base-0 input (a zero is present) is still detected.
    assert np.allclose(achieved_proportions(np.array([0, 1, 2]), 3),
                       [1 / 3, 1 / 3, 1 / 3])


def test_symmetric_is_a_beta_bell_not_equal_probability_cuts() -> None:
    """SYMMETRIC and UNIFORM are different styles, and must stay different.

    ``docs/ARCHITECTURE.md`` used to describe SYMMETRIC as "equal-probability
    normal cuts", which is the definition of UNIFORM -- the two styles would
    have been the same thing under two names.  The implementation is the
    sensible one: SYMMETRIC is the equal-width discretisation of
    ``Beta(3.5, 3.5)``, a centred bell.
    """
    sym = proportions_for_style(5, ResponseStyle.SYMMETRIC)
    uni = proportions_for_style(5, ResponseStyle.UNIFORM)
    print(f"\n  SYMMETRIC K=5 = {np.round(sym, 4)}"
          f"\n  UNIFORM   K=5 = {np.round(uni, 4)}")
    assert np.allclose(uni, 0.2)
    assert not np.allclose(sym, uni, atol=0.05)
    # Bell-shaped: the middle category carries the most mass, the tails least,
    # and the profile is symmetric about the centre.
    assert sym.argmax() == 2
    assert np.allclose(sym, sym[::-1], atol=1e-12)
    assert sym[2] > 2.0 * sym[0]
    assert STYLE_BETA_PARAMS[ResponseStyle.SYMMETRIC] == (3.5, 3.5)
    # And the Beta shape is written down where a reader will look for it.
    assert "Beta(3.5, 3.5)" in (thresholds_for_style.__doc__ or "")

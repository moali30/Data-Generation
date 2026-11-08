"""
tests.test_power
================
Numerical correctness of :mod:`semsim.power.analysis`.

Nothing here checks "the code runs".  Every assertion is against a published
worked example, a closed form, or an independent recomputation:

*   **MacCallum, Browne & Sugawara (1996), Table 4** is reproduced entry by
    entry for both the test of close fit and the test of not-close fit.  That
    is the headline check: it can only pass if the null distribution is a
    **non-central** chi-square with ``lambda_0 = (N - 1) df epsilon_0^2``.  A
    central ``chi2.ppf`` critical value is explicitly shown to be wrong, by
    recomputing the published numbers with it and watching them collapse.
*   **Satorra & Saris (1985)** noncentrality is checked against a closed form.
    For two independent one-factor blocks whose population blocks are each
    exactly unidimensional, the restricted model (the path fixed to zero,
    hence the factors uncorrelated) has a block-diagonal implied matrix, the
    ML discrepancy separates, and its minimum is
    ``log|S_XX| + log|S_YY| - log|S|`` exactly.
*   **Monte Carlo** results are checked against the Satorra-Saris analytic
    power for the same model, which is a genuinely independent route to the
    same quantity.

A note on the task brief
------------------------
The brief quoted MacCallum et al. Table 4 as "1,238 at df=2; 464 at df=8; 314
at df=16; 231 at df=30; 171 at df=55; 132 at df=90".  Those ``N`` values are
(mostly) real Table 4 entries, but they are paired with the wrong degrees of
freedom: in the published table 1 238 belongs to ``df = 6``, 314 to
``df = 30``, 231 to ``df = 45`` and 132 to ``df = 100``.
:func:`test_maccallum_table_4_brief_pairs_are_mismatched` documents the
discrepancy by printing both columns, so the mismatch is recorded rather than
hidden; the correctness assertions use the published pairs.

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
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, ncx2

from semsim.core import examples
from semsim.core.compile import compile_model
from semsim.core.errors import InfeasibleTargetError, SpecificationError
from semsim.core.implied import implied_covariance
from semsim.core.spec import (
    ConstructRole,
    EffectSize,
    ModelSpec,
    PowerMethod,
    ScaleKind,
)
from semsim.power import analysis as power


# --------------------------------------------------------------------------- #
# Published reference values
# --------------------------------------------------------------------------- #

#: MacCallum, Browne & Sugawara (1996), Table 4, left panel: minimum N for
#: power .80 in the test of CLOSE fit (alpha = .05, eps_0 = .05, eps_a = .08).
MBS_TABLE4_CLOSE_FIT = {
    2: 3488, 4: 1807, 6: 1238, 8: 954, 10: 782, 12: 666, 14: 585, 16: 522,
    18: 473, 20: 435, 25: 363, 30: 314, 35: 279, 40: 252, 45: 231, 50: 214,
    55: 200, 60: 188, 65: 177, 70: 168, 75: 161, 80: 154, 85: 148, 90: 142,
    95: 137, 100: 132,
}

#: Same table, right panel: test of NOT-CLOSE fit (eps_0 = .05, eps_a = .01).
MBS_TABLE4_NOT_CLOSE_FIT = {
    2: 2382, 4: 1426, 6: 1069, 8: 875, 10: 750, 12: 663, 14: 598, 16: 547,
    20: 474, 30: 366, 40: 307, 50: 268, 55: 253, 60: 240, 70: 220, 80: 204,
    90: 190, 100: 178,
}

#: The (df, N) pairs the task brief quoted.  Kept so the mismatch is on the
#: record; see the module docstring.
BRIEF_PAIRS = {2: 1238, 8: 464, 16: 314, 30: 231, 55: 171, 90: 132}

TOLERANCE = 0.03


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _two_block_spec(beta: float, *, n_items: int = 4,
                    loading: float = 0.80) -> ModelSpec:
    """``X -> Y``, one four-item factor each, continuous, no extras.

    The restricted model (``beta`` fixed to zero) then has a block-diagonal
    implied covariance matrix, which is what makes the closed-form check in
    :func:`test_satorra_saris_matches_closed_form` possible.
    """
    spec = ModelSpec(seed=20240808)
    spec.add_construct("X", label_ar="المتغير المستقل", label_en="X",
                       role=ConstructRole.INDEPENDENT, n_items=n_items,
                       loading=loading)
    spec.add_construct("Y", label_ar="المتغير التابع", label_en="Y",
                       role=ConstructRole.DEPENDENT, n_items=n_items,
                       loading=loading)
    spec.add_path("X", "Y", effect_size=EffectSize.CUSTOM, coefficient=beta)
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    spec.distribution.use_non_normal_engine = False
    spec.ensure_unique_item_names()
    return spec


def _closed_form_restricted_fmin(sigma: np.ndarray, k: int) -> float:
    """Minimum ``F_ML`` of a block-diagonal restricted model, in closed form.

    For ``Sigma_r = blockdiag(A, B)``,

    ``F = log|A| + log|B| + tr(S_AA A^-1) + tr(S_BB B^-1) - log|S| - p``

    which separates, so each block is minimised at ``A = S_AA``,
    ``B = S_BB``.  Both blocks of the population matrix are exactly
    one-factor, so the restricted model attains them, leaving
    ``F = log|S_AA| + log|S_BB| - log|S|``.

    References
    ----------
    Joreskog, K. G. (1969). A general approach to confirmatory maximum
        likelihood factor analysis. *Psychometrika*, 34(2), 183-202.
    Satorra, A., & Saris, W. E. (1985). Power of the likelihood ratio test in
        covariance structure analysis. *Psychometrika*, 50(1), 83-90.
    """
    sign_a, log_a = np.linalg.slogdet(sigma[:k, :k])
    sign_b, log_b = np.linalg.slogdet(sigma[k:, k:])
    sign_s, log_s = np.linalg.slogdet(sigma)
    assert sign_a > 0 and sign_b > 0 and sign_s > 0
    return float(log_a + log_b - log_s)


# --------------------------------------------------------------------------- #
# 1. MacCallum et al. (1996), Table 4 -- the headline check
# --------------------------------------------------------------------------- #


def test_maccallum_table_4_close_fit():
    """Reproduce Table 4 (test of close fit) within 3 % at every tabulated df."""
    rows = []
    worst = 0.0
    for df, published in sorted(MBS_TABLE4_CLOSE_FIT.items()):
        obtained = power.n_for_close_fit_power(
            df, 0.80, rmsea_null=0.05, rmsea_alt=0.08, alpha=0.05)
        rel = abs(obtained - published) / published
        worst = max(worst, rel)
        rows.append((df, published, obtained, 100 * rel,
                     power.power_close_fit(df, obtained)))
    print("\nMacCallum, Browne & Sugawara (1996) Table 4 -- test of close fit")
    print("  df   published   obtained   dev%    power@obtained")
    for df, pub, obt, dev, pw in rows:
        print(f"{df:4d}   {pub:9d}   {obt:8d}   {dev:5.2f}   {pw:.4f}")
    print(f"  worst deviation: {100 * worst:.2f} %")
    for df, pub, obt, dev, _pw in rows:
        assert dev <= 100 * TOLERANCE, (
            f"df={df}: obtained N={obt}, published N={pub} "
            f"({dev:.2f} % off)")


def test_maccallum_table_4_not_close_fit():
    """Reproduce Table 4 (test of not-close fit) within 3 %."""
    rows = []
    for df, published in sorted(MBS_TABLE4_NOT_CLOSE_FIT.items()):
        obtained = power.n_for_not_close_fit_power(
            df, 0.80, rmsea_null=0.05, rmsea_alt=0.01, alpha=0.05)
        rel = abs(obtained - published) / published
        rows.append((df, published, obtained, 100 * rel))
    print("\nMacCallum et al. (1996) Table 4 -- test of NOT-close fit")
    print("  df   published   obtained   dev%")
    for df, pub, obt, dev in rows:
        print(f"{df:4d}   {pub:9d}   {obt:8d}   {dev:5.2f}")
    for df, pub, obt, dev in rows:
        assert dev <= 100 * TOLERANCE, (
            f"df={df}: obtained N={obt}, published N={pub}")


def test_maccallum_table_4_brief_pairs_are_mismatched():
    """Record that the brief's (df, N) pairs are not the published pairs.

    The brief's ``N`` values are real Table 4 entries, but attached to the
    wrong ``df``.  This test asserts the *documented* fact -- that each brief
    ``N`` is reproduced at some other df -- so the discrepancy is on the
    record and cannot be mistaken for a bug in the formula.
    """
    print("\nTask-brief pairs versus the published table")
    print("  df   brief N   obtained N   dev%    df where brief N is correct")
    inverse = {n: df for df, n in MBS_TABLE4_CLOSE_FIT.items()}
    for df, brief_n in sorted(BRIEF_PAIRS.items()):
        obtained = power.n_for_close_fit_power(df, 0.80)
        dev = 100 * abs(obtained - brief_n) / brief_n
        true_df = inverse.get(brief_n, "not a tabulated value")
        print(f"{df:4d}   {brief_n:7d}   {obtained:10d}   {dev:5.1f}   "
              f"{true_df}")
    # The four brief values that are genuine Table 4 entries belong to these df.
    assert power.n_for_close_fit_power(6, 0.80) == 1238
    assert power.n_for_close_fit_power(30, 0.80) == 314
    assert power.n_for_close_fit_power(45, 0.80) == 231
    assert power.n_for_close_fit_power(100, 0.80) == 132


def test_central_null_would_be_wrong():
    """A central chi-square critical value destroys the published values.

    This is the bug the module exists to avoid.  Recomputing Table 4 with
    ``chi2.ppf(1 - alpha, df)`` in place of ``ncx2.ppf(1 - alpha, df,
    lambda_0)`` must land far from the published N -- demonstrating that the
    passing test above really does hinge on the non-central null.
    """
    def wrong_power(df: int, n: int) -> float:
        crit = float(chi2.ppf(0.95, df))                # <- the classic bug
        lam_alt = (n - 1) * df * 0.08 ** 2
        return 1.0 - float(ncx2.cdf(crit, df, lam_alt))

    def wrong_n(df: int) -> int:
        lo, hi = 2, 200_000
        while lo < hi:
            mid = (lo + hi) // 2
            if wrong_power(df, mid) >= 0.80:
                hi = mid
            else:
                lo = mid + 1
        return lo

    print("\nWhat the central-null bug would produce")
    print("  df   published   central-null N   ratio")
    for df in (2, 8, 30, 100):
        pub = MBS_TABLE4_CLOSE_FIT[df]
        bad = wrong_n(df)
        print(f"{df:4d}   {pub:9d}   {bad:14d}   {bad / pub:5.2f}")
        assert abs(bad - pub) / pub > 0.20, (
            f"df={df}: the central-null calculation accidentally agrees, so "
            f"this test proves nothing")


# --------------------------------------------------------------------------- #
# 2. Monotonicity and bisection tightness
# --------------------------------------------------------------------------- #


def test_power_increases_in_n_and_in_df():
    """Close-fit power is monotone increasing in N and in df, over a grid."""
    n_grid = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
    df_grid = [1, 2, 3, 5, 8, 12, 16, 20, 30, 45, 60, 90, 120, 200]

    print("\nMonotonicity in N (df = 8):")
    previous = -1.0
    for n in n_grid:
        value = power.power_close_fit(8, n)
        print(f"  N={n:5d}  power={value:.6f}")
        assert value > previous, f"power fell going to N={n}"
        previous = value

    print("Monotonicity in df (N = 300):")
    previous = -1.0
    for df in df_grid:
        value = power.power_close_fit(df, 300)
        print(f"  df={df:4d} power={value:.6f}")
        assert value > previous, f"power fell going to df={df}"
        previous = value

    # The whole grid, not just the two slices.  Once power rounds to 1.0 in
    # double precision the increments are necessarily zero, so strictness is
    # required only below saturation.
    grid = np.array([[power.power_close_fit(df, n) for n in n_grid]
                     for df in df_grid])
    d_n = np.diff(grid, axis=1)
    d_df = np.diff(grid, axis=0)
    assert np.all(d_n >= -1e-12), "power fell with N somewhere"
    assert np.all(d_df >= -1e-12), "power fell with df somewhere"
    unsaturated_n = grid[:, :-1] < 0.999
    unsaturated_df = grid[:-1, :] < 0.999
    assert np.all(d_n[unsaturated_n] > 0), "not strictly monotone in N"
    assert np.all(d_df[unsaturated_df] > 0), "not strictly monotone in df"
    print(f"  full {len(df_grid)}x{len(n_grid)} grid monotone in both "
          f"directions: min(dP/dN)={d_n.min():.3e}, "
          f"min(dP/ddf)={d_df.min():.3e}; strict below saturation with "
          f"min(dP/dN)={d_n[unsaturated_n].min():.3e}, "
          f"min(dP/ddf)={d_df[unsaturated_df].min():.3e}")


def test_bisection_is_tight():
    """At the returned N power >= .80; at N - 1 it is < .80."""
    print("\nBisection tightness")
    print("  df      N   power(N)   power(N-1)")
    for df in (1, 2, 5, 8, 16, 30, 55, 90, 150):
        n = power.n_for_close_fit_power(df, 0.80)
        at_n = power.power_close_fit(df, n)
        at_n_minus = power.power_close_fit(df, n - 1)
        print(f"{df:4d}   {n:5d}   {at_n:.6f}   {at_n_minus:.6f}")
        assert at_n >= 0.80, f"df={df}: power({n}) = {at_n} < 0.80"
        assert at_n_minus < 0.80, (
            f"df={df}: power({n - 1}) = {at_n_minus} >= 0.80, so N={n} is "
            f"not the smallest adequate sample size")


def test_power_equals_alpha_when_hypotheses_coincide():
    """With eps_a = eps_0 the test has exactly its nominal size."""
    for df in (2, 8, 40):
        for n in (50, 500, 5000):
            for alpha in (0.01, 0.05, 0.10):
                value = power.power_close_fit(
                    df, n, rmsea_null=0.05, rmsea_alt=0.05, alpha=alpha)
                assert value == pytest.approx(alpha, abs=1e-9), (
                    f"df={df}, N={n}, alpha={alpha}: size {value}")
    print("\nSize check: power == alpha exactly when eps_a == eps_0.")


def test_bad_arguments_raise_bilingual_errors():
    """User mistakes surface as SemSimError subclasses with Arabic text."""
    with pytest.raises(SpecificationError) as exc:
        power.power_close_fit(0, 300)
    assert exc.value.message_ar and exc.value.message_ar != exc.value.message_en
    assert any("؀" <= ch <= "ۿ" for ch in exc.value.message_ar)
    for call in (lambda: power.power_close_fit(8, 1),
                 lambda: power.power_close_fit(8, 300, alpha=0.0),
                 lambda: power.power_close_fit(8, 300, alpha=1.0),
                 lambda: power.power_close_fit(8, 300, rmsea_alt=-0.1),
                 lambda: power.power_close_fit(-3, 300),
                 lambda: power.n_for_close_fit_power(8, 1.5)):
        with pytest.raises(SpecificationError):
            call()
    with pytest.raises(InfeasibleTargetError):
        # Null and alternative coincide: no N can reach .80.
        power.n_for_close_fit_power(8, 0.80, rmsea_alt=0.05, max_n=10_000)
    print("\nInvalid arguments raise SpecificationError / "
          "InfeasibleTargetError with real Arabic messages.")


def test_noncentrality_formula():
    """lambda = (N - 1) df eps^2, checked against hand arithmetic."""
    assert power.noncentrality(8, 401, 0.08) == pytest.approx(
        400 * 8 * 0.0064)
    assert power.noncentrality(30, 101, 0.05) == pytest.approx(
        100 * 30 * 0.0025)
    print("\nNoncentrality: (N-1)*df*eps^2 reproduced exactly.")


# --------------------------------------------------------------------------- #
# 3. Satorra & Saris (1985)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("beta", [0.50, 0.30, 0.15, 0.02])
def test_satorra_saris_matches_closed_form(beta):
    """The restricted population discrepancy equals its closed form.

    An independent recomputation, not a re-run of the same code: the module
    obtains ``F`` by numerically minimising ``F_ML`` over the restricted
    model's 17 free parameters, while the reference value comes from three
    log-determinants.
    """
    spec = _two_block_spec(beta)
    cm = compile_model(spec)
    sigma = implied_covariance(cm)
    detail = power.path_power_detail(spec, cm, spec.paths[0], use_cache=False)
    reference = _closed_form_restricted_fmin(sigma, 4)
    print(f"\nbeta={beta:.2f}: restricted F (refit) = {detail.ncp_per_case:.12f}, "
          f"closed form = {reference:.12f}, "
          f"relative difference = "
          f"{abs(detail.ncp_per_case - reference) / reference:.3e}, "
          f"method = {detail.method}, "
          f"df change = {detail.restricted_df_change}")
    assert detail.method == "satorra_saris"
    assert detail.restricted_df_change == 1
    assert detail.ncp_per_case == pytest.approx(reference, rel=1e-6)


def test_satorra_saris_power_strong_and_null_paths():
    """A beta of 0.50 at N = 500 is near-certain; 0.02 at N = 100 is ~alpha."""
    strong = _two_block_spec(0.50)
    cm_strong = compile_model(strong)
    pw_strong = power.satorra_saris_power(strong, cm_strong,
                                          strong.paths[0], 500)
    detail_strong = power.path_power_detail(strong, cm_strong,
                                            strong.paths[0])

    null = _two_block_spec(0.02)
    cm_null = compile_model(null)
    pw_null = power.satorra_saris_power(null, cm_null, null.paths[0], 100)
    detail_null = power.path_power_detail(null, cm_null, null.paths[0])

    print("\nSatorra-Saris per-path power")
    print(f"  beta=0.50, N=500: F={detail_strong.ncp_per_case:.6f}, "
          f"lambda={detail_strong.ncp(500):.3f}, power={pw_strong:.6f}")
    print(f"  beta=0.02, N=100: F={detail_null.ncp_per_case:.8f}, "
          f"lambda={detail_null.ncp(100):.5f}, power={pw_null:.6f}")

    assert pw_strong > 0.99, f"power for beta=0.50 at N=500 was {pw_strong}"
    assert 0.05 <= pw_null < 0.08, (
        f"power for beta=0.02 at N=100 was {pw_null}, which is not 'near "
        f"alpha = 0.05'")
    # A zero-information path has power exactly alpha.
    assert 1.0 - float(ncx2.cdf(chi2.ppf(0.95, 1), 1, 0.0)) == pytest.approx(
        0.05, abs=1e-9)


def test_satorra_saris_power_monotone_and_n_for_path_power_tight():
    """Per-path power rises with N, and the bisection brackets .80 exactly."""
    spec = _two_block_spec(0.30)
    cm = compile_model(spec)
    path = spec.paths[0]
    previous = -1.0
    print("\nPer-path power against N (beta = 0.30)")
    for n in (25, 50, 100, 150, 200, 300, 500, 1000):
        value = power.satorra_saris_power(spec, cm, path, n)
        print(f"  N={n:5d}  power={value:.6f}")
        assert value > previous
        previous = value
    needed = power.n_for_path_power(spec, cm, path, 0.80)
    at_n = power.satorra_saris_power(spec, cm, path, needed)
    below = power.satorra_saris_power(spec, cm, path, needed - 1)
    print(f"  n_for_path_power(.80) = {needed}: power={at_n:.6f}, "
          f"power(N-1)={below:.6f}")
    assert at_n >= 0.80 and below < 0.80

    # Cross-check against the closed-form noncentrality: solve for N by hand.
    reference_f = _closed_form_restricted_fmin(implied_covariance(cm), 4)
    crit = float(chi2.ppf(0.95, 1))
    hand = next(n for n in range(2, 100_000)
                if 1.0 - float(ncx2.cdf(crit, 1, (n - 1) * reference_f))
                >= 0.80)
    print(f"  independent brute-force search gives N={hand}")
    assert needed == hand


def test_zero_coefficient_path_is_infeasible():
    """A path of exactly zero can never be detected, and says so in Arabic."""
    spec = _two_block_spec(0.0)
    cm = compile_model(spec)
    with pytest.raises(InfeasibleTargetError) as exc:
        power.n_for_path_power(spec, cm, spec.paths[0], 0.80)
    assert any("؀" <= ch <= "ۿ" for ch in exc.value.message_ar)
    detail = power.path_power_detail(spec, cm, spec.paths[0])
    print(f"\nbeta=0: F={detail.ncp_per_case:.3e}, "
          f"power at N=10000 = "
          f"{power.satorra_saris_power(spec, cm, spec.paths[0], 10_000):.6f}")
    assert detail.ncp_per_case == pytest.approx(0.0, abs=1e-12)
    assert power.satorra_saris_power(
        spec, cm, spec.paths[0], 10_000) == pytest.approx(0.05, abs=1e-6)


def test_wald_fallback_is_recorded_in_notes():
    """When the restricted refit is unavailable, the fallback is reported."""
    spec = _two_block_spec(0.30)
    cm = compile_model(spec)
    analytic, coefficient = power._analytic_wald_ncp(cm, spec.paths[0])
    # Latent regression with one perfectly measured predictor:
    # avar(beta) = (1 - beta^2) / 1, so lambda/(N-1) = beta^2 / (1 - beta^2).
    expected = 0.30 ** 2 / (1.0 - 0.30 ** 2)
    print(f"\nAnalytic Wald F = {analytic:.8f}, "
          f"closed form beta^2/(1-beta^2) = {expected:.8f}")
    assert coefficient == pytest.approx(0.30)
    assert analytic == pytest.approx(expected, rel=1e-9)

    notes_en: list[str] = []
    notes_ar: list[str] = []

    def exploding_estimate(*_a, **_kw):
        raise ValueError("estimator deliberately broken for this test")

    import semsim.analysis.estimator as est_module
    original = est_module.estimate
    est_module.estimate = exploding_estimate   # type: ignore[assignment]
    try:
        value = power.satorra_saris_power(
            spec, cm, spec.paths[0], 300,
            notes=notes_en, notes_ar=notes_ar)
    finally:
        est_module.estimate = original         # type: ignore[assignment]
    print(f"  power with the estimator broken: {value:.6f}")
    for en in notes_en:
        print(f"  note(en): {en}")
    for ar in notes_ar:
        print(f"  note(ar): {ar}")
    assert notes_en and notes_ar
    assert any("FALLBACK" in n for n in notes_en)
    assert any(any("؀" <= ch <= "ۿ" for ch in n) for n in notes_ar)
    assert 0.0 < value < 1.0


# --------------------------------------------------------------------------- #
# 4. Monte Carlo (Muthen & Muthen, 2002)
# --------------------------------------------------------------------------- #


def test_monte_carlo_minimal_cfa_shape_and_runtime():
    """50 replications on minimal_cfa: one row per monitored path, under 60 s."""
    spec = examples.minimal_cfa()
    cm = compile_model(spec)
    seen: list[float] = []
    started = time.perf_counter()
    frame = power.monte_carlo_power(spec, n=400, replications=50, seed=20240808,
                                    progress=lambda _m, f: seen.append(f))
    elapsed = time.perf_counter() - started

    targets = power._monte_carlo_targets(spec, cm)
    print(f"\nMonte Carlo on minimal_cfa (N=400, 50 replications): "
          f"{elapsed:.2f} s, {len(frame)} row(s), "
          f"{len(seen)} progress callbacks")
    print(frame.to_string())

    assert elapsed < 60.0, f"took {elapsed:.1f} s"
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(power.MONTE_CARLO_COLUMNS)
    # minimal_cfa has no structural paths; its one monitored structural
    # parameter is the F1-F2 latent covariance.
    assert len(frame) == len(targets) == 1
    assert len(seen) == 50 and seen[-1] == pytest.approx(1.0)
    row = frame.iloc[0]
    assert int(row["n_converged"]) == 50
    assert row["population"] == pytest.approx(0.40, abs=1e-9)
    assert abs(row["relative_bias"]) < 0.10, "Muthen & Muthen: bias under 10 %"
    assert 0.85 <= row["coverage_95"] <= 1.0
    assert row["power"] == pytest.approx(1.0)


def test_monte_carlo_interruptible():
    """A progress callback returning False stops the study early."""
    spec = examples.minimal_cfa()
    calls = {"n": 0}

    def stop_after_five(_message: str, _fraction: float):
        calls["n"] += 1
        return False if calls["n"] >= 5 else None

    frame = power.monte_carlo_power(spec, n=300, replications=500, seed=3,
                                    progress=stop_after_five)
    print(f"\nInterrupted after {calls['n']} replications; frame reports "
          f"n_converged={int(frame.iloc[0]['n_converged'])} of "
          f"replications={int(frame.iloc[0]['replications'])}")
    assert calls["n"] == 5
    assert int(frame.iloc[0]["n_converged"]) == 5
    assert int(frame.iloc[0]["replications"]) == 500


def test_monte_carlo_power_agrees_with_satorra_saris():
    """Empirical power tracks the analytic Satorra-Saris value.

    Two completely different routes to the same number: 300 simulated
    datasets with a full ML fit each, against a non-central chi-square
    evaluated at the population discrepancy.  With 300 replications the
    binomial standard error of an empirical proportion near .60 is about
    .028, so a tolerance of 0.09 (roughly three standard errors) is the right
    size of window.
    """
    spec = _two_block_spec(0.20, n_items=4, loading=0.78)
    cm = compile_model(spec)
    analytic = power.satorra_saris_power(spec, cm, spec.paths[0], 200)
    frame = power.monte_carlo_power(spec, n=200, replications=300,
                                    seed=20240909, cm=cm)
    row = frame.iloc[0]
    empirical = float(row["power"])
    se = math.sqrt(max(empirical * (1 - empirical), 1e-9) / 300)
    print(f"\nbeta=0.20 at N=200: Satorra-Saris power = {analytic:.4f}, "
          f"Monte Carlo power = {empirical:.4f} "
          f"(binomial SE {se:.4f}, {int(row['n_converged'])}/300 converged)")
    print(frame.to_string())
    assert abs(empirical - analytic) < 0.09, (
        f"empirical {empirical:.4f} vs analytic {analytic:.4f}")
    # Muthen & Muthen's reporting criteria.
    assert abs(row["relative_bias"]) < 0.10
    assert 0.90 <= row["coverage_95"] <= 0.99, (
        f"coverage {row['coverage_95']:.3f} is outside Muthen & Muthen's "
        f"0.91-0.98 window (allowing a little Monte Carlo slack)")


def test_monte_carlo_rejects_impossible_requests():
    """Bad arguments raise, rather than returning a meaningless frame."""
    spec = examples.minimal_cfa()
    with pytest.raises(SpecificationError):
        power.monte_carlo_power(spec, n=400, replications=0)
    with pytest.raises(SpecificationError):
        power.monte_carlo_power(spec, n=2, replications=5)
    with pytest.raises(SpecificationError):
        power.monte_carlo_power(spec, n=400, replications=5, alpha=0.0)
    print("\nMonte Carlo rejects replications=0, N=2 and alpha=0.")


def test_monte_carlo_is_reproducible_and_seed_dependent():
    """The same seed gives the same numbers; a different seed does not."""
    spec = _two_block_spec(0.25)
    a = power.monte_carlo_power(spec, n=200, replications=12, seed=99)
    b = power.monte_carlo_power(spec, n=200, replications=12, seed=99)
    c = power.monte_carlo_power(spec, n=200, replications=12, seed=100)
    print(f"\nseed 99 twice: mean_estimate "
          f"{a.iloc[0]['mean_estimate']:.10f} / "
          f"{b.iloc[0]['mean_estimate']:.10f}; seed 100: "
          f"{c.iloc[0]['mean_estimate']:.10f}")
    assert a.iloc[0]["mean_estimate"] == pytest.approx(
        b.iloc[0]["mean_estimate"], abs=0.0)
    assert a.iloc[0]["mean_estimate"] != pytest.approx(
        c.iloc[0]["mean_estimate"], abs=1e-12)


# --------------------------------------------------------------------------- #
# 5. Rules of thumb and the assembled analysis
# --------------------------------------------------------------------------- #


def test_rules_of_thumb_arithmetic():
    """The heuristics are exactly the published ratios, not approximations."""
    spec = examples.simple_mediation()
    rot = power.rules_of_thumb(spec)
    n_free = spec.count_free_parameters()
    n_items = spec.n_items
    print(f"\nsimple_mediation: {n_free} free parameters, {n_items} items")
    print(f"  {rot}")
    assert rot["n_per_parameter"] == 10 * n_free
    assert rot["n_per_item"] == 10 * n_items
    assert rot["minimum_200"] == 200
    assert rot["absolute_floor_5_to_1"] == 5 * n_free
    custom = power.rules_of_thumb(spec, per_parameter=20, per_item=15)
    assert custom["n_per_parameter"] == 20 * n_free
    assert custom["n_per_item"] == 15 * n_items


def test_analyse_minimal_cfa_consistency():
    """analyse() returns numbers that agree with the primitives it uses."""
    spec = examples.minimal_cfa()
    result = power.analyse(spec)
    print(f"\nanalyse(minimal_cfa): method={result.method.value}, "
          f"df={result.df}, n_free={result.n_free}, "
          f"N={result.n_recommended}, "
          f"power={result.power_at_recommended:.6f}, "
          f"binding={result.binding_constraint}")
    for en, ar in zip(result.notes_en, result.notes_ar):
        print(f"  EN: {en}")
        print(f"  AR: {ar}")
    assert result.df == spec.degrees_of_freedom() == 8
    assert result.n_free == spec.count_free_parameters() == 13
    assert result.n_from_method == power.n_for_close_fit_power(8, 0.80)
    assert result.power_at_recommended == pytest.approx(
        power.power_close_fit(8, result.n_recommended))
    assert result.power_at_recommended >= 0.80
    assert result.rule_of_thumb == power.rules_of_thumb(spec)
    assert result.per_path_power is None          # no structural paths
    # Bilingual, parallel, and concrete about this model's numbers.
    assert len(result.notes_en) == len(result.notes_ar) >= 4
    assert any("13 free parameters" in n for n in result.notes_en)
    assert any("130" in n for n in result.notes_en)
    assert all(any("؀" <= ch <= "ۿ" for ch in n)
               for n in result.notes_ar)
    # The curve is a genuine power curve, monotone and containing the answer.
    curve = result.curve
    assert list(curve.columns) == ["n", "power", "power_not_close_fit"]
    assert curve["n"].is_monotonic_increasing
    assert curve["power"].is_monotonic_increasing
    assert result.n_recommended in set(curve["n"])
    match = curve.loc[curve["n"] == result.n_recommended, "power"].iloc[0]
    assert match == pytest.approx(result.power_at_recommended)


def test_analyse_honours_the_floors():
    """The recommendation never drops below Kline's 200 or the 5:1 floor."""
    spec = examples.second_order_hierarchy()
    result = power.analyse(spec)
    rot = result.rule_of_thumb
    print(f"\nanalyse(second_order_hierarchy): df={result.df}, "
          f"n_free={result.n_free}, formal N={result.n_from_method}, "
          f"recommended N={result.n_recommended}, "
          f"binding={result.binding_constraint}, floors={rot}")
    assert result.n_recommended >= 200
    assert result.n_recommended >= rot["absolute_floor_5_to_1"]
    assert result.n_recommended >= result.n_from_method
    assert result.binding_constraint in {
        "formal_method", "minimum_200", "absolute_floor_5_to_1",
        "satorra_saris", "min_n_guard_rail", "max_n_guard_rail"}


def test_analyse_fixed_mode_is_honoured_and_warns():
    """mode='fixed' keeps the user's N and lists every criterion it misses."""
    spec = examples.simple_mediation()
    spec.power.mode = "fixed"
    spec.power.n = 120
    result = power.analyse(spec)
    print(f"\nanalyse(simple_mediation, fixed N=120): "
          f"N={result.n_recommended}, "
          f"power={result.power_at_recommended:.4f}, "
          f"binding={result.binding_constraint}")
    for en in result.notes_en:
        print(f"  EN: {en}")
    assert result.n_recommended == 120
    assert result.binding_constraint == "fixed_by_user"
    assert result.power_at_recommended == pytest.approx(
        power.power_close_fit(result.df, 120))
    assert any("falls short" in n for n in result.notes_en)
    assert len(result.notes_en) == len(result.notes_ar)


def test_analyse_per_path_table_matches_the_primitives():
    """Every per-path row agrees with satorra_saris_power / n_for_path_power."""
    spec = examples.simple_mediation()
    cm = compile_model(spec)
    result = power.analyse(spec, cm)
    table = result.per_path_power
    assert table is not None
    print(f"\nanalyse(simple_mediation): N={result.n_recommended}, "
          f"binding={result.binding_constraint}")
    print(table.to_string())
    assert list(table.columns) == list(power.PER_PATH_COLUMNS)
    assert len(table) == len(spec.paths)
    for _, row in table.iterrows():
        path = next(p for p in spec.paths if p.id == row["path_id"])
        direct = power.satorra_saris_power(spec, cm, path,
                                           result.n_recommended)
        assert row["power"] == pytest.approx(direct, abs=1e-12)
        assert row["ncp_at_n"] == pytest.approx(
            (result.n_recommended - 1) * row["ncp_per_case"])
        if not pd.isna(row["n_required"]):
            needed = int(row["n_required"])
            assert needed == power.n_for_path_power(
                spec, cm, path, 0.80, max_n=spec.power.max_n)
            assert power.satorra_saris_power(spec, cm, path, needed) >= 0.80
    # The whole point of adding Satorra-Saris: the weakest path drives N.
    assert result.n_recommended >= int(table["n_required"].dropna().max())


def test_analyse_degenerate_df_falls_back_to_heuristics():
    """A just-identified model has no global power, and analyse() says so."""
    spec = ModelSpec(seed=1)
    spec.add_construct("A", role=ConstructRole.INDEPENDENT, n_items=2,
                       loading=0.8)
    spec.add_construct("B", role=ConstructRole.INDEPENDENT, n_items=1,
                       loading=0.8)
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS
    spec.ensure_unique_item_names()
    result = power.analyse(spec)
    print(f"\ndegenerate model: df={result.df}, method={result.method.value}, "
          f"N={result.n_recommended}, power={result.power_at_recommended}")
    for en, ar in zip(result.notes_en, result.notes_ar):
        print(f"  EN: {en}")
        print(f"  AR: {ar}")
    assert result.df <= 0
    assert result.method is PowerMethod.RULE_OF_THUMB
    assert math.isnan(result.power_at_recommended)
    assert result.curve.empty
    assert result.n_recommended == max(result.rule_of_thumb.values())
    assert any("saturated or under-identified" in n for n in result.notes_en)


def test_power_result_to_table_is_bilingual():
    """to_table() labels the criteria in whichever language is asked for."""
    result = power.analyse(examples.minimal_cfa())
    en = result.to_table("en")
    ar = result.to_table("ar")
    print("\n" + en.to_string())
    print(ar.to_string())
    assert list(en.columns) == ["criterion", "n"] == list(ar.columns)
    assert len(en) == len(ar) == len(result.rule_of_thumb) + 2
    assert en["n"].tolist() == ar["n"].tolist()
    assert all(any("؀" <= ch <= "ۿ" for ch in c)
               for c in ar["criterion"])
    assert en.iloc[-1]["n"] == result.n_recommended


def test_no_global_random_state_is_touched():
    """Nothing in the module reads or advances numpy's global RNG."""
    np.random.seed(4242)
    before = np.random.get_state()[1][:8].copy()        # type: ignore[index]
    spec = _two_block_spec(0.30)
    cm = compile_model(spec)
    power.satorra_saris_power(spec, cm, spec.paths[0], 300)
    power.monte_carlo_power(spec, n=200, replications=4, seed=7, cm=cm)
    power.analyse(examples.minimal_cfa())
    after = np.random.get_state()[1][:8]                # type: ignore[index]
    print(f"\nglobal numpy RNG state unchanged: {np.array_equal(before, after)}")
    assert np.array_equal(before, after)


# --------------------------------------------------------------------------- #
# Stand-alone runner, for environments without pytest
# --------------------------------------------------------------------------- #


def _main() -> int:  # pragma: no cover - manual entry point
    import traceback

    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        marks = getattr(fn, "pytestmark", [])
        cases = [()]
        for mark in marks:
            if mark.name == "parametrize":
                cases = [(v,) for v in mark.args[1]]
        for args in cases:
            label = f"{name}{args if args else ''}"
            try:
                fn(*args)
            except Exception:
                failures += 1
                print(f"FAIL {label}")
                traceback.print_exc()
            else:
                print(f"PASS {label}")
    print(f"\n{len(tests)} test functions, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

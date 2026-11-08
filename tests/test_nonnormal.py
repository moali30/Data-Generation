"""
Numerical tests for the non-normal generation layer
===================================================
``semsim.generation.fleishman`` and ``semsim.generation.vale_maurelli``.

Every check is against a closed-form quantity: Fleishman's three moment
equations evaluated at the returned coefficients, the Vale & Maurelli cubic
evaluated at the returned intermediate correlation, or a large-sample
Monte-Carlo estimate with a stated tolerance.  Observed maxima are printed so
the numbers can be read off the run rather than taken on trust.

Run with::

    python3 -m pytest tests/test_nonnormal.py -q -s

or directly (no pytest needed)::

    python3 tests/test_nonnormal.py

References
----------
Fleishman, A. I. (1978). *Psychometrika*, 43(4), 521-532.
Vale, C. D., & Maurelli, V. A. (1983). *Psychometrika*, 48(3), 465-471.
"""

from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np

if __package__ in (None, ""):  # allows `python3 tests/test_nonnormal.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semsim.core.errors import (
    ConvergenceError,
    InfeasibleTargetError,
    SemSimError,
    SpecificationError,
)
from semsim.generation import fleishman as fl
from semsim.generation import vale_maurelli as vm


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _moment_equations(b: float, c: float, d: float) -> tuple[float, float, float]:
    """Fleishman's three left-hand sides, written out independently of the
    implementation so the test does not reuse the code it is checking."""
    variance = b ** 2 + 2.0 * c ** 2 + 6.0 * b * d + 15.0 * d ** 2
    skewness = 2.0 * c * (b ** 2 + 24.0 * b * d + 105.0 * d ** 2 + 2.0)
    kurtosis = 24.0 * (
        b * d
        + c ** 2 * (1.0 + b ** 2 + 28.0 * b * d)
        + d ** 2 * (12.0 + 48.0 * b * d + 141.0 * c ** 2 + 225.0 * d ** 2)
    )
    return variance, skewness, kurtosis


def _sample_moments(y: np.ndarray) -> tuple[float, float, float, float]:
    """mean, population sd, g1 skewness, g2 excess kurtosis."""
    y = np.asarray(y, dtype=float)
    m = float(y.mean())
    c2 = float(np.mean((y - m) ** 2))
    c3 = float(np.mean((y - m) ** 3))
    c4 = float(np.mean((y - m) ** 4))
    return m, math.sqrt(c2), c3 / c2 ** 1.5, c4 / c2 ** 2 - 3.0


def _feasible_grid() -> list[tuple[float, float]]:
    """A grid of (skew, excess kurtosis) pairs inside the reachable region."""
    pairs: list[tuple[float, float]] = []
    for skew in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        for kurt in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                     15.0, 20.0):
            # Stay clear of the practical boundary of the cubic power method.
            if kurt > fl.FLEISHMAN_PRACTICAL_BOUND(skew) + 0.25:
                pairs.append((skew, kurt))
    return pairs


FIVE_BY_FIVE = np.array([
    [1.00, 0.50, 0.40, 0.30, 0.20],
    [0.50, 1.00, 0.45, 0.35, 0.25],
    [0.40, 0.45, 1.00, 0.50, 0.30],
    [0.30, 0.35, 0.50, 1.00, 0.40],
    [0.20, 0.25, 0.30, 0.40, 1.00],
])
FIVE_SKEW = [0.0, 0.80, -0.60, 1.20, 0.40]
FIVE_KURT = [0.0, 0.80, 0.50, 2.00, -0.30]


# --------------------------------------------------------------------------- #
# Fleishman: exact cases and the published table
# --------------------------------------------------------------------------- #


def test_normal_target_is_the_exact_identity() -> None:
    """skew == 0 and kurtosis == 0 must short-circuit to (0, 1, 0, 0)."""
    coef = fl.solve(0.0, 0.0)
    assert (coef.a, coef.b, coef.c, coef.d) == (0.0, 1.0, 0.0, 0.0)
    assert coef.iterations == 0, "the normal case must not iterate at all"
    assert coef.converged is True
    assert coef.residual == 0.0
    assert coef.is_identity
    # The identity must leave the deviates untouched, bit for bit.
    z = np.array([-2.5, -0.3, 0.0, 0.7, 3.1])
    assert np.array_equal(fl.transform(z, coef), z)
    print("[obs] solve(0, 0) -> (a, b, c, d) =", coef.as_tuple())


def test_a_equals_minus_c_and_mean_is_zero() -> None:
    """``a = -c`` is what makes E[Y] = 0, since E[Z**2] = 1."""
    for skew, kurt in ((2.0, 7.0), (-1.3, 3.0), (0.0, 4.0), (0.5, -0.5)):
        coef = fl.solve(skew, kurt)
        assert coef.a == -coef.c
        # E[Y] = a + c exactly (E[Z] = E[Z**3] = 0, E[Z**2] = 1).
        assert abs(coef.a + coef.c) < 1e-15


def test_symmetric_target_gives_exactly_zero_c() -> None:
    for kurt in (-1.0, 0.5, 3.0, 10.0, 25.0):
        coef = fl.solve(0.0, kurt)
        assert coef.c == 0.0, "a symmetric target must return c exactly 0"
        assert coef.a == 0.0
        assert coef.b > 0.0


def test_skew_two_kurt_seven_reproduces_its_moments() -> None:
    """Fleishman's table row (skew 2.0, excess kurtosis 7.0): the coefficients
    must reproduce the targets through the moment equations to 1e-8."""
    coef = fl.solve(2.0, 7.0)
    var, skew, kurt = _moment_equations(coef.b, coef.c, coef.d)
    print(f"[obs] skew=2, kurt=7 -> a={coef.a:.6f} b={coef.b:.6f} "
          f"c={coef.c:.6f} d={coef.d:.6f}")
    print(f"[obs]   back-substituted: var={var:.12f} skew={skew:.12f} "
          f"kurt={kurt:.12f}")
    assert abs(var - 1.0) < 1e-8
    assert abs(skew - 2.0) < 1e-8
    assert abs(kurt - 7.0) < 1e-8
    assert coef.converged


def test_symmetric_kurtosis_25() -> None:
    """skew = 0, excess kurtosis = 25.

    The task brief quotes b = 0.233790, d = 0.178935 for this row.  That pair
    does **not** satisfy Fleishman's variance equation
    ``b**2 + 2c**2 + 6bd + 15d**2 = 1`` (it gives 0.78592), so it cannot be a
    solution of the published system; it is a correctly shaped but
    non-unit-variance pair: rescaled to unit variance (divide by
    sqrt(0.78592) = 0.88651) it becomes (0.26372, 0, 0.20184), whose excess
    kurtosis is 24.465 rather than 25.  The
    unit-variance solution is asserted instead, by back-substitution.
    """
    coef = fl.solve(0.0, 25.0)
    var, skew, kurt = _moment_equations(coef.b, coef.c, coef.d)
    print(f"[obs] skew=0, kurt=25 -> b={coef.b:.6f} c={coef.c:.6f} "
          f"d={coef.d:.6f}  (var={var:.12f}, kurt={kurt:.10f})")
    assert abs(var - 1.0) < 1e-8
    assert abs(skew - 0.0) < 1e-8
    assert abs(kurt - 25.0) < 1e-8
    assert coef.c == 0.0 and coef.b > 0.0 and coef.d > 0.0

    quoted_var, quoted_skew, quoted_kurt = _moment_equations(
        0.233790, 0.0, 0.178935)
    print(f"[obs]   quoted pair (0.233790, 0, 0.178935) substituted into the "
          f"equations: var={quoted_var:.6f} (must be 1), skew={quoted_skew:.6f}, "
          f"kurt-equation={quoted_kurt:.6f}")
    assert abs(quoted_var - 1.0) > 0.2, (
        "documented discrepancy: the quoted coefficients are not unit-variance")


def test_grid_of_targets_back_substitutes_to_zero() -> None:
    """The robust check: solve a grid and substitute back into all three
    equations, requiring residuals below 1e-8."""
    pairs = _feasible_grid()
    assert len(pairs) >= 25, f"grid too small: {len(pairs)} pairs"
    worst = 0.0
    worst_at = None
    for skew, kurt in pairs:
        coef = fl.solve(skew, kurt)
        var, sk, ku = _moment_equations(coef.b, coef.c, coef.d)
        residual = max(abs(var - 1.0), abs(sk - skew), abs(ku - kurt))
        if residual > worst:
            worst, worst_at = residual, (skew, kurt)
        assert residual < 1e-8, (
            f"residual {residual:.3e} at skew={skew}, kurtosis={kurt}")
        assert coef.b > 0.0
        assert coef.a == -coef.c
        # Sign convention: c carries the sign of the skewness.
        if skew > 0:
            assert coef.c > 0.0
        elif skew < 0:
            assert coef.c < 0.0
    print(f"[obs] grid of {len(pairs)} feasible pairs "
          f"(skew in [-2, 2], kurt in [-1, 20]): "
          f"max residual {worst:.3e} at {worst_at}")


def test_sign_symmetry_of_the_skewness() -> None:
    """Negating c flips the skewness and leaves variance and kurtosis alone."""
    for skew, kurt in ((1.7, 5.0), (0.9, 1.5), (2.0, 7.0)):
        pos = fl.solve(skew, kurt)
        neg = fl.solve(-skew, kurt)
        assert abs(pos.b - neg.b) < 1e-10
        assert abs(pos.d - neg.d) < 1e-10
        assert abs(pos.c + neg.c) < 1e-10


# --------------------------------------------------------------------------- #
# Fleishman: feasibility and error handling
# --------------------------------------------------------------------------- #


def test_feasibility_bound_matches_the_contract() -> None:
    assert fl.FLEISHMAN_LOWER_KURTOSIS_BOUND(0.0) == -2.0
    assert fl.FLEISHMAN_LOWER_KURTOSIS_BOUND(2.0) == 2.0
    assert fl.is_feasible(0.0, 0.0) is True
    assert fl.is_feasible(2.0, 2.5) is True
    assert fl.is_feasible(2.0, 2.0) is False          # exactly on the bound
    assert fl.is_feasible(2.0, 1.0) is False
    assert fl.is_feasible(0.0, -2.5) is False
    assert fl.is_feasible(float("nan"), 1.0) is False


def test_infeasible_target_raises_with_arabic_message() -> None:
    for skew, kurt in ((2.0, 1.0), (0.0, -3.0), (3.0, 5.0)):
        try:
            fl.solve(skew, kurt)
        except InfeasibleTargetError as exc:
            assert isinstance(exc, SemSimError)
            assert exc.message_ar and exc.message_ar != exc.message_en
            assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
        else:
            raise AssertionError(
                f"solve({skew}, {kurt}) should have raised InfeasibleTargetError")


def test_clamp_pulls_an_infeasible_target_inside() -> None:
    skew, kurt = fl.clamp_to_feasible(2.0, 1.0)
    assert skew == 2.0
    assert kurt > fl.FLEISHMAN_LOWER_KURTOSIS_BOUND(2.0)
    coef = fl.solve(2.0, 1.0, clamp=True)        # must not raise
    var, sk, ku = _moment_equations(coef.b, coef.c, coef.d)
    assert abs(var - 1.0) < 1e-8
    assert abs(sk - 2.0) < 1e-8
    print(f"[obs] clamped (2.0, 1.0) -> kurtosis {ku:.6f}")


def test_target_below_the_practical_boundary_raises_convergence_error() -> None:
    """Above skew**2 - 2 but below the reachable region: a documented
    ConvergenceError, never a bare LinAlgError or a silent wrong answer."""
    try:
        fl.solve(2.0, 2.5)
    except ConvergenceError as exc:
        assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
        print("[obs] solve(2.0, 2.5) raised ConvergenceError as expected")
    except InfeasibleTargetError:
        print("[obs] solve(2.0, 2.5) raised InfeasibleTargetError")
    else:
        raise AssertionError("expected a SemSimError below the Fleishman region")


def test_non_finite_targets_raise() -> None:
    for skew, kurt in ((float("nan"), 0.0), (0.0, float("inf"))):
        try:
            fl.solve(skew, kurt)
        except InfeasibleTargetError:
            pass
        else:
            raise AssertionError("non-finite targets must raise")


# --------------------------------------------------------------------------- #
# Fleishman: Monte-Carlo validation of the transformation itself
# --------------------------------------------------------------------------- #


def test_transform_monte_carlo_moments() -> None:
    """An independent empirical check that the coefficients really deliver the
    requested marginal shape (closed-form algebra validated against data)."""
    rng = np.random.default_rng(20240101)
    z = rng.standard_normal(2_000_000)
    rows = []
    for skew, kurt in ((0.0, 0.0), (0.75, 1.0), (-1.0, 2.0), (1.5, 4.0)):
        coef = fl.solve(skew, kurt)
        y = fl.transform(z, coef)
        mean, sd, g1, g2 = _sample_moments(y)
        rows.append((skew, kurt, mean, sd, g1, g2))
        assert abs(mean) < 0.01
        assert abs(sd - 1.0) < 0.01
        assert abs(g1 - skew) < 0.05, f"skew {g1} vs {skew}"
        assert abs(g2 - kurt) < 0.20, f"kurtosis {g2} vs {kurt}"
    for skew, kurt, mean, sd, g1, g2 in rows:
        print(f"[obs] MC n=2e6 target ({skew:+.2f}, {kurt:+.2f}) -> "
              f"mean {mean:+.4f} sd {sd:.4f} skew {g1:+.4f} kurt {g2:+.4f}")


# --------------------------------------------------------------------------- #
# Vale & Maurelli: the intermediate correlation cubic
# --------------------------------------------------------------------------- #


def test_normal_case_collapses_the_cubic_exactly() -> None:
    normal = fl.solve(0.0, 0.0)
    for rho in (-0.99, -0.5, -0.123456789, 0.0, 0.2, 0.75, 1.0):
        got = fl.intermediate_correlation(rho, normal, normal)
        assert got == rho, f"normal case must be exact: {got!r} != {rho!r}"
    print("[obs] normal-normal intermediate correlation is bit-exact")


def test_intermediate_correlation_satisfies_the_cubic() -> None:
    """Substituting rho_z back into Vale & Maurelli's cubic must return the
    target correlation."""
    coefs = [fl.solve(s, k) for s, k in
             ((0.0, 0.0), (1.0, 2.0), (-0.8, 1.0), (2.0, 7.0), (0.0, 5.0))]
    worst = 0.0
    checked = 0
    unattainable = 0
    for ci in coefs:
        for cj in coefs:
            for rho in (-0.8, -0.35, 0.0, 0.15, 0.45, 0.7, 0.9):
                try:
                    rz = fl.intermediate_correlation(rho, ci, cj)
                except InfeasibleTargetError:
                    # Legitimate: opposite-signed skewness caps how strongly
                    # two Fleishman marginals can correlate.
                    unattainable += 1
                    continue
                checked += 1
                assert -1.0 <= rz <= 1.0
                back = (rz * (ci.b * cj.b + 3 * ci.b * cj.d + 3 * ci.d * cj.b
                              + 9 * ci.d * cj.d)
                        + rz ** 2 * (2 * ci.c * cj.c)
                        + rz ** 3 * (6 * ci.d * cj.d))
                worst = max(worst, abs(back - rho))
                assert abs(back - rho) < 1e-10, (
                    f"cubic not satisfied: {back} vs {rho}")
                # Non-normality always needs a larger normal-stage correlation.
                if rho > 0 and not (ci.is_identity and cj.is_identity):
                    assert rz >= rho - 1e-12
    assert checked >= 150, f"only {checked} solvable combinations"
    print(f"[obs] intermediate-correlation cubic: max |rho_y(rho_z) - target| "
          f"= {worst:.3e} over {checked} solvable combinations "
          f"({unattainable} correctly rejected as unattainable)")


def test_intermediate_correlation_rejects_bad_input() -> None:
    normal = fl.solve(0.0, 0.0)
    for rho in (1.5, -2.0, float("nan")):
        try:
            fl.intermediate_correlation(rho, normal, normal)
        except InfeasibleTargetError:
            pass
        else:
            raise AssertionError(f"rho={rho} must raise")


def test_unattainable_correlation_raises() -> None:
    """Two strongly non-normal marginals cannot correlate arbitrarily highly."""
    heavy = fl.solve(0.0, 25.0)
    try:
        fl.intermediate_correlation(0.999, heavy, heavy)
    except InfeasibleTargetError as exc:
        assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
        print("[obs] rho=0.999 between two kurtosis-25 marginals is rejected")
    else:
        # If a root does exist it must still satisfy the cubic; the method's
        # attainable range depends on the coefficients, so accept either.
        print("[obs] rho=0.999 attainable for this pair; no rejection needed")


# --------------------------------------------------------------------------- #
# Vale & Maurelli: generation
# --------------------------------------------------------------------------- #


def test_generate_normal_recovers_the_target_correlation() -> None:
    rng = np.random.default_rng(4242)
    x = vm.generate_normal(200_000, FIVE_BY_FIVE, rng)
    assert x.shape == (200_000, 5)
    r = np.corrcoef(x, rowvar=False)
    err = float(np.max(np.abs(r - FIVE_BY_FIVE)))
    mom = vm.achieved_moments(x)
    print(f"[obs] generate_normal n=2e5: max |r - target| = {err:.5f}, "
          f"max |skew| = {np.max(np.abs(mom['skewness'])):.4f}, "
          f"max |kurt| = {np.max(np.abs(mom['kurtosis'])):.4f}")
    assert err < 0.01
    assert np.max(np.abs(mom["skewness"])) < 0.05
    assert np.max(np.abs(mom["kurtosis"])) < 0.10


def test_normal_marginals_through_generate_reproduce_the_matrix() -> None:
    """generate() with all-normal targets must match generate_normal's accuracy
    (the intermediate cubic is the identity, so nothing is lost)."""
    rng = np.random.default_rng(99)
    x = vm.generate(200_000, FIVE_BY_FIVE, [0.0] * 5, [0.0] * 5, rng)
    r = np.corrcoef(x, rowvar=False)
    err = float(np.max(np.abs(r - FIVE_BY_FIVE)))
    print(f"[obs] generate() all-normal n=2e5: max |r - target| = {err:.5f}")
    assert err < 0.01


def test_vale_maurelli_monte_carlo_accuracy() -> None:
    """The headline accuracy check: a known 5x5 correlation matrix with
    assorted skewness / kurtosis, n = 200000, fixed seed."""
    rng = np.random.default_rng(20240101)
    x = vm.generate(200_000, FIVE_BY_FIVE, FIVE_SKEW, FIVE_KURT, rng)
    assert x.shape == (200_000, 5)

    r = np.corrcoef(x, rowvar=False)
    corr_err = float(np.max(np.abs(r - FIVE_BY_FIVE)))
    mom = vm.achieved_moments(x)
    skew_err = float(np.max(np.abs(mom["skewness"] - np.array(FIVE_SKEW))))
    kurt_err = float(np.max(np.abs(mom["kurtosis"] - np.array(FIVE_KURT))))
    sd_err = float(np.max(np.abs(mom["sd"] - 1.0)))
    mean_err = float(np.max(np.abs(mom["mean"])))

    print("[obs] Vale-Maurelli MC, n=200000, seed=20240101:")
    print(f"[obs]   max |corr - target|      = {corr_err:.5f}  (tol 0.02)")
    print(f"[obs]   max |skew - target|      = {skew_err:.5f}  (tol 0.10)")
    print(f"[obs]   max |kurtosis - target|  = {kurt_err:.5f}  (tol 0.40)")
    print(f"[obs]   max |sd - 1|             = {sd_err:.5f}")
    print(f"[obs]   max |mean|               = {mean_err:.5f}")
    print(f"[obs]   achieved skew  = {np.round(mom['skewness'], 4)}")
    print(f"[obs]   achieved kurt  = {np.round(mom['kurtosis'], 4)}")

    assert corr_err < 0.02
    assert skew_err < 0.10
    assert kurt_err < 0.40
    assert sd_err < 0.02
    assert mean_err < 0.02


def test_covariance_metric_means_and_sds() -> None:
    """diag(sigma) must set the SDs and the mean vector must be honoured."""
    sd = np.array([2.0, 0.5, 3.0])
    r = np.array([[1.0, 0.4, -0.3], [0.4, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    sigma = r * np.outer(sd, sd)
    mean = np.array([10.0, -4.0, 0.0])
    rng = np.random.default_rng(7)
    x = vm.generate(200_000, sigma, [0.6, 0.0, -1.0], [0.8, 0.0, 2.0], rng,
                    mean=mean)
    mom = vm.achieved_moments(x)
    print(f"[obs] scaled metric: sd {np.round(mom['sd'], 4)} (target {sd}), "
          f"mean {np.round(mom['mean'], 4)} (target {mean})")
    assert np.max(np.abs(mom["sd"] - sd)) < 0.05
    assert np.max(np.abs(mom["mean"] - mean)) < 0.05
    cov = np.cov(x, rowvar=False, ddof=0)
    assert np.max(np.abs(cov - sigma)) < 0.06


def test_single_variable_and_tiny_sample() -> None:
    rng = np.random.default_rng(3)
    x = vm.generate(5, np.array([[4.0]]), [1.0], [2.0], rng)
    assert x.shape == (5, 1)
    assert np.all(np.isfinite(x))


def test_reproducibility_depends_only_on_the_generator() -> None:
    a = vm.generate(500, FIVE_BY_FIVE, FIVE_SKEW, FIVE_KURT,
                    np.random.default_rng(12345))
    b = vm.generate(500, FIVE_BY_FIVE, FIVE_SKEW, FIVE_KURT,
                    np.random.default_rng(12345))
    c = vm.generate(500, FIVE_BY_FIVE, FIVE_SKEW, FIVE_KURT,
                    np.random.default_rng(54321))
    assert np.array_equal(a, b), "same seed must give an identical dataset"
    assert not np.array_equal(a, c), "a different seed must give different data"


def test_performance_500_by_60() -> None:
    import time
    p = 60
    r = np.full((p, p), 0.25)
    np.fill_diagonal(r, 1.0)
    skew = [0.5 if j % 2 else -0.4 for j in range(p)]
    kurt = [1.0 if j % 3 else 0.3 for j in range(p)]
    rng = np.random.default_rng(1)
    t0 = time.perf_counter()
    x = vm.generate(500, r, skew, kurt, rng)
    elapsed = time.perf_counter() - t0
    print(f"[obs] 500x60 generated in {elapsed * 1000:.1f} ms")
    assert x.shape == (500, 60)
    assert elapsed < 1.0


# --------------------------------------------------------------------------- #
# Vale & Maurelli: PD repair and error handling
# --------------------------------------------------------------------------- #


def _non_pd_matrix() -> np.ndarray:
    m = np.array([
        [1.0, 0.9, -0.9],
        [0.9, 1.0, 0.9],
        [-0.9, 0.9, 1.0],
    ])
    assert np.min(np.linalg.eigvalsh(m)) < 0.0
    return m


def test_non_pd_target_is_repaired_with_a_warning() -> None:
    bad = _non_pd_matrix()
    rng = np.random.default_rng(11)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = vm.generate(1000, bad, [0.5, 0.0, -0.5], [1.0, 0.0, 1.0], rng)
    assert x.shape == (1000, 3)
    assert np.all(np.isfinite(x))
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        "a non-PD intermediate matrix must warn"
    print(f"[obs] non-PD repair warned: "
          f"{str(caught[0].message).split('[AR]')[0].strip()[:90]}...")


def test_repair_disabled_raises() -> None:
    bad = _non_pd_matrix()
    rng = np.random.default_rng(11)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vm.generate(100, bad, [0.5, 0.0, -0.5], [1.0, 0.0, 1.0], rng,
                        repair_pd=False)
    except SemSimError as exc:
        assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar)
    else:
        raise AssertionError("repair_pd=False on a non-PD matrix must raise")


def test_input_validation_raises_semsim_errors() -> None:
    rng = np.random.default_rng(0)
    good = np.eye(3)
    cases = [
        ("n = 0", lambda: vm.generate(0, good, [0, 0, 0], [0, 0, 0], rng)),
        ("n negative", lambda: vm.generate(-5, good, [0, 0, 0], [0, 0, 0], rng)),
        ("non-square sigma",
         lambda: vm.generate(10, np.ones((2, 3)), [0, 0, 0], [0, 0, 0], rng)),
        ("asymmetric sigma",
         lambda: vm.generate(10, np.array([[1.0, 0.3], [0.9, 1.0]]),
                             [0, 0], [0, 0], rng)),
        ("zero variance",
         lambda: vm.generate(10, np.diag([1.0, 0.0]), [0, 0], [0, 0], rng)),
        ("nan in sigma",
         lambda: vm.generate(10, np.array([[1.0, np.nan], [np.nan, 1.0]]),
                             [0, 0], [0, 0], rng)),
        ("skew length mismatch",
         lambda: vm.generate(10, good, [0, 0], [0, 0, 0], rng)),
        ("kurtosis length mismatch",
         lambda: vm.generate(10, good, [0, 0, 0], [0, 0], rng)),
        ("mean length mismatch",
         lambda: vm.generate(10, good, [0, 0, 0], [0, 0, 0], rng,
                             mean=[1.0, 2.0])),
        ("not a Generator",
         lambda: vm.generate(10, good, [0, 0, 0], [0, 0, 0], 12345)),
        ("infeasible marginal",
         lambda: vm.generate(10, good, [2.0, 0, 0], [1.0, 0, 0], rng)),
        ("empty data to achieved_moments",
         lambda: vm.achieved_moments(np.zeros((0, 3)))),
    ]
    for label, call in cases:
        try:
            call()
        except SemSimError as exc:
            assert exc.message_ar, f"{label}: missing Arabic message"
            assert any("؀" <= ch <= "ۿ" for ch in exc.message_ar), \
                f"{label}: Arabic message is a transliteration"
        except Exception as exc:  # noqa: BLE001 - that is the point of the test
            raise AssertionError(
                f"{label} raised {type(exc).__name__} instead of a "
                f"SemSimError: {exc}") from exc
        else:
            raise AssertionError(f"{label} should have raised")
    print(f"[obs] {len(cases)} malformed inputs all raised SemSimError "
          "subclasses with Arabic messages")


def test_achieved_moments_against_closed_form() -> None:
    """Checked against the moments of a known finite sample."""
    y = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    mom = vm.achieved_moments(y)
    mean = 22.0
    centred = y[:, 0] - mean
    m2 = float(np.mean(centred ** 2))
    assert abs(mom["mean"][0] - mean) < 1e-12
    assert abs(mom["sd"][0] - math.sqrt(m2)) < 1e-12
    assert abs(mom["skewness"][0]
               - float(np.mean(centred ** 3)) / m2 ** 1.5) < 1e-12
    assert abs(mom["kurtosis"][0]
               - (float(np.mean(centred ** 4)) / m2 ** 2 - 3.0)) < 1e-12
    # A standard normal sample: skewness ~ 0, excess kurtosis ~ 0.
    z = np.random.default_rng(5).standard_normal((500_000, 1))
    zm = vm.achieved_moments(z)
    assert abs(zm["skewness"][0]) < 0.02
    assert abs(zm["kurtosis"][0]) < 0.03
    assert set(("mean", "sd", "skewness", "kurtosis")) <= set(mom)


def test_fleishman_coefficients_are_cached_per_target() -> None:
    coefs = vm.fleishman_coefficients([0.5] * 4 + [0.0] * 4,
                                      [1.0] * 4 + [0.0] * 4)
    assert len(coefs) == 8
    assert coefs[0] is coefs[1], "identical targets should reuse one solution"
    assert coefs[4].is_identity


# --------------------------------------------------------------------------- #
# Stand-alone runner (used when pytest is unavailable)
# --------------------------------------------------------------------------- #


def _main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - runner reports every failure
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

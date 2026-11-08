"""
Numerical tests for :mod:`semsim.analysis.estimator`.

Every assertion here checks a *number* against a closed form, a published
identity or an independent recomputation:

* parameter recovery -- the gold standard: draw a large sample from the exact
  population ``Sigma(theta)`` and check the standardised estimates come back
  to the population values the specification asked for;
* population fit -- feed the engine its own ``Sigma(theta)`` as if it were a
  sample covariance matrix; ``F`` and chi-square must be ~0, which is only
  possible if the marker parameterisation spans the model exactly;
* degrees of freedom -- ``p (p + 1) / 2 - n_free``, equal to 8 for the
  minimal CFA;
* standard errors -- the expected-information and observed-information
  formulae must agree, and must shrink as ``1 / sqrt(N)``;
* the duplication matrix and the two algebraically identical forms of the
  expected information must agree to machine precision.

The file is runnable both under pytest and directly with ``python3``.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, norm

from semsim.core import examples
from semsim.core.compile import compile_model
from semsim.core.errors import IdentificationError, SpecificationError
from semsim.core.implied import implied_covariance
from semsim.core.spec import EstimatorKind, ModelSpec
from semsim.analysis import estimator as est
from semsim.analysis.estimator import (EstimationResult, duplication_matrix,
                                       estimate, ml_discrepancy,
                                       uls_discrepancy, unvech, vech)


# --------------------------------------------------------------------------- #
# Fixtures built by hand so the file also runs without pytest
# --------------------------------------------------------------------------- #


def exact_covariance_data(sigma: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``(n, p)`` data whose sample covariance is *exactly* ``sigma``.

    Orthonormalising a mean-centred random matrix gives ``Q`` with
    ``Q'Q = I`` and zero column means; scaling by ``sqrt(n - 1)`` and
    multiplying by a Cholesky factor of ``sigma`` therefore produces a sample
    covariance (``ddof=1``) equal to ``sigma`` to machine precision.  That is
    what turns an ordinary call to :func:`estimate` into a population fit.
    """
    rng = np.random.default_rng(seed)
    p = sigma.shape[0]
    a = rng.normal(size=(n, p))
    a -= a.mean(axis=0)
    q, _ = np.linalg.qr(a)
    z = q * math.sqrt(n - 1.0)
    return z @ np.linalg.cholesky(sigma).T


def sample_data(sigma: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` draws from ``N(0, sigma)`` using an explicit generator."""
    rng = np.random.default_rng(seed)
    dist = multivariate_normal(mean=np.zeros(sigma.shape[0]), cov=sigma)
    return np.asarray(dist.rvs(size=n, random_state=rng), dtype=float)


_CACHE: Dict[Tuple, object] = {}


def fitted(example: str, n: int, seed: int, *, exact: bool = False,
           estimator: EstimatorKind = EstimatorKind.ML,
           observed_se: bool = True) -> Tuple[EstimationResult, object,
                                              np.ndarray]:
    """Fit one example once and memoise the result."""
    key = (example, n, seed, exact, estimator, observed_se)
    if key not in _CACHE:
        spec = examples.build(example)
        cm = compile_model(spec)
        sigma = implied_covariance(cm)
        maker = exact_covariance_data if exact else sample_data
        values = maker(sigma, n, seed)
        frame = pd.DataFrame(values, columns=cm.item_names)
        result = estimate(frame, spec, estimator=estimator,
                          observed_se=observed_se)
        _CACHE[key] = (result, cm, sigma)
    return _CACHE[key]  # type: ignore[return-value]


def _std_of(table: pd.DataFrame, source: str, target: str) -> float:
    row = table[(table["source"] == source) & (table["target"] == target)]
    assert len(row) == 1, f"expected one {source}->{target} row, got {len(row)}"
    return float(row["std"].iloc[0])


# --------------------------------------------------------------------------- #
# 1. Degrees of freedom
# --------------------------------------------------------------------------- #


def test_minimal_cfa_degrees_of_freedom():
    """df = p(p+1)/2 - n_free = 21 - 13 = 8 for the minimal two-factor CFA."""
    result, cm, _ = fitted("minimal_cfa", 400, 101, exact=True)
    p = cm.n_items
    moments = p * (p + 1) // 2
    assert p == 6
    assert moments == 21
    assert result.n_free == 13, f"n_free was {result.n_free}"
    assert result.df == 8, f"df was {result.df}"
    assert result.df == moments - result.n_free


@pytest.mark.parametrize("name", ["minimal_cfa", "simple_mediation",
                                  "second_order", "messy_reality"])
def test_df_identity_holds(name):
    """df is always ``p(p+1)/2 - n_free``, whatever the model."""
    result, cm, _ = fitted(name, 400, 202, exact=True)
    moments = cm.n_items * (cm.n_items + 1) // 2
    assert result.df == moments - result.n_free
    assert result.df >= 0


# --------------------------------------------------------------------------- #
# 2. Population fit: the sharpest test of the parameterisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["minimal_cfa", "simple_mediation",
                                  "second_order", "messy_reality"])
def test_population_fit_drives_f_to_zero(name):
    """Fitting the exact population Sigma must give F ~ 0 and chi-square ~ 0."""
    result, _, sigma = fitted(name, 400, 303, exact=True)
    assert np.allclose(result.sample_cov, sigma, atol=1e-10)
    assert result.converged
    assert result.fmin < 1e-6, f"{name}: fmin = {result.fmin!r}"
    assert result.chi_square < 1e-3, f"{name}: chi2 = {result.chi_square!r}"


def test_population_fit_reproduces_the_standardised_solution_exactly():
    """At a population fit the standardised solution *is* the specification."""
    result, _, _ = fitted("minimal_cfa", 400, 303, exact=True)
    std = result.loadings["std"].to_numpy(dtype=float)
    assert np.allclose(std, 0.80, atol=1e-6), np.round(std, 8)
    assert abs(result.standardised["F1~~F2"] - 0.40) < 1e-6


def test_population_fit_recovers_second_order_loadings_exactly():
    result, _, _ = fitted("second_order", 400, 303, exact=True)
    table = result.second_order_loadings
    assert table is not None and len(table) == 3
    std = table["std"].to_numpy(dtype=float)
    assert np.allclose(std, 0.82, atol=1e-6), np.round(std, 8)


def test_r_squared_matches_the_closed_form():
    """R-squared = 1 - Psi_ii / Phi_ii equals the path algebra of the model."""
    result, _, _ = fitted("simple_mediation", 400, 303, exact=True)
    # ENG ~ 0.50 SAT                       -> R2 = 0.25
    # PERF ~ 0.30 ENG + 0.15 SAT, r = 0.50 -> R2 = .09 + .0225 + 2*.3*.15*.5
    assert abs(result.r_squared["ENG"] - 0.25) < 1e-6
    assert abs(result.r_squared["PERF"] - 0.1575) < 1e-6


# --------------------------------------------------------------------------- #
# 3. Parameter recovery from real samples
# --------------------------------------------------------------------------- #


def test_minimal_cfa_parameter_recovery_at_n_5000():
    """Every standardised loading within 0.03 of 0.80, factor r within 0.04."""
    result, _, _ = fitted("minimal_cfa", 5000, 20240101)
    assert result.converged
    std = result.loadings["std"].to_numpy(dtype=float)
    worst = float(np.max(np.abs(std - 0.80)))
    assert worst < 0.03, f"loadings {np.round(std, 4)} (worst dev {worst:.4f})"
    phi = result.standardised["F1~~F2"]
    assert abs(phi - 0.40) < 0.04, f"factor correlation {phi:.4f}"


def test_mediation_path_recovery_at_n_2000():
    """The three standardised paths come back within 0.05 of 0.50/0.30/0.15."""
    result, _, _ = fitted("simple_mediation", 2000, 777)
    assert result.converged
    table = result.path_estimates
    targets = {("SAT", "ENG"): 0.50, ("ENG", "PERF"): 0.30,
               ("SAT", "PERF"): 0.15}
    for (source, target), expected in targets.items():
        got = _std_of(table, source, target)
        assert abs(got - expected) < 0.05, (
            f"{source}->{target}: {got:.4f} vs {expected}")


def test_second_order_loading_recovery_at_n_2000():
    """The three second-order loadings come back within 0.06 of 0.82."""
    result, _, _ = fitted("second_order", 2000, 4242)
    assert result.converged
    std = result.second_order_loadings["std"].to_numpy(dtype=float)
    worst = float(np.max(np.abs(std - 0.82)))
    assert worst < 0.06, f"second-order loadings {np.round(std, 4)}"


# --------------------------------------------------------------------------- #
# 4. Standard errors
# --------------------------------------------------------------------------- #


def test_expected_and_observed_standard_errors_agree():
    """Expected and observed information must agree to within 5%."""
    result, _, _ = fitted("simple_mediation", 1000, 31415)
    assert result.standard_errors_observed, "observed SEs were not computed"
    worst_label, worst = "", 0.0
    for label, a in result.standard_errors.items():
        b = result.standard_errors_observed[label]
        assert np.isfinite(a) and np.isfinite(b), label
        relative = abs(a - b) / a
        if relative > worst:
            worst_label, worst = label, relative
    assert worst < 0.05, f"worst disagreement {worst:.4%} on {worst_label}"


def test_standard_errors_shrink_as_sqrt_n():
    """SE(4000) / SE(1000) must be near sqrt(999/3999) = 0.4998."""
    small, _, _ = fitted("minimal_cfa", 1000, 5, exact=True, observed_se=False)
    large, _, _ = fitted("minimal_cfa", 4000, 5, exact=True, observed_se=False)
    expected = math.sqrt(999.0 / 3999.0)
    ratios = np.asarray([large.standard_errors[k] / small.standard_errors[k]
                         for k in small.standard_errors], dtype=float)
    assert np.all(np.isfinite(ratios))
    worst = float(np.max(np.abs(ratios / 0.5 - 1.0)))
    assert worst < 0.15, f"ratios {np.round(ratios, 4)}"
    assert abs(float(np.median(ratios)) - expected) < 0.01


def test_wald_tests_follow_from_the_estimates():
    """z = est / SE and p = 2 * norm.sf(|z|), recomputed independently."""
    result, _, _ = fitted("simple_mediation", 1000, 31415)
    for label, estimate_value in result.parameters.items():
        se = result.standard_errors[label]
        z = estimate_value / se
        assert abs(result.z_values[label] - z) < 1e-10, label
        assert abs(result.p_values[label] - 2.0 * norm.sf(abs(z))) < 1e-12


def test_loading_standard_errors_are_absent_for_fixed_markers():
    result, _, _ = fitted("minimal_cfa", 400, 303, exact=True)
    markers = result.loadings[result.loadings["fixed"]]
    assert len(markers) == 2
    assert np.allclose(markers["est"].to_numpy(dtype=float), 1.0)
    assert np.all(np.isnan(markers["se"].to_numpy(dtype=float)))


# --------------------------------------------------------------------------- #
# 5. Linear algebra identities
# --------------------------------------------------------------------------- #


def test_duplication_matrix_identity():
    """vec(A) = D vech(A) and unvech(vech(A)) = A."""
    rng = np.random.default_rng(11)
    for p in (1, 2, 5, 7):
        a = rng.normal(size=(p, p))
        a = a + a.T
        d = duplication_matrix(p)
        assert d.shape == (p * p, p * (p + 1) // 2)
        assert np.allclose(d @ vech(a), a.reshape(-1, order="F"))
        assert np.allclose(unvech(vech(a), p), a)


def test_expected_information_two_forms_agree():
    """D-form and trace-form expected information agree to machine precision.

    ``Delta' D' (Sigma^-1 (x) Sigma^-1) D Delta`` must equal the matrix with
    entries ``tr(dSigma_a Sigma^-1 dSigma_b Sigma^-1)`` (Magnus & Neudecker,
    1999), which is an independent recomputation of the whole information
    matrix.
    """
    result, cm, _ = fitted("minimal_cfa", 1000, 5, exact=True,
                           observed_se=False)
    par = est._build_parameterisation(cm, [])
    delta = est._sigma_jacobian(par, result.theta)
    kron_form = est._information_from_delta(delta, result.sigma_hat,
                                            use_kronecker=True)
    trace_form = est._information_from_delta(delta, result.sigma_hat,
                                             use_kronecker=False)
    scale = float(np.max(np.abs(kron_form)))
    assert np.allclose(kron_form, trace_form, atol=1e-9 * scale), (
        float(np.max(np.abs(kron_form - trace_form))))


def test_sigma_hat_reproduces_the_model_equations():
    """Sigma_hat = Lambda Phi Lambda' + Theta, recomputed from the estimates."""
    result, cm, _ = fitted("simple_mediation", 400, 303, exact=True)
    par = est._build_parameterisation(cm, [])
    Lambda, B, Psi, Theta = par.matrices(result.theta)
    inv = np.linalg.inv(np.eye(par.m) - B)
    phi = inv @ Psi @ inv.T
    sigma = Lambda @ phi @ Lambda.T + Theta
    assert np.allclose(sigma, result.sigma_hat, atol=1e-10)
    assert np.allclose(phi, result.latent_covariance, atol=1e-10)


# --------------------------------------------------------------------------- #
# 6. Discrepancy functions
# --------------------------------------------------------------------------- #


def test_ml_discrepancy_is_zero_at_equality_and_positive_elsewhere():
    rng = np.random.default_rng(13)
    a = rng.normal(size=(8, 5))
    s = a.T @ a / 8.0
    assert abs(ml_discrepancy(s, s)) < 1e-12
    perturbed = s + 0.05 * np.eye(5)
    assert ml_discrepancy(s, perturbed) > 0.0


def test_ml_discrepancy_known_closed_form():
    """For Sigma = c S the discrepancy is p (log c + 1/c - 1)."""
    rng = np.random.default_rng(17)
    a = rng.normal(size=(12, 4))
    s = a.T @ a / 12.0
    for c in (0.5, 1.5, 3.0):
        expected = 4.0 * (math.log(c) + 1.0 / c - 1.0)
        assert abs(ml_discrepancy(s, c * s) - expected) < 1e-10


def test_uls_discrepancy_known_closed_form():
    s = np.eye(3)
    sigma = np.eye(3) * 1.2
    # 0.5 * tr((S - Sigma)^2) = 0.5 * 3 * 0.2^2
    assert abs(uls_discrepancy(s, sigma) - 0.06) < 1e-12


def test_non_positive_definite_sigma_returns_infinity():
    """A non-PD Sigma returns +inf; it never raises, so the search backs off."""
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert ml_discrepancy(np.eye(2), bad) == float("inf")
    assert uls_discrepancy(np.eye(2), bad) == float("inf")
    assert est.dwls_discrepancy(np.eye(2), bad, np.ones(3)) == float("inf")


@pytest.mark.parametrize("kind", [EstimatorKind.ULS, EstimatorKind.DWLS,
                                  EstimatorKind.MLR])
def test_other_estimators_also_reach_the_population_solution(kind):
    result, _, _ = fitted("minimal_cfa", 500, 909, exact=True, estimator=kind,
                          observed_se=False)
    assert result.converged
    assert result.fmin < 1e-6, f"{kind.value}: fmin = {result.fmin!r}"
    std = result.loadings["std"].to_numpy(dtype=float)
    assert np.allclose(std, 0.80, atol=1e-5), np.round(std, 6)


# --------------------------------------------------------------------------- #
# 7. Starting values and the optimiser chain
# --------------------------------------------------------------------------- #


def _problem_for(cm, sample_cov):
    par = est._build_parameterisation(cm, [])
    logdet, _ = est._logdet_cholesky(sample_cov)
    problem = est._Problem(par=par, kind=EstimatorKind.ML,
                           sample_cov=sample_cov, logdet_sample=float(logdet),
                           weights=est._normal_theory_weights(sample_cov))
    return par, problem


def test_optimiser_recovers_the_optimum_from_a_perturbed_start():
    """Shifting every start value by up to 30% still lands on F ~ 0."""
    result, cm, sigma = fitted("minimal_cfa", 1000, 5, exact=True,
                               observed_se=False)
    par, problem = _problem_for(cm, result.sample_cov)
    rng = np.random.default_rng(3)
    x0 = par.clip(par.start_vector() * (1.0 + rng.uniform(-0.3, 0.3, par.q)))
    assert problem.value(x0) > 1e-3, "the perturbation did not move F"
    _, fmin, converged, _, _ = est._run_optimiser(problem, x0)
    assert converged
    assert fmin < 1e-6, f"fmin from a perturbed start = {fmin!r}"


def test_two_step_start_alone_reaches_the_optimum():
    """The factor-analytic fallback start converges without the compiled one."""
    result, cm, _ = fitted("minimal_cfa", 1000, 5, exact=True,
                           observed_se=False)
    par, problem = _problem_for(cm, result.sample_cov)
    start = est._two_step_start(par, result.sample_cov, None)
    assert np.all(np.isfinite(start))
    _, fmin, converged, _, _ = est._run_optimiser(problem, start)
    assert converged
    assert fmin < 1e-6, f"fmin from the two-step start = {fmin!r}"


def test_variance_parameters_are_bounded_away_from_zero():
    result, cm, _ = fitted("minimal_cfa", 400, 303, exact=True)
    par = est._build_parameterisation(cm, [])
    for (low, _high), prm in zip(par.bounds(), par.parameters):
        if prm.kind in (est.KIND_LATENT_VARIANCE, est.KIND_RESIDUAL_VARIANCE):
            assert low == est.VARIANCE_FLOOR
        else:
            assert low is None
    for label, value in result.parameters.items():
        if result.parameter_kinds[label] in (est.KIND_LATENT_VARIANCE,
                                             est.KIND_RESIDUAL_VARIANCE):
            assert value >= est.VARIANCE_FLOOR


# --------------------------------------------------------------------------- #
# 8. Latent interactions and user-facing failures
# --------------------------------------------------------------------------- #


def test_latent_interaction_is_omitted_and_reported():
    """A moderation model fits its main effects and says so in Arabic."""
    spec = examples.moderation()
    cm = compile_model(spec)
    values = exact_covariance_data(implied_covariance(cm), 400, 11)
    result = estimate(pd.DataFrame(values, columns=cm.item_names), spec,
                      observed_se=False)
    assert result.fmin < 1e-6
    assert "INTERACTION_NOT_ESTIMATED" in result.message
    assert "التفاعل" in result.message or "تفاعل" in result.message
    labels = " ".join(result.parameters)
    assert ":" not in labels, "no interaction term should appear as a parameter"


def test_missing_item_column_raises_a_bilingual_specification_error():
    spec = examples.minimal_cfa()
    cm = compile_model(spec)
    frame = pd.DataFrame(exact_covariance_data(implied_covariance(cm), 200, 2),
                         columns=cm.item_names)
    with pytest.raises(SpecificationError) as info:
        estimate(frame.drop(columns=[cm.item_names[1]]), spec)
    assert cm.item_names[1] in info.value.message_en
    assert "البنود" in info.value.message_ar


def test_sample_smaller_than_the_item_count_raises_identification_error():
    spec = examples.minimal_cfa()
    cm = compile_model(spec)
    frame = pd.DataFrame(exact_covariance_data(implied_covariance(cm), 200, 2),
                         columns=cm.item_names)
    with pytest.raises(IdentificationError) as info:
        estimate(frame.iloc[:4], spec)
    assert "مصفوفة تغاير العينة" in info.value.message_ar


def test_negative_degrees_of_freedom_raises_identification_error():
    spec = ModelSpec(seed=1)
    spec.add_construct("A", n_items=2, loading=0.8)
    spec.ensure_unique_item_names()
    cm = compile_model(spec)
    frame = pd.DataFrame(exact_covariance_data(implied_covariance(cm), 100, 3),
                         columns=cm.item_names)
    with pytest.raises(IdentificationError) as info:
        estimate(frame, spec)
    assert "درجات الحرية" in info.value.message_ar


def test_singular_sample_covariance_raises_specification_error():
    spec = examples.minimal_cfa()
    cm = compile_model(spec)
    frame = pd.DataFrame(exact_covariance_data(implied_covariance(cm), 200, 2),
                         columns=cm.item_names)
    frame[cm.item_names[2]] = frame[cm.item_names[0]]
    with pytest.raises(SpecificationError) as info:
        estimate(frame, spec)
    assert "مفردة" in info.value.message_ar


def test_missing_values_are_deleted_listwise_and_reported():
    spec = examples.minimal_cfa()
    cm = compile_model(spec)
    frame = pd.DataFrame(exact_covariance_data(implied_covariance(cm), 200, 2),
                         columns=cm.item_names)
    frame.iloc[0, 0] = np.nan
    result = estimate(frame, spec, observed_se=False)
    assert result.n == 199
    assert "LISTWISE_DELETION" in result.message
    assert "الحذف بالقائمة" in result.message


# --------------------------------------------------------------------------- #
# Direct execution fallback
# --------------------------------------------------------------------------- #


def _run_without_pytest() -> int:  # pragma: no cover - convenience path
    import inspect

    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        signature = inspect.signature(function)
        marks = getattr(function, "pytestmark", [])
        cases = [{}]
        for mark in marks:
            if mark.name == "parametrize":
                key = mark.args[0]
                cases = [{key: value} for value in mark.args[1]]
        for case in cases:
            label = f"{name}{tuple(case.values()) if case else ''}"
            try:
                function(**{k: v for k, v in case.items()
                            if k in signature.parameters})
                print(f"PASS {label}")
            except Exception as exc:
                failures += 1
                print(f"FAIL {label}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_run_without_pytest())

"""
Numerical tests for :mod:`semsim.targeting.cudeck_browne`.

Nothing here settles for "the code ran".  Every claim the module makes is
checked against a closed form or against an independent recomputation:

* the realised population RMSEA is recomputed from scratch as
  ``sqrt(F_ML(Sigma_0, Sigma(theta)) / df)`` with ``F_ML`` written out by hand
  from ``log|Sigma| - log|Sigma_0| + tr(Sigma_0 Sigma^-1) - p``, and has to
  match the request to within 1e-6 for four targets on three models;
* ``population_discrepancy`` is checked against the hand-worked closed form
  ``-log(a) - log(b) + a + b - 2`` for ``Sigma = I_2``,
  ``Sigma_0 = diag(a, b)``;
* ``ml_weight_matrix`` is checked against the literal
  ``0.5 D' (Sigma^-1 (x) Sigma^-1) D`` with the Kronecker product formed
  explicitly;
* the parameter-preservation claim -- the whole point of Cudeck & Browne -- is
  checked three independent ways: the analytic gradient
  ``tr(Sigma^-1 (Sigma - Sigma_0) Sigma^-1 dSigma/dtheta_i)`` must be zero at
  ``theta``, the numerical Hessian there must be positive definite, and the
  model must be *re-fitted* to a dataset whose sample covariance is exactly
  ``Sigma_0`` (so :func:`semsim.analysis.estimator.estimate` sees ``Sigma_0``
  as its ``S``) and give back ``theta`` and ``F_0``;
* the ``"residual"`` method is checked to *fail* that gradient test, which is
  what proves the projection is doing the work rather than decorating it.

Run with::

    python3 -m pytest tests/test_cudeck_browne.py -q

The file also works as a plain script (``python3 tests/test_cudeck_browne.py``)
and then prints PASS/FAIL per check and exits non-zero on failure.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semsim.core import examples                                  # noqa: E402
from semsim.core.compile import compile_model                     # noqa: E402
from semsim.core.errors import (InfeasibleTargetError,            # noqa: E402
                                SemSimError, SpecificationError)
from semsim.core.implied import (implied_covariance,              # noqa: E402
                                 is_positive_definite)
from semsim.targeting import cudeck_browne as cb                  # noqa: E402

TARGETS = (0.02, 0.045, 0.07, 0.10)
MODELS: Tuple[Tuple[str, Callable[[], object]], ...] = (
    ("minimal_cfa", examples.minimal_cfa),
    ("simple_mediation", examples.simple_mediation),
    ("second_order_hierarchy", examples.second_order_hierarchy),
)

#: Compiled models are reused across checks; compiling is pure and cheap but
#: the Jacobians are not.
_CACHE: Dict[str, object] = {}


def _compiled(name: str):
    if name not in _CACHE:
        builder = dict(MODELS)[name]
        _CACHE[name] = compile_model(builder())
    return _CACHE[name]


def _say(text: str) -> None:
    """Print an observed value so the run is self-documenting."""
    print(text, flush=True)


# --------------------------------------------------------------------------- #
# Independent recomputations (deliberately not importing the module's helpers)
# --------------------------------------------------------------------------- #


def _f_ml_by_hand(sigma_0: np.ndarray, sigma: np.ndarray) -> float:
    """``log|Sigma| - log|Sigma_0| + tr(Sigma_0 Sigma^-1) - p``, written out.

    Uses ``slogdet`` and an explicit inverse rather than anything from the
    module under test, so a sign slip or a swapped argument there cannot hide.
    """
    s0 = np.asarray(sigma_0, dtype=float)
    s = np.asarray(sigma, dtype=float)
    p = s.shape[0]
    sign_s, logdet_s = np.linalg.slogdet(s)
    sign_0, logdet_0 = np.linalg.slogdet(s0)
    assert sign_s > 0 and sign_0 > 0, (sign_s, sign_0)
    return float(logdet_s - logdet_0
                 + np.trace(s0 @ np.linalg.inv(s)) - p)


def _rmsea_by_hand(sigma_0: np.ndarray, sigma: np.ndarray, df: int) -> float:
    return math.sqrt(_f_ml_by_hand(sigma_0, sigma) / float(df))


def _analytic_gradient(pmap, sigma_0: np.ndarray,
                       theta: Optional[np.ndarray] = None) -> np.ndarray:
    """``dF_ML(Sigma_0, Sigma(theta)) / d theta`` at ``theta``.

    ``dF = tr((Sigma^-1 - Sigma^-1 Sigma_0 Sigma^-1) dSigma)``, so the
    gradient entry for parameter ``i`` is
    ``tr(Sigma^-1 (Sigma - Sigma_0) Sigma^-1 dSigma/dtheta_i)``.  Zero for
    every ``i`` *is* the statement "theta is a stationary point", and combined
    with a positive-definite Hessian it is the statement "theta is the exact
    minimiser".
    """
    base = pmap.theta0 if theta is None else np.asarray(theta, dtype=float)
    sigma = pmap.sigma_of(base)
    assert sigma is not None
    inv = np.linalg.inv(sigma)
    middle = inv @ (sigma - np.asarray(sigma_0, dtype=float)) @ inv
    delta = cb.model_jacobian(pmap, theta=base)
    p = sigma.shape[0]
    return np.asarray(
        [float(np.trace(middle @ cb.unvech(delta[:, i], p)))
         for i in range(delta.shape[1])], dtype=float)


def _numerical_hessian(pmap, sigma_0: np.ndarray, step: float = 1e-4
                       ) -> np.ndarray:
    """Hessian of ``F_ML(Sigma_0, Sigma(.))`` at ``pmap.theta0``."""
    t0 = np.asarray(pmap.theta0, dtype=float)
    q = t0.size

    def f(x: np.ndarray) -> float:
        sigma = pmap.sigma_of(x)
        assert sigma is not None
        return _f_ml_by_hand(sigma_0, sigma)

    h = np.asarray([step * max(1.0, abs(float(v))) for v in t0], dtype=float)
    base = f(t0)
    hess = np.zeros((q, q), dtype=float)
    for i in range(q):
        for j in range(i, q):
            if i == j:
                up, down = t0.copy(), t0.copy()
                up[i] += h[i]
                down[i] -= h[i]
                hess[i, i] = (f(up) - 2.0 * base + f(down)) / (h[i] ** 2)
            else:
                pp, pm, mp, mm = (t0.copy(), t0.copy(), t0.copy(), t0.copy())
                pp[i] += h[i]; pp[j] += h[j]
                pm[i] += h[i]; pm[j] -= h[j]
                mp[i] -= h[i]; mp[j] += h[j]
                mm[i] -= h[i]; mm[j] -= h[j]
                value = ((f(pp) - f(pm) - f(mp) + f(mm))
                         / (4.0 * h[i] * h[j]))
                hess[i, j] = hess[j, i] = value
    return 0.5 * (hess + hess.T)


def _gauss_newton_refit(pmap, sigma_0: np.ndarray, start: np.ndarray,
                        max_iter: int = 100) -> Tuple[np.ndarray, float]:
    """Minimise ``F_ML(Sigma_0, Sigma(theta))`` over ``pmap`` from ``start``.

    Gauss-Newton with a halving line search, the natural iteration for an ML
    covariance-structure problem: the step solves
    ``(Delta' W Delta) d = Delta' W (vech(Sigma_0) - vech(Sigma))``.
    """
    theta = np.asarray(start, dtype=float).copy()
    for _ in range(max_iter):
        sigma = pmap.sigma_of(theta)
        if sigma is None:
            break
        current = _f_ml_by_hand(sigma_0, sigma)
        weight = cb.ml_weight_matrix(sigma)
        delta = cb.model_jacobian(pmap, theta=theta)
        residual = cb.vech(sigma_0) - cb.vech(sigma)
        info = delta.T @ weight @ delta
        rhs = delta.T @ weight @ residual
        try:
            direction = np.linalg.solve(info, rhs)
        except np.linalg.LinAlgError:
            direction = np.linalg.pinv(info) @ rhs
        lam = 1.0
        improved = False
        for _ in range(50):
            candidate = theta + lam * direction
            trial = pmap.sigma_of(candidate)
            if trial is not None:
                value = _f_ml_by_hand(sigma_0, trial)
                if value < current - 1e-18:
                    theta = candidate
                    current = value
                    improved = True
                    break
            lam *= 0.5
        if not improved or np.max(np.abs(lam * direction)) < 1e-14:
            break
    sigma = pmap.sigma_of(theta)
    assert sigma is not None
    return theta, _f_ml_by_hand(sigma_0, sigma)


def _dataset_with_exact_covariance(sigma_0: np.ndarray, n: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """A sample of size ``n`` whose unbiased covariance equals ``sigma_0``.

    Any centred sample can be linearly re-coloured so its sample covariance is
    exactly the target: ``Y = X L_x^-1' L_0'`` with ``L_x L_x' = S_x`` and
    ``L_0 L_0' = Sigma_0``.  This lets the real estimator be driven with
    ``S == Sigma_0`` so the parameter-preservation claim is tested on the
    population quantity, with no sampling error in the way.
    """
    p = sigma_0.shape[0]
    x = rng.standard_normal((n, p))
    x -= x.mean(axis=0)
    s_x = x.T @ x / (n - 1)
    return x @ np.linalg.inv(np.linalg.cholesky(s_x)).T \
        @ np.linalg.cholesky(sigma_0).T


# --------------------------------------------------------------------------- #
# 1. Linear-algebra building blocks, against their definitions
# --------------------------------------------------------------------------- #


def test_vech_unvech_and_duplication_match_their_definitions() -> None:
    rng = np.random.default_rng(0)
    for p in (1, 2, 3, 5, 7):
        a = rng.standard_normal((p, p))
        a = 0.5 * (a + a.T)
        v = cb.vech(a)
        assert v.size == p * (p + 1) // 2
        assert np.allclose(cb.unvech(v, p), a, atol=0.0, rtol=0.0)
        d = cb.duplication_matrix(p)
        assert np.allclose(d @ v, a.reshape(-1, order="F"))
    _say("duplication/vech identities exact for p in {1,2,3,5,7}")


def test_ml_weight_matrix_equals_the_explicit_kronecker_form() -> None:
    rng = np.random.default_rng(1)
    worst = 0.0
    for p in (2, 3, 4, 6):
        a = rng.standard_normal((p, p))
        sigma = a @ a.T + p * np.eye(p)
        inv = np.linalg.inv(sigma)
        d = cb.duplication_matrix(p)
        reference = 0.5 * d.T @ np.kron(inv, inv) @ d
        got = cb.ml_weight_matrix(sigma)
        worst = max(worst, float(np.max(np.abs(got - reference))))
        assert np.allclose(got, reference, atol=1e-10, rtol=0.0), p
        assert np.allclose(got, got.T, atol=1e-12)
        assert np.linalg.eigvalsh(got).min() > 0.0
    _say(f"ml_weight_matrix vs 0.5 D'(S^-1 (x) S^-1) D: max |diff| = {worst:.3e}")


def test_population_discrepancy_matches_the_closed_form() -> None:
    # Sigma = I_2, Sigma_0 = diag(a, b):
    #   F = log|I| - log(ab) + tr(diag(a, b)) - 2 = a + b - log a - log b - 2
    sigma = np.eye(2)
    for a, b in ((1.0, 1.0), (1.5, 0.8), (2.0, 2.0), (0.25, 4.0)):
        closed = a + b - math.log(a) - math.log(b) - 2.0
        got = cb.population_discrepancy(np.diag([a, b]), sigma)
        assert abs(got - closed) < 1e-14, (a, b, got, closed)
    assert cb.population_discrepancy(sigma, sigma) == 0.0
    # Not positive definite -> +inf, never an exception.
    assert cb.population_discrepancy(np.diag([-1.0, 1.0]), sigma) == float("inf")
    _say("population_discrepancy matches a + b - log a - log b - 2 to 1e-14")


def test_rmsea_and_discrepancy_are_exact_inverses() -> None:
    for df in (1, 8, 74, 164):
        for r in (0.001, 0.02, 0.045, 0.07, 0.1, 0.3):
            f0 = cb.discrepancy_for_rmsea(r, df)
            assert abs(f0 - df * r * r) < 1e-15
            assert abs(cb.rmsea_from_discrepancy(f0, df) - r) < 1e-14
    assert cb.discrepancy_for_rmsea(0.05, 0) == 0.0
    assert cb.rmsea_from_discrepancy(1.0, 0) == 0.0
    assert cb.rmsea_from_discrepancy(-1.0, 10) == 0.0
    _say("F_0 = df * rmsea^2 and its inverse agree to 1e-14")


def test_tangent_projector_is_idempotent_and_w_orthogonal() -> None:
    cm = _compiled("minimal_cfa")
    sigma = implied_covariance(cm)
    pmap = cb.free_parameter_map(cm)
    delta = cb.model_jacobian(pmap)
    weight = cb.ml_weight_matrix(sigma)
    proj = cb.tangent_projector(delta, weight)
    assert np.allclose(proj @ proj, proj, atol=1e-8), \
        float(np.max(np.abs(proj @ proj - proj)))
    # W P symmetric <=> P self-adjoint in the W metric.
    wp = weight @ proj
    assert np.allclose(wp, wp.T, atol=1e-6 * float(np.max(np.abs(wp))))
    # P leaves the tangent space alone and (I - P) annihilates it.
    assert np.allclose(proj @ delta, delta, atol=1e-8)
    # trace(P) == q == rank of the tangent space.
    assert abs(float(np.trace(proj)) - pmap.q) < 1e-6, float(np.trace(proj))
    _say(f"projector idempotent, trace(P) = {float(np.trace(proj)):.9f}, "
         f"q = {pmap.q}")


def test_builtin_and_estimator_parameterisations_span_the_same_tangent_space(
) -> None:
    """The construction must not depend on which mapping was available."""
    for name, _ in MODELS:
        cm = _compiled(name)
        sigma = implied_covariance(cm)
        weight = cb.ml_weight_matrix(sigma)
        borrowed = cb.free_parameter_map(cm, prefer_estimator=True)
        own = cb.free_parameter_map(cm, prefer_estimator=False)
        assert np.allclose(borrowed.sigma_of(borrowed.theta0), sigma, atol=1e-12)
        assert np.allclose(own.sigma_of(own.theta0), sigma, atol=1e-12)
        assert borrowed.q == own.q, (name, borrowed.q, own.q)
        p_a = cb.tangent_projector(cb.model_jacobian(borrowed), weight)
        p_b = cb.tangent_projector(cb.model_jacobian(own), weight)
        gap = float(np.max(np.abs(p_a - p_b)))
        assert gap < 1e-6, (name, gap)
        _say(f"{name}: parameterisations '{borrowed.source}' and "
             f"'{own.source}' both q = {own.q}, projector max |diff| = "
             f"{gap:.3e}")


# --------------------------------------------------------------------------- #
# 2. The headline property: the realised RMSEA hits the target
# --------------------------------------------------------------------------- #


def test_achieved_rmsea_matches_target_on_every_model_and_target() -> None:
    worst = 0.0
    for name, _ in MODELS:
        cm = _compiled(name)
        sigma = implied_covariance(cm)
        for target in TARGETS:
            out = cb.inject_model_error(
                cm, target, rng=np.random.default_rng(20240101))
            independent = _rmsea_by_hand(out.sigma_0, sigma, out.df)
            error = abs(independent - target)
            worst = max(worst, error)
            _say(f"{name:24s} target={target:<6.3f} df={out.df:<4d} "
                 f"reported={out.achieved_rmsea:.12f} "
                 f"recomputed={independent:.12f} |err|={error:.3e} "
                 f"k={out.scale:.6f} F_0={out.achieved_fmin:.12f}")
            assert out.converged, name
            assert error < 1e-6, (name, target, independent)
            # The module's own report must agree with the recomputation.
            assert abs(out.achieved_rmsea - independent) < 1e-9
            # F_0 = df * target^2 exactly.
            assert abs(out.achieved_fmin - out.df * target ** 2) < 1e-9
            assert out.target_rmsea == target
            assert out.method == "cudeck_browne"
    _say(f"worst |recomputed RMSEA - target| over 12 cases = {worst:.3e}")
    assert worst < 1e-6


def test_sigma_0_is_symmetric_and_positive_definite_everywhere() -> None:
    for name, _ in MODELS:
        cm = _compiled(name)
        for target in TARGETS:
            out = cb.inject_model_error(
                cm, target, rng=np.random.default_rng(777))
            s0 = out.sigma_0
            assert np.array_equal(s0, s0.T), (name, target)
            eigs = np.linalg.eigvalsh(s0)
            assert is_positive_definite(s0), (name, target, eigs.min())
            assert eigs.min() > 0.0
            assert np.allclose(np.diag(s0), np.diag(s0))  # finite, no nan
            _say(f"{name:24s} target={target:<6.3f} "
                 f"min eig(Sigma_0) = {eigs.min():.6f} "
                 f"cond = {eigs.max() / eigs.min():.2f}")


def test_more_misfit_needs_a_bigger_step_along_the_same_direction() -> None:
    """``F`` is strictly increasing in ``k``, so ``k`` must be monotone."""
    cm = _compiled("simple_mediation")
    scales = [cb.inject_model_error(cm, t, rng=np.random.default_rng(5)).scale
              for t in TARGETS]
    assert all(b > a for a, b in zip(scales, scales[1:])), scales
    _say(f"k for targets {TARGETS} = "
         + ", ".join(f"{s:.6f}" for s in scales))


def test_sigma_0_keeps_a_unit_diagonal_because_theta_is_free() -> None:
    """A closed-form consequence of the projection, worth pinning down.

    Every residual variance is a free parameter, so ``dSigma/dTheta_jj = E_jj``
    and the tangent space contains all ``p`` diagonal directions.  The
    projection therefore removes the whole diagonal component of the random
    direction: ``diag(E) == 0`` and ``Sigma_0`` stays a correlation matrix with
    a unit diagonal, which is what the Vale-Maurelli generator downstream
    assumes.  (The unprojected ``'residual'`` method has no such property.)
    """
    for name, _ in MODELS:
        cm = _compiled(name)
        out = cb.inject_model_error(cm, 0.10, rng=np.random.default_rng(13))
        diag_e = float(np.max(np.abs(np.diag(out.direction))))
        off_e = float(np.max(np.abs(out.direction - np.diag(np.diag(
            out.direction)))))
        diag_gap = float(np.max(np.abs(np.diag(out.sigma_0) - 1.0)))
        _say(f"{name:24s} max |diag(E)| = {diag_e:.3e} "
             f"(max |offdiag(E)| = {off_e:.3e}), "
             f"max |diag(Sigma_0) - 1| = {diag_gap:.3e}")
        assert diag_e < 1e-9, (name, diag_e)
        assert diag_gap < 1e-9, (name, diag_gap)
    plain = cb.inject_model_error(_compiled("minimal_cfa"), 0.10,
                                 rng=np.random.default_rng(13),
                                 method="residual")
    assert float(np.max(np.abs(np.diag(plain.direction)))) > 1e-3
    _say("the unprojected 'residual' direction does perturb the diagonal")


# --------------------------------------------------------------------------- #
# 3. The point of the whole exercise: theta is still the exact minimiser
# --------------------------------------------------------------------------- #


def test_theta_is_a_stationary_point_of_the_perturbed_discrepancy() -> None:
    """Analytic first-order condition, the exact algebraic statement."""
    for name, _ in MODELS:
        cm = _compiled(name)
        pmap = cb.free_parameter_map(cm)
        out = cb.inject_model_error(cm, 0.07, rng=np.random.default_rng(11))
        grad = _analytic_gradient(pmap, out.sigma_0)
        # Scale: the gradient is O(k * ||W|| * ||Delta||); compare against the
        # same quantity for an unprojected direction (see the residual test).
        assert np.max(np.abs(grad)) < 1e-10, (name, np.max(np.abs(grad)))
        _say(f"{name:24s} max |dF/dtheta| at theta = "
             f"{np.max(np.abs(grad)):.3e} (q = {pmap.q})")


def test_the_stationary_point_is_a_minimum_not_a_saddle() -> None:
    cm = _compiled("minimal_cfa")
    pmap = cb.free_parameter_map(cm)
    out = cb.inject_model_error(cm, 0.07, rng=np.random.default_rng(11))
    hess = _numerical_hessian(pmap, out.sigma_0)
    smallest = float(np.linalg.eigvalsh(hess).min())
    assert smallest > 0.0, smallest
    _say(f"minimal_cfa: smallest Hessian eigenvalue at theta = {smallest:.6f}")


def test_refitting_the_model_to_sigma_0_recovers_theta_and_f_0() -> None:
    """The parameter-preservation test, run through the real estimator.

    A dataset is built whose sample covariance is *exactly* ``Sigma_0``, so
    :func:`semsim.analysis.estimator.estimate` minimises
    ``F_ML(Sigma_0, Sigma(.))`` on the population quantity.  The recovered
    parameters must be ``theta`` and the attained discrepancy must be ``F_0``.
    """
    try:
        from semsim.analysis import estimator as est
    except Exception as exc:                                  # pragma: no cover
        raise AssertionError(
            "semsim.analysis.estimator is not importable, so the "
            f"parameter-preservation test could not be run: {exc!r}") from exc

    for name, builder in MODELS:
        spec = builder()
        cm = compile_model(spec)
        pmap = cb.free_parameter_map(cm)
        assert pmap.source == "estimator", pmap.source
        out = cb.inject_model_error(cm, 0.07, rng=np.random.default_rng(11))
        data = _dataset_with_exact_covariance(
            out.sigma_0, 600, np.random.default_rng(2024))
        frame = pd.DataFrame(data, columns=list(cm.item_names))
        result = est.estimate(frame, spec, start_from=cm)

        reproduced = float(np.max(np.abs(result.sample_cov - out.sigma_0)))
        assert reproduced < 1e-10, (name, reproduced)
        assert result.converged, (name, result.message)
        assert result.df == out.df, (name, result.df, out.df)

        got = np.asarray([result.parameters[label] for label in pmap.labels],
                         dtype=float)
        deviation = float(np.max(np.abs(got - pmap.theta0)))
        f_gap = abs(float(result.fmin) - out.achieved_fmin)
        _say(f"{name:24s} refit: max |theta_hat - theta| = {deviation:.3e}, "
             f"|F_hat - F_0| = {f_gap:.3e} "
             f"(F_0 = {out.achieved_fmin:.12f}, F_hat = {result.fmin:.12f})")
        assert deviation < 1e-4, (name, deviation)
        assert f_gap < 1e-6, (name, f_gap)


def test_gauss_newton_refit_from_a_perturbed_start_returns_to_theta() -> None:
    """An independent optimiser, started away from ``theta``, must come back."""
    for name, _ in MODELS:
        cm = _compiled(name)
        pmap = cb.free_parameter_map(cm)
        out = cb.inject_model_error(cm, 0.045, rng=np.random.default_rng(31))
        rng = np.random.default_rng(99)
        start = pmap.theta0 + 0.02 * rng.standard_normal(pmap.theta0.size)
        recovered, attained = _gauss_newton_refit(pmap, out.sigma_0, start)
        deviation = float(np.max(np.abs(recovered - pmap.theta0)))
        f_gap = abs(attained - out.achieved_fmin)
        _say(f"{name:24s} Gauss-Newton from a 0.02-sd perturbation: "
             f"max |theta_hat - theta| = {deviation:.3e}, "
             f"|F_hat - F_0| = {f_gap:.3e}")
        assert deviation < 1e-4, (name, deviation)
        assert f_gap < 1e-6, (name, f_gap)


def test_residual_method_hits_the_rmsea_but_moves_the_minimiser() -> None:
    """Why the projection matters: without it ``theta`` is no longer optimal."""
    cm = _compiled("minimal_cfa")
    sigma = implied_covariance(cm)
    pmap = cb.free_parameter_map(cm)
    projected = cb.inject_model_error(
        cm, 0.07, rng=np.random.default_rng(11), method="cudeck_browne")
    plain = cb.inject_model_error(
        cm, 0.07, rng=np.random.default_rng(11), method="residual")

    # Both land on the requested population RMSEA...
    for out in (projected, plain):
        assert abs(_rmsea_by_hand(out.sigma_0, sigma, out.df) - 0.07) < 1e-6
    assert plain.method == "residual"
    assert plain.parameterisation == ""
    assert plain.warnings and "UNPROJECTED_ERROR" in plain.warnings[0]

    g_proj = float(np.max(np.abs(_analytic_gradient(pmap, projected.sigma_0))))
    g_plain = float(np.max(np.abs(_analytic_gradient(pmap, plain.sigma_0))))
    _say(f"max |dF/dtheta| at theta: cudeck_browne = {g_proj:.3e}, "
         f"residual = {g_plain:.3e} (ratio {g_plain / max(g_proj, 1e-300):.3g})")
    assert g_proj < 1e-10
    assert g_plain > 1e-4, g_plain

    # ...and refitting the unprojected population really does move theta.
    moved, _ = _gauss_newton_refit(pmap, plain.sigma_0, pmap.theta0)
    shift = float(np.max(np.abs(moved - pmap.theta0)))
    _say(f"refitting the 'residual' population shifts theta by {shift:.3e}")
    assert shift > 1e-4, shift


# --------------------------------------------------------------------------- #
# 4. Degenerate requests
# --------------------------------------------------------------------------- #


def test_zero_or_negative_target_returns_sigma_bit_identical() -> None:
    for name, _ in MODELS:
        cm = _compiled(name)
        sigma = implied_covariance(cm)
        for target in (0.0, -0.01, -1.0):
            out = cb.inject_model_error(
                cm, target, rng=np.random.default_rng(4))
            assert np.array_equal(out.sigma_0, sigma), (name, target)
            assert out.achieved_rmsea == 0.0
            assert out.achieved_fmin == 0.0
            assert out.scale == 0.0
            assert out.converged
            assert out.direction is None
            assert not out.perturbed
    _say("target_rmsea <= 0 returns Sigma(theta) bit-identically "
         "(np.array_equal) on all three models")


def test_method_none_returns_sigma_bit_identical() -> None:
    cm = _compiled("minimal_cfa")
    sigma = implied_covariance(cm)
    for spelling in ("none", "None", "", "off"):
        out = cb.inject_model_error(cm, 0.08, method=spelling,
                                    rng=np.random.default_rng(4))
        assert np.array_equal(out.sigma_0, sigma), spelling
        assert out.achieved_rmsea == 0.0 and out.scale == 0.0
    _say("method='none' (and its spellings) returns Sigma(theta) untouched")


def test_non_positive_df_returns_sigma_bit_identical() -> None:
    cm = _compiled("minimal_cfa")
    sigma = implied_covariance(cm)
    for df in (0, -3):
        out = cb.inject_model_error(cm, 0.08, df=df,
                                    rng=np.random.default_rng(4))
        assert np.array_equal(out.sigma_0, sigma), df
        assert out.achieved_rmsea == 0.0
        assert out.df == df
    _say("df <= 0 returns Sigma(theta) untouched with achieved_rmsea = 0")


def test_explicit_df_is_honoured() -> None:
    cm = _compiled("minimal_cfa")
    sigma = implied_covariance(cm)
    out = cb.inject_model_error(cm, 0.05, df=20,
                                rng=np.random.default_rng(6))
    assert out.df == 20
    assert abs(out.achieved_fmin - 20 * 0.05 ** 2) < 1e-10
    assert abs(_rmsea_by_hand(out.sigma_0, sigma, 20) - 0.05) < 1e-6
    # An explicit df is the caller's decision and is taken as given; only the
    # *derived* default is cross-checked against ModelSpec.degrees_of_freedom.
    default = cb.inject_model_error(cm, 0.05, rng=np.random.default_rng(6))
    assert default.df == 8
    _say(f"df=20 override: F_0 = {out.achieved_fmin:.12f} "
         f"(= 20 * 0.05^2 = {20 * 0.05 ** 2:.12f}); "
         f"derived default df = {default.df}")


def test_unknown_method_raises_a_bilingual_specification_error() -> None:
    cm = _compiled("minimal_cfa")
    try:
        cb.inject_model_error(cm, 0.05, method="magic")
    except SpecificationError as exc:
        assert "magic" in exc.message_en
        assert _is_arabic(exc.message_ar), exc.message_ar
        _say("unknown method -> SpecificationError with an Arabic message")
    else:
        raise AssertionError("an unknown method must raise")


def test_non_finite_target_raises() -> None:
    cm = _compiled("minimal_cfa")
    for bad in (float("nan"), float("inf")):
        try:
            cb.inject_model_error(cm, bad)
        except SpecificationError as exc:
            assert _is_arabic(exc.message_ar)
        else:
            raise AssertionError(f"target {bad} must raise")
    _say("NaN / inf targets raise SpecificationError")


# --------------------------------------------------------------------------- #
# 5. Reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_twice_gives_the_same_matrix() -> None:
    cm = _compiled("simple_mediation")
    a = cb.inject_model_error(cm, 0.06, rng=np.random.default_rng(2024))
    b = cb.inject_model_error(cm, 0.06, rng=np.random.default_rng(2024))
    assert np.array_equal(a.sigma_0, b.sigma_0)
    assert a.scale == b.scale
    assert a.achieved_fmin == b.achieved_fmin
    _say("same rng seed -> bit-identical Sigma_0 (np.array_equal)")


def test_different_seeds_give_different_matrices_with_the_same_rmsea() -> None:
    cm = _compiled("simple_mediation")
    sigma = implied_covariance(cm)
    out_a = cb.inject_model_error(cm, 0.06, rng=np.random.default_rng(1))
    out_b = cb.inject_model_error(cm, 0.06, rng=np.random.default_rng(2))
    gap = float(np.max(np.abs(out_a.sigma_0 - out_b.sigma_0)))
    assert gap > 1e-3, gap
    r_a = _rmsea_by_hand(out_a.sigma_0, sigma, out_a.df)
    r_b = _rmsea_by_hand(out_b.sigma_0, sigma, out_b.df)
    assert abs(r_a - 0.06) < 1e-6 and abs(r_b - 0.06) < 1e-6
    _say(f"seeds 1 and 2: max |Sigma_0 difference| = {gap:.4f}, "
         f"RMSEA {r_a:.12f} and {r_b:.12f}")


def test_rng_defaults_to_the_spec_seed_and_is_reproducible() -> None:
    spec = examples.minimal_cfa()
    assert spec.seed is not None
    cm_a = compile_model(spec)
    cm_b = compile_model(examples.minimal_cfa())
    out_a = cb.inject_model_error(cm_a, 0.05)
    out_b = cb.inject_model_error(cm_b, 0.05)
    assert np.allclose(out_a.sigma_0, out_b.sigma_0, atol=0.0, rtol=0.0)
    expected = cb.inject_model_error(
        cm_a, 0.05, rng=np.random.default_rng(spec.seed))
    assert np.array_equal(out_a.sigma_0, expected.sigma_0)
    _say(f"rng=None reuses ModelSpec.seed ({spec.seed}) and is reproducible")


# --------------------------------------------------------------------------- #
# 6. Infeasible targets
# --------------------------------------------------------------------------- #


def _is_arabic(text: str) -> bool:
    """At least a dozen Arabic letters, i.e. a real sentence not a token."""
    letters = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return letters >= 12


def test_impossible_target_raises_infeasible_with_an_arabic_message() -> None:
    cm = _compiled("minimal_cfa")      # df = 8, p = 6: very little room
    for target in (3.0, 5.0, 25.0):
        try:
            out = cb.inject_model_error(
                cm, target, rng=np.random.default_rng(3))
        except InfeasibleTargetError as exc:
            assert _is_arabic(exc.message_ar), exc.message_ar
            assert "max_achievable_rmsea" in exc.context
            _say(f"target={target}: InfeasibleTargetError, "
                 f"max achievable RMSEA = "
                 f"{exc.context['max_achievable_rmsea']:.6f}; "
                 f"AR: {exc.message_ar[:70]}...")
        else:
            raise AssertionError(
                f"RMSEA {target} on df=8 must be infeasible, got "
                f"min eig = {out.min_eigenvalue}")


def test_an_infeasible_target_never_returns_a_non_pd_matrix() -> None:
    """Across many seeds: either a usable matrix, or a typed error -- never a
    matrix that is not positive definite."""
    cm = _compiled("minimal_cfa")
    raised = 0
    returned = 0
    worst_eig = float("inf")
    for seed in range(25):
        try:
            out = cb.inject_model_error(cm, 1.2,
                                        rng=np.random.default_rng(seed))
        except InfeasibleTargetError:
            raised += 1
        else:
            returned += 1
            assert is_positive_definite(out.sigma_0), seed
            assert out.min_eigenvalue > 0.0, seed
            assert abs(out.achieved_rmsea - 1.2) < 1e-6, seed
            worst_eig = min(worst_eig, out.min_eigenvalue)
    assert raised + returned == 25
    _say(f"RMSEA 1.2 on df=8 over 25 seeds: {returned} usable matrices "
         f"(smallest eigenvalue seen {worst_eig:.3e}), "
         f"{raised} InfeasibleTargetError; no non-PD matrix ever returned")

    # Far beyond what df = 8 can carry: every seed must refuse, with a typed
    # error, rather than hand back something unusable.
    for seed in range(25):
        try:
            cb.inject_model_error(cm, 3.0, rng=np.random.default_rng(seed))
        except InfeasibleTargetError as exc:
            assert exc.context["max_achievable_rmsea"] < 3.0
        else:
            raise AssertionError(f"RMSEA 3.0 on df=8 must fail (seed {seed})")
    _say("RMSEA 3.0 on df=8: InfeasibleTargetError for all 25 seeds")


def test_saturated_model_is_reported_as_infeasible() -> None:
    """A direction orthogonal to a space-filling tangent space cannot exist."""
    from semsim.core.spec import LatentCovariance, ModelSpec, ScaleKind

    spec = ModelSpec(seed=7)
    a = spec.add_construct("A", n_items=2, loading=0.7)
    b = spec.add_construct("B", n_items=1, loading=0.8)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=a.id, construct_b=b.id, correlation=0.3))
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS
    spec.ensure_unique_item_names()
    cm = compile_model(spec)
    p = cm.n_items
    pmap = cb.free_parameter_map(cm)
    df = p * (p + 1) // 2 - pmap.q
    _say(f"saturated fixture: p = {p}, p* = {p * (p + 1) // 2}, "
         f"q = {pmap.q}, df = {df}")
    out = cb.inject_model_error(cm, 0.05, rng=np.random.default_rng(1))
    if df <= 0:
        assert out.scale == 0.0
        assert np.array_equal(out.sigma_0, implied_covariance(cm))
        assert any("ZERO_DF" in w for w in out.warnings), out.warnings
        _say("df <= 0 fixture: injection skipped and ZERO_DF warned")
    else:  # pragma: no cover - depends on the identification bookkeeping
        assert abs(out.achieved_rmsea - 0.05) < 1e-6


def test_errors_are_semsim_errors_not_raw_tracebacks() -> None:
    """Nothing user-triggerable may escape as a bare numpy / scipy exception."""
    cm = _compiled("minimal_cfa")
    bad_inputs = [
        dict(target_rmsea=50.0),
        dict(target_rmsea=0.05, method="nope"),
        dict(target_rmsea=float("nan")),
    ]
    for kwargs in bad_inputs:
        target = kwargs.pop("target_rmsea")
        try:
            cb.inject_model_error(cm, target,
                                  rng=np.random.default_rng(0), **kwargs)
        except SemSimError as exc:
            assert _is_arabic(exc.message_ar), (kwargs, exc.message_ar)
        except Exception as exc:                                # noqa: BLE001
            raise AssertionError(
                f"{kwargs} leaked a {type(exc).__name__}: {exc}") from exc
    _say("every bad input surfaces as a SemSimError with an Arabic message")


# --------------------------------------------------------------------------- #
# 7. Reporting surface
# --------------------------------------------------------------------------- #


def test_result_fields_are_consistent() -> None:
    cm = _compiled("second_order_hierarchy")
    out = cb.inject_model_error(cm, 0.045, rng=np.random.default_rng(8))
    assert out.direction is not None
    assert abs(float(np.linalg.norm(out.direction, ord="fro")) - 1.0) < 1e-12
    assert np.allclose(out.direction, out.direction.T, atol=0.0, rtol=0.0)
    rebuilt = implied_covariance(cm) + out.scale * out.direction
    assert np.allclose(rebuilt, out.sigma_0, atol=1e-14)
    assert abs(out.rmsea_error) < 1e-9
    assert out.perturbed
    assert out.n_free == cb.free_parameter_map(cm).q
    assert out.parameterisation in ("estimator", "builtin")
    assert out.iterations > 0
    _say(f"second_order_hierarchy: ||E||_F = "
         f"{float(np.linalg.norm(out.direction, ord='fro')):.15f}, "
         f"Sigma(theta) + k E reproduces sigma_0 to 1e-14, "
         f"n_free = {out.n_free}, iterations = {out.iterations}")


# --------------------------------------------------------------------------- #
# Script mode (pytest may be absent)
# --------------------------------------------------------------------------- #


def _main() -> int:
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

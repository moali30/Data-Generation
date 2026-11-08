"""
semsim.targeting.calibration
============================
The closed loop that makes the realised output match what was asked for.

Injecting a population discrepancy of ``F_0 = df * RMSEA**2`` is exact at the
population level -- and still lands short once a finite sample is fitted.
The reason is the estimator: maximum likelihood finds the parameter vector
that minimises the discrepancy *for that sample*, which is never worse and
usually better than the value at the population parameters.  So the realised
RMSEA sits systematically below the injected one, by a margin that depends on
the model's size and the sample's.

Rather than derive a correction, the loop measures it.  It generates, fits,
compares, scales the injected discrepancy by the ratio it observed, and goes
round again.  Two or three passes are normally enough, and the history is
returned so the researcher can see the adjustment rather than take it on
trust.

Path significance is handled in the same spirit: a draw whose p-values
contradict the requested pattern is rejected and re-drawn with a fresh seed,
up to ``significance_target.max_resamples``.  Rejection sampling is honest
about what it is -- the returned result says how many draws it took.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from semsim.core.compile import CompiledModel, compile_model
from semsim.core.errors import SemSimError
from semsim.core import implied as implied_module
from semsim.core.spec import FitLevel, ModelSpec
from semsim.generation.engine import GeneratedDataset, generate_dataset
from semsim.targeting import significance as significance_module

ProgressFn = Optional[Callable[[str, float], None]]

#: Never let one round move the injected discrepancy by more than this factor;
#: an unlucky draw should nudge the target, not throw it across the room.
MAX_STEP = 3.0
#: Damping on the multiplicative update, so the loop converges instead of
#: oscillating around the target.
DAMPING = 0.75


@dataclass
class CalibrationResult:
    """The accepted draw, plus the record of how it was reached."""

    dataset: GeneratedDataset
    estimation: Any = None
    fit: Any = None
    rounds: int = 0
    draws: int = 1
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    significance_ok: bool = True
    significance_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    plan: Optional[significance_module.SignificancePlan] = None
    converged: bool = False
    target_rmsea: float = 0.0
    realised_rmsea: float = float("nan")
    warnings: List[str] = field(default_factory=list)

    @property
    def within_tolerance(self) -> bool:
        if not math.isfinite(self.realised_rmsea):
            return False
        return abs(self.realised_rmsea - self.target_rmsea) <= self._tolerance

    _tolerance: float = 0.006

    def summary(self) -> Dict[str, Any]:
        return {
            "rounds": self.rounds,
            "draws": self.draws,
            "target_rmsea": self.target_rmsea,
            "realised_rmsea": self.realised_rmsea,
            "within_tolerance": self.within_tolerance,
            "significance_ok": self.significance_ok,
            "converged": self.converged,
        }


def calibrate(spec: ModelSpec, *, n: Optional[int] = None,
              seed: Optional[int] = None,
              progress: ProgressFn = None,
              estimate_fn: Optional[Callable[..., Any]] = None
              ) -> CalibrationResult:
    """Generate, fit, compare and adjust until the targets are met.

    Parameters
    ----------
    spec
        The model to simulate.
    n
        Sample size; defaults to ``spec.power.n``.
    seed
        Overrides ``spec.seed``.  Each rejected draw advances the seed, so the
        whole run stays reproducible from this one number.
    progress
        ``progress(message_key, fraction)`` for a GUI.
    estimate_fn
        Injection point for tests; defaults to
        :func:`semsim.analysis.estimator.estimate`.
    """
    def report(key: str, fraction: float) -> None:
        if progress is not None:
            progress(key, fraction)

    warnings: List[str] = []
    n_cases = int(n if n is not None else spec.power.n)
    base_seed = int(seed if seed is not None
                    else (spec.seed if spec.seed is not None else 0))

    report("status.compiling", 0.03)
    compiled = compile_model(spec)
    df = spec.degrees_of_freedom(compiled)

    plan = significance_module.build_plan(spec, compiled, n_cases,
                                          compute_power=False)
    working = significance_module.apply_plan(spec, plan)
    working.power.n = n_cases
    if plan.coefficients:
        compiled = compile_model(working)

    target_rmsea = _target_rmsea(spec)
    tolerance = float(spec.fit_target.tolerance)
    max_rounds = max(int(spec.fit_target.max_calibration_rounds), 1)
    targeting_fit = bool(spec.fit_target.enabled
                         and spec.fit_target.method not in ("none", "")
                         and target_rmsea > 0.0 and df > 0)
    if spec.fit_target.enabled and df <= 0:
        warnings.append(
            "FIT_TARGET_SKIPPED: the model is saturated (df = 0), so there is "
            "no lack of fit to target.")

    estimate = estimate_fn or _default_estimator()
    max_resamples = (max(int(spec.significance_target.max_resamples), 0)
                     if spec.significance_target.enabled else 0)

    history: List[Dict[str, Any]] = []
    injected = target_rmsea
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    draws = 0
    converged = False

    for round_index in range(1, max_rounds + 1):
        report("status.calibrating", 0.1 + 0.8 * (round_index - 1) / max_rounds)
        attempt, draws_used = _draw_until_significant(
            working, compiled, n_cases, base_seed + 1000 * round_index,
            injected if targeting_fit else 0.0, df, estimate,
            max_resamples, warnings)
        draws += draws_used
        realised = attempt["rmsea"]
        history.append({
            "round": round_index,
            "injected_rmsea": injected,
            "realised_rmsea": realised,
            "target_rmsea": target_rmsea,
            "deviation": realised - target_rmsea,
            "draws": draws_used,
            "significance_ok": attempt["significance_ok"],
        })

        score = _score(attempt, target_rmsea, targeting_fit)
        if best is None or score < best[0]:
            best = (score, attempt)

        if not targeting_fit:
            converged = attempt["significance_ok"] or max_resamples == 0
            break
        if abs(realised - target_rmsea) <= tolerance and attempt["significance_ok"]:
            converged = True
            break
        injected = _next_injection(injected, realised, target_rmsea, warnings)

    assert best is not None
    attempt = best[1]
    significance_ok, table = significance_module.check_realised(
        attempt["estimation"], spec, spec.significance_target.alpha)

    result = CalibrationResult(
        dataset=attempt["dataset"],
        estimation=attempt["estimation"],
        fit=attempt["fit"],
        rounds=len(history),
        draws=draws,
        history=pd.DataFrame(history),
        significance_ok=significance_ok,
        significance_table=table,
        plan=plan,
        converged=converged,
        target_rmsea=target_rmsea,
        realised_rmsea=attempt["rmsea"],
        warnings=warnings + list(attempt["dataset"].warnings),
    )
    result._tolerance = tolerance
    if targeting_fit and not result.within_tolerance:
        warnings.append(
            f"FIT_TARGET_MISSED: after {len(history)} round(s) the realised "
            f"RMSEA was {attempt['rmsea']:.4f} against a target of "
            f"{target_rmsea:.4f}. Raise the round limit or widen the "
            "tolerance.")
        result.warnings = warnings + list(attempt["dataset"].warnings)
    return result


# --------------------------------------------------------------------------- #
# One round
# --------------------------------------------------------------------------- #


def _draw_until_significant(working: ModelSpec, compiled: CompiledModel,
                            n: int, seed: int, injected_rmsea: float,
                            df: int, estimate: Callable[..., Any],
                            max_resamples: int,
                            warnings: List[str]) -> Tuple[Dict[str, Any], int]:
    """Draw, fit, and re-draw while the p-value pattern is wrong."""
    sigma = _generating_matrix(compiled, injected_rmsea, df, seed, warnings)
    fallback: Optional[Dict[str, Any]] = None
    attempts = max_resamples + 1

    for attempt_index in range(attempts):
        dataset = generate_dataset(working, n=n, seed=seed + attempt_index,
                                   sigma_override=sigma, compiled=compiled)
        estimation, fit = _fit(dataset, working, estimate, warnings)
        ok = True
        if estimation is not None and max_resamples:
            ok, _table = significance_module.check_realised(
                estimation, working, working.significance_target.alpha)
        record = {
            "dataset": dataset,
            "estimation": estimation,
            "fit": fit,
            "rmsea": float(getattr(fit, "rmsea", float("nan"))
                           if fit is not None else float("nan")),
            "significance_ok": ok,
        }
        if ok:
            return record, attempt_index + 1
        if fallback is None:
            fallback = record
    if max_resamples:
        warnings.append(
            f"SIGNIFICANCE_NOT_REACHED: {attempts} draws did not reproduce the "
            "requested pattern of significant and non-significant paths. The "
            "closest draw was kept. Raising the sample size is usually the "
            "fix; rejection sampling cannot create power that is not there.")
    return (fallback if fallback is not None else record), attempts


def _generating_matrix(compiled: CompiledModel, injected_rmsea: float,
                       df: int, seed: int,
                       warnings: List[str]) -> Optional[np.ndarray]:
    if injected_rmsea <= 0.0 or df <= 0:
        return None
    try:
        from semsim.targeting.cudeck_browne import inject_model_error
    except ImportError:
        warnings.append("MODEL_ERROR_UNAVAILABLE: the model-error module is "
                        "not installed; the data will fit perfectly.")
        return None
    try:
        result = inject_model_error(compiled, injected_rmsea, df=df,
                                    rng=np.random.default_rng(seed))
    except SemSimError as exc:
        warnings.append(f"MODEL_ERROR_SKIPPED: {exc.message_en}")
        return None
    return result.sigma_0


def _fit(dataset: GeneratedDataset, spec: ModelSpec,
         estimate: Callable[..., Any],
         warnings: List[str]) -> Tuple[Any, Any]:
    if estimate is None:
        return None, None
    try:
        estimation = estimate(dataset.data, spec,
                              estimator=spec.analysis.estimator)
    except Exception as exc:
        warnings.append(f"ESTIMATION_FAILED: {type(exc).__name__}: {exc}")
        return None, None
    try:
        from semsim.analysis.fit_indices import compute as compute_fit
        fit = compute_fit(estimation.sample_cov, estimation.sigma_hat,
                          estimation.n, estimation.n_free)
    except Exception as exc:
        warnings.append(f"FIT_INDICES_FAILED: {type(exc).__name__}: {exc}")
        fit = None
    return estimation, fit


# --------------------------------------------------------------------------- #
# The update rule
# --------------------------------------------------------------------------- #


def _next_injection(injected: float, realised: float, target: float,
                    warnings: List[str]) -> float:
    """Scale the injected discrepancy by what the last round actually gave.

    RMSEA grows as the square root of the discrepancy, so a multiplicative
    update on RMSEA is the natural one.  It is damped and bounded: a single
    unlucky draw should move the target, not overturn it.
    """
    if not math.isfinite(realised) or realised <= 1e-9:
        # The draw fit perfectly; push harder rather than divide by zero.
        return min(injected * 1.8 + 0.01, 0.30)
    ratio = target / realised
    ratio = max(1.0 / MAX_STEP, min(MAX_STEP, ratio))
    damped = ratio ** DAMPING
    return float(np.clip(injected * damped, 1e-4, 0.30))


def _score(attempt: Dict[str, Any], target: float, targeting_fit: bool) -> float:
    """Lower is better; used to keep the best round when none converges."""
    penalty = 0.0 if attempt["significance_ok"] else 1.0
    if not targeting_fit:
        return penalty
    realised = attempt["rmsea"]
    if not math.isfinite(realised):
        return penalty + 10.0
    return penalty + abs(realised - target)


def _target_rmsea(spec: ModelSpec) -> float:
    target = spec.fit_target
    if target.level is FitLevel.CUSTOM:
        return float(target.rmsea)
    return float(target.level.target_rmsea)


def _default_estimator() -> Optional[Callable[..., Any]]:
    try:
        from semsim.analysis.estimator import estimate
    except ImportError:
        return None
    return estimate


__all__ = ["CalibrationResult", "calibrate"]

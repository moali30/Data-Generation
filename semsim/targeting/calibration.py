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

Every draw comes from one simulated population, which varies a little
around the plan (:mod:`semsim.generation.heterogeneity`).  That variation can
leave a weak path the researcher wants significant smaller than planned, too
small for the sample to show it, and then no number of further samples from
that population helps.  So when a population misses a requested p-value, up
to ``EXTRA_POPULATIONS`` other populations around the same plan are tried,
each as plausible as the first.  When none reaches the pattern, the draw
closest to it is kept: the fewest misses, then the p-values nearest alpha.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from semsim.core import cancel as _cancel
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
#: How many further populations are tried when the first one leaves a path
#: the researcher wants significant out of the sample's reach.
EXTRA_POPULATIONS = 2
#: Seed offsets for the further populations and for their samples, so the
#: first population and its draws are exactly those of a single-population run.
POPULATION_SEED_STRIDE = 7_368_787
SAMPLE_SEED_STRIDE = 500_000


@dataclass
class CalibrationResult:
    """The accepted draw, plus the record of how it was reached."""

    dataset: GeneratedDataset
    estimation: Any = None
    fit: Any = None
    rounds: int = 0
    draws: int = 1
    populations: int = 1
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
            "populations": self.populations,
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
    # Control paths are structural paths: targeted and checked like the rest.
    spec = spec.expanded()
    compiled = compile_model(spec)
    df = spec.degrees_of_freedom(compiled)

    plan = significance_module.build_plan(spec, compiled, n_cases,
                                          compute_power=False)
    planned = significance_module.apply_plan(spec, plan)

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
    checking = bool(spec.significance_target.enabled)
    max_resamples = (max(int(spec.significance_target.max_resamples), 0)
                     if checking else 0)
    # Another population only differs from the first when the population
    # varies around the plan at all.
    from semsim.generation.heterogeneity import realise_population
    varies = bool(getattr(planned.heterogeneity, "enabled", False))
    populations = 1 + (EXTRA_POPULATIONS if varies and max_resamples else 0)

    history: List[Dict[str, Any]] = []
    injected = target_rmsea
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    draws = 0
    converged = False
    populations_used = 0

    for population_index in range(populations):
        populations_used = population_index + 1
        last_population = population_index == populations - 1
        # Every round and every re-draw of one pass samples from the same
        # population, so rejection sampling selects among samples of it.
        working = realise_population(
            planned, base_seed + POPULATION_SEED_STRIDE * population_index)
        working.power.n = n_cases
        compiled = compile_model(working)
        hopeless: Dict[str, int] = {}
        sample_seed = base_seed + SAMPLE_SEED_STRIDE * population_index
        span = (0.1, 0.9) if population_index == 0 else (0.9, 0.97)

        def fraction(done: float, _span: Tuple[float, float] = span) -> float:
            return _span[0] + (_span[1] - _span[0]) * done / max_rounds

        for round_index in range(1, max_rounds + 1):
            report("status.calibrating", fraction(round_index - 1))
            # Once a path is known to be out of reach, rejection sampling has
            # nothing left to try, so later rounds draw once and move on.
            stuck = any(c >= HOPELESS_AFTER for c in hopeless.values())
            resamples = 0 if stuck else max_resamples

            def on_draw(done: int, total: int, _round: int = round_index,
                        _fraction: Callable[[float], float] = fraction) -> None:
                # Each fit is the slow part, so the bar moves once per draw
                # rather than once per round -- and Stop is honoured between
                # them.
                report("status.calibrating",
                       _fraction((_round - 1) + done / max(total, 1)))

            attempt, draws_used = _draw_until_significant(
                working, compiled, n_cases, sample_seed + 1000 * round_index,
                injected if targeting_fit else 0.0, df, estimate,
                resamples, warnings, hopeless, on_draw=on_draw,
                check=checking)
            draws += draws_used
            realised = attempt["rmsea"]
            history.append({
                "round": len(history) + 1,
                "population": population_index + 1,
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

            # Significance is settled when it was met, when nothing was asked
            # of it, or when this population has shown it cannot meet it.
            settled = (attempt["significance_ok"] or max_resamples == 0
                       or any(c >= HOPELESS_AFTER for c in hopeless.values()))
            if not targeting_fit:
                converged = attempt["significance_ok"] or max_resamples == 0
                break
            if abs(realised - target_rmsea) <= tolerance and settled:
                converged = attempt["significance_ok"] or max_resamples == 0
                break
            injected = _next_injection(injected, realised, target_rmsea, warnings)
            if not attempt["significance_ok"] and not last_population:
                # A fresh population is a better bet than more samples from
                # one that keeps missing.
                break

        if best is not None and best[1]["significance_ok"]:
            break

    assert best is not None
    attempt = best[1]
    significance_ok, table = significance_module.check_realised(
        attempt["estimation"], spec, spec.significance_target.alpha)

    if significance_ok:
        # A population that missed and was replaced leaves nothing to report.
        warnings = [w for w in warnings if not w.startswith(_MISS_CODES)]
    elif populations_used > 1:
        warnings = [w for w in warnings
                    if not w.startswith("SIGNIFICANCE_NOT_REACHED")]
        warnings.append(
            f"SIGNIFICANCE_NOT_REACHED: {draws} draw(s) from "
            f"{populations_used} simulated populations did not reproduce the "
            "requested pattern of significant and non-significant paths. The "
            "closest draw was kept. Rejection sampling cannot create power "
            "that is not there.")

    result = CalibrationResult(
        dataset=attempt["dataset"],
        estimation=attempt["estimation"],
        fit=attempt["fit"],
        rounds=len(history),
        draws=draws,
        populations=populations_used,
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


#: After this many consecutive draws in which a path misses alpha by a wide
#: margin, stop re-drawing: the sample does not have the power, and no number
#: of further draws will conjure it.
HOPELESS_AFTER = 3
#: "A wide margin" means a p-value this many times alpha.
HOPELESS_RATIO = 5.0
#: Warnings about a missed significance pattern.
_MISS_CODES = ("UNDERPOWERED_PATH", "SIGNIFICANCE_NOT_REACHED")


def _draw_until_significant(working: ModelSpec, compiled: CompiledModel,
                            n: int, seed: int, injected_rmsea: float,
                            df: int, estimate: Callable[..., Any],
                            max_resamples: int,
                            warnings: List[str],
                            hopeless: Optional[Dict[str, int]] = None,
                            on_draw: Optional[Callable[[int, int], None]] = None,
                            check: Optional[bool] = None
                            ) -> Tuple[Dict[str, Any], int]:
    """Draw, fit, and re-draw while the p-value pattern is wrong.

    Rejection sampling only helps when the requested pattern is reachable.
    When a path flagged significant keeps landing far above alpha, the sample
    is underpowered for it and further draws only burn time -- a complex model
    costs tens of seconds per fit, so twenty-five futile draws is a quarter of
    an hour of a frozen progress bar. The loop watches for that and stops,
    saying what the actual fix is.

    ``check`` says whether to test the p-values at all (by default, whenever
    re-drawing is allowed); a draw is then scored by how far it is from the
    requested pattern, and when none matches, the closest one is returned.
    """
    sigma = _generating_matrix(compiled, injected_rmsea, df, seed, warnings)
    if check is None:
        check = bool(max_resamples)
    fallback: Optional[Dict[str, Any]] = None
    attempts = max_resamples + 1
    alpha = working.significance_target.alpha
    # Shared across calibration rounds: a path the sample cannot detect in
    # round one cannot detect it in round four either, and re-learning that
    # every round costs three full re-fits each time.
    if hopeless is None:
        hopeless = {}
    used = 0

    for attempt_index in range(attempts):
        _cancel.check()
        if on_draw is not None and attempt_index:
            on_draw(attempt_index, attempts)
        used = attempt_index + 1
        dataset = generate_dataset(working, n=n, seed=seed + attempt_index,
                                   sigma_override=sigma, compiled=compiled)
        estimation, fit = _fit(dataset, working, estimate, warnings)
        ok = True
        table = pd.DataFrame()
        if estimation is not None and check:
            ok, table = significance_module.check_realised(
                estimation, working, alpha)
        record = {
            "dataset": dataset,
            "estimation": estimation,
            "fit": fit,
            "rmsea": float(getattr(fit, "rmsea", float("nan"))
                           if fit is not None else float("nan")),
            "significance_ok": ok,
            "miss": miss_distance(table, alpha) if not ok else (0, 0.0),
        }
        if ok:
            return record, used
        if fallback is None or record["miss"] < fallback["miss"]:
            fallback = record
        if not max_resamples:
            break
        for label in _underpowered(table, alpha):
            hopeless[label] = hopeless.get(label, 0) + 1
        stuck = [label for label, count in hopeless.items()
                 if count >= HOPELESS_AFTER]
        if stuck:
            _warn_once(warnings,
                       f"UNDERPOWERED_PATH: {', '.join(sorted(stuck))} missed "
                       f"alpha by a wide margin in {HOPELESS_AFTER} "
                       "consecutive draws, so re-drawing was stopped. At this "
                       "sample size the effect is not detectable; raise N (the "
                       "power page will say by how much) or raise the "
                       "coefficient.")
            break
    if max_resamples:
        _warn_once(warnings,
                   f"SIGNIFICANCE_NOT_REACHED: {used} draw(s) did not reproduce "
                   "the requested pattern of significant and non-significant "
                   "paths. The closest draw was kept. Rejection sampling "
                   "cannot create power that is not there.")
    return (fallback if fallback is not None else record), used


def miss_distance(table: pd.DataFrame, alpha: float) -> Tuple[int, float]:
    """How far a draw is from the requested pattern of p-values.

    Returns ``(misses, distance)``: the number of paths on the wrong side of
    alpha, then the sum over those paths of how many orders of magnitude
    their p-value is from alpha.  Tuples compare in that order, so the draw
    with fewer misses wins and, among equals, the one whose misses are
    nearest to alpha.
    """
    if table is None or table.empty or "p" not in table.columns:
        return (0, 0.0)
    misses, distance = 0, 0.0
    for _index, row in table.iterrows():
        if row.get("requested") == row.get("realised"):
            continue
        misses += 1
        try:
            p_value = float(row.get("p"))
        except (TypeError, ValueError):
            p_value = float("nan")
        if not math.isfinite(p_value):
            distance += _MAX_MISS
            continue
        p_value = min(max(p_value, 1e-300), 1.0)
        ratio = (p_value / alpha if row.get("requested") == "significant"
                 else alpha / p_value)
        distance += min(abs(math.log10(ratio)), _MAX_MISS)
    return (misses, float(distance))


#: Cap on one path's contribution to :func:`miss_distance`.
_MAX_MISS = 5.0


def _underpowered(table: pd.DataFrame, alpha: float) -> List[str]:
    """Paths asked to be significant whose p-value is nowhere near alpha."""
    if table is None or table.empty or "p" not in table.columns:
        return []
    out: List[str] = []
    for _index, row in table.iterrows():
        if row.get("requested") != "significant":
            continue
        try:
            p_value = float(row.get("p"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(p_value) and p_value > alpha * HOPELESS_RATIO:
            out.append(f"{row.get('source')} -> {row.get('target')}")
    return out


def _warn_once(warnings: List[str], message: str) -> None:
    """Each round would otherwise repeat the same sentence."""
    code = message.split(":", 1)[0]
    if any(w.startswith(code) for w in warnings):
        return
    warnings.append(message)


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
        estimation = estimate(dataset.model_frame, spec,
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
    """Lower is better; used to keep the best round when none converges.

    Significance comes first -- fewer missed paths, then p-values nearer
    alpha -- and the RMSEA only decides between draws that miss alike.
    """
    misses, distance = attempt.get("miss", (0, 0.0))
    penalty = 0.0 if attempt["significance_ok"] else (
        10.0 * max(misses, 1) + min(distance, 9.0))
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

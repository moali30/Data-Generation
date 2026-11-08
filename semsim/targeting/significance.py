"""
semsim.targeting.significance
=============================
Decide what each path's generating coefficient should be, and check afterwards
whether the sample actually delivered the outcome the researcher asked for.

The brief the engine implements is "make this path significant with a medium
effect, and that one non-significant".  Those are two different requests:

*   **Significant.**  The coefficient the researcher chose is the population
    value; what has to be arranged is a sample large enough that it clears
    alpha with the requested probability.  That is a power question, answered
    by :mod:`semsim.power.analysis`.
*   **Non-significant.**  Setting the coefficient to exactly zero would be a
    lie of a different kind: real null effects are small, not absent, and a
    column of exact zeros is visible in any residual plot.  The path gets
    ``significance_target.null_coefficient`` -- small enough to miss alpha at
    a realistic N, large enough to look like data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from semsim.core.spec import EffectSize, Interaction, ModelSpec, Path


@dataclass
class SignificancePlan:
    """What to generate with, and how big a sample it needs."""

    #: ``path_or_interaction_id -> coefficient to generate with``.
    coefficients: Dict[str, float] = field(default_factory=dict)
    #: Smallest N at which every path flagged significant reaches the
    #: requested power.
    n_required: int = 0
    per_path: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: List[str] = field(default_factory=list)

    @property
    def n_targeted_significant(self) -> int:
        if self.per_path.empty:
            return 0
        return int((self.per_path["requested"] == "significant").sum())

    @property
    def n_targeted_null(self) -> int:
        if self.per_path.empty:
            return 0
        return int((self.per_path["requested"] == "not significant").sum())


def build_plan(spec: ModelSpec, cm: Any, n: int, *,
               compute_power: bool = True) -> SignificancePlan:
    """Work out the generating coefficients and the sample size they need.

    Parameters
    ----------
    spec
        The model as specified.
    cm
        A compiled model, used for the Satorra-Saris power computation.
    n
        The sample size currently intended.
    compute_power
        Whether to run the Satorra-Saris power computation for each path.
        It refits the model once per path, which is far too slow for a
        calibration loop that calls this on every round, so that caller
        switches it off and asks for power once, separately.

    Returns
    -------
    SignificancePlan
    """
    policy = spec.significance_target
    rows: List[Dict[str, Any]] = []
    coefficients: Dict[str, float] = {}
    notes: List[str] = []
    n_required = int(n)

    effects: List[Tuple[Any, str, str, str]] = []
    for path in spec.paths:
        effects.append((path, "path", path.source, path.target))
    for interaction in spec.interactions:
        effects.append((interaction, "interaction",
                        interaction.predictor, interaction.target))

    for effect, kind, source_id, target_id in effects:
        requested = ("significant" if effect.target_significant
                     else "not significant")
        specified = float(effect.resolved_coefficient())
        if effect.target_significant:
            coefficient = specified
        else:
            # A null path keeps a trace rather than being nailed to zero.
            coefficient = float(np.sign(specified or 1.0)
                                * abs(policy.null_coefficient))
        coefficients[effect.id] = coefficient

        needed: Optional[int] = None
        power_at_n: Optional[float] = None
        if (compute_power and effect.target_significant
                and policy.enabled and kind == "path"):
            needed, power_at_n = _power_for_path(spec, cm, effect, n,
                                                 policy.alpha,
                                                 policy.target_power, notes)
            if needed is not None:
                n_required = max(n_required, needed)

        rows.append({
            "source": _label(spec, source_id),
            "target": _label(spec, target_id),
            "kind": kind,
            "effect_size": effect.effect_size.value,
            "specified": specified,
            "coefficient": coefficient,
            "requested": requested,
            "n_required": needed if needed is not None else np.nan,
            "power": power_at_n if power_at_n is not None else np.nan,
        })

    if not spec.significance_target.enabled:
        notes.append("Significance targeting is switched off: every path is "
                     "generated at the coefficient you specified.")
    return SignificancePlan(coefficients=coefficients,
                            n_required=int(n_required),
                            per_path=pd.DataFrame(rows), notes=notes)


def _power_for_path(spec: ModelSpec, cm: Any, path: Path, n: int,
                    alpha: float, target_power: float,
                    notes: List[str]) -> Tuple[Optional[int], Optional[float]]:
    try:
        from semsim.power import analysis as power_analysis
    except ImportError:
        return None, None
    try:
        power_at_n = float(power_analysis.satorra_saris_power(
            spec, cm, path, n, alpha=alpha))
        needed = int(power_analysis.n_for_path_power(
            spec, cm, path, target_power=target_power, alpha=alpha))
        return needed, power_at_n
    except Exception as exc:                 # never let power maths stop a run
        notes.append(f"Power for the path {path.source} -> {path.target} "
                     f"could not be computed ({type(exc).__name__}); the "
                     "sample size was left as specified.")
        return None, None


def _label(spec: ModelSpec, construct_id: str) -> str:
    construct = spec.construct(construct_id)
    return construct.name if construct is not None else construct_id


def apply_plan(spec: ModelSpec, plan: SignificancePlan) -> ModelSpec:
    """A copy of ``spec`` carrying the plan's generating coefficients.

    The researcher's own specification is never mutated: the plan's
    coefficients describe what to *generate* with, and the specification
    continues to describe what was asked for.
    """
    out = copy.deepcopy(spec)
    for path in out.paths:
        if path.id in plan.coefficients:
            path.coefficient = plan.coefficients[path.id]
            path.coefficient_is_manual = True
            path.effect_size = EffectSize.CUSTOM
    for interaction in out.interactions:
        if interaction.id in plan.coefficients:
            interaction.coefficient = plan.coefficients[interaction.id]
            interaction.coefficient_is_manual = True
            interaction.effect_size = EffectSize.CUSTOM
    return out


def check_realised(estimation: Any, spec: ModelSpec,
                   alpha: Optional[float] = None) -> Tuple[bool, pd.DataFrame]:
    """Did every targeted path land on the right side of alpha?

    Returns ``(all_matched, table)``.  The table has one row per path with the
    requested outcome, the realised p-value and whether they agree, which is
    what the results page shows and what the calibration loop tests.
    """
    if alpha is None:
        alpha = spec.significance_target.alpha
    estimates = getattr(estimation, "path_estimates", None)
    if estimates is None or len(estimates) == 0:
        return False, pd.DataFrame()

    by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _index, row in estimates.iterrows():
        by_pair[(str(row.get("source")), str(row.get("target")))] = row

    rows: List[Dict[str, Any]] = []
    all_matched = True
    for path in spec.paths:
        source = _label(spec, path.source)
        target = _label(spec, path.target)
        row = by_pair.get((source, target))
        if row is None:
            continue
        p_value = float(row.get("p", np.nan))
        realised = "significant" if p_value < alpha else "not significant"
        requested = ("significant" if path.target_significant
                     else "not significant")
        matched = realised == requested
        all_matched &= matched
        rows.append({
            "source": source,
            "target": target,
            "specified": float(path.resolved_coefficient()),
            "estimate": float(row.get("std", row.get("est", np.nan))),
            "p": p_value,
            "requested": requested,
            "realised": realised,
            "verdict": "pass" if matched else "fail",
        })
    return all_matched, pd.DataFrame(rows)


__all__ = ["SignificancePlan", "build_plan", "apply_plan", "check_realised"]

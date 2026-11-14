"""
semsim.generation.heterogeneity
===============================
Draw one realistic population around the researcher's specification.

A specification is a theory, and a theory is tidy: every item of a construct
loads 0.78, every item answers in the same symmetric bell, and every path is a
round number.  Data drawn straight from it carry the tidiness into the file.
Every item mean sits on the scale midpoint, every standard deviation is the
same, every loading is the same, and a reader who has seen real questionnaire
data recognises it as simulated at a glance.

Real scales are not like that, for reasons that are well documented:

* **Constructs differ in level.**  Satisfaction with a service and trust in
  its provider are not answered around the same point of the scale.
* **Items differ in difficulty.**  In item response theory each item of a
  graded scale has its own location: some statements are easy to agree with
  and some are not (Samejima, 1969; Embretson & Reise, 2000).  Their means and
  skewness differ accordingly.
* **Items differ in spread.**  Some statements split respondents, others
  gather them in the middle, so item standard deviations differ.
* **Loadings and paths are not round.**  The population a sample is drawn from
  is never exactly the hypothesised one; the values a researcher plans are
  approximations of it (MacCallum & Tucker, 1991, on model error in
  population).

This module draws one population per run.  It works on a copy of the
specification and changes four things:

1. each Likert item gets its own cut points, shifted by its construct's level
   plus its own difficulty and stretched by its own spread;
2. each loading and second-order loading moves a little around its planned
   value;
3. each path the researcher wants significant, and each correlation between
   exogenous constructs, moves a little around its planned value, never far
   enough to change its sign or halve it;
4. nothing else: paths that must stay null, moderation terms, explicit cut
   points and explicit category proportions are left exactly as given.

The population comes from a random stream of its own, keyed on the run seed,
so the same seed gives the same population, and the draws that significance
targeting rejects and repeats all sample from that one population.  Any
change that would make the model harder to compile than the original (an
over-explained construct, a loading row that must be rescaled) is undone, so
the variation never turns a valid model into an invalid one.

References
----------
Samejima, F. (1969). Estimation of latent ability using a response pattern of
    graded scores. *Psychometrika Monograph Supplement*, 34(4, Pt. 2), 100.
Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for
    Psychologists*. Erlbaum.
MacCallum, R. C., & Tucker, L. R. (1991). Representing sources of error in
    the common-factor model: Implications for theory and practice.
    *Psychological Bulletin*, 109(3), 502-511.
Rhemtulla, M., Brosseau-Liard, P. E., & Savalei, V. (2012). When can
    categorical variables be treated as continuous? *Psychological Methods*,
    17(3), 354-373.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from semsim.core.errors import SemSimError
from semsim.core.spec import (
    Construct, EffectSize, HeterogeneitySpec, Item, ModelSpec, ScaleKind,
)

#: Mixed into the seed so the population has random streams of its own, one
#: per part, so changing the path variation never moves the item levels.
_STREAM = 7919
_LEVELS, _LOADINGS, _PATHS = 1, 2, 3

#: A draw is cut off at this many standard deviations, so one unlucky item
#: cannot land at the edge of the scale.
_TRUNCATE = 2.5

#: Bounds on a varied loading.
_LOADING_MIN, _LOADING_MAX = 0.30, 0.95

#: A varied path keeps at least this share of its planned size and at most
#: the reciprocal share above it, so a hypothesis keeps its direction and its
#: rough strength.
_PATH_KEEP = 0.75

#: Paths smaller than this are left alone: they are nulls or near-nulls.
_PATH_MIN = 0.05

#: Smallest share a category may fall to after the cut points move, unless
#: the planned style already gives it less.  An empty category breaks the
#: polychoric correlations.
_CATEGORY_FLOOR = 0.004

__all__ = ["realise_population", "population_stream"]


def population_stream(seed: int, part: int) -> np.random.Generator:
    """The random stream for one part of the population of a given run.

    Parameters
    ----------
    seed : int
        The run's seed.
    part : int
        Which part: response levels, loadings or paths.

    Returns
    -------
    numpy.random.Generator
        A generator independent of the ones that draw the responses.

    References
    ----------
    O'Neill, M. E. (2014). PCG: A family of simple fast space-efficient
        statistically good algorithms for random number generation. Harvey
        Mudd College Technical Report HMC-CS-2014-0905.
    """
    return np.random.default_rng([abs(int(seed)), _STREAM, int(part)])


def _normal(rng: np.random.Generator, sd: float) -> float:
    """One normal draw with standard deviation ``sd``, truncated."""
    if sd <= 0.0:
        return 0.0
    return float(np.clip(rng.standard_normal(), -_TRUNCATE, _TRUNCATE) * sd)


def realise_population(spec: ModelSpec, seed: int) -> ModelSpec:
    """A copy of ``spec`` describing one realistic population around it.

    Parameters
    ----------
    spec : ModelSpec
        The model as specified.  It is never modified.
    seed : int
        The run's seed; the same seed gives the same population.

    Returns
    -------
    ModelSpec
        ``spec`` itself when variation is switched off, otherwise a deep copy
        whose items carry their own cut points and whose loadings, paths and
        correlations have moved around their planned values.  The copy has
        variation switched off, so passing it through again changes nothing.

    References
    ----------
    MacCallum, R. C., & Tucker, L. R. (1991). Representing sources of error
        in the common-factor model: Implications for theory and practice.
        *Psychological Bulletin*, 109(3), 502-511.
    """
    settings = getattr(spec, "heterogeneity", None)
    if settings is None or not settings.enabled:
        return spec
    out = copy.deepcopy(spec)
    out.heterogeneity.enabled = False

    _vary_response_levels(out, settings, population_stream(seed, _LEVELS))

    baseline = _compile_warnings(spec)
    if baseline is None:
        # The model does not compile as specified; leave the structure alone
        # and let the caller report the real problem.
        return out

    saved = _snapshot_loadings(out)
    _vary_loadings(out, settings, population_stream(seed, _LOADINGS))
    if not _compiles_as_well(out, baseline):
        _restore_loadings(out, saved)

    saved_paths = _snapshot_structure(out)
    _vary_structure(out, settings, population_stream(seed, _PATHS))
    if not _compiles_as_well(out, baseline):
        _restore_structure(out, saved_paths)
    return out


# --------------------------------------------------------------------------- #
# Response levels
# --------------------------------------------------------------------------- #


def _questionnaire_constructs(spec: ModelSpec) -> List[Construct]:
    return [c for c in spec.constructs if not c.covariate]


def _vary_response_levels(spec: ModelSpec, settings: HeterogeneitySpec,
                          rng: np.random.Generator) -> None:
    """Give each item its own location and spread on the response scale."""
    for construct in _questionnaire_constructs(spec):
        level = _normal(rng, settings.construct_level_sd)
        if construct.is_second_order:
            groups = [(level + _normal(rng, settings.item_level_sd / 2.0),
                       dimension.items) for dimension in construct.dimensions]
        else:
            groups = [(level, construct.items)]
        for group_level, items in groups:
            for item in items:
                shift = group_level + _normal(rng, settings.item_level_sd)
                stretch = float(np.exp(_normal(rng, settings.item_spread_sd)))
                _move_item(item, shift, stretch)


def _move_item(item: Item, shift: float, stretch: float) -> None:
    """Move one item's response distribution up by ``shift`` (z units) and
    widen its cut points by ``stretch``.

    A positive shift lowers every cut point, so more of the latent response
    falls in the higher categories and the item mean rises.  Stretching the
    cut points about their own centre gathers answers in the middle
    categories (``stretch > 1``) or pushes them to the ends (``< 1``).
    """
    scale = item.scale
    if scale.kind is ScaleKind.CONTINUOUS:
        item.scale = replace(scale, mean=float(scale.mean + shift * scale.sd),
                             sd=float(scale.sd / stretch))
        return
    if scale.thresholds or scale.category_proportions:
        # The researcher fixed this item's distribution exactly.
        return
    from semsim.generation import thresholds as th
    try:
        tau = th.scale_thresholds(scale)
    except SemSimError:
        return
    planned = th.category_probabilities(tau)
    # A skewed style can already leave an end category nearly empty; the move
    # may thin it further, but never below half its planned share.
    floor = min(_CATEGORY_FLOOR, 0.5 * float(planned.min()))
    centre = float(tau.mean())

    def moved(t: float) -> np.ndarray:
        """The cut points after ``t`` of the full move (0 = as planned)."""
        return (centre + (1.0 + (stretch - 1.0) * t) * (tau - centre)
                - shift * t)

    def allowed(t: float) -> bool:
        return float(th.category_probabilities(moved(t)).min()) >= floor

    t = 1.0
    if not allowed(t):
        # Too far for this scale: take the largest part of the move that
        # keeps every category populated.
        low, high = 0.0, 1.0
        for _step in range(30):
            middle = 0.5 * (low + high)
            if allowed(middle):
                low = middle
            else:
                high = middle
        t = low
    item.scale = replace(scale, thresholds=[float(v) for v in moved(t)])


# --------------------------------------------------------------------------- #
# Loadings
# --------------------------------------------------------------------------- #


def _snapshot_loadings(spec: ModelSpec) -> Dict[str, float]:
    saved: Dict[str, float] = {}
    for construct in _questionnaire_constructs(spec):
        for item in construct.all_items():
            saved[item.id] = item.loading
        for dimension in construct.dimensions:
            saved[dimension.id] = dimension.second_order_loading
    return saved


def _restore_loadings(spec: ModelSpec, saved: Dict[str, float]) -> None:
    for construct in _questionnaire_constructs(spec):
        for item in construct.all_items():
            item.loading = saved.get(item.id, item.loading)
        for dimension in construct.dimensions:
            dimension.second_order_loading = saved.get(
                dimension.id, dimension.second_order_loading)


def _varied_loading(value: float, sd: float, rng: np.random.Generator) -> float:
    if sd <= 0.0 or abs(value) < 1e-9:
        return value
    sign = 1.0 if value >= 0 else -1.0
    size = abs(value) + _normal(rng, sd)
    size = float(np.clip(size, _LOADING_MIN, _LOADING_MAX))
    if abs(value) > _LOADING_MAX:
        size = abs(value)
    return sign * size


def _vary_loadings(spec: ModelSpec, settings: HeterogeneitySpec,
                   rng: np.random.Generator) -> None:
    for construct in _questionnaire_constructs(spec):
        for item in construct.all_items():
            item.loading = _varied_loading(item.loading, settings.loading_sd, rng)
        for dimension in construct.dimensions:
            dimension.second_order_loading = _varied_loading(
                dimension.second_order_loading, settings.loading_sd, rng)


# --------------------------------------------------------------------------- #
# Paths and correlations
# --------------------------------------------------------------------------- #


def _snapshot_structure(spec: ModelSpec) -> Dict[str, tuple]:
    saved: Dict[str, tuple] = {}
    for path in spec.paths:
        saved[path.id] = (path.coefficient, path.coefficient_is_manual,
                          path.effect_size)
    for index, cov in enumerate(spec.latent_covariances):
        saved[f"cov{index}"] = (cov.correlation,)
    return saved


def _restore_structure(spec: ModelSpec, saved: Dict[str, tuple]) -> None:
    for path in spec.paths:
        if path.id in saved:
            (path.coefficient, path.coefficient_is_manual,
             path.effect_size) = saved[path.id]
    for index, cov in enumerate(spec.latent_covariances):
        if f"cov{index}" in saved:
            cov.correlation = saved[f"cov{index}"][0]


def _varied_effect(value: float, step: float) -> Optional[float]:
    """``value`` moved by ``step``, keeping its sign and rough size."""
    if step == 0.0 or abs(value) < _PATH_MIN:
        return None
    low, high = sorted((value * _PATH_KEEP, value / _PATH_KEEP))
    return float(np.clip(value + step, low, high))


def _vary_structure(spec: ModelSpec, settings: HeterogeneitySpec,
                    rng: np.random.Generator) -> None:
    covariates = {c.id for c in spec.constructs if c.covariate}
    for path in spec.paths:
        # A step is drawn for every path, used or not, so that whether one
        # path varies never moves the others.
        step = _normal(rng, settings.path_sd)
        if not path.target_significant or path.source in covariates:
            continue
        moved = _varied_effect(float(path.resolved_coefficient()), step)
        if moved is None:
            continue
        path.coefficient = moved
        path.coefficient_is_manual = True
        path.effect_size = EffectSize.CUSTOM
    for cov in spec.latent_covariances:
        step = _normal(rng, settings.path_sd)
        # Correlations involving a control term are fixed by how the sample
        # was drawn (independent quotas, mutually exclusive dummies).
        if cov.construct_a in covariates or cov.construct_b in covariates:
            continue
        moved = _varied_effect(float(cov.correlation), step)
        if moved is not None:
            cov.correlation = float(np.clip(moved, -0.95, 0.95))


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def _compile_warnings(spec: ModelSpec) -> Optional[int]:
    """How many warnings compiling ``spec`` raises, or ``None`` if it fails."""
    from semsim.core.compile import compile_model
    try:
        return len(compile_model(spec).warnings)
    except Exception:  # noqa: BLE001 - any failure means "do not vary"
        return None


def _compiles_as_well(spec: ModelSpec, baseline: int) -> bool:
    count = _compile_warnings(spec)
    return count is not None and count <= baseline

"""
semsim.generation.engine
========================
Turn a :class:`~semsim.core.spec.ModelSpec` into a dataset.

Two generators, because no single one covers the ground:

**Covariance engine** (the default).  Build the model-implied covariance
matrix, optionally perturb it to the requested amount of model error, then draw
from it with Vale & Maurelli's (1983) method so each column lands on its target
skewness and kurtosis while the whole matrix lands on its target correlations.
This is what the simulation literature does and it gives the tightest control
over the distributions.

**Score engine.**  Draw the latent variables themselves and push them through
the structural equations.  Slower and slightly looser on the marginals, but it
is the only way to produce a *latent interaction*: the product of two latent
variables has no expression in a covariance matrix, so a moderation hypothesis
cannot be simulated any other way.

Non-normality under the score engine uses an exact result rather than an
approximation.  For an item ``y = sum_k lambda_k eta_k + e`` with normal
latents, independent error and unit total variance,

    skew(y)     = skew(e) * theta ** 1.5
    kurtosis(y) = kurt(e) * theta ** 2

so the error term is given the shape that makes the *item* land on target:
``skew(e) = target_skew / theta ** 1.5``.  Where that is outside Fleishman's
feasible region the request is clamped and a warning is recorded, never
silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from semsim.core import cancel as _cancel
from semsim.core.compile import CompiledModel, compile_model
from semsim.core.errors import SemSimError, SpecificationError
from semsim.core import implied as implied_module
from semsim.core.spec import (
    GenerationEngine, Item, ModelSpec, ScaleKind,
)

ProgressFn = Optional[Callable[[str, float], None]]


@dataclass
class GeneratedDataset:
    """Everything one simulation run produced."""

    data: pd.DataFrame
    compiled: CompiledModel
    sigma_population: np.ndarray
    sigma_generating: np.ndarray
    n: int
    seed: int
    engine_used: GenerationEngine
    latent_scores: Optional[pd.DataFrame] = None
    continuous_shadow: Optional[pd.DataFrame] = None
    noise_report: object = None
    model_error: object = None
    warnings: List[str] = field(default_factory=list)
    #: Category codes per respondent, and any dummy columns.  Kept apart from
    #: ``data`` so reliability and item statistics see questionnaire items only.
    demographics: Optional[pd.DataFrame] = None

    @property
    def item_names(self) -> List[str]:
        return list(self.data.columns)

    @property
    def model_frame(self) -> pd.DataFrame:
        """The columns the fitted model needs: control terms, then items."""
        wanted = self.compiled.spec.control_columns() \
            if not self.compiled.covariate_rows else [
                self.compiled.item_names[j] for j in self.compiled.covariate_rows]
        if self.demographics is None or not wanted:
            return self.data
        return pd.concat([self.demographics[wanted], self.data], axis=1)

    @property
    def full_frame(self) -> pd.DataFrame:
        """Everything a researcher gets in a file: demographics, then items."""
        if self.demographics is None or self.demographics.empty:
            return self.data
        return pd.concat([self.demographics, self.data], axis=1)

    @property
    def has_missing(self) -> bool:
        return bool(self.data.isna().to_numpy().any())

    def summary(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "items": self.data.shape[1],
            "seed": self.seed,
            "engine": self.engine_used.value,
            "missing_cells": int(self.data.isna().to_numpy().sum()),
            "warnings": len(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def generate_dataset(spec: ModelSpec, *, n: Optional[int] = None,
                     seed: Optional[int] = None,
                     progress: ProgressFn = None,
                     sigma_override: Optional[np.ndarray] = None,
                     compiled: Optional[CompiledModel] = None
                     ) -> GeneratedDataset:
    """Simulate one dataset.

    Parameters
    ----------
    spec
        The model to simulate.
    n
        Sample size; defaults to ``spec.power.n``.
    seed
        Overrides ``spec.seed``.  The same seed always reproduces the same
        dataset, which is the point of carrying one at all.
    progress
        Called as ``progress(message_key, fraction)`` so a GUI can show where
        the run has got to.
    sigma_override
        Generate from this matrix instead of the model-implied one.  The
        calibration loop uses it to hand back a perturbed matrix without
        recompiling.
    compiled
        A pre-compiled model, to avoid compiling twice in a calibration loop.
    """
    def report(key: str, fraction: float) -> None:
        _cancel.check()
        if progress is not None:
            progress(key, fraction)

    warnings: List[str] = []
    n_cases = int(n if n is not None else spec.power.n)
    if n_cases < 5:
        raise SpecificationError(
            f"A sample of {n_cases} cases is too small to simulate.",
            f"حجم العينة ({n_cases}) أصغر من أن يُحاكى.",
            n=n_cases)
    actual_seed = int(seed if seed is not None
                      else (spec.seed if spec.seed is not None else 0))
    rng = np.random.default_rng(actual_seed)

    report("status.compiling", 0.05)
    if compiled is None and sigma_override is None:
        # A caller that hands over a compiled model has already drawn its
        # population (the calibration loop does, once for all its draws).
        from semsim.generation.heterogeneity import realise_population
        spec = realise_population(spec, actual_seed)
    cm = compiled if compiled is not None else compile_model(spec)
    warnings.extend(cm.warnings)

    demographics = None
    drawn = None
    if spec.demographics:
        from semsim.generation.demographics import draw_demographics_jointly
        drawn = draw_demographics_jointly(spec, n_cases, actual_seed)
        demographics = drawn.frame
        warnings.extend(drawn.notes)
    controls = _standardised_controls(cm, demographics)

    sigma_population = implied_module.implied_covariance(cm)
    sigma_generating = sigma_population
    model_error = None

    if sigma_override is not None:
        sigma_generating = np.asarray(sigma_override, dtype=float)
    elif _wants_model_error(spec):
        report("status.calibrating", 0.15)
        sigma_generating, model_error, note = _inject_model_error(
            spec, cm, rng, warnings)
        if note:
            warnings.append(note)

    engine = spec.resolved_engine()
    report("status.generating", 0.30)

    if engine is GenerationEngine.SCORES:
        y_star, eta = _generate_from_scores(spec, cm, n_cases, rng, warnings,
                                            controls)
        if sigma_generating is not sigma_population:
            # The scores are drawn from the structural equations, so they
            # would ignore the misfit injected into the generating matrix
            # and fit the model almost perfectly whatever RMSEA was asked for.
            y_star = _impose_covariance(y_star, sigma_population,
                                        sigma_generating, cm)
    elif controls is not None:
        y_star = _generate_given_controls(spec, cm, sigma_generating, controls,
                                          rng, warnings)
        eta = None
    else:
        y_star = _generate_from_covariance(spec, cm, sigma_generating,
                                           n_cases, rng, warnings)
        eta = None

    # From here on only the questionnaire items are scored and contaminated.
    # The control terms keep the exact codes the quotas assigned.
    full_star = y_star
    items_cm = cm
    if cm.covariate_rows:
        y_star = y_star[:, :cm.n_substantive_items]
        items_cm = _questionnaire_view(cm)

    report("status.generating", 0.55)
    observed = _apply_scales(spec, items_cm, y_star, warnings)

    report("status.generating", 0.70)
    noise_report = None
    if spec.noise.enabled:
        try:
            from semsim.generation.noise import apply_noise
            observed, noise_report = apply_noise(observed, spec, items_cm, rng)
        except ImportError:
            warnings.append("NOISE_UNAVAILABLE: the noise module is not "
                            "installed; the dataset carries no contamination.")

    data = pd.DataFrame(observed, columns=items_cm.item_names)
    if demographics is not None:
        # Regroup comparison variables by the finished answers, so a t-test
        # or ANOVA on them finds the differences the researcher planned.
        from semsim.generation.group_differences import apply_group_differences
        demographics, notes = apply_group_differences(spec, data, demographics,
                                                      actual_seed, drawn=drawn)
        warnings.extend(notes)
    shadow = (pd.DataFrame(y_star, columns=[f"{c}_star"
                                            for c in items_cm.item_names])
              if spec.export.include_continuous_shadow else None)
    latent = None
    if eta is not None and spec.export.include_latent_scores:
        latent = pd.DataFrame(eta, columns=cm.factor_names)
    elif spec.export.include_latent_scores:
        latent = _factor_score_estimates(cm, full_star)
    if latent is not None and cm.covariate_factors:
        latent = latent.drop(columns=[latent.columns[f]
                                      for f in cm.covariate_factors])

    report("status.done", 1.0)
    return GeneratedDataset(
        data=data, compiled=cm,
        sigma_population=sigma_population,
        sigma_generating=sigma_generating,
        n=n_cases, seed=actual_seed, engine_used=engine,
        latent_scores=latent, continuous_shadow=shadow,
        noise_report=noise_report, model_error=model_error,
        warnings=warnings, demographics=demographics,
    )


# --------------------------------------------------------------------------- #
# Control terms
# --------------------------------------------------------------------------- #


def _standardised_controls(cm: CompiledModel,
                           demographics: Optional[pd.DataFrame]
                           ) -> Optional[np.ndarray]:
    """Control terms as z-scores, in the order of ``cm.covariate_rows``.

    Standardising on the sample's own mean and variance is what makes the
    requested coefficient the standardised coefficient of the observed coded
    variable, whatever the coding and whatever the quota.
    """
    if not cm.covariate_rows:
        return None
    if demographics is None:
        raise SpecificationError(
            "The model uses control variables but no demographics were drawn.",
            "النموذج يستخدم متغيرات ضابطة لكن لم يتم توليد المتغيرات "
            "الديموغرافية.")
    names = [cm.item_names[j] for j in cm.covariate_rows]
    values = demographics[names].to_numpy(dtype=float)
    spread = values.std(axis=0, ddof=1)
    constant = [name for name, sd in zip(names, spread) if sd < 1e-9]
    if constant:
        raise SpecificationError(
            "These control terms have no variance in this sample, so they "
            f"cannot have an effect: {', '.join(constant)}. Raise the sample "
            "size or the share of the smallest category.",
            "هذه الحدود الضابطة بلا تباين في هذه العينة، فلا يمكن أن يكون لها "
            f"أثر: {'، '.join(constant)}. زِد حجم العينة أو نسبة أصغر فئة.",
            terms=constant)
    return (values - values.mean(axis=0)) / spread


def _questionnaire_view(cm: CompiledModel) -> CompiledModel:
    """The compiled model as seen by the scoring and noise steps."""
    p = cm.n_substantive_items
    return replace(
        cm, item_ids=cm.item_ids[:p], item_names=cm.item_names[:p],
        Lambda=cm.Lambda[:p], Theta=cm.Theta[:p, :p],
        item_index={iid: j for iid, j in cm.item_index.items() if j < p},
        home_factor=cm.home_factor[:p], covariate_rows=[],
        covariate_factors=[])


def _generate_given_controls(spec: ModelSpec, cm: CompiledModel,
                             sigma: np.ndarray, controls: np.ndarray,
                             rng: np.random.Generator,
                             warnings: List[str]) -> np.ndarray:
    """Draw the items conditionally on the control terms already assigned.

    With ``c`` the standardised control terms and ``S_cc`` their sample
    correlation, the items are ``y = W c + e`` where ``W = Sigma_yc S_cc^-1``
    and ``e`` has covariance ``Sigma_yy - W Sigma_cy``.  Each control keeps its
    categorical distribution, which a joint normal draw could not do, and each
    target regression recovers the requested coefficient on the observed coded
    variable, so no correction for dichotomising is needed.

    References
    ----------
    Anderson, T. W. (2003). *An Introduction to Multivariate Statistical
        Analysis* (3rd ed.), section 2.5. Wiley.
    """
    p = sigma.shape[0]
    rows = np.arange(cm.n_substantive_items)
    cols = np.asarray(cm.covariate_rows, dtype=int)
    S_cc = np.atleast_2d(np.corrcoef(controls, rowvar=False))
    sigma_yc = sigma[np.ix_(rows, cols)]
    W = np.linalg.solve(S_cc, sigma_yc.T).T
    conditional = sigma[np.ix_(rows, rows)] - W @ sigma_yc.T
    conditional = 0.5 * (conditional + conditional.T)
    sd = np.sqrt(np.clip(np.diag(conditional), 1e-9, None))
    correlation = conditional / np.outer(sd, sd)
    np.fill_diagonal(correlation, 1.0)
    if not implied_module.is_positive_definite(correlation):
        correlation = implied_module.nearest_positive_definite(correlation)
        np.fill_diagonal(correlation, 1.0)

    from semsim.generation import vale_maurelli as vm
    n = controls.shape[0]
    skew, kurt = _marginal_targets(spec, cm)
    skew, kurt = skew[rows], kurt[rows]
    if (spec.distribution.use_non_normal_engine
            and (np.any(np.abs(skew) > 1e-9) or np.any(np.abs(kurt) > 1e-9))):
        try:
            errors = vm.generate(n, correlation, skew, kurt, rng)
        except SemSimError as exc:
            warnings.append(
                f"NON_NORMAL_FALLBACK: {exc.message_en} Falling back on a "
                "normal draw, so the requested skewness and kurtosis were not "
                "applied.")
            errors = vm.generate_normal(n, correlation, rng)
    else:
        errors = vm.generate_normal(n, correlation, rng)

    out = np.zeros((n, p))
    out[:, rows] = controls @ W.T + errors * sd
    out[:, cols] = controls
    return out


# --------------------------------------------------------------------------- #
# Model error
# --------------------------------------------------------------------------- #


def _wants_model_error(spec: ModelSpec) -> bool:
    target = spec.fit_target
    return bool(target.enabled
                and target.method not in ("none", "")
                and _target_rmsea(spec) > 0.0
                and spec.degrees_of_freedom() > 0)


def _target_rmsea(spec: ModelSpec) -> float:
    target = spec.fit_target
    if target.level.value == "custom":
        return float(target.rmsea)
    return float(target.level.target_rmsea)


def _inject_model_error(spec: ModelSpec, cm: CompiledModel,
                        rng: np.random.Generator, warnings: List[str]):
    """Perturb the implied matrix to the requested amount of misfit."""
    try:
        from semsim.targeting.cudeck_browne import inject_model_error
    except ImportError:
        return (implied_module.implied_covariance(cm), None,
                "MODEL_ERROR_UNAVAILABLE: the model-error module is not "
                "installed; the data will fit the model perfectly.")
    try:
        result = inject_model_error(
            cm, _target_rmsea(spec),
            df=spec.degrees_of_freedom(cm),
            rng=rng, method=spec.fit_target.method)
    except SemSimError as exc:
        return (implied_module.implied_covariance(cm), None,
                f"MODEL_ERROR_SKIPPED: {exc.message_en}")
    return result.sigma_0, result, ""


# --------------------------------------------------------------------------- #
# Covariance engine
# --------------------------------------------------------------------------- #


def _generate_from_covariance(spec: ModelSpec, cm: CompiledModel,
                              sigma: np.ndarray, n: int,
                              rng: np.random.Generator,
                              warnings: List[str]) -> np.ndarray:
    skewness, kurtosis = _marginal_targets(spec, cm)
    use_non_normal = (spec.distribution.use_non_normal_engine
                      and (np.any(np.abs(skewness) > 1e-9)
                           or np.any(np.abs(kurtosis) > 1e-9)))
    from semsim.generation import vale_maurelli as vm
    if not use_non_normal:
        return vm.generate_normal(n, sigma, rng)
    try:
        return vm.generate(n, sigma, skewness, kurtosis, rng)
    except SemSimError as exc:
        warnings.append(
            f"NON_NORMAL_FALLBACK: {exc.message_en} Falling back on a normal "
            "draw, so the requested skewness and kurtosis were not applied.")
        return vm.generate_normal(n, sigma, rng)


def _marginal_targets(spec: ModelSpec,
                      cm: CompiledModel) -> Tuple[np.ndarray, np.ndarray]:
    """Per-item skewness and excess kurtosis for the continuous draw.

    An ordinal item takes its shape from its thresholds, so asking for a
    skewed *latent* response as well would double-count the asymmetry and
    break the normality the polychoric estimator assumes.  Those items are
    drawn normal and shaped by :func:`_apply_scales`.
    """
    distribution = spec.distribution
    skew = np.zeros(len(cm.item_ids))
    kurt = np.zeros(len(cm.item_ids))
    for item in spec.all_items():
        index = cm.item_index.get(item.id)
        if index is None:
            continue
        if item.scale.kind is ScaleKind.LIKERT:
            continue
        skew[index] = (item.skewness if abs(item.skewness) > 1e-12
                       else distribution.default_skewness)
        kurt[index] = (item.kurtosis if abs(item.kurtosis) > 1e-12
                       else distribution.default_kurtosis)
    if distribution.clamp_to_feasible_region:
        skew, kurt = _clamp_feasible(skew, kurt)
    return skew, kurt


def _clamp_feasible(skew: np.ndarray,
                    kurt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fleishman has no real solution below ``kurtosis = skew**2 - 2``."""
    floor = skew ** 2 - 2.0 + 0.05
    return skew, np.maximum(kurt, floor)


# --------------------------------------------------------------------------- #
# Score engine
# --------------------------------------------------------------------------- #


def _symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """``matrix ** power`` for a symmetric positive (semi)definite matrix."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.clip(values, 1e-10, None)
    return (vectors * values ** power) @ vectors.T


def _impose_covariance(y_star: np.ndarray, sigma_model: np.ndarray,
                       sigma_target: np.ndarray,
                       cm: CompiledModel) -> np.ndarray:
    """Give score-generated data the covariance of the misfit matrix.

    Data drawn through the structural equations has, in the population, the
    model-implied covariance ``sigma_model`` -- moderation adds variance to
    an outcome but no covariance with the constructs that form the product.
    Post-multiplying by ``A = sigma_model^(-1/2) sigma_target^(1/2)`` turns
    that population covariance into ``sigma_target`` exactly
    (``A' sigma_model A = sigma_target``), the same matrix the covariance
    engine draws from.  The symmetric roots make ``A`` the transformation
    closest to the identity, so with the small perturbation a fit target
    asks for each column keeps its own content, and the interaction its
    effect.  Sampling error is untouched: only the population moment moves.

    Control columns hold the exact codes their quotas assigned, so they are
    left alone and only the questionnaire block is transformed.
    """
    covariates = set(cm.covariate_rows)
    keep = [j for j in range(y_star.shape[1]) if j not in covariates]
    if not keep:
        return y_star
    block = np.ix_(keep, keep)
    transform = (_symmetric_power(sigma_model[block], -0.5)
                 @ _symmetric_power(sigma_target[block], 0.5))
    out = np.array(y_star, dtype=float, copy=True)
    out[:, keep] = y_star[:, keep] @ transform
    return out


def _generate_from_scores(spec: ModelSpec, cm: CompiledModel, n: int,
                          rng: np.random.Generator,
                          warnings: List[str],
                          controls: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Draw the latents and push them through the structural equations."""
    m = len(cm.factor_ids)
    eta = np.zeros((n, m))

    # Control terms are exogenous too, but their values are already assigned.
    given = set(cm.covariate_factors)
    exogenous = [i for i in range(m)
                 if not bool(cm.endogenous[i]) and i not in given]
    endogenous = [i for i in range(m) if bool(cm.endogenous[i])]
    if controls is not None:
        by_row = {cm.home_factor[j]: k for k, j in enumerate(cm.covariate_rows)}
        for factor, k in by_row.items():
            eta[:, factor] = controls[:, k]

    if exogenous:
        psi_exo = cm.Psi[np.ix_(exogenous, exogenous)]
        psi_exo = implied_module.nearest_positive_definite(psi_exo) \
            if not implied_module.is_positive_definite(psi_exo) else psi_exo
        chol = np.linalg.cholesky(psi_exo)
        eta[:, exogenous] = rng.standard_normal((n, len(exogenous))) @ chol.T

    interactions = _interaction_terms(spec, cm)
    order = _topological_order(cm, endogenous)
    for index in order:
        predictors = np.nonzero(np.abs(cm.B[index]) > 1e-12)[0]
        linear = (eta[:, predictors] @ cm.B[index, predictors]
                  if predictors.size else np.zeros(n))
        product = np.zeros(n)
        extra_variance = 0.0
        for predictor, moderator, coefficient in interactions.get(index, []):
            term = eta[:, predictor] * eta[:, moderator]
            product += coefficient * term
            # For jointly normal standardised X and M with correlation r,
            # Var(X*M) = 1 + r**2 and Cov(X, X*M) = 0, so the product adds
            # variance without covarying with its own constituents.
            r = float(cm.Phi[predictor, moderator])
            extra_variance += coefficient ** 2 * (1.0 + r ** 2)
        explained = float(cm.B[index, predictors] @ cm.Phi[np.ix_(predictors, predictors)]
                          @ cm.B[index, predictors]) if predictors.size else 0.0
        disturbance = 1.0 - explained - extra_variance
        if disturbance <= 1e-6:
            warnings.append(
                f"INTERACTION_HEYWOOD: the predictors of "
                f"'{cm.factor_names[index]}' explain more than its total "
                f"variance once the interaction term is included; the "
                f"disturbance variance was floored.")
            disturbance = 1e-6
        eta[:, index] = (linear + product
                         + rng.standard_normal(n) * math.sqrt(disturbance))

    y_star = eta @ cm.Lambda.T + _draw_errors(spec, cm, n, rng, warnings)
    # Unit variance is only guaranteed in expectation here; rescale so the
    # thresholds downstream see the metric they were computed for.
    sd = y_star.std(axis=0, ddof=1)
    sd[sd < 1e-9] = 1.0
    y_star = (y_star - y_star.mean(axis=0)) / sd
    if controls is not None:
        y_star[:, cm.covariate_rows] = controls
    return y_star, eta


def _interaction_terms(spec: ModelSpec, cm: CompiledModel
                       ) -> Dict[int, List[Tuple[int, int, float]]]:
    out: Dict[int, List[Tuple[int, int, float]]] = {}
    for interaction in spec.interactions:
        target = cm.factor_index.get(interaction.target)
        predictor = cm.factor_index.get(interaction.predictor)
        moderator = cm.factor_index.get(interaction.moderator)
        if None in (target, predictor, moderator):
            continue
        out.setdefault(target, []).append(
            (predictor, moderator, interaction.resolved_coefficient()))
    return out


def _topological_order(cm: CompiledModel, endogenous: Sequence[int]) -> List[int]:
    """Endogenous latents ordered so every predictor is drawn first."""
    remaining = list(endogenous)
    done = {i for i in range(len(cm.factor_ids)) if i not in set(endogenous)}
    order: List[int] = []
    for _ in range(len(remaining) + 1):
        progressed = False
        for index in list(remaining):
            predictors = np.nonzero(np.abs(cm.B[index]) > 1e-12)[0]
            if all(p in done for p in predictors):
                order.append(index)
                done.add(index)
                remaining.remove(index)
                progressed = True
        if not remaining:
            break
        if not progressed:
            # A feedback loop: no ordering exists. Emit the rest in index
            # order so something is produced rather than nothing.
            order.extend(remaining)
            break
    return order


def _draw_errors(spec: ModelSpec, cm: CompiledModel, n: int,
                 rng: np.random.Generator,
                 warnings: List[str]) -> np.ndarray:
    """Unique errors, shaped so each ITEM lands on its target moments."""
    theta = cm.Theta
    p = theta.shape[0]
    sd = np.sqrt(np.clip(np.diag(theta), 1e-12, None))
    correlation = theta / np.outer(sd, sd)
    np.fill_diagonal(correlation, 1.0)
    if not implied_module.is_positive_definite(correlation):
        correlation = implied_module.nearest_positive_definite(correlation)
    chol = np.linalg.cholesky(correlation)
    z = rng.standard_normal((n, p)) @ chol.T

    skew_target, kurt_target = _marginal_targets(spec, cm)
    if not (spec.distribution.use_non_normal_engine
            and (np.any(np.abs(skew_target) > 1e-9)
                 or np.any(np.abs(kurt_target) > 1e-9))):
        return z * sd

    from semsim.generation import fleishman
    for j in range(p):
        if abs(skew_target[j]) < 1e-9 and abs(kurt_target[j]) < 1e-9:
            continue
        theta_jj = float(theta[j, j])
        if theta_jj < 1e-6:
            continue
        # skew(y) = skew(e) * theta**1.5 and kurt(y) = kurt(e) * theta**2 when
        # the latent part is normal and independent of the error.
        need_skew = skew_target[j] / theta_jj ** 1.5
        need_kurt = kurt_target[j] / theta_jj ** 2
        if not fleishman.is_feasible(need_skew, need_kurt):
            warnings.append(
                f"SHAPE_INFEASIBLE: item '{cm.item_names[j]}' would need an "
                f"error skewness of {need_skew:.2f} to reach the requested "
                f"item skewness of {skew_target[j]:.2f}; that is outside "
                "Fleishman's feasible region, so the shape was reduced.")
            need_kurt = max(need_kurt, need_skew ** 2 - 2.0 + 0.05)
            if not fleishman.is_feasible(need_skew, need_kurt):
                continue
        try:
            coefficients = fleishman.solve(need_skew, need_kurt)
        except SemSimError:
            continue
        z[:, j] = fleishman.transform(z[:, j], coefficients)
        column = z[:, j]
        spread = column.std(ddof=1)
        if spread > 1e-9:
            z[:, j] = (column - column.mean()) / spread
    return z * sd


def _factor_score_estimates(cm: CompiledModel, y_star: np.ndarray
                            ) -> Optional[pd.DataFrame]:
    """Thurstone regression factor scores, for the covariance engine.

    The covariance engine never draws the latents, so there are no true scores
    to export.  These are estimates and are labelled as such.
    """
    try:
        sigma = cm.Lambda @ cm.Phi @ cm.Lambda.T + cm.Theta
        weights = np.linalg.solve(sigma, cm.Lambda @ cm.Phi)
        scores = y_star @ weights
    except np.linalg.LinAlgError:
        return None
    return pd.DataFrame(scores, columns=[f"{name}_hat"
                                         for name in cm.factor_names])


# --------------------------------------------------------------------------- #
# Scales
# --------------------------------------------------------------------------- #


def _apply_scales(spec: ModelSpec, cm: CompiledModel, y_star: np.ndarray,
                  warnings: List[str]) -> np.ndarray:
    """Turn the continuous responses into the scores a respondent would give."""
    from semsim.generation import thresholds as th
    n, p = y_star.shape
    covariates = spec.covariate_item_ids()
    items = [item for item in spec.all_items() if item.id not in covariates]
    any_ordinal = any(item.scale.kind is ScaleKind.LIKERT for item in items)
    out = np.empty((n, p), dtype=int if any_ordinal else float)
    if any_ordinal:
        out = np.empty((n, p), dtype=float)

    for item in items:
        index = cm.item_index.get(item.id)
        if index is None:
            continue
        # No explicit flip here: the compiler already gave a reverse-worded
        # item a negative loading, so its continuous response already runs
        # the other way.  Its response style, though, describes its
        # construct: a satisfied sample that mostly agrees with the positive
        # items mostly DISAGREES with "I am unhappy".  So a style-shaped
        # reverse item takes the mirror image of the style's cut points.
        # Explicit cut points or proportions describe the raw answers and are
        # used as given (the population draw writes mirrored ones itself).
        scale = item.scale
        if (item.reverse_coded and scale.kind is ScaleKind.LIKERT
                and not scale.thresholds and not scale.category_proportions):
            tau = th.scale_thresholds(scale)
            scale = replace(scale, thresholds=[float(v) for v in -tau[::-1]])
        out[:, index] = th.apply_scale(y_star[:, index], scale)

    if any_ordinal and all(item.scale.kind is ScaleKind.LIKERT
                           for item in items):
        out = out.astype(int)
    return out


__all__ = ["GeneratedDataset", "generate_dataset"]

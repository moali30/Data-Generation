"""
semsim.core.spec
================
The canonical, serialisable description of a Structural Equation Model to be
simulated.  Every other module in the package consumes *this* contract and
nothing else, so it is deliberately free of numpy / Qt / IO dependencies.

Vocabulary
----------
Construct   A latent variable the researcher theorises about.  May be
            first-order (items load on it directly) or second-order
            (first-order *dimensions* load on it).
Dimension   A first-order factor nested inside a second-order construct.
Item        One observed indicator (a questionnaire question).
Path        A directed structural effect between two constructs.

Design notes
------------
*   The user-facing hierarchy is Construct -> Dimension -> Item.  The
    numerical core flattens this into a single latent vector ``eta`` with one
    unified regression matrix ``B`` (see :mod:`semsim.core.compile`), where
    second-order loadings become ordinary latent regressions.  Keeping the two
    representations apart means the UI can stay hierarchical while the maths
    stays uniform.
*   All loadings / path coefficients are **completely standardised** values.
    The compiler derives the implied residual variances so every latent and
    every item has unit total variance.  This is what makes "set this path to
    0.30" mean what a researcher expects it to mean.
*   Nothing here performs validation beyond cheap structural invariants;
    semantic checks (identification, cycles, variance bounds) live in
    :mod:`semsim.core.validation` so they can produce rich bilingual
    diagnostics.
"""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SPEC_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class ConstructRole(str, Enum):
    """The job a construct performs inside the structural model."""

    INDEPENDENT = "independent"      # exogenous predictor
    DEPENDENT = "dependent"          # endogenous outcome
    MEDIATOR = "mediator"            # endogenous and a predictor
    MODERATOR = "moderator"          # participates in an interaction term
    CONTROL = "control"              # exogenous covariate

    @property
    def is_exogenous_by_default(self) -> bool:
        return self in (ConstructRole.INDEPENDENT, ConstructRole.CONTROL,
                        ConstructRole.MODERATOR)


class ConstructOrder(str, Enum):
    FIRST = "first"    # items attach directly to the construct
    SECOND = "second"  # dimensions attach to the construct, items to dimensions


class ScaleKind(str, Enum):
    CONTINUOUS = "continuous"
    LIKERT = "likert"


class ResponseStyle(str, Enum):
    """Shape of the categorical response distribution for a Likert item."""

    SYMMETRIC = "symmetric"            # bell-shaped, centred
    UNIFORM = "uniform"                # equal category probabilities
    POSITIVE_SKEW = "positive_skew"    # mass on low categories (floor effect)
    NEGATIVE_SKEW = "negative_skew"    # mass on high categories (ceiling)
    EXTREME_POSITIVE = "extreme_positive"
    EXTREME_NEGATIVE = "extreme_negative"
    BIMODAL = "bimodal"                # polarised opinions
    CUSTOM = "custom"                  # explicit proportions / thresholds


class EffectSize(str, Enum):
    """Conventional standardised effect magnitudes (Cohen, 1988; Keith, 2015)."""

    NONE = "none"
    TRIVIAL = "trivial"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    CUSTOM = "custom"

    @property
    def default_coefficient(self) -> float:
        return {
            EffectSize.NONE: 0.00,
            EffectSize.TRIVIAL: 0.03,
            EffectSize.WEAK: 0.15,
            EffectSize.MEDIUM: 0.30,
            EffectSize.STRONG: 0.50,
            EffectSize.VERY_STRONG: 0.70,
            EffectSize.CUSTOM: 0.30,
        }[self]


class FitLevel(str, Enum):
    """Target bands for global fit, following Hu & Bentler (1999) and
    Browne & Cudeck (1993)."""

    PERFECT = "perfect"        # saturated / no injected model error
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"              # for demonstrating what misfit looks like
    CUSTOM = "custom"

    @property
    def target_rmsea(self) -> float:
        return {
            FitLevel.PERFECT: 0.000,
            FitLevel.EXCELLENT: 0.020,
            FitLevel.GOOD: 0.045,
            FitLevel.ACCEPTABLE: 0.070,
            FitLevel.MARGINAL: 0.090,
            FitLevel.POOR: 0.120,
            FitLevel.CUSTOM: 0.045,
        }[self]

    @property
    def expected_cfi_floor(self) -> float:
        return {
            FitLevel.PERFECT: 1.000,
            FitLevel.EXCELLENT: 0.980,
            FitLevel.GOOD: 0.950,
            FitLevel.ACCEPTABLE: 0.900,
            FitLevel.MARGINAL: 0.870,
            FitLevel.POOR: 0.800,
            FitLevel.CUSTOM: 0.950,
        }[self]


class GenerationEngine(str, Enum):
    AUTO = "auto"
    COVARIANCE = "covariance"   # Vale-Maurelli from the implied matrix
    SCORES = "scores"           # simulate latent scores through the equations


class EstimatorKind(str, Enum):
    ML = "ml"            # maximum likelihood on the covariance matrix
    MLR = "mlr"          # ML with Satorra-Bentler robust corrections
    ULS = "uls"          # unweighted least squares
    DWLS = "dwls"        # diagonally weighted LS on polychoric correlations


class MissingMechanism(str, Enum):
    NONE = "none"
    MCAR = "mcar"
    MAR = "mar"


class DemographicKind(str, Enum):
    """Measurement level of a demographic variable."""

    NOMINAL = "nominal"
    ORDINAL = "ordinal"


class DemographicRole(str, Enum):
    """What the researcher means to do with a demographic variable."""

    DESCRIPTIVE = "descriptive"
    CONTROL = "control"
    COMPARISON = "comparison"
    MGA = "mga"


class GroupEffect(str, Enum):
    """Conventional sizes of a difference between groups (Cohen, 1988).

    Two groups are sized by Cohen's d and three or more by eta squared, the
    measures the comparison table reports for each.
    """

    NONE = "none"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    CUSTOM = "custom"

    def default_value(self, n_groups: int) -> float:
        """The size's conventional value: d for two groups, eta squared otherwise."""
        if n_groups <= 2:
            return {GroupEffect.NONE: 0.0, GroupEffect.SMALL: 0.20,
                    GroupEffect.MEDIUM: 0.50, GroupEffect.LARGE: 0.80,
                    GroupEffect.CUSTOM: 0.30}[self]
        return {GroupEffect.NONE: 0.0, GroupEffect.SMALL: 0.01,
                GroupEffect.MEDIUM: 0.06, GroupEffect.LARGE: 0.14,
                GroupEffect.CUSTOM: 0.04}[self]


class GroupDirection(str, Enum):
    """Which end of a comparison variable's category list scores higher."""

    ASCENDING = "ascending"     # the last category scores highest
    DESCENDING = "descending"   # the first category scores highest


#: Largest d and eta squared a group difference may ask for.
GROUP_D_MAX = 2.0
GROUP_ETA_MAX = 0.40


class AssociationStrength(str, Enum):
    """Conventional strengths of an association, as Cramér's V (Cohen, 1988)."""

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CUSTOM = "custom"

    @property
    def default_value(self) -> float:
        return {AssociationStrength.NONE: 0.0, AssociationStrength.WEAK: 0.10,
                AssociationStrength.MODERATE: 0.30,
                AssociationStrength.STRONG: 0.50,
                AssociationStrength.CUSTOM: 0.30}[self]


class AssociationDirection(str, Enum):
    """How the category lists of two associated variables line up."""

    POSITIVE = "positive"   # the last categories of both go together
    NEGATIVE = "negative"   # the last of the first goes with the first of the second


#: Largest Cramér's V an association may ask for.
ASSOCIATION_V_MAX = 0.90


class PowerMethod(str, Enum):
    RMSEA_CLOSE_FIT = "rmsea_close_fit"   # MacCallum, Browne & Sugawara (1996)
    SATORRA_SARIS = "satorra_saris"       # power for a single parameter
    MONTE_CARLO = "monte_carlo"           # empirical, Muthen & Muthen (2002)
    RULE_OF_THUMB = "rule_of_thumb"       # N:q and N:item heuristics


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Bilingual:
    """A label carried in both interface languages."""

    ar: str = ""
    en: str = ""

    def get(self, lang: str = "en") -> str:
        primary = self.ar if lang == "ar" else self.en
        return primary or self.en or self.ar

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.ar or self.en)


# --------------------------------------------------------------------------- #
# Measurement layer
# --------------------------------------------------------------------------- #


@dataclass
class ScaleSpec:
    """How one item's continuous latent response is turned into a score."""

    kind: ScaleKind = ScaleKind.LIKERT
    n_categories: int = 5
    response_style: ResponseStyle = ResponseStyle.SYMMETRIC
    #: Explicit z-thresholds, length ``n_categories - 1``.  When supplied these
    #: override ``response_style`` and ``category_proportions``.
    thresholds: Optional[List[float]] = None
    #: Target marginal proportions, length ``n_categories``; converted to
    #: thresholds through the inverse normal CDF of the cumulative sums.
    category_proportions: Optional[List[float]] = None
    #: Reported metric for continuous items (and the labels shown for Likert).
    mean: float = 0.0
    sd: float = 1.0
    min_label: Bilingual = field(default_factory=Bilingual)
    max_label: Bilingual = field(default_factory=Bilingual)

    @property
    def is_ordinal(self) -> bool:
        return self.kind is ScaleKind.LIKERT

    @property
    def n_thresholds(self) -> int:
        return max(self.n_categories - 1, 0) if self.is_ordinal else 0


@dataclass
class Item:
    """One observed indicator."""

    id: str = field(default_factory=lambda: _new_id("it"))
    name: str = ""                       # column name in the exported dataset
    label: Bilingual = field(default_factory=Bilingual)
    #: Completely standardised primary loading on the item's home factor.
    loading: float = 0.75
    #: Marginal non-normality targets for the *observed* item.  Excess kurtosis
    #: (normal == 0).  Ignored for ordinal items whose shape comes from the
    #: thresholds instead.
    skewness: float = 0.0
    kurtosis: float = 0.0
    scale: ScaleSpec = field(default_factory=ScaleSpec)
    #: Reverse-worded item: the exported score is flipped on the response scale.
    reverse_coded: bool = False
    #: Extra unique variance beyond ``1 - loading**2`` (unreliability noise).
    extra_error_variance: float = 0.0
    notes: str = ""

    def clone(self, **overrides: Any) -> "Item":
        data = {**asdict_shallow(self), **overrides}
        data.pop("id", None)
        return Item(id=_new_id("it"), **data)


@dataclass
class Dimension:
    """A first-order factor nested inside a second-order construct."""

    id: str = field(default_factory=lambda: _new_id("dim"))
    name: str = ""
    label: Bilingual = field(default_factory=Bilingual)
    items: List[Item] = field(default_factory=list)
    #: Completely standardised loading of this dimension on its parent
    #: construct.  Only meaningful for second-order constructs.
    second_order_loading: float = 0.75

    @property
    def n_items(self) -> int:
        return len(self.items)


@dataclass
class Construct:
    """A latent variable in the researcher's conceptual model."""

    id: str = field(default_factory=lambda: _new_id("c"))
    name: str = ""
    label: Bilingual = field(default_factory=Bilingual)
    role: ConstructRole = ConstructRole.INDEPENDENT
    order: ConstructOrder = ConstructOrder.FIRST
    #: Used when ``order is FIRST``.
    items: List[Item] = field(default_factory=list)
    #: Used when ``order is SECOND``.
    dimensions: List[Dimension] = field(default_factory=list)
    notes: str = ""
    #: Stands for an observed control term; set only by ``ModelSpec.expanded``.
    covariate: bool = False

    # -- convenience ------------------------------------------------------- #

    @property
    def is_second_order(self) -> bool:
        return self.order is ConstructOrder.SECOND

    def all_items(self) -> List[Item]:
        if self.is_second_order:
            return [it for d in self.dimensions for it in d.items]
        return list(self.items)

    @property
    def n_items(self) -> int:
        return len(self.all_items())

    def first_order_factor_ids(self) -> List[str]:
        """Ids of the latent factors that items attach to for this construct."""
        if self.is_second_order:
            return [d.id for d in self.dimensions]
        return [self.id]


# --------------------------------------------------------------------------- #
# Complex-reality layer
# --------------------------------------------------------------------------- #


@dataclass
class CrossLoading:
    """A secondary loading of one item onto a factor that is not its home."""

    item_id: str = ""
    factor_id: str = ""     # construct id or dimension id
    loading: float = 0.20


@dataclass
class CrossLoadingPolicy:
    """Rules for auto-generating cross-loadings to mimic fuzzy item wording."""

    enabled: bool = False
    #: Share of items that receive at least one cross-loading.
    proportion_of_items: float = 0.20
    #: Absolute magnitude drawn uniformly from this closed interval.
    magnitude_range: Tuple[float, float] = (0.10, 0.25)
    #: Only cross-load onto factors that belong to a *different* construct.
    across_constructs_only: bool = True
    #: Allow negative cross-loadings (opposite-signed contamination).
    allow_negative: bool = False
    #: Cap on how many extra factors a single item may load on.
    max_per_item: int = 1


@dataclass
class ErrorCovariance:
    """A correlated residual between two items."""

    item_a: str = ""
    item_b: str = ""
    #: Residual *correlation* in the standardised metric, converted to a
    #: covariance by the compiler using each item's unique variance.
    correlation: float = 0.20
    reason: Bilingual = field(default_factory=Bilingual)


@dataclass
class CommonMethodBias:
    """An unmeasured latent method factor loading on every (or some) item."""

    enabled: bool = False
    #: Completely standardised loading of the method factor on affected items.
    loading: float = 0.30
    #: Restrict to these item ids; empty means every item.
    item_ids: List[str] = field(default_factory=list)
    label: Bilingual = field(default_factory=lambda: Bilingual(
        ar="عامل طريقة القياس المشترك", en="Common method factor"))


# --------------------------------------------------------------------------- #
# Structural layer
# --------------------------------------------------------------------------- #


@dataclass
class Path:
    """A directed structural effect between two constructs."""

    id: str = field(default_factory=lambda: _new_id("p"))
    source: str = ""        # construct id
    target: str = ""        # construct id
    effect_size: EffectSize = EffectSize.MEDIUM
    #: Completely standardised coefficient.  Authoritative when
    #: ``effect_size is CUSTOM``; otherwise refreshed from the enum default
    #: unless the user has edited it (tracked by ``coefficient_is_manual``).
    coefficient: float = 0.30
    coefficient_is_manual: bool = False
    #: Whether the simulation should aim for p < alpha on this path.
    target_significant: bool = True
    label: str = ""

    def resolved_coefficient(self) -> float:
        if self.effect_size is EffectSize.CUSTOM or self.coefficient_is_manual:
            return self.coefficient
        return self.effect_size.default_coefficient


@dataclass
class LatentCovariance:
    """A free covariance between two exogenous constructs."""

    construct_a: str = ""
    construct_b: str = ""
    correlation: float = 0.30


@dataclass
class Interaction:
    """A latent moderation term: ``target ~ ... + b * predictor:moderator``."""

    id: str = field(default_factory=lambda: _new_id("ix"))
    predictor: str = ""
    moderator: str = ""
    target: str = ""
    effect_size: EffectSize = EffectSize.WEAK
    coefficient: float = 0.15
    coefficient_is_manual: bool = False
    target_significant: bool = True

    def resolved_coefficient(self) -> float:
        if self.effect_size is EffectSize.CUSTOM or self.coefficient_is_manual:
            return self.coefficient
        return self.effect_size.default_coefficient


@dataclass
class IndirectEffect:
    """A mediation chain the researcher wants reported (and optionally
    targeted for significance)."""

    id: str = field(default_factory=lambda: _new_id("ind"))
    #: Ordered construct ids, e.g. ``["X", "M", "Y"]``.
    chain: List[str] = field(default_factory=list)
    target_significant: bool = True
    label: str = ""


# --------------------------------------------------------------------------- #
# Realism / noise layer
# --------------------------------------------------------------------------- #


@dataclass
class NoiseSpec:
    """Deliberate imperfections that keep simulated data from looking synthetic."""

    enabled: bool = True
    #: Multiplicative inflation of every unique variance (1.0 == untouched).
    #: Values above 1 attenuate reliability the way real measurement does.
    unique_variance_inflation: float = 1.0
    #: Share of respondents who answer without reading (flat or random lines).
    careless_proportion: float = 0.0
    careless_style: str = "random"          # "random" | "straightline" | "mixed"
    #: Share of cells replaced by multivariate outliers.
    outlier_proportion: float = 0.0
    outlier_magnitude_sd: float = 3.5
    #: Missing data.
    missing_mechanism: MissingMechanism = MissingMechanism.NONE
    missing_proportion: float = 0.0
    #: Construct whose score drives MAR missingness (empty == first exogenous).
    mar_driver: str = ""
    #: Jitter added to every item before categorisation, in SD units.  Small
    #: values break the perfect-fit artefact without disturbing the model.
    measurement_jitter_sd: float = 0.0


#: The named amounts of real-world variation, as ``(construct_level_sd,
#: item_level_sd, item_spread_sd, loading_sd, path_sd, tail_share)``.
#: ``realistic`` is what a published questionnaire study typically shows:
#: construct means a few tenths of a point apart on a five-point scale, item
#: means within a construct a tenth or two apart, loadings and paths a few
#: hundredths away from the values the researcher planned, and a few answers
#: at the far end of even a strongly skewed item.
HETEROGENEITY_PRESETS: Dict[str, Tuple[float, ...]] = {
    "mild": (0.15, 0.08, 0.05, 0.03, 0.015, 0.025),
    "realistic": (0.30, 0.15, 0.10, 0.05, 0.03, 0.05),
    "strong": (0.45, 0.25, 0.15, 0.08, 0.05, 0.08),
}


@dataclass
class HeterogeneitySpec:
    """How far the simulated population departs from the textbook model.

    A specification gives every item of a construct the same loading and the
    same response style, and every path a round number.  Data drawn straight
    from that has every item mean at the same value, identical standard
    deviations and identical loadings, which no real questionnaire shows.
    These settings draw one population around the specification instead; see
    :mod:`semsim.generation.heterogeneity`.  The first five values are
    standard deviations: three on the z-scale of the latent response, two on
    the standardised coefficient.  ``tail_share`` is a proportion.
    """

    enabled: bool = True
    #: Constructs differ in their average level (one construct sits higher on
    #: the scale than another).
    construct_level_sd: float = 0.30
    #: Items differ in how easy they are to agree with, around their
    #: construct's level.
    item_level_sd: float = 0.15
    #: Items differ in how spread out their answers are (log of the stretch
    #: applied to the cut points).
    item_spread_sd: float = 0.10
    #: Population loadings scatter around the planned loading.
    loading_sd: float = 0.05
    #: Population paths and correlations scatter around the planned values.
    path_sd: float = 0.03
    #: Share of each item's answers spread evenly over all categories, so even
    #: a strongly skewed item has a few answers at the far end of the scale,
    #: as every real sample does.
    tail_share: float = 0.05

    def apply_preset(self, name: str) -> None:
        """Set the amounts from a named preset, or switch off for ``none``."""
        if name == "none":
            self.enabled = False
            return
        values = HETEROGENEITY_PRESETS.get(name)
        if values is None:
            return
        self.enabled = True
        (self.construct_level_sd, self.item_level_sd, self.item_spread_sd,
         self.loading_sd, self.path_sd, self.tail_share) = values

    def preset_name(self) -> str:
        """The preset these settings match, ``none``, or ``custom``."""
        if not self.enabled:
            return "none"
        current = (self.construct_level_sd, self.item_level_sd,
                   self.item_spread_sd, self.loading_sd, self.path_sd,
                   self.tail_share)
        for name, values in HETEROGENEITY_PRESETS.items():
            if all(abs(a - b) < 1e-9 for a, b in zip(current, values)):
                return name
        return "custom"


@dataclass
class DistributionSpec:
    """Global non-normality controls; per-item values override these."""

    #: Applied to every continuous item that leaves skewness/kurtosis at 0.
    default_skewness: float = 0.0
    default_kurtosis: float = 0.0
    #: Vale-Maurelli needs the intermediate correlation solve; disable to fall
    #: back on plain multivariate normal (much faster, normal marginals).
    use_non_normal_engine: bool = True
    #: Hard bound enforced by validation: Fleishman solutions exist only in a
    #: region roughly described by ``kurtosis > skew**2 - 2``.
    clamp_to_feasible_region: bool = True


# --------------------------------------------------------------------------- #
# Targeting layer
# --------------------------------------------------------------------------- #


@dataclass
class FitTargetSpec:
    """How much model error to inject so the fitted indices land where the
    researcher wants them."""

    enabled: bool = True
    level: FitLevel = FitLevel.GOOD
    #: Authoritative when ``level is CUSTOM``.
    rmsea: float = 0.045
    #: Optional secondary targets used by the calibration loop as soft goals.
    cfi: Optional[float] = None
    srmr: Optional[float] = None
    #: Cudeck & Browne (1992) keeps the specified parameters as the exact
    #: minimiser; "residual" simply perturbs the implied matrix (faster, less
    #: principled).
    method: str = "cudeck_browne"          # "cudeck_browne" | "residual" | "none"
    #: Maximum calibration passes when matching the empirical indices.
    max_calibration_rounds: int = 6
    #: Acceptable absolute deviation of the realised RMSEA from the target.
    tolerance: float = 0.006


@dataclass
class SignificanceTargetSpec:
    """Global policy for making targeted paths reach (or miss) significance."""

    enabled: bool = True
    alpha: float = 0.05
    #: Desired probability that a path flagged ``target_significant`` really is
    #: significant in the generated sample.
    target_power: float = 0.80
    #: Coefficient assigned to paths that must stay non-significant.
    null_coefficient: float = 0.02
    #: Re-draw the dataset until every targeted path matches its flag, up to
    #: this many attempts.  0 disables rejection sampling.  Each attempt costs
    #: a full re-fit, which on a large model is tens of seconds, so the loop
    #: also gives up early on a path the sample cannot detect at all.
    max_resamples: int = 8
    #: When rejection sampling, also require non-targeted paths to stay null.
    enforce_null_paths: bool = True


@dataclass
class PowerSpec:
    """Sample-size determination."""

    mode: str = "suggest"              # "suggest" | "fixed"
    n: int = 300
    target_power: float = 0.80
    alpha: float = 0.05
    method: PowerMethod = PowerMethod.RMSEA_CLOSE_FIT
    #: Null / alternative RMSEA for the test of close fit.
    rmsea_null: float = 0.05
    rmsea_alt: float = 0.08
    #: Monte-Carlo settings.
    mc_replications: int = 200
    #: Guard rails applied to any suggestion.
    min_n: int = 50
    max_n: int = 5000
    #: Heuristic ratios reported alongside the formal analysis.
    ratio_per_parameter: int = 10
    ratio_per_item: int = 10


# --------------------------------------------------------------------------- #
# Analysis / output layer
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisSpec:
    """What to estimate and report once data exist."""

    estimator: EstimatorKind = EstimatorKind.ML
    #: Fit the measurement model on polychoric correlations for ordinal data.
    use_polychoric_for_ordinal: bool = True
    report_reliability: bool = True
    report_validity: bool = True
    report_htmt: bool = True
    report_fit: bool = True
    report_descriptives: bool = True
    report_normality_tests: bool = True
    #: Bootstrap resamples for indirect-effect confidence intervals (0 = skip).
    bootstrap_samples: int = 0
    htmt_threshold: float = 0.85


@dataclass
class ExportSpec:
    formats: List[str] = field(default_factory=lambda: ["csv"])
    include_latent_scores: bool = False
    include_continuous_shadow: bool = False   # pre-categorisation y*
    #: Emit ready-to-run syntax for other packages.
    emit_lavaan_syntax: bool = True
    emit_mplus_syntax: bool = False
    emit_amos_hint: bool = False
    decimal_places: int = 4


# --------------------------------------------------------------------------- #
# Demographic and control variables
# --------------------------------------------------------------------------- #


@dataclass
class DemographicCategory:
    """One answer option of a demographic variable and its share of the sample."""

    id: str = field(default_factory=lambda: _new_id("cat"))
    label: Bilingual = field(default_factory=Bilingual)
    #: Share of the sample, between zero and one.
    proportion: float = 0.5


@dataclass
class ControlPath:
    """The effect of one control term on one endogenous construct."""

    id: str = field(default_factory=lambda: _new_id("cp"))
    #: :attr:`DemographicTerm.key` of the term this effect starts from.
    term: str = ""
    target: str = ""        # construct id
    effect_size: EffectSize = EffectSize.WEAK
    coefficient: float = 0.15
    coefficient_is_manual: bool = False
    target_significant: bool = True

    def resolved_coefficient(self) -> float:
        if self.effect_size is EffectSize.CUSTOM or self.coefficient_is_manual:
            return self.coefficient
        return self.effect_size.default_coefficient


@dataclass
class GroupDifference:
    """How the groups of one comparison variable differ on one construct.

    The size is Cohen's d when the variable has two categories and eta
    squared when it has three or more.  The groups are ordered along the
    category list, so with three or more the means rise (or fall) from the
    first category to the last.
    """

    id: str = field(default_factory=lambda: _new_id("gd"))
    target: str = ""        # construct id
    size: GroupEffect = GroupEffect.SMALL
    #: Read only when ``size`` is :attr:`GroupEffect.CUSTOM`.
    value: float = 0.30
    direction: GroupDirection = GroupDirection.ASCENDING
    target_significant: bool = True

    def resolved_value(self, n_groups: int) -> float:
        """The requested d (two groups) or eta squared (three or more)."""
        raw = (float(self.value) if self.size is GroupEffect.CUSTOM
               else self.size.default_value(n_groups))
        ceiling = GROUP_D_MAX if n_groups <= 2 else GROUP_ETA_MAX
        return min(max(raw, 0.0), ceiling)

    def eta_squared(self, shares: Sequence[float]) -> float:
        """The requested difference as a share of the construct's variance.

        Parameters
        ----------
        shares : sequence of float
            The groups' shares of the sample, summing to one.

        References
        ----------
        Cohen, J. (1988). *Statistical Power Analysis for the Behavioral
            Sciences* (2nd ed.), chapter 2. Erlbaum.
        """
        value = self.resolved_value(len(shares))
        if len(shares) > 2:
            return value
        balance = float(shares[0]) * float(shares[1])
        return value * value * balance / (1.0 + value * value * balance)

    @property
    def wants_significance(self) -> bool:
        """Whether the sample should show the difference as significant."""
        return self.target_significant and self.size is not GroupEffect.NONE


@dataclass
class DemographicAssociation:
    """Two demographic variables that go together, such as age and experience.

    The strength is Cramér's V of their cross-table.  The categories are
    matched in list order: with a positive direction the last categories of
    the two variables go together, with a negative one the last category of
    the first goes with the first category of the second.
    """

    id: str = field(default_factory=lambda: _new_id("da"))
    first: str = ""         # demographic variable id
    second: str = ""        # demographic variable id
    strength: AssociationStrength = AssociationStrength.MODERATE
    #: Read only when ``strength`` is :attr:`AssociationStrength.CUSTOM`.
    value: float = 0.30
    direction: AssociationDirection = AssociationDirection.POSITIVE
    target_significant: bool = True

    def resolved_value(self) -> float:
        """The requested Cramér's V."""
        raw = (float(self.value) if self.strength is AssociationStrength.CUSTOM
               else self.strength.default_value)
        return min(max(raw, 0.0), ASSOCIATION_V_MAX)

    @property
    def wants_significance(self) -> bool:
        """Whether the chi-square test should come out significant."""
        return (self.target_significant
                and self.strength is not AssociationStrength.NONE)


@dataclass(frozen=True)
class DemographicTerm:
    """One column a demographic variable contributes to a regression.

    A binary or ordinal variable has one term, its own codes.  A nominal
    variable with ``k >= 3`` categories has ``k - 1`` dummy terms, each
    comparing one category with the first.
    """

    key: str
    column: str
    #: Position of the category a dummy term flags; ``None`` for the raw codes.
    category: Optional[int] = None


@dataclass
class DemographicVariable:
    """A respondent characteristic: age, gender, education, job title."""

    id: str = field(default_factory=lambda: _new_id("dv"))
    name: str = ""                       # column name in the exported dataset
    label: Bilingual = field(default_factory=Bilingual)
    kind: DemographicKind = DemographicKind.NOMINAL
    role: DemographicRole = DemographicRole.DESCRIPTIVE
    categories: List[DemographicCategory] = field(default_factory=list)
    #: Used only when ``role is CONTROL``.
    control_paths: List[ControlPath] = field(default_factory=list)
    #: Used only when ``role is COMPARISON``.
    group_differences: List[GroupDifference] = field(default_factory=list)

    @property
    def is_dummy_coded(self) -> bool:
        """Whether the variable enters a regression as dummy columns."""
        return self.kind is DemographicKind.NOMINAL and len(self.categories) >= 3

    def terms(self) -> List[DemographicTerm]:
        """The regression columns this variable provides, in column order."""
        if not self.is_dummy_coded:
            return [DemographicTerm(key="", column=self.name)]
        return [DemographicTerm(key=cat.id,
                                column=f"{self.name}_{position + 1}",
                                category=position)
                for position, cat in enumerate(self.categories) if position]

    def normalised_proportions(self) -> List[float]:
        """Category shares rescaled to sum to one exactly."""
        raw = [max(float(c.proportion), 0.0) for c in self.categories]
        total = sum(raw)
        if total <= 0.0:
            return [1.0 / len(raw)] * len(raw) if raw else []
        return [value / total for value in raw]

    def reconcile_control_paths(self) -> None:
        """Repoint control paths whose term stopped existing.

        Changing a variable's categories or level can change its terms, and a
        path left pointing at a term that has gone would generate nothing.
        """
        valid = [term.key for term in self.terms()]
        for path in self.control_paths:
            if path.term not in valid and valid:
                path.term = valid[0]


# --------------------------------------------------------------------------- #
# The whole project
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    """A complete, self-contained simulation request."""

    # -- identity ---------------------------------------------------------- #
    spec_version: str = SPEC_VERSION
    title: Bilingual = field(default_factory=lambda: Bilingual(
        ar="نموذج محاكاة جديد", en="New simulation model"))
    description: str = ""
    author: str = ""
    created_at: str = ""
    seed: Optional[int] = 20240101

    # -- measurement ------------------------------------------------------- #
    constructs: List[Construct] = field(default_factory=list)

    # -- complex realities ------------------------------------------------- #
    cross_loadings: List[CrossLoading] = field(default_factory=list)
    cross_loading_policy: CrossLoadingPolicy = field(default_factory=CrossLoadingPolicy)
    error_covariances: List[ErrorCovariance] = field(default_factory=list)
    common_method_bias: CommonMethodBias = field(default_factory=CommonMethodBias)

    # -- structure --------------------------------------------------------- #
    paths: List[Path] = field(default_factory=list)
    latent_covariances: List[LatentCovariance] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    indirect_effects: List[IndirectEffect] = field(default_factory=list)

    # -- generation -------------------------------------------------------- #
    distribution: DistributionSpec = field(default_factory=DistributionSpec)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    heterogeneity: HeterogeneitySpec = field(default_factory=HeterogeneitySpec)
    engine: GenerationEngine = GenerationEngine.AUTO

    # -- targeting --------------------------------------------------------- #
    power: PowerSpec = field(default_factory=PowerSpec)
    fit_target: FitTargetSpec = field(default_factory=FitTargetSpec)
    significance_target: SignificanceTargetSpec = field(
        default_factory=SignificanceTargetSpec)

    # -- output ------------------------------------------------------------ #
    analysis: AnalysisSpec = field(default_factory=AnalysisSpec)
    export: ExportSpec = field(default_factory=ExportSpec)

    # -- demographics ------------------------------------------------------ #
    demographics: List[DemographicVariable] = field(default_factory=list)
    #: ``True`` fills each category's quota exactly; ``False`` draws every
    #: respondent independently with the category shares as weights, so the
    #: achieved counts vary the way a collected sample's do.
    quota_exact: bool = False
    #: Pairs of demographic variables that go together.
    demographic_associations: List[DemographicAssociation] = field(
        default_factory=list)

    # ------------------------------------------------------------------ #
    # Demographics and the model they enter
    # ------------------------------------------------------------------ #

    def demographic(self, vid: str) -> Optional[DemographicVariable]:
        for variable in self.demographics:
            if variable.id == vid:
                return variable
        return None

    def active_control_terms(self) -> List[Tuple[DemographicVariable,
                                                 DemographicTerm]]:
        """Control terms that have at least one path into the model."""
        out: List[Tuple[DemographicVariable, DemographicTerm]] = []
        for variable in self.demographics:
            if variable.role is not DemographicRole.CONTROL:
                continue
            used = {path.term for path in variable.control_paths}
            out.extend((variable, term) for term in variable.terms()
                       if term.key in used)
        return out

    def demographic_columns(self) -> List[str]:
        """Every demographic column of a generated dataset, in order."""
        out: List[str] = []
        for variable in self.demographics:
            out.append(variable.name)
            if variable.role is DemographicRole.CONTROL and variable.is_dummy_coded:
                out.extend(term.column for term in variable.terms())
        return out

    def control_target_ids(self) -> List[str]:
        """Constructs a control may point at: the dependent and mediating ones."""
        receiving = set(self.endogenous_construct_ids())
        return [c.id for c in self.constructs if not c.covariate
                and (c.id in receiving
                     or c.role in (ConstructRole.DEPENDENT, ConstructRole.MEDIATOR))]

    def control_columns(self) -> List[str]:
        """Columns the fitted model needs beyond the questionnaire items."""
        return [term.column for _variable, term in self.active_control_terms()]

    def covariate_item_ids(self) -> set:
        """Ids of the indicators that stand for control terms."""
        return {item.id for construct in self.constructs if construct.covariate
                for item in construct.items}

    def expanded(self) -> "ModelSpec":
        """This model with its control terms written in as model variables.

        Each active control term becomes an exogenous variable measured
        without error by a single indicator, which is how a SEM treats an
        observed covariate.  The term's paths become ordinary structural
        paths, so the compiler, the estimator, significance targeting and the
        degrees of freedom all handle them without a special case, and adding
        a control leaves the standardised substantive paths where they were:
        the target's disturbance variance gives up the variance the control
        explains.

        Control terms may covary freely with each other and with the
        exogenous constructs, as covariates do when a model is fitted.  Their
        population correlation with the constructs is zero, because quota
        assignment is independent of the answers.  Dummies of one variable
        correlate negatively by construction, and that value is written in.

        The expansion is idempotent, and a model with no active control is
        returned as is, so it compiles and generates exactly as before.

        Returns
        -------
        ModelSpec
            ``self`` when there is nothing to expand, otherwise a deep copy.

        References
        ----------
        Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
            chapter 7. Wiley. (Single indicators with fixed error variance.)
        Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied
            Multiple Regression/Correlation Analysis* (3rd ed.), chapter 8.
            Erlbaum. (Dummy coding and the correlations among dummies.)
        """
        if any(construct.covariate for construct in self.constructs):
            return self
        active = self.active_control_terms()
        if not active:
            return self

        out = copy.deepcopy(self)
        term_ids: List[str] = []
        shares: Dict[str, Tuple[str, float]] = {}
        sources: Dict[str, Tuple[DemographicVariable, DemographicTerm]] = {}
        for variable, term in active:
            suffix = term.key or "main"
            cid = f"__ctl_{variable.id}_{suffix}"
            term_ids.append(cid)
            sources[cid] = (variable, term)
            if term.category is not None:
                shares[cid] = (variable.id,
                               variable.normalised_proportions()[term.category])
            label = variable.label
            if term.category is not None:
                category = variable.categories[term.category].label
                label = Bilingual(ar=f"{variable.label.ar}: {category.ar}",
                                  en=f"{variable.label.en}: {category.en}")
            out.constructs.append(Construct(
                id=cid, name=term.column, label=label,
                role=ConstructRole.CONTROL, covariate=True,
                items=[Item(id=f"{cid}_i", name=term.column, loading=1.0,
                            scale=ScaleSpec(kind=ScaleKind.CONTINUOUS))]))
            for path in variable.control_paths:
                if path.term == term.key:
                    out.paths.append(Path(
                        id=path.id, source=cid, target=path.target,
                        effect_size=path.effect_size,
                        coefficient=path.coefficient,
                        coefficient_is_manual=path.coefficient_is_manual,
                        target_significant=path.target_significant))

        linked = None
        if self.demographic_associations:
            from semsim.core.associations import latent_structure, term_correlation
            linked = latent_structure(self)
        for a, first in enumerate(term_ids):
            for second in term_ids[a + 1:]:
                correlation = 0.0
                if first in shares and second in shares \
                        and shares[first][0] == shares[second][0]:
                    pa, pb = shares[first][1], shares[second][1]
                    denominator = max((1.0 - pa) * (1.0 - pb), 1e-12)
                    correlation = max(-math.sqrt(pa * pb / denominator), -0.999)
                elif linked is not None:
                    # Associated variables: their terms correlate as their
                    # cross-table says, which the data will show.
                    (va, ta), (vb, tb) = sources[first], sources[second]
                    i, j = linked.index(va.id), linked.index(vb.id)
                    if va.id != vb.id and i is not None and j is not None:
                        correlation = term_correlation(
                            va, ta, vb, tb, float(linked.matrix[i, j]))
                out.latent_covariances.append(LatentCovariance(
                    construct_a=first, construct_b=second,
                    correlation=correlation))
        covariates = set(term_ids)
        for exogenous in out.exogenous_construct_ids():
            if exogenous in covariates:
                continue
            for term_id in term_ids:
                out.latent_covariances.append(LatentCovariance(
                    construct_a=term_id, construct_b=exogenous,
                    correlation=0.0))
        return out

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def construct(self, cid: str) -> Optional[Construct]:
        for c in self.constructs:
            if c.id == cid:
                return c
        return None

    def construct_by_name(self, name: str) -> Optional[Construct]:
        for c in self.constructs:
            if c.name == name:
                return c
        return None

    def dimension(self, did: str) -> Optional[Dimension]:
        for c in self.constructs:
            for d in c.dimensions:
                if d.id == did:
                    return d
        return None

    def item(self, iid: str) -> Optional[Item]:
        for it in self.all_items():
            if it.id == iid:
                return it
        return None

    def all_items(self) -> List[Item]:
        return [it for c in self.constructs for it in c.all_items()]

    def all_item_names(self) -> List[str]:
        return [it.name for it in self.all_items()]

    def iter_items_with_owner(self) -> Iterator[Tuple[Item, Construct, Optional[Dimension]]]:
        """Yield ``(item, construct, dimension_or_None)`` for every item."""
        for c in self.constructs:
            if c.is_second_order:
                for d in c.dimensions:
                    for it in d.items:
                        yield it, c, d
            else:
                for it in c.items:
                    yield it, c, None

    def home_factor_of(self, item_id: str) -> Optional[str]:
        """Id of the factor an item loads on primarily."""
        for it, c, d in self.iter_items_with_owner():
            if it.id == item_id:
                return d.id if d is not None else c.id
        return None

    def all_factor_ids(self) -> List[str]:
        """Every latent node: constructs plus the dimensions inside them."""
        out: List[str] = []
        for c in self.constructs:
            out.append(c.id)
            out.extend(d.id for d in c.dimensions)
        return out

    def factor_label(self, fid: str, lang: str = "en") -> str:
        c = self.construct(fid)
        if c is not None:
            return c.label.get(lang) or c.name or fid
        d = self.dimension(fid)
        if d is not None:
            return d.label.get(lang) or d.name or fid
        return fid

    # ------------------------------------------------------------------ #
    # Derived counts
    # ------------------------------------------------------------------ #

    @property
    def n_items(self) -> int:
        return len(self.all_items())

    @property
    def n_constructs(self) -> int:
        return len(self.constructs)

    @property
    def n_factors(self) -> int:
        return len(self.all_factor_ids())

    def endogenous_construct_ids(self) -> List[str]:
        """Constructs that receive at least one structural path."""
        targets = {p.target for p in self.paths}
        targets |= {ix.target for ix in self.interactions}
        return [c.id for c in self.constructs if c.id in targets]

    def exogenous_construct_ids(self) -> List[str]:
        endo = set(self.endogenous_construct_ids())
        return [c.id for c in self.constructs if c.id not in endo]

    def incoming_paths(self, cid: str) -> List[Path]:
        return [p for p in self.paths if p.target == cid]

    def outgoing_paths(self, cid: str) -> List[Path]:
        return [p for p in self.paths if p.source == cid]

    def has_interactions(self) -> bool:
        return bool(self.interactions)

    def resolved_engine(self) -> GenerationEngine:
        """Which generator will actually run."""
        if self.engine is not GenerationEngine.AUTO:
            return self.engine
        # Latent products cannot be expressed in a covariance matrix, so any
        # moderation forces the score-based path.
        if self.has_interactions():
            return GenerationEngine.SCORES
        if self.noise.enabled and (
            self.noise.careless_proportion > 0
            or self.noise.outlier_proportion > 0
        ):
            return GenerationEngine.COVARIANCE  # contamination applied post-hoc
        return GenerationEngine.COVARIANCE

    # ------------------------------------------------------------------ #
    # Degrees of freedom (CFA + structural, ML on the covariance matrix)
    # ------------------------------------------------------------------ #

    def expected_auto_cross_loadings(self) -> int:
        """How many cross-loadings the auto policy will draw.

        The actual draw happens in the compiler with a seeded generator, so
        this is the count that policy implies, not a figure read back from a
        compiled model.  Pass a :class:`CompiledModel` to
        :meth:`count_free_parameters` when the exact number matters.
        """
        policy = self.cross_loading_policy
        n_items = self.n_items - len(self.covariate_item_ids())
        if not policy.enabled or n_items == 0:
            return 0
        affected = round(policy.proportion_of_items * n_items)
        return int(affected * max(policy.max_per_item, 1))

    def count_free_parameters(self, compiled: Any = None) -> int:
        """Free parameters in the model as it would be *fitted*.

        Counted under the completely standardised identification this spec
        works in: every latent has unit variance by construction, so an
        exogenous latent's variance is a scaling constraint and an endogenous
        latent's disturbance variance is derived from its predictors
        (``1 - explained``).  Neither is estimated, so neither is counted --
        adding one free disturbance per endogenous construct on top of its
        free loadings double-counts and understates the degrees of freedom.
        The marker-variable parameterisation an estimator uses reaches the
        same total: it fixes one loading per factor and frees that factor's
        variance instead.

        A second-order loading is a *regression of a dimension on its
        construct*, so it lives in ``B`` alongside the structural paths, not
        in ``Lambda``.  Latent interactions are excluded: a product of two
        latent variables cannot be expressed in a covariance matrix, so no
        covariance-structure estimator fits one and it contributes no
        parameter to the fitted model's degrees of freedom.

        Parameters
        ----------
        compiled
            An optional :class:`~semsim.core.compile.CompiledModel`.  When
            given, the auto-generated cross-loadings it actually drew are
            counted instead of the number the policy implies.
        """
        full = self.expanded()
        if full is not self:
            return full.count_free_parameters(compiled)
        # Count what is ESTIMATED, not what happens to be non-zero. A path
        # whose population coefficient is exactly zero -- an effect the
        # researcher wants to come out non-significant -- is still a free
        # parameter, so reading the count off the non-zero entries of B would
        # drop it silently and overstate the degrees of freedom.
        n_auto = (len(getattr(compiled, "auto_cross_loadings", ()))
                  if compiled is not None
                  else self.expected_auto_cross_loadings())
        n_load = self._declared_item_loadings() + n_auto
        n_struct = (sum(len(c.dimensions) for c in self.constructs)
                    + len(self.paths))
        # A control term is observed without error, so it has no residual.
        n_theta = (self.n_items - len(self.covariate_item_ids())
                   + len(self.error_covariances))
        n_cov = len(self.latent_covariances)
        return n_load + n_struct + n_theta + n_cov

    def _declared_item_loadings(self) -> int:
        """Loadings of items on factors, excluding second-order loadings and
        the auto policy's draws."""
        n_load = self.n_items + len(self.cross_loadings)
        if self.common_method_bias.enabled:
            n_load += (len(self.common_method_bias.item_ids)
                       or self.n_items - len(self.covariate_item_ids()))
        return n_load

    def count_moments(self) -> int:
        p = self.expanded().n_items
        return p * (p + 1) // 2

    def degrees_of_freedom(self, compiled: Any = None) -> int:
        return self.count_moments() - self.count_free_parameters(compiled)

    # ------------------------------------------------------------------ #
    # Convenience builders used by the UI and the tests
    # ------------------------------------------------------------------ #

    def add_construct(
        self,
        name: str,
        *,
        label_ar: str = "",
        label_en: str = "",
        role: ConstructRole = ConstructRole.INDEPENDENT,
        n_items: int = 4,
        loading: float = 0.75,
        n_categories: int = 5,
    ) -> Construct:
        """Create a first-order construct with ``n_items`` auto-named items."""
        c = Construct(
            name=name,
            label=Bilingual(ar=label_ar or name, en=label_en or name),
            role=role,
            order=ConstructOrder.FIRST,
        )
        for k in range(1, n_items + 1):
            c.items.append(Item(
                name=f"{name}{k}",
                label=Bilingual(ar=f"{label_ar or name} {k}",
                                en=f"{label_en or name} {k}"),
                loading=loading,
                scale=ScaleSpec(n_categories=n_categories),
            ))
        self.constructs.append(c)
        return c

    def add_second_order_construct(
        self,
        name: str,
        *,
        label_ar: str = "",
        label_en: str = "",
        role: ConstructRole = ConstructRole.INDEPENDENT,
        n_dimensions: int = 3,
        items_per_dimension: int = 4,
        loading: float = 0.75,
        second_order_loading: float = 0.80,
        n_categories: int = 5,
    ) -> Construct:
        c = Construct(
            name=name,
            label=Bilingual(ar=label_ar or name, en=label_en or name),
            role=role,
            order=ConstructOrder.SECOND,
        )
        for j in range(1, n_dimensions + 1):
            d = Dimension(
                name=f"{name}_D{j}",
                label=Bilingual(ar=f"{label_ar or name} - البعد {j}",
                                en=f"{label_en or name} - Dimension {j}"),
                second_order_loading=second_order_loading,
            )
            for k in range(1, items_per_dimension + 1):
                d.items.append(Item(
                    name=f"{name}{j}{k}",
                    label=Bilingual(ar=f"{label_ar or name} {j}.{k}",
                                    en=f"{label_en or name} {j}.{k}"),
                    loading=loading,
                    scale=ScaleSpec(n_categories=n_categories),
                ))
            c.dimensions.append(d)
        self.constructs.append(c)
        return c

    def add_path(
        self,
        source: str,
        target: str,
        *,
        effect_size: EffectSize = EffectSize.MEDIUM,
        coefficient: Optional[float] = None,
        significant: bool = True,
    ) -> Path:
        """``source`` / ``target`` accept either construct ids or names."""
        sid = self._resolve_construct_ref(source)
        tid = self._resolve_construct_ref(target)
        p = Path(
            source=sid,
            target=tid,
            effect_size=effect_size,
            target_significant=significant,
        )
        if coefficient is not None:
            p.coefficient = coefficient
            p.coefficient_is_manual = True
        else:
            p.coefficient = effect_size.default_coefficient
        self.paths.append(p)
        return p

    def add_interaction(
        self,
        predictor: str,
        moderator: str,
        target: str,
        *,
        effect_size: EffectSize = EffectSize.WEAK,
        coefficient: Optional[float] = None,
        significant: bool = True,
    ) -> Interaction:
        ix = Interaction(
            predictor=self._resolve_construct_ref(predictor),
            moderator=self._resolve_construct_ref(moderator),
            target=self._resolve_construct_ref(target),
            effect_size=effect_size,
            target_significant=significant,
        )
        ix.coefficient = (coefficient if coefficient is not None
                          else effect_size.default_coefficient)
        ix.coefficient_is_manual = coefficient is not None
        self.interactions.append(ix)
        return ix

    def add_error_covariance(self, item_a: str, item_b: str,
                             correlation: float = 0.2) -> ErrorCovariance:
        ec = ErrorCovariance(
            item_a=self._resolve_item_ref(item_a),
            item_b=self._resolve_item_ref(item_b),
            correlation=correlation,
        )
        self.error_covariances.append(ec)
        return ec

    def add_cross_loading(self, item: str, factor: str,
                          loading: float = 0.2) -> CrossLoading:
        cl = CrossLoading(
            item_id=self._resolve_item_ref(item),
            factor_id=self._resolve_factor_ref(factor),
            loading=loading,
        )
        self.cross_loadings.append(cl)
        return cl

    # -- reference resolution --------------------------------------------- #

    def _resolve_construct_ref(self, ref: str) -> str:
        if self.construct(ref) is not None:
            return ref
        c = self.construct_by_name(ref)
        if c is not None:
            return c.id
        raise KeyError(f"Unknown construct reference: {ref!r}")

    def _resolve_item_ref(self, ref: str) -> str:
        if self.item(ref) is not None:
            return ref
        for it in self.all_items():
            if it.name == ref:
                return it.id
        raise KeyError(f"Unknown item reference: {ref!r}")

    def _resolve_factor_ref(self, ref: str) -> str:
        if ref in self.all_factor_ids():
            return ref
        c = self.construct_by_name(ref)
        if c is not None:
            return c.id
        for cc in self.constructs:
            for d in cc.dimensions:
                if d.name == ref:
                    return d.id
        raise KeyError(f"Unknown factor reference: {ref!r}")

    # ------------------------------------------------------------------ #
    # Naming hygiene
    # ------------------------------------------------------------------ #

    def add_demographic(
        self,
        name: str,
        categories: Sequence[Tuple[str, float]],
        *,
        kind: DemographicKind = DemographicKind.NOMINAL,
        role: DemographicRole = DemographicRole.DESCRIPTIVE,
        label: Optional[Bilingual] = None,
    ) -> DemographicVariable:
        """Add a demographic variable from ``(label, proportion)`` pairs."""
        variable = DemographicVariable(
            name=name, label=label or Bilingual(ar=name, en=name),
            kind=kind, role=role,
            categories=[DemographicCategory(label=Bilingual(ar=text, en=text),
                                            proportion=share)
                        for text, share in categories])
        self.demographics.append(variable)
        return variable

    def add_control_path(
        self,
        variable: DemographicVariable,
        target: str,
        *,
        term: str = "",
        coefficient: Optional[float] = None,
        effect_size: EffectSize = EffectSize.WEAK,
        target_significant: bool = True,
    ) -> ControlPath:
        """Add the effect of a control term on a construct (id or name)."""
        path = ControlPath(
            term=term, target=self._resolve_construct_ref(target),
            effect_size=(EffectSize.CUSTOM if coefficient is not None
                         else effect_size),
            coefficient=(coefficient if coefficient is not None
                         else effect_size.default_coefficient),
            target_significant=target_significant)
        variable.control_paths.append(path)
        return path

    def add_group_difference(
        self,
        variable: DemographicVariable,
        target: str,
        *,
        size: GroupEffect = GroupEffect.SMALL,
        value: Optional[float] = None,
        direction: GroupDirection = GroupDirection.ASCENDING,
        target_significant: bool = True,
    ) -> GroupDifference:
        """Make a comparison variable's groups differ on a construct (id or name).

        ``value`` is an exact d (two categories) or eta squared (three or
        more) and overrides ``size``.
        """
        difference = GroupDifference(
            target=self._resolve_construct_ref(target),
            size=GroupEffect.CUSTOM if value is not None else size,
            value=float(value) if value is not None else 0.30,
            direction=direction, target_significant=target_significant)
        variable.group_differences.append(difference)
        return difference

    def add_demographic_association(
        self,
        first: "DemographicVariable | str",
        second: "DemographicVariable | str",
        *,
        strength: AssociationStrength = AssociationStrength.MODERATE,
        value: Optional[float] = None,
        direction: AssociationDirection = AssociationDirection.POSITIVE,
        target_significant: bool = True,
    ) -> DemographicAssociation:
        """Link two demographic variables (objects, ids or column names).

        ``value`` is an exact Cramér's V and overrides ``strength``.
        """
        def resolve(ref: "DemographicVariable | str") -> str:
            if isinstance(ref, DemographicVariable):
                return ref.id
            for variable in self.demographics:
                if ref in (variable.id, variable.name):
                    return variable.id
            return str(ref)

        association = DemographicAssociation(
            first=resolve(first), second=resolve(second),
            strength=AssociationStrength.CUSTOM if value is not None else strength,
            value=float(value) if value is not None else 0.30,
            direction=direction, target_significant=target_significant)
        self.demographic_associations.append(association)
        return association

    def ensure_unique_item_names(self) -> None:
        """Make every exported column name unique and syntactically safe."""
        seen: Dict[str, int] = {}
        for idx, it in enumerate(self.all_items(), start=1):
            base = _sanitise_name(it.name) or f"V{idx}"
            if base in seen:
                seen[base] += 1
                it.name = f"{base}_{seen[base]}"
            else:
                seen[base] = 0
                it.name = base


def _sanitise_name(name: str) -> str:
    """Keep names usable as identifiers in R / Mplus / SPSS."""
    out = []
    for ch in (name or "").strip():
        if ch.isalnum() and ch.isascii():
            out.append(ch)
        elif ch in "._":
            out.append(ch)
        elif ch.isspace() or ch == "-":
            out.append("_")
    cleaned = "".join(out).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = "V" + cleaned
    return cleaned[:32]


def asdict_shallow(obj: Any) -> Dict[str, Any]:
    """``dataclasses.asdict`` recurses and destroys nested dataclasses; this
    keeps the first level intact so ``clone`` can reuse live sub-objects."""
    return {f: getattr(obj, f) for f in obj.__dataclass_fields__}  # type: ignore[attr-defined]


__all__ = [
    "SPEC_VERSION",
    "ConstructRole", "ConstructOrder", "ScaleKind", "ResponseStyle",
    "EffectSize", "FitLevel", "GenerationEngine", "EstimatorKind",
    "MissingMechanism", "PowerMethod",
    "Bilingual", "ScaleSpec", "Item", "Dimension", "Construct",
    "CrossLoading", "CrossLoadingPolicy", "ErrorCovariance", "CommonMethodBias",
    "Path", "LatentCovariance", "Interaction", "IndirectEffect",
    "NoiseSpec", "HeterogeneitySpec", "HETEROGENEITY_PRESETS",
    "DistributionSpec",
    "FitTargetSpec", "SignificanceTargetSpec", "PowerSpec",
    "AnalysisSpec", "ExportSpec", "ModelSpec",
    "DemographicKind", "DemographicRole", "DemographicCategory",
    "GroupEffect", "GroupDirection", "GroupDifference", "GROUP_D_MAX",
    "GROUP_ETA_MAX", "AssociationStrength", "AssociationDirection",
    "DemographicAssociation", "ASSOCIATION_V_MAX",
    "DemographicTerm", "DemographicVariable", "ControlPath",
]

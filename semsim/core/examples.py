"""
semsim.core.examples
====================
Ready-made model specifications.

These are not toys.  Each one is a design a reviewer would recognise from the
applied literature, which makes them useful for three jobs at once: a starting
point for the researcher, a demonstration of a particular feature, and a
regression fixture for the test suite.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from semsim.core.spec import (
    Bilingual, ConstructRole, EffectSize, FitLevel, GenerationEngine,
    IndirectEffect, LatentCovariance, MissingMechanism, ModelSpec,
    PowerMethod, ResponseStyle, ScaleKind,
)


# --------------------------------------------------------------------------- #
# 1. Simple mediation
# --------------------------------------------------------------------------- #


def simple_mediation() -> ModelSpec:
    """X -> M -> Y with a direct path: the canonical mediation design.

    Three first-order constructs, five-point Likert items, mild negative skew
    of the kind self-report attitude scales nearly always show.
    """
    spec = ModelSpec(
        title=Bilingual(ar="نموذج وساطة: الرضا والاندماج والأداء",
                        en="Mediation: satisfaction, engagement, performance"),
        description=(
            "Job satisfaction raises work engagement, which in turn raises "
            "task performance. A small direct path is retained so partial "
            "mediation can be demonstrated."),
        seed=20240101,
    )
    sat = spec.add_construct(
        "SAT", label_ar="الرضا الوظيفي", label_en="Job satisfaction",
        role=ConstructRole.INDEPENDENT, n_items=5, loading=0.78)
    eng = spec.add_construct(
        "ENG", label_ar="الاندماج الوظيفي", label_en="Work engagement",
        role=ConstructRole.MEDIATOR, n_items=5, loading=0.80)
    perf = spec.add_construct(
        "PERF", label_ar="الأداء الوظيفي", label_en="Task performance",
        role=ConstructRole.DEPENDENT, n_items=4, loading=0.75)

    spec.add_path("SAT", "ENG", effect_size=EffectSize.STRONG)
    spec.add_path("ENG", "PERF", effect_size=EffectSize.MEDIUM)
    spec.add_path("SAT", "PERF", effect_size=EffectSize.WEAK)
    spec.indirect_effects.append(IndirectEffect(
        chain=[sat.id, eng.id, perf.id],
        label="SAT -> ENG -> PERF"))

    for item in spec.all_items():
        item.scale.response_style = ResponseStyle.NEGATIVE_SKEW
    spec.distribution.default_skewness = -0.45
    spec.distribution.default_kurtosis = 0.25

    spec.power.mode = "suggest"
    spec.fit_target.level = FitLevel.GOOD
    spec.ensure_unique_item_names()
    return spec


# --------------------------------------------------------------------------- #
# 2. Second-order hierarchy
# --------------------------------------------------------------------------- #


def second_order_hierarchy() -> ModelSpec:
    """A second-order construct built from three dimensions.

    Service quality as a higher-order construct with tangibility, reliability
    and empathy dimensions -- the structure the user's brief calls a
    "complex human concept made of several main dimensions".
    """
    spec = ModelSpec(
        title=Bilingual(ar="نموذج هرمي: جودة الخدمة ورضا العميل والولاء",
                        en="Hierarchical: service quality, satisfaction, loyalty"),
        description=(
            "Service quality is a second-order construct with three "
            "first-order dimensions. It predicts customer satisfaction, which "
            "predicts loyalty."),
        seed=20240202,
    )
    sq = spec.add_second_order_construct(
        "SQ", label_ar="جودة الخدمة", label_en="Service quality",
        role=ConstructRole.INDEPENDENT,
        n_dimensions=3, items_per_dimension=4,
        loading=0.76, second_order_loading=0.82)
    dim_labels = [
        ("الملموسية", "Tangibles"),
        ("الاعتمادية", "Reliability"),
        ("التعاطف", "Empathy"),
    ]
    for dim, (ar, en) in zip(sq.dimensions, dim_labels):
        dim.label = Bilingual(ar=ar, en=en)

    csat = spec.add_construct(
        "CSAT", label_ar="رضا العميل", label_en="Customer satisfaction",
        role=ConstructRole.MEDIATOR, n_items=4, loading=0.81)
    loy = spec.add_construct(
        "LOY", label_ar="الولاء", label_en="Loyalty",
        role=ConstructRole.DEPENDENT, n_items=4, loading=0.79)

    spec.add_path("SQ", "CSAT", effect_size=EffectSize.STRONG)
    spec.add_path("CSAT", "LOY", effect_size=EffectSize.STRONG)
    spec.add_path("SQ", "LOY", effect_size=EffectSize.TRIVIAL, significant=False)
    spec.indirect_effects.append(IndirectEffect(
        chain=[sq.id, csat.id, loy.id], label="SQ -> CSAT -> LOY"))

    for item in spec.all_items():
        item.scale.n_categories = 7
    spec.fit_target.level = FitLevel.GOOD
    spec.ensure_unique_item_names()
    return spec


# --------------------------------------------------------------------------- #
# 3. Moderation
# --------------------------------------------------------------------------- #


def moderation() -> ModelSpec:
    """A latent interaction: stress -> burnout, moderated by social support.

    Forces the score-based generator, because a product of two latent
    variables cannot be expressed in a covariance matrix.
    """
    spec = ModelSpec(
        title=Bilingual(ar="نموذج تعديل: ضغوط العمل والاحتراق والدعم",
                        en="Moderation: stress, burnout, social support"),
        description=(
            "Work stress predicts burnout. Perceived social support weakens "
            "that relationship, so the interaction coefficient is negative."),
        seed=20240303,
    )
    spec.add_construct(
        "STR", label_ar="ضغوط العمل", label_en="Work stress",
        role=ConstructRole.INDEPENDENT, n_items=5, loading=0.77)
    spec.add_construct(
        "SUP", label_ar="الدعم الاجتماعي المُدرك", label_en="Perceived support",
        role=ConstructRole.MODERATOR, n_items=4, loading=0.80)
    spec.add_construct(
        "BO", label_ar="الاحتراق النفسي", label_en="Burnout",
        role=ConstructRole.DEPENDENT, n_items=5, loading=0.82)

    spec.add_path("STR", "BO", effect_size=EffectSize.STRONG)
    spec.add_path("SUP", "BO", effect_size=EffectSize.MEDIUM, coefficient=-0.28)
    spec.add_interaction("STR", "SUP", "BO", coefficient=-0.18)
    spec.latent_covariances.append(LatentCovariance(
        construct_a=spec.construct_by_name("STR").id,
        construct_b=spec.construct_by_name("SUP").id,
        correlation=-0.20))

    # Stress items pile up at the low end in a healthy workforce.
    for item in spec.construct_by_name("STR").items:
        item.scale.response_style = ResponseStyle.POSITIVE_SKEW
    for item in spec.construct_by_name("BO").items:
        item.scale.response_style = ResponseStyle.POSITIVE_SKEW

    spec.engine = GenerationEngine.SCORES
    spec.fit_target.level = FitLevel.ACCEPTABLE
    spec.ensure_unique_item_names()
    return spec


# --------------------------------------------------------------------------- #
# 4. Messy reality
# --------------------------------------------------------------------------- #


def messy_reality() -> ModelSpec:
    """Everything the realism layer can do, switched on at once.

    Cross-loadings, correlated residuals, a common method factor, careless
    respondents, outliers, MAR missingness, strong non-normality and a
    deliberately merely-acceptable fit.  Useful for stress-testing the whole
    pipeline, and for teaching what messy data looks like.
    """
    spec = ModelSpec(
        title=Bilingual(ar="نموذج واقعي معقّد: النية السلوكية",
                        en="Complex realistic model: behavioural intention"),
        description=(
            "A TPB-style model carrying every complication the engine can "
            "simulate, so a researcher can see how each one shows up in the "
            "reliability, validity and fit output."),
        seed=20240404,
    )
    att = spec.add_construct(
        "ATT", label_ar="الاتجاه", label_en="Attitude",
        role=ConstructRole.INDEPENDENT, n_items=4, loading=0.74)
    norm = spec.add_construct(
        "NORM", label_ar="المعايير الذاتية", label_en="Subjective norms",
        role=ConstructRole.INDEPENDENT, n_items=4, loading=0.70)
    pbc = spec.add_construct(
        "PBC", label_ar="التحكم السلوكي المُدرك", label_en="Perceived control",
        role=ConstructRole.INDEPENDENT, n_items=4, loading=0.72)
    intent = spec.add_construct(
        "INT", label_ar="النية السلوكية", label_en="Behavioural intention",
        role=ConstructRole.MEDIATOR, n_items=4, loading=0.80)
    beh = spec.add_construct(
        "BEH", label_ar="السلوك الفعلي", label_en="Actual behaviour",
        role=ConstructRole.DEPENDENT, n_items=3, loading=0.76)
    spec.ensure_unique_item_names()

    spec.add_path("ATT", "INT", effect_size=EffectSize.MEDIUM)
    spec.add_path("NORM", "INT", effect_size=EffectSize.WEAK)
    spec.add_path("PBC", "INT", effect_size=EffectSize.MEDIUM)
    spec.add_path("INT", "BEH", effect_size=EffectSize.STRONG)
    spec.add_path("PBC", "BEH", effect_size=EffectSize.WEAK)
    spec.add_path("ATT", "BEH", effect_size=EffectSize.NONE, significant=False)
    spec.indirect_effects.append(IndirectEffect(
        chain=[att.id, intent.id, beh.id], label="ATT -> INT -> BEH"))

    spec.latent_covariances += [
        LatentCovariance(construct_a=att.id, construct_b=norm.id, correlation=0.35),
        LatentCovariance(construct_a=att.id, construct_b=pbc.id, correlation=0.30),
        LatentCovariance(construct_a=norm.id, construct_b=pbc.id, correlation=0.25),
    ]

    # Two items that genuinely tap a neighbouring construct.
    spec.add_cross_loading("ATT1", "NORM", 0.22)
    spec.add_cross_loading("PBC2", "INT", 0.18)
    # Two pairs worded almost identically.
    spec.add_error_covariance("ATT3", "ATT4", 0.24)
    spec.add_error_covariance("INT1", "INT2", 0.20)

    spec.common_method_bias.enabled = True
    spec.common_method_bias.loading = 0.26

    spec.cross_loading_policy.enabled = True
    spec.cross_loading_policy.proportion_of_items = 0.15
    spec.cross_loading_policy.magnitude_range = (0.08, 0.18)

    spec.distribution.default_skewness = -0.70
    spec.distribution.default_kurtosis = 0.60
    for item in spec.all_items():
        item.scale.response_style = ResponseStyle.NEGATIVE_SKEW
    for item in spec.construct_by_name("BEH").items:
        item.scale.n_categories = 5
        item.scale.response_style = ResponseStyle.BIMODAL
    spec.all_items()[2].reverse_coded = True
    spec.all_items()[9].reverse_coded = True

    spec.noise.enabled = True
    spec.noise.unique_variance_inflation = 1.08
    spec.noise.careless_proportion = 0.04
    spec.noise.careless_style = "mixed"
    spec.noise.outlier_proportion = 0.01
    spec.noise.outlier_magnitude_sd = 3.5
    spec.noise.missing_mechanism = MissingMechanism.MAR
    spec.noise.missing_proportion = 0.05
    spec.noise.mar_driver = att.id

    spec.fit_target.level = FitLevel.ACCEPTABLE
    spec.power.mode = "suggest"
    spec.power.method = PowerMethod.RMSEA_CLOSE_FIT
    spec.analysis.report_htmt = True
    spec.analysis.bootstrap_samples = 0
    spec.export.formats = ["csv", "xlsx"]
    spec.export.include_latent_scores = True
    return spec


# --------------------------------------------------------------------------- #
# 5. Minimal CFA
# --------------------------------------------------------------------------- #


def minimal_cfa() -> ModelSpec:
    """Two correlated factors, three items each: the smallest identified CFA.

    df = 21 - 13 = 8.  Fast enough to be the default fixture in unit tests.
    """
    spec = ModelSpec(
        title=Bilingual(ar="نموذج قياس أدنى (عاملان)",
                        en="Minimal two-factor CFA"),
        seed=12345,
    )
    a = spec.add_construct("F1", label_ar="العامل الأول", label_en="Factor 1",
                           role=ConstructRole.INDEPENDENT, n_items=3,
                           loading=0.80)
    b = spec.add_construct("F2", label_ar="العامل الثاني", label_en="Factor 2",
                           role=ConstructRole.INDEPENDENT, n_items=3,
                           loading=0.80)
    spec.latent_covariances.append(LatentCovariance(
        construct_a=a.id, construct_b=b.id, correlation=0.40))
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS
    spec.fit_target.level = FitLevel.PERFECT
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    spec.distribution.use_non_normal_engine = False
    spec.ensure_unique_item_names()
    return spec


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

EXAMPLES: Dict[str, Tuple[str, Callable[[], ModelSpec]]] = {
    "minimal_cfa": ("ex.minimal", minimal_cfa),
    "simple_mediation": ("ex.mediation", simple_mediation),
    "second_order": ("ex.second_order", second_order_hierarchy),
    "moderation": ("ex.moderation", moderation),
    "messy_reality": ("ex.full", messy_reality),
}


def example_names() -> List[str]:
    return list(EXAMPLES)


def build(name: str) -> ModelSpec:
    """Instantiate an example by registry key."""
    try:
        return EXAMPLES[name][1]()
    except KeyError:
        raise KeyError(
            f"Unknown example {name!r}; available: {', '.join(EXAMPLES)}") from None


def all_examples() -> Dict[str, ModelSpec]:
    return {name: build(name) for name in EXAMPLES}


__all__ = [
    "simple_mediation", "second_order_hierarchy", "moderation",
    "messy_reality", "minimal_cfa",
    "EXAMPLES", "example_names", "build", "all_examples",
]

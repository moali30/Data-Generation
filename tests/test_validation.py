"""
Numerical tests for :mod:`semsim.core.validation`.

Every diagnostic code published in ``docs/ARCHITECTURE.md`` gets a spec
engineered to trigger exactly that code, and every assertion is against a
closed-form number (degrees of freedom counted by hand, a quadratic form
``b' R b`` evaluated by hand, the Fleishman boundary ``kurtosis = skew^2 - 2``,
the eigenvalues of a 2x2 symmetric matrix ``a +/- b``) rather than against
"it ran".

Runnable either with pytest::

    python3 -m pytest tests/test_validation.py -q

or, when pytest is unavailable, directly::

    python3 tests/test_validation.py
"""

from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semsim.core.spec import (  # noqa: E402
    ConstructRole,
    EffectSize,
    ErrorCovariance,
    Interaction,
    LatentCovariance,
    MissingMechanism,
    ModelSpec,
    Path,
    ScaleKind,
)
from semsim.core.validation import (  # noqa: E402
    CONDITION_NUMBER_LIMIT,
    HEYWOOD_FLOOR,
    LOW_COMMUNALITY_FLOOR,
    MIN_CASES_PER_PARAMETER,
    Diagnostic,
    Severity,
    ValidationReport,
    validate,
    validate_compiled,
)

ARABIC = re.compile(r"[؀-ۿ]")

#: Every code the integration contract requires from this module.
REQUIRED_SPEC_CODES = [
    "NO_CONSTRUCTS", "CONSTRUCT_NO_ITEMS", "TOO_FEW_ITEMS",
    "SECOND_ORDER_TOO_FEW_DIMS", "DUPLICATE_ITEM_NAME", "LOADING_OUT_OF_RANGE",
    "NEGATIVE_DF", "ZERO_DF", "CYCLE_IN_PATHS", "SELF_PATH", "DUPLICATE_PATH",
    "PATH_UNKNOWN_CONSTRUCT", "MODERATOR_NOT_IN_INTERACTION",
    "INTERACTION_TARGET_EXOGENOUS", "MEDIATOR_WITHOUT_IN_OR_OUT",
    "DEPENDENT_WITHOUT_PREDICTOR", "ERRCOV_SAME_ITEM", "ERRCOV_OUT_OF_RANGE",
    "CROSSLOAD_HOME_FACTOR", "HEYWOOD_RISK", "R2_EXCEEDS_ONE",
    "LIKERT_CATEGORY_RANGE", "PROPORTIONS_DONT_SUM",
    "INFEASIBLE_SKEW_KURTOSIS", "SAMPLE_SIZE_TOO_SMALL",
    "FIT_TARGET_IMPOSSIBLE_AT_ZERO_DF", "MISSING_PROPORTION_RANGE",
]
REQUIRED_COMPILED_CODES = [
    "HEYWOOD_CASE", "SIGMA_NOT_PD", "R2_EXCEEDS_ONE",
    "SIGMA_ILL_CONDITIONED", "LOW_COMMUNALITY",
]

#: Filled in as the tests run, then audited by ``test_every_code_is_covered``
#: and ``test_all_diagnostics_are_truly_bilingual``.
SEEN: dict = {}


def _record(report: ValidationReport) -> ValidationReport:
    for d in report.diagnostics:
        SEEN.setdefault(d.code, d)
    return report


# --------------------------------------------------------------------------- #
# Spec builders
# --------------------------------------------------------------------------- #


def clean_spec() -> ModelSpec:
    """A well-formed mediation model: X -> M -> Y plus a direct X -> Y.

    12 items, so 12*13/2 = 78 moments and
    12 loadings + 12 unique variances + 3 paths + 2 disturbances = 29 free
    parameters, i.e. df = 49.
    """
    s = ModelSpec()
    s.add_construct("X", n_items=4, loading=0.75, role=ConstructRole.INDEPENDENT)
    s.add_construct("M", n_items=4, loading=0.75, role=ConstructRole.MEDIATOR)
    s.add_construct("Y", n_items=4, loading=0.75, role=ConstructRole.DEPENDENT)
    s.add_path("X", "M", effect_size=EffectSize.MEDIUM)
    s.add_path("M", "Y", effect_size=EffectSize.MEDIUM)
    s.add_path("X", "Y", effect_size=EffectSize.WEAK)
    return s


def _two_factor_six_items() -> ModelSpec:
    """6 items on 2 constructs, one path: 21 moments, 14 free parameters."""
    s = ModelSpec()
    s.add_construct("A", n_items=3, loading=0.75)
    s.add_construct("B", n_items=3, loading=0.75, role=ConstructRole.DEPENDENT)
    s.add_path("A", "B")
    return s


def _distinct_pairs(names, k):
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append((names[i], names[j]))
            if len(out) == k:
                return out
    return out


# --------------------------------------------------------------------------- #
# A clean spec is clean
# --------------------------------------------------------------------------- #


def test_clean_spec_has_no_errors():
    s = clean_spec()
    assert s.count_moments() == 78, s.count_moments()
    assert s.count_free_parameters() == 29, s.count_free_parameters()
    assert s.degrees_of_freedom() == 49, s.degrees_of_freedom()

    report = validate(s)
    assert report.errors() == [], [d.code for d in report.errors()]
    assert report.ok is True
    # a genuinely clean model should not even raise a warning
    assert report.diagnostics == [], [d.code for d in report.diagnostics]


def test_validate_never_raises_on_garbage():
    class Junk:
        constructs = "not a list of constructs"
        paths = None

    report = validate(Junk())          # type: ignore[arg-type]
    assert isinstance(report, ValidationReport)
    report2 = validate(ModelSpec(constructs=[]))
    assert report2.has("NO_CONSTRUCTS")
    _record(report2)


def test_validate_is_fast_enough_for_every_keystroke():
    s = ModelSpec()
    for k in range(20):
        s.add_construct(f"C{k}", n_items=4, loading=0.70)
    for k in range(1, 20):
        s.add_path(f"C{k - 1}", f"C{k}", effect_size=EffectSize.WEAK)
    s.power.n = 2000
    validate(s)                                    # warm up
    t0 = time.perf_counter()
    for _ in range(10):
        validate(s)
    elapsed_ms = (time.perf_counter() - t0) / 10 * 1000.0
    print(f"    [timing] 20-construct / 80-item validate = {elapsed_ms:.3f} ms")
    assert elapsed_ms < 20.0, elapsed_ms


# --------------------------------------------------------------------------- #
# Measurement-layer codes
# --------------------------------------------------------------------------- #


def test_no_constructs():
    report = _record(validate(ModelSpec()))
    assert report.has("NO_CONSTRUCTS")
    assert not report.ok


def test_construct_no_items():
    s = ModelSpec()
    s.add_construct("A", n_items=0)
    report = _record(validate(s))
    assert report.has("CONSTRUCT_NO_ITEMS")
    assert "A" in report.by_code("CONSTRUCT_NO_ITEMS")[0].message_en


def test_too_few_items():
    s = ModelSpec()
    s.add_construct("A", n_items=2)
    s.add_construct("B", n_items=4)
    report = _record(validate(s))
    hits = report.by_code("TOO_FEW_ITEMS")
    assert len(hits) == 1, [d.where for d in hits]
    assert "only 2 item" in hits[0].message_en


def test_second_order_too_few_dims():
    s = ModelSpec()
    s.add_second_order_construct("SO", n_dimensions=2, items_per_dimension=4)
    report = _record(validate(s))
    hits = report.by_code("SECOND_ORDER_TOO_FEW_DIMS")
    assert len(hits) == 1
    assert "only 2 dimension" in hits[0].message_en
    # three dimensions is enough
    s2 = ModelSpec()
    s2.add_second_order_construct("SO", n_dimensions=3, items_per_dimension=4)
    assert not validate(s2).has("SECOND_ORDER_TOO_FEW_DIMS")


def test_duplicate_item_name():
    s = clean_spec()
    items = s.all_items()
    items[5].name = items[0].name
    report = _record(validate(s))
    hits = report.by_code("DUPLICATE_ITEM_NAME")
    assert len(hits) == 1
    assert items[0].name in hits[0].message_en


def test_loading_out_of_range():
    s = clean_spec()
    s.all_items()[0].loading = 1.05
    report = _record(validate(s))
    hits = report.by_code("LOADING_OUT_OF_RANGE")
    assert len(hits) == 1
    assert "1.050" in hits[0].message_en
    # the boundary itself is exclusive: 0.999 is legal, 1.0 is not
    s.all_items()[0].loading = 0.999
    assert not validate(s).has("LOADING_OUT_OF_RANGE")
    s.all_items()[0].loading = 1.0
    assert validate(s).has("LOADING_OUT_OF_RANGE")


# --------------------------------------------------------------------------- #
# Identification: hand-counted degrees of freedom
# --------------------------------------------------------------------------- #


def test_negative_df_at_hand_counted_boundary():
    s = _two_factor_six_items()
    names = [it.name for it in s.all_items()]
    assert s.count_moments() == 21
    assert s.count_free_parameters() == 14      # 6 + 6 + 1 path + 1 psi
    for a, b in _distinct_pairs(names, 8):
        s.add_error_covariance(a, b, correlation=0.15)
    # each residual covariance adds exactly one free parameter: 14 + 8 = 22
    assert s.count_free_parameters() == 22, s.count_free_parameters()
    assert s.degrees_of_freedom() == -1, s.degrees_of_freedom()
    report = _record(validate(s))
    assert report.has("NEGATIVE_DF")
    assert not report.ok
    msg = report.by_code("NEGATIVE_DF")[0].message_en
    assert "(-1)" in msg and "22 free parameters" in msg and "21 known" in msg


def test_zero_df_and_fit_target_impossible():
    s = _two_factor_six_items()
    names = [it.name for it in s.all_items()]
    for a, b in _distinct_pairs(names, 7):
        s.add_error_covariance(a, b, correlation=0.15)
    assert s.count_free_parameters() == 21
    assert s.degrees_of_freedom() == 0, s.degrees_of_freedom()
    report = _record(validate(s))
    assert report.has("ZERO_DF")
    assert not report.has("NEGATIVE_DF")
    # fit_target defaults to enabled / GOOD (RMSEA 0.045), unreachable at df 0
    assert s.fit_target.enabled and s.fit_target.method == "cudeck_browne"
    assert report.has("FIT_TARGET_IMPOSSIBLE_AT_ZERO_DF")
    assert "0.045" in report.by_code("FIT_TARGET_IMPOSSIBLE_AT_ZERO_DF")[0].message_en
    # ... and is not raised once df is positive again
    assert not validate(clean_spec()).has("FIT_TARGET_IMPOSSIBLE_AT_ZERO_DF")


def test_sample_size_too_small_at_the_five_per_parameter_boundary():
    s = clean_spec()
    q = s.count_free_parameters()
    need = MIN_CASES_PER_PARAMETER * q
    assert (q, need) == (29, 145)

    s.power.n = need
    assert not validate(s).has("SAMPLE_SIZE_TOO_SMALL")
    s.power.n = need - 1
    report = _record(validate(s))
    assert report.has("SAMPLE_SIZE_TOO_SMALL")
    msg = report.by_code("SAMPLE_SIZE_TOO_SMALL")[0].message_en
    assert "N = 144" in msg and "N = 145" in msg


# --------------------------------------------------------------------------- #
# Structural-layer codes
# --------------------------------------------------------------------------- #


def test_cycle_in_paths():
    s = clean_spec()
    s.add_path("Y", "X", effect_size=EffectSize.WEAK)       # closes X -> M -> Y -> X
    report = _record(validate(s))
    hits = report.by_code("CYCLE_IN_PATHS")
    assert len(hits) == 1
    for name in ("X", "M", "Y"):
        assert name in hits[0].message_en
    assert not validate(clean_spec()).has("CYCLE_IN_PATHS")


def test_self_path():
    s = clean_spec()
    xid = s.construct_by_name("X").id
    s.paths.append(Path(source=xid, target=xid, coefficient=0.3))
    report = _record(validate(s))
    assert report.has("SELF_PATH")
    assert not report.ok
    # a self path is not reported as a cycle as well
    assert not report.has("CYCLE_IN_PATHS")


def test_duplicate_path():
    s = clean_spec()
    s.add_path("X", "M", effect_size=EffectSize.MEDIUM)
    report = _record(validate(s))
    hits = report.by_code("DUPLICATE_PATH")
    assert len(hits) == 1
    assert "'X' -> 'M'" in hits[0].message_en


def test_path_unknown_construct():
    s = clean_spec()
    s.paths.append(Path(source="c_does_not_exist", target=s.construct_by_name("Y").id))
    report = _record(validate(s))
    hits = report.by_code("PATH_UNKNOWN_CONSTRUCT")
    assert len(hits) == 1
    assert "c_does_not_exist" in hits[0].message_en


def test_moderator_not_in_interaction():
    s = clean_spec()
    s.add_construct("W", n_items=4, loading=0.75, role=ConstructRole.MODERATOR)
    s.add_path("W", "Y", effect_size=EffectSize.WEAK)
    report = _record(validate(s))
    hits = report.by_code("MODERATOR_NOT_IN_INTERACTION")
    assert len(hits) == 1
    assert "W" in hits[0].message_en
    # adding the interaction clears it
    s.add_interaction("X", "W", "Y", effect_size=EffectSize.WEAK)
    assert not validate(s).has("MODERATOR_NOT_IN_INTERACTION")


def test_interaction_target_exogenous():
    s = ModelSpec()
    s.add_construct("X", n_items=4, loading=0.75)
    s.add_construct("W", n_items=4, loading=0.75, role=ConstructRole.MODERATOR)
    s.add_construct("Y", n_items=4, loading=0.75, role=ConstructRole.DEPENDENT)
    s.add_interaction("X", "W", "Y", effect_size=EffectSize.WEAK)
    report = _record(validate(s))
    hits = report.by_code("INTERACTION_TARGET_EXOGENOUS")
    assert len(hits) == 1
    assert "'Y'" in hits[0].message_en
    # with both main effects present the interaction is fine
    s.add_path("X", "Y")
    s.add_path("W", "Y")
    assert not validate(s).has("INTERACTION_TARGET_EXOGENOUS")


def test_mediator_without_in_or_out():
    s = ModelSpec()
    s.add_construct("X", n_items=4, loading=0.75)
    s.add_construct("M", n_items=4, loading=0.75, role=ConstructRole.MEDIATOR)
    s.add_construct("Y", n_items=4, loading=0.75, role=ConstructRole.DEPENDENT)
    s.add_path("X", "M")
    s.add_path("X", "Y")
    report = _record(validate(s))
    hits = report.by_code("MEDIATOR_WITHOUT_IN_OR_OUT")
    assert len(hits) == 1
    assert "no outgoing path" in hits[0].message_en
    s.add_path("M", "Y")
    assert not validate(s).has("MEDIATOR_WITHOUT_IN_OR_OUT")


def test_dependent_without_predictor():
    s = ModelSpec()
    s.add_construct("X", n_items=4, loading=0.75)
    s.add_construct("Y", n_items=4, loading=0.75, role=ConstructRole.DEPENDENT)
    report = _record(validate(s))
    hits = report.by_code("DEPENDENT_WITHOUT_PREDICTOR")
    assert len(hits) == 1
    assert "'Y'" in hits[0].message_en


# --------------------------------------------------------------------------- #
# Complex-reality codes
# --------------------------------------------------------------------------- #


def test_errcov_same_item():
    s = clean_spec()
    iid = s.all_items()[0].id
    s.error_covariances.append(ErrorCovariance(item_a=iid, item_b=iid, correlation=0.2))
    report = _record(validate(s))
    assert report.has("ERRCOV_SAME_ITEM")
    assert not report.ok


def test_errcov_out_of_range():
    s = clean_spec()
    names = [it.name for it in s.all_items()]
    s.add_error_covariance(names[0], names[1], correlation=1.4)
    report = _record(validate(s))
    hits = report.by_code("ERRCOV_OUT_OF_RANGE")
    assert len(hits) == 1
    assert "1.4" in hits[0].message_en
    # |r| < 1 is admissible, |r| == 1 is not
    s.error_covariances[0].correlation = 0.99
    assert not validate(s).has("ERRCOV_OUT_OF_RANGE")
    s.error_covariances[0].correlation = -1.0
    assert validate(s).has("ERRCOV_OUT_OF_RANGE")


def test_crossload_home_factor():
    s = clean_spec()
    item = s.all_items()[0]
    s.add_cross_loading(item.name, "X", loading=0.2)      # X is its home factor
    report = _record(validate(s))
    hits = report.by_code("CROSSLOAD_HOME_FACTOR")
    assert len(hits) == 1
    assert hits[0].where == item.id
    # the same cross-loading onto a different construct is legal
    s.cross_loadings.clear()
    s.add_cross_loading(item.name, "M", loading=0.2)
    assert not validate(s).has("CROSSLOAD_HOME_FACTOR")


# --------------------------------------------------------------------------- #
# Variance bounds: closed-form communality and R^2
# --------------------------------------------------------------------------- #


def test_heywood_risk_from_closed_form_communality():
    s = clean_spec()
    item = s.all_items()[0]
    item.loading = 0.9
    s.add_cross_loading(item.name, "M", loading=0.5)
    # communality = 0.9^2 + 0.5^2 = 0.81 + 0.25 = 1.06  (orthogonal factors)
    report = _record(validate(s))
    hits = report.by_code("HEYWOOD_RISK")
    assert len(hits) == 1, [d.message_en for d in hits]
    assert "1.060" in hits[0].message_en, hits[0].message_en
    assert hits[0].where == item.id

    # 0.9^2 + 0.4^2 = 0.97 < 1 -> admissible
    s.cross_loadings[0].loading = 0.4
    assert not validate(s).has("HEYWOOD_RISK")

    # the method factor adds its own square: 0.9^2 + 0.4^2 + 0.3^2 = 1.06
    s.common_method_bias.enabled = True
    s.common_method_bias.loading = 0.3
    rep2 = validate(s)
    assert rep2.has("HEYWOOD_RISK")
    assert "1.060" in rep2.by_code("HEYWOOD_RISK")[0].message_en


def test_r2_exceeds_one_from_closed_form_quadratic_form():
    s = ModelSpec()
    s.add_construct("X1", n_items=4, loading=0.70)
    s.add_construct("X2", n_items=4, loading=0.70)
    s.add_construct("Y", n_items=4, loading=0.70, role=ConstructRole.DEPENDENT)
    x1 = s.construct_by_name("X1").id
    x2 = s.construct_by_name("X2").id
    s.add_path("X1", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.7)
    s.add_path("X2", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.7)

    # orthogonal predictors: R^2 = 0.7^2 + 0.7^2 = 0.98 < 1  -> admissible
    assert not validate(s).has("R2_EXCEEDS_ONE")

    # correlated predictors: R^2 = b' R b = 0.49 + 0.49 + 2*0.7*0.7*0.5 = 1.47
    s.latent_covariances.append(
        LatentCovariance(construct_a=x1, construct_b=x2, correlation=0.5))
    b = np.array([0.7, 0.7])
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert abs(float(b @ R @ b) - 1.47) < 1e-12

    report = _record(validate(s))
    hits = report.by_code("R2_EXCEEDS_ONE")
    assert len(hits) == 1
    assert "1.470" in hits[0].message_en, hits[0].message_en
    assert not report.ok


# --------------------------------------------------------------------------- #
# Scale and distribution codes
# --------------------------------------------------------------------------- #


def test_likert_category_range():
    s = clean_spec()
    s.all_items()[0].scale.n_categories = 15
    report = _record(validate(s))
    hits = report.by_code("LIKERT_CATEGORY_RANGE")
    assert len(hits) == 1
    assert "15 Likert categories" in hits[0].message_en
    for k, legal in ((1, False), (2, True), (11, True), (12, False)):
        s.all_items()[0].scale.n_categories = k
        assert validate(s).has("LIKERT_CATEGORY_RANGE") is (not legal), k


def test_proportions_dont_sum():
    s = clean_spec()
    sc = s.all_items()[0].scale
    sc.n_categories = 5
    sc.category_proportions = [0.3, 0.3, 0.3, 0.3, 0.3]        # sums to 1.5
    report = _record(validate(s))
    hits = report.by_code("PROPORTIONS_DONT_SUM")
    assert len(hits) == 1
    assert "1.5000" in hits[0].message_en
    sc.category_proportions = [0.1, 0.2, 0.4, 0.2, 0.1]        # sums to 1.0
    assert not validate(s).has("PROPORTIONS_DONT_SUM")


def test_infeasible_skew_kurtosis_at_the_fleishman_boundary():
    s = clean_spec()
    item = s.all_items()[0]
    item.scale.kind = ScaleKind.CONTINUOUS
    item.skewness = 2.0
    s.distribution.clamp_to_feasible_region = False
    bound = 2.0 ** 2 - 2.0                         # = 2.0
    assert bound == 2.0

    item.kurtosis = 1.9                            # <= bound -> no real solution
    report = _record(validate(s))
    hits = report.by_code("INFEASIBLE_SKEW_KURTOSIS")
    assert len(hits) == 1
    assert hits[0].severity is Severity.ERROR
    assert "2.000" in hits[0].message_en

    item.kurtosis = 2.0                            # exactly on the boundary
    assert validate(s).has("INFEASIBLE_SKEW_KURTOSIS")
    item.kurtosis = 2.5                            # strictly above -> feasible
    assert not validate(s).has("INFEASIBLE_SKEW_KURTOSIS")

    # with clamping on, the same request is a warning rather than an error
    item.kurtosis = 1.9
    s.distribution.clamp_to_feasible_region = True
    hits = validate(s).by_code("INFEASIBLE_SKEW_KURTOSIS")
    assert len(hits) == 1 and hits[0].severity is Severity.WARNING


def test_missing_proportion_range():
    s = clean_spec()
    s.noise.missing_mechanism = MissingMechanism.MCAR
    s.noise.missing_proportion = 1.4
    report = _record(validate(s))
    hits = report.by_code("MISSING_PROPORTION_RANGE")
    assert len(hits) == 1
    assert hits[0].severity is Severity.ERROR
    s.noise.missing_proportion = 0.15
    assert not validate(s).has("MISSING_PROPORTION_RANGE")


# --------------------------------------------------------------------------- #
# Post-compilation checks
# --------------------------------------------------------------------------- #


class FakeCompiled:
    """Minimal duck-typed stand-in for ``semsim.core.compile.CompiledModel``."""

    def __init__(self, Lambda, Phi, Theta, *, Psi=None, B=None,
                 endogenous=None, item_names=None, factor_names=None,
                 warnings=None):
        self.Lambda = np.asarray(Lambda, dtype=float)
        self.Phi = np.asarray(Phi, dtype=float)
        self.Theta = np.asarray(Theta, dtype=float)
        m = self.Phi.shape[0]
        self.Psi = np.eye(m) if Psi is None else np.asarray(Psi, dtype=float)
        self.B = np.zeros((m, m)) if B is None else np.asarray(B, dtype=float)
        self.endogenous = (np.zeros(m, dtype=bool) if endogenous is None
                           else np.asarray(endogenous, dtype=bool))
        p = self.Lambda.shape[0]
        self.item_names = item_names or [f"Y{i + 1}" for i in range(p)]
        self.item_ids = list(self.item_names)
        self.factor_names = factor_names or [f"F{i + 1}" for i in range(m)]
        self.factor_ids = list(self.factor_names)
        self.warnings = list(warnings or [])


def _two_factor_compiled(loading=0.8, phi=0.3, theta=None):
    Lambda = np.zeros((6, 2))
    Lambda[:3, 0] = loading
    Lambda[3:, 1] = loading
    Phi = np.array([[1.0, phi], [phi, 1.0]])
    comm = loading ** 2
    Theta = np.diag(np.full(6, 1.0 - comm if theta is None else theta))
    return FakeCompiled(Lambda, Phi, Theta)


def test_compiled_clean_model_has_no_errors():
    cm = _two_factor_compiled(loading=0.8, phi=0.3)
    # communality = 0.8^2 = 0.64, unique variance = 0.36, Sigma has unit diagonal
    Sigma = cm.Lambda @ cm.Phi @ cm.Lambda.T + cm.Theta
    assert np.allclose(np.diag(Sigma), 1.0)
    report = _record(validate_compiled(cm))
    assert report.errors() == [], [d.code for d in report.errors()]
    assert not report.has("LOW_COMMUNALITY")
    assert not report.has("SIGMA_ILL_CONDITIONED")
    assert report.ok


def test_compiled_heywood_case_realised_in_theta_and_psi():
    cm = _two_factor_compiled()
    cm.Theta[0, 0] = HEYWOOD_FLOOR          # exactly the compiler's clamp floor
    cm.endogenous = np.array([False, True])
    cm.Psi = np.diag([1.0, HEYWOOD_FLOOR])
    cm.warnings = ["Heywood clamp applied to F2"]
    report = _record(validate_compiled(cm))
    hits = report.by_code("HEYWOOD_CASE")
    assert len(hits) == 2, [d.message_en for d in hits]
    wheres = sorted(d.where for d in hits)
    assert wheres == ["F2", "Y1"], wheres
    assert all(d.severity is Severity.ERROR for d in hits)
    assert report.has("COMPILE_WARNING")
    assert report.by_code("COMPILE_WARNING")[0].severity is Severity.INFO
    _record(report)


def test_compiled_sigma_not_positive_definite_with_closed_form_eigenvalue():
    # Sigma = [[1.1, 1.2], [1.2, 1.1]];  eigenvalues of [[a, b], [b, a]] are
    # a + b and a - b, i.e. 2.3 and -0.1.
    Lambda = np.eye(2)
    Phi = np.array([[1.0, 1.2], [1.2, 1.0]])
    Theta = np.diag([0.1, 0.1])
    cm = FakeCompiled(Lambda, Phi, Theta)
    Sigma = Lambda @ Phi @ Lambda.T + Theta
    eig = np.linalg.eigvalsh(Sigma)
    assert abs(float(eig.min()) - (-0.1)) < 1e-12
    assert abs(float(eig.max()) - 2.3) < 1e-12

    report = _record(validate_compiled(cm))
    hits = report.by_code("SIGMA_NOT_PD")
    assert len(hits) == 1
    assert "-1.000e-01" in hits[0].message_en, hits[0].message_en
    assert not report.ok
    assert not report.has("HEYWOOD_CASE")          # Theta = 0.1 is admissible


def test_compiled_sigma_ill_conditioned_with_closed_form_condition_number():
    # Sigma = [[100.5, r], [r, 100.5]] with r = 100.5 - 1e-8.
    # eigenvalues 100.5 +/- r  ->  cond = (201 - 1e-8) / 1e-8 = 2.01e10 > 1e10.
    gap = 1e-8
    Lambda = np.diag([10.0, 10.0])
    r = 100.5 - gap
    Phi = np.array([[1.0, r / 100.0], [r / 100.0, 1.0]])
    Theta = np.diag([0.5, 0.5])
    cm = FakeCompiled(Lambda, Phi, Theta)
    Sigma = Lambda @ Phi @ Lambda.T + Theta
    eig = np.linalg.eigvalsh(Sigma)
    cond = float(eig.max() / eig.min())
    assert eig.min() > 1e-10, eig.min()            # still positive definite
    assert abs(cond / 2.01e10 - 1.0) < 0.02, cond
    assert cond > CONDITION_NUMBER_LIMIT

    report = _record(validate_compiled(cm))
    hits = report.by_code("SIGMA_ILL_CONDITIONED")
    assert len(hits) == 1, [d.code for d in report.diagnostics]
    assert hits[0].severity is Severity.WARNING
    assert not report.has("SIGMA_NOT_PD")


def test_compiled_r2_exceeds_one():
    cm = _two_factor_compiled()
    cm.endogenous = np.array([False, True])
    cm.B = np.array([[0.0, 0.0], [1.2, 0.0]])      # F2 regressed on F1 at 1.2
    cm.Phi = np.eye(2)
    # R^2 = b' Phi b = 1.2^2 = 1.44
    report = _record(validate_compiled(cm))
    hits = report.by_code("R2_EXCEEDS_ONE")
    assert len(hits) == 1
    assert "1.4400" in hits[0].message_en, hits[0].message_en
    assert hits[0].where == "F2"
    # a legal coefficient of 0.6 gives R^2 = 0.36 and no diagnostic
    cm.B = np.array([[0.0, 0.0], [0.6, 0.0]])
    assert not validate_compiled(cm).has("R2_EXCEEDS_ONE")


def test_compiled_low_communality():
    cm = _two_factor_compiled(loading=0.4, phi=0.3)
    # communality = 0.4^2 = 0.16 < 0.20 for all six items
    comm = cm.Lambda @ cm.Phi @ cm.Lambda.T
    assert np.allclose(np.diag(comm), 0.16)
    assert 0.16 < LOW_COMMUNALITY_FLOOR
    report = _record(validate_compiled(cm))
    hits = report.by_code("LOW_COMMUNALITY")
    assert len(hits) == 6, len(hits)
    assert "0.160" in hits[0].message_en
    assert report.errors() == [], [d.code for d in report.errors()]
    # at loading 0.45 the communality is 0.2025 and the flag clears
    cm2 = _two_factor_compiled(loading=0.45, phi=0.3)
    assert not validate_compiled(cm2).has("LOW_COMMUNALITY")


def test_validate_compiled_never_raises():
    class Broken:
        Lambda = "nonsense"
        Phi = None

    report = validate_compiled(Broken())           # type: ignore[arg-type]
    assert isinstance(report, ValidationReport)
    # a singular Lambda/Theta pair must not leak a LinAlgError either
    cm = FakeCompiled(np.ones((3, 1)), np.array([[1.0]]), np.zeros((3, 3)))
    rep = validate_compiled(cm)
    assert rep.has("SIGMA_NOT_PD")
    _record(rep)


# --------------------------------------------------------------------------- #
# Contract-level audits
# --------------------------------------------------------------------------- #


def test_every_code_is_covered():
    missing = [c for c in REQUIRED_SPEC_CODES + REQUIRED_COMPILED_CODES
               if c not in SEEN]
    assert not missing, f"codes never triggered by any test: {missing}"


def test_all_diagnostics_are_truly_bilingual():
    assert len(SEEN) >= len(REQUIRED_SPEC_CODES) + len(REQUIRED_COMPILED_CODES)
    for code, d in sorted(SEEN.items()):
        assert isinstance(d, Diagnostic)
        assert d.message_en.strip(), code
        assert d.message_ar.strip(), code
        assert d.message_ar != d.message_en, code
        assert ARABIC.search(d.message_ar), f"{code}: Arabic message has no Arabic script"
        assert not ARABIC.search(d.message_en), f"{code}: English message has Arabic script"
        assert d.hint_en.strip(), f"{code}: empty English hint"
        assert d.hint_ar.strip(), f"{code}: empty Arabic hint"
        assert d.hint_ar != d.hint_en, code
        assert ARABIC.search(d.hint_ar), f"{code}: Arabic hint has no Arabic script"
        # the Arabic message must be real prose, not a transliterated stub
        assert len(d.message_ar) >= 25, f"{code}: Arabic message too short"
        assert d.message(lang="ar") == d.message_ar
        assert d.message(lang="en") == d.message_en


def test_report_api_matches_the_contract():
    rep = ValidationReport([
        Diagnostic(Severity.ERROR, "A", "e", "ع", hint_en="h", hint_ar="ه"),
        Diagnostic(Severity.WARNING, "B", "e", "ع"),
        Diagnostic(Severity.INFO, "C", "e", "ع"),
    ])
    assert rep.ok is False
    assert [d.code for d in rep.errors()] == ["A"]
    assert [d.code for d in rep.warnings()] == ["B"]
    assert [d.code for d in rep.infos()] == ["C"]
    assert rep.codes() == ["A", "B", "C"]
    assert ValidationReport([]).ok is True
    assert Severity.ERROR.value == "error"
    assert Severity.WARNING.value == "warning"
    assert Severity.INFO.value == "info"
    merged = rep.merge(ValidationReport([Diagnostic(Severity.INFO, "D", "e", "ع")]))
    assert merged.codes() == ["A", "B", "C", "D"]
    fields = list(Diagnostic.__dataclass_fields__)
    assert fields == ["severity", "code", "message_en", "message_ar",
                      "where", "hint_en", "hint_ar"], fields


# --------------------------------------------------------------------------- #
# Stand-alone runner (pytest is not installed in every environment)
# --------------------------------------------------------------------------- #


def _main() -> int:
    ordered = [
        # the two audits must run last, they inspect what the others recorded
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
        and k not in ("test_every_code_is_covered",
                      "test_all_diagnostics_are_truly_bilingual")
    ]
    ordered += [test_every_code_is_covered, test_all_diagnostics_are_truly_bilingual]

    failures = 0
    for fn in ordered:
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")
    total = len(ordered)
    print(f"\n{total - failures}/{total} checks passed"
          f"{'' if not failures else f' ({failures} FAILED)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

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

import math
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
    "NON_FINITE_VALUE",
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
    assert s.count_free_parameters() == 27, s.count_free_parameters()
    assert s.degrees_of_freedom() == 51, s.degrees_of_freedom()

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
    # 6 loadings + 6 unique variances + 1 structural path. The latent
    # variances are scaling constraints, not estimates, so they are not here.
    assert s.count_free_parameters() == 13, s.count_free_parameters()
    for a, b in _distinct_pairs(names, 9):
        s.add_error_covariance(a, b, correlation=0.15)
    # each residual covariance adds exactly one free parameter: 13 + 9 = 22
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
    for a, b in _distinct_pairs(names, 8):
        s.add_error_covariance(a, b, correlation=0.15)
    assert s.count_free_parameters() == 21, s.count_free_parameters()
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
    assert (q, need) == (27, 135)

    s.power.n = need
    assert not validate(s).has("SAMPLE_SIZE_TOO_SMALL")
    s.power.n = need - 1
    report = _record(validate(s))
    assert report.has("SAMPLE_SIZE_TOO_SMALL")
    msg = report.by_code("SAMPLE_SIZE_TOO_SMALL")[0].message_en
    assert "N = 134" in msg and "N = 135" in msg


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
    """Communality is ``lambda' Phi lambda``, not ``sum(lambda^2)``.

    This test used to assert the orthogonal-factor sum of squares (0.81 + 0.25
    = 1.06) for a cross-loading onto ``M``, but in ``clean_spec`` the path
    X -> M = 0.30 makes X and M correlate 0.30, so the cross term
    ``2 * lambda_h * lambda_c * r`` is part of the communality the compiler
    actually realises (ARCHITECTURE step 6). The old expectation was the bug,
    not the contract, so the numbers below are the implied-metric ones.
    """
    s = clean_spec()
    item = s.all_items()[0]
    item.loading = 0.9
    s.add_cross_loading(item.name, "M", loading=0.5)
    # implied corr(X, M) = 0.30, so
    # communality = 0.81 + 0.25 + 2*0.9*0.5*0.30 = 1.06 + 0.27 = 1.33
    report = _record(validate(s))
    hits = report.by_code("HEYWOOD_RISK")
    assert len(hits) == 1, [d.message_en for d in hits]
    assert "1.330" in hits[0].message_en, hits[0].message_en
    assert hits[0].where == item.id

    # 0.81 + 0.04 + 2*0.9*0.2*0.30 = 0.85 + 0.108 = 0.958 < 1 -> admissible
    s.cross_loadings[0].loading = 0.2
    assert not validate(s).has("HEYWOOD_RISK")

    # the method factor is orthogonal to every substantive factor, so it adds
    # exactly its own square: 0.958 + 0.3^2 = 1.048
    s.common_method_bias.enabled = True
    s.common_method_bias.loading = 0.3
    rep2 = validate(s)
    assert rep2.has("HEYWOOD_RISK")
    assert "1.048" in rep2.by_code("HEYWOOD_RISK")[0].message_en, \
        rep2.by_code("HEYWOOD_RISK")[0].message_en


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
# Regressions: the implied latent correlations (not just the declared ones)
# --------------------------------------------------------------------------- #


def test_r2_uses_the_correlations_the_structural_model_induces():
    """A mediation model correlates its own predictors.

    X -> M = 0.80 means X and M correlate 0.80 even though no
    ``LatentCovariance`` says so, so the R^2 of Y is
    ``b' Phi b`` with ``Phi[X, M] = 0.80``, not ``b' I b``.  Building R from
    the declared covariances alone made this model - whose disturbance
    variance is unambiguously negative, and which the compiler has to repair
    with a Heywood clamp - validate completely clean.
    """
    s = ModelSpec()
    s.add_construct("X", n_items=4, loading=0.7)
    s.add_construct("M", n_items=4, loading=0.7, role=ConstructRole.MEDIATOR)
    s.add_construct("Y", n_items=4, loading=0.7, role=ConstructRole.DEPENDENT)
    s.add_path("X", "M", effect_size=EffectSize.CUSTOM, coefficient=0.80)
    s.add_path("X", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.70)
    s.add_path("M", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.70)

    b = np.array([0.70, 0.70])
    R = np.array([[1.0, 0.80], [0.80, 1.0]])
    assert abs(float(b @ R @ b) - 1.764) < 1e-12        # 0.98 + 0.784

    report = _record(validate(s))
    hits = report.by_code("R2_EXCEEDS_ONE")
    assert len(hits) == 1, [d.message_en for d in hits]
    assert "1.764" in hits[0].message_en, hits[0].message_en
    assert hits[0].where == s.construct_by_name("Y").id
    assert not report.ok

    # M itself is fine: R^2 = 0.80^2 = 0.64
    assert hits[0].where != s.construct_by_name("M").id
    # and the same model with admissible coefficients raises nothing
    s.paths[1].coefficient = 0.20
    s.paths[2].coefficient = 0.30
    assert not validate(s).has("R2_EXCEEDS_ONE")


def test_r2_matches_the_compiler_on_a_chain_of_three_predictors():
    """The validator's quadratic form and the compiler's must agree."""
    try:
        from semsim.core.compile import compile_model
    except Exception as exc:                      # pragma: no cover
        print(f"    [skip] semsim.core.compile unavailable: {exc!r}")
        return

    s = ModelSpec()
    s.add_construct("X", n_items=3, loading=0.7)
    s.add_construct("M1", n_items=3, loading=0.7, role=ConstructRole.MEDIATOR)
    s.add_construct("M2", n_items=3, loading=0.7, role=ConstructRole.MEDIATOR)
    s.add_construct("Y", n_items=3, loading=0.7, role=ConstructRole.DEPENDENT)
    s.add_path("X", "M1", effect_size=EffectSize.CUSTOM, coefficient=0.60)
    s.add_path("M1", "M2", effect_size=EffectSize.CUSTOM, coefficient=0.55)
    s.add_path("X", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.35)
    s.add_path("M1", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.30)
    s.add_path("M2", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.25)
    assert not validate(s).has("R2_EXCEEDS_ONE")

    cm = compile_model(s)
    yi = cm.factor_index[s.construct_by_name("Y").id]
    compiled_r2 = float(cm.B[yi] @ cm.Phi @ cm.B[yi])
    # The validator's own number, read back through its diagnostic: push the
    # same model over the edge by a known factor and check the figure it
    # prints equals the compiler's quadratic form, scaled the same way.
    scale = 1.0 / math.sqrt(compiled_r2)
    for p in s.paths:
        if p.target == s.construct_by_name("Y").id:
            p.coefficient = p.coefficient * scale * 1.02
    hits = validate(s).by_code("R2_EXCEEDS_ONE")
    assert len(hits) == 1, [d.message_en for d in hits]
    expected = compiled_r2 * (scale * 1.02) ** 2         # = 1.02^2 = 1.0404
    assert abs(expected - 1.0404) < 1e-9, expected
    assert f"{expected:.3f}" in hits[0].message_en, hits[0].message_en


def test_duplicate_paths_are_not_summed_and_follow_the_compiler():
    """Two X -> Y paths at 0.50 compile to 0.50, not to 1.00.

    ARCHITECTURE compile step 2 assigns ``B[t, s] = resolved_coefficient()``,
    so the last declaration wins.  Summing them inside the validator reported
    R2_EXCEEDS_ONE (1.00) on a model whose compiled R^2 is 0.25.
    """
    s = ModelSpec()
    s.add_construct("X", n_items=4, loading=0.7)
    s.add_construct("Y", n_items=4, loading=0.7, role=ConstructRole.DEPENDENT)
    s.add_path("X", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.50)
    s.add_path("X", "Y", effect_size=EffectSize.CUSTOM, coefficient=0.50)

    report = _record(validate(s))
    assert not report.has("R2_EXCEEDS_ONE"), [
        d.message_en for d in report.by_code("R2_EXCEEDS_ONE")]
    # the user still has to be told that the earlier path is being ignored
    dups = report.by_code("DUPLICATE_PATH")
    assert len(dups) == 1
    assert dups[0].where == s.paths[1].id
    assert "ignored" in dups[0].message_en

    # the surviving coefficient is the LAST one, so the order matters
    s.paths[1].coefficient = 0.995
    assert not validate(s).has("R2_EXCEEDS_ONE")        # 0.995^2 = 0.990
    s.paths[0].coefficient = 0.10                        # the ignored one
    assert not validate(s).has("R2_EXCEEDS_ONE")
    s.paths[1].coefficient = 1.01                        # the live one
    assert validate(s).has("R2_EXCEEDS_ONE")             # 1.01^2 = 1.020

    try:
        from semsim.core.compile import compile_model
    except Exception as exc:                      # pragma: no cover
        print(f"    [skip] semsim.core.compile unavailable: {exc!r}")
        return
    s.paths[0].coefficient = 0.50
    s.paths[1].coefficient = 0.50
    cm = compile_model(s)
    yi = cm.factor_index[s.construct_by_name("Y").id]
    xi = cm.factor_index[s.construct_by_name("X").id]
    assert abs(float(cm.B[yi, xi]) - 0.50) < 1e-12, cm.B[yi, xi]


def test_heywood_risk_cross_loading_resolves_the_implied_correlation():
    """``2*lambda_h*lambda_c*r`` needs the *implied* r, in both directions.

    Two dimensions of one second-order construct correlate
    ``sol_1 * sol_2`` through their parent, and two constructs correlate
    through the structural paths between them.  Resolving ``r`` against the
    declared latent covariances only returned 0 in both cases, so the
    communality came out below 1 and no Heywood case was reported.
    """
    # (a) two dimensions of the same parent: r = 0.9 * 0.9 = 0.81
    s = ModelSpec()
    c = s.add_second_order_construct("SO", n_dimensions=3,
                                     items_per_dimension=4, loading=0.7,
                                     second_order_loading=0.9)
    item = c.dimensions[0].items[0]
    item.loading = 0.8
    s.add_cross_loading(item.id, c.dimensions[1].id, loading=0.5)
    # 0.64 + 0.25 + 2*0.8*0.5*0.81 = 0.89 + 0.648 = 1.538
    report = _record(validate(s))
    hits = report.by_code("HEYWOOD_RISK")
    assert len(hits) == 1, [d.message_en for d in hits]
    assert "1.538" in hits[0].message_en, hits[0].message_en
    assert hits[0].where == item.id

    # (b) the same cross-loading onto a genuinely orthogonal construct keeps
    #     the plain sum of squares: 0.64 + 0.25 = 0.89 < 1
    s2 = ModelSpec()
    c2 = s2.add_second_order_construct("SO", n_dimensions=3,
                                       items_per_dimension=4, loading=0.7,
                                       second_order_loading=0.9)
    s2.add_construct("Z", n_items=4, loading=0.7)
    it2 = c2.dimensions[0].items[0]
    it2.loading = 0.8
    s2.add_cross_loading(it2.id, s2.construct_by_name("Z").id, loading=0.5)
    assert not validate(s2).has("HEYWOOD_RISK")

    # (c) two constructs correlated only through a structural path:
    #     clean_spec has X -> M = 0.30 and M -> Y = 0.30, X -> Y = 0.15, so
    #     implied corr(X, Y) = 0.15 + 0.30*0.30 = 0.24 and the communality is
    #     0.81 + 0.16 + 2*0.9*0.4*0.24 = 0.97 + 0.1728 = 1.1428
    s3 = clean_spec()
    it3 = s3.all_items()[0]
    it3.loading = 0.9
    s3.add_cross_loading(it3.id, s3.construct_by_name("Y").id, loading=0.4)
    hits3 = validate(s3).by_code("HEYWOOD_RISK")
    assert len(hits3) == 1, [d.message_en for d in hits3]
    assert "1.143" in hits3[0].message_en, hits3[0].message_en


def test_duplicate_cross_loadings_are_not_summed_either():
    """compile.py writes ``Lambda[j, fi] = cl.loading``: the last one wins."""
    s = clean_spec()
    item = s.all_items()[0]
    item.loading = 0.7
    s.add_cross_loading(item.id, s.construct_by_name("M").id, loading=0.6)
    s.add_cross_loading(item.id, s.construct_by_name("M").id, loading=0.1)
    # live row is (0.7 on X, 0.1 on M): 0.49 + 0.01 + 2*0.7*0.1*0.30 = 0.542
    assert not validate(s).has("HEYWOOD_RISK")


# --------------------------------------------------------------------------- #
# Regression: NaN / infinity must not validate clean
# --------------------------------------------------------------------------- #


def test_non_finite_values_are_reported():
    """``_num`` silently coerced NaN / inf to 0.0 and nothing said so.

    A spec carrying a non-finite number used to produce zero diagnostics and
    then poison Lambda, Sigma, the Cholesky factor and every generated column.
    """
    s = clean_spec()
    assert not validate(s).has("NON_FINITE_VALUE")

    items = s.all_items()
    items[0].loading = float("nan")
    items[1].extra_error_variance = float("inf")
    items[2].scale.thresholds = [-1.0, 0.0, float("nan"), 1.0]
    s.paths[0].effect_size = EffectSize.CUSTOM
    s.paths[0].coefficient = float("inf")
    s.add_error_covariance(items[4].id, items[5].id, correlation=float("nan"))
    s.latent_covariances.append(LatentCovariance(
        construct_a=s.construct_by_name("X").id,
        construct_b=s.construct_by_name("M").id,
        correlation=float("-inf")))
    s.noise.missing_mechanism = MissingMechanism.MCAR
    s.noise.missing_proportion = float("nan")

    report = _record(validate(s))
    hits = report.by_code("NON_FINITE_VALUE")
    assert len(hits) == 7, [d.message_en for d in hits]
    assert all(d.severity is Severity.ERROR for d in hits)
    assert not report.ok

    by_field = {d.message_en: d for d in hits}
    assert any("standardised loading" in m and f"'{items[0].name}'" in m
               for m in by_field), list(by_field)
    assert any("extra error variance" in m and "+infinity" in m
               for m in by_field), list(by_field)
    assert any("threshold number 3" in m and "NaN" in m
               for m in by_field), list(by_field)
    assert any("the path 'X' -> 'M'" in m for m in by_field), list(by_field)
    assert any("residual covariance between items" in m for m in by_field)
    assert any("free covariance between 'X' and 'M'" in m and "-infinity" in m
               for m in by_field), list(by_field)
    assert any("proportion of missing data" in m for m in by_field)

    # every finding points at the object that carries the bad number
    wheres = {d.where for d in hits}
    assert items[0].id in wheres
    assert items[2].id in wheres
    assert s.paths[0].id in wheres

    # and the spec-level checks still ran instead of crashing
    assert isinstance(report, ValidationReport)
    assert validate(s).has("NON_FINITE_VALUE")


def test_non_finite_value_covers_the_global_settings_too():
    s = clean_spec()
    s.distribution.default_kurtosis = float("nan")
    s.fit_target.rmsea = float("inf")
    s.significance_target.alpha = float("nan")
    s.power.n = float("nan")
    hits = _record(validate(s)).by_code("NON_FINITE_VALUE")
    fields = " | ".join(d.message_en for d in hits)
    assert len(hits) == 4, fields
    for needle in ("default excess kurtosis", "target RMSEA",
                   "significance level alpha", "planned sample size"):
        assert needle in fields, (needle, fields)
    # a value that simply is not a number (an empty editor) is not reported
    # here: the field-specific range checks describe that case.
    s2 = clean_spec()
    s2.all_items()[0].loading = ""                   # type: ignore[assignment]
    assert not validate(s2).has("NON_FINITE_VALUE")


# --------------------------------------------------------------------------- #
# Regression: the sample-size check must be monotone in N
# --------------------------------------------------------------------------- #


def test_sample_size_check_is_monotone_in_n():
    """``N < 5q`` is the contract; N = 0 and N = -50 satisfy it.

    The old ``n > 0`` guard made the check non-monotone: N = 1 and N = 134
    were flagged while N = 0 and N = -50 - strictly worse - were not.  Both
    are now reported under SAMPLE_SIZE_TOO_SMALL.
    """
    s = clean_spec()
    q = s.count_free_parameters()
    need = MIN_CASES_PER_PARAMETER * q
    assert (q, need) == (27, 135)

    flagged = []
    for n in (-50, -1, 0, 1, 2, 134, 135, 136, 1000):
        s.power.n = n
        flagged.append(validate(s).has("SAMPLE_SIZE_TOO_SMALL"))
    assert flagged == [True, True, True, True, True, True, False, False, False], \
        flagged

    s.power.n = 0
    report = _record(validate(s))
    hits = report.by_code("SAMPLE_SIZE_TOO_SMALL")
    assert len(hits) == 1
    assert "N = 0" in hits[0].message_en and "N = 135" in hits[0].message_en
    assert "cannot be drawn" in hits[0].message_en
    assert hits[0].severity is Severity.WARNING

    s.power.n = -50
    msg = validate(s).by_code("SAMPLE_SIZE_TOO_SMALL")[0].message_en
    assert "N = -50" in msg, msg


# --------------------------------------------------------------------------- #
# Regression: Diagnostic.where is an id, not a display name
# --------------------------------------------------------------------------- #


def test_compiled_where_is_the_id_not_the_display_name():
    """``where`` is the construct / item / path id per ARCHITECTURE.

    ``validate_compiled`` used to put the display name there, so the UI could
    not select the offending object (and two items may even share a name).
    The name belongs in the message, which is where a human reads it.
    """
    cm = _two_factor_compiled(loading=0.4, phi=0.3)
    cm.item_names = ["Sat1", "Sat2", "Sat3", "Loy1", "Loy2", "Loy3"]
    cm.item_ids = [f"it_{k}" for k in range(1, 7)]
    cm.factor_names = ["Satisfaction", "Loyalty"]
    cm.factor_ids = ["c_sat", "c_loy"]
    cm.Theta[0, 0] = HEYWOOD_FLOOR
    cm.endogenous = np.array([False, True])
    cm.Psi = np.diag([1.0, HEYWOOD_FLOOR])
    cm.B = np.array([[0.0, 0.0], [1.2, 0.0]])
    cm.Phi = np.eye(2)

    report = _record(validate_compiled(cm))
    known_ids = set(cm.item_ids) | set(cm.factor_ids)
    display_names = set(cm.item_names) | set(cm.factor_names)
    for d in report.diagnostics:
        if d.code == "COMPILE_WARNING":
            continue
        assert d.where in known_ids, (d.code, d.where)
        assert d.where not in display_names, (d.code, d.where)

    heywood = report.by_code("HEYWOOD_CASE")
    assert sorted(d.where for d in heywood) == ["c_loy", "it_1"], \
        [d.where for d in heywood]
    # the human-readable name is still in the message
    assert any("Sat1" in d.message_en and "Sat1" in d.message_ar
               for d in heywood)
    assert any("Loyalty" in d.message_en for d in heywood)

    r2 = report.by_code("R2_EXCEEDS_ONE")
    assert len(r2) == 1 and r2[0].where == "c_loy"
    assert "Loyalty" in r2[0].message_en

    low = report.by_code("LOW_COMMUNALITY")
    assert sorted(d.where for d in low) == cm.item_ids, [d.where for d in low]
    assert "Sat1" in low[0].message_en


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

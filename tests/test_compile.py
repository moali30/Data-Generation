"""
Numerical tests for :mod:`semsim.core.compile` and :mod:`semsim.core.implied`.

Every assertion here is against a closed-form value worked out by hand from the
standardisation algebra, not merely against "the code ran":

* ``Phi`` and ``Sigma`` must have an exact unit diagonal for every model shape;
* a two-factor model with loadings 0.8 and a path of 0.5 must imply an
  inter-item correlation of ``0.8 * 0.5 * 0.8 = 0.32`` across factors and
  ``0.8 * 0.8 = 0.64`` within one;
* a second-order construct with dimension loadings 0.8 and item loadings 0.7
  must imply ``0.7 * 0.8 * 0.8 * 0.7 = 0.3136`` between sibling dimensions;
* mediation 0.5 -> 0.4 must imply an indirect effect of 0.20;
* a factor with k items all loading lambda must give ``AVE == lambda ** 2``
  and ``CR == (k lambda)^2 / ((k lambda)^2 + k (1 - lambda^2))``.

Run with::

    python3 -m pytest tests/test_compile.py -q

The file also works as a plain script (``python3 tests/test_compile.py``) and
then prints PASS/FAIL per check and exits non-zero on failure.
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings

import numpy as np

# Allow ``python3 tests/test_compile.py`` as a fallback when pytest is absent.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semsim.core import implied
from semsim.core.compile import HEYWOOD_FLOOR, compile_model
from semsim.core.errors import SpecificationError
from semsim.core.spec import (
    CommonMethodBias,
    ConstructRole,
    CrossLoading,
    ErrorCovariance,
    LatentCovariance,
    ModelSpec,
)

ATOL = 1e-9


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _first_order() -> ModelSpec:
    spec = ModelSpec(seed=7)
    spec.add_construct("X", n_items=4, loading=0.8)
    spec.add_construct("Y", n_items=4, loading=0.8, role=ConstructRole.DEPENDENT)
    spec.add_path("X", "Y", coefficient=0.5)
    return spec


def _two_factor_simple() -> ModelSpec:
    """Loadings all 0.8, one path of 0.5 -- the hand-algebra reference model."""
    spec = ModelSpec(seed=11)
    spec.add_construct("F1", n_items=3, loading=0.8)
    spec.add_construct("F2", n_items=3, loading=0.8, role=ConstructRole.DEPENDENT)
    spec.add_path("F1", "F2", coefficient=0.5)
    return spec


def _second_order() -> ModelSpec:
    """Three dimensions loading 0.8 on the construct, items loading 0.7."""
    spec = ModelSpec(seed=13)
    spec.add_second_order_construct(
        "SO", n_dimensions=3, items_per_dimension=3,
        loading=0.7, second_order_loading=0.8)
    return spec


def _mediation() -> ModelSpec:
    spec = ModelSpec(seed=17)
    spec.add_construct("X", n_items=3, loading=0.8)
    spec.add_construct("M", n_items=3, loading=0.8, role=ConstructRole.MEDIATOR)
    spec.add_construct("Y", n_items=3, loading=0.8, role=ConstructRole.DEPENDENT)
    spec.add_path("X", "M", coefficient=0.5)
    spec.add_path("M", "Y", coefficient=0.4)
    return spec


def _double_mediation() -> ModelSpec:
    spec = ModelSpec(seed=19)
    spec.add_construct("X", n_items=3, loading=0.8)
    spec.add_construct("M1", n_items=3, loading=0.8, role=ConstructRole.MEDIATOR)
    spec.add_construct("M2", n_items=3, loading=0.8, role=ConstructRole.MEDIATOR)
    spec.add_construct("Y", n_items=3, loading=0.8, role=ConstructRole.DEPENDENT)
    spec.add_path("X", "M1", coefficient=0.5)
    spec.add_path("M1", "M2", coefficient=0.4)
    spec.add_path("M2", "Y", coefficient=0.3)
    spec.add_path("X", "Y", coefficient=0.2)
    return spec


def _correlated_exogenous() -> ModelSpec:
    spec = ModelSpec(seed=23)
    a = spec.add_construct("A", n_items=3, loading=0.75)
    b = spec.add_construct("B", n_items=3, loading=0.75)
    y = spec.add_construct("Y", n_items=3, loading=0.75,
                           role=ConstructRole.DEPENDENT)
    spec.add_path("A", "Y", coefficient=0.4)
    spec.add_path("B", "Y", coefficient=0.3)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=a.id, construct_b=b.id, correlation=0.35))
    assert y is not None
    return spec


def _cross_loaded() -> ModelSpec:
    spec = ModelSpec(seed=29)
    f1 = spec.add_construct("F1", n_items=3, loading=0.8)
    f2 = spec.add_construct("F2", n_items=3, loading=0.7,
                            role=ConstructRole.DEPENDENT)
    spec.add_path("F1", "F2", coefficient=0.5)
    spec.cross_loadings.append(
        CrossLoading(item_id=f1.items[0].id, factor_id=f2.id, loading=0.30))
    return spec


def _error_covariances() -> ModelSpec:
    spec = ModelSpec(seed=31)
    f = spec.add_construct("F", n_items=4, loading=0.7)
    spec.add_construct("G", n_items=4, loading=0.7, role=ConstructRole.DEPENDENT)
    spec.add_path("F", "G", coefficient=0.45)
    spec.error_covariances.append(
        ErrorCovariance(item_a=f.items[0].id, item_b=f.items[1].id,
                        correlation=0.30))
    return spec


def _method_factor() -> ModelSpec:
    spec = ModelSpec(seed=37)
    spec.add_construct("P", n_items=3, loading=0.8)
    spec.add_construct("Q", n_items=3, loading=0.7, role=ConstructRole.DEPENDENT)
    spec.add_path("P", "Q", coefficient=0.4)
    spec.common_method_bias = CommonMethodBias(enabled=True, loading=0.30)
    return spec


def _heywood() -> ModelSpec:
    """Two predictors of 0.9 each, correlated 0.8: R^2 = 2.916."""
    spec = ModelSpec(seed=41)
    a = spec.add_construct("A", n_items=3, loading=0.7)
    b = spec.add_construct("B", n_items=3, loading=0.7)
    spec.add_construct("Y", n_items=3, loading=0.7, role=ConstructRole.DEPENDENT)
    spec.add_path("A", "Y", coefficient=0.9)
    spec.add_path("B", "Y", coefficient=0.9)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=a.id, construct_b=b.id, correlation=0.8))
    return spec


def _non_recursive() -> ModelSpec:
    """A reciprocal loop: A -> B at 0.5 and B -> A at 0.3."""
    spec = ModelSpec(seed=43)
    spec.add_construct("A", n_items=3, loading=0.8)
    spec.add_construct("B", n_items=3, loading=0.8)
    spec.add_path("A", "B", coefficient=0.5)
    spec.add_path("B", "A", coefficient=0.3)
    return spec


ALL_BUILDERS = {
    "first_order": _first_order,
    "two_factor_simple": _two_factor_simple,
    "second_order": _second_order,
    "mediation": _mediation,
    "double_mediation": _double_mediation,
    "correlated_exogenous": _correlated_exogenous,
    "cross_loaded": _cross_loaded,
    "error_covariances": _error_covariances,
    "method_factor": _method_factor,
    "heywood": _heywood,
    "non_recursive": _non_recursive,
}


# --------------------------------------------------------------------------- #
# The acid tests: unit diagonals everywhere
# --------------------------------------------------------------------------- #


def test_phi_has_unit_diagonal_for_every_model_shape() -> None:
    for name, build in ALL_BUILDERS.items():
        cm = compile_model(build())
        assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL), name
        assert np.allclose(cm.Phi, cm.Phi.T, atol=ATOL), name


def test_sigma_has_unit_diagonal_for_every_model_shape() -> None:
    for name, build in ALL_BUILDERS.items():
        cm = compile_model(build())
        sigma = implied.implied_covariance(cm)
        assert np.allclose(np.diag(sigma), 1.0, atol=ATOL), name
        assert np.allclose(sigma, sigma.T, atol=ATOL), name


def test_sigma_is_symmetric_and_positive_definite() -> None:
    for name, build in ALL_BUILDERS.items():
        cm = compile_model(build())
        sigma = implied.implied_covariance(cm)
        assert implied.is_positive_definite(sigma), name
        assert np.linalg.eigvalsh(sigma).min() > 0.0, name
        assert implied.is_positive_definite(cm.Phi), name


def test_phi_matches_the_lisrel_identity() -> None:
    """Phi must really be ``(I - B)^-1 Psi (I - B)^-T``, not a patched matrix."""
    for name, build in ALL_BUILDERS.items():
        cm = compile_model(build())
        inv = np.linalg.inv(np.eye(cm.n_factors) - cm.B)
        assert np.allclose(cm.Phi, inv @ cm.Psi @ inv.T, atol=1e-8), name


def test_auto_generated_cross_loadings_keep_the_unit_diagonal() -> None:
    spec = _first_order()
    spec.cross_loading_policy.enabled = True
    spec.cross_loading_policy.proportion_of_items = 0.5
    spec.cross_loading_policy.magnitude_range = (0.15, 0.30)
    cm = compile_model(spec, np.random.default_rng(2024))
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)
    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    # Four of the eight items picked up a secondary loading.
    extra = sum(1 for j in range(cm.n_items)
                if np.count_nonzero(cm.Lambda[j]) > 1)
    assert extra == 4, extra


# --------------------------------------------------------------------------- #
# Closed-form measurement algebra
# --------------------------------------------------------------------------- #


def test_two_factor_hand_algebra_032_and_064() -> None:
    cm = compile_model(_two_factor_simple())
    sigma = implied.implied_covariance(cm)
    f1 = cm.items_of(cm.factor_ids[cm.factor_index[
        cm.spec.construct_by_name("F1").id]])
    f2 = cm.items_of(cm.spec.construct_by_name("F2").id)
    assert len(f1) == 3 and len(f2) == 3

    # Phi: the structural path is the latent correlation for a bivariate model.
    assert abs(cm.Phi[cm.factor_index[cm.spec.construct_by_name("F2").id],
                      cm.factor_index[cm.spec.construct_by_name("F1").id]]
               - 0.5) < ATOL

    within = [sigma[a, b] for a in f1 for b in f1 if a != b]
    within += [sigma[a, b] for a in f2 for b in f2 if a != b]
    across = [sigma[a, b] for a in f1 for b in f2]

    assert np.allclose(within, 0.8 * 0.8, atol=ATOL), within
    assert np.allclose(across, 0.8 * 0.5 * 0.8, atol=ATOL), across
    assert abs(np.mean(across) - 0.32) < ATOL
    assert abs(np.mean(within) - 0.64) < ATOL

    # Residual variance is exactly 1 - 0.64 for every item.
    assert np.allclose(np.diag(cm.Theta), 1.0 - 0.64, atol=ATOL)
    # Disturbance variance of the endogenous factor is 1 - 0.5 ** 2.
    f2_idx = cm.factor_index[cm.spec.construct_by_name("F2").id]
    assert abs(cm.Psi[f2_idx, f2_idx] - 0.75) < ATOL


def test_second_order_sibling_dimension_item_correlation_is_03136() -> None:
    cm = compile_model(_second_order())
    sigma = implied.implied_covariance(cm)
    construct = cm.spec.constructs[0]
    dims = [d.id for d in construct.dimensions]
    rows = {d: cm.items_of(d) for d in dims}

    # Sibling dimensions correlate 0.8 * 0.8 = 0.64 at the latent level.
    i0, i1 = cm.factor_index[dims[0]], cm.factor_index[dims[1]]
    assert abs(cm.Phi[i0, i1] - 0.64) < ATOL

    sibling = [sigma[a, b] for a in rows[dims[0]] for b in rows[dims[1]]]
    assert np.allclose(sibling, 0.7 * 0.8 * 0.8 * 0.7, atol=ATOL), sibling
    assert abs(np.mean(sibling) - 0.3136) < ATOL

    same = [sigma[a, b] for a in rows[dims[0]] for b in rows[dims[0]] if a != b]
    assert np.allclose(same, 0.49, atol=ATOL), same

    # A dimension explains 0.64 of itself, so its disturbance is 0.36.
    assert abs(cm.Psi[i0, i0] - 0.36) < ATOL


def test_cross_loading_contribution_is_exact() -> None:
    cm = compile_model(_cross_loaded())
    sigma = implied.implied_covariance(cm)
    f1 = cm.spec.construct_by_name("F1")
    f2 = cm.spec.construct_by_name("F2")
    j = cm.item_index[f1.items[0].id]
    other_f1 = cm.item_index[f1.items[1].id]
    f2_rows = cm.items_of(f2.id)

    # Communality = 0.8^2 + 0.3^2 + 2 * 0.8 * 0.3 * 0.5 = 0.97.
    assert abs(cm.Theta[j, j] - (1.0 - 0.97)) < ATOL
    # With the F2 item: 0.8 * 0.5 * 0.7 + 0.3 * 1 * 0.7 = 0.49.
    assert np.allclose(sigma[j, f2_rows], 0.49, atol=ATOL)
    # With a clean F1 item: 0.8 * 0.8 + 0.3 * 0.5 * 0.8 = 0.76.
    assert abs(sigma[j, other_f1] - 0.76) < ATOL


def test_error_covariance_uses_the_standardised_formula() -> None:
    spec = _error_covariances()
    cm = compile_model(spec)
    ec = spec.error_covariances[0]
    a = cm.item_index[ec.item_a]
    b = cm.item_index[ec.item_b]
    expected = 0.30 * math.sqrt(cm.Theta[a, a] * cm.Theta[b, b])
    assert abs(cm.Theta[a, b] - expected) < ATOL
    assert abs(cm.Theta[b, a] - expected) < ATOL
    # Theta[j, j] == 1 - 0.49, so the covariance is 0.30 * 0.51.
    assert abs(cm.Theta[a, a] - 0.51) < ATOL
    assert abs(cm.Theta[a, b] - 0.30 * 0.51) < ATOL
    # And the two items' total correlation is the common part plus the residual.
    sigma = implied.implied_covariance(cm)
    assert abs(sigma[a, b] - (0.49 + 0.30 * 0.51)) < ATOL


def test_common_method_factor_adds_a_shared_component() -> None:
    cm = compile_model(_method_factor())
    sigma = implied.implied_covariance(cm)
    assert cm.method_factor_index is not None
    p_rows = cm.items_of(cm.spec.construct_by_name("P").id)
    q_rows = cm.items_of(cm.spec.construct_by_name("Q").id)

    # The method factor is exogenous and uncorrelated with the substantive ones.
    mi = cm.method_factor_index
    off = np.delete(cm.Phi[mi], mi)
    assert np.allclose(off, 0.0, atol=ATOL)

    # Communality of a P item: 0.8^2 + 0.3^2 = 0.73.
    assert abs(cm.Theta[p_rows[0], p_rows[0]] - (1.0 - 0.73)) < ATOL
    # P-Q correlation: 0.8 * 0.4 * 0.7 + 0.3 * 0.3 = 0.224 + 0.09 = 0.314.
    assert np.allclose(sigma[np.ix_(p_rows, q_rows)], 0.314, atol=ATOL)
    # P-P correlation: 0.8 * 0.8 + 0.3 * 0.3 = 0.73.
    assert abs(sigma[p_rows[0], p_rows[1]] - 0.73) < ATOL


def test_unreliability_attenuates_the_loading_and_renormalises() -> None:
    spec = ModelSpec(seed=5)
    f = spec.add_construct("F", n_items=4, loading=0.8)
    f.items[0].extra_error_variance = 0.20
    cm = compile_model(spec)
    j = cm.item_index[f.items[0].id]
    # Theta was 0.36, inflated to 0.56, total 1.20.
    assert abs(cm.Lambda[j, cm.home_factor[j]] - 0.8 / math.sqrt(1.2)) < ATOL
    assert abs(cm.Theta[j, j] - 0.56 / 1.2) < ATOL
    sigma = implied.implied_covariance(cm)
    assert abs(sigma[j, j] - 1.0) < ATOL
    # The attenuated correlation with a clean sibling: (0.8/sqrt(1.2)) * 0.8.
    other = cm.item_index[f.items[1].id]
    assert abs(sigma[j, other] - 0.8 / math.sqrt(1.2) * 0.8) < ATOL


def test_global_variance_inflation_attenuates_every_item() -> None:
    spec = ModelSpec(seed=5)
    f = spec.add_construct("F", n_items=3, loading=0.8)
    spec.noise.unique_variance_inflation = 1.5
    cm = compile_model(spec)
    total = 0.64 + 0.36 * 1.5           # 1.18
    assert np.allclose(cm.Lambda[:, cm.home_factor[0]],
                       0.8 / math.sqrt(total), atol=ATOL)
    assert np.allclose(np.diag(cm.Theta), 0.54 / total, atol=ATOL)
    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert abs(sigma[0, 1] - 0.64 / total) < ATOL
    assert f is not None


# --------------------------------------------------------------------------- #
# Structural effects
# --------------------------------------------------------------------------- #


def test_mediation_total_and_indirect_effects_are_020() -> None:
    spec = _mediation()
    cm = compile_model(spec)
    x = spec.construct_by_name("X").id
    mm = spec.construct_by_name("M").id
    y = spec.construct_by_name("Y").id

    total = implied.total_effects(cm)
    xi, mi, yi = (cm.factor_index[x], cm.factor_index[mm], cm.factor_index[y])

    assert abs(total[yi, xi] - 0.20) < ATOL     # no direct path, so total == indirect
    assert abs(total[mi, xi] - 0.50) < ATOL
    assert abs(total[yi, mi] - 0.40) < ATOL
    assert np.allclose(np.diag(total), 0.0, atol=ATOL)

    assert abs(implied.indirect_effect(cm, [x, mm, y]) - 0.20) < ATOL
    assert abs(implied.direct_effect(cm, x, mm) - 0.50) < ATOL
    assert abs(implied.direct_effect(cm, x, y) - 0.0) < ATOL

    chains = implied.all_indirect_effects(cm, x, y)
    assert len(chains) == 1
    assert chains[0][0] == [x, mm, y]
    assert abs(chains[0][1] - 0.20) < ATOL

    # And the latent correlation implied by the chain: r(X, Y) = 0.5 * 0.4.
    assert abs(cm.Phi[yi, xi] - 0.20) < ATOL


def test_two_mediator_chains_and_total_effect_decomposition() -> None:
    spec = _double_mediation()
    cm = compile_model(spec)
    x = spec.construct_by_name("X").id
    y = spec.construct_by_name("Y").id
    m1 = spec.construct_by_name("M1").id
    m2 = spec.construct_by_name("M2").id

    chains = implied.all_indirect_effects(cm, x, y, max_len=4)
    assert len(chains) == 1
    assert chains[0][0] == [x, m1, m2, y]
    assert abs(chains[0][1] - 0.5 * 0.4 * 0.3) < ATOL

    # max_len = 3 admits no chain with two mediators.
    assert implied.all_indirect_effects(cm, x, y, max_len=3) == []

    total = implied.total_effects(cm)
    expected = 0.2 + 0.5 * 0.4 * 0.3
    assert abs(total[cm.factor_index[y], cm.factor_index[x]] - expected) < ATOL
    assert abs(implied.indirect_effect(cm, [x, m1, m2, y]) - 0.06) < ATOL


def test_indirect_effect_rejects_a_broken_chain() -> None:
    spec = _mediation()
    cm = compile_model(spec)
    x = spec.construct_by_name("X").id
    y = spec.construct_by_name("Y").id
    try:
        implied.indirect_effect(cm, [x, y])
    except SpecificationError as exc:
        assert exc.message_ar and exc.message_ar != exc.message_en
    else:
        raise AssertionError("a two-link chain must raise")

    try:
        implied.indirect_effect(cm, [y, x, spec.construct_by_name("M").id])
    except SpecificationError as exc:
        assert "مسار" in exc.message_ar
    else:
        raise AssertionError("a non-existent link must raise")


# --------------------------------------------------------------------------- #
# Topological ordering and the cyclic fallback
# --------------------------------------------------------------------------- #


def test_recursive_models_give_a_lower_triangular_b() -> None:
    for name in ("first_order", "two_factor_simple", "second_order",
                 "mediation", "double_mediation", "correlated_exogenous",
                 "cross_loaded", "error_covariances", "method_factor"):
        cm = compile_model(ALL_BUILDERS[name]())
        assert np.allclose(np.triu(cm.B), 0.0, atol=ATOL), name
        assert cm.is_recursive, name


def test_topological_order_survives_a_reversed_specification() -> None:
    """Declaring Y before X must still produce a lower-triangular B."""
    spec = ModelSpec(seed=3)
    spec.add_construct("Y", n_items=3, loading=0.8, role=ConstructRole.DEPENDENT)
    spec.add_construct("M", n_items=3, loading=0.8, role=ConstructRole.MEDIATOR)
    spec.add_construct("X", n_items=3, loading=0.8)
    spec.add_path("X", "M", coefficient=0.5)
    spec.add_path("M", "Y", coefficient=0.4)
    cm = compile_model(spec)
    assert np.allclose(np.triu(cm.B), 0.0, atol=ATOL)
    order = [cm.factor_index[spec.construct_by_name(n).id]
             for n in ("X", "M", "Y")]
    assert order == sorted(order), order
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)


def test_non_recursive_model_falls_back_to_the_fixed_point_solution() -> None:
    spec = _non_recursive()
    cm = compile_model(spec)
    assert not cm.is_recursive
    assert any("CYCLE_IN_PATHS" in w for w in cm.warnings), cm.warnings
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)

    # Closed form for the reciprocal pair A <-> B with a = 0.3 (B on A) and
    # b = 0.5 (A on B).  det(I - B) = 1 - ab = 0.85, and the unit-variance
    # constraints are   s_A + a^2 s_B = det^2 ,  b^2 s_A + s_B = det^2 .
    a, b = 0.3, 0.5
    det2 = (1.0 - a * b) ** 2
    mat = np.array([[1.0, a ** 2], [b ** 2, 1.0]])
    s_a, s_b = np.linalg.solve(mat, np.array([det2, det2]))
    r = (b * s_a + a * s_b) / det2

    ai = cm.factor_index[spec.construct_by_name("A").id]
    bi = cm.factor_index[spec.construct_by_name("B").id]
    assert abs(cm.Psi[ai, ai] - s_a) < 1e-9, (cm.Psi[ai, ai], s_a)
    assert abs(cm.Psi[bi, bi] - s_b) < 1e-9, (cm.Psi[bi, bi], s_b)
    assert abs(cm.Phi[ai, bi] - r) < 1e-9, (cm.Phi[ai, bi], r)
    assert abs(r - 0.6956521739130435) < 1e-12, r

    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert implied.is_positive_definite(sigma)
    # Items across the loop correlate 0.8 * r * 0.8.
    rows_a = cm.items_of(spec.construct_by_name("A").id)
    rows_b = cm.items_of(spec.construct_by_name("B").id)
    assert np.allclose(sigma[np.ix_(rows_a, rows_b)], 0.64 * r, atol=1e-9)


def test_correlated_disturbance_between_endogenous_latents_still_standardises() -> None:
    spec = ModelSpec(seed=47)
    spec.add_construct("X", n_items=3, loading=0.8)
    m1 = spec.add_construct("M1", n_items=3, loading=0.8,
                            role=ConstructRole.MEDIATOR)
    m2 = spec.add_construct("M2", n_items=3, loading=0.8,
                            role=ConstructRole.MEDIATOR)
    spec.add_path("X", "M1", coefficient=0.5)
    spec.add_path("X", "M2", coefficient=0.4)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=m1.id, construct_b=m2.id, correlation=0.25))
    cm = compile_model(spec)
    assert any("ENDOGENOUS_COVARIANCE" in w for w in cm.warnings), cm.warnings
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)
    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert implied.is_positive_definite(sigma)


def test_correlated_exogenous_block_is_copied_into_phi() -> None:
    spec = _correlated_exogenous()
    cm = compile_model(spec)
    a = cm.factor_index[spec.construct_by_name("A").id]
    b = cm.factor_index[spec.construct_by_name("B").id]
    y = cm.factor_index[spec.construct_by_name("Y").id]
    assert abs(cm.Phi[a, b] - 0.35) < ATOL
    # R^2 = 0.4^2 + 0.3^2 + 2 * 0.4 * 0.3 * 0.35 = 0.25 + 0.084 = 0.334.
    assert abs(cm.Psi[y, y] - (1.0 - 0.334)) < ATOL
    # Cov(Y, A) = 0.4 * 1 + 0.3 * 0.35 = 0.505.
    assert abs(cm.Phi[y, a] - 0.505) < ATOL
    # Cov(Y, B) = 0.4 * 0.35 + 0.3 * 1 = 0.44.
    assert abs(cm.Phi[y, b] - 0.44) < ATOL


def _realised_disturbance_correlation(cm, id_a: str, id_b: str) -> float:
    """Correlation actually carried by Psi between two latents' disturbances."""
    a = cm.factor_index[id_a]
    b = cm.factor_index[id_b]
    return float(cm.Psi[a, b] / math.sqrt(cm.Psi[a, a] * cm.Psi[b, b]))


def test_latent_covariance_is_realised_as_a_correlation_not_a_covariance() -> None:
    """``LatentCovariance.correlation`` is a correlation for every pair.

    Regression test. ``Psi[a, b]`` used to be set to ``r`` itself, which is
    only the right covariance when both disturbance variances are one, i.e.
    when both latents are exogenous. An endogenous latent has
    ``Psi[i, i] = 1 - R^2 < 1``, so the unscaled entry realised
    ``r / sqrt(Psi_aa Psi_bb)`` -- bigger than ``r``, and for the model below it
    came out as 1.1111, an impossible correlation that left ``Psi`` indefinite.
    The entry has to be ``r * sqrt(Psi_aa Psi_bb)``, taken after the diagonals
    are known.
    """
    # -- exogenous <-> exogenous: both disturbance variances are 1, so the fix
    #    must leave this case exactly as it was.
    cm = compile_model(_correlated_exogenous())
    spec = cm.spec
    a_id = spec.construct_by_name("A").id
    b_id = spec.construct_by_name("B").id
    assert abs(_realised_disturbance_correlation(cm, a_id, b_id) - 0.35) < ATOL
    assert abs(cm.Psi[cm.factor_index[a_id], cm.factor_index[b_id]] - 0.35) < ATOL

    # -- exogenous <-> endogenous.  Y = 0.8 X2 + zeta_Y with X2 exogenous and
    #    unit variance, so Psi_YY = 1 - 0.64 = 0.36 and the covariance that
    #    realises corr(A, zeta_Y) = 0.40 is 0.40 * sqrt(1 * 0.36) = 0.24.
    spec = ModelSpec(seed=101)
    a = spec.add_construct("A", n_items=3, loading=0.8)
    spec.add_construct("X2", n_items=3, loading=0.8)
    y = spec.add_construct("Y", n_items=3, loading=0.8,
                           role=ConstructRole.DEPENDENT)
    spec.add_path("X2", "Y", coefficient=0.8)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=a.id, construct_b=y.id, correlation=0.40))
    cm = compile_model(spec)
    ai, yi = cm.factor_index[a.id], cm.factor_index[y.id]
    assert abs(cm.Psi[yi, yi] - 0.36) < ATOL, cm.Psi[yi, yi]
    assert abs(cm.Psi[ai, yi] - 0.24) < ATOL, cm.Psi[ai, yi]
    assert abs(_realised_disturbance_correlation(cm, a.id, y.id) - 0.40) < ATOL
    # Cov(A, Y) = 0.8 * Cov(A, X2) + Cov(A, zeta_Y) = 0 + 0.24.
    assert abs(cm.Phi[ai, yi] - 0.24) < ATOL, cm.Phi[ai, yi]

    # -- endogenous <-> endogenous with *unequal* disturbance variances, the
    #    sharpest discriminator: Psi_M1M1 = 1 - 0.8^2 = 0.36 and
    #    Psi_M2M2 = 1 - 0.5^2 = 0.75, so the entry is
    #    0.40 * sqrt(0.36 * 0.75) = 0.2078460969...  The old code wrote 0.40
    #    and realised 0.7698.
    spec = ModelSpec(seed=103)
    spec.add_construct("X", n_items=3, loading=0.8)
    m1 = spec.add_construct("M1", n_items=3, loading=0.8,
                            role=ConstructRole.MEDIATOR)
    m2 = spec.add_construct("M2", n_items=3, loading=0.8,
                            role=ConstructRole.MEDIATOR)
    spec.add_path("X", "M1", coefficient=0.8)
    spec.add_path("X", "M2", coefficient=0.5)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=m1.id, construct_b=m2.id, correlation=0.40))
    cm = compile_model(spec)
    # The warning naming the reinterpretation must survive the fix.
    assert any("ENDOGENOUS_COVARIANCE" in w for w in cm.warnings), cm.warnings
    i1, i2 = cm.factor_index[m1.id], cm.factor_index[m2.id]
    xi = cm.factor_index[spec.construct_by_name("X").id]
    assert abs(cm.Psi[i1, i1] - 0.36) < ATOL, cm.Psi[i1, i1]
    assert abs(cm.Psi[i2, i2] - 0.75) < ATOL, cm.Psi[i2, i2]
    expected = 0.40 * math.sqrt(0.36 * 0.75)
    assert abs(cm.Psi[i1, i2] - expected) < ATOL, (cm.Psi[i1, i2], expected)
    assert abs(_realised_disturbance_correlation(cm, m1.id, m2.id) - 0.40) < ATOL
    # Cov(M1, M2) = 0.8 * 0.5 * Var(X) + Psi[M1, M2].
    assert abs(cm.Phi[i1, i2] - (0.40 + expected)) < ATOL, cm.Phi[i1, i2]
    # Partialling X back out of Phi must return the requested 0.40 exactly.
    rho = cm.Phi
    partial = ((rho[i1, i2] - rho[i1, xi] * rho[i2, xi])
               / math.sqrt((1.0 - rho[i1, xi] ** 2) * (1.0 - rho[i2, xi] ** 2)))
    assert abs(partial - 0.40) < ATOL, partial
    # An over-large off-diagonal is what used to make Psi indefinite.
    assert implied.is_positive_definite(cm.Psi), cm.Psi
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)
    assert implied.is_positive_definite(implied.implied_covariance(cm))


def test_disturbance_correlation_matches_the_closed_form_solution() -> None:
    """The solved disturbance variance agrees with the algebra by hand.

    A chain C0 -> C1 (0.839) -> C2 (0.819) with corr(zeta_1, zeta_2) = -0.091.
    Psi_11 = 1 - 0.839^2 directly, and the unit-variance constraint on C2 is

        0.819^2 + v + 2 * 0.819 * (-0.091) * sqrt(Psi_11) * sqrt(v) = 1,

    a quadratic in sqrt(v) whose positive root is the only admissible answer.
    The cross term is what makes this a real test: it exists only because the
    off-diagonal of Psi is scaled by the disturbance standard deviations, and
    it is also what makes d Phi_22 / d Psi_22 differ from one, so this model
    exercises the corrected Newton slope as well as the scaling.
    """
    spec = ModelSpec(seed=137)
    spec.add_construct("C0", n_items=3, loading=0.8)
    c1 = spec.add_construct("C1", n_items=3, loading=0.8,
                            role=ConstructRole.MEDIATOR)
    c2 = spec.add_construct("C2", n_items=3, loading=0.8,
                            role=ConstructRole.DEPENDENT)
    spec.add_path("C0", "C1", coefficient=0.839)
    spec.add_path("C1", "C2", coefficient=0.819)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=c1.id, construct_b=c2.id,
                         correlation=-0.091))
    cm = compile_model(spec)
    i1, i2 = cm.factor_index[c1.id], cm.factor_index[c2.id]

    psi_11 = 1.0 - 0.839 ** 2
    assert abs(cm.Psi[i1, i1] - psi_11) < ATOL, cm.Psi[i1, i1]
    # Positive root of  v - 2*0.819*0.091*sqrt(psi_11)*sqrt(v) + 0.819^2 - 1 = 0.
    half_b = 0.819 * 0.091 * math.sqrt(psi_11)
    root = half_b + math.sqrt(half_b ** 2 + 1.0 - 0.819 ** 2)
    assert abs(cm.Psi[i2, i2] - root ** 2) < 1e-9, (cm.Psi[i2, i2], root ** 2)
    assert abs(_realised_disturbance_correlation(cm, c1.id, c2.id)
               - (-0.091)) < ATOL
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)
    assert implied.is_positive_definite(cm.Psi), cm.Psi


def test_disturbance_correlation_holds_through_a_feedback_loop() -> None:
    """The rescaling must survive the cyclic fixed-point solver too.

    Regression test. The fixed-point pass rewrites ``diag(Psi)`` on every
    sweep; the off-diagonal is a function of that diagonal, so it has to be
    rebuilt each time. Leaving it alone let the realised correlation drift with
    the diagonal, far away from the requested value.
    """
    spec = ModelSpec(seed=107)
    a = spec.add_construct("A", n_items=3, loading=0.8)
    b = spec.add_construct("B", n_items=3, loading=0.8)
    spec.add_construct("C", n_items=3, loading=0.8, role=ConstructRole.DEPENDENT)
    spec.add_path("A", "B", coefficient=0.5)
    spec.add_path("B", "A", coefficient=0.3)
    spec.add_path("B", "C", coefficient=0.4)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=a.id, construct_b=b.id, correlation=-0.45))
    cm = compile_model(spec)
    assert not cm.is_recursive
    assert abs(_realised_disturbance_correlation(cm, a.id, b.id) - (-0.45)) < ATOL
    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)
    assert implied.is_positive_definite(cm.Psi), cm.Psi
    assert implied.is_positive_definite(implied.implied_covariance(cm))


# --------------------------------------------------------------------------- #
# Heywood handling
# --------------------------------------------------------------------------- #


def test_latent_heywood_case_clamps_warns_and_stays_positive_definite() -> None:
    cm = compile_model(_heywood())          # must not raise
    spec = cm.spec
    a = cm.factor_index[spec.construct_by_name("A").id]
    b = cm.factor_index[spec.construct_by_name("B").id]
    y = cm.factor_index[spec.construct_by_name("Y").id]

    assert any("HEYWOOD" in w for w in cm.warnings), cm.warnings
    assert any("هيوود" in w for w in cm.warnings), "the warning must be bilingual"

    # R^2 was 0.81 + 0.81 + 2 * 0.81 * 0.8 = 2.916; the shrink factor is
    # sqrt((1 - floor) / 2.916).
    explained = 2.916
    shrink = math.sqrt((1.0 - HEYWOOD_FLOOR) / explained)
    assert abs(cm.B[y, a] - 0.9 * shrink) < 1e-12
    assert abs(cm.B[y, b] - 0.9 * shrink) < 1e-12
    assert abs(cm.Psi[y, y] - HEYWOOD_FLOOR) < 1e-15

    assert np.allclose(np.diag(cm.Phi), 1.0, atol=ATOL)
    assert implied.is_positive_definite(cm.Phi)
    sigma = implied.implied_covariance(cm)
    assert implied.is_positive_definite(sigma)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    # Retained R^2 is exactly 1 - floor.
    bvec = np.array([cm.B[y, a], cm.B[y, b]])
    block = cm.Phi[np.ix_([a, b], [a, b])]
    assert abs(float(bvec @ block @ bvec) - (1.0 - HEYWOOD_FLOOR)) < 1e-12


def test_item_heywood_case_rescales_the_loading_row() -> None:
    spec = ModelSpec(seed=53)
    f = spec.add_construct("F", n_items=4, loading=0.8)
    f.items[0].loading = 1.20              # communality 1.44 > 1
    cm = compile_model(spec)
    j = cm.item_index[f.items[0].id]
    assert any("HEYWOOD_ITEM" in w for w in cm.warnings), cm.warnings
    assert abs(cm.Theta[j, j] - HEYWOOD_FLOOR) < 1e-15
    assert abs(cm.Lambda[j, cm.home_factor[j]]
               - 1.20 * math.sqrt((1.0 - HEYWOOD_FLOOR) / 1.44)) < 1e-12
    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert implied.is_positive_definite(sigma)


# --------------------------------------------------------------------------- #
# Positive-definiteness utilities
# --------------------------------------------------------------------------- #


def test_nearest_positive_definite_is_idempotent_on_a_pd_matrix() -> None:
    rng = np.random.default_rng(101)
    a = rng.normal(size=(6, 6))
    pd = a @ a.T + 3.0 * np.eye(6)
    assert implied.is_positive_definite(pd)
    once = implied.nearest_positive_definite(pd)
    twice = implied.nearest_positive_definite(once)
    assert np.allclose(once, pd, atol=1e-12)
    assert np.allclose(twice, once, atol=1e-12)
    assert np.allclose(once, once.T, atol=0.0)


def test_nearest_positive_definite_fixes_one_negative_eigenvalue() -> None:
    # A correlation-shaped matrix with a single negative eigenvalue.
    broken = np.array([
        [1.00, 0.95, 0.95],
        [0.95, 1.00, -0.60],
        [0.95, -0.60, 1.00],
    ])
    eig = np.linalg.eigvalsh(broken)
    assert eig.min() < 0.0, eig

    fixed = implied.nearest_positive_definite(broken)
    assert np.allclose(fixed, fixed.T, atol=0.0)
    assert np.linalg.eigvalsh(fixed).min() > 0.0
    assert implied.is_positive_definite(fixed)
    # The diagonal (the variances) must be preserved exactly.
    assert np.allclose(np.diag(fixed), 1.0, atol=1e-12)
    # And it is a repair, not a reset to the identity.
    assert abs(fixed[0, 1]) > 0.3


def test_is_positive_definite_rejects_asymmetric_and_singular_input() -> None:
    assert not implied.is_positive_definite(np.array([[1.0, 0.4], [0.9, 1.0]]))
    assert not implied.is_positive_definite(np.zeros((3, 3)))
    assert not implied.is_positive_definite(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert implied.is_positive_definite(np.eye(4))


def test_condition_number_of_identity_is_one() -> None:
    assert abs(implied.condition_number(np.eye(5)) - 1.0) < 1e-12
    m = np.diag([4.0, 1.0, 0.5])
    assert abs(implied.condition_number(m) - 8.0) < 1e-12
    assert implied.condition_number(np.array([[1.0, 1.0], [1.0, 1.0]])) == float("inf")


def test_condition_number_reports_a_numerically_singular_matrix_as_inf() -> None:
    """Singularity is relative to the largest eigenvalue, not to exact zero.

    Regression test. Singularity used to be decided with
    ``min(|eigvalsh|) <= 0.0``, but LAPACK returns a tiny value of either sign
    for a zero eigenvalue, so the test missed it whenever the sign came out
    positive. ``np.outer([1, 2, 3], [1, 2, 3])`` is exactly rank one; its zero
    eigenvalue comes back as +1.9e-16 and the function reported a finite
    7.36e16 for a matrix with no inverse.
    """
    rank_one = np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert np.linalg.matrix_rank(rank_one) == 1
    assert float(np.abs(np.linalg.eigvalsh(rank_one)).min()) > 0.0, (
        "the eigenvalue this test is about is supposed to be a tiny positive "
        "float, not an exact zero")
    assert implied.condition_number(rank_one) == float("inf")
    # A rank-deficient matrix with a larger scale must behave the same way: the
    # tolerance has to be relative, not an absolute floor.
    assert implied.condition_number(1e6 * rank_one) == float("inf")
    assert implied.condition_number(np.zeros((4, 4))) == float("inf")
    # ... while a genuinely ill-conditioned but invertible matrix stays finite.
    nearly = np.diag([1.0, 1e-9, 1.0])
    assert abs(implied.condition_number(nearly) - 1e9) < 1.0


def test_population_htmt_is_nan_when_a_monotrait_mean_is_not_positive() -> None:
    """Each monotrait mean is guarded on its own, not only their product.

    Regression test. The guard used to be ``mono[a] * mono[b] > 0``, which
    passes when BOTH means are negative -- and the geometric mean of two
    negative within-factor averages is not a reliability of anything. The case
    is reachable: the compiler negates the loading row of every reverse-worded
    item, so a factor whose items are mostly reverse-worded has a negative mean
    monotrait correlation until the researcher recodes them. Here both factors
    are in that state and the old code returned a confident 0.1333.
    """
    spec = ModelSpec(seed=109)
    fa = spec.add_construct("FA", n_items=3, loading=0.8)
    fb = spec.add_construct("FB", n_items=3, loading=0.8)
    spec.latent_covariances.append(
        LatentCovariance(construct_a=fa.id, construct_b=fb.id, correlation=0.4))
    for construct in (fa, fb):
        construct.items[0].reverse_coded = True
        construct.items[1].reverse_coded = True
    cm = compile_model(spec)

    # Both monotrait means really are negative: (+0.64 - 0.64 - 0.64) / 3.
    corr = implied.implied_correlation(cm)
    iu = np.triu_indices(3, k=1)
    for construct in (fa, fb):
        rows = cm.items_of(construct.id)
        mono = float(np.mean(corr[np.ix_(rows, rows)][iu]))
        assert abs(mono - (-0.64 / 3.0)) < 1e-12, mono

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        htmt, ids = implied.population_htmt(cm)
    assert len(ids) == 2
    assert math.isnan(htmt[0, 1]) and math.isnan(htmt[1, 0]), htmt
    assert np.allclose(np.diag(htmt), 1.0, atol=0.0)
    # The reason has to be recorded, naming both offending factors, in both
    # interface languages.
    reasons = [str(w.message) for w in caught
               if "HTMT_MONOTRAIT_NOT_POSITIVE" in str(w.message)]
    assert len(reasons) == 1, [str(w.message) for w in caught]
    assert fa.id in reasons[0] and fb.id in reasons[0]
    assert "متوسط الارتباطات الداخلية" in reasons[0]

    # One negative mean is enough; this side was already handled and must stay.
    spec2 = ModelSpec(seed=113)
    ga = spec2.add_construct("GA", n_items=3, loading=0.8)
    spec2.add_construct("GB", n_items=3, loading=0.8)
    ga.items[0].reverse_coded = True
    ga.items[1].reverse_coded = True
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        htmt2, _ = implied.population_htmt(compile_model(spec2))
    assert math.isnan(htmt2[0, 1]), htmt2
    assert any("HTMT_MONOTRAIT_NOT_POSITIVE" in str(w.message) for w in caught2)

    # And a clean model must still produce a number, with no warning at all.
    with warnings.catch_warnings(record=True) as caught3:
        warnings.simplefilter("always")
        htmt3, _ = implied.population_htmt(compile_model(_two_factor_simple()))
    assert abs(htmt3[0, 1] - 0.5) < 1e-12
    assert not [w for w in caught3
                if "HTMT_MONOTRAIT_NOT_POSITIVE" in str(w.message)]


def test_reverse_coded_items_get_negative_loadings_and_correlations() -> None:
    """A reverse-worded item runs against its factor until it is recoded.

    The generation engine leans on this: a simulated dataset has to be able to
    reproduce what a forgotten recode looks like in practice -- a negative
    item-total correlation and a collapsed alpha -- so the negated loading row
    and the negative sibling correlations are load-bearing, not cosmetic.
    """
    spec = ModelSpec(seed=127)
    f = spec.add_construct("F", n_items=4, loading=0.8)
    spec.add_construct("G", n_items=3, loading=0.7, role=ConstructRole.DEPENDENT)
    spec.add_path("F", "G", coefficient=0.5)
    f.items[0].reverse_coded = True
    cm = compile_model(spec)
    assert any("REVERSE_CODED" in w for w in cm.warnings), cm.warnings

    rows = cm.items_of(f.id)
    reversed_row = cm.item_index[f.items[0].id]
    siblings = [j for j in rows if j != reversed_row]
    home = cm.home_factor[reversed_row]
    # The whole loading row flips sign, the forward-worded ones do not.
    assert cm.Lambda[reversed_row, home] < 0.0, cm.Lambda[reversed_row]
    assert abs(cm.Lambda[reversed_row, home] + 0.8) < ATOL
    for j in siblings:
        assert cm.Lambda[j, cm.home_factor[j]] > 0.0

    # Implied correlations: reverse-against-sibling is -0.64, and the item
    # still explains the same share of its own variance (Theta is unchanged).
    sigma = implied.implied_correlation(cm)
    for j in siblings:
        assert sigma[reversed_row, j] < 0.0, sigma[reversed_row, j]
        assert abs(sigma[reversed_row, j] + 0.64) < ATOL
    # Across factors it is negative too: -0.8 * 0.5 * 0.7.
    for j in cm.items_of(spec.construct_by_name("G").id):
        assert abs(sigma[reversed_row, j] + 0.28) < ATOL, sigma[reversed_row, j]
    # Standardisation is untouched by the sign flip.
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert implied.is_positive_definite(sigma)
    assert abs(cm.Theta[reversed_row, reversed_row] - (1.0 - 0.64)) < ATOL


def test_theta_with_an_impossible_residual_pattern_is_repaired() -> None:
    spec = ModelSpec(seed=59)
    f = spec.add_construct("F", n_items=4, loading=0.5)
    ids = [it.id for it in f.items]
    for a, b, r in ((0, 1, 0.95), (0, 2, -0.95), (1, 2, 0.95)):
        spec.error_covariances.append(
            ErrorCovariance(item_a=ids[a], item_b=ids[b], correlation=r))
    cm = compile_model(spec)
    assert any("PD_REPAIR" in w for w in cm.warnings), cm.warnings
    assert implied.is_positive_definite(cm.Theta)
    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert implied.is_positive_definite(sigma)


# --------------------------------------------------------------------------- #
# Population reliability / validity
# --------------------------------------------------------------------------- #


def test_population_reliability_matches_the_closed_form() -> None:
    for k in (3, 4, 5, 6):
        for lam in (0.6, 0.7, 0.8, 0.9):
            spec = ModelSpec(seed=61)
            f = spec.add_construct("F", n_items=k, loading=lam)
            cm = compile_model(spec)
            rel = implied.population_reliability(cm)[f.id]
            expected_ave = lam ** 2
            numerator = (k * lam) ** 2
            expected_cr = numerator / (numerator + k * (1.0 - lam ** 2))
            assert abs(rel["ave"] - expected_ave) < 1e-12, (k, lam)
            assert abs(rel["cr"] - expected_cr) < 1e-12, (k, lam)
            assert abs(rel["omega"] - expected_cr) < 1e-12, (k, lam)
            assert abs(rel["alpha_implied"] - expected_cr) < 1e-12, (k, lam)
            assert rel["n_items"] == float(k)


def test_population_reliability_worked_example_k4_lambda08() -> None:
    spec = ModelSpec(seed=67)
    f = spec.add_construct("F", n_items=4, loading=0.8)
    cm = compile_model(spec)
    rel = implied.population_reliability(cm)[f.id]
    assert round(rel["ave"], 4) == 0.64
    assert round(rel["cr"], 4) == 0.8767
    assert round(rel["alpha_implied"], 4) == 0.8767
    assert round(rel["omega"], 4) == 0.8767


def test_second_order_construct_reports_dimension_level_reliability() -> None:
    spec = _second_order()
    cm = compile_model(spec)
    rel = implied.population_reliability(cm)
    construct = spec.constructs[0]
    assert construct.id in rel
    top = rel[construct.id]
    # Three dimensions loading 0.8: AVE = 0.64, CR = 2.4^2/(2.4^2 + 3*0.36).
    assert abs(top["ave"] - 0.64) < 1e-12
    assert abs(top["cr"] - (2.4 ** 2 / (2.4 ** 2 + 3 * 0.36))) < 1e-12
    assert top["n_items"] == 3.0
    for d in construct.dimensions:
        assert abs(rel[d.id]["ave"] - 0.49) < 1e-12


def test_population_htmt_recovers_the_factor_correlation() -> None:
    cm = compile_model(_two_factor_simple())
    htmt, ids = implied.population_htmt(cm)
    assert len(ids) == 2
    assert np.allclose(np.diag(htmt), 1.0, atol=0.0)
    # Heterotrait mean 0.32, monotrait means 0.64 each -> 0.32 / 0.64 = 0.5.
    assert abs(htmt[0, 1] - 0.5) < 1e-12
    assert abs(htmt[0, 1] - htmt[1, 0]) < 1e-15


def test_implied_correlation_equals_implied_covariance_here() -> None:
    for name, build in ALL_BUILDERS.items():
        cm = compile_model(build())
        assert np.allclose(implied.implied_correlation(cm),
                           implied.implied_covariance(cm), atol=1e-10), name
        assert np.allclose(np.diag(implied.implied_correlation(cm)), 1.0,
                           atol=0.0), name


def test_latent_covariance_returns_phi() -> None:
    cm = compile_model(_mediation())
    assert np.allclose(implied.latent_covariance(cm), cm.Phi, atol=0.0)


# --------------------------------------------------------------------------- #
# Reproducibility, bookkeeping and errors
# --------------------------------------------------------------------------- #


def test_compilation_is_reproducible_under_an_explicit_generator() -> None:
    def build() -> ModelSpec:
        spec = _first_order()
        spec.cross_loading_policy.enabled = True
        spec.cross_loading_policy.proportion_of_items = 0.5
        spec.cross_loading_policy.allow_negative = True
        return spec

    one = compile_model(build(), np.random.default_rng(777))
    two = compile_model(build(), np.random.default_rng(777))
    three = compile_model(build(), np.random.default_rng(778))
    assert np.allclose(one.Lambda, two.Lambda, atol=0.0)
    assert not np.allclose(one.Lambda, three.Lambda, atol=0.0)
    # Seeded from spec.seed when no generator is supplied.
    assert np.allclose(compile_model(build()).Lambda,
                       compile_model(build()).Lambda, atol=0.0)


def test_shapes_and_index_bookkeeping() -> None:
    spec = _method_factor()
    cm = compile_model(spec)
    p, m = cm.n_items, cm.n_factors
    assert p == 6 and m == 3                      # P, Q and the method factor
    assert cm.Lambda.shape == (p, m)
    assert cm.B.shape == cm.Psi.shape == cm.Phi.shape == (m, m)
    assert cm.Theta.shape == (p, p)
    assert cm.endogenous.shape == (m,)
    assert cm.endogenous.dtype == bool
    assert len(cm.item_ids) == len(cm.item_names) == p
    assert len(cm.factor_ids) == len(cm.factor_names) == m
    assert len(cm.home_factor) == p
    assert cm.item_names == [it.name for it in spec.all_items()]
    assert all(cm.item_index[i] == k for k, i in enumerate(cm.item_ids))
    assert all(cm.factor_index[f] == k for k, f in enumerate(cm.factor_ids))
    q = cm.factor_index[spec.construct_by_name("Q").id]
    assert bool(cm.endogenous[q])
    assert not bool(cm.endogenous[cm.method_factor_index])


def test_second_order_endogenous_mask_marks_dimensions() -> None:
    spec = _second_order()
    cm = compile_model(spec)
    construct = spec.constructs[0]
    assert not bool(cm.endogenous[cm.factor_index[construct.id]])
    for d in construct.dimensions:
        assert bool(cm.endogenous[cm.factor_index[d.id]])


def test_user_errors_raise_semsim_errors_with_arabic_messages() -> None:
    arabic = range(0x0600, 0x06FF)

    def has_arabic(text: str) -> bool:
        return any(ord(ch) in arabic for ch in text)

    # Residual correlation of magnitude >= 1.
    spec = ModelSpec(seed=71)
    f = spec.add_construct("F", n_items=3, loading=0.7)
    spec.error_covariances.append(
        ErrorCovariance(item_a=f.items[0].id, item_b=f.items[1].id,
                        correlation=1.0))
    try:
        compile_model(spec)
    except SpecificationError as exc:
        assert has_arabic(exc.message_ar)
    else:
        raise AssertionError("|r| >= 1 must raise SpecificationError")

    # Unknown item in a cross-loading.
    spec = ModelSpec(seed=73)
    spec.add_construct("F", n_items=3, loading=0.7)
    spec.cross_loadings.append(
        CrossLoading(item_id="nope", factor_id=spec.constructs[0].id,
                     loading=0.2))
    try:
        compile_model(spec)
    except SpecificationError as exc:
        assert has_arabic(exc.message_ar)
    else:
        raise AssertionError("an unknown item must raise SpecificationError")

    # Empty model.
    try:
        compile_model(ModelSpec(seed=79))
    except SpecificationError as exc:
        assert has_arabic(exc.message_ar)
    else:
        raise AssertionError("an empty model must raise SpecificationError")

    # Negative extra error variance.
    spec = ModelSpec(seed=83)
    g = spec.add_construct("G", n_items=3, loading=0.7)
    g.items[0].extra_error_variance = -0.1
    try:
        compile_model(spec)
    except SpecificationError as exc:
        assert has_arabic(exc.message_ar)
    else:
        raise AssertionError("a negative error variance must raise")

    # Unknown factor requested from implied.*
    cm = compile_model(_mediation())
    try:
        implied.direct_effect(cm, "ghost", cm.factor_ids[0])
    except SpecificationError as exc:
        assert has_arabic(exc.message_ar)
    else:
        raise AssertionError("an unknown factor id must raise")


def test_clean_model_compiles_without_warnings() -> None:
    assert compile_model(_two_factor_simple()).warnings == []
    assert compile_model(_second_order()).warnings == []


def test_sixty_item_model_compiles_well_under_a_second() -> None:
    spec = ModelSpec(seed=89)
    for k in range(5):
        spec.add_construct(f"C{k}", n_items=12, loading=0.75)
    for k in range(1, 5):
        spec.add_path(f"C{k - 1}", f"C{k}", coefficient=0.35)
    start = time.perf_counter()
    cm = compile_model(spec)
    sigma = implied.implied_covariance(cm)
    elapsed = time.perf_counter() - start
    assert cm.n_items == 60
    assert np.allclose(np.diag(sigma), 1.0, atol=ATOL)
    assert implied.is_positive_definite(sigma)
    assert elapsed < 0.5, elapsed


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

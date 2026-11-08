"""Numerical tests for :mod:`semsim.generation.noise`.

Every check prints the observed number so a failure is self-explanatory, and
every assertion is against a closed-form or known quantity rather than "it
ran": the careless row count is compared with ``round(proportion * n)``, the
realised MCAR rate with the requested rate, the MAR intercept with the
analytic solution of a symmetric two-point driver, and the injected outlier
values with ``mean +/- magnitude * sd``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import expit

from semsim.core.errors import ConvergenceError, SpecificationError
from semsim.core.spec import (
    MissingMechanism,
    ModelSpec,
    ScaleKind,
    ScaleSpec,
)
from semsim.generation.noise import (
    MAR_LOGIT_SLOPE,
    NoiseReport,
    _calibrate_intercept,
    _ordinal_flags_and_bounds,
    _scales_for,
    apply_noise,
    inject_careless,
    inject_missing,
    inject_outliers,
)
from semsim.generation.vale_maurelli import achieved_moments

SEED = 20240101


def _likert_data(n: int, p: int, k: int = 5, seed: int = SEED) -> np.ndarray:
    """A plausible integer Likert block, centred and bell-shaped."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p))
    cuts = np.array([-1.5, -0.5, 0.5, 1.5])[: k - 1]
    return (np.searchsorted(cuts, z.ravel()).reshape(n, p) + 1).astype(np.int64)


def _scales(p: int, k: int = 5) -> list[ScaleSpec]:
    return [ScaleSpec(kind=ScaleKind.LIKERT, n_categories=k) for _ in range(p)]


def _spec_with_noise(**noise_kw) -> ModelSpec:
    spec = ModelSpec(seed=SEED)
    spec.add_construct("X", n_items=3, n_categories=5)
    spec.add_construct("Y", n_items=3, n_categories=5)
    for key, value in noise_kw.items():
        setattr(spec.noise, key, value)
    return spec


# --------------------------------------------------------------------------- #
# 1. Careless responders
# --------------------------------------------------------------------------- #


def test_careless_row_count_is_exact_and_rows_unique() -> None:
    n, p, prop = 500, 10, 0.06
    data = _likert_data(n, p)
    rng = np.random.default_rng(SEED)
    out, rows = inject_careless(data, _scales(p), prop, "mixed", rng)

    expected = round(prop * n)
    print(f"careless rows: observed={len(rows)} expected={expected}")
    assert len(rows) == expected == 30
    assert len(set(rows)) == len(rows), "careless rows drawn with replacement"
    assert all(0 <= r < n for r in rows)

    changed = np.flatnonzero((out != data).any(axis=1))
    print(f"rows that actually changed: {changed.size} (subset of careless: "
          f"{set(changed.tolist()) <= set(rows)})")
    assert set(changed.tolist()) <= set(rows), "a non-careless row was touched"


def test_straightline_rows_are_constant_across_all_items() -> None:
    n, p = 400, 12
    data = _likert_data(n, p)
    rng = np.random.default_rng(7)
    out, rows = inject_careless(data, _scales(p), 0.25, "straightline", rng)

    block = out[rows, :]
    spreads = block.max(axis=1) - block.min(axis=1)
    print(f"straightline rows={len(rows)}; max within-row spread="
          f"{int(spreads.max())}; distinct categories used="
          f"{sorted(set(block[:, 0].tolist()))}")
    assert len(rows) == round(0.25 * n) == 100
    assert int(spreads.max()) == 0, "a straight-liner was not flat"
    assert block.min() >= 1 and block.max() <= 5
    assert out.dtype == data.dtype
    # Not every flat-liner picks the same category.
    assert len(set(block[:, 0].tolist())) > 1


def test_mixed_splits_rows_between_the_two_archetypes() -> None:
    n, p = 1000, 8
    data = _likert_data(n, p)
    rng = np.random.default_rng(11)
    out, rows = inject_careless(data, _scales(p), 0.20, "mixed", rng)

    block = out[rows, :]
    flat = (block.max(axis=1) - block.min(axis=1)) == 0
    print(f"mixed: n_careless={len(rows)} flat={int(flat.sum())} "
          f"random={int((~flat).sum())}")
    assert len(rows) == 200
    # Exactly half are straight-liners by construction; a random responder can
    # coincidentally be flat, with probability 8 * (1/5)**7 ~ 1e-4 per row.
    assert 100 <= int(flat.sum()) <= 103


def test_random_careless_respects_each_items_own_scale() -> None:
    n = 6000
    ks = [3, 5, 7]
    scales = [ScaleSpec(kind=ScaleKind.LIKERT, n_categories=k) for k in ks]
    data = np.full((n, 3), 2, dtype=np.int64)
    rng = np.random.default_rng(3)
    out, rows = inject_careless(data, scales, 1.0, "random", rng)

    assert len(rows) == n
    for j, k in enumerate(ks):
        col = out[:, j]
        counts = np.bincount(col, minlength=k + 1)[1:]
        share = counts / n
        print(f"item {j} (K={k}): min={col.min()} max={col.max()} "
              f"shares={np.round(share, 4).tolist()} expected={1 / k:.4f}")
        assert col.min() == 1 and col.max() == k
        assert np.all(np.abs(share - 1 / k) < 0.02)


def test_careless_rejects_bad_input_with_arabic_messages() -> None:
    data = _likert_data(50, 4)
    rng = np.random.default_rng(0)
    for bad in (-0.1, 1.5, float("nan")):
        try:
            inject_careless(data, _scales(4), bad, "random", rng)
        except SpecificationError as exc:
            assert exc.message_ar and exc.message_ar != exc.message_en
        else:  # pragma: no cover - must raise
            raise AssertionError(f"proportion {bad} was accepted")
    try:
        inject_careless(data, _scales(4), 0.1, "lazy", rng)
    except SpecificationError as exc:
        print(f"unknown style -> ar: {exc.message_ar}")
        assert "straightline" in exc.message_ar
    else:  # pragma: no cover - must raise
        raise AssertionError("unknown careless style was accepted")


# --------------------------------------------------------------------------- #
# 2. Outliers
# --------------------------------------------------------------------------- #


def test_ordinal_outliers_stay_inside_the_legal_range() -> None:
    n, p, k = 800, 10, 5
    data = _likert_data(n, p, k)
    rng = np.random.default_rng(5)
    out, n_cells = inject_outliers(data, 0.05, 3.5, rng, scales=_scales(p, k))

    expected_cells = round(0.05 * n * p)
    print(f"outlier cells: observed={n_cells} expected={expected_cells}; "
          f"value range after injection=[{out.min()}, {out.max()}]; "
          f"unique values={sorted(set(np.unique(out).tolist()))}")
    assert n_cells == expected_cells == 400
    assert out.min() >= 1 and out.max() <= k
    assert out.dtype == data.dtype
    assert set(np.unique(out).tolist()) <= set(range(1, k + 1))

    # 3.5 SD away from a mean near 3 on a 1-5 scale can only land on an
    # endpoint, so every changed cell must be a floor or a ceiling response.
    changed = out != data
    print(f"changed cells={int(changed.sum())}; distinct values at changed "
          f"cells={sorted(set(out[changed].tolist()))}")
    assert set(out[changed].tolist()) <= {1, k}


def test_continuous_outliers_sit_exactly_at_mean_plus_minus_magnitude() -> None:
    rng = np.random.default_rng(17)
    n, p, mag = 400, 5, 4.0
    data = rng.normal(loc=10.0, scale=2.0, size=(n, p))
    before = data.copy()
    out, n_cells = inject_outliers(data, 0.02, mag, np.random.default_rng(2))

    mean = before.mean(axis=0)
    sd = before.std(axis=0)
    changed = np.flatnonzero((out != before).ravel())
    rows, cols = np.unravel_index(changed, before.shape)
    hi = mean[cols] + mag * sd[cols]
    lo = mean[cols] - mag * sd[cols]
    obs = out[rows, cols]
    err = np.minimum(np.abs(obs - hi), np.abs(obs - lo)).max()
    print(f"outlier cells={n_cells} (expected {round(0.02 * n * p)}); "
          f"max |observed - (mean +/- {mag} sd)| = {err:.3e}; "
          f"signs: +{int((obs > mean[cols]).sum())} / "
          f"-{int((obs < mean[cols]).sum())}")
    assert n_cells == round(0.02 * n * p) == 40
    assert changed.size == n_cells
    assert err < 1e-12
    # Both signs occur.
    assert 5 <= int((obs > mean[cols]).sum()) <= n_cells - 5
    assert np.array_equal(before, data), "input mutated in place"


def test_outlier_magnitude_must_be_positive() -> None:
    data = _likert_data(40, 3)
    try:
        inject_outliers(data, 0.1, 0.0, np.random.default_rng(0))
    except SpecificationError as exc:
        print(f"magnitude 0 -> ar: {exc.message_ar}")
        assert exc.message_ar != exc.message_en
    else:  # pragma: no cover - must raise
        raise AssertionError("a zero outlier magnitude was accepted")


# --------------------------------------------------------------------------- #
# 3. Missingness
# --------------------------------------------------------------------------- #


def test_mcar_realised_rate_matches_the_request() -> None:
    n, p, prop = 5000, 8, 0.12
    data = _likert_data(n, p, seed=99)
    rng = np.random.default_rng(SEED)
    out, n_missing = inject_missing(data, MissingMechanism.MCAR, prop, rng)

    realised = n_missing / (n * p)
    print(f"MCAR: requested={prop} realised={realised:.5f} "
          f"|diff|={abs(realised - prop):.5f} cells={n_missing}")
    assert abs(realised - prop) < 0.01
    assert np.isnan(out).sum() == n_missing
    assert out.dtype == np.float64

    # Row-wise independence: the count of missing cells per row must look
    # binomial(p, prop), not clumped.  Variance check as a cheap proxy.
    per_row = np.isnan(out).sum(axis=1)
    print(f"per-row missing: mean={per_row.mean():.4f} (expected "
          f"{p * prop:.4f}) var={per_row.var():.4f} (expected "
          f"{p * prop * (1 - prop):.4f})")
    assert abs(per_row.mean() - p * prop) < 0.1
    assert abs(per_row.var() - p * prop * (1 - prop)) < 0.1


def test_mar_intercept_calibration_matches_closed_form() -> None:
    # A symmetric two-point driver and a target of exactly 1/2 force the
    # logistic intercept to 0 by symmetry: expit(s) + expit(-s) = 1.
    z = np.array([1.0, -1.0] * 500)
    a = _calibrate_intercept(z, MAR_LOGIT_SLOPE, 0.5)
    print(f"closed form: intercept={a:.3e} (expected 0); realised rate="
          f"{expit(a + MAR_LOGIT_SLOPE * z).mean():.12f}")
    assert abs(a) < 1e-9
    assert abs(expit(a + MAR_LOGIT_SLOPE * z).mean() - 0.5) < 1e-12

    # And for an arbitrary driver the calibrated mean probability equals the
    # requested proportion to numerical precision.
    rng = np.random.default_rng(4)
    z2 = rng.standard_normal(2000)
    z2 = (z2 - z2.mean()) / z2.std()
    for target in (0.05, 0.2, 0.45):
        a2 = _calibrate_intercept(z2, MAR_LOGIT_SLOPE, target)
        achieved = float(expit(a2 + MAR_LOGIT_SLOPE * z2).mean())
        print(f"target={target} intercept={a2:.6f} mean p_i={achieved:.12f}")
        assert abs(achieved - target) < 1e-9


def test_mar_missingness_is_associated_with_the_driver() -> None:
    n, p, prop = 5000, 8, 0.15
    data = _likert_data(n, p, seed=123).astype(float)
    driver = data[:, 0].copy() + data[:, 1].copy()   # an observed composite
    out, n_missing = inject_missing(
        data, MissingMechanism.MAR, prop, np.random.default_rng(SEED), driver
    )

    realised = n_missing / (n * p)
    any_missing = np.isnan(out).any(axis=1)
    g1 = driver[any_missing]
    g0 = driver[~any_missing]
    se = np.sqrt(g1.var(ddof=1) / g1.size + g0.var(ddof=1) / g0.size)
    t = (g1.mean() - g0.mean()) / se
    print(f"MAR: requested={prop} realised={realised:.5f} "
          f"|diff|={abs(realised - prop):.5f}")
    print(f"driver mean: incomplete rows={g1.mean():.4f} (n={g1.size}) "
          f"complete rows={g0.mean():.4f} (n={g0.size}) t={t:.3f}")
    assert abs(realised - prop) < 0.02
    assert abs(t) > 3.0, f"MAR is not associated with the driver (t={t:.3f})"
    # Direction: a positive slope means high scorers skip more items.
    assert t > 0

    # Per-row rate must be monotone in the driver: top decile loses more than
    # the bottom decile.
    order = np.argsort(driver)
    bottom = np.isnan(out[order[: n // 10]]).mean()
    top = np.isnan(out[order[-(n // 10):]]).mean()
    print(f"missing rate: bottom decile={bottom:.4f} top decile={top:.4f}")
    assert top > bottom + 0.05


def test_mar_rejects_a_constant_or_mis_sized_driver() -> None:
    data = _likert_data(100, 4).astype(float)
    rng = np.random.default_rng(0)
    try:
        inject_missing(data, MissingMechanism.MAR, 0.1, rng,
                       np.zeros(100))
    except SpecificationError as exc:
        print(f"constant driver -> ar: {exc.message_ar}")
        assert exc.message_ar != exc.message_en
    else:  # pragma: no cover - must raise
        raise AssertionError("a constant MAR driver was accepted")
    try:
        inject_missing(data, MissingMechanism.MAR, 0.1, rng, np.arange(7.0))
    except SpecificationError as exc:
        print(f"wrong-length driver -> ar: {exc.message_ar}")
    else:  # pragma: no cover - must raise
        raise AssertionError("a mis-sized MAR driver was accepted")
    try:
        inject_missing(data, "sometimes", 0.1, rng)
    except SpecificationError as exc:
        print(f"unknown mechanism -> ar: {exc.message_ar}")
        assert "mcar" in exc.message_ar
    else:  # pragma: no cover - must raise
        raise AssertionError("an unknown mechanism was accepted")
    assert ConvergenceError is not None  # imported for the calibration guard


def test_mechanism_none_and_zero_proportion_are_no_ops() -> None:
    data = _likert_data(80, 5)
    rng = np.random.default_rng(1)
    out, k = inject_missing(data, MissingMechanism.NONE, 0.3, rng)
    print(f"mechanism=none -> missing={k}, dtype preserved={out.dtype}")
    assert k == 0 and np.array_equal(out, data) and out.dtype == data.dtype
    out, k = inject_missing(data, MissingMechanism.MCAR, 0.0, rng)
    assert k == 0 and np.array_equal(out, data) and out.dtype == data.dtype


# --------------------------------------------------------------------------- #
# 4. apply_noise orchestration
# --------------------------------------------------------------------------- #


def test_apply_noise_zero_everywhere_is_bit_identical() -> None:
    spec = _spec_with_noise(
        careless_proportion=0.0,
        outlier_proportion=0.0,
        missing_mechanism=MissingMechanism.NONE,
        missing_proportion=0.0,
    )
    data = _likert_data(200, 6)
    before = data.copy()
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))
    print(f"zero noise: identical={np.array_equal(out, before)} "
          f"dtype={out.dtype} report={report}")
    assert np.array_equal(out, before)
    assert out.dtype == before.dtype
    assert out.tobytes() == before.tobytes()
    assert (report.n_careless, report.n_outlier_cells,
            report.n_missing_cells) == (0, 0, 0)
    assert report.applied == []
    assert np.array_equal(data, before), "input mutated in place"


def test_apply_noise_disabled_is_a_no_op() -> None:
    spec = _spec_with_noise(enabled=False, careless_proportion=0.5,
                            outlier_proportion=0.5)
    data = _likert_data(50, 6)
    out, report = apply_noise(data, spec, None, np.random.default_rng(0))
    assert np.array_equal(out, data) and isinstance(report, NoiseReport)
    assert report.applied == []


def test_apply_noise_runs_all_three_and_never_mutates_the_input() -> None:
    spec = _spec_with_noise(
        careless_proportion=0.05,
        careless_style="straightline",
        outlier_proportion=0.02,
        outlier_magnitude_sd=3.5,
        missing_mechanism=MissingMechanism.MCAR,
        missing_proportion=0.08,
    )
    n, p = 1000, 6
    data = _likert_data(n, p)
    before = data.copy()
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))

    realised_missing = report.n_missing_cells / (n * p)
    print(f"apply_noise: careless={report.n_careless} "
          f"(expected {round(0.05 * n)}), outlier_cells="
          f"{report.n_outlier_cells} (expected {round(0.02 * n * p)}), "
          f"missing={report.n_missing_cells} rate={realised_missing:.4f}")
    print(f"applied trace: {report.applied}")
    assert report.n_careless == round(0.05 * n) == 50
    assert len(set(report.careless_rows)) == report.n_careless
    assert report.n_outlier_cells == round(0.02 * n * p) == 120
    assert abs(realised_missing - 0.08) < 0.01
    assert len(report.applied) == 3
    assert np.isnan(out).sum() == report.n_missing_cells
    # Observed values other than nan stay inside the 1..5 Likert range.
    finite = out[np.isfinite(out)]
    print(f"observed range after all noise: [{finite.min()}, {finite.max()}]")
    assert finite.min() >= 1 and finite.max() <= 5
    assert np.array_equal(data, before), "apply_noise mutated its input"
    assert data.dtype == before.dtype


def test_apply_noise_mar_uses_the_requested_driver_construct() -> None:
    spec = _spec_with_noise(
        missing_mechanism=MissingMechanism.MAR,
        missing_proportion=0.15,
    )
    spec.noise.mar_driver = spec.constructs[0].id
    n, p = 4000, 6
    data = _likert_data(n, p, seed=321)
    before = data.copy()
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))

    # apply_noise drives MAR from the construct's observed composite score.
    driver = before[:, :3].mean(axis=1)
    any_missing = np.isnan(out).any(axis=1)
    g1, g0 = driver[any_missing], driver[~any_missing]
    se = np.sqrt(g1.var(ddof=1) / g1.size + g0.var(ddof=1) / g0.size)
    t = (g1.mean() - g0.mean()) / se
    realised = report.n_missing_cells / (n * p)
    print(f"apply_noise MAR: realised={realised:.5f} (requested 0.15), "
          f"driver mean incomplete={g1.mean():.4f} complete={g0.mean():.4f} "
          f"t={t:.3f}")
    assert abs(realised - 0.15) < 0.02
    assert abs(t) > 3.0
    assert np.array_equal(data, before)


def test_apply_noise_is_reproducible_and_fast_on_500x60() -> None:
    # 60 items across the two constructs, so the driver composite exists.
    spec = ModelSpec(seed=SEED)
    spec.add_construct("X", n_items=30, n_categories=5)
    spec.add_construct("Y", n_items=30, n_categories=7)
    spec.noise.careless_proportion = 0.06
    spec.noise.careless_style = "mixed"
    spec.noise.outlier_proportion = 0.02
    spec.noise.outlier_magnitude_sd = 3.5
    spec.noise.missing_mechanism = MissingMechanism.MAR
    spec.noise.missing_proportion = 0.05
    spec.noise.mar_driver = spec.constructs[0].id

    data = np.concatenate(
        [_likert_data(500, 30, 5, seed=1), _likert_data(500, 30, 7, seed=2)],
        axis=1,
    )
    t0 = time.perf_counter()
    out_a, rep_a = apply_noise(data, spec, None, np.random.default_rng(SEED))
    elapsed = time.perf_counter() - t0
    out_b, rep_b = apply_noise(data, spec, None, np.random.default_rng(SEED))

    print(f"500x60 apply_noise took {elapsed * 1000:.1f} ms; "
          f"report={rep_a}")
    assert elapsed < 0.5
    assert np.array_equal(np.nan_to_num(out_a, nan=-999.0),
                          np.nan_to_num(out_b, nan=-999.0))
    assert rep_a.careless_rows == rep_b.careless_rows
    assert rep_a.n_missing_cells == rep_b.n_missing_cells
    # Per-item category ranges are respected even though the two constructs
    # use different scales.
    finite_left = out_a[:, :30][np.isfinite(out_a[:, :30])]
    finite_right = out_a[:, 30:][np.isfinite(out_a[:, 30:])]
    print(f"left block range=[{finite_left.min()}, {finite_left.max()}] "
          f"(K=5); right block range=[{finite_right.min()}, "
          f"{finite_right.max()}] (K=7)")
    assert finite_left.min() >= 1 and finite_left.max() <= 5
    assert finite_right.min() >= 1 and finite_right.max() <= 7


def test_bad_matrix_shape_is_rejected() -> None:
    try:
        apply_noise(np.arange(10.0), _spec_with_noise(), None,
                    np.random.default_rng(0))
    except SpecificationError as exc:
        print(f"1-D input -> ar: {exc.message_ar}")
        assert exc.message_ar != exc.message_en
    else:  # pragma: no cover - must raise
        raise AssertionError("a 1-D dataset was accepted")


# --------------------------------------------------------------------------- #
# 5. Regression tests for the repaired defects
# --------------------------------------------------------------------------- #


class _UnresolvableCM:
    """A compiled model whose ``item_ids`` do not exist in the spec.

    This is what a stale or mismatched compile looks like: ids renamed or
    regenerated after compilation, or a matrix that came from a different spec
    object.  ``spec.item(id)`` answers ``None`` for each of them.
    """

    def __init__(self, p: int) -> None:
        self.item_ids = [f"no-such-item-{j}" for j in range(p)]
        self.item_index: dict = {}


def test_scales_for_falls_back_when_every_item_id_is_unresolvable() -> None:
    """A list of ``None`` is not an empty list.

    The old fallback guard was ``if not items:``, which a list made entirely
    of ``None`` passes happily, so when ``cm.item_ids`` could not be resolved
    against the spec the fallback never ran and every column came back with no
    scale at all.
    """
    spec = _spec_with_noise()
    scales = _scales_for(spec, _UnresolvableCM(6), 6)
    is_ordinal, lower, upper = _ordinal_flags_and_bounds(scales, 6)
    print(f"\n  resolved scales: {sum(s is not None for s in scales)}/6 "
          f"non-None; is_ordinal={is_ordinal.tolist()}")
    assert all(s is not None for s in scales)
    assert is_ordinal.all()
    assert np.allclose(lower, 1.0) and np.allclose(upper, 5.0)

    # The real consequence: without the fallback every column read as
    # "continuous, unbounded", so inject_outliers skipped the Likert clip and
    # wrote values such as -1.9 or 7.9 into a 1..5 item.
    data = _likert_data(300, 6, k=5, seed=77)
    out, n_cells = inject_outliers(data, 0.25, 3.5, np.random.default_rng(5),
                                   scales=scales)
    finite = out[np.isfinite(out)]
    print(f"  outlier range with the recovered scales = "
          f"[{finite.min()}, {finite.max()}] over {n_cells} cells")
    assert n_cells > 0
    assert finite.min() >= 1 and finite.max() <= 5


def test_apply_noise_clips_likert_outliers_through_a_stale_compiled_model() -> None:
    """End to end: a mismatched ``cm`` must not yield illegal item codes."""
    spec = _spec_with_noise(outlier_proportion=0.3, outlier_magnitude_sd=4.0)
    data = _likert_data(400, 6, k=5, seed=78)
    out, report = apply_noise(data, spec, _UnresolvableCM(6),
                              np.random.default_rng(SEED))
    finite = out[np.isfinite(out)]
    print(f"\n  apply_noise with an unresolvable cm: "
          f"{report.n_outlier_cells} outlier cells, "
          f"range [{finite.min()}, {finite.max()}]")
    assert report.n_outlier_cells > 0
    assert finite.min() >= 1 and finite.max() <= 5


def test_apply_noise_rejects_a_proportion_outside_the_unit_interval() -> None:
    """A negative proportion used to be silently treated as zero.

    The step guards were ``> 0.0``, which a negative value fails exactly as
    0.0 does, so ``careless_proportion = -0.05`` switched that contamination
    off without a word -- while a value above 1 raised, because it reached the
    ``inject_*`` call that checks it.  The docstring promised a
    ``SpecificationError`` for anything outside ``[0, 1]`` either way, and a
    sign slip or a percent/proportion mix-up is the likelier of the two
    mistakes: the researcher gets a pristine dataset and believes it is
    contaminated.
    """
    data = _likert_data(60, 6)
    checked = 0
    for field in ("careless_proportion", "outlier_proportion",
                  "missing_proportion"):
        for bad in (-0.25, -1e-9, 1.5, float("nan")):
            spec = _spec_with_noise(missing_mechanism=MissingMechanism.MCAR)
            setattr(spec.noise, field, bad)
            try:
                apply_noise(data, spec, None, np.random.default_rng(0))
            except SpecificationError as exc:
                assert exc.message_ar and exc.message_ar != exc.message_en
                checked += 1
            else:  # pragma: no cover - must raise
                raise AssertionError(
                    f"{field}={bad!r} was accepted instead of raising"
                )
    print(f"\n  12 out-of-range proportions all raised "
          f"SpecificationError ({checked} checked)")
    assert checked == 12


def test_apply_noise_leaves_the_mar_driver_fully_observed() -> None:
    """MAR must depend on data the analyst can actually see.

    Reading the driver composite before any cell is blanked is necessary but
    not sufficient: missingness used to be applied to every column, the driver
    construct's own items included, so the quantity the mechanism depended on
    was unrecoverable from the delivered data for a large share of
    respondents.  That makes the mechanism MNAR for those rows and voids the
    assumption every MAR-valid method (FIML, multiple imputation) is tested
    against.
    """
    spec = _spec_with_noise(
        missing_mechanism=MissingMechanism.MAR,
        missing_proportion=0.2,
    )
    spec.noise.mar_driver = spec.constructs[0].id      # items 0..2 of 6
    n, p = 3000, 6
    data = _likert_data(n, p, seed=424)
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))

    driver_nan = int(np.isnan(out[:, :3]).sum())
    other_nan = int(np.isnan(out[:, 3:]).sum())
    realised = report.n_missing_cells / (n * p)
    print(f"\n  driver columns: {driver_nan} blanked cells; "
          f"other columns: {other_nan}; realised overall rate "
          f"{realised:.4f} (requested 0.2)")
    # The driver composite is recomputable for every single respondent.
    assert driver_nan == 0
    assert other_nan == report.n_missing_cells
    # Holding three of six columns out would halve the realised rate, so the
    # rate on the remaining columns is scaled up and the overall share still
    # matches the request.
    assert abs(realised - 0.2) < 0.02
    # And the mechanism is still genuinely driven by the driver.
    driver = data[:, :3].mean(axis=1)
    any_missing = np.isnan(out).any(axis=1)
    g1, g0 = driver[any_missing], driver[~any_missing]
    se = np.sqrt(g1.var(ddof=1) / g1.size + g0.var(ddof=1) / g0.size)
    t = (g1.mean() - g0.mean()) / se
    print(f"  driver mean: incomplete={g1.mean():.4f} "
          f"complete={g0.mean():.4f} t={t:.3f}")
    assert abs(t) > 3.0


def test_apply_noise_mar_driver_spanning_every_column_still_works() -> None:
    """One-construct model: there is nothing to hold out, so do not try."""
    spec = ModelSpec(seed=SEED)
    spec.add_construct("X", n_items=6, n_categories=5)
    spec.noise.enabled = True
    spec.noise.missing_mechanism = MissingMechanism.MAR
    spec.noise.missing_proportion = 0.1
    spec.noise.mar_driver = spec.constructs[0].id
    n, p = 1500, 6
    data = _likert_data(n, p, seed=425)
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))
    realised = report.n_missing_cells / (n * p)
    print(f"\n  driver spans all {p} columns: realised {realised:.4f} "
          "(requested 0.1)")
    assert abs(realised - 0.1) < 0.02
    assert report.n_missing_cells == int(np.isnan(out).sum())


def test_mcar_is_unaffected_by_the_driver_hold_out() -> None:
    """MCAR has no driver, so every column stays eligible."""
    spec = _spec_with_noise(missing_mechanism=MissingMechanism.MCAR,
                            missing_proportion=0.2)
    n, p = 2000, 6
    data = _likert_data(n, p, seed=426)
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))
    per_column = np.isnan(out).mean(axis=0)
    print(f"\n  MCAR per-column missing rates = {np.round(per_column, 4)}")
    assert report.n_missing_cells == int(np.isnan(out).sum())
    assert abs(report.n_missing_cells / (n * p) - 0.2) < 0.02
    assert per_column.min() > 0.15, "no column may be held out under MCAR"


def test_apply_noise_holds_the_fallback_driver_column_out_too() -> None:
    """The guarantee must not depend on which driver resolution fired.

    ``apply_noise`` resolves the MAR driver in three steps: the requested
    construct, then the first exogenous construct, then the first column.  The
    third was reached by leaving ``driver=None`` for ``inject_missing`` to
    resolve internally, which meant ``apply_noise`` did not know which column
    it was and could not hold it out -- so the "driver stays observed"
    guarantee quietly lapsed for exactly the specs that were already in
    trouble.  Naming it explicitly costs nothing and keeps the mechanism
    identical, because ``_resolve_driver`` standardises whatever it is given.
    """
    class _BareNoise:
        enabled = True
        careless_proportion = 0.0
        outlier_proportion = 0.0
        careless_style = "random"
        outlier_magnitude_sd = 3.5
        missing_mechanism = MissingMechanism.MAR
        missing_proportion = 0.15
        mar_driver = ""

    class _BareSpec:
        """A duck-typed spec with no resolvable constructs at all."""

        noise = _BareNoise()

        def all_items(self):
            return []

    n, p = 2000, 6
    data = _likert_data(n, p, seed=427)
    out, report = apply_noise(data, _BareSpec(), None,
                              np.random.default_rng(SEED))
    per_column = np.isnan(out).sum(axis=0)
    realised = report.n_missing_cells / (n * p)
    print(f"\n  first-column fallback: per-column nan={per_column.tolist()}, "
          f"realised={realised:.4f} (requested 0.15)")
    assert per_column[0] == 0, "the fallback driver column was blanked"
    assert (per_column[1:] > 0).all()
    assert abs(realised - 0.15) < 0.02
    # Still a genuine MAR mechanism, driven by that first column.
    driver = data[:, 0].astype(float)
    any_missing = np.isnan(out).any(axis=1)
    g1, g0 = driver[any_missing], driver[~any_missing]
    se = np.sqrt(g1.var(ddof=1) / g1.size + g0.var(ddof=1) / g0.size)
    print(f"  driver(col 0) mean: incomplete={g1.mean():.4f} "
          f"complete={g0.mean():.4f}")
    assert abs((g1.mean() - g0.mean()) / se) > 3.0


def test_apply_noise_exogenous_fallback_driver_is_held_out() -> None:
    """Second resolution step: an unknown ``mar_driver`` id."""
    spec = _spec_with_noise(
        missing_mechanism=MissingMechanism.MAR,
        missing_proportion=0.15,
    )
    spec.noise.mar_driver = "no-such-construct"
    n, p = 2000, 6
    data = _likert_data(n, p, seed=428)
    out, report = apply_noise(data, spec, None, np.random.default_rng(SEED))
    per_column = np.isnan(out).sum(axis=0)
    realised = report.n_missing_cells / (n * p)
    print(f"\n  exogenous fallback: per-column nan={per_column.tolist()}, "
          f"realised={realised:.4f} (requested 0.15)")
    # The first exogenous construct owns columns 0..2.
    assert per_column[:3].sum() == 0
    assert (per_column[3:] > 0).all()
    assert abs(realised - 0.15) < 0.02


def test_achieved_moments_propagate_nan_from_injected_missingness() -> None:
    """A NaN column must not be reported as a perfectly normal one.

    ``vale_maurelli.achieved_moments`` computed its shape statistics with
    ``np.where(m2 > 0.0, ..., 0.0)``, and every comparison against NaN is
    False, so the *else* branch was taken for a column whose ``m2`` was NaN:
    skewness 0.0 and excess kurtosis 0.0, indistinguishable from a normal
    column, while ``mean`` and ``sd`` for the same column correctly read NaN.
    NaN is exactly what :func:`inject_missing` writes, so this misreported the
    moments of the engine's own output whenever missingness was switched on.
    The test lives here because ``inject_missing`` is where the NaN comes
    from.
    """
    rng = np.random.default_rng(31)
    # A strongly skewed column, so "skewness 0.0" is unmistakably wrong.
    x = np.column_stack([
        rng.standard_normal(2000),
        rng.exponential(size=2000),
        rng.standard_normal(2000),
    ])
    clean = achieved_moments(x)
    assert clean["skewness"][1] > 1.0, clean["skewness"]

    holed, n_missing = inject_missing(
        x, MissingMechanism.MCAR, 0.05, np.random.default_rng(SEED)
    )
    assert n_missing > 0
    moments = achieved_moments(holed)
    print(f"\n  complete-data skewness = {np.round(clean['skewness'], 4)}"
          f"\n  after {n_missing} blanked cells: "
          f"skew={np.round(moments['skewness'], 4)} "
          f"kurt={np.round(moments['kurtosis'], 4)}")
    assert np.all(np.isnan(moments["skewness"])), moments["skewness"]
    assert np.all(np.isnan(moments["kurtosis"])), moments["kurtosis"]
    assert np.all(np.isnan(moments["skew"]))
    assert np.all(np.isnan(moments["excess_kurtosis"]))
    # Dropping the incomplete rows recovers the complete-data answer.
    kept = holed[~np.isnan(holed).any(axis=1)]
    recovered = achieved_moments(kept)
    assert np.all(np.isfinite(recovered["skewness"]))
    assert abs(recovered["skewness"][1] - clean["skewness"][1]) < 0.3


def test_achieved_moments_still_report_zero_shape_for_a_constant_column() -> None:
    """A genuinely constant column has no shape; that is not the NaN case."""
    x = np.column_stack([np.full(100, 3.0), np.linspace(-1.0, 1.0, 100)])
    moments = achieved_moments(x)
    print(f"\n  constant column: sd={moments['sd'][0]:.4f} "
          f"skew={moments['skewness'][0]:.4f} "
          f"kurt={moments['kurtosis'][0]:.4f}")
    assert moments["sd"][0] == 0.0
    assert moments["skewness"][0] == 0.0
    assert moments["kurtosis"][0] == 0.0
    assert np.isfinite(moments["skewness"]).all()


if __name__ == "__main__":  # pragma: no cover - manual runner
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
            else:
                print(f"PASS {name}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


def test_a_missing_rate_above_the_holdout_ceiling_is_reported() -> None:
    """Holding the MAR driver out caps the attainable missing rate.

    Scaling the remaining columns up is clamped at 1, so once the request
    exceeds ``1 - len(driver)/p`` it simply cannot be met. That is a
    defensible trade-off -- missingness that depends on data the researcher
    cannot see is not MAR -- but a caller reading only the report had no way
    to see the realised rate fall short, so the shortfall is recorded with
    its numbers and its remedies.
    """
    import numpy as np

    from semsim.core.compile import compile_model
    from semsim.core.spec import ConstructRole, MissingMechanism, ModelSpec
    from semsim.generation.noise import _mar_driver_columns, apply_noise

    def build(requested: float) -> ModelSpec:
        spec = ModelSpec(seed=3)
        driver = spec.add_construct("D", n_items=3, loading=0.8)
        spec.add_construct("O", role=ConstructRole.DEPENDENT, n_items=3,
                           loading=0.8)
        spec.ensure_unique_item_names()
        spec.add_path("D", "O")
        spec.noise.enabled = True
        spec.noise.careless_proportion = 0.0
        spec.noise.outlier_proportion = 0.0
        spec.noise.missing_mechanism = MissingMechanism.MAR
        spec.noise.missing_proportion = requested
        spec.noise.mar_driver = driver.id
        return spec

    data = np.random.default_rng(1).integers(1, 6, size=(600, 6)).astype(float)
    ceiling = 0.5          # three driver columns out of six

    for requested, expect_note in ((0.10, False), (0.30, False),
                                   (0.49, False), (0.60, True),
                                   (0.90, True), (1.00, True)):
        spec = build(requested)
        compiled = compile_model(spec)
        cols = _mar_driver_columns(spec, compiled, spec.noise.mar_driver, 6)
        assert len(cols) == 3, cols
        out, report = apply_noise(data.copy(), spec, compiled,
                                  np.random.default_rng(7))
        realised = float(np.isnan(out).mean())
        notes = [line for line in report.applied if "requested" in line]

        # The driver must survive intact whatever was asked for: that is the
        # whole reason the ceiling exists.
        assert not np.isnan(out[:, cols]).any(), requested

        if expect_note:
            assert notes, f"no shortfall recorded for {requested}"
            assert abs(realised - ceiling) < 0.01, realised
            assert f"{requested:.3f}" in notes[0]
            assert f"{ceiling:.3f}" in notes[0]
        else:
            assert not notes, notes
            assert abs(realised - requested) < 0.02, (requested, realised)

"""
End-to-end contract tests.

These are deliberately not unit tests.  Each module has its own suite proving
its own maths; what is checked here is that the pieces still mean the same
thing to each other -- that the degrees of freedom the specification reports
are the ones the estimator uses, that a dataset generated from a model fits
that model, that the seed really does reproduce a run, and that every export
path produces a file something else can read.

They run on the bundled example models, so a regression in any layer shows up
as a failure on a realistic design rather than on a toy.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from semsim.core import examples
from semsim.core.compile import compile_model
from semsim.core.errors import SemSimError
from semsim.core.spec import FitLevel, GenerationEngine, ScaleKind
from semsim.core.validation import validate
from semsim.generation.engine import generate_dataset
from semsim.io import project as project_io

ALL_EXAMPLES = list(examples.example_names())


# --------------------------------------------------------------------------- #
# The bundled models
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_EXAMPLES)
def test_example_validates_clean(name: str) -> None:
    """Every shipped example must be a model a researcher could actually run."""
    report = validate(examples.build(name))
    errors = [d.code for d in report.errors()]
    assert not errors, f"{name} raised {errors}"


@pytest.mark.parametrize("name", ALL_EXAMPLES)
def test_example_compiles_to_a_correlation_matrix(name: str) -> None:
    from semsim.core import implied

    cm = compile_model(examples.build(name))
    phi = implied.latent_covariance(cm)
    sigma = implied.implied_covariance(cm)
    assert np.allclose(np.diag(phi), 1.0, atol=1e-9), name
    assert np.allclose(np.diag(sigma), 1.0, atol=1e-9), name
    assert np.allclose(sigma, sigma.T, atol=1e-12), name
    assert implied.is_positive_definite(sigma), name


@pytest.mark.parametrize("name", ALL_EXAMPLES)
def test_specification_df_matches_the_estimator(name: str) -> None:
    """The headline number in the interface must be the one the fit uses.

    If these drift apart, every chi-square, RMSEA and CFI the program reports
    is computed against the wrong degrees of freedom.
    """
    from semsim.analysis.estimator import estimate
    from semsim.core import implied

    spec = examples.build(name)
    cm = compile_model(spec)
    sigma = implied.implied_covariance(cm)
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=400),
        columns=cm.item_names)
    result = estimate(data, spec)
    assert result.df == spec.degrees_of_freedom(cm), (
        f"{name}: spec says {spec.degrees_of_freedom(cm)}, "
        f"estimator says {result.df}")
    assert result.n_free == spec.count_free_parameters(cm)


@pytest.mark.parametrize("name", ALL_EXAMPLES)
def test_example_generates(name: str) -> None:
    spec = examples.build(name)
    dataset = generate_dataset(spec, n=200)
    assert dataset.data.shape == (200, spec.n_items)
    assert list(dataset.data.columns) == [i.name for i in spec.all_items()]
    ordinal = [i for i in spec.all_items() if i.scale.kind is ScaleKind.LIKERT]
    for item in ordinal:
        column = dataset.data[item.name].dropna()
        if column.empty:
            continue
        assert column.min() >= 1, f"{name}/{item.name} below the scale"
        assert column.max() <= item.scale.n_categories, \
            f"{name}/{item.name} above the scale"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_the_same_seed_reproduces_the_dataset_exactly() -> None:
    spec = examples.simple_mediation()
    first = generate_dataset(spec, n=300)
    second = generate_dataset(spec, n=300)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.seed == second.seed


def test_a_different_seed_produces_a_different_dataset() -> None:
    spec = examples.simple_mediation()
    first = generate_dataset(spec, n=300, seed=1)
    second = generate_dataset(spec, n=300, seed=2)
    assert not first.data.equals(second.data)


def test_compilation_is_deterministic_including_the_auto_policy() -> None:
    """The cross-loading policy draws at random; the same spec must still
    compile to the same matrices, or a saved project would not reopen the
    same way.

    Note the two halves. Compiling ONE spec twice must give identical ids,
    because they are the same objects. Building the spec twice gives fresh
    uuids by design, so only the numbers can be compared there.
    """
    spec = examples.messy_reality()
    first, second = compile_model(spec), compile_model(spec)
    assert np.allclose(first.Lambda, second.Lambda, atol=0.0)
    assert first.auto_cross_loadings == second.auto_cross_loadings

    rebuilt = compile_model(examples.messy_reality())
    assert np.allclose(first.Lambda, rebuilt.Lambda, atol=0.0)
    assert len(first.auto_cross_loadings) == len(rebuilt.auto_cross_loadings)
    assert [first.item_names[first.item_index[i]]
            for i, _f in first.auto_cross_loadings] == \
           [rebuilt.item_names[rebuilt.item_index[i]]
            for i, _f in rebuilt.auto_cross_loadings]


# --------------------------------------------------------------------------- #
# Generation fidelity
# --------------------------------------------------------------------------- #


def test_generated_correlations_match_the_generating_matrix() -> None:
    """On continuous items the realised correlations must be the target ones,
    within sampling error."""
    spec = examples.minimal_cfa()
    spec.power.n = 4000
    dataset = generate_dataset(spec)
    realised = np.corrcoef(dataset.data.to_numpy().T)
    error = float(np.abs(realised - dataset.sigma_generating).max())
    # 4 standard errors of a correlation at n = 4000 is about 0.063.
    assert error < 0.07, f"max |r - sigma| = {error:.4f}"


def test_reverse_coded_items_run_against_their_factor() -> None:
    """A reverse-worded item that has not been recoded must correlate
    negatively with its siblings -- that is the whole point of the flag, and
    the diagnostic a researcher is meant to notice."""
    spec = examples.minimal_cfa()
    spec.power.n = 2000
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS
    spec.constructs[0].items[0].reverse_coded = True
    dataset = generate_dataset(spec)
    names = [i.name for i in spec.constructs[0].items]
    reversed_name, sibling = names[0], names[1]
    r = float(dataset.data[reversed_name].corr(dataset.data[sibling]))
    assert r < -0.3, f"reverse-coded item correlated {r:+.3f} with its sibling"


def test_a_latent_interaction_is_recoverable_from_the_scores() -> None:
    """The moderation example must actually contain its interaction.

    A product of two latent variables cannot live in a covariance matrix, so
    this is the test that the score engine is doing its job.
    """
    spec = examples.moderation()
    spec.power.n = 4000
    spec.export.include_latent_scores = True
    dataset = generate_dataset(spec)
    assert dataset.engine_used is GenerationEngine.SCORES
    eta = dataset.latent_scores
    assert eta is not None and eta.shape[1] >= 3
    x = eta.iloc[:, 0].to_numpy()
    m = eta.iloc[:, 1].to_numpy()
    y = eta.iloc[:, 2].to_numpy()
    design = np.column_stack([np.ones_like(x), x, m, x * m])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert abs(beta[1] - 0.50) < 0.08, f"main effect {beta[1]:+.3f}"
    assert abs(beta[2] + 0.28) < 0.08, f"moderator effect {beta[2]:+.3f}"
    assert abs(beta[3] + 0.18) < 0.08, f"interaction {beta[3]:+.3f}"


def test_noise_lands_where_it_was_asked_to() -> None:
    spec = examples.messy_reality()
    spec.power.n = 2000
    spec.noise.missing_proportion = 0.08
    spec.noise.careless_proportion = 0.05
    dataset = generate_dataset(spec)
    realised_missing = float(dataset.data.isna().to_numpy().mean())
    assert abs(realised_missing - 0.08) < 0.02, realised_missing
    report = dataset.noise_report
    assert report is not None
    # The share is of the whole file: 105 careless on top of 2000 is 5.0%.
    assert getattr(report, "n_careless", 0) == round(2000 * 0.05 / 0.95)
    assert len(dataset.full_frame) == 2000 + report.n_careless


# --------------------------------------------------------------------------- #
# The data fits the model it came from
# --------------------------------------------------------------------------- #


def test_data_generated_without_model_error_fits_almost_perfectly() -> None:
    """The sharpest end-to-end check there is: if a dataset drawn from
    Sigma(theta) does not fit the model that produced it, something between
    the compiler and the estimator disagrees."""
    from semsim.analysis.estimator import estimate
    from semsim.analysis.fit_indices import compute

    spec = examples.simple_mediation()
    spec.power.n = 1500
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    spec.distribution.use_non_normal_engine = False
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS

    dataset = generate_dataset(spec)
    result = estimate(dataset.data, spec)
    fit = compute(result.sample_cov, result.sigma_hat, result.n, result.n_free)
    assert result.converged
    assert fit.rmsea < 0.035, f"RMSEA {fit.rmsea:.4f}"
    assert fit.cfi > 0.975, f"CFI {fit.cfi:.4f}"
    assert fit.srmr < 0.055, f"SRMR {fit.srmr:.4f}"


def test_population_parameters_are_recovered() -> None:
    from semsim.analysis.estimator import estimate

    spec = examples.simple_mediation()
    spec.power.n = 3000
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    spec.distribution.use_non_normal_engine = False
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS

    dataset = generate_dataset(spec)
    result = estimate(dataset.data, spec)
    by_pair = {(r["source"], r["target"]): r["std"]
               for _i, r in result.path_estimates.iterrows()}
    assert abs(by_pair[("SAT", "ENG")] - 0.50) < 0.06
    assert abs(by_pair[("ENG", "PERF")] - 0.30) < 0.06
    assert abs(by_pair[("SAT", "PERF")] - 0.15) < 0.06
    loadings = result.loadings["std"].to_numpy()
    assert loadings.min() > 0.65, f"lowest standardised loading {loadings.min():.3f}"


@pytest.mark.parametrize("level,expected", [
    (FitLevel.EXCELLENT, 0.020),
    (FitLevel.GOOD, 0.045),
    (FitLevel.ACCEPTABLE, 0.070),
])
def test_fit_targeting_lands_near_its_target(level: FitLevel,
                                             expected: float) -> None:
    """Calibration must put the realised RMSEA where the researcher asked."""
    from semsim.targeting.calibration import calibrate

    spec = examples.simple_mediation()
    spec.power.n = 800
    spec.fit_target.level = level
    spec.fit_target.max_calibration_rounds = 4
    spec.fit_target.tolerance = 0.008
    spec.significance_target.max_resamples = 0
    spec.noise.enabled = False
    spec.distribution.use_non_normal_engine = False
    for item in spec.all_items():
        item.scale.kind = ScaleKind.CONTINUOUS

    result = calibrate(spec)
    assert result.fit is not None
    assert abs(result.realised_rmsea - expected) < 0.012, (
        f"asked for {expected}, got {result.realised_rmsea:.4f} "
        f"after {result.rounds} round(s)")


def test_fit_targeting_off_gives_a_near_perfect_fit() -> None:
    from semsim.targeting.calibration import calibrate

    spec = examples.minimal_cfa()
    spec.power.n = 600
    spec.fit_target.enabled = False
    result = calibrate(spec)
    assert result.realised_rmsea < 0.03


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_EXAMPLES)
def test_project_round_trip_is_exact(name: str) -> None:
    spec = examples.build(name)
    with tempfile.TemporaryDirectory() as directory:
        path = project_io.save(spec, os.path.join(directory, name))
        reloaded = project_io.load(path)
    assert project_io.to_dict(spec)["model"] == project_io.to_dict(reloaded)["model"]


def test_a_reloaded_project_generates_the_same_data() -> None:
    """Saving and reopening must not change a single value, or a shared
    project is not the same study."""
    spec = examples.messy_reality()
    with tempfile.TemporaryDirectory() as directory:
        path = project_io.save(spec, os.path.join(directory, "p"))
        reloaded = project_io.load(path)
    pd.testing.assert_frame_equal(
        generate_dataset(spec, n=250).data,
        generate_dataset(reloaded, n=250).data)


def test_report_is_self_contained() -> None:
    from semsim.io.report import ReportBundle, render_html

    spec = examples.simple_mediation()
    dataset = generate_dataset(spec, n=200)
    bundle = ReportBundle(spec=spec, dataset=dataset,
                          generated_at="2026-01-01 00:00")
    for lang in ("ar", "en"):
        html = render_html(bundle, lang=lang)
        assert "<html" in html and "</html>" in html
        assert 'src="http' not in html and "href=\"http" not in html
        assert f'dir="{"rtl" if lang == "ar" else "ltr"}"' in html
        assert "<style>" in html


# --------------------------------------------------------------------------- #
# Failure behaviour
# --------------------------------------------------------------------------- #


def test_an_impossible_sample_size_is_refused_politely() -> None:
    spec = examples.minimal_cfa()
    with pytest.raises(SemSimError) as excinfo:
        generate_dataset(spec, n=2)
    assert excinfo.value.message_ar
    assert excinfo.value.message_ar != excinfo.value.message_en


def test_every_error_carries_a_real_arabic_message() -> None:
    """A user-facing error with an English-only message is a bug: the
    interface has nothing to show an Arabic-speaking researcher."""
    from semsim.core import errors as error_module

    for name in error_module.__all__:
        cls = getattr(error_module, name)
        if not isinstance(cls, type) or not issubclass(cls, SemSimError):
            continue
        instance = cls("English text", "نص عربي")
        assert instance.message("ar") == "نص عربي"
        assert instance.message("en") == "English text"


def test_validation_is_fast_enough_to_run_on_every_keystroke() -> None:
    import time

    spec = examples.messy_reality()
    start = time.perf_counter()
    for _ in range(20):
        validate(spec)
    per_call_ms = (time.perf_counter() - start) / 20 * 1000
    assert per_call_ms < 25.0, f"{per_call_ms:.1f} ms per call"

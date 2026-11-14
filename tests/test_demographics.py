"""
Demographic variables: quotas, roles, and controls that enter the model.

The promises checked here are the ones a researcher relies on: a 60/40 quota
gives 60/40, a control path comes back at the size and significance requested,
the substantive paths are left alone, and a model without controls generates
the same numbers it always did.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from semsim.analysis import demographics as analysis
from semsim.analysis.estimator import estimate
from semsim.core import examples
from semsim.core.compile import compile_model
from semsim.core.spec import (
    DemographicKind, DemographicRole, EffectSize, MissingMechanism, ModelSpec,
)
from semsim.core.validation import validate
from semsim.generation.demographics import allocate_quota
from semsim.generation.engine import generate_dataset
from semsim.io import exporters, project
from semsim.io.report import ReportBundle, render_html
from semsim.targeting.calibration import calibrate

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def plain_model():
    spec = examples.simple_mediation()
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    # Most tests here count cases per category, which only a quota design
    # fixes in advance; weighted sampling has tests of its own.
    spec.quota_exact = True
    return spec


def with_controls(**kwargs):
    spec = plain_model()
    target = spec.constructs[-1]
    gender = spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)],
                                  role=DemographicRole.CONTROL)
    spec.add_control_path(gender, target.id, coefficient=0.20, **kwargs)
    return spec


def test_several_controls_are_estimated_not_left_at_the_start():
    # Small dummy variances once stopped L-BFGS-B on its first step, and the
    # generating values came back labelled as converged estimates.
    spec = plain_model()
    mediator, outcome = spec.constructs[1], spec.constructs[2]
    gender = spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)],
                                  role=DemographicRole.CONTROL)
    spec.add_control_path(gender, mediator.id)
    job = spec.add_demographic("Job", [("a", 0.5), ("b", 0.35), ("c", 0.15)],
                               role=DemographicRole.CONTROL)
    first, second = job.terms()
    spec.add_control_path(job, mediator.id, term=first.key)
    spec.add_control_path(job, outcome.id, term=first.key)
    spec.add_control_path(job, outcome.id, term=second.key)

    ds = generate_dataset(spec, seed=7)
    fitted = estimate(ds.model_frame, spec, estimator=spec.analysis.estimator)

    assert fitted.converged
    assert fitted.n_iter > 1
    assert "OPTIMISER: start" not in fitted.message
    assert fitted.fmin < 0.6


def codes(ds, name):
    return ds.demographics[name].value_counts().sort_index().tolist()


# --------------------------------------------------------------------------- #
# Quotas
# --------------------------------------------------------------------------- #


def test_quota_is_exact():
    spec = plain_model()
    spec.quota_exact = True
    spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)])
    spec.add_demographic("Edu", [("b", 0.7), ("p", 0.3)],
                         kind=DemographicKind.ORDINAL)
    ds = generate_dataset(spec, n=300, seed=1)
    assert codes(ds, "Gender") == [180, 120]
    assert codes(ds, "Edu") == [210, 90]


def test_quota_sums_to_n_when_shares_do_not_divide_evenly():
    counts = allocate_quota([1 / 3, 1 / 3, 1 / 3], 100)
    assert counts.sum() == 100 and sorted(counts) == [33, 33, 34]


def test_weighted_sampling_is_the_default():
    # A collected sample never lands exactly on its target shares, so new
    # models draw respondents at random unless a quota design is asked for.
    assert ModelSpec().quota_exact is False


def test_weighted_sampling_is_random_but_on_target():
    spec = plain_model()
    spec.quota_exact = False
    spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)])
    first = generate_dataset(spec, n=4000, seed=1).demographics["Gender"]
    second = generate_dataset(spec, n=4000, seed=2).demographics["Gender"]
    assert first.value_counts().iloc[0] != second.value_counts().iloc[0]
    assert abs(float((first == 1).mean()) - 0.6) < 0.04


def test_same_seed_gives_the_same_people():
    spec = with_controls()
    a = generate_dataset(spec, n=200, seed=9)
    b = generate_dataset(spec, n=200, seed=9)
    assert a.demographics.equals(b.demographics) and a.data.equals(b.data)


# --------------------------------------------------------------------------- #
# Nothing changes for a model without controls
# --------------------------------------------------------------------------- #


def test_descriptive_variable_leaves_the_answers_untouched():
    spec = plain_model()
    before = generate_dataset(spec, n=300, seed=5).data
    spec.add_demographic("Gender", [("m", 0.5), ("f", 0.5)])
    after = generate_dataset(spec, n=300, seed=5)
    assert after.data.equals(before)
    assert spec.expanded() is spec


def test_a_model_without_demographics_has_no_extra_columns():
    ds = generate_dataset(plain_model(), n=100, seed=5)
    assert ds.demographics is None
    assert ds.model_frame is ds.data and ds.full_frame is ds.data


# --------------------------------------------------------------------------- #
# Controls in the model
# --------------------------------------------------------------------------- #


def test_expansion_is_idempotent_and_leaves_the_original_alone():
    spec = with_controls()
    n_before = len(spec.constructs)
    full = spec.expanded()
    assert full.expanded() is full
    assert len(spec.constructs) == n_before
    assert len(full.constructs) == n_before + 1


def test_control_without_a_path_stays_out_of_the_model():
    spec = plain_model()
    spec.add_demographic("Gender", [("m", 0.5), ("f", 0.5)],
                         role=DemographicRole.CONTROL)
    assert spec.expanded() is spec


def test_control_is_observed_without_error():
    cm = compile_model(with_controls())
    row = cm.covariate_rows[0]
    assert row == cm.n_items - 1 and cm.Theta[row, row] == 0.0
    assert cm.Lambda[row, cm.covariate_factors[0]] == 1.0
    assert np.allclose(np.diag(cm.Phi), 1.0)


def test_degrees_of_freedom_match_the_estimator():
    spec = with_controls()
    ds = generate_dataset(spec, n=500, seed=2)
    result = estimate(ds.model_frame, spec, use_polychoric=False)
    assert result.df == spec.degrees_of_freedom(ds.compiled)


def test_control_coefficient_and_substantive_paths_are_recovered():
    spec = with_controls()
    # Recovery is checked against clean data: answering styles add a little
    # method variance on purpose, which shaves about 0.006 off this
    # coefficient and is tested on its own.
    spec.respondent_styles.enabled = False
    ds = generate_dataset(spec, n=6000, seed=11)
    result = estimate(ds.model_frame, spec, use_polychoric=False)
    table = result.path_estimates.set_index(["source", "target"])
    assert table.loc[("Gender", "PERF"), "std"] == pytest.approx(0.20, abs=0.03)
    assert table.loc[("Gender", "PERF"), "p"] < 0.05
    for path in spec.paths:
        source = spec.construct(path.source).name
        target = spec.construct(path.target).name
        assert table.loc[(source, target), "std"] == pytest.approx(
            path.resolved_coefficient(), abs=0.05)


def test_a_control_lowers_the_targets_disturbance_only():
    bare = compile_model(plain_model())
    full = compile_model(with_controls())
    name = "PERF"
    assert full.B[full.factor_index[full.spec.constructs[2].id],
                  full.factor_names.index("ENG")] == pytest.approx(
        bare.B[bare.factor_names.index(name), bare.factor_names.index("ENG")])
    assert (full.Psi[full.factor_names.index(name), full.factor_names.index(name)]
            < bare.Psi[bare.factor_names.index(name), bare.factor_names.index(name)])


def test_nominal_with_three_categories_becomes_two_dummies():
    spec = plain_model()
    sector = spec.add_demographic("Sector",
                                  [("a", 0.5), ("b", 0.3), ("c", 0.2)],
                                  role=DemographicRole.CONTROL)
    assert [t.column for t in sector.terms()] == ["Sector_2", "Sector_3"]
    for term in sector.terms():
        spec.add_control_path(sector, spec.constructs[-1].id, term=term.key,
                              coefficient=-0.2)
    ds = generate_dataset(spec, n=1000, seed=3)
    assert list(ds.demographics.columns) == ["Sector", "Sector_2", "Sector_3"]
    assert ds.demographics["Sector_2"].sum() == 300
    assert ds.demographics["Sector_3"].sum() == 200
    cm = ds.compiled
    a, b = cm.covariate_factors
    expected = -np.sqrt(0.3 * 0.2 / (0.7 * 0.8))
    assert cm.Phi[a, b] == pytest.approx(expected)
    result = estimate(ds.model_frame, spec, use_polychoric=False)
    assert result.converged and result.df == spec.degrees_of_freedom(cm)


def test_control_works_under_the_score_engine():
    spec = with_controls()
    a, b, c = spec.constructs
    spec.add_interaction(a.name, b.name, c.name, coefficient=0.1)
    assert spec.resolved_engine().value == "scores"
    ds = generate_dataset(spec, n=800, seed=4)
    assert codes(ds, "Gender") == [480, 320]
    assert ds.data.shape[1] == spec.n_items


def test_noise_touches_items_only():
    spec = with_controls()
    spec.noise.enabled = True
    spec.noise.missing_mechanism = MissingMechanism.MCAR
    spec.noise.missing_proportion = 0.05
    ds = generate_dataset(spec, n=400, seed=4)
    assert ds.has_missing
    assert not ds.demographics.isna().to_numpy().any()
    assert codes(ds, "Gender") == [240, 160]


def test_significance_target_controls_the_p_value():
    spec = with_controls(target_significant=False)
    spec.fit_target.enabled = True
    spec.significance_target.enabled = True
    result = calibrate(spec, n=800)
    row = result.significance_table.set_index("source").loc["Gender"]
    assert row["requested"] == "not significant"
    assert row["verdict"] == "pass" and row["p"] >= 0.05


def test_power_covers_control_paths():
    from semsim.power import analysis as power
    spec = with_controls()
    result = power.analyse(spec)
    assert "Gender" in set(result.per_path_power["source"])


# --------------------------------------------------------------------------- #
# Roles produce results
# --------------------------------------------------------------------------- #


def role_model():
    spec = plain_model()
    spec.add_demographic("Sector", [("a", 0.5), ("b", 0.3), ("c", 0.2)],
                         role=DemographicRole.COMPARISON)
    spec.add_demographic("Edu", [("x", 0.7), ("y", 0.3)],
                         kind=DemographicKind.ORDINAL,
                         role=DemographicRole.COMPARISON)
    spec.add_demographic("Region", [("n", 0.5), ("s", 0.5)],
                         role=DemographicRole.MGA)
    return spec


@pytest.fixture(scope="module")
def roles():
    spec = role_model()
    ds = generate_dataset(spec, n=600, seed=3)
    return spec, ds, analysis.analyse(spec, ds)


def test_frequencies_report_target_and_achieved(roles):
    _spec, _ds, result = roles
    row = result.frequencies[result.frequencies["variable"] == "Sector"].iloc[1]
    assert row["target_percent"] == pytest.approx(30.0)
    assert row["n"] == 180 and row["achieved_percent"] == pytest.approx(30.0)


def test_comparison_uses_t_for_two_groups_and_anova_for_three(roles):
    spec, _ds, result = roles
    tests = result.comparisons.groupby("variable")["test"].first().to_dict()
    assert tests == {"Sector": "anova_f", "Edu": "welch_t"}
    assert len(result.comparisons) == 2 * len(spec.constructs)


def test_welch_matches_scipy():
    from scipy import stats
    a, b = np.array([1.0, 2.0, 4.0, 5.0]), np.array([2.0, 3.0, 3.5, 6.0, 8.0])
    t, _df, p, _d = analysis._welch(a, b)
    reference = stats.ttest_ind(a, b, equal_var=False)
    assert t == pytest.approx(reference.statistic)
    assert p == pytest.approx(reference.pvalue)


def test_anova_matches_scipy():
    from scipy import stats
    groups = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]),
              np.array([5.0, 6.0, 9.0, 3.0])]
    f, _a, _b, p, _eta = analysis._anova(groups)
    reference = stats.f_oneway(*groups)
    assert f == pytest.approx(reference.statistic)
    assert p == pytest.approx(reference.pvalue)


def test_invariance_models_are_nested(roles):
    spec, _ds, result = roles
    table = result.invariance
    assert list(table["model_en"].str.split().str[0]) == [
        "Configural", "Metric", "Structural"]
    assert table["df"].is_monotonic_increasing
    assert table["chi_square"].is_monotonic_increasing
    cm = compile_model(spec)
    first_loadings = sum(1 for j, f in enumerate(cm.home_factor)) - cm.n_factors
    assert table["delta_df"].iloc[1] == first_loadings
    assert table["delta_df"].iloc[2] == len(spec.paths)
    assert table["invariant"].iloc[1] is not None


def test_equal_populations_look_invariant(roles):
    _spec, _ds, result = roles
    assert result.invariance["delta_cfi"].iloc[1:].abs().max() < 0.02


def test_group_paths_are_per_group(roles):
    spec, _ds, result = roles
    assert len(result.group_paths) == 2 * len(spec.paths)


def test_small_group_is_reported_instead_of_failing():
    spec = plain_model()
    spec.add_demographic("R", [("big", 0.97), ("tiny", 0.03)],
                         role=DemographicRole.MGA)
    ds = generate_dataset(spec, n=300, seed=3)
    result = analysis.analyse(spec, ds)
    assert result.invariance.empty and "too small" in result.notes[0]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def codes_of(spec):
    return {d.code for d in validate(spec).diagnostics}


def test_percentages_must_reach_one_hundred():
    spec = plain_model()
    spec.add_demographic("G", [("a", 0.6), ("b", 0.3)])
    assert "DEMO_PROPORTIONS_SUM" in codes_of(spec)


def test_column_names_must_not_collide():
    spec = plain_model()
    spec.add_demographic(spec.all_item_names()[0], [("a", 0.5), ("b", 0.5)])
    assert "DEMO_NAME_DUPLICATE" in codes_of(spec)


def test_a_dummy_may_not_collide_with_another_column():
    spec = plain_model()
    spec.add_demographic("S_2", [("a", 0.5), ("b", 0.5)])
    spec.add_demographic("S", [("a", 0.5), ("b", 0.3), ("c", 0.2)],
                         role=DemographicRole.CONTROL)
    assert "DEMO_NAME_DUPLICATE" in codes_of(spec)


def test_control_without_a_path_is_flagged():
    spec = plain_model()
    spec.add_demographic("G", [("a", 0.5), ("b", 0.5)],
                         role=DemographicRole.CONTROL)
    assert "CONTROL_NO_PATH" in codes_of(spec)


def test_control_cannot_point_at_an_exogenous_construct():
    spec = plain_model()
    variable = spec.add_demographic("G", [("a", 0.5), ("b", 0.5)],
                                    role=DemographicRole.CONTROL)
    spec.add_control_path(variable, spec.constructs[0].id)
    assert "CONTROL_BAD_TARGET" in codes_of(spec)


def test_a_category_smaller_than_one_case_is_flagged():
    spec = plain_model()
    spec.power.n = 100
    spec.add_demographic("G", [("a", 0.995), ("b", 0.005)])
    assert "DEMO_CATEGORY_BELOW_ONE" in codes_of(spec)


def test_a_valid_model_has_no_demographic_errors():
    report = validate(with_controls())
    assert not [d for d in report.diagnostics
                if d.code.startswith(("DEMO_", "CONTROL_"))]


# --------------------------------------------------------------------------- #
# Saving and exporting
# --------------------------------------------------------------------------- #


def test_project_round_trip(tmp_path):
    spec = with_controls()
    # The non-default value, so a round trip that dropped it would show.
    spec.quota_exact = True
    path = project.save(spec, tmp_path / "model")
    again = project.load(path)
    variable = again.demographics[0]
    assert again.quota_exact is True
    assert variable.role is DemographicRole.CONTROL
    assert variable.categories[0].proportion == pytest.approx(0.6)
    assert variable.control_paths[0].effect_size is EffectSize.CUSTOM
    assert variable.control_paths[0].resolved_coefficient() == pytest.approx(0.2)
    assert again.degrees_of_freedom() == spec.degrees_of_freedom()


def test_csv_has_demographics_first(tmp_path):
    spec = with_controls()
    ds = generate_dataset(spec, n=100, seed=1)
    path = tmp_path / "d.csv"
    exporters.export_csv(ds, path)
    frame = pd.read_csv(path)
    assert list(frame.columns)[0] == "Gender"
    assert frame.shape[1] == 1 + spec.n_items


def test_spss_carries_category_labels(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    spec = with_controls()
    ds = generate_dataset(spec, n=100, seed=1)
    path = tmp_path / "d.sav"
    exporters.export_sav(ds, path, spec, lang="en")
    _frame, meta = pyreadstat.read_sav(str(path))
    assert meta.variable_value_labels["Gender"] == {1.0: "m", 2.0: "f"}


def test_workbook_has_a_codebook(tmp_path):
    from openpyxl import load_workbook
    spec = with_controls()
    ds = generate_dataset(spec, n=100, seed=1)
    path = tmp_path / "d.xlsx"
    exporters.export_xlsx(ds, path, lang="en")
    workbook = load_workbook(path)
    assert "Codebook" in workbook.sheetnames
    assert workbook["Data"]["A1"].value == "Gender"


def test_lavaan_treats_a_control_as_an_observed_predictor():
    spec = with_controls()
    text = exporters.lavaan_syntax(spec)
    assert "Gender =~" not in text
    regression = [ln for ln in text.splitlines() if "PERF ~ " in ln][0]
    assert "*Gender" in regression
    assert "fixed.x = FALSE" in text


def test_lavaan_offers_the_multi_group_call():
    assert 'group = "Region"' in exporters.lavaan_syntax(role_model())


def test_mplus_names_match_the_data_file(tmp_path):
    spec = with_controls()
    spec.add_demographic("Edu", [("a", 0.5), ("b", 0.5)])
    text = exporters.mplus_syntax(spec)
    names_block = text.split("NAMES ARE")[1].split(";")[0].split()
    ds = generate_dataset(spec, n=50, seed=1)
    assert len(names_block) == ds.full_frame.shape[1]
    assert names_block[0] == "Gender" and names_block[1] == "Edu"
    use_block = text.split("USEVARIABLES ARE")[1].split(";")[0].split()
    assert "Gender" in use_block and "Edu" not in use_block
    assert "Gender BY" not in text


def test_report_has_a_demographics_section():
    spec = role_model()
    ds = generate_dataset(spec, n=300, seed=1)
    html = render_html(ReportBundle(
        spec=spec, dataset=ds, demographics=analysis.analyse(spec, ds)), lang="en")
    assert "Demographics and controls" in html and "Frequencies" in html


def test_report_omits_the_section_without_demographics():
    html = render_html(ReportBundle(spec=plain_model()), lang="en")
    assert "Demographics and controls" not in html


# --------------------------------------------------------------------------- #
# The screen
# --------------------------------------------------------------------------- #


@pytest.fixture
def window():
    from PySide6.QtWidgets import QApplication
    from semsim.ui import theme
    from semsim.ui.app import MainWindow
    from semsim.ui.i18n import translator
    app = QApplication.instance() or QApplication(["semsim-test"])
    translator.set_language("ar")
    theme.apply(app, theme.LIGHT, rtl=True)
    win = MainWindow(plain_model())
    win.show()
    app.processEvents()
    yield win
    win.state._dirty = False
    win.close()
    app.processEvents()


def settle():
    from PySide6.QtWidgets import QApplication
    QApplication.instance().processEvents()


def editor(win):
    return win._pages[0]._demographics


def test_add_buttons_add_a_card(window):
    page = editor(window)
    assert page.empty.isVisibleTo(page)
    page.add_button.click()
    settle()
    assert len(window.state.spec.demographics) == 1 and len(page._cards) == 1
    assert not page.empty.isVisibleTo(page)
    page.template_button.click()
    settle()
    variable = window.state.spec.demographics[1]
    assert variable.name == "Gender" and variable.label.ar == "النوع"
    assert [c.proportion for c in variable.categories] == [0.6, 0.4]


def _buttons(card):
    from PySide6.QtWidgets import QPushButton
    return {b.text(): b for b in card.findChildren(QPushButton)}


def test_category_controls_change_the_model(window):
    from semsim.ui.i18n import translator
    page = editor(window)
    page.add_button.click()
    settle()
    variable = window.state.spec.demographics[0]
    _buttons(page._cards[0])[translator.t("dem.add_category")].click()
    settle()
    assert len(variable.categories) == 3
    _buttons(page._cards[0])[translator.t("dem.equal")].click()
    settle()
    assert sum(c.proportion for c in variable.categories) == pytest.approx(1.0)


def test_total_turns_red_until_it_reaches_one_hundred(window):
    from PySide6.QtWidgets import QDoubleSpinBox, QLabel
    page = editor(window)
    page.template_button.click()
    settle()
    card = page._cards[0]
    box = card.findChildren(QDoubleSpinBox)[0]
    box.setValue(50.0)
    settle()
    assert window.state.spec.demographics[0].categories[0].proportion == pytest.approx(0.5)
    labels = [lab for lab in card.findChildren(QLabel)
              if lab.property("role") == "error"]
    assert labels and "90.0" in labels[0].text()


def _combo(card, enum_cls):
    from semsim.ui.widgets.common import EnumCombo
    return [c for c in card.findChildren(EnumCombo) if c._enum is enum_cls][0]


def test_choosing_control_opens_the_path_list(window):
    from semsim.ui.i18n import translator
    page = editor(window)
    page.template_button.click()
    settle()
    combo = _combo(page._cards[0], DemographicRole)
    combo.set_value(DemographicRole.CONTROL)
    combo.value_changed.emit(DemographicRole.CONTROL)
    settle()
    variable = window.state.spec.demographics[0]
    assert variable.role is DemographicRole.CONTROL
    _buttons(page._cards[0])[translator.t("dem.add_control_path")].click()
    settle()
    assert len(variable.control_paths) == 1
    assert variable.control_paths[0].target in window.state.spec.control_target_ids()
    assert window.state.spec.expanded() is not window.state.spec


def test_deleting_a_construct_drops_its_control_paths(window):
    spec = window.state.spec
    variable = spec.add_demographic("G", [("a", 0.5), ("b", 0.5)],
                                    role=DemographicRole.CONTROL)
    target = spec.constructs[-1]
    spec.add_control_path(variable, target.id)
    page = window._pages[0]
    page.tree.setCurrentItem(page._find_node(target.id))
    settle()
    page._remove_selected() if hasattr(page, "_remove_selected") else page._on_remove()
    settle()
    assert variable.control_paths == []


def test_language_switch_rebuilds_the_cards(window):
    from semsim.ui.i18n import translator
    page = editor(window)
    page.template_button.click()
    settle()
    translator.set_language("en")
    settle()
    assert len(page._cards) == 1
    translator.set_language("ar")
    settle()


def test_full_run_fills_the_results_tab(window):
    from semsim.ui.app import SimulationWorker
    spec = window.state.spec
    variable = spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)],
                                    role=DemographicRole.CONTROL)
    spec.add_control_path(variable, spec.constructs[-1].id)
    spec.add_demographic("Edu", [("a", 0.5), ("b", 0.5)],
                         role=DemographicRole.COMPARISON)
    out = SimulationWorker(spec, "generate")._run_pipeline()
    assert out.get("estimation_error") is None, out.get("estimation_error")
    assert out.get("demographics_error") is None, out.get("demographics_error")
    assert "Gender" in set(out["estimation"].path_estimates["source"])
    window.state.set_results(dataset=out["dataset"],
                             estimation=out["estimation"], fit=out["fit"],
                             validity=out["validity"],
                             descriptives=out["descriptives"],
                             demographics=out["demographics"])
    settle()
    results = next(p for p in window._pages if hasattr(p, "_fill_demographics"))
    assert results._panels["dem.frequencies"].isVisibleTo(results._tabs.widget(7))
    assert not results._demographics_empty.isVisibleTo(results._tabs.widget(7))
    assert not out["validity"] is None

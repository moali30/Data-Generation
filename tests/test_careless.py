"""Careless respondents: in the file, screened out of the analysis.

A collected questionnaire file holds some responses given without reading.
By default the simulator adds them on top of the planned sample, scattered
through the file, so the researcher screens them out and analyses the
planned n, on which every target was calibrated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semsim.core import examples
from semsim.core.spec import ModelSpec
from semsim.core.validation import validate
from semsim.generation import careless as careless_module
from semsim.generation.engine import generate_dataset
from semsim.generation.noise import careless_patterns, inject_careless
from semsim.io import ai_template, exporters
from semsim.io.report import ReportBundle, render_html


def _model(n_items: int = 5) -> ModelSpec:
    spec = ModelSpec(seed=7)
    x = spec.add_construct("X", n_items=n_items, n_categories=5)
    y = spec.add_construct("Y", n_items=n_items, n_categories=5)
    spec.add_path(x.id, y.id, coefficient=0.5)
    return spec


def _likert(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 6, size=(n, p)).astype(float)


class _Scale:
    is_ordinal = True
    n_categories = 5


# --------------------------------------------------------------------------- #
# How many
# --------------------------------------------------------------------------- #


def test_the_share_is_of_the_whole_file():
    spec = _model()
    assert careless_module.extra_count(spec, 400) == 21   # 21 / 421 = 5.0%
    spec.noise.careless_proportion = 0.10
    assert careless_module.extra_count(spec, 450) == 50   # 50 / 500


@pytest.mark.parametrize("change", ["off", "zero", "mixed_in"])
def test_nothing_is_added_unless_asked(change):
    spec = _model()
    if change == "off":
        spec.noise.enabled = False
    elif change == "zero":
        spec.noise.careless_proportion = 0.0
    else:
        spec.noise.careless_screened = False
    assert careless_module.extra_count(spec, 400) == 0


# --------------------------------------------------------------------------- #
# The file and the sample
# --------------------------------------------------------------------------- #


def test_the_file_holds_the_sample_plus_the_careless_respondents():
    ds = generate_dataset(_model(), n=400, seed=3)
    block = ds.careless
    assert len(ds.data) == 400 and ds.n == 400
    assert block.n == 21 and ds.n_in_file == 421
    assert len(ds.full_frame) == 421
    assert block.positions == sorted(set(block.positions))
    assert 0 <= block.positions[0] and block.positions[-1] < 421
    assert block.file_rows == [p + 1 for p in block.positions]
    assert sum(block.count(k) for k in ("straightline", "partial", "random")) == 21
    assert ds.noise_report.n_careless == 21
    assert ds.noise_report.careless_rows == block.positions
    assert ds.summary()["rows_in_file"] == 421


def test_careless_rows_sit_at_their_places_and_the_rest_keep_their_order():
    ds = generate_dataset(_model(), n=200, seed=4)
    full = ds.full_frame
    block = ds.careless
    mark = np.zeros(len(full), dtype=bool)
    mark[block.positions] = True
    assert full[mark].reset_index(drop=True).equals(block.frame())
    assert full[~mark].reset_index(drop=True).equals(
        ds.clean_frame.reset_index(drop=True))
    assert (full.dtypes == ds.clean_frame.dtypes).all()


def test_adding_careless_respondents_never_changes_the_sample():
    spec = _model()
    spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)])
    with_them = generate_dataset(spec, n=300, seed=9)
    spec.noise.careless_proportion = 0.0
    without = generate_dataset(spec, n=300, seed=9)
    assert without.careless is None
    assert with_them.data.equals(without.data)
    assert with_them.demographics.equals(without.demographics)


def test_careless_respondents_answer_the_personal_questions_like_others():
    spec = _model()
    spec.add_demographic("Gender", [("m", 0.6), ("f", 0.4)])
    spec.noise.careless_proportion = 0.10
    ds = generate_dataset(spec, n=900, seed=2)
    codes = ds.careless.demographics["Gender"]
    assert set(codes) <= set(ds.demographics["Gender"])
    assert 0.4 < (codes == 1).mean() < 0.8


def test_latent_scores_line_up_with_the_file():
    spec = _model()
    spec.export.include_latent_scores = True
    ds = generate_dataset(spec, n=150, seed=1)
    scores = ds.in_file_order(ds.latent_scores)
    assert len(scores) == ds.n_in_file
    assert scores.iloc[ds.careless.positions].isna().all().all()


def test_mixing_them_into_the_sample_keeps_n_rows():
    spec = _model()
    spec.noise.careless_screened = False
    ds = generate_dataset(spec, n=400, seed=3)
    assert ds.careless is None
    assert len(ds.full_frame) == 400
    assert ds.noise_report.n_careless == 20
    flat = ds.data.to_numpy().std(axis=1) == 0
    assert flat.any()


# --------------------------------------------------------------------------- #
# What they look like
# --------------------------------------------------------------------------- #


def test_each_kind_answers_as_it_should():
    spec = _model(n_items=8)
    spec.noise.careless_proportion = 0.2
    ds = generate_dataset(spec, n=800, seed=5)
    block = ds.careless
    items = block.items.to_numpy(dtype=float)
    assert ((items >= 1) & (items <= 5)).all()
    for row, kind in zip(items, block.kinds):
        if kind == "straightline":
            assert np.unique(row).size == 1
        elif kind == "partial":
            tail = row[-3:]
            assert np.unique(tail).size == 1
    kinds = pd.Series(block.kinds).value_counts(normalize=True)
    assert set(kinds.index) == {"straightline", "partial", "random"}
    assert all(0.15 < share < 0.55 for share in kinds)


def test_straight_lines_are_mostly_middle_or_agreeing():
    reference = np.clip(np.rint(np.random.default_rng(1).normal(3.8, 0.9,
                                                                (500, 10))), 1, 5)
    block = reference[:400].copy()
    rng = np.random.default_rng(2)
    out, kinds = careless_patterns(block, [_Scale()] * 10, rng,
                                   reference=reference)
    flat = np.array([k == "straightline" for k in kinds])
    codes = out[flat, 0]
    assert (codes >= 3).mean() > 0.85
    assert (codes == 1).mean() < 0.05


def test_a_late_straight_liner_keeps_the_start():
    reference = _likert(300, 12, seed=4)
    rng = np.random.default_rng(6)
    out, kinds = careless_patterns(reference[:200], [_Scale()] * 12, rng,
                                   reference=reference)
    late = [i for i, k in enumerate(kinds) if k == "partial"]
    assert late
    for i in late:
        changed = np.flatnonzero(out[i] != reference[i])
        if changed.size:
            assert changed.min() >= 4    # the first 40% stay as answered


def test_the_realistic_style_works_inside_the_sample_too():
    data = _likert(200, 10)
    rng = np.random.default_rng(0)
    out, rows = inject_careless(data, [_Scale()] * 10, 0.1, "realistic", rng)
    assert len(rows) == 20
    untouched = np.setdiff1d(np.arange(200), rows)
    assert np.array_equal(out[untouched], data[untouched])


def test_the_older_styles_still_work_on_top_of_the_sample():
    spec = _model()
    spec.noise.careless_style = "straightline"
    ds = generate_dataset(spec, n=200, seed=8)
    assert set(ds.careless.kinds) == {"straightline"}
    assert (ds.careless.items.to_numpy().std(axis=1) == 0).all()


# --------------------------------------------------------------------------- #
# Telling the researcher
# --------------------------------------------------------------------------- #


def test_exports_carry_the_file_and_say_which_rows_to_remove(tmp_path):
    ds = generate_dataset(_model(), n=120, seed=3)
    csv = tmp_path / "data.csv"
    exporters.export_csv(ds, csv)
    assert len(pd.read_csv(csv)) == ds.n_in_file
    table = exporters.specification_table(ds, lang="en")
    values = dict(zip(table.iloc[:, 0], table.iloc[:, 1]))
    assert values["Careless respondents (remove before analysis)"] == ds.careless.n
    assert values["Their rows in the file"] == ", ".join(
        str(r) for r in ds.careless.file_rows)
    script = tmp_path / "data.R"
    exporters.export_rds_script(ds, script)
    text = script.read_text(encoding="utf-8")
    assert "careless_rows <- c(" in text
    assert str(ds.careless.file_rows[0]) in text


def test_the_report_says_how_many_were_screened_out():
    spec = _model()
    ds = generate_dataset(spec, n=200, seed=3)
    html = render_html(ReportBundle(spec=spec, dataset=ds), lang="en")
    assert "Data screening" in html
    assert f"the file holds {ds.n_in_file} responses" in html


def test_a_high_share_is_questioned():
    spec = _model()
    spec.noise.careless_proportion = 0.25
    codes = {d.code for d in validate(spec).diagnostics}
    assert "CARELESS_SHARE_HIGH" in codes
    spec.noise.careless_proportion = 0.05
    codes = {d.code for d in validate(spec).diagnostics}
    assert "CARELESS_SHARE_HIGH" not in codes


def _template(**realism):
    payload = ai_template.build_template()
    payload["filled"] = True
    payload["realism"].update(realism)
    return payload


def test_the_ai_template_expects_a_few_careless_respondents():
    spec = ai_template.from_payload(_template()).spec
    assert spec.noise.careless_proportion == pytest.approx(0.05)
    assert spec.noise.careless_screened
    payload = _template()
    del payload["realism"]["careless_percent"]
    del payload["realism"]["careless_removed"]
    spec = ai_template.from_payload(payload).spec
    assert spec.noise.careless_proportion == pytest.approx(0.05)


def test_the_ai_template_can_mix_them_in():
    spec = ai_template.from_payload(
        _template(careless_percent=3, careless_removed=False)).spec
    assert spec.noise.careless_proportion == pytest.approx(0.03)
    assert not spec.noise.careless_screened


def test_the_ai_template_rejects_a_non_boolean_choice():
    with pytest.raises(ai_template.AiTemplateError) as caught:
        ai_template.from_payload(_template(careless_removed="yes"))
    assert any("careless_removed" in p.where for p in caught.value.problems)


def test_the_messy_example_adds_its_careless_respondents_on_top():
    spec = examples.messy_reality()
    ds = generate_dataset(spec, n=300)
    assert ds.careless is not None and ds.careless.n > 0
    assert len(ds.data) == 300

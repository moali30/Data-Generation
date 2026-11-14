"""
Respondent answering styles.

In collected data each respondent uses the scale in a way of their own: some
tick the ends, some stay in the middle, some agree with everything.  These
tests pin down that each respondent's style shows consistently across the
items, that it leaves the averages and the model nearly where they were, and
that it can be switched off.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from semsim.analysis import estimator as est
from semsim.core import examples
from semsim.core.spec import ModelSpec, ResponseStyle, ScaleKind
from semsim.generation.engine import generate_dataset
from semsim.generation.respondent_styles import (
    draw_respondent_styles, neutral_point,
)
from semsim.io import ai_template, project


def model(level: str = "realistic") -> ModelSpec:
    spec = examples.simple_mediation()
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    spec.heterogeneity.enabled = False
    for item in spec.all_items():
        item.scale.response_style = ResponseStyle.NEGATIVE_SKEW
    spec.constructs[0].items[-1].reverse_coded = True
    spec.respondent_styles.apply_preset(level)
    return spec


def test_styles_are_on_for_a_new_model():
    assert ModelSpec().respondent_styles.preset_name() == "realistic"
    assert ModelSpec().respondent_styles.active


def test_presets_round_trip():
    settings = ModelSpec().respondent_styles
    for name in ("mild", "realistic", "strong"):
        settings.apply_preset(name)
        assert settings.preset_name() == name
    settings.apply_preset("none")
    assert settings.preset_name() == "none" and not settings.active
    settings.apply_preset("mild")
    settings.acquiescence_sd = 0.3
    assert settings.preset_name() == "custom"


def test_switched_off_draws_nothing():
    assert draw_respondent_styles(model("none"), 100, 1) is None


def test_the_neutral_point_is_the_middle_of_the_scale():
    assert neutral_point(np.array([-1.5, -0.5, 0.5, 1.5])) == 0.0
    assert neutral_point(np.array([-1.0, 0.2, 1.0])) == pytest.approx(0.2)


def test_the_average_respondent_is_unchanged():
    styles = draw_respondent_styles(model(), 200000, 3)
    assert styles.stretch.mean() == pytest.approx(1.0, abs=0.01)
    assert styles.shift.mean() == pytest.approx(0.0, abs=0.01)


def test_only_the_answers_move():
    plain, styled = model("none"), model("realistic")
    plain.export.include_continuous_shadow = True
    styled.export.include_continuous_shadow = True
    before = generate_dataset(plain, n=500, seed=4)
    after = generate_dataset(styled, n=500, seed=4)
    assert before.continuous_shadow.equals(after.continuous_shadow)
    assert not before.data.equals(after.data)
    assert np.abs(before.data.mean() - after.data.mean()).max() < 0.08


def test_extreme_respondents_pick_the_ends_on_every_construct():
    spec = model("strong")
    n, seed = 3000, 7
    data = generate_dataset(spec, n=n, seed=seed).data.to_numpy(dtype=float)
    stretch = draw_respondent_styles(spec, n, seed).stretch
    ends = ((data == 1) | (data == 5)).mean(axis=1)
    middle = (data == 3).mean(axis=1)
    assert np.corrcoef(stretch, ends)[0, 1] > 0.3
    assert np.corrcoef(stretch, middle)[0, 1] < -0.15


def test_acquiescent_respondents_agree_with_reverse_items_too():
    spec = model("strong")
    n, seed = 3000, 8
    data = generate_dataset(spec, n=n, seed=seed).data
    shift = draw_respondent_styles(spec, n, seed).shift
    reverse = spec.constructs[0].items[-1].name
    # Raw agreement with the reverse-worded statement rises with the shift.
    assert np.corrcoef(shift, data[reverse])[0, 1] > 0.05
    plain = generate_dataset(model("none"), n=n, seed=seed).data
    names = [item.name for item in spec.constructs[0].items]

    def reverse_link(frame):
        return np.corrcoef(frame[reverse], frame[names[:-1]].mean(axis=1))[0, 1]

    # Agreeing with both kinds of item weakens their negative correlation.
    assert reverse_link(data) > reverse_link(plain)


def test_the_paths_barely_move():
    planned, styled = [], []
    for seed in range(3):
        for level, sink in (("none", planned), ("realistic", styled)):
            spec = model(level)
            dataset = generate_dataset(spec, n=2000, seed=seed)
            fitted = est.estimate(dataset.model_frame, spec,
                                  estimator=spec.analysis.estimator)
            sink.append(fitted.path_estimates["est"].to_numpy())
    assert np.abs(np.mean(styled, axis=0) - np.mean(planned, axis=0)).max() < 0.05


def test_continuous_items_are_left_alone():
    plain, styled = model("none"), model("strong")
    for spec in (plain, styled):
        for item in spec.all_items():
            item.scale.kind = ScaleKind.CONTINUOUS
    assert (generate_dataset(plain, n=200, seed=2).data
            .equals(generate_dataset(styled, n=200, seed=2).data))


def test_project_file_keeps_the_settings(tmp_path):
    spec = model("strong")
    again = project.load(project.save(spec, tmp_path / "styled"))
    assert again.respondent_styles.preset_name() == "strong"


def filled(**realism):
    payload = ai_template.build_template()
    payload["filled"] = True
    payload["realism"].update(realism)
    return payload


@pytest.mark.parametrize("level", ["none", "mild", "realistic", "strong"])
def test_template_reads_every_level(level):
    spec = ai_template.from_payload(filled(respondent_styles=level)).spec
    assert spec.respondent_styles.preset_name() == level


def test_template_defaults_to_realistic_and_rejects_nonsense():
    payload = filled()
    payload["realism"].pop("respondent_styles")
    assert (ai_template.from_payload(payload).spec.respondent_styles
            .preset_name() == "realistic")
    with pytest.raises(ai_template.AiTemplateError) as info:
        ai_template.from_text(json.dumps(filled(respondent_styles="wild")))
    assert any(p.where == "realism.respondent_styles"
               for p in info.value.problems)

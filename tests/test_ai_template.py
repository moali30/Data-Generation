"""
The JSON template an AI assistant fills in.

The promises checked here: the shipped template is itself a valid model, an
answer wrapped in chat text still reads, each mistake is reported with its
location, and everything in the document reaches the model.
"""

from __future__ import annotations

import copy
import json

import pytest

from semsim.core.spec import (
    ConstructOrder, DemographicRole, EffectSize, FitLevel, MissingMechanism,
)
from semsim.generation.engine import generate_dataset
from semsim.io import ai_template as ai
from semsim.io import project


def filled():
    document = ai.build_template()
    document["filled"] = True
    return document


def problems_of(document):
    with pytest.raises(ai.AiTemplateError) as caught:
        ai.from_payload(document)
    return {p.where: p for p in caught.value.problems}


def test_shipped_template_is_a_model_that_generates():
    result = ai.from_text(ai.template_text())

    assert [c.name for c in result.spec.constructs] == ["SQ", "SAT", "LOY"]
    assert any(w.where == "filled" for w in result.warnings)
    data = generate_dataset(result.spec, seed=3)
    assert data.n == 400
    assert "Gender" in data.full_frame.columns


def test_everything_in_the_document_reaches_the_model():
    spec = ai.from_payload(filled()).spec

    assert spec.power.n == 400 and spec.power.mode == "fixed"
    quality, satisfaction, loyalty = spec.constructs
    assert quality.order is ConstructOrder.SECOND
    assert len(quality.dimensions) == 3 and quality.n_items == 9
    assert satisfaction.items[0].label.en.startswith("Overall")
    assert satisfaction.items[2].reverse_coded
    assert satisfaction.items[2].loading == pytest.approx(0.72)
    assert spec.paths[0].resolved_coefficient() == pytest.approx(0.50)
    assert spec.paths[1].resolved_coefficient() == pytest.approx(0.42)
    assert spec.indirect_effects[0].chain == [c.id for c in spec.constructs]
    gender, sector, education = spec.demographics
    assert gender.normalised_proportions() == pytest.approx([0.6, 0.4])
    assert sector.control_paths[0].term == sector.categories[1].id
    assert sector.control_paths[0].resolved_coefficient() == pytest.approx(0.12)
    assert education.role is DemographicRole.COMPARISON
    assert spec.fit_target.level is FitLevel.GOOD


@pytest.mark.parametrize("wrap", [
    "Here is the file:\n```json\n{}\n```\nTell me if you need changes.",
    "﻿{}",
])
def test_answer_wrapped_in_chat_text_still_reads(wrap):
    text = wrap.replace("{}", json.dumps(filled(), ensure_ascii=False))
    assert ai.from_text(text).spec.power.n == 400


def test_broken_json_reports_its_position():
    with pytest.raises(ai.AiTemplateError) as caught:
        ai.parse_text('{"format": ')
    assert "line 1" in caught.value.problems[0].where


def test_all_mistakes_are_reported_together_with_locations():
    document = filled()
    document["constructs"][1]["loding"] = 0.8
    document["paths"][0]["to"] = "SATT"
    document["paths"][2]["effect"] = "huge"
    document["project"]["sample_size"] = None

    found = problems_of(document)

    assert "'loading'" in found["constructs[1].loding"].message_en
    assert "'SAT'" in found["paths[0].to"].message_en
    assert "paths[2].effect" in found
    assert "project.sample_size" in found
    for problem in found.values():
        assert problem.message_ar


def test_other_json_is_refused():
    assert "format" in problems_of({"a": 1})


def test_percentages_must_sum_to_100():
    document = filled()
    document["demographics"]["variables"][0]["categories"][0]["percent"] = 70
    assert "DEMO_PROPORTIONS_SUM" in problems_of(document)


def test_control_paths_on_a_dummy_variable_must_name_a_category():
    document = filled()
    sector = document["demographics"]["variables"][1]
    path = sector["control_paths"][0]

    del path["category"]
    assert "Public" not in problems_of(copy.deepcopy(document))[
        "demographics.variables[1].control_paths[0].category"].message_en

    path["category"] = "Public"
    assert "reference" in problems_of(copy.deepcopy(document))[
        "demographics.variables[1].control_paths[0].category"].message_en


def test_category_is_refused_on_a_single_effect_variable():
    document = filled()
    document["demographics"]["variables"][0]["control_paths"][0]["category"] = "Male"
    assert ("demographics.variables[0].control_paths[0].category"
            in problems_of(document))


def test_control_paths_need_the_control_role():
    document = filled()
    variable = document["demographics"]["variables"][0]
    variable["role"] = "descriptive"
    assert "demographics.variables[0].control_paths" in problems_of(document)


def test_a_factor_needs_exactly_one_way_of_giving_items():
    document = filled()
    document["constructs"][2]["items"] = ["a", "b", "c"]
    assert "constructs[2]" in problems_of(document)


def test_duplicate_names_are_refused():
    document = filled()
    document["constructs"][2]["name"] = "SAT"
    assert "constructs[2].name" in problems_of(document)


def test_realism_and_fit_values_are_applied():
    document = filled()
    document["fit_target"] = {"rmsea": 0.06, "cfi": 0.93}
    document["realism"].update(missing_percent=5, careless_percent=2,
                               common_method_loading=0.25)
    spec = ai.from_payload(document).spec

    assert spec.fit_target.level is FitLevel.CUSTOM
    assert spec.fit_target.rmsea == pytest.approx(0.06)
    assert spec.fit_target.cfi == pytest.approx(0.93)
    assert spec.noise.missing_proportion == pytest.approx(0.05)
    assert spec.noise.missing_mechanism is MissingMechanism.MCAR
    assert spec.common_method_bias.enabled


def test_coefficient_takes_priority_over_effect():
    document = filled()
    document["paths"][0].update(effect="weak", coefficient=0.33)
    path = ai.from_payload(document).spec.paths[0]
    assert path.effect_size is EffectSize.CUSTOM
    assert path.resolved_coefficient() == pytest.approx(0.33)


def test_imported_model_saves_and_reopens(tmp_path):
    spec = ai.from_payload(filled()).spec
    again = project.load(project.save(spec, tmp_path / "m"))
    assert [v.name for v in again.demographics] == ["Gender", "Sector", "Education"]
    assert again.degrees_of_freedom() == spec.degrees_of_freedom()


def test_problem_list_for_the_assistant_asks_for_the_whole_file():
    text = ai.problems_for_ai([ai.TemplateProblem("paths[0].to", "Bad.", "سيئ.")])
    assert "complete JSON" in text and "paths[0].to: Bad." in text


# -- moderation, written the way the instructions say ----------------------- #


def _moderation_study():
    """A trust-and-intention study with three moderators, as an assistant
    following the instructions would write it."""
    document = filled()

    def construct(name, ar, role):
        return {"name": name, "label_ar": ar, "label_en": name, "role": role,
                "loading": 0.78, "scale": "likert", "scale_points": 5,
                "response_style": "negative_skew", "n_items": 4}

    document["constructs"] = [
        construct("PU", "الفائدة المدركة", "independent"),
        construct("FC", "التسهيلات المتاحة", "independent"),
        construct("AGT", "حوكمة وشفافية الذكاء الاصطناعي", "independent"),
        construct("PIM", "إدارة الهوية الحافظة للخصوصية", "independent"),
        construct("TR", "الثقة المدركة", "mediator"),
        construct("BI", "النية السلوكية للاستخدام", "dependent"),
        construct("DL", "الإلمام الرقمي", "moderator"),
        construct("TC", "تعقيد المهمة", "moderator"),
        construct("VU", "طوعية الاستخدام", "moderator"),
    ]
    document["paths"] = [
        {"from": "PU", "to": "BI", "effect": "medium", "significant": True},
        {"from": "FC", "to": "BI", "effect": "weak", "significant": True},
        {"from": "AGT", "to": "TR", "effect": "medium", "significant": True},
        {"from": "PIM", "to": "TR", "effect": "weak", "significant": True},
        {"from": "TR", "to": "BI", "effect": "medium", "significant": True},
        {"from": "DL", "to": "BI", "effect": "weak", "significant": True},
        {"from": "TC", "to": "BI", "effect": "weak", "significant": True},
        {"from": "VU", "to": "BI", "effect": "weak", "significant": True},
    ]
    document["correlations"] = [{"a": "PU", "b": "FC", "r": 0.40},
                                {"a": "AGT", "b": "PIM", "r": 0.50}]
    document["interactions"] = [
        {"predictor": "PU", "moderator": "DL", "target": "BI", "effect": "weak"},
        {"predictor": "PU", "moderator": "TC", "target": "BI", "effect": "weak"},
        {"predictor": "TR", "moderator": "VU", "target": "BI", "effect": "weak"},
    ]
    document["indirect_effects"] = [
        {"chain": ["AGT", "TR", "BI"], "significant": True},
        {"chain": ["PIM", "TR", "BI"], "significant": True}]
    document["demographics"] = {"exact_quotas": True, "variables": []}
    return document


def test_the_instructions_spell_out_how_a_moderation_is_written():
    text = json.dumps(ai.build_template()["ai_instructions"])
    for key in ('\\"predictor\\"', '\\"moderator\\"', '\\"target\\"',
                '\\"a\\"', '\\"r\\"'):
        assert key in text
    assert "Never write a moderation as a plain path" in text


def test_a_moderation_study_imports_and_draws_its_moderators_on_their_paths():
    from PySide6.QtWidgets import QApplication
    from semsim.ui.widgets.path_diagram import build_layout

    QApplication.instance() or QApplication(["semsim-test"])
    spec = ai.from_payload(_moderation_study()).spec
    assert len(spec.interactions) == 3
    assert len(spec.latent_covariances) == 2

    layout = build_layout(spec, show_items=False, rtl=True)
    ids = {c.name: c.id for c in spec.constructs}
    flow = [layout.nodes[ids[n]] for n in ("PU", "FC", "AGT", "PIM", "TR", "BI")]
    top = min(n.rect.top() for n in flow)
    bottom = max(n.rect.bottom() for n in flow)
    for name in ("DL", "TC", "VU"):
        node = layout.nodes[ids[name]]
        assert node.rect.bottom() < top or node.rect.top() > bottom, name
    landed = [e for e in layout.edges if e.kind == "interaction"]
    assert len(landed) == 3 and all(e.anchor is not None for e in landed)


def test_a_moderation_without_its_path_is_sent_back():
    document = _moderation_study()
    document["paths"] = [p for p in document["paths"]
                         if (p["from"], p["to"]) != ("PU", "BI")]
    found = problems_of(document)
    assert any("PU -> BI" in p.message_en for p in found.values())


def test_a_moderator_that_moderates_nothing_is_sent_back():
    document = _moderation_study()
    document["interactions"] = [i for i in document["interactions"]
                                if i["moderator"] != "VU"]
    found = problems_of(document)
    assert "constructs.VU.role" in found

"""
Group differences on comparison variables.

Demographics drawn apart from the answers make every t-test and ANOVA null by
construction.  These tests pin down that a planned difference shows in the
comparison table at about the size asked, in the direction asked, significant
or not as asked, and that nothing else about the dataset moves: the answers,
the group sizes and every other column stay exactly as drawn.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from semsim.analysis.demographics import compare_groups
from semsim.core import examples
from semsim.core.spec import (
    DemographicKind, DemographicRole, GroupDirection, GroupEffect, ModelSpec,
)
from semsim.core.validation import validate
from semsim.generation.engine import generate_dataset
from semsim.generation.group_differences import between_group_variance
from semsim.io import ai_template, project


def model(shares=(0.6, 0.4), *, significance: bool = True) -> ModelSpec:
    spec = examples.simple_mediation()
    spec.fit_target.enabled = False
    spec.significance_target.enabled = significance
    variable = spec.add_demographic(
        "Group", [(f"G{i + 1}", share) for i, share in enumerate(shares)],
        role=DemographicRole.COMPARISON)
    variable.kind = DemographicKind.ORDINAL
    return spec


def comparison(spec: ModelSpec, seed: int, n: int = 400):
    dataset = generate_dataset(spec, n=n, seed=seed)
    table = compare_groups(spec, dataset.data, dataset.demographics)
    return dataset, table.set_index("construct")


def first(spec: ModelSpec):
    return spec.demographics[0], spec.constructs[0]


def test_between_group_variance_of_a_median_split():
    # Halves of a standard normal have means of -+0.798.
    assert between_group_variance([0.5, 0.5]) == pytest.approx(2 / np.pi, rel=1e-6)
    assert between_group_variance([1.0]) == 0.0


def test_two_groups_reach_the_requested_d_on_average():
    spec = model(significance=False)
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, size=GroupEffect.MEDIUM)
    sizes = [comparison(spec, seed)[1].loc[target.name, "effect_size"]
             for seed in range(12)]
    # d is first group minus second, so an ascending difference is negative.
    assert all(d < 0 for d in sizes)
    assert np.mean(sizes) == pytest.approx(-0.5, abs=0.08)


def test_descending_puts_the_first_group_on_top():
    spec = model(significance=False)
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, size=GroupEffect.MEDIUM,
                              direction=GroupDirection.DESCENDING)
    sizes = [comparison(spec, seed)[1].loc[target.name, "effect_size"]
             for seed in range(6)]
    assert all(d > 0 for d in sizes)


def test_three_groups_reach_the_requested_eta_squared_and_rise_in_order():
    spec = model((0.3, 0.45, 0.25), significance=False)
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, size=GroupEffect.MEDIUM)
    etas, rising = [], 0
    for seed in range(10):
        dataset, table = comparison(spec, seed)
        etas.append(table.loc[target.name, "effect_size"])
        codes = dataset.demographics["Group"].to_numpy()
        score = dataset.data[[i.name for i in target.items]].mean(axis=1).to_numpy()
        means = [score[codes == g].mean() for g in (1, 2, 3)]
        rising += means[0] < means[1] < means[2]
    assert np.mean(etas) == pytest.approx(0.06, abs=0.02)
    assert rising >= 8


def test_a_custom_value_is_used():
    spec = model(significance=False)
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, value=1.0)
    sizes = [comparison(spec, seed, n=600)[1].loc[target.name, "effect_size"]
             for seed in range(6)]
    assert np.mean(sizes) == pytest.approx(-1.0, abs=0.12)


def test_small_differences_are_significant_when_asked():
    spec = model()
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, size=GroupEffect.SMALL)
    for seed in range(8):
        dataset, table = comparison(spec, seed)
        assert table.loc[target.name, "significant"], seed
        assert not any("GROUP_DIFFERENCE_SIGNIFICANCE" in w
                       for w in dataset.warnings)


def test_no_difference_stays_non_significant():
    spec = model()
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, size=GroupEffect.NONE)
    for seed in range(8):
        assert not comparison(spec, seed)[1].loc[target.name, "significant"]


def test_a_construct_can_be_held_level_while_a_correlated_one_differs():
    spec = model()
    variable = spec.demographics[0]
    differs, level = spec.constructs[0], spec.constructs[1]
    spec.add_group_difference(variable, differs.id, size=GroupEffect.MEDIUM)
    spec.add_group_difference(variable, level.id, size=GroupEffect.NONE)
    for seed in range(5):
        table = comparison(spec, seed)[1]
        assert table.loc[differs.name, "significant"]
        assert not table.loc[level.name, "significant"]


def test_answers_and_group_sizes_do_not_move():
    plain = model(significance=False)
    varied = model(significance=False)
    variable, target = first(varied)
    varied.add_group_difference(variable, target.id, size=GroupEffect.LARGE)
    for exact in (False, True):
        plain.quota_exact = varied.quota_exact = exact
        before = generate_dataset(plain, n=300, seed=3)
        after = generate_dataset(varied, n=300, seed=3)
        assert before.data.equals(after.data)
        assert (np.bincount(before.demographics["Group"])
                == np.bincount(after.demographics["Group"])).all()
        assert not before.demographics["Group"].equals(after.demographics["Group"])


def test_other_demographics_are_left_alone():
    plain = model()
    plain.add_demographic("Sector", [("a", 0.5), ("b", 0.5)])
    varied = model()
    varied.add_demographic("Sector", [("a", 0.5), ("b", 0.5)])
    variable, target = first(varied)
    varied.add_group_difference(variable, target.id, size=GroupEffect.MEDIUM)
    before = generate_dataset(plain, n=300, seed=9).demographics
    after = generate_dataset(varied, n=300, seed=9).demographics
    assert before["Sector"].equals(after["Sector"])


def test_differences_are_reproducible_from_the_seed():
    spec = model()
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id)
    one = generate_dataset(spec, n=300, seed=21).demographics
    two = generate_dataset(spec, n=300, seed=21).demographics
    assert one.equals(two)


def test_only_a_comparison_variable_is_regrouped():
    plain = model()
    plain.demographics[0].role = DemographicRole.DESCRIPTIVE
    varied = model()
    variable, target = first(varied)
    varied.add_group_difference(variable, target.id, size=GroupEffect.LARGE)
    variable.role = DemographicRole.DESCRIPTIVE
    assert (generate_dataset(plain, n=300, seed=4).demographics
            .equals(generate_dataset(varied, n=300, seed=4).demographics))


def test_project_file_keeps_the_differences(tmp_path):
    spec = model()
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id, size=GroupEffect.LARGE,
                              direction=GroupDirection.DESCENDING,
                              target_significant=False)
    again = project.load(project.save(spec, tmp_path / "groups"))
    difference = again.demographics[0].group_differences[0]
    assert difference.target == target.id
    assert difference.size is GroupEffect.LARGE
    assert difference.direction is GroupDirection.DESCENDING
    assert difference.target_significant is False


def test_validation_catches_a_lost_target_and_a_repeat():
    spec = model()
    variable, target = first(spec)
    spec.add_group_difference(variable, target.id)
    spec.add_group_difference(variable, target.id)
    codes = {d.code for d in validate(spec)}
    assert "GROUP_DIFFERENCE_DUPLICATE" in codes
    variable.group_differences[0].target = "gone"
    assert "GROUP_DIFFERENCE_BAD_TARGET" in {d.code for d in validate(spec)}


def test_effect_values_follow_the_number_of_groups():
    spec = model()
    variable, target = first(spec)
    difference = spec.add_group_difference(variable, target.id,
                                           size=GroupEffect.LARGE)
    assert difference.resolved_value(2) == pytest.approx(0.8)
    assert difference.resolved_value(3) == pytest.approx(0.14)
    assert difference.eta_squared([0.5, 0.5]) == pytest.approx(0.16 / 1.16)
    difference.size = GroupEffect.NONE
    assert not difference.wants_significance


# --------------------------------------------------------------------------- #
# The AI template
# --------------------------------------------------------------------------- #


def template_with(differences, role="comparison"):
    payload = ai_template.build_template()
    payload["filled"] = True
    education = next(v for v in payload["demographics"]["variables"]
                     if v["name"] == "Education")
    education["role"] = role
    education["differences"] = differences
    return payload


def test_template_example_shows_a_difference():
    payload = ai_template.build_template()
    payload["filled"] = True
    spec = ai_template.from_payload(payload).spec
    education = next(v for v in spec.demographics if v.name == "Education")
    assert education.role is DemographicRole.COMPARISON
    difference = education.group_differences[0]
    assert difference.size is GroupEffect.SMALL
    assert spec.construct_by_name("SAT").id == difference.target


def test_template_reads_size_value_and_direction():
    spec = ai_template.from_payload(template_with([
        {"on": "SAT", "size": "large", "direction": "descending",
         "significant": False},
        {"on": "LOY", "value": 0.35}])).spec
    education = next(v for v in spec.demographics if v.name == "Education")
    large, custom = education.group_differences
    assert large.size is GroupEffect.LARGE
    assert large.direction is GroupDirection.DESCENDING
    assert not large.target_significant
    assert custom.size is GroupEffect.CUSTOM and custom.value == pytest.approx(0.35)


@pytest.mark.parametrize("differences, where", [
    ([{"on": "Nothing"}], "differences[0].on"),
    ([{"on": "SAT", "size": "huge"}], "differences[0].size"),
    ([{"on": "SAT", "direction": "up"}], "differences[0].direction"),
    ([{"on": "SAT"}, {"on": "SAT"}], "differences[1].on"),
])
def test_template_rejects_a_bad_difference(differences, where):
    with pytest.raises(ai_template.AiTemplateError) as info:
        ai_template.from_text(json.dumps(template_with(differences)))
    assert any(p.where.endswith(where) for p in info.value.problems)


def test_template_rejects_differences_on_another_role():
    with pytest.raises(ai_template.AiTemplateError) as info:
        ai_template.from_text(json.dumps(
            template_with([{"on": "SAT"}], role="descriptive")))
    assert any(p.where.endswith("differences") for p in info.value.problems)

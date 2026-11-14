"""
Associations between demographic variables.

Drawn one at a time, demographics are independent, so every chi-square test
between them is null: in such a sample age says nothing about experience.
These tests pin down that a linked pair shows about the Cramér's V asked for,
in the direction asked for, significant or not as asked; that each variable
keeps its shares; that nothing unlinked moves; and that the model knows the
correlation of linked control variables.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from semsim.analysis.demographics import analyse
from semsim.core import examples
from semsim.core.associations import (
    cell_probabilities, chi_square, cramers_v, latent_correlation,
    latent_structure, reachable_v,
)
from semsim.core.spec import (
    AssociationDirection, AssociationStrength, DemographicKind,
    DemographicRole, GroupEffect, ModelSpec,
)
from semsim.core.validation import validate
from semsim.generation.engine import generate_dataset
from semsim.io import ai_template, project

AGE = [("<30", 0.30), ("30-45", 0.45), (">45", 0.25)]
EXPERIENCE = [("<5", 0.35), ("5-10", 0.40), (">10", 0.25)]
JOB = [("Admin", 0.50), ("Tech", 0.35), ("Manager", 0.15)]


def model(*, significance: bool = False, exact: bool = False) -> ModelSpec:
    spec = examples.simple_mediation()
    spec.fit_target.enabled = False
    spec.significance_target.enabled = significance
    spec.quota_exact = exact
    spec.add_demographic("Age", AGE, kind=DemographicKind.ORDINAL)
    spec.add_demographic("Experience", EXPERIENCE, kind=DemographicKind.ORDINAL)
    spec.add_demographic("Job", JOB)
    spec.add_demographic("Gender", [("Male", 0.6), ("Female", 0.4)])
    return spec


def variable(spec: ModelSpec, name: str):
    return next(v for v in spec.demographics if v.name == name)


def crosstab(spec: ModelSpec, dataset, first: str, second: str):
    a, b = variable(spec, first), variable(spec, second)
    return chi_square(dataset.demographics[first].to_numpy(),
                      dataset.demographics[second].to_numpy(),
                      len(a.categories), len(b.categories))


# --------------------------------------------------------------------------- #
# The copula
# --------------------------------------------------------------------------- #


def test_a_median_split_follows_the_tetrachoric_formula():
    # For two median splits phi = (2 / pi) * arcsin(r).
    for r in (0.2, 0.5, 0.8):
        cells = cell_probabilities([0.5, 0.5], [0.5, 0.5], r)
        assert cramers_v(cells) == pytest.approx(2 / np.pi * np.arcsin(r), abs=1e-4)


def test_the_latent_correlation_gives_back_the_v_asked_for():
    for v in (0.1, 0.3, 0.5):
        r = latent_correlation([s for _c, s in AGE], [s for _c, s in JOB], v)
        assert cramers_v(cell_probabilities([s for _c, s in AGE],
                                             [s for _c, s in JOB], r)) \
            == pytest.approx(v, abs=1e-4)
    assert latent_correlation([0.5, 0.5], [0.5, 0.5], 0.0) == 0.0
    assert reachable_v([0.9, 0.1], [0.5, 0.5]) < 0.5


def test_a_chain_of_links_implies_the_missing_one():
    spec = model()
    spec.add_demographic_association("Age", "Experience", value=0.5)
    spec.add_demographic_association("Experience", "Job", value=0.3)
    structure = latent_structure(spec)
    matrix = structure.matrix
    age, exp, job = (structure.index(variable(spec, n).id)
                     for n in ("Age", "Experience", "Job"))
    # The maximum-determinant completion of a chain is the product.
    assert matrix[age, job] == pytest.approx(matrix[age, exp] * matrix[exp, job],
                                             abs=1e-6)
    assert not structure.notes


def test_links_that_cannot_hold_together_are_softened():
    spec = model()
    spec.add_demographic_association("Age", "Experience", value=0.8)
    spec.add_demographic_association("Experience", "Job", value=0.8)
    spec.add_demographic_association("Age", "Job", value=0.8,
                                     direction=AssociationDirection.NEGATIVE)
    structure = latent_structure(spec)
    assert structure.notes
    assert np.linalg.eigvalsh(structure.matrix).min() > 0
    assert "ASSOCIATIONS_INCONSISTENT" in {d.code for d in validate(spec)}


# --------------------------------------------------------------------------- #
# Generated data
# --------------------------------------------------------------------------- #


def test_linked_pairs_reach_the_requested_v_on_average():
    spec = model()
    spec.add_demographic_association("Age", "Experience",
                                     strength=AssociationStrength.STRONG)
    spec.add_demographic_association("Job", "Gender",
                                     strength=AssociationStrength.MODERATE)
    strong, moderate = [], []
    for seed in range(10):
        dataset = generate_dataset(spec, n=500, seed=seed)
        strong.append(crosstab(spec, dataset, "Age", "Experience")[3])
        moderate.append(crosstab(spec, dataset, "Job", "Gender")[3])
    assert np.mean(strong) == pytest.approx(0.50, abs=0.04)
    assert np.mean(moderate) == pytest.approx(0.30, abs=0.04)


def test_the_direction_says_which_categories_go_together():
    spec = model()
    spec.add_demographic_association("Age", "Experience",
                                     strength=AssociationStrength.STRONG)
    spec.add_demographic_association("Gender", "Job",
                                     strength=AssociationStrength.MODERATE,
                                     direction=AssociationDirection.NEGATIVE)
    d = generate_dataset(spec, n=4000, seed=1).demographics
    assert np.corrcoef(d["Age"], d["Experience"])[0, 1] > 0.4
    # Negative: the last gender category goes with the first job category.
    assert (d["Job"][d["Gender"] == 2] == 1).mean() \
        > (d["Job"][d["Gender"] == 1] == 1).mean() + 0.1


@pytest.mark.parametrize("exact", [False, True])
def test_each_variable_keeps_its_shares(exact):
    spec = model(exact=exact)
    spec.add_demographic_association("Age", "Experience",
                                     strength=AssociationStrength.STRONG)
    n = 400
    d = generate_dataset(spec, n=n, seed=6).demographics
    for name, categories in (("Age", AGE), ("Experience", EXPERIENCE)):
        counts = np.bincount(d[name], minlength=4)[1:]
        shares = np.array([s for _c, s in categories])
        if exact:
            assert counts.tolist() == np.round(shares * n).astype(int).tolist()
        else:
            assert np.abs(counts / n - shares).max() < 0.07


def test_unlinked_columns_and_the_answers_do_not_move():
    plain = model()
    linked = model()
    linked.add_demographic_association("Age", "Experience",
                                       strength=AssociationStrength.STRONG)
    before = generate_dataset(plain, n=300, seed=2)
    after = generate_dataset(linked, n=300, seed=2)
    assert before.data.equals(after.data)
    for name in ("Job", "Gender"):
        assert before.demographics[name].equals(after.demographics[name])
    assert not before.demographics["Age"].equals(after.demographics["Age"])


def test_weak_links_are_significant_when_asked():
    spec = model(significance=True)
    spec.add_demographic_association("Job", "Gender",
                                     strength=AssociationStrength.WEAK)
    for seed in range(8):
        dataset = generate_dataset(spec, n=400, seed=seed)
        assert crosstab(spec, dataset, "Job", "Gender")[2] < 0.05, seed


def test_no_association_stays_non_significant():
    spec = model(significance=True)
    spec.add_demographic_association("Age", "Experience",
                                     strength=AssociationStrength.STRONG)
    spec.add_demographic_association("Experience", "Gender",
                                     strength=AssociationStrength.NONE)
    for seed in range(8):
        dataset = generate_dataset(spec, n=400, seed=seed)
        assert crosstab(spec, dataset, "Experience", "Gender")[2] >= 0.05, seed


def test_draws_are_reproducible_from_the_seed():
    spec = model(significance=True)
    spec.add_demographic_association("Age", "Experience")
    one = generate_dataset(spec, n=300, seed=17).demographics
    two = generate_dataset(spec, n=300, seed=17).demographics
    assert one.equals(two)


def test_a_regrouped_comparison_variable_keeps_its_association():
    spec = model(significance=True)
    gender = variable(spec, "Gender")
    gender.role = DemographicRole.COMPARISON
    spec.add_group_difference(gender, spec.constructs[0].id,
                              size=GroupEffect.SMALL)
    spec.add_demographic_association("Job", "Gender",
                                     strength=AssociationStrength.MODERATE)
    values = []
    for seed in range(6):
        dataset = generate_dataset(spec, n=400, seed=seed)
        test = crosstab(spec, dataset, "Job", "Gender")
        assert test[2] < 0.05
        values.append(test[3])
    assert np.mean(values) == pytest.approx(0.30, abs=0.06)


def test_the_model_knows_how_linked_controls_correlate():
    spec = model()
    gender, job = variable(spec, "Gender"), variable(spec, "Job")
    for control in (gender, job):
        control.role = DemographicRole.CONTROL
        for term in control.terms():
            spec.add_control_path(control, spec.constructs[-1].id,
                                  term=term.key, coefficient=0.10)
    spec.add_demographic_association(gender, job,
                                     strength=AssociationStrength.STRONG)
    expanded = spec.expanded()
    names = {c.id: c.name for c in expanded.constructs if c.covariate}
    d = generate_dataset(spec, n=20000, seed=3).demographics
    checked = 0
    for covariance in expanded.latent_covariances:
        a, b = covariance.construct_a, covariance.construct_b
        if a in names and b in names:
            sample = np.corrcoef(d[names[a]], d[names[b]])[0, 1]
            assert covariance.correlation == pytest.approx(sample, abs=0.03)
            checked += 1
    assert checked == 3


def test_the_analysis_reports_each_link():
    spec = model()
    spec.add_demographic_association("Age", "Experience",
                                     strength=AssociationStrength.STRONG)
    dataset = generate_dataset(spec, n=400, seed=5)
    table = analyse(spec, dataset).associations
    assert table[["variable_a", "variable_b"]].values.tolist() == [["Age", "Experience"]]
    row = table.iloc[0]
    assert row["df"] == 4 and row["significant"] and row["cramers_v"] > 0.35


# --------------------------------------------------------------------------- #
# Validation and files
# --------------------------------------------------------------------------- #


def test_validation_catches_bad_and_repeated_links():
    spec = model()
    spec.add_demographic_association("Age", "Experience")
    spec.add_demographic_association("Experience", "Age")
    codes = {d.code for d in validate(spec)}
    assert "ASSOCIATION_DUPLICATE" in codes
    spec.demographic_associations[0].second = spec.demographic_associations[0].first
    assert "ASSOCIATION_BAD_VARIABLE" in {d.code for d in validate(spec)}


def test_validation_says_when_a_v_is_out_of_reach():
    spec = model()
    spec.add_demographic("Rare", [("yes", 0.05), ("no", 0.95)])
    spec.add_demographic_association("Rare", "Gender", value=0.8)
    assert "ASSOCIATION_OUT_OF_REACH" in {d.code for d in validate(spec)}


def test_project_file_keeps_the_links(tmp_path):
    spec = model()
    spec.add_demographic_association("Gender", "Job",
                                     strength=AssociationStrength.WEAK,
                                     direction=AssociationDirection.NEGATIVE,
                                     target_significant=False)
    again = project.load(project.save(spec, tmp_path / "linked"))
    association = again.demographic_associations[0]
    assert association.first == variable(spec, "Gender").id
    assert association.strength is AssociationStrength.WEAK
    assert association.direction is AssociationDirection.NEGATIVE
    assert association.target_significant is False


def filled(associations):
    payload = ai_template.build_template()
    payload["filled"] = True
    payload["demographics"]["associations"] = associations
    return payload


def test_template_example_links_two_variables():
    spec = ai_template.from_payload(filled(
        ai_template.build_template()["demographics"]["associations"])).spec
    association = spec.demographic_associations[0]
    assert variable(spec, "Education").id == association.first
    assert association.strength is AssociationStrength.WEAK
    assert association.direction is AssociationDirection.NEGATIVE


def test_template_reads_an_exact_value():
    spec = ai_template.from_payload(filled([
        {"between": ["Gender", "Sector"], "value": 0.25}])).spec
    association = spec.demographic_associations[0]
    assert association.strength is AssociationStrength.CUSTOM
    assert association.resolved_value() == pytest.approx(0.25)


@pytest.mark.parametrize("associations", [
    [{"between": ["Gender"]}],
    [{"between": ["Gender", "Nobody"]}],
    [{"between": ["Gender", "Gender"]}],
    [{"between": ["Gender", "Sector"], "strength": "huge"}],
    [{"between": ["Gender", "Sector"]}, {"between": ["Sector", "Gender"]}],
])
def test_template_rejects_a_bad_link(associations):
    with pytest.raises(ai_template.AiTemplateError) as info:
        ai_template.from_text(json.dumps(filled(associations)))
    assert any("associations" in p.where for p in info.value.problems)

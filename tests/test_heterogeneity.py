"""
Real-world variation: the population drawn around a specification.

Data drawn straight from a specification give every item of a construct the
same mean, spread and loading, and a reader spots it as simulated.  These
tests pin down that the variation is there when asked for, absent when not,
reproducible from the seed, and never strong enough to change what the
researcher asked of a path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from semsim.core import examples
from semsim.core.compile import compile_model
from semsim.core.spec import EffectSize, ModelSpec
from semsim.generation.engine import generate_dataset
from semsim.generation.heterogeneity import realise_population
from semsim.io import ai_template, project


def model(enabled: bool = True) -> ModelSpec:
    spec = examples.simple_mediation()
    spec.fit_target.enabled = False
    spec.noise.enabled = False
    spec.heterogeneity.enabled = enabled
    return spec


def item_means(spec: ModelSpec, seed: int = 5, n: int = 2000) -> np.ndarray:
    return generate_dataset(spec, n=n, seed=seed).data.mean().to_numpy()


def test_variation_is_on_for_a_new_model():
    assert ModelSpec().heterogeneity.enabled
    assert ModelSpec().heterogeneity.preset_name() == "realistic"


def test_without_variation_every_item_answers_alike():
    means = item_means(model(enabled=False))
    # Same style, same loading: only sampling error separates the items.
    assert means.max() - means.min() < 0.12


def test_with_variation_items_and_constructs_differ():
    means = item_means(model(enabled=True))
    assert means.max() - means.min() > 0.25
    sds = generate_dataset(model(), n=2000, seed=5).data.std().to_numpy()
    assert sds.max() - sds.min() > 0.05


def test_population_is_reproducible_and_leaves_the_spec_alone():
    spec = model()
    before = [item.loading for item in spec.all_items()]
    first = realise_population(spec, 11)
    second = realise_population(spec, 11)
    other = realise_population(spec, 12)
    assert [i.loading for i in spec.all_items()] == before
    assert all(i.scale.thresholds is None for i in spec.all_items())
    assert ([i.scale.thresholds for i in first.all_items()]
            == [i.scale.thresholds for i in second.all_items()])
    assert ([i.loading for i in first.all_items()]
            != [i.loading for i in other.all_items()])
    # The copy has absorbed its variation, so it is not varied twice.
    assert not first.heterogeneity.enabled
    assert realise_population(first, 11) is first


def test_switched_off_returns_the_same_object():
    spec = model(enabled=False)
    assert realise_population(spec, 3) is spec


def test_loadings_scatter_but_stay_plausible():
    spec = model()
    population = realise_population(spec, 4)
    planned = np.array([i.loading for i in spec.all_items()])
    varied = np.array([i.loading for i in population.all_items()])
    assert np.std(varied) > 0.01
    assert np.all(np.abs(varied - planned) <= 2.5 * 0.05 + 1e-9)
    assert np.all((varied >= 0.30) & (varied <= 0.95))


def test_paths_keep_their_sign_and_most_of_their_size():
    spec = model()
    spec.heterogeneity.path_sd = 0.2   # far more than any preset
    for seed in range(20):
        population = realise_population(spec, seed)
        for planned, varied in zip(spec.paths, population.paths):
            b = planned.resolved_coefficient()
            v = varied.resolved_coefficient()
            assert np.sign(v) == np.sign(b)
            assert 0.75 * abs(b) - 1e-9 <= abs(v) <= abs(b) / 0.75 + 1e-9


def test_null_paths_stay_null():
    spec = model()
    spec.paths[0].target_significant = False
    spec.paths[0].effect_size = EffectSize.TRIVIAL
    population = realise_population(spec, 1)
    assert (population.paths[0].resolved_coefficient()
            == spec.paths[0].resolved_coefficient())


@pytest.mark.parametrize("style", ["symmetric", "negative_skew", "positive_skew",
                                   "extreme_negative", "bimodal", "uniform"])
def test_every_item_moves_whatever_the_style(style):
    # A skewed style leaves an end category nearly empty; moving towards it
    # was once refused outright, so half the items never varied.
    from semsim.core.spec import ResponseStyle
    from semsim.generation import thresholds as th
    spec = model()
    for item in spec.all_items():
        item.scale.response_style = ResponseStyle(style)
    for seed in range(5):
        population = realise_population(spec, seed)
        planned = th.scale_thresholds(spec.all_items()[0].scale)
        moved = [np.asarray(i.scale.thresholds) for i in population.all_items()]
        assert all(m is not None and m.size == planned.size for m in moved)
        assert sum(not np.allclose(m, planned) for m in moved) == len(moved)


def test_explicit_cut_points_are_kept_exactly():
    spec = model()
    item = spec.all_items()[0]
    item.scale.thresholds = [-1.5, -0.5, 0.5, 1.5]
    population = realise_population(spec, 2)
    assert population.all_items()[0].scale.thresholds == [-1.5, -0.5, 0.5, 1.5]


def test_no_category_is_emptied():
    spec = model()
    spec.heterogeneity.apply_preset("strong")
    data = generate_dataset(spec, n=3000, seed=8).data
    for column in data:
        assert data[column].nunique() == 5, column


def test_population_compiles_as_cleanly_as_the_model():
    spec = model()
    spec.heterogeneity.apply_preset("strong")
    baseline = len(compile_model(spec).warnings)
    for seed in range(10):
        assert len(compile_model(realise_population(spec, seed)).warnings) <= baseline


def test_control_correlations_are_not_varied():
    spec = model()
    spec.quota_exact = True
    sector = spec.add_demographic("Sector", [("a", 0.5), ("b", 0.3), ("c", 0.2)])
    from semsim.core.spec import DemographicRole
    sector.role = DemographicRole.CONTROL
    for term in sector.terms():
        spec.add_control_path(sector, spec.constructs[-1].id, term=term.key,
                              coefficient=0.15)
    expanded = spec.expanded()
    population = realise_population(expanded, 3)
    controls = {c.id for c in expanded.constructs if c.covariate}
    for planned, varied in zip(expanded.latent_covariances,
                               population.latent_covariances):
        if planned.construct_a in controls or planned.construct_b in controls:
            assert varied.correlation == planned.correlation
    for planned, varied in zip(expanded.paths, population.paths):
        if planned.source in controls:
            assert varied.resolved_coefficient() == planned.resolved_coefficient()


def test_presets_round_trip():
    settings = ModelSpec().heterogeneity
    for name in ("mild", "realistic", "strong"):
        settings.apply_preset(name)
        assert settings.preset_name() == name
    settings.apply_preset("none")
    assert settings.preset_name() == "none" and not settings.enabled
    settings.apply_preset("mild")
    settings.path_sd = 0.07
    assert settings.preset_name() == "custom"


def test_project_file_keeps_the_settings(tmp_path):
    spec = model()
    spec.heterogeneity.apply_preset("strong")
    path = project.save(spec, tmp_path / "varied")
    again = project.load(path)
    assert again.heterogeneity.preset_name() == "strong"


# --------------------------------------------------------------------------- #
# The AI template
# --------------------------------------------------------------------------- #


def filled_template(**realism):
    payload = ai_template.build_template()
    payload["filled"] = True
    payload["realism"].update(realism)
    return payload


def test_template_defaults_to_realistic_variation_and_random_quotas():
    payload = filled_template()
    payload["realism"].pop("variation")
    payload["demographics"].pop("exact_quotas")
    spec = ai_template.from_payload(payload).spec
    assert spec.heterogeneity.preset_name() == "realistic"
    assert spec.quota_exact is False


def test_template_example_shows_the_realistic_choices():
    payload = ai_template.build_template()
    assert payload["realism"]["variation"] == "realistic"
    assert payload["demographics"]["exact_quotas"] is False
    assert "realism.variation" in payload["allowed_values"]


@pytest.mark.parametrize("level", ["none", "mild", "realistic", "strong"])
def test_template_reads_every_variation_level(level):
    spec = ai_template.from_payload(filled_template(variation=level)).spec
    assert spec.heterogeneity.preset_name() == level


def test_template_rejects_an_unknown_variation_level():
    with pytest.raises(ai_template.AiTemplateError) as info:
        ai_template.from_text(json.dumps(filled_template(variation="huge")))
    assert any(p.where == "realism.variation" for p in info.value.problems)


# --------------------------------------------------------------------------- #
# Reverse-worded items and the ends of the scale
# --------------------------------------------------------------------------- #


def reversed_model(enabled: bool) -> ModelSpec:
    from semsim.core.spec import ResponseStyle
    spec = model(enabled)
    for item in spec.all_items():
        item.scale.response_style = ResponseStyle.NEGATIVE_SKEW
    spec.constructs[0].items[-1].reverse_coded = True
    return spec


@pytest.mark.parametrize("enabled", [False, True])
def test_a_reverse_item_is_mostly_disagreed_with(enabled):
    # A sample that mostly agrees with "I am satisfied" mostly disagrees with
    # "I am unhappy": the raw reverse item sits low, and recoding it lines it
    # up with its construct.
    spec = reversed_model(enabled)
    data = generate_dataset(spec, n=3000, seed=4).data
    construct = spec.constructs[0]
    names = [item.name for item in construct.items]
    reverse, forward = names[-1], names[:-1]
    assert data[reverse].mean() < 2.6
    assert abs((6 - data[reverse]).mean() - data[forward].mean().mean()) < 0.5
    assert np.corrcoef(data[reverse], data[forward].sum(axis=1))[0, 1] < 0


def test_a_reverse_item_loads_more_weakly():
    spec = reversed_model(True)
    for seed in range(5):
        population = realise_population(spec, seed)
        planned = spec.constructs[0].items[-1].loading
        assert population.constructs[0].items[-1].loading < planned


def test_every_category_is_used_on_a_skewed_scale():
    # A Beta-shaped agreement scale gives "strongly disagree" about one answer
    # in six hundred; a real sample has a few on every item.
    data = generate_dataset(reversed_model(True), n=1000, seed=2).data
    counts = data.apply(lambda c: c.value_counts().reindex(range(1, 6),
                                                           fill_value=0))
    assert int(counts.to_numpy().min()) >= 1


def test_tail_share_is_part_of_the_presets():
    settings = ModelSpec().heterogeneity
    assert settings.tail_share == pytest.approx(0.05)
    settings.tail_share = 0.10
    assert settings.preset_name() == "custom"

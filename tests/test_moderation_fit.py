"""A moderation model reaches its fit target.

Moderation forces the score-based generator, which draws the latents through
the structural equations.  It used to ignore the misfit injected into the
generating matrix, so every moderation model fit almost perfectly -- RMSEA
near zero whatever was asked for -- and calibration pushed the injection up
round after round without effect.
"""

from __future__ import annotations

import numpy as np
import pytest

from semsim.core import examples, implied
from semsim.core.compile import compile_model
from semsim.core.spec import GenerationEngine
from semsim.generation import engine
from semsim.targeting.calibration import calibrate
from semsim.targeting.cudeck_browne import inject_model_error


def _moderation(n: int = 500):
    spec = examples.moderation()
    spec.power.n = n
    spec.significance_target.enabled = False
    assert spec.resolved_engine() is GenerationEngine.SCORES
    return spec


def test_score_data_take_on_the_misfit_matrix():
    spec = _moderation()
    cm = compile_model(spec)
    sigma = implied.implied_covariance(cm)
    sigma_0 = inject_model_error(cm, 0.06, df=spec.degrees_of_freedom(cm),
                                 rng=np.random.default_rng(4)).sigma_0
    y, _ = engine._generate_from_scores(spec, cm, 60000,
                                        np.random.default_rng(1), [], None)
    moved = engine._impose_covariance(y, sigma, sigma_0, cm)
    before = np.abs(np.cov(y, rowvar=False) - sigma_0).max()
    after = np.abs(np.cov(moved, rowvar=False) - sigma_0).max()
    assert after < 0.02 < before


def test_the_moderation_survives():
    spec = _moderation()
    cm = compile_model(spec)
    sigma = implied.implied_covariance(cm)
    sigma_0 = inject_model_error(cm, 0.06, df=spec.degrees_of_freedom(cm),
                                 rng=np.random.default_rng(4)).sigma_0
    y, _ = engine._generate_from_scores(spec, cm, 60000,
                                        np.random.default_rng(1), [], None)
    moved = engine._impose_covariance(y, sigma, sigma_0, cm)
    (ix,) = spec.interactions[:1]
    ids = {c.id: c for c in spec.constructs}

    def composite(data, key):
        cols = [cm.item_index[i.id] for i in ids[key].items]
        values = data[:, cols].mean(axis=1)
        return (values - values.mean()) / values.std()

    def product_slope(data):
        x, m = composite(data, ix.predictor), composite(data, ix.moderator)
        t = composite(data, ix.target)
        design = np.column_stack([np.ones_like(x), x, m, x * m])
        return np.linalg.lstsq(design, t, rcond=None)[0][3]

    # The example's moderation is a buffering (negative) one.
    assert abs(product_slope(y)) > 0.05
    assert product_slope(moved) == pytest.approx(product_slope(y), abs=0.02)


def test_a_moderation_model_lands_on_its_rmsea_target():
    result = calibrate(_moderation())
    assert result.target_rmsea > 0.02
    assert abs(result.realised_rmsea - result.target_rmsea) <= 0.012

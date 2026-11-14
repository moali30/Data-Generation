"""A population that leaves a requested path out of reach is replaced.

The calibration loop samples from one simulated population that varies a
little around the plan.  When that variation shrinks a weak path the
researcher wants significant, re-drawing samples cannot help; another
population around the same plan can.  When nothing reaches the requested
pattern, the draw kept is the closest one, not merely the first.
"""

from __future__ import annotations

import pandas as pd
import pytest

from semsim.io import ai_template
from semsim.targeting import calibration
from semsim.targeting.calibration import _score, calibrate, miss_distance

_MISS_CODES = ("UNDERPOWERED_PATH", "SIGNIFICANCE_NOT_REACHED")


def _template(seed: int, n: int = 400):
    payload = ai_template.build_template()
    payload["filled"] = True
    spec = ai_template.from_payload(payload).spec
    spec.seed = seed
    spec.power.n = n
    return spec


def _row(p: float, requested: str = "significant") -> dict:
    realised = "significant" if p < 0.05 else "not significant"
    return {"source": "A", "target": "B", "p": p,
            "requested": requested, "realised": realised}


# --------------------------------------------------------------------------- #
# Scoring a draw
# --------------------------------------------------------------------------- #


def test_a_matching_draw_has_no_distance():
    table = pd.DataFrame([_row(0.01), _row(0.40, "not significant")])
    assert miss_distance(table, 0.05) == (0, 0.0)


def test_fewer_misses_beat_nearer_misses():
    one_far = pd.DataFrame([_row(0.90), _row(0.001)])
    two_near = pd.DataFrame([_row(0.06), _row(0.07)])
    assert miss_distance(one_far, 0.05) < miss_distance(two_near, 0.05)


def test_among_equal_misses_the_nearer_p_value_wins():
    near = miss_distance(pd.DataFrame([_row(0.07)]), 0.05)
    far = miss_distance(pd.DataFrame([_row(0.60)]), 0.05)
    assert near[0] == far[0] == 1
    assert near < far


def test_a_null_that_came_out_significant_counts_too():
    table = pd.DataFrame([_row(0.0001, "not significant")])
    misses, distance = miss_distance(table, 0.05)
    assert misses == 1
    assert distance == pytest.approx(2.69897, abs=1e-4)


def test_the_round_score_puts_significance_before_rmsea():
    near = {"significance_ok": False, "miss": (1, 0.1), "rmsea": 0.09}
    far = {"significance_ok": False, "miss": (1, 1.2), "rmsea": 0.05}
    ok = {"significance_ok": True, "miss": (0, 0.0), "rmsea": 0.09}
    scores = [_score(a, 0.05, True) for a in (ok, near, far)]
    assert scores == sorted(scores)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_a_population_that_misses_is_replaced():
    # At this seed the first population shrinks SQ -> LOY (planned 0.15)
    # to about 0.11, which n = 400 cannot show; the second one can.
    result = calibrate(_template(502))
    assert result.significance_ok
    assert result.populations == 2
    assert result.summary()["populations"] == 2
    assert not any(w.startswith(_MISS_CODES) for w in result.warnings)
    assert set(result.history["population"]) == {1, 2}


def test_a_population_that_meets_the_plan_is_kept():
    result = calibrate(_template(501))
    assert result.significance_ok
    assert result.populations == 1


def test_without_variation_there_is_no_other_population_to_try():
    spec = _template(500, n=80)
    spec.heterogeneity.enabled = False
    result = calibrate(spec)
    assert not result.significance_ok
    assert result.populations == 1


def test_an_underpowered_plan_still_says_so_after_every_population():
    result = calibrate(_template(500, n=80))
    assert not result.significance_ok
    assert result.populations == 1 + calibration.EXTRA_POPULATIONS
    missed = [w for w in result.warnings
              if w.startswith("SIGNIFICANCE_NOT_REACHED")]
    assert len(missed) == 1
    assert f"{result.populations} simulated populations" in missed[0]
    assert any(w.startswith("UNDERPOWERED_PATH") for w in result.warnings)

"""
semsim.generation.respondent_styles
===================================
How each simulated respondent uses the rating scale.

Two respondents with the same opinion do not tick the same box.  One marks
"strongly agree" wherever they agree at all, another never leaves the middle
of the scale, and a third agrees with nearly everything, including the
reverse-worded statements that say the opposite.  Data in which every
respondent uses the scale the same way look simulated: no one ever answers 5
on every item, and nobody's answers sit all at 3.

Each respondent draws two traits from a stream of their own:

* an extreme-versus-midpoint stretch ``s = exp(sd_e * z - sd_e^2 / 2)``,
  which pushes that respondent's latent responses away from each item's
  neutral point (``s > 1``: more 1s and 5s) or towards it (``s < 1``: more
  middle answers).  Its mean is one, so the average respondent is unchanged;
* an acquiescence shift ``a = sd_a * z`` added to every rating item in the
  direction of agreeing with the statement as written.  A reverse-worded
  item's latent response already runs in that direction (its loading is
  negative), so an acquiescent respondent agrees with "I am unhappy" too, and
  the reverse items correlate less strongly, negatively, with the others, as
  they do in real surveys.

The neutral point of an item is the middle of its cut points: the centre of
the middle category, or the middle cut point of an even scale.

Both traits add a little variance that every item shares, the method
variance a Harman single-factor test picks up in collected data.  At the
realistic level it moves a standardised path by about 0.02.

References
----------
Baumgartner, H., & Steenkamp, J.-B. E. M. (2001). Response styles in
    marketing research: A cross-national investigation. *Journal of Marketing
    Research*, 38(2), 143-156.
Weijters, B., Geuens, M., & Schillewaert, N. (2010). The individual
    consistency of acquiescence and extreme response style in self-report
    questionnaires. *Applied Psychological Measurement*, 34(2), 105-121.
Böckenholt, U. (2012). Modeling multiple response processes in judgment and
    choice. *Psychological Methods*, 17(4), 665-678.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from semsim.core.spec import ModelSpec

#: Mixed into the seed so answering styles draw from their own random stream.
_STREAM = 32452843

__all__ = ["RespondentStyles", "draw_respondent_styles", "neutral_point"]


def neutral_point(thresholds: np.ndarray) -> float:
    """The latent value at the middle of a rating scale's cut points."""
    tau = np.asarray(thresholds, dtype=float)
    return float(np.median(tau)) if tau.size else 0.0


@dataclass
class RespondentStyles:
    """Each respondent's stretch and acquiescence shift.

    Attributes
    ----------
    stretch : numpy.ndarray
        Length-``n`` multipliers around each item's neutral point.
    shift : numpy.ndarray
        Length-``n`` shifts towards agreeing, on the latent z-scale.
    """

    stretch: np.ndarray
    shift: np.ndarray

    def apply(self, column: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """One item's latent responses as these respondents would give them."""
        middle = neutral_point(thresholds)
        return middle + self.stretch * (column - middle) + self.shift


def draw_respondent_styles(spec: ModelSpec, n: int, seed: int
                           ) -> Optional[RespondentStyles]:
    """Draw every respondent's answering style, or ``None`` when switched off.

    Parameters
    ----------
    spec : ModelSpec
        Supplies :attr:`ModelSpec.respondent_styles`.
    n : int
        Sample size.
    seed : int
        The run's seed.

    Returns
    -------
    RespondentStyles or None
    """
    settings = spec.respondent_styles
    if not settings.active:
        return None
    rng = np.random.default_rng([abs(int(seed)), _STREAM])
    extreme = max(float(settings.extreme_sd), 0.0)
    agree = max(float(settings.acquiescence_sd), 0.0)
    z = rng.standard_normal((2, int(n)))
    return RespondentStyles(
        stretch=np.exp(extreme * z[0] - 0.5 * extreme * extreme),
        shift=agree * z[1])

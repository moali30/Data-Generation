"""
semsim.generation.careless
==========================
Careless respondents that come on top of the planned sample.

A questionnaire collected online always brings back some responses given
without reading: a row of identical answers, a row that goes flat half-way
down, answers that have nothing to do with the questions.  A researcher
screens them out before analysing (Meade & Craig, 2012; Curran, 2016) and
reports how many went.  So the file a study starts from holds more rows than
the sample it analyses.

When ``spec.noise.careless_screened`` is on, the engine generates the planned
sample as usual -- every target is calibrated on it -- and this module adds
``k`` careless respondents to it, so that ``k / (n + k)`` is the requested
``careless_proportion``.  They are scattered through the file at random
positions, as they arrive in a real collection.  Their answers start as those
of respondents drawn from the attentive sample's own correlation structure,
so a late straight-liner's first answers look like everyone else's, and are
then made careless by :func:`semsim.generation.noise.careless_patterns`.
Their demographic answers are copied from attentive respondents' rows, so the
file's category shares and links stay as planned; people answer the short
personal questions even when they do not read the long ones.

Everything here draws from a random stream of its own, so adding careless
respondents never changes the attentive sample.

References
----------
Curran, P. G. (2016). Methods for the detection of carelessly invalid
    responses in survey data. *Journal of Experimental Social Psychology*,
    66, 4-19.
Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey
    data. *Psychological Methods*, 17(3), 437-455.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from semsim.core.spec import ModelSpec

__all__ = ["CarelessBlock", "extra_count", "draw_careless", "insert_rows"]

#: Random stream of its own, so the attentive sample never changes.
_STREAM = 49979687

#: Largest share of a file that can be careless; beyond it the "sample" is
#: mostly noise.  Validation keeps requests well below this.
_MAX_SHARE = 0.5


@dataclass
class CarelessBlock:
    """The careless respondents added to a file, and where they sit in it."""

    #: Their questionnaire answers, one row each, columns as ``dataset.data``.
    items: pd.DataFrame
    #: Their demographic answers, columns as ``dataset.demographics``.
    demographics: Optional[pd.DataFrame]
    #: Zero-based row of each one in the file, ascending.
    positions: List[int] = field(default_factory=list)
    #: ``"straightline"``, ``"partial"`` or ``"random"`` for each one.
    kinds: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.positions)

    @property
    def file_rows(self) -> List[int]:
        """One-based row numbers, as a spreadsheet or SPSS shows them."""
        return [int(p) + 1 for p in self.positions]

    def count(self, kind: str) -> int:
        return sum(1 for k in self.kinds if k == kind)

    def frame(self) -> pd.DataFrame:
        """Demographics then items, as the file carries them."""
        if self.demographics is None or self.demographics.empty:
            return self.items
        return pd.concat([self.demographics.reset_index(drop=True),
                          self.items.reset_index(drop=True)], axis=1)


def extra_count(spec: ModelSpec, n: int) -> int:
    """How many careless respondents to add to a sample of ``n``.

    ``careless_proportion`` is their share of the whole file, so ``k`` solves
    ``k / (n + k) = share``.  Zero unless noise is on, the share is positive
    and careless respondents are added on top rather than mixed in.
    """
    noise = getattr(spec, "noise", None)
    if noise is None or not bool(getattr(noise, "enabled", False)):
        return 0
    if not bool(getattr(noise, "careless_screened", False)):
        return 0
    try:
        share = float(getattr(noise, "careless_proportion", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(share) or share <= 0.0:
        return 0
    share = min(share, _MAX_SHARE)
    return int(np.rint(n * share / (1.0 - share)))


def draw_careless(spec: ModelSpec, y_star: np.ndarray, data: pd.DataFrame,
                  demographics: Optional[pd.DataFrame],
                  scales: Sequence[Any], score: Callable[[np.ndarray], np.ndarray],
                  k: int, seed: int) -> Optional[CarelessBlock]:
    """Draw ``k`` careless respondents to add to the attentive sample.

    Parameters
    ----------
    spec
        The model; ``spec.noise.careless_style`` picks the kinds.
    y_star
        The attentive sample's latent responses to the items, ``(n, p)``.
    data
        Its answers, ``(n, p)``, used as the reference for how people answer.
    demographics
        Its demographic frame, or None.
    scales
        One scale per item column.
    score
        Turns latent responses into answers the way the engine does.
    k
        How many to draw; see :func:`extra_count`.
    seed
        The run's seed.
    """
    if k <= 0 or data.shape[1] == 0:
        return None
    from semsim.generation import noise as noise_module

    rng = np.random.default_rng([abs(int(seed)), _STREAM])
    n = data.shape[0]
    latent = np.asarray(y_star, dtype=float)
    centre = latent.mean(axis=0)
    spread = np.atleast_2d(np.cov(latent, rowvar=False))
    first_answers = score(rng.multivariate_normal(centre, spread, size=k,
                                                  method="eigh"))
    reference = data.to_numpy(dtype=float)

    style = str(getattr(spec.noise, "careless_style", "realistic") or "realistic")
    style = style.strip().lower()
    if style == "realistic":
        block, kinds = noise_module.careless_patterns(
            first_answers, list(scales), rng, reference=reference)
    else:
        block, _rows = noise_module.inject_careless(
            first_answers, list(scales), 1.0, style, rng)
        block = np.asarray(block, dtype=float)
        kinds = ["straightline" if np.nanstd(row) == 0 else "random"
                 for row in block]

    items = pd.DataFrame(block, columns=data.columns)
    for column in data.columns:
        if pd.api.types.is_integer_dtype(data[column].dtype):
            items[column] = np.rint(items[column]).astype(data[column].dtype)

    people = None
    if demographics is not None and not demographics.empty:
        donors = rng.integers(0, n, size=k)
        people = demographics.iloc[donors].reset_index(drop=True)

    positions = sorted(int(p) for p in rng.choice(n + k, size=k, replace=False))
    return CarelessBlock(items=items, demographics=people,
                         positions=positions, kinds=list(kinds))


def insert_rows(frame: pd.DataFrame, extra: Optional[pd.DataFrame],
                positions: Sequence[int]) -> pd.DataFrame:
    """``frame`` with the rows of ``extra`` placed at ``positions``.

    ``extra`` of None inserts empty rows, for frames the careless respondents
    have no values in (latent scores, the continuous shadow).
    """
    positions = list(positions)
    if not positions:
        return frame
    k = len(positions)
    if extra is None:
        extra = pd.DataFrame(np.nan, index=range(k), columns=frame.columns)
    total = len(frame) + k
    mark = np.zeros(total, dtype=bool)
    mark[positions] = True
    order = np.empty(total, dtype=int)
    order[~mark] = np.arange(len(frame))
    order[mark] = len(frame) + np.arange(k)
    combined = pd.concat([frame.reset_index(drop=True),
                          extra[frame.columns].reset_index(drop=True)],
                         ignore_index=True)
    return combined.iloc[order].reset_index(drop=True)

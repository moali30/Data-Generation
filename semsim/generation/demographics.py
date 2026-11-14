"""
semsim.generation.demographics
==============================
Assign demographic categories to simulated respondents.

Two sampling schemes are offered, because they answer different questions.
*Quota sampling* fills every category to its exact share, so a 60/40 design at
``n = 300`` yields 180 and 120 cases, which is what a researcher running a
quota design reports.  *Weighted random sampling* draws each respondent
independently with the shares as weights, so the realised split varies from
sample to sample the way it does under probability sampling.

Assignments for different variables are independent unless the researcher
links them.  Linked variables are cut from correlated latent scores (see
:mod:`semsim.core.associations`), so age goes with experience the way it does
in a real sample, and when significance targeting is on the scores are
redrawn until each chi-square test is significant or not as asked.  Every
draw comes from a stream of its own, so adding a demographic variable never
changes the questionnaire answers of a model that gains no control path, and
linking two variables never changes the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from semsim.core.associations import (
    LatentStructure, chi_square, latent_structure, valid_associations,
)
from semsim.core.errors import SpecificationError
from semsim.core.spec import DemographicRole, DemographicVariable, ModelSpec

#: Mixed into the seed so demographics draw from their own random stream.
_STREAM = 104729

#: Most redraws spent on making the associations' significance match.
MAX_ATTEMPTS = 200

__all__ = ["allocate_quota", "draw_categories", "draw_demographics",
           "demographic_stream", "DemographicDraw", "draw_demographics_jointly",
           "MAX_ATTEMPTS"]


@dataclass
class DemographicDraw:
    """Drawn demographic columns and the latent scores behind linked ones.

    Attributes
    ----------
    frame : pandas.DataFrame
        The columns :func:`draw_demographics` returns.
    structure : LatentStructure
        The latent correlations of the associated variables.
    scores : numpy.ndarray or None
        ``n x m`` latent scores of those variables, in ``structure.ids``
        order; ``None`` when nothing is associated.
    notes : list of str
        Bilingual remarks for the run's warnings.
    """

    frame: pd.DataFrame
    structure: LatentStructure = field(default_factory=LatentStructure)
    scores: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


def demographic_stream(seed: int) -> np.random.Generator:
    """The random stream demographics use for a given run seed.

    Parameters
    ----------
    seed : int
        The run's seed.

    Returns
    -------
    numpy.random.Generator
        A generator independent of the one that draws the item responses.

    References
    ----------
    O'Neill, M. E. (2014). PCG: A family of simple fast space-efficient
        statistically good algorithms for random number generation. Harvey
        Mudd College Technical Report HMC-CS-2014-0905.
    """
    return np.random.default_rng([abs(int(seed)), _STREAM])


def allocate_quota(shares: List[float], n: int) -> np.ndarray:
    """Whole-number category counts closest to the requested shares.

    Uses the largest-remainder method: every category gets the integer part of
    its quota, and the cases left over go to the largest fractional parts.

    Parameters
    ----------
    shares : list of float
        Category shares, summing to one.
    n : int
        Sample size.

    Returns
    -------
    numpy.ndarray
        Integer counts summing to ``n``.

    References
    ----------
    Balinski, M. L., & Young, H. P. (2001). *Fair Representation: Meeting the
        Ideal of One Man, One Vote* (2nd ed.). Brookings Institution Press.
    """
    quotas = np.asarray(shares, dtype=float) * int(n)
    # Rounding first keeps 0.07 * 100 == 7.000000000000001 from counting as a
    # fractional part that competes for a leftover case.
    quotas = np.round(quotas, 9)
    counts = np.floor(quotas).astype(int)
    leftover = int(n) - int(counts.sum())
    if leftover > 0:
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:leftover]] += 1
    return counts


def draw_categories(variable: DemographicVariable, n: int,
                    rng: np.random.Generator, *, exact: bool) -> np.ndarray:
    """Category codes ``1..k`` for ``n`` respondents.

    Parameters
    ----------
    variable : DemographicVariable
        The variable to assign.
    n : int
        Sample size.
    rng : numpy.random.Generator
        Source of randomness.
    exact : bool
        Fill exact quotas when ``True``; draw with weights when ``False``.

    Returns
    -------
    numpy.ndarray
        Length-``n`` integer codes in random order.

    References
    ----------
    Moser, C. A. (1952). Quota sampling. *Journal of the Royal Statistical
        Society, Series A*, 115(3), 411-423.
    """
    shares = variable.normalised_proportions()
    k = len(shares)
    if exact:
        counts = allocate_quota(shares, n)
        codes = np.repeat(np.arange(1, k + 1), counts)
        return rng.permutation(codes)
    return rng.choice(np.arange(1, k + 1), size=int(n), p=shares)


def _codes_from_scores(score: np.ndarray, variable: DemographicVariable,
                       exact: bool) -> np.ndarray:
    """Categories as slices of a latent score, in list order."""
    shares = np.asarray(variable.normalised_proportions())
    if exact:
        counts = allocate_quota(list(shares), score.size)
        codes = np.empty(score.size, dtype=int)
        codes[np.argsort(score, kind="stable")] = np.repeat(
            np.arange(1, shares.size + 1), counts)
        return codes
    from scipy import stats as _stats
    cuts = _stats.norm.ppf(np.clip(np.cumsum(shares)[:-1], 1e-12, 1 - 1e-12))
    return np.searchsorted(cuts, score, side="right").astype(int) + 1


def _draw_linked(spec: ModelSpec, n: int, seed: int
                 ) -> "tuple[Dict[str, np.ndarray], LatentStructure, Optional[np.ndarray], List[str]]":
    """Codes of every associated variable, cut from correlated latent scores."""
    structure = latent_structure(spec)
    if not structure.ids:
        return {}, structure, None, []
    variables = {v.id: v for v in spec.demographics}
    associations = valid_associations(spec)
    rng = np.random.default_rng([abs(int(seed)), _STREAM, 1])
    factor = np.linalg.cholesky(structure.matrix)
    attempts = MAX_ATTEMPTS if spec.significance_target.enabled else 1
    alpha = float(spec.significance_target.alpha)
    best = None
    for _ in range(attempts):
        scores = rng.standard_normal((int(n), len(structure.ids))) @ factor.T
        codes = {vid: _codes_from_scores(scores[:, j], variables[vid],
                                         spec.quota_exact)
                 for j, vid in enumerate(structure.ids)}
        hits = 0
        for association in associations:
            first, second = variables[association.first], variables[association.second]
            test = chi_square(codes[first.id], codes[second.id],
                              len(first.categories), len(second.categories))
            if test is None or (test[2] < alpha) == association.wants_significance:
                hits += 1
        if best is None or hits > best[0]:
            best = (hits, codes, scores)
        if hits == len(associations):
            break
    notes = list(structure.notes)
    if attempts > 1 and best[0] < len(associations):
        notes.append(
            "ASSOCIATION_SIGNIFICANCE: not every association between the "
            "demographic variables reached the significance asked of it in "
            "this sample; a larger sample or a stronger association would. | "
            "لم تصل كل الارتباطات بين المتغيرات الديموغرافية إلى الدلالة "
            "المطلوبة في هذه العينة؛ عينة أكبر أو ارتباط أقوى يحقق ذلك.")
    return best[1], structure, best[2], notes


def draw_demographics_jointly(spec: ModelSpec, n: int, seed: int
                              ) -> DemographicDraw:
    """Draw every demographic variable, linked ones together.

    Parameters
    ----------
    spec : ModelSpec
        The model; only its demographic section is read.
    n : int
        Sample size.
    seed : int
        The run's seed.

    Returns
    -------
    DemographicDraw
        The columns, and the latent scores the group-difference step needs
        to keep an association when it regroups a comparison variable.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If a variable has no categories.

    References
    ----------
    Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied
        Multiple Regression/Correlation Analysis* (3rd ed.), chapter 8.
        Erlbaum.
    Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
    """
    rng = demographic_stream(seed)
    drawn: Dict[str, np.ndarray] = {}
    for variable in spec.demographics:
        if not variable.categories:
            raise SpecificationError(
                f"The demographic variable {variable.name!r} has no categories.",
                f"المتغير الديموغرافي «{variable.name}» ليس له فئات.",
                variable=variable.name)
        # Every variable consumes its draw, linked or not, so linking two
        # variables leaves the others exactly as they were.
        drawn[variable.id] = draw_categories(variable, n, rng,
                                             exact=spec.quota_exact)
    linked, structure, scores, notes = _draw_linked(spec, n, seed)
    drawn.update(linked)

    columns = {}
    for variable in spec.demographics:
        codes = drawn[variable.id]
        columns[variable.name] = codes
        if variable.role is DemographicRole.CONTROL and variable.is_dummy_coded:
            for term in variable.terms():
                columns[term.column] = (codes == term.category + 1).astype(int)
    frame = pd.DataFrame(columns, index=range(int(n)))
    return DemographicDraw(frame=frame, structure=structure, scores=scores,
                           notes=notes)


def draw_demographics(spec: ModelSpec, n: int, seed: int) -> pd.DataFrame:
    """Draw every demographic variable for one simulated sample.

    Parameters
    ----------
    spec : ModelSpec
        The model; only its demographic section is read.
    n : int
        Sample size.
    seed : int
        The run's seed.

    Returns
    -------
    pandas.DataFrame
        One integer column per variable holding the codes ``1..k``, and, for
        a dummy-coded control, one 0/1 column per dummy term.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If a variable has no categories.

    References
    ----------
    Cohen, J., Cohen, P., West, S. G., & Aiken, L. S. (2003). *Applied
        Multiple Regression/Correlation Analysis* (3rd ed.), chapter 8.
        Erlbaum.
    """
    return draw_demographics_jointly(spec, n, seed).frame

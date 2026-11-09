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

Assignments for different variables are independent, and they come from a
stream of their own, so adding a demographic variable never changes the
questionnaire answers of a model that gains no control path.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from semsim.core.errors import SpecificationError
from semsim.core.spec import DemographicRole, DemographicVariable, ModelSpec

#: Mixed into the seed so demographics draw from their own random stream.
_STREAM = 104729

__all__ = ["allocate_quota", "draw_categories", "draw_demographics",
           "demographic_stream"]


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
    rng = demographic_stream(seed)
    columns = {}
    for variable in spec.demographics:
        if not variable.categories:
            raise SpecificationError(
                f"The demographic variable {variable.name!r} has no categories.",
                f"المتغير الديموغرافي «{variable.name}» ليس له فئات.",
                variable=variable.name)
        codes = draw_categories(variable, n, rng, exact=spec.quota_exact)
        columns[variable.name] = codes
        if variable.role is DemographicRole.CONTROL and variable.is_dummy_coded:
            for term in variable.terms():
                columns[term.column] = (codes == term.category + 1).astype(int)
    return pd.DataFrame(columns, index=range(int(n)))

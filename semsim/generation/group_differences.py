"""
semsim.generation.group_differences
===================================
Make the groups of a comparison variable differ the way the researcher planned.

Demographic categories are drawn independently of the answers, so every t-test
and ANOVA on a comparison variable comes out null by construction, and a
reader who runs them sees a sample in which no group differs from any other on
anything.  Real samples are not like that.

The answers are left exactly as they were drawn.  What changes is who belongs
to which group: each respondent gets a propensity that correlates with the
target constructs' composite scores, and the group sizes already drawn are
filled in order of that propensity.  The item data, the covariance matrix and
every SEM result are therefore untouched, and only the comparison column
moves.

With one target, ``u = rho * z + sqrt(1 - rho^2) * e``, and cutting ``u`` into
groups of shares ``p`` gives the construct a between-group share of variance
``eta^2 = rho^2 * V``, where ``V`` is the variance of the group means of a
standard normal cut at those shares.  So ``rho = sqrt(eta^2 / V)``.  Several
targets share the propensity: their correlations with ``u`` are set through
the inverse of their correlation matrix, so a construct asked to show no
difference really shows none even when it correlates with one that does.

The noise ``e`` comes from a stream of its own.  When significance targeting is
on, it is redrawn until each difference is significant or not as asked, the
way a sample is redrawn for a path, and the closest draw is kept otherwise.

A comparison variable linked to other demographic variables keeps the link:
the part of its latent score the linked variables predict joins the
propensity, so regrouping by the answers does not undo the association, and
the redraws also check the chi-square test of each link.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _stats

from semsim.core.associations import chi_square, valid_associations
from semsim.core.spec import (
    DemographicRole, DemographicVariable, GroupDifference, GroupDirection,
    ModelSpec,
)

if TYPE_CHECKING:  # pragma: no cover
    from semsim.generation.demographics import DemographicDraw

#: Mixed into the seed so group assignment draws from its own random stream.
_STREAM = 15485863

#: Most redraws of the propensity noise spent on matching significance.
MAX_ATTEMPTS = 200

#: Largest share of the propensity's variance the constructs may explain.
#: Above it a request is out of reach and is scaled back.
_MAX_EXPLAINED = 0.95

__all__ = ["apply_group_differences", "between_group_variance", "MAX_ATTEMPTS"]


def between_group_variance(shares: Sequence[float]) -> float:
    """Variance of the group means of a standard normal cut at ``shares``.

    Parameters
    ----------
    shares : sequence of float
        Group shares in order along the variable, summing to one.

    Returns
    -------
    float
        ``sum p_g m_g^2``, where ``m_g`` is the mean of a standard normal
        truncated to group ``g``'s slice.

    References
    ----------
    Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). *Continuous
        Univariate Distributions* (2nd ed., Vol. 1), section 13.10. Wiley.
    """
    p = np.asarray(shares, dtype=float)
    p = p[p > 0]
    if p.size < 2:
        return 0.0
    p = p / p.sum()
    cuts = _stats.norm.ppf(np.clip(np.cumsum(p)[:-1], 1e-12, 1 - 1e-12))
    density = np.concatenate([[0.0], _stats.norm.pdf(cuts), [0.0]])
    means = (density[:-1] - density[1:]) / p
    return float(np.sum(p * means * means))


def _stream(seed: int, position: int) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)), _STREAM, int(position)])


def _standardised(score: np.ndarray) -> np.ndarray:
    """z-scores with missing composites set to the mean."""
    finite = np.isfinite(score)
    if finite.sum() < 2:
        return np.zeros_like(score)
    mean = float(score[finite].mean())
    sd = float(score[finite].std(ddof=1))
    out = np.zeros_like(score)
    if sd > 1e-12:
        out[finite] = (score[finite] - mean) / sd
    return out


def _assign(propensity: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Codes ``1..k`` given in order of the propensity, ``counts[g]`` each."""
    order = np.argsort(propensity, kind="stable")
    codes = np.empty(propensity.size, dtype=int)
    codes[order] = np.repeat(np.arange(1, counts.size + 1), counts)
    return codes


def _p_value(score: np.ndarray, codes: np.ndarray, k: int) -> Optional[float]:
    """The comparison table's test of ``score`` across the groups."""
    from semsim.analysis.demographics import _anova, _welch
    groups = []
    for code in range(1, k + 1):
        values = score[(codes == code) & np.isfinite(score)]
        if values.size >= 2 and float(values.var(ddof=1)) > 1e-12:
            groups.append(values)
    if len(groups) < 2:
        return None
    if len(groups) == 2:
        return _welch(groups[0], groups[1])[2]
    return _anova(groups)[3]


def _targets(variable: DemographicVariable, spec: ModelSpec
             ) -> List[GroupDifference]:
    """The variable's differences on constructs that exist, one per construct."""
    known = {c.id for c in spec.constructs if not c.covariate}
    out: Dict[str, GroupDifference] = {}
    for difference in variable.group_differences:
        if difference.target in known and difference.target not in out:
            out[difference.target] = difference
    return list(out.values())


def _correlations(differences: Sequence[GroupDifference], shares: np.ndarray
                  ) -> np.ndarray:
    """Each target's correlation with the propensity, before feasibility."""
    spread = between_group_variance(shares)
    out = np.zeros(len(differences))
    if spread <= 0.0:
        return out
    for j, difference in enumerate(differences):
        rho = np.sqrt(min(difference.eta_squared(list(shares)) / spread, 1.0))
        sign = 1.0 if difference.direction is GroupDirection.ASCENDING else -1.0
        out[j] = sign * rho
    return out


def _linked_part(variable: DemographicVariable,
                 drawn: Optional["DemographicDraw"]
                 ) -> Tuple[Optional[np.ndarray], float]:
    """What the linked variables' latent scores say about this one's.

    Returns the conditional mean ``E[l | linked]`` for every respondent and
    its variance, or ``(None, 0)`` when the variable is linked to nothing.
    """
    if drawn is None or drawn.scores is None:
        return None, 0.0
    own = drawn.structure.index(variable.id)
    if own is None:
        return None, 0.0
    others = [j for j in range(len(drawn.structure.ids)) if j != own]
    if not others:
        return None, 0.0
    matrix = drawn.structure.matrix
    weights = np.linalg.solve(matrix[np.ix_(others, others)], matrix[others, own])
    explained = float(matrix[own, others] @ weights)
    return drawn.scores[:, others] @ weights, explained


def _reassign(spec: ModelSpec, variable: DemographicVariable,
              codes: np.ndarray, data: pd.DataFrame, rng: np.random.Generator,
              current: Optional[pd.DataFrame] = None,
              drawn: Optional["DemographicDraw"] = None
              ) -> Tuple[np.ndarray, List[str]]:
    """New codes for one variable, keeping its group sizes."""
    from semsim.analysis.demographics import _composite

    notes: List[str] = []
    k = len(variable.categories)
    n = codes.size
    counts = np.bincount(codes, minlength=k + 1)[1:k + 1]
    if counts.sum() != n or np.count_nonzero(counts) < 2:
        return codes, notes

    differences, scores = [], []
    constructs = {c.id: c for c in spec.constructs}
    for difference in _targets(variable, spec):
        score = _composite(constructs[difference.target], data)
        if score is not None and np.isfinite(score).sum() >= 2:
            differences.append(difference)
            scores.append(score)
    if not differences:
        return codes, notes

    z = np.column_stack([_standardised(s) for s in scores])
    rho = _correlations(differences, counts / n)
    corr = np.atleast_2d(np.corrcoef(z, rowvar=False)) if z.shape[1] > 1 else np.ones((1, 1))
    corr = corr + np.eye(corr.shape[0]) * 1e-6
    weights = np.linalg.solve(corr, rho)
    explained = float(rho @ weights)
    if explained > _MAX_EXPLAINED:
        scale = np.sqrt(_MAX_EXPLAINED / explained)
        rho, weights, explained = rho * scale, weights * scale, _MAX_EXPLAINED
        notes.append(
            f"GROUP_DIFFERENCE_REDUCED: the differences asked of "
            f"'{variable.name}' cannot all hold together given how its target "
            "constructs correlate, so they were scaled back. | لا يمكن أن "
            f"تتحقق كل الفروق المطلوبة في «{variable.name}» معًا بسبب "
            "الارتباط بين المتغيرات المستهدفة، فخُفِّضت.")
    signal = z @ weights
    linked, linked_share = _linked_part(variable, drawn)
    if linked is not None:
        total = explained + linked_share
        if total > _MAX_EXPLAINED:
            shrink = np.sqrt(_MAX_EXPLAINED / total)
            signal, linked = signal * shrink, linked * shrink
            explained, linked_share = explained * shrink ** 2, linked_share * shrink ** 2
        signal = signal + linked
        explained += linked_share
    residual = np.sqrt(max(1.0 - explained, 0.0))

    links = []
    if current is not None:
        others = {v.id: v for v in spec.demographics}
        for association in valid_associations(spec):
            if variable.id not in (association.first, association.second):
                continue
            other = others[association.second if association.first == variable.id
                           else association.first]
            if other.name in current.columns:
                links.append((association, other,
                              current[other.name].to_numpy(dtype=int)))

    attempts = MAX_ATTEMPTS if spec.significance_target.enabled else 1
    alpha = float(spec.significance_target.alpha)
    wanted = len(differences) + len(links)
    best, best_hits = codes, -1
    for _ in range(attempts):
        candidate = _assign(signal + residual * rng.standard_normal(n), counts)
        hits = 0
        for difference, score in zip(differences, scores):
            p = _p_value(score, candidate, k)
            if p is None or (p < alpha) == difference.wants_significance:
                hits += 1
        for association, other, other_codes in links:
            test = chi_square(candidate, other_codes, k, len(other.categories))
            if test is None or (test[2] < alpha) == association.wants_significance:
                hits += 1
        if hits > best_hits:
            best, best_hits = candidate, hits
        if hits == wanted:
            break
    if attempts > 1 and best_hits < wanted:
        notes.append(
            f"GROUP_DIFFERENCE_SIGNIFICANCE: not every difference or "
            f"association of '{variable.name}' reached the significance asked "
            "of it in this sample; a larger sample or a larger effect would. "
            f"| لم تصل كل فروق «{variable.name}» أو ارتباطاته إلى الدلالة "
            "المطلوبة في هذه العينة؛ عينة أكبر أو أثر أكبر يحقق ذلك.")
    return best, notes


def apply_group_differences(spec: ModelSpec, data: pd.DataFrame,
                            demographics: Optional[pd.DataFrame], seed: int,
                            drawn: Optional["DemographicDraw"] = None
                            ) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Regroup the respondents of every comparison variable with differences.

    Parameters
    ----------
    spec : ModelSpec
        Supplies the comparison variables, their differences and the
        constructs whose composites they act on.
    data : pandas.DataFrame
        The finished item responses, recoding of reverse items and missing
        values included, as the comparison table will read them.
    demographics : pandas.DataFrame or None
        The drawn demographic columns.  Their group sizes are kept.
    seed : int
        The run's seed.
    drawn : DemographicDraw, optional
        The latent scores behind associated variables, so a regrouped
        comparison variable keeps its associations.

    Returns
    -------
    tuple
        The demographics with the comparison columns regrouped (a copy when
        anything changed), and bilingual notes for the run's warnings.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
        (2nd ed.). Erlbaum.
    Olsson, U., Drasgow, F., & Dorans, N. J. (1982). The polyserial
        correlation coefficient. *Psychometrika*, 47(3), 337-347.
    """
    if demographics is None or demographics.empty:
        return demographics, []
    notes: List[str] = []
    out = demographics
    for position, variable in enumerate(spec.demographics):
        if (variable.role is not DemographicRole.COMPARISON
                or not variable.group_differences
                or variable.name not in demographics.columns):
            continue
        codes = demographics[variable.name].to_numpy(dtype=int)
        regrouped, extra = _reassign(spec, variable, codes, data,
                                     _stream(seed, position), out, drawn)
        notes.extend(extra)
        if regrouped is not codes:
            if out is demographics:
                out = demographics.copy()
            out[variable.name] = regrouped
    return out, notes

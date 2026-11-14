"""
semsim.core.associations
========================
How two demographic variables go together.

In a real sample older respondents have more years of experience and
postgraduates hold more senior posts, so a chi-square test between such
variables is significant.  Drawn one at a time, demographics are independent
and every such test is null.

Each associated variable is given a standard normal latent score, and its
categories are slices of that score at its shares, in list order.  The latent
scores of linked variables correlate, which is a Gaussian copula: each
variable keeps its shares exactly, and the cross-table of two variables is the
bivariate normal mass of each pair of slices.  The strength a researcher asks
for is Cramér's V, the effect size a chi-square test reports, and the latent
correlation that gives it is found by bisection.

Pairs nobody linked are filled in by the maximum-determinant completion of the
latent correlation matrix, the least committal one: a variable linked to two
others carries an implied link between them (age to experience to job grade
makes age go with job grade), as it would in a real population.

References
----------
Cramér, H. (1946). *Mathematical Methods of Statistics*, section 21.9.
    Princeton University Press.
Nelsen, R. B. (2006). *An Introduction to Copulas* (2nd ed.). Springer.
Grone, R., Johnson, C. R., Sá, E. M., & Wolkowicz, H. (1984). Positive
    definite completions of partial Hermitian matrices. *Linear Algebra and
    its Applications*, 58, 109-124.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _stats

if TYPE_CHECKING:  # pragma: no cover
    from semsim.core.spec import (
        DemographicAssociation, DemographicTerm, DemographicVariable, ModelSpec,
    )

#: Largest latent correlation used; beyond it the cross-table barely changes.
LATENT_MAX = 0.99

#: Bounds standing in for infinity in the bivariate normal distribution.
_EDGE = 9.0

__all__ = ["cell_probabilities", "chi_square", "cramers_v", "latent_correlation",
           "reachable_v", "LatentStructure", "latent_structure",
           "term_correlation", "valid_associations", "LATENT_MAX"]


def _cuts(shares: Sequence[float]) -> np.ndarray:
    """Slice edges of a standard normal for these shares, ends included."""
    p = np.asarray(shares, dtype=float)
    p = p / p.sum()
    inner = _stats.norm.ppf(np.clip(np.cumsum(p)[:-1], 1e-12, 1 - 1e-12))
    return np.concatenate([[-_EDGE], np.clip(inner, -_EDGE, _EDGE), [_EDGE]])


def cell_probabilities(shares_a: Sequence[float], shares_b: Sequence[float],
                       r: float) -> np.ndarray:
    """The population cross-table of two variables linked by latent ``r``.

    Parameters
    ----------
    shares_a, shares_b : sequence of float
        Category shares of the row and the column variable.
    r : float
        Correlation of their latent scores.

    Returns
    -------
    numpy.ndarray
        ``k x m`` cell probabilities summing to one.
    """
    a, b = _cuts(shares_a), _cuts(shares_b)
    if abs(r) < 1e-12:
        pa = np.diff(_stats.norm.cdf(a))
        pb = np.diff(_stats.norm.cdf(b))
        return np.outer(pa, pb)
    grid = np.array([[x, y] for x in a for y in b])
    dist = _stats.multivariate_normal(mean=[0.0, 0.0],
                                      cov=[[1.0, r], [r, 1.0]])
    cdf = np.asarray(dist.cdf(grid)).reshape(a.size, b.size)
    cells = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
    cells = np.clip(cells, 0.0, None)
    return cells / cells.sum()


def cramers_v(table: np.ndarray) -> float:
    """Cramér's V of a cross-table of counts or probabilities.

    Rows and columns that are entirely empty are dropped first, as a
    statistics package does.
    """
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    p = t / t.sum()
    expected = np.outer(p.sum(axis=1), p.sum(axis=0))
    phi2 = float(np.sum((p - expected) ** 2 / expected))
    return float(np.sqrt(phi2 / (min(p.shape) - 1)))


def reachable_v(shares_a: Sequence[float], shares_b: Sequence[float]) -> float:
    """The strongest V these shares allow, at the largest latent correlation."""
    return cramers_v(cell_probabilities(shares_a, shares_b, LATENT_MAX))


def latent_correlation(shares_a: Sequence[float], shares_b: Sequence[float],
                       v: float) -> float:
    """The non-negative latent correlation whose cross-table has Cramér's V ``v``.

    Returns :data:`LATENT_MAX` when ``v`` is beyond reach for these shares.
    """
    if v <= 0.0:
        return 0.0
    if v >= reachable_v(shares_a, shares_b):
        return LATENT_MAX
    low, high = 0.0, LATENT_MAX
    for _ in range(40):
        middle = 0.5 * (low + high)
        if cramers_v(cell_probabilities(shares_a, shares_b, middle)) < v:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def valid_associations(spec: "ModelSpec") -> List["DemographicAssociation"]:
    """Associations between two distinct variables that exist, once per pair."""
    usable = {v.id for v in spec.demographics if len(v.categories) >= 2}
    seen, out = set(), []
    for association in spec.demographic_associations:
        pair = frozenset((association.first, association.second))
        if (len(pair) == 2 and pair <= usable and pair not in seen):
            seen.add(pair)
            out.append(association)
    return out


@dataclass
class LatentStructure:
    """The latent correlation matrix of every associated variable.

    Attributes
    ----------
    ids : list of str
        Variable ids, in the spec's order, naming the rows and columns.
    matrix : numpy.ndarray
        Their latent correlations, positive definite.
    notes : list of str
        Bilingual remarks, such as links that could not all hold together.
    """

    ids: List[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    notes: List[str] = field(default_factory=list)

    def index(self, vid: str) -> Optional[int]:
        return self.ids.index(vid) if vid in self.ids else None


def _complete(matrix: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Maximum-determinant completion of the entries not ``fixed``.

    The determinant is quadratic in any one off-diagonal entry, so each free
    entry is set to its maximiser in turn until the matrix settles.
    """
    out = matrix.copy()
    free = [(i, j) for i in range(out.shape[0]) for j in range(i + 1, out.shape[0])
            if not fixed[i, j]]
    if not free:
        return out

    def det_at(i: int, j: int, x: float) -> float:
        out[i, j] = out[j, i] = x
        return float(np.linalg.det(out))

    for _ in range(50):
        moved = 0.0
        for i, j in free:
            old = out[i, j]
            low, mid, high = det_at(i, j, -1.0), det_at(i, j, 0.0), det_at(i, j, 1.0)
            curvature = 0.5 * (high + low) - mid
            slope = 0.5 * (high - low)
            best = (-slope / (2.0 * curvature) if curvature < -1e-15 else old)
            best = float(np.clip(best, -LATENT_MAX, LATENT_MAX))
            out[i, j] = out[j, i] = best
            moved = max(moved, abs(best - old))
        if moved < 1e-10:
            break
    return out


def latent_structure(spec: "ModelSpec") -> LatentStructure:
    """The latent correlations that give every association its Cramér's V.

    Parameters
    ----------
    spec : ModelSpec
        Supplies the demographic variables and their associations.

    Returns
    -------
    LatentStructure
        Empty when the model has no usable association.
    """
    from semsim.core.spec import AssociationDirection

    associations = valid_associations(spec)
    if not associations:
        return LatentStructure()
    linked = {vid for a in associations for vid in (a.first, a.second)}
    variables = [v for v in spec.demographics if v.id in linked]
    ids = [v.id for v in variables]
    shares = {v.id: v.normalised_proportions() for v in variables}
    m = len(ids)
    matrix = np.eye(m)
    fixed = np.eye(m, dtype=bool)
    for association in associations:
        i, j = ids.index(association.first), ids.index(association.second)
        r = latent_correlation(shares[association.first],
                               shares[association.second],
                               association.resolved_value())
        if association.direction is AssociationDirection.NEGATIVE:
            r = -r
        matrix[i, j] = matrix[j, i] = r
        fixed[i, j] = fixed[j, i] = True
    matrix = _complete(matrix, fixed)

    notes: List[str] = []
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < 1e-6:
        values = np.clip(values, 1e-3, None)
        matrix = vectors @ np.diag(values) @ vectors.T
        scale = np.sqrt(np.diag(matrix))
        matrix = matrix / np.outer(scale, scale)
        notes.append(
            "ASSOCIATIONS_INCONSISTENT: the associations between the "
            "demographic variables cannot all hold together, so they were "
            "softened to the nearest set that can. | لا يمكن أن تتحقق كل "
            "الارتباطات بين المتغيرات الديموغرافية معًا، فخُفِّفت إلى أقرب "
            "مجموعة ممكنة.")
    return LatentStructure(ids=ids, matrix=matrix, notes=notes)


def term_correlation(variable_a: "DemographicVariable", term_a: "DemographicTerm",
                     variable_b: "DemographicVariable", term_b: "DemographicTerm",
                     r: float) -> float:
    """Population correlation of two regression columns of linked variables.

    A term is either the variable's codes ``1..k`` or a 0/1 dummy flagging
    one category, so its moments follow from the shares and the two
    variables' cross-table.
    """
    pa = np.asarray(variable_a.normalised_proportions())
    pb = np.asarray(variable_b.normalised_proportions())
    cells = cell_probabilities(pa, pb, r)

    def values(variable: "DemographicVariable", term: "DemographicTerm"
               ) -> np.ndarray:
        k = len(variable.categories)
        if term.category is None:
            return np.arange(1, k + 1, dtype=float)
        out = np.zeros(k)
        out[term.category] = 1.0
        return out

    x, y = values(variable_a, term_a), values(variable_b, term_b)
    mean_x, mean_y = float(pa @ x), float(pb @ y)
    var_x = float(pa @ (x - mean_x) ** 2)
    var_y = float(pb @ (y - mean_y) ** 2)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    covariance = float((x - mean_x) @ cells @ (y - mean_y))
    return float(np.clip(covariance / np.sqrt(var_x * var_y), -0.999, 0.999))


def chi_square(codes_a: np.ndarray, codes_b: np.ndarray, k: int, m: int
               ) -> Optional[Tuple[float, int, float, float]]:
    """Pearson's chi-square test of independence of two coded columns.

    Parameters
    ----------
    codes_a, codes_b : numpy.ndarray
        Codes ``1..k`` and ``1..m``; anything else (missing) is skipped.
    k, m : int
        Number of categories of each variable.

    Returns
    -------
    tuple or None
        ``(chi_square, df, p, cramers_v)``, or ``None`` when fewer than two
        categories of either variable occur.

    References
    ----------
    Pearson, K. (1900). On the criterion that a given system of deviations
        from the probable... *Philosophical Magazine*, 50(302), 157-175.
    """
    a = np.asarray(codes_a)
    b = np.asarray(codes_b)
    keep = (a >= 1) & (a <= k) & (b >= 1) & (b <= m)
    table = np.zeros((k, m))
    np.add.at(table, (a[keep].astype(int) - 1, b[keep].astype(int) - 1), 1.0)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    statistic, p, df, _expected = _stats.chi2_contingency(table, correction=False)
    return float(statistic), int(df), float(p), cramers_v(table)

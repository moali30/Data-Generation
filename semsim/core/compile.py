"""
semsim.core.compile
===================
Flattens the hierarchical :class:`~semsim.core.spec.ModelSpec` into a single
latent vector plus the four LISREL-style matrices, in the **completely
standardised** metric: every latent variable and every observed item ends up
with unit total variance.

That metric is the whole point.  It is what makes "set this path to 0.30" mean
*standardised* 0.30 to the researcher, what makes ``Sigma(theta)`` a
correlation matrix the Vale-Maurelli generator can consume directly, and what
makes population reliability a closed-form read-off rather than a simulation.

The algebra
-----------
Latent model::

    eta   = B eta + zeta          Phi = (I - B)^-1 Psi (I - B)^-T

Measurement model::

    y     = Lambda eta + eps      Theta = Cov(eps)

Implied covariance::

    Sigma = Lambda Phi Lambda' + Theta

``B[i, j]`` is the effect **of** latent ``j`` **on** latent ``i``.  Latent
nodes are ordered topologically, so a recursive model yields a strictly
lower-triangular ``B`` and the disturbance variances can be derived in a single
forward pass.  A non-recursive (cyclic) model, or one with correlated
disturbances between endogenous latents, falls back on a damped fixed-point
iteration that rescales ``diag(Psi)`` until ``diag(Phi)`` is one.

References
----------
Joreskog, K. G. (1970). A general method for analysis of covariance
    structures. *Biometrika*, 57(2), 239-251.
Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*.
    Scientific Software International.
Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from semsim.core.errors import ConvergenceError, SpecificationError
from semsim.core.implied import is_positive_definite, nearest_positive_definite
from semsim.core.spec import ModelSpec

#: Smallest residual / disturbance variance the compiler will tolerate.  A
#: specification that implies less than this is a Heywood case and gets clamped
#: (with a warning) rather than producing a non-positive-definite matrix.
HEYWOOD_FLOOR = 1e-3

#: Convergence tolerance on ``diag(Phi) == 1`` for the cyclic fallback.
PHI_DIAG_TOL = 1e-10

#: Maximum passes of the fixed-point iteration before giving up.
MAX_FIXED_POINT_PASSES = 500

#: Smallest Newton slope the fixed-point solver will divide a step by. Guards
#: against a step that explodes (or reverses) when a strong correlated
#: disturbance makes ``d Phi[i, i] / d Psi[i, i]`` approach zero.
_MIN_NEWTON_SLOPE = 0.1

__all__ = [
    "HEYWOOD_FLOOR", "PHI_DIAG_TOL", "MAX_FIXED_POINT_PASSES",
    "CompiledModel", "compile_model",
]


# --------------------------------------------------------------------------- #
# The compiled artefact
# --------------------------------------------------------------------------- #


@dataclass
class CompiledModel:
    """Numerical form of a :class:`~semsim.core.spec.ModelSpec`.

    Attributes
    ----------
    spec : ModelSpec
        The specification this was compiled from (kept by reference).
    item_ids, item_names : list of str
        Length ``p``, in dataset column order.
    factor_ids, factor_names : list of str
        Length ``m``: constructs, dimensions and the optional method factor,
        in topological order.
    Lambda : numpy.ndarray
        ``(p, m)`` completely standardised loading matrix.
    B : numpy.ndarray
        ``(m, m)`` structural coefficients; ``B[i, j]`` is the effect of ``j``
        on ``i``.  Strictly lower triangular for a recursive model.
    Psi : numpy.ndarray
        ``(m, m)`` latent disturbance covariance.
    Theta : numpy.ndarray
        ``(p, p)`` item residual covariance.
    Phi : numpy.ndarray
        ``(m, m)`` implied latent covariance, unit diagonal.
    endogenous : numpy.ndarray
        ``(m,)`` boolean mask of latents that receive at least one path.
    item_index, factor_index : dict of str to int
        Reverse lookups into the two orderings.
    method_factor_index : int or None
        Column of the common-method factor in ``Lambda`` / ``B``.
    home_factor : list of int
        Length ``p``; the index of each item's primary factor.
    warnings : list of str
        Bilingual notes about Heywood clamping, repaired matrices and
        specifications the compiler had to reinterpret.
    """

    spec: ModelSpec
    item_ids: List[str]
    item_names: List[str]
    factor_ids: List[str]
    factor_names: List[str]
    Lambda: np.ndarray
    B: np.ndarray
    Psi: np.ndarray
    Theta: np.ndarray
    Phi: np.ndarray
    endogenous: np.ndarray
    item_index: Dict[str, int]
    factor_index: Dict[str, int]
    method_factor_index: Optional[int]
    home_factor: List[int]
    #: ``(item_id, factor_id)`` pairs drawn by the cross-loading policy, as
    #: distinct from the ones the user declared. Counting free parameters from
    #: the non-zero entries of ``Lambda`` would miss a loading that happens to
    #: be exactly zero, so the pattern is recorded rather than inferred.
    auto_cross_loadings: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Rows of ``Lambda`` / columns of the data that are observed control
    #: terms.  They always follow the questionnaire items.
    covariate_rows: List[int] = field(default_factory=list)
    #: Latent indices of those control terms.
    covariate_factors: List[int] = field(default_factory=list)

    # -- convenience ------------------------------------------------------- #

    @property
    def n_items(self) -> int:
        """Number of observed items, ``p``."""
        return len(self.item_ids)

    @property
    def n_factors(self) -> int:
        """Number of latent nodes, ``m``."""
        return len(self.factor_ids)

    @property
    def n_substantive_items(self) -> int:
        """Number of questionnaire items, control terms excluded."""
        return len(self.item_ids) - len(self.covariate_rows)

    @property
    def is_recursive(self) -> bool:
        """Whether ``B`` is strictly lower triangular (no feedback loops)."""
        return bool(np.allclose(np.triu(self.B), 0.0))

    def items_of(self, factor_id: str) -> List[int]:
        """Row indices of the items whose home factor is ``factor_id``."""
        fi = self.factor_index[factor_id]
        return [j for j, h in enumerate(self.home_factor) if h == fi]


# --------------------------------------------------------------------------- #
# Small internal helpers
# --------------------------------------------------------------------------- #


def _bilingual(en: str, ar: str) -> str:
    """One warning string carrying both interface languages."""
    return f"{en} | {ar}"


def _topological_order(n: int, edges: Iterable[Tuple[int, int]]
                       ) -> Tuple[List[int], bool]:
    """Kahn's algorithm with a deterministic tie-break.

    Parameters
    ----------
    n : int
        Number of nodes.
    edges : iterable of (int, int)
        ``(source, target)`` pairs, i.e. ``source`` affects ``target``.

    Returns
    -------
    (list of int, bool)
        A node order in which every source precedes its targets *and* every
        structurally exogenous node precedes every endogenous one, plus a flag
        that is ``True`` when the graph contains a cycle (in which case the
        remaining nodes are appended in their original order).

    References
    ----------
    Kahn, A. B. (1962). Topological sorting of large networks.
        *Communications of the ACM*, 5(11), 558-562.
    """
    unique = sorted({(int(s), int(t)) for s, t in edges})
    indegree = [0] * n
    successors: List[List[int]] = [[] for _ in range(n)]
    for s, t in unique:
        successors[s].append(t)
        indegree[t] += 1

    # Every structurally exogenous latent must be emitted before any
    # endogenous one. A plain Kahn heap breaks ties by index, so for
    # X -> Y with X, Y, Z at indices 0, 1, 2 it pops X, then Y (which has
    # just become ready) BEFORE Z. Step 4 then takes Z's exogenous branch and
    # writes Phi[Z, j] = Psi[Z, j] for every already-processed j -- which is
    # zero for Y -- silently discarding the correlation the structural model
    # induces between Y and Z. With a declared covariance X ~ Z = 0.5 and
    # X -> Y = 0.6, Phi[Y, Z] came out 0.0 where the algebra gives 0.3, and
    # the resulting model could be inadmissible with no warning at all.
    #
    # Ranking by (became-ready-later, index) fixes it: every node that starts
    # with indegree zero carries rank 0 and so precedes every node that only
    # becomes ready once its predecessors are placed.
    heap = [(0, i) for i in range(n) if indegree[i] == 0]
    heapq.heapify(heap)
    order: List[int] = []
    while heap:
        _rank, node = heapq.heappop(heap)
        order.append(node)
        for t in successors[node]:
            indegree[t] -= 1
            if indegree[t] == 0:
                heapq.heappush(heap, (1, t))

    if len(order) < n:
        placed = set(order)
        order.extend(i for i in range(n) if i not in placed)
        return order, True
    return order, False


def _repair_symmetric(matrix: np.ndarray, label_en: str, label_ar: str,
                      warnings: List[str]) -> np.ndarray:
    """Make ``matrix`` positive definite while preserving its diagonal exactly."""
    if is_positive_definite(matrix):
        return matrix
    target = np.diag(matrix).copy()
    repaired = nearest_positive_definite(matrix, eps=max(HEYWOOD_FLOOR * 1e-3, 1e-8))
    np.fill_diagonal(repaired, target)
    if not is_positive_definite(repaired):
        repaired = nearest_positive_definite(repaired,
                                             eps=max(HEYWOOD_FLOOR * 1e-3, 1e-8))
        np.fill_diagonal(repaired, target)
    warnings.append(_bilingual(
        f"PD_REPAIR: {label_en} was not positive definite and was repaired with "
        "Higham's (2002) nearest-positive-definite projection; the variances on "
        "its diagonal were preserved.",
        f"إصلاح التحديد الموجب: {label_ar} لم تكن موجبة التحديد، فتم إصلاحها "
        "بطريقة هيغام (2002) لأقرب مصفوفة موجبة التحديد مع الحفاظ على التباينات "
        "الموجودة على قطرها.",
    ))
    return repaired


def _apply_disturbance_correlations(Psi: np.ndarray, psi_corr: np.ndarray,
                                    pairs: Sequence[Tuple[int, int]]) -> None:
    """Re-express the requested disturbance *correlations* as covariances.

    ``LatentCovariance.correlation`` is a correlation, not a covariance, so the
    entry that belongs in ``Psi`` is ``r * sd_a * sd_b`` with
    ``sd_i = sqrt(Psi[i, i])``.  Writing ``r`` itself is only right when both
    disturbance variances are one, i.e. when both latents are exogenous.  An
    endogenous latent has ``Psi[i, i] = 1 - R^2 < 1``, so the unscaled entry
    implies a disturbance correlation of ``r / sqrt(Psi_aa Psi_bb)``, which is
    strictly larger than ``r`` in magnitude and can exceed one outright — a
    non-positive-definite ``Psi`` that the engine then generates data from.

    Writing the scaled entry has a second benefit: because ``Psi`` is then
    ``D^(1/2) psi_corr D^(1/2)`` with ``D = diag(Psi)`` positive, it inherits
    positive definiteness from ``psi_corr`` no matter how far the fixed-point
    iteration drives the disturbance variances down.

    Parameters
    ----------
    Psi : numpy.ndarray
        ``(m, m)`` disturbance covariance; its off-diagonal entries for
        ``pairs`` are overwritten in place from its current diagonal.
    psi_corr : numpy.ndarray
        ``(m, m)`` matrix of requested disturbance *correlations*, unit
        diagonal, as settled (and possibly PD-repaired) in step 3.
    pairs : sequence of (int, int)
        The ``(a, b)`` index pairs carrying a requested correlation.
    """
    for a, b in pairs:
        sd = float(np.sqrt(max(float(Psi[a, a]), 0.0) * max(float(Psi[b, b]), 0.0)))
        Psi[a, b] = Psi[b, a] = float(psi_corr[a, b]) * sd


# --------------------------------------------------------------------------- #
# Step 1: the latent node list
# --------------------------------------------------------------------------- #


def _build_nodes(spec: ModelSpec) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Latent node ids and names before topological ordering.

    Returns
    -------
    (list of str, list of str, dict)
        Node ids, display names, and a map from factor id to the id of the
        construct that owns it (a construct owns itself).
    """
    ids: List[str] = []
    names: List[str] = []
    owner: Dict[str, str] = {}
    for c in spec.constructs:
        ids.append(c.id)
        names.append(c.name or c.label.get("en") or c.id)
        owner[c.id] = c.id
        for d in c.dimensions:
            ids.append(d.id)
            names.append(d.name or d.label.get("en") or d.id)
            owner[d.id] = c.id
    return ids, names, owner


# --------------------------------------------------------------------------- #
# Step 4 fallback: fixed-point solve for a cyclic / correlated-disturbance model
# --------------------------------------------------------------------------- #


def _solve_phi_fixed_point(B: np.ndarray, Psi: np.ndarray,
                           endogenous: np.ndarray, factor_names: Sequence[str],
                           warnings: List[str],
                           psi_corr: Optional[np.ndarray] = None,
                           psi_pairs: Sequence[Tuple[int, int]] = (),
                           ) -> np.ndarray:
    """Rescale ``diag(Psi)`` until ``diag(Phi)`` is one, then return ``Phi``.

    Used for non-recursive models and for models with correlated disturbances
    between endogenous latents, where the single forward pass of the recursive
    algorithm does not hold.  Each pass evaluates
    ``Phi = (I - B)^-1 Psi (I - B)^-T`` and applies the Newton correction
    ``Psi[i, i] += (1 - Phi[i, i]) / slope_i``, where ``slope_i`` is the exact
    derivative ``d Phi[i, i] / d Psi[i, i]``::

        A       = (I - B)^-1,   g = sqrt(diag(Psi)),   R = psi_corr
        slope_i = A_ii^2 + A_ii * sum_(l != i) A_il R_il g_l / g_i

    A unit slope — what an earlier version of this solver assumed — is only
    right for an uncoupled latent of a recursive model, where ``A_ii`` is one
    and the sum is empty.  It is wrong inside a feedback loop (``A_ii != 1``)
    and badly wrong for a correlated disturbance, whose contribution to
    ``Phi[i, i]`` goes as ``sqrt(Psi[i, i])`` and so has a slope that grows
    without bound as the disturbance variance shrinks; the undamped step then
    oscillated past the solution until the pass budget ran out.  When the
    correction would drive a disturbance variance below ``HEYWOOD_FLOOR`` the
    corresponding row of ``B`` is shrunk instead, which is the latent Heywood
    repair.

    Parameters
    ----------
    B : numpy.ndarray
        ``(m, m)`` structural matrix; modified in place when a row is shrunk.
    Psi : numpy.ndarray
        ``(m, m)`` disturbance covariance; its diagonal is modified in place.
    endogenous : numpy.ndarray
        ``(m,)`` boolean mask of latents whose variance may be rescaled.
    factor_names : Sequence[str]
        Display names, used in the warning text.
    warnings : list of str
        Appended to in place.
    psi_corr : numpy.ndarray, optional
        ``(m, m)`` requested disturbance *correlations* with a unit diagonal.
        Every pass rescales the off-diagonal entries listed in ``psi_pairs``
        from the freshly updated diagonal, because the diagonal moves on every
        pass and a correlation written as a raw covariance would silently drift
        with it.
    psi_pairs : sequence of (int, int), optional
        The index pairs in ``psi_corr`` that carry a requested correlation.

    Returns
    -------
    numpy.ndarray
        ``(m, m)`` implied latent covariance with a unit diagonal.

    Raises
    ------
    semsim.core.errors.ConvergenceError
        If ``MAX_FIXED_POINT_PASSES`` passes do not reach ``PHI_DIAG_TOL``.

    References
    ----------
    Joreskog, K. G. (1973). A general method for estimating a linear structural
        equation system. In A. S. Goldberger & O. D. Duncan (Eds.),
        *Structural Equation Models in the Social Sciences* (pp. 85-112).
        Seminar Press.
    Bentler, P. M., & Freeman, E. H. (1983). Tests for stability in linear
        structural equation systems. *Psychometrika*, 48(1), 143-145.
    """
    m = B.shape[0]
    eye = np.eye(m)
    shrunk: Set[int] = set()
    # The entry Psi[a, b] must track the diagonal, so the pass that changes a
    # disturbance variance has to rewrite the covariance built from it. Doing
    # it once up front as well keeps the very first evaluation of Phi
    # consistent with the diagonal it is derived from.
    if psi_corr is not None and len(psi_pairs):
        _apply_disturbance_correlations(Psi, psi_corr, psi_pairs)

    for _ in range(MAX_FIXED_POINT_PASSES):
        try:
            inv = np.linalg.inv(eye - B)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceError(
                "The structural matrix (I - B) became singular while solving "
                "for the latent covariance of a non-recursive model.",
                "أصبحت مصفوفة المسارات (I - B) غير قابلة للعكس أثناء حساب "
                "تغايرات المتغيرات الكامنة في نموذج غير تعاقبي؛ قلّل معاملات "
                "المسارات داخل الحلقة.",
                detail=str(exc),
            ) from exc

        Phi = inv @ Psi @ inv.T
        Phi = 0.5 * (Phi + Phi.T)
        diag = np.diag(Phi).copy()
        error = float(np.max(np.abs(diag - 1.0)))
        if error < PHI_DIAG_TOL:
            scale = 1.0 / np.sqrt(np.clip(diag, 1e-300, None))
            Phi = Phi * np.outer(scale, scale)
            np.fill_diagonal(Phi, 1.0)
            return 0.5 * (Phi + Phi.T)

        # Snapshot of the disturbance standard deviations this pass started
        # from, used for the Newton slope below; Psi[i, i] is written in place
        # inside the loop, so reading it back would mix two passes.
        sd_in = np.sqrt(np.clip(np.diag(Psi), 0.0, None))

        for i in range(m):
            if not endogenous[i]:
                continue
            # d Phi[i, i] / d Psi[i, i] is one only for an uncoupled node of a
            # recursive model. With ``Psi = D^(1/2) R D^(1/2)`` and
            # ``A = (I - B)^-1``, Phi[i, i] = sum_kl A_ik A_il R_kl g_k g_l with
            # g = sqrt(diag(Psi)), so the exact slope is
            # ``A_ii^2 + A_ii * S_i / g_i`` with
            # ``S_i = sum_(l != i) A_il R_il g_l``. The correlated-disturbance
            # term grows without bound as g_i shrinks, which is exactly the
            # regime a strong incoming path puts an endogenous latent in, so
            # the old undamped step (slope fixed at one) overshot and
            # oscillated until the pass budget ran out on models that do have a
            # solution. Dividing by the true slope is a proper Newton step and
            # converges on them.
            slope = 1.0
            if sd_in[i] > 0.0:
                a_ii = float(inv[i, i])
                coupling = 0.0
                if psi_corr is not None:
                    for pa, pb in psi_pairs:
                        partner = pb if pa == i else (pa if pb == i else None)
                        if partner is None:
                            continue
                        coupling += (float(inv[i, partner])
                                     * float(psi_corr[i, partner])
                                     * sd_in[partner])
                slope = a_ii * a_ii + a_ii * coupling / sd_in[i]
            # A slope at or below zero means the linearisation points the wrong
            # way; fall back on a bounded positive step and let the next pass
            # re-linearise rather than marching off.
            slope = max(slope, _MIN_NEWTON_SLOPE)
            proposed = Psi[i, i] + (1.0 - diag[i]) / slope
            if proposed < HEYWOOD_FLOOR:
                explained = max(diag[i] - Psi[i, i], HEYWOOD_FLOOR)
                factor = float(np.sqrt((1.0 - HEYWOOD_FLOOR) / explained))
                if factor < 1.0:
                    B[i, :] *= factor
                    if i not in shrunk:
                        shrunk.add(i)
                        warnings.append(_bilingual(
                            f"HEYWOOD_LATENT: the predictors of "
                            f"'{factor_names[i]}' explain more than its total "
                            "variance inside a feedback loop; its incoming "
                            f"coefficients were shrunk by {factor:.4f} and its "
                            f"disturbance variance clamped to {HEYWOOD_FLOOR:g}.",
                            f"حالة هيوود كامنة: المتغيرات المتنبئة بالمتغير "
                            f"«{factor_names[i]}» تفسّر أكثر من تباينه الكلي داخل "
                            f"حلقة تغذية راجعة، فتم تقليص معاملاته الداخلة بمقدار "
                            f"{factor:.4f} وتثبيت تباين البواقي عند "
                            f"{HEYWOOD_FLOOR:g}.",
                        ))
                Psi[i, i] = HEYWOOD_FLOOR
            else:
                Psi[i, i] = proposed

        # diag(Psi) has just moved, so the correlated-disturbance covariances
        # derived from it are stale; rebuild them before the next evaluation of
        # Phi. Without this the realised disturbance correlation drifts away
        # from the one the researcher asked for, and the drift is upward, which
        # is the direction that breaks positive definiteness.
        if psi_corr is not None and len(psi_pairs):
            _apply_disturbance_correlations(Psi, psi_corr, psi_pairs)

    raise ConvergenceError(
        "Could not give every latent variable unit variance in this "
        f"non-recursive model after {MAX_FIXED_POINT_PASSES} passes. Reduce the "
        "coefficients inside the feedback loop or respecify the model as "
        "recursive.",
        "تعذّر جعل تباين كل متغير كامن مساويًا للواحد في هذا النموذج غير التعاقبي "
        f"بعد {MAX_FIXED_POINT_PASSES} محاولة. قلّل معاملات المسارات داخل الحلقة "
        "التغذية الراجعة أو أعد صياغة النموذج ليكون تعاقبيًا.",
        passes=MAX_FIXED_POINT_PASSES,
    )


# --------------------------------------------------------------------------- #
# Step 5 helper: automatic cross-loadings
# --------------------------------------------------------------------------- #


def _auto_cross_loadings(spec: ModelSpec, Lambda: np.ndarray,
                         item_ids: Sequence[str], home_factor: Sequence[int],
                         factor_ids: Sequence[str], owner_of: Dict[str, str],
                         item_construct: Sequence[str],
                         method_index: Optional[int],
                         explicit_rows: Set[int],
                         rng: np.random.Generator,
                         warnings: List[str],
                         drawn: Optional[List[Tuple[str, str]]] = None,
                         covariate_rows: Sequence[int] = (),
                         covariate_factors: Sequence[int] = ()) -> None:
    """Draw cross-loadings from ``spec.cross_loading_policy`` in place.

    Mimics fuzzy item wording: a random share of items picks up a small
    secondary loading on a factor that is not its home.

    References
    ----------
    Marsh, H. W., Muthen, B., Asparouhov, T., Ludtke, O., Robitzsch, A.,
        Morin, A. J. S., & Trautwein, U. (2009). Exploratory structural
        equation modeling. *Structural Equation Modeling*, 16(3), 439-476.
    """
    policy = spec.cross_loading_policy
    # A control is an observed characteristic, not a worded question: it picks
    # up no cross-loading and no question cross-loads on it.
    p = len(item_ids) - len(covariate_rows)
    if not policy.enabled or p == 0:
        return

    low, high = policy.magnitude_range
    low, high = float(min(low, high)), float(max(low, high))
    if not (0.0 <= low <= high < 1.0):
        raise SpecificationError(
            "The cross-loading magnitude range must lie inside [0, 1).",
            "مدى حجم التحميلات المتقاطعة يجب أن يكون داخل الفترة [0، 1).",
            magnitude_range=policy.magnitude_range,
        )

    eligible = [j for j in range(p) if j not in explicit_rows]
    wanted = int(round(float(policy.proportion_of_items) * p))
    wanted = max(0, min(wanted, len(eligible)))
    if wanted == 0:
        return

    chosen = rng.choice(np.asarray(eligible, dtype=int), size=wanted,
                        replace=False)
    max_per_item = max(int(policy.max_per_item), 0)

    for j in sorted(int(x) for x in chosen):
        own_construct = item_construct[j]
        candidates = [
            fi for fi in range(len(factor_ids))
            if fi != home_factor[j]
            and (method_index is None or fi != method_index)
            and fi not in covariate_factors
            and not (policy.across_constructs_only
                     and owner_of.get(factor_ids[fi]) == own_construct)
        ]
        if not candidates:
            continue
        k = min(max_per_item, len(candidates))
        if k == 0:
            continue
        picks = rng.choice(np.asarray(candidates, dtype=int), size=k,
                           replace=False)
        for fi in np.atleast_1d(picks):
            magnitude = float(rng.uniform(low, high))
            if policy.allow_negative and bool(rng.integers(0, 2)):
                magnitude = -magnitude
            Lambda[j, int(fi)] = magnitude
            if drawn is not None:
                drawn.append((item_ids[j], factor_ids[int(fi)]))

    warnings.append(_bilingual(
        f"AUTO_CROSSLOADINGS: {wanted} item(s) received an automatically drawn "
        "cross-loading from the cross-loading policy.",
        f"تحميلات متقاطعة تلقائية: تم توليد تحميل متقاطع لـ {wanted} من البنود "
        "وفقًا لسياسة التحميلات المتقاطعة.",
    ))


# --------------------------------------------------------------------------- #
# The compiler
# --------------------------------------------------------------------------- #


def compile_model(spec: ModelSpec,
                  rng: Optional[np.random.Generator] = None) -> CompiledModel:
    """Compile a :class:`~semsim.core.spec.ModelSpec` into LISREL matrices.

    Performs the eight-step standardisation documented in
    ``docs/ARCHITECTURE.md``: build and topologically order the latent nodes,
    populate ``B`` from second-order loadings and structural paths, set the
    exogenous block of ``Psi`` from the requested latent correlations, derive
    every endogenous disturbance variance so that ``diag(Phi) == 1``, build
    ``Lambda`` from primary loadings, cross-loadings and the method factor,
    derive ``diag(Theta)`` from the implied communalities, apply unreliability
    by inflating and renormalising, and finally install the correlated
    residuals.

    Heywood cases (explained variance at or above one, communality at or above
    one) are repaired by shrinking the offending coefficients and clamping the
    residual variance to ``HEYWOOD_FLOOR``, with a bilingual warning naming the
    variable; they never raise.

    Parameters
    ----------
    spec : ModelSpec
        The specification to compile.
    rng : numpy.random.Generator, optional
        Source of randomness for automatic cross-loadings.  When omitted a
        generator is seeded from ``spec.seed`` so compilation stays
        reproducible.  The global ``numpy.random`` state is never touched.

    Returns
    -------
    CompiledModel
        Matrices in the completely standardised metric: ``diag(Phi) == 1`` and
        ``diag(Lambda Phi Lambda' + Theta) == 1``.

    Raises
    ------
    semsim.core.errors.SpecificationError
        For a specification the compiler cannot interpret: an unknown item,
        factor or construct reference, a residual correlation of magnitude one
        or more, a negative error variance, or a non-positive variance
        inflation.
    semsim.core.errors.ConvergenceError
        If a non-recursive model cannot be standardised.

    Examples
    --------
    >>> import numpy as np
    >>> from semsim.core.spec import ModelSpec
    >>> from semsim.core.compile import compile_model
    >>> spec = ModelSpec()
    >>> _ = spec.add_construct("X", n_items=3, loading=0.8)
    >>> _ = spec.add_construct("Y", n_items=3, loading=0.8)
    >>> _ = spec.add_path("X", "Y", coefficient=0.5)
    >>> cm = compile_model(spec)
    >>> bool(np.allclose(np.diag(cm.Phi), 1.0))
    True

    References
    ----------
    Joreskog, K. G. (1970). A general method for analysis of covariance
        structures. *Biometrika*, 57(2), 239-251.
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        chapters 3-4. Wiley.
    Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*.
        Scientific Software International.
    """
    warnings: List[str] = []
    spec = spec.expanded()
    if rng is None:
        rng = np.random.default_rng(spec.seed)

    # ---------------------------------------------------------------- Step 1
    raw_ids, raw_names, owner_of = _build_nodes(spec)
    cmb = spec.common_method_bias
    use_method = bool(cmb.enabled)
    if use_method:
        raw_ids.append("__method__")
        raw_names.append(cmb.label.get("en") or "Common method factor")
        owner_of["__method__"] = "__method__"

    m = len(raw_ids)
    if m == 0:
        raise SpecificationError(
            "The model contains no constructs, so there is nothing to compile.",
            "النموذج لا يحتوي على أي متغير كامن، لذلك لا يوجد ما يمكن تصريفه.",
        )
    raw_index = {fid: i for i, fid in enumerate(raw_ids)}

    # Edges in the *pre-ordering* index space: (source, target).
    edges: List[Tuple[int, int]] = []
    for c in spec.constructs:
        for d in c.dimensions:
            edges.append((raw_index[c.id], raw_index[d.id]))
    for path in spec.paths:
        if path.source not in raw_index or path.target not in raw_index:
            raise SpecificationError(
                f"Path {path.id!r} refers to an unknown construct "
                f"({path.source!r} -> {path.target!r}).",
                f"المسار «{path.id}» يشير إلى متغير كامن غير موجود "
                f"(«{path.source}» ← «{path.target}»).",
                path_id=path.id,
            )
        if path.source == path.target:
            warnings.append(_bilingual(
                f"SELF_PATH: path {path.id!r} points a construct at itself and "
                "was ignored.",
                f"مسار ذاتي: المسار «{path.id}» يربط المتغير بنفسه، وقد تم "
                "تجاهله.",
            ))
            continue
        edges.append((raw_index[path.source], raw_index[path.target]))

    order, has_cycle = _topological_order(m, edges)
    if has_cycle:
        warnings.append(_bilingual(
            "CYCLE_IN_PATHS: the structural model is non-recursive (it contains "
            "a feedback loop), so the latent covariances were obtained by "
            "fixed-point iteration rather than a single forward pass.",
            "حلقة في المسارات: النموذج البنائي غير تعاقبي (يحتوي على حلقة تغذية "
            "راجعة)، لذلك تم حساب تغايرات المتغيرات الكامنة بالتكرار حتى الاستقرار "
            "بدلًا من مرور أمامي واحد.",
        ))

    factor_ids = [raw_ids[k] for k in order]
    factor_names = [raw_names[k] for k in order]
    factor_index = {fid: i for i, fid in enumerate(factor_ids)}
    method_factor_index = factor_index["__method__"] if use_method else None

    # ---------------------------------------------------------------- Step 2
    B = np.zeros((m, m), dtype=float)
    for c in spec.constructs:
        ci = factor_index[c.id]
        for d in c.dimensions:
            B[factor_index[d.id], ci] = float(d.second_order_loading)
    for path in spec.paths:
        if path.source == path.target:
            continue
        ti = factor_index[path.target]
        si = factor_index[path.source]
        if B[ti, si] != 0.0:
            warnings.append(_bilingual(
                f"DUPLICATE_PATH: more than one path targets "
                f"'{factor_names[ti]}' from '{factor_names[si]}'; the last one "
                "wins.",
                f"مسار مكرر: يوجد أكثر من مسار من «{factor_names[si]}» إلى "
                f"«{factor_names[ti]}»، وقد تم الاعتماد على الأخير منها.",
            ))
        B[ti, si] = float(path.resolved_coefficient())

    endogenous = np.any(B != 0.0, axis=1)

    if spec.interactions:
        warnings.append(_bilingual(
            f"INTERACTIONS_NOT_IN_SIGMA: {len(spec.interactions)} latent "
            "interaction term(s) cannot be represented in a covariance "
            "structure and are applied by the score-based generator instead, so "
            "Sigma(theta) carries the main effects only.",
            f"تفاعلات غير ممثَّلة في مصفوفة التغاير: عدد "
            f"{len(spec.interactions)} من حدود التفاعل الكامنة لا يمكن تمثيلها "
            "في بنية التغاير، ويتم تطبيقها عبر مولّد الدرجات الكامنة؛ لذلك تحتوي "
            "مصفوفة التغاير المُشتقة على الآثار الرئيسة فقط.",
        ))

    # ---------------------------------------------------------------- Step 3
    Psi = np.zeros((m, m), dtype=float)
    np.fill_diagonal(Psi, 1.0)
    correlated_disturbance = False
    for lc in spec.latent_covariances:
        if lc.construct_a not in factor_index or lc.construct_b not in factor_index:
            raise SpecificationError(
                "A latent covariance refers to an unknown construct "
                f"({lc.construct_a!r}, {lc.construct_b!r}).",
                "أحد التغايرات بين المتغيرات الكامنة يشير إلى متغير غير موجود "
                f"(«{lc.construct_a}»، «{lc.construct_b}»).",
            )
        a = factor_index[lc.construct_a]
        b = factor_index[lc.construct_b]
        if a == b:
            warnings.append(_bilingual(
                f"LATENT_COV_SAME: a covariance between '{factor_names[a]}' and "
                "itself was ignored.",
                f"تغاير مع النفس: تم تجاهل التغاير المحدَّد بين المتغير "
                f"«{factor_names[a]}» ونفسه.",
            ))
            continue
        r = float(lc.correlation)
        if abs(r) >= 1.0:
            raise SpecificationError(
                f"The correlation between {factor_names[a]!r} and "
                f"{factor_names[b]!r} must be smaller than 1 in absolute value; "
                f"{r} was supplied.",
                f"الارتباط بين «{factor_names[a]}» و«{factor_names[b]}» يجب أن "
                f"تكون قيمته المطلقة أصغر من 1، والقيمة المُدخلة هي {r}.",
                correlation=r,
            )
        if endogenous[a] or endogenous[b]:
            correlated_disturbance = True
            warnings.append(_bilingual(
                f"ENDOGENOUS_COVARIANCE: '{factor_names[a]}' and "
                f"'{factor_names[b]}' are not both exogenous, so the requested "
                f"correlation of {r:g} was installed as a correlated "
                "disturbance and the latent covariances were solved "
                "iteratively.",
                f"تغاير بين متغيرين داخليين: «{factor_names[a]}» و"
                f"«{factor_names[b]}» ليسا كلاهما خارجيَّين، لذلك تم تمثيل "
                f"الارتباط المطلوب ({r:g}) كارتباط بين بواقي المتغيرين، وتم حساب "
                "التغايرات الكامنة بالتكرار.",
            ))
        Psi[a, b] = Psi[b, a] = r

    Psi = _repair_symmetric(
        Psi, "the latent disturbance covariance matrix Psi",
        "مصفوفة تغاير بواقي المتغيرات الكامنة (Psi)", warnings)

    # At this point diag(Psi) is still all ones, so Psi *is* the matrix of
    # requested disturbance correlations (post-repair, which only ever touches
    # the off-diagonal). Keep that correlation metric aside: the step-4 passes
    # below replace the diagonal with 1 - R^2 for every endogenous latent, and
    # each off-diagonal then has to be rebuilt as r * sd_a * sd_b so the
    # correlation the researcher typed is the correlation actually realised.
    psi_corr = Psi.copy()
    psi_pairs = [(int(a), int(b))
                 for a, b in zip(*np.triu_indices(m, k=1))
                 if psi_corr[a, b] != 0.0]

    # ---------------------------------------------------------------- Step 4
    if has_cycle or correlated_disturbance:
        Phi = _solve_phi_fixed_point(B, Psi, endogenous, factor_names, warnings,
                                     psi_corr=psi_corr, psi_pairs=psi_pairs)
    else:
        Phi = np.zeros((m, m), dtype=float)
        processed: List[int] = []
        for i in range(m):          # B is lower triangular in this ordering
            predictors = np.flatnonzero(B[i] != 0.0)
            if predictors.size == 0:
                Phi[i, i] = 1.0
                for j in processed:
                    Phi[i, j] = Phi[j, i] = Psi[i, j]
                processed.append(i)
                continue
            b = B[i, predictors].copy()
            block = Phi[np.ix_(predictors, predictors)]
            explained = float(b @ block @ b)
            if explained > 1.0 - HEYWOOD_FLOOR:
                original = np.array2string(b, precision=4, separator=", ")
                r_squared = explained
                factor = float(np.sqrt((1.0 - HEYWOOD_FLOOR) / max(explained, 1e-300)))
                b = b * factor
                B[i, predictors] = b
                retained = np.array2string(b, precision=4, separator=", ")
                Psi[i, i] = HEYWOOD_FLOOR
                explained = 1.0 - HEYWOOD_FLOOR
                warnings.append(_bilingual(
                    f"HEYWOOD_LATENT: the predictors of '{factor_names[i]}' "
                    f"explain R-squared = {r_squared:.4f} of its unit variance, "
                    "leaving no room for a disturbance. The incoming "
                    f"coefficients were shrunk from {original} to {retained} and "
                    f"its disturbance variance clamped to {HEYWOOD_FLOOR:g}.",
                    f"حالة هيوود كامنة: المتغيرات المتنبئة بالمتغير "
                    f"«{factor_names[i]}» تفسّر من تباينه الوحدوي ما يساوي "
                    f"معامل تحديد قدره {r_squared:.4f}، فلا يتبقى له تباين "
                    "للبواقي. تم تقليص المعاملات الداخلة من "
                    f"{original} إلى {retained} وتثبيت تباين بواقيه عند "
                    f"{HEYWOOD_FLOOR:g}.",
                ))
            else:
                Psi[i, i] = 1.0 - explained
            # Psi[i, i] is only known now, so any requested correlation that
            # involves node i has to be turned into a covariance at this point
            # and not in step 3. A model reaching this branch has both ends of
            # every requested correlation exogenous (anything else sets
            # ``correlated_disturbance`` and goes through the fixed-point
            # solver), so today this is a no-op guard that keeps the branch
            # correct if that gating is ever widened.
            if psi_pairs:
                _apply_disturbance_correlations(
                    Psi, psi_corr, [(a, b) for a, b in psi_pairs if i in (a, b)])
            for j in processed:
                Phi[i, j] = Phi[j, i] = float(b @ Phi[predictors, j]) + Psi[i, j]
            Phi[i, i] = 1.0
            processed.append(i)
        Phi = 0.5 * (Phi + Phi.T)
        np.fill_diagonal(Phi, 1.0)

    if not is_positive_definite(Phi):
        Phi = _repair_symmetric(
            Phi, "the implied latent covariance matrix Phi",
            "مصفوفة التغاير المُشتقة للمتغيرات الكامنة (Phi)", warnings)
        np.fill_diagonal(Phi, 1.0)

    # ---------------------------------------------------------------- Step 5
    items = list(spec.all_items())
    p = len(items)
    item_ids = [it.id for it in items]
    item_names = [it.name or it.id for it in items]
    item_index = {iid: j for j, iid in enumerate(item_ids)}

    Lambda = np.zeros((p, m), dtype=float)
    home_factor: List[int] = [0] * p
    item_construct: List[str] = [""] * p
    for j, (item, construct, dimension) in enumerate(spec.iter_items_with_owner()):
        home_id = dimension.id if dimension is not None else construct.id
        fi = factor_index[home_id]
        home_factor[j] = fi
        item_construct[j] = construct.id
        Lambda[j, fi] = float(item.loading)

    explicit_rows: Set[int] = set()
    for cl in spec.cross_loadings:
        if cl.item_id not in item_index:
            raise SpecificationError(
                f"A cross-loading refers to an unknown item ({cl.item_id!r}).",
                f"أحد التحميلات المتقاطعة يشير إلى بند غير موجود («{cl.item_id}»).",
                item_id=cl.item_id,
            )
        if cl.factor_id not in factor_index:
            raise SpecificationError(
                f"A cross-loading refers to an unknown factor ({cl.factor_id!r}).",
                "أحد التحميلات المتقاطعة يشير إلى عامل كامن غير موجود "
                f"(«{cl.factor_id}»).",
                factor_id=cl.factor_id,
            )
        j = item_index[cl.item_id]
        fi = factor_index[cl.factor_id]
        if fi == home_factor[j]:
            warnings.append(_bilingual(
                f"CROSSLOAD_HOME_FACTOR: item '{item_names[j]}' cross-loads on "
                "its own home factor; the entry was ignored.",
                f"تحميل متقاطع على العامل الأصلي: البند «{item_names[j]}» "
                "يُحمَّل متقاطعًا على عامله الأصلي نفسه، وقد تم تجاهل هذا الإدخال.",
            ))
            continue
        Lambda[j, fi] = float(cl.loading)
        explicit_rows.add(j)

    covariate_ids = spec.covariate_item_ids()
    covariate_rows = [j for j, iid in enumerate(item_ids) if iid in covariate_ids]
    covariate_set = set(covariate_rows)
    covariate_factors = sorted(home_factor[j] for j in covariate_rows)

    auto_cross_loadings: List[Tuple[str, str]] = []
    _auto_cross_loadings(spec, Lambda, item_ids, home_factor, factor_ids,
                         owner_of, item_construct, method_factor_index,
                         explicit_rows, rng, warnings, auto_cross_loadings,
                         covariate_rows, covariate_factors)

    if use_method:
        if cmb.item_ids:
            affected = []
            for iid in cmb.item_ids:
                if iid not in item_index:
                    raise SpecificationError(
                        "The common-method factor refers to an unknown item "
                        f"({iid!r}).",
                        "عامل طريقة القياس المشترك يشير إلى بند غير موجود "
                        f"(«{iid}»).",
                        item_id=iid,
                    )
                affected.append(item_index[iid])
        else:
            affected = [j for j in range(p) if j not in covariate_set]
        for j in affected:
            Lambda[j, method_factor_index] = float(cmb.loading)

    # ---------------------------------------------------------------- Step 6
    Theta = np.zeros((p, p), dtype=float)
    communality = np.einsum("ij,jk,ik->i", Lambda, Phi, Lambda)
    for j in range(p):
        if j in covariate_set:
            continue        # observed without error: Theta[j, j] stays zero
        if communality[j] > 1.0 - HEYWOOD_FLOOR:
            factor = float(np.sqrt((1.0 - HEYWOOD_FLOOR)
                                   / max(communality[j], 1e-300)))
            Lambda[j, :] *= factor
            communality[j] = 1.0 - HEYWOOD_FLOOR
            Theta[j, j] = HEYWOOD_FLOOR
            warnings.append(_bilingual(
                f"HEYWOOD_ITEM: the loadings of item '{item_names[j]}' imply a "
                "communality of 1 or more, which leaves no unique variance. The "
                f"whole loading row was scaled by {factor:.4f} and the unique "
                f"variance clamped to {HEYWOOD_FLOOR:g}.",
                f"حالة هيوود في بند: تحميلات البند «{item_names[j]}» تفيد بأن "
                "تباينه المشترك يساوي 1 أو يزيد، فلا يتبقى له تباين فريد. تم "
                f"ضرب صف تحميلاته بالمعامل {factor:.4f} وتثبيت تباينه الفريد عند "
                f"{HEYWOOD_FLOOR:g}.",
            ))
        else:
            Theta[j, j] = 1.0 - communality[j]

    # ---------------------------------------------------------------- Step 7
    inflation = (float(spec.noise.unique_variance_inflation)
                 if spec.noise.enabled else 1.0)
    if inflation <= 0.0:
        raise SpecificationError(
            "noise.unique_variance_inflation must be greater than zero; "
            f"{inflation} was supplied.",
            "معامل تضخيم التباين الفريد (unique_variance_inflation) يجب أن يكون "
            f"أكبر من صفر، والقيمة المُدخلة هي {inflation}.",
            inflation=inflation,
        )
    for j, item in enumerate(items):
        if j in covariate_set:
            continue
        extra = float(item.extra_error_variance)
        if extra < 0.0:
            raise SpecificationError(
                f"Item {item_names[j]!r} has a negative extra error variance "
                f"({extra}); unique variance cannot be negative.",
                f"البند «{item_names[j]}» له تباين خطأ إضافي سالب ({extra})، "
                "والتباين الفريد لا يمكن أن يكون سالبًا.",
                item=item_names[j],
            )
        if inflation == 1.0 and extra == 0.0:
            continue
        inflated = Theta[j, j] * inflation + extra
        total = communality[j] + inflated
        Lambda[j, :] /= np.sqrt(total)
        Theta[j, j] = inflated / total
        communality[j] = communality[j] / total

    # ---------------------------------------------------------------- Step 8
    for ec in spec.error_covariances:
        for ref in (ec.item_a, ec.item_b):
            if ref not in item_index:
                raise SpecificationError(
                    f"A residual covariance refers to an unknown item ({ref!r}).",
                    f"أحد تغايرات البواقي يشير إلى بند غير موجود («{ref}»).",
                    item_id=ref,
                )
        a = item_index[ec.item_a]
        b = item_index[ec.item_b]
        if a == b:
            warnings.append(_bilingual(
                f"ERRCOV_SAME_ITEM: a residual covariance of item "
                f"'{item_names[a]}' with itself was ignored.",
                f"تغاير بواقٍ مع النفس: تم تجاهل تغاير بواقي البند "
                f"«{item_names[a]}» مع نفسه.",
            ))
            continue
        r = float(ec.correlation)
        if abs(r) >= 1.0:
            raise SpecificationError(
                f"The residual correlation between {item_names[a]!r} and "
                f"{item_names[b]!r} must be smaller than 1 in absolute value; "
                f"{r} was supplied.",
                f"ارتباط البواقي بين البندين «{item_names[a]}» و"
                f"«{item_names[b]}» يجب أن تكون قيمته المطلقة أصغر من 1، "
                f"والقيمة المُدخلة هي {r}.",
                correlation=r,
            )
        Theta[a, b] = Theta[b, a] = r * float(np.sqrt(Theta[a, a] * Theta[b, b]))

    # Control terms have zero residual variance, so only the questionnaire
    # items can be, or need to be, positive definite here.
    measured = np.array([j for j in range(p) if j not in covariate_set], dtype=int)
    Theta[np.ix_(measured, measured)] = _repair_symmetric(
        Theta[np.ix_(measured, measured)],
        "the item residual covariance matrix Theta",
        "مصفوفة تغاير بواقي البنود (Theta)", warnings)

    # A reverse-worded item is answered on an inverted scale, so its raw
    # responses correlate NEGATIVELY with its factor until the researcher
    # recodes them. Modelling that as a negated loading keeps Sigma, the
    # generated data and any estimator fitted to them consistent, and lets a
    # simulated dataset reproduce the diagnostic a forgotten recode produces:
    # a negative item-total correlation and a collapsed alpha.
    reversed_items = [it.id for it in spec.all_items() if it.reverse_coded]
    for iid in reversed_items:
        j = item_index.get(iid)
        if j is not None:
            Lambda[j, :] *= -1.0
    if reversed_items:
        warnings.append(_bilingual(
            f"REVERSE_CODED: {len(reversed_items)} item(s) are reverse-worded, "
            "so their loadings carry a negative sign and their raw responses "
            "run against their factor until they are recoded.",
            f"بنود معكوسة الصياغة: {len(reversed_items)} من البنود معكوسة، "
            "لذا تحمل تشبعاتها إشارة سالبة وتسير استجاباتها الخام عكس عاملها "
            "إلى أن يُعاد ترميزها.",
        ))

    compiled = CompiledModel(
        spec=spec,
        item_ids=item_ids,
        item_names=item_names,
        factor_ids=factor_ids,
        factor_names=factor_names,
        Lambda=Lambda,
        B=B,
        Psi=Psi,
        Theta=Theta,
        Phi=Phi,
        endogenous=np.any(B != 0.0, axis=1),
        item_index=item_index,
        factor_index=factor_index,
        method_factor_index=method_factor_index,
        home_factor=home_factor,
        auto_cross_loadings=auto_cross_loadings,
        warnings=warnings,
        covariate_rows=covariate_rows,
        covariate_factors=covariate_factors,
    )
    return compiled

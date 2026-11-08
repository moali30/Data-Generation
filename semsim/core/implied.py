"""
semsim.core.implied
===================
Everything that can be read off a :class:`~semsim.core.compile.CompiledModel`
without touching data: the model-implied latent covariance ``Phi``, the
model-implied observed covariance ``Sigma(theta)``, total / direct / indirect
effects, population reliability and validity coefficients, and the linear
algebra guards (positive-definiteness test, Higham repair, condition number).

The compiler leaves every latent variable and every observed item with unit
total variance, so ``Phi`` is a latent *correlation* matrix and ``Sigma`` is an
observed *correlation* matrix.  Nothing in this module changes that: the
functions here only read the compiled matrices.

Notation follows the LISREL "all-y" convention used throughout the engine::

    eta   = B eta + zeta          Phi   = (I - B)^-1 Psi (I - B)^-T
    y     = Lambda eta + eps      Sigma = Lambda Phi Lambda' + Theta

References
----------
Joreskog, K. G., & Sorbom, D. (1996). *LISREL 8: User's Reference Guide*.
    Scientific Software International.
Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
"""

from __future__ import annotations

import warnings as _warnings
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import numpy as np

from semsim.core.errors import SpecificationError

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from semsim.core.compile import CompiledModel


__all__ = [
    "latent_covariance", "implied_covariance", "implied_correlation",
    "total_effects", "direct_effect", "indirect_effect",
    "all_indirect_effects", "population_reliability", "population_htmt",
    "is_positive_definite", "nearest_positive_definite", "condition_number",
]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _symmetrise(m: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (m + m.T)`` as a fresh float array."""
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise SpecificationError(
            "A square matrix is required.",
            "المطلوب مصفوفة مربعة الشكل.",
            shape=getattr(a, "shape", None),
        )
    return 0.5 * (a + a.T)


def _min_eigenvalue(m: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric part of ``m`` (``-inf`` on failure)."""
    try:
        return float(np.linalg.eigvalsh(_symmetrise(m)).min())
    except np.linalg.LinAlgError:
        return float("-inf")


def _resolve_factor(cm: "CompiledModel", ref: str) -> int:
    """Map a factor id (or, as a convenience, a factor name) to its index."""
    index = getattr(cm, "factor_index", {}) or {}
    if ref in index:
        return int(index[ref])
    names = list(getattr(cm, "factor_names", []) or [])
    if ref in names:
        return names.index(ref)
    raise SpecificationError(
        f"No latent variable with id or name {ref!r} exists in the compiled model.",
        f"لا يوجد متغير كامن بالمعرّف أو الاسم «{ref}» في النموذج المُصرَّف.",
        reference=ref,
    )


def _inverse_i_minus_b(B: np.ndarray) -> np.ndarray:
    """``(I - B)^-1``, raising a user-facing error instead of ``LinAlgError``."""
    B = np.asarray(B, dtype=float)
    eye = np.eye(B.shape[0])
    try:
        return np.linalg.inv(eye - B)
    except np.linalg.LinAlgError as exc:  # non-invertible feedback loop
        raise SpecificationError(
            "The structural coefficient matrix (I - B) is singular, so total "
            "effects cannot be computed. Reduce the coefficients inside the "
            "feedback loop or respecify the model as recursive.",
            "مصفوفة المسارات (I - B) غير قابلة للعكس، لذلك لا يمكن حساب الآثار "
            "الكلية. قلّل معاملات المسارات داخل الحلقة التغذية الراجعة أو أعد "
            "صياغة النموذج ليكون تعاقبيًا.",
            detail=str(exc),
        ) from exc


def _factor_items(cm: "CompiledModel") -> Dict[int, List[int]]:
    """Items grouped by the index of their home (primary) factor."""
    out: Dict[int, List[int]] = {}
    for j, fi in enumerate(getattr(cm, "home_factor", []) or []):
        out.setdefault(int(fi), []).append(j)
    return out


def _second_order_children(cm: "CompiledModel") -> Dict[int, List[int]]:
    """Dimension indices grouped by the index of their parent construct."""
    out: Dict[int, List[int]] = {}
    spec = getattr(cm, "spec", None)
    if spec is None:
        return out
    index = getattr(cm, "factor_index", {}) or {}
    for c in getattr(spec, "constructs", []) or []:
        if not getattr(c, "is_second_order", False):
            continue
        parent = index.get(getattr(c, "id", None))
        if parent is None:
            continue
        kids = [index[d.id] for d in getattr(c, "dimensions", []) if d.id in index]
        if kids:
            out[int(parent)] = kids
    return out


def _mean_offdiag(block: np.ndarray) -> float:
    """Mean of the strictly upper-triangular entries of a square block."""
    k = block.shape[0]
    if k < 2:
        return float("nan")
    iu = np.triu_indices(k, k=1)
    return float(np.mean(block[iu]))


# --------------------------------------------------------------------------- #
# Implied moments
# --------------------------------------------------------------------------- #


def latent_covariance(cm: "CompiledModel") -> np.ndarray:
    """Model-implied latent covariance matrix ``Phi``.

    Because the compiler works in the completely standardised metric, ``Phi``
    has a unit diagonal and therefore *is* the latent correlation matrix.

    Parameters
    ----------
    cm : CompiledModel
        A model produced by :func:`semsim.core.compile.compile_model`.

    Returns
    -------
    numpy.ndarray
        ``(m, m)`` symmetric matrix, ``Phi = (I - B)^-1 Psi (I - B)^-T``.

    References
    ----------
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*,
        chapter 4. Wiley.
    """
    return _symmetrise(cm.Phi)


def implied_covariance(cm: "CompiledModel") -> np.ndarray:
    """Model-implied covariance matrix of the observed items.

    Evaluates ``Sigma(theta) = Lambda Phi Lambda' + Theta``.  Every item has
    unit total variance in the compiled metric, so ``diag(Sigma) == 1`` and
    ``Sigma`` is already a correlation matrix.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` symmetric matrix in dataset column order.

    References
    ----------
    Joreskog, K. G. (1970). A general method for analysis of covariance
        structures. *Biometrika*, 57(2), 239-251.
    """
    Lambda = np.asarray(cm.Lambda, dtype=float)
    Phi = _symmetrise(cm.Phi)
    Theta = _symmetrise(cm.Theta)
    return _symmetrise(Lambda @ Phi @ Lambda.T + Theta)


def implied_correlation(cm: "CompiledModel") -> np.ndarray:
    """Model-implied correlation matrix of the observed items.

    Identical to :func:`implied_covariance` up to floating point for a
    correctly compiled model, but rescaled explicitly so the result is a true
    correlation matrix even after a positive-definiteness repair.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.

    Returns
    -------
    numpy.ndarray
        ``(p, p)`` correlation matrix with an exact unit diagonal.

    References
    ----------
    Joreskog, K. G. (1970). A general method for analysis of covariance
        structures. *Biometrika*, 57(2), 239-251.
    """
    sigma = implied_covariance(cm)
    d = np.sqrt(np.clip(np.diag(sigma), 1e-300, None))
    corr = sigma / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return _symmetrise(corr)


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #


def total_effects(cm: "CompiledModel") -> np.ndarray:
    """Total structural effects between every ordered pair of latents.

    ``T = (I - B)^-1 - I``, so ``T[i, j]`` is the total (direct plus all
    indirect) standardised effect **of** latent ``j`` **on** latent ``i`` --
    the same orientation as ``B``.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.

    Returns
    -------
    numpy.ndarray
        ``(m, m)`` matrix of total effects with a zero diagonal for a
        recursive model.

    References
    ----------
    Fox, J. (1980). Effect analysis in structural equation models.
        *Sociological Methods & Research*, 9(1), 3-28.
    Bollen, K. A. (1987). Total, direct, and indirect effects in structural
        equation models. *Sociological Methodology*, 17, 37-69.
    """
    B = np.asarray(cm.B, dtype=float)
    return _inverse_i_minus_b(B) - np.eye(B.shape[0])


def direct_effect(cm: "CompiledModel", source_id: str, target_id: str) -> float:
    """Standardised direct effect of ``source_id`` on ``target_id``.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.
    source_id : str
        Id (or name) of the predictor latent variable.
    target_id : str
        Id (or name) of the outcome latent variable.

    Returns
    -------
    float
        ``B[target, source]``; ``0.0`` when no such path exists.

    References
    ----------
    Bollen, K. A. (1987). Total, direct, and indirect effects in structural
        equation models. *Sociological Methodology*, 17, 37-69.
    """
    s = _resolve_factor(cm, source_id)
    t = _resolve_factor(cm, target_id)
    return float(np.asarray(cm.B, dtype=float)[t, s])


def indirect_effect(cm: "CompiledModel", chain: Sequence[str]) -> float:
    """Specific indirect effect along one mediation chain.

    The classical product-of-coefficients estimator: the effect of a chain
    ``X -> M -> Y`` is ``b_MX * b_YM``.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.
    chain : Sequence[str]
        Ordered latent ids (or names), from the antecedent to the outcome;
        at least three entries.

    Returns
    -------
    float
        Product of the direct effects along the chain.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If the chain has fewer than three links, repeats a latent variable, or
        contains a step that is not an actual path in the model.

    References
    ----------
    Sobel, M. E. (1982). Asymptotic confidence intervals for indirect effects
        in structural equation models. *Sociological Methodology*, 13, 290-312.
    Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
        distinction in social psychological research. *Journal of Personality
        and Social Psychology*, 51(6), 1173-1182.
    """
    links = list(chain)
    if len(links) < 3:
        raise SpecificationError(
            "A mediation chain needs at least three variables "
            "(antecedent, mediator, outcome).",
            "سلسلة الوساطة تحتاج إلى ثلاثة متغيرات على الأقل "
            "(المتغير المستقل، والوسيط، والتابع).",
            chain=links,
        )
    idx = [_resolve_factor(cm, ref) for ref in links]
    if len(set(idx)) != len(idx):
        raise SpecificationError(
            "A mediation chain may not visit the same latent variable twice.",
            "لا يجوز أن تمرّ سلسلة الوساطة بالمتغير الكامن نفسه أكثر من مرة.",
            chain=links,
        )
    B = np.asarray(cm.B, dtype=float)
    product = 1.0
    for a, b in zip(idx[:-1], idx[1:]):
        coef = float(B[b, a])
        if coef == 0.0:
            raise SpecificationError(
                f"There is no direct path from {links[idx.index(a)]!r} to "
                f"{links[idx.index(b)]!r}, so this indirect effect is undefined.",
                f"لا يوجد مسار مباشر من «{links[idx.index(a)]}» إلى "
                f"«{links[idx.index(b)]}»، لذلك لا يمكن حساب الأثر غير المباشر "
                "لهذه السلسلة.",
                chain=links,
            )
        product *= coef
    return float(product)


def all_indirect_effects(
    cm: "CompiledModel",
    source_id: str,
    target_id: str,
    max_len: int = 4,
) -> List[Tuple[List[str], float]]:
    """Enumerate every specific indirect effect from ``source_id`` to ``target_id``.

    Walks the directed graph encoded in ``B`` and returns one entry per simple
    path that passes through at least one mediator.  The direct path (a chain
    of two nodes) is deliberately excluded.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.
    source_id : str
        Id (or name) of the antecedent latent variable.
    target_id : str
        Id (or name) of the outcome latent variable.
    max_len : int, default 4
        Maximum number of *nodes* in a chain, so ``4`` allows up to two
        mediators.

    Returns
    -------
    list of (list of str, float)
        ``(chain_of_factor_ids, product_of_coefficients)`` sorted by
        descending absolute effect.

    References
    ----------
    Bollen, K. A. (1987). Total, direct, and indirect effects in structural
        equation models. *Sociological Methodology*, 17, 37-69.
    Fox, J. (1980). Effect analysis in structural equation models.
        *Sociological Methods & Research*, 9(1), 3-28.
    """
    s = _resolve_factor(cm, source_id)
    t = _resolve_factor(cm, target_id)
    B = np.asarray(cm.B, dtype=float)
    ids = list(cm.factor_ids)
    limit = max(int(max_len), 0)
    found: List[Tuple[List[str], float]] = []

    def walk(node: int, visited: List[int], coefficient: float) -> None:
        if len(visited) >= limit:
            return
        for nxt in np.flatnonzero(B[:, node] != 0.0):
            nxt = int(nxt)
            if nxt in visited:
                continue
            value = coefficient * float(B[nxt, node])
            path = visited + [nxt]
            if nxt == t:
                if len(path) >= 3:
                    found.append(([ids[k] for k in path], float(value)))
                continue
            walk(nxt, path, value)

    walk(s, [s], 1.0)
    found.sort(key=lambda pair: (-abs(pair[1]), len(pair[0])))
    return found


# --------------------------------------------------------------------------- #
# Population reliability / validity
# --------------------------------------------------------------------------- #


def _reliability_block(
    loadings: np.ndarray,
    errors: np.ndarray,
    corr_block: np.ndarray,
    error_block: np.ndarray,
) -> Dict[str, float]:
    """Shared arithmetic for item-based and dimension-based reliability."""
    k = int(loadings.size)
    sum_l = float(np.sum(loadings))
    sum_l_sq = sum_l ** 2
    sum_err = float(np.sum(errors))
    cr = sum_l_sq / (sum_l_sq + sum_err) if (sum_l_sq + sum_err) > 0 else float("nan")
    ave = float(np.mean(loadings ** 2)) if k else float("nan")

    if k >= 2:
        total = float(np.sum(corr_block))
        trace = float(np.trace(corr_block))
        alpha = (k / (k - 1.0)) * (1.0 - trace / total) if total > 0 else float("nan")
        err_total = float(np.sum(error_block))
        omega = (sum_l_sq / (sum_l_sq + err_total)
                 if (sum_l_sq + err_total) > 0 else float("nan"))
    else:
        alpha = float("nan")
        omega = cr

    return {
        "ave": ave,
        "cr": float(cr),
        "alpha_implied": float(alpha),
        "omega": float(omega),
        "n_items": float(k),
    }


def population_reliability(cm: "CompiledModel") -> Dict[str, Dict[str, float]]:
    """Population reliability and convergent validity for every factor.

    For a first-order factor (or a dimension of a second-order construct) the
    indicators are its items, with loadings taken from the compiled ``Lambda``
    and error variances from ``diag(Theta)``.  For a second-order construct the
    indicators are its dimensions, with loadings from ``B`` and disturbance
    variances from ``diag(Psi)``.

    Because the model is completely standardised:

    * ``ave = mean(lambda ** 2)``;
    * ``cr  = (sum lambda) ** 2 / ((sum lambda) ** 2 + sum theta)``;
    * ``alpha_implied`` is Cronbach's alpha evaluated on the *implied*
      correlation block, i.e. the population value the data will converge on;
    * ``omega`` is McDonald's total omega, which additionally charges the
      correlated residuals to the error term.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.

    Returns
    -------
    dict
        ``{factor_id: {"ave", "cr", "alpha_implied", "omega", "n_items"}}``.
        The common-method factor is excluded; it is a nuisance factor, not a
        construct the researcher reports.

    References
    ----------
    Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation models
        with unobservable variables and measurement error. *Journal of
        Marketing Research*, 18(1), 39-50.
    Joreskog, K. G. (1971). Statistical analysis of sets of congeneric tests.
        *Psychometrika*, 36(2), 109-133.
    Cronbach, L. J. (1951). Coefficient alpha and the internal structure of
        tests. *Psychometrika*, 16(3), 297-334.
    McDonald, R. P. (1999). *Test Theory: A Unified Treatment*. Erlbaum.
    """
    Lambda = np.asarray(cm.Lambda, dtype=float)
    Theta = _symmetrise(cm.Theta)
    Psi = _symmetrise(cm.Psi)
    Phi = _symmetrise(cm.Phi)
    sigma = implied_covariance(cm)
    method = getattr(cm, "method_factor_index", None)

    out: Dict[str, Dict[str, float]] = {}

    for fi, rows in sorted(_factor_items(cm).items()):
        if method is not None and fi == method:
            continue
        rows_arr = np.asarray(rows, dtype=int)
        loadings = Lambda[rows_arr, fi]
        errors = np.diag(Theta)[rows_arr]
        block = sigma[np.ix_(rows_arr, rows_arr)]
        err_block = Theta[np.ix_(rows_arr, rows_arr)]
        out[cm.factor_ids[fi]] = _reliability_block(
            loadings, errors, block, err_block)

    for parent, kids in sorted(_second_order_children(cm).items()):
        kids_arr = np.asarray(kids, dtype=int)
        loadings = np.asarray(cm.B, dtype=float)[kids_arr, parent]
        errors = np.diag(Psi)[kids_arr]
        block = Phi[np.ix_(kids_arr, kids_arr)]
        err_block = Psi[np.ix_(kids_arr, kids_arr)]
        out[cm.factor_ids[parent]] = _reliability_block(
            loadings, errors, block, err_block)

    return out


def population_htmt(cm: "CompiledModel") -> Tuple[np.ndarray, List[str]]:
    """Population heterotrait-monotrait ratio of correlations.

    ``HTMT[i, j]`` is the mean of the between-factor item correlations divided
    by the geometric mean of the two within-factor mean item correlations, all
    evaluated on the model-implied correlation matrix.  Factors measured by
    fewer than two items have no monotrait average and are omitted.

    A pair whose monotrait mean is not positive on *either* side gets ``nan``,
    with the reason appended to ``cm.warnings``
    instead of a number, and the reason is recorded as a ``RuntimeWarning``
    naming the offending factors; the ratio has no meaning there, and a factor
    built mostly from reverse-worded items is the common way to land in it.

    Parameters
    ----------
    cm : CompiledModel
        A compiled model.

    Returns
    -------
    (numpy.ndarray, list of str)
        The ``(f, f)`` HTMT matrix (unit diagonal) and the factor ids it is
        indexed by, in compiled factor order.

    References
    ----------
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
        assessing discriminant validity in variance-based structural equation
        modeling. *Journal of the Academy of Marketing Science*, 43(1),
        115-135.
    """
    corr = implied_correlation(cm)
    method = getattr(cm, "method_factor_index", None)
    groups = [(fi, rows) for fi, rows in sorted(_factor_items(cm).items())
              if len(rows) >= 2 and not (method is not None and fi == method)]
    ids = [cm.factor_ids[fi] for fi, _ in groups]
    f = len(groups)
    htmt = np.ones((f, f), dtype=float)
    mono = [_mean_offdiag(corr[np.ix_(rows, rows)]) for _, rows in groups]
    for a in range(f):
        for b in range(a + 1, f):
            rows_a = groups[a][1]
            rows_b = groups[b][1]
            hetero = float(np.mean(corr[np.ix_(rows_a, rows_b)]))
            # Each monotrait mean must be positive *on its own*. Guarding only
            # the product ``mono[a] * mono[b]`` passes when BOTH means are
            # negative, and the geometric mean of two negative averages is not
            # a monotrait reliability of anything: the ratio it produces is
            # arbitrary in both sign and size. That case is reachable, not
            # hypothetical — the compiler negates the loading row of every
            # reverse-worded item (the REVERSE_CODED block in compile.py), so a
            # factor whose items are mostly reverse-worded has a negative mean
            # within-factor correlation until the researcher recodes them.
            # NaN is the honest answer, with the reason recorded as a warning
            # so the caller can report the missing recode rather than a number.
            offending = [i for i in (a, b) if not mono[i] > 0.0]
            if offending:
                htmt[a, b] = htmt[b, a] = float("nan")
                named = ", ".join(f"{ids[i]} ({mono[i]:.4f})" for i in offending)
                named_ar = "، ".join(f"«{ids[i]}» ({mono[i]:.4f})"
                                     for i in offending)
                # Recorded on the compiled model rather than raised as a
                # Python warning. This is a documented, expected outcome for
                # a model with reverse-worded items, not an exceptional
                # condition: raising RuntimeWarning printed to stderr behind
                # a desktop window where nobody sees it, and turned into an
                # error under the project's filterwarnings setting. Every
                # other diagnostic in the engine travels through this list.
                note = (
                    "HTMT_MONOTRAIT_NOT_POSITIVE: the HTMT ratio between "
                    f"{ids[a]!r} and {ids[b]!r} is undefined because the mean "
                    "within-factor (monotrait) correlation is not positive "
                    f"for {named}; NaN was returned. The usual cause is "
                    "reverse-worded items that have not been recoded. | "
                    "نسبة HTMT غير معرَّفة: تعذّر حساب النسبة بين العاملين "
                    f"«{ids[a]}» و«{ids[b]}» لأن متوسط الارتباطات الداخلية "
                    f"(أحادية السمة) غير موجب لدى {named_ar}؛ وقد أُعيدت القيمة "
                    "NaN لأن الجذر التربيعي لحاصل ضرب متوسطين سالبين لا يحمل "
                    "أي معنى قياسي. والسبب الأغلب هو وجود بنود معكوسة الصياغة "
                    "لم يُعَد ترميزها.")
                recorded = getattr(cm, "warnings", None)
                if isinstance(recorded, list) and note not in recorded:
                    recorded.append(note)
                continue
            htmt[a, b] = htmt[b, a] = hetero / float(np.sqrt(mono[a] * mono[b]))
    return htmt, ids


# --------------------------------------------------------------------------- #
# Linear-algebra guards
# --------------------------------------------------------------------------- #


def is_positive_definite(m: np.ndarray, tol: float = 1e-10) -> bool:
    """Whether ``m`` is symmetric positive definite.

    Parameters
    ----------
    m : numpy.ndarray
        Square matrix.
    tol : float, default 1e-10
        Minimum eigenvalue required; also the tolerance on symmetry.

    Returns
    -------
    bool
        ``True`` when ``m`` is symmetric to within ``tol`` and its smallest
        eigenvalue exceeds ``tol``.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.),
        section 4.2. Johns Hopkins University Press.
    """
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0:
        return False
    if not np.all(np.isfinite(a)):
        return False
    if not np.allclose(a, a.T, atol=max(tol, 1e-12), rtol=0.0):
        return False
    return _min_eigenvalue(a) > tol


def nearest_positive_definite(m: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Nearest positive-definite matrix with the original diagonal preserved.

    Implements Higham's (2002) alternating-projections algorithm with Dykstra's
    correction: alternately project onto the cone of positive semi-definite
    matrices (eigenvalue clipping at ``eps``) and onto the affine set of
    matrices whose diagonal equals ``diag(m)``.  A final eigenvalue clip and a
    shrink of the off-diagonal entries guarantee strict positive definiteness
    while keeping the diagonal exact, which is what the compiler needs so that
    every item keeps unit total variance.

    The function is idempotent: an input that is already positive definite is
    returned unchanged apart from symmetrisation.

    Parameters
    ----------
    m : numpy.ndarray
        Square, (approximately) symmetric matrix.
    eps : float, default 1e-8
        Floor imposed on the eigenvalues of the result.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-definite matrix of the same shape.

    Raises
    ------
    semsim.core.errors.SpecificationError
        If ``m`` is not square or contains non-finite entries.

    References
    ----------
    Higham, N. J. (2002). Computing the nearest correlation matrix -- a problem
        from finance. *IMA Journal of Numerical Analysis*, 22(3), 329-343.
    Dykstra, R. L. (1983). An algorithm for restricted least squares
        regression. *Journal of the American Statistical Association*, 78(384),
        837-842.
    """
    a = _symmetrise(m)
    if not np.all(np.isfinite(a)):
        raise SpecificationError(
            "The matrix to repair contains non-finite entries.",
            "المصفوفة المطلوب إصلاحها تحتوي على قيم غير منتهية (NaN أو ما لا نهاية).",
        )
    if is_positive_definite(a, tol=eps):
        return a

    target_diag = np.diag(a).copy()
    target_diag = np.where(target_diag > eps, target_diag, eps)

    y = a.copy()
    delta_s = np.zeros_like(a)
    for _ in range(200):
        r = y - delta_s
        vals, vecs = np.linalg.eigh(_symmetrise(r))
        x = _symmetrise((vecs * np.clip(vals, eps, None)) @ vecs.T)
        delta_s = x - r
        y = x.copy()
        np.fill_diagonal(y, target_diag)
        if np.linalg.norm(y - x, ord="fro") <= 1e-12 * max(
                np.linalg.norm(y, ord="fro"), 1.0):
            break

    y = _symmetrise(y)
    np.fill_diagonal(y, target_diag)

    if is_positive_definite(y, tol=eps):
        return y

    # Final guarantee: shrink the off-diagonal block towards the diagonal.
    off = y - np.diag(np.diag(y))
    diag_only = np.diag(target_diag)
    lo, hi = 0.0, 1.0
    best = diag_only
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cand = _symmetrise(diag_only + (1.0 - mid) * off)
        if is_positive_definite(cand, tol=eps):
            best = cand
            hi = mid
        else:
            lo = mid
    return best


def condition_number(m: np.ndarray) -> float:
    """Spectral condition number of ``m``.

    Parameters
    ----------
    m : numpy.ndarray
        Square matrix.

    Returns
    -------
    float
        ``|lambda_max| / |lambda_min|``, or ``inf`` for a singular matrix.
        Singularity is judged with numpy's own relative tolerance,
        ``max(shape) * eps * |lambda_max|``, so a matrix that is singular only
        up to floating-point noise is still reported as singular.
        Values above roughly ``1e4`` signal an empirically under-identified or
        near-collinear covariance structure.

    References
    ----------
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.),
        section 2.6.2. Johns Hopkins University Press.
    """
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0:
        raise SpecificationError(
            "A square matrix is required to compute a condition number.",
            "حساب رقم الحالة يتطلب مصفوفة مربعة الشكل.",
            shape=getattr(a, "shape", None),
        )
    if not np.all(np.isfinite(a)):
        return float("inf")
    try:
        vals = np.abs(np.linalg.eigvalsh(_symmetrise(a)))
    except np.linalg.LinAlgError:
        return float("inf")
    largest = float(vals.max())
    smallest = float(vals.min())
    # A *numerically* singular matrix almost never produces an eigenvalue of
    # exactly zero: LAPACK returns a tiny value of either sign, so testing
    # ``smallest <= 0.0`` misses it half the time. np.outer([1, 2, 3], [1, 2, 3])
    # is exactly rank one, yet its zero eigenvalue comes back as +1.9e-16 and
    # the old test let it through, reporting a finite 7.36e16 for a matrix with
    # no inverse. Compare against numpy's own default singularity tolerance
    # (``max(shape) * eps * largest``, as used by matrix_rank and pinv) so that
    # "singular" means the same thing here as everywhere else in the stack.
    tol = max(a.shape) * float(np.finfo(float).eps) * largest
    if smallest <= tol:
        return float("inf")
    return float(largest / smallest)

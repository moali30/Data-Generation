"""
semsim.generation.fleishman
===========================
Fleishman's (1978) power-method transformation of a standard normal deviate
into a variable with a prescribed skewness and (excess) kurtosis, plus the
Vale & Maurelli (1983) *intermediate correlation* cubic that makes the
transformation usable inside a multivariate generator.

The transformation is

.. math::

    Y = a + b Z + c Z^2 + d Z^3, \\qquad Z \\sim N(0, 1), \\qquad a = -c,

and ``a = -c`` is exactly the choice that makes :math:`E[Y] = 0`, because
:math:`E[Z^2] = 1`.  Requiring :math:`Var(Y) = 1` and matching the third and
fourth standardised moments gives Fleishman's three equations in
:math:`(b, c, d)`

.. math::

    b^2 + 2c^2 + 6bd + 15d^2 &= 1 \\\\
    2c\\,(b^2 + 24bd + 105d^2 + 2) &= \\gamma_1 \\\\
    24\\,[\\,bd + c^2(1 + b^2 + 28bd)
        + d^2(12 + 48bd + 141c^2 + 225d^2)\\,] &= \\gamma_2

with :math:`\\gamma_1` the skewness and :math:`\\gamma_2` the *excess*
kurtosis (normal :math:`\\Rightarrow 0`).  The system is solved by
Newton-Raphson with the analytic Jacobian.

Two structural facts about the system are used for robustness:

*   Negating ``c`` alone flips the sign of the skewness and leaves both the
    variance and the kurtosis untouched, so every target is solved for
    ``|gamma_1|`` and the sign is restored afterwards.  This keeps ``b > 0``
    and removes one spurious branch from the search.
*   ``gamma_1 == 0`` forces ``c == 0`` (the skewness equation's bracket is
    ``b^2 + 24bd + 105d^2 + 2 >= 2 - 144/105 > 0``), so symmetric targets are
    solved on the two-dimensional sub-system and ``c`` comes out exactly zero.

References
----------
Fleishman, A. I. (1978). A method for simulating non-normal distributions.
    *Psychometrika*, 43(4), 521-532.
Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
    distributions. *Psychometrika*, 48(3), 465-471.
Headrick, T. C. (2010). *Statistical Simulation: Power Method Polynomials and
    Other Transformations*. Chapman & Hall/CRC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from semsim.core.errors import ConvergenceError, InfeasibleTargetError

__all__ = [
    "FLEISHMAN_LOWER_KURTOSIS_BOUND",
    "FLEISHMAN_PRACTICAL_BOUND",
    "FleishmanCoefficients",
    "is_feasible",
    "is_practically_feasible",
    "clamp_to_feasible",
    "solve",
    "transform",
    "moment_residuals",
    "intermediate_correlation",
]


#: Nominal lower bound on the excess kurtosis for a given skewness.  Outside
#: it Fleishman's system has no real solution at all; the *practical* boundary
#: of the power method is a little higher (see
#: :data:`FLEISHMAN_PRACTICAL_BOUND`).
FLEISHMAN_LOWER_KURTOSIS_BOUND = lambda skew: skew ** 2 - 2  # noqa: E731

#: Headrick's (2010) empirical description of the boundary of the region the
#: third-order power method can actually reach.  Used only for diagnostics and
#: for :func:`clamp_to_feasible`, never to reject a target outright.
FLEISHMAN_PRACTICAL_BOUND = lambda skew: -1.2264489 + 1.6410373 * skew ** 2  # noqa: E731


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class FleishmanCoefficients:
    """Solution of Fleishman's system for one variable.

    Attributes
    ----------
    a, b, c, d : float
        Polynomial coefficients of ``Y = a + b Z + c Z**2 + d Z**3``, with
        ``a == -c`` so that ``E[Y] == 0`` and ``Var(Y) == 1``.
    skewness : float
        The requested skewness (third standardised moment).
    kurtosis : float
        The requested *excess* kurtosis (fourth standardised moment minus 3).
    converged : bool
        ``True`` when the final residual is below the requested tolerance.
    iterations : int
        Newton-Raphson iterations actually spent (0 for the normal
        short-circuit).
    residual : float
        Infinity norm of the three moment equations at the solution.
    """

    a: float
    b: float
    c: float
    d: float
    skewness: float
    kurtosis: float
    converged: bool
    iterations: int
    residual: float

    # -- convenience ------------------------------------------------------- #

    @property
    def is_identity(self) -> bool:
        """``True`` when the polynomial is the identity (a normal variable)."""
        return (self.a, self.b, self.c, self.d) == (0.0, 1.0, 0.0, 0.0)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return ``(a, b, c, d)``."""
        return (self.a, self.b, self.c, self.d)


# --------------------------------------------------------------------------- #
# Feasibility
# --------------------------------------------------------------------------- #


def is_feasible(skewness: float, kurtosis: float) -> bool:
    """Is the (skewness, excess kurtosis) pair inside Fleishman's region?

    Applies the necessary condition shared by every standardised distribution,
    ``kurtosis > skewness**2 - 2``.  It is necessary but not quite sufficient:
    the third-order power method reaches a slightly smaller region, described
    by :func:`is_practically_feasible`.

    Parameters
    ----------
    skewness : float
        Target third standardised moment.
    kurtosis : float
        Target *excess* kurtosis.

    Returns
    -------
    bool
        ``True`` when a real solution is not ruled out a priori.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    s = float(skewness)
    k = float(kurtosis)
    if not (np.isfinite(s) and np.isfinite(k)):
        return False
    return k > FLEISHMAN_LOWER_KURTOSIS_BOUND(s)


def is_practically_feasible(skewness: float, kurtosis: float) -> bool:
    """Is the pair inside the region the cubic power method can actually hit?

    Parameters
    ----------
    skewness : float
        Target third standardised moment.
    kurtosis : float
        Target *excess* kurtosis.

    Returns
    -------
    bool
        ``True`` when the pair lies above Headrick's empirical boundary.

    References
    ----------
    Headrick, T. C. (2010). *Statistical Simulation: Power Method Polynomials
        and Other Transformations*. Chapman & Hall/CRC, chapter 2.
    """
    s = float(skewness)
    k = float(kurtosis)
    if not (np.isfinite(s) and np.isfinite(k)):
        return False
    return k > FLEISHMAN_PRACTICAL_BOUND(s)


def clamp_to_feasible(skewness: float, kurtosis: float,
                      *, margin: float = 0.05) -> Tuple[float, float]:
    """Move an infeasible target just inside the reachable region.

    The skewness is kept; the kurtosis is raised to ``margin`` above the
    boundary, which is the behaviour requested by
    :attr:`semsim.core.spec.DistributionSpec.clamp_to_feasible_region`.

    Parameters
    ----------
    skewness : float
        Target third standardised moment; returned unchanged (clipped to a
        finite range).
    kurtosis : float
        Target *excess* kurtosis.
    margin : float, default 0.05
        Distance kept above the boundary.

    Returns
    -------
    tuple of (float, float)
        The clamped ``(skewness, kurtosis)``.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    s = float(skewness)
    k = float(kurtosis)
    if not np.isfinite(s):
        s = 0.0
    if not np.isfinite(k):
        k = 0.0
    floor = max(FLEISHMAN_LOWER_KURTOSIS_BOUND(s),
                FLEISHMAN_PRACTICAL_BOUND(s)) + float(margin)
    if k < floor:
        k = floor
    return s, k


# --------------------------------------------------------------------------- #
# The moment equations and their Jacobian
# --------------------------------------------------------------------------- #


def _equations(b: float, c: float, d: float,
               skewness: float, kurtosis: float) -> np.ndarray:
    """Fleishman's three equations, written as residuals (target subtracted)."""
    bd = b * d
    c2 = c * c
    d2 = d * d
    f1 = b * b + 2.0 * c2 + 6.0 * bd + 15.0 * d2 - 1.0
    f2 = 2.0 * c * (b * b + 24.0 * bd + 105.0 * d2 + 2.0) - skewness
    f3 = 24.0 * (bd
                 + c2 * (1.0 + b * b + 28.0 * bd)
                 + d2 * (12.0 + 48.0 * bd + 141.0 * c2 + 225.0 * d2)) - kurtosis
    return np.array([f1, f2, f3], dtype=float)


def _jacobian(b: float, c: float, d: float) -> np.ndarray:
    """Analytic Jacobian ``d(f1, f2, f3) / d(b, c, d)`` of :func:`_equations`."""
    c2 = c * c
    d2 = d * d
    d3 = d2 * d
    bd = b * d

    j = np.empty((3, 3), dtype=float)

    # f1 = b^2 + 2c^2 + 6bd + 15d^2 - 1
    j[0, 0] = 2.0 * b + 6.0 * d
    j[0, 1] = 4.0 * c
    j[0, 2] = 6.0 * b + 30.0 * d

    # f2 = 2c (b^2 + 24bd + 105d^2 + 2) - skew
    j[1, 0] = 2.0 * c * (2.0 * b + 24.0 * d)
    j[1, 1] = 2.0 * (b * b + 24.0 * bd + 105.0 * d2 + 2.0)
    j[1, 2] = 2.0 * c * (24.0 * b + 210.0 * d)

    # f3 = 24 [ bd + c^2 (1 + b^2 + 28bd) + d^2 (12 + 48bd + 141c^2 + 225d^2) ]
    j[2, 0] = 24.0 * (d + 2.0 * b * c2 + 28.0 * c2 * d + 48.0 * d3)
    j[2, 1] = 24.0 * (2.0 * c * (1.0 + b * b + 28.0 * bd) + 282.0 * c * d2)
    j[2, 2] = 24.0 * (b + 28.0 * b * c2 + 24.0 * d
                      + 144.0 * b * d2 + 282.0 * c2 * d + 900.0 * d3)
    return j


def moment_residuals(coef: "FleishmanCoefficients | Sequence[float]",
                     skewness: float, kurtosis: float) -> np.ndarray:
    """Substitute coefficients back into Fleishman's three equations.

    This is the definitive correctness check on a solution: all three entries
    must be zero for the polynomial to deliver unit variance and the requested
    third and fourth standardised moments.

    Parameters
    ----------
    coef : FleishmanCoefficients or sequence of float
        Either a solved :class:`FleishmanCoefficients` or a raw ``(a, b, c, d)``
        / ``(b, c, d)`` sequence.
    skewness : float
        Target skewness.
    kurtosis : float
        Target *excess* kurtosis.

    Returns
    -------
    numpy.ndarray
        Length-3 array ``[variance_eq, skewness_eq, kurtosis_eq]`` of residuals.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    if isinstance(coef, FleishmanCoefficients):
        b, c, d = coef.b, coef.c, coef.d
    else:
        vals = [float(v) for v in coef]
        if len(vals) == 4:
            _, b, c, d = vals
        elif len(vals) == 3:
            b, c, d = vals
        else:
            raise InfeasibleTargetError(
                "Fleishman coefficients must be given as (a, b, c, d) or (b, c, d).",
                "يجب تمرير معاملات فليشمان على صورة (a, b, c, d) أو (b, c, d).",
                n_values=len(vals),
            )
    return _equations(b, c, d, float(skewness), float(kurtosis))


# --------------------------------------------------------------------------- #
# Newton-Raphson
# --------------------------------------------------------------------------- #


def _newton(start: Tuple[float, float, float], skewness: float, kurtosis: float,
            tol: float, max_iter: int) -> Tuple[np.ndarray, float, int]:
    """Damped Newton-Raphson on the three equations.

    Returns the final ``(b, c, d)``, the residual infinity norm and the number
    of iterations used.  Never raises: a failed attempt simply returns a large
    residual so the caller can try another starting point.
    """
    x = np.array(start, dtype=float)
    f = _equations(x[0], x[1], x[2], skewness, kurtosis)
    res = float(np.max(np.abs(f)))
    used = 0
    for it in range(1, int(max_iter) + 1):
        used = it
        if res <= tol:
            break
        jac = _jacobian(x[0], x[1], x[2])
        try:
            step = np.linalg.solve(jac, -f)
        except np.linalg.LinAlgError:
            # Singular Jacobian: fall back on the minimum-norm least-squares
            # step, which is well defined even on the fold of the surface.
            step, *_ = np.linalg.lstsq(jac, -f, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        # Backtracking line search on the residual norm keeps the iteration
        # from flying out of the region when the Jacobian is nearly singular.
        scale = 1.0
        improved = False
        for _ in range(40):
            cand = x + scale * step
            if np.all(np.isfinite(cand)):
                f_cand = _equations(cand[0], cand[1], cand[2], skewness, kurtosis)
                res_cand = float(np.max(np.abs(f_cand)))
                if np.isfinite(res_cand) and res_cand < res:
                    x, f, res = cand, f_cand, res_cand
                    improved = True
                    break
            scale *= 0.5
        if not improved:
            break
    return x, res, used


def _symmetric_equations(b: float, d: float, kurtosis: float) -> np.ndarray:
    """Variance and kurtosis equations with ``c == 0`` (symmetric targets)."""
    bd = b * d
    d2 = d * d
    return np.array([
        b * b + 6.0 * bd + 15.0 * d2 - 1.0,
        24.0 * (bd + d2 * (12.0 + 48.0 * bd + 225.0 * d2)) - kurtosis,
    ], dtype=float)


def _symmetric_jacobian(b: float, d: float) -> np.ndarray:
    d2 = d * d
    d3 = d2 * d
    return np.array([
        [2.0 * b + 6.0 * d, 6.0 * b + 30.0 * d],
        [24.0 * (d + 48.0 * d3), 24.0 * (b + 24.0 * d + 144.0 * b * d2 + 900.0 * d3)],
    ], dtype=float)


def _newton_symmetric(start: Tuple[float, float], kurtosis: float,
                      tol: float, max_iter: int) -> Tuple[np.ndarray, float, int]:
    """Damped Newton-Raphson on the two-equation symmetric sub-system."""
    x = np.array(start, dtype=float)
    f = _symmetric_equations(x[0], x[1], kurtosis)
    res = float(np.max(np.abs(f)))
    used = 0
    for it in range(1, int(max_iter) + 1):
        used = it
        if res <= tol:
            break
        jac = _symmetric_jacobian(x[0], x[1])
        try:
            step = np.linalg.solve(jac, -f)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(jac, -f, rcond=None)
        if not np.all(np.isfinite(step)):
            break
        scale = 1.0
        improved = False
        for _ in range(40):
            cand = x + scale * step
            if np.all(np.isfinite(cand)):
                f_cand = _symmetric_equations(cand[0], cand[1], kurtosis)
                res_cand = float(np.max(np.abs(f_cand)))
                if np.isfinite(res_cand) and res_cand < res:
                    x, f, res = cand, f_cand, res_cand
                    improved = True
                    break
            scale *= 0.5
        if not improved:
            break
    return x, res, used


def _starting_points(skewness: float, kurtosis: float
                     ) -> list[Tuple[float, float, float]]:
    """Fleishman's own heuristic start, followed by progressive fallbacks."""
    c0 = skewness / 6.0
    starts: list[Tuple[float, float, float]] = [
        (1.0, c0, 0.0),                      # the published starting heuristic
        (1.0, c0, kurtosis / 200.0),
        (0.95, c0 * 0.8, 0.01),
        (0.85, c0 * 0.6, 0.04),
        (0.70, c0 * 0.5, 0.08),
        (0.50, c0 * 0.4, 0.14),
        (0.35, c0 * 0.3, 0.19),
        (0.25, c0 * 0.2, 0.23),
        (1.05, c0 * 1.2, -0.02),
        (1.10, 0.0, -0.05),
    ]
    return starts


def solve(skewness: float, kurtosis: float, *,
          tol: float = 1e-10, max_iter: int = 500,
          clamp: bool = False) -> FleishmanCoefficients:
    """Solve Fleishman's system for one variable.

    Parameters
    ----------
    skewness : float
        Target skewness (third standardised moment) of the generated variable.
    kurtosis : float
        Target *excess* kurtosis (fourth standardised moment minus 3); the
        normal distribution is ``0``.
    tol : float, default 1e-10
        Convergence tolerance on the infinity norm of the three residuals.
    max_iter : int, default 500
        Maximum Newton-Raphson iterations per starting point.
    clamp : bool, default False
        When ``True`` an infeasible target is moved just inside the reachable
        region instead of raising, mirroring
        :attr:`semsim.core.spec.DistributionSpec.clamp_to_feasible_region`.

    Returns
    -------
    FleishmanCoefficients
        Coefficients with ``a == -c``, ``E[Y] == 0`` and ``Var(Y) == 1``.
        ``skewness == 0`` and ``kurtosis == 0`` short-circuit to the exact
        identity ``(0, 1, 0, 0)`` without a single iteration.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        If the pair violates ``kurtosis > skewness**2 - 2`` (and ``clamp`` is
        ``False``), or if either value is not finite.
    semsim.core.errors.ConvergenceError
        If every starting point fails to reach a usable solution, which in
        practice means the target sits on or below the practical boundary of
        the cubic power method.

    Notes
    -----
    The sign symmetry ``(b, c, d) -> (b, -c, d)`` flips the skewness and leaves
    the variance and kurtosis alone, so the search always runs on
    ``abs(skewness)`` and the sign is restored at the end.  For a symmetric
    target the two-equation sub-system with ``c = 0`` is solved instead, which
    returns ``c`` as exactly zero rather than as a value near zero.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
        distributions. *Psychometrika*, 48(3), 465-471.
    """
    skew_in = float(skewness)
    kurt_in = float(kurtosis)

    if not (np.isfinite(skew_in) and np.isfinite(kurt_in)):
        raise InfeasibleTargetError(
            "Skewness and kurtosis targets must be finite numbers; "
            f"got skewness={skewness!r}, kurtosis={kurtosis!r}.",
            "يجب أن تكون قيمتا الالتواء والتفلطح أعدادًا منتهية؛ "
            f"القيم الممرَّرة هي الالتواء={skewness!r} والتفلطح={kurtosis!r}.",
            skewness=skewness, kurtosis=kurtosis,
        )

    if clamp:
        skew_in, kurt_in = clamp_to_feasible(skew_in, kurt_in)

    if not is_feasible(skew_in, kurt_in):
        bound = FLEISHMAN_LOWER_KURTOSIS_BOUND(skew_in)
        raise InfeasibleTargetError(
            f"No real Fleishman solution exists for skewness={skew_in:.4f} with "
            f"excess kurtosis={kurt_in:.4f}: the kurtosis must exceed "
            f"skewness**2 - 2 = {bound:.4f}. Raise the kurtosis or reduce the "
            "skewness.",
            f"لا يوجد حل حقيقي لمعادلات فليشمان عند التواء={skew_in:.4f} "
            f"وتفلطح={kurt_in:.4f}؛ إذ يجب أن يكون التفلطح أكبر من "
            f"(مربّع الالتواء ناقص 2) أي أكبر من {bound:.4f}. "
            "ارفع قيمة التفلطح أو خفّض قيمة الالتواء.",
            skewness=skew_in, kurtosis=kurt_in, lower_bound=bound,
        )

    # -- the normal short-circuit: exact, no iteration -------------------- #
    if skew_in == 0.0 and kurt_in == 0.0:
        return FleishmanCoefficients(
            a=0.0, b=1.0, c=0.0, d=0.0,
            skewness=0.0, kurtosis=0.0,
            converged=True, iterations=0, residual=0.0,
        )

    target_skew = abs(skew_in)
    flip = skew_in < 0.0

    best_x: np.ndarray | None = None
    best_res = np.inf
    best_iters = 0

    if target_skew == 0.0:
        # Symmetric target: c is exactly zero, solve the 2x2 sub-system.
        for b0, d0 in ((1.0, 0.0), (1.0, kurt_in / 200.0), (0.9, 0.02),
                       (0.7, 0.08), (0.5, 0.14), (0.3, 0.20), (1.1, -0.05)):
            x2, res, iters = _newton_symmetric((b0, d0), kurt_in, tol, max_iter)
            if res < best_res:
                best_res, best_iters = res, iters
                best_x = np.array([x2[0], 0.0, x2[1]], dtype=float)
            if best_res <= tol:
                break
    else:
        for start in _starting_points(target_skew, kurt_in):
            x3, res, iters = _newton(start, target_skew, kurt_in, tol, max_iter)
            if res < best_res:
                best_res, best_iters, best_x = res, iters, x3
            if best_res <= tol:
                break

        if best_res > tol:
            # Last resort: a trust-region least-squares polish from the best
            # Newton point, which copes with the nearly singular fold at the
            # edge of the feasible region.
            polished = _least_squares_polish(best_x, target_skew, kurt_in)
            if polished is not None:
                x3, res = polished
                if res < best_res:
                    best_res, best_x = res, x3

    if best_x is None or not np.all(np.isfinite(best_x)):
        raise ConvergenceError(
            f"Newton-Raphson produced no usable Fleishman solution for "
            f"skewness={skew_in:.4f}, excess kurtosis={kurt_in:.4f}.",
            f"تعذّر على طريقة نيوتن-رافسون إيجاد حل لمعادلات فليشمان عند "
            f"التواء={skew_in:.4f} وتفلطح={kurt_in:.4f}.",
            skewness=skew_in, kurtosis=kurt_in,
        )

    # A residual this large means the point is outside the region the cubic
    # power method can actually reach, even though it clears skew**2 - 2.
    if best_res > 1e-5:
        raise ConvergenceError(
            f"Fleishman's system did not converge for skewness={skew_in:.4f}, "
            f"excess kurtosis={kurt_in:.4f} (residual {best_res:.3e}). The pair "
            "lies on or below the practical boundary of the cubic power method "
            f"(kurtosis > {FLEISHMAN_PRACTICAL_BOUND(skew_in):.4f} for this "
            "skewness).",
            f"لم تتقارب معادلات فليشمان عند التواء={skew_in:.4f} "
            f"وتفلطح={kurt_in:.4f} (مقدار الخطأ المتبقّي {best_res:.3e}). "
            "القيمتان تقعان على حدود المنطقة التي يمكن لكثير حدود فليشمان "
            "من الدرجة الثالثة الوصول إليها أو خارجها؛ يلزم أن يكون التفلطح "
            f"أكبر من {FLEISHMAN_PRACTICAL_BOUND(skew_in):.4f} عند هذا "
            "الالتواء. ارفع التفلطح قليلًا أو خفّض الالتواء.",
            skewness=skew_in, kurtosis=kurt_in, residual=best_res,
        )

    b, c, d = (float(best_x[0]), float(best_x[1]), float(best_x[2]))
    if b < 0.0:
        # The full-negation branch (b, c, d) -> (-b, -c, -d) preserves the
        # variance and the kurtosis and flips the skewness twice over, so it is
        # the same distribution; normalise to b > 0.
        b, c, d = -b, -c, -d
    if flip:
        c = -c
    if c == 0.0:
        c = 0.0          # kill a negative zero so a == -c == 0.0 exactly
    return FleishmanCoefficients(
        a=-c if c != 0.0 else 0.0, b=b, c=c, d=d,
        skewness=skew_in, kurtosis=kurt_in,
        converged=bool(best_res <= tol), iterations=int(best_iters),
        residual=float(best_res),
    )


def _least_squares_polish(x0: np.ndarray | None, skewness: float,
                          kurtosis: float) -> Tuple[np.ndarray, float] | None:
    """Optional scipy trust-region polish; ``None`` when it cannot help."""
    if x0 is None:
        return None
    try:  # pragma: no cover - scipy is a hard dependency, guard is cheap
        from scipy.optimize import least_squares
    except Exception:
        return None
    try:
        res = least_squares(
            lambda v: _equations(v[0], v[1], v[2], skewness, kurtosis),
            np.asarray(x0, dtype=float),
            jac=lambda v: _jacobian(v[0], v[1], v[2]),
            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000,
        )
    except Exception:
        return None
    f = _equations(res.x[0], res.x[1], res.x[2], skewness, kurtosis)
    return np.asarray(res.x, dtype=float), float(np.max(np.abs(f)))


# --------------------------------------------------------------------------- #
# The transformation itself
# --------------------------------------------------------------------------- #


def transform(z: np.ndarray, coef: FleishmanCoefficients) -> np.ndarray:
    """Apply ``Y = a + b Z + c Z**2 + d Z**3`` to standard normal deviates.

    Parameters
    ----------
    z : numpy.ndarray
        Standard normal deviates of any shape.
    coef : FleishmanCoefficients
        Coefficients from :func:`solve`.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``z``, with population mean 0, population
        variance 1 and the requested skewness and excess kurtosis.

    Notes
    -----
    Evaluated by Horner's rule, so one pass over the data and no temporary
    powers of ``z``.

    References
    ----------
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    arr = np.asarray(z, dtype=float)
    if coef.c == 0.0 and coef.d == 0.0:
        return arr * coef.b + coef.a
    return coef.a + arr * (coef.b + arr * (coef.c + arr * coef.d))


# --------------------------------------------------------------------------- #
# Vale & Maurelli's intermediate correlation
# --------------------------------------------------------------------------- #


def intermediate_correlation(target_rho: float, ci: FleishmanCoefficients,
                             cj: FleishmanCoefficients) -> float:
    """Normal-stage correlation that yields ``target_rho`` after transformation.

    Vale & Maurelli (1983) show that if ``Z_i`` and ``Z_j`` are standard
    normal with correlation ``rho_z``, the Fleishman-transformed variables
    correlate

    .. math::

        \\rho_y = \\rho_z (b_i b_j + 3 b_i d_j + 3 d_i b_j + 9 d_i d_j)
                + \\rho_z^2 (2 c_i c_j) + \\rho_z^3 (6 d_i d_j).

    This function inverts that cubic for ``rho_z``.

    Parameters
    ----------
    target_rho : float
        Desired correlation between the two transformed variables, in
        ``[-1, 1]``.
    ci, cj : FleishmanCoefficients
        Solved coefficients of the two variables.

    Returns
    -------
    float
        The real root in ``[-1, 1]`` closest to ``target_rho``.  When both
        variables are normal (``c == d == 0``, ``b == 1``) the cubic collapses
        to the identity and the target is returned bit-for-bit unchanged.

    Raises
    ------
    semsim.core.errors.InfeasibleTargetError
        If ``abs(target_rho) > 1``, or if the cubic has no real root inside
        ``[-1, 1]`` -- which means the requested correlation is unattainable
        for that pair of marginal distributions.

    References
    ----------
    Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
        distributions. *Psychometrika*, 48(3), 465-471.
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
        *Psychometrika*, 43(4), 521-532.
    """
    rho = float(target_rho)
    if not np.isfinite(rho) or abs(rho) > 1.0 + 1e-12:
        raise InfeasibleTargetError(
            f"A target correlation must lie in [-1, 1]; got {target_rho!r}.",
            "يجب أن تقع قيمة الارتباط المستهدف بين -1 و 1؛ "
            f"القيمة الممرَّرة هي {target_rho!r}.",
            target_rho=target_rho,
        )
    rho = float(np.clip(rho, -1.0, 1.0))

    k1 = (ci.b * cj.b + 3.0 * ci.b * cj.d + 3.0 * ci.d * cj.b
          + 9.0 * ci.d * cj.d)
    k2 = 2.0 * ci.c * cj.c
    k3 = 6.0 * ci.d * cj.d

    # Exact short-circuits: zero correlation needs zero correlation, and two
    # normal marginals need no adjustment at all.
    if rho == 0.0:
        return 0.0
    if k2 == 0.0 and k3 == 0.0:
        if k1 == 0.0:
            raise InfeasibleTargetError(
                "The two variables are degenerate (zero linear coefficient), so "
                f"no intermediate correlation can produce rho = {rho:.4f}.",
                "المتغيّران منحلّان (المعامل الخطي يساوي صفرًا)، ولذلك لا يوجد "
                f"ارتباط وسيط يمكنه إنتاج ارتباط مقداره {rho:.4f}.",
                target_rho=rho,
            )
        root = rho / k1
        if abs(root) > 1.0 + 1e-9:
            raise InfeasibleTargetError(
                f"The correlation {rho:.4f} is unattainable for this pair of "
                "marginal distributions: it needs an intermediate correlation "
                f"of {root:.4f}, outside [-1, 1].",
                f"الارتباط المطلوب {rho:.4f} غير قابل للتحقّق بين متغيّرين بهذين "
                f"التوزيعين، إذ يحتاج إلى ارتباط وسيط مقداره {root:.4f} وهو "
                "خارج المدى [-1، 1]. خفّض قيمة الارتباط المطلوب أو اجعل "
                "توزيعي المتغيّرين أقرب إلى الاعتدال.",
                target_rho=rho, required=root,
            )
        return float(np.clip(root, -1.0, 1.0))

    coeffs = [k3, k2, k1, -rho]
    # Drop leading zeros so numpy.roots gets a well-posed problem.
    while coeffs and abs(coeffs[0]) < 1e-300:
        coeffs.pop(0)
    if len(coeffs) < 2:
        raise InfeasibleTargetError(
            f"The intermediate-correlation equation degenerates for rho = "
            f"{rho:.4f} with these marginal distributions.",
            f"معادلة الارتباط الوسيط تنحلّ عند ارتباط مقداره {rho:.4f} "
            "مع هذين التوزيعين؛ غيّر قيم الالتواء والتفلطح أو الارتباط المطلوب.",
            target_rho=rho,
        )

    try:
        roots = np.roots(coeffs)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical guard
        raise InfeasibleTargetError(
            f"Could not solve the intermediate-correlation cubic for rho = "
            f"{rho:.4f}.",
            f"تعذّر حلّ معادلة الارتباط الوسيط من الدرجة الثالثة عند ارتباط "
            f"مقداره {rho:.4f}.",
            target_rho=rho,
        ) from exc

    tolerance = 1e-8
    candidates: list[float] = []
    for r in np.atleast_1d(roots):
        if abs(float(np.imag(r))) > tolerance * max(1.0, abs(float(np.real(r)))):
            continue
        real = float(np.real(r))
        if -1.0 - 1e-9 <= real <= 1.0 + 1e-9:
            candidates.append(float(np.clip(real, -1.0, 1.0)))

    if not candidates:
        raise InfeasibleTargetError(
            f"No intermediate correlation in [-1, 1] produces a correlation of "
            f"{rho:.4f} between variables with skewness "
            f"({ci.skewness:.3f}, {cj.skewness:.3f}) and excess kurtosis "
            f"({ci.kurtosis:.3f}, {cj.kurtosis:.3f}).",
            f"لا يوجد ارتباط وسيط بين -1 و 1 يُنتج ارتباطًا مقداره {rho:.4f} "
            f"بين متغيّرين التواؤهما ({ci.skewness:.3f}، {cj.skewness:.3f}) "
            f"وتفلطحهما ({ci.kurtosis:.3f}، {cj.kurtosis:.3f}). "
            "خفّض قيمة الارتباط المطلوب أو اجعل التوزيعين أقرب إلى الاعتدال.",
            target_rho=rho,
        )

    # Several real roots can sit inside the unit interval; the admissible one
    # is the branch continuous with the normal case, i.e. nearest the target.
    best = min(candidates, key=lambda r: (abs(r - rho), abs(r)))
    return float(best)

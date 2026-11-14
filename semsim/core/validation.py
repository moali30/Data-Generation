"""
semsim.core.validation
======================
Bilingual (Arabic / English) diagnostics for a :class:`~semsim.core.spec.ModelSpec`.

Two entry points
----------------
:func:`validate`
    Pure-Python, allocation-light structural checks on the *specification*.
    It is wired to the UI's ``textChanged`` signals, so it must be fast
    (< 20 ms for a 20-construct model) and it **must never raise** --- a
    half-finished, self-contradictory spec is precisely the input it exists to
    describe.
:func:`validate_compiled`
    The post-compilation checks that need the actual matrices: Heywood cases
    that were really realised, a non-positive-definite ``Sigma``, an
    :math:`R^2` above one, a near-singular implied matrix, and items whose
    communality is so low they measure almost nothing.

Design notes
------------
*   Every diagnostic carries a *stable machine code* (so the UI, the HTML
    report and the test-suite can key off it), an English message, a genuinely
    idiomatic Arabic message written in the vocabulary an Arabic-speaking
    methodologist actually uses (``المتغير الكامن``, ``التشبع``,
    ``درجات الحرية``, ``حالة هيوود``, ``البعد``, ``المتغير الوسيط``,
    ``المتغير المعدِّل``, ``الصدق التمايزي``) with the English technical term in
    parentheses where that is the convention, and an *actionable* bilingual
    hint that names the knob to turn.
*   Severity convention: ``ERROR`` means the model cannot be compiled or
    estimated as described; ``WARNING`` means it will run but the result is
    methodologically questionable; ``INFO`` is commentary (e.g. warnings the
    compiler already emitted).
*   Nothing in this module calls the global ``numpy.random``; nothing here is
    stochastic at all.

References
----------
Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley.
Kline, R. B. (2016). *Principles and Practice of Structural Equation
Modeling* (4th ed.). Guilford.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spec import _sanitise_name, DemographicRole
from .spec import (
    ConstructOrder,
    ConstructRole,
    MissingMechanism,
    ModelSpec,
    ScaleKind,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, compile.py may not exist yet
    from .compile import CompiledModel


# --------------------------------------------------------------------------- #
# Tunable thresholds (conventional values, each with a source)
# --------------------------------------------------------------------------- #

#: Three indicators are the classical minimum for a factor to be identified in
#: isolation (Bollen, 1989, p. 244; Kline, 2016, ch. 9).
MIN_ITEMS_PER_FACTOR = 3
#: A second-order construct with fewer than three first-order dimensions needs
#: extra constraints at the upper level (Rindskopf & Rose, 1988).
MIN_DIMENSIONS_SECOND_ORDER = 3
#: Supported Likert widths.
LIKERT_MIN_CATEGORIES = 2
LIKERT_MAX_CATEGORIES = 11
#: Bentler & Chou (1987): five cases per free parameter is the absolute floor.
MIN_CASES_PER_PARAMETER = 5
#: Hair et al. (2010): an item below this communality explains almost nothing.
LOW_COMMUNALITY_FLOOR = 0.20
#: Above this, ``Sigma`` is numerically near-singular.
CONDITION_NUMBER_LIMIT = 1e10
#: Mirrors ``compile.HEYWOOD_FLOOR``; a variance pinned here was clamped.
HEYWOOD_FLOOR = 1e-3

_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Diagnostic containers
# --------------------------------------------------------------------------- #


class Severity(str, Enum):
    """How badly a diagnostic blocks the simulation."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    """One bilingual finding about a model specification.

    Attributes
    ----------
    severity : Severity
        ``ERROR`` blocks generation, ``WARNING`` does not, ``INFO`` is
        commentary.
    code : str
        Stable machine code, e.g. ``"DF_NEGATIVE"``.  Never localised.
    message_en, message_ar : str
        The finding, in both interface languages.
    where : str
        Id (construct / dimension / item / path) the finding attaches to, so
        the UI can highlight the offending row.
    hint_en, hint_ar : str
        What the researcher should change.
    """

    severity: Severity
    code: str
    message_en: str
    message_ar: str
    where: str = ""
    hint_en: str = ""
    hint_ar: str = ""

    # -- convenience used by the UI / report ------------------------------- #

    def message(self, lang: str = "en") -> str:
        """Return the message in ``lang`` (``"ar"`` or ``"en"``)."""
        return self.message_ar if lang == "ar" else self.message_en

    def hint(self, lang: str = "en") -> str:
        """Return the hint in ``lang`` (``"ar"`` or ``"en"``)."""
        return self.hint_ar if lang == "ar" else self.hint_en

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR


@dataclass
class ValidationReport:
    """The complete set of diagnostics produced by one validation pass."""

    diagnostics: List[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when no ``ERROR`` diagnostic is present."""
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)

    def errors(self) -> List[Diagnostic]:
        """Every ``ERROR`` diagnostic."""
        return [d for d in self.diagnostics if d.severity is Severity.ERROR]

    def warnings(self) -> List[Diagnostic]:
        """Every ``WARNING`` diagnostic."""
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    def infos(self) -> List[Diagnostic]:
        """Every ``INFO`` diagnostic."""
        return [d for d in self.diagnostics if d.severity is Severity.INFO]

    # -- lookups ----------------------------------------------------------- #

    def codes(self) -> List[str]:
        """Codes in the order they were raised (duplicates preserved)."""
        return [d.code for d in self.diagnostics]

    def has(self, code: str) -> bool:
        """Whether ``code`` was raised at least once."""
        return any(d.code == code for d in self.diagnostics)

    def by_code(self, code: str) -> List[Diagnostic]:
        """Every diagnostic carrying ``code``."""
        return [d for d in self.diagnostics if d.code == code]

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        """Return a new report holding this report's diagnostics then ``other``'s."""
        return ValidationReport(list(self.diagnostics) + list(other.diagnostics))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.diagnostics)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.diagnostics)

    def __bool__(self) -> bool:  # pragma: no cover - avoid the empty-list trap
        return True


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _d(
    severity: Severity,
    code: str,
    en: str,
    ar: str,
    where: str = "",
    hint_en: str = "",
    hint_ar: str = "",
) -> Diagnostic:
    return Diagnostic(
        severity=severity,
        code=code,
        message_en=en,
        message_ar=ar,
        where=where,
        hint_en=hint_en,
        hint_ar=hint_ar,
    )


def _label(obj: Any, fallback: str = "?") -> str:
    """Best human-readable name for a construct / dimension / item."""
    if obj is None:
        return fallback
    name = getattr(obj, "name", "") or ""
    if name:
        return str(name)
    lab = getattr(obj, "label", None)
    if lab is not None:
        try:
            text = lab.get("en") or lab.get("ar")
        except Exception:  # pragma: no cover - duck-typed label
            text = ""
        if text:
            return str(text)
    return str(getattr(obj, "id", "") or fallback)


def _num(value: Any, default: float = 0.0) -> float:
    """Coerce a possibly-garbage UI value to a finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _find_cycle(edges: Sequence[Tuple[str, str]]) -> Optional[List[str]]:
    """Return one directed cycle in ``edges`` (excluding self-loops), or ``None``.

    Iterative three-colour depth-first search, so a pathological UI graph
    cannot blow the Python recursion limit.
    """
    adj: Dict[str, List[str]] = {}
    nodes: List[str] = []
    for src, dst in edges:
        if src == dst:
            continue
        if src not in adj:
            adj[src] = []
            nodes.append(src)
        if dst not in adj:
            adj[dst] = []
            nodes.append(dst)
        adj[src].append(dst)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {n: WHITE for n in nodes}

    for root in nodes:
        if colour[root] != WHITE:
            continue
        stack: List[Tuple[str, int]] = [(root, 0)]
        path: List[str] = [root]
        colour[root] = GREY
        while stack:
            node, idx = stack[-1]
            neighbours = adj[node]
            if idx < len(neighbours):
                stack[-1] = (node, idx + 1)
                nxt = neighbours[idx]
                state = colour.get(nxt, WHITE)
                if state == GREY:
                    start = path.index(nxt)
                    return path[start:] + [nxt]
                if state == WHITE:
                    colour[nxt] = GREY
                    stack.append((nxt, 0))
                    path.append(nxt)
            else:
                colour[node] = BLACK
                stack.pop()
                path.pop()
    return None


def _symmetric_pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# --------------------------------------------------------------------------- #
# The latent correlations the structural model *implies*
# --------------------------------------------------------------------------- #
#
# Reading the latent correlations off the declared ``LatentCovariance`` rows
# only -- and assuming ``R = I`` everywhere else -- is wrong, because a
# structural model induces correlations of its own: in ``X -> M`` with a
# coefficient of 0.80 the pair (X, M) correlates 0.80 even though the
# researcher never declared a covariance, and two dimensions of one
# second-order construct correlate through their parent.  Both the quadratic
# form behind ``R2_EXCEEDS_ONE`` (``b' Phi_PP b``) and the one behind
# ``HEYWOOD_RISK`` (``lambda' Phi lambda``) are evaluated in that implied
# metric by :mod:`semsim.core.compile`, so validation must use the same matrix
# or it reports Heywood cases the compiler does not see and misses the ones it
# does.


def _kahn_order(n: int, edges: Sequence[Tuple[int, int]],
                endogenous: Sequence[bool]) -> Tuple[List[int], bool]:
    """Topological node order, every exogenous latent first.

    ``edges`` are ``(source, target)`` pairs.  Exogenous nodes (no incoming
    edge) are emitted before any endogenous one so that the forward
    accumulation below always has every predictor's row of ``Phi`` already
    filled in; a plain Kahn queue can interleave the two and silently treat a
    not-yet-processed predictor as uncorrelated.  The second return value is
    ``True`` when the graph contains a cycle, in which case the nodes inside
    the cycle are appended in index order.
    """
    indegree = [0] * n
    successors: List[List[int]] = [[] for _ in range(n)]
    for s, t in sorted({(int(s), int(t)) for s, t in edges}):
        successors[s].append(t)
        indegree[t] += 1

    ready = [(1 if endogenous[i] else 0, i) for i in range(n) if indegree[i] == 0]
    ready.sort()
    order: List[int] = []
    while ready:
        _, node = ready.pop(0)
        order.append(node)
        for t in successors[node]:
            indegree[t] -= 1
            if indegree[t] == 0:
                ready.append((1 if endogenous[t] else 0, t))
                ready.sort()

    if len(order) < n:
        placed = set(order)
        order.extend(i for i in range(n) if i not in placed)
        return order, True
    return order, False


@dataclass
class _LatentSolution:
    """The latent correlation matrix a specification implies.

    Attributes
    ----------
    index : dict
        Factor id (construct *or* dimension) -> row in ``Phi``.
    Phi : numpy.ndarray
        ``(m, m)`` implied latent correlation matrix, unit diagonal.
    explained : dict
        Factor id -> ``b' Phi_PP b``, the structural :math:`R^2` *before* the
        compiler's Heywood clamp, i.e. the number the researcher asked for.
    coefficients : dict
        Target id -> ``{source id: coefficient}`` after duplicate paths have
        been resolved the way the compiler resolves them (the last path with a
        given source and target wins).
    duplicates : dict
        Target id -> ``{source id: how many paths declared that effect}``.
    cyclic : bool
        ``True`` when the structural graph is non-recursive, in which case
        ``Phi`` is a lower bound rather than the exact solution (the compiler
        reaches it by fixed-point iteration).
    """

    index: Dict[str, int]
    Phi: np.ndarray
    explained: Dict[str, float]
    coefficients: Dict[str, Dict[str, float]]
    duplicates: Dict[str, Dict[str, int]]
    cyclic: bool

    def correlation(self, a: str, b: str) -> float:
        """Implied correlation between two latent factors (0 if unknown)."""
        ia = self.index.get(a)
        ib = self.index.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(self.Phi[ia, ib])


def _solve_latent_correlations(spec: ModelSpec) -> _LatentSolution:
    """Reproduce ``compile.compile_model``'s steps 2-4 without compiling.

    Walks the latent nodes (constructs plus the dimensions inside them) in
    topological order and accumulates ``Phi[i, j] = b_i . Phi[P, j] + Psi[i, j]``
    exactly as ``docs/ARCHITECTURE.md`` step 4 prescribes, including the
    Heywood repair (shrink ``b`` and clamp the disturbance) so that a latent
    further down the chain still sees admissible correlations.

    Cost is O(m^2) with ``m`` the number of latent nodes plus one tiny
    quadratic form per endogenous node, which keeps :func:`validate` far
    inside its 20 ms budget on a 20-construct model.
    """
    constructs = list(getattr(spec, "constructs", None) or [])
    lut = _construct_lookup(spec)

    ids: List[str] = []
    index: Dict[str, int] = {}
    dims_of: List[Tuple[str, List[Any]]] = []
    for c in constructs:
        cid = str(getattr(c, "id", "") or "")
        if not cid or cid in index:
            continue
        index[cid] = len(ids)
        ids.append(cid)
        dims = list(getattr(c, "dimensions", None) or [])
        dims_of.append((cid, dims))
        for dim in dims:
            did = str(getattr(dim, "id", "") or "")
            if not did or did in index:
                continue
            index[did] = len(ids)
            ids.append(did)

    m = len(ids)
    B = np.zeros((m, m), dtype=float)
    edges: List[Tuple[int, int]] = []

    # A second-order loading is a regression of the dimension ON the construct
    # (ARCHITECTURE step 2), which is what makes two dimensions of one parent
    # correlated: Phi[d1, d2] = sol_1 * sol_2.
    for cid, dims in dims_of:
        ci = index[cid]
        for dim in dims:
            di = index.get(str(getattr(dim, "id", "") or ""))
            if di is None or di == ci:
                continue
            B[di, ci] = _num(getattr(dim, "second_order_loading", 0.0))
            edges.append((ci, di))

    coefficients: Dict[str, Dict[str, float]] = {}
    duplicates: Dict[str, Dict[str, int]] = {}
    for p in list(getattr(spec, "paths", None) or []):
        cs = lut.get(str(getattr(p, "source", "") or ""))
        ct = lut.get(str(getattr(p, "target", "") or ""))
        if cs is None or ct is None:
            continue
        sid = str(getattr(cs, "id", "") or "")
        tid = str(getattr(ct, "id", "") or "")
        if sid == tid:
            continue
        si, ti = index.get(sid), index.get(tid)
        if si is None or ti is None:
            continue
        try:
            coef = _num(p.resolved_coefficient())
        except Exception:
            coef = _num(getattr(p, "coefficient", 0.0))
        # Assignment, not accumulation: compile.py step 2 writes
        # ``B[t, s] = path.resolved_coefficient()``, so of two paths with the
        # same source and target the last one is the model that gets compiled.
        B[ti, si] = coef
        coefficients.setdefault(tid, {})[sid] = coef
        row = duplicates.setdefault(tid, {})
        row[sid] = row.get(sid, 0) + 1
        edges.append((si, ti))

    Psi = np.eye(m, dtype=float)
    for lc in list(getattr(spec, "latent_covariances", None) or []):
        ca = lut.get(str(getattr(lc, "construct_a", "") or ""))
        cb = lut.get(str(getattr(lc, "construct_b", "") or ""))
        if ca is None or cb is None:
            continue
        ia = index.get(str(getattr(ca, "id", "") or ""))
        ib = index.get(str(getattr(cb, "id", "") or ""))
        if ia is None or ib is None or ia == ib:
            continue
        r = _num(getattr(lc, "correlation", 0.0))
        if abs(r) >= 1.0:
            continue
        Psi[ia, ib] = Psi[ib, ia] = r

    endogenous = [bool(np.any(B[i] != 0.0)) for i in range(m)]
    order, cyclic = _kahn_order(m, edges, endogenous)

    Phi = np.zeros((m, m), dtype=float)
    explained: Dict[str, float] = {}
    done: List[int] = []
    done_mask = np.zeros(m, dtype=bool)
    for i in order:
        # In a cyclic model a predictor may not be processed yet; leaving it
        # out understates R^2, which is the safe direction for a diagnostic.
        preds = [int(j) for j in np.flatnonzero(B[i] != 0.0) if done_mask[j]]
        if preds:
            idx = np.asarray(preds, dtype=int)
            b = B[i, idx]
            r2 = float(b @ Phi[np.ix_(idx, idx)] @ b)
            explained[ids[i]] = r2
            if r2 > 1.0 - HEYWOOD_FLOOR:
                b = b * math.sqrt((1.0 - HEYWOOD_FLOOR) / max(r2, 1e-300))
            for j in done:
                Phi[i, j] = Phi[j, i] = float(b @ Phi[idx, j]) + Psi[i, j]
        else:
            explained[ids[i]] = 0.0
            for j in done:
                Phi[i, j] = Phi[j, i] = Psi[i, j]
        Phi[i, i] = 1.0
        done.append(i)
        done_mask[i] = True

    return _LatentSolution(index=index, Phi=Phi, explained=explained,
                           coefficients=coefficients, duplicates=duplicates,
                           cyclic=cyclic)


# --------------------------------------------------------------------------- #
# Specification-level validation
# --------------------------------------------------------------------------- #


def validate(spec: ModelSpec) -> ValidationReport:
    """Check a :class:`~semsim.core.spec.ModelSpec` and describe every problem.

    The function is total: any internal failure is reported as a diagnostic
    rather than raised, because the UI calls it on every keystroke while the
    spec is still half-written.

    Parameters
    ----------
    spec : ModelSpec
        The specification to inspect.  May be arbitrarily malformed.

    Returns
    -------
    ValidationReport
        All findings, ordered measurement -> structure -> distribution ->
        targeting.  ``report.ok`` is ``True`` when nothing fatal was found.

    Notes
    -----
    Complexity is linear in items + paths + cross-loadings, with one
    :math:`k \\times k` quadratic form per endogenous construct (``k`` = its
    number of predictors), so a 20-construct model validates in well under a
    millisecond.

    Examples
    --------
    >>> from semsim.core.spec import ModelSpec
    >>> report = validate(ModelSpec())
    >>> report.has("NO_CONSTRUCTS")
    True
    >>> report.ok
    False

    References
    ----------
    Bollen, K. A. (1989). *Structural Equations with Latent Variables*
    (identification rules, pp. 88-104, 244-246). Wiley.
    Kline, R. B. (2016). *Principles and Practice of Structural Equation
    Modeling* (4th ed.), ch. 6-9 (degrees of freedom, recursive models,
    Heywood cases). Guilford.
    Bentler, P. M., & Chou, C.-P. (1987). Practical issues in structural
    modeling. *Sociological Methods & Research*, 16, 78-117.
    Fleishman, A. I. (1978). A method for simulating non-normal distributions.
    *Psychometrika*, 43, 521-532.
    Cudeck, R., & Browne, M. W. (1992). Constructing a covariance matrix that
    yields a specified minimizer. *Psychometrika*, 57, 357-369.
    Aiken, L. S., & West, S. G. (1991). *Multiple Regression: Testing and
    Interpreting Interactions*. Sage.
    """
    out: List[Diagnostic] = []
    try:
        _check_non_finite(spec, out)
        _check_measurement(spec, out)
        _check_complex_reality(spec, out)
        _check_structure(spec, out)
        _check_variance_bounds(spec, out)
        _check_scales(spec, out)
        _check_distribution(spec, out)
        _check_identification(spec, out)
        _check_targeting(spec, out)
        _check_noise(spec, out)
        _check_demographics(spec, out)
    except Exception as exc:  # pragma: no cover - last-resort safety net
        out.append(_d(
            Severity.ERROR,
            "VALIDATOR_FAILED",
            f"The validator could not finish inspecting this model: {exc!r}.",
            f"تعذَّر على المدقِّق إكمال فحص النموذج: {exc!r}.",
            hint_en="Please report this model; meanwhile undo the last edit.",
            hint_ar="أبلِغ عن هذا النموذج، وتراجَع مؤقتًا عن آخر تعديل أجريته.",
        ))
    return ValidationReport(out)


# -- non-finite numbers ----------------------------------------------------- #
#
# ``_num`` deliberately swallows a non-finite value so that :func:`validate`
# can keep inspecting a half-typed spec, but swallowing it silently is a
# falsehood: a NaN loading used to validate completely clean and then
# propagate through ``Lambda``, ``Sigma``, the Cholesky factor and the
# generated data, where it surfaces as an empty column nobody can explain.
# Every numeric field this module reads is therefore screened here first.


def _non_finite_token(v: float) -> Tuple[str, str]:
    """Human-readable name of a non-finite float, in both languages."""
    if math.isnan(v):
        return "NaN (not a number)", "قيمة غير معرَّفة (NaN)"
    if v > 0.0:
        return "+infinity", "ما لا نهاية (+∞)"
    return "-infinity", "سالب ما لا نهاية (−∞)"


def _flag_non_finite(value: Any, out: List[Diagnostic], *,
                     field_en: str, field_ar: str,
                     owner_en: str, owner_ar: str, where: str = "") -> bool:
    """Append a ``NON_FINITE_VALUE`` error when ``value`` is NaN or infinite.

    Returns ``True`` when a diagnostic was raised.  A value that is not a
    number at all (an empty string, ``None``) is *not* reported here: the UI
    binds those fields to numeric editors, and the field-specific range checks
    already describe the fallback that ``_num`` substitutes.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if math.isfinite(v):
        return False
    tok_en, tok_ar = _non_finite_token(v)
    out.append(_d(
        Severity.ERROR,
        "NON_FINITE_VALUE",
        f"The {field_en} of {owner_en} is {tok_en}, not a finite real number. "
        f"The validator reads it as 0, and every quantity derived from it - "
        f"the implied covariance matrix, the fit indices, the generated data - "
        f"would be meaningless.",
        f"قيمة {field_ar} في {owner_ar} هي {tok_ar}، وليست عددًا حقيقيًا "
        f"منتهيًا. ويقرأ المدقِّق هذه القيمة صفرًا، ومن ثَمَّ تفقد كل الكميات "
        f"المشتقة منها معناها: مصفوفة التغاير المُشتقة، ومؤشرات المطابقة، "
        f"والبيانات المولَّدة.",
        where=where,
        hint_en="Enter a finite number in this field; a NaN usually comes "
                "from an empty cell or a division by zero in the step that "
                "produced the value.",
        hint_ar="أدخِل عددًا منتهيًا في هذا الحقل؛ فالقيمة غير المعرَّفة تنشأ "
                "عادةً من خلية فارغة أو من قسمة على صفر في الخطوة التي "
                "أنتجت القيمة.",
    ))
    return True


def _check_non_finite(spec: ModelSpec, out: List[Diagnostic]) -> None:
    """Screen every numeric field :func:`validate` reads for NaN / infinity."""
    constructs = list(getattr(spec, "constructs", None) or [])
    lut = _construct_lookup(spec)

    # -- items and their scales -------------------------------------------- #
    try:
        owned: List[Tuple[Any, Any, Any]] = list(spec.iter_items_with_owner())
    except Exception:  # pragma: no cover - duck-typed spec
        owned = [(it, None, None) for it in _safe_items(spec)]

    for it, construct, _dim in owned:
        iid = str(getattr(it, "id", "") or "")
        iname = _label(it)
        cname = _label(construct) if construct is not None else ""
        own_en = f"item '{iname}'" + (f" of construct '{cname}'" if cname else "")
        own_ar = (f"البند «{iname}»"
                  + (f" التابع للمتغير الكامن «{cname}»" if cname else ""))
        for attr, f_en, f_ar in (
            ("loading", "standardised loading", "التشبع المعياري"),
            ("skewness", "target skewness", "الالتواء المستهدف"),
            ("kurtosis", "target excess kurtosis", "التفلطح الزائد المستهدف"),
            ("extra_error_variance", "extra error variance",
             "تباين الخطأ الإضافي"),
        ):
            _flag_non_finite(getattr(it, attr, 0.0), out, field_en=f_en,
                             field_ar=f_ar, owner_en=own_en, owner_ar=own_ar,
                             where=iid)
        sc = getattr(it, "scale", None)
        if sc is None:
            continue
        for attr, f_en, f_ar in (
            ("mean", "reported mean", "المتوسط المُعلَن"),
            ("sd", "reported standard deviation", "الانحراف المعياري المُعلَن"),
        ):
            _flag_non_finite(getattr(sc, attr, 0.0), out, field_en=f_en,
                             field_ar=f_ar, owner_en=own_en, owner_ar=own_ar,
                             where=iid)
        for k, v in enumerate(list(getattr(sc, "thresholds", None) or [])):
            _flag_non_finite(v, out, field_en=f"threshold number {k + 1}",
                             field_ar=f"الحد الفاصل رقم {k + 1}",
                             owner_en=own_en, owner_ar=own_ar, where=iid)
        for k, v in enumerate(list(getattr(sc, "category_proportions", None) or [])):
            _flag_non_finite(v, out,
                             field_en=f"proportion of category {k + 1}",
                             field_ar=f"نسبة الفئة رقم {k + 1}",
                             owner_en=own_en, owner_ar=own_ar, where=iid)

    # -- second-order loadings --------------------------------------------- #
    for c in constructs:
        cname = _label(c)
        for dim in list(getattr(c, "dimensions", None) or []):
            dname = _label(dim)
            _flag_non_finite(
                getattr(dim, "second_order_loading", 0.0), out,
                field_en="second-order loading",
                field_ar="تشبع البُعد على المتغير الكامن من الرتبة الثانية",
                owner_en=f"dimension '{dname}' of construct '{cname}'",
                owner_ar=f"البُعد «{dname}» التابع للمتغير الكامن «{cname}»",
                where=str(getattr(dim, "id", "") or ""))

    # -- cross-loadings and their policy ----------------------------------- #
    item_by_id = {str(getattr(it, "id", "") or ""): it for it in _safe_items(spec)}
    for cl in list(getattr(spec, "cross_loadings", None) or []):
        iid = str(getattr(cl, "item_id", "") or "")
        fid = str(getattr(cl, "factor_id", "") or "")
        iname = _label(item_by_id.get(iid), iid or "?")
        try:
            fname = spec.factor_label(fid, "en") or fid
        except Exception:
            fname = fid
        _flag_non_finite(
            getattr(cl, "loading", 0.0), out,
            field_en="cross-loading", field_ar="التشبع المتقاطع",
            owner_en=f"item '{iname}' on factor '{fname}'",
            owner_ar=f"البند «{iname}» على العامل «{fname}»",
            where=iid)

    pol = getattr(spec, "cross_loading_policy", None)
    if pol is not None and getattr(pol, "enabled", False):
        pol_en = "the automatic cross-loading policy"
        pol_ar = "سياسة التشبعات المتقاطعة التلقائية"
        _flag_non_finite(getattr(pol, "proportion_of_items", 0.0), out,
                         field_en="proportion of affected items",
                         field_ar="نسبة البنود المتأثرة",
                         owner_en=pol_en, owner_ar=pol_ar)
        try:
            bounds = list(tuple(getattr(pol, "magnitude_range", ()) or ()))
        except Exception:
            bounds = []
        for k, v in enumerate(bounds[:2]):
            edge_en = "lower" if k == 0 else "upper"
            edge_ar = "الأدنى" if k == 0 else "الأعلى"
            _flag_non_finite(v, out,
                             field_en=f"{edge_en} end of the magnitude range",
                             field_ar=f"الطرف {edge_ar} لمدى المقدار",
                             owner_en=pol_en, owner_ar=pol_ar)

    # -- residual covariances and the method factor ------------------------ #
    for ec in list(getattr(spec, "error_covariances", None) or []):
        a = str(getattr(ec, "item_a", "") or "")
        b = str(getattr(ec, "item_b", "") or "")
        na = _label(item_by_id.get(a), a or "?")
        nb = _label(item_by_id.get(b), b or "?")
        _flag_non_finite(
            getattr(ec, "correlation", 0.0), out,
            field_en="residual correlation", field_ar="معامل ارتباط الأخطاء",
            owner_en=f"the residual covariance between items '{na}' and '{nb}'",
            owner_ar=f"تغاير أخطاء البندين «{na}» و«{nb}»",
            where=a)

    cmb = getattr(spec, "common_method_bias", None)
    if cmb is not None and getattr(cmb, "enabled", False):
        _flag_non_finite(
            getattr(cmb, "loading", 0.0), out,
            field_en="loading", field_ar="التشبع",
            owner_en="the common method factor",
            owner_ar="عامل طريقة القياس المشترك")

    # -- structural coefficients ------------------------------------------- #
    for p in list(getattr(spec, "paths", None) or []):
        pid = str(getattr(p, "id", "") or "")
        sname = _label(lut.get(str(getattr(p, "source", "") or "")),
                       str(getattr(p, "source", "") or "?"))
        tname = _label(lut.get(str(getattr(p, "target", "") or "")),
                       str(getattr(p, "target", "") or "?"))
        try:
            coef = p.resolved_coefficient()
        except Exception:
            coef = getattr(p, "coefficient", 0.0)
        _flag_non_finite(
            coef, out,
            field_en="standardised coefficient", field_ar="المعامل المعياري",
            owner_en=f"the path '{sname}' -> '{tname}'",
            owner_ar=f"المسار من «{sname}» إلى «{tname}»",
            where=pid)

    for ix in list(getattr(spec, "interactions", None) or []):
        ixid = str(getattr(ix, "id", "") or "")
        pname = _label(lut.get(str(getattr(ix, "predictor", "") or "")),
                       str(getattr(ix, "predictor", "") or "?"))
        mname = _label(lut.get(str(getattr(ix, "moderator", "") or "")),
                       str(getattr(ix, "moderator", "") or "?"))
        tname = _label(lut.get(str(getattr(ix, "target", "") or "")),
                       str(getattr(ix, "target", "") or "?"))
        try:
            coef = ix.resolved_coefficient()
        except Exception:
            coef = getattr(ix, "coefficient", 0.0)
        _flag_non_finite(
            coef, out,
            field_en="standardised coefficient", field_ar="المعامل المعياري",
            owner_en=f"the interaction '{pname}' x '{mname}' -> '{tname}'",
            owner_ar=f"الحد التفاعلي «{pname}» × «{mname}» على «{tname}»",
            where=ixid)

    for lc in list(getattr(spec, "latent_covariances", None) or []):
        a = str(getattr(lc, "construct_a", "") or "")
        b = str(getattr(lc, "construct_b", "") or "")
        na = _label(lut.get(a), a or "?")
        nb = _label(lut.get(b), b or "?")
        _flag_non_finite(
            getattr(lc, "correlation", 0.0), out,
            field_en="latent correlation",
            field_ar="معامل الارتباط بين المتغيرين الكامنين",
            owner_en=f"the free covariance between '{na}' and '{nb}'",
            owner_ar=f"التغاير الحر بين «{na}» و«{nb}»")

    # -- global generation / targeting settings ---------------------------- #
    sections: List[Tuple[Any, str, str, Sequence[Tuple[str, str, str]]]] = [
        (getattr(spec, "distribution", None),
         "the global distribution settings", "إعدادات التوزيع العامة",
         (("default_skewness", "default skewness", "الالتواء الافتراضي"),
          ("default_kurtosis", "default excess kurtosis",
           "التفلطح الزائد الافتراضي"))),
        (getattr(spec, "noise", None),
         "the noise settings", "إعدادات التشويش",
         (("unique_variance_inflation", "unique-variance inflation factor",
           "معامل تضخيم التباين الفريد"),
          ("careless_proportion", "proportion of careless responders",
           "نسبة المستجيبين غير المبالين"),
          ("outlier_proportion", "proportion of outlier cells",
           "نسبة الخلايا الشاذة"),
          ("missing_proportion", "proportion of missing data",
           "نسبة البيانات المفقودة"),
          ("measurement_jitter_sd", "measurement jitter (in SD units)",
           "مقدار الاهتزاز القياسي (بوحدات الانحراف المعياري)"))),
        (getattr(spec, "fit_target", None),
         "the fit target", "هدف المطابقة",
         (("rmsea", "target RMSEA", "قيمة RMSEA المستهدفة"),
          ("cfi", "target CFI", "قيمة CFI المستهدفة"),
          ("srmr", "target SRMR", "قيمة SRMR المستهدفة"),
          ("tolerance", "calibration tolerance", "هامش التفاوت في المعايرة"))),
        (getattr(spec, "significance_target", None),
         "the significance target", "هدف الدلالة الإحصائية",
         (("alpha", "significance level alpha", "مستوى الدلالة alpha"),
          ("target_power", "target power", "القوة الإحصائية المستهدفة"),
          ("null_coefficient", "null coefficient",
           "معامل المسارات غير الدالة"))),
        (getattr(spec, "power", None),
         "the sample-size settings", "إعدادات حجم العينة",
         (("n", "planned sample size", "حجم العينة المخطَّط"),
          ("target_power", "target power", "القوة الإحصائية المستهدفة"),
          ("alpha", "significance level alpha", "مستوى الدلالة alpha"),
          ("rmsea_null", "null RMSEA", "قيمة RMSEA الصفرية"),
          ("rmsea_alt", "alternative RMSEA", "قيمة RMSEA البديلة"))),
    ]
    for section, own_en, own_ar, fields in sections:
        if section is None:
            continue
        for attr, f_en, f_ar in fields:
            value = getattr(section, attr, None)
            if value is None:
                continue        # an unset optional target is not a bad number
            _flag_non_finite(value, out, field_en=f_en, field_ar=f_ar,
                             owner_en=own_en, owner_ar=own_ar)


# -- measurement layer ------------------------------------------------------ #


def _check_measurement(spec: ModelSpec, out: List[Diagnostic]) -> None:
    constructs = list(getattr(spec, "constructs", None) or [])

    if not constructs:
        out.append(_d(
            Severity.ERROR,
            "NO_CONSTRUCTS",
            "The model contains no constructs; there is nothing to simulate.",
            "لا يحتوي النموذج على أي متغير كامن (latent construct)، فليس هناك "
            "ما يمكن محاكاته.",
            hint_en="Add at least one construct and give it three or more items.",
            hint_ar="أضِف متغيرًا كامنًا واحدًا على الأقل، وزوِّده بثلاثة بنود "
                    "(items) أو أكثر.",
        ))
        return

    for c in constructs:
        cname = _label(c)
        cid = str(getattr(c, "id", "") or "")
        second_order = getattr(c, "order", None) is ConstructOrder.SECOND

        if second_order:
            dims = list(getattr(c, "dimensions", None) or [])
            n_dims = len(dims)
            if n_dims == 0:
                out.append(_d(
                    Severity.ERROR,
                    "CONSTRUCT_NO_ITEMS",
                    f"Second-order construct '{cname}' has no dimensions and "
                    f"therefore no items.",
                    f"المتغير الكامن من الرتبة الثانية «{cname}» لا يتضمن أي "
                    f"بُعد (dimension)، وبالتالي لا يرتبط به أي بند قياس.",
                    where=cid,
                    hint_en="Add three dimensions, each with three or more items.",
                    hint_ar="أضِف ثلاثة أبعاد، ولكل بُعد ثلاثة بنود على الأقل.",
                ))
            elif n_dims < MIN_DIMENSIONS_SECOND_ORDER:
                out.append(_d(
                    Severity.WARNING,
                    "SECOND_ORDER_TOO_FEW_DIMS",
                    f"Second-order construct '{cname}' has only {n_dims} "
                    f"dimension(s); with fewer than "
                    f"{MIN_DIMENSIONS_SECOND_ORDER} the second-order level is "
                    f"under-identified unless extra constraints are imposed.",
                    f"المتغير الكامن من الرتبة الثانية «{cname}» يتضمن "
                    f"{n_dims} بُعدًا فقط؛ وبأقل من "
                    f"{MIN_DIMENSIONS_SECOND_ORDER} أبعاد يبقى مستوى الرتبة "
                    f"الثانية غير محدَّد (under-identified) ما لم تُضَف قيود "
                    f"إضافية على التشبعات.",
                    where=cid,
                    hint_en="Add a third dimension, or constrain the "
                            "second-order loadings to be equal.",
                    hint_ar="أضِف بُعدًا ثالثًا، أو ساوِ بين تشبعات الأبعاد على "
                            "المتغير الكامن من الرتبة الثانية (قيد تساوٍ).",
                ))
            for d in dims:
                _check_factor_items(d, out, kind="dimension", owner=cname)
        else:
            if getattr(c, "dimensions", None):
                out.append(_d(
                    Severity.WARNING,
                    "FIRST_ORDER_WITH_DIMENSIONS",
                    f"Construct '{cname}' is first-order but carries "
                    f"dimensions; the dimensions will be ignored.",
                    f"المتغير الكامن «{cname}» من الرتبة الأولى لكنه يحمل "
                    f"أبعادًا (dimensions)، وسيتم تجاهل هذه الأبعاد.",
                    where=cid,
                    hint_en="Switch the construct to second-order, or delete "
                            "its dimensions.",
                    hint_ar="حوِّل المتغير إلى الرتبة الثانية، أو احذف أبعاده.",
                ))
            _check_factor_items(c, out, kind="construct", owner="")

    # duplicate / empty exported column names -------------------------------
    seen: Dict[str, str] = {}
    try:
        items = list(spec.all_items())
    except Exception:
        items = []
    for it in items:
        iid = str(getattr(it, "id", "") or "")
        raw = getattr(it, "name", "") or ""
        name = str(raw).strip()
        if not name:
            out.append(_d(
                Severity.ERROR,
                "EMPTY_ITEM_NAME",
                f"An item (id {iid or 'unknown'}) has no name, so it cannot "
                f"become a column in the exported dataset.",
                f"أحد البنود (المُعرِّف {iid or 'غير معروف'}) بلا اسم، فلا يمكن "
                f"أن يصبح عمودًا في ملف البيانات المُصدَّر.",
                where=iid,
                hint_en="Give the item a short ASCII name such as SAT1.",
                hint_ar="امنح البند اسمًا لاتينيًا قصيرًا مثل SAT1.",
            ))
            continue
        key = name.lower()
        if key in seen:
            out.append(_d(
                Severity.ERROR,
                "DUPLICATE_ITEM_NAME",
                f"Item name '{name}' is used more than once; dataset columns "
                f"must be unique.",
                f"اسم البند «{name}» مكرَّر؛ ولا يصح أن يتكرَّر اسم العمود "
                f"نفسه في ملف البيانات.",
                where=iid,
                hint_en="Rename one of the two items, or run the automatic "
                        "name-uniquifier.",
                hint_ar="غيِّر اسم أحد البندين، أو استخدم أداة توحيد الأسماء "
                        "التلقائية.",
            ))
        else:
            seen[key] = iid


def _check_factor_items(factor: Any, out: List[Diagnostic], *,
                        kind: str, owner: str) -> None:
    """Item-count checks for one first-order factor (construct or dimension)."""
    fname = _label(factor)
    fid = str(getattr(factor, "id", "") or "")
    items = list(getattr(factor, "items", None) or [])
    n = len(items)

    if n == 0:
        if kind == "dimension":
            out.append(_d(
                Severity.ERROR,
                "DIMENSION_NO_ITEMS",
                f"Dimension '{fname}' of construct '{owner}' has no items.",
                f"البُعد «{fname}» التابع للمتغير الكامن «{owner}» لا يرتبط به "
                f"أي بند قياس (item).",
                where=fid,
                hint_en=f"Add at least {MIN_ITEMS_PER_FACTOR} items to this "
                        f"dimension, or delete it.",
                hint_ar=f"أضِف {MIN_ITEMS_PER_FACTOR} بنود على الأقل لهذا "
                        f"البُعد، أو احذفه.",
            ))
        else:
            out.append(_d(
                Severity.ERROR,
                "CONSTRUCT_NO_ITEMS",
                f"Construct '{fname}' has no items, so it has no measurement "
                f"model.",
                f"المتغير الكامن «{fname}» لا يرتبط به أي بند قياس (item)، "
                f"فلا يوجد له نموذج قياس.",
                where=fid,
                hint_en=f"Add at least {MIN_ITEMS_PER_FACTOR} items to this "
                        f"construct.",
                hint_ar=f"أضِف {MIN_ITEMS_PER_FACTOR} بنود على الأقل لقياس هذا "
                        f"المتغير الكامن.",
            ))
    elif n < MIN_ITEMS_PER_FACTOR:
        noun_en = "Dimension" if kind == "dimension" else "Construct"
        noun_ar = "البُعد" if kind == "dimension" else "المتغير الكامن"
        out.append(_d(
            Severity.WARNING,
            "TOO_FEW_ITEMS",
            f"{noun_en} '{fname}' has only {n} item(s); a factor needs "
            f"{MIN_ITEMS_PER_FACTOR} indicators to be identified in isolation.",
            f"{noun_ar} «{fname}» يقوم على {n} بند فقط؛ ويحتاج العامل إلى "
            f"{MIN_ITEMS_PER_FACTOR} مؤشرات على الأقل ليكون محدَّدًا "
            f"(identified) بمعزل عن بقية النموذج.",
            where=fid,
            hint_en=f"Raise the count to {MIN_ITEMS_PER_FACTOR}, or impose an "
                    f"equality constraint on two loadings.",
            hint_ar=f"ارفع عدد البنود إلى {MIN_ITEMS_PER_FACTOR}، أو ثبِّت "
                    f"تشبعين (loadings) متساويين لتحقيق التحديد.",
        ))

    # loading range -------------------------------------------------------- #
    for it in items:
        lam = _num(getattr(it, "loading", 0.0))
        if abs(lam) >= 1.0:
            iname = _label(it)
            out.append(_d(
                Severity.ERROR,
                "LOADING_OUT_OF_RANGE",
                f"Standardised loading of item '{iname}' is {lam:.3f}; in the "
                f"completely standardised metric |lambda| must be below 1.",
                f"التشبع المعياري (standardised loading) للبند «{iname}» يساوي "
                f"{lam:.3f}، وهو خارج المدى المسموح؛ إذ يجب أن تكون قيمة "
                f"التشبع المطلقة أصغر من 1 في القياس المعياري الكامل.",
                where=str(getattr(it, "id", "") or ""),
                hint_en="Set the loading between 0.30 and 0.95.",
                hint_ar="اضبط التشبع في المدى من 0.30 إلى 0.95.",
            ))

    # second-order loading range ------------------------------------------- #
    if kind == "dimension":
        sol = _num(getattr(factor, "second_order_loading", 0.0))
        if abs(sol) >= 1.0:
            out.append(_d(
                Severity.ERROR,
                "LOADING_OUT_OF_RANGE",
                f"Second-order loading of dimension '{fname}' is {sol:.3f}; "
                f"|lambda| must be below 1.",
                f"تشبع البُعد «{fname}» على المتغير الكامن من الرتبة الثانية "
                f"يساوي {sol:.3f}، ويجب أن تكون قيمته المطلقة أصغر من 1.",
                where=fid,
                hint_en="Set the second-order loading between 0.40 and 0.95.",
                hint_ar="اضبط تشبع البُعد في المدى من 0.40 إلى 0.95.",
            ))


# -- complex-reality layer -------------------------------------------------- #


def _check_complex_reality(spec: ModelSpec, out: List[Diagnostic]) -> None:
    item_ids = {str(getattr(it, "id", "")): it for it in _safe_items(spec)}
    factor_ids = set(_safe_factor_ids(spec))

    # cross-loadings -------------------------------------------------------- #
    for cl in list(getattr(spec, "cross_loadings", None) or []):
        iid = str(getattr(cl, "item_id", "") or "")
        fid = str(getattr(cl, "factor_id", "") or "")
        lam = _num(getattr(cl, "loading", 0.0))
        item = item_ids.get(iid)
        iname = _label(item, iid or "?")

        if item is None:
            out.append(_d(
                Severity.ERROR,
                "CROSSLOAD_UNKNOWN_ITEM",
                f"A cross-loading refers to item id '{iid}', which is not in "
                f"the model.",
                f"أحد التشبعات المتقاطعة (cross-loading) يشير إلى بند غير "
                f"موجود في النموذج (المُعرِّف: {iid}).",
                where=iid,
                hint_en="Delete the cross-loading or pick an existing item.",
                hint_ar="احذف هذا التشبع المتقاطع أو اختر بندًا موجودًا فعلاً.",
            ))
            continue
        if fid not in factor_ids:
            out.append(_d(
                Severity.ERROR,
                "CROSSLOAD_UNKNOWN_FACTOR",
                f"The cross-loading of item '{iname}' targets factor id "
                f"'{fid}', which is not in the model.",
                f"التشبع المتقاطع للبند «{iname}» يشير إلى عامل غير موجود في "
                f"النموذج (المُعرِّف: {fid}).",
                where=iid,
                hint_en="Pick one of the model's constructs or dimensions.",
                hint_ar="اختر أحد المتغيرات الكامنة أو الأبعاد المعرَّفة في "
                        "النموذج.",
            ))
        else:
            home = _safe_home_factor(spec, iid)
            if home is not None and fid == home:
                out.append(_d(
                    Severity.ERROR,
                    "CROSSLOAD_HOME_FACTOR",
                    f"Item '{iname}' has a cross-loading on its own home "
                    f"factor; a cross-loading must point at a different "
                    f"factor.",
                    f"البند «{iname}» له تشبع متقاطع على عامله الأصلي نفسه؛ "
                    f"والتشبع المتقاطع يجب أن يكون على عامل آخر غير العامل "
                    f"الأم (home factor).",
                    where=iid,
                    hint_en="Point the cross-loading at another factor, or "
                            "edit the item's primary loading instead.",
                    hint_ar="وجِّه التشبع المتقاطع إلى عامل آخر، أو عدِّل "
                            "التشبع الأساسي للبند بدلاً من ذلك.",
                ))
        if abs(lam) >= 1.0:
            out.append(_d(
                Severity.ERROR,
                "LOADING_OUT_OF_RANGE",
                f"Cross-loading of item '{iname}' is {lam:.3f}; |lambda| must "
                f"be below 1.",
                f"التشبع المتقاطع للبند «{iname}» يساوي {lam:.3f}، ويجب أن "
                f"تكون قيمته المطلقة أصغر من 1.",
                where=iid,
                hint_en="Keep cross-loadings in the 0.10-0.30 range.",
                hint_ar="أبقِ التشبعات المتقاطعة في المدى من 0.10 إلى 0.30.",
            ))

    # cross-loading policy -------------------------------------------------- #
    pol = getattr(spec, "cross_loading_policy", None)
    if pol is not None and getattr(pol, "enabled", False):
        prop = _num(getattr(pol, "proportion_of_items", 0.0))
        if not (0.0 <= prop <= 1.0):
            out.append(_d(
                Severity.ERROR,
                "CROSSLOAD_POLICY_RANGE",
                f"The automatic cross-loading policy asks for a proportion of "
                f"{prop}, which is outside [0, 1].",
                f"سياسة التشبعات المتقاطعة التلقائية تطلب نسبة {prop} من "
                f"البنود، وهي خارج المدى [0, 1].",
                hint_en="Choose a proportion between 0 and 0.4.",
                hint_ar="اختر نسبة بين 0 و0.4.",
            ))
        lo, hi = 0.0, 0.0
        try:
            lo, hi = (_num(v) for v in tuple(getattr(pol, "magnitude_range", (0.1, 0.25))))
        except Exception:
            lo, hi = 0.1, 0.25
        if hi < lo or abs(hi) >= 1.0 or abs(lo) >= 1.0:
            out.append(_d(
                Severity.ERROR,
                "CROSSLOAD_POLICY_RANGE",
                f"The automatic cross-loading magnitude range ({lo}, {hi}) is "
                f"empty or reaches |lambda| >= 1.",
                f"مدى مقدار التشبعات المتقاطعة التلقائية ({lo}, {hi}) مدى فارغ "
                f"أو يبلغ قيمة تشبع مطلقة ≥ 1.",
                hint_en="Use an increasing range inside (0, 1), e.g. "
                        "(0.10, 0.25).",
                hint_ar="استخدم مدى متزايدًا داخل (0, 1)، مثل (0.10, 0.25).",
            ))

    # error covariances ----------------------------------------------------- #
    seen_pairs: Dict[Tuple[str, str], int] = {}
    for ec in list(getattr(spec, "error_covariances", None) or []):
        a = str(getattr(ec, "item_a", "") or "")
        b = str(getattr(ec, "item_b", "") or "")
        r = _num(getattr(ec, "correlation", 0.0))
        na = _label(item_ids.get(a), a or "?")
        nb = _label(item_ids.get(b), b or "?")

        if a and a == b:
            out.append(_d(
                Severity.ERROR,
                "ERRCOV_SAME_ITEM",
                f"A residual covariance is declared between item '{na}' and "
                f"itself; that quantity is the item's error *variance*, which "
                f"the compiler derives automatically.",
                f"هناك تغاير أخطاء (error covariance) محدَّد بين البند «{na}» "
                f"ونفسه؛ وهذه الكمية هي تباين خطأ البند لا تغايره، ويستنبطها "
                f"المُجمِّع تلقائيًا.",
                where=a,
                hint_en="Pick two different items, or raise the item's "
                        "extra error variance instead.",
                hint_ar="اختر بندين مختلفين، أو ارفع «تباين الخطأ الإضافي» "
                        "للبند نفسه بدلاً من ذلك.",
            ))
            continue
        for ref, nm in ((a, na), (b, nb)):
            if ref not in item_ids:
                out.append(_d(
                    Severity.ERROR,
                    "ERRCOV_UNKNOWN_ITEM",
                    f"A residual covariance refers to item id '{ref}', which "
                    f"is not in the model.",
                    f"أحد تغايرات الأخطاء يشير إلى بند غير موجود في النموذج "
                    f"(المُعرِّف: {ref}).",
                    where=ref,
                    hint_en="Delete the residual covariance or pick an "
                            "existing item.",
                    hint_ar="احذف تغاير الأخطاء أو اختر بندًا موجودًا فعلاً.",
                ))
        if not (abs(r) < 1.0):
            out.append(_d(
                Severity.ERROR,
                "ERRCOV_OUT_OF_RANGE",
                f"The residual correlation between '{na}' and '{nb}' is {r}, "
                f"outside the admissible interval (-1, 1).",
                f"معامل ارتباط أخطاء البندين «{na}» و«{nb}» يساوي {r}، وهو "
                f"خارج المدى المسموح (-1, 1).",
                where=a,
                hint_en="Use a residual correlation between -0.60 and 0.60; "
                        "0.20-0.30 is the usual magnitude.",
                hint_ar="استخدم ارتباط أخطاء بين -0.60 و0.60، والمعتاد في "
                        "الدراسات من 0.20 إلى 0.30.",
            ))
        if a in item_ids and b in item_ids:
            key = _symmetric_pair_key(a, b)
            seen_pairs[key] = seen_pairs.get(key, 0) + 1
            if seen_pairs[key] == 2:
                out.append(_d(
                    Severity.WARNING,
                    "ERRCOV_DUPLICATE",
                    f"The residual covariance between '{na}' and '{nb}' is "
                    f"declared more than once; only the last one survives.",
                    f"تغاير الأخطاء بين البندين «{na}» و«{nb}» مكرَّر، "
                    f"وسيُعتمد آخر قيمة فقط.",
                    where=a,
                    hint_en="Delete the duplicate row.",
                    hint_ar="احذف السطر المكرَّر.",
                ))

    # common method bias ---------------------------------------------------- #
    cmb = getattr(spec, "common_method_bias", None)
    if cmb is not None and getattr(cmb, "enabled", False):
        lam = _num(getattr(cmb, "loading", 0.0))
        if abs(lam) >= 1.0:
            out.append(_d(
                Severity.ERROR,
                "LOADING_OUT_OF_RANGE",
                f"The common-method-factor loading is {lam:.3f}; |lambda| "
                f"must be below 1.",
                f"تشبع عامل طريقة القياس المشترك (common method factor) يساوي "
                f"{lam:.3f}، ويجب أن تكون قيمته المطلقة أصغر من 1.",
                hint_en="A method loading of 0.20-0.40 reproduces the bias "
                        "levels reported in the literature.",
                hint_ar="تشبع عامل الطريقة بين 0.20 و0.40 يعيد مستويات التحيز "
                        "الشائعة في الأدبيات.",
            ))
        for ref in list(getattr(cmb, "item_ids", None) or []):
            if str(ref) not in item_ids:
                out.append(_d(
                    Severity.ERROR,
                    "CMB_UNKNOWN_ITEM",
                    f"The common method factor is restricted to item id "
                    f"'{ref}', which is not in the model.",
                    f"عامل طريقة القياس المشترك مقصور على بند غير موجود في "
                    f"النموذج (المُعرِّف: {ref}).",
                    where=str(ref),
                    hint_en="Clear the restriction list to load on every item.",
                    hint_ar="أفرِغ قائمة البنود ليُحمَّل العامل على جميع البنود.",
                ))


# -- structural layer ------------------------------------------------------- #


def _construct_lookup(spec: ModelSpec) -> Dict[str, Any]:
    """Map every usable reference (id first, then name) to its construct."""
    lut: Dict[str, Any] = {}
    for c in list(getattr(spec, "constructs", None) or []):
        name = str(getattr(c, "name", "") or "")
        if name and name not in lut:
            lut[name] = c
    for c in list(getattr(spec, "constructs", None) or []):
        cid = str(getattr(c, "id", "") or "")
        if cid:
            lut[cid] = c
    return lut


def _check_structure(spec: ModelSpec, out: List[Diagnostic]) -> None:
    lut = _construct_lookup(spec)
    paths = list(getattr(spec, "paths", None) or [])
    interactions = list(getattr(spec, "interactions", None) or [])

    edges: List[Tuple[str, str]] = []
    seen_edge: Dict[Tuple[str, str], int] = {}
    incoming: Dict[str, int] = {}
    outgoing: Dict[str, int] = {}

    for p in paths:
        pid = str(getattr(p, "id", "") or "")
        s = str(getattr(p, "source", "") or "")
        t = str(getattr(p, "target", "") or "")
        cs, ct = lut.get(s), lut.get(t)

        for ref, who_en, who_ar in ((s, "source", "المصدر"), (t, "target", "الهدف")):
            if ref not in lut:
                out.append(_d(
                    Severity.ERROR,
                    "PATH_UNKNOWN_CONSTRUCT",
                    f"A structural path has a {who_en} ('{ref}') that is not a "
                    f"construct in this model.",
                    f"أحد المسارات البنائية يشير إلى متغير كامن غير موجود في "
                    f"النموذج في موضع {who_ar} (المُعرِّف: {ref}).",
                    where=pid,
                    hint_en="Re-pick the source and target from the construct "
                            "list, or delete the path.",
                    hint_ar="أعِد اختيار المتغير الكامن المصدر والهدف من قائمة "
                            "المتغيرات المعرَفة، أو احذف المسار.",
                ))
        if cs is None or ct is None:
            continue

        sid = str(getattr(cs, "id", "") or s)
        tid = str(getattr(ct, "id", "") or t)
        sname, tname = _label(cs), _label(ct)

        if sid == tid:
            out.append(_d(
                Severity.ERROR,
                "SELF_PATH",
                f"The path from '{sname}' to itself is a self-regression, "
                f"which is not defined in structural equation modelling.",
                f"المسار يربط المتغير الكامن «{sname}» بنفسه؛ والانحدار الذاتي "
                f"غير معرَّف في نماذج المعادلات البنائية.",
                where=pid,
                hint_en="Delete this path, or choose a different target "
                        "construct.",
                hint_ar="احذف هذا المسار، أو اختر متغيرًا هدفًا مختلفًا.",
            ))
            continue

        key = (sid, tid)
        seen_edge[key] = seen_edge.get(key, 0) + 1
        if seen_edge[key] == 2:
            out.append(_d(
                Severity.WARNING,
                "DUPLICATE_PATH",
                f"The path '{sname}' -> '{tname}' is declared more than once; "
                f"the compiler writes one coefficient per cell of B, so the "
                f"last declaration wins and the earlier one is silently "
                f"ignored rather than added to it.",
                f"المسار من «{sname}» إلى «{tname}» مكرَّر؛ ولأن المُجمِّع "
                f"يكتب معاملًا واحدًا في خلية مصفوفة المسارات فإن آخر مسار "
                f"مُعلَن هو الذي يُعتمد، ويُهمَل المسار الأسبق دون أن يُضاف "
                f"إليه.",
                where=pid,
                hint_en="Keep one path only and put the whole effect size in "
                        "it; the two coefficients are not summed.",
                hint_ar="احتفظ بمسار واحد فقط واجمع حجم التأثير كله فيه، "
                        "فالمعاملان لا يُجمعان.",
            ))
        edges.append((sid, tid))
        outgoing[sid] = outgoing.get(sid, 0) + 1
        incoming[tid] = incoming.get(tid, 0) + 1

    # interactions ---------------------------------------------------------- #
    moderators_used: set = set()
    for ix in interactions:
        ixid = str(getattr(ix, "id", "") or "")
        pred = str(getattr(ix, "predictor", "") or "")
        mod = str(getattr(ix, "moderator", "") or "")
        tgt = str(getattr(ix, "target", "") or "")
        cpred, cmod, ctgt = lut.get(pred), lut.get(mod), lut.get(tgt)

        missing = [r for r, c in ((pred, cpred), (mod, cmod), (tgt, ctgt)) if c is None]
        for ref in missing:
            out.append(_d(
                Severity.ERROR,
                "INTERACTION_UNKNOWN_CONSTRUCT",
                f"An interaction term refers to construct '{ref}', which is "
                f"not in the model.",
                f"أحد الحدود التفاعلية (interaction term) يشير إلى متغير كامن "
                f"غير موجود في النموذج (المُعرِّف: {ref}).",
                where=ixid,
                hint_en="Re-pick the predictor, the moderator and the target.",
                hint_ar="أعِد اختيار المتغير المستقل والمتغير المعدِّل "
                        "والمتغير الهدف.",
            ))
        if missing:
            continue

        if cmod is not None:
            moderators_used.add(str(getattr(cmod, "id", "") or mod))
        tid = str(getattr(ctgt, "id", "") or tgt)
        tname = _label(ctgt)

        if incoming.get(tid, 0) == 0:
            out.append(_d(
                Severity.ERROR,
                "INTERACTION_TARGET_EXOGENOUS",
                f"The target of the interaction "
                f"'{_label(cpred)}' x '{_label(cmod)}' is '{tname}', which "
                f"receives no structural path and is therefore exogenous; a "
                f"moderation cannot be interpreted without the two main "
                f"effects in the same equation.",
                f"هدف الحد التفاعلي «{_label(cpred)}» × «{_label(cmod)}» هو "
                f"المتغير «{tname}»، وهو متغير خارجي (exogenous) لا يستقبل أي "
                f"مسار بنائي؛ ولا يصح تقدير أثر التعديل دون إدراج الأثرين "
                f"الرئيسين للمتغير المستقل والمتغير المعدِّل في المعادلة "
                f"نفسها.",
                where=ixid,
                hint_en="Add a path from the predictor and a path from the "
                        "moderator to the target before adding the product "
                        "term.",
                hint_ar="أضِف مسارًا من المتغير المستقل ومسارًا من المتغير "
                        "المعدِّل إلى المتغير الهدف قبل إضافة حد التفاعل.",
            ))

    # role consistency ------------------------------------------------------ #
    for c in list(getattr(spec, "constructs", None) or []):
        cid = str(getattr(c, "id", "") or "")
        cname = _label(c)
        role = getattr(c, "role", None)
        n_in = incoming.get(cid, 0)
        n_out = outgoing.get(cid, 0)

        if role is ConstructRole.MODERATOR and cid not in moderators_used:
            out.append(_d(
                Severity.WARNING,
                "MODERATOR_NOT_IN_INTERACTION",
                f"Construct '{cname}' is flagged as a moderator but does not "
                f"appear in any interaction term, so it is currently just an "
                f"ordinary exogenous predictor.",
                f"المتغير المعدِّل (moderator) «{cname}» مصنَّف كمتغير معدِّل "
                f"لكنه لا يدخل في أي حد تفاعلي (interaction term)، فهو "
                f"يُعامَل حاليًا كمتغير مستقل عادي.",
                where=cid,
                hint_en="Add an interaction that uses this moderator, or "
                        "change its role to 'independent'.",
                hint_ar="أضِف حدًّا تفاعليًا يضم هذا المتغير المعدِّل، أو غيِّر "
                        "دوره إلى «متغير مستقل».",
            ))

        if role is ConstructRole.MEDIATOR and (n_in == 0 or n_out == 0):
            if n_in == 0 and n_out == 0:
                lack_en = "neither an incoming nor an outgoing path"
                lack_ar = "مسارًا داخلاً ولا مسارًا خارجًا"
            elif n_in == 0:
                lack_en = "no incoming path"
                lack_ar = "المسار الداخل إليه"
            else:
                lack_en = "no outgoing path"
                lack_ar = "المسار الخارج منه"
            out.append(_d(
                Severity.WARNING,
                "MEDIATOR_WITHOUT_IN_OR_OUT",
                f"Mediator '{cname}' has {lack_en}; mediation needs both arms "
                f"(X -> M and M -> Y) for an indirect effect to exist.",
                f"المتغير الوسيط (mediator) «{cname}» ينقصه {lack_ar}؛ "
                f"والوساطة تقتضي وجود مسار داخل إلى الوسيط ومسار خارج منه حتى "
                f"يتحقَّق أثر غير مباشر.",
                where=cid,
                hint_en="Complete the chain: a path from the predictor to the "
                        "mediator and a path from the mediator to the outcome.",
                hint_ar="أكمِل سلسلة الوساطة: مسار من المتغير المستقل إلى "
                        "الوسيط، ومسار من الوسيط إلى المتغير التابع.",
            ))

        if role is ConstructRole.DEPENDENT and n_in == 0:
            out.append(_d(
                Severity.WARNING,
                "DEPENDENT_WITHOUT_PREDICTOR",
                f"Dependent construct '{cname}' receives no structural path, "
                f"so it stays exogenous and has no explained variance (R^2).",
                f"المتغير الكامن التابع «{cname}» لا يستقبل أي مسار بنائي، "
                f"فيبقى متغيرًا خارجيًا (exogenous) ولا يكون له تباين مُفسَّر "
                f"(R²).",
                where=cid,
                hint_en="Add at least one path from a predictor to this "
                        "construct.",
                hint_ar="أضِف مسارًا واحدًا على الأقل من متغير مستقل إلى هذا "
                        "المتغير التابع.",
            ))

    # cycles ---------------------------------------------------------------- #
    cycle = _find_cycle(edges)
    if cycle is not None:
        names = " -> ".join(_label(lut.get(n), n) for n in cycle)
        names_ar = " ← ".join(_label(lut.get(n), n) for n in reversed(cycle))
        out.append(_d(
            Severity.WARNING,
            "CYCLE_IN_PATHS",
            f"The structural model contains a feedback loop: {names}. A "
            f"non-recursive model needs instruments or extra constraints "
            f"before it is identified, and the compiler must fall back on "
            f"fixed-point iteration.",
            f"يوجد في النموذج البنائي حلقة سببية (feedback loop) بين "
            f"المتغيرات الكامنة: {names_ar}. والنموذج غير التعاقبي "
            f"(non-recursive) يحتاج إلى متغيرات مُعرِّفة (instruments) أو قيود "
            f"إضافية حتى يكون محدَّدًا، كما يضطر المُجمِّع إلى حل تكراري "
            f"بالنقطة الثابتة.",
            where="",
            hint_en="Delete one path in the loop, or add an instrument for "
                    "each equation inside it.",
            hint_ar="احذف أحد مسارات الحلقة، أو أضِف متغيرًا مُعرِّفًا لكل "
                    "معادلة داخل الحلقة.",
        ))

    # latent covariances ---------------------------------------------------- #
    endo = set()
    try:
        endo = set(spec.endogenous_construct_ids())
    except Exception:
        endo = {t for _, t in edges}
    for lc in list(getattr(spec, "latent_covariances", None) or []):
        a = str(getattr(lc, "construct_a", "") or "")
        b = str(getattr(lc, "construct_b", "") or "")
        r = _num(getattr(lc, "correlation", 0.0))
        ca, cb = lut.get(a), lut.get(b)
        for ref, c in ((a, ca), (b, cb)):
            if c is None:
                out.append(_d(
                    Severity.ERROR,
                    "LATCOV_UNKNOWN_CONSTRUCT",
                    f"A latent covariance refers to construct '{ref}', which "
                    f"is not in the model.",
                    f"أحد تغايرات المتغيرات الكامنة يشير إلى متغير غير موجود "
                    f"في النموذج (المُعرِّف: {ref}).",
                    hint_en="Delete the covariance or pick an existing "
                            "construct.",
                    hint_ar="احذف التغاير أو اختر متغيرًا كامنًا موجودًا فعلاً.",
                ))
        if ca is None or cb is None:
            continue
        aid = str(getattr(ca, "id", "") or a)
        bid = str(getattr(cb, "id", "") or b)
        if not (abs(r) < 1.0):
            out.append(_d(
                Severity.ERROR,
                "LATCOV_OUT_OF_RANGE",
                f"The latent correlation between '{_label(ca)}' and "
                f"'{_label(cb)}' is {r}, outside (-1, 1).",
                f"معامل الارتباط بين المتغيرين الكامنين «{_label(ca)}» "
                f"و«{_label(cb)}» يساوي {r}، وهو خارج المدى (-1, 1).",
                hint_en="Use a correlation between -0.85 and 0.85; above 0.85 "
                        "discriminant validity collapses.",
                hint_ar="استخدم ارتباطًا بين -0.85 و0.85؛ وما فوق 0.85 ينهار "
                        "معه الصدق التمايزي (discriminant validity).",
            ))
        if aid in endo or bid in endo:
            out.append(_d(
                Severity.WARNING,
                "LATCOV_ENDOGENOUS",
                f"The free covariance between '{_label(ca)}' and "
                f"'{_label(cb)}' involves an endogenous construct; it will be "
                f"treated as a correlated disturbance, not as an exogenous "
                f"covariance.",
                f"التغاير الحر بين «{_label(ca)}» و«{_label(cb)}» يضم متغيرًا "
                f"كامنًا داخليًا (endogenous)، وسيُعامَل على أنه ارتباط بين "
                f"البواقي (correlated disturbance) لا تغايرًا بين متغيرين "
                f"خارجيين.",
                hint_en="Declare free covariances between exogenous "
                        "constructs only, or accept the correlated "
                        "disturbance.",
                hint_ar="اقصر التغايرات الحرة على المتغيرات الخارجية، أو اقبل "
                        "تفسيرها كارتباط بين البواقي.",
            ))

    # indirect effects ------------------------------------------------------ #
    edge_set = set(edges)
    for ind in list(getattr(spec, "indirect_effects", None) or []):
        chain = [str(x) for x in (getattr(ind, "chain", None) or [])]
        iid = str(getattr(ind, "id", "") or "")
        resolved = [lut.get(x) for x in chain]
        if len(chain) < 3 or any(c is None for c in resolved):
            out.append(_d(
                Severity.ERROR,
                "INDIRECT_CHAIN_BROKEN",
                f"The indirect effect {chain!r} is not a valid mediation "
                f"chain: it needs at least three known constructs.",
                f"سلسلة الأثر غير المباشر {chain!r} ليست سلسلة وساطة صحيحة: "
                f"فهي تحتاج ثلاثة متغيرات كامنة معروفة على الأقل.",
                where=iid,
                hint_en="Build the chain as predictor -> mediator -> outcome.",
                hint_ar="ابنِ السلسلة على النحو: مستقل ← وسيط ← تابع.",
            ))
            continue
        ids = [str(getattr(c, "id", "")) for c in resolved]
        for u, v in zip(ids, ids[1:]):
            if (u, v) not in edge_set:
                out.append(_d(
                    Severity.ERROR,
                    "INDIRECT_CHAIN_BROKEN",
                    f"The indirect effect names the link "
                    f"'{_label(lut.get(u), u)}' -> "
                    f"'{_label(lut.get(v), v)}', but no such path exists; the "
                    f"indirect effect would be exactly zero.",
                    f"سلسلة الأثر غير المباشر تفترض وجود مسار من "
                    f"«{_label(lut.get(u), u)}» إلى "
                    f"«{_label(lut.get(v), v)}»، وهذا المسار غير موجود؛ "
                    f"فالأثر غير المباشر سيكون صفرًا بالضرورة.",
                    where=iid,
                    hint_en="Add the missing path, or shorten the chain.",
                    hint_ar="أضِف المسار الناقص، أو اختصر السلسلة.",
                ))
                break


# -- variance bounds (Heywood risk, R^2) ------------------------------------ #


def _check_variance_bounds(spec: ModelSpec, out: List[Diagnostic]) -> None:
    lut = _construct_lookup(spec)

    # ---- the latent correlations the model itself implies ---------------- #
    # Declared LatentCovariance rows are only part of the story; everything
    # else the structural model induces (X -> M, two dimensions of one
    # second-order construct, a chain X -> M -> Y) lives in this matrix.
    solution = _solve_latent_correlations(spec)

    # ---- communality per item (Heywood risk) ----------------------------- #
    cmb = getattr(spec, "common_method_bias", None)
    cmb_on = bool(cmb is not None and getattr(cmb, "enabled", False))
    cmb_loading = _num(getattr(cmb, "loading", 0.0)) if cmb_on else 0.0
    cmb_items = {str(x) for x in (getattr(cmb, "item_ids", None) or [])} if cmb_on else set()

    #: item id -> {factor id: loading}.  A dict, not a list, because
    #: compile.py writes ``Lambda[j, fi] = cl.loading``: two cross-loadings of
    #: one item on one factor are not additive, the last one simply wins.
    cross_by_item: Dict[str, Dict[str, float]] = {}
    for cl in list(getattr(spec, "cross_loadings", None) or []):
        iid = str(getattr(cl, "item_id", "") or "")
        fid = str(getattr(cl, "factor_id", "") or "")
        cross_by_item.setdefault(iid, {})[fid] = _num(getattr(cl, "loading", 0.0))

    for it in _safe_items(spec):
        iid = str(getattr(it, "id", "") or "")
        iname = _label(it)
        home = _safe_home_factor(spec, iid)
        lam_h = _num(getattr(it, "loading", 0.0))

        # The item's row of Lambda, as (factor id, loading) pairs, then the
        # quadratic form ARCHITECTURE step 6 specifies:
        # communality = Lambda[j] @ Phi @ Lambda[j].
        row: List[Tuple[str, float]] = []
        if home is not None:
            row.append((home, lam_h))
        for fid, lam_c in cross_by_item.get(iid, {}).items():
            if home is not None and fid == home:
                continue          # compile.py ignores a home-factor cross-loading
            if fid not in solution.index:
                continue          # unknown factor: CROSSLOAD_UNKNOWN_FACTOR covers it
            row.append((fid, lam_c))

        if row:
            comm = 0.0
            for fa, la in row:
                for fb, lb in row:
                    comm += la * lb * (1.0 if fa == fb
                                       else solution.correlation(fa, fb))
        else:
            # No resolvable home factor (a malformed spec): fall back on the
            # orthogonal sum of squares rather than reporting nothing at all.
            comm = lam_h * lam_h + sum(
                v * v for f, v in cross_by_item.get(iid, {}).items())
        # The method factor is a separate latent with a unit variance and no
        # paths or covariances, so it is orthogonal to every other factor and
        # contributes exactly its squared loading.
        if cmb_on and (not cmb_items or iid in cmb_items):
            comm += cmb_loading * cmb_loading

        if comm >= 1.0 - _TOL:
            out.append(_d(
                Severity.ERROR,
                "HEYWOOD_RISK",
                f"The implied communality of item '{iname}' is {comm:.3f} "
                f"(>= 1), which forces a negative unique variance - a Heywood "
                f"case.",
                f"الشيوع (communality) المُستنبَط للبند «{iname}» يساوي "
                f"{comm:.3f} أي ≥ 1، وهذا يفرض تباين خطأ سالبًا، وهي حالة "
                f"هيوود (Heywood case).",
                where=iid,
                hint_en="Lower the primary loading, the cross-loadings, or "
                        "the common-method loading until the implied "
                        "communality is below 1; remember that a cross-loading "
                        "on a correlated factor adds 2*lambda_h*lambda_c*r, "
                        "not just its own square.",
                hint_ar="اخفض التشبع الأساسي أو التشبعات المتقاطعة أو تشبع "
                        "عامل الطريقة المشتركة حتى يصبح الشيوع المُستنبَط أقل "
                        "من 1، مع ملاحظة أن التشبع المتقاطع على عامل مرتبط "
                        "بالعامل الأم يضيف المقدار 2×التشبع الأساسي×التشبع "
                        "المتقاطع×معامل الارتباط، لا مربعه فقط.",
            ))

    # ---- R^2 per endogenous construct ------------------------------------ #
    # ``solution.explained`` already holds ``b' Phi_PP b`` for every latent
    # node, evaluated in the implied metric and with duplicate paths resolved
    # the compiler's way (last one wins).
    ix_extra: Dict[str, float] = {}
    for ix in list(getattr(spec, "interactions", None) or []):
        ct = lut.get(str(getattr(ix, "target", "") or ""))
        if ct is None:
            continue
        try:
            coef = _num(ix.resolved_coefficient())
        except Exception:
            coef = _num(getattr(ix, "coefficient", 0.0))
        tid = str(getattr(ct, "id", ""))
        ix_extra[tid] = ix_extra.get(tid, 0.0) + coef * coef

    targets = list(solution.coefficients.keys())
    targets += [t for t in ix_extra if t not in solution.coefficients]
    for tid in targets:
        explained = solution.explained.get(tid, 0.0) + ix_extra.get(tid, 0.0)
        if explained >= 1.0 - _TOL:
            cname = _label(lut.get(tid), tid)
            out.append(_d(
                Severity.ERROR,
                "R2_EXCEEDS_ONE",
                f"The implied R-squared of construct '{cname}' is "
                f"{explained:.3f} (>= 1), so its disturbance variance would "
                f"be negative - a Heywood case at the structural level.",
                f"معامل التحديد R² المُستنبَط للمتغير الكامن «{cname}» يساوي "
                f"{explained:.3f} أي ≥ 1، فيصبح تباين البواقي (disturbance "
                f"variance) سالبًا، وهي حالة هيوود على المستوى البنائي.",
                where=tid,
                hint_en="Lower the incoming path coefficients, or lower the "
                        "correlations among its predictors - including the "
                        "correlations the model itself induces, such as the "
                        "one between X and M in X -> M -> Y.",
                hint_ar="اخفض معاملات المسارات الداخلة إلى هذا المتغير، أو "
                        "اخفض الارتباطات بين المتغيرات المتنبئة به، ومنها "
                        "الارتباطات التي يولّدها النموذج نفسه مثل الارتباط بين "
                        "المتغير المستقل والمتغير الوسيط في سلسلة الوساطة.",
            ))


# -- scales ----------------------------------------------------------------- #


def _check_scales(spec: ModelSpec, out: List[Diagnostic]) -> None:
    for it in _safe_items(spec):
        iid = str(getattr(it, "id", "") or "")
        iname = _label(it)
        sc = getattr(it, "scale", None)
        if sc is None:
            continue
        ordinal = getattr(sc, "kind", None) is ScaleKind.LIKERT
        k = getattr(sc, "n_categories", 5)
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = -1

        if ordinal and not (LIKERT_MIN_CATEGORIES <= k <= LIKERT_MAX_CATEGORIES):
            out.append(_d(
                Severity.ERROR,
                "LIKERT_CATEGORY_RANGE",
                f"Item '{iname}' asks for {k} Likert categories; the "
                f"supported range is "
                f"{LIKERT_MIN_CATEGORIES}-{LIKERT_MAX_CATEGORIES}.",
                f"عدد فئات مقياس ليكرت للبند «{iname}» يساوي {k}، وهو خارج "
                f"المدى المدعوم من {LIKERT_MIN_CATEGORIES} إلى "
                f"{LIKERT_MAX_CATEGORIES} فئة.",
                where=iid,
                hint_en="Use 5 or 7 categories, the two commonest widths in "
                        "survey research.",
                hint_ar="استخدم 5 أو 7 فئات، وهما الأكثر شيوعًا في بحوث "
                        "الاستبانات.",
            ))

        props = getattr(sc, "category_proportions", None)
        if props:
            vals = [_num(v, float("nan")) for v in props]
            if any(math.isnan(v) for v in vals):
                vals = [0.0 if math.isnan(v) else v for v in vals]
            if ordinal and k > 0 and len(vals) != k:
                out.append(_d(
                    Severity.ERROR,
                    "PROPORTIONS_LENGTH_MISMATCH",
                    f"Item '{iname}' supplies {len(vals)} category "
                    f"proportions for {k} categories.",
                    f"البند «{iname}» يقدِّم {len(vals)} نسبة فئوية في حين أن "
                    f"عدد الفئات {k}.",
                    where=iid,
                    hint_en="Supply exactly one proportion per category.",
                    hint_ar="قدِّم نسبة واحدة لكل فئة بالضبط.",
                ))
            total = sum(vals)
            if abs(total - 1.0) > 1e-6 or any(v < 0.0 for v in vals):
                out.append(_d(
                    Severity.ERROR,
                    "PROPORTIONS_DONT_SUM",
                    f"The category proportions of item '{iname}' sum to "
                    f"{total:.4f} instead of 1, or include a negative share.",
                    f"نسب الفئات للبند «{iname}» مجموعها {total:.4f} ولا يساوي "
                    f"1، أو تتضمَّن نسبة سالبة.",
                    where=iid,
                    hint_en="Rescale the proportions so they sum to 1.00, or "
                            "clear them and let the response style generate "
                            "the thresholds.",
                    hint_ar="عدِّل النسب حتى يكون مجموعها 1.00، أو اتركها "
                            "فارغة ليستنبط البرنامج الحدود الفاصلة من نمط "
                            "الاستجابة.",
                ))

        thr = getattr(sc, "thresholds", None)
        if thr:
            tvals = [_num(v) for v in thr]
            if ordinal and k > 0 and len(tvals) != k - 1:
                out.append(_d(
                    Severity.ERROR,
                    "THRESHOLDS_LENGTH_MISMATCH",
                    f"Item '{iname}' supplies {len(tvals)} thresholds; "
                    f"{k} categories need exactly {max(k - 1, 0)}.",
                    f"البند «{iname}» يقدِّم {len(tvals)} حدًّا فاصلاً "
                    f"(threshold) في حين أن {k} فئات تحتاج "
                    f"{max(k - 1, 0)} حدًّا بالضبط.",
                    where=iid,
                    hint_en="Give n_categories - 1 thresholds, or clear them.",
                    hint_ar="قدِّم عددًا من الحدود يساوي عدد الفئات ناقص واحد، "
                            "أو اتركها فارغة.",
                ))
            if any(tvals[i + 1] <= tvals[i] for i in range(len(tvals) - 1)):
                out.append(_d(
                    Severity.ERROR,
                    "THRESHOLDS_NOT_INCREASING",
                    f"The thresholds of item '{iname}' are not strictly "
                    f"increasing, so at least one category has zero or "
                    f"negative probability.",
                    f"الحدود الفاصلة (thresholds) للبند «{iname}» غير متزايدة "
                    f"تزايدًا تامًّا، ما يعني أن فئة واحدة على الأقل احتمالها "
                    f"صفر أو سالب.",
                    where=iid,
                    hint_en="Sort the thresholds in ascending order with no "
                            "ties.",
                    hint_ar="رتِّب الحدود الفاصلة تصاعديًا دون تكرار أي قيمة.",
                ))

        if not ordinal and _num(getattr(sc, "sd", 1.0)) <= 0.0:
            out.append(_d(
                Severity.ERROR,
                "CONTINUOUS_SD_NOT_POSITIVE",
                f"Continuous item '{iname}' has a reported SD of "
                f"{_num(getattr(sc, 'sd', 1.0))}; it must be positive.",
                f"البند المتصل «{iname}» له انحراف معياري مُعلَن يساوي "
                f"{_num(getattr(sc, 'sd', 1.0))}، ويجب أن يكون موجبًا.",
                where=iid,
                hint_en="Set the SD to a positive value (1.0 keeps the "
                        "z-metric).",
                hint_ar="اضبط الانحراف المعياري على قيمة موجبة (والقيمة 1.0 "
                        "تُبقي المقياس المعياري z).",
            ))


# -- distribution ----------------------------------------------------------- #


def _check_distribution(spec: ModelSpec, out: List[Diagnostic]) -> None:
    dist = getattr(spec, "distribution", None)
    clamp = bool(getattr(dist, "clamp_to_feasible_region", True)) if dist else True
    sev = Severity.WARNING if clamp else Severity.ERROR
    default_sk = _num(getattr(dist, "default_skewness", 0.0)) if dist else 0.0
    default_ku = _num(getattr(dist, "default_kurtosis", 0.0)) if dist else 0.0

    def _flag(sk: float, ku: float, where: str, what_en: str, what_ar: str) -> None:
        bound = sk * sk - 2.0
        if ku <= bound + _TOL:
            out.append(_d(
                sev,
                "INFEASIBLE_SKEW_KURTOSIS",
                f"{what_en} asks for skewness {sk:g} with excess kurtosis "
                f"{ku:g}; Fleishman's polynomial has no real solution unless "
                f"kurtosis > skewness^2 - 2 = {bound:.3f}.",
                f"{what_ar} يطلب التواءً (skewness) يساوي {sk:g} مع تفلطح "
                f"زائد (excess kurtosis) يساوي {ku:g}، ولا يوجد حل حقيقي "
                f"لمتعددة حدود فليشمان (Fleishman 1978) إلا إذا كان التفلطح "
                f"أكبر من (مربع الالتواء − 2) = {bound:.3f}.",
                where=where,
                hint_en=f"Raise the kurtosis above {bound:.3f}, or lower the "
                        f"skewness.",
                hint_ar=f"ارفع قيمة التفلطح فوق {bound:.3f}، أو اخفض قيمة "
                        f"الالتواء.",
            ))

    if default_sk or default_ku:
        _flag(default_sk, default_ku, "",
              "The global default distribution",
              "التوزيع الافتراضي العام")

    for it in _safe_items(spec):
        sc = getattr(it, "scale", None)
        if sc is not None and getattr(sc, "kind", None) is ScaleKind.LIKERT:
            continue  # shape comes from the thresholds, not Fleishman
        sk = _num(getattr(it, "skewness", 0.0)) or default_sk
        ku = _num(getattr(it, "kurtosis", 0.0)) or default_ku
        if sk == 0.0 and ku == 0.0:
            continue
        _flag(sk, ku, str(getattr(it, "id", "") or ""),
              f"Item '{_label(it)}'", f"البند «{_label(it)}»")


# -- identification / df ---------------------------------------------------- #


def _check_identification(spec: ModelSpec, out: List[Diagnostic]) -> None:
    try:
        q = int(spec.count_free_parameters())
        moments = int(spec.count_moments())
        df = int(spec.degrees_of_freedom())
    except Exception:
        return

    if df < 0:
        out.append(_d(
            Severity.ERROR,
            "NEGATIVE_DF",
            f"Degrees of freedom are negative ({df}): the model estimates "
            f"{q} free parameters from only {moments} known elements of the "
            f"covariance matrix, so it is under-identified.",
            f"درجات الحرية سالبة ({df}): فالنموذج يقدِّر {q} معلمة حرة من "
            f"{moments} عنصرًا معلومًا فقط في مصفوفة التغاير، أي أنه غير "
            f"محدَّد (under-identified).",
            hint_en="Add items, or remove paths, cross-loadings and residual "
                    "covariances until df >= 1.",
            hint_ar="أضِف بنودًا، أو احذف مسارات وتشبعات متقاطعة وتغايرات "
                    "أخطاء حتى تصبح درجات الحرية ≥ 1.",
        ))
    elif df == 0:
        out.append(_d(
            Severity.WARNING,
            "ZERO_DF",
            f"Degrees of freedom are zero: the model is just-identified, so "
            f"every fit index is degenerate (chi-square = 0, RMSEA = 0, "
            f"CFI = 1 by construction) and tells you nothing.",
            "درجات الحرية تساوي صفرًا: النموذج محدَّد تمامًا "
            "(just-identified)، ولذلك تكون مؤشرات المطابقة بلا معنى: كاي "
            "تربيع = 0 وRMSEA = 0 وCFI = 1 بحكم البناء لا بحكم جودة النموذج.",
            hint_en="Free at least one degree of freedom (drop a residual "
                    "covariance, or add an item) if you want to evaluate fit.",
            hint_ar="حرِّر درجة حرية واحدة على الأقل (بحذف تغاير خطأ أو إضافة "
                    "بند) إذا أردت تقويم مطابقة النموذج.",
        ))

    # sample size ----------------------------------------------------------- #
    power = getattr(spec, "power", None)
    n = 0
    try:
        n = int(getattr(power, "n", 0) or 0)
    except (TypeError, ValueError):
        n = 0
    need = MIN_CASES_PER_PARAMETER * q
    # The contract's condition is literally ``N < 5 * free parameters``, and
    # N = 0 or N = -50 satisfy it just as N = 134 does.  The old ``n > 0``
    # guard made the check non-monotone: tightening the sample size from 1 to
    # 0 made the warning disappear, which is exactly backwards.  A non-positive
    # N is reported under the same code (there is no sample at all), with the
    # extra sentence the researcher needs.
    if q > 0 and n < need:
        if n <= 0:
            extra_en = (" A sample size of zero or less cannot be drawn at "
                        "all, so nothing can be generated or estimated.")
            extra_ar = (" وحجم عينة يساوي صفرًا أو أقل لا يمكن سحبه أصلًا، "
                        "فلا يمكن توليد بيانات ولا تقدير النموذج.")
        else:
            extra_en = extra_ar = ""
        out.append(_d(
            Severity.WARNING,
            "SAMPLE_SIZE_TOO_SMALL",
            f"The planned sample size (N = {n}) is below five cases per free "
            f"parameter: {q} parameters require at least N = {need}."
            f"{extra_en}",
            f"حجم العينة المخطَّط (N = {n}) أقل من خمس حالات لكل معلمة حرة: "
            f"فـ {q} معلمة تتطلب {need} حالة على الأقل.{extra_ar}",
            hint_en=f"Raise N to at least {need} (the safer 10:1 rule asks "
                    f"for {10 * q}).",
            hint_ar=f"ارفع حجم العينة إلى {need} على الأقل (والقاعدة الأكثر "
                    f"تحفظًا 10 حالات لكل معلمة تطلب {10 * q} حالة).",
        ))


# -- targeting -------------------------------------------------------------- #


def _check_demographics(spec: ModelSpec, out: List[Diagnostic]) -> None:
    """Quotas that add up, columns that do not collide, controls that point somewhere."""
    variables = list(getattr(spec, "demographics", None) or [])
    if not variables:
        return
    _check_associations(spec, out)
    n = int(getattr(getattr(spec, "power", None), "n", 0) or 0)
    taken: Dict[str, str] = {}
    for item in spec.all_items():
        taken[item.name] = "item"
    for construct in spec.constructs:
        taken.setdefault(construct.name, "construct")
    targets = set(spec.control_target_ids())
    p = spec.n_items

    for variable in variables:
        where = variable.id
        name = (variable.name or "").strip()
        shown = name or (variable.label.get("en") or "?")
        if not name:
            out.append(_d(
                Severity.ERROR, "DEMO_NAME_EMPTY",
                "A demographic variable has no column name.",
                "أحد المتغيرات الديموغرافية بلا اسم عمود.",
                where=where,
                hint_en="Give it a short name such as Gender.",
                hint_ar="اكتب له اسمًا مختصرًا مثل Gender."))
        elif _sanitise_name(name) != name:
            out.append(_d(
                Severity.WARNING, "DEMO_NAME_UNSAFE",
                f"The column name '{name}' uses characters R, SPSS or Mplus may "
                "reject.",
                f"اسم العمود «{name}» يحتوي على رموز قد يرفضها R أو SPSS أو Mplus.",
                where=where,
                hint_en="Use English letters, digits and underscores.",
                hint_ar="استخدم حروفًا إنجليزية وأرقامًا وشرطة سفلية."))

        columns = [name] if name else []
        if variable.role is DemographicRole.CONTROL and variable.is_dummy_coded and name:
            columns += [term.column for term in variable.terms()]
        for column in columns:
            if column in taken:
                out.append(_d(
                    Severity.ERROR, "DEMO_NAME_DUPLICATE",
                    f"The column name '{column}' is already used in this model.",
                    f"اسم العمود «{column}» مستخدم بالفعل في هذا النموذج.",
                    where=where,
                    hint_en="Rename the variable so every column is unique.",
                    hint_ar="غيّر اسم المتغير حتى يكون كل عمود فريدًا."))
            taken[column] = "demographic"

        categories = list(variable.categories)
        if len(categories) < 2:
            out.append(_d(
                Severity.ERROR, "DEMO_TOO_FEW_CATEGORIES",
                f"'{shown}' needs at least two categories.",
                f"المتغير «{shown}» يحتاج إلى فئتين على الأقل.",
                where=where))
            continue

        total = sum(float(c.proportion) for c in categories)
        if abs(total - 1.0) > 0.005:
            out.append(_d(
                Severity.ERROR, "DEMO_PROPORTIONS_SUM",
                f"The percentages of '{shown}' add up to {total * 100:.1f}%, not "
                "100%.",
                f"نِسب فئات «{shown}» مجموعها {total * 100:.1f}% وليس 100%.",
                where=where,
                hint_en="Adjust them, or press Divide equally.",
                hint_ar="عدّلها، أو اضغط «توزيع متساوٍ»."))
        for category in categories:
            text = category.label.get("en") or category.label.get("ar")
            share = float(category.proportion)
            if share <= 0.0:
                out.append(_d(
                    Severity.ERROR, "DEMO_EMPTY_CATEGORY",
                    f"The category '{text}' of '{shown}' has a share of zero.",
                    f"الفئة «{text}» من «{shown}» نسبتها صفر.",
                    where=where,
                    hint_en="Give it a share or remove it.",
                    hint_ar="حدّد لها نسبة أو احذفها."))
            elif n and n * share < 1.0:
                out.append(_d(
                    Severity.ERROR, "DEMO_CATEGORY_BELOW_ONE",
                    f"At a sample of {n}, the category '{text}' of '{shown}' "
                    "would hold less than one respondent.",
                    f"عند عينة حجمها {n} ستضم الفئة «{text}» من «{shown}» أقل "
                    "من مستجيب واحد.",
                    where=where,
                    hint_en="Raise its share or the sample size.",
                    hint_ar="زِد نسبتها أو حجم العينة."))
            elif (n and variable.role is DemographicRole.MGA
                    and n * share <= p + 1):
                out.append(_d(
                    Severity.WARNING, "MGA_GROUP_TOO_SMALL",
                    f"The group '{text}' of '{shown}' would hold about "
                    f"{n * share:.0f} cases, too few to fit a model with {p} "
                    "items on its own, so invariance will not be tested.",
                    f"المجموعة «{text}» من «{shown}» ستضم نحو {n * share:.0f} "
                    f"حالة، وهذا أقل من اللازم لتقدير نموذج به {p} بندًا، فلن "
                    "يُختبر الثبات.",
                    where=where,
                    hint_en=f"Each group needs more than {p + 1} cases.",
                    hint_ar=f"كل مجموعة تحتاج أكثر من {p + 1} حالة."))

        if variable.role is DemographicRole.COMPARISON:
            _check_group_differences(spec, variable, shown, out)
        if variable.role is not DemographicRole.CONTROL:
            continue
        if not variable.control_paths:
            out.append(_d(
                Severity.WARNING, "CONTROL_NO_PATH",
                f"'{shown}' is a control variable with no path, so it has no "
                "effect on the model.",
                f"المتغير الضابط «{shown}» بلا مسار، فلن يؤثر في النموذج.",
                where=where,
                hint_en="Add a path to the construct it should be controlled for.",
                hint_ar="أضف مسارًا إلى المتغير المراد عزل أثره عنه."))
        valid_terms = {term.key for term in variable.terms()}
        seen = set()
        for path in variable.control_paths:
            if path.term not in valid_terms:
                out.append(_d(
                    Severity.ERROR, "CONTROL_UNKNOWN_TERM",
                    f"A path of '{shown}' starts from a category that no longer "
                    "exists.",
                    f"أحد مسارات «{shown}» يبدأ من فئة لم تعد موجودة.",
                    where=path.id))
            if path.target not in targets:
                out.append(_d(
                    Severity.ERROR, "CONTROL_BAD_TARGET",
                    f"A path of '{shown}' does not end at a dependent or "
                    "mediating construct.",
                    f"أحد مسارات «{shown}» لا ينتهي عند متغير تابع أو وسيط.",
                    where=path.id,
                    hint_en="Point it at a construct that receives a path.",
                    hint_ar="وجّهه إلى متغير يستقبل مسارًا."))
            if (path.term, path.target) in seen:
                out.append(_d(
                    Severity.ERROR, "CONTROL_DUPLICATE_PATH",
                    f"'{shown}' has the same path twice.",
                    f"المتغير «{shown}» له المسار نفسه مرتين.",
                    where=path.id))
            seen.add((path.term, path.target))
            if abs(float(path.resolved_coefficient())) >= 1.0:
                out.append(_d(
                    Severity.ERROR, "CONTROL_COEFFICIENT_RANGE",
                    f"A path of '{shown}' has a standardised coefficient of 1 "
                    "or more.",
                    f"أحد مسارات «{shown}» معامله المعياري 1 أو أكثر.",
                    where=path.id))


def _check_associations(spec: ModelSpec, out: List[Diagnostic]) -> None:
    """Links between two variables that exist, once each, within reach."""
    associations = list(getattr(spec, "demographic_associations", None) or [])
    if not associations:
        return
    from semsim.core.associations import latent_structure, reachable_v
    variables = {v.id: v for v in spec.demographics}
    seen = set()
    for association in associations:
        first = variables.get(association.first)
        second = variables.get(association.second)
        if first is None or second is None or first is second:
            out.append(_d(
                Severity.ERROR, "ASSOCIATION_BAD_VARIABLE",
                "An association does not join two different demographic "
                "variables.",
                "أحد الارتباطات لا يربط بين متغيرين ديموغرافيين مختلفين.",
                where=association.id,
                hint_en="Choose two variables or remove the association.",
                hint_ar="اختر متغيرين أو احذف الارتباط."))
            continue
        pair = frozenset((first.id, second.id))
        shown = f"{first.name} - {second.name}"
        if pair in seen:
            out.append(_d(
                Severity.ERROR, "ASSOCIATION_DUPLICATE",
                f"'{shown}' are linked twice.",
                f"المتغيران «{shown}» مربوطان مرتين.",
                where=association.id,
                hint_en="Keep one of the two associations.",
                hint_ar="أبقِ ارتباطًا واحدًا منهما."))
            continue
        seen.add(pair)
        if len(first.categories) < 2 or len(second.categories) < 2:
            continue
        wanted = association.resolved_value()
        reach = reachable_v(first.normalised_proportions(),
                            second.normalised_proportions())
        if wanted > reach + 1e-6:
            out.append(_d(
                Severity.WARNING, "ASSOCIATION_OUT_OF_REACH",
                f"With these category shares, '{shown}' can reach a Cramér's V "
                f"of about {reach:.2f}, not {wanted:.2f}.",
                f"مع نِسب الفئات الحالية يصل ارتباط «{shown}» إلى معامل V "
                f"قدره {reach:.2f} تقريبًا، وليس {wanted:.2f}.",
                where=association.id,
                hint_en="Lower the strength, or bring the shares closer to "
                        "each other.",
                hint_ar="خفّض قوة الارتباط، أو قرّب نِسب الفئات من بعضها."))
    try:
        notes = latent_structure(spec).notes
    except Exception:
        notes = []
    if notes:
        out.append(_d(
            Severity.WARNING, "ASSOCIATIONS_INCONSISTENT",
            "The associations between the demographic variables cannot all "
            "hold together, so they will be softened.",
            "لا يمكن أن تتحقق كل الارتباطات بين المتغيرات الديموغرافية معًا، "
            "فستُخفَّف.",
            hint_en="A strong link from A to B and from B to C implies a link "
                    "from A to C; avoid asking for the opposite.",
            hint_ar="الارتباط القوي بين أ وب وبين ب وج يفرض ارتباطًا بين أ "
                    "وج؛ تجنّب طلب عكسه."))


def _check_group_differences(spec: ModelSpec, variable: Any, shown: str,
                             out: List[Diagnostic]) -> None:
    """Differences that point at a construct, once each."""
    known = {c.id for c in spec.constructs if not c.covariate}
    seen = set()
    for difference in variable.group_differences:
        if difference.target not in known:
            out.append(_d(
                Severity.ERROR, "GROUP_DIFFERENCE_BAD_TARGET",
                f"A group difference of '{shown}' points at a construct that "
                "no longer exists.",
                f"أحد فروق «{shown}» يشير إلى متغير كامن لم يعد موجودًا.",
                where=difference.id,
                hint_en="Choose another construct or remove the difference.",
                hint_ar="اختر متغيرًا آخر أو احذف الفرق."))
        elif difference.target in seen:
            out.append(_d(
                Severity.ERROR, "GROUP_DIFFERENCE_DUPLICATE",
                f"'{shown}' has two differences on the same construct.",
                f"المتغير «{shown}» له فرقان على المتغير الكامن نفسه.",
                where=difference.id,
                hint_en="Keep one of them.",
                hint_ar="أبقِ واحدًا منهما."))
        seen.add(difference.target)


def _check_targeting(spec: ModelSpec, out: List[Diagnostic]) -> None:
    try:
        df = int(spec.degrees_of_freedom())
    except Exception:
        df = 1

    ft = getattr(spec, "fit_target", None)
    if ft is not None and getattr(ft, "enabled", False):
        method = str(getattr(ft, "method", "cudeck_browne") or "none")
        level = getattr(ft, "level", None)
        target = _num(getattr(ft, "rmsea", 0.0))
        try:
            if level is not None and getattr(level, "name", "") != "CUSTOM":
                target = _num(level.target_rmsea, target)
        except Exception:
            pass
        if method != "none" and target > 0.0 and df <= 0:
            out.append(_d(
                Severity.WARNING,
                "FIT_TARGET_IMPOSSIBLE_AT_ZERO_DF",
                f"A target RMSEA of {target:g} cannot be reached with "
                f"df = {df}: injected model error (Cudeck & Browne, 1992) "
                f"needs positive degrees of freedom, and at df <= 0 the fit "
                f"is perfect by construction.",
                f"لا يمكن تحقيق قيمة RMSEA المستهدفة ({target:g}) عند درجات "
                f"حرية = {df}: فحقن خطأ النموذج بطريقة Cudeck & Browne (1992) "
                f"يحتاج درجات حرية موجبة، وعند درجات حرية ≤ 0 تكون المطابقة "
                f"تامة بحكم البناء.",
                hint_en="Free degrees of freedom by removing parameters, or "
                        "set the fit-target method to 'none'.",
                hint_ar="حرِّر درجات حرية بحذف بعض المعالم، أو اختر «بلا خطأ "
                        "نموذج» في استهداف المطابقة.",
            ))
        if not (0.0 <= target < 1.0):
            out.append(_d(
                Severity.ERROR,
                "FIT_TARGET_OUT_OF_RANGE",
                f"The target RMSEA is {target}, outside the admissible "
                f"interval [0, 1).",
                f"قيمة RMSEA المستهدفة تساوي {target}، وهي خارج المدى المسموح "
                f"[0, 1).",
                hint_en="Use 0.02 (excellent) to 0.09 (marginal).",
                hint_ar="استخدم قيمة من 0.02 (ممتاز) إلى 0.09 (حدّي).",
            ))

    st = getattr(spec, "significance_target", None)
    if st is not None and getattr(st, "enabled", False):
        alpha = _num(getattr(st, "alpha", 0.05))
        if not (0.0 < alpha < 1.0):
            out.append(_d(
                Severity.ERROR,
                "ALPHA_OUT_OF_RANGE",
                f"The significance level alpha is {alpha}, outside (0, 1).",
                f"مستوى الدلالة (alpha) يساوي {alpha}، وهو خارج المدى (0, 1).",
                hint_en="Use 0.05, or 0.01 for a stricter test.",
                hint_ar="استخدم 0.05، أو 0.01 لاختبار أكثر تشدُّدًا.",
            ))


# -- noise ------------------------------------------------------------------ #


def _check_noise(spec: ModelSpec, out: List[Diagnostic]) -> None:
    noise = getattr(spec, "noise", None)
    if noise is None or not getattr(noise, "enabled", True):
        return

    mech = getattr(noise, "missing_mechanism", MissingMechanism.NONE)
    prop = _num(getattr(noise, "missing_proportion", 0.0))
    if mech is not MissingMechanism.NONE or prop != 0.0:
        if not (0.0 <= prop < 1.0):
            out.append(_d(
                Severity.ERROR,
                "MISSING_PROPORTION_RANGE",
                f"The requested proportion of missing data is {prop}, outside "
                f"the admissible interval [0, 1).",
                f"نسبة البيانات المفقودة المطلوبة تساوي {prop}، وهي خارج "
                f"المدى المسموح [0, 1).",
                hint_en="Choose a proportion between 0 and 0.30; above that "
                        "the model can no longer be estimated reliably.",
                hint_ar="اختر نسبة فقدان بين 0 و0.30؛ فما فوق ذلك يتعذَّر "
                        "تقدير النموذج بثقة.",
            ))
        elif mech is MissingMechanism.NONE and prop > 0.0:
            out.append(_d(
                Severity.WARNING,
                "MISSING_PROPORTION_RANGE",
                f"A missing-data proportion of {prop} was requested but the "
                f"mechanism is 'none', so no values will be deleted.",
                f"طُلبت نسبة فقدان {prop} لكن آلية الفقدان مضبوطة على «بلا»، "
                f"ولذلك لن تُحذف أي قيمة فعليًا.",
                hint_en="Pick MCAR or MAR, or set the proportion back to 0.",
                hint_ar="اختر آلية MCAR أو MAR، أو أعِد النسبة إلى الصفر.",
            ))

    if mech is MissingMechanism.MAR:
        driver = str(getattr(noise, "mar_driver", "") or "")
        if driver and driver not in _construct_lookup(spec):
            out.append(_d(
                Severity.ERROR,
                "MAR_DRIVER_UNKNOWN",
                f"MAR missingness is driven by construct '{driver}', which is "
                f"not in the model.",
                f"آلية الفقدان MAR مبنية على المتغير الكامن «{driver}» وهو غير "
                f"موجود في النموذج.",
                hint_en="Pick an existing construct, or leave the driver "
                        "empty to use the first exogenous construct.",
                hint_ar="اختر متغيرًا كامنًا موجودًا، أو اترك الحقل فارغًا "
                        "لاستخدام أول متغير خارجي.",
            ))

    for attr, label_en, label_ar in (
        ("careless_proportion", "careless responders", "المستجيبين غير المبالين"),
        ("outlier_proportion", "outlier cells", "الخلايا الشاذة"),
    ):
        v = _num(getattr(noise, attr, 0.0))
        if not (0.0 <= v < 1.0):
            out.append(_d(
                Severity.ERROR,
                "NOISE_PROPORTION_RANGE",
                f"The proportion of {label_en} is {v}, outside [0, 1).",
                f"نسبة {label_ar} تساوي {v}، وهي خارج المدى [0, 1).",
                hint_en="Use 0 to 0.10; published carelessness rates rarely "
                        "exceed 10%.",
                hint_ar="استخدم نسبة بين 0 و0.10، فالنسب المنشورة لا تتجاوز "
                        "عادة 10%.",
            ))

    infl = _num(getattr(noise, "unique_variance_inflation", 1.0), 1.0)
    if infl <= 0.0:
        out.append(_d(
            Severity.ERROR,
            "UNIQUE_VARIANCE_INFLATION_RANGE",
            f"The unique-variance inflation factor is {infl}; it must be "
            f"positive (1.0 leaves the unique variances untouched).",
            f"معامل تضخيم التباين الفريد يساوي {infl}، ويجب أن يكون موجبًا "
            f"(والقيمة 1.0 تترك التباينات الفريدة كما هي).",
            hint_en="Use 1.0 for no attenuation, up to about 1.5 for weak "
                    "measurement.",
            hint_ar="استخدم 1.0 لعدم التوهين، وحتى نحو 1.5 لمحاكاة قياس ضعيف.",
        ))


# -- small spec accessors that must not raise ------------------------------- #


def _safe_items(spec: ModelSpec) -> List[Any]:
    try:
        return list(spec.all_items())
    except Exception:  # pragma: no cover - duck-typed spec
        return []


def _safe_factor_ids(spec: ModelSpec) -> List[str]:
    try:
        return [str(x) for x in spec.all_factor_ids()]
    except Exception:  # pragma: no cover - duck-typed spec
        return []


def _safe_home_factor(spec: ModelSpec, item_id: str) -> Optional[str]:
    try:
        home = spec.home_factor_of(item_id)
    except Exception:  # pragma: no cover - duck-typed spec
        return None
    return str(home) if home else None


# --------------------------------------------------------------------------- #
# Post-compilation validation
# --------------------------------------------------------------------------- #


def validate_compiled(cm: "CompiledModel") -> ValidationReport:
    """Check the matrices a :func:`semsim.core.compile.compile_model` produced.

    These are the checks that cannot be made on the specification alone,
    because they depend on the realised ``Phi``, ``Theta`` and ``Sigma``:

    ``HEYWOOD_CASE``
        A unique variance in ``Theta`` or a disturbance variance in ``Psi``
        that is non-positive, or that sits on the compiler's clamping floor
        (``HEYWOOD_FLOOR``) and therefore *was* a Heywood case.
    ``SIGMA_NOT_PD``
        ``Sigma = Lambda Phi Lambda' + Theta`` is not positive definite, so no
        sample can be drawn from it.
    ``R2_EXCEEDS_ONE``
        An endogenous latent whose explained variance ``b' Phi_PP b`` exceeds
        one.
    ``SIGMA_ILL_CONDITIONED``
        ``cond(Sigma) > 1e10``: near-singular, usually a pair of latents
        correlated so highly that discriminant validity is gone.
    ``LOW_COMMUNALITY``
        An item whose communality is below 0.20, i.e. one that measures almost
        nothing.

    Parameters
    ----------
    cm : CompiledModel
        The compiled model.  Duck-typed: only ``Lambda``, ``Phi``, ``Theta``,
        ``Psi``, ``B``, ``endogenous``, ``item_names``/``item_ids``,
        ``factor_names``/``factor_ids`` and ``warnings`` are read, and every
        one of them is optional.

    Returns
    -------
    ValidationReport
        The post-compilation findings.  Like :func:`validate`, this function
        never raises; a ``numpy.linalg.LinAlgError`` becomes a diagnostic.

    References
    ----------
    Heywood, H. B. (1931). On finite sequences of real numbers.
    *Proceedings of the Royal Society A*, 134, 486-501.
    Higham, N. J. (2002). Computing the nearest correlation matrix - a problem
    from finance. *IMA Journal of Numerical Analysis*, 22, 329-343.
    Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2010).
    *Multivariate Data Analysis* (7th ed.), ch. 3 (communality floor). Pearson.
    Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion for
    assessing discriminant validity. *Journal of the Academy of Marketing
    Science*, 43, 115-135.
    """
    out: List[Diagnostic] = []
    try:
        _check_compiled(cm, out)
    except Exception as exc:  # pragma: no cover - last-resort safety net
        out.append(_d(
            Severity.ERROR,
            "VALIDATOR_FAILED",
            f"The compiled model could not be inspected: {exc!r}.",
            f"تعذَّر فحص النموذج بعد بنائه: {exc!r}.",
            hint_en="Recompile the model; if it persists, report the spec.",
            hint_ar="أعِد بناء النموذج، وإذا تكرَّر الخطأ فأبلِغ عن المواصفة.",
        ))
    return ValidationReport(out)


def _as_matrix(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        if arr.ndim == 2 and arr.size:
            return arr
        return None
    return arr


def _names(cm: Any, *attrs: str, size: int = 0, prefix: str = "V") -> List[str]:
    for attr in attrs:
        vals = getattr(cm, attr, None)
        if vals is not None:
            try:
                out = [str(v) for v in vals]
            except TypeError:
                continue
            if len(out) >= size:
                return out
    return [f"{prefix}{i + 1}" for i in range(size)]


def _check_compiled(cm: Any, out: List[Diagnostic]) -> None:
    Lambda = _as_matrix(getattr(cm, "Lambda", None))
    Phi = _as_matrix(getattr(cm, "Phi", None))
    Theta = _as_matrix(getattr(cm, "Theta", None))
    Psi = _as_matrix(getattr(cm, "Psi", None))
    B = _as_matrix(getattr(cm, "B", None))

    p = int(Theta.shape[0]) if Theta is not None else (
        int(Lambda.shape[0]) if Lambda is not None else 0)
    m = int(Phi.shape[0]) if Phi is not None else (
        int(Lambda.shape[1]) if Lambda is not None else 0)
    item_names = _names(cm, "item_names", "item_ids", size=p, prefix="Y")
    factor_names = _names(cm, "factor_names", "factor_ids", size=m, prefix="F")
    # ``Diagnostic.where`` is declared in docs/ARCHITECTURE.md as the
    # construct / item / path *id*, and the UI selects the offending object by
    # it.  A display name there never matches anything (two items may even
    # share a name), so the id goes in ``where`` and the name stays in the
    # message, where a human reads it.
    item_keys = _names(cm, "item_ids", "item_names", size=p, prefix="Y")
    factor_keys = _names(cm, "factor_ids", "factor_names", size=m, prefix="F")

    # -- compiler warnings surfaced as INFO -------------------------------- #
    for w in list(getattr(cm, "warnings", None) or []):
        out.append(_d(
            Severity.INFO,
            "COMPILE_WARNING",
            f"Compiler note: {w}",
            f"ملاحظة من مُجمِّع النموذج: {w}",
            hint_en="The compiler adjusted the specification to keep every "
                    "variance admissible.",
            hint_ar="عدَّل المُجمِّع المواصفة تلقائيًا للحفاظ على تباينات "
                    "مقبولة.",
        ))

    # -- Heywood cases realised in Theta ----------------------------------- #
    if Theta is not None:
        theta_d = np.diag(Theta)
        for j in range(min(p, theta_d.size)):
            v = float(theta_d[j])
            if v <= HEYWOOD_FLOOR * (1.0 + 1e-6):
                name = item_names[j] if j < len(item_names) else f"Y{j + 1}"
                key = item_keys[j] if j < len(item_keys) else name
                out.append(_d(
                    Severity.ERROR,
                    "HEYWOOD_CASE",
                    f"Item '{name}' ended up with a unique variance of "
                    f"{v:.6f}: a Heywood case that the compiler had to clamp "
                    f"to the floor of {HEYWOOD_FLOOR:g}.",
                    f"انتهى البند «{name}» بتباين فريد (unique variance) يساوي "
                    f"{v:.6f}، وهي حالة هيوود (Heywood case) محقَّقة فعلاً "
                    f"اضطر المُجمِّع إلى تثبيتها عند الحد الأدنى "
                    f"{HEYWOOD_FLOOR:g}.",
                    where=key,
                    hint_en="Lower this item's loading (or its cross-loadings "
                            "and the method loading) so its communality stays "
                            "below 1.",
                    hint_ar="اخفض تشبع هذا البند (أو تشبعاته المتقاطعة وتشبع "
                            "عامل الطريقة) حتى يبقى شيوعه أقل من 1.",
                ))

    # -- Heywood cases realised in Psi (latent disturbances) --------------- #
    endog = getattr(cm, "endogenous", None)
    endo_mask: Optional[np.ndarray]
    try:
        endo_mask = np.asarray(endog, dtype=bool) if endog is not None else None
    except (TypeError, ValueError):
        endo_mask = None
    if Psi is not None:
        psi_d = np.diag(Psi)
        for i in range(min(m, psi_d.size)):
            if endo_mask is not None and i < endo_mask.size and not bool(endo_mask[i]):
                continue
            v = float(psi_d[i])
            if v <= HEYWOOD_FLOOR * (1.0 + 1e-6):
                name = factor_names[i] if i < len(factor_names) else f"F{i + 1}"
                key = factor_keys[i] if i < len(factor_keys) else name
                out.append(_d(
                    Severity.ERROR,
                    "HEYWOOD_CASE",
                    f"Latent variable '{name}' ended up with a disturbance "
                    f"variance of {v:.6f}: a structural Heywood case, i.e. "
                    f"its predictors explain (almost) all of its variance.",
                    f"انتهى المتغير الكامن «{name}» بتباين بواقٍ (disturbance "
                    f"variance) يساوي {v:.6f}، وهي حالة هيوود على المستوى "
                    f"البنائي: أي أن متنبئاته تفسِّر تباينه كله تقريبًا.",
                    where=key,
                    hint_en="Lower the path coefficients entering this latent "
                            "variable.",
                    hint_ar="اخفض معاملات المسارات الداخلة إلى هذا المتغير "
                            "الكامن.",
                ))

    # -- R^2 above one ----------------------------------------------------- #
    if B is not None and Phi is not None and B.shape[0] == Phi.shape[0]:
        for i in range(B.shape[0]):
            if endo_mask is not None and i < endo_mask.size and not bool(endo_mask[i]):
                continue
            b = B[i]
            if not np.any(b):
                continue
            r2 = float(b @ Phi @ b)
            if r2 > 1.0 + 1e-8:
                name = factor_names[i] if i < len(factor_names) else f"F{i + 1}"
                key = factor_keys[i] if i < len(factor_keys) else name
                out.append(_d(
                    Severity.ERROR,
                    "R2_EXCEEDS_ONE",
                    f"The realised R-squared of latent variable '{name}' is "
                    f"{r2:.4f}, above 1; its disturbance variance is "
                    f"negative.",
                    f"معامل التحديد R² المحقَّق للمتغير الكامن «{name}» يساوي "
                    f"{r2:.4f} أي أكبر من 1، ومن ثَمَّ يكون تباين البواقي "
                    f"سالبًا.",
                    where=key,
                    hint_en="Lower the incoming path coefficients, or reduce "
                            "the correlations among the predictors.",
                    hint_ar="اخفض معاملات المسارات الداخلة، أو اخفض الارتباطات "
                            "بين المتغيرات المتنبئة.",
                ))

    # -- Sigma: positive definiteness, conditioning, communalities --------- #
    if Lambda is None or Phi is None or Theta is None:
        return
    if Lambda.shape[1] != Phi.shape[0] or Lambda.shape[0] != Theta.shape[0]:
        out.append(_d(
            Severity.ERROR,
            "COMPILED_SHAPE_MISMATCH",
            f"The compiled matrices do not line up: Lambda is "
            f"{Lambda.shape}, Phi is {Phi.shape}, Theta is {Theta.shape}.",
            f"أبعاد مصفوفات النموذج المبني غير متوافقة: Lambda بأبعاد "
            f"{Lambda.shape}، وPhi بأبعاد {Phi.shape}، وTheta بأبعاد "
            f"{Theta.shape}.",
            hint_en="Recompile the model from the specification.",
            hint_ar="أعِد بناء النموذج من المواصفة.",
        ))
        return

    try:
        comm = Lambda @ Phi @ Lambda.T
        Sigma = comm + Theta
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals = np.linalg.eigvalsh(Sigma)
    except np.linalg.LinAlgError as exc:
        out.append(_d(
            Severity.ERROR,
            "SIGMA_NOT_PD",
            f"The implied covariance matrix Sigma could not be decomposed "
            f"({exc}), so it cannot be used to generate data.",
            f"تعذَّر تحليل المصفوفة التغايرية الضمنية Σ ({exc})، فلا يمكن "
            f"استخدامها في توليد البيانات.",
            hint_en="Repair Sigma with implied.nearest_positive_definite, or "
                    "lower the latent correlations.",
            hint_ar="أصلِح المصفوفة Σ بأقرب مصفوفة معرَّفة موجبة، أو اخفض "
                    "الارتباطات بين المتغيرات الكامنة.",
        ))
        return

    lo = float(eigvals.min())
    hi = float(eigvals.max())
    if lo <= 1e-10:
        out.append(_d(
            Severity.ERROR,
            "SIGMA_NOT_PD",
            f"The implied covariance matrix Sigma is not positive definite: "
            f"its smallest eigenvalue is {lo:.3e}. No sample can be drawn "
            f"from it until it is repaired.",
            f"المصفوفة التغايرية الضمنية Σ غير معرَّفة موجبة (not positive "
            f"definite): أصغر قيمة ذاتية لها {lo:.3e}، ولا يمكن سحب عينة "
            f"منها قبل إصلاحها.",
            hint_en="Lower the latent correlations and the residual "
                    "covariances, or let the compiler project Sigma onto the "
                    "nearest positive-definite matrix (Higham, 2002).",
            hint_ar="اخفض الارتباطات بين المتغيرات الكامنة وتغايرات الأخطاء، "
                    "أو اسمح للمُجمِّع بإسقاط Σ على أقرب مصفوفة معرَّفة موجبة "
                    "(Higham, 2002).",
        ))
    else:
        cond = hi / lo
        if cond > CONDITION_NUMBER_LIMIT:
            out.append(_d(
                Severity.WARNING,
                "SIGMA_ILL_CONDITIONED",
                f"The condition number of the implied matrix Sigma is "
                f"{cond:.3e}, above the {CONDITION_NUMBER_LIMIT:.0e} limit: "
                f"Sigma is near-singular, which usually means two latent "
                f"variables are correlated so highly that discriminant "
                f"validity is gone.",
                f"رقم شرط (condition number) المصفوفة الضمنية Σ يساوي "
                f"{cond:.3e}، وهو يتجاوز الحد {CONDITION_NUMBER_LIMIT:.0e}؛ "
                f"أي أن المصفوفة تكاد تكون منفردة (near-singular)، وغالبًا "
                f"يرجع ذلك إلى ارتباط شديد بين متغيرين كامنين يُفقد معه الصدق "
                f"التمايزي (discriminant validity).",
                hint_en="Lower the latent correlations below 0.85, or merge "
                        "the two constructs that overlap.",
                hint_ar="اخفض الارتباطات بين المتغيرات الكامنة إلى ما دون "
                        "0.85، أو ادمج المتغيرين المتداخلين في متغير واحد.",
            ))

    comm_d = np.diag(comm)
    for j in range(min(p, comm_d.size)):
        c = float(comm_d[j])
        if c < LOW_COMMUNALITY_FLOOR:
            name = item_names[j] if j < len(item_names) else f"Y{j + 1}"
            key = item_keys[j] if j < len(item_keys) else name
            out.append(_d(
                Severity.WARNING,
                "LOW_COMMUNALITY",
                f"Item '{name}' has a communality of {c:.3f}, below the "
                f"{LOW_COMMUNALITY_FLOOR:.2f} floor: the factors explain less "
                f"than {100 * LOW_COMMUNALITY_FLOOR:.0f}% of its variance, so "
                f"it measures almost nothing.",
                f"شيوع البند «{name}» (communality) يساوي {c:.3f}، وهو أقل من "
                f"الحد {LOW_COMMUNALITY_FLOOR:.2f}: أي أن العوامل تفسِّر أقل "
                f"من {100 * LOW_COMMUNALITY_FLOOR:.0f}% من تباين البند، فهو "
                f"لا يكاد يقيس شيئًا.",
                where=key,
                hint_en="Raise this item's loading above 0.45, or drop the "
                        "item from the scale.",
                hint_ar="ارفع تشبع هذا البند فوق 0.45، أو احذف البند من "
                        "المقياس.",
            ))


__all__ = [
    "Severity",
    "Diagnostic",
    "ValidationReport",
    "validate",
    "validate_compiled",
    "MIN_ITEMS_PER_FACTOR",
    "MIN_DIMENSIONS_SECOND_ORDER",
    "LIKERT_MIN_CATEGORIES",
    "LIKERT_MAX_CATEGORIES",
    "MIN_CASES_PER_PARAMETER",
    "LOW_COMMUNALITY_FLOOR",
    "CONDITION_NUMBER_LIMIT",
    "HEYWOOD_FLOOR",
]

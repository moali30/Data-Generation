"""
semsim.ui.widgets.path_diagram
==============================
A drawn picture of the model: constructs, dimensions, items and the paths
between them.

Conventions a reader of SEM diagrams expects, and which this honours:

*   Latent variables are ellipses, observed items are rectangles.  That is not
    decoration -- it is the notation, and a diagram that gets it wrong is
    harder to read than no diagram.
*   Single-headed arrows are regressions, double-headed arcs are covariances,
    and a moderation is drawn as an arrow that lands *on the path it
    moderates*, with a dot where it lands.  Moderators therefore sit in a band
    above or below the model, next to that path, instead of in the column of
    predictors where their arrow would lie on top of the path it modifies.
*   Coefficients sit on their arrow, left-to-right, never mirrored.
*   Flow follows the interface direction by default -- left to right in
    English, right to left in Arabic -- with a toggle, because conventions in
    published Arabic work differ and the researcher's own journal wins.

Built on QGraphicsScene so panning, zooming and PNG/SVG export come for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGraphicsItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from semsim.core.spec import Construct, ConstructRole, ModelSpec
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.widgets.common import Notice, TButton, TCheckBox

# Geometry, in scene units.
CONSTRUCT_MIN_W, CONSTRUCT_MAX_W = 178.0, 240.0
CONSTRUCT_H1, CONSTRUCT_H2 = 76.0, 94.0   # one-line / two-line label
#: Kept for callers that size things off a "typical" construct.
CONSTRUCT_W, CONSTRUCT_H = CONSTRUCT_MIN_W, CONSTRUCT_H1
DIMENSION_W, DIMENSION_H = 118.0, 46.0
ITEM_W, ITEM_H = 56.0, 24.0
CONTROL_W, CONTROL_H = 132.0, 40.0
COLUMN_GAP = 190.0        # clear space between two layers of constructs
DIM_GAP = 64.0            # construct to its dimensions
DIM_V_GAP = 22.0          # between sibling dimensions
ITEM_LINK_GAP = 48.0      # factor to its indicator column
STACK_GAP = 26.0          # factor to an indicator row above or below it
NODE_GAP = 40.0
ITEM_GAP = 6.0
BAND_GAP = 92.0           # model to the moderator band above or below it
BAND_SPACING = 36.0       # between two moderators in the same band
LABEL_PAD = 52.0          # ellipse width not available to its label

#: Where along a path (0 = predictor, 1 = outcome) a moderation arrow may
#: land, in order of preference.  0.5 is left free for the path's own label.
LANDING_SLOTS = (0.70, 0.32, 0.84, 0.18, 0.60, 0.42)


@dataclass
class _Node:
    key: str
    label: str
    kind: str                 # "construct" | "dimension" | "item" | "control"
    role: Optional[ConstructRole] = None
    layer: int = 0
    order: int = 0
    x: float = 0.0
    y: float = 0.0
    w: float = CONSTRUCT_W
    h: float = CONSTRUCT_H
    #: Short code shown under the label ("PU").
    code: str = ""
    #: Colour family: "exo", "endo" or "moderator".
    tone: str = "exo"

    @property
    def rect(self) -> QRectF:
        return QRectF(self.x - self.w / 2, self.y - self.h / 2, self.w, self.h)

    @property
    def centre(self) -> QPointF:
        return QPointF(self.x, self.y)


@dataclass
class _Edge:
    source: str
    target: str
    label: str = ""
    kind: str = "path"        # "path" | "covariance" | "loading" | "interaction"
    strength: float = 0.0
    dashed: bool = False
    #: A moderator's own effect on the outcome it moderates: drawn quieter, so
    #: the moderation arrow -- the hypothesis -- is what the eye follows.
    muted: bool = False
    #: Interactions only: the point on the moderated path the arrow lands on.
    anchor: Optional[QPointF] = None


@dataclass
class DiagramLayout:
    nodes: Dict[str, _Node]
    edges: List[_Edge]
    bounds: QRectF
    direction: float = 1.0
    legend: Optional[QRectF] = None
    legend_rows: List[Tuple[str, str]] = field(default_factory=list)
    #: Extra reach for same-column covariance arcs, to clear an indicator
    #: column hanging off the side they are drawn on.
    arc_clearance: float = 0.0

    def __iter__(self):
        # Old callers unpack ``nodes, edges, bounds``.
        return iter((self.nodes, self.edges, self.bounds))


# --------------------------------------------------------------------------- #
# Fonts and sizing
# --------------------------------------------------------------------------- #


def _label_font() -> QFont:
    font = QFont()
    font.setPointSizeF(10.5)
    font.setBold(True)
    return font


def _sub_font() -> QFont:
    font = QFont()
    font.setPointSizeF(8.0)
    return font


def _construct_size(labels: Sequence[str]) -> Tuple[float, float]:
    """One ellipse size for every construct, fitted to the longest label.

    A uniform size is what published diagrams use, and it keeps columns
    straight.  Labels that still do not fit on one line wrap to two, and every
    ellipse grows to the two-line height so none looks squeezed.
    """
    metrics = QFontMetricsF(_label_font())
    widths = [metrics.horizontalAdvance(text) for text in labels if text]
    widest = max(widths, default=0.0)
    width = min(max(widest + LABEL_PAD + 12, CONSTRUCT_MIN_W), CONSTRUCT_MAX_W)
    wraps = any(w > width - LABEL_PAD for w in widths)
    return width, (CONSTRUCT_H2 if wraps else CONSTRUCT_H1)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def _assign_layers(spec: ModelSpec) -> Dict[str, int]:
    """Longest-path layering over the structural graph.

    Cycles cannot be layered; the loop breaks after ``n`` relaxation passes and
    whatever depth each node reached is used, which still draws something
    sensible for a non-recursive model.
    """
    layer = {c.id: 0 for c in spec.constructs}
    edges = [(p.source, p.target) for p in spec.paths
             if p.source in layer and p.target in layer]
    for _ in range(len(layer) + 1):
        changed = False
        for source, target in edges:
            if layer[target] < layer[source] + 1:
                layer[target] = layer[source] + 1
                changed = True
        if not changed:
            break
    # A control sits one column before the construct it adjusts, so its arrow
    # is short instead of crossing the whole model.
    for construct in spec.constructs:
        if construct.covariate:
            targets = [layer[target] for source, target in edges
                       if source == construct.id]
            if targets:
                layer[construct.id] = max(min(targets) - 1, 0)
    return layer


def _banded_moderators(spec: ModelSpec) -> Set[str]:
    """Moderators that leave the columns for a band above or below.

    Only a pure moderator qualifies: exogenous, not itself the predictor of
    an interaction, and moderating a path whose two ends are both drawn.
    Anything else has a place in the flow and keeps it.
    """
    ids = {c.id: c for c in spec.constructs}
    has_incoming = {p.target for p in spec.paths}
    predictors = {ix.predictor for ix in spec.interactions}
    out: Set[str] = set()
    for ix in spec.interactions:
        moderator = ids.get(ix.moderator)
        if (moderator is None or moderator.covariate
                or ix.moderator in has_incoming or ix.moderator in predictors
                or ix.predictor not in ids or ix.target not in ids
                or ix.predictor == ix.moderator):
            continue
        out.add(ix.moderator)
    return out


def build_layout(spec: ModelSpec, *, show_items: bool,
                 rtl: bool) -> DiagramLayout:
    """Lay the model out in layers along the flow direction.

    Indicators follow the convention of published path diagrams: they hang off
    the OUTER edge for the first and last layers, and sit above and below the
    construct for the layers in between.  Putting every indicator on one side
    would park them squarely across the structural arrows that have to reach
    the next layer.

    Within a column each construct sits level with the constructs that feed
    it, so a mediator lines up between its predictors and its outcome instead
    of wherever the order of entry happened to put it.
    """
    lang = translator.language
    spec = spec.expanded()
    layers = _assign_layers(spec)
    banded = _banded_moderators(spec)
    moderator_ids = banded | {c.id for c in spec.constructs
                              if c.role is ConstructRole.MODERATOR}
    endogenous = {p.target for p in spec.paths}

    core = [c for c in spec.constructs if c.id not in banded]
    by_layer: Dict[int, List[Construct]] = {}
    for construct in core:
        by_layer.setdefault(layers.get(construct.id, 0), []).append(construct)

    labels = [c.label.get(lang) or c.name for c in spec.constructs
              if not c.covariate]
    node_w, node_h = _construct_size(labels)

    nodes: Dict[str, _Node] = {}
    edges: List[_Edge] = []
    direction = -1.0 if rtl else 1.0
    layer_indices = sorted(by_layer)
    first_layer = layer_indices[0] if layer_indices else 0
    last_layer = layer_indices[-1] if layer_indices else 0

    def side_for(layer_index: int) -> str:
        """Where this layer's indicators go."""
        if not show_items:
            return "none"
        if layer_index == first_layer:
            return "behind"          # outer edge, upstream of the flow
        if layer_index == last_layer:
            return "ahead"           # outer edge, downstream of the flow
        return "stacked"             # above and below, clear of the arrows

    # -- horizontal: how much room each layer needs, then cumulative x ------ #
    widths: Dict[int, Tuple[float, float]] = {}      # layer -> (behind, ahead)
    for layer_index in layer_indices:
        second_order = any(c.is_second_order for c in by_layer[layer_index])
        side = side_for(layer_index)
        satellite = 0.0
        if second_order:
            satellite += DIM_GAP + DIMENSION_W
        if side in ("behind", "ahead"):
            satellite += ITEM_LINK_GAP + ITEM_W
        behind = satellite if side == "behind" or (
            second_order and side != "ahead") else 0.0
        ahead = satellite if side == "ahead" else 0.0
        if second_order and side == "stacked":
            behind, ahead = DIM_GAP + DIMENSION_W, 0.0
        widths[layer_index] = (behind, ahead)

    x_of: Dict[int, float] = {}
    cursor = 0.0
    for position, layer_index in enumerate(layer_indices):
        behind, ahead = widths[layer_index]
        if position:
            cursor += direction * (behind + node_w / 2)
        x_of[layer_index] = cursor
        cursor += direction * (node_w / 2 + ahead + COLUMN_GAP)

    # -- vertical: level with the constructs that feed each one ------------ #
    feeders: Dict[str, List[str]] = {}
    for path in spec.paths:
        feeders.setdefault(path.target, []).append(path.source)
    y_of: Dict[str, float] = {}
    for layer_index in layer_indices:
        column = by_layer[layer_index]
        side = side_for(layer_index)
        blocks = {c.id: _block_height(c, side, node_h) for c in column}
        y_of.update(_place_column(column, blocks, feeders, y_of))

    for layer_index in layer_indices:
        side = side_for(layer_index)
        for construct in by_layer[layer_index]:
            x = x_of[layer_index]
            y = y_of[construct.id]
            label = construct.label.get(lang) or construct.name
            if construct.covariate:
                nodes[construct.id] = _Node(
                    key=construct.id, label=label, kind="control",
                    layer=layer_index, x=x, y=y, w=CONTROL_W, h=CONTROL_H)
                continue
            nodes[construct.id] = _Node(
                key=construct.id, label=label, kind="construct",
                role=construct.role, layer=layer_index, x=x, y=y,
                w=node_w, h=node_h, code=_code(construct, label),
                tone=_tone(construct.id, moderator_ids, endogenous))

            if construct.is_second_order:
                _place_dimensions(nodes, edges, construct, x, y, node_w,
                                  direction, side, show_items, lang)
            elif show_items:
                _place_item_group(nodes, edges, construct.items, construct.id,
                                  x, y, node_w, node_h, direction, side)

    # -- moderators: in a band, beside the path each one moderates ---------- #
    anchors = _place_moderators(spec, nodes, edges, banded, node_w, node_h,
                                show_items, lang, moderator_ids, endogenous)

    moderated: Dict[str, Set[str]] = {}
    for ix in spec.interactions:
        moderated.setdefault(ix.moderator, set()).add(ix.target)
    for path in spec.paths:
        if path.source in nodes and path.target in nodes:
            coefficient = path.resolved_coefficient()
            edges.append(_Edge(path.source, path.target,
                               f"{coefficient:+.2f}", kind="path",
                               strength=abs(coefficient),
                               dashed=not path.target_significant,
                               muted=path.target in moderated.get(
                                   path.source, ())))
    for cov in spec.latent_covariances:
        # Controls may covary freely; drawing every pair would bury the model.
        if nodes.get(cov.construct_a) and nodes[cov.construct_a].kind == "control":
            continue
        if nodes.get(cov.construct_b) and nodes[cov.construct_b].kind == "control":
            continue
        if cov.construct_a in nodes and cov.construct_b in nodes:
            edges.append(_Edge(cov.construct_a, cov.construct_b,
                               f"{cov.correlation:+.2f}", kind="covariance",
                               strength=abs(cov.correlation)))
    for ix in spec.interactions:
        if ix.moderator in nodes and ix.target in nodes:
            coefficient = ix.resolved_coefficient()
            edges.append(_Edge(ix.moderator, ix.target, f"{coefficient:+.2f}",
                               kind="interaction", strength=abs(coefficient),
                               dashed=not ix.target_significant,
                               anchor=anchors.get(ix.id)))

    layout = DiagramLayout(nodes=nodes, edges=edges, bounds=QRectF(),
                           direction=direction,
                           arc_clearance=(ITEM_LINK_GAP + ITEM_W + 6.0
                                          if show_items else 0.0))
    _place_legend(layout)
    layout.bounds = _bounds(layout)
    return layout


def _code(construct: Construct, label: str) -> str:
    """The short name shown under the label, when it says something new."""
    name = (construct.name or "").strip()
    return "" if not name or name == label else name


def _tone(key: str, moderators: Set[str], endogenous: Set[str]) -> str:
    if key in moderators:
        return "moderator"
    if key in endogenous:
        return "endo"
    return "exo"


def _place_column(column: List[Construct], blocks: Dict[str, float],
                  feeders: Dict[str, List[str]],
                  placed: Dict[str, float]) -> Dict[str, float]:
    """Vertical centres for one column.

    Each construct wants to sit at the mean height of the constructs feeding
    it; one fed by nothing already placed keeps its stacking slot.  Wishes are
    then honoured in order with the minimum spacing enforced, and the column
    is shifted as a whole so it moved as little as possible overall.
    """
    total = sum(blocks.values()) + NODE_GAP * max(len(column) - 1, 0)
    top = -total / 2.0
    wish: Dict[str, float] = {}
    for construct in column:
        height = blocks[construct.id]
        stacked = top + height / 2.0
        top += height + NODE_GAP
        sources = [placed[s] for s in feeders.get(construct.id, ())
                   if s in placed]
        wish[construct.id] = (sum(sources) / len(sources)) if sources else stacked

    order = sorted(column, key=lambda c: wish[c.id])
    out: Dict[str, float] = {}
    bottom = -math.inf
    for construct in order:
        half = blocks[construct.id] / 2.0
        y = max(wish[construct.id], bottom + NODE_GAP + half)
        out[construct.id] = y
        bottom = y + half
    if out:
        shift = sum(out[k] - wish[k] for k in out) / len(out)
        for key in out:
            out[key] -= shift
    return out


def _place_moderators(spec: ModelSpec, nodes: Dict[str, _Node],
                      edges: List[_Edge], banded: Set[str], node_w: float,
                      node_h: float, show_items: bool, lang: str,
                      moderator_ids: Set[str],
                      endogenous: Set[str]) -> Dict[str, QPointF]:
    """Put each banded moderator over (or under) the path it moderates.

    The arrow lands on the path itself, at a slot clear of the path's label
    and of other moderators' arrows, and drops straight down (or up) to it.
    A slot whose drop would cut through a construct is skipped.  Returns the
    landing point of every interaction, keyed by its id.
    """
    anchors: Dict[str, QPointF] = {}
    if not banded:
        return anchors
    core = list(nodes.values())
    if core:
        top = min(n.rect.top() for n in core)
        bottom = max(n.rect.bottom() for n in core)
        constructs = [n for n in core if n.kind == "construct"] or core
        middle = sum(n.y for n in constructs) / len(constructs)
    else:
        top = bottom = middle = 0.0
    item_room = (STACK_GAP + ITEM_H) if show_items else 0.0
    top_y = top - BAND_GAP - node_h / 2 - item_room
    bottom_y = bottom + BAND_GAP + node_h / 2 + item_room
    obstacles = [n.rect.adjusted(-8, -8, 8, 8) for n in core
                 if n.kind != "item"]

    used: Dict[Tuple[str, str], List[float]] = {}
    bands: Dict[str, List[Tuple[str, float]]] = {"top": [], "bottom": []}
    ids = {c.id: c for c in spec.constructs}
    placed: Set[str] = set()

    for ix in spec.interactions:
        if ix.moderator not in banded:
            continue
        predictor, target = nodes.get(ix.predictor), nodes.get(ix.target)
        if predictor is None or target is None:
            continue
        start = _edge_point(predictor, target.centre)
        end = _edge_point(target, predictor.centre)
        segment = (ix.predictor, ix.target)
        taken = used.setdefault(segment, [])
        if ix.moderator in placed:
            # A second interaction of a moderator already placed: land it on
            # its own path, from wherever the moderator already is.
            t = next((s for s in LANDING_SLOTS if s not in taken),
                     LANDING_SLOTS[0])
            taken.append(t)
            anchors[ix.id] = _lerp(start, end, t)
            continue

        choice: Optional[Tuple[float, QPointF, str]] = None
        for t in LANDING_SLOTS:
            if t in taken:
                continue
            point = _lerp(start, end, t)
            band = "top" if point.y() <= middle else "bottom"
            for side in (band, "bottom" if band == "top" else "top"):
                edge_y = top if side == "top" else bottom
                drop = QLineF(point.x(), point.y(), point.x(), edge_y)
                if not any(_segment_hits(drop, r) for r in obstacles):
                    choice = (t, point, side)
                    break
            if choice is not None:
                break
        if choice is None:
            t = next((s for s in LANDING_SLOTS if s not in taken),
                     LANDING_SLOTS[0])
            point = _lerp(start, end, t)
            choice = (t, point, "top" if point.y() <= middle else "bottom")
        t, point, side = choice
        taken.append(t)
        anchors[ix.id] = point
        bands[side].append((ix.moderator, point.x()))
        placed.add(ix.moderator)

    # A moderator whose paths could not be drawn still has to appear.
    for key in sorted(banded - placed, key=lambda k: list(ids).index(k)):
        bands["top"].append((key, 0.0))

    for side, members in bands.items():
        members.sort(key=lambda m: m[1])
        xs = [x for _key, x in members]
        # Keep neighbours apart, then centre each run on its arrows.
        for i in range(1, len(xs)):
            xs[i] = max(xs[i], xs[i - 1] + node_w + BAND_SPACING)
        if xs:
            drift = sum(x - m[1] for x, m in zip(xs, members)) / len(xs)
            xs = [x - drift for x in xs]
        y = top_y if side == "top" else bottom_y
        for (key, _wanted), x in zip(members, xs):
            construct = ids[key]
            label = construct.label.get(lang) or construct.name
            nodes[key] = _Node(
                key=key, label=label, kind="construct", role=construct.role,
                x=x, y=y, w=node_w, h=node_h, code=_code(construct, label),
                tone=_tone(key, moderator_ids, endogenous))
            if show_items and construct.items:
                row_y = (y - node_h / 2 - STACK_GAP - ITEM_H / 2 if side == "top"
                         else y + node_h / 2 + STACK_GAP + ITEM_H / 2)
                _place_items_row(nodes, edges, construct.items, x, row_y, key)
    return anchors


def _lerp(a: QPointF, b: QPointF, t: float) -> QPointF:
    return QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)


def _segment_hits(line: QLineF, rect: QRectF) -> bool:
    """Whether a segment passes through a rectangle (Liang-Barsky clip)."""
    x0, y0, x1, y1 = line.x1(), line.y1(), line.x2(), line.y2()
    dx, dy = x1 - x0, y1 - y0
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, x0 - rect.left()), (dx, rect.right() - x0),
                 (-dy, y0 - rect.top()), (dy, rect.bottom() - y0)):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            lo = max(lo, r)
        else:
            hi = min(hi, r)
        if lo > hi:
            return False
    return True


def _block_height(construct: Construct, side: str,
                  node_h: float = CONSTRUCT_H) -> float:
    """Vertical room one construct needs, indicators included."""
    if construct.covariate:
        return CONTROL_H
    item_run = ITEM_H + ITEM_GAP
    if construct.is_second_order:
        dims = construct.dimensions or [None]
        if side in ("behind", "ahead"):
            per = [max(len(d.items) * item_run if d else 0.0,
                       DIMENSION_H + 14) for d in dims]
        else:
            per = [DIMENSION_H + 14 + (STACK_GAP + item_run
                                       if side == "stacked" and d and d.items
                                       else 0.0) for d in dims]
        return max(sum(per) + DIM_V_GAP * (len(per) - 1), node_h)
    if side in ("behind", "ahead"):
        return max(len(construct.items) * item_run, node_h)
    if side == "stacked" and construct.items:
        return node_h + 2 * (STACK_GAP + ITEM_H)
    return node_h


def _place_dimensions(nodes, edges, construct: Construct, x: float, y: float,
                      node_w: float, direction: float, side: str,
                      show_items: bool, lang: str) -> None:
    dims = construct.dimensions
    if not dims:
        return
    dim_x = x - direction * (node_w / 2 + DIM_GAP + DIMENSION_W / 2)
    item_run = ITEM_H + ITEM_GAP
    if show_items and side in ("behind", "ahead"):
        heights = [max(len(d.items) * item_run, DIMENSION_H + 14) for d in dims]
    elif show_items and side == "stacked":
        heights = [DIMENSION_H + 14 + (STACK_GAP + item_run if d.items else 0)
                   for d in dims]
    else:
        heights = [DIMENSION_H + 14 for d in dims]
    total = sum(heights) + DIM_V_GAP * (len(dims) - 1)
    top = y - total / 2.0
    for dimension, height in zip(dims, heights):
        dim_y = top + height / 2.0
        top += height + DIM_V_GAP
        nodes[dimension.id] = _Node(
            key=dimension.id,
            label=dimension.label.get(lang) or dimension.name,
            kind="dimension", x=dim_x, y=dim_y,
            w=DIMENSION_W, h=DIMENSION_H)
        edges.append(_Edge(construct.id, dimension.id,
                           f"{dimension.second_order_loading:.2f}",
                           kind="loading"))
        if show_items:
            _place_item_group(nodes, edges, dimension.items, dimension.id,
                              dim_x, dim_y, DIMENSION_W, DIMENSION_H,
                              direction, side)


def _place_item_group(nodes, edges, items, parent_key: str, x: float, y: float,
                      parent_w: float, parent_h: float, direction: float,
                      side: str) -> None:
    """Indicators in a column beside the factor, or split above and below it."""
    if not items:
        return
    if side == "behind":
        _place_items(nodes, edges, items,
                     x - direction * (parent_w / 2 + ITEM_LINK_GAP + ITEM_W / 2),
                     y, parent_key)
    elif side == "ahead":
        _place_items(nodes, edges, items,
                     x + direction * (parent_w / 2 + ITEM_LINK_GAP + ITEM_W / 2),
                     y, parent_key)
    else:
        half = (len(items) + 1) // 2
        _place_items_row(nodes, edges, items[:half], x,
                         y - parent_h / 2 - STACK_GAP - ITEM_H / 2, parent_key)
        _place_items_row(nodes, edges, items[half:], x,
                         y + parent_h / 2 + STACK_GAP + ITEM_H / 2, parent_key)


def _place_items_row(nodes, edges, items, centre_x: float, y: float,
                     parent_key: str) -> None:
    if not items:
        return
    span = len(items) * (ITEM_W + ITEM_GAP)
    cursor = centre_x - span / 2.0 + ITEM_W / 2.0
    for item in items:
        nodes[item.id] = _Node(key=item.id, label=item.name, kind="item",
                               x=cursor, y=y, w=ITEM_W, h=ITEM_H)
        edges.append(_Edge(parent_key, item.id, f"{item.loading:.2f}",
                           kind="loading"))
        cursor += ITEM_W + ITEM_GAP


def _place_items(nodes: Dict[str, _Node], edges: List[_Edge], items,
                 x: float, centre_y: float, parent_key: str) -> None:
    if not items:
        return
    span = len(items) * (ITEM_H + ITEM_GAP)
    cursor = centre_y - span / 2.0 + ITEM_H / 2.0
    for item in items:
        nodes[item.id] = _Node(key=item.id, label=item.name, kind="item",
                               x=x, y=cursor, w=ITEM_W, h=ITEM_H)
        edges.append(_Edge(parent_key, item.id, f"{item.loading:.2f}",
                           kind="loading"))
        cursor += ITEM_H + ITEM_GAP


# -- legend and bounds ------------------------------------------------------ #

LEGEND_ROW_H = 26.0
LEGEND_SAMPLE_W = 46.0
LEGEND_PAD = 14.0


def _legend_font() -> QFont:
    font = QFont()
    font.setPointSizeF(8.5)
    return font


def _place_legend(layout: DiagramLayout) -> None:
    """A key in the downstream top corner, listing only what is drawn."""
    kinds = {e.kind for e in layout.edges}
    rows: List[Tuple[str, str]] = []
    if any(e.kind == "path" and not e.dashed and not e.muted
           for e in layout.edges):
        rows.append(("path", "s.diagram.lg.path"))
    if any(e.kind == "path" and e.dashed for e in layout.edges):
        rows.append(("null", "s.diagram.lg.null"))
    if "interaction" in kinds:
        rows.append(("interaction", "s.diagram.lg.moderation"))
    if any(e.kind == "path" and e.muted and not e.dashed for e in layout.edges):
        rows.append(("main", "s.diagram.lg.main"))
    if "covariance" in kinds:
        rows.append(("covariance", "s.diagram.lg.covariance"))
    if len(rows) < 2 or not layout.nodes:
        layout.legend, layout.legend_rows = None, []
        return

    metrics = QFontMetricsF(_legend_font())
    title = QFont(_legend_font())
    title.setBold(True)
    text_w = max([metrics.horizontalAdvance(translator.t(key))
                  for _kind, key in rows]
                 + [QFontMetricsF(title).horizontalAdvance(
                     translator.t("s.diagram.legend"))])
    width = LEGEND_PAD * 2 + LEGEND_SAMPLE_W + 12 + text_w
    height = LEGEND_PAD * 2 + LEGEND_ROW_H * (len(rows) + 1)

    rects = [n.rect for n in layout.nodes.values()]
    top = min(r.top() for r in rects)
    if layout.direction < 0:
        left = min(r.left() for r in rects)
    else:
        left = max(r.right() for r in rects) - width
    box = QRectF(left, top, width, height)
    if any(box.adjusted(-18, -18, 18, 18).intersects(r) for r in rects):
        box.moveTop(top - height - 36)
    layout.legend, layout.legend_rows = box, rows


def _bounds(layout: DiagramLayout) -> QRectF:
    if not layout.nodes:
        return QRectF(-200, -120, 400, 240)
    bounds: Optional[QRectF] = None
    for node in layout.nodes.values():
        bounds = node.rect if bounds is None else bounds.united(node.rect)
    assert bounds is not None
    for edge in layout.edges:
        if edge.kind == "covariance":
            a, b = layout.nodes[edge.source], layout.nodes[edge.target]
            path, apex = _covariance_path(a, b, layout.direction,
                                          _centroid(layout.nodes),
                                          layout.arc_clearance)
            bounds = bounds.united(path.boundingRect())
            bounds = bounds.united(QRectF(apex.x() - 34, apex.y() - 14, 68, 28))
    if layout.legend is not None:
        bounds = bounds.united(layout.legend)
    return bounds.adjusted(-50, -50, 50, 50)


def _centroid(nodes: Dict[str, _Node]) -> QPointF:
    """Middle of the construct nodes, used to bow covariance arcs outward."""
    points = [(n.x, n.y) for n in nodes.values() if n.kind == "construct"]
    if not points:
        return QPointF(0.0, 0.0)
    return QPointF(sum(p[0] for p in points) / len(points),
                   sum(p[1] for p in points) / len(points))


# --------------------------------------------------------------------------- #
# Painting
# --------------------------------------------------------------------------- #


class _DiagramScene(QGraphicsScene):
    """Paints the whole diagram in ``drawBackground``.

    One painter pass beats hundreds of QGraphicsItems here: the diagram is
    redrawn on every edit to a loading, and item churn at that rate is visible.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = DiagramLayout(nodes={}, edges=[], bounds=QRectF())
        self._theme = theme.LIGHT

    @property
    def _nodes(self) -> Dict[str, _Node]:
        return self._layout.nodes

    @property
    def _edges(self) -> List[_Edge]:
        return self._layout.edges

    def set_model(self, layout, edges: Optional[List[_Edge]] = None,
                  bounds: Optional[QRectF] = None) -> None:
        if not isinstance(layout, DiagramLayout):
            # The old three-argument form: (nodes, edges, bounds).
            layout = DiagramLayout(nodes=layout, edges=edges or [],
                                   bounds=bounds or QRectF())
        self._layout = layout
        self.setSceneRect(layout.bounds)
        self.update()

    def set_theme(self, name: str) -> None:
        self._theme = name
        self.update()

    # -- drawing ---------------------------------------------------------- #

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        pal = theme.palette(self._theme)
        painter.fillRect(rect, QColor(pal.diagram_bg))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self._draw_grid(painter, rect, pal)

        labels: List[Tuple[List[QPointF], str, bool]] = []
        order = {"loading": 0, "covariance": 1, "path": 2, "interaction": 3}
        for edge in sorted(self._edges, key=lambda e: order.get(e.kind, 2)):
            spot = self._draw_edge(painter, edge, pal)
            if spot is not None:
                labels.append(spot)
        for node in self._nodes.values():
            self._draw_node(painter, node, pal)

        # Labels last, so no line or ellipse covers a number, and each one
        # slides along its edge until it no longer sits on another.
        taken: List[QRectF] = [n.rect.adjusted(2, 2, -2, -2)
                               for n in self._nodes.values()
                               if n.kind in ("construct", "control",
                                             "dimension")]
        for candidates, text, small in labels:
            _draw_label(painter, pal, candidates, text, small, taken)
        if self._layout.legend is not None:
            self._draw_legend(painter, pal)

    def _draw_grid(self, painter: QPainter, rect: QRectF, pal) -> None:
        pen = QPen(QColor(pal.diagram_grid))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        step = 40.0
        x = math.floor(rect.left() / step) * step
        while x < rect.right():
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
            x += step
        y = math.floor(rect.top() / step) * step
        while y < rect.bottom():
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
            y += step

    def _draw_node(self, painter: QPainter, node: _Node, pal) -> None:
        rect = node.rect
        if node.kind in ("item", "control"):
            fill, border = pal.diagram_item, pal.diagram_edge
        elif node.kind == "dimension":
            fill, border = pal.diagram_dimension, pal.diagram_node_border
        elif node.tone == "moderator":
            fill, border = pal.diagram_moderator, pal.diagram_moderator_border
        elif node.tone == "endo":
            fill, border = pal.diagram_node_endo, pal.diagram_node_border
        else:
            fill, border = pal.diagram_node, pal.diagram_node_border
        painter.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border))
        pen.setWidthF(2.4 if node.kind == "construct" else 1.4)
        painter.setPen(pen)
        if node.kind in ("item", "control"):
            # Observed variables are rectangles; latent ones are ellipses.
            painter.drawRoundedRect(rect, 3, 3)
        else:
            painter.drawEllipse(rect)

        if node.kind != "construct":
            painter.setPen(QPen(QColor(pal.diagram_node_text)))
            font = QFont()
            font.setPointSizeF(8.5 if node.kind == "item" else 9.5)
            painter.setFont(font)
            flags = int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap)
            painter.drawText(rect.adjusted(3, 1, -3, -1), flags,
                             _ltr(node.label) if node.kind == "item"
                             else node.label)
            return
        self._draw_construct_text(painter, node, pal)

    def _draw_construct_text(self, painter: QPainter, node: _Node,
                             pal) -> None:
        """Name in bold, and "code · role" under it, centred as one block
        inside the ellipse -- never on its outline."""
        rect = node.rect
        text_w = rect.width() - LABEL_PAD
        label_font = _label_font()
        wrap = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                   | Qt.TextFlag.TextWordWrap)
        label_h = QFontMetricsF(label_font).boundingRect(
            QRectF(0, 0, text_w, 1000), wrap, node.label).height()
        subtitle = _subtitle(node)
        sub_font = _sub_font()
        sub_h = QFontMetricsF(sub_font).height() if subtitle else 0.0
        gap = 3.0 if subtitle else 0.0
        top = node.y - (label_h + gap + sub_h) / 2.0

        painter.setFont(label_font)
        painter.setPen(QPen(QColor(pal.diagram_node_text)))
        painter.drawText(QRectF(node.x - text_w / 2, top, text_w, label_h),
                         wrap, node.label)
        if subtitle:
            painter.setFont(sub_font)
            painter.setPen(QPen(QColor(pal.text_muted)))
            painter.drawText(QRectF(rect.left(), top + label_h + gap,
                                    rect.width(), sub_h),
                             int(Qt.AlignmentFlag.AlignCenter), subtitle)

    def _draw_edge(self, painter: QPainter, edge: _Edge, pal
                   ) -> Optional[Tuple[List[QPointF], str, bool]]:
        """Draw one connector; return where its label may go."""
        source = self._nodes.get(edge.source)
        target = self._nodes.get(edge.target)
        if source is None or target is None:
            return None
        if edge.kind == "loading":
            colour = QColor(pal.diagram_edge)
            width = 1.2
        elif edge.kind == "interaction":
            colour = QColor(pal.diagram_moderator_border)
            width = 2.2
        elif edge.dashed:
            colour = QColor(pal.diagram_edge_null)
            width = 1.6
        elif edge.muted:
            colour = QColor(pal.diagram_edge_null)
            width = 1.4
        else:
            colour = QColor(pal.diagram_edge_strong)
            width = 1.8 + 1.2 * min(edge.strength, 0.8)

        pen = QPen(colour)
        pen.setWidthF(width)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        if edge.dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if edge.kind == "covariance":
            path, apex = _covariance_path(source, target,
                                          self._layout.direction,
                                          _centroid(self._nodes),
                                          self._layout.arc_clearance)
            painter.drawPath(path)
            _arrow_head(painter, colour, path.pointAtPercent(0.04),
                        path.pointAtPercent(0.0), size=9.0)
            _arrow_head(painter, colour, path.pointAtPercent(0.96),
                        path.pointAtPercent(1.0), size=9.0)
            return [apex], edge.label, False

        if edge.kind == "interaction" and edge.anchor is not None:
            # The arrow lands on the moderated path, marked with a dot.
            anchor = edge.anchor
            start = _edge_point(source, anchor)
            length = QLineF(start, anchor).length() or 1.0
            stop = _lerp(start, anchor, max(0.0, 1.0 - 5.0 / length))
            painter.drawLine(QLineF(start, stop))
            _arrow_head(painter, colour, start, stop)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(anchor, 4.5, 4.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            return ([_lerp(start, anchor, t) for t in (0.5, 0.38, 0.62)],
                    edge.label, False)

        start = _edge_point(source, target.centre)
        end = _edge_point(target, source.centre)
        painter.drawLine(QLineF(start, _lerp(start, end, 0.985)))
        _arrow_head(painter, colour, start, end,
                    size=7.0 if edge.kind == "loading" else 11.0)
        if not edge.label:
            return None
        if edge.kind == "loading":
            return [_lerp(start, end, 0.5)], edge.label, True
        return ([_lerp(start, end, t) for t in (0.5, 0.42, 0.58, 0.34, 0.66,
                                                0.26, 0.74)],
                edge.label, edge.muted)

    def _draw_legend(self, painter: QPainter, pal) -> None:
        box = self._layout.legend
        assert box is not None
        painter.setBrush(QBrush(QColor(pal.diagram_bg)))
        pen = QPen(QColor(pal.diagram_label_border))
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.drawRoundedRect(box, 6, 6)

        rtl = translator.is_rtl
        align = (Qt.AlignmentFlag.AlignRight if rtl
                 else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter
        inner = box.adjusted(LEGEND_PAD, LEGEND_PAD, -LEGEND_PAD, -LEGEND_PAD)
        title = QFont(_legend_font())
        title.setBold(True)
        painter.setFont(title)
        painter.setPen(QPen(QColor(pal.text)))
        painter.drawText(QRectF(inner.left(), inner.top(), inner.width(),
                                LEGEND_ROW_H), int(align),
                         translator.t("s.diagram.legend"))
        painter.setFont(_legend_font())
        for index, (kind, key) in enumerate(self._layout.legend_rows):
            y = inner.top() + LEGEND_ROW_H * (index + 1)
            mid = y + LEGEND_ROW_H / 2
            if rtl:
                sample_l = inner.left()
                text_rect = QRectF(sample_l + LEGEND_SAMPLE_W + 12, y,
                                   inner.width() - LEGEND_SAMPLE_W - 12,
                                   LEGEND_ROW_H)
            else:
                sample_l = inner.right() - LEGEND_SAMPLE_W
                text_rect = QRectF(inner.left(), y,
                                   inner.width() - LEGEND_SAMPLE_W - 12,
                                   LEGEND_ROW_H)
            a = QPointF(sample_l, mid)
            b = QPointF(sample_l + LEGEND_SAMPLE_W, mid)
            self._legend_sample(painter, pal, kind, a, b)
            painter.setPen(QPen(QColor(pal.text)))
            painter.setFont(_legend_font())
            painter.drawText(text_rect, int(align), translator.t(key))

    def _legend_sample(self, painter: QPainter, pal, kind: str, a: QPointF,
                       b: QPointF) -> None:
        colours = {"path": pal.diagram_edge_strong,
                   "null": pal.diagram_edge_null,
                   "main": pal.diagram_edge_null,
                   "interaction": pal.diagram_moderator_border,
                   "covariance": pal.diagram_edge_strong}
        colour = QColor(colours[kind])
        pen = QPen(colour)
        pen.setWidthF(2.2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        if kind == "null":
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind == "covariance":
            path = QPainterPath(a)
            path.quadTo(QPointF((a.x() + b.x()) / 2, a.y() - 16), b)
            painter.drawPath(path)
            _arrow_head(painter, colour, path.pointAtPercent(0.1), a, size=8.0)
            _arrow_head(painter, colour, path.pointAtPercent(0.9), b, size=8.0)
            return
        # Arrows point the way the diagram flows.
        start, end = (b, a) if self._layout.direction < 0 else (a, b)
        painter.drawLine(QLineF(start, _lerp(start, end, 0.9)))
        _arrow_head(painter, colour, start, end, size=10.0)
        if kind == "interaction":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(start, 3.5, 3.5)
            painter.setBrush(Qt.BrushStyle.NoBrush)


def _subtitle(node: _Node) -> str:
    """ "PU · مستقل" -- the code on the left, the role on the right."""
    role = (translator.enum("role", node.role)
            if node.role is not None else "")
    if node.code and role:
        if translator.is_rtl:
            # Embed right-to-left so the role reads first, on the right.
            return "‫" + role + " · " + node.code + "‬"
        return f"{node.code} · {role}"
    return node.code or role


def _edge_point(node: _Node, towards: QPointF) -> QPointF:
    """Where a line from ``towards`` meets this node's outline."""
    dx = towards.x() - node.x
    dy = towards.y() - node.y
    if dx == 0 and dy == 0:
        return node.centre
    if node.kind in ("item", "control"):
        # Rectangle: clip to the nearer edge.
        scale_x = (node.w / 2) / abs(dx) if dx else float("inf")
        scale_y = (node.h / 2) / abs(dy) if dy else float("inf")
        scale = min(scale_x, scale_y)
    else:
        # Ellipse: solve for the parametric intersection.
        a, b = node.w / 2, node.h / 2
        scale = 1.0 / math.sqrt((dx / a) ** 2 + (dy / b) ** 2)
    return QPointF(node.x + dx * scale, node.y + dy * scale)


def _arrow_head(painter: QPainter, colour: QColor, start: QPointF,
                end: QPointF, size: float = 11.0) -> None:
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    tip = end
    spread = math.pi / 8
    left = QPointF(tip.x() - size * math.cos(angle - spread),
                   tip.y() - size * math.sin(angle - spread))
    right = QPointF(tip.x() - size * math.cos(angle + spread),
                    tip.y() - size * math.sin(angle + spread))
    painter.setBrush(QBrush(colour))
    painter.setPen(QPen(colour))
    painter.drawPolygon(QPolygonF([tip, left, right]))
    painter.setBrush(Qt.BrushStyle.NoBrush)


def _covariance_path(a: _Node, b: _Node, direction: float,
                     centre: QPointF,
                     clearance: float = 0.0) -> Tuple[QPainterPath, QPointF]:
    """A two-headed arc and the apex where its label goes.

    Two constructs in the same column are joined on their upstream side --
    the side no structural arrow leaves from -- by an arc that bows outward
    in proportion to how far apart they are, so the arcs of a whole column
    nest instead of crossing.  Constructs elsewhere get a plain bowed arc,
    curving away from the middle of the model.
    """
    if abs(a.x - b.x) < 1.0:
        side = -direction or 1.0
        upper, lower = (a, b) if a.y <= b.y else (b, a)
        # With an indicator column on this side, leave from near the bottom
        # of the ellipse and swing out through the gap below the indicators.
        angle = math.radians(66 if clearance else 28)

        def anchor(node: _Node, downward: bool) -> QPointF:
            return QPointF(node.x + side * (node.w / 2) * math.cos(angle),
                           node.y + (1 if downward else -1)
                           * (node.h / 2) * math.sin(angle))

        start = anchor(upper, True)
        end = anchor(lower, False)
        reach = 34.0 + clearance + 0.10 * abs(lower.y - upper.y)
        apex_x = upper.x + side * (upper.w / 2 + reach)
        mid_y = (start.y() + end.y()) / 2
        path = QPainterPath(start)
        if clearance:
            gap_top = upper.y + upper.h / 2 + 8
            gap_bottom = lower.y - lower.h / 2 - 8
            path.cubicTo(QPointF(start.x(), gap_top + 18),
                         QPointF(apex_x, gap_top),
                         QPointF(apex_x, mid_y))
            path.cubicTo(QPointF(apex_x, gap_bottom),
                         QPointF(end.x(), gap_bottom - 18), end)
        else:
            mid_x = (start.x() + end.x()) / 2
            path.quadTo(QPointF(2 * apex_x - mid_x, mid_y), end)
        if a.y > b.y:
            path = path.toReversed()
        return path, QPointF(apex_x, mid_y)
    path, apex = _arc(_edge_point(a, b.centre), _edge_point(b, a.centre),
                      away_from=centre)
    return path, apex


def _arc(start: QPointF, end: QPointF,
         away_from: Optional[QPointF] = None) -> Tuple[QPainterPath, QPointF]:
    """A bowed two-headed connector, with its mid-point for the label.

    ``away_from`` is a point the bow should curve away from -- normally the
    centre of the diagram, so a covariance between two constructs stacked in
    the same column arcs out into empty space instead of across the nodes it
    connects.
    """
    path = QPainterPath(start)
    dx, dy = end.x() - start.x(), end.y() - start.y()
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    bulge = min(120.0, max(36.0, length * 0.22))
    mid = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
    if away_from is not None:
        # Flip the normal if it points back towards the thing we are avoiding.
        if (nx * (mid.x() - away_from.x())
                + ny * (mid.y() - away_from.y())) < 0:
            nx, ny = -nx, -ny
    control = QPointF(mid.x() + nx * bulge * 2.0, mid.y() + ny * bulge * 2.0)
    path.quadTo(control, end)
    return path, QPointF(mid.x() + nx * bulge, mid.y() + ny * bulge)


#: U+202D LEFT-TO-RIGHT OVERRIDE ... U+202C POP DIRECTIONAL FORMATTING.
#: A signed coefficient is a number, not prose: "+0.30" must never render as
#: "0.30+" because the surrounding paragraph happens to be Arabic.
def _ltr(text: str) -> str:
    return "‭" + text + "‬"


def _draw_label(painter: QPainter, pal, candidates, text: str,
                small: bool = False,
                taken: Optional[List[QRectF]] = None) -> None:
    """A coefficient in a small box, at the first candidate spot that does
    not cover another label or a construct."""
    if isinstance(candidates, QPointF):
        candidates = [candidates]
    text = _ltr(text)
    font = QFont()
    font.setPointSizeF(7.5 if small else 9.0)
    font.setBold(not small)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    width = metrics.horizontalAdvance(text) + (8 if small else 14)
    height = metrics.height() + (2 if small else 6)

    def box_at(point: QPointF) -> QRectF:
        return QRectF(point.x() - width / 2, point.y() - height / 2,
                      width, height)

    box = box_at(candidates[0])
    if taken is not None:
        for point in candidates:
            trial = box_at(point)
            if not any(trial.intersects(r) for r in taken):
                box = trial
                break
        taken.append(box.adjusted(-3, -3, 3, 3))

    painter.setBrush(QBrush(QColor(pal.diagram_bg)))
    if small:
        painter.setPen(Qt.PenStyle.NoPen)
    else:
        pen = QPen(QColor(pal.diagram_label_border))
        pen.setWidthF(1.0)
        painter.setPen(pen)
    painter.drawRoundedRect(box, 4, 4)
    painter.setPen(QPen(QColor(pal.text)))
    # Coefficients are numbers: they never mirror, whatever the interface does.
    painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)


# --------------------------------------------------------------------------- #
# View
# --------------------------------------------------------------------------- #


class _ZoomView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # The diagram is a figure, not prose: it must not mirror.
        theme.keep_ltr(self)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class PathDiagramView(QWidget):
    """The diagram tab: a toolbar plus the drawing."""

    def __init__(self, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._theme = theme.LIGHT

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        self._show_items = TCheckBox("s.diagram.show_items")
        self._show_items.setChecked(False)
        self._show_items.toggled.connect(lambda _c: self.refresh())
        bar.addWidget(self._show_items)
        self._mirror = TCheckBox("s.diagram.mirror")
        self._mirror.setChecked(translator.is_rtl)
        self._mirror.toggled.connect(lambda _c: self.refresh())
        bar.addWidget(self._mirror)
        bar.addStretch(1)
        self._notice = Notice()
        bar.addWidget(self._notice)
        self._fit = TButton("s.diagram.fit", role="flat")
        self._fit.clicked.connect(self.fit)
        bar.addWidget(self._fit)
        self._export = TButton("s.diagram.export", role="flat")
        self._export.clicked.connect(self.export_image)
        bar.addWidget(self._export)
        lay.addLayout(bar)

        self._fit_pending = True
        self.scene = _DiagramScene(self)
        self.view = _ZoomView(self.scene, self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Expanding)
        lay.addWidget(self.view, 1)

        translator.language_changed.connect(lambda _l: self.refresh())
        self.refresh()

    # -- api --------------------------------------------------------------- #

    def set_theme(self, name: str) -> None:
        self._theme = name
        self.scene.set_theme(name)

    def refresh(self) -> None:
        layout = build_layout(
            self.state.spec,
            show_items=self._show_items.isChecked(),
            rtl=self._mirror.isChecked())
        self.scene.set_model(layout)
        self.fit()

    def fit(self) -> None:
        rect = self.scene.sceneRect()
        if not rect.isValid():
            return
        # fitInView against a viewport that has not been laid out yet yields a
        # near-zero transform and a diagram the size of a postage stamp, so
        # wait until the viewport has real pixels.
        viewport = self.view.viewport()
        if viewport is None or viewport.width() < 32 or viewport.height() < 32:
            self._fit_pending = True
            return
        self._fit_pending = False
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        # Never blow a small model up past life size.
        scale = self.view.transform().m11()
        if scale > 1.4:
            self.view.resetTransform()
            self.view.scale(1.4, 1.4)
            self.view.centerOn(rect.center())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(self, "_fit_pending", True):
            self.fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_fit_pending", False):
            self.fit()

    def export_image(self, path: str = "") -> str:
        """Render the diagram to a PNG at twice the on-screen resolution."""
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, translator.t("act.export"), "model.png",
                "PNG image (*.png)")
        if not path:
            return ""
        from PySide6.QtGui import QImage
        rect = self.scene.sceneRect()
        scale = 2.0
        image = QImage(int(rect.width() * scale), int(rect.height() * scale),
                       QImage.Format.Format_ARGB32)
        image.fill(QColor(theme.palette(self._theme).diagram_bg))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.scene.render(painter, QRectF(image.rect()), rect)
        painter.end()
        if not image.save(path):
            self._notice.show_key("s.diagram.save_failed", role="error")
            return ""
        self._notice.show_key("s.diagram.saved")
        return path


__all__ = ["PathDiagramView", "build_layout", "DiagramLayout"]

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
    and a moderation is drawn as an arrow that lands *on another arrow*.
*   Coefficients sit on their arrow, left-to-right, never mirrored.
*   Flow follows the interface direction by default -- left to right in
    English, right to left in Arabic -- with a toggle, because conventions in
    published Arabic work differ and the researcher's own journal wins.

Built on QGraphicsScene so panning, zooming and PNG/SVG export come for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGraphicsItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from semsim.core.spec import Construct, ConstructRole, ModelSpec
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.widgets.common import TButton, TCheckBox

# Geometry, in scene units.
CONSTRUCT_W, CONSTRUCT_H = 148.0, 62.0
DIMENSION_W, DIMENSION_H = 118.0, 46.0
ITEM_W, ITEM_H = 56.0, 24.0
LAYER_GAP = 230.0
NODE_GAP = 34.0
ITEM_GAP = 6.0


@dataclass
class _Node:
    key: str
    label: str
    kind: str                 # "construct" | "dimension" | "item"
    role: Optional[ConstructRole] = None
    layer: int = 0
    order: int = 0
    x: float = 0.0
    y: float = 0.0
    w: float = CONSTRUCT_W
    h: float = CONSTRUCT_H

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
    return layer


def build_layout(spec: ModelSpec, *, show_items: bool,
                 rtl: bool) -> Tuple[Dict[str, _Node], List[_Edge], QRectF]:
    lang = translator.language
    layers = _assign_layers(spec)
    by_layer: Dict[int, List[Construct]] = {}
    for construct in spec.constructs:
        by_layer.setdefault(layers.get(construct.id, 0), []).append(construct)

    nodes: Dict[str, _Node] = {}
    edges: List[_Edge] = []
    direction = -1.0 if rtl else 1.0

    for layer_index in sorted(by_layer):
        column = by_layer[layer_index]
        # Height of this column, so it can be centred vertically.
        heights = []
        for construct in column:
            if construct.is_second_order:
                block = max(len(construct.dimensions), 1) * (DIMENSION_H + NODE_GAP)
                if show_items:
                    block = sum(
                        max(len(d.items), 1) * (ITEM_H + ITEM_GAP) + NODE_GAP
                        for d in construct.dimensions) or block
                heights.append(max(block, CONSTRUCT_H))
            else:
                block = (len(construct.items) * (ITEM_H + ITEM_GAP)
                         if show_items else CONSTRUCT_H)
                heights.append(max(block, CONSTRUCT_H))
        total = sum(heights) + NODE_GAP * max(len(column) - 1, 0)
        cursor = -total / 2.0

        for construct, block_height in zip(column, heights):
            x = direction * layer_index * LAYER_GAP
            y = cursor + block_height / 2.0
            cursor += block_height + NODE_GAP
            label = construct.label.get(lang) or construct.name
            nodes[construct.id] = _Node(
                key=construct.id, label=label, kind="construct",
                role=construct.role, layer=layer_index, x=x, y=y)

            if construct.is_second_order:
                dims = construct.dimensions
                span = len(dims) * (DIMENSION_H + NODE_GAP)
                dim_cursor = y - span / 2.0 + DIMENSION_H / 2.0
                for dimension in dims:
                    dim_x = x + direction * (LAYER_GAP * 0.52)
                    dim_label = dimension.label.get(lang) or dimension.name
                    nodes[dimension.id] = _Node(
                        key=dimension.id, label=dim_label, kind="dimension",
                        layer=layer_index, x=dim_x, y=dim_cursor,
                        w=DIMENSION_W, h=DIMENSION_H)
                    edges.append(_Edge(construct.id, dimension.id,
                                       f"{dimension.second_order_loading:.2f}",
                                       kind="loading"))
                    if show_items:
                        _place_items(nodes, edges, dimension.items,
                                     dim_x + direction * (LAYER_GAP * 0.42),
                                     dim_cursor, dimension.id)
                    dim_cursor += DIMENSION_H + NODE_GAP
            elif show_items:
                _place_items(nodes, edges, construct.items,
                             x + direction * (LAYER_GAP * 0.5), y, construct.id)

    for path in spec.paths:
        if path.source in nodes and path.target in nodes:
            coefficient = path.resolved_coefficient()
            edges.append(_Edge(path.source, path.target,
                               f"{coefficient:+.2f}", kind="path",
                               strength=abs(coefficient),
                               dashed=not path.target_significant))
    for cov in spec.latent_covariances:
        if cov.construct_a in nodes and cov.construct_b in nodes:
            edges.append(_Edge(cov.construct_a, cov.construct_b,
                               f"{cov.correlation:+.2f}", kind="covariance",
                               strength=abs(cov.correlation)))
    for interaction in spec.interactions:
        if interaction.moderator in nodes and interaction.target in nodes:
            edges.append(_Edge(interaction.moderator, interaction.target,
                               f"{interaction.resolved_coefficient():+.2f}",
                               kind="interaction",
                               strength=abs(interaction.resolved_coefficient())))

    if nodes:
        xs = [n.rect for n in nodes.values()]
        bounds = xs[0]
        for r in xs[1:]:
            bounds = bounds.united(r)
        bounds = bounds.adjusted(-60, -60, 60, 60)
    else:
        bounds = QRectF(-200, -120, 400, 240)
    return nodes, edges, bounds


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
        self._nodes: Dict[str, _Node] = {}
        self._edges: List[_Edge] = []
        self._theme = theme.LIGHT

    def set_model(self, nodes: Dict[str, _Node], edges: List[_Edge],
                  bounds: QRectF) -> None:
        self._nodes, self._edges = nodes, edges
        self.setSceneRect(bounds)
        self.update()

    def set_theme(self, name: str) -> None:
        self._theme = name
        self.update()

    # -- drawing ---------------------------------------------------------- #

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        pal = theme.palette(self._theme)
        painter.fillRect(rect, QColor(pal.diagram_bg))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._draw_grid(painter, rect, pal)
        for edge in self._edges:
            self._draw_edge(painter, edge, pal)
        for node in self._nodes.values():
            self._draw_node(painter, node, pal)

    def _centroid(self) -> QPointF:
        """Middle of the construct nodes, used to bow covariance arcs outward."""
        points = [(n.x, n.y) for n in self._nodes.values()
                  if n.kind == "construct"]
        if not points:
            return QPointF(0.0, 0.0)
        return QPointF(sum(p[0] for p in points) / len(points),
                       sum(p[1] for p in points) / len(points))

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
        if node.kind == "item":
            fill, border = pal.diagram_item, pal.diagram_edge
        elif node.kind == "dimension":
            fill, border = pal.diagram_dimension, pal.diagram_node_border
        else:
            fill, border = pal.diagram_node, pal.diagram_node_border
        painter.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border))
        pen.setWidthF(2.0 if node.kind == "construct" else 1.4)
        painter.setPen(pen)
        if node.kind == "item":
            # Observed variables are rectangles; latent ones are ellipses.
            painter.drawRoundedRect(rect, 3, 3)
        else:
            painter.drawEllipse(rect)

        painter.setPen(QPen(QColor(pal.diagram_node_text)))
        font = QFont()
        font.setPointSizeF(8.5 if node.kind == "item" else 9.5)
        font.setBold(node.kind == "construct")
        painter.setFont(font)
        flags = int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap)
        text_rect = (rect.adjusted(6, 0, -6, -13) if node.kind == "construct"
                     else rect.adjusted(3, 1, -3, -1))
        painter.drawText(text_rect, flags,
                         _ltr(node.label) if node.kind == "item"
                         else node.label)

        if node.kind == "construct" and node.role is not None:
            badge = translator.enum("role", node.role)
            small = QFont()
            small.setPointSizeF(7.5)
            painter.setFont(small)
            painter.setPen(QPen(QColor(pal.text_muted)))
            painter.drawText(
                QRectF(rect.left(), rect.bottom() - 15, rect.width(), 13),
                int(Qt.AlignmentFlag.AlignCenter), badge)

    def _draw_edge(self, painter: QPainter, edge: _Edge, pal) -> None:
        source = self._nodes.get(edge.source)
        target = self._nodes.get(edge.target)
        if source is None or target is None:
            return
        if edge.kind == "loading":
            colour = QColor(pal.diagram_edge)
            width = 1.2
        elif edge.dashed:
            colour = QColor(pal.diagram_edge_null)
            width = 1.4
        else:
            colour = QColor(pal.diagram_edge_strong)
            width = 1.6 + 2.4 * min(edge.strength, 0.8)

        pen = QPen(colour)
        pen.setWidthF(width)
        if edge.dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        start = _edge_point(source, target.centre)
        end = _edge_point(target, source.centre)

        if edge.kind == "covariance":
            path, mid = _arc(start, end, away_from=self._centroid())
            painter.drawPath(path)
            _arrow_head(painter, colour, path.pointAtPercent(0.03),
                        path.pointAtPercent(0.0))
            _arrow_head(painter, colour, path.pointAtPercent(0.97),
                        path.pointAtPercent(1.0))
        elif edge.kind == "interaction":
            # A moderation lands on the middle of the path it modifies, so draw
            # it as a dashed line ending in an arrow part-way along.
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawLine(QLineF(start, end))
            _arrow_head(painter, colour, start, end)
            mid = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
        else:
            painter.drawLine(QLineF(start, end))
            _arrow_head(painter, colour, start, end)
            mid = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)

        if edge.label and edge.kind != "loading":
            _draw_label(painter, pal, mid, edge.label)
        elif edge.label and edge.kind == "loading":
            _draw_label(painter, pal, QPointF((start.x() + end.x()) / 2,
                                              (start.y() + end.y()) / 2),
                        edge.label, small=True)


def _edge_point(node: _Node, towards: QPointF) -> QPointF:
    """Where a line from ``towards`` meets this node's outline."""
    dx = towards.x() - node.x
    dy = towards.y() - node.y
    if dx == 0 and dy == 0:
        return node.centre
    if node.kind == "item":
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
                end: QPointF, size: float = 9.0) -> None:
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    tip = end
    left = QPointF(tip.x() - size * math.cos(angle - math.pi / 7),
                   tip.y() - size * math.sin(angle - math.pi / 7))
    right = QPointF(tip.x() - size * math.cos(angle + math.pi / 7),
                    tip.y() - size * math.sin(angle + math.pi / 7))
    painter.setBrush(QBrush(colour))
    painter.setPen(QPen(colour))
    painter.drawPolygon(QPolygonF([tip, left, right]))
    painter.setBrush(Qt.BrushStyle.NoBrush)


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
    bulge = min(170.0, max(46.0, length * 0.34))
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
    return "\u202d" + text + "\u202c"


def _draw_label(painter: QPainter, pal, point: QPointF, text: str,
                small: bool = False) -> None:
    text = _ltr(text)
    font = QFont()
    font.setPointSizeF(7.5 if small else 8.5)
    font.setBold(not small)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    width = metrics.horizontalAdvance(text) + 8
    height = metrics.height() + 2
    box = QRectF(point.x() - width / 2, point.y() - height / 2, width, height)
    painter.setBrush(QBrush(QColor(pal.diagram_bg)))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(box, 3, 3)
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
        nodes, edges, bounds = build_layout(
            self.state.spec,
            show_items=self._show_items.isChecked(),
            rtl=self._mirror.isChecked())
        self.scene.set_model(nodes, edges, bounds)
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
        image.save(path)
        return path


__all__ = ["PathDiagramView", "build_layout"]

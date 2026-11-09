"""
semsim.ui.pages.measurement
===========================
Stage 1 -- Model Specification Architecture.

Two halves of one job, so they live in one page under two tabs:

*   **Measurement** -- the construct / dimension / item hierarchy, edited as a
    tree with a context-sensitive editor beside it.  A second-order construct
    is a construct whose ``order`` is SECOND; switching the combo converts the
    existing items into a first dimension rather than discarding them, because
    losing a researcher's typed-in items to a dropdown change is unforgivable.
*   **Structural** -- paths, exogenous covariances, latent interactions and the
    mediation chains to report.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QSplitter, QStackedWidget, QTabWidget, QToolButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from semsim.core.spec import (
    Bilingual, Construct, ConstructOrder, ConstructRole, Dimension, Item,
    ScaleKind, ScaleSpec, ResponseStyle,
)
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.core.validation import (
    MIN_DIMENSIONS_SECOND_ORDER, MIN_ITEMS_PER_FACTOR,
)
from semsim.ui.pages.base import StagePage
from semsim.ui.pages.structural import StructuralEditor
from semsim.ui.widgets.common import (
    Card, Collapsible, EnumCombo, Field, InfoBanner, TButton, TCheckBox,
    ValueSlider, float_spin, int_spin, separator,
)

NODE_KIND = Qt.ItemDataRole.UserRole + 1
NODE_ID = Qt.ItemDataRole.UserRole + 2


class MeasurementPage(StagePage):
    TITLE_KEY = "stage.1.title"
    SUBTITLE_KEY = "stage.1.subtitle"

    def build(self) -> None:
        self.add_banner("d.thresholds.explain") if False else None
        tabs = QTabWidget()
        self._tabs = tabs
        self.add(tabs)

        self._measurement_tab = self._build_measurement_tab()
        tabs.addTab(self._measurement_tab, "")

        self._structural = StructuralEditor(self.state)
        tabs.addTab(self._structural, "")

        from semsim.ui.pages.demographics import DemographicsEditor
        self._demographics = DemographicsEditor(self.state)
        tabs.addTab(self._demographics, "")

        try:
            from semsim.ui.widgets.path_diagram import PathDiagramView
            self._diagram = PathDiagramView(self.state)
            tabs.addTab(self._diagram, "")
        except Exception:
            self._diagram = None

        self._retranslate_tabs()
        translator.language_changed.connect(lambda _l: self._retranslate_tabs())
        self.rebuild()

    def _retranslate_tabs(self) -> None:
        try:
            self._tabs.setTabText(0, translator.t("stage.1.title"))
            self._tabs.setTabText(1, translator.t("s.structural"))
            self._tabs.setTabText(2, translator.t("dem.tab"))
            if self._diagram is not None:
                self._tabs.setTabText(3, translator.t("s.diagram"))
            self._btn_up.setToolTip(translator.t("m.move_up"))
            self._btn_down.setToolTip(translator.t("m.move_down"))
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Measurement tab
    # ------------------------------------------------------------------ #

    def _build_measurement_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self._btn_add = TButton("m.add_construct", role="primary")
        self._btn_add.clicked.connect(self._add_construct)
        toolbar.addWidget(self._btn_add)
        self._btn_add2 = TButton("m.add_second_order")
        self._btn_add2.clicked.connect(self._add_second_order)
        toolbar.addWidget(self._btn_add2)
        self._btn_dup = TButton("m.duplicate")
        self._btn_dup.clicked.connect(self._duplicate_selected)
        toolbar.addWidget(self._btn_dup)
        self._btn_del = TButton("m.remove", role="danger")
        self._btn_del.clicked.connect(self._remove_selected)
        toolbar.addWidget(self._btn_del)
        toolbar.addStretch(1)
        self._btn_up = QToolButton()
        self._btn_up.setText("↑")
        self._btn_up.clicked.connect(lambda: self._move_selected(-1))
        toolbar.addWidget(self._btn_up)
        self._btn_down = QToolButton()
        self._btn_down.setText("↓")
        self._btn_down.clicked.connect(lambda: self._move_selected(1))
        toolbar.addWidget(self._btn_down)
        lay.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(300)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.currentItemChanged.connect(lambda *_a: self._on_select())
        splitter.addWidget(self.tree)

        self.editor_stack = QStackedWidget()
        self._empty_editor = _EmptyEditor()
        self._construct_editor = _ConstructEditor(self)
        self._dimension_editor = _DimensionEditor(self)
        self._item_editor = _ItemEditor(self)
        for widget in (self._empty_editor, self._construct_editor,
                       self._dimension_editor, self._item_editor):
            self.editor_stack.addWidget(widget)
        splitter.addWidget(self.editor_stack)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 620])
        lay.addWidget(splitter, 1)
        return page

    # ------------------------------------------------------------------ #
    # Tree
    # ------------------------------------------------------------------ #

    def rebuild(self) -> None:
        self._populate_tree()
        self._on_select()
        if hasattr(self, "_structural"):
            self._structural.rebuild()
        if hasattr(self, "_demographics"):
            self._demographics.rebuild()
        if getattr(self, "_diagram", None) is not None:
            self._diagram.refresh()

    def refresh(self) -> None:
        self._refresh_tree_labels()
        current = self.editor_stack.currentWidget()
        loader = getattr(current, "reload", None)
        if callable(loader):
            loader()
        if hasattr(self, "_structural"):
            self._structural.refresh()
        if hasattr(self, "_demographics"):
            self._demographics.refresh()

    def set_theme(self, name: str) -> None:
        super().set_theme(name)
        if getattr(self, "_diagram", None) is not None:
            self._diagram.set_theme(name)

    def _populate_tree(self) -> None:
        keep_kind, keep_id = self._selected()
        self.tree.blockSignals(True)
        self.tree.clear()
        lang = translator.language
        for construct in self.spec.constructs:
            node = QTreeWidgetItem(self.tree)
            node.setData(0, NODE_KIND, "construct")
            node.setData(0, NODE_ID, construct.id)
            self._label_construct(node, construct, lang)
            if construct.is_second_order:
                for dimension in construct.dimensions:
                    dim_node = QTreeWidgetItem(node)
                    dim_node.setData(0, NODE_KIND, "dimension")
                    dim_node.setData(0, NODE_ID, dimension.id)
                    self._label_dimension(dim_node, dimension, lang)
                    for item in dimension.items:
                        self._add_item_node(dim_node, item)
            else:
                for item in construct.items:
                    self._add_item_node(node, item)
            node.setExpanded(True)
        self.tree.blockSignals(False)
        self._restore_selection(keep_kind, keep_id)

    def _add_item_node(self, parent: QTreeWidgetItem, item: Item) -> None:
        node = QTreeWidgetItem(parent)
        node.setData(0, NODE_KIND, "item")
        node.setData(0, NODE_ID, item.id)
        self._label_item(node, item)

    def _label_construct(self, node: QTreeWidgetItem, construct: Construct,
                         lang: str) -> None:
        label = construct.label.get(lang) or construct.name
        role = translator.enum("role", construct.role)
        order = "²" if construct.is_second_order else ""
        node.setText(0, f"{construct.name}{order}  —  {label}   [{role}]")

    def _label_dimension(self, node: QTreeWidgetItem, dimension: Dimension,
                         lang: str) -> None:
        label = dimension.label.get(lang) or dimension.name
        node.setText(0, f"{dimension.name}  —  {label}   "
                        f"(λ={dimension.second_order_loading:.2f})")

    def _label_item(self, node: QTreeWidgetItem, item: Item) -> None:
        mark = " ↻" if item.reverse_coded else ""
        node.setText(0, f"{item.name}{mark}   λ={item.loading:.2f}")

    def _refresh_tree_labels(self) -> None:
        lang = translator.language
        for i in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(i)
            construct = self.spec.construct(node.data(0, NODE_ID))
            if construct is None:
                continue
            self._label_construct(node, construct, lang)
            for j in range(node.childCount()):
                child = node.child(j)
                kind = child.data(0, NODE_KIND)
                if kind == "dimension":
                    dimension = self.spec.dimension(child.data(0, NODE_ID))
                    if dimension is not None:
                        self._label_dimension(child, dimension, lang)
                    for k in range(child.childCount()):
                        grand = child.child(k)
                        item = self.spec.item(grand.data(0, NODE_ID))
                        if item is not None:
                            self._label_item(grand, item)
                else:
                    item = self.spec.item(child.data(0, NODE_ID))
                    if item is not None:
                        self._label_item(child, item)

    def _selected(self) -> tuple:
        node = self.tree.currentItem()
        if node is None:
            return "", ""
        return node.data(0, NODE_KIND) or "", node.data(0, NODE_ID) or ""

    def _restore_selection(self, kind: str, identifier: str) -> None:
        if identifier:
            found = self._find_node(identifier)
            if found is not None:
                self.tree.setCurrentItem(found)
                return
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _find_node(self, identifier: str) -> Optional[QTreeWidgetItem]:
        stack = [self.tree.topLevelItem(i)
                 for i in range(self.tree.topLevelItemCount())]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.data(0, NODE_ID) == identifier:
                return node
            stack.extend(node.child(i) for i in range(node.childCount()))
        return None

    def _on_select(self) -> None:
        kind, identifier = self._selected()
        if kind == "construct":
            self._construct_editor.load(identifier)
            self.editor_stack.setCurrentWidget(self._construct_editor)
        elif kind == "dimension":
            self._dimension_editor.load(identifier)
            self.editor_stack.setCurrentWidget(self._dimension_editor)
        elif kind == "item":
            self._item_editor.load(identifier)
            self.editor_stack.setCurrentWidget(self._item_editor)
        else:
            self.editor_stack.setCurrentWidget(self._empty_editor)
        self._btn_dup.setEnabled(bool(kind))
        self._btn_del.setEnabled(bool(kind))
        siblings, index = self._siblings_of(kind, identifier)
        self._btn_up.setEnabled(index > 0)
        self._btn_down.setEnabled(0 <= index < len(siblings) - 1)

    def node_changed(self, identifier: str) -> None:
        """An editor changed a label; refresh just that node's text."""
        node = self._find_node(identifier)
        if node is None:
            return
        lang = translator.language
        kind = node.data(0, NODE_KIND)
        if kind == "construct":
            construct = self.spec.construct(identifier)
            if construct:
                self._label_construct(node, construct, lang)
        elif kind == "dimension":
            dimension = self.spec.dimension(identifier)
            if dimension:
                self._label_dimension(node, dimension, lang)
        else:
            item = self.spec.item(identifier)
            if item:
                self._label_item(node, item)

    # ------------------------------------------------------------------ #
    # Structure commands
    # ------------------------------------------------------------------ #

    def _next_name(self, stem: str = "F") -> str:
        existing = {c.name for c in self.spec.constructs}
        index = len(existing) + 1
        while f"{stem}{index}" in existing:
            index += 1
        return f"{stem}{index}"

    def _select(self, identifier: str) -> None:
        """Show a node that was just created, which may be below the fold."""
        node = self._find_node(identifier)
        if node is not None:
            self.tree.setCurrentItem(node)
            self.tree.scrollToItem(node)

    def _add_construct(self) -> None:
        name = self._next_name()
        construct = self.spec.add_construct(
            name, label_ar=name, label_en=name,
            role=ConstructRole.INDEPENDENT, n_items=4)
        self.spec.ensure_unique_item_names()
        self.edit(structural=True)
        self._select(construct.id)

    def _add_second_order(self) -> None:
        name = self._next_name("H")
        construct = self.spec.add_second_order_construct(
            name, label_ar=name, label_en=name,
            role=ConstructRole.INDEPENDENT, n_dimensions=3,
            items_per_dimension=4)
        self.spec.ensure_unique_item_names()
        self.edit(structural=True)
        self._select(construct.id)

    def _duplicate_selected(self) -> None:
        kind, identifier = self._selected()
        if kind == "construct":
            source = self.spec.construct(identifier)
            if source is None:
                return
            import copy
            clone = copy.deepcopy(source)
            clone.id = Construct().id
            clone.name = self._next_name(source.name.rstrip("0123456789") or "F")
            for dimension in clone.dimensions:
                dimension.id = Dimension().id
                for item in dimension.items:
                    item.id = Item().id
            for item in clone.items:
                item.id = Item().id
            self.spec.constructs.append(clone)
            self.spec.ensure_unique_item_names()
            self.edit(structural=True)
            self._select(clone.id)
            return
        elif kind == "item":
            for construct in self.spec.constructs:
                for container in ([construct] + list(construct.dimensions)):
                    items = container.items
                    for index, item in enumerate(items):
                        if item.id == identifier:
                            import copy
                            clone = copy.deepcopy(item)
                            clone.id = Item().id
                            items.insert(index + 1, clone)
                            self.spec.ensure_unique_item_names()
                            self.edit(structural=True)
                            self._select(clone.id)
                            return
            return
        elif kind == "dimension":
            for construct in self.spec.constructs:
                for index, dimension in enumerate(construct.dimensions):
                    if dimension.id == identifier:
                        import copy
                        clone = copy.deepcopy(dimension)
                        clone.id = Dimension().id
                        clone.name = f"{construct.name}_D{len(construct.dimensions)+1}"
                        for item in clone.items:
                            item.id = Item().id
                        construct.dimensions.insert(index + 1, clone)
                        self.spec.ensure_unique_item_names()
                        self.edit(structural=True)
                        self._select(clone.id)
                        return
            return
        self.edit(structural=True)

    def _remove_selected(self) -> None:
        kind, identifier = self._selected()
        if not kind:
            return
        if kind == "construct":
            self.spec.constructs = [c for c in self.spec.constructs
                                    if c.id != identifier]
            # Drop every path and covariance that pointed at it, or the model
            # would carry references to a construct that no longer exists.
            self.spec.paths = [p for p in self.spec.paths
                               if identifier not in (p.source, p.target)]
            for variable in self.spec.demographics:
                variable.control_paths = [p for p in variable.control_paths
                                          if p.target != identifier]
            self.spec.latent_covariances = [
                lc for lc in self.spec.latent_covariances
                if identifier not in (lc.construct_a, lc.construct_b)]
            self.spec.interactions = [
                ix for ix in self.spec.interactions
                if identifier not in (ix.predictor, ix.moderator, ix.target)]
            self.spec.indirect_effects = [
                ie for ie in self.spec.indirect_effects
                if identifier not in ie.chain]
        elif kind == "dimension":
            for construct in self.spec.constructs:
                construct.dimensions = [d for d in construct.dimensions
                                        if d.id != identifier]
        elif kind == "item":
            for construct in self.spec.constructs:
                construct.items = [i for i in construct.items if i.id != identifier]
                for dimension in construct.dimensions:
                    dimension.items = [i for i in dimension.items
                                       if i.id != identifier]
        self._prune_item_references()
        self.edit(structural=True)

    def _prune_item_references(self) -> None:
        """Remove cross-loadings and error covariances naming a deleted item."""
        live_items = {i.id for i in self.spec.all_items()}
        live_factors = set(self.spec.all_factor_ids())
        self.spec.cross_loadings = [
            cl for cl in self.spec.cross_loadings
            if cl.item_id in live_items and cl.factor_id in live_factors]
        self.spec.error_covariances = [
            ec for ec in self.spec.error_covariances
            if ec.item_a in live_items and ec.item_b in live_items]
        self.spec.common_method_bias.item_ids = [
            i for i in self.spec.common_method_bias.item_ids if i in live_items]

    def _siblings_of(self, kind: str, identifier: str) -> Tuple[List[Any], int]:
        """The list holding the selected node and its place in it."""
        constructs = self.spec.constructs
        if kind == "construct":
            groups = [constructs]
        elif kind == "dimension":
            groups = [c.dimensions for c in constructs]
        elif kind == "item":
            groups = [c.items for c in constructs]
            groups += [d.items for c in constructs for d in c.dimensions]
        else:
            groups = []
        for siblings in groups:
            for index, node in enumerate(siblings):
                if node.id == identifier:
                    return siblings, index
        return [], -1

    def _move_selected(self, delta: int) -> None:
        kind, identifier = self._selected()
        siblings, index = self._siblings_of(kind, identifier)
        target = index + delta
        if index < 0 or not 0 <= target < len(siblings):
            return
        siblings[index], siblings[target] = siblings[target], siblings[index]
        self.edit(structural=True)


# --------------------------------------------------------------------------- #
# Editors
# --------------------------------------------------------------------------- #


class _EmptyEditor(QWidget):
    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        label = QLabel()
        theme.set_role(label, "hint")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)
        translator.language_changed.connect(
            lambda _l: label.setText(translator.t("m.empty_hint")))
        label.setText(translator.t("m.empty_hint"))


class _EditorBase(QWidget):
    def __init__(self, page: MeasurementPage) -> None:
        super().__init__()
        self.page = page
        self._id = ""
        self._loading = False
        from semsim.ui.widgets.common import ScrollPage
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = ScrollPage()
        outer.addWidget(self.scroll)
        self.build()

    def build(self) -> None: ...

    def load(self, identifier: str) -> None:
        self._id = identifier
        self.reload()

    def reload(self) -> None: ...

    def changed(self, structural: bool = False) -> None:
        if self._loading:
            return
        self.page.node_changed(self._id)
        self.page.edit(structural=structural)


class _ConstructEditor(_EditorBase):

    def build(self) -> None:
        card = Card("m.construct")
        self.scroll.add(card)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._on_name)
        card.add(Field("m.name", self.name, hint_key="m.name.hint"))
        self.label_ar = QLineEdit()
        self.label_ar.textEdited.connect(lambda _t: self._on_labels())
        card.add(Field("m.label_ar", self.label_ar, ltr_editor=False))
        self.label_en = QLineEdit()
        self.label_en.textEdited.connect(lambda _t: self._on_labels())
        card.add(Field("m.label_en", self.label_en))
        self.role = EnumCombo(ConstructRole, "role")
        self.role.value_changed.connect(self._on_role)
        card.add(Field("m.role", self.role))
        self.order = EnumCombo(ConstructOrder, "m.order")
        self.order.value_changed.connect(self._on_order)
        card.add(Field("m.order", self.order))

        self.bulk = Card("m.items")
        self.scroll.add(self.bulk)
        self.n_items = int_spin(1, 60, 4)
        self.n_items.valueChanged.connect(self._on_n_items)
        self._n_items_field = Field("m.n_items", self.n_items)
        self.bulk.add(self._n_items_field)
        self.n_dims = int_spin(1, 12, 3)
        self.n_dims.valueChanged.connect(self._on_n_dims)
        self._n_dims_field = Field("m.n_dimensions", self.n_dims)
        self.bulk.add(self._n_dims_field)
        self.uniform = ValueSlider(0.0, 0.99, 0.75)
        self.uniform.value_changed.connect(self._on_uniform)
        self.bulk.add(Field("m.uniform_loadings", self.uniform,
                            hint_key="m.loading.hint"))
        row = QHBoxLayout()
        self.vary = TButton("m.vary_loadings", role="flat")
        self.vary.clicked.connect(self._on_vary)
        row.addWidget(self.vary)
        row.addStretch(1)
        self.bulk.add(row)

        scale_card = Card("d.scale")
        self.scroll.add(scale_card)
        self.scale_kind = EnumCombo(ScaleKind, "d.scale")
        self.scale_kind.value_changed.connect(self._on_scale)
        scale_card.add(Field("d.scale_kind", self.scale_kind))
        self.n_cats = int_spin(2, 11, 5)
        self.n_cats.valueChanged.connect(lambda _v: self._on_scale())
        scale_card.add(Field("d.n_categories", self.n_cats))
        self.style = EnumCombo(ResponseStyle, "style")
        self.style.value_changed.connect(lambda _v: self._on_scale())
        scale_card.add(Field("d.response_style", self.style,
                             hint_key="d.thresholds.explain"))
        self.scroll.add_stretch()

    def reload(self) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None:
            return
        self._loading = True
        self.name.setText(construct.name)
        self.label_ar.setText(construct.label.ar)
        self.label_en.setText(construct.label.en)
        self.role.set_value(construct.role)
        self.order.set_value(construct.order)
        second = construct.is_second_order
        self.bulk.set_title_key("m.dimensions" if second else "m.items")
        self._n_dims_field.setVisible(second)
        self._n_items_field.setVisible(not second)
        if second:
            self.n_dims.setValue(len(construct.dimensions))
        else:
            self.n_items.setValue(len(construct.items))
        items = construct.all_items()
        if items:
            self.uniform.set_value(items[0].loading)
            scale = items[0].scale
            self.scale_kind.set_value(scale.kind)
            self.n_cats.setValue(scale.n_categories)
            self.style.set_value(scale.response_style)
            self.n_cats.setEnabled(scale.kind is ScaleKind.LIKERT)
            self.style.setEnabled(scale.kind is ScaleKind.LIKERT)
        self._loading = False

    # -- handlers ---------------------------------------------------------- #

    def _on_name(self) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None or self._loading:
            return
        construct.name = self.name.text().strip() or construct.name
        for index, item in enumerate(construct.all_items(), start=1):
            item.name = f"{construct.name}{index}"
        self.page.spec.ensure_unique_item_names()
        self.changed(structural=True)

    def _on_labels(self) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None:
            return
        construct.label = Bilingual(ar=self.label_ar.text(),
                                    en=self.label_en.text())
        self.changed()

    def _on_role(self, role: ConstructRole) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None:
            return
        construct.role = role
        self.changed()

    def _on_order(self, order: ConstructOrder) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None or construct.order is order:
            return
        if order is ConstructOrder.SECOND:
            self._split_into_dimensions(construct)
        else:
            # construct.items is normally empty here.  It is kept because
            # projects saved by 1.0.1 can hold items on a second-order construct.
            construct.items = [*construct.items,
                               *(i for d in construct.dimensions for i in d.items)]
            construct.dimensions = []
            construct.order = order
        self.page.spec.ensure_unique_item_names()
        self.changed(structural=True)

    def _split_into_dimensions(self, construct: Construct) -> None:
        """Turn the items into the smallest set of dimensions that can be fitted.

        Nothing the researcher typed is lost: the items are dealt out in order,
        and new ones are added only where a dimension would be too small.
        """
        items = list(construct.items)
        per_dimension = max(
            MIN_ITEMS_PER_FACTOR,
            math.ceil(len(items) / MIN_DIMENSIONS_SECOND_ORDER))
        dimensions = []
        for index in range(1, MIN_DIMENSIONS_SECOND_ORDER + 1):
            dimensions.append(Dimension(
                name=f"{construct.name}_D{index}",
                label=Bilingual(ar=f"{construct.label.ar} - البعد {index}",
                                en=f"{construct.label.en} - Dimension {index}"),
                items=items[:per_dimension]))
            items = items[per_dimension:]
        construct.order = ConstructOrder.SECOND
        construct.dimensions = dimensions
        construct.items = []
        for dimension in dimensions:
            self._resize_items(construct, dimension.items, per_dimension)

    def _on_n_items(self, count: int) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None or self._loading or construct.is_second_order:
            return
        self._resize_items(construct, construct.items, count)
        self.page.spec.ensure_unique_item_names()
        self.changed(structural=True)

    def _on_n_dims(self, count: int) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None or self._loading or not construct.is_second_order:
            return
        current = construct.dimensions
        template_items = len(current[0].items) if current else 4
        while len(current) > count:
            current.pop()
        while len(current) < count:
            index = len(current) + 1
            dimension = Dimension(
                name=f"{construct.name}_D{index}",
                label=Bilingual(ar=f"{construct.label.ar} - البعد {index}",
                                en=f"{construct.label.en} - Dimension {index}"),
                second_order_loading=(current[0].second_order_loading
                                      if current else 0.80))
            current.append(dimension)
            self._resize_items(construct, dimension.items, template_items)
        self.page.spec.ensure_unique_item_names()
        self.changed(structural=True)

    def _resize_items(self, construct: Construct, items: List[Item],
                      count: int) -> None:
        existing = construct.all_items()
        model = items[0] if items else (existing[0] if existing else None)
        loading = model.loading if model else self.uniform.value()
        scale = model.scale if model else ScaleSpec()
        while len(items) > count:
            items.pop()
        while len(items) < count:
            index = len(construct.all_items()) + 1
            items.append(Item(
                name=f"{construct.name}{index}",
                label=Bilingual(ar=f"{construct.label.ar} {index}",
                                en=f"{construct.label.en} {index}"),
                loading=loading,
                scale=ScaleSpec(kind=scale.kind,
                                n_categories=scale.n_categories,
                                response_style=scale.response_style)))

    def _on_uniform(self, value: float) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None or self._loading:
            return
        for item in construct.all_items():
            item.loading = value
        self.page._refresh_tree_labels()
        self.changed()

    def _on_vary(self) -> None:
        import numpy as np
        construct = self.page.spec.construct(self._id)
        if construct is None:
            return
        # A fresh generator: seeding it from the model gave the same numbers
        # on every press, so the second press appeared to do nothing.  The
        # drawn loadings are stored in the model, so a run stays reproducible.
        rng = np.random.default_rng()
        centre = self.uniform.value()
        for item in construct.all_items():
            item.loading = float(np.clip(
                rng.normal(centre, 0.06), 0.30, 0.95))
        self.page._refresh_tree_labels()
        self.changed()

    def _on_scale(self, *_a) -> None:
        construct = self.page.spec.construct(self._id)
        if construct is None or self._loading:
            return
        kind = self.scale_kind.value()
        self.n_cats.setEnabled(kind is ScaleKind.LIKERT)
        self.style.setEnabled(kind is ScaleKind.LIKERT)
        for item in construct.all_items():
            item.scale.kind = kind
            item.scale.n_categories = self.n_cats.value()
            item.scale.response_style = self.style.value()
        self.changed()


class _DimensionEditor(_EditorBase):

    def build(self) -> None:
        card = Card("m.dimension")
        self.scroll.add(card)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._on_name)
        card.add(Field("m.name", self.name))
        self.label_ar = QLineEdit()
        self.label_ar.textEdited.connect(lambda _t: self._on_labels())
        card.add(Field("m.label_ar", self.label_ar, ltr_editor=False))
        self.label_en = QLineEdit()
        self.label_en.textEdited.connect(lambda _t: self._on_labels())
        card.add(Field("m.label_en", self.label_en))
        self.loading = ValueSlider(0.0, 0.99, 0.80)
        self.loading.value_changed.connect(self._on_loading)
        card.add(Field("m.second_order_loading", self.loading,
                       hint_key="m.loading.hint"))
        self.n_items = int_spin(1, 60, 4)
        self.n_items.valueChanged.connect(self._on_n_items)
        card.add(Field("m.n_items", self.n_items))
        self.scroll.add_stretch()

    def reload(self) -> None:
        dimension = self.page.spec.dimension(self._id)
        if dimension is None:
            return
        self._loading = True
        self.name.setText(dimension.name)
        self.label_ar.setText(dimension.label.ar)
        self.label_en.setText(dimension.label.en)
        self.loading.set_value(dimension.second_order_loading)
        self.n_items.setValue(len(dimension.items))
        self._loading = False

    def _on_name(self) -> None:
        dimension = self.page.spec.dimension(self._id)
        if dimension is None or self._loading:
            return
        dimension.name = self.name.text().strip() or dimension.name
        self.changed()

    def _on_labels(self) -> None:
        dimension = self.page.spec.dimension(self._id)
        if dimension is None:
            return
        dimension.label = Bilingual(ar=self.label_ar.text(),
                                    en=self.label_en.text())
        self.changed()

    def _on_loading(self, value: float) -> None:
        dimension = self.page.spec.dimension(self._id)
        if dimension is None or self._loading:
            return
        dimension.second_order_loading = value
        self.changed()

    def _on_n_items(self, count: int) -> None:
        dimension = self.page.spec.dimension(self._id)
        if dimension is None or self._loading:
            return
        items = dimension.items
        loading = items[0].loading if items else 0.75
        scale = items[0].scale if items else ScaleSpec()
        while len(items) > count:
            items.pop()
        while len(items) < count:
            index = len(items) + 1
            items.append(Item(
                name=f"{dimension.name}_{index}",
                label=Bilingual(ar=f"{dimension.label.ar} {index}",
                                en=f"{dimension.label.en} {index}"),
                loading=loading,
                scale=ScaleSpec(kind=scale.kind,
                                n_categories=scale.n_categories,
                                response_style=scale.response_style)))
        self.page.spec.ensure_unique_item_names()
        self.changed(structural=True)


class _ItemEditor(_EditorBase):

    def build(self) -> None:
        card = Card("m.item")
        self.scroll.add(card)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._on_name)
        card.add(Field("m.name", self.name, hint_key="m.name.hint"))
        self.label_ar = QLineEdit()
        self.label_ar.textEdited.connect(lambda _t: self._on_labels())
        card.add(Field("m.label_ar", self.label_ar, ltr_editor=False))
        self.label_en = QLineEdit()
        self.label_en.textEdited.connect(lambda _t: self._on_labels())
        card.add(Field("m.label_en", self.label_en))
        self.loading = ValueSlider(-0.99, 0.99, 0.75)
        self.loading.value_changed.connect(self._on_loading)
        card.add(Field("m.loading", self.loading, hint_key="m.loading.hint"))
        self.reverse = TCheckBox("m.reverse_coded")
        self.reverse.toggled.connect(self._on_reverse)
        card.add(self.reverse)

        scale_card = Card("d.scale")
        self.scroll.add(scale_card)
        self.kind = EnumCombo(ScaleKind, "d.scale")
        self.kind.value_changed.connect(lambda _v: self._on_scale())
        scale_card.add(Field("d.scale_kind", self.kind))
        self.n_cats = int_spin(2, 11, 5)
        self.n_cats.valueChanged.connect(lambda _v: self._on_scale())
        scale_card.add(Field("d.n_categories", self.n_cats))
        self.style = EnumCombo(ResponseStyle, "style")
        self.style.value_changed.connect(lambda _v: self._on_scale())
        scale_card.add(Field("d.response_style", self.style))
        self.mean = float_spin(-10.0, 10.0, 0.0, decimals=2, step=0.1)
        self.mean.valueChanged.connect(lambda _v: self._on_scale())
        self._mean_field = Field("d.mean", self.mean)
        scale_card.add(self._mean_field)
        self.sd = float_spin(0.01, 10.0, 1.0, decimals=2, step=0.1)
        self.sd.valueChanged.connect(lambda _v: self._on_scale())
        self._sd_field = Field("d.sd", self.sd)
        scale_card.add(self._sd_field)

        shape = Collapsible("d.nonnormality")
        self.scroll.add(shape)
        shape.add(InfoBanner("d.nonnormality.explain"))
        self.skew = float_spin(-3.0, 3.0, 0.0, decimals=2, step=0.1)
        self.skew.valueChanged.connect(lambda _v: self._on_shape())
        shape.add(Field("d.skewness", self.skew))
        self.kurt = float_spin(-2.0, 25.0, 0.0, decimals=2, step=0.1)
        self.kurt.valueChanged.connect(lambda _v: self._on_shape())
        self._kurt_field = Field("d.kurtosis", self.kurt,
                                 hint_key="d.kurtosis.hint")
        shape.add(self._kurt_field)
        self.extra_error = float_spin(0.0, 0.8, 0.0, decimals=2, step=0.05)
        self.extra_error.valueChanged.connect(lambda _v: self._on_shape())
        shape.add(Field("m.extra_error", self.extra_error))
        self.scroll.add_stretch()

    def reload(self) -> None:
        item = self.page.spec.item(self._id)
        if item is None:
            return
        self._loading = True
        self.name.setText(item.name)
        self.label_ar.setText(item.label.ar)
        self.label_en.setText(item.label.en)
        self.loading.set_value(item.loading)
        self.reverse.setChecked(item.reverse_coded)
        self.kind.set_value(item.scale.kind)
        self.n_cats.setValue(item.scale.n_categories)
        self.style.set_value(item.scale.response_style)
        self.mean.setValue(item.scale.mean)
        self.sd.setValue(item.scale.sd)
        self.skew.setValue(item.skewness)
        self.kurt.setValue(item.kurtosis)
        self.extra_error.setValue(item.extra_error_variance)
        self._apply_kind_visibility(item.scale.kind)
        self._check_feasible()
        self._loading = False

    def _apply_kind_visibility(self, kind: ScaleKind) -> None:
        ordinal = kind is ScaleKind.LIKERT
        self.n_cats.setEnabled(ordinal)
        self.style.setEnabled(ordinal)
        self._mean_field.setVisible(not ordinal)
        self._sd_field.setVisible(not ordinal)

    def _check_feasible(self) -> None:
        """Fleishman has no real solution when kurtosis <= skew^2 - 2."""
        skew, kurt = self.skew.value(), self.kurt.value()
        if kurt <= skew * skew - 2.0:
            self._kurt_field.set_error(translator.t("d.infeasible"))
        else:
            self._kurt_field.set_error("")

    def _on_name(self) -> None:
        item = self.page.spec.item(self._id)
        if item is None or self._loading:
            return
        item.name = self.name.text().strip() or item.name
        self.page.spec.ensure_unique_item_names()
        self.changed()

    def _on_labels(self) -> None:
        item = self.page.spec.item(self._id)
        if item is None:
            return
        item.label = Bilingual(ar=self.label_ar.text(), en=self.label_en.text())
        self.changed()

    def _on_loading(self, value: float) -> None:
        item = self.page.spec.item(self._id)
        if item is None or self._loading:
            return
        item.loading = value
        self.changed()

    def _on_reverse(self, checked: bool) -> None:
        item = self.page.spec.item(self._id)
        if item is None or self._loading:
            return
        item.reverse_coded = checked
        self.changed()

    def _on_scale(self) -> None:
        item = self.page.spec.item(self._id)
        if item is None or self._loading:
            return
        item.scale.kind = self.kind.value()
        item.scale.n_categories = self.n_cats.value()
        item.scale.response_style = self.style.value()
        item.scale.mean = self.mean.value()
        item.scale.sd = self.sd.value()
        self._apply_kind_visibility(item.scale.kind)
        self.changed()

    def _on_shape(self) -> None:
        item = self.page.spec.item(self._id)
        if item is None or self._loading:
            return
        item.skewness = self.skew.value()
        item.kurtosis = self.kurt.value()
        item.extra_error_variance = self.extra_error.value()
        self._check_feasible()
        self.changed()


__all__ = ["MeasurementPage"]

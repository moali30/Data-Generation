"""
semsim.ui.pages.base
====================
Shared scaffolding for the six stage pages.

Every page follows the same contract so the main window can treat them
interchangeably:

``__init__(state)``   build the widgets
``rebuild()``         the model was replaced wholesale -- re-create anything
                      that mirrors its structure
``refresh()``         values changed but the structure did not -- re-read them
``set_theme(name)``   re-colour anything that paints its own colours

The two-phase split matters for responsiveness.  Typing in a loading box must
not tear down and rebuild a tree of widgets, so edits call
``state.touch()`` (values only) and only add/remove operations call
``state.touch(structural=True)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.state import AppState
from semsim.ui.widgets.common import (
    Card, Collapsible, Field, InfoBanner, PageTitle, ScrollPage, TButton,
    TCheckBox, separator,
)


class StagePage(QWidget):
    """Base class for the six stage pages."""

    #: Overridden by subclasses.
    TITLE_KEY = ""
    SUBTITLE_KEY = ""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._updating = False          # guards against feedback loops
        self._theme = theme.LIGHT

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QWidget()
        hlay = QVBoxLayout(header)
        hlay.setContentsMargins(20, 16, 20, 4)
        hlay.addWidget(PageTitle(self.TITLE_KEY, self.SUBTITLE_KEY))
        outer.addWidget(header)
        outer.addWidget(separator())
        self.scroll = ScrollPage()
        outer.addWidget(self.scroll, 1)

        self.build()
        self.state.structure_changed.connect(self._on_structure_changed)
        self.state.results_changed.connect(self._on_results_changed)
        translator.language_changed.connect(lambda _l: self.refresh())

    # -- lifecycle -------------------------------------------------------- #

    def build(self) -> None:
        """Create the page's widgets.  Called once from ``__init__``."""

    def rebuild(self) -> None:
        """The spec was replaced; re-create structure-dependent widgets."""
        self.refresh()

    def refresh(self) -> None:
        """Re-read values from the spec into the existing widgets."""

    def set_theme(self, name: str) -> None:
        self._theme = name

    # -- helpers ---------------------------------------------------------- #

    def _on_structure_changed(self) -> None:
        if self._updating:
            return
        self.rebuild()

    def _on_results_changed(self) -> None:
        pass

    @property
    def spec(self):
        return self.state.spec

    def edit(self, structural: bool = False) -> None:
        """Record an edit made by one of this page's widgets."""
        if self._updating:
            return
        self.state.touch(structural=structural)

    def quiet(self) -> "_Quiet":
        """Context manager: apply values to widgets without echoing edits back."""
        return _Quiet(self)

    def add(self, widget_or_layout: Any) -> Any:
        return self.scroll.add(widget_or_layout)

    def add_banner(self, key: str, kind: str = "info") -> InfoBanner:
        banner = InfoBanner(key, kind=kind)
        self.scroll.add(banner)
        return banner

    def add_card(self, title_key: str = "", *, both: bool = False) -> Card:
        card = Card(title_key, both=both)
        self.scroll.add(card)
        return card


class _Quiet:
    def __init__(self, page: StagePage) -> None:
        self._page = page
        self._previous = False

    def __enter__(self) -> StagePage:
        self._previous = self._page._updating
        self._page._updating = True
        return self._page

    def __exit__(self, *exc: Any) -> None:
        self._page._updating = self._previous


# --------------------------------------------------------------------------- #
# Editable row lists
# --------------------------------------------------------------------------- #


class RowList(QWidget):
    """A vertical list of editable rows with an add button.

    Model collections -- paths, cross-loadings, error covariances -- are edited
    as rows of live controls rather than through a table delegate.  A path row
    needs two combo boxes, an enum combo, a slider and a checkbox; building
    that with delegates costs far more code than it saves, and the controls
    behave better.
    """

    #: Emitted after a row is added or removed.
    changed = Signal()

    def __init__(self, add_key: str, *, empty_key: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: List[QWidget] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._header: Optional[QWidget] = None
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        lay.addWidget(self._rows_host)

        self._empty = QLabel()
        theme.set_role(self._empty, "hint")
        self._empty_key = empty_key or "v.no_issues"
        lay.addWidget(self._empty)

        bottom = QHBoxLayout()
        self._add_button = TButton(add_key, role="flat")
        bottom.addWidget(self._add_button)
        bottom.addStretch(1)
        lay.addLayout(bottom)

        translator.language_changed.connect(lambda _l: self._retranslate())
        self._retranslate()
        self._update_empty()

    @property
    def add_button(self) -> TButton:
        return self._add_button

    def set_header(self, labels: Sequence[Tuple[str, int]]) -> None:
        """Column captions above the rows: ``[(i18n_key, stretch), ...]``."""
        if self._header is not None:
            self._header.deleteLater()
        self._header = QWidget()
        hl = QHBoxLayout(self._header)
        hl.setContentsMargins(4, 0, 4, 0)
        hl.setSpacing(8)
        self._header_labels: List[Tuple[QLabel, str]] = []
        for key, stretch in labels:
            label = QLabel()
            theme.set_role(label, "hint")
            hl.addWidget(label, stretch)
            self._header_labels.append((label, key))
        layout = self.layout()
        layout.insertWidget(0, self._header)
        self._retranslate()

    def add_row(self, widget: QWidget) -> QWidget:
        self._rows.append(widget)
        self._rows_layout.addWidget(widget)
        self._update_empty()
        return widget

    def remove_row(self, widget: QWidget) -> None:
        if widget in self._rows:
            self._rows.remove(widget)
        self._rows_layout.removeWidget(widget)
        widget.setParent(None)
        widget.deleteLater()
        self._update_empty()

    def clear(self) -> None:
        for widget in list(self._rows):
            self._rows_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._rows.clear()
        self._update_empty()

    def rows(self) -> List[QWidget]:
        return list(self._rows)

    def _update_empty(self) -> None:
        self._empty.setVisible(not self._rows)
        if self._header is not None:
            self._header.setVisible(bool(self._rows))

    def _retranslate(self) -> None:
        try:
            self._empty.setText(translator.t(self._empty_key))
            for label, key in getattr(self, "_header_labels", []):
                label.setText(translator.t(key))
        except RuntimeError:
            pass


class Row(QFrame):
    """One row inside a :class:`RowList`, with a trailing delete button."""

    removed = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(4, 2, 4, 2)
        self.layout_.setSpacing(8)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.layout_.addWidget(widget, stretch)
        return widget

    def add_delete_button(self) -> QToolButton:
        button = QToolButton()
        button.setText("×")
        button.setAutoRaise(True)
        button.setFixedWidth(26)
        button.setToolTip(translator.t("m.remove"))
        button.clicked.connect(lambda: self.removed.emit(self))
        self.layout_.addWidget(button)
        return button


def construct_choices(spec, *, roles=None, exclude=None) -> List[Tuple[str, str]]:
    """``[(construct_id, display_label), ...]`` for a chooser combo."""
    lang = translator.language
    out: List[Tuple[str, str]] = []
    for construct in spec.constructs:
        if roles is not None and construct.role not in roles:
            continue
        if exclude is not None and construct.id in exclude:
            continue
        label = construct.label.get(lang) or construct.name or construct.id
        out.append((construct.id, f"{construct.name} - {label}"
                    if construct.name and label != construct.name else label))
    return out


def factor_choices(spec) -> List[Tuple[str, str]]:
    """Constructs *and* the dimensions inside them, for cross-loading targets."""
    lang = translator.language
    out: List[Tuple[str, str]] = []
    for construct in spec.constructs:
        label = construct.label.get(lang) or construct.name
        out.append((construct.id, label))
        for dimension in construct.dimensions:
            dim_label = dimension.label.get(lang) or dimension.name
            out.append((dimension.id, f"    {label} ‹ {dim_label}"))
    return out


def item_choices(spec) -> List[Tuple[str, str]]:
    lang = translator.language
    out: List[Tuple[str, str]] = []
    for item, construct, dimension in spec.iter_items_with_owner():
        owner = (dimension.label.get(lang) or dimension.name) if dimension \
            else (construct.label.get(lang) or construct.name)
        out.append((item.id, f"{item.name}  ({owner})"))
    return out


class IdCombo(QWidget):
    """A combo box over ``(id, label)`` pairs that survives a rebuild.

    Rebuilding the list must not silently change the selection, so the current
    id is restored after repopulating, and the row is marked invalid when its
    id has gone away (the construct it pointed at was deleted).
    """

    value_changed = Signal(str)

    def __init__(self, choices: Optional[Sequence[Tuple[str, str]]] = None,
                 parent: Optional[QWidget] = None) -> None:
        from PySide6.QtWidgets import QComboBox
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        lay.addWidget(self._combo)
        self._combo.currentIndexChanged.connect(
            lambda _i: self.value_changed.emit(self.value()))
        if choices:
            self.set_choices(choices)

    def set_choices(self, choices: Sequence[Tuple[str, str]],
                    keep: str = "") -> None:
        current = keep or self.value()
        blocked = self._combo.blockSignals(True)
        self._combo.clear()
        for identifier, label in choices:
            self._combo.addItem(label, identifier)
        if current:
            index = self._combo.findData(current)
            if index >= 0:
                self._combo.setCurrentIndex(index)
            else:
                # The target was deleted; say so instead of silently
                # repointing the row at whatever is first in the list.
                self._combo.insertItem(0, translator.t("v.missing_ref"), current)
                self._combo.setCurrentIndex(0)
        self._combo.blockSignals(blocked)

    def value(self) -> str:
        data = self._combo.currentData()
        return str(data) if data is not None else ""

    def set_value(self, identifier: str) -> None:
        index = self._combo.findData(identifier)
        if index >= 0:
            blocked = self._combo.blockSignals(True)
            self._combo.setCurrentIndex(index)
            self._combo.blockSignals(blocked)

    @property
    def combo(self):
        return self._combo


__all__ = [
    "StagePage", "RowList", "Row", "IdCombo",
    "construct_choices", "factor_choices", "item_choices",
]

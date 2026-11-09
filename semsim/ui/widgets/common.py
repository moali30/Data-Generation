"""
semsim.ui.widgets.common
========================
Small reusable widgets.

Two conventions run through this module:

*   **Live retranslation.**  Anything that shows text takes an i18n *key*, not a
    string, and re-reads it when :data:`semsim.ui.i18n.translator` emits
    ``language_changed``.  Switching to English never needs the window rebuilt.
*   **Direction discipline.**  Labels and prose follow the interface direction;
    numbers, identifiers and anything that would read wrong mirrored are pinned
    left-to-right with :func:`semsim.ui.theme.keep_ltr`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, QSize, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from semsim.ui import theme
from semsim.ui.i18n import translator


# --------------------------------------------------------------------------- #
# Retranslation plumbing
# --------------------------------------------------------------------------- #


class Retranslatable:
    """Mixin: re-read every i18n key when the language changes.

    Subclasses implement :meth:`retranslate`.  The connection is made once, in
    :meth:`hook_language`, and Qt drops it automatically when the widget dies.
    """

    def hook_language(self) -> None:
        translator.language_changed.connect(self._on_language_changed)
        self.retranslate()

    def _on_language_changed(self, _lang: str) -> None:
        try:
            self.retranslate()
        except RuntimeError:
            pass          # the C++ side is gone; nothing to retranslate

    def retranslate(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #


class TKey(QLabel, Retranslatable):
    """A label driven by an i18n key."""

    def __init__(self, key: str, *, role: str = "", both: bool = False,
                 parent: Optional[QWidget] = None, **fmt: Any) -> None:
        super().__init__(parent)
        self._key = key
        self._both = both
        self._fmt = fmt
        if role:
            theme.set_role(self, role)
        self.setWordWrap(True)
        self.hook_language()

    def set_key(self, key: str, **fmt: Any) -> None:
        self._key = key
        self._fmt = fmt or self._fmt
        self.retranslate()

    def retranslate(self) -> None:
        text = (translator.both(self._key) if self._both
                else translator.t(self._key, **self._fmt))
        self.setText(text)


class Heading(TKey):
    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(key, role="h2", parent=parent)


class PageTitle(QWidget, Retranslatable):
    """Big title plus a muted subtitle, used at the top of each stage page."""

    def __init__(self, title_key: str, subtitle_key: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title_key = title_key
        self._subtitle_key = subtitle_key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 8)
        lay.setSpacing(2)
        self._title = QLabel()
        theme.set_role(self._title, "h1")
        lay.addWidget(self._title)
        self._subtitle = QLabel()
        theme.set_role(self._subtitle, "subtitle")
        self._subtitle.setWordWrap(True)
        lay.addWidget(self._subtitle)
        self.hook_language()

    def retranslate(self) -> None:
        self._title.setText(translator.t(self._title_key))
        if self._subtitle_key:
            self._subtitle.setText(translator.t(self._subtitle_key))
            self._subtitle.setVisible(True)
        else:
            self._subtitle.setVisible(False)


class InfoBanner(QFrame, Retranslatable):
    """A short explanation of what a section is for and why it matters."""

    def __init__(self, key: str, *, kind: str = "info",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        self._kind = kind
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("infoBanner")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)
        self._icon = QLabel()
        self._icon.setFixedWidth(18)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignTop
                                | Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self._icon)
        self._text = QLabel()
        self._text.setWordWrap(True)
        # A word-wrapped label inside a horizontal layout reports the
        # height of its UNwrapped text unless height may follow width.
        policy = self._text.sizePolicy()
        policy.setHeightForWidth(True)
        self._text.setSizePolicy(policy)
        lay.addWidget(self._text, 1)
        self._apply_kind()
        self.hook_language()

    def _apply_kind(self) -> None:
        pal = theme.palette(current_theme())
        colour = {"info": pal.info, "warning": pal.warning,
                  "danger": pal.danger, "success": pal.success}.get(
                      self._kind, pal.info)
        glyph = {"info": "i", "warning": "!", "danger": "!",
                 "success": "✓"}.get(self._kind, "i")
        self._icon.setText(glyph)
        self._icon.setStyleSheet(f"color: {colour}; font-weight: 700;")
        # Qt stylesheets do not mirror border-left, so pick the side that
        # is actually the leading edge for the current direction.  The rule
        # is scoped to this frame by name: QLabel derives from QFrame, so a
        # bare "QFrame" selector would draw the bar on the labels inside too.
        side = "right" if translator.is_rtl else "left"
        self.setStyleSheet(
            f"QFrame#infoBanner {{ background: {pal.surface_alt};"
            f" border-{side}: 3px solid {colour}; border-radius: 6px; }}")

    def set_kind(self, kind: str) -> None:
        self._kind = kind
        self._apply_kind()

    def retranslate(self) -> None:
        self._text.setText(translator.t(self._key))
        self._apply_kind()


class VerdictChip(QLabel):
    """A small coloured pass / fail / borderline badge."""

    def __init__(self, verdict: str = "pass", text: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum,
                           QSizePolicy.Policy.Fixed)
        self.set_verdict(verdict, text)

    def set_verdict(self, verdict: str, text: str = "") -> None:
        bg, fg = theme.verdict_colours(current_theme(), verdict)
        self.setText(text or translator.t(f"verdict.{verdict}"))
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 9px;"
            f" padding: 2px 10px; font-weight: 600;")


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


_CURRENT_THEME = theme.LIGHT


def set_current_theme(name: str) -> None:
    """Record the active theme so widgets drawing their own colours agree."""
    global _CURRENT_THEME
    _CURRENT_THEME = name


def current_theme() -> str:
    return _CURRENT_THEME


class Card(QFrame, Retranslatable):
    """A titled panel.  ``body`` is the layout callers add their widgets to."""

    def __init__(self, title_key: str = "", *, both: bool = False,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title_key = title_key
        self._both = both
        theme.set_role(self, "card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)
        self._title = QLabel()
        theme.set_role(self._title, "h2")
        outer.addWidget(self._title)
        self._title.setVisible(bool(title_key))
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)
        self.hook_language()

    def add(self, widget_or_layout: Any) -> Any:
        if isinstance(widget_or_layout, QLayout):
            self.body.addLayout(widget_or_layout)
        else:
            self.body.addWidget(widget_or_layout)
        return widget_or_layout

    def set_title_key(self, key: str) -> None:
        self._title_key = key
        self._title.setVisible(bool(key))
        self.retranslate()

    def retranslate(self) -> None:
        if self._title_key:
            self._title.setText(translator.both(self._title_key) if self._both
                                else translator.t(self._title_key))


class Collapsible(QWidget, Retranslatable):
    """A section that folds away.  Advanced controls live in these."""

    toggled_open = Signal(bool)

    def __init__(self, title_key: str, *, expanded: bool = False,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title_key = title_key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._button = QToolButton()
        self._button.setObjectName("sectionHeader")
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setAutoRaise(True)
        self._button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.ArrowType.DownArrow if expanded
                                  else arrow_for_direction())
        self._button.clicked.connect(self._on_click)
        lay.addWidget(self._button)
        self._content = QWidget()
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(16, 4, 16, 4)
        self.body.setSpacing(8)
        self._content.setVisible(expanded)
        lay.addWidget(self._content)
        self.hook_language()

    def _on_click(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._button.setArrowType(Qt.ArrowType.DownArrow if checked
                                  else arrow_for_direction())
        self.toggled_open.emit(checked)

    def add(self, widget_or_layout: Any) -> Any:
        if isinstance(widget_or_layout, QLayout):
            self.body.addLayout(widget_or_layout)
        else:
            self.body.addWidget(widget_or_layout)
        return widget_or_layout

    def set_expanded(self, expanded: bool) -> None:
        self._button.setChecked(expanded)
        self._on_click(expanded)

    def retranslate(self) -> None:
        # Padded on both sides: the arrow is drawn after the text under RTL.
        self._button.setText(f"  {translator.t(self._title_key)}  ")
        self._button.setArrowType(Qt.ArrowType.DownArrow
                                  if self._button.isChecked()
                                  else arrow_for_direction())


def arrow_for_direction() -> Qt.ArrowType:
    """Collapsed sections point *into* the text, which flips under RTL."""
    return (Qt.ArrowType.LeftArrow if translator.is_rtl
            else Qt.ArrowType.RightArrow)


class ScrollPage(QScrollArea):
    """A vertically scrolling page body."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(20, 16, 20, 20)
        self.body.setSpacing(14)
        self.setWidget(inner)

    def add(self, widget_or_layout: Any) -> Any:
        if isinstance(widget_or_layout, QLayout):
            self.body.addLayout(widget_or_layout)
        else:
            self.body.addWidget(widget_or_layout)
        return widget_or_layout

    def add_stretch(self) -> None:
        self.body.addStretch(1)


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    theme.set_role(line, "separator")
    line.setFixedHeight(1)
    return line


class Notice(QLabel):
    """A short confirmation beside the control that caused it.

    Some actions change nothing where the user is looking: applying a scale to
    every item, say, or writing an image to disk.  Without a word back they are
    indistinguishable from a button that does nothing.  The text clears itself
    so that it never outlives the action it describes.
    """

    DURATION_MS = 6000

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        theme.set_role(self, "success")
        self.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._expire)

    def show_key(self, key: str, *, role: str = "success", **fmt: Any) -> None:
        theme.set_role(self, role)
        self.setText(translator.t(key, **fmt))
        self.setVisible(True)
        self._timer.start(self.DURATION_MS)

    def _expire(self) -> None:
        self.clear()
        self.setVisible(False)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


_MIN_LABEL_WIDTH = 110


class Field(QWidget, Retranslatable):
    """A labelled input with an optional hint underneath.

    The label follows the interface direction; the editor is pinned LTR when it
    holds a number or an identifier.
    """

    def __init__(self, label_key: str, editor: QWidget, *,
                 hint_key: str = "", ltr_editor: bool = True,
                 label_width: int = 220,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label_key = label_key
        self._hint_key = hint_key
        self.editor = editor
        if ltr_editor and not _is_choice(editor):
            theme.keep_ltr(editor)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(10)
        self._label = QLabel()
        self._label.setWordWrap(True)
        # Only a full-width form row gets a column of labels and a cap on its
        # editor; a compact field inside a table row has neither to spare.
        cap = _editor_width_cap(editor) if label_width else 0
        if label_width:
            # A range, not a fixed width.  The labels still line up, because
            # every row in a card is the same width, and a narrow pane takes
            # its room from them before it has to scroll sideways.
            self._label.setMinimumWidth(min(label_width, _MIN_LABEL_WIDTH))
            self._label.setMaximumWidth(label_width)
        if cap:
            editor.setMaximumWidth(cap)
        row.addWidget(self._label, 40 if label_width else 0)
        row.addWidget(editor, 60 if label_width else 1)
        if cap:
            row.addStretch(1)
        outer.addLayout(row)

        self._hint = QLabel()
        theme.set_role(self._hint, "hint")
        self._hint.setWordWrap(True)
        self._hint.setVisible(bool(hint_key))
        outer.addWidget(self._hint)
        self.hook_language()

    def set_hint_key(self, key: str) -> None:
        self._hint_key = key
        self._hint.setVisible(bool(key))
        self.retranslate()

    def set_error(self, message: str = "") -> None:
        """Show a red message under the field, or clear it.

        Clearing used to delegate to :meth:`retranslate`, which only rewrites
        the hint when the field was built with a ``hint_key``. A field without
        one therefore kept its red error text on screen after the problem was
        fixed -- the worst kind of stale UI, because it says the input is
        still wrong when it is not. The label is now emptied explicitly and
        the standing hint restored only if there is one.
        """
        if message:
            theme.set_role(self._hint, "error")
            self._hint.setText(message)
            self._hint.setVisible(True)
            self.editor.setProperty("invalid", True)
        else:
            theme.set_role(self._hint, "hint")
            self.editor.setProperty("invalid", False)
            if self._hint_key:
                self._hint.setText(translator.t(self._hint_key))
                self._hint.setVisible(True)
            else:
                self._hint.clear()
                self._hint.setVisible(False)
        style = self.editor.style()
        if style is not None:
            style.unpolish(self.editor)
            style.polish(self.editor)

    def retranslate(self) -> None:
        self._label.setText(translator.t(self._label_key))
        if self._hint_key:
            self._hint.setText(translator.t(self._hint_key))
            self._hint.setVisible(True)


def let_combo_shrink(combo: QComboBox) -> None:
    """Stop a combo's longest entry from setting the width of the whole page.

    The drop-down list still opens wide enough to read every entry.
    """
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)


class EnumCombo(QComboBox, Retranslatable):
    """A combo box bound to an enum, labelled through the i18n catalogue."""

    value_changed = Signal(object)

    def __init__(self, enum_cls: Any, prefix: str, *,
                 members: Optional[Sequence[Any]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._enum = enum_cls
        self._prefix = prefix
        self._members: List[Any] = list(members if members is not None
                                        else list(enum_cls))
        for member in self._members:
            self.addItem("", member)
        let_combo_shrink(self)
        self.currentIndexChanged.connect(self._emit)
        self.hook_language()

    def _emit(self, _index: int) -> None:
        self.value_changed.emit(self.value())

    def value(self) -> Any:
        # Not currentData(): every enum in the spec derives from str, so Qt
        # hands back the bare string and an ``is`` test against the member
        # fails.  Switching a construct to second order died exactly there.
        index = self.currentIndex()
        return self._members[index] if 0 <= index < len(self._members) else None

    def set_value(self, member: Any) -> None:
        for i, m in enumerate(self._members):
            if m == member:
                blocked = self.blockSignals(True)
                self.setCurrentIndex(i)
                self.blockSignals(blocked)
                return

    def retranslate(self) -> None:
        for i, member in enumerate(self._members):
            value = getattr(member, "value", member)
            self.setItemText(i, translator.t(f"{self._prefix}.{value}"))


class ValueSlider(QWidget):
    """A slider and a spin box editing one float, kept in step.

    Used for loadings and path coefficients, where researchers want to drag to
    explore but type to be exact.
    """

    value_changed = Signal(float)

    def __init__(self, minimum: float = 0.0, maximum: float = 1.0,
                 value: float = 0.5, *, decimals: int = 2,
                 step: float = 0.01, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._min, self._max = minimum, maximum
        self._scale = 10 ** decimals
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(round(minimum * self._scale)))
        self._slider.setMaximum(int(round(maximum * self._scale)))
        self._slider.setSingleStep(max(1, int(round(step * self._scale))))
        self._slider.setPageStep(max(1, int(round(step * self._scale * 10))))
        theme.keep_ltr(self._slider)
        lay.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(step)
        self._spin.setFixedWidth(86)
        self._spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        _finish_spin(self._spin)
        lay.addWidget(self._spin)

        self._slider.valueChanged.connect(self._from_slider)
        self._spin.valueChanged.connect(self._from_spin)
        self.set_value(value)

    def _from_slider(self, raw: int) -> None:
        value = raw / self._scale
        if abs(value - self._spin.value()) > 1e-9:
            blocked = self._spin.blockSignals(True)
            self._spin.setValue(value)
            self._spin.blockSignals(blocked)
        self.value_changed.emit(value)

    def _from_spin(self, value: float) -> None:
        raw = int(round(value * self._scale))
        if raw != self._slider.value():
            blocked = self._slider.blockSignals(True)
            self._slider.setValue(raw)
            self._slider.blockSignals(blocked)
        self.value_changed.emit(value)

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, value: float) -> None:
        value = max(self._min, min(self._max, float(value)))
        blocked_s = self._slider.blockSignals(True)
        blocked_b = self._spin.blockSignals(True)
        self._spin.setValue(value)
        self._slider.setValue(int(round(value * self._scale)))
        self._slider.blockSignals(blocked_s)
        self._spin.blockSignals(blocked_b)

    def set_range(self, minimum: float, maximum: float) -> None:
        self._min, self._max = minimum, maximum
        self._spin.setRange(minimum, maximum)
        self._slider.setMinimum(int(round(minimum * self._scale)))
        self._slider.setMaximum(int(round(maximum * self._scale)))


def _finish_spin(box: QAbstractSpinBox) -> QAbstractSpinBox:
    """Pin a spin box left-to-right and keep its digits off the border.

    The stylesheet's padding does not reach the line edit inside a spin box.
    """
    theme.keep_ltr(box)
    box.lineEdit().setTextMargins(6, 0, 2, 0)
    return box


def _is_choice(editor: QWidget) -> bool:
    """A combo holds words, which should read in the interface's direction."""
    return (isinstance(editor, QComboBox)
            or isinstance(getattr(editor, "combo", None), QComboBox))


def _editor_width_cap(editor: QWidget) -> int:
    """How wide an editor may grow, or 0 for no limit.

    On a maximised window an uncapped editor runs to well over a thousand
    pixels, which puts a two-digit number a long way from its label.
    """
    if isinstance(editor, QAbstractSpinBox):
        return 170
    if isinstance(editor, ValueSlider):
        return 520
    if _is_choice(editor):
        return 420
    if isinstance(editor, QLineEdit):
        return 460
    return 0


def int_spin(minimum: int, maximum: int, value: int, *,
             step: int = 1, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return _finish_spin(box)


def float_spin(minimum: float, maximum: float, value: float, *,
               decimals: int = 3, step: float = 0.01,
               suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return _finish_spin(box)


def percent_spin(value: float, *, maximum: float = 1.0) -> QDoubleSpinBox:
    """A 0..1 proportion shown as a percentage."""
    box = QDoubleSpinBox()
    box.setDecimals(1)
    box.setRange(0.0, maximum * 100.0)
    box.setSingleStep(0.5)
    box.setValue(value * 100.0)
    box.setSuffix(" %")
    return _finish_spin(box)


def percent_value(box: QDoubleSpinBox) -> float:
    return float(box.value()) / 100.0


class TCheckBox(QCheckBox, Retranslatable):
    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        self.hook_language()

    def retranslate(self) -> None:
        self.setText(translator.t(self._key))


class TButton(QPushButton, Retranslatable):
    def __init__(self, key: str, *, role: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        if role:
            theme.set_role(self, role)
        self.hook_language()

    def set_key(self, key: str) -> None:
        self._key = key
        self.retranslate()

    def retranslate(self) -> None:
        self.setText(translator.t(self._key))


class StatBadge(QFrame):
    """A compact "label: value" tile for the model summary strip."""

    def __init__(self, label_key: str, value: str = "-",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label_key = label_key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        self._value = QLabel(value)
        theme.set_role(self._value, "mono")
        self._value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self._value.font()
        font.setPointSize(font.pointSize() + 3)
        font.setBold(True)
        self._value.setFont(font)
        lay.addWidget(self._value)
        self._label = QLabel()
        theme.set_role(self._label, "hint")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._label)
        translator.language_changed.connect(lambda _l: self._retranslate())
        self._retranslate()

    def _retranslate(self) -> None:
        try:
            self._label.setText(translator.t(self._label_key))
        except RuntimeError:
            pass

    def set_value(self, value: Any, *, kind: str = "") -> None:
        self._value.setText(str(value))
        if kind:
            _bg, fg = theme.verdict_colours(current_theme(), kind)
            self._value.setStyleSheet(f"color: {fg};")
        else:
            self._value.setStyleSheet("")


__all__ = [
    "Retranslatable", "TKey", "Heading", "PageTitle", "InfoBanner",
    "VerdictChip", "Card", "Collapsible", "ScrollPage", "separator",
    "Notice", "Field", "EnumCombo", "ValueSlider", "int_spin", "float_spin",
    "percent_spin", "percent_value", "TCheckBox", "TButton", "StatBadge",
    "set_current_theme", "current_theme", "arrow_for_direction",
]

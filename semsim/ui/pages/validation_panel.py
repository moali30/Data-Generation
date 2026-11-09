"""
semsim.ui.pages.validation_panel
================================
The model-check dialog.

``MainWindow.show_validation`` opens this on the validation chip and on
*Model -> Validate model*, so it is a :class:`QDialog` rather than a stage
page.  It shows the diagnostics that
:func:`semsim.core.validation.validate` produced for the current spec,
grouped into errors, warnings and notes.

Three conventions from the rest of the interface are kept here:

*   **Nothing is read off the model twice.**  The report is taken from
    :attr:`AppState.validation`, which is cached and debounced; *Re-check*
    asks the state to recompute and rebuilds the list.
*   **Live retranslation.**  Chrome comes from the catalogue through
    ``TKey`` / ``TButton``; the diagnostic text itself is bilingual in the
    :class:`~semsim.core.validation.Diagnostic`, so each card re-reads
    ``message(lang)`` and ``hint(lang)`` when the language changes.
*   **Direction discipline.**  Messages and hints are prose and follow the
    interface direction; the stable code, the location id and the per-section
    counts are identifiers and numbers, so they are pinned left-to-right.
    Qt does not mirror ``border-left``, so the severity edge picks its side
    from ``translator.is_rtl``, exactly as ``InfoBanner`` does.

The dialog never assumes validation is available: ``state.validation`` is
``None`` when :mod:`semsim.core.validation` failed to import, and that case
shows one explanatory note instead of an empty list.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.widgets.common import (
    InfoBanner, Retranslatable, ScrollPage, TButton, TKey, current_theme,
    separator,
)

#: ``(severity value, heading key, banner/colour kind)`` in display order.
SECTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("error", "v.errors", "danger"),
    ("warning", "v.warnings", "warning"),
    ("info", "v.info", "info"),
)

_SEVERITY_KEY = {
    "error": "v.severity.error",
    "warning": "v.severity.warning",
    "info": "v.severity.info",
}


# --------------------------------------------------------------------------- #
# Diagnostic access helpers
# --------------------------------------------------------------------------- #


def severity_of(diagnostic: Any) -> str:
    """``"error"`` / ``"warning"`` / ``"info"`` for a diagnostic.

    ``Severity`` is a ``str`` enum, but the report is duck-typed here so the
    dialog survives a validation module that reports plain strings.
    """
    raw = getattr(diagnostic, "severity", "info")
    value = getattr(raw, "value", raw)
    text = str(value).lower()
    return text if text in _SEVERITY_KEY else "info"


def message_of(diagnostic: Any, lang: str) -> str:
    getter = getattr(diagnostic, "message", None)
    if callable(getter):
        try:
            return str(getter(lang) or "")
        except Exception:
            pass
    attr = "message_ar" if lang == "ar" else "message_en"
    return str(getattr(diagnostic, attr, "") or "")


def hint_of(diagnostic: Any, lang: str) -> str:
    getter = getattr(diagnostic, "hint", None)
    if callable(getter):
        try:
            return str(getter(lang) or "")
        except Exception:
            pass
    attr = "hint_ar" if lang == "ar" else "hint_en"
    return str(getattr(diagnostic, attr, "") or "")


def code_of(diagnostic: Any) -> str:
    return str(getattr(diagnostic, "code", "") or "")


def where_of(diagnostic: Any) -> str:
    return str(getattr(diagnostic, "where", "") or "")


def matches(diagnostic: Any, needle: str) -> bool:
    """Whether the filter text occurs in the code or in either message."""
    if not needle:
        return True
    needle = needle.casefold()
    haystacks = (
        code_of(diagnostic), where_of(diagnostic),
        message_of(diagnostic, "ar"), message_of(diagnostic, "en"),
        hint_of(diagnostic, "ar"), hint_of(diagnostic, "en"),
    )
    return any(needle in text.casefold() for text in haystacks if text)


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def drop_language_hooks(widget: QWidget) -> None:
    """Unhook ``widget`` and its children from ``language_changed``.

    Cards are rebuilt on every keystroke in the filter box, so the discarded
    ones must stop listening before they are deleted: a slot left bound to a
    widget whose C++ side has gone is a crash waiting for the next language
    switch, not a catchable error.
    """
    for target in [widget] + widget.findChildren(QWidget):
        slot = getattr(target, "_on_language_changed", None)
        if slot is None:
            continue
        try:
            translator.language_changed.disconnect(slot)
        except (RuntimeError, TypeError):
            pass




class _SeverityChip(QLabel, Retranslatable):
    """A small outlined badge naming the severity of one diagnostic."""

    def __init__(self, severity: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._severity = severity
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum,
                           QSizePolicy.Policy.Fixed)
        self.hook_language()

    def retranslate(self) -> None:
        pal = theme.palette(current_theme())
        colour = {"error": pal.danger, "warning": pal.warning,
                  "info": pal.info}.get(self._severity, pal.info)
        self.setText(translator.t(_SEVERITY_KEY.get(self._severity, "v.info")))
        self.setStyleSheet(
            f"color: {colour}; border: 1px solid {colour};"
            f" border-radius: 9px; padding: 1px 9px; font-weight: 600;")


class DiagnosticCard(QFrame, Retranslatable):
    """One diagnostic: message, stable code, location and hint.

    The message and the hint come from the diagnostic itself rather than the
    catalogue, so they are re-read on ``language_changed`` instead of being
    driven by a key.
    """

    def __init__(self, diagnostic: Any,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._diag = diagnostic
        self._severity = severity_of(diagnostic)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # A stylesheet set on a widget also reaches its children, and QLabel
        # is a QFrame, so a bare ``QFrame { ... }`` rule would paint the
        # severity edge around every label in the card.  Name the frame and
        # select it by that name.
        self.setObjectName("diagnosticCard")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._chip = _SeverityChip(self._severity)
        top.addWidget(self._chip)
        self._code = QLabel()
        theme.set_role(self._code, "mono")
        theme.keep_ltr(self._code)
        self._code.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._code.setText(code_of(diagnostic))
        top.addWidget(self._code)
        top.addStretch(1)
        outer.addLayout(top)

        self._message = QLabel()
        self._message.setWordWrap(True)
        policy = self._message.sizePolicy()
        policy.setHeightForWidth(True)
        self._message.setSizePolicy(policy)
        outer.addWidget(self._message)

        self._where_row = QWidget()
        where_lay = QHBoxLayout(self._where_row)
        where_lay.setContentsMargins(0, 0, 0, 0)
        where_lay.setSpacing(6)
        self._where_label = TKey("v.where", role="hint")
        self._where_label.setWordWrap(False)
        where_lay.addWidget(self._where_label)
        self._where_value = QLabel(where_of(diagnostic))
        theme.set_role(self._where_value, "mono")
        theme.keep_ltr(self._where_value)
        self._where_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        where_lay.addWidget(self._where_value)
        where_lay.addStretch(1)
        self._where_row.setVisible(bool(where_of(diagnostic)))
        outer.addWidget(self._where_row)

        self._hint_row = QWidget()
        hint_lay = QVBoxLayout(self._hint_row)
        hint_lay.setContentsMargins(0, 0, 0, 0)
        hint_lay.setSpacing(1)
        self._hint_label = TKey("v.hint", role="hint")
        hint_lay.addWidget(self._hint_label)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        hint_policy = self._hint.sizePolicy()
        hint_policy.setHeightForWidth(True)
        self._hint.setSizePolicy(hint_policy)
        hint_lay.addWidget(self._hint)
        outer.addWidget(self._hint_row)

        self.hook_language()

    def retranslate(self) -> None:
        lang = translator.language
        self._message.setText(message_of(self._diag, lang))
        hint = hint_of(self._diag, lang)
        self._hint.setText(hint)
        self._hint_row.setVisible(bool(hint))
        self._code.setToolTip(translator.t("v.code"))
        self._apply_colours()

    def _apply_colours(self) -> None:
        pal = theme.palette(current_theme())
        colour = {"error": pal.danger, "warning": pal.warning,
                  "info": pal.info}.get(self._severity, pal.info)
        # Qt stylesheets do not mirror border-left, so the leading edge is
        # chosen from the interface direction.
        side = "right" if translator.is_rtl else "left"
        self.setStyleSheet(
            f"QFrame#diagnosticCard {{ background: {pal.surface_alt};"
            f" border-{side}: 3px solid {colour}; border-radius: 6px; }}")


class DiagnosticSection(QWidget, Retranslatable):
    """A collapsible group of cards with its count in the heading.

    ``Collapsible`` in ``widgets.common`` drives its caption from a single key
    and cannot carry the count, and the count is a number that must not
    mirror, so the header is assembled here: translated caption, then an
    LTR-pinned numeral.
    """

    def __init__(self, heading_key: str, *, expanded: bool = True,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._heading_key = heading_key
        self._count = 0
        #: Set once the reader folds the section away by hand, so refilling
        #: the list (a keystroke in the filter box) does not unfold it again.
        self._user_collapsed = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        header = QWidget()
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(6)
        self._button = QToolButton()
        self._button.setObjectName("sectionHeader")
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setAutoRaise(True)
        self._button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.clicked.connect(self._on_user_toggle)
        header_lay.addWidget(self._button)
        self._count_label = QLabel()
        theme.set_role(self._count_label, "mono")
        theme.keep_ltr(self._count_label)
        header_lay.addWidget(self._count_label)
        header_lay.addStretch(1)
        lay.addWidget(header)

        self._content = QWidget()
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(10, 2, 10, 6)
        self.body.setSpacing(8)
        self._content.setVisible(expanded)
        lay.addWidget(self._content)

        self.hook_language()

    # -- content ---------------------------------------------------------- #

    def clear(self) -> None:
        while self.body.count():
            item = self.body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                drop_language_hooks(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._count = 0

    def add_card(self, card: QWidget) -> QWidget:
        self.body.addWidget(card)
        self._count += 1
        self.retranslate()
        return card

    @property
    def count(self) -> int:
        return self._count

    @property
    def user_collapsed(self) -> bool:
        return self._user_collapsed

    def set_expanded(self, expanded: bool) -> None:
        self._button.setChecked(expanded)
        self._on_click(expanded)

    def _on_user_toggle(self, checked: bool) -> None:
        self._user_collapsed = not checked
        self._on_click(checked)

    def _on_click(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self.retranslate()

    def _arrow(self) -> Qt.ArrowType:
        if self._button.isChecked():
            return Qt.ArrowType.DownArrow
        return (Qt.ArrowType.LeftArrow if translator.is_rtl
                else Qt.ArrowType.RightArrow)

    def retranslate(self) -> None:
        self._button.setText("  " + translator.t(self._heading_key))
        self._button.setArrowType(self._arrow())
        self._count_label.setText(str(self._count))


# --------------------------------------------------------------------------- #
# The dialog
# --------------------------------------------------------------------------- #


class ValidationDialog(QDialog):
    """The model-check report, opened from the window's validation chip."""

    def __init__(self, state: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._report: Any = None
        self._cards: List[DiagnosticCard] = []

        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(520, 380)
        self.resize(760, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        head_lay = QVBoxLayout(head)
        head_lay.setContentsMargins(18, 14, 18, 10)
        head_lay.setSpacing(8)
        self._title = TKey("v.title", role="h1")
        head_lay.addWidget(self._title)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._filter = QLineEdit()
        theme.keep_ltr(self._filter)
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(lambda _t: self._populate())
        controls.addWidget(self._filter, 1)
        self._recheck = TButton("v.recheck", role="primary")
        self._recheck.clicked.connect(self._on_recheck)
        controls.addWidget(self._recheck)
        self._copy = TButton("act.copy")
        self._copy.clicked.connect(self._on_copy)
        controls.addWidget(self._copy)
        head_lay.addLayout(controls)
        outer.addWidget(head)
        outer.addWidget(separator())

        self.scroll = ScrollPage()
        outer.addWidget(self.scroll, 1)

        self._blocked = InfoBanner("v.blocked", kind="danger")
        self.scroll.add(self._blocked)
        # One banner per empty-state, rather than one banner whose key is
        # swapped: each keeps retranslating itself from its own key.
        self._note_ok = InfoBanner("v.no_issues", kind="success")
        self.scroll.add(self._note_ok)
        self._note_unavailable = InfoBanner("v.unavailable", kind="warning")
        self.scroll.add(self._note_unavailable)
        self._note_no_match = InfoBanner("v.no_match", kind="info")
        self.scroll.add(self._note_no_match)

        self._sections: dict = {}
        for severity, heading_key, _kind in SECTIONS:
            section = DiagnosticSection(heading_key,
                                        expanded=severity != "info")
            self._sections[severity] = section
            self.scroll.add(section)
        self.scroll.add_stretch()

        foot = QWidget()
        foot_lay = QHBoxLayout(foot)
        foot_lay.setContentsMargins(18, 10, 18, 14)
        foot_lay.setSpacing(8)
        self._status = QLabel()
        self._status_key = ""
        theme.set_role(self._status, "hint")
        foot_lay.addWidget(self._status)
        foot_lay.addStretch(1)
        self._close = TButton("act.close")
        self._close.clicked.connect(self.accept)
        foot_lay.addWidget(self._close)
        outer.addWidget(separator())
        outer.addWidget(foot)

        translator.language_changed.connect(self._on_language_changed)
        self._retranslate()
        self._report = self._read_report()
        self._populate()

    # -- report ----------------------------------------------------------- #

    def _read_report(self) -> Any:
        """The cached report, or one computed here if the state has none."""
        report = None
        try:
            report = getattr(self.state, "validation", None)
        except Exception:
            report = None
        if report is not None:
            return report
        # The state caches ``None`` both when the module is missing and when a
        # pass raised, so try once directly before giving up on it.
        try:
            from semsim.core.validation import validate
        except Exception:
            return None
        try:
            return validate(self.state.spec)
        except Exception:
            return None

    def _diagnostics(self) -> List[Any]:
        report = self._report
        if report is None:
            return []
        diagnostics = getattr(report, "diagnostics", None)
        if diagnostics is None:
            try:
                diagnostics = list(report)
            except Exception:
                return []
        return list(diagnostics)

    def _on_recheck(self) -> None:
        revalidate = getattr(self.state, "_revalidate", None)
        if callable(revalidate):
            try:
                revalidate()
            except Exception:
                pass
        self._report = self._read_report()
        self._set_status("v.rechecked")
        self._populate()

    # -- rendering -------------------------------------------------------- #

    def _populate(self) -> None:
        needle = self._filter.text().strip()
        diagnostics = self._diagnostics()
        for section in self._sections.values():
            section.clear()
        self._cards.clear()

        shown = 0
        for diagnostic in diagnostics:
            if not matches(diagnostic, needle):
                continue
            section = self._sections.get(severity_of(diagnostic))
            if section is None:
                continue
            card = DiagnosticCard(diagnostic)
            section.add_card(card)
            self._cards.append(card)
            shown += 1

        has_errors = any(severity_of(d) == "error" for d in diagnostics)
        self._blocked.setVisible(has_errors)

        for severity, _key, _kind in SECTIONS:
            section = self._sections[severity]
            section.setVisible(section.count > 0)
            if (section.count and severity != "info"
                    and not section.user_collapsed):
                section.set_expanded(True)

        unavailable = self._report is None
        self._note_unavailable.setVisible(unavailable)
        self._note_ok.setVisible(not unavailable and not diagnostics)
        self._note_no_match.setVisible(
            not unavailable and bool(diagnostics) and not shown)

        self._copy.setEnabled(bool(diagnostics))

    # -- clipboard -------------------------------------------------------- #

    def report_text(self) -> str:
        """The whole report as plain text, one block per diagnostic."""
        lang = translator.language
        lines: List[str] = [translator.t("v.title"), ""]
        diagnostics = self._diagnostics()
        if not diagnostics:
            lines.append(translator.t("v.no_issues")
                         if self._report is not None
                         else translator.t("v.unavailable"))
            return "\n".join(lines).strip() + "\n"
        for severity, heading_key, _kind in SECTIONS:
            group = [d for d in diagnostics if severity_of(d) == severity]
            if not group:
                continue
            lines.append(f"== {translator.t(heading_key)} ({len(group)}) ==")
            lines.append("")
            for diagnostic in group:
                lines.append(f"{code_of(diagnostic)}  "
                             f"[{translator.t(_SEVERITY_KEY[severity])}]")
                lines.append(message_of(diagnostic, lang))
                where = where_of(diagnostic)
                if where:
                    lines.append(f"{translator.t('v.where')}: {where}")
                hint = hint_of(diagnostic, lang)
                if hint:
                    lines.append(f"{translator.t('v.hint')}: {hint}")
                lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _on_copy(self) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.report_text())
        self._set_status("v.copied")

    def _set_status(self, key: str) -> None:
        self._status_key = key
        self._status.setText(translator.t(key))

    # -- retranslation ---------------------------------------------------- #

    def _on_language_changed(self, _lang: str) -> None:
        try:
            self._retranslate()
        except RuntimeError:
            pass

    def _retranslate(self) -> None:
        self.setWindowTitle(translator.t("v.title"))
        self._filter.setPlaceholderText(translator.t("v.filter"))
        self._copy.setToolTip(translator.t("v.copy_report"))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft
                                if translator.is_rtl
                                else Qt.LayoutDirection.LeftToRight)
        if self._status_key:
            self._status.setText(translator.t(self._status_key))


__all__ = ["ValidationDialog", "DiagnosticCard", "DiagnosticSection",
           "SECTIONS", "severity_of", "message_of", "hint_of",
           "drop_language_hooks"]

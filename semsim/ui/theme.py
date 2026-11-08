"""
semsim.ui.theme
===============
Palette, typography and the Qt stylesheet.

Two concerns live here that are easy to get wrong in an Arabic desktop app:

*   **Fonts.**  Qt's default Latin font renders Arabic acceptably on Windows but
    the metrics are poor at small sizes.  The stack below prefers the faces that
    actually ship with Windows and handle Arabic shaping well, and falls back
    through Linux/macOS equivalents so the app looks right while developing.
*   **Mirroring.**  ``Qt.RightToLeft`` flips layouts, but it must NOT flip
    numbers, statistical symbols or the path diagram.  Helpers here mark the
    widgets that have to stay left-to-right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
    from PySide6.QtWidgets import QApplication, QWidget
    _HAVE_QT = True
except Exception:  # pragma: no cover - headless
    _HAVE_QT = False
    Qt = QColor = QFont = QFontDatabase = QPalette = QApplication = QWidget = None  # type: ignore


LIGHT = "light"
DARK = "dark"

#: Faces that shape Arabic well, in preference order, with Latin fallbacks.
ARABIC_FONT_STACK: List[str] = [
    "Segoe UI",          # Windows: good Arabic coverage and metrics
    "Dubai",             # Windows 10+ Arabic UI face
    "Tahoma",            # the old reliable for Arabic on Windows
    "Noto Sans Arabic",
    "Cairo",
    "Amiri",
    "DejaVu Sans",
    "Arial",
]

#: Tabular figures for statistics tables -- columns must line up.
MONO_FONT_STACK: List[str] = [
    "Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Menlo", "Courier New",
]


@dataclass(frozen=True)
class Palette:
    """Semantic colour roles, resolved per theme."""

    name: str
    window: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    text_inverse: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    success: str
    warning: str
    danger: str
    info: str
    # Verdict chips in the results tables.
    verdict_pass_bg: str
    verdict_pass_fg: str
    verdict_fail_bg: str
    verdict_fail_fg: str
    verdict_borderline_bg: str
    verdict_borderline_fg: str
    # Path diagram.
    diagram_bg: str
    diagram_node: str
    diagram_node_border: str
    diagram_node_text: str
    diagram_dimension: str
    diagram_item: str
    diagram_edge: str
    diagram_edge_strong: str
    diagram_edge_null: str
    diagram_grid: str


LIGHT_PALETTE = Palette(
    name=LIGHT,
    window="#f4f6f8",
    surface="#ffffff",
    surface_alt="#eef1f5",
    border="#d4dae1",
    text="#17202a",
    text_muted="#5b6b7c",
    text_inverse="#ffffff",
    accent="#1b5e8c",
    accent_hover="#21719f",
    accent_pressed="#154b70",
    accent_text="#ffffff",
    success="#1d7a4c",
    warning="#9a6400",
    danger="#a62b2b",
    info="#1b5e8c",
    verdict_pass_bg="#e2f2e8",
    verdict_pass_fg="#14603a",
    verdict_fail_bg="#fbe6e6",
    verdict_fail_fg="#8c2222",
    verdict_borderline_bg="#fdf1dc",
    verdict_borderline_fg="#7d5100",
    diagram_bg="#fbfcfd",
    diagram_node="#dcebf6",
    diagram_node_border="#1b5e8c",
    diagram_node_text="#12374f",
    diagram_dimension="#e8f0e6",
    diagram_item="#f2f4f7",
    diagram_edge="#54657a",
    diagram_edge_strong="#1b5e8c",
    diagram_edge_null="#aab6c2",
    diagram_grid="#e9edf2",
)

DARK_PALETTE = Palette(
    name=DARK,
    window="#15191e",
    surface="#1d2328",
    surface_alt="#252c33",
    border="#333c45",
    text="#e8edf2",
    text_muted="#9aa8b5",
    text_inverse="#15191e",
    accent="#4a9fd4",
    accent_hover="#5cafe2",
    accent_pressed="#3a87b8",
    accent_text="#0f1419",
    success="#52c48a",
    warning="#e0b05c",
    danger="#e07878",
    info="#4a9fd4",
    verdict_pass_bg="#1d3a2c",
    verdict_pass_fg="#7fd9a8",
    verdict_fail_bg="#3d2222",
    verdict_fail_fg="#f0a0a0",
    verdict_borderline_bg="#3a2f1c",
    verdict_borderline_fg="#e8c98a",
    diagram_bg="#181d22",
    diagram_node="#23333f",
    diagram_node_border="#4a9fd4",
    diagram_node_text="#d6e6f2",
    diagram_dimension="#23342a",
    diagram_item="#242b31",
    diagram_edge="#8b9aa8",
    diagram_edge_strong="#4a9fd4",
    diagram_edge_null="#4a545e",
    diagram_grid="#222a31",
)

PALETTES: Dict[str, Palette] = {LIGHT: LIGHT_PALETTE, DARK: DARK_PALETTE}


def palette(theme: str = LIGHT) -> Palette:
    return PALETTES.get(theme, LIGHT_PALETTE)


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #


def resolve_font_family(stack: List[str]) -> str:
    """First family in ``stack`` that is actually installed."""
    if not _HAVE_QT:
        return stack[-1]
    try:
        available = set(QFontDatabase.families())
    except Exception:  # pragma: no cover
        return stack[-1]
    for family in stack:
        if family in available:
            return family
    return stack[-1]


def ui_font(point_size: int = 10, bold: bool = False) -> "QFont":
    font = QFont(resolve_font_family(ARABIC_FONT_STACK), point_size)
    font.setBold(bold)
    return font


def mono_font(point_size: int = 10) -> "QFont":
    font = QFont(resolve_font_family(MONO_FONT_STACK), point_size)
    # Tabular figures so decimal points line up down a statistics column.
    try:
        font.setStyleHint(QFont.StyleHint.Monospace)
    except Exception:  # pragma: no cover
        pass
    return font


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #


def stylesheet(theme: str = LIGHT, base_point_size: int = 10) -> str:
    """Qt stylesheet for the whole application."""
    c = palette(theme)
    family = resolve_font_family(ARABIC_FONT_STACK)
    mono = resolve_font_family(MONO_FONT_STACK)
    return f"""
* {{
    font-family: "{family}";
    font-size: {base_point_size}pt;
}}
QWidget {{
    background: {c.window};
    color: {c.text};
}}
QMainWindow, QDialog {{ background: {c.window}; }}

QLabel {{ background: transparent; }}
QLabel[role="h1"] {{ font-size: {base_point_size + 7}pt; font-weight: 600; }}
QLabel[role="h2"] {{ font-size: {base_point_size + 3}pt; font-weight: 600; }}
QLabel[role="subtitle"] {{ color: {c.text_muted}; font-size: {base_point_size}pt; }}
QLabel[role="hint"] {{ color: {c.text_muted}; font-size: {base_point_size - 1}pt; }}
QLabel[role="mono"] {{ font-family: "{mono}"; }}
QLabel[role="error"] {{ color: {c.danger}; }}
QLabel[role="warning"] {{ color: {c.warning}; }}
QLabel[role="success"] {{ color: {c.success}; }}

QFrame[role="card"] {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 8px;
}}
QFrame[role="separator"] {{ background: {c.border}; max-height: 1px; border: none; }}

QGroupBox {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 8px;
    color: {c.text};
}}

QPushButton {{
    background: {c.surface};
    color: {c.text};
    border: 1px solid {c.border};
    border-radius: 6px;
    padding: 6px 16px;
    min-height: 22px;
}}
QPushButton:hover {{ background: {c.surface_alt}; }}
QPushButton:pressed {{ background: {c.border}; }}
QPushButton:disabled {{ color: {c.text_muted}; background: {c.surface_alt}; }}
QPushButton[role="primary"] {{
    background: {c.accent};
    color: {c.accent_text};
    border: 1px solid {c.accent};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {c.accent_hover}; border-color: {c.accent_hover}; }}
QPushButton[role="primary"]:pressed {{ background: {c.accent_pressed}; }}
QPushButton[role="primary"]:disabled {{
    background: {c.surface_alt}; color: {c.text_muted}; border-color: {c.border};
}}
QPushButton[role="danger"] {{ color: {c.danger}; border-color: {c.danger}; }}
QPushButton[role="flat"] {{ background: transparent; border: none; color: {c.accent}; }}
QPushButton[role="flat"]:hover {{ text-decoration: underline; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {c.surface};
    color: {c.text};
    border: 1px solid {c.border};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
    selection-background-color: {c.accent};
    selection-color: {c.accent_text};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {c.accent}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {c.surface_alt}; color: {c.text_muted};
}}
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"] {{
    border: 1px solid {c.danger};
}}
QSpinBox, QDoubleSpinBox {{ font-family: "{mono}"; }}

QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {c.surface};
    color: {c.text};
    border: 1px solid {c.border};
    selection-background-color: {c.accent};
    selection-color: {c.accent_text};
    outline: none;
}}

QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {c.border};
    background: {c.surface};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {c.accent}; border-color: {c.accent};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background: {c.surface_alt};
}}

QTabWidget::pane {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 8px;
    top: -1px;
}}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {c.text_muted};
    padding: 8px 18px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{
    background: {c.surface};
    color: {c.text};
    border-color: {c.border};
    border-bottom-color: {c.surface};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ color: {c.text}; }}

QTableView, QTreeView, QListView {{
    background: {c.surface};
    alternate-background-color: {c.surface_alt};
    color: {c.text};
    border: 1px solid {c.border};
    border-radius: 6px;
    gridline-color: {c.border};
    selection-background-color: {c.accent};
    selection-color: {c.accent_text};
    outline: none;
}}
QTableView {{ font-family: "{mono}"; }}
QTreeView, QListView {{ font-family: "{family}"; }}
QHeaderView::section {{
    background: {c.surface_alt};
    color: {c.text};
    border: none;
    border-right: 1px solid {c.border};
    border-bottom: 1px solid {c.border};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {c.surface_alt}; border: none; }}
QTreeView::item, QListView::item {{ padding: 4px 2px; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: {c.border}; border-radius: 5px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {c.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{
    background: {c.surface_alt};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: {c.text};
}}
QProgressBar::chunk {{ background: {c.accent}; border-radius: 4px; }}

QMenuBar {{ background: {c.surface}; border-bottom: 1px solid {c.border}; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; }}
QMenuBar::item:selected {{ background: {c.surface_alt}; }}
QMenu {{
    background: {c.surface};
    color: {c.text};
    border: 1px solid {c.border};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 24px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c.accent}; color: {c.accent_text}; }}
QMenu::item:disabled {{ color: {c.text_muted}; }}
QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 8px; }}

QStatusBar {{ background: {c.surface}; border-top: 1px solid {c.border}; color: {c.text_muted}; }}
QStatusBar::item {{ border: none; }}

QToolTip {{
    background: {c.surface};
    color: {c.text};
    border: 1px solid {c.border};
    border-radius: 6px;
    padding: 6px 8px;
}}

QSplitter::handle {{ background: {c.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

QToolBar {{ background: {c.surface}; border: none; border-bottom: 1px solid {c.border}; spacing: 4px; padding: 4px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; }}
QToolButton:hover {{ background: {c.surface_alt}; border-color: {c.border}; }}
QToolButton:checked {{ background: {c.accent}; color: {c.accent_text}; }}
"""


# --------------------------------------------------------------------------- #
# Application helpers
# --------------------------------------------------------------------------- #


def apply(app: "QApplication", theme: str = LIGHT, rtl: bool = True,
          base_point_size: int = 10) -> None:
    """Apply palette, font, direction and stylesheet to a running app."""
    if not _HAVE_QT:
        return
    c = palette(theme)
    app.setStyle("Fusion")
    app.setFont(ui_font(base_point_size))
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl
                           else Qt.LayoutDirection.LeftToRight)

    qp = QPalette()
    qp.setColor(QPalette.ColorRole.Window, QColor(c.window))
    qp.setColor(QPalette.ColorRole.WindowText, QColor(c.text))
    qp.setColor(QPalette.ColorRole.Base, QColor(c.surface))
    qp.setColor(QPalette.ColorRole.AlternateBase, QColor(c.surface_alt))
    qp.setColor(QPalette.ColorRole.Text, QColor(c.text))
    qp.setColor(QPalette.ColorRole.Button, QColor(c.surface))
    qp.setColor(QPalette.ColorRole.ButtonText, QColor(c.text))
    qp.setColor(QPalette.ColorRole.Highlight, QColor(c.accent))
    qp.setColor(QPalette.ColorRole.HighlightedText, QColor(c.accent_text))
    qp.setColor(QPalette.ColorRole.ToolTipBase, QColor(c.surface))
    qp.setColor(QPalette.ColorRole.ToolTipText, QColor(c.text))
    qp.setColor(QPalette.ColorRole.PlaceholderText, QColor(c.text_muted))
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                QColor(c.text_muted))
    qp.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                QColor(c.text_muted))
    app.setPalette(qp)
    app.setStyleSheet(stylesheet(theme, base_point_size))


def keep_ltr(widget: "QWidget") -> "QWidget":
    """Pin a widget to left-to-right.

    Numbers, model syntax, column names and the path diagram must not mirror
    when the interface is Arabic: a correlation matrix read right-to-left is
    simply wrong, and ``lavaan`` syntax has to be copy-pasteable.
    """
    if _HAVE_QT and widget is not None:
        widget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    return widget


def set_role(widget: "QWidget", role: str) -> "QWidget":
    """Tag a widget so the stylesheet's ``[role="..."]`` rules apply."""
    if _HAVE_QT and widget is not None:
        widget.setProperty("role", role)
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)
    return widget


def verdict_colours(theme: str, verdict: str) -> "tuple[str, str]":
    """``(background, foreground)`` for a pass/fail/borderline chip."""
    c = palette(theme)
    return {
        "pass": (c.verdict_pass_bg, c.verdict_pass_fg),
        "fail": (c.verdict_fail_bg, c.verdict_fail_fg),
        "borderline": (c.verdict_borderline_bg, c.verdict_borderline_fg),
    }.get(verdict, (c.surface_alt, c.text_muted))


__all__ = [
    "LIGHT", "DARK", "Palette", "LIGHT_PALETTE", "DARK_PALETTE", "PALETTES",
    "palette", "ARABIC_FONT_STACK", "MONO_FONT_STACK", "resolve_font_family",
    "ui_font", "mono_font", "stylesheet", "apply", "keep_ltr", "set_role",
    "verdict_colours",
]

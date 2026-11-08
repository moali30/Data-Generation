"""
semsim.ui.widgets.tables
========================
Displaying pandas frames in Qt.

Statistics tables have requirements an ordinary table view does not meet:

*   Decimals must line up, so numeric cells render monospaced, right-aligned
    and padded to a fixed precision -- ``0.80`` not ``0.8``.
*   A p-value column wants significance stars and emphasis, and a verdict
    column wants its pass/fail colour.
*   The table itself must stay left-to-right even when the interface is
    Arabic: a correlation matrix read right-to-left is simply the wrong matrix.
    Header *text* is still translated; only the column order is pinned.
*   Copying must produce something that pastes into Excel or Word intact.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QSortFilterProxyModel
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableView, QVBoxLayout, QWidget,
)

from semsim.ui import theme
from semsim.ui.i18n import translator

#: Columns whose name matches one of these get special rendering.
P_VALUE_COLUMNS = {"p", "p_value", "pvalue", "p-value", "sig", "pclose",
                   "shapiro_p", "p_close"}
VERDICT_COLUMNS = {"verdict", "pass", "result", "decision", "status",
                   "requested", "realised", "significance_ok"}
#: Default decimal places by column name fragment.
PRECISION_HINTS: Dict[str, int] = {
    "p": 4, "chi": 3, "rmsea": 4, "cfi": 3, "tli": 3, "srmr": 4, "gfi": 3,
    "alpha": 3, "ave": 3, "cr": 3, "omega": 3, "htmt": 3, "loading": 3,
    "est": 3, "se": 4, "z": 3, "std": 3, "mean": 3, "sd": 3, "power": 3,
    "skew": 3, "kurt": 3, "r2": 3, "r_squared": 3, "correlation": 3,
    "injected_rmsea": 4, "realised_rmsea": 4, "target_rmsea": 4,
    "deviation": 4, "specified": 3, "estimate": 3,
}
DEFAULT_PRECISION = 3

#: Column names that map onto catalogue keys without the caller saying so.
DEFAULT_HEADER_KEYS: Dict[str, str] = {
    "index": "r.index", "value": "r.value", "threshold": "r.threshold",
    "verdict": "r.verdict", "interpretation": "r.interpretation",
    "estimate": "r.estimate", "est": "r.estimate", "se": "r.se", "z": "r.z",
    "p": "r.p", "std": "r.std", "lower": "r.ci_lower", "upper": "r.ci_upper",
    "source": "s.source", "target": "s.target", "item": "m.item",
    "factor": "m.construct", "name": "m.name", "n": "r.n", "n_items": "m.n_items",
    "label": "r.index", "specified": "t.specified", "realised": "t.achieved",
    "requested": "t.requested", "kind": "m.construct",
    "injected_rmsea": "t.injected", "realised_rmsea": "t.realised_rmsea",
    "target_rmsea": "t.target_rmsea", "deviation": "t.deviation",
    "draws": "t.draws", "significance_ok": "t.significance_ok",
    "n_required": "p.recommended",
    "mean": "r.mean", "sd": "r.sd", "median": "r.median",
    "minimum": "r.min", "maximum": "r.max", "n_missing": "r.missing",
    "skewness": "d.skewness", "kurtosis": "d.kurtosis",
    "cronbach_alpha": "r.alpha", "alpha": "r.alpha",
    "composite_reliability": "r.cr", "cr": "r.cr", "ave": "r.ave",
    "omega": "r.omega", "omega_total": "r.omega", "htmt": "r.htmt",
    "power": "p.achieved_power", "df": "p.df", "loading": "m.loading",
    "coefficient": "s.coefficient", "correlation": "s.correlation",
    "r_squared": "r.r_squared", "floor_pct": "r.floor",
    "ceiling_pct": "r.ceiling", "round": "t.round",
}
#: Columns holding counts render without decimals.
INTEGER_COLUMNS = {"n", "df", "n_items", "n_free", "count", "n_missing",
                   "iterations", "round", "replications", "baseline_df",
                   "draws", "rounds", "n_required", "n_recommended",
                   "n_categories", "n_cases"}


#: Fragment matching is only safe for fragments long enough to be distinctive.
#: "p" appears inside "Task performance"; matching on it would print a factor's
#: item count as "4.0000".
MIN_FRAGMENT_LENGTH = 3


def _precision_for(column: str) -> int:
    name = str(column).strip().lower()
    if name in INTEGER_COLUMNS:
        return 0
    if name in PRECISION_HINTS:
        return PRECISION_HINTS[name]
    for fragment, places in PRECISION_HINTS.items():
        if len(fragment) >= MIN_FRAGMENT_LENGTH and fragment in name:
            return places
    return DEFAULT_PRECISION


def format_number(value: Any, places: int) -> str:
    """Fixed-precision rendering that keeps a column's decimals aligned."""
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return ""
    if math.isinf(f):
        return "∞" if f > 0 else "-∞"
    if places == 0:
        return f"{int(round(f)):d}"
    # A p-value that rounds to zero is reported as a bound, never as "0.0000":
    # "p = 0" is not a claim any statistician makes.
    return f"{f:.{places}f}"


def _translate_verdict(value: Any) -> str:
    """Render a verdict cell in the interface language.

    Accepts the strings the analysis modules emit, plus plain booleans, and
    passes anything unrecognised (a dash, a note) through untouched.
    """
    if isinstance(value, (bool, np.bool_)):
        return translator.t("verdict.pass" if value else "verdict.fail")
    key = str(value).strip().lower()
    mapped = {
        "pass": "verdict.pass", "true": "verdict.pass", "yes": "verdict.pass",
        "ok": "verdict.pass", "met": "verdict.pass",
        "fail": "verdict.fail", "false": "verdict.fail", "no": "verdict.fail",
        "not met": "verdict.fail",
        "borderline": "verdict.borderline", "marginal": "verdict.borderline",
        "significant": "verdict.significant",
        "not significant": "verdict.not_significant",
        "non-significant": "verdict.not_significant",
    }.get(key)
    return translator.t(mapped) if mapped else str(value)


def significance_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


class DataFrameModel(QAbstractTableModel):
    """A read-only Qt model over a pandas DataFrame."""

    def __init__(self, frame: Optional[pd.DataFrame] = None, *,
                 precision: Optional[Dict[str, int]] = None,
                 header_keys: Optional[Dict[str, str]] = None,
                 show_stars: bool = True,
                 max_rows: Optional[int] = None,
                 row_key_column: Optional[str] = "index",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame() if frame is None else frame
        self._precision = dict(precision or {})
        self._header_keys = {**DEFAULT_HEADER_KEYS, **(header_keys or {})}
        # In a long-format table ("index", "value") the natural precision
        # belongs to the ROW -- df wants none, RMSEA wants four -- so the
        # statistic's own name drives it when such a column exists.
        self._row_key_column = row_key_column
        self._show_stars = show_stars
        self._max_rows = max_rows
        self._theme = theme.LIGHT
        translator.language_changed.connect(self._on_language_changed)

    # -- data ------------------------------------------------------------ #

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = pd.DataFrame() if frame is None else frame
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def set_theme(self, name: str) -> None:
        self._theme = name
        if self.rowCount():
            top = self.index(0, 0)
            bottom = self.index(self.rowCount() - 1, max(self.columnCount() - 1, 0))
            self.dataChanged.emit(top, bottom, [Qt.ItemDataRole.ForegroundRole])

    def _on_language_changed(self, _lang: str) -> None:
        try:
            self.headerDataChanged.emit(
                Qt.Orientation.Horizontal, 0, max(self.columnCount() - 1, 0))
        except RuntimeError:
            pass

    # -- Qt model interface ---------------------------------------------- #

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        n = len(self._frame.index)
        return min(n, self._max_rows) if self._max_rows else n

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.columns)

    def _column_name(self, col: int) -> str:
        return str(self._frame.columns[col])

    def _row_precision(self, row: int) -> Optional[int]:
        """Precision implied by this row's statistic name, if any."""
        key = self._row_key_column
        if not key or key not in self._frame.columns:
            return None
        try:
            label = str(self._frame[key].iloc[row])
        except (IndexError, KeyError):
            return None
        name = label.strip().lower()
        if name in INTEGER_COLUMNS:
            return 0
        if name in PRECISION_HINTS:
            return PRECISION_HINTS[name]
        for fragment, places in PRECISION_HINTS.items():
            if len(fragment) >= MIN_FRAGMENT_LENGTH and fragment in name:
                return places
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row >= self.rowCount() or col >= self.columnCount():
            return None
        name = self._column_name(col)
        value = self._frame.iloc[row, col]
        lower = name.strip().lower()
        pal = theme.palette(self._theme)

        if role == Qt.ItemDataRole.DisplayRole:
            if lower in VERDICT_COLUMNS:
                return _translate_verdict(value)
            if isinstance(value, (bool, np.bool_)):
                return translator.t("act.yes" if value else "act.no")
            if isinstance(value, (int, float, np.integer, np.floating)):
                places = self._precision.get(name)
                if places is None:
                    places = self._row_precision(row)
                if places is None:
                    places = _precision_for(name)
                text = format_number(value, places)
                if self._show_stars and lower in P_VALUE_COLUMNS:
                    try:
                        f = float(value)
                        if not math.isnan(f):
                            if f < 10 ** (-places):
                                text = "<" + format_number(10 ** (-places), places)
                            text += significance_stars(f)
                    except (TypeError, ValueError):
                        pass
                return text
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return ""
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
                return int(Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.ForegroundRole:
            if lower in VERDICT_COLUMNS:
                key = str(value).strip().lower()
                mapped = {"pass": "pass", "true": "pass", "yes": "pass",
                          "fail": "fail", "false": "fail", "no": "fail",
                          "borderline": "borderline"}.get(key, "")
                if mapped:
                    _bg, fg = theme.verdict_colours(self._theme, mapped)
                    return QBrush(QColor(fg))
            if lower in P_VALUE_COLUMNS:
                try:
                    f = float(value)
                    if not math.isnan(f):
                        return QBrush(QColor(
                            pal.success if f < 0.05 else pal.text_muted))
                except (TypeError, ValueError):
                    pass
            return None

        if role == Qt.ItemDataRole.FontRole:
            if lower in VERDICT_COLUMNS or lower in P_VALUE_COLUMNS:
                font = QFont()
                font.setBold(True)
                return font
            return None

        if role == Qt.ItemDataRole.ToolTipRole:
            return self.data(index, Qt.ItemDataRole.DisplayRole)

        if role == Qt.ItemDataRole.UserRole:
            return value          # raw value, used for sorting

        return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if section >= len(self._frame.columns):
                return None
            name = self._column_name(section)
            key = self._header_keys.get(name)
            return translator.t(key) if key else name
        if section >= len(self._frame.index):
            return None
        return str(self._frame.index[section])

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

    # -- export ----------------------------------------------------------- #

    def to_tsv(self, rows: Optional[Sequence[int]] = None) -> str:
        """Tab-separated text that pastes cleanly into Excel or Word."""
        cols = list(self._frame.columns)
        header_key = self._header_keys
        heads = [translator.t(header_key[c]) if c in header_key else str(c)
                 for c in cols]
        lines = ["\t".join(heads)]
        indices = rows if rows is not None else range(self.rowCount())
        for r in indices:
            cells = []
            for c in range(len(cols)):
                cells.append(str(self.data(self.index(r, c),
                                           Qt.ItemDataRole.DisplayRole) or ""))
            lines.append("\t".join(cells))
        return "\n".join(lines)


class _SortProxy(QSortFilterProxyModel):
    """Sort on the raw value, not the formatted string."""

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        a = self.sourceModel().data(left, Qt.ItemDataRole.UserRole)
        b = self.sourceModel().data(right, Qt.ItemDataRole.UserRole)
        try:
            fa, fb = float(a), float(b)
            if math.isnan(fa):
                return False
            if math.isnan(fb):
                return True
            return fa < fb
        except (TypeError, ValueError):
            return str(a) < str(b)


class DataFrameView(QTableView):
    """A table view configured the way a statistics table should behave."""

    def __init__(self, frame: Optional[pd.DataFrame] = None, *,
                 precision: Optional[Dict[str, int]] = None,
                 header_keys: Optional[Dict[str, str]] = None,
                 sortable: bool = True,
                 stretch_last: bool = True,
                 max_rows: Optional[int] = None,
                 row_key_column: Optional[str] = "index",
                 hide_columns: Optional[Sequence[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hidden = list(hide_columns or [])
        self._model = DataFrameModel(frame, precision=precision,
                                     header_keys=header_keys,
                                     max_rows=max_rows,
                                     row_key_column=row_key_column, parent=self)
        self._proxy = _SortProxy(self)
        self._proxy.setSourceModel(self._model)
        self.setModel(self._proxy)

        # A matrix must not mirror: its columns carry meaning in order.
        theme.keep_ltr(self)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setWordWrap(False)
        self.setShowGrid(True)
        self.setSortingEnabled(sortable)
        if sortable:
            # Qt sorts by column 0 the moment sorting is enabled. A fit-index
            # table has a meaningful row order, so start unsorted and let the
            # user opt in by clicking a header.
            self.sortByColumn(-1, Qt.SortOrder.AscendingOrder)
            self.horizontalHeader().setSortIndicatorShown(False)
            self.horizontalHeader().sortIndicatorChanged.connect(
                lambda *_a: self.horizontalHeader().setSortIndicatorShown(True))
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(stretch_last)
        self._stretch_last = stretch_last
        header.setHighlightSections(False)

        copy = QShortcut(QKeySequence.StandardKey.Copy, self)
        copy.activated.connect(self.copy_selection)

    # -- data ------------------------------------------------------------ #

    def set_frame(self, frame: pd.DataFrame) -> None:
        self._model.set_frame(frame)
        self.resizeColumnsToContents()
        # ResizeToContents leaves no slack, so the final column ends up under
        # the scrollbar. Give every column a little breathing room instead.
        header = self.horizontalHeader()
        for col in range(self._model.columnCount()):
            header.resizeSection(col, header.sectionSize(col) + 14)
        # A hidden column can still drive row-aware precision: the fit table
        # keys its decimals off the statistic's machine code while showing the
        # reader its translated name.
        names = [str(c) for c in self._model.frame.columns]
        for col, name in enumerate(names):
            self.setColumnHidden(col, name in self._hidden)

    @property
    def frame(self) -> pd.DataFrame:
        return self._model.frame

    def set_theme(self, name: str) -> None:
        self._model.set_theme(name)

    def set_index_visible(self, visible: bool) -> None:
        self.verticalHeader().setVisible(visible)

    # -- clipboard -------------------------------------------------------- #

    def copy_selection(self) -> None:
        """Copy the selected block, or the whole table when nothing is selected."""
        selection = self.selectionModel()
        clip = QGuiApplication.clipboard()
        if clip is None:
            return
        if selection is None or not selection.hasSelection():
            clip.setText(self._model.to_tsv())
            return
        indexes = sorted(selection.selectedIndexes(),
                         key=lambda ix: (ix.row(), ix.column()))
        if not indexes:
            clip.setText(self._model.to_tsv())
            return
        rows: Dict[int, Dict[int, str]] = {}
        for ix in indexes:
            text = ix.data(Qt.ItemDataRole.DisplayRole)
            rows.setdefault(ix.row(), {})[ix.column()] = str(text or "")
        cols = sorted({c for r in rows.values() for c in r})
        lines = []
        for r in sorted(rows):
            lines.append("\t".join(rows[r].get(c, "") for c in cols))
        clip.setText("\n".join(lines))

    def ideal_height(self, max_rows: int = 14) -> int:
        """Height that shows up to ``max_rows`` without an inner scrollbar."""
        rows = min(self._proxy.rowCount(), max_rows)
        header = self.horizontalHeader().height()
        return header + rows * self.verticalHeader().defaultSectionSize() + 8


class TablePanel(QWidget):
    """A titled card holding one table, sized to its content."""

    def __init__(self, title_key: str, frame: Optional[pd.DataFrame] = None,
                 *, max_rows: int = 14, both: bool = False,
                 hide_columns: Optional[Sequence[str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        from semsim.ui.widgets.common import Card
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._card = Card(title_key, both=both)
        lay.addWidget(self._card)
        self.table = DataFrameView(frame, hide_columns=hide_columns)
        self._card.add(self.table)
        self._max_rows = max_rows
        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.table.set_frame(frame)
        height = min(self.table.ideal_height(self._max_rows), 420)
        self.table.setMinimumHeight(height)
        # Without a ceiling a three-row table stretches to fill the page.
        self.table.setMaximumHeight(max(height, 90))

    def set_theme(self, name: str) -> None:
        self.table.set_theme(name)


__all__ = [
    "DataFrameModel", "DataFrameView", "TablePanel",
    "format_number", "significance_stars", "DEFAULT_HEADER_KEYS",
]

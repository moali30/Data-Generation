"""
semsim.ui.pages.power
=====================
Stage 4 -- Power Analysis & Sample Size.

Three sections, read top to bottom the way the decision is actually made:

*   **Model complexity** -- the three numbers that drive every formal power
    calculation: observed moments, free parameters and the degrees of freedom
    left over.  They are read straight off the spec and re-read on every
    ``spec_changed``, so the strip never lags behind an edit made on another
    page.  A model with ``df <= 0`` has no global fit to have power *against*,
    so the badge turns red and a warning says why.
*   **Settings** -- what :class:`~semsim.core.spec.PowerSpec` holds.
*   **Results** -- the recommendation, the heuristics, the per-path table and a
    power curve.  The analysis runs on a :class:`QThread` worker, because a
    Monte-Carlo run is tens of seconds of work and a frozen window looks like a
    crashed one.  :mod:`semsim.power.analysis` is imported *inside* the worker
    and guarded: when it is missing the page falls back to the rules of thumb
    it can compute itself and says so.
"""

from __future__ import annotations

import copy
import math
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QRadioButton, QSizePolicy, QVBoxLayout, QWidget,
)

from semsim.core.spec import PowerMethod
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.pages.base import StagePage
from semsim.ui.widgets.common import (
    Card, Field, InfoBanner, Retranslatable, StatBadge, TButton, TKey,
    current_theme, float_spin, int_spin,
)
from semsim.ui.widgets.tables import TablePanel

#: Kline's (2016) floor, used by the inline fallback when the analysis module
#: is unavailable.  The real module carries its own copy of the same constant.
KLINE_MINIMUM_N = 200

#: ``PowerMethod`` member -> catalogue key.  The enum *values* are
#: ``rmsea_close_fit`` / ``satorra_saris`` / ..., while the catalogue keys are
#: ``p.method.close_fit`` / ``p.method.satorra`` / ..., so the combo is built
#: from explicit pairs rather than from a key prefix.
METHOD_KEYS: Tuple[Tuple[PowerMethod, str], ...] = (
    (PowerMethod.RMSEA_CLOSE_FIT, "p.method.close_fit"),
    (PowerMethod.SATORRA_SARIS, "p.method.satorra"),
    (PowerMethod.MONTE_CARLO, "p.method.monte_carlo"),
    (PowerMethod.RULE_OF_THUMB, "p.method.rule"),
)

def _as_method(value: Any) -> PowerMethod:
    """Coerce whatever the combo handed back into a real enum member.

    ``PowerMethod`` mixes in :class:`str`, and Qt stores a ``str`` subclass in
    item data as a plain string, so ``currentData()`` comes back as
    ``"monte_carlo"`` rather than the member.  The analysis module compares its
    method with ``is``, so the member has to be restored before it is written
    into the spec.
    """
    try:
        return PowerMethod(getattr(value, "value", value))
    except (ValueError, TypeError):
        return PowerMethod.RMSEA_CLOSE_FIT


def _fit_table(panel: TablePanel, max_rows: int) -> None:
    """Make one table exactly as tall and as wide as it needs to be.

    ``DataFrameView.set_frame`` pads every column, the last one included, so a
    two-column table can end up a few pixels wider than its viewport -- which
    clips the final digit of a right-aligned number.  Stretching the last
    section instead makes it land exactly on the edge, and pinning the height
    drops the empty strip under the final row.
    """
    table = panel.table
    header = table.horizontalHeader()
    if header.count():
        # ``stretchLastSection`` only ever *grows* the final column, so it has
        # to be turned off before the stretch resize mode can shrink it.
        header.setStretchLastSection(False)
        header.setSectionResizeMode(header.count() - 1,
                                    QHeaderView.ResizeMode.Stretch)
    height = table.ideal_height(max_rows)
    table.setMinimumHeight(height)
    table.setMaximumHeight(height)


#: ``PowerResult.rule_of_thumb`` key -> catalogue key.
RULE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("n_per_parameter", "p.rule_per_param"),
    ("n_per_item", "p.rule_per_item"),
    ("minimum_200", "p.kline_min"),
    ("absolute_floor_5_to_1", "p.floor_5_to_1"),
)


# --------------------------------------------------------------------------- #
# Small widgets this page needs and the common set does not carry
# --------------------------------------------------------------------------- #


class _TRadio(QRadioButton, Retranslatable):
    """A radio button driven by an i18n key."""

    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        self.hook_language()

    def retranslate(self) -> None:
        self.setText(translator.t(self._key))


class _KeyedCombo(QComboBox, Retranslatable):
    """A combo over ``(member, i18n_key)`` pairs.

    :class:`~semsim.ui.widgets.common.EnumCombo` builds its keys as
    ``prefix.value``, which the power-method catalogue entries do not follow,
    so the pairs are spelled out instead of guessed.
    """

    value_changed = Signal(object)

    def __init__(self, pairs: Sequence[Tuple[Any, str]],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pairs: List[Tuple[Any, str]] = list(pairs)
        for member, _key in self._pairs:
            self.addItem("", member)
        self.currentIndexChanged.connect(
            lambda _i: self.value_changed.emit(self.value()))
        self.hook_language()

    def value(self) -> Any:
        return self.currentData()

    def set_value(self, member: Any) -> None:
        for index, (candidate, _key) in enumerate(self._pairs):
            if candidate == member:
                blocked = self.blockSignals(True)
                self.setCurrentIndex(index)
                self.blockSignals(blocked)
                return

    def retranslate(self) -> None:
        for index, (_member, key) in enumerate(self._pairs):
            self.setItemText(index, translator.t(key))


class _PowerCurveView(QWidget):
    """The power curve, painted by hand.

    Pinned left-to-right: a curve whose horizontal axis counts *up* to the
    right is the only way anyone reads a power curve, in Arabic as in English.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        theme.keep_ltr(self)
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self._theme = current_theme()
        self._points: List[Tuple[int, float]] = []
        self._recommended = 0
        self._reference = 0.80
        translator.language_changed.connect(self._on_language_changed)

    def _on_language_changed(self, _lang: str) -> None:
        try:
            self.update()
        except RuntimeError:          # pragma: no cover - widget already gone
            pass

    def set_theme(self, name: str) -> None:
        self._theme = name
        self.update()

    def set_curve(self, frame: Any, recommended: int = 0,
                  reference: float = 0.80) -> bool:
        """Load ``(n, power)`` pairs.  Returns False when there is nothing to draw."""
        points: List[Tuple[int, float]] = []
        columns = list(getattr(frame, "columns", ()))
        if frame is not None and len(frame) and "n" in columns and "power" in columns:
            for raw_n, raw_power in zip(frame["n"], frame["power"]):
                try:
                    n = int(raw_n)
                    power = float(raw_power)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(power):
                    continue
                points.append((n, min(max(power, 0.0), 1.0)))
        points.sort(key=lambda pair: pair[0])
        self._points = points
        self._recommended = int(recommended or 0)
        self._reference = float(reference) if math.isfinite(reference) else 0.80
        self.update()
        return bool(points)

    # -- painting -------------------------------------------------------- #

    def paintEvent(self, event) -> None:     # noqa: N802 - Qt naming
        pal = theme.palette(self._theme)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(pal.surface))
            left, right, top, bottom = 54.0, 20.0, 22.0, 42.0
            width = max(float(self.width()) - left - right, 20.0)
            height = max(float(self.height()) - top - bottom, 20.0)
            plot = QRectF(left, top, width, height)
            painter.fillRect(plot, QColor(pal.diagram_bg))
            if not self._points:
                painter.end()
                return

            xs = [p[0] for p in self._points]
            x_min, x_max = float(min(xs)), float(max(xs))
            if x_max <= x_min:
                x_max = x_min + 1.0

            def x_of(n: float) -> float:
                return plot.left() + (n - x_min) / (x_max - x_min) * plot.width()

            def y_of(power: float) -> float:
                return plot.bottom() - power * plot.height()

            painter.setFont(theme.mono_font(8))
            # Horizontal grid, power 0.0 .. 1.0
            for step in range(6):
                value = step / 5.0
                y = y_of(value)
                painter.setPen(QPen(QColor(pal.diagram_grid), 1.0))
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                painter.setPen(QPen(QColor(pal.text_muted), 1.0))
                painter.drawText(
                    QRectF(2.0, y - 9.0, plot.left() - 8.0, 18.0),
                    int(Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter),
                    f"{value:.1f}")
            # Vertical grid and the sample sizes under it
            for step in range(5):
                fraction = step / 4.0
                x = plot.left() + fraction * plot.width()
                painter.setPen(QPen(QColor(pal.diagram_grid), 1.0))
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
                painter.setPen(QPen(QColor(pal.text_muted), 1.0))
                painter.drawText(
                    QRectF(x - 34.0, plot.bottom() + 4.0, 68.0, 16.0),
                    int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                    f"{int(round(x_min + fraction * (x_max - x_min)))}")

            painter.setPen(QPen(QColor(pal.border), 1.0))
            painter.drawRect(plot)

            # The level the recommendation aims at.
            y_ref = y_of(min(max(self._reference, 0.0), 1.0))
            painter.setPen(QPen(QColor(pal.warning), 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(plot.left(), y_ref),
                             QPointF(plot.right(), y_ref))
            painter.setPen(QPen(QColor(pal.warning), 1.0))
            painter.drawText(
                QRectF(plot.left() + 6.0, y_ref - 17.0, 70.0, 15.0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{self._reference:.2f}")

            # The recommended N.
            if self._recommended and x_min <= self._recommended <= x_max:
                x_rec = x_of(float(self._recommended))
                painter.setPen(QPen(QColor(pal.success), 1.4, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(x_rec, plot.top()),
                                 QPointF(x_rec, plot.bottom()))
                label = QRectF(0.0, plot.top() - 19.0, 120.0, 16.0)
                label.moveLeft(min(max(x_rec - 60.0, plot.left()),
                                   plot.right() - 120.0))
                painter.setPen(QPen(QColor(pal.success), 1.0))
                painter.drawText(
                    label,
                    int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                    f"N = {self._recommended}")

            polygon = QPolygonF([QPointF(x_of(float(n)), y_of(power))
                                 for n, power in self._points])
            painter.setPen(QPen(QColor(pal.accent), 2.0))
            painter.drawPolyline(polygon)

            # Axis captions, from the catalogue, in the interface language.
            painter.setFont(theme.ui_font(8))
            painter.setPen(QPen(QColor(pal.text_muted), 1.0))
            painter.drawText(
                QRectF(plot.left(), float(self.height()) - 21.0,
                       plot.width(), 18.0),
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                translator.t("p.n"))
            painter.drawText(
                QRectF(2.0, 2.0, plot.width(), 16.0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                translator.t("p.axis_power"))
        finally:
            if painter.isActive():
                painter.end()


# --------------------------------------------------------------------------- #
# Background work
# --------------------------------------------------------------------------- #


class _PowerWorker(QObject):
    """Runs :func:`semsim.power.analysis.analyse` off the GUI thread."""

    finished = Signal(object)
    unavailable = Signal()
    failed = Signal(str, str)

    def __init__(self, spec: Any) -> None:
        super().__init__()
        self._spec = spec

    def run(self) -> None:
        try:
            from semsim.power.analysis import analyse
        except Exception:
            # The analysis layer is not in this build; the page degrades to the
            # heuristics it can work out on its own.
            self.unavailable.emit()
            return
        try:
            self.finished.emit(analyse(self._spec))
        except Exception as exc:                  # noqa: BLE001 - reported, not swallowed
            detail = "".join(
                traceback.format_exception_only(type(exc), exc)).strip()
            message_ar = str(getattr(exc, "message_ar", "") or detail)
            message_en = str(getattr(exc, "message_en", "") or detail)
            self.failed.emit(message_ar, message_en)


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


class PowerPage(StagePage):
    TITLE_KEY = "stage.4.title"
    SUBTITLE_KEY = "stage.4.subtitle"

    def build(self) -> None:
        self._thread: Optional[QThread] = None
        self._worker: Optional[_PowerWorker] = None
        self._result: Any = None
        self._stale = False
        self._error: Tuple[str, str] = ("", "")

        self.add_banner("p.explain")
        self._build_complexity()
        self._build_settings()
        self._build_results()
        self.scroll.add_stretch()

        # The complexity strip describes the model as it is *now*, so it
        # follows every value edit, not only structural ones.
        self.state.spec_changed.connect(self._on_spec_changed)
        self.refresh()

    def _on_spec_changed(self) -> None:
        self._refresh_complexity()
        # A recommendation computed from an earlier version of the model is
        # not a recommendation for this one.  The numbers stay on screen --
        # throwing them away mid-edit helps nobody -- but they stop claiming
        # to be current.
        if self._result is not None and not self._stale:
            self._stale = True
            self._status.set_key("p.stale")
            self._status.setVisible(True)

    # ------------------------------------------------------------------ #
    # Section 1 -- model complexity
    # ------------------------------------------------------------------ #

    def _build_complexity(self) -> None:
        card = self.add_card("p.complexity")
        strip = QHBoxLayout()
        strip.setSpacing(10)
        self._badge_moments = StatBadge("p.moments")
        self._badge_free = StatBadge("p.free_params")
        self._badge_df = StatBadge("p.df")
        for badge in (self._badge_moments, self._badge_free, self._badge_df):
            theme.set_role(badge, "card")
            badge.setMinimumWidth(150)
            strip.addWidget(badge)
        strip.addStretch(1)
        card.add(strip)
        self._df_warning = InfoBanner("p.df.negative", kind="warning")
        self._df_warning.setVisible(False)
        card.add(self._df_warning)

    def _complexity_numbers(self) -> Tuple[int, int, int]:
        """``(moments, free parameters, df)`` -- never raises."""
        try:
            moments = int(self.spec.count_moments())
        except Exception:
            moments = 0
        try:
            free = int(self.spec.count_free_parameters())
        except Exception:
            free = 0
        return moments, free, moments - free

    def _refresh_complexity(self) -> None:
        moments, free, df = self._complexity_numbers()
        self._badge_moments.set_value(moments)
        self._badge_free.set_value(free)
        self._badge_df.set_value(df, kind="fail" if df <= 0 else "")
        self._df_warning.setVisible(df <= 0)

    # ------------------------------------------------------------------ #
    # Section 2 -- settings
    # ------------------------------------------------------------------ #

    def _build_settings(self) -> None:
        card = self.add_card("p.settings")

        self._mode_suggest = _TRadio("p.mode.suggest")
        self._mode_fixed = _TRadio("p.mode.fixed")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_suggest)
        self._mode_group.addButton(self._mode_fixed)
        mode_host = QWidget()
        # A bare QWidget picks up the window colour from the global stylesheet,
        # which would paint a grey strip across the white card behind it.
        mode_host.setStyleSheet("background: transparent;")
        mode_row = QHBoxLayout(mode_host)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(18)
        mode_row.addWidget(self._mode_suggest)
        mode_row.addWidget(self._mode_fixed)
        mode_row.addStretch(1)
        self._mode_fixed.toggled.connect(self._on_mode)
        card.add(Field("p.mode", mode_host, ltr_editor=False))

        self._n = int_spin(10, 100_000, 300, step=10)
        self._n.valueChanged.connect(self._on_n)
        card.add(Field("p.n", self._n, hint_key="p.n.hint"))

        self._target_power = float_spin(0.50, 0.995, 0.80,
                                        decimals=2, step=0.05)
        self._target_power.valueChanged.connect(lambda _v: self._on_values())
        card.add(Field("p.target_power", self._target_power))

        self._alpha = float_spin(0.001, 0.200, 0.05, decimals=3, step=0.005)
        self._alpha.valueChanged.connect(lambda _v: self._on_values())
        card.add(Field("p.alpha", self._alpha))

        self._method = _KeyedCombo(METHOD_KEYS)
        self._method.value_changed.connect(self._on_method)
        card.add(Field("p.method", self._method, ltr_editor=False))

        self._rmsea_null = float_spin(0.0, 0.30, 0.05, decimals=3, step=0.005)
        self._rmsea_null.valueChanged.connect(lambda _v: self._on_values())
        card.add(Field("p.rmsea_null", self._rmsea_null))

        self._rmsea_alt = float_spin(0.0, 0.30, 0.08, decimals=3, step=0.005)
        self._rmsea_alt.valueChanged.connect(lambda _v: self._on_values())
        card.add(Field("p.rmsea_alt", self._rmsea_alt,
                       hint_key="p.rmsea.hint"))

        self._mc_reps = int_spin(20, 5_000, 200, step=10)
        self._mc_reps.valueChanged.connect(lambda _v: self._on_values())
        self._mc_field = Field("p.mc_reps", self._mc_reps,
                               hint_key="p.mc_reps.hint")
        card.add(self._mc_field)

    def _update_method_visibility(self, method: Any) -> None:
        self._mc_field.setVisible(_as_method(method) is PowerMethod.MONTE_CARLO)

    # -- handlers -------------------------------------------------------- #

    def _on_mode(self, _checked: bool = False) -> None:
        if self._updating:
            return
        fixed = self._mode_fixed.isChecked()
        self.spec.power.mode = "fixed" if fixed else "suggest"
        self._n.setEnabled(fixed)
        self.edit()

    def _on_n(self, value: int) -> None:
        if self._updating:
            return
        self.spec.power.n = int(value)
        self.edit()

    def _on_method(self, method: Any) -> None:
        if self._updating:
            return
        if method is None:
            return
        resolved = _as_method(method)
        self.spec.power.method = resolved
        self._update_method_visibility(resolved)
        self.edit()

    def _on_values(self) -> None:
        if self._updating:
            return
        power = self.spec.power
        power.target_power = float(self._target_power.value())
        power.alpha = float(self._alpha.value())
        power.rmsea_null = float(self._rmsea_null.value())
        power.rmsea_alt = float(self._rmsea_alt.value())
        power.mc_replications = int(self._mc_reps.value())
        self.edit()

    # ------------------------------------------------------------------ #
    # Section 3 -- results
    # ------------------------------------------------------------------ #

    def _build_results(self) -> None:
        card = self.add_card("p.results")

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._compute = TButton("p.compute", role="primary")
        self._compute.clicked.connect(self._compute_now)
        buttons.addWidget(self._compute)
        self._apply_button = TButton("p.apply_n")
        self._apply_button.clicked.connect(self._apply_recommendation)
        buttons.addWidget(self._apply_button)
        buttons.addStretch(1)
        card.add(buttons)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)          # indeterminate: analyse() has no steps
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        theme.keep_ltr(self._progress)
        card.add(self._progress)

        self._status = TKey("p.not_computed", role="hint")
        card.add(self._status)
        self._error_label = QLabel()
        theme.set_role(self._error_label, "error")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        card.add(self._error_label)

        badges = QHBoxLayout()
        badges.setSpacing(10)
        self._badge_n = StatBadge("p.recommended")
        theme.set_role(self._badge_n, "card")
        # The headline number of the whole stage; the caption stays small.
        self._badge_n.setStyleSheet('QLabel[role="mono"] { font-size: 24pt; }')
        self._badge_n.setMinimumWidth(190)
        badges.addWidget(self._badge_n)
        self._badge_power = StatBadge("p.achieved_power")
        theme.set_role(self._badge_power, "card")
        self._badge_power.setMinimumWidth(160)
        badges.addWidget(self._badge_power)
        badges.addStretch(1)
        card.add(badges)

        self._rules = TablePanel("p.rules_of_thumb", max_rows=8)
        self.add(self._rules)
        self._paths = TablePanel("p.per_path", max_rows=10)
        self._paths.setVisible(False)
        self.add(self._paths)

        self._curve_card = Card("p.curve")
        self._curve = _PowerCurveView()
        self._curve_card.add(self._curve)
        self._curve_card.setVisible(False)
        self.add(self._curve_card)

        self._notes_card = Card("p.notes")
        self._notes = QLabel()
        self._notes.setWordWrap(True)
        theme.set_role(self._notes, "hint")
        self._notes_card.add(self._notes)
        self._notes_card.setVisible(False)
        self.add(self._notes_card)

    # -- running --------------------------------------------------------- #

    def _compute_now(self) -> None:
        if self._thread is not None:
            return
        self._error = ("", "")
        self._stale = False
        self._error_label.setVisible(False)
        self._compute.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setVisible(True)
        self._status.set_key("p.computing")
        self._thread = QThread(self)
        self._worker = _PowerWorker(copy.deepcopy(self.spec))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_computed)
        self._worker.unavailable.connect(self._on_unavailable)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _teardown_thread(self) -> None:
        self._progress.setVisible(False)
        self._compute.setEnabled(True)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None

    def _on_computed(self, result: Any) -> None:
        self._teardown_thread()
        self._result = result
        self._stale = False
        self._status.setVisible(False)
        self._apply_results()

    def _on_unavailable(self) -> None:
        self._teardown_thread()
        self._result = None
        self._stale = False
        self._status.setVisible(True)
        self._status.set_key("p.unavailable")
        self._apply_results()

    def _on_failed(self, message_ar: str, message_en: str) -> None:
        self._teardown_thread()
        self._result = None
        self._stale = False
        self._status.setVisible(True)
        self._status.set_key("p.failed")
        self._error = (message_ar, message_en)
        self._refresh_error_text()
        self._apply_results()

    def _refresh_error_text(self) -> None:
        message_ar, message_en = self._error
        text = message_ar if translator.is_rtl else message_en
        self._error_label.setText(text)
        self._error_label.setVisible(bool(text))

    def _apply_recommendation(self) -> None:
        recommended = self._recommended_n()
        if recommended <= 0:
            return
        self.spec.power.n = recommended
        self.spec.power.mode = "fixed"
        with self.quiet():
            self._mode_fixed.setChecked(True)
            self._n.setEnabled(True)
            self._n.setValue(recommended)
        self.edit()

    # -- display --------------------------------------------------------- #

    def _rules_of_thumb(self) -> Dict[str, int]:
        """The result's heuristics, or the ones this page can work out alone."""
        if self._result is not None:
            table = getattr(self._result, "rule_of_thumb", None)
            if isinstance(table, dict) and table:
                return {str(k): int(v) for k, v in table.items()}
        _moments, free, _df = self._complexity_numbers()
        items = int(getattr(self.spec, "n_items", 0) or 0)
        ratio_param = int(getattr(self.spec.power, "ratio_per_parameter", 10) or 10)
        ratio_item = int(getattr(self.spec.power, "ratio_per_item", 10) or 10)
        return {
            "n_per_parameter": ratio_param * free,
            "n_per_item": ratio_item * items,
            "minimum_200": KLINE_MINIMUM_N,
        }

    def _recommended_n(self) -> int:
        if self._result is not None:
            return int(getattr(self._result, "n_recommended", 0) or 0)
        values = [v for v in self._rules_of_thumb().values() if v > 0]
        return max(values) if values else 0

    def _rules_frame(self) -> pd.DataFrame:
        """Criterion / N, with the criterion column headed in the catalogue's words."""
        table = self._rules_of_thumb()
        criterion = translator.t("p.criterion")
        rows = [(translator.t(key), int(table[name]))
                for name, key in RULE_KEYS if name in table]
        for name, value in table.items():
            if name not in {n for n, _k in RULE_KEYS}:
                rows.append((str(name), int(value)))
        if not rows:
            return pd.DataFrame({criterion: [], "n": []})
        return pd.DataFrame({criterion: [r[0] for r in rows],
                             "n": [r[1] for r in rows]})

    def _per_path_frame(self) -> Optional[pd.DataFrame]:
        frame = getattr(self._result, "per_path_power", None)
        if frame is None or not len(frame):
            return None
        columns = list(getattr(frame, "columns", ()))

        def name_of(identifier: Any) -> str:
            construct = self.spec.construct(str(identifier))
            return construct.name if construct is not None else str(identifier)

        out: Dict[str, List[Any]] = {}
        if "source" in columns:
            out["source"] = [name_of(v) for v in frame["source"]]
        if "target" in columns:
            out["target"] = [name_of(v) for v in frame["target"]]
        if "coefficient" in columns:
            out["coefficient"] = list(frame["coefficient"])
        if "n_required" in columns:
            # "n" is a known integer column, so it renders without decimals.
            out["n"] = [None if pd.isna(v) else int(v)
                        for v in frame["n_required"]]
        if "power" in columns:
            out["power"] = list(frame["power"])
        return pd.DataFrame(out) if out else None

    def _notes_text(self) -> str:
        key = "notes_ar" if translator.is_rtl else "notes_en"
        notes = getattr(self._result, key, None) or []
        lines = [str(note).strip() for note in notes if str(note).strip()]
        return "\n".join(f"•  {line}" for line in lines)

    def _apply_results(self) -> None:
        result = self._result
        recommended = self._recommended_n()
        target = float(getattr(self.spec.power, "target_power", 0.80) or 0.80)
        if result is not None:
            target = float(getattr(result, "target_power", target) or target)
        self._badge_n.set_value(recommended if recommended > 0 else "—")
        self._apply_button.setEnabled(recommended > 0)

        achieved = float(getattr(result, "power_at_recommended", float("nan"))
                         if result is not None else float("nan"))
        if math.isfinite(achieved):
            self._badge_power.set_value(
                f"{achieved:.3f}",
                kind="pass" if achieved >= target - 1e-9 else "borderline")
        else:
            self._badge_power.set_value("—")

        self._rules.set_frame(self._rules_frame())
        _fit_table(self._rules, 8)

        paths = self._per_path_frame() if result is not None else None
        if paths is not None:
            self._paths.set_frame(paths)
            _fit_table(self._paths, 10)
        self._paths.setVisible(paths is not None)

        curve = getattr(result, "curve", None) if result is not None else None
        drawn = self._curve.set_curve(curve, recommended, target)
        self._curve_card.setVisible(drawn)

        notes = self._notes_text() if result is not None else ""
        self._notes.setText(notes)
        self._notes_card.setVisible(bool(notes))

    # ------------------------------------------------------------------ #
    # StagePage contract
    # ------------------------------------------------------------------ #

    def rebuild(self) -> None:
        """The model was replaced; an analysis of the old one means nothing."""
        self._result = None
        self._stale = False
        self._error = ("", "")
        self._error_label.setVisible(False)
        self._status.set_key("p.not_computed")
        self._status.setVisible(True)
        self.refresh()

    def refresh(self) -> None:
        power = self.spec.power
        with self.quiet():
            fixed = str(getattr(power, "mode", "suggest")) == "fixed"
            self._mode_fixed.setChecked(fixed)
            self._mode_suggest.setChecked(not fixed)
            self._n.setValue(int(power.n))
            self._n.setEnabled(fixed)
            self._target_power.setValue(float(power.target_power))
            self._alpha.setValue(float(power.alpha))
            self._method.set_value(power.method)
            self._rmsea_null.setValue(float(power.rmsea_null))
            self._rmsea_alt.setValue(float(power.rmsea_alt))
            self._mc_reps.setValue(int(power.mc_replications))
            self._update_method_visibility(power.method)
        self._refresh_complexity()
        self._refresh_error_text()
        self._apply_results()

    def set_theme(self, name: str) -> None:
        super().set_theme(name)
        self._curve.set_theme(name)
        self._rules.set_theme(name)
        self._paths.set_theme(name)

    def _on_results_changed(self) -> None:
        """A full run carries a power result of its own; show it."""
        result = getattr(self.state.results, "power", None)
        if result is not None and result is not self._result:
            self._result = result
            self._stale = False
            self._error = ("", "")
            self._error_label.setVisible(False)
            self._status.setVisible(False)
            self._apply_results()


__all__ = ["PowerPage"]

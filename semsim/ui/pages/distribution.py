"""
semsim.ui.pages.distribution
============================
Stage 3 -- the data distribution and generation engine.

Four decisions live here, in the order a researcher makes them:

*   **Global non-normality** -- the default skewness and kurtosis every item
    inherits, and whether the Vale-Maurelli machinery runs at all.  The pair is
    checked live, because Fleishman's polynomial has no real solution below
    ``kurtosis = skew**2 - 2`` and discovering that only when the run fails is
    no help to anybody.
*   **The threshold preview** -- a response style and a category count, with
    the resulting category proportions drawn as bars.  The bars are painted by
    hand in :class:`_ProportionChart`: matplotlib cannot repaint fast enough to
    follow a spin box, and this chart has to follow one.
*   **Real-world variation** -- how far constructs and items differ from one
    another, and how far the population loadings and paths scatter around the
    planned values, so the data do not carry the uniformity of a textbook model.
*   **Noise injection** -- the deliberate imperfections (careless respondents,
    outliers, missingness, jitter) that keep a simulated matrix from fitting
    too perfectly.
*   **Reproducibility** -- the seed, and which generator will actually run.
    ``engine`` is a request; :meth:`ModelSpec.resolved_engine` is the answer,
    and the page says so out loud when the two differ.

Direction discipline: every number, the proportions line and the bar chart are
pinned left-to-right, so category 1 is the leftmost bar in Arabic as well.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from semsim.core.spec import (
    GenerationEngine, Item, MissingMechanism, ResponseStyle,
)
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.pages.base import IdCombo, StagePage, construct_choices
from semsim.ui.widgets.common import (
    EnumCombo, Field, InfoBanner, Notice, TButton, TCheckBox, TKey,
    current_theme, float_spin, int_spin, percent_spin, percent_value,
)

#: ``NoiseSpec.careless_style`` is a bare string, not an enum; these are its
#: three legal values, each with a catalogue entry of its own.
CARELESS_STYLES: Tuple[str, str, str] = ("random", "straightline", "mixed")

#: The variation presets offered, plus ``custom`` for hand-set amounts.
VARIATION_LEVELS: Tuple[str, ...] = ("none", "mild", "realistic", "strong",
                                     "custom")

#: numpy's default_rng accepts anything up to 2**32-1; a spin box is int-sized.
MAX_SEED: int = 2_147_483_647


def enum_value(combo: EnumCombo, enum_cls, default):
    """The combo's selection as a real enum member.

    Qt stores the item data as a QVariant, and every enum in the spec derives
    from :class:`str`, so what comes back out of a combo is the bare value
    ``"scores"`` rather than ``GenerationEngine.SCORES``.  That reads the same
    in a comparison with ``==`` and differently with ``is`` -- and
    :meth:`ModelSpec.resolved_engine` tests with ``is`` -- so the member is
    rebuilt here before it ever reaches the spec.
    """
    try:
        return enum_cls(combo.value())
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# The bar chart
# --------------------------------------------------------------------------- #


class _ProportionChart(QWidget):
    """Category proportions as bars, painted directly.

    The chart is redrawn on every keystroke of the category spin box, so it
    draws itself with :class:`QPainter` instead of going through a plotting
    library.  It is pinned left-to-right: the lowest category is the leftmost
    bar whatever the interface direction, because a Likert scale drawn
    backwards reads as a different distribution.
    """

    #: Room above the bars for the percentage labels, and below for the codes.
    TOP_PAD = 20
    BOTTOM_PAD = 20
    SIDE_PAD = 12
    GAP = 6

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._proportions: List[float] = []
        self._theme = current_theme()
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setFixedHeight(170)
        theme.keep_ltr(self)

    def set_proportions(self, proportions: Sequence[float]) -> None:
        self._proportions = [max(0.0, float(p)) for p in proportions]
        self.update()

    def set_theme(self, name: str) -> None:
        self._theme = name
        self.update()

    # -- painting --------------------------------------------------------- #

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pal = theme.palette(self._theme)

        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(pal.border), 1.0))
        painter.setBrush(QColor(pal.surface_alt))
        painter.drawRoundedRect(frame, 6.0, 6.0)

        proportions = self._proportions
        if not proportions:
            return

        font = painter.font()
        font.setPointSize(max(7, font.pointSize() - 1))
        painter.setFont(font)

        count = len(proportions)
        plot_left = frame.left() + self.SIDE_PAD
        plot_width = frame.width() - 2 * self.SIDE_PAD
        baseline = frame.bottom() - self.BOTTOM_PAD
        plot_height = baseline - frame.top() - self.TOP_PAD
        if plot_width <= 0 or plot_height <= 0:
            return

        gap = self.GAP if count > 1 else 0
        bar_width = max(2.0, (plot_width - gap * (count - 1)) / count)
        peak = max(proportions) or 1.0

        painter.setPen(QPen(QColor(pal.border), 1.0))
        painter.drawLine(int(plot_left), int(baseline),
                         int(plot_left + plot_width), int(baseline))

        for index, proportion in enumerate(proportions):
            left = plot_left + index * (bar_width + gap)
            height = plot_height * (proportion / peak)
            bar = QRectF(left, baseline - height, bar_width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(pal.accent))
            painter.drawRoundedRect(bar, 3.0, 3.0)

            label_box = QRectF(left - gap, baseline - height - self.TOP_PAD,
                               bar_width + 2 * gap, self.TOP_PAD)
            painter.setPen(QPen(QColor(pal.text)))
            painter.drawText(label_box,
                             int(Qt.AlignmentFlag.AlignHCenter
                                 | Qt.AlignmentFlag.AlignBottom),
                             f"{proportion * 100:.0f}%")

            code_box = QRectF(left - gap, baseline + 2,
                              bar_width + 2 * gap, self.BOTTOM_PAD - 2)
            painter.setPen(QPen(QColor(pal.text_muted)))
            painter.drawText(code_box,
                             int(Qt.AlignmentFlag.AlignHCenter
                                 | Qt.AlignmentFlag.AlignTop),
                             str(index + 1))


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


class DistributionPage(StagePage):
    """Stage 3 -- data distribution and generation engine."""

    TITLE_KEY = "stage.3.title"
    SUBTITLE_KEY = "stage.3.subtitle"

    def build(self) -> None:
        #: True once the researcher has touched the preview controls, after
        #: which a refresh stops dragging them back to the first item's scale.
        self._preview_touched = False
        self._noise_editors: List[QWidget] = []

        self._build_global()
        self._build_preview()
        self._build_variation()
        self._build_respondent_styles()
        self._build_noise()
        self._build_reproducibility()
        self.scroll.add_stretch()
        self.rebuild()

    # ------------------------------------------------------------------ #
    # 1. Global distribution
    # ------------------------------------------------------------------ #

    def _build_global(self) -> None:
        card = self.add_card("d.nonnormality")
        card.add(InfoBanner("d.nonnormality.explain"))

        self.skew = float_spin(-3.0, 3.0, 0.0, decimals=2, step=0.1)
        self.skew.valueChanged.connect(lambda _v: self._on_distribution())
        self._skew_field = Field("d.default_skew", self.skew)
        card.add(self._skew_field)

        self.kurt = float_spin(-2.0, 25.0, 0.0, decimals=2, step=0.1)
        self.kurt.valueChanged.connect(lambda _v: self._on_distribution())
        self._kurt_field = Field("d.default_kurt", self.kurt,
                                 hint_key="d.kurtosis.hint")
        card.add(self._kurt_field)

        self.use_nonnormal = TCheckBox("d.use_nonnormal")
        self.use_nonnormal.toggled.connect(lambda _c: self._on_distribution())
        card.add(self.use_nonnormal)

        self.clamp = TCheckBox("d.clamp_feasible")
        self.clamp.toggled.connect(lambda _c: self._on_distribution())
        card.add(self.clamp)

    def _on_distribution(self) -> None:
        if self._updating:
            return
        dist = self.spec.distribution
        dist.default_skewness = float(self.skew.value())
        dist.default_kurtosis = float(self.kurt.value())
        dist.use_non_normal_engine = bool(self.use_nonnormal.isChecked())
        dist.clamp_to_feasible_region = bool(self.clamp.isChecked())
        self._check_feasible()
        self.edit()

    def _check_feasible(self) -> None:
        """Fleishman has no real solution when ``kurtosis <= skew**2 - 2``."""
        skew = float(self.skew.value())
        kurt = float(self.kurt.value())
        if kurt <= skew * skew - 2.0:
            self._kurt_field.set_error(translator.t("d.infeasible"))
        else:
            self._kurt_field.set_error("")

    # ------------------------------------------------------------------ #
    # 2. Likert / threshold preview
    # ------------------------------------------------------------------ #

    def _build_preview(self) -> None:
        card = self.add_card("d.threshold_preview")
        card.add(InfoBanner("d.thresholds.explain"))

        self.style = EnumCombo(ResponseStyle, "style")
        self.style.value_changed.connect(lambda _v: self._on_preview_changed())
        card.add(Field("d.response_style", self.style))

        self.n_cats = int_spin(2, 11, 5)
        self.n_cats.valueChanged.connect(lambda _v: self._on_preview_changed())
        card.add(Field("d.n_categories", self.n_cats))

        self.chart = _ProportionChart()
        card.add(self.chart)

        self.proportions = QLabel()
        theme.set_role(self.proportions, "mono")
        self.proportions.setWordWrap(True)
        card.add(Field("d.proportions", self.proportions))

        self.preview_hint = TKey("d.thresholds.unavailable", role="hint")
        self.preview_hint.setVisible(False)
        card.add(self.preview_hint)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_apply_all = TButton("d.apply_all_items", role="primary")
        self.btn_apply_all.clicked.connect(self._apply_to_all_items)
        row.addWidget(self.btn_apply_all)
        self.apply_notice = Notice()
        row.addWidget(self.apply_notice)
        row.addStretch(1)
        card.add(row)

        target = QWidget()
        target_row = QHBoxLayout(target)
        target_row.setContentsMargins(0, 0, 0, 0)
        target_row.setSpacing(8)
        self.apply_target = IdCombo()
        target_row.addWidget(self.apply_target, 1)
        self.btn_apply_construct = TButton("d.apply_to_construct")
        self.btn_apply_construct.clicked.connect(self._apply_to_construct)
        target_row.addWidget(self.btn_apply_construct)
        card.add(Field("m.construct", target, ltr_editor=False))

    def _on_preview_changed(self) -> None:
        if not self._updating:
            self._preview_touched = True
        self._update_preview()

    def _proportions_for(self, n_categories: int,
                        style: ResponseStyle) -> Tuple[List[float], str]:
        """``(proportions, hint_key)`` for the preview.

        The generation package may not be present yet, and
        :attr:`ResponseStyle.CUSTOM` legitimately carries no shape of its own,
        so both cases fall back on equal proportions and say which it was.
        """
        flat = [1.0 / n_categories] * n_categories
        try:
            from semsim.generation.thresholds import (
                category_probabilities, thresholds_for_style,
            )
        except Exception:
            return flat, "d.thresholds.unavailable"
        try:
            tau = thresholds_for_style(n_categories, style)
            return [float(p) for p in category_probabilities(tau)], ""
        except Exception:
            return flat, ("d.custom_style.hint"
                          if style is ResponseStyle.CUSTOM
                          else "d.thresholds.unavailable")

    def _update_preview(self) -> None:
        n_categories = int(self.n_cats.value())
        style = enum_value(self.style, ResponseStyle, ResponseStyle.SYMMETRIC)
        proportions, hint_key = self._proportions_for(n_categories, style)
        self.chart.set_proportions(proportions)
        # Digits only, pinned left-to-right by the Field: nothing to translate.
        self.proportions.setText("   ".join(
            f"{index}: {value:.3f}"
            for index, value in enumerate(proportions, start=1)))
        if hint_key:
            self.preview_hint.set_key(hint_key)
            self.preview_hint.setVisible(True)
        else:
            self.preview_hint.setVisible(False)

    def _apply_scale(self, items: Sequence[Item]) -> None:
        if not items:
            return
        style = enum_value(self.style, ResponseStyle, ResponseStyle.SYMMETRIC)
        n_categories = int(self.n_cats.value())
        for item in items:
            item.scale.response_style = style
            item.scale.n_categories = n_categories
        self.edit()
        self.apply_notice.show_key("d.applied", n=len(items))

    def _apply_to_all_items(self) -> None:
        self._apply_scale(self.spec.all_items())

    def _apply_to_construct(self) -> None:
        construct = self.spec.construct(self.apply_target.value())
        if construct is None:
            return
        self._apply_scale(construct.all_items())

    # ------------------------------------------------------------------ #
    # 3. Real-world variation
    # ------------------------------------------------------------------ #

    def _build_variation(self) -> None:
        card = self.add_card("d.variation")
        card.add(InfoBanner("d.variation.explain"))

        self.variation = EnumCombo(str, "d.variation", members=VARIATION_LEVELS)
        self.variation.value_changed.connect(self._on_variation_preset)
        card.add(Field("d.variation.level", self.variation))

        def amount(maximum: float, step: float) -> "object":
            box = float_spin(0.0, maximum, 0.0, decimals=2, step=step)
            box.valueChanged.connect(lambda _v: self._on_variation_amounts())
            return box

        self.construct_level_sd = amount(1.0, 0.05)
        card.add(Field("d.variation.construct_level", self.construct_level_sd,
                       hint_key="d.variation.construct_level.hint"))
        self.item_level_sd = amount(1.0, 0.05)
        card.add(Field("d.variation.item_level", self.item_level_sd,
                       hint_key="d.variation.item_level.hint"))
        self.item_spread_sd = amount(0.5, 0.01)
        card.add(Field("d.variation.item_spread", self.item_spread_sd,
                       hint_key="d.variation.item_spread.hint"))
        self.loading_sd = amount(0.2, 0.01)
        card.add(Field("d.variation.loading", self.loading_sd,
                       hint_key="d.variation.loading.hint"))
        self.path_sd = amount(0.15, 0.01)
        card.add(Field("d.variation.path", self.path_sd,
                       hint_key="d.variation.path.hint"))
        self.tail_share = percent_spin(0.0, maximum=0.2)
        self.tail_share.valueChanged.connect(
            lambda _v: self._on_variation_amounts())
        card.add(Field("d.variation.tail", self.tail_share,
                       hint_key="d.variation.tail.hint"))
        self._variation_editors = [
            self.construct_level_sd, self.item_level_sd, self.item_spread_sd,
            self.loading_sd, self.path_sd, self.tail_share,
        ]

    def _on_variation_preset(self, level: object) -> None:
        if self._updating:
            return
        settings = self.spec.heterogeneity
        level = str(level or "realistic")
        if level == "custom":
            # Keep the current amounts as the starting point for editing.
            settings.enabled = True
        else:
            settings.apply_preset(level)
        self._show_variation()
        self.edit()

    def _on_variation_amounts(self) -> None:
        if self._updating:
            return
        settings = self.spec.heterogeneity
        settings.enabled = True
        settings.construct_level_sd = float(self.construct_level_sd.value())
        settings.item_level_sd = float(self.item_level_sd.value())
        settings.item_spread_sd = float(self.item_spread_sd.value())
        settings.loading_sd = float(self.loading_sd.value())
        settings.path_sd = float(self.path_sd.value())
        settings.tail_share = percent_value(self.tail_share)
        with self.quiet():
            self.variation.set_value(settings.preset_name())
        self._update_variation_gating()
        self.edit()

    def _show_variation(self) -> None:
        settings = self.spec.heterogeneity
        with self.quiet():
            self.variation.set_value(settings.preset_name())
            self.construct_level_sd.setValue(settings.construct_level_sd)
            self.item_level_sd.setValue(settings.item_level_sd)
            self.item_spread_sd.setValue(settings.item_spread_sd)
            self.loading_sd.setValue(settings.loading_sd)
            self.path_sd.setValue(settings.path_sd)
            self.tail_share.setValue(settings.tail_share * 100.0)
        self._update_variation_gating()

    def _update_variation_gating(self) -> None:
        enabled = bool(self.spec.heterogeneity.enabled)
        for editor in self._variation_editors:
            editor.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Respondent answering styles
    # ------------------------------------------------------------------ #

    def _build_respondent_styles(self) -> None:
        card = self.add_card("d.styles")
        card.add(InfoBanner("d.styles.explain"))
        self.styles_level = EnumCombo(str, "d.variation", members=VARIATION_LEVELS)
        self.styles_level.value_changed.connect(self._on_styles_preset)
        card.add(Field("d.styles.level", self.styles_level))
        self.extreme_sd = float_spin(0.0, 0.6, 0.0, decimals=2, step=0.05)
        self.extreme_sd.valueChanged.connect(lambda _v: self._on_styles_amounts())
        card.add(Field("d.styles.extreme", self.extreme_sd,
                       hint_key="d.styles.extreme.hint"))
        self.acquiescence_sd = float_spin(0.0, 0.4, 0.0, decimals=2, step=0.01)
        self.acquiescence_sd.valueChanged.connect(
            lambda _v: self._on_styles_amounts())
        card.add(Field("d.styles.acquiescence", self.acquiescence_sd,
                       hint_key="d.styles.acquiescence.hint"))

    def _on_styles_preset(self, level: object) -> None:
        if self._updating:
            return
        settings = self.spec.respondent_styles
        level = str(level or "realistic")
        if level == "custom":
            settings.enabled = True
        else:
            settings.apply_preset(level)
        self._show_styles()
        self.edit()

    def _on_styles_amounts(self) -> None:
        if self._updating:
            return
        settings = self.spec.respondent_styles
        settings.enabled = True
        settings.extreme_sd = float(self.extreme_sd.value())
        settings.acquiescence_sd = float(self.acquiescence_sd.value())
        with self.quiet():
            self.styles_level.set_value(settings.preset_name())
        self._update_styles_gating()
        self.edit()

    def _show_styles(self) -> None:
        settings = self.spec.respondent_styles
        with self.quiet():
            self.styles_level.set_value(settings.preset_name())
            self.extreme_sd.setValue(settings.extreme_sd)
            self.acquiescence_sd.setValue(settings.acquiescence_sd)
        self._update_styles_gating()

    def _update_styles_gating(self) -> None:
        enabled = bool(self.spec.respondent_styles.enabled)
        self.extreme_sd.setEnabled(enabled)
        self.acquiescence_sd.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # 4. Noise injection
    # ------------------------------------------------------------------ #

    def _build_noise(self) -> None:
        card = self.add_card("d.noise")
        card.add(InfoBanner("d.noise.explain"))

        self.noise_enabled = TCheckBox("d.noise.enable")
        self.noise_enabled.toggled.connect(lambda _c: self._on_noise())
        card.add(self.noise_enabled)

        self.inflation = float_spin(1.0, 2.0, 1.0, decimals=2, step=0.05)
        self.inflation.valueChanged.connect(lambda _v: self._on_noise())
        card.add(Field("d.unique_inflation", self.inflation,
                       hint_key="d.unique_inflation.hint"))

        card.add(InfoBanner("d.careless.explain"))
        self.careless = percent_spin(0.0)
        self.careless.valueChanged.connect(lambda _v: self._on_noise())
        card.add(Field("d.careless", self.careless))

        self.careless_style = EnumCombo(str, "d.careless",
                                        members=CARELESS_STYLES)
        self.careless_style.value_changed.connect(lambda _v: self._on_noise())
        card.add(Field("d.careless_style", self.careless_style))

        self.outliers = percent_spin(0.0)
        self.outliers.valueChanged.connect(lambda _v: self._on_noise())
        card.add(Field("d.outliers", self.outliers))

        self.outlier_sd = float_spin(1.0, 10.0, 3.5, decimals=2, step=0.1)
        self.outlier_sd.valueChanged.connect(lambda _v: self._on_noise())
        card.add(Field("d.outlier_magnitude", self.outlier_sd))

        self.missing_mechanism = EnumCombo(MissingMechanism, "d.missing")
        self.missing_mechanism.value_changed.connect(lambda _v: self._on_noise())
        card.add(Field("d.missing_mechanism", self.missing_mechanism))

        self.missing_proportion = percent_spin(0.0)
        self.missing_proportion.valueChanged.connect(lambda _v: self._on_noise())
        card.add(Field("d.missing_proportion", self.missing_proportion))

        self.mar_driver = IdCombo()
        self.mar_driver.value_changed.connect(lambda _v: self._on_noise())
        card.add(Field("d.mar_driver", self.mar_driver, ltr_editor=False))

        self.jitter = float_spin(0.0, 2.0, 0.0, decimals=3, step=0.01)
        self.jitter.valueChanged.connect(lambda _v: self._on_noise())
        card.add(Field("d.jitter", self.jitter))

        self._noise_editors = [
            self.inflation, self.careless, self.careless_style, self.outliers,
            self.outlier_sd, self.missing_mechanism, self.missing_proportion,
            self.mar_driver, self.jitter,
        ]

    def _on_noise(self) -> None:
        if self._updating:
            return
        noise = self.spec.noise
        noise.enabled = bool(self.noise_enabled.isChecked())
        noise.unique_variance_inflation = float(self.inflation.value())
        noise.careless_proportion = percent_value(self.careless)
        noise.careless_style = str(self.careless_style.value() or "random")
        noise.outlier_proportion = percent_value(self.outliers)
        noise.outlier_magnitude_sd = float(self.outlier_sd.value())
        noise.missing_mechanism = enum_value(
            self.missing_mechanism, MissingMechanism, MissingMechanism.NONE)
        noise.missing_proportion = percent_value(self.missing_proportion)
        noise.mar_driver = self.mar_driver.value()
        noise.measurement_jitter_sd = float(self.jitter.value())
        self._update_noise_gating()
        self.edit()

    def _update_noise_gating(self) -> None:
        """Grey out what the current switches make irrelevant."""
        enabled = bool(self.noise_enabled.isChecked())
        for editor in self._noise_editors:
            editor.setEnabled(enabled)
        # Only MAR needs a driver: MCAR is indifferent and NONE drops nothing.
        mechanism = enum_value(self.missing_mechanism, MissingMechanism,
                               MissingMechanism.NONE)
        self.mar_driver.setEnabled(enabled
                                   and mechanism is MissingMechanism.MAR)

    # ------------------------------------------------------------------ #
    # 5. Reproducibility
    # ------------------------------------------------------------------ #

    def _build_reproducibility(self) -> None:
        card = self.add_card("d.reproducibility")

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(8)
        self.seed = int_spin(0, MAX_SEED, 20240101)
        self.seed.valueChanged.connect(lambda _v: self._on_seed())
        seed_layout.addWidget(self.seed, 1)
        self.btn_new_seed = TButton("d.new_seed")
        self.btn_new_seed.clicked.connect(self._new_seed)
        seed_layout.addWidget(self.btn_new_seed)
        theme.keep_ltr(seed_row)
        card.add(Field("d.seed", seed_row, hint_key="d.seed.hint",
                       ltr_editor=False))

        self.engine = EnumCombo(GenerationEngine, "d.engine")
        self.engine.value_changed.connect(lambda _v: self._on_engine())
        card.add(Field("d.engine", self.engine, hint_key="d.engine.hint"))

        # Deliberately not the ``mono`` role: that font has no Arabic
        # shaping, and the engine names are prose in both languages.
        self.resolved = QLabel()
        self.resolved.setWordWrap(True)
        resolved_font = self.resolved.font()
        resolved_font.setBold(True)
        self.resolved.setFont(resolved_font)
        card.add(Field("d.engine.resolved", self.resolved, ltr_editor=False))

        self.note_forced = InfoBanner("d.engine.forced_scores", kind="warning")
        self.note_forced.setVisible(False)
        card.add(self.note_forced)
        self.note_auto = InfoBanner("d.engine.auto_resolved")
        self.note_auto.setVisible(False)
        card.add(self.note_auto)

    def _on_seed(self) -> None:
        if self._updating:
            return
        self.spec.seed = int(self.seed.value())
        self.edit()

    def _new_seed(self) -> None:
        import secrets
        value = secrets.randbelow(MAX_SEED) + 1
        with self.quiet():
            self.seed.setValue(value)
        self.spec.seed = value
        self.edit()

    def _on_engine(self) -> None:
        if self._updating:
            return
        self.spec.engine = enum_value(self.engine, GenerationEngine,
                                      GenerationEngine.AUTO)
        self._update_engine_note()
        self.edit()

    def _update_engine_note(self) -> None:
        try:
            resolved = self.spec.resolved_engine()
        except Exception:
            resolved = self.spec.engine
        self.resolved.setText(translator.enum("d.engine", resolved))
        chosen = enum_value(self.engine, GenerationEngine, GenerationEngine.AUTO)
        # A latent product cannot live in a covariance matrix, so moderation
        # is the one thing that overrides the researcher's choice.
        if self.spec.has_interactions():
            self.note_forced.set_kind(
                "info" if resolved is GenerationEngine.SCORES else "warning")
            self.note_forced.setVisible(True)
            self.note_auto.setVisible(False)
        elif chosen is not resolved:
            self.note_forced.setVisible(False)
            self.note_auto.setVisible(True)
        else:
            self.note_forced.setVisible(False)
            self.note_auto.setVisible(False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def rebuild(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        spec = self.spec
        dist, noise = spec.distribution, spec.noise
        choices = construct_choices(spec)
        with self.quiet():
            self.skew.setValue(dist.default_skewness)
            self.kurt.setValue(dist.default_kurtosis)
            self.use_nonnormal.setChecked(dist.use_non_normal_engine)
            self.clamp.setChecked(dist.clamp_to_feasible_region)

            self.noise_enabled.setChecked(noise.enabled)
            self.inflation.setValue(noise.unique_variance_inflation)
            self.careless.setValue(noise.careless_proportion * 100.0)
            self.careless_style.set_value(noise.careless_style)
            self.outliers.setValue(noise.outlier_proportion * 100.0)
            self.outlier_sd.setValue(noise.outlier_magnitude_sd)
            self.missing_mechanism.set_value(noise.missing_mechanism)
            self.missing_proportion.setValue(noise.missing_proportion * 100.0)
            self.jitter.setValue(noise.measurement_jitter_sd)

            # The construct labels are language-dependent, so the choosers are
            # repopulated here rather than only on a structural change.
            self.apply_target.set_choices(choices, stored=False)
            has_items = bool(spec.all_items())
            needs_items = "" if has_items else translator.t("x.needs_items")
            for button in (self.btn_apply_all, self.btn_apply_construct):
                button.setEnabled(has_items)
                button.setToolTip(needs_items)
            self.mar_driver.set_choices(choices, keep=noise.mar_driver)

            self.seed.setValue(int(spec.seed or 0))
            self.engine.set_value(spec.engine)

            if not self._preview_touched:
                items = spec.all_items()
                if items:
                    scale = items[0].scale
                    self.style.set_value(scale.response_style)
                    self.n_cats.setValue(scale.n_categories)

        self._check_feasible()
        self._update_preview()
        self._show_variation()
        self._show_styles()
        self._update_noise_gating()
        self._update_engine_note()

    def set_theme(self, name: str) -> None:
        super().set_theme(name)
        self.chart.set_theme(name)


__all__ = ["DistributionPage", "enum_value"]

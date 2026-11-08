"""
semsim.ui.pages.targeting
=========================
Stage 5 -- fit indices and significance targeting.

This is where the researcher says what the *output* should look like, and the
engine works backwards to a covariance matrix that produces it:

*   **Model fit target.**  A fit level (Hu & Bentler, 1999) carries a
    conventional RMSEA and an expected CFI floor, so picking "good" writes
    ``0.045`` into the spec and says out loud what else that implies.  Editing
    the RMSEA by hand makes the number the researcher's own, so the level
    combo flips to CUSTOM rather than going on claiming a conventional band --
    the same contract the effect-size combo and the coefficient slider keep in
    :mod:`semsim.ui.pages.structural`.
    A model with no degrees of freedom has no fit to target at all, and the
    page says so instead of offering controls that cannot work.
*   **Significance targeting.**  Alpha, the power each targeted path should
    reach, the coefficient given to paths that must stay null, and how hard
    the rejection sampler may try.
*   **Per-path targets.**  Every path and interaction on one screen with its
    effect size, its coefficient and the outcome asked of it.  This is the
    scenario sheet: the researcher reads down it to see which hypotheses the
    generated dataset will support.
*   **Calibration result.**  What the last run actually achieved, shown only
    once a run has produced it.

Direction discipline: the prose follows the interface, while construct names,
coefficients and every table stay left-to-right -- a calibration log read
right-to-left is the wrong log.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from semsim.core.spec import FitLevel, Interaction, Path
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.pages.base import Row, RowList, StagePage
from semsim.ui.widgets.common import (
    Card, EnumCombo, Field, InfoBanner, StatBadge, TCheckBox, TKey,
    VerdictChip, float_spin, int_spin,
)
from semsim.ui.widgets.tables import TablePanel

#: ``FitTargetSpec.method`` is a bare string, not an enum.  These are its legal
#: values paired with the catalogue suffix each one is labelled by, so the
#: combo stays translated without the spec having to grow an enum.
METHOD_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("cudeck", "cudeck_browne"),
    ("residual", "residual"),
    ("none", "none"),
)
METHOD_SUFFIXES: Tuple[str, ...] = tuple(suffix for suffix, _v in METHOD_CHOICES)
METHOD_TO_SPEC: Dict[str, str] = dict(METHOD_CHOICES)
SPEC_TO_METHOD: Dict[str, str] = {v: k for k, v in METHOD_CHOICES}

#: Columns a calibration history is expected to carry.  A table is never left
#: holding a frame with no columns at all: :class:`DataFrameModel` announces a
#: header change for section 0 when the language switches, and on an empty
#: frame there is no section 0 for the sort proxy to map.
EMPTY_HISTORY_COLUMNS: Tuple[str, ...] = ("round", "rmsea", "cfi", "srmr")


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EMPTY_HISTORY_COLUMNS))


#: The requested outcome of one path, labelled through ``verdict.*``.
OUTCOME_SIGNIFICANT = "significant"
OUTCOME_NULL = "not_significant"
OUTCOME_CHOICES: Tuple[str, str] = (OUTCOME_SIGNIFICANT, OUTCOME_NULL)

#: Conventional cut-offs (Hu & Bentler, 1999; Kline, 2016).  The symbols are
#: mathematical notation, not prose, so they are not translated.
FIT_THRESHOLDS: Tuple[Tuple[str, str], ...] = (
    ("r.rmsea", "≤ 0.06"),
    ("r.cfi", "≥ 0.95"),
    ("r.tli", "≥ 0.95"),
    ("r.srmr", "≤ 0.08"),
    ("r.chi2_df", "≤ 3"),
)


def enum_value(combo: EnumCombo, enum_cls: Any, default: Any) -> Any:
    """The combo's selection as a real enum member.

    Qt stores item data as a QVariant and every enum in the spec derives from
    :class:`str`, so what comes back out of a combo is the bare value
    ``"good"`` rather than ``FitLevel.GOOD``.  That compares equal with ``==``
    and differently with ``is``, and the properties on :class:`FitLevel` are
    reached through the member, so the member is rebuilt here before it ever
    touches the spec.
    """
    try:
        return enum_cls(combo.value())
    except (ValueError, TypeError):
        return default


def _size_to_content(panel: TablePanel, max_rows: int) -> None:
    """Keep a short table from claiming a scroll area's default height.

    :class:`TablePanel` sets a minimum; the view's own size hint is a good
    deal taller than five rows, which leaves a band of empty table under the
    last row.  Capping the height at the same ideal removes it.
    """
    table = panel.table
    table.setMaximumHeight(table.ideal_height(max_rows))


def _mono(text: str, *, role: str = "mono") -> QLabel:
    """A label for an identifier or a number: pinned left-to-right.

    Pinning the label alone would also pin its *alignment*, so a construct
    name would sit at the left of a cell whose Arabic column caption sits at
    the right.  The content stays left-to-right -- ``+0.50`` must not come out
    as ``0.50+`` -- while the alignment follows the interface, so every cell
    lines up under its own heading in both languages.
    """
    label = QLabel(text)
    theme.set_role(label, role)
    theme.keep_ltr(label)
    label.setAlignment(
        (Qt.AlignmentFlag.AlignRight if translator.is_rtl
         else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
    return label


class TargetingPage(StagePage):
    TITLE_KEY = "stage.5.title"
    SUBTITLE_KEY = "stage.5.subtitle"

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def build(self) -> None:
        self._build_fit_card()
        self._build_thresholds_panel()
        self._build_significance_card()
        self._build_paths_card()
        self._build_calibration_section()
        self.scroll.add_stretch()
        self.rebuild()

    # -- 1. model fit target --------------------------------------------- #

    def _build_fit_card(self) -> None:
        card = self.add_card("t.fit_target")
        card.add(InfoBanner("t.fit_target.explain"))

        self._saturated_banner = InfoBanner("t.saturated", kind="warning")
        self._saturated_banner.setVisible(False)
        card.add(self._saturated_banner)

        self.enable_fit = TCheckBox("t.enable_fit")
        self.enable_fit.toggled.connect(self._on_enable_fit)
        card.add(self.enable_fit)

        self.level = EnumCombo(FitLevel, "fit")
        self.level.value_changed.connect(lambda _v: self._on_level())
        card.add(Field("t.level", self.level))
        self.level_hint = TKey("t.level.implies", role="hint",
                               rmsea="-", cfi="-")
        card.add(self.level_hint)

        self.rmsea = float_spin(0.0, 0.30, 0.045, decimals=3, step=0.005)
        self.rmsea.valueChanged.connect(lambda _v: self._on_rmsea())
        card.add(Field("t.target_rmsea", self.rmsea))

        self.use_cfi = TCheckBox("t.use_cfi")
        self.use_cfi.toggled.connect(lambda _c: self._on_secondary())
        card.add(self.use_cfi)
        self.cfi = float_spin(0.50, 1.0, 0.95, decimals=3, step=0.005)
        self.cfi.valueChanged.connect(lambda _v: self._on_secondary())
        self._cfi_field = Field("t.target_cfi", self.cfi)
        card.add(self._cfi_field)

        self.use_srmr = TCheckBox("t.use_srmr")
        self.use_srmr.toggled.connect(lambda _c: self._on_secondary())
        card.add(self.use_srmr)
        self.srmr = float_spin(0.0, 0.30, 0.08, decimals=3, step=0.005)
        self.srmr.valueChanged.connect(lambda _v: self._on_secondary())
        self._srmr_field = Field("t.target_srmr", self.srmr)
        card.add(self._srmr_field)

        self.method = EnumCombo(str, "t.method", members=METHOD_SUFFIXES)
        self.method.value_changed.connect(lambda _v: self._on_method())
        card.add(Field("t.method", self.method))

        self.tolerance = float_spin(0.001, 0.050, 0.006, decimals=3, step=0.001)
        self.tolerance.valueChanged.connect(lambda _v: self._on_rounds())
        card.add(Field("t.tolerance", self.tolerance))

        self.max_rounds = int_spin(1, 50, 6)
        self.max_rounds.valueChanged.connect(lambda _v: self._on_rounds())
        card.add(Field("t.max_rounds", self.max_rounds))

        self._fit_controls: List[QWidget] = [
            self.level, self.rmsea, self.use_cfi, self.cfi, self.use_srmr,
            self.srmr, self.method, self.tolerance, self.max_rounds,
        ]

    def _build_thresholds_panel(self) -> None:
        """The cut-offs the researcher is aiming at, always on screen."""
        self._thresholds = TablePanel("t.thresholds", max_rows=6)
        self.add(self._thresholds)

    def _threshold_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "index": [translator.t(key) for key, _v in FIT_THRESHOLDS],
            "threshold": [value for _k, value in FIT_THRESHOLDS],
        })

    # -- 2. significance targeting --------------------------------------- #

    def _build_significance_card(self) -> None:
        card = self.add_card("t.significance")
        card.add(InfoBanner("t.significance.explain"))

        self.enable_significance = TCheckBox("t.enable_significance")
        self.enable_significance.toggled.connect(self._on_enable_significance)
        card.add(self.enable_significance)

        self.alpha = float_spin(0.001, 0.200, 0.05, decimals=3, step=0.005)
        self.alpha.valueChanged.connect(lambda _v: self._on_significance())
        card.add(Field("p.alpha", self.alpha))

        self.target_power = float_spin(0.50, 0.999, 0.80, decimals=2, step=0.05)
        self.target_power.valueChanged.connect(lambda _v: self._on_significance())
        card.add(Field("p.target_power", self.target_power))

        self.null_coefficient = float_spin(0.0, 0.200, 0.02, decimals=3,
                                           step=0.005)
        self.null_coefficient.valueChanged.connect(
            lambda _v: self._on_significance())
        card.add(Field("t.null_coefficient", self.null_coefficient))

        self.max_resamples = int_spin(0, 500, 25)
        self.max_resamples.valueChanged.connect(lambda _v: self._on_significance())
        card.add(Field("t.max_resamples", self.max_resamples))

        self.enforce_null = TCheckBox("t.enforce_null")
        self.enforce_null.toggled.connect(lambda _c: self._on_significance())
        card.add(self.enforce_null)

        self._significance_controls: List[QWidget] = [
            self.alpha, self.target_power, self.null_coefficient,
            self.max_resamples, self.enforce_null,
        ]

    # -- 3. per-path targets --------------------------------------------- #

    def _build_paths_card(self) -> None:
        card = self.add_card("t.per_path")
        self.path_rows = RowList("s.add_path", empty_key="t.no_paths")
        # Paths are created in stage 1; this sheet only aims them.
        self.path_rows.add_button.setVisible(False)
        self.path_rows.set_header([("s.source", 4), ("s.target", 4),
                                   ("s.effect_size", 3), ("s.coefficient", 2),
                                   ("t.requested", 4)])
        card.add(self.path_rows)
        self.path_summary = TKey("t.targets_summary", role="hint",
                                 sig=0, null=0, total=0)
        card.add(self.path_summary)

    def _rebuild_path_rows(self) -> None:
        self.path_rows.clear()
        for path in self.spec.paths:
            self.path_rows.add_row(self._path_row(path))
        for interaction in self.spec.interactions:
            self.path_rows.add_row(self._interaction_row(interaction))

    def _name_of(self, cid: str) -> str:
        construct = self.spec.construct(cid)
        if construct is not None:
            return construct.name or construct.id
        dimension = self.spec.dimension(cid)
        if dimension is not None:
            return dimension.name or dimension.id
        return translator.t("v.missing_ref")

    def _label_of(self, cid: str) -> str:
        return self.spec.factor_label(cid, translator.language)

    def _outcome_combo(self, target: Any) -> EnumCombo:
        """A combo writing back to ``target.target_significant``."""
        combo = EnumCombo(str, "verdict", members=OUTCOME_CHOICES)
        combo.set_value(OUTCOME_SIGNIFICANT if target.target_significant
                        else OUTCOME_NULL)

        def on_outcome(value: Any) -> None:
            target.target_significant = (str(value) == OUTCOME_SIGNIFICANT)
            self.edit()
            self._update_path_summary()

        combo.value_changed.connect(on_outcome)
        return combo

    def _path_row(self, path: Path) -> Row:
        row = Row()
        source = _mono(self._name_of(path.source))
        source.setToolTip(self._label_of(path.source))
        row.add(source, 4)
        target = _mono(self._name_of(path.target))
        target.setToolTip(self._label_of(path.target))
        row.add(target, 4)
        row.add(QLabel(translator.enum("effect", path.effect_size)), 3)
        row.add(_mono(f"{path.resolved_coefficient():+.2f}"), 2)
        row.add(self._outcome_combo(path), 4)
        return row

    def _interaction_row(self, interaction: Interaction) -> Row:
        row = Row()
        source = _mono(f"{self._name_of(interaction.predictor)} × "
                       f"{self._name_of(interaction.moderator)}")
        source.setToolTip(f"{self._label_of(interaction.predictor)} × "
                          f"{self._label_of(interaction.moderator)}")
        row.add(source, 4)
        target = _mono(self._name_of(interaction.target))
        target.setToolTip(self._label_of(interaction.target))
        row.add(target, 4)
        row.add(QLabel(translator.enum("effect", interaction.effect_size)), 3)
        row.add(_mono(f"{interaction.resolved_coefficient():+.2f}"), 2)
        row.add(self._outcome_combo(interaction), 4)
        return row

    def _update_path_summary(self) -> None:
        targets = list(self.spec.paths) + list(self.spec.interactions)
        significant = sum(1 for t in targets if t.target_significant)
        self.path_summary.set_key("t.targets_summary",
                                  sig=significant,
                                  null=len(targets) - significant,
                                  total=len(targets))

    # -- 4. calibration result ------------------------------------------- #

    def _build_calibration_section(self) -> None:
        self._calibration_card = Card("t.achieved")
        self.add(self._calibration_card)
        strip = QHBoxLayout()
        strip.setSpacing(8)
        self._badge_target = StatBadge("t.target_rmsea")
        self._badge_realised = StatBadge("t.realised_rmsea")
        self._badge_rounds = StatBadge("t.rounds_used")
        for badge in (self._badge_target, self._badge_realised,
                      self._badge_rounds):
            theme.set_role(badge, "card")
            strip.addWidget(badge)
        strip.addStretch(1)
        self._calibration_card.add(strip)

        chips = QHBoxLayout()
        chips.setSpacing(8)
        self._chip_tolerance = VerdictChip("pass")
        self._chip_paths = VerdictChip("pass")
        chips.addWidget(self._chip_tolerance)
        chips.addWidget(self._chip_paths)
        chips.addStretch(1)
        self._calibration_card.add(chips)

        self._calibration_table = TablePanel("t.calibration", empty_history())
        self.add(self._calibration_table)

        self._calibration_hint = TKey("t.no_calibration", role="hint")
        self.add(self._calibration_hint)

    def _on_results_changed(self) -> None:
        self._update_calibration()

    def _update_calibration(self) -> None:
        calibration = getattr(self.state.results, "calibration", None)
        present = calibration is not None
        self._calibration_card.setVisible(present)
        self._calibration_table.setVisible(present)
        self._calibration_hint.setVisible(not present)
        if not present:
            return

        target = float(self._target_rmsea())
        realised = self._realised_rmsea(calibration)
        tolerance = float(getattr(self.spec.fit_target, "tolerance", 0.006))
        self._badge_target.set_value(f"{target:.3f}")
        rounds = getattr(calibration, "rounds", None)
        if rounds is None:
            history = getattr(calibration, "history", None)
            rounds = len(history) if history is not None else "-"
        self._badge_rounds.set_value(rounds)

        if realised is None:
            self._badge_realised.set_value("-")
            self._chip_tolerance.set_verdict(
                "borderline", translator.t("t.outside_tolerance"))
        else:
            inside = abs(realised - target) <= tolerance
            self._badge_realised.set_value(
                f"{realised:.3f}", kind="pass" if inside else "fail")
            self._chip_tolerance.set_verdict(
                "pass" if inside else "fail",
                translator.t("t.within_tolerance" if inside
                             else "t.outside_tolerance"))

        matched = self._paths_matched(calibration)
        if matched is None:
            self._chip_paths.set_verdict("borderline",
                                         translator.t("t.paths_unmatched"))
        else:
            self._chip_paths.set_verdict(
                "pass" if matched else "fail",
                translator.t("t.paths_matched" if matched
                             else "t.paths_unmatched"))

        self._calibration_table.set_frame(self._history_frame(calibration))
        _size_to_content(self._calibration_table, 14)

    def _history_frame(self, calibration: Any) -> pd.DataFrame:
        """The calibration log as a frame, whatever shape it arrives in.

        The calibration module may not exist yet, and a build that has one may
        report its history as a list of records rather than a frame.  Anything
        unreadable degrades to the empty log rather than taking the page down.
        """
        history = getattr(calibration, "history", None)
        frame: pd.DataFrame
        if isinstance(history, pd.DataFrame):
            frame = history
        elif history is None:
            frame = empty_history()
        else:
            try:
                frame = pd.DataFrame(list(history))
            except Exception:
                frame = empty_history()
        # A frame with no columns at all would leave the view without a
        # section 0 to re-label when the language changes.
        return frame if len(frame.columns) else empty_history()

    def _realised_rmsea(self, calibration: Any) -> Optional[float]:
        """The RMSEA the run actually landed on, however it is reported.

        The calibration module may not exist yet, and when it does it may
        report a discrepancy function rather than an index, so every route is
        tried and a missing one simply yields ``None``.
        """
        for attribute in ("realised_rmsea", "achieved_rmsea", "rmsea",
                          "final_rmsea"):
            value = getattr(calibration, attribute, None)
            if isinstance(value, (int, float)):
                return float(value)
        fit = getattr(self.state.results, "fit", None)
        value = getattr(fit, "rmsea", None)
        if isinstance(value, (int, float)):
            return float(value)
        discrepancy = getattr(calibration, "discrepancy", None)
        if isinstance(discrepancy, (int, float)):
            try:
                from semsim.targeting.cudeck_browne import rmsea_from_discrepancy
            except Exception:
                return None
            try:
                return float(rmsea_from_discrepancy(
                    float(discrepancy), self.spec.degrees_of_freedom()))
            except Exception:
                return None
        return None

    def _paths_matched(self, calibration: Any) -> Optional[bool]:
        for attribute in ("paths_matched", "all_targets_met", "targets_met",
                          "significance_matched"):
            value = getattr(calibration, attribute, None)
            if isinstance(value, bool):
                return value
        unmatched = getattr(calibration, "unmatched_paths", None)
        if unmatched is not None:
            try:
                return not len(unmatched)
            except TypeError:
                return None
        return None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def rebuild(self) -> None:
        with self.quiet():
            self._load_fit()
            self._load_significance()
            self._rebuild_path_rows()
        self._thresholds.set_frame(self._threshold_frame())
        _size_to_content(self._thresholds, len(FIT_THRESHOLDS) + 1)
        self._update_path_summary()
        self._update_availability()
        self._update_calibration()

    def refresh(self) -> None:
        # Every row carries translated outcome labels and effect-size text, so
        # a language switch rebuilds the sheet rather than patching it.
        self.rebuild()

    def set_theme(self, name: str) -> None:
        super().set_theme(name)
        self._thresholds.set_theme(name)
        self._calibration_table.set_theme(name)

    # ------------------------------------------------------------------ #
    # Loading the spec into the widgets
    # ------------------------------------------------------------------ #

    def _target_rmsea(self) -> float:
        """The RMSEA the spec is really aiming at.

        ``rmsea`` is authoritative for a custom level; every other level reads
        its conventional value off the enum.
        """
        fit = self.spec.fit_target
        level = FitLevel(fit.level)
        if level is FitLevel.CUSTOM:
            return float(fit.rmsea)
        return float(level.target_rmsea)

    def _load_fit(self) -> None:
        fit = self.spec.fit_target
        self.enable_fit.setChecked(bool(fit.enabled))
        level = FitLevel(fit.level)
        self.level.set_value(level)
        self.rmsea.setValue(float(fit.rmsea))
        self.use_cfi.setChecked(fit.cfi is not None)
        if fit.cfi is not None:
            self.cfi.setValue(float(fit.cfi))
        self.use_srmr.setChecked(fit.srmr is not None)
        if fit.srmr is not None:
            self.srmr.setValue(float(fit.srmr))
        self.method.set_value(SPEC_TO_METHOD.get(fit.method, "cudeck"))
        self.tolerance.setValue(float(fit.tolerance))
        self.max_rounds.setValue(int(fit.max_calibration_rounds))
        self._update_level_hint(level)

    def _load_significance(self) -> None:
        target = self.spec.significance_target
        self.enable_significance.setChecked(bool(target.enabled))
        self.alpha.setValue(float(target.alpha))
        self.target_power.setValue(float(target.target_power))
        self.null_coefficient.setValue(float(target.null_coefficient))
        self.max_resamples.setValue(int(target.max_resamples))
        self.enforce_null.setChecked(bool(target.enforce_null_paths))

    def _update_level_hint(self, level: FitLevel) -> None:
        self.level_hint.set_key("t.level.implies",
                                rmsea=f"{level.target_rmsea:.3f}",
                                cfi=f"{level.expected_cfi_floor:.3f}")

    def _degrees_of_freedom(self) -> Optional[int]:
        try:
            return int(self.spec.degrees_of_freedom())
        except Exception:
            return None

    def _update_availability(self) -> None:
        """A saturated model has no fit to target, so say so and stand down."""
        df = self._degrees_of_freedom()
        saturated = df is not None and df <= 0
        self._saturated_banner.setVisible(saturated)
        self.enable_fit.setEnabled(not saturated)
        enabled = (not saturated) and self.enable_fit.isChecked()
        for widget in self._fit_controls:
            widget.setEnabled(enabled)
        if enabled:
            self._cfi_field.editor.setEnabled(self.use_cfi.isChecked())
            self._srmr_field.editor.setEnabled(self.use_srmr.isChecked())

        on = self.enable_significance.isChecked()
        for widget in self._significance_controls:
            widget.setEnabled(on)

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    def _on_enable_fit(self, checked: bool) -> None:
        self.spec.fit_target.enabled = bool(checked)
        self._update_availability()
        self.edit()

    def _on_level(self) -> None:
        fit = self.spec.fit_target
        level = enum_value(self.level, FitLevel, FitLevel.GOOD)
        fit.level = level
        self._update_level_hint(level)
        if level is not FitLevel.CUSTOM:
            # Picking a band writes its conventional RMSEA, so the spin box
            # must not go on showing the previous number.
            fit.rmsea = float(level.target_rmsea)
            blocked = self.rmsea.blockSignals(True)
            self.rmsea.setValue(fit.rmsea)
            self.rmsea.blockSignals(blocked)
        self.edit()

    def _on_rmsea(self) -> None:
        # Typing an RMSEA makes the target the researcher's own, so the level
        # combo must stop claiming a conventional band.
        fit = self.spec.fit_target
        fit.rmsea = float(self.rmsea.value())
        if FitLevel(fit.level) is not FitLevel.CUSTOM:
            fit.level = FitLevel.CUSTOM
            self.level.set_value(FitLevel.CUSTOM)
            self._update_level_hint(FitLevel.CUSTOM)
        self.edit()

    def _on_secondary(self) -> None:
        fit = self.spec.fit_target
        fit.cfi = float(self.cfi.value()) if self.use_cfi.isChecked() else None
        fit.srmr = (float(self.srmr.value()) if self.use_srmr.isChecked()
                    else None)
        self._cfi_field.editor.setEnabled(self.use_cfi.isChecked())
        self._srmr_field.editor.setEnabled(self.use_srmr.isChecked())
        self.edit()

    def _on_method(self) -> None:
        suffix = str(self.method.value() or "cudeck")
        self.spec.fit_target.method = METHOD_TO_SPEC.get(suffix, "cudeck_browne")
        self.edit()

    def _on_rounds(self) -> None:
        fit = self.spec.fit_target
        fit.tolerance = float(self.tolerance.value())
        fit.max_calibration_rounds = int(self.max_rounds.value())
        self.edit()

    def _on_enable_significance(self, checked: bool) -> None:
        self.spec.significance_target.enabled = bool(checked)
        self._update_availability()
        self.edit()

    def _on_significance(self) -> None:
        target = self.spec.significance_target
        target.alpha = float(self.alpha.value())
        target.target_power = float(self.target_power.value())
        target.null_coefficient = float(self.null_coefficient.value())
        target.max_resamples = int(self.max_resamples.value())
        target.enforce_null_paths = bool(self.enforce_null.isChecked())
        self.edit()


__all__ = ["TargetingPage"]

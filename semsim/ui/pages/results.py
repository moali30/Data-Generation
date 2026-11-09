"""
semsim.ui.pages.results
=======================
Stage 6 -- Validation & Fit Output.

What this page has to get right is not the numbers, which arrive already
computed, but the *reading* of them.  A researcher looking at a results screen
needs three things at once: the value, the threshold it is judged against, and
whether it cleared it.  Every table here carries all three, and the headline
indices are repeated as large tiles so the verdict is legible from across a
desk.

Nothing is shown before a run has happened.  An empty page that explains what
is missing beats a page of dashes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget,
)

from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.pages.base import StagePage
from semsim.ui.widgets.common import (
    Card, InfoBanner, ScrollPage, StatBadge, TButton, VerdictChip, separator,
)
from semsim.ui.widgets.tables import TablePanel

#: Conventional cut-offs, used for the headline tiles.
HEADLINE = [
    ("rmsea", "r.rmsea", 0.06, "below"),
    ("cfi", "r.cfi", 0.95, "above"),
    ("tli", "r.tli", 0.95, "above"),
    ("srmr", "r.srmr", 0.08, "below"),
    ("chi_square_df_ratio", "r.chi2_df", 3.0, "below"),
]


class ResultsPage(StagePage):
    TITLE_KEY = "stage.6.title"
    SUBTITLE_KEY = "stage.6.subtitle"

    # ------------------------------------------------------------------ #

    def build(self) -> None:
        self._empty = InfoBanner("r.no_results", kind="info")
        self.scroll.add(self._empty)

        self._tiles_card = Card("r.fit", both=True)
        self._tiles_row = QHBoxLayout()
        self._tiles_row.setSpacing(8)
        self._tiles_card.add(self._tiles_row)
        self._tiles: Dict[str, StatBadge] = {}
        for key, label_key, _threshold, _dir in HEADLINE:
            badge = StatBadge(label_key, "-", wrap=True)
            self._tiles[key] = badge
            self._tiles_row.addWidget(badge, 1)
        self._verdict = VerdictChip("pass")
        self._tiles_row.addWidget(self._verdict)
        self.scroll.add(self._tiles_card)

        self._tabs = QTabWidget()
        self.scroll.add(self._tabs)

        self._panels: Dict[str, TablePanel] = {}
        self._tab_keys: List[str] = []

        self._tabs.addTab(self._build_fit_tab(), "")
        self._tab_keys.append("r.fit")
        self._tabs.addTab(self._build_reliability_tab(), "")
        self._tab_keys.append("r.reliability")
        self._tabs.addTab(self._build_validity_tab(), "")
        self._tab_keys.append("r.validity")
        self._tabs.addTab(self._build_estimates_tab(), "")
        self._tab_keys.append("r.estimates")
        self._tabs.addTab(self._build_descriptives_tab(), "")
        self._tab_keys.append("r.descriptives")
        self._tabs.addTab(self._build_data_tab(), "")
        self._tab_keys.append("r.data_preview")
        self._tabs.addTab(self._build_targeting_tab(), "")
        self._tab_keys.append("t.calibration")
        self._tabs.addTab(self._build_demographics_tab(), "")
        self._tab_keys.append("dem.tab")

        actions = QHBoxLayout()
        self._export_button = TButton("act.export", role="primary")
        self._export_button.clicked.connect(self._on_export)
        actions.addWidget(self._export_button)
        self._report_button = TButton("act.save_report")
        self._report_button.clicked.connect(self._on_save_report)
        actions.addWidget(self._report_button)
        self._rerun_button = TButton("act.rerun")
        self._rerun_button.clicked.connect(self._on_rerun)
        self.state.busy_changed.connect(
            lambda busy: self._rerun_button.setEnabled(not busy))
        actions.addWidget(self._rerun_button)
        actions.addStretch(1)
        self.scroll.add(actions)
        self.scroll.add_stretch()

        translator.language_changed.connect(lambda _l: self._retranslate_tabs())
        self._retranslate_tabs()
        self.refresh()

    def _panel(self, key: str, host: ScrollPage, *, max_rows: int = 14,
               both: bool = False,
               hide_columns: Optional[List[str]] = None) -> TablePanel:
        panel = TablePanel(key, max_rows=max_rows, both=both,
                           hide_columns=hide_columns)
        host.add(panel)
        self._panels[key] = panel
        return panel

    def _tab(self) -> ScrollPage:
        page = ScrollPage()
        return page

    # -- tabs -------------------------------------------------------------- #

    def _build_fit_tab(self) -> QWidget:
        page = self._tab()
        # "index" carries the machine code (rmsea, cfi, ...). It stays in the
        # frame so each row gets the right number of decimals, but the reader
        # sees the translated label instead.
        self._panel("r.fit", page, max_rows=20, both=True,
                    hide_columns=["index"])
        self._fit_note = QLabel()
        theme.set_role(self._fit_note, "hint")
        self._fit_note.setWordWrap(True)
        page.add(self._fit_note)
        page.add_stretch()
        return page

    def _build_reliability_tab(self) -> QWidget:
        page = self._tab()
        page.add(InfoBanner("r.reliability.explain"))
        self._panel("r.reliability", page, max_rows=16, both=True)
        self._panel("r.item_total", page, max_rows=20, both=True)
        page.add_stretch()
        return page

    def _build_validity_tab(self) -> QWidget:
        page = self._tab()
        self._panel("r.fornell", page, max_rows=14, both=True)
        self._panel("r.htmt", page, max_rows=14, both=True)
        self._panel("r.loadings", page, max_rows=24, both=True)
        self._validity_flags = QLabel()
        theme.set_role(self._validity_flags, "warning")
        self._validity_flags.setWordWrap(True)
        page.add(self._validity_flags)
        page.add_stretch()
        return page

    def _build_estimates_tab(self) -> QWidget:
        page = self._tab()
        self._panel("r.paths", page, max_rows=16, both=True)
        self._panel("r.indirect", page, max_rows=10, both=True)
        self._panel("r.r_squared", page, max_rows=10, both=True)
        self._panel("r.estimates", page, max_rows=24, both=True)
        page.add_stretch()
        return page

    def _build_descriptives_tab(self) -> QWidget:
        page = self._tab()
        self._panel("r.descriptives", page, max_rows=24, both=True)
        self._normality = QLabel()
        theme.set_role(self._normality, "hint")
        self._normality.setWordWrap(True)
        page.add(self._normality)
        page.add_stretch()
        return page

    def _build_data_tab(self) -> QWidget:
        page = self._tab()
        self._panel("r.data_preview", page, max_rows=25)
        self._data_note = QLabel()
        theme.set_role(self._data_note, "hint")
        page.add(self._data_note)
        page.add_stretch()
        return page

    def _build_targeting_tab(self) -> QWidget:
        page = self._tab()
        self._panel("t.per_path", page, max_rows=14, both=True)
        self._panel("t.calibration", page, max_rows=12, both=True)
        self._targeting_note = QLabel()
        theme.set_role(self._targeting_note, "hint")
        self._targeting_note.setWordWrap(True)
        page.add(self._targeting_note)
        page.add_stretch()
        return page

    def _build_demographics_tab(self) -> QWidget:
        page = self._tab()
        self._demographics_empty = InfoBanner("dem.none")
        page.add(self._demographics_empty)
        self._panel("dem.frequencies", page, max_rows=30)
        self._panel("dem.comparisons", page, max_rows=30)
        self._panel("dem.invariance", page, max_rows=20)
        self._panel("dem.group_paths", page, max_rows=30)
        self._demographics_note = QLabel()
        theme.set_role(self._demographics_note, "hint")
        self._demographics_note.setWordWrap(True)
        page.add(self._demographics_note)
        page.add_stretch()
        return page

    def _retranslate_tabs(self) -> None:
        try:
            for index, key in enumerate(self._tab_keys):
                self._tabs.setTabText(index, translator.t(key))
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Refresh
    # ------------------------------------------------------------------ #

    def _on_results_changed(self) -> None:
        self.refresh()

    def set_theme(self, name: str) -> None:
        super().set_theme(name)
        for panel in self._panels.values():
            panel.set_theme(name)

    def refresh(self) -> None:
        results = self.state.results
        has = results.dataset is not None
        self._empty.setVisible(not has)
        self._tiles_card.setVisible(has)
        self._tabs.setVisible(has)
        for button in (self._export_button, self._report_button):
            button.setEnabled(has)
        if not has:
            return

        self._fill_headline(results.fit)
        self._fill_fit(results.fit)
        self._fill_reliability(results.validity)
        self._fill_validity(results.validity)
        self._fill_estimates(results.estimation, results.dataset)
        self._fill_descriptives(results.descriptives)
        self._fill_data(results.dataset)
        self._fill_targeting(results.calibration, results.estimation)
        self._fill_demographics(results.demographics)

    # -- fillers ----------------------------------------------------------- #

    def _fill_headline(self, fit: Any) -> None:
        if fit is None:
            for badge in self._tiles.values():
                badge.set_value("-")
            self._verdict.setVisible(False)
            return
        self._verdict.setVisible(True)
        for key, _label, threshold, direction in HEADLINE:
            value = getattr(fit, key, None)
            if value is None or not np.isfinite(value):
                self._tiles[key].set_value("-")
                continue
            passed = (value <= threshold if direction == "below"
                      else value >= threshold)
            places = 4 if key in ("rmsea", "srmr") else 3
            self._tiles[key].set_value(f"{value:.{places}f}",
                                       kind="pass" if passed else "fail")
        try:
            level = fit.verdict()
            self._verdict.set_verdict(
                {"excellent": "pass", "good": "pass", "perfect": "pass",
                 "acceptable": "borderline", "marginal": "borderline"}
                .get(level.value, "fail"),
                translator.t(f"fit.{level.value}"))
        except Exception:
            self._verdict.setVisible(False)

    def _fill_fit(self, fit: Any) -> None:
        if fit is None:
            self._panels["r.fit"].set_frame(pd.DataFrame())
            self._fit_note.setText("")
            return
        try:
            table = _localise(fit.to_table(translator.language))
        except Exception:
            table = pd.DataFrame()
        self._panels["r.fit"].set_frame(table)
        self._fit_note.setText(
            f"N = {fit.n}   |   {translator.t('p.df')} = {fit.df}   |   "
            f"{translator.t('p.free_params')} = {fit.n_free}   |   "
            f"χ² = {fit.chi_square:.3f} (p = {fit.p_value:.4f})   |   "
            f"RMSEA 90% CI = [{fit.rmsea_ci_lower:.4f}, "
            f"{fit.rmsea_ci_upper:.4f}]   |   PCLOSE = {fit.rmsea_pclose:.4f}")

    def _fill_reliability(self, validity: Any) -> None:
        if validity is None or not getattr(validity, "per_factor", None):
            self._panels["r.reliability"].set_frame(pd.DataFrame())
            self._panels["r.item_total"].set_frame(pd.DataFrame())
            return
        rows = []
        item_rows = []
        for factor in validity.per_factor:
            rows.append({
                "index": factor.factor_name,
                "n_items": factor.n_items,
                "alpha": factor.cronbach_alpha,
                "cr": factor.composite_reliability,
                "omega": factor.omega_total,
                "ave": factor.ave,
                "verdict": _reliability_verdict(factor),
            })
            for name, value in (factor.item_total_correlations or {}).items():
                item_rows.append({
                    "item": name,
                    "factor": factor.factor_name,
                    "correlation": value,
                    "alpha": (factor.alpha_if_item_dropped or {}).get(name, np.nan),
                    "verdict": "pass" if value >= 0.30 else "fail",
                })
        self._panels["r.reliability"].set_frame(pd.DataFrame(rows))
        self._panels["r.item_total"].set_frame(pd.DataFrame(item_rows))

    def _fill_validity(self, validity: Any) -> None:
        if validity is None:
            for key in ("r.fornell", "r.htmt", "r.loadings"):
                self._panels[key].set_frame(pd.DataFrame())
            self._validity_flags.setText("")
            return
        self._panels["r.fornell"].set_frame(
            _reset(getattr(validity, "fornell_larcker", None)))
        self._panels["r.htmt"].set_frame(
            _reset(getattr(validity, "htmt", None)))
        self._panels["r.loadings"].set_frame(
            _reset(getattr(validity, "cross_loading_table", None)))
        flags = list(getattr(validity, "flags", []) or [])
        verdicts = []
        if not getattr(validity, "fornell_larcker_passed", True):
            verdicts.append(translator.t("r.fornell") + ": "
                            + translator.t("verdict.fail"))
        if not getattr(validity, "htmt_passed", True):
            verdicts.append(translator.t("r.htmt") + ": "
                            + translator.t("verdict.fail"))
        self._validity_flags.setText("  |  ".join(verdicts + flags[:4]))

    def _fill_estimates(self, estimation: Any, dataset: Any) -> None:
        if estimation is None:
            for key in ("r.paths", "r.indirect", "r.r_squared", "r.estimates"):
                self._panels[key].set_frame(pd.DataFrame())
            return
        paths = _reset(getattr(estimation, "path_estimates", None))
        if not paths.empty:
            keep = [c for c in ("source", "target", "est", "se", "z", "p", "std")
                    if c in paths.columns]
            paths = paths[keep].copy()
            if "p" in paths.columns:
                paths["verdict"] = np.where(paths["p"] < 0.05,
                                            "significant", "not significant")
        self._panels["r.paths"].set_frame(paths)
        self._panels["r.indirect"].set_frame(
            _indirect_table(self.spec, estimation, dataset))
        r2 = getattr(estimation, "r_squared", None) or {}
        self._panels["r.r_squared"].set_frame(pd.DataFrame(
            [{"index": k, "value": v} for k, v in r2.items()]))
        loadings = _reset(getattr(estimation, "loadings", None))
        if not loadings.empty:
            keep = [c for c in ("item", "factor", "est", "se", "z", "p", "std")
                    if c in loadings.columns]
            loadings = loadings[keep]
        self._panels["r.estimates"].set_frame(loadings)

    def _fill_descriptives(self, descriptives: Any) -> None:
        if descriptives is None:
            self._panels["r.descriptives"].set_frame(pd.DataFrame())
            self._normality.setText("")
            return
        try:
            table = _localise(descriptives.to_table(translator.language))
        except Exception:
            table = pd.DataFrame()
        self._panels["r.descriptives"].set_frame(table)
        verdict = (translator.t("verdict.pass")
                   if descriptives.multivariate_normal
                   else translator.t("verdict.fail"))
        self._normality.setText(
            f"{translator.t('r.mardia')}:  skewness = "
            f"{descriptives.mardia_skewness:.3f} "
            f"(p = {descriptives.mardia_skewness_p:.4g}),  kurtosis = "
            f"{descriptives.mardia_kurtosis:.3f} "
            f"(p = {descriptives.mardia_kurtosis_p:.4g})   —   {verdict}")

    def _fill_demographics(self, analysis: Any) -> None:
        keys = ("dem.frequencies", "dem.comparisons", "dem.invariance",
                "dem.group_paths")
        self._demographics_empty.setVisible(analysis is None)
        if analysis is None:
            for key in keys:
                self._panels[key].set_frame(pd.DataFrame())
                self._panels[key].setVisible(False)
            self._demographics_note.setText("")
            return

        def yes_no(value: Any) -> str:
            if value is None or value != value:
                return ""
            return translator.t("dem.yes" if value else "dem.no")

        frequencies = _localise(analysis.frequencies)
        if not frequencies.empty:
            frequencies["role"] = frequencies["role"].map(
                lambda v: translator.t(f"drole.{v}"))
        comparisons = _localise(analysis.comparisons)
        if not comparisons.empty:
            comparisons["test"] = comparisons["test"].map(
                lambda v: translator.t(f"dem.test.{v}"))
            comparisons["effect_kind"] = comparisons["effect_kind"].map(
                lambda v: translator.t(f"dem.kind.{v}"))
            comparisons["significant"] = comparisons["significant"].map(yes_no)
        invariance = _localise(analysis.invariance)
        if not invariance.empty:
            invariance["invariant"] = invariance["invariant"].map(yes_no)
        frames = (frequencies, comparisons, invariance,
                  _localise(analysis.group_paths))
        for key, frame in zip(keys, frames):
            self._panels[key].set_frame(frame)
            self._panels[key].setVisible(not frame.empty)
        side = 1 if translator.is_rtl else 0
        self._demographics_note.setText("\n".join(
            note.split(" | ")[side] if " | " in note else note
            for note in analysis.notes))

    def _fill_data(self, dataset: Any) -> None:
        data = getattr(dataset, "data", None)
        if data is None:
            self._panels["r.data_preview"].set_frame(pd.DataFrame())
            self._data_note.setText("")
            return
        preview = getattr(dataset, "full_frame", data).head(100).copy()
        preview.insert(0, "index", np.arange(1, len(preview) + 1))
        self._panels["r.data_preview"].set_frame(preview)
        missing = int(data.isna().to_numpy().sum())
        total = data.size or 1
        self._data_note.setText(
            f"{translator.t('r.n')} = {len(data)}   |   "
            f"{translator.t('m.items')} = {data.shape[1]}   |   "
            f"{translator.t('r.missing')} = {missing} "
            f"({missing / total * 100:.2f}%)   |   "
            f"{translator.t('d.seed')} = {getattr(dataset, 'seed', '-')}   |   "
            f"{translator.t('d.engine')} = "
            f"{translator.t('d.engine.' + getattr(dataset, 'engine_used', None).value)}"
            if getattr(dataset, "engine_used", None) is not None else "")

    def _fill_targeting(self, calibration: Any, estimation: Any) -> None:
        table = pd.DataFrame()
        if calibration is not None:
            table = _reset(getattr(calibration, "significance_table", None))
        elif estimation is not None:
            try:
                from semsim.targeting.significance import check_realised
                _ok, table = check_realised(estimation, self.spec)
            except Exception:
                table = pd.DataFrame()
        self._panels["t.per_path"].set_frame(table)
        history = (_reset(getattr(calibration, "history", None))
                   if calibration is not None else pd.DataFrame())
        self._panels["t.calibration"].set_frame(history)
        if calibration is None:
            self._targeting_note.setText("")
            return
        summary = calibration.summary() if hasattr(calibration, "summary") else {}
        parts = [
            f"{translator.t('t.round')} = {summary.get('rounds', '-')}",
            f"{translator.t('t.requested')} RMSEA = "
            f"{summary.get('target_rmsea', float('nan')):.4f}",
            f"{translator.t('t.achieved')} RMSEA = "
            f"{summary.get('realised_rmsea', float('nan')):.4f}",
            (translator.t("verdict.pass") if summary.get("within_tolerance")
             else translator.t("verdict.fail")),
        ]
        self._targeting_note.setText("   |   ".join(parts))

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _window(self) -> Any:
        widget = self
        while widget is not None and not hasattr(widget, "export_results"):
            widget = widget.parentWidget()
        return widget

    def _on_export(self) -> None:
        window = self._window()
        if window is not None:
            window.export_results()

    def _on_rerun(self) -> None:
        window = self._window()
        if window is not None and hasattr(window, "run"):
            window.run("full")

    def _on_save_report(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _filter = QFileDialog.getSaveFileName(
            self, translator.t("act.save_report"), "report.html",
            "HTML (*.html)")
        if not path:
            return
        try:
            from semsim.io.report import ReportBundle, write_html
            results = self.state.results
            bundle = ReportBundle(
                spec=self.spec, dataset=results.dataset,
                estimation=results.estimation, fit=results.fit,
                validity=results.validity, descriptives=results.descriptives,
                power=results.power, calibration=results.calibration,
                demographics=results.demographics)
            write_html(bundle, path, lang=translator.language)
        except Exception as exc:
            QMessageBox.warning(self, translator.t("status.failed"), str(exc))
            return
        QMessageBox.information(
            self, translator.t("act.save_report"),
            f"{translator.t('dlg.export.done')}:\n{path}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _localise(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the active language's column from each ``*_ar`` / ``*_en`` pair.

    The analysis modules return both so a report can be rendered in either
    language without recomputing; the screen only ever shows one.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    suffix = "_ar" if translator.is_rtl else "_en"
    drop = "_en" if translator.is_rtl else "_ar"
    out = frame.copy()
    renames = {}
    for column in list(out.columns):
        name = str(column)
        if name.endswith(drop):
            base = name[: -len(drop)]
            if f"{base}{suffix}" in out.columns:
                out = out.drop(columns=[column])
                continue
        if name.endswith(suffix):
            renames[column] = name[: -len(suffix)]
    return out.rename(columns=renames)


def _reset(frame: Any) -> pd.DataFrame:
    """A DataFrame with its index promoted to a column, so it is visible."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    if frame.index.name or not isinstance(frame.index, pd.RangeIndex):
        out = frame.reset_index()
        out.columns = ["index"] + list(out.columns[1:])
        return out
    return frame


def _reliability_verdict(factor: Any) -> str:
    """Pass when alpha and CR clear 0.70 and AVE clears 0.50."""
    alpha_ok = factor.cronbach_alpha >= 0.70
    cr_ok = factor.composite_reliability >= 0.70
    ave_ok = factor.ave >= 0.50
    if alpha_ok and cr_ok and ave_ok:
        return "pass"
    if (factor.cronbach_alpha >= 0.60 and factor.composite_reliability >= 0.60
            and factor.ave >= 0.45):
        return "borderline"
    return "fail"


def _indirect_table(spec: Any, estimation: Any, dataset: Any) -> pd.DataFrame:
    """Each declared mediation chain, as the product along its estimated path."""
    if not getattr(spec, "indirect_effects", None):
        return pd.DataFrame()
    estimates = _reset(getattr(estimation, "path_estimates", None))
    if estimates.empty:
        return pd.DataFrame()
    lookup: Dict[tuple, Dict[str, float]] = {}
    for _i, row in estimates.iterrows():
        lookup[(str(row.get("source")), str(row.get("target")))] = row

    rows = []
    for indirect in spec.indirect_effects:
        names = []
        product = 1.0
        ok = True
        for cid in indirect.chain:
            construct = spec.construct(cid)
            names.append(construct.name if construct else "?")
        for source, target in zip(names, names[1:]):
            row = lookup.get((source, target))
            if row is None:
                ok = False
                break
            product *= float(row.get("std", row.get("est", np.nan)))
        if not ok:
            continue
        specified = 1.0
        for source, target in zip(indirect.chain, indirect.chain[1:]):
            match = next((p for p in spec.paths
                          if p.source == source and p.target == target), None)
            specified *= match.resolved_coefficient() if match else float("nan")
        rows.append({
            "index": " → ".join(names),
            "specified": specified,
            "estimate": product,
            "requested": ("significant" if indirect.target_significant
                          else "not significant"),
        })
    return pd.DataFrame(rows)


__all__ = ["ResultsPage"]

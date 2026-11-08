"""
semsim.ui.app
=============
The main window: six stage pages, a live model summary, and the run plumbing.

Navigation is a list rather than a strict wizard.  The six stages are ordered
the way the methodology is ordered, but a researcher refining a design moves
back and forth constantly, so every page stays reachable at all times.  What
the window does enforce is honesty about state: the summary strip and the
validation badge always describe the model as it is right now, and the Run
button is disabled while the specification contains errors.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from semsim.core import examples as example_models
from semsim.core.errors import SemSimError
from semsim.core.spec import ModelSpec
from semsim.io import project as project_io
from semsim.ui import theme
from semsim.ui.i18n import ARABIC, ENGLISH, LANGUAGE_NAMES, translator
from semsim.ui.state import AppState, RecentProjects
from semsim.ui.widgets.common import (
    StatBadge, TButton, VerdictChip, separator, set_current_theme,
)

#: (page module, class name, title key, subtitle key)
PAGE_SPECS: List[Tuple[str, str, str, str]] = [
    ("semsim.ui.pages.measurement", "MeasurementPage", "stage.1.title", "stage.1.subtitle"),
    ("semsim.ui.pages.complexity", "ComplexityPage", "stage.2.title", "stage.2.subtitle"),
    ("semsim.ui.pages.distribution", "DistributionPage", "stage.3.title", "stage.3.subtitle"),
    ("semsim.ui.pages.power", "PowerPage", "stage.4.title", "stage.4.subtitle"),
    ("semsim.ui.pages.targeting", "TargetingPage", "stage.5.title", "stage.5.subtitle"),
    ("semsim.ui.pages.results", "ResultsPage", "stage.6.title", "stage.6.subtitle"),
]


# --------------------------------------------------------------------------- #
# Background work
# --------------------------------------------------------------------------- #


class SimulationWorker(QObject):
    """Runs the pipeline off the GUI thread.

    A full calibration run can take tens of seconds; blocking the event loop
    for that long makes the window look crashed, and the user has no way to
    stop it.
    """

    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, spec: ModelSpec, mode: str = "full") -> None:
        super().__init__()
        self._spec = spec
        self._mode = mode
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _check(self) -> None:
        if self._cancelled:
            raise SemSimError("Cancelled by the user.", "أُلغيت العملية.")

    def run(self) -> None:
        try:
            result = self._run_pipeline()
            if not self._cancelled:
                self.finished.emit(result)
        except SemSimError as exc:
            self.failed.emit(exc.message_ar, exc.message_en)
        except Exception as exc:  # pragma: no cover - defensive
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail, detail)

    def _run_pipeline(self) -> Dict[str, object]:
        """Generate, estimate and report.  Each stage is optional so the app
        still works while a module is unavailable."""
        out: Dict[str, object] = {}

        def step(key: str, fraction: float) -> None:
            self._check()
            self.progress.emit(key, fraction)

        step("status.compiling", 0.05)
        from semsim.core.compile import compile_model
        compiled = compile_model(self._spec)
        out["compiled"] = compiled

        if self._mode in ("full", "power"):
            step("status.ready", 0.12)
            try:
                from semsim.power import analysis as power_analysis
                out["power"] = power_analysis.analyse(self._spec, compiled)
            except Exception:
                out["power"] = None
            if self._mode == "power":
                return out

        step("status.generating", 0.25)
        if self._mode == "full" and getattr(self._spec.fit_target, "enabled", False):
            try:
                from semsim.targeting.calibration import calibrate
                calib = calibrate(self._spec, progress=lambda m, f: self._emit(m, f))
                out["calibration"] = calib
                out["dataset"] = calib.dataset
                out["estimation"] = calib.estimation
                out["fit"] = calib.fit
            except ImportError:
                pass
            except Exception as exc:
                out["calibration_error"] = str(exc)

        if "dataset" not in out:
            from semsim.generation.engine import generate_dataset
            out["dataset"] = generate_dataset(
                self._spec, progress=lambda m, f: self._emit(m, f))

        dataset = out["dataset"]
        step("status.estimating", 0.70)
        if "estimation" not in out:
            try:
                from semsim.analysis.estimator import estimate
                est = estimate(dataset.data, self._spec,
                               estimator=self._spec.analysis.estimator)
                out["estimation"] = est
                from semsim.analysis.fit_indices import compute as compute_fit
                out["fit"] = compute_fit(est.sample_cov, est.sigma_hat,
                                         est.n, est.n_free)
            except Exception as exc:
                out["estimation_error"] = str(exc)

        step("status.reporting", 0.88)
        try:
            from semsim.analysis.reliability import analyse as analyse_validity
            out["validity"] = analyse_validity(
                dataset.data, self._spec, out.get("estimation"),
                htmt_threshold=self._spec.analysis.htmt_threshold)
        except Exception:
            out["validity"] = None
        try:
            from semsim.analysis.descriptives import describe
            out["descriptives"] = describe(dataset.data, self._spec)
        except Exception:
            out["descriptives"] = None

        step("status.done", 1.0)
        return out

    def _emit(self, message: str, fraction: float) -> None:
        self._check()
        self.progress.emit(message, fraction)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):

    def __init__(self, spec: Optional[ModelSpec] = None,
                 theme_name: str = theme.LIGHT) -> None:
        super().__init__()
        self.state = AppState(spec if spec is not None
                              else example_models.simple_mediation())
        self.recent = RecentProjects()
        self._theme = theme_name
        self._thread: Optional[QThread] = None
        self._worker: Optional[SimulationWorker] = None
        self._pages: List[QWidget] = []
        self._page_errors: Dict[int, str] = {}

        self.setMinimumSize(1120, 740)
        self.setWindowIcon(application_icon())
        self._build_ui()
        self._build_menu()
        self._connect()
        translator.language_changed.connect(self._retranslate)
        self._retranslate()
        self._refresh_summary()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.nav = QListWidget()
        self.nav.setFixedWidth(260)
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setSpacing(2)
        splitter.addWidget(self.nav)

        self.stack = QStackedWidget()
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        outer.addWidget(separator())
        outer.addWidget(self._build_summary_strip())
        self.setCentralWidget(central)

        self._load_pages()

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.status_label = QLabel()
        bar = self.statusBar()
        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.progress_bar)

    def _build_summary_strip(self) -> QWidget:
        strip = QWidget()
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(6)

        self.badges: Dict[str, StatBadge] = {}
        for key, label in (("constructs", "m.constructs"),
                           ("items", "m.items"),
                           ("paths", "s.paths"),
                           ("params", "p.free_params"),
                           ("df", "p.df"),
                           ("n", "p.n")):
            badge = StatBadge(label, "-")
            self.badges[key] = badge
            lay.addWidget(badge)

        lay.addStretch(1)
        self.validation_chip = VerdictChip("pass")
        self.validation_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.validation_chip)

        self.cancel_button = TButton("menu.run.cancel", role="danger")
        self.cancel_button.setVisible(False)
        lay.addWidget(self.cancel_button)

        self.run_button = TButton("act.run", role="primary")
        self.run_button.setMinimumWidth(170)
        lay.addWidget(self.run_button)
        return strip

    def _load_pages(self) -> None:
        """Import each page, substituting a placeholder for any that is not
        ready yet, so a partial build still runs."""
        import importlib
        for index, (module_name, class_name, title_key, _sub) in enumerate(PAGE_SPECS):
            widget: QWidget
            try:
                module = importlib.import_module(module_name)
                widget = getattr(module, class_name)(self.state)
            except Exception as exc:
                self._page_errors[index] = (
                    f"{type(exc).__name__}: {exc}")
                widget = _PlaceholderPage(title_key, self._page_errors[index])
            self._pages.append(widget)
            self.stack.addWidget(widget)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.nav.addItem(item)
        self.nav.setCurrentRow(0)

    def _build_menu(self) -> None:
        self._actions: Dict[str, QAction] = {}

        def action(key: str, slot, shortcut: Optional[QKeySequence] = None,
                   checkable: bool = False) -> QAction:
            act = QAction(self)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.setCheckable(checkable)
            act.triggered.connect(slot)
            self._actions[key] = act
            return act

        bar = self.menuBar()
        self._menus: Dict[str, object] = {}

        file_menu = bar.addMenu("")
        self._menus["menu.file"] = file_menu
        file_menu.addAction(action("menu.file.new", self.new_project,
                                   QKeySequence.StandardKey.New))
        file_menu.addAction(action("menu.file.open", self.open_project,
                                   QKeySequence.StandardKey.Open))
        self.recent_menu = file_menu.addMenu("")
        self._menus["menu.file.recent"] = self.recent_menu
        file_menu.addSeparator()
        file_menu.addAction(action("menu.file.save", self.save_project,
                                   QKeySequence.StandardKey.Save))
        file_menu.addAction(action("menu.file.save_as", self.save_project_as,
                                   QKeySequence.StandardKey.SaveAs))
        file_menu.addSeparator()
        file_menu.addAction(action("menu.file.export", self.export_results))
        file_menu.addSeparator()
        file_menu.addAction(action("menu.file.exit", self.close,
                                   QKeySequence.StandardKey.Quit))

        model_menu = bar.addMenu("")
        self._menus["menu.model"] = model_menu
        model_menu.addAction(action("menu.model.validate", self.show_validation))
        self.example_menu = model_menu.addMenu("")
        self._menus["menu.model.example"] = self.example_menu
        self._example_actions: List[Tuple[QAction, str]] = []
        for name, (label_key, _factory) in example_models.EXAMPLES.items():
            act = QAction(self)
            act.triggered.connect(lambda _c=False, n=name: self.load_example(n))
            self.example_menu.addAction(act)
            self._example_actions.append((act, label_key))
        model_menu.addSeparator()
        model_menu.addAction(action("menu.model.clear", self.clear_model))

        run_menu = bar.addMenu("")
        self._menus["menu.run"] = run_menu
        run_menu.addAction(action("menu.run.full", lambda: self.run("full"),
                                  QKeySequence("F5")))
        run_menu.addAction(action("menu.run.simulate", lambda: self.run("generate")))
        run_menu.addAction(action("menu.run.estimate", lambda: self.run("estimate")))
        run_menu.addSeparator()
        run_menu.addAction(action("menu.run.cancel", self.cancel_run,
                                  QKeySequence("Esc")))

        view_menu = bar.addMenu("")
        self._menus["menu.view"] = view_menu
        lang_menu = view_menu.addMenu("")
        self._menus["menu.view.language"] = lang_menu
        self._lang_actions: Dict[str, QAction] = {}
        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)
        for code in (ARABIC, ENGLISH):
            act = QAction(LANGUAGE_NAMES[code], self)
            act.setCheckable(True)
            act.setChecked(code == translator.language)
            act.triggered.connect(lambda _c=False, c=code: self.set_language(c))
            lang_group.addAction(act)
            lang_menu.addAction(act)
            self._lang_actions[code] = act

        theme_menu = view_menu.addMenu("")
        self._menus["menu.view.theme"] = theme_menu
        self._theme_actions: Dict[str, QAction] = {}
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for name, key in ((theme.LIGHT, "menu.view.theme.light"),
                          (theme.DARK, "menu.view.theme.dark")):
            act = QAction(self)
            act.setCheckable(True)
            act.setChecked(name == self._theme)
            act.triggered.connect(lambda _c=False, n=name: self.set_theme(n))
            theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[name] = act
            self._actions[key] = act

        help_menu = bar.addMenu("")
        self._menus["menu.help"] = help_menu
        help_menu.addAction(action("menu.help.about", self.show_about))

    def _connect(self) -> None:
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.run_button.clicked.connect(lambda: self.run("full"))
        self.cancel_button.clicked.connect(self.cancel_run)
        self.validation_chip.mousePressEvent = (  # type: ignore[assignment]
            lambda _e: self.show_validation())
        self.state.spec_changed.connect(self._refresh_summary)
        self.state.structure_changed.connect(self._refresh_summary)
        self.state.project_changed.connect(self._refresh_title)
        self.state.results_changed.connect(self._refresh_summary)

    # ------------------------------------------------------------------ #
    # Live summary
    # ------------------------------------------------------------------ #

    def _refresh_summary(self) -> None:
        spec = self.state.spec
        try:
            df = spec.degrees_of_freedom()
            params = spec.count_free_parameters()
        except Exception:
            df, params = 0, 0
        self.badges["constructs"].set_value(spec.n_constructs)
        self.badges["items"].set_value(spec.n_items)
        self.badges["paths"].set_value(len(spec.paths) + len(spec.interactions))
        self.badges["params"].set_value(params)
        self.badges["df"].set_value(df, kind="fail" if df <= 0 else "")
        self.badges["n"].set_value(spec.power.n)

        report = self.state.validation
        if report is None:
            self.validation_chip.set_verdict("borderline", translator.t("v.title"))
        elif report.ok:
            n_warn = len(report.warnings())
            if n_warn:
                self.validation_chip.set_verdict(
                    "borderline", f"{translator.t('v.warnings')}: {n_warn}")
            else:
                self.validation_chip.set_verdict("pass", translator.t("v.no_issues"))
        else:
            self.validation_chip.set_verdict(
                "fail", f"{translator.t('v.errors')}: {len(report.errors())}")
        self.run_button.setEnabled(self.state.can_run and not self.state.busy)

    def _refresh_title(self) -> None:
        name = self.state.display_name or translator.t("menu.file.new")
        mark = "*" if self.state.dirty else ""
        self.setWindowTitle(f"{mark}{name} - {translator.t('app.title')}")

    # ------------------------------------------------------------------ #
    # Translation
    # ------------------------------------------------------------------ #

    def _retranslate(self, *_a) -> None:
        rtl = translator.is_rtl
        app = QApplication.instance()
        if app is not None:
            app.setLayoutDirection(Qt.LayoutDirection.RightToLeft if rtl
                                   else Qt.LayoutDirection.LeftToRight)
        for index, (_m, _c, title_key, sub_key) in enumerate(PAGE_SPECS):
            item = self.nav.item(index)
            if item is not None:
                item.setText(f"{index + 1}.  {translator.t(title_key)}")
                item.setToolTip(translator.t(sub_key))
        for key, act in self._actions.items():
            act.setText(translator.t(key))
        for key, menu in self._menus.items():
            menu.setTitle(translator.t(key))
        for act, label_key in self._example_actions:
            act.setText(translator.t(label_key))
        self._refresh_title()
        self._refresh_summary()
        self._refresh_recent_menu()
        self.status_label.setText(translator.t("status.ready"))

    def set_language(self, code: str) -> None:
        translator.set_language(code)
        act = self._lang_actions.get(code)
        if act is not None:
            act.setChecked(True)

    def set_theme(self, name: str) -> None:
        self._theme = name
        set_current_theme(name)
        app = QApplication.instance()
        if app is not None:
            theme.apply(app, name, rtl=translator.is_rtl)
        for page in self._pages:
            setter = getattr(page, "set_theme", None)
            if callable(setter):
                setter(name)
        act = self._theme_actions.get(name)
        if act is not None:
            act.setChecked(True)

    # ------------------------------------------------------------------ #
    # Project commands
    # ------------------------------------------------------------------ #

    def _confirm_discard(self) -> bool:
        if not self.state.dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle(translator.t("dlg.unsaved.title"))
        box.setText(translator.t("dlg.unsaved.body"))
        box.setStandardButtons(QMessageBox.StandardButton.Save
                               | QMessageBox.StandardButton.Discard
                               | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            return self.save_project()
        return choice == QMessageBox.StandardButton.Discard

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.state.new_project()
        self._rebuild_pages()

    def load_example(self, name: str) -> None:
        if not self._confirm_discard():
            return
        self.state.set_spec(example_models.build(name))
        self._rebuild_pages()

    def clear_model(self) -> None:
        answer = QMessageBox.question(
            self, translator.t("menu.model.clear"),
            translator.t("dlg.confirm_clear"))
        if answer == QMessageBox.StandardButton.Yes:
            self.state.new_project()
            self._rebuild_pages()

    def open_project(self, path: str = "") -> None:
        if not self._confirm_discard():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, translator.t("menu.file.open"), "",
                f"SEM Simulation (*{project_io.PROJECT_SUFFIX});;All files (*)")
        if not path:
            return
        try:
            warnings = self.state.load(path)
        except SemSimError as exc:
            self._error_box(exc)
            self.recent.remove(path)
            return
        self.recent.add(path)
        self._refresh_recent_menu()
        self._rebuild_pages()
        if warnings:
            QMessageBox.information(self, translator.t("menu.file.open"),
                                    "\n".join(warnings[:6]))

    def save_project(self) -> bool:
        if not self.state.path:
            return self.save_project_as()
        try:
            self.state.save()
        except SemSimError as exc:
            self._error_box(exc)
            return False
        self.recent.add(self.state.path or "")
        self._refresh_recent_menu()
        return True

    def save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, translator.t("menu.file.save_as"), "",
            f"SEM Simulation (*{project_io.PROJECT_SUFFIX})")
        if not path:
            return False
        try:
            saved = self.state.save(path)
        except SemSimError as exc:
            self._error_box(exc)
            return False
        self.recent.add(saved)
        self._refresh_recent_menu()
        return True

    def _refresh_recent_menu(self) -> None:
        menu = getattr(self, "recent_menu", None)
        if menu is None:
            return
        menu.clear()
        entries = self.recent.describe()
        if not entries:
            act = QAction(translator.t("v.no_issues"), self)
            act.setEnabled(False)
            menu.addAction(act)
            return
        for entry in entries:
            label = (entry.get("title_ar") if translator.is_rtl
                     else entry.get("title_en")) or os.path.basename(entry["path"])
            act = QAction(f"{label}   ({entry.get('n_items', 0)})", self)
            act.setToolTip(entry["path"])
            act.triggered.connect(
                lambda _c=False, p=entry["path"]: self.open_project(p))
            menu.addAction(act)

    def export_results(self) -> None:
        if not self.state.results.has_data:
            QMessageBox.information(self, translator.t("dlg.export.title"),
                                    translator.t("v.blocked"))
            return
        try:
            from semsim.ui.pages.export_dialog import ExportDialog
        except Exception:
            directory = QFileDialog.getExistingDirectory(
                self, translator.t("dlg.export.directory"))
            if not directory:
                return
            try:
                from semsim.io.exporters import export_all
                written = export_all(self.state.results.dataset, directory,
                                     self.state.spec)
                QMessageBox.information(
                    self, translator.t("dlg.export.title"),
                    f"{translator.t('dlg.export.done')}:\n" + "\n".join(written))
            except Exception as exc:
                QMessageBox.warning(self, translator.t("dlg.export.title"), str(exc))
            return
        ExportDialog(self.state, self).exec()

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #

    def run(self, mode: str = "full") -> None:
        if self.state.busy:
            return
        report = self.state.validation
        if report is not None and not report.ok:
            self.show_validation()
            return

        self.state.set_busy(True)
        self.run_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self._thread = QThread(self)
        self._worker = SimulationWorker(self.state.spec, mode)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def cancel_run(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText(translator.t("status.cancelled"))

    def _on_progress(self, key: str, fraction: float) -> None:
        text = translator.t(key) if key.startswith("status.") else key
        self.status_label.setText(text)
        if fraction >= 0:
            self.progress_bar.setValue(int(fraction * 100))

    def _teardown_thread(self) -> None:
        self.state.set_busy(False)
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None
        self._refresh_summary()

    def _on_finished(self, result: Dict[str, object]) -> None:
        self._teardown_thread()
        self.state.set_results(
            dataset=result.get("dataset"),
            estimation=result.get("estimation"),
            fit=result.get("fit"),
            validity=result.get("validity"),
            descriptives=result.get("descriptives"),
            power=result.get("power"),
            calibration=result.get("calibration"),
        )
        self.status_label.setText(translator.t("status.done"))
        soft_errors = [v for k, v in result.items() if k.endswith("_error")]
        if soft_errors:
            QMessageBox.warning(self, translator.t("status.failed"),
                                "\n".join(str(e) for e in soft_errors))
        self.nav.setCurrentRow(len(PAGE_SPECS) - 1)

    def _on_failed(self, message_ar: str, message_en: str) -> None:
        self._teardown_thread()
        self.status_label.setText(translator.t("status.failed"))
        QMessageBox.warning(self, translator.t("status.failed"),
                            message_ar if translator.is_rtl else message_en)

    # ------------------------------------------------------------------ #
    # Dialogs
    # ------------------------------------------------------------------ #

    def show_validation(self) -> None:
        try:
            from semsim.ui.pages.validation_panel import ValidationDialog
            ValidationDialog(self.state, self).exec()
            return
        except Exception:
            pass
        report = self.state.validation
        if report is None:
            return
        lines = []
        for diag in report.diagnostics[:30]:
            msg = diag.message_ar if translator.is_rtl else diag.message_en
            lines.append(f"[{diag.severity.value}] {msg}")
        QMessageBox.information(self, translator.t("v.title"),
                                "\n\n".join(lines) or translator.t("v.no_issues"))

    def show_about(self) -> None:
        QMessageBox.about(
            self, translator.t("menu.help.about"),
            f"<b>{translator.t('app.title')}</b><br><br>"
            f"{translator.t('dlg.about.body')}<br><br>"
            f"<i>{translator.t('app.tagline')}</i>")

    def _error_box(self, exc: SemSimError) -> None:
        QMessageBox.warning(
            self, translator.t("status.failed"),
            exc.message_ar if translator.is_rtl else exc.message_en)

    def _rebuild_pages(self) -> None:
        for page in self._pages:
            rebuild = getattr(page, "rebuild", None)
            if callable(rebuild):
                try:
                    rebuild()
                except Exception:
                    pass
        self._refresh_summary()
        self._refresh_title()

    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.state.busy:
            self.cancel_run()
        if self._confirm_discard():
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(2000)
            event.accept()
        else:
            event.ignore()


class _PlaceholderPage(QWidget):
    """Shown when a stage page fails to import, with the reason."""

    def __init__(self, title_key: str, detail: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 28, 28, 28)
        title = QLabel()
        theme.set_role(title, "h1")
        lay.addWidget(title)
        body = QLabel(detail)
        theme.set_role(body, "error")
        body.setWordWrap(True)
        theme.keep_ltr(body)
        lay.addWidget(body)
        lay.addStretch(1)
        translator.language_changed.connect(
            lambda _l: title.setText(translator.t(title_key)))
        title.setText(translator.t(title_key))


def application_icon() -> "QIcon":
    """The window and taskbar icon, loaded from the bundled resource.

    Falls back on an empty icon rather than raising: a missing icon is a
    cosmetic problem, not a reason the application cannot start. The path is
    resolved relative to this module so it works both from a source checkout
    and from inside a PyInstaller one-file bundle, where ``__file__`` points
    into the extracted temporary tree.
    """
    from PySide6.QtGui import QIcon
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("icon.png", "semsim.ico"):
        candidate = os.path.join(here, "resources", name)
        if os.path.exists(candidate):
            return QIcon(candidate)
    return QIcon()


def create_app(argv: Optional[List[str]] = None) -> Tuple[QApplication, MainWindow]:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("SEM Data Simulation Engine")
    app.setOrganizationName("semsim")
    app.setWindowIcon(application_icon())
    set_current_theme(theme.LIGHT)
    theme.apply(app, theme.LIGHT, rtl=translator.is_rtl)
    window = MainWindow()
    return app, window


__all__ = ["MainWindow", "SimulationWorker", "create_app",
           "application_icon", "PAGE_SPECS"]

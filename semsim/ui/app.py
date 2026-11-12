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

import copy
import os
import sys
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from semsim.core import cancel as run_cancel
from semsim.core import examples as example_models
from semsim.core.errors import SemSimError
from semsim.core.spec import ModelSpec
from semsim.io import ai_template
from semsim.io import project as project_io
from semsim.ui import theme
from semsim.ui.i18n import ARABIC, ENGLISH, LANGUAGE_NAMES, translator
from semsim.ui.state import AppState, RecentProjects, Results
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
    cancelled = Signal()
    #: Emitted last, whatever the outcome, so the thread can be shut down
    #: even after the window has stopped listening to this worker.
    done = Signal()

    def __init__(self, spec: ModelSpec, mode: str = "full",
                 prior: Optional[Results] = None) -> None:
        super().__init__()
        self._spec = spec
        self._mode = mode
        #: The previous run's results, which an "estimate" run refits.
        self._prior = prior
        #: Polled by the estimator, the calibration loop and the generator
        #: (see :mod:`semsim.core.cancel`), so Stop lands within one
        #: objective evaluation instead of at the end of a calibration round.
        self._token = threading.Event()

    def cancel(self) -> None:
        self._token.set()

    @property
    def is_cancelled(self) -> bool:
        return self._token.is_set()

    def _check(self) -> None:
        if self._token.is_set():
            raise run_cancel.Cancelled()

    def run(self) -> None:
        run_cancel.install(self._token)
        try:
            result = self._run_pipeline()
            # Cancelled after the last checkpoint: it still has to end as a
            # cancellation, not hand back results the user said to drop.
            self._check()
            self.finished.emit(result)
        except run_cancel.Cancelled:
            self.cancelled.emit()
        except SemSimError as exc:
            if self._token.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(exc.message_ar, exc.message_en)
        except Exception as exc:  # pragma: no cover - defensive
            if self._token.is_set():
                self.cancelled.emit()
            else:
                detail = "".join(
                    traceback.format_exception_only(type(exc), exc)).strip()
                self.failed.emit(detail, detail)
        finally:
            run_cancel.install(None)
            self.done.emit()

    def _sub(self, lo: float, hi: float):
        """Map a stage's own 0..1 progress into its slice of the whole run,
        so the bar never runs backwards when a stage starts counting again."""
        def emit(message: str, fraction: float) -> None:
            if message == "status.done":
                # The stage is done; the run is not.
                return self._check()
            f = min(max(float(fraction), 0.0), 1.0)
            self._emit(message, lo + (hi - lo) * f)
        return emit

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
            step("status.power", 0.06)
            try:
                from semsim.power import analysis as power_analysis
                out["power"] = power_analysis.analyse(
                    self._spec, compiled,
                    progress=self._sub(0.06, 0.20 if self._mode == "full"
                                       else 0.95))
            except Exception:
                out["power"] = None
            if self._mode == "power":
                return out

        if self._mode == "estimate":
            # Refit the data on hand.  How that data came about has not
            # changed, so its power analysis and calibration still stand.
            out["dataset"] = self._prior.dataset
            out["power"] = self._prior.power
            out["calibration"] = self._prior.calibration
        else:
            step("status.generating", 0.20)

        if self._mode == "full" and getattr(self._spec.fit_target, "enabled", False):
            try:
                from semsim.targeting.calibration import calibrate
                calib = calibrate(self._spec, progress=self._sub(0.20, 0.85))
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
                self._spec, progress=self._sub(0.20, 0.35))

        dataset = out["dataset"]
        step("status.estimating", 0.85 if "estimation" in out else 0.35)
        if "estimation" not in out:
            try:
                from semsim.analysis.estimator import estimate
                est = estimate(dataset.model_frame, self._spec,
                               estimator=self._spec.analysis.estimator)
                out["estimation"] = est
                from semsim.analysis.fit_indices import compute as compute_fit
                out["fit"] = compute_fit(est.sample_cov, est.sigma_hat,
                                         est.n, est.n_free)
            except Exception as exc:
                out["estimation_error"] = str(exc)

        step("status.reporting", 0.92)
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
        try:
            from semsim.analysis.demographics import analyse as analyse_groups
            out["demographics"] = analyse_groups(self._spec, dataset)
        except Exception as exc:
            out["demographics"] = None
            out["demographics_error"] = str(exc)

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
        #: Runs the user stopped that are still unwinding in the background:
        #: kept referenced until their thread ends, or Qt would destroy a
        #: running thread.
        self._orphans: List[Tuple[QThread, SimulationWorker]] = []
        self._run_started = 0.0
        self._progress_value = 0.0
        self._eta_total: Optional[float] = None
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._refresh_eta)
        self._run_mode = "full"
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
        self.nav.setObjectName("stageNav")
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
        self.eta_label = QLabel()
        self.eta_label.setVisible(False)
        bar = self.statusBar()
        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.eta_label)
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
        ai_menu = file_menu.addMenu("")
        self._menus["menu.file.ai"] = ai_menu
        ai_menu.addAction(action("menu.file.ai.save", self.save_ai_template))
        ai_menu.addAction(action("menu.file.ai.copy", self.copy_ai_template))
        ai_menu.addSeparator()
        ai_menu.addAction(action("menu.file.ai.import_file", self.import_ai_file))
        ai_menu.addAction(action("menu.file.ai.import_paste", self.import_ai_paste))
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
        self.state.busy_changed.connect(self._sync_busy)
        self._sync_busy(self.state.busy)

    def _sync_busy(self, busy: bool) -> None:
        """Only offer what can actually happen right now.

        Stop does nothing while idle and a second run is refused during the
        first, so both would otherwise be commands that ignore the user.
        """
        self._actions["menu.run.cancel"].setEnabled(busy)
        for key in ("menu.run.full", "menu.run.simulate", "menu.run.estimate"):
            self._actions[key].setEnabled(not busy)

    def _flash_status(self, key: str, **fmt: object) -> None:
        self.statusBar().showMessage(translator.t(key, **fmt), 6000)

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
        self.run_button.setToolTip("" if self.state.can_run
                                   else translator.t("v.blocked"))

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
        self.nav.setCurrentRow(0)

    def load_example(self, name: str) -> None:
        if not self._confirm_discard():
            return
        self.state.set_spec(example_models.build(name))
        self._rebuild_pages()
        self.nav.setCurrentRow(0)
        self._flash_status("status.loaded")

    def clear_model(self) -> None:
        answer = QMessageBox.question(
            self, translator.t("menu.model.clear"),
            translator.t("dlg.confirm_clear"))
        if answer == QMessageBox.StandardButton.Yes:
            self.state.new_project()
            self._rebuild_pages()
            self.nav.setCurrentRow(0)
            self._flash_status("status.cleared")

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
        self.nav.setCurrentRow(0)
        self._flash_status("status.loaded")
        if warnings:
            QMessageBox.information(self, translator.t("menu.file.open"),
                                    "\n".join(warnings[:6]))

    def save_ai_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, translator.t("menu.file.ai.save"), "semsim_ai_template.json",
            "JSON (*.json)")
        if not path:
            return
        try:
            saved = ai_template.save_template(path)
        except SemSimError as exc:
            self._error_box(exc)
            return
        QMessageBox.information(self, translator.t("menu.file.ai"),
                                translator.t("ai.saved_help", path=saved))

    def copy_ai_template(self) -> None:
        QApplication.clipboard().setText(ai_template.template_text())
        self._flash_status("status.ai_copied")

    def import_ai_file(self, path: str = "") -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, translator.t("menu.file.ai.import_file"), "",
                "JSON (*.json);;All files (*)")
        if path:
            self._import_ai(lambda: ai_template.load_file(path))

    def import_ai_paste(self) -> None:
        text = QApplication.clipboard().text()
        if not text.strip():
            QMessageBox.information(self, translator.t("menu.file.ai"),
                                    translator.t("ai.clipboard_empty"))
            return
        self._import_ai(lambda: ai_template.from_text(text))

    def _import_ai(self, read) -> None:
        # Read first: a rejected template must not cost the user their project.
        try:
            result = read()
        except ai_template.AiTemplateError as exc:
            self._show_ai_problems("ai.problems_title", "ai.problems_intro",
                                   exc.problems)
            return
        except SemSimError as exc:
            self._error_box(exc)
            return
        if not self._confirm_discard():
            return
        self.state.set_spec(result.spec, dirty=True)
        self._rebuild_pages()
        self.nav.setCurrentRow(0)
        self._flash_status("status.ai_imported")
        if result.warnings:
            self._show_ai_problems("ai.warnings_title", "ai.warnings_intro",
                                   result.warnings)

    def _show_ai_problems(self, title_key: str, intro_key: str, problems) -> None:
        from semsim.ui.widgets.problems_dialog import ProblemsDialog
        lang = ARABIC if translator.is_rtl else ENGLISH
        ProblemsDialog(translator.t(title_key), translator.t(intro_key),
                       "\n".join(p.text(lang) for p in problems),
                       ai_template.problems_for_ai(problems), self).exec()

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
        self._flash_status("status.saved", name=self.state.display_name)
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
        self._flash_status("status.saved", name=self.state.display_name)
        return True

    def _refresh_recent_menu(self) -> None:
        menu = getattr(self, "recent_menu", None)
        if menu is None:
            return
        menu.clear()
        entries = self.recent.describe()
        if not entries:
            act = QAction(translator.t("menu.file.recent.empty"), self)
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
                                    translator.t("dlg.export.no_data"))
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
        if mode == "estimate":
            blocker = self._estimation_blocker()
            if blocker:
                QMessageBox.information(
                    self, translator.t("menu.run.estimate"), blocker)
                return

        self._run_mode = mode
        self.state.set_busy(True)
        self.run_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._run_started = time.monotonic()
        self._progress_value = 0.0
        self._eta_total = None
        self.eta_label.setVisible(True)
        self._refresh_eta()
        self._eta_timer.start()

        thread = QThread(self)
        worker = SimulationWorker(self.state.spec, mode,
                                  copy.copy(self.state.results))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Every slot checks that the signal comes from the current run: a
        # stopped run may still be unwinding and must not touch the window.
        worker.progress.connect(
            lambda k, f, w=worker: w is self._worker and self._on_progress(k, f))
        worker.finished.connect(
            lambda r, w=worker: w is self._worker and self._on_finished(r))
        worker.failed.connect(
            lambda a, e, w=worker: w is self._worker and self._on_failed(a, e))
        worker.cancelled.connect(
            lambda w=worker: w is self._worker and self._on_cancelled())
        worker.done.connect(thread.quit)
        thread.finished.connect(lambda t=thread: self._release_thread(t))
        self._thread, self._worker = thread, worker
        thread.start()

    def _estimation_blocker(self) -> str:
        """Why the data on hand cannot be refitted; empty when it can."""
        dataset = self.state.results.dataset
        if dataset is None:
            return translator.t("run.estimate.no_data")
        columns = set(getattr(dataset, "full_frame", dataset.data).columns)
        missing = [item.name for item in self.state.spec.all_items()
                   if item.name not in columns and item.id not in columns]
        missing += [name for name in self.state.spec.control_columns()
                    if name not in columns]
        if missing:
            return translator.t("run.estimate.stale", names=", ".join(missing))
        return ""

    def cancel_run(self) -> None:
        """Stop the run now.

        The worker is told to stop and polls for it inside the estimator, so
        it unwinds within a fraction of a second -- but the window does not
        wait for that.  It lets go of the run immediately, so the controls
        come back at once and a new run can start while the old one finishes
        unwinding in the background.
        """
        worker, thread = self._worker, self._thread
        if worker is None:
            return
        worker.cancel()
        if thread is not None and thread.isRunning():
            self._orphans.append((thread, worker))
        self._thread = None
        self._worker = None
        self._teardown_thread()
        self.status_label.setText(translator.t("status.cancelled"))

    def _release_thread(self, thread: QThread) -> None:
        """A thread has ended: drop it, whether current or stopped earlier."""
        before = len(self._orphans)
        self._orphans = [(t, w) for t, w in self._orphans if t is not thread]
        if len(self._orphans) != before:
            thread.deleteLater()

    def _on_progress(self, key: str, fraction: float) -> None:
        text = translator.t(key) if key.startswith("status.") else key
        self.status_label.setText(text)
        if fraction < 0:
            return
        # Never run backwards: a stage restarting its own count is not the
        # run losing ground, and a bar that jumps back ruins the estimate.
        fraction = max(self._progress_value, min(float(fraction), 1.0))
        self._progress_value = fraction
        self.progress_bar.setValue(int(fraction * 100))
        elapsed = time.monotonic() - self._run_started
        if fraction >= 0.03 and elapsed >= 1.0:
            estimate = elapsed / fraction
            # Smooth the jumps between stages of very different cost.
            if self._eta_total is not None:
                estimate = 0.6 * estimate + 0.4 * self._eta_total
            self._eta_total = estimate
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        """Elapsed time, and how long is probably left, ticking every second."""
        if not self.state.busy:
            return
        elapsed = time.monotonic() - self._run_started
        parts = [translator.t("status.elapsed", t=_clock(elapsed))]
        if self._eta_total is None:
            parts.append(translator.t("status.eta_pending"))
        else:
            remaining = self._eta_total - elapsed
            if remaining <= 1.0:
                parts.append(translator.t("status.eta_soon"))
            else:
                parts.append(translator.t("status.eta", t=_clock(remaining)))
        self.eta_label.setText("  ·  ".join(parts))

    def _teardown_thread(self) -> None:
        self.state.set_busy(False)
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)
        self._eta_timer.stop()
        self.eta_label.setVisible(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread.deleteLater()
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
            demographics=result.get("demographics"),
        )
        self.status_label.setText(translator.t(
            "status.estimated" if self._run_mode == "estimate" else "status.done"))
        soft_errors = [v for k, v in result.items() if k.endswith("_error")]
        if soft_errors:
            QMessageBox.warning(self, translator.t("status.failed"),
                                "\n".join(str(e) for e in soft_errors))
        self.nav.setCurrentRow(len(PAGE_SPECS) - 1)

    def _on_cancelled(self) -> None:
        self._teardown_thread()
        self.status_label.setText(translator.t("status.cancelled"))

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
        except ImportError:
            pass
        else:
            ValidationDialog(self.state, self).exec()
            return
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
                    # One page failing must not stop the others rebuilding,
                    # but the failure still has to reach the user.
                    sys.excepthook(*sys.exc_info())
        self._refresh_summary()
        self._refresh_title()

    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.state.busy:
            self.cancel_run()
        if self._confirm_discard():
            for thread, _worker in list(self._orphans):
                thread.quit()
                thread.wait(2000)
            event.accept()
        else:
            event.ignore()


def _clock(seconds: float) -> str:
    """``m:ss``, or ``h:mm:ss`` past an hour."""
    total = max(int(round(seconds)), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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


def install_exception_hook() -> None:
    """Show an error raised inside a button handler instead of losing it.

    Qt carries on after a Python exception in a slot, and a windowed build has
    no console for the traceback, so the only symptom would be a button that
    appears to do nothing.
    """
    showing = [False]

    def hook(kind, value, tb) -> None:
        if not issubclass(kind, Exception):
            sys.__excepthook__(kind, value, tb)
            return
        detail = "".join(traceback.format_exception(kind, value, tb))
        if sys.stderr is not None:
            sys.stderr.write(detail)
        # A signal delivered to a widget that has already been deleted says
        # nothing about the action the user took.
        if isinstance(value, RuntimeError) and "already deleted" in str(value):
            return
        app = QApplication.instance()
        if showing[0] or app is None or QThread.currentThread() != app.thread():
            return
        showing[0] = True
        try:
            box = QMessageBox(app.activeWindow())
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(translator.t("status.failed"))
            box.setText(translator.t("err.unexpected"))
            box.setInformativeText(f"{kind.__name__}: {value}")
            box.setDetailedText(detail)
            box.exec()
        finally:
            showing[0] = False

    sys.excepthook = hook


def create_app(argv: Optional[List[str]] = None) -> Tuple[QApplication, MainWindow]:
    install_exception_hook()
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("SEM Data Simulation Engine")
    app.setOrganizationName("semsim")
    app.setWindowIcon(application_icon())
    set_current_theme(theme.LIGHT)
    theme.apply(app, theme.LIGHT, rtl=translator.is_rtl)
    window = MainWindow()
    return app, window


__all__ = ["MainWindow", "SimulationWorker", "create_app",
           "application_icon", "install_exception_hook", "PAGE_SPECS"]

"""
semsim.ui.pages.export_dialog
=============================
The export dialog, opened from *File -> Export data…*
(:meth:`semsim.ui.app.MainWindow.export_results`).

It is a :class:`QDialog` rather than a stage page because it acts on the
*results* of a run, not on the specification -- but it keeps the same four
conventions as the stage pages:

*   **The specification is the model.**  Every switch writes straight into
    ``spec.export`` (an :class:`~semsim.core.spec.ExportSpec`) and calls
    ``state.touch()``, so the chosen formats are saved with the project and the
    next run exports the same way.  Loading values back into the widgets runs
    inside :meth:`ExportDialog.quiet` so it never echoes an edit.
*   **Live retranslation.**  Nothing user-visible is hardcoded; every label
    comes from the catalogue through ``TKey`` / ``Field`` / ``Card`` /
    ``TButton`` / ``TCheckBox``, so the Arabic/English switch keeps working
    while the dialog is open.
*   **Direction discipline.**  Prose follows the interface direction; the
    destination path, the file-name stem, the decimal count and the list of
    files written are identifiers and numbers, so they are pinned
    left-to-right.
*   **Writing happens off the GUI thread.**  An ``.xlsx`` of 5,000 rows takes
    long enough to make the window look crashed, so the writers run in a
    :class:`QThread` worker behind a progress bar, exactly as
    :class:`semsim.ui.app.SimulationWorker` does.

:mod:`semsim.io.exporters` is imported lazily *inside* the worker and probed
through a guarded import here.  When that module is unavailable the dialog
degrades instead of failing: the CSV is written with pandas directly, the other
formats are disabled, and a banner says why.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QVBoxLayout, QWidget,
)

from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.widgets.common import (
    Card, Field, InfoBanner, ScrollPage, TButton, TCheckBox, TKey, int_spin,
    separator,
)

#: ``(format id written into spec.export.formats, catalogue key)``, in the
#: order the checkboxes appear and the files are written.
FORMATS: Tuple[Tuple[str, str], ...] = (
    ("csv", "dlg.export.csv"),
    ("xlsx", "dlg.export.xlsx"),
    ("sav", "dlg.export.sav"),
    ("lavaan", "dlg.export.lavaan"),
    ("mplus", "dlg.export.mplus"),
)

#: Formats that need :mod:`semsim.io.exporters`; a bare pandas fallback can
#: only write the CSV.
NEEDS_EXPORTERS = frozenset({"xlsx", "sav", "lavaan", "mplus"})

#: How many files each format produces, for the "n files will be written" hint.
#: Mplus needs two: the free-format data file and the input file that reads it.
FILES_PER_FORMAT: Dict[str, int] = {
    "csv": 1, "xlsx": 1, "sav": 1, "lavaan": 1, "mplus": 2,
}

DEFAULT_STEM = "simulation"


def safe_stem(raw: str) -> str:
    """A file-name stem R, SPSS and Mplus can all open.

    Arabic file names are perfectly legal on disk, but Mplus reads its data
    file name out of a fixed-format ``FILE IS`` line and several R toolchains
    still stumble on non-ASCII paths, so the stem is reduced to ASCII rather
    than failing later inside someone else's parser.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or DEFAULT_STEM


def exporters_available() -> bool:
    """True when :mod:`semsim.io.exporters` can be imported.

    Guarded: the IO layer may not exist yet in a partial build, and the dialog
    must still open.
    """
    try:
        import importlib

        importlib.import_module("semsim.io.exporters")
        return True
    except Exception:
        return False


class ExportFailure(Exception):
    """A bilingual failure raised by the worker itself."""

    def __init__(self, message_ar: str, message_en: str) -> None:
        super().__init__(message_en)
        self.message_ar = message_ar
        self.message_en = message_en


def failure_messages(exc: BaseException) -> Tuple[str, str]:
    """``(arabic, english)`` for anything the worker can raise.

    :class:`~semsim.core.errors.ExportError` and the rest of the ``SemSimError``
    family already carry both languages; everything else has only one message,
    and showing it in both places is more honest than inventing a translation.
    """
    arabic = getattr(exc, "message_ar", "") or ""
    english = getattr(exc, "message_en", "") or ""
    if arabic or english:
        return arabic or english, english or arabic
    detail = str(exc).strip() or type(exc).__name__
    return detail, detail


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


class _FrameShim:
    """Minimal dataset-like wrapper, so a bare frame can go through a writer."""

    __slots__ = ("data", "seed")

    def __init__(self, data: Any, seed: Any = None) -> None:
        self.data = data
        self.seed = seed


class ExportWorker(QObject):
    """Writes the chosen files off the GUI thread.

    Lives in its own :class:`QThread`.  It touches the specification read-only
    and never touches a widget, so the only traffic back to the interface is
    through these three signals.
    """

    #: ``(catalogue key or plain text, fraction 0..1)``
    progress = Signal(str, float)
    #: The list of paths actually written.
    finished = Signal(object)
    #: ``(arabic message, english message)``
    failed = Signal(str, str)

    def __init__(self, dataset: Any, spec: Any, directory: str, stem: str,
                 formats: List[str], decimal_places: int,
                 include_latent: bool, include_shadow: bool,
                 lang: str = "ar") -> None:
        super().__init__()
        self._dataset = dataset
        self._spec = spec
        self._directory = directory
        self._base = safe_stem(stem)
        self._formats = list(formats)
        self._places = max(int(decimal_places), 0)
        self._latent = bool(include_latent)
        self._shadow = bool(include_shadow)
        self._lang = lang
        self._cancelled = False
        self._written: List[str] = []
        self._done = 0
        self._total = 1

    def cancel(self) -> None:
        self._cancelled = True

    # -- lifecycle -------------------------------------------------------- #

    def run(self) -> None:
        try:
            written = self._write()
        except Exception as exc:  # every failure is reported, never raised out
            # A cancellation is not a failure: the dialog asked for it, and an
            # error box over a window the user has just closed is noise.
            if not self._cancelled:
                self.failed.emit(*failure_messages(exc))
            return
        if not self._cancelled:
            self.finished.emit(written)

    # -- helpers ---------------------------------------------------------- #

    def _check(self) -> None:
        if self._cancelled:
            raise ExportFailure("أُلغي التصدير قبل اكتماله.",
                                "The export was cancelled before it finished.")

    def _path(self, suffix: str) -> str:
        return os.path.join(self._directory, f"{self._base}{suffix}")

    def _tick(self) -> None:
        self._done += 1
        self.progress.emit("dlg.export.running",
                           min(self._done / float(self._total), 1.0))

    def _frame(self) -> Any:
        data = getattr(self._dataset, "data", None)
        if data is None:
            raise ExportFailure(
                "لا يحتوي ناتج المحاكاة على جدول بيانات، فلا شيء لتصديره؛ "
                "شغّل المحاكاة أولًا.",
                "The simulation result carries no data table, so there is "
                "nothing to export; run the simulation first.")
        return data

    def _extra_frame(self, attribute: str) -> Any:
        frame = getattr(self._dataset, attribute, None)
        if frame is None:
            return None
        try:
            if getattr(frame, "empty", False):
                return None
        except Exception:
            return None
        return frame

    # -- writing ---------------------------------------------------------- #

    def _write(self) -> List[str]:
        if not self._directory or not os.path.isdir(self._directory):
            raise ExportFailure(
                f"مجلد الحفظ غير موجود: «{self._directory}»؛ اختر مجلدًا "
                f"موجودًا ولديك صلاحية الكتابة فيه.",
                f"The destination folder does not exist: {self._directory!r}. "
                f"Choose a folder that exists and that you may write to.")

        exporters = self._load_exporters()
        formats = [f for f in self._formats
                   if exporters is not None or f not in NEEDS_EXPORTERS]
        if not formats:
            formats = ["csv"]

        latent = self._extra_frame("latent_scores") if self._latent else None
        shadow = (self._extra_frame("continuous_shadow")
                  if self._shadow else None)
        self._total = max(sum(FILES_PER_FORMAT.get(f, 1) for f in formats)
                          + int(latent is not None) + int(shadow is not None), 1)
        self._done = 0
        self.progress.emit("dlg.export.running", 0.0)

        if exporters is None:
            self._write_csv_with_pandas(self._frame(), self._path(".csv"))
            if latent is not None:
                self._write_csv_with_pandas(
                    latent, self._path("_latent_scores.csv"))
            if shadow is not None:
                self._write_csv_with_pandas(
                    shadow, self._path("_continuous.csv"))
            return self._written

        for name in formats:
            self._check()
            if name == "csv":
                target = self._path(".csv")
                exporters.export_csv(self._dataset, target,
                                     decimal_places=self._places)
                self._record(target)
            elif name == "xlsx":
                target = self._path(".xlsx")
                exporters.export_xlsx(self._dataset, target, lang=self._lang)
                self._record(target)
            elif name == "sav":
                target = self._path(".sav")
                exporters.export_sav(self._dataset, target, self._spec,
                                     lang=self._lang)
                self._record(target)
            elif name == "lavaan":
                target = self._path("_lavaan.R")
                self._write_text(
                    target, exporters.lavaan_syntax(self._spec, lang=self._lang))
                self._record(target)
            elif name == "mplus":
                data_path = self._path(".dat")
                exporters.export_mplus_dat(self._dataset, data_path,
                                           decimal_places=self._places)
                self._record(data_path)
                self._check()
                input_path = self._path(".inp")
                self._write_text(
                    input_path,
                    exporters.mplus_syntax(self._spec,
                                           os.path.basename(data_path),
                                           lang=self._lang))
                self._record(input_path)

        if latent is not None:
            self._check()
            target = self._path("_latent_scores.csv")
            exporters.export_csv(_FrameShim(latent), target,
                                 decimal_places=self._places)
            self._record(target)
        if shadow is not None:
            self._check()
            target = self._path("_continuous.csv")
            exporters.export_csv(_FrameShim(shadow), target,
                                 decimal_places=self._places)
            self._record(target)
        return self._written

    def _load_exporters(self) -> Any:
        """The IO layer, or ``None`` when this build does not ship it."""
        try:
            from semsim.io import exporters
        except Exception:
            return None
        return exporters

    def _record(self, path: str) -> None:
        self._written.append(str(path))
        self._tick()

    def _write_text(self, path: str, text: str) -> None:
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except OSError as exc:
            raise ExportFailure(
                f"تعذّرت الكتابة في الملف «{path}»: {exc.strerror or exc}.",
                f"Could not write {path!r}: {exc.strerror or exc}.") from exc

    def _write_csv_with_pandas(self, frame: Any, path: str) -> None:
        """The fallback writer used when the IO layer is unavailable."""
        try:
            import pandas as pd  # noqa: F401  (imported for the error message)
        except Exception as exc:
            raise ExportFailure(
                "مكتبة pandas غير متاحة، فلا يمكن كتابة ملف CSV.",
                "pandas is unavailable, so no CSV can be written.") from exc
        try:
            frame.to_csv(path, index=False, na_rep="",
                         float_format=f"%.{self._places}f",
                         encoding="utf-8-sig")
        except (OSError, ValueError, UnicodeError) as exc:
            raise ExportFailure(
                f"تعذّرت كتابة ملف CSV في «{path}»: {exc}.",
                f"Could not write the CSV to {path!r}: {exc}.") from exc
        self._record(path)


# --------------------------------------------------------------------------- #
# Dialog
# --------------------------------------------------------------------------- #


class ExportDialog(QDialog):
    """Choose what to write and where, then write it off the GUI thread."""

    def __init__(self, state: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._updating = False
        self._directory = ""
        self._thread: Optional[QThread] = None
        self._worker: Optional[ExportWorker] = None
        self._boxes: Dict[str, TCheckBox] = {}
        self._aborted = False
        self._have_exporters = exporters_available()

        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(540, 420)
        self.resize(720, 640)

        self.build()
        translator.language_changed.connect(self._on_language_changed)
        self._retranslate()
        self.refresh()

    # ------------------------------------------------------------------ #
    # Model access
    # ------------------------------------------------------------------ #

    @property
    def spec(self) -> Any:
        return self.state.spec

    @property
    def export_spec(self) -> Any:
        return self.spec.export

    @property
    def dataset(self) -> Any:
        results = getattr(self.state, "results", None)
        return getattr(results, "dataset", None)

    def edit(self) -> None:
        """Record a value change, exactly as a stage page would."""
        if self._updating:
            return
        try:
            self.state.touch()
        except Exception:
            pass

    @contextmanager
    def quiet(self) -> Iterator["ExportDialog"]:
        """Apply values to widgets without echoing an edit back into the spec."""
        previous = self._updating
        self._updating = True
        try:
            yield self
        finally:
            self._updating = previous

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        head_lay = QVBoxLayout(head)
        head_lay.setContentsMargins(18, 14, 18, 10)
        head_lay.setSpacing(2)
        head_lay.addWidget(TKey("dlg.export.title", role="h1"))
        outer.addWidget(head)
        outer.addWidget(separator())

        self.scroll = ScrollPage()
        outer.addWidget(self.scroll, 1)

        self._no_data = InfoBanner("dlg.export.no_data", kind="danger")
        self.scroll.add(self._no_data)
        self._limited = InfoBanner("dlg.export.limited", kind="warning")
        self._limited.setVisible(not self._have_exporters)
        self.scroll.add(self._limited)

        self.scroll.add(self._build_formats_card())
        self.scroll.add(self._build_options_card())
        self.scroll.add(self._build_destination_card())
        self.scroll.add(self._build_result_card())
        self.scroll.add_stretch()

        outer.addWidget(separator())
        outer.addWidget(self._build_footer())

    def _build_formats_card(self) -> Card:
        card = Card("dlg.export.formats")
        for name, key in FORMATS:
            box = TCheckBox(key)
            box.toggled.connect(lambda _c, n=name: self._on_formats(n))
            if name in NEEDS_EXPORTERS and not self._have_exporters:
                box.setEnabled(False)
            card.add(box)
            self._boxes[name] = box
        return card

    def _build_options_card(self) -> Card:
        card = Card("dlg.export.options")
        self._latent = TCheckBox("dlg.export.include_latent")
        self._latent.toggled.connect(self._on_latent)
        card.add(self._latent)
        self._shadow = TCheckBox("dlg.export.include_shadow")
        self._shadow.toggled.connect(self._on_shadow)
        card.add(self._shadow)
        self._decimals = int_spin(0, 10, 4)
        self._decimals.valueChanged.connect(self._on_decimals)
        card.add(Field("dlg.export.decimals", self._decimals,
                       hint_key="dlg.export.decimals.hint"))
        return card

    def _build_destination_card(self) -> Card:
        card = Card("dlg.export.directory")
        row = QHBoxLayout()
        row.setSpacing(8)
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        self._dir_edit.setPlaceholderText("")
        theme.keep_ltr(self._dir_edit)
        row.addWidget(self._dir_edit, 1)
        self._browse = TButton("act.browse")
        self._browse.clicked.connect(self._on_browse)
        row.addWidget(self._browse)
        card.add(row)

        self._stem = QLineEdit()
        self._stem.textChanged.connect(lambda _t: self._update_actions())
        # What the field shows is what the files are called: the stem is
        # reduced to the ASCII the writers can use, rather than letting the
        # researcher believe a name the files will not carry.
        self._stem.editingFinished.connect(self._normalise_stem)
        card.add(Field("dlg.export.stem", self._stem,
                       hint_key="dlg.export.stem.hint"))
        return card

    def _build_result_card(self) -> Card:
        card = Card("dlg.export.done")
        self._result_folder = QLabel()
        theme.set_role(self._result_folder, "mono")
        self._result_folder.setWordWrap(True)
        theme.keep_ltr(self._result_folder)
        card.add(self._result_folder)
        self._result_files = QLabel()
        theme.set_role(self._result_files, "mono")
        self._result_files.setWordWrap(True)
        theme.keep_ltr(self._result_files)
        card.add(self._result_files)
        open_row = QHBoxLayout()
        self._open_folder = TButton("dlg.export.open_folder", role="flat")
        self._open_folder.clicked.connect(self._on_open_folder)
        open_row.addWidget(self._open_folder)
        open_row.addStretch(1)
        card.add(open_row)
        card.setVisible(False)
        self._result = card
        return card

    def _build_footer(self) -> QWidget:
        foot = QWidget()
        lay = QVBoxLayout(foot)
        lay.setContentsMargins(18, 10, 18, 14)
        lay.setSpacing(8)

        self._hint = TKey("dlg.export.need_format", role="hint")
        lay.addWidget(self._hint)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setVisible(False)
        theme.keep_ltr(self._progress)
        lay.addWidget(self._progress)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        self._close_button = TButton("act.close")
        self._close_button.clicked.connect(self.reject)
        buttons.addWidget(self._close_button)
        self._export_button = TButton("act.export", role="primary")
        self._export_button.setDefault(True)
        self._export_button.clicked.connect(self._on_export)
        buttons.addWidget(self._export_button)
        lay.addLayout(buttons)
        return foot

    # ------------------------------------------------------------------ #
    # Loading values back
    # ------------------------------------------------------------------ #

    def refresh(self) -> None:
        """Re-read ``spec.export`` into the widgets."""
        export = self.export_spec
        chosen = {str(f).lower() for f in (getattr(export, "formats", None) or [])}
        # The two syntax switches live beside ``formats`` in the spec, so a
        # project saved before they were folded into the list still shows them.
        if getattr(export, "emit_lavaan_syntax", False):
            chosen.add("lavaan")
        if getattr(export, "emit_mplus_syntax", False):
            chosen.add("mplus")
        with self.quiet():
            for name, box in self._boxes.items():
                box.setChecked(name in chosen and box.isEnabled())
            self._latent.setChecked(
                bool(getattr(export, "include_latent_scores", False)))
            self._shadow.setChecked(
                bool(getattr(export, "include_continuous_shadow", False)))
            self._decimals.setValue(
                int(getattr(export, "decimal_places", 4) or 0))
            if not self._stem.text().strip():
                self._stem.setText(self._default_stem())
        self._update_actions()

    def _default_stem(self) -> str:
        """The saved project's own file name, or a neutral fallback.

        Naming the output after the project is what makes a folder of exports
        readable months later; an unsaved project has no name to borrow, so it
        falls back to ``"simulation"`` rather than to the model title, which
        is a sentence and would make an unwieldy file name.
        """
        name = ""
        try:
            name = str(getattr(self.state, "display_name", "") or "")
        except Exception:
            name = ""
        if name:
            name = os.path.splitext(name)[0]
        return safe_stem(name)

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    def chosen_formats(self) -> List[str]:
        """The ticked formats, in catalogue order."""
        return [name for name, _key in FORMATS
                if self._boxes[name].isChecked()]

    def _on_formats(self, _name: str) -> None:
        if self._updating:
            return
        chosen = self.chosen_formats()
        export = self.export_spec
        export.formats = list(chosen)
        export.emit_lavaan_syntax = "lavaan" in chosen
        export.emit_mplus_syntax = "mplus" in chosen
        self.edit()
        self._update_actions()

    def _on_latent(self, checked: bool) -> None:
        if self._updating:
            return
        self.export_spec.include_latent_scores = bool(checked)
        self.edit()
        self._update_actions()

    def _on_shadow(self, checked: bool) -> None:
        if self._updating:
            return
        self.export_spec.include_continuous_shadow = bool(checked)
        self.edit()
        self._update_actions()

    def _on_decimals(self, value: int) -> None:
        if self._updating:
            return
        self.export_spec.decimal_places = int(value)
        self.edit()

    def _on_browse(self) -> None:
        start = self._directory or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(
            self, translator.t("dlg.export.directory"), start)
        if not chosen:
            return
        self.set_directory(chosen)

    def set_directory(self, directory: str) -> None:
        """Set the destination without opening a file dialog (used by tests)."""
        self._directory = str(directory or "")
        self._dir_edit.setText(self._directory)
        # The box is too narrow for a deep path, so the whole one is a tooltip.
        self._dir_edit.setToolTip(self._directory)
        self._update_actions()

    def _normalise_stem(self) -> None:
        cleaned = safe_stem(self._stem.text())
        if cleaned != self._stem.text():
            self._stem.setText(cleaned)

    def _on_open_folder(self) -> None:
        if not self._directory:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._directory))

    # ------------------------------------------------------------------ #
    # Enabling and the hint that says what is missing
    # ------------------------------------------------------------------ #

    def expected_file_count(self) -> int:
        count = sum(FILES_PER_FORMAT.get(name, 1)
                    for name in self.chosen_formats())
        dataset = self.dataset
        if self._latent.isChecked() and self._has_frame(dataset, "latent_scores"):
            count += 1
        if self._shadow.isChecked() and self._has_frame(dataset,
                                                        "continuous_shadow"):
            count += 1
        return count

    @staticmethod
    def _has_frame(dataset: Any, attribute: str) -> bool:
        frame = getattr(dataset, attribute, None)
        if frame is None:
            return False
        try:
            return not bool(getattr(frame, "empty", False))
        except Exception:
            return False

    def _set_hint(self, key: str, *, error: bool = False, **fmt: Any) -> None:
        theme.set_role(self._hint, "error" if error else "hint")
        style = self._hint.style()
        if style is not None:
            style.unpolish(self._hint)
            style.polish(self._hint)
        self._hint.set_key(key, **fmt)

    def _update_actions(self) -> None:
        busy = self._thread is not None
        has_data = self.dataset is not None
        self._no_data.setVisible(not has_data)
        has_format = bool(self.chosen_formats())
        has_directory = bool(self._directory) and os.path.isdir(self._directory)

        if busy:
            self._set_hint("dlg.export.running")
        elif not has_data:
            self._set_hint("dlg.export.no_data", error=True)
        elif not has_format:
            self._set_hint("dlg.export.need_format", error=True)
        elif not has_directory:
            self._set_hint("dlg.export.need_directory", error=True)
        else:
            self._set_hint("dlg.export.ready", n=self.expected_file_count())

        self._export_button.setEnabled(
            has_data and has_format and has_directory and not busy)
        self._browse.setEnabled(not busy)
        # Formats needing the IO layer stay disabled while it is missing.
        for name, box in self._boxes.items():
            allowed = self._have_exporters or name not in NEEDS_EXPORTERS
            box.setEnabled(allowed and not busy)
        self._latent.setEnabled(not busy)
        self._shadow.setEnabled(not busy)
        self._decimals.setEnabled(not busy)
        self._stem.setEnabled(not busy)

    # ------------------------------------------------------------------ #
    # Running the export
    # ------------------------------------------------------------------ #

    def _on_export(self) -> None:
        if self._thread is not None:
            return
        dataset = self.dataset
        if dataset is None:
            self._update_actions()
            return
        formats = self.chosen_formats()
        if not formats or not self._directory:
            self._update_actions()
            return

        self._aborted = False
        self._normalise_stem()
        self._result.setVisible(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        self._worker = ExportWorker(
            dataset, self.spec, self._directory,
            self._stem.text() or self._default_stem(), formats,
            int(self._decimals.value()), self._latent.isChecked(),
            self._shadow.isChecked(), translator.language)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._update_actions()
        self._thread.start()

    def _on_progress(self, key: str, fraction: float) -> None:
        if key.startswith("dlg.") or key.startswith("status."):
            self._set_hint(key)
        if fraction >= 0:
            self._progress.setValue(int(round(fraction * 100)))

    def _teardown(self, *, cancel: bool = False) -> None:
        if cancel and self._worker is not None:
            self._aborted = True
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None
        self._progress.setVisible(False)

    def _on_finished(self, written: Any) -> None:
        self._teardown()
        paths = [str(p) for p in (written or [])]
        self._show_written(paths)
        self._update_actions()

    def _on_failed(self, message_ar: str, message_en: str) -> None:
        self._teardown()
        self._update_actions()
        if self._aborted or not self.isVisible():
            return          # the dialog is gone; there is nobody to tell
        QMessageBox.warning(
            self, translator.t("status.failed"),
            message_ar if translator.is_rtl else message_en)

    def _show_written(self, paths: List[str]) -> None:
        self._result_folder.setText(self._directory)
        self._result_files.setText("\n".join(os.path.basename(p)
                                             for p in paths))
        self._result.setVisible(bool(paths))
        self._open_folder.setEnabled(bool(self._directory))
        if paths:
            # The card sits below the destination row, so on a short window it
            # lands off-screen; a successful export must be seen, not hunted.
            # The scroll is deferred one tick: the card has only just become
            # visible, so the scroll area does not yet know how tall it is.
            QTimer.singleShot(0, self._scroll_to_result)

    def _scroll_to_result(self) -> None:
        try:
            self.scroll.ensureWidgetVisible(self._result)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Retranslation and shutdown
    # ------------------------------------------------------------------ #

    def _on_language_changed(self, _lang: str) -> None:
        try:
            self._retranslate()
            self._update_actions()
        except RuntimeError:
            pass

    def _retranslate(self) -> None:
        self.setWindowTitle(translator.t("dlg.export.title"))

    def closeEvent(self, event: QCloseEvent) -> None:
        self._teardown(cancel=True)
        super().closeEvent(event)

    def reject(self) -> None:
        self._teardown(cancel=True)
        super().reject()


__all__ = [
    "ExportDialog", "ExportWorker", "ExportFailure", "FORMATS",
    "NEEDS_EXPORTERS", "safe_stem", "exporters_available",
]

"""
semsim.ui.state
===============
The single mutable object the interface edits, plus the signals that keep the
pages, the validation panel and the summary bar in step.

The six pages all read and write one :class:`~semsim.core.spec.ModelSpec`.
Rather than have them talk to each other, each page mutates the spec and calls
:meth:`AppState.touch`; everything that displays derived information listens for
``spec_changed``.  That keeps the pages independent and makes "the degrees of
freedom updated while I was typing" work without special-casing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from semsim.core.spec import ModelSpec
from semsim.io import project as project_io


@dataclass
class Results:
    """Everything produced by a run.  All optional: a page shows what it has."""

    dataset: Any = None            # GeneratedDataset
    estimation: Any = None         # EstimationResult
    fit: Any = None                # FitIndices
    validity: Any = None           # ValidityReport
    descriptives: Any = None       # DescriptivesReport
    power: Any = None              # PowerResult
    calibration: Any = None        # CalibrationResult
    warnings: List[str] = field(default_factory=list)

    def clear(self) -> None:
        self.dataset = None
        self.estimation = None
        self.fit = None
        self.validity = None
        self.descriptives = None
        self.power = None
        self.calibration = None
        self.warnings = []

    @property
    def has_data(self) -> bool:
        return self.dataset is not None

    @property
    def has_fit(self) -> bool:
        return self.fit is not None


class AppState(QObject):
    """Owns the model, the results and the dirty flag."""

    #: The specification changed in a way that affects derived numbers.
    spec_changed = Signal()
    #: Structure changed (construct/item/path added or removed) -- pages that
    #: build widget trees rebuild on this, not on every keystroke.
    structure_changed = Signal()
    #: A run finished (or results were cleared).
    results_changed = Signal()
    #: Saved / loaded / modified -- the title bar listens.
    project_changed = Signal()
    #: Long-running work: (message key or text, fraction 0..1).
    progress = Signal(str, float)
    #: A background run ended badly.
    run_failed = Signal(str)

    #: Validation is cheap but not free; coalesce bursts of keystrokes.
    VALIDATE_DEBOUNCE_MS = 120

    def __init__(self, spec: Optional[ModelSpec] = None) -> None:
        super().__init__()
        self._spec = spec if spec is not None else ModelSpec()
        self._results = Results()
        self._path: Optional[str] = None
        self._dirty = False
        self._validation: Any = None
        self._busy = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.VALIDATE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._revalidate)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def set_spec(self, spec: ModelSpec, *, path: Optional[str] = None,
                 dirty: bool = False) -> None:
        self._spec = spec
        self._path = path
        self._dirty = dirty
        self._results.clear()
        self._validation = None
        self._revalidate()
        self.structure_changed.emit()
        self.results_changed.emit()
        self.project_changed.emit()

    def touch(self, *, structural: bool = False) -> None:
        """Record an edit.

        ``structural=True`` means a construct, dimension, item or path was added
        or removed, so pages that mirror the model in widgets must rebuild.
        """
        self._dirty = True
        self._debounce.start()
        self.spec_changed.emit()
        if structural:
            self.structure_changed.emit()
        self.project_changed.emit()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @property
    def validation(self) -> Any:
        """The latest :class:`ValidationReport`, computed lazily."""
        if self._validation is None:
            self._revalidate()
        return self._validation

    def _revalidate(self) -> None:
        try:
            from semsim.core.validation import validate
        except Exception:
            self._validation = None
            return
        try:
            self._validation = validate(self._spec)
        except Exception as exc:  # validation must never take the UI down
            self._validation = None
            self._results.warnings.append(f"validation failed: {exc}")

    @property
    def can_run(self) -> bool:
        report = self.validation
        if report is None:
            return True          # validation unavailable; let the run try
        return bool(getattr(report, "ok", True))

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #

    @property
    def results(self) -> Results:
        return self._results

    def set_results(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._results, key):
                setattr(self._results, key, value)
        self.results_changed.emit()

    def clear_results(self) -> None:
        self._results.clear()
        self.results_changed.emit()

    # ------------------------------------------------------------------ #
    # Busy state
    # ------------------------------------------------------------------ #

    @property
    def busy(self) -> bool:
        return self._busy

    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    def report_progress(self, message: str, fraction: float = -1.0) -> None:
        self.progress.emit(message, fraction)

    # ------------------------------------------------------------------ #
    # Project file
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def display_name(self) -> str:
        if self._path:
            return os.path.basename(self._path)
        return ""

    def save(self, path: Optional[str] = None) -> str:
        target = path or self._path
        if not target:
            raise ValueError("no path to save to")
        self._path = project_io.save(self._spec, target)
        target = self._path
        self._dirty = False
        self.project_changed.emit()
        return target

    def load(self, path: str) -> List[str]:
        spec = project_io.load(path)
        warnings = list(project_io.LAST_LOAD_WARNINGS)
        self.set_spec(spec, path=path, dirty=False)
        return warnings

    def new_project(self, spec: Optional[ModelSpec] = None) -> None:
        self.set_spec(spec if spec is not None else ModelSpec(),
                      path=None, dirty=False)


class RecentProjects:
    """A short most-recently-used list, persisted next to the user's settings."""

    MAX = 8

    def __init__(self, settings_path: Optional[str] = None) -> None:
        self._path = settings_path or self._default_path()
        self._items: List[str] = []
        self.load()

    @staticmethod
    def _default_path() -> str:
        base = (os.environ.get("APPDATA")
                or os.path.expanduser("~/.config"))
        return os.path.join(base, "semsim", "recent.txt")

    def load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                self._items = [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            self._items = []
        self._items = [p for p in self._items if os.path.exists(p)][:self.MAX]

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self._items))
        except OSError:
            pass          # a missing recents file is never worth an error box

    def add(self, path: str) -> None:
        path = os.path.abspath(path)
        self._items = [p for p in self._items if p != path]
        self._items.insert(0, path)
        self._items = self._items[:self.MAX]
        self.save()

    def remove(self, path: str) -> None:
        self._items = [p for p in self._items if p != os.path.abspath(path)]
        self.save()

    def items(self) -> List[str]:
        return list(self._items)

    def describe(self) -> List[dict]:
        return [project_io.describe_project(p) for p in self._items]


__all__ = ["Results", "AppState", "RecentProjects"]

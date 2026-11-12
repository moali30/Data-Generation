"""Stopping a run, the time-left estimate, and the diagram's moderator band.

The Stop button used to set a flag that was read only between calibration
rounds -- minutes apart on a real model -- and the cancellation it raised was
then caught by the calibration step's own ``except Exception`` and turned into
a soft error, so the run carried on.  These tests pin the fix down.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from semsim.core import cancel, examples  # noqa: E402
from semsim.core.spec import ConstructRole  # noqa: E402
from semsim.targeting.calibration import calibrate  # noqa: E402
from semsim.ui import theme  # noqa: E402
from semsim.ui.app import MainWindow, _clock  # noqa: E402
from semsim.ui.i18n import translator  # noqa: E402


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication(["semsim-test"])
    theme.apply(app, theme.LIGHT, rtl=True)
    return app


def _wait(application, condition, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        application.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return condition()


# -- the engine ------------------------------------------------------------- #


def test_cancelled_is_not_an_exception_a_step_can_swallow():
    # Every "except Exception" in the pipeline must let it through.
    assert not issubclass(cancel.Cancelled, Exception)


def test_a_check_without_a_token_costs_nothing():
    cancel.install(None)
    cancel.check()


def test_calibration_stops_within_a_moment():
    # A cold import of SciPy cannot be interrupted part-way; it happens once
    # per process and is not what this test is about.
    import semsim.analysis.estimator  # noqa: F401
    import semsim.targeting.cudeck_browne  # noqa: F401

    spec = examples.build("messy_reality")      # minutes of calibration
    token = threading.Event()
    timer = threading.Timer(0.4, token.set)
    cancel.install(token)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(cancel.Cancelled):
            calibrate(spec)
    finally:
        cancel.install(None)
        timer.cancel()
    assert time.monotonic() - started < 3.0


# -- the window ------------------------------------------------------------- #


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a)))
    win = MainWindow(examples.build("messy_reality"))
    win.warnings = warnings
    win.show()
    application.processEvents()
    yield win
    win.state._dirty = False
    win.close()
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    translator.language_changed.disconnect()


def test_stop_gives_the_window_back_at_once(application, window):
    window.run("full")
    assert window.state.busy
    assert window.eta_label.isVisible()
    # Let the run get into its stride before stopping it.
    assert _wait(application, lambda: window._progress_value >= 0.08, 60)

    window.cancel_button.click()
    # No waiting on the worker: the controls are back immediately.
    assert not window.state.busy
    assert not window.cancel_button.isVisible()
    assert not window.eta_label.isVisible()
    assert window.status_label.text() == translator.t("status.cancelled")

    # And the abandoned run really does stop, rather than grinding on.
    assert _wait(application, lambda: not window._orphans, 10)
    assert not window.warnings
    assert window.state.results.dataset is None


def test_the_time_left_appears_once_there_is_progress(application, window):
    window.run("full")
    try:
        window._run_started -= 10          # pretend ten seconds have passed
        window._on_progress("status.calibrating", 0.5)
        text = window.eta_label.text()
        assert translator.t("status.elapsed", t="0:10") in text
        # Half done in ten seconds: about ten seconds left.
        assert "0:1" in text.split("·")[-1] or "0:09" in text
    finally:
        window.cancel_run()
        _wait(application, lambda: not window._orphans, 10)


def test_progress_never_runs_backwards(application, window):
    window.run("full")
    try:
        window._on_progress("status.calibrating", 0.6)
        window._on_progress("status.generating", 0.3)
        assert window.progress_bar.value() == 60
    finally:
        window.cancel_run()
        _wait(application, lambda: not window._orphans, 10)


@pytest.mark.parametrize("seconds, text", [
    (0, "0:00"), (59.6, "1:00"), (125, "2:05"), (3725, "1:02:05")])
def test_clock(seconds, text):
    assert _clock(seconds) == text


# -- the diagram ------------------------------------------------------------ #


def _moderated_model():
    spec = examples.build("simple_mediation")
    spec.add_construct("MOD", label_ar="معدل", label_en="Moderator",
                       role=ConstructRole.MODERATOR, n_items=3)
    spec.add_path("MOD", "PERF", coefficient=0.10)
    spec.add_interaction("ENG", "MOD", "PERF")
    return spec


@pytest.mark.parametrize("rtl", [False, True])
def test_a_moderator_sits_off_the_flow_and_lands_on_its_path(application, rtl):
    from semsim.ui.widgets.path_diagram import build_layout

    spec = _moderated_model()
    layout = build_layout(spec, show_items=False, rtl=rtl)
    ids = {c.name: c.id for c in spec.constructs}
    moderator = layout.nodes[ids["MOD"]]
    core = [layout.nodes[ids[n]] for n in ("SAT", "ENG", "PERF")]
    # Above or below the whole model, never in a column of predictors.
    assert (moderator.rect.bottom() < min(n.rect.top() for n in core)
            or moderator.rect.top() > max(n.rect.bottom() for n in core))
    assert moderator.tone == "moderator"

    (edge,) = [e for e in layout.edges if e.kind == "interaction"]
    assert edge.anchor is not None
    eng, perf = layout.nodes[ids["ENG"]], layout.nodes[ids["PERF"]]
    # The arrow lands on the ENG -> PERF segment, strictly between its ends.
    assert min(eng.x, perf.x) < edge.anchor.x() < max(eng.x, perf.x)
    t = (edge.anchor.x() - eng.x) / (perf.x - eng.x)
    expected_y = eng.y + t * (perf.y - eng.y)
    assert abs(edge.anchor.y() - expected_y) < 12.0
    # Its own effect on the outcome is drawn, but quieter.
    (main,) = [e for e in layout.edges
               if e.kind == "path" and e.source == ids["MOD"]]
    assert main.muted


def test_a_mediator_lines_up_with_what_feeds_it(application):
    from semsim.core.spec import ModelSpec
    from semsim.ui.widgets.path_diagram import build_layout

    spec = ModelSpec(seed=1)
    for name in ("A", "B", "C", "D"):
        spec.add_construct(name, role=ConstructRole.INDEPENDENT, n_items=3)
    spec.add_construct("M", role=ConstructRole.MEDIATOR, n_items=3)
    spec.add_construct("Y", role=ConstructRole.DEPENDENT, n_items=3)
    spec.add_path("C", "M")
    spec.add_path("D", "M")
    spec.add_path("A", "Y")
    spec.add_path("M", "Y")
    layout = build_layout(spec, show_items=False, rtl=False)
    node = {c.name: layout.nodes[c.id] for c in spec.constructs}
    # M is fed by the lower two predictors, so it sits level with them.
    assert node["M"].y == pytest.approx((node["C"].y + node["D"].y) / 2)


def test_long_labels_fit_inside_their_ellipse(application):
    from PySide6.QtGui import QFontMetricsF
    from semsim.ui.widgets.path_diagram import (
        LABEL_PAD, _label_font, build_layout)

    translator.set_language("ar")
    spec = examples.build("simple_mediation")
    spec.constructs[0].label.ar = "حوكمة وشفافية الذكاء الاصطناعي"
    layout = build_layout(spec, show_items=False, rtl=True)
    node = layout.nodes[spec.constructs[0].id]
    metrics = QFontMetricsF(_label_font())
    words = node.label.split()
    # Every word fits on a line, so wrapping never spills past the outline.
    assert all(metrics.horizontalAdvance(w) <= node.w - LABEL_PAD
               for w in words)

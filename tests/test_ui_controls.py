"""
Every control in the interface has to answer the person pressing it.

A button may do its work, or be greyed out with the reason shown.  What it may
not do is accept the click and change nothing, which is how a dead handler, a
swallowed exception and a precondition that quietly returns all look from the
chair.  These tests drive the real window headlessly.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QLocale  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton, QApplication, QDialog, QFileDialog, QLabel, QMenu,
    QMessageBox, QWidget,
)

from semsim.core import examples  # noqa: E402
from semsim.core.spec import ConstructOrder  # noqa: E402
from semsim.ui import theme  # noqa: E402
from semsim.ui.app import (  # noqa: E402
    MainWindow, SimulationWorker, install_exception_hook,
)
from semsim.ui.i18n import translator  # noqa: E402
from semsim.ui.state import Results  # noqa: E402
from semsim.ui.pages.base import IdCombo  # noqa: E402
from semsim.ui.widgets.common import EnumCombo, Field, float_spin  # noqa: E402


#: The window fixture replaces ``run`` so clicks never start a simulation.
REAL_RUN = MainWindow.run


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication(["semsim-test"])
    theme.apply(app, theme.LIGHT, rtl=True)
    return app


@pytest.fixture
def events(monkeypatch, tmp_path):
    """Replace every modal call with a recording, so nothing can block."""
    seen = []

    def recorder(name, answer=None):
        def fake(*_args, **_kwargs):
            seen.append(name)
            return answer
        return staticmethod(fake)

    monkeypatch.setattr(QMessageBox, "information",
                        recorder("info", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "warning",
                        recorder("warning", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question",
                        recorder("question", QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "about", recorder("about"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        recorder("open-file", ("", "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        recorder("pick-dir", str(tmp_path)))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        recorder("save-file", (str(tmp_path / "output.png"), "")))
    monkeypatch.setattr(
        QDialog, "exec",
        lambda self, *_a: seen.append(f"dialog:{type(self).__name__}") or 0)
    monkeypatch.setattr(QMenu, "exec", lambda self, *_a: seen.append("menu"))
    return seen


@pytest.fixture
def errors(monkeypatch):
    """Exceptions raised inside slots, which Qt would otherwise step over."""
    caught = []
    monkeypatch.setattr(
        sys, "excepthook",
        lambda kind, value, tb: caught.append(
            "".join(traceback.format_exception(kind, value, tb))))
    return caught


@pytest.fixture
def window(application, events, errors, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    # Recorded, not run: a real simulation would outlive the click under test.
    monkeypatch.setattr(MainWindow, "run",
                        lambda self, mode="full": events.append(f"run:{mode}"))
    sheet = application.styleSheet()
    win = MainWindow(examples.simple_mediation())
    win.resize(1320, 860)
    win.show()
    application.processEvents()
    yield win
    win.state._dirty = False
    win.close()
    win.deleteLater()
    # processEvents() leaves deferred deletes alone; without this every window
    # stays alive and each later restyle has to walk all of them.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()
    # The translator lives for the whole process and would go on calling the
    # widgets of every window already closed.
    translator.language_changed.disconnect()
    if translator.language != "ar" or application.styleSheet() != sheet:
        translator.set_language("ar")
        theme.apply(application, theme.LIGHT, rtl=True)


def settle() -> None:
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()


def snapshot(root: QWidget, win: MainWindow):
    parts = [repr(win.state.spec), win.state.dirty, win.nav.currentRow(),
             QApplication.clipboard().text(),
             QApplication.instance().styleSheet(), translator.language]
    for widget in [root] + root.findChildren(QWidget):
        row = [type(widget).__name__, widget.isVisibleTo(root),
               widget.isEnabled()]
        if isinstance(widget, (QAbstractButton, QLabel)):
            row.append(widget.text())
        if isinstance(widget, QAbstractButton):
            row.append(widget.isChecked())
        parts.append(tuple(row))
    return parts


def press_every_button(win, index, events, errors):
    """Press each enabled button on one page against a freshly reset model."""
    silent, pressed = [], 0
    page = win._pages[index]
    win.nav.setCurrentRow(index)
    settle()
    total = len(page.findChildren(QAbstractButton))
    for position in range(total):
        win.state.set_spec(examples.simple_mediation())
        win._rebuild_pages()
        settle()
        buttons = page.findChildren(QAbstractButton)
        if position >= len(buttons):
            continue
        button = buttons[position]
        if not button.isEnabled() or not button.isVisibleTo(page):
            continue
        if button.isChecked() and button.autoExclusive():
            continue          # choosing the option already chosen
        label = f"{type(button).__name__}[{getattr(button, '_key', '')}]"
        events.clear()
        before = snapshot(page, win)
        button.click()
        settle()
        pressed += 1
        assert not errors, f"{label} raised:\n{errors[0]}"
        if not events and before == snapshot(page, win):
            silent.append(label)
    return pressed, silent


@pytest.mark.parametrize("index", range(6))
def test_no_button_ignores_the_click(window, events, errors, index):
    pressed, silent = press_every_button(window, index, events, errors)
    assert pressed > 0
    assert silent == []


def test_every_menu_command_responds(window, events, errors):
    for key, action in window._actions.items():
        if key == "menu.file.exit" or not action.isEnabled():
            continue
        if action.isChecked() and action.actionGroup() is not None:
            continue          # choosing the option already chosen
        events.clear()
        before = snapshot(window, window)
        action.trigger()
        settle()
        assert not errors, f"{key} raised:\n{errors[0]}"
        assert events or before != snapshot(window, window), key


def test_stop_is_only_offered_during_a_run(window):
    stop = window._actions["menu.run.cancel"]
    assert not stop.isEnabled()
    window.state.set_busy(True)
    assert stop.isEnabled()
    assert not window._actions["menu.run.full"].isEnabled()
    window.state.set_busy(False)
    assert not stop.isEnabled()


# -- stage 1 ---------------------------------------------------------------- #


def test_arrows_move_items_as_well_as_constructs(window):
    page = window._pages[0]
    items = window.state.spec.constructs[0].items
    first, second = items[0].id, items[1].id
    page.tree.setCurrentItem(page._find_node(first))
    assert not page._btn_up.isEnabled()
    assert page._btn_down.isEnabled()

    page._btn_down.click()
    items = window.state.spec.constructs[0].items
    assert [items[0].id, items[1].id] == [second, first]
    assert page._selected()[1] == first
    assert page._btn_up.isEnabled()


def test_arrows_stop_at_the_ends_of_the_list(window):
    page = window._pages[0]
    last = window.state.spec.constructs[-1]
    page.tree.setCurrentItem(page._find_node(last.id))
    assert page._btn_up.isEnabled()
    assert not page._btn_down.isEnabled()


def test_a_new_construct_is_selected(window):
    page = window._pages[0]
    before = {c.id for c in window.state.spec.constructs}
    page._btn_add.click()
    added = [c for c in window.state.spec.constructs if c.id not in before]
    assert len(added) == 1
    assert page._selected() == ("construct", added[0].id)


def test_varying_loadings_changes_them_every_time(window):
    page = window._pages[0]
    construct = window.state.spec.constructs[0]
    page.tree.setCurrentItem(page._find_node(construct.id))
    page._construct_editor.vary.click()
    first = [item.loading for item in construct.items]
    page._construct_editor.vary.click()
    assert [item.loading for item in construct.items] != first


def test_choice_lists_hand_back_the_real_enum_member(application):
    combo = EnumCombo(ConstructOrder, "m.order")
    combo.setCurrentIndex(1)
    assert combo.value() is ConstructOrder.SECOND


def switch_order(window, construct, index):
    page = window._pages[0]
    page.tree.setCurrentItem(page._find_node(construct.id))
    page._construct_editor.order.setCurrentIndex(index)
    settle()
    return page


def test_second_order_puts_the_items_inside_dimensions(window):
    construct = window.state.spec.constructs[0]
    typed = [item.id for item in construct.items]

    page = switch_order(window, construct, 1)

    assert construct.is_second_order
    assert construct.items == []
    assert len(construct.dimensions) == 3
    assert all(len(d.items) >= 3 for d in construct.dimensions)
    assert [item.id for item in construct.all_items()][:len(typed)] == typed
    assert window.state.can_run

    node = page._find_node(construct.id)
    assert "²" in node.text(0)
    assert node.child(0).text(0).startswith(f"{construct.name}_D1")
    assert node.child(0).childCount() == 3

    editor = page._construct_editor
    assert editor._n_dims_field.isVisibleTo(page)
    assert not editor._n_items_field.isVisibleTo(page)
    assert editor.bulk._title.text() == translator.t("m.dimensions")


def test_second_order_adds_no_items_when_there_are_enough(window):
    construct = window.state.spec.constructs[0]
    construct.items.extend(construct.items[0].__class__(
        name=f"EXTRA{n}", loading=0.7) for n in range(7))
    typed = [item.id for item in construct.items]
    assert len(typed) == 12

    switch_order(window, construct, 1)

    assert [item.id for item in construct.all_items()] == typed
    assert [len(d.items) for d in construct.dimensions] == [4, 4, 4]


def test_returning_to_first_order_keeps_every_item(window):
    construct = window.state.spec.constructs[0]
    switch_order(window, construct, 1)
    ids = [item.id for item in construct.all_items()]

    page = switch_order(window, construct, 0)

    assert not construct.is_second_order
    assert construct.dimensions == []
    assert [item.id for item in construct.items] == ids
    assert page._construct_editor._n_items_field.isVisibleTo(page)


def test_a_project_saved_with_the_old_fault_keeps_its_items(window):
    # The old code stored "second" while leaving the items on the construct.
    construct = window.state.spec.constructs[0]
    ids = [item.id for item in construct.items]
    construct.order = ConstructOrder.SECOND
    window._pages[0].rebuild()
    settle()

    switch_order(window, construct, 0)

    assert not construct.is_second_order
    assert [item.id for item in construct.items] == ids


def test_new_items_are_numbered_on_from_the_existing_ones(window):
    construct = window.state.spec.constructs[0]
    switch_order(window, construct, 1)
    names = [item.name for item in construct.all_items()]
    assert names == [f"{construct.name}{n}" for n in range(1, len(names) + 1)]


# Real side-scroll is measured against the Windows fonts by hand: the offscreen
# platform these tests run on draws text at about twice its width.  What can be
# pinned here is the mechanism, which does not depend on any font.
@pytest.mark.parametrize("build", [
    lambda: EnumCombo(ConstructOrder, "m.order"),
    lambda: IdCombo([("a", "a")]).combo,
])
def test_a_long_choice_does_not_widen_its_list(application, build):
    combo = build()
    before = combo.minimumSizeHint().width()
    combo.addItem("x" * 300)
    assert combo.minimumSizeHint().width() == before


def test_field_labels_give_way_in_a_narrow_pane(application):
    field = Field("m.order", QLabel("x"), label_width=220)
    label = field._label
    assert label.minimumWidth() < label.maximumWidth() == 220


def test_covariance_says_why_it_is_unavailable(window):
    editor = window._pages[0]._structural
    assert len(window.state.spec.exogenous_construct_ids()) == 1
    assert not editor.covariances.add_button.isEnabled()
    assert (editor.covariances._blocked.text()
            == translator.t("s.block.two_exogenous"))


@pytest.mark.parametrize("list_name, collection", [
    ("paths", "paths"),
    ("interactions", "interactions"),
    ("indirects", "indirect_effects"),
])
def test_add_buttons_run_out_rather_than_repeat(window, list_name, collection):
    editor = window._pages[0]._structural
    rows = getattr(editor, list_name)
    for _ in range(40):
        if not rows.add_button.isEnabled():
            break
        rows.add_button.click()
    assert not rows.add_button.isEnabled()
    assert rows._blocked.text()

    report = window.state.spec
    values = getattr(report, collection)
    if collection == "paths":
        keys = [frozenset((p.source, p.target)) for p in values]
    elif collection == "interactions":
        keys = [(frozenset((i.predictor, i.moderator)), i.target) for i in values]
    else:
        keys = [tuple(i.chain) for i in values]
    assert len(keys) == len(set(keys))


def test_mediation_is_never_offered_without_real_paths(window):
    editor = window._pages[0]._structural
    window.state.spec.paths.clear()
    window.state.spec.indirect_effects.clear()
    editor.rebuild()
    assert not editor.indirects.add_button.isEnabled()
    assert (editor.indirects._blocked.text()
            == translator.t("s.block.no_chain"))


# -- estimating -------------------------------------------------------------- #


def test_estimate_refits_the_data_on_hand(application):
    spec = examples.simple_mediation()
    generated = SimulationWorker(spec, "generate")._run_pipeline()
    prior = Results(dataset=generated["dataset"], power="power",
                    calibration="calibration")

    refit = SimulationWorker(spec, "estimate", prior)._run_pipeline()

    assert refit["dataset"] is generated["dataset"]
    assert (refit["power"], refit["calibration"]) == ("power", "calibration")
    assert refit["estimation"] is not None
    assert refit["fit"] is not None


def test_estimate_without_data_says_why(window, events):
    REAL_RUN(window, "estimate")
    assert "info" in events
    assert not window.state.busy


def test_estimate_on_data_from_another_model_says_why(window, events):
    window.state.results.dataset = SimpleNamespace(
        data=pd.DataFrame({"unrelated": [1.0, 2.0]}))
    REAL_RUN(window, "estimate")
    assert "info" in events
    assert not window.state.busy


# -- confirmations ---------------------------------------------------------- #


def test_applying_a_scale_is_confirmed(window):
    page = window._pages[2]
    page.btn_apply_all.click()
    assert page.apply_notice.isVisibleTo(page)
    assert str(window.state.spec.n_items) in page.apply_notice.text()


def test_applying_to_a_construct_survives_a_new_model(window):
    # The chooser used to keep pointing at the previous model's construct, so
    # after File > New the button found nothing and returned without a word.
    page = window._pages[2]
    window.state.set_spec(examples.simple_mediation())
    window._rebuild_pages()
    settle()
    ids = {construct.id for construct in window.state.spec.constructs}
    assert page.apply_target.value() in ids
    page.btn_apply_construct.click()
    assert page.apply_notice.isVisibleTo(page)


def test_variation_level_drives_the_amounts(window, errors):
    page = window._pages[2]
    settings = window.state.spec.heterogeneity
    assert page.variation.value() == "realistic"
    page.variation.setCurrentIndex(page.variation.findData("strong"))
    assert settings.enabled and settings.preset_name() == "strong"
    assert page.path_sd.value() == pytest.approx(settings.path_sd)

    page.variation.setCurrentIndex(page.variation.findData("none"))
    assert not settings.enabled
    assert not page.loading_sd.isEnabled()

    page.variation.setCurrentIndex(page.variation.findData("mild"))
    page.path_sd.setValue(0.07)
    assert settings.path_sd == pytest.approx(0.07)
    assert page.variation.value() == "custom"
    assert not errors


def test_rechecking_the_model_is_confirmed(window):
    from semsim.ui.pages.validation_panel import ValidationDialog
    dialog = ValidationDialog(window.state, window)
    dialog._recheck.click()
    assert dialog._status.text() == translator.t("v.rechecked")
    dialog.deleteLater()


def test_empty_recent_menu_says_so(window):
    texts = [a.text() for a in window.recent_menu.actions()]
    assert texts == [translator.t("menu.file.recent.empty")]


def test_results_page_does_not_claim_the_model_has_errors(window):
    page = window._pages[5]
    assert page._empty._key == "r.no_results"


def test_saving_is_confirmed(window, tmp_path):
    path = tmp_path / "model.semsim"
    window.state.save(str(path))
    window.save_project()
    assert window.state.display_name in window.statusBar().currentMessage()


# -- failures must be visible ------------------------------------------------ #


def test_an_error_in_a_button_handler_reaches_the_user(application, events,
                                                       monkeypatch):
    # The hook replaces sys.excepthook; pytest's copy is restored afterwards.
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    install_exception_hook()
    try:
        raise ValueError("boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert "dialog:QMessageBox" in events


def test_stale_widget_signals_do_not_alarm_the_user(application, events,
                                                    monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    install_exception_hook()
    try:
        raise RuntimeError("Internal C++ object (QLabel) already deleted.")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert events == []


def test_language_switch_survives_a_rebuild(window, errors):
    window._pages[0]._structural.paths.add_button.click()
    window._rebuild_pages()
    settle()
    window.set_language("en")
    settle()
    window.set_language("ar")
    settle()
    assert errors == []


# -- appearance -------------------------------------------------------------- #


def test_arabic_table_cells_read_right_to_left(application):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QStyleOptionViewItem

    from semsim.ui.widgets.tables import DataFrameView
    view = DataFrameView(pd.DataFrame({"a": ["مقدار سوء المطابقة.", "> .05"]}))
    directions = []
    for row in range(2):
        option = QStyleOptionViewItem()
        option.direction = view.layoutDirection()
        view.itemDelegate().initStyleOption(option, view.model().index(row, 0))
        directions.append(option.direction)
    assert directions == [Qt.LayoutDirection.RightToLeft,
                          Qt.LayoutDirection.LeftToRight]
    assert view.layoutDirection() == Qt.LayoutDirection.LeftToRight


def test_numbers_use_western_digits(application):
    assert QLocale().toString(0.78, "f", 2) == "0.78"
    box = float_spin(0.0, 1.0, 0.78, decimals=2)
    assert re.fullmatch(r"0\.78", box.text())


@pytest.mark.parametrize("name", [theme.LIGHT, theme.DARK])
def test_every_indicator_image_exists(application, name):
    urls = re.findall(r'url\("([^"]+)"\)', theme.stylesheet(name))
    assert len(urls) >= 8
    for url in urls:
        assert os.path.exists(url), url


def test_every_stage_has_an_arabic_subtitle():
    from semsim.ui.i18n import STRINGS
    for stage in range(1, 7):
        arabic, _english = STRINGS[f"stage.{stage}.subtitle"]
        assert re.search(r"[؀-ۿ]", arabic), stage


def test_a_comparison_variable_gets_a_difference_editor(window, errors):
    from semsim.core.spec import DemographicRole, GroupEffect
    spec = window.state.spec
    variable = spec.add_demographic("Gender", [("Male", 0.6), ("Female", 0.4)],
                                    role=DemographicRole.COMPARISON)
    editor = window._pages[0]._demographics
    editor.rebuild()
    settle()
    add = [b for b in editor.findChildren(QAbstractButton)
           if getattr(b, "_key", "") == "dem.add_difference"]
    assert len(add) == 1
    add[0].click()
    settle()
    assert len(variable.group_differences) == 1
    difference = variable.group_differences[0]
    assert difference.target == spec.constructs[0].id

    sizes = [c for c in editor.findChildren(EnumCombo) if c._prefix == "geffect"]
    assert len(sizes) == 1
    sizes[0].setCurrentIndex(list(GroupEffect).index(GroupEffect.LARGE))
    assert difference.size is GroupEffect.LARGE
    assert difference.resolved_value(2) == pytest.approx(0.8)

    directions = [c for c in editor.findChildren(IdCombo)
                  if c.combo.findData("ascending") >= 0]
    assert len(directions) == 1
    assert "Female" in directions[0].combo.itemText(0)
    directions[0].combo.setCurrentIndex(directions[0].combo.findData("descending"))
    assert difference.direction.value == "descending"
    assert not errors


def test_two_demographics_can_be_linked(window, errors):
    from semsim.core.spec import AssociationStrength
    spec = window.state.spec
    age = spec.add_demographic("Age", [("Young", 0.5), ("Old", 0.5)])
    spec.add_demographic("Experience", [("Low", 0.5), ("High", 0.5)])
    editor = window._pages[0]._demographics
    editor.rebuild()
    settle()
    add = [b for b in editor.findChildren(QAbstractButton)
           if getattr(b, "_key", "") == "dem.add_association"]
    assert len(add) == 1
    add[0].click()
    settle()
    assert len(spec.demographic_associations) == 1
    association = spec.demographic_associations[0]
    assert {association.first, association.second} == {
        v.id for v in spec.demographics}

    strengths = [c for c in editor.findChildren(EnumCombo)
                 if c._prefix == "astrength"]
    strengths[0].setCurrentIndex(
        list(AssociationStrength).index(AssociationStrength.STRONG))
    assert association.strength is AssociationStrength.STRONG
    together = [c for c in editor.findChildren(IdCombo)
                if c.combo.findData("positive") >= 0]
    assert "Old" in together[0].combo.itemText(0)

    # Deleting a variable takes its links with it.
    editor._remove_variable(age)
    settle()
    assert spec.demographic_associations == []
    assert not errors


def test_style_level_drives_the_amounts(window, errors):
    page = window._pages[2]
    settings = window.state.spec.respondent_styles
    assert page.styles_level.value() == "realistic"
    page.styles_level.setCurrentIndex(page.styles_level.findData("strong"))
    assert settings.preset_name() == "strong"
    assert page.extreme_sd.value() == pytest.approx(settings.extreme_sd)
    page.styles_level.setCurrentIndex(page.styles_level.findData("none"))
    assert not settings.enabled and not page.extreme_sd.isEnabled()
    page.styles_level.setCurrentIndex(page.styles_level.findData("mild"))
    page.acquiescence_sd.setValue(0.25)
    assert settings.acquiescence_sd == pytest.approx(0.25)
    assert page.styles_level.value() == "custom"
    assert not errors


def test_careless_respondents_default_to_being_screened_out(window, errors):
    page = window._pages[2]
    noise = window.state.spec.noise
    assert page.careless_style.value() == "realistic"
    assert page.careless_screened.isChecked() and noise.careless_screened
    page.careless_screened.setChecked(False)
    assert not noise.careless_screened
    page.careless_style.setCurrentIndex(
        page.careless_style.findData("straightline"))
    assert noise.careless_style == "straightline"
    page.noise_enabled.setChecked(False)
    assert not page.careless_screened.isEnabled()
    assert not errors

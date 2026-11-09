"""
semsim.ui.pages.demographics
============================
The demographics tab of stage 1.

Each variable is one card: its name, its measurement level and its role at the
top, then its categories with their percentages, and, for a control variable,
the paths it sends into the model.  A running total under the percentages
turns red until they reach 100, because a quota that does not add up is the
mistake this screen most needs to catch while it is being made.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QToolButton,
    QVBoxLayout, QWidget,
)

from semsim.core.spec import (
    Bilingual, ControlPath, DemographicCategory, DemographicKind,
    DemographicRole, DemographicVariable, EffectSize, ModelSpec,
)
from semsim.ui import theme
from semsim.ui.i18n import STRINGS, translator
from semsim.ui.pages.base import IdCombo, Row, RowList, construct_choices
from semsim.ui.state import AppState
from semsim.ui.widgets.common import (
    Card, EnumCombo, InfoBanner, ScrollPage, TButton, TCheckBox, TKey,
    ValueSlider, percent_spin, percent_value,
)

MAX_CATEGORIES = 20
#: How far the percentages may stray from 100 before the total turns red.
TOTAL_TOLERANCE = 0.005

#: Ready-made variables: (template key, name, level, labels with shares).
TEMPLATES: Sequence[Tuple[str, str, DemographicKind,
                          Sequence[Tuple[str, str, float]]]] = (
    ("dem.tpl.gender", "Gender", DemographicKind.NOMINAL,
     (("ذكر", "Male", 0.60), ("أنثى", "Female", 0.40))),
    ("dem.tpl.age", "Age", DemographicKind.ORDINAL,
     (("أقل من 30 سنة", "Under 30", 0.30),
      ("من 30 إلى 45 سنة", "30 to 45", 0.45),
      ("أكثر من 45 سنة", "Over 45", 0.25))),
    ("dem.tpl.education", "Education", DemographicKind.ORDINAL,
     (("بكالوريوس", "Bachelor's", 0.70),
      ("دراسات عليا", "Postgraduate", 0.30))),
    ("dem.tpl.job", "JobTitle", DemographicKind.NOMINAL,
     (("إداري", "Administrative", 0.50), ("فني", "Technical", 0.35),
      ("قيادي", "Managerial", 0.15))),
    ("dem.tpl.experience", "Experience", DemographicKind.ORDINAL,
     (("أقل من 5 سنوات", "Under 5 years", 0.35),
      ("من 5 إلى 10 سنوات", "5 to 10 years", 0.40),
      ("أكثر من 10 سنوات", "Over 10 years", 0.25))),
    ("dem.tpl.sector", "Sector", DemographicKind.NOMINAL,
     (("عام", "Public", 0.50), ("خاص", "Private", 0.35),
      ("مختلط", "Mixed", 0.15))),
)


def _set_label(label: Bilingual, text: str) -> None:
    """Write the label for the language on screen, leaving the other alone."""
    if translator.language == "ar":
        label.ar = text
    else:
        label.en = text


def term_choices(variable: DemographicVariable) -> List[Tuple[str, str]]:
    """``[(term key, text)]`` for the term chooser of a control path."""
    lang = translator.language
    name = variable.label.get(lang) or variable.name
    out: List[Tuple[str, str]] = []
    for term in variable.terms():
        if term.category is None:
            out.append((term.key, name))
            continue
        out.append((term.key, translator.t(
            "dem.term_vs", name=name,
            category=variable.categories[term.category].label.get(lang),
            reference=variable.categories[0].label.get(lang))))
    return out


class DemographicsEditor(QWidget):
    """Define demographic variables, their quotas and their roles."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._updating = False
        self._cards: List[Card] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = ScrollPage()
        outer.addWidget(self.scroll)

        self.scroll.add(InfoBanner("dem.explain"))

        settings = Card("dem.settings")
        self.quota_exact = TCheckBox("dem.quota_exact")
        self.quota_exact.toggled.connect(self._on_quota)
        settings.add(self.quota_exact)
        self.quota_hint = TKey("dem.quota_exact.on", role="hint")
        settings.add(self.quota_hint)

        actions = QHBoxLayout()
        self.add_button = TButton("dem.add", role="primary")
        self.add_button.clicked.connect(self._add_variable)
        actions.addWidget(self.add_button)
        self.template = IdCombo()
        actions.addWidget(self.template, 1)
        self.template_button = TButton("dem.add_template")
        self.template_button.clicked.connect(self._add_template)
        actions.addWidget(self.template_button)
        settings.add(actions)
        self.scroll.add(settings)

        self.empty = InfoBanner("dem.empty")
        self.scroll.add(self.empty)
        self.host = QWidget()
        self.cards_layout = QVBoxLayout(self.host)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)
        self.scroll.add(self.host)
        self.scroll.add_stretch()

        translator.language_changed.connect(lambda _l: self.rebuild())
        self.rebuild()

    # ------------------------------------------------------------------ #

    @property
    def spec(self) -> ModelSpec:
        return self.state.spec

    def edit(self, structural: bool = False) -> None:
        if self._updating:
            return
        self.state.touch(structural=structural)

    def rebuild(self) -> None:
        self._updating = True
        try:
            self.quota_exact.setChecked(bool(self.spec.quota_exact))
            self._show_quota_hint()
            self.template.set_choices(
                [(key, translator.t(key)) for key, *_rest in TEMPLATES],
                stored=False)
            for card in self._cards:
                self.cards_layout.removeWidget(card)
                card.setParent(None)
                card.deleteLater()
            self._cards = []
            for variable in self.spec.demographics:
                card = self._variable_card(variable)
                self._cards.append(card)
                self.cards_layout.addWidget(card)
            self.empty.setVisible(not self.spec.demographics)
        finally:
            self._updating = False

    def refresh(self) -> None:
        self.rebuild()

    # ------------------------------------------------------------------ #
    # Sampling scheme and adding variables
    # ------------------------------------------------------------------ #

    def _show_quota_hint(self) -> None:
        self.quota_hint.set_key("dem.quota_exact.on" if self.spec.quota_exact
                                else "dem.quota_exact.off")

    def _on_quota(self, checked: bool) -> None:
        self.spec.quota_exact = bool(checked)
        self._show_quota_hint()
        self.edit()

    def _unique_name(self, base: str) -> str:
        taken = set(self.spec.demographic_columns()) | set(self.spec.all_item_names())
        taken |= {c.name for c in self.spec.constructs}
        name, counter = base, 1
        while name in taken:
            counter += 1
            name = f"{base}{counter}"
        return name

    def _add_variable(self) -> None:
        number = len(self.spec.demographics) + 1
        first = translator.t("dem.category_n", n=1)
        second = translator.t("dem.category_n", n=2)
        self.spec.demographics.append(DemographicVariable(
            name=self._unique_name(f"Demo{number}"),
            label=Bilingual(ar=translator.t("dem.variable_n", n=number),
                            en=translator.t("dem.variable_n", n=number)),
            categories=[
                DemographicCategory(label=Bilingual(ar=first, en=first),
                                    proportion=0.5),
                DemographicCategory(label=Bilingual(ar=second, en=second),
                                    proportion=0.5)]))
        self.edit(structural=True)

    def _add_template(self) -> None:
        key = self.template.value()
        for template_key, name, kind, categories in TEMPLATES:
            if template_key != key:
                continue
            ar, en = STRINGS[template_key]
            self.spec.demographics.append(DemographicVariable(
                name=self._unique_name(name),
                label=Bilingual(ar=ar, en=en), kind=kind,
                categories=[DemographicCategory(
                    label=Bilingual(ar=cat_ar, en=cat_en), proportion=share)
                    for cat_ar, cat_en, share in categories]))
            self.edit(structural=True)
            return

    def _remove_variable(self, variable: DemographicVariable) -> None:
        self.spec.demographics = [v for v in self.spec.demographics
                                  if v is not variable]
        self.edit(structural=True)

    # ------------------------------------------------------------------ #
    # One variable
    # ------------------------------------------------------------------ #

    def _variable_card(self, variable: DemographicVariable) -> Card:
        lang = translator.language
        card = Card()

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        for column, key in enumerate(("m.name", "dem.label", "dem.kind", "dem.role")):
            grid.addWidget(TKey(key, role="hint"), 0, column)

        name = QLineEdit(variable.name)
        theme.keep_ltr(name)
        label = QLineEdit(variable.label.get(lang))
        kind = EnumCombo(DemographicKind, "dkind")
        kind.set_value(variable.kind)
        role = EnumCombo(DemographicRole, "drole")
        role.set_value(variable.role)
        for column, widget in enumerate((name, label, kind, role)):
            grid.addWidget(widget, 1, column)
            grid.setColumnStretch(column, 1)

        delete = QToolButton()
        delete.setText("×")
        delete.setAutoRaise(True)
        delete.setFixedWidth(26)
        delete.setToolTip(translator.t("m.remove"))
        delete.clicked.connect(lambda: self._remove_variable(variable))
        grid.addWidget(delete, 1, 4)
        card.add(grid)

        hint = TKey(f"dem.role_hint.{variable.role.value}", role="hint")
        card.add(hint)

        categories = RowList("dem.add_category", empty_key="dem.no_categories")
        categories.set_header([("dem.category", 6), ("dem.percent", 3)])
        card.add(categories)

        footer = QHBoxLayout()
        total = QLabel()
        footer.addWidget(total, 1)
        equal = TButton("dem.equal", role="flat")
        footer.addWidget(equal)
        card.add(footer)

        def update_total() -> None:
            value = sum(float(c.proportion) for c in variable.categories)
            good = abs(value - 1.0) <= TOTAL_TOLERANCE
            theme.set_role(total, "success" if good else "error")
            total.setText(translator.t(
                "dem.total_ok" if good else "dem.total_bad",
                total=f"{value * 100.0:.1f}"))

        term_combos: List[IdCombo] = []

        def refresh_terms() -> None:
            choices = term_choices(variable)
            for combo in term_combos:
                combo.set_choices(choices)

        for category in variable.categories:
            categories.add_row(self._category_row(
                variable, category, update_total, refresh_terms))
        update_total()
        categories.add_button.clicked.connect(lambda: self._add_category(variable))
        categories.set_add_blocked(
            "dem.block.max_categories"
            if len(variable.categories) >= MAX_CATEGORIES else "")

        def on_equal() -> None:
            share = 1.0 / len(variable.categories)
            for category in variable.categories:
                category.proportion = share
            self.edit(structural=True)

        equal.clicked.connect(on_equal)

        if variable.role is DemographicRole.CONTROL:
            card.add(self._control_paths(variable, term_combos))

        def on_name(text: str) -> None:
            variable.name = text.strip()
            self.edit()

        def on_label(text: str) -> None:
            _set_label(variable.label, text)
            refresh_terms()
            self.edit()

        def on_kind(value: DemographicKind) -> None:
            variable.kind = value
            variable.reconcile_control_paths()
            self.edit(structural=True)

        def on_role(value: DemographicRole) -> None:
            variable.role = value
            variable.reconcile_control_paths()
            self.edit(structural=True)

        name.textEdited.connect(on_name)
        label.textEdited.connect(on_label)
        kind.value_changed.connect(on_kind)
        role.value_changed.connect(on_role)
        return card

    # ------------------------------------------------------------------ #
    # Categories
    # ------------------------------------------------------------------ #

    def _category_row(self, variable: DemographicVariable,
                      category: DemographicCategory,
                      update_total: Callable[[], None],
                      refresh_terms: Callable[[], None]) -> Row:
        row = Row()
        label = QLineEdit(category.label.get(translator.language))
        row.add(label, 6)
        share = percent_spin(category.proportion)
        row.add(share, 3)
        delete = row.add_delete_button()
        if len(variable.categories) <= 2:
            delete.setEnabled(False)
            delete.setToolTip(translator.t("dem.block.min_categories"))

        def on_label(text: str) -> None:
            _set_label(category.label, text)
            refresh_terms()
            self.edit()

        def on_share(_value: float) -> None:
            category.proportion = percent_value(share)
            update_total()
            self.edit()

        label.textEdited.connect(on_label)
        share.valueChanged.connect(on_share)
        row.removed.connect(lambda _r: self._remove_category(variable, category))
        return row

    def _add_category(self, variable: DemographicVariable) -> None:
        number = len(variable.categories) + 1
        text = translator.t("dem.category_n", n=number)
        variable.categories.append(DemographicCategory(
            label=Bilingual(ar=text, en=text), proportion=0.0))
        variable.reconcile_control_paths()
        self.edit(structural=True)

    def _remove_category(self, variable: DemographicVariable,
                         category: DemographicCategory) -> None:
        variable.categories = [c for c in variable.categories if c is not category]
        variable.reconcile_control_paths()
        self.edit(structural=True)

    # ------------------------------------------------------------------ #
    # Control paths
    # ------------------------------------------------------------------ #

    def _targets(self) -> List[Tuple[str, str]]:
        allowed = set(self.spec.control_target_ids())
        return [(cid, text) for cid, text in construct_choices(self.spec)
                if cid in allowed]

    def _control_paths(self, variable: DemographicVariable,
                       term_combos: List[IdCombo]) -> QWidget:
        host = QFrame()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(TKey("dem.control_paths", role="h2"))
        paths = RowList("dem.add_control_path", empty_key="dem.no_control_paths")
        paths.set_header([("dem.term", 3), ("s.target", 3), ("s.effect_size", 3),
                          ("s.coefficient", 4), ("s.significant", 2)])
        layout.addWidget(paths)

        for path in variable.control_paths:
            paths.add_row(self._control_row(variable, path, term_combos))
        paths.add_button.clicked.connect(lambda: self._add_control(variable))
        if not self._targets():
            paths.set_add_blocked("dem.block.no_target")
        elif self._free_control(variable) is None:
            paths.set_add_blocked("dem.block.all_controls")
        return host

    def _free_control(self, variable: DemographicVariable
                      ) -> Optional[Tuple[str, str]]:
        """The first term and target not already joined by a path."""
        used = {(p.term, p.target) for p in variable.control_paths}
        for term in variable.terms():
            for target, _text in self._targets():
                if (term.key, target) not in used:
                    return term.key, target
        return None

    def _add_control(self, variable: DemographicVariable) -> None:
        free = self._free_control(variable)
        if free is None:
            return
        variable.control_paths.append(ControlPath(term=free[0], target=free[1]))
        self.edit(structural=True)

    def _control_row(self, variable: DemographicVariable, path: ControlPath,
                     term_combos: List[IdCombo]) -> Row:
        row = Row()
        term = IdCombo(term_choices(variable))
        term.set_value(path.term)
        term.setEnabled(variable.is_dummy_coded)
        term_combos.append(term)
        row.add(term, 3)
        target = IdCombo(self._targets())
        target.set_choices(self._targets(), keep=path.target)
        row.add(target, 3)
        effect = EnumCombo(EffectSize, "effect")
        effect.set_value(path.effect_size)
        row.add(effect, 3)
        slider = ValueSlider(-0.95, 0.95, path.resolved_coefficient())
        row.add(slider, 4)
        significant = TCheckBox("s.significant")
        significant.setChecked(path.target_significant)
        row.add(significant, 2)
        row.add_delete_button()

        def on_term(value: str) -> None:
            path.term = value
            self.edit()

        def on_target(value: str) -> None:
            path.target = value
            self.edit()

        def on_effect(size: EffectSize) -> None:
            path.effect_size = size
            if size is not EffectSize.CUSTOM:
                path.coefficient_is_manual = False
                path.coefficient = size.default_coefficient
                blocked = slider.blockSignals(True)
                slider.set_value(path.coefficient)
                slider.blockSignals(blocked)
            self.edit()

        def on_coefficient(value: float) -> None:
            path.coefficient = value
            path.coefficient_is_manual = True
            if path.effect_size is not EffectSize.CUSTOM:
                path.effect_size = EffectSize.CUSTOM
                effect.set_value(EffectSize.CUSTOM)
            self.edit()

        def on_significant(checked: bool) -> None:
            path.target_significant = checked
            self.edit()

        term.value_changed.connect(on_term)
        target.value_changed.connect(on_target)
        effect.value_changed.connect(on_effect)
        slider.value_changed.connect(on_coefficient)
        significant.toggled.connect(on_significant)
        row.removed.connect(lambda _r: self._remove_control(variable, path))
        return row

    def _remove_control(self, variable: DemographicVariable,
                        path: ControlPath) -> None:
        variable.control_paths = [p for p in variable.control_paths
                                  if p is not path]
        self.edit(structural=True)


__all__ = ["DemographicsEditor", "TEMPLATES", "term_choices"]

"""
semsim.ui.pages.structural
==========================
The structural half of stage 1: paths, exogenous covariances, latent
interactions and the mediation chains to report.

Each collection is a :class:`~semsim.ui.pages.base.RowList` of live controls.
A path row carries two construct choosers, an effect-size combo, a coefficient
slider and a significance switch; the combo and the slider stay in step, so
picking "medium" writes 0.30 and dragging the slider flips the combo to
"custom" rather than silently contradicting itself.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from semsim.core.spec import (
    EffectSize, IndirectEffect, Interaction, LatentCovariance, ModelSpec, Path,
)
from semsim.ui import theme
from semsim.ui.i18n import translator
from semsim.ui.pages.base import IdCombo, Row, RowList, construct_choices
from semsim.ui.state import AppState
from semsim.ui.widgets.common import (
    Card, EnumCombo, InfoBanner, ScrollPage, TButton, TCheckBox, ValueSlider,
)


class StructuralEditor(QWidget):
    """The structural-model tab."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._updating = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = ScrollPage()
        outer.addWidget(self.scroll)

        self.paths = RowList("s.add_path", empty_key="s.paths")
        self.paths.set_header([("s.source", 3), ("s.target", 3),
                               ("s.effect_size", 3), ("s.coefficient", 4),
                               ("s.significant", 2)])
        self.paths.add_button.clicked.connect(self._add_path)
        card = Card("s.paths")
        card.add(self.paths)
        self.scroll.add(card)

        self.covariances = RowList("s.add_latent_cov", empty_key="s.latent_cov")
        self.covariances.set_header([("s.source", 3), ("s.target", 3),
                                     ("s.correlation", 4)])
        self.covariances.add_button.clicked.connect(self._add_covariance)
        cov_card = Card("s.latent_cov")
        cov_card.add(self.covariances)
        self.scroll.add(cov_card)

        self.interactions = RowList("s.add_interaction", empty_key="s.interactions")
        self.interactions.set_header([("s.predictor", 3), ("s.moderator", 3),
                                      ("s.target", 3), ("s.coefficient", 4),
                                      ("s.significant", 2)])
        self.interactions.add_button.clicked.connect(self._add_interaction)
        ix_card = Card("s.interactions")
        ix_card.add(InfoBanner("d.engine.hint"))
        ix_card.add(self.interactions)
        self.scroll.add(ix_card)

        self.indirects = RowList("s.add_indirect", empty_key="s.indirect")
        self.indirects.set_header([("s.chain", 8), ("s.significant", 2)])
        self.indirects.add_button.clicked.connect(self._add_indirect)
        ind_card = Card("s.indirect")
        ind_card.add(self.indirects)
        self.scroll.add(ind_card)
        self.scroll.add_stretch()

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
            self.paths.clear()
            for path in self.spec.paths:
                self.paths.add_row(self._path_row(path))
            self.covariances.clear()
            for cov in self.spec.latent_covariances:
                self.covariances.add_row(self._covariance_row(cov))
            self.interactions.clear()
            for interaction in self.spec.interactions:
                self.interactions.add_row(self._interaction_row(interaction))
            self.indirects.clear()
            for indirect in self.spec.indirect_effects:
                self.indirects.add_row(self._indirect_row(indirect))
        finally:
            self._updating = False

    def refresh(self) -> None:
        """Only the labels can change without a structural edit."""
        self.rebuild()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    def _add_path(self) -> None:
        if len(self.spec.constructs) < 2:
            return
        source, target = self.spec.constructs[0], self.spec.constructs[-1]
        self.spec.paths.append(Path(
            source=source.id, target=target.id,
            effect_size=EffectSize.MEDIUM,
            coefficient=EffectSize.MEDIUM.default_coefficient))
        self.edit(structural=True)

    def _path_row(self, path: Path) -> Row:
        row = Row()
        choices = construct_choices(self.spec)
        source = IdCombo(choices); source.set_value(path.source)
        row.add(source, 3)
        target = IdCombo(choices); target.set_value(path.target)
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

        def on_source(value: str) -> None:
            path.source = value
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
            # Dragging the slider makes the coefficient the user's own, so the
            # effect-size combo must stop claiming a conventional value.
            path.coefficient = value
            path.coefficient_is_manual = True
            if path.effect_size is not EffectSize.CUSTOM:
                path.effect_size = EffectSize.CUSTOM
                effect.set_value(EffectSize.CUSTOM)
            self.edit()

        source.value_changed.connect(on_source)
        target.value_changed.connect(on_target)
        effect.value_changed.connect(on_effect)
        slider.value_changed.connect(on_coefficient)
        significant.toggled.connect(
            lambda checked: (setattr(path, "target_significant", checked),
                             self.edit()))
        row.removed.connect(lambda _r: self._remove_path(path))
        return row

    def _remove_path(self, path: Path) -> None:
        self.spec.paths = [p for p in self.spec.paths if p is not path]
        self.edit(structural=True)

    # ------------------------------------------------------------------ #
    # Exogenous covariances
    # ------------------------------------------------------------------ #

    def _add_covariance(self) -> None:
        exogenous = self.spec.exogenous_construct_ids()
        if len(exogenous) < 2:
            return
        self.spec.latent_covariances.append(LatentCovariance(
            construct_a=exogenous[0], construct_b=exogenous[1],
            correlation=0.30))
        self.edit(structural=True)

    def _covariance_row(self, cov: LatentCovariance) -> Row:
        row = Row()
        choices = construct_choices(self.spec)
        first = IdCombo(choices); first.set_value(cov.construct_a)
        row.add(first, 3)
        second = IdCombo(choices); second.set_value(cov.construct_b)
        row.add(second, 3)
        slider = ValueSlider(-0.95, 0.95, cov.correlation)
        row.add(slider, 4)
        row.add_delete_button()
        first.value_changed.connect(
            lambda v: (setattr(cov, "construct_a", v), self.edit()))
        second.value_changed.connect(
            lambda v: (setattr(cov, "construct_b", v), self.edit()))
        slider.value_changed.connect(
            lambda v: (setattr(cov, "correlation", v), self.edit()))
        row.removed.connect(lambda _r: self._remove_covariance(cov))
        return row

    def _remove_covariance(self, cov: LatentCovariance) -> None:
        self.spec.latent_covariances = [
            c for c in self.spec.latent_covariances if c is not cov]
        self.edit(structural=True)

    # ------------------------------------------------------------------ #
    # Interactions
    # ------------------------------------------------------------------ #

    def _add_interaction(self) -> None:
        if len(self.spec.constructs) < 3:
            return
        ids = [c.id for c in self.spec.constructs]
        self.spec.interactions.append(Interaction(
            predictor=ids[0], moderator=ids[1], target=ids[-1],
            effect_size=EffectSize.WEAK,
            coefficient=EffectSize.WEAK.default_coefficient))
        self.edit(structural=True)

    def _interaction_row(self, interaction: Interaction) -> Row:
        row = Row()
        choices = construct_choices(self.spec)
        predictor = IdCombo(choices); predictor.set_value(interaction.predictor)
        row.add(predictor, 3)
        moderator = IdCombo(choices); moderator.set_value(interaction.moderator)
        row.add(moderator, 3)
        target = IdCombo(choices); target.set_value(interaction.target)
        row.add(target, 3)
        slider = ValueSlider(-0.95, 0.95, interaction.resolved_coefficient())
        row.add(slider, 4)
        significant = TCheckBox("s.significant")
        significant.setChecked(interaction.target_significant)
        row.add(significant, 2)
        row.add_delete_button()
        predictor.value_changed.connect(
            lambda v: (setattr(interaction, "predictor", v), self.edit()))
        moderator.value_changed.connect(
            lambda v: (setattr(interaction, "moderator", v), self.edit()))
        target.value_changed.connect(
            lambda v: (setattr(interaction, "target", v), self.edit()))

        def on_coefficient(value: float) -> None:
            interaction.coefficient = value
            interaction.coefficient_is_manual = True
            interaction.effect_size = EffectSize.CUSTOM
            self.edit()

        slider.value_changed.connect(on_coefficient)
        significant.toggled.connect(
            lambda checked: (setattr(interaction, "target_significant", checked),
                             self.edit()))
        row.removed.connect(lambda _r: self._remove_interaction(interaction))
        return row

    def _remove_interaction(self, interaction: Interaction) -> None:
        self.spec.interactions = [
            i for i in self.spec.interactions if i is not interaction]
        self.edit(structural=True)

    # ------------------------------------------------------------------ #
    # Indirect effects
    # ------------------------------------------------------------------ #

    def _add_indirect(self) -> None:
        chain = self._suggest_chain()
        if not chain:
            return
        self.spec.indirect_effects.append(
            IndirectEffect(chain=chain, label=self._chain_label(chain)))
        self.edit(structural=True)

    def _suggest_chain(self) -> List[str]:
        """Offer the first real X -> M -> Y the paths already describe."""
        outgoing: Dict[str, List[str]] = {}
        for path in self.spec.paths:
            outgoing.setdefault(path.source, []).append(path.target)
        for source, middles in outgoing.items():
            for middle in middles:
                for end in outgoing.get(middle, []):
                    if end != source:
                        return [source, middle, end]
        ids = [c.id for c in self.spec.constructs]
        return ids[:3] if len(ids) >= 3 else []

    def _chain_label(self, chain: List[str]) -> str:
        names = []
        for cid in chain:
            construct = self.spec.construct(cid)
            names.append(construct.name if construct else "?")
        return " → ".join(names)

    def _indirect_row(self, indirect: IndirectEffect) -> Row:
        row = Row()
        label = QLabel(self._chain_label(indirect.chain))
        theme.keep_ltr(label)
        theme.set_role(label, "mono")
        row.add(label, 8)
        significant = TCheckBox("s.significant")
        significant.setChecked(indirect.target_significant)
        row.add(significant, 2)
        row.add_delete_button()
        significant.toggled.connect(
            lambda checked: (setattr(indirect, "target_significant", checked),
                             self.edit()))
        row.removed.connect(lambda _r: self._remove_indirect(indirect))
        return row

    def _remove_indirect(self, indirect: IndirectEffect) -> None:
        self.spec.indirect_effects = [
            i for i in self.spec.indirect_effects if i is not indirect]
        self.edit(structural=True)


__all__ = ["StructuralEditor"]

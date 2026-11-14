"""
semsim.io.ai_template
=====================
A JSON file that an AI assistant fills in from a plain description of a study.

:func:`build_template` returns one self-contained document: the instructions
the assistant needs, the allowed values, and a worked example in the exact
shape the program reads.  The researcher gives that document to an assistant,
describes the data they need, and brings the filled document back.

:func:`from_text` turns the filled document into a
:class:`~semsim.core.spec.ModelSpec`.  It reports all the document's problems
together, each with its location, so the list can be handed back to the
assistant.  Model rules such as category percentages are checked once the
document itself reads cleanly, so they can appear on a second attempt.
"""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from semsim.core.errors import SemSimError
from semsim.core.spec import (
    Bilingual, Construct, ConstructOrder, ConstructRole, DemographicCategory,
    DemographicKind, DemographicRole, DemographicVariable, Dimension,
    ASSOCIATION_V_MAX, AssociationDirection, AssociationStrength,
    DemographicAssociation,
    RESPONDENT_STYLE_PRESETS,
    EffectSize, FitLevel, GROUP_D_MAX, GROUP_ETA_MAX, GroupDifference,
    GroupDirection, GroupEffect, HETEROGENEITY_PRESETS, IndirectEffect, Item,
    LatentCovariance,
    MissingMechanism, ModelSpec, ResponseStyle, ScaleKind, ScaleSpec,
)
from semsim.core.validation import validate

FORMAT = "semsim-ai-template"
TEMPLATE_VERSION = 1

#: Sections that speak to the assistant and carry no model content.
_GUIDANCE_KEYS = ("how_to_use_ar", "ai_instructions", "allowed_values")
_TOP_KEYS = (
    "format", "template_version", "filled", *_GUIDANCE_KEYS, "project",
    "constructs", "paths", "correlations", "interactions", "indirect_effects",
    "demographics", "fit_target", "significance", "realism",
)
_EFFECTS = tuple(e.value for e in EffectSize if e is not EffectSize.CUSTOM)
_STYLES = tuple(s.value for s in ResponseStyle if s is not ResponseStyle.CUSTOM)
_ROLES = (ConstructRole.INDEPENDENT.value, ConstructRole.MEDIATOR.value,
          ConstructRole.DEPENDENT.value, ConstructRole.MODERATOR.value)
_FIT_LEVELS = tuple(f.value for f in FitLevel if f is not FitLevel.CUSTOM)
_VARIATIONS = ("none", *HETEROGENEITY_PRESETS)
_STYLE_LEVELS = ("none", *RESPONDENT_STYLE_PRESETS)


@dataclass(frozen=True)
class TemplateProblem:
    """One thing wrong with a filled template, and where it is."""

    where: str
    message_en: str
    message_ar: str

    def text(self, lang: str = "en") -> str:
        """Return ``location: message`` in the requested language.

        Parameters
        ----------
        lang : str
            ``"ar"`` for Arabic; anything else gives English.

        Returns
        -------
        str
            The location followed by the message.

        References
        ----------
        Bray, T. (2017). *The JavaScript Object Notation (JSON) Data
            Interchange Format*. RFC 8259.
        """
        message = self.message_ar if lang == "ar" else self.message_en
        line = f"{self.where}: {message}" if self.where else message
        # Without the mark a location in Latin letters makes the line run
        # left to right and scrambles the Arabic after it.
        return f"\u200f{line}" if lang == "ar" else line


class AiTemplateError(SemSimError):
    """A filled template could not be turned into a model."""

    def __init__(self, problems: Sequence[TemplateProblem]) -> None:
        self.problems = list(problems)
        count = len(self.problems)
        super().__init__(
            f"The filled template has {count} problem(s).",
            f"القالب المعبّأ فيه {count} مشكلة.",
        )


@dataclass
class ImportResult:
    """A model read from a template, with anything worth a second look."""

    spec: ModelSpec
    warnings: List[TemplateProblem]


# --------------------------------------------------------------------------- #
# The template
# --------------------------------------------------------------------------- #


_INSTRUCTIONS: Dict[str, Any] = {
    "your_job": (
        "This file is the input form of SEM Simulator, a program that generates "
        "simulated (synthetic) questionnaire data from a structural equation "
        "model. The user will describe the data they need. Fill this form so it "
        "matches that description exactly."
    ),
    "how_to_work": [
        "Talk to the user in the language they write in.",
        "If something that changes the model is missing (the variables, their "
        "roles, the paths between them, or the sample size), ask a short "
        "question before answering. Use the defaults below for everything else.",
        "Before the JSON, list any value you chose yourself so the user can "
        "correct it.",
    ],
    "how_to_answer": [
        "Return the complete JSON as one code block or file, with the same "
        "top-level keys. It must be valid JSON with no comments.",
        "Keep 'format' and 'template_version' unchanged and set 'filled' to true.",
        "Replace every example value. No example construct, item or category "
        "may remain unless the user asked for it.",
        "'how_to_use_ar', 'ai_instructions' and 'allowed_values' may be kept or "
        "removed. Use only the keys shown in this file.",
        "If the user sends you a list of problems from the program, fix all of "
        "them and return the complete JSON again.",
    ],
    "names": [
        "Every 'name' uses English letters, digits and underscores, starts with "
        "a letter, is short (SAT, ENG, Gender) and is unique across constructs, "
        "dimensions and demographics. Names become column names, and the path "
        "diagram shows each construct's name under its label, so use the "
        "abbreviations the user's study uses (PU, TR, BI).",
        "Give every construct, dimension and category both 'label_ar' and "
        "'label_en'.",
    ],
    "constructs": [
        "A first-order construct has either 'items' or 'n_items'. A second-order "
        "construct has 'dimensions' instead, and each dimension has either "
        "'items' or 'n_items'.",
        "If the user supplied question wording, copy it into 'items' word for "
        "word. If not, use 'n_items' and do not invent wording unless asked.",
        "An item is a string, or an object with 'text_ar', 'text_en' and "
        "optionally 'name', 'loading' and 'reverse'.",
        "Use at least 3 items per factor. Use 3 or more dimensions per "
        "second-order construct when the theory allows it.",
        "'loading' is the standardised loading, between 0.40 and 0.95. Use "
        "0.70 to 0.85 unless told otherwise; 0.71 or above gives AVE of 0.50.",
        "'role': independent has no incoming path; mediator has incoming and "
        "outgoing paths; dependent has incoming paths; moderator is a "
        "construct that changes the strength of another path and appears as "
        "'moderator' in 'interactions'. Every moderator construct must be used "
        "in 'interactions'. Age, gender and similar variables go in "
        "'demographics', not in 'constructs'.",
        "'scale' is likert or continuous. 'scale_points' is 5 unless told "
        "otherwise.",
        "'response_style' sets where a construct's answers sit on the scale. "
        "Real attitude and evaluation scales (satisfaction, quality, trust, "
        "intention, attitudes towards something the respondent chose) lean "
        "towards agreement, so use negative_skew for them, which gives means "
        "of about 3.6 to 4.1 on a 5-point scale. Use symmetric only for a "
        "construct the user expects to sit near the middle of the scale, and "
        "positive_skew for one most respondents disagree with or rarely "
        "experience (stress, burnout, intention to leave). Choose each "
        "construct's style on its own merits.",
    ],
    "paths": [
        "'from' and 'to' are construct names. Use 'effect', or 'coefficient' for "
        "an exact standardised value; 'coefficient' takes priority.",
        "effect values: trivial 0.03, weak 0.15, medium 0.30, strong 0.50, "
        "very_strong 0.70, none 0.00.",
        "'significant': false generates a near-zero path. Use it only for a "
        "hypothesis the user wants rejected.",
        "The explained variance of each construct must stay below 1. Keep the "
        "sum of squared coefficients into one construct at or below about 0.70.",
        "Each 'paths' entry has the form {\"from\": \"PU\", \"to\": \"BI\", "
        "\"effect\": \"medium\", \"significant\": true}.",
        "'correlations' links two independent constructs; 'r' is their "
        "correlation. Each entry has the form {\"a\": \"PU\", \"b\": \"FC\", "
        "\"r\": 0.40}. Independent constructs not listed are uncorrelated.",
        "List mediation chains in 'indirect_effects' so they are reported, for "
        "example {\"chain\": [\"AGT\", \"TR\", \"BI\"], \"significant\": true}.",
        "A moderation hypothesis such as 'DL moderates the effect of PU on BI' "
        "is written in 'interactions' as {\"predictor\": \"PU\", \"moderator\": "
        "\"DL\", \"target\": \"BI\", \"effect\": \"weak\", \"significant\": "
        "true}. 'predictor' and 'target' are the two ends of the moderated "
        "path. Never write a moderation as a plain path from the moderator to "
        "the target: that is a different hypothesis.",
        "The moderated path itself (PU -> BI above) must also be listed in "
        "'paths'. Also add the moderator's own main effect on the target "
        "(DL -> BI) to 'paths', with effect weak unless the user gives a value, "
        "because a moderation model always includes it. The diagram draws "
        "the moderator above or below the model with its arrow landing on the "
        "moderated path.",
        "One moderator may moderate several paths: add one 'interactions' entry "
        "for each.",
        "A model with interactions reaches its fit target like any other, so "
        "do not warn the user that it cannot.",
    ],
    "demographics": [
        "'percent' values of one variable must sum to 100.",
        "'role': control enters the structural model through 'control_paths'; "
        "descriptive is only reported; comparison gets a t-test or ANOVA; mga "
        "gets multi-group analysis.",
        "'level' is nominal for unordered categories and ordinal for ordered "
        "ones such as education or age bands.",
        "Only role control may have 'control_paths', and it needs at least one. "
        "'to' is a construct with incoming effects (mediator or dependent).",
        "A nominal variable with 3 or more categories is dummy-coded and its "
        "first category is the reference. Each of its control paths must name "
        "another category in 'category', using the exact category label.",
        "Any other variable has one effect, so leave 'category' out.",
        "'exact_quotas' false (the default) draws each respondent at random "
        "with the category shares as weights, so the counts vary the way a "
        "collected sample's do (317, 204, 158 rather than 320, 200, 160). Set "
        "it to true only when the user says the sample was filled by quota.",
        "A comparison variable may list 'differences': how its groups differ "
        "on a construct. 'on' names the construct. 'size' is none, small, "
        "medium or large: Cohen's d of 0.2, 0.5 or 0.8 for two categories, "
        "eta squared of 0.01, 0.06 or 0.14 for three or more. Give 'value' "
        "instead of 'size' only when the user names an exact d or eta squared.",
        "'direction' ascending makes the last category score highest and "
        "descending the first, with the categories in between in order, so "
        "list the categories in the order the difference runs. For Gender "
        "with Male then Female, ascending means females score higher.",
        "'significant' true makes the t-test or ANOVA significant, false makes "
        "it non-significant; size none is never significant. When the user "
        "hypothesises differences by a demographic (for example 'satisfaction "
        "differs by gender'), make it a comparison variable and add the "
        "difference. Real demographic differences are mostly small, so use "
        "small unless the user says otherwise. Constructs without a listed "
        "difference differ only as far as they correlate with a listed one.",
        "'associations' (inside 'demographics', next to 'variables') links two "
        "demographic variables that go together in real samples, such as age "
        "and years of experience, or education and job level. 'between' names "
        "the two variables. 'strength' weak, moderate or strong is a Cramér's "
        "V of 0.10, 0.30 or 0.50; give 'value' instead only for an exact V. "
        "'direction' positive puts the last categories of both lists together "
        "(older with more experienced); negative puts the last category of "
        "the first with the first category of the second. 'significant' "
        "works as for differences; strength none keeps the two independent.",
        "Add an association when the user describes one or when it is "
        "obvious from the categories (age bands and experience bands are "
        "strongly linked). Leave unrelated variables such as gender and age "
        "unlinked.",
    ],
    "defaults": [
        "'sample_size' is required. If the user gives no number, use 300 for "
        "medium effects and 500 or more when weak effects must be significant.",
        "fit_target.level good targets RMSEA of about 0.045. Add 'rmsea', 'cfi' "
        "or 'srmr' only if the user names a value.",
        "All 'realism' percentages are 0 unless the user asks for imperfections.",
        "'realism.variation' makes the data look collected rather than "
        "simulated: constructs and items get different means and spreads, and "
        "the population loadings and paths scatter a little around the planned "
        "values, as they do in any real sample. Keep it realistic unless the "
        "user asks for textbook-clean data (none), slightly uneven data (mild) "
        "or very uneven data (strong).",
        "'realism.respondent_styles' gives each respondent a way of using the "
        "scale: some pick the ends, some keep to the middle, some agree with "
        "everything. Keep it realistic unless the user asks for clean data "
        "(none) or for strong response styles (strong).",
        "Empty optional sections may be left as [] or {}.",
    ],
}

_ALLOWED_VALUES: Dict[str, Any] = {
    "constructs.role": list(_ROLES),
    "constructs.scale": [ScaleKind.LIKERT.value, ScaleKind.CONTINUOUS.value],
    "constructs.response_style": list(_STYLES),
    "effect": list(_EFFECTS),
    "demographics.level": [k.value for k in DemographicKind],
    "demographics.role": [r.value for r in DemographicRole],
    "demographics.differences.size": [g.value for g in GroupEffect
                                      if g is not GroupEffect.CUSTOM],
    "demographics.differences.direction": [d.value for d in GroupDirection],
    "demographics.associations.strength": [a.value for a in AssociationStrength
                                           if a is not AssociationStrength.CUSTOM],
    "demographics.associations.direction": [d.value for d in AssociationDirection],
    "fit_target.level": list(_FIT_LEVELS),
    "realism.missing_mechanism": [MissingMechanism.MCAR.value,
                                  MissingMechanism.MAR.value],
    "realism.variation": list(_VARIATIONS),
    "realism.respondent_styles": list(_STYLE_LEVELS),
}


def build_template() -> Dict[str, Any]:
    """Return the template: instructions, allowed values and a worked example.

    The example is a valid model, so the document shows the assistant every
    section in the exact form the program reads.

    Returns
    -------
    dict
        A JSON-safe dictionary.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    return {
        "format": FORMAT,
        "template_version": TEMPLATE_VERSION,
        "filled": False,
        "how_to_use_ar": (
            "أرسل هذا الملف إلى مساعد الذكاء الاصطناعي واشرح له البيانات التي "
            "تحتاجها، ثم استورد الملف المعبّأ من قائمة ملف ← قالب الذكاء الاصطناعي."
        ),
        "ai_instructions": _INSTRUCTIONS,
        "allowed_values": _ALLOWED_VALUES,
        "project": {
            "title_ar": "مثال: جودة الخدمة والرضا والولاء",
            "title_en": "Example: service quality, satisfaction, loyalty",
            "description": "Replace this example with the user's study.",
            "sample_size": 400,
        },
        "constructs": [
            {
                "name": "SQ", "label_ar": "جودة الخدمة",
                "label_en": "Service quality", "role": "independent",
                "loading": 0.78, "scale": "likert", "scale_points": 5,
                "response_style": "negative_skew",
                "dimensions": [
                    {"name": "SQ_REL", "label_ar": "الاعتمادية",
                     "label_en": "Reliability", "second_order_loading": 0.80,
                     "n_items": 3},
                    {"name": "SQ_RES", "label_ar": "الاستجابة",
                     "label_en": "Responsiveness", "second_order_loading": 0.80,
                     "n_items": 3},
                    {"name": "SQ_EMP", "label_ar": "التعاطف",
                     "label_en": "Empathy", "second_order_loading": 0.75,
                     "n_items": 3},
                ],
            },
            {
                "name": "SAT", "label_ar": "الرضا", "label_en": "Satisfaction",
                "role": "mediator", "loading": 0.80, "scale": "likert",
                "scale_points": 5, "response_style": "negative_skew",
                "items": [
                    {"text_ar": "أنا راضٍ عن الخدمة بشكل عام.",
                     "text_en": "Overall, I am satisfied with the service."},
                    {"text_ar": "الخدمة تلبي توقعاتي.",
                     "text_en": "The service meets my expectations."},
                    {"text_ar": "لست سعيدًا بتجربتي مع الخدمة.",
                     "text_en": "I am unhappy with my experience.",
                     "reverse": True, "loading": 0.72},
                ],
            },
            {
                "name": "LOY", "label_ar": "الولاء", "label_en": "Loyalty",
                "role": "dependent", "loading": 0.76, "scale": "likert",
                "scale_points": 5, "response_style": "symmetric", "n_items": 4,
            },
        ],
        "paths": [
            {"from": "SQ", "to": "SAT", "effect": "strong", "significant": True},
            {"from": "SAT", "to": "LOY", "coefficient": 0.42,
             "significant": True},
            {"from": "SQ", "to": "LOY", "effect": "weak", "significant": True},
        ],
        "correlations": [],
        "interactions": [],
        "indirect_effects": [{"chain": ["SQ", "SAT", "LOY"],
                              "significant": True}],
        "demographics": {
            "exact_quotas": False,
            "variables": [
                {"name": "Gender", "label_ar": "النوع", "label_en": "Gender",
                 "level": "nominal", "role": "control",
                 "categories": [
                     {"label_ar": "ذكر", "label_en": "Male", "percent": 60},
                     {"label_ar": "أنثى", "label_en": "Female", "percent": 40},
                 ],
                 "control_paths": [{"to": "LOY", "effect": "weak",
                                    "significant": True}]},
                {"name": "Sector", "label_ar": "القطاع", "label_en": "Sector",
                 "level": "nominal", "role": "control",
                 "categories": [
                     {"label_ar": "عام", "label_en": "Public", "percent": 50},
                     {"label_ar": "خاص", "label_en": "Private", "percent": 35},
                     {"label_ar": "مختلط", "label_en": "Mixed", "percent": 15},
                 ],
                 "control_paths": [{"to": "SAT", "category": "Private",
                                    "coefficient": 0.12, "significant": True}]},
                {"name": "Education", "label_ar": "المستوى التعليمي",
                 "label_en": "Education", "level": "ordinal",
                 "role": "comparison",
                 "categories": [
                     {"label_ar": "بكالوريوس", "label_en": "Bachelor's",
                      "percent": 70},
                     {"label_ar": "دراسات عليا", "label_en": "Postgraduate",
                      "percent": 30},
                 ],
                 "differences": [{"on": "SAT", "size": "small",
                                  "direction": "ascending",
                                  "significant": True}]},
            ],
            "associations": [{"between": ["Education", "Sector"],
                              "strength": "weak", "direction": "negative",
                              "significant": True}],
        },
        "fit_target": {"level": "good"},
        "significance": {"alpha": 0.05},
        "realism": {
            "missing_percent": 0, "missing_mechanism": "mcar",
            "careless_percent": 0, "outlier_percent": 0,
            "common_method_loading": 0, "skewness": -0.3, "kurtosis": 0.2,
            "variation": "realistic", "respondent_styles": "realistic",
        },
    }


def template_text() -> str:
    """Return the template as UTF-8-ready, indented JSON text.

    Returns
    -------
    str
        The document, with Arabic left readable.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    return json.dumps(build_template(), ensure_ascii=False, indent=2)


def save_template(path: "str | os.PathLike[str]") -> str:
    """Write the template to ``path`` and return the path written.

    Parameters
    ----------
    path : str or os.PathLike
        Destination; ``.json`` is appended when missing.

    Returns
    -------
    str
        The path actually written.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    target = str(path)
    if not target.lower().endswith(".json"):
        target += ".json"
    try:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(template_text())
    except OSError as exc:
        raise SemSimError(f"Could not write the template: {exc}",
                          f"تعذّر حفظ القالب: {exc}") from exc
    return target


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def parse_text(text: str) -> Dict[str, Any]:
    """Read an assistant's answer as a JSON object.

    Assistants often wrap the document in a code fence or add a sentence
    around it, so when the text is not JSON as it stands, the span from the
    first ``{`` to the last ``}`` is tried.

    Parameters
    ----------
    text : str
        The file content or pasted answer.

    Returns
    -------
    dict
        The parsed document.

    Raises
    ------
    AiTemplateError
        When no JSON object can be read.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    body = text.lstrip("﻿").strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as first:
        start, end = body.find("{"), body.rfind("}")
        failure = first
        payload = None
        if 0 <= start < end:
            try:
                payload = json.loads(body[start:end + 1])
            except json.JSONDecodeError as second:
                failure = second
        if payload is None:
            raise AiTemplateError([TemplateProblem(
                f"line {failure.lineno}, column {failure.colno}",
                f"This is not valid JSON: {failure.msg}.",
                f"هذا ليس JSON صالحًا: {failure.msg}.",
            )]) from failure
    if not isinstance(payload, dict):
        raise AiTemplateError([TemplateProblem(
            "", "The document must be one JSON object.",
            "يجب أن يكون المستند كائن JSON واحدًا.")])
    return payload


class _Reader:
    """Typed access to the document that records problems instead of raising."""

    def __init__(self) -> None:
        self.problems: List[TemplateProblem] = []
        self.warnings: List[TemplateProblem] = []

    def fail(self, where: str, en: str, ar: str) -> None:
        self.problems.append(TemplateProblem(where, en, ar))

    def keys(self, obj: Dict[str, Any], allowed: Iterable[str],
             where: str) -> None:
        names = tuple(allowed)
        for key in obj:
            if key.startswith("_") or key in names:
                continue
            hint = difflib.get_close_matches(key, names, n=1)
            more_en = f" Did you mean '{hint[0]}'?" if hint else ""
            more_ar = f" هل تقصد '{hint[0]}'؟" if hint else ""
            self.fail(f"{where}.{key}" if where else key,
                      f"This key is not part of the template.{more_en}",
                      f"هذا المفتاح ليس جزءًا من القالب.{more_ar}")

    def text(self, obj: Dict[str, Any], key: str, where: str,
             required: bool = False) -> str:
        value = obj.get(key)
        if value is None or value == "":
            if required:
                self.fail(f"{where}.{key}", "A value is required.",
                          "هذه القيمة مطلوبة.")
            return ""
        if not isinstance(value, str):
            self.fail(f"{where}.{key}", "This must be text.", "يجب أن يكون نصًا.")
            return ""
        return value.strip()

    def number(self, obj: Dict[str, Any], key: str, where: str, low: float,
               high: float, default: Optional[float] = None) -> Optional[float]:
        if obj.get(key) is None:
            return default
        value = obj[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(f"{where}.{key}", "This must be a number.",
                      "يجب أن يكون رقمًا.")
            return default
        if not low <= value <= high:
            self.fail(f"{where}.{key}",
                      f"{value} is outside the range {low} to {high}.",
                      f"القيمة {value} خارج النطاق من {low} إلى {high}.")
            return default
        return float(value)

    def integer(self, obj: Dict[str, Any], key: str, where: str, low: int,
                high: int) -> Optional[int]:
        value = self.number(obj, key, where, low, high)
        if value is None:
            return None
        if int(value) != value:
            self.fail(f"{where}.{key}", "This must be a whole number.",
                      "يجب أن يكون رقمًا صحيحًا.")
            return None
        return int(value)

    def flag(self, obj: Dict[str, Any], key: str, where: str,
             default: bool) -> bool:
        value = obj.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            self.fail(f"{where}.{key}", "This must be true or false.",
                      "يجب أن تكون true أو false.")
            return default
        return value

    def choice(self, obj: Dict[str, Any], key: str, where: str,
               allowed: Sequence[str], default: str) -> str:
        value = obj.get(key)
        if value is None:
            return default
        if value not in allowed:
            options = ", ".join(allowed)
            self.fail(f"{where}.{key}",
                      f"'{value}' is not allowed. Use one of: {options}.",
                      f"القيمة '{value}' غير مسموحة. استخدم إحدى القيم: {options}.")
            return default
        return value

    def objects(self, obj: Dict[str, Any], key: str, where: str
                ) -> List[Tuple[str, Dict[str, Any]]]:
        value = obj.get(key)
        path = f"{where}.{key}" if where else key
        if value is None:
            return []
        if not isinstance(value, list):
            self.fail(path, "This must be a list.", "يجب أن تكون قائمة.")
            return []
        out: List[Tuple[str, Dict[str, Any]]] = []
        for index, entry in enumerate(value):
            if isinstance(entry, dict):
                out.append((f"{path}[{index}]", entry))
            else:
                self.fail(f"{path}[{index}]", "This must be an object.",
                          "يجب أن يكون كائنًا.")
        return out

    def section(self, obj: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = obj.get(key)
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.fail(key, "This must be an object.", "يجب أن يكون كائنًا.")
            return {}
        return value

    def effect(self, obj: Dict[str, Any], where: str, default: str
               ) -> Tuple[EffectSize, Optional[float]]:
        coefficient = self.number(obj, "coefficient", where, -0.99, 0.99)
        if coefficient is not None:
            return EffectSize.CUSTOM, coefficient
        return EffectSize(self.choice(obj, "effect", where, _EFFECTS, default)), None

    def reference(self, value: Any, known: Dict[str, str], where: str
                  ) -> Optional[str]:
        if isinstance(value, str) and value.strip() in known:
            return known[value.strip()]
        hint = difflib.get_close_matches(str(value), list(known), n=1)
        more_en = f" Did you mean '{hint[0]}'?" if hint else ""
        more_ar = f" هل تقصد '{hint[0]}'؟" if hint else ""
        self.fail(where, f"No construct is named '{value}'.{more_en}",
                  f"لا يوجد متغير كامن باسم '{value}'.{more_ar}")
        return None


def _label(reader: _Reader, obj: Dict[str, Any], where: str,
           fallback: str = "") -> Bilingual:
    arabic = reader.text(obj, "label_ar", where)
    english = reader.text(obj, "label_en", where)
    return Bilingual(ar=arabic or english or fallback,
                     en=english or arabic or fallback)


def _read_items(reader: _Reader, owner: Dict[str, Any], where: str, stem: str,
                label: Bilingual, loading: float, scale: ScaleSpec,
                ) -> List[Item]:
    """Build a factor's items from ``items`` or ``n_items``."""
    has_list, has_count = owner.get("items") is not None, owner.get("n_items") is not None
    if has_list == has_count:
        reader.fail(where, "Give either 'items' or 'n_items', exactly one of them.",
                    "اذكر 'items' أو 'n_items'، واحدًا منهما فقط.")
        return []

    def fresh_scale() -> ScaleSpec:
        return ScaleSpec(kind=scale.kind, n_categories=scale.n_categories,
                         response_style=scale.response_style)

    if has_count:
        count = reader.integer(owner, "n_items", where, 1, 100) or 0
        return [Item(name=f"{stem}{k}",
                     label=Bilingual(ar=f"{label.ar} {k}", en=f"{label.en} {k}"),
                     loading=loading, scale=fresh_scale())
                for k in range(1, count + 1)]

    entries = owner["items"]
    if not isinstance(entries, list):
        reader.fail(f"{where}.items", "This must be a list.", "يجب أن تكون قائمة.")
        return []
    items: List[Item] = []
    for index, entry in enumerate(entries, start=1):
        path = f"{where}.items[{index - 1}]"
        if isinstance(entry, str):
            items.append(Item(name=f"{stem}{index}",
                              label=Bilingual(ar=entry, en=entry),
                              loading=loading, scale=fresh_scale()))
            continue
        if not isinstance(entry, dict):
            reader.fail(path, "An item must be text or an object.",
                        "يجب أن يكون السؤال نصًا أو كائنًا.")
            continue
        reader.keys(entry, ("name", "text_ar", "text_en", "loading", "reverse"), path)
        arabic = reader.text(entry, "text_ar", path)
        english = reader.text(entry, "text_en", path)
        if not arabic and not english:
            reader.fail(path, "An item needs 'text_ar' or 'text_en'.",
                        "السؤال يحتاج إلى 'text_ar' أو 'text_en'.")
        items.append(Item(
            name=reader.text(entry, "name", path) or f"{stem}{index}",
            label=Bilingual(ar=arabic or english, en=english or arabic),
            loading=float(reader.number(entry, "loading", path, 0.05, 0.99, loading)),
            reverse_coded=reader.flag(entry, "reverse", path, False),
            scale=fresh_scale()))
    return items


def _read_construct(reader: _Reader, entry: Dict[str, Any], where: str,
                    used_names: Dict[str, str]) -> Optional[Construct]:
    reader.keys(entry, ("name", "label_ar", "label_en", "role", "loading", "scale",
                        "scale_points", "response_style", "items", "n_items",
                        "dimensions"), where)
    name = reader.text(entry, "name", where, required=True)
    if not name:
        return None

    def claim(value: str, path: str) -> None:
        if value in used_names:
            reader.fail(path, f"The name '{value}' is used more than once.",
                        f"الاسم '{value}' مستخدم أكثر من مرة.")
        used_names[value] = path

    claim(name, f"{where}.name")
    label = _label(reader, entry, where, name)
    role = reader.choice(entry, "role", where, _ROLES, ConstructRole.INDEPENDENT.value)
    loading = float(reader.number(entry, "loading", where, 0.05, 0.99, 0.75))
    kind = reader.choice(entry, "scale", where,
                         (ScaleKind.LIKERT.value, ScaleKind.CONTINUOUS.value),
                         ScaleKind.LIKERT.value)
    points = reader.integer(entry, "scale_points", where, 2, 11) or 5
    style = reader.choice(entry, "response_style", where, _STYLES,
                          ResponseStyle.SYMMETRIC.value)
    scale = ScaleSpec(kind=ScaleKind(kind), n_categories=points,
                      response_style=ResponseStyle(style))
    construct = Construct(name=name, label=label, role=ConstructRole(role))

    if entry.get("dimensions") is None:
        construct.items = _read_items(reader, entry, where, name, label, loading, scale)
        return construct

    if entry.get("items") is not None or entry.get("n_items") is not None:
        reader.fail(where, "Give 'dimensions' or items, not both.",
                    "اذكر 'dimensions' أو الأسئلة، وليس الاثنين معًا.")
    construct.order = ConstructOrder.SECOND
    for position, (path, raw) in enumerate(
            reader.objects(entry, "dimensions", where), start=1):
        reader.keys(raw, ("name", "label_ar", "label_en", "second_order_loading",
                          "items", "n_items"), path)
        dimension_name = reader.text(raw, "name", path, required=True) or f"{name}_D{position}"
        claim(dimension_name, f"{path}.name")
        dimension_label = _label(reader, raw, path, dimension_name)
        construct.dimensions.append(Dimension(
            name=dimension_name, label=dimension_label,
            second_order_loading=float(reader.number(
                raw, "second_order_loading", path, 0.05, 0.99, 0.80)),
            items=_read_items(reader, raw, path, f"{name}{position}",
                              dimension_label, loading, scale)))
    return construct


def _read_demographics(reader: _Reader, spec: ModelSpec, payload: Dict[str, Any],
                       constructs: Dict[str, str], used_names: Dict[str, str]) -> None:
    section = reader.section(payload, "demographics")
    reader.keys(section, ("exact_quotas", "variables", "associations"),
                "demographics")
    spec.quota_exact = reader.flag(section, "exact_quotas", "demographics", False)

    for where, entry in reader.objects(section, "variables", "demographics"):
        reader.keys(entry, ("name", "label_ar", "label_en", "level", "role",
                            "categories", "control_paths", "differences"), where)
        name = reader.text(entry, "name", where, required=True)
        if not name:
            continue
        if name in used_names:
            reader.fail(f"{where}.name", f"The name '{name}' is used more than once.",
                        f"الاسم '{name}' مستخدم أكثر من مرة.")
        used_names[name] = where
        level = reader.choice(entry, "level", where, [k.value for k in DemographicKind],
                              DemographicKind.NOMINAL.value)
        role = reader.choice(entry, "role", where, [r.value for r in DemographicRole],
                             DemographicRole.DESCRIPTIVE.value)
        variable = DemographicVariable(
            name=name, label=_label(reader, entry, where, name),
            kind=DemographicKind(level), role=DemographicRole(role))
        for path, raw in reader.objects(entry, "categories", where):
            reader.keys(raw, ("label_ar", "label_en", "percent"), path)
            if not (raw.get("label_ar") or raw.get("label_en")):
                reader.fail(path, "A category needs 'label_ar' or 'label_en'.",
                            "الفئة تحتاج إلى 'label_ar' أو 'label_en'.")
            if raw.get("percent") is None:
                reader.fail(f"{path}.percent", "A value is required.",
                            "هذه القيمة مطلوبة.")
            percent = reader.number(raw, "percent", path, 0, 100, 0.0)
            variable.categories.append(DemographicCategory(
                label=_label(reader, raw, path), proportion=float(percent) / 100.0))
        spec.demographics.append(variable)
        _read_control_paths(reader, spec, entry, where, variable, constructs)
        _read_differences(reader, entry, where, variable, constructs)
    _read_associations(reader, spec, section)


def _read_associations(reader: _Reader, spec: ModelSpec,
                       section: Dict[str, Any]) -> None:
    variables = {v.name: v for v in spec.demographics}
    strengths = [a.value for a in AssociationStrength
                 if a is not AssociationStrength.CUSTOM]
    seen: Dict[frozenset, str] = {}
    for path, raw in reader.objects(section, "associations", "demographics"):
        reader.keys(raw, ("between", "strength", "value", "direction",
                          "significant"), path)
        between = raw.get("between")
        if (not isinstance(between, list) or len(between) != 2
                or not all(isinstance(name, str) for name in between)):
            reader.fail(f"{path}.between",
                        "Give the names of two demographic variables.",
                        "اذكر اسمي متغيرين ديموغرافيين.")
            continue
        names = [name.strip() for name in between]
        missing = [name for name in names if name not in variables]
        for name in missing:
            hint = difflib.get_close_matches(name, list(variables), n=1)
            more_en = f" Did you mean '{hint[0]}'?" if hint else ""
            more_ar = f" هل تقصد '{hint[0]}'؟" if hint else ""
            reader.fail(f"{path}.between",
                        f"No demographic variable is named '{name}'.{more_en}",
                        f"لا يوجد متغير ديموغرافي باسم '{name}'.{more_ar}")
        if missing:
            continue
        if names[0] == names[1]:
            reader.fail(f"{path}.between", "Name two different variables.",
                        "اذكر متغيرين مختلفين.")
            continue
        strength = AssociationStrength(reader.choice(
            raw, "strength", path, strengths, AssociationStrength.MODERATE.value))
        value = reader.number(raw, "value", path, 0.0, ASSOCIATION_V_MAX)
        direction = reader.choice(raw, "direction", path,
                                  [d.value for d in AssociationDirection],
                                  AssociationDirection.POSITIVE.value)
        significant = reader.flag(raw, "significant", path, True)
        pair = frozenset(names)
        if pair in seen:
            reader.fail(f"{path}.between",
                        f"These two are already linked in {seen[pair]}.",
                        f"هذان المتغيران مربوطان بالفعل في {seen[pair]}.")
            continue
        seen[pair] = path
        association = DemographicAssociation(
            first=variables[names[0]].id, second=variables[names[1]].id,
            strength=strength, direction=AssociationDirection(direction),
            target_significant=significant)
        if value is not None:
            association.strength = AssociationStrength.CUSTOM
            association.value = float(value)
        spec.demographic_associations.append(association)


def _read_differences(reader: _Reader, entry: Dict[str, Any], where: str,
                      variable: DemographicVariable,
                      constructs: Dict[str, str]) -> None:
    entries = reader.objects(entry, "differences", where)
    if entries and variable.role is not DemographicRole.COMPARISON:
        reader.fail(f"{where}.differences",
                    "Only a variable with role 'comparison' can have differences.",
                    "المتغير ذو الدور 'comparison' فقط يمكن أن تكون له فروق.")
        return
    seen: Dict[str, str] = {}
    sizes = [g.value for g in GroupEffect if g is not GroupEffect.CUSTOM]
    ceiling = GROUP_D_MAX if len(variable.categories) <= 2 else GROUP_ETA_MAX
    for path, raw in entries:
        reader.keys(raw, ("on", "size", "value", "direction", "significant"), path)
        target = reader.reference(raw.get("on"), constructs, f"{path}.on")
        size = GroupEffect(reader.choice(raw, "size", path, sizes,
                                         GroupEffect.SMALL.value))
        value = reader.number(raw, "value", path, 0.0, ceiling)
        direction = reader.choice(raw, "direction", path,
                                  [d.value for d in GroupDirection],
                                  GroupDirection.ASCENDING.value)
        significant = reader.flag(raw, "significant", path, True)
        if target is None:
            continue
        if target in seen:
            reader.fail(f"{path}.on",
                        f"'{raw.get('on')}' already has a difference in {seen[target]}.",
                        f"'{raw.get('on')}' له فرق بالفعل في {seen[target]}.")
            continue
        seen[target] = path
        difference = GroupDifference(
            target=target, size=size, direction=GroupDirection(direction),
            target_significant=significant)
        if value is not None:
            difference.size, difference.value = GroupEffect.CUSTOM, float(value)
        variable.group_differences.append(difference)


def _read_control_paths(reader: _Reader, spec: ModelSpec, entry: Dict[str, Any],
                        where: str, variable: DemographicVariable,
                        constructs: Dict[str, str]) -> None:
    entries = reader.objects(entry, "control_paths", where)
    if entries and variable.role is not DemographicRole.CONTROL:
        reader.fail(f"{where}.control_paths",
                    "Only a variable with role 'control' can have control paths.",
                    "المتغير ذو الدور 'control' فقط يمكن أن تكون له مسارات ضبط.")
        return
    for path, raw in entries:
        reader.keys(raw, ("to", "category", "effect", "coefficient", "significant"), path)
        target = reader.reference(raw.get("to"), constructs, f"{path}.to")
        term = ""
        if variable.is_dummy_coded:
            term = _dummy_term(reader, raw, path, variable)
        elif raw.get("category") is not None:
            reader.fail(f"{path}.category",
                        "This variable has one effect, so remove 'category'.",
                        "هذا المتغير له أثر واحد، لذا احذف 'category'.")
        effect, coefficient = reader.effect(raw, path, EffectSize.WEAK.value)
        significant = reader.flag(raw, "significant", path, True)
        if target is None or term is None:
            continue
        spec.add_control_path(
            variable, target, term=term, coefficient=coefficient,
            effect_size=effect if coefficient is None else EffectSize.WEAK,
            target_significant=significant)


def _dummy_term(reader: _Reader, raw: Dict[str, Any], path: str,
                variable: DemographicVariable) -> Optional[str]:
    """Key of the dummy term named by ``category``, or ``None`` after a problem."""
    wanted = raw.get("category")
    first = variable.categories[0].label
    labels = [c.label.en for c in variable.categories[1:]]
    options = ", ".join(labels)
    if wanted in (first.ar, first.en):
        reader.fail(f"{path}.category",
                    f"'{wanted}' is the reference category. Use one of: {options}.",
                    f"'{wanted}' هي الفئة المرجعية. استخدم إحدى الفئات: {options}.")
        return None
    for term in variable.terms():
        label = variable.categories[term.category].label
        if wanted in (label.ar, label.en):
            return term.key
    reader.fail(f"{path}.category",
                f"This variable has 3 or more categories, so name one of: {options}.",
                f"هذا المتغير له 3 فئات أو أكثر، لذا اذكر إحدى الفئات: {options}.")
    return None


def _read_structure(reader: _Reader, spec: ModelSpec, payload: Dict[str, Any],
                    known: Dict[str, str]) -> None:
    for where, raw in reader.objects(payload, "paths", ""):
        reader.keys(raw, ("from", "to", "effect", "coefficient", "significant"), where)
        source = reader.reference(raw.get("from"), known, f"{where}.from")
        target = reader.reference(raw.get("to"), known, f"{where}.to")
        effect, coefficient = reader.effect(raw, where, EffectSize.MEDIUM.value)
        significant = reader.flag(raw, "significant", where, True)
        if source and target:
            spec.add_path(source, target, effect_size=effect, coefficient=coefficient,
                          significant=significant)

    for where, raw in reader.objects(payload, "correlations", ""):
        reader.keys(raw, ("a", "b", "r"), where)
        first = reader.reference(raw.get("a"), known, f"{where}.a")
        second = reader.reference(raw.get("b"), known, f"{where}.b")
        if raw.get("r") is None:
            reader.fail(f"{where}.r", "A value is required.", "هذه القيمة مطلوبة.")
        value = reader.number(raw, "r", where, -0.99, 0.99, 0.0)
        if first and second:
            spec.latent_covariances.append(LatentCovariance(
                construct_a=first, construct_b=second, correlation=float(value)))

    drawn = {(p.source, p.target) for p in spec.paths}
    for where, raw in reader.objects(payload, "interactions", ""):
        reader.keys(raw, ("predictor", "moderator", "target", "effect", "coefficient",
                          "significant"), where)
        ids = [reader.reference(raw.get(k), known, f"{where}.{k}")
               for k in ("predictor", "moderator", "target")]
        effect, coefficient = reader.effect(raw, where, EffectSize.WEAK.value)
        significant = reader.flag(raw, "significant", where, True)
        if all(ids) and (ids[0], ids[2]) not in drawn:
            # The interaction modifies the predictor -> target path; without
            # that path there is nothing to moderate and nothing to draw on.
            reader.fail(
                f"{where}.predictor",
                f"Add the moderated path {raw.get('predictor')} -> "
                f"{raw.get('target')} to 'paths'.",
                f"أضف المسار الذي يعدّله المعدِّل {raw.get('predictor')} -> "
                f"{raw.get('target')} إلى 'paths'.")
        if all(ids):
            spec.add_interaction(ids[0], ids[1], ids[2], effect_size=effect,
                                 coefficient=coefficient, significant=significant)

    moderating = {ix.moderator for ix in spec.interactions}
    for construct in spec.constructs:
        if (construct.role is ConstructRole.MODERATOR
                and construct.id not in moderating):
            reader.fail(
                f"constructs.{construct.name}.role",
                f"{construct.name} is a moderator but moderates nothing. Add "
                "an 'interactions' entry naming the path it moderates, or give "
                "it role independent if it is an ordinary predictor.",
                f"{construct.name} دوره معدِّل لكنه لا يعدّل أي علاقة. أضف عنصرًا "
                "في 'interactions' يحدد المسار الذي يعدّله، أو اجعل دوره "
                "independent إذا كان متغيرًا مستقلًا عاديًا.")

    for where, raw in reader.objects(payload, "indirect_effects", ""):
        reader.keys(raw, ("chain", "significant"), where)
        chain = raw.get("chain")
        if not isinstance(chain, list) or len(chain) < 3:
            reader.fail(f"{where}.chain", "This must list at least 3 construct names.",
                        "يجب أن تذكر 3 متغيرات كامنة على الأقل.")
            continue
        ids = [reader.reference(name, known, f"{where}.chain[{i}]")
               for i, name in enumerate(chain)]
        significant = reader.flag(raw, "significant", where, True)
        if all(ids):
            spec.indirect_effects.append(IndirectEffect(
                chain=ids, target_significant=significant,
                label=" -> ".join(str(name) for name in chain)))


def _read_settings(reader: _Reader, spec: ModelSpec, payload: Dict[str, Any]) -> None:
    fit = reader.section(payload, "fit_target")
    reader.keys(fit, ("level", "rmsea", "cfi", "srmr"), "fit_target")
    spec.fit_target.level = FitLevel(reader.choice(
        fit, "level", "fit_target", _FIT_LEVELS, FitLevel.GOOD.value))
    rmsea = reader.number(fit, "rmsea", "fit_target", 0.0, 0.3)
    if rmsea is not None:
        spec.fit_target.level = FitLevel.CUSTOM
        spec.fit_target.rmsea = rmsea
    spec.fit_target.cfi = reader.number(fit, "cfi", "fit_target", 0.0, 1.0)
    spec.fit_target.srmr = reader.number(fit, "srmr", "fit_target", 0.0, 0.3)

    significance = reader.section(payload, "significance")
    reader.keys(significance, ("alpha",), "significance")
    spec.significance_target.alpha = float(reader.number(
        significance, "alpha", "significance", 0.001, 0.2, 0.05))

    realism = reader.section(payload, "realism")
    reader.keys(realism, ("missing_percent", "missing_mechanism", "careless_percent",
                          "outlier_percent", "common_method_loading", "skewness",
                          "kurtosis", "variation", "respondent_styles"), "realism")
    missing = float(reader.number(realism, "missing_percent", "realism", 0, 50, 0.0))
    mechanism = reader.choice(
        realism, "missing_mechanism", "realism",
        (MissingMechanism.MCAR.value, MissingMechanism.MAR.value),
        MissingMechanism.MCAR.value)
    spec.noise.missing_proportion = missing / 100.0
    spec.noise.missing_mechanism = (MissingMechanism(mechanism) if missing > 0
                                    else MissingMechanism.NONE)
    spec.noise.careless_proportion = float(reader.number(
        realism, "careless_percent", "realism", 0, 50, 0.0)) / 100.0
    spec.noise.outlier_proportion = float(reader.number(
        realism, "outlier_percent", "realism", 0, 20, 0.0)) / 100.0
    method = float(reader.number(realism, "common_method_loading", "realism", 0, 0.9, 0.0))
    spec.common_method_bias.enabled = method > 0
    if method > 0:
        spec.common_method_bias.loading = method
    spec.distribution.default_skewness = float(reader.number(
        realism, "skewness", "realism", -3, 3, 0.0))
    spec.distribution.default_kurtosis = float(reader.number(
        realism, "kurtosis", "realism", -2, 20, 0.0))
    spec.heterogeneity.apply_preset(reader.choice(
        realism, "variation", "realism", _VARIATIONS, "realistic"))
    spec.respondent_styles.apply_preset(reader.choice(
        realism, "respondent_styles", "realism", _STYLE_LEVELS, "realistic"))


def from_payload(payload: Dict[str, Any]) -> ImportResult:
    """Build a model from a filled template.

    Parameters
    ----------
    payload : dict
        The parsed document.

    Returns
    -------
    ImportResult
        The model and any warnings.

    Raises
    ------
    AiTemplateError
        Carrying every problem found, each with its location.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    reader = _Reader()
    if payload.get("format") != FORMAT:
        raise AiTemplateError([TemplateProblem(
            "format", f"This must be '{FORMAT}'. Is this the program's template?",
            f"يجب أن تكون القيمة '{FORMAT}'. هل هذا هو قالب البرنامج؟")])
    if payload.get("template_version") != TEMPLATE_VERSION:
        reader.fail("template_version", f"This must be {TEMPLATE_VERSION}.",
                    f"يجب أن تكون القيمة {TEMPLATE_VERSION}.")
    reader.keys(payload, _TOP_KEYS, "")
    if payload.get("filled") is not True:
        reader.warnings.append(TemplateProblem(
            "filled", "'filled' is not true, so this may still be the example.",
            "قيمة 'filled' ليست true، فقد يكون هذا ما زال المثال الأصلي."))

    project = reader.section(payload, "project")
    reader.keys(project, ("title_ar", "title_en", "description", "sample_size", "seed"),
                "project")
    spec = ModelSpec()
    arabic, english = (reader.text(project, "title_ar", "project"),
                       reader.text(project, "title_en", "project"))
    if arabic or english:
        spec.title = Bilingual(ar=arabic or english, en=english or arabic)
    spec.description = reader.text(project, "description", "project")
    if project.get("sample_size") is None:
        reader.fail("project.sample_size", "A value is required.", "هذه القيمة مطلوبة.")
    size = reader.integer(project, "sample_size", "project", 30, 1000000)
    if size is not None:
        spec.power.n = size
        spec.power.mode = "fixed"
    seed = reader.integer(project, "seed", "project", 0, 2147483647)
    if seed is not None:
        spec.seed = seed

    used_names: Dict[str, str] = {}
    entries = reader.objects(payload, "constructs", "")
    if not entries:
        reader.fail("constructs", "At least one construct is required.",
                    "مطلوب متغير كامن واحد على الأقل.")
    for where, entry in entries:
        construct = _read_construct(reader, entry, where, used_names)
        if construct is not None:
            spec.constructs.append(construct)
    known = {c.name: c.id for c in spec.constructs}
    _read_structure(reader, spec, payload, known)
    _read_demographics(reader, spec, payload, known, used_names)
    _read_settings(reader, spec, payload)

    if reader.problems:
        raise AiTemplateError(reader.problems)

    spec.ensure_unique_item_names()
    report = validate(spec)
    errors = [TemplateProblem(d.code, d.message_en, d.message_ar) for d in report.errors()]
    if errors:
        raise AiTemplateError(errors)
    reader.warnings.extend(TemplateProblem(d.code, d.message_en, d.message_ar)
                           for d in report.warnings())
    return ImportResult(spec=spec, warnings=reader.warnings)


def from_text(text: str) -> ImportResult:
    """Build a model from an assistant's answer.

    Parameters
    ----------
    text : str
        File content or pasted answer.

    Returns
    -------
    ImportResult
        The model and any warnings.

    Raises
    ------
    AiTemplateError
        Carrying every problem found.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    return from_payload(parse_text(text))


def load_file(path: "str | os.PathLike[str]") -> ImportResult:
    """Build a model from a filled template on disk.

    Parameters
    ----------
    path : str or os.PathLike
        The file to read.

    Returns
    -------
    ImportResult
        The model and any warnings.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    try:
        with open(str(path), "r", encoding="utf-8-sig") as handle:
            return from_text(handle.read())
    except (OSError, UnicodeDecodeError) as exc:
        raise SemSimError(f"Could not read the file: {exc}",
                          f"تعذّر قراءة الملف: {exc}") from exc


def problems_for_ai(problems: Sequence[TemplateProblem]) -> str:
    """Return a problem list worded to be pasted back to the assistant.

    Parameters
    ----------
    problems : sequence of TemplateProblem
        The problems reported by the program.

    Returns
    -------
    str
        A request followed by one line per problem.

    References
    ----------
    Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange
        Format*. RFC 8259.
    """
    lines = "\n".join(f"- {p.text('en')}" for p in problems)
    return ("SEM Simulator rejected the template for the reasons below. Fix all "
            f"of them and return the complete JSON again.\n{lines}")


__all__ = [
    "FORMAT", "TEMPLATE_VERSION", "TemplateProblem", "AiTemplateError",
    "ImportResult", "build_template", "template_text", "save_template",
    "parse_text", "from_payload", "from_text", "load_file", "problems_for_ai",
]

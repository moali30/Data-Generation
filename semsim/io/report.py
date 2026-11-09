"""
semsim.io.report
================
A self-contained bilingual HTML report.

Everything is inline -- styles, charts, fonts fall back to the system stack --
so the file can be emailed, opened offline, or archived beside a thesis
without a single external request.  Charts are hand-written SVG rather than
rendered images, which keeps them sharp at any zoom and keeps the file small.

The report mirrors the six methodology stages, and every table that reports a
statistic carries three things next to it: the conventional threshold, whether
the value cleared it, and a sentence saying what that means.  A number without
its criterion is not a finding.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from semsim.core.errors import ExportError
from semsim.core.spec import ModelSpec, ScaleKind

# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


@dataclass
class ReportBundle:
    """Everything the report can draw on.  All but ``spec`` may be None."""

    spec: ModelSpec
    dataset: Any = None
    estimation: Any = None
    fit: Any = None
    validity: Any = None
    descriptives: Any = None
    power: Any = None
    calibration: Any = None
    generated_at: Optional[str] = None
    demographics: Any = None

    def timestamp(self) -> str:
        return self.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Small bilingual helpers
# --------------------------------------------------------------------------- #


#: Column name used for a verdict chip. It is language-neutral on purpose:
#: the header is translated at render time, and the chip renderer keys off it.
_VERDICT = "\u2713"


def _t(lang: str, ar: str, en: str) -> str:
    return ar if lang == "ar" else en


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: Any, places: int = 3) -> str:
    """Numbers always read left-to-right, whatever the page direction."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if math.isnan(f):
        return "&ndash;"
    if math.isinf(f):
        return "&infin;" if f > 0 else "&minus;&infin;"
    if places == 0:
        return f'<span dir="ltr">{int(round(f))}</span>'
    return f'<span dir="ltr">{f:.{places}f}</span>'


def _p(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "&ndash;"
    if math.isnan(f):
        return "&ndash;"
    text = "&lt; .0001" if f < 1e-4 else f"{f:.4f}"
    stars = "***" if f < .001 else ("**" if f < .01 else ("*" if f < .05 else ""))
    return f'<span dir="ltr">{text}{stars}</span>'


def _chip(verdict: str, lang: str) -> str:
    labels = {
        "pass": ("مستوفى", "Pass"),
        "fail": ("غير مستوفى", "Fail"),
        "borderline": ("حدّي", "Borderline"),
        "info": ("وصفي", "Descriptive"),
    }
    ar, en = labels.get(verdict, labels["info"])
    return f'<span class="chip {verdict}">{_esc(_t(lang, ar, en))}</span>'


# --------------------------------------------------------------------------- #
# Inline SVG charts
# --------------------------------------------------------------------------- #


def _svg_bars(values: Sequence[float], labels: Sequence[str], *,
              width: int = 420, height: int = 150,
              colour: str = "#1b5e8c", fmt: str = "{:.0%}") -> str:
    """A simple bar chart.  Always drawn left-to-right: category 1 first."""
    if not len(values):
        return ""
    top = max(max(values), 1e-9)
    pad_l, pad_b, pad_t = 8, 22, 18
    plot_h = height - pad_b - pad_t
    slot = (width - 2 * pad_l) / len(values)
    bar_w = slot * 0.68
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" '
             f'width="100%" height="{height}" dir="ltr" '
             f'role="img" aria-hidden="true">']
    for index, value in enumerate(values):
        h = max(plot_h * (value / top), 1.0)
        x = pad_l + slot * index + (slot - bar_w) / 2
        y = pad_t + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                     f'height="{h:.1f}" rx="2" fill="{colour}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" '
                     f'class="svg-val">{_esc(fmt.format(value))}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" '
                     f'y="{height - 6:.1f}" class="svg-lab">'
                     f'{_esc(labels[index])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_fit(fit: Any, lang: str) -> str:
    """Each headline index against its threshold, as a proportion of it."""
    if fit is None:
        return ""
    rows = [
        ("RMSEA", getattr(fit, "rmsea", float("nan")), 0.06, "below"),
        ("SRMR", getattr(fit, "srmr", float("nan")), 0.08, "below"),
        ("CFI", getattr(fit, "cfi", float("nan")), 0.95, "above"),
        ("TLI", getattr(fit, "tli", float("nan")), 0.95, "above"),
        ("&chi;&sup2;/df", getattr(fit, "chi_square_df_ratio", float("nan")),
         3.0, "below"),
    ]
    width, row_h, pad = 460, 26, 6
    height = pad * 2 + row_h * len(rows)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" '
             f'width="100%" height="{height}" dir="ltr" role="img">']
    label_w, track_x = 70, 86
    track_w = width - track_x - 68
    for index, (name, value, threshold, direction) in enumerate(rows):
        y = pad + row_h * index
        if not math.isfinite(value):
            continue
        passed = value <= threshold if direction == "below" else value >= threshold
        colour = "#1d7a4c" if passed else "#a62b2b"
        # Scale so the threshold always sits at 60% of the track: that makes
        # the mark a reference line rather than an arbitrary position.
        scale = (track_w * 0.6) / max(threshold, 1e-9)
        bar = min(max(value * scale, 1.0), track_w)
        parts.append(f'<text x="{label_w}" y="{y + 15}" class="svg-lab" '
                     f'text-anchor="end">{name}</text>')
        parts.append(f'<rect x="{track_x}" y="{y + 6}" width="{track_w}" '
                     f'height="11" rx="5" fill="#eef1f5"/>')
        parts.append(f'<rect x="{track_x}" y="{y + 6}" width="{bar:.1f}" '
                     f'height="11" rx="5" fill="{colour}"/>')
        mark = track_x + track_w * 0.6
        parts.append(f'<line x1="{mark:.1f}" y1="{y + 2}" x2="{mark:.1f}" '
                     f'y2="{y + 21}" stroke="#5b6b7c" stroke-width="1.5" '
                     f'stroke-dasharray="3 2"/>')
        parts.append(f'<text x="{track_x + track_w + 6}" y="{y + 15}" '
                     f'class="svg-val" text-anchor="start">{value:.4g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_curve(curve: Any, recommended: Optional[int], target: float,
               lang: str) -> str:
    """The power curve, with the 0.80 reference and the recommended N marked."""
    if curve is None or not isinstance(curve, pd.DataFrame) or curve.empty:
        return ""
    if "n" not in curve.columns or "power" not in curve.columns:
        return ""
    xs = curve["n"].to_numpy(dtype=float)
    ys = curve["power"].to_numpy(dtype=float)
    width, height = 480, 200
    pad_l, pad_r, pad_t, pad_b = 42, 12, 14, 28
    x0, x1 = float(xs.min()), float(xs.max())
    span = max(x1 - x0, 1.0)
    px = lambda v: pad_l + (v - x0) / span * (width - pad_l - pad_r)
    py = lambda v: pad_t + (1.0 - min(max(v, 0.0), 1.0)) * (height - pad_t - pad_b)
    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" '
             f'width="100%" height="{height}" dir="ltr" role="img">']
    for level in (0.2, 0.4, 0.6, 0.8, 1.0):
        y = py(level)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                     f'y2="{y:.1f}" stroke="#e9edf2" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="svg-lab" '
                     f'text-anchor="end">{level:.1f}</text>')
    y_target = py(target)
    parts.append(f'<line x1="{pad_l}" y1="{y_target:.1f}" '
                 f'x2="{width - pad_r}" y2="{y_target:.1f}" stroke="#9a6400" '
                 f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#1b5e8c" '
                 f'stroke-width="2.2"/>')
    if recommended is not None and x0 <= recommended <= x1:
        x = px(float(recommended))
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
                     f'y2="{height - pad_b}" stroke="#1d7a4c" '
                     f'stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="svg-val" '
                     f'text-anchor="middle" fill="#1d7a4c">'
                     f'N = {recommended}</text>')
    parts.append(f'<text x="{pad_l}" y="{height - 8}" class="svg-lab" '
                 f'text-anchor="start">{int(x0)}</text>')
    parts.append(f'<text x="{width - pad_r}" y="{height - 8}" class="svg-lab" '
                 f'text-anchor="end">{int(x1)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Table rendering
# --------------------------------------------------------------------------- #


def _table(frame: pd.DataFrame, *, headers: Optional[Sequence[str]] = None,
           precision: Optional[Dict[str, int]] = None,
           verdict_columns: Sequence[str] = ("verdict",),
           p_columns: Sequence[str] = ("p", "p_value", "pclose"),
           lang: str = "ar", max_rows: Optional[int] = None) -> str:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return f'<p class="muted">{_esc(_t(lang, "لا توجد بيانات.", "No data."))}</p>'
    body = frame if max_rows is None else frame.head(max_rows)
    precision = precision or {}
    # Column names in this report are already translated, so the English
    # "is this a count" list cannot recognise them. Decide from the values:
    # a column whose numbers are all whole prints without decimals.
    integral = {str(c) for c in body.columns if _all_integral(body[c])}
    heads = headers if headers is not None else [str(c) for c in body.columns]
    out = ['<div class="scroll"><table><thead><tr>']
    out += [f"<th>{_esc(_t(lang, 'الحكم', 'Verdict') if h == _VERDICT else h)}</th>"
            for h in heads]
    out.append("</tr></thead><tbody>")
    for _index, row in body.iterrows():
        out.append("<tr>")
        for column in body.columns:
            name = str(column)
            value = row[column]
            if name in verdict_columns:
                out.append(f"<td>{_chip(_verdict_of(value), lang)}</td>")
            elif name in p_columns:
                out.append(f'<td class="num">{_p(value)}</td>')
            elif isinstance(value, (int, float, np.integer, np.floating)) \
                    and not isinstance(value, bool):
                places = precision.get(
                    name, 0 if (_is_count(name) or name in integral) else 3)
                out.append(f'<td class="num">{_num(value, places)}</td>')
            else:
                out.append(f"<td>{_esc(value)}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    if max_rows is not None and len(frame) > max_rows:
        out.append(f'<p class="muted">{_esc(_t(lang, f"عُرض {max_rows} صفًا من أصل {len(frame)}.", f"Showing {max_rows} of {len(frame)} rows."))}</p>')
    return "".join(out)


def _all_integral(column: pd.Series) -> bool:
    """True when every value in a numeric column is a whole number."""
    try:
        values = pd.to_numeric(column, errors="coerce").dropna()
    except (TypeError, ValueError):
        return False
    if values.empty or column.dtype == bool:
        return False
    return bool(np.all(np.isclose(values % 1, 0.0)))


def _is_count(name: str) -> bool:
    return name.strip().lower() in {
        "n", "df", "n_items", "n_free", "count", "n_missing", "round",
        "draws", "n_required", "n_recommended", "n_categories"}


def _verdict_of(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "pass" if value else "fail"
    key = str(value).strip().lower()
    if key in {"pass", "true", "yes", "significant", "ok"}:
        return "pass"
    if key in {"fail", "false", "no", "not significant", "non-significant"}:
        return "fail"
    if key in {"borderline", "marginal"}:
        return "borderline"
    return "info"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _section(number: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="sub">{_esc(subtitle)}</p>' if subtitle else ""
    return (f'<section><h2><span class="num-badge">{_esc(number)}</span>'
            f'{_esc(title)}</h2>{sub}{body}</section>')


def _specification_section(bundle: ReportBundle, lang: str) -> str:
    spec = bundle.spec
    rows = []
    for construct in spec.constructs:
        rows.append({
            _t(lang, "المتغير", "Construct"): construct.name,
            _t(lang, "التسمية", "Label"): construct.label.get(lang),
            _t(lang, "الدور", "Role"): _role_label(construct.role, lang),
            _t(lang, "الرتبة", "Order"): _t(lang, "الثانية", "Second")
                if construct.is_second_order else _t(lang, "الأولى", "First"),
            _t(lang, "الأبعاد", "Dimensions"): len(construct.dimensions),
            _t(lang, "الأسئلة", "Items"): construct.n_items,
        })
    table = _table(pd.DataFrame(rows), lang=lang)

    path_rows = []
    for path in spec.paths:
        source = spec.construct(path.source)
        target = spec.construct(path.target)
        path_rows.append({
            _t(lang, "من", "From"): source.name if source else path.source,
            _t(lang, "إلى", "To"): target.name if target else path.target,
            _t(lang, "قوة التأثير", "Effect size"):
                _effect_label(path.effect_size, lang),
            _t(lang, "المعامل", "Coefficient"): path.resolved_coefficient(),
            _VERDICT: "pass" if path.target_significant else "info",
        })
    for interaction in spec.interactions:
        predictor = spec.construct(interaction.predictor)
        moderator = spec.construct(interaction.moderator)
        target = spec.construct(interaction.target)
        path_rows.append({
            _t(lang, "من", "From"):
                f"{predictor.name if predictor else '?'} × "
                f"{moderator.name if moderator else '?'}",
            _t(lang, "إلى", "To"): target.name if target else "?",
            _t(lang, "قوة التأثير", "Effect size"):
                _effect_label(interaction.effect_size, lang),
            _t(lang, "المعامل", "Coefficient"):
                interaction.resolved_coefficient(),
            _VERDICT: "pass" if interaction.target_significant else "info",
        })
    paths = (_table(pd.DataFrame(path_rows), lang=lang,
                    verdict_columns=(_VERDICT,)) if path_rows else "")

    counts = (
        f'<div class="tiles">'
        f'{_tile(_t(lang, "المتغيرات الكامنة", "Constructs"), spec.n_constructs)}'
        f'{_tile(_t(lang, "الأسئلة", "Items"), spec.n_items)}'
        f'{_tile(_t(lang, "المسارات", "Paths"), len(spec.paths) + len(spec.interactions))}'
        f'{_tile(_t(lang, "المعالم المقدَّرة", "Free parameters"), spec.count_free_parameters())}'
        f'{_tile(_t(lang, "درجات الحرية", "Degrees of freedom"), spec.degrees_of_freedom())}'
        f'</div>')
    heading = f'<h3>{_esc(_t(lang, "النموذج الهيكلي", "Structural model"))}</h3>'
    return _section("1", _t(lang, "بناء مواصفات النموذج", "Model specification"),
                    counts + table + (heading + paths if paths else ""),
                    "Model Specification Architecture")


def _tile(label: str, value: Any) -> str:
    return (f'<div class="tile"><div class="tile-v" dir="ltr">{_esc(value)}'
            f'</div><div class="tile-l">{_esc(label)}</div></div>')


def _demographics_section(bundle: ReportBundle, lang: str) -> str:
    """Quotas achieved, control paths, group comparisons and invariance."""
    analysis = bundle.demographics
    if analysis is None:
        return ""
    suffix = "_ar" if lang == "ar" else "_en"
    pieces: List[str] = []

    def heading(ar: str, en: str) -> str:
        return f'<h3>{_esc(_t(lang, ar, en))}</h3>'

    table = analysis.frequencies
    rows = [{
        _t(lang, "المتغير", "Variable"): row["variable"],
        _t(lang, "الفئة", "Category"): row[f"category{suffix}"],
        _t(lang, "المطلوب %", "Target %"): row["target_percent"],
        "n": row["n"],
        _t(lang, "المحقق %", "Achieved %"): row["achieved_percent"],
    } for _i, row in table.iterrows()]
    pieces.append(heading("التكرارات", "Frequencies")
                  + _table(pd.DataFrame(rows), lang=lang))

    controls = [{
        _t(lang, "المتغير الضابط", "Control"): term.column,
        _t(lang, "المتغير المتأثر", "Target"):
            bundle.spec.factor_label(path.target, lang),
        _t(lang, "المعامل المعياري", "Standardised coefficient"):
            path.resolved_coefficient(),
        _t(lang, "مستهدف أن يكون دالًا", "Target significant"):
            _t(lang, "نعم", "Yes") if path.target_significant
            else _t(lang, "لا", "No"),
    } for variable, term in bundle.spec.active_control_terms()
        for path in variable.control_paths if path.term == term.key]
    if controls:
        pieces.append(heading("مسارات المتغيرات الضابطة", "Control paths")
                      + _table(pd.DataFrame(controls), lang=lang))

    table = analysis.comparisons
    if not table.empty:
        rows = [{
            _t(lang, "المتغير", "Variable"): row["variable"],
            _t(lang, "المتغير الكامن", "Construct"): row["construct"],
            _t(lang, "الاختبار", "Test"): (
                _t(lang, "t (ويلش)", "Welch t") if row["test"] == "welch_t"
                else "F"),
            _t(lang, "الإحصاءة", "Statistic"): row["statistic"],
            "p": row["p"],
            _t(lang, "حجم الأثر", "Effect size"): row["effect_size"],
            _t(lang, "المتوسطات", "Means"): row[f"means{suffix}"],
        } for _i, row in table.iterrows()]
        pieces.append(heading("اختبارات الفروق", "Group comparisons")
                      + _table(pd.DataFrame(rows), lang=lang))

    table = analysis.invariance
    if not table.empty:
        rows = [{
            _t(lang, "المتغير", "Variable"): row["variable"],
            _t(lang, "النموذج", "Model"): row[f"model{suffix}"],
            "χ²": row["chi_square"], "df": row["df"], "CFI": row["cfi"],
            "RMSEA": row["rmsea"], "Δχ²": row["delta_chi_square"],
            "Δdf": row["delta_df"], "ΔCFI": row["delta_cfi"],
        } for _i, row in table.iterrows()]
        pieces.append(heading("ثبات النموذج عبر المجموعات",
                              "Multi-group invariance")
                      + _table(pd.DataFrame(rows), lang=lang))
    return _section("8", _t(lang, "الديموغرافيا والمتغيرات الضابطة",
                            "Demographics and controls"),
                    "".join(pieces), "Demographics & Control Variables")


def _complexity_section(bundle: ReportBundle, lang: str) -> str:
    spec = bundle.spec
    pieces: List[str] = []
    if spec.cross_loadings:
        rows = [{
            _t(lang, "السؤال", "Item"): (spec.item(cl.item_id).name
                                         if spec.item(cl.item_id) else cl.item_id),
            _t(lang, "على المتغير", "On factor"):
                spec.factor_label(cl.factor_id, lang),
            _t(lang, "التشبع", "Loading"): cl.loading,
        } for cl in spec.cross_loadings]
        pieces.append(f'<h3>{_esc(_t(lang, "التأثيرات المتقاطعة", "Cross-loadings"))}</h3>'
                      + _table(pd.DataFrame(rows), lang=lang))
    if spec.cross_loading_policy.enabled:
        policy = spec.cross_loading_policy
        pieces.append(
            f'<p class="note">{_esc(_t(lang, f"تم توليد تحميلات متقاطعة تلقائيًا لنسبة {policy.proportion_of_items:.0%} من الأسئلة، بحجم بين {policy.magnitude_range[0]:.2f} و{policy.magnitude_range[1]:.2f}.", f"Cross-loadings were auto-generated for {policy.proportion_of_items:.0%} of items, with magnitudes between {policy.magnitude_range[0]:.2f} and {policy.magnitude_range[1]:.2f}."))}</p>')
    if spec.error_covariances:
        rows = [{
            _t(lang, "السؤال الأول", "First item"):
                (spec.item(ec.item_a).name if spec.item(ec.item_a) else ec.item_a),
            _t(lang, "السؤال الثاني", "Second item"):
                (spec.item(ec.item_b).name if spec.item(ec.item_b) else ec.item_b),
            _t(lang, "الارتباط", "Correlation"): ec.correlation,
        } for ec in spec.error_covariances]
        pieces.append(f'<h3>{_esc(_t(lang, "التغاير بين الأخطاء", "Error covariances"))}</h3>'
                      + _table(pd.DataFrame(rows), lang=lang))
    if spec.common_method_bias.enabled:
        affected = len(spec.common_method_bias.item_ids) or spec.n_items
        pieces.append(
            f'<p class="note">{_esc(_t(lang, f"عامل طريقة قياس مشترك بتشبع {spec.common_method_bias.loading:.2f} على {affected} سؤالًا.", f"A common method factor loading {spec.common_method_bias.loading:.2f} on {affected} items."))}</p>')
    if not pieces:
        pieces.append(f'<p class="muted">{_esc(_t(lang, "لم تُفعَّل أي تعقيدات.", "No complications were enabled."))}</p>')
    return _section("2", _t(lang, "محاكاة تعقيدات الواقع", "Complex realities"),
                    "".join(pieces), "Complex Realities Simulation")


def _distribution_section(bundle: ReportBundle, lang: str) -> str:
    spec = bundle.spec
    pieces: List[str] = []
    dataset = bundle.dataset
    if dataset is not None:
        pieces.append(
            f'<div class="tiles">'
            f'{_tile(_t(lang, "حجم العينة", "Sample size"), getattr(dataset, "n", "-"))}'
            f'{_tile(_t(lang, "بذرة العشوائية", "Seed"), getattr(dataset, "seed", "-"))}'
            f'{_tile(_t(lang, "محرك التوليد", "Engine"), getattr(getattr(dataset, "engine_used", None), "value", "-"))}'
            f'{_tile(_t(lang, "الخلايا المفقودة", "Missing cells"), int(dataset.data.isna().to_numpy().sum()) if hasattr(dataset, "data") else 0)}'
            f'</div>')
    if bundle.descriptives is not None:
        try:
            table = _drop_other_language(bundle.descriptives.to_table(lang), lang)
            # The full table carries twenty columns; a reader wants eight.
            # The rest stay in the exported spreadsheet.
            wanted = ["item", "label", "n", "n_missing", "mean", "sd",
                      "minimum", "maximum", "skewness", "kurtosis"]
            keep = [c for c in wanted if c in table.columns]
            if len(keep) >= 6:
                table = table[keep]
        except Exception:
            table = pd.DataFrame()
        pieces.append(f'<h3>{_esc(_t(lang, "الإحصاءات الوصفية", "Descriptive statistics"))}</h3>'
                      + _table(table, lang=lang, max_rows=60))
        d = bundle.descriptives
        verdict = "pass" if d.multivariate_normal else "fail"
        pieces.append(
            f'<p class="note">{_esc(_t(lang, "اختبار مارديا للاعتدالية المتعددة", "Mardia multivariate normality"))}: '
            f'skew = {_num(d.mardia_skewness, 3)} (p = {_p(d.mardia_skewness_p)}), '
            f'kurtosis = {_num(d.mardia_kurtosis, 3)} (p = {_p(d.mardia_kurtosis_p)}) '
            f'{_chip(verdict, lang)}</p>')
    chart = _category_charts(bundle, lang)
    if chart:
        pieces.append(f'<h3>{_esc(_t(lang, "توزيع الاستجابات", "Response distributions"))}</h3>' + chart)
    return _section("3", _t(lang, "توليد البيانات والتوزيعات",
                            "Distributions and generation"),
                    "".join(pieces), "Data Distribution & Generation Engine")


def _category_charts(bundle: ReportBundle, lang: str, limit: int = 6) -> str:
    dataset = bundle.dataset
    if dataset is None or not hasattr(dataset, "data"):
        return ""
    spec = bundle.spec
    ordinal = [item for item in spec.all_items()
               if item.scale.kind is ScaleKind.LIKERT]
    if not ordinal:
        return ""
    cards = []
    for item in ordinal[:limit]:
        if item.name not in dataset.data.columns:
            continue
        column = dataset.data[item.name].dropna()
        k = item.scale.n_categories
        counts = [float((column == c).sum()) for c in range(1, k + 1)]
        total = sum(counts) or 1.0
        proportions = [c / total for c in counts]
        cards.append(
            f'<div class="card"><h4>{_esc(item.name)} &mdash; '
            f'{_esc(item.label.get(lang))}</h4>'
            + _svg_bars(proportions, [str(c) for c in range(1, k + 1)])
            + "</div>")
    if not cards:
        return ""
    more = ""
    if len(ordinal) > limit:
        more = (f'<p class="muted">{_esc(_t(lang, f"عُرضت أول {limit} أسئلة من أصل {len(ordinal)}.", f"Showing the first {limit} of {len(ordinal)} items."))}</p>')
    return f'<div class="grid">{"".join(cards)}</div>{more}'


def _power_section(bundle: ReportBundle, lang: str) -> str:
    power = bundle.power
    spec = bundle.spec
    if power is None:
        return _section("4", _t(lang, "تحليل القوة وحجم العينة",
                                "Power and sample size"),
                        f'<p class="muted">{_esc(_t(lang, "لم يُجرَ تحليل القوة.", "Power analysis was not run."))}</p>',
                        "Power Analysis & Sample Size")
    tiles = (f'<div class="tiles">'
             f'{_tile(_t(lang, "الحجم الموصى به", "Recommended N"), power.n_recommended)}'
             f'{_tile(_t(lang, "القوة المتحققة", "Power"), f"{power.power_at_recommended:.3f}")}'
             f'{_tile(_t(lang, "درجات الحرية", "df"), power.df)}'
             f'{_tile("&alpha;", power.alpha)}'
             f'</div>')
    curve = _svg_curve(getattr(power, "curve", None), power.n_recommended,
                       getattr(power, "target_power", 0.80), lang)
    rules = getattr(power, "rule_of_thumb", None) or {}
    rule_table = _table(pd.DataFrame(
        [{_t(lang, "القاعدة", "Rule"): k, _t(lang, "الحجم", "N"): v}
         for k, v in rules.items()]), lang=lang) if rules else ""
    notes = getattr(power, "notes_ar" if lang == "ar" else "notes_en", []) or []
    note_html = "".join(f'<li>{_esc(n)}</li>' for n in notes[:8])
    note_block = f"<ul class='notes'>{note_html}</ul>" if note_html else ""
    per_path = _table(_drop_other_language(
        getattr(power, "per_path_power", None), lang), lang=lang)
    return _section("4", _t(lang, "تحليل القوة وحجم العينة",
                            "Power and sample size"),
                    tiles + curve + rule_table + per_path + note_block,
                    "Power Analysis & Sample Size")


def _targeting_section(bundle: ReportBundle, lang: str) -> str:
    spec = bundle.spec
    pieces: List[str] = []
    target = spec.fit_target
    pieces.append(
        f'<p class="note">{_esc(_t(lang, "المستوى المستهدف", "Target level"))}: '
        f'<b>{_esc(_fit_level_label(target.level, lang))}</b> '
        f'(RMSEA &asymp; {_num(target.level.target_rmsea, 3)}), '
        f'{_esc(_t(lang, "الأسلوب", "method"))}: {_esc(target.method)}</p>')
    calibration = bundle.calibration
    if calibration is not None:
        summary = calibration.summary() if hasattr(calibration, "summary") else {}
        verdict = "pass" if summary.get("within_tolerance") else "fail"
        pieces.append(
            f'<p class="note">{_esc(_t(lang, "المتحقق", "Realised"))} RMSEA = '
            f'{_num(summary.get("realised_rmsea"), 4)} '
            f'{_esc(_t(lang, "بعد", "after"))} {_esc(summary.get("rounds", "-"))} '
            f'{_esc(_t(lang, "دورة معايرة", "calibration round(s)"))} '
            f'{_chip(verdict, lang)}</p>')
        pieces.append(f'<h3>{_esc(_t(lang, "سجل المعايرة", "Calibration log"))}</h3>'
                      + _table(getattr(calibration, "history", None), lang=lang,
                               precision={"injected_rmsea": 4,
                                          "realised_rmsea": 4,
                                          "target_rmsea": 4, "deviation": 4}))
        pieces.append(f'<h3>{_esc(_t(lang, "حالة كل مسار", "Per-path status"))}</h3>'
                      + _table(getattr(calibration, "significance_table", None),
                               lang=lang,
                               verdict_columns=("verdict", "requested", "realised")))
    return _section("5", _t(lang, "معايرة الدلالة وجودة التوفيق",
                            "Significance and fit targeting"),
                    "".join(pieces), "Fit Indices & Significance Targeting")


def _validation_section(bundle: ReportBundle, lang: str) -> str:
    pieces: List[str] = []
    fit = bundle.fit
    if fit is not None:
        try:
            level = fit.verdict()
            label = _fit_level_label(level, lang)
        except Exception:
            label = ""
        pieces.append(
            f'<div class="tiles">'
            f'{_tile("RMSEA", f"{fit.rmsea:.4f}")}'
            f'{_tile("CFI", f"{fit.cfi:.3f}")}'
            f'{_tile("TLI", f"{fit.tli:.3f}")}'
            f'{_tile("SRMR", f"{fit.srmr:.4f}")}'
            f'{_tile("&chi;&sup2;/df", f"{fit.chi_square_df_ratio:.3f}")}'
            f'</div>')
        if label:
            pieces.append(f'<p class="note">{_esc(_t(lang, "الحكم العام", "Overall verdict"))}: <b>{_esc(label)}</b></p>')
        pieces.append(_svg_fit(fit, lang))
        try:
            table = _drop_other_language(fit.to_table(lang), lang)
            table = table.drop(columns=[c for c in ("index",)
                                        if c in table.columns])
        except Exception:
            table = pd.DataFrame()
        pieces.append(_table(table, lang=lang))
    validity = bundle.validity
    if validity is not None and getattr(validity, "per_factor", None):
        rows = []
        for factor in validity.per_factor:
            rows.append({
                _t(lang, "المتغير", "Construct"): factor.factor_name,
                _t(lang, "عدد الأسئلة", "Items"): factor.n_items,
                _t(lang, "ألفا", "Alpha"): factor.cronbach_alpha,
                "CR": factor.composite_reliability,
                "&omega;": factor.omega_total,
                "AVE": factor.ave,
                _VERDICT: ("pass" if (factor.cronbach_alpha >= .70
                                       and factor.composite_reliability >= .70
                                       and factor.ave >= .50)
                            else ("borderline" if factor.ave >= .45 else "fail")),
            })
        pieces.append(f'<h3>{_esc(_t(lang, "مؤشرات الثبات والصدق التقاربي", "Reliability and convergent validity"))}</h3>'
                      + _table(pd.DataFrame(rows), lang=lang,
                               verdict_columns=(_VERDICT,)))
        pieces.append(f'<h3>{_esc(_t(lang, "معيار فورنيل-لاركر", "Fornell-Larcker criterion"))}</h3>'
                      + _table(_reset_index(getattr(validity, "fornell_larcker", None)), lang=lang)
                      + f'<p class="note">{_chip("pass" if validity.fornell_larcker_passed else "fail", lang)} '
                      + _esc(_t(lang, "الجذر التربيعي لـ AVE على القطر مقابل الارتباطات خارجه.",
                                "Square root of AVE on the diagonal against the correlations off it.")) + "</p>")
        pieces.append(f'<h3>HTMT</h3>'
                      + _table(_reset_index(getattr(validity, "htmt", None)), lang=lang)
                      + f'<p class="note">{_chip("pass" if validity.htmt_passed else "fail", lang)} '
                      + _esc(_t(lang, f"المعيار: أقل من {validity.htmt_threshold:.2f}.",
                                f"Criterion: below {validity.htmt_threshold:.2f}.")) + "</p>")
    estimation = bundle.estimation
    if estimation is not None:
        paths = _reset_index(getattr(estimation, "path_estimates", None))
        if not paths.empty:
            keep = [c for c in ("source", "target", "est", "se", "z", "p", "std")
                    if c in paths.columns]
            pieces.append(f'<h3>{_esc(_t(lang, "معاملات المسار", "Path coefficients"))}</h3>'
                          + _table(paths[keep], lang=lang))
        loadings = _reset_index(getattr(estimation, "loadings", None))
        if not loadings.empty:
            keep = [c for c in ("item", "factor", "est", "se", "z", "p", "std")
                    if c in loadings.columns]
            pieces.append(f'<h3>{_esc(_t(lang, "التشبعات", "Loadings"))}</h3>'
                          + _table(loadings[keep], lang=lang, max_rows=60))
        r2 = getattr(estimation, "r_squared", None) or {}
        if r2:
            pieces.append(f'<h3>R&sup2;</h3>' + _table(pd.DataFrame(
                [{_t(lang, "المتغير", "Construct"): k, "R²": v}
                 for k, v in r2.items()]), lang=lang))
    if not pieces:
        pieces.append(f'<p class="muted">{_esc(_t(lang, "لم تُجرَ عملية تقدير.", "No estimation was run."))}</p>')
    return _section("6", _t(lang, "التقييم واستخراج المؤشرات",
                            "Validation and output"),
                    "".join(pieces), "Validation & Fit Output")


def _syntax_section(bundle: ReportBundle, lang: str) -> str:
    try:
        from semsim.io.exporters import lavaan_syntax
        syntax = lavaan_syntax(bundle.spec)
    except Exception:
        return ""
    return _section("A", _t(lang, "صيغة lavaan لإعادة التقدير في R",
                            "lavaan syntax for re-estimation in R"),
                    f'<pre dir="ltr">{_esc(syntax)}</pre>')


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #


def _role_label(role: Any, lang: str) -> str:
    return {
        "independent": ("مستقل", "Independent"),
        "dependent": ("تابع", "Dependent"),
        "mediator": ("وسيط", "Mediator"),
        "moderator": ("معدِّل", "Moderator"),
        "control": ("ضابط", "Control"),
    }.get(getattr(role, "value", role), ("", ""))[0 if lang == "ar" else 1]


def _effect_label(effect: Any, lang: str) -> str:
    return {
        "none": ("لا يوجد", "None"), "trivial": ("ضئيل", "Trivial"),
        "weak": ("ضعيف", "Weak"), "medium": ("متوسط", "Medium"),
        "strong": ("قوي", "Strong"), "very_strong": ("قوي جدًا", "Very strong"),
        "custom": ("مخصّص", "Custom"),
    }.get(getattr(effect, "value", effect), ("", ""))[0 if lang == "ar" else 1]


def _fit_level_label(level: Any, lang: str) -> str:
    return {
        "perfect": ("مثالي", "Perfect"), "excellent": ("ممتاز", "Excellent"),
        "good": ("جيد جدًا", "Good"), "acceptable": ("مقبول", "Acceptable"),
        "marginal": ("حدّي", "Marginal"), "poor": ("ضعيف", "Poor"),
        "custom": ("مخصّص", "Custom"),
    }.get(getattr(level, "value", level), ("", ""))[0 if lang == "ar" else 1]


def _drop_other_language(frame: Any, lang: str) -> pd.DataFrame:
    """Keep only the active language's half of each ``*_ar`` / ``*_en`` pair."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    keep, drop = ("_ar", "_en") if lang == "ar" else ("_en", "_ar")
    out = frame.copy()
    renames = {}
    for column in list(out.columns):
        name = str(column)
        if name.endswith(drop) and f"{name[:-len(drop)]}{keep}" in out.columns:
            out = out.drop(columns=[column])
        elif name.endswith(keep):
            renames[column] = name[:-len(keep)]
    return out.rename(columns=renames)


def _reset_index(frame: Any) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.index, pd.RangeIndex) and not frame.index.name:
        return frame
    return frame.reset_index()


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #

_CSS = """
:root{--bg:#f4f6f8;--surface:#fff;--alt:#eef1f5;--border:#d4dae1;--text:#17202a;
--muted:#5b6b7c;--accent:#1b5e8c;--ok:#1d7a4c;--bad:#a62b2b;--warn:#9a6400;}
*{box-sizing:border-box}
body{margin:0;padding:0;background:var(--bg);color:var(--text);
font-family:"Segoe UI","Dubai",Tahoma,"Noto Sans Arabic","DejaVu Sans",Arial,sans-serif;
font-size:14px;line-height:1.65;}
.wrap{max-width:1040px;margin:0 auto;padding:28px 20px 64px;}
header.doc{background:var(--accent);color:#fff;border-radius:12px;padding:22px 26px;margin-bottom:22px;}
header.doc h1{margin:0 0 6px;font-size:24px;}
header.doc .meta{opacity:.9;font-size:13px;}
header.doc .meta span{margin-inline-end:18px;white-space:nowrap;}
section{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:20px 22px;margin-bottom:18px;}
h2{margin:0 0 2px;font-size:19px;display:flex;align-items:center;gap:10px;}
h3{margin:22px 0 8px;font-size:15px;color:var(--accent);}
h4{margin:0 0 6px;font-size:13px;}
.num-badge{display:inline-flex;align-items:center;justify-content:center;
min-width:26px;height:26px;border-radius:7px;background:var(--accent);color:#fff;
font-size:13px;font-weight:700;}
p.sub{margin:0 0 14px;color:var(--muted);font-size:12px;letter-spacing:.3px;}
p.note{background:var(--alt);border-radius:8px;padding:9px 12px;margin:10px 0;}
p.muted,.muted{color:var(--muted);}
ul.notes{margin:10px 0;padding-inline-start:20px;color:var(--muted);}
.scroll{overflow-x:auto;margin:8px 0 4px;}
table{border-collapse:collapse;width:100%;font-size:13px;}
th,td{border-bottom:1px solid var(--border);padding:7px 10px;text-align:start;}
thead th{background:var(--alt);font-weight:700;white-space:nowrap;}
tbody tr:nth-child(even){background:#fafbfc;}
td.num{text-align:end;font-family:Consolas,"DejaVu Sans Mono",monospace;white-space:nowrap;}
.chip{display:inline-block;border-radius:9px;padding:1px 9px;font-size:12px;font-weight:700;}
.chip.pass{background:#e2f2e8;color:#14603a;}
.chip.fail{background:#fbe6e6;color:#8c2222;}
.chip.borderline{background:#fdf1dc;color:#7d5100;}
.chip.info{background:var(--alt);color:var(--muted);}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:4px 0 14px;}
.tile{flex:1 1 118px;background:var(--alt);border-radius:9px;padding:10px 12px;text-align:center;}
.tile-v{font-size:19px;font-weight:700;font-family:Consolas,"DejaVu Sans Mono",monospace;}
.tile-l{font-size:11px;color:var(--muted);margin-top:2px;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;}
.card{border:1px solid var(--border);border-radius:9px;padding:10px 12px;}
svg.chart{display:block;margin:6px 0;}
.svg-lab{font-size:10px;fill:#5b6b7c;text-anchor:middle;}
.svg-val{font-size:10px;fill:#17202a;text-anchor:middle;
font-family:Consolas,"DejaVu Sans Mono",monospace;}
pre{background:#11161b;color:#e8edf2;border-radius:9px;padding:14px 16px;
overflow-x:auto;font-size:12.5px;line-height:1.5;}
footer{color:var(--muted);font-size:12px;text-align:center;padding-top:10px;}
@media print{body{background:#fff}section{break-inside:avoid;box-shadow:none}
.wrap{max-width:none;padding:0}header.doc{background:#fff;color:#17202a;
border:1px solid var(--border)}}
"""


def render_html(bundle: ReportBundle, *, lang: str = "ar",
                embed_charts: bool = True) -> str:
    """Build the whole report as one self-contained HTML string."""
    spec = bundle.spec
    direction = "rtl" if lang == "ar" else "ltr"
    title = spec.title.get(lang) or _t(lang, "تقرير محاكاة", "Simulation report")
    n = getattr(bundle.dataset, "n", spec.power.n)
    seed = getattr(bundle.dataset, "seed", spec.seed)

    sections = [
        _specification_section(bundle, lang),
        _complexity_section(bundle, lang),
        _distribution_section(bundle, lang),
        _power_section(bundle, lang),
        _targeting_section(bundle, lang),
        _validation_section(bundle, lang),
        _syntax_section(bundle, lang),
        _demographics_section(bundle, lang),
    ]
    meta = (f'<span>{_esc(_t(lang, "التاريخ", "Date"))}: '
            f'<span dir="ltr">{_esc(bundle.timestamp())}</span></span>'
            f'<span>N = <span dir="ltr">{_esc(n)}</span></span>'
            f'<span>{_esc(_t(lang, "البذرة", "Seed"))}: '
            f'<span dir="ltr">{_esc(seed)}</span></span>'
            f'<span>{_esc(_t(lang, "الأسئلة", "Items"))}: '
            f'<span dir="ltr">{spec.n_items}</span></span>')
    footer = _t(lang,
                "أُنشئ بواسطة محرك محاكاة بيانات النمذجة بالمعادلات البنائية.",
                "Generated by the SEM Data Simulation Engine.")
    return (
        f'<!doctype html>\n<html lang="{lang}" dir="{direction}">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f'<div class="wrap">\n<header class="doc"><h1>{_esc(title)}</h1>'
        f'<div class="meta">{meta}</div></header>\n'
        + "\n".join(s for s in sections if s)
        + f'\n<footer>{_esc(footer)}</footer>\n</div>\n</body>\n</html>\n')


def write_html(bundle: ReportBundle, path: Any, *, lang: str = "ar",
               embed_charts: bool = True) -> str:
    """Render and write the report, returning the path written."""
    target = str(path)
    if not target.lower().endswith((".html", ".htm")):
        target += ".html"
    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(render_html(bundle, lang=lang, embed_charts=embed_charts))
    except OSError as exc:
        raise ExportError(
            f"Could not write the report: {exc}",
            f"تعذّر حفظ التقرير: {exc}", path=target) from exc
    return target


def render_markdown(bundle: ReportBundle, *, lang: str = "ar") -> str:
    """A plain-text summary, for pasting into notes or an email."""
    spec = bundle.spec
    lines = [f"# {spec.title.get(lang)}", ""]
    lines.append(f"- {_t(lang, 'التاريخ', 'Date')}: {bundle.timestamp()}")
    lines.append(f"- N: {getattr(bundle.dataset, 'n', spec.power.n)}")
    lines.append(f"- {_t(lang, 'الأسئلة', 'Items')}: {spec.n_items}")
    lines.append(f"- {_t(lang, 'درجات الحرية', 'Degrees of freedom')}: "
                 f"{spec.degrees_of_freedom()}")
    fit = bundle.fit
    if fit is not None:
        lines += ["", f"## {_t(lang, 'جودة المطابقة', 'Model fit')}", "",
                  f"| {_t(lang, 'المؤشر', 'Index')} | "
                  f"{_t(lang, 'القيمة', 'Value')} | "
                  f"{_t(lang, 'المعيار', 'Threshold')} |",
                  "| --- | --- | --- |",
                  f"| RMSEA | {fit.rmsea:.4f} | <= .06 |",
                  f"| CFI | {fit.cfi:.3f} | >= .95 |",
                  f"| TLI | {fit.tli:.3f} | >= .95 |",
                  f"| SRMR | {fit.srmr:.4f} | <= .08 |",
                  f"| chi2/df | {fit.chi_square_df_ratio:.3f} | < 3 |"]
    validity = bundle.validity
    if validity is not None and getattr(validity, "per_factor", None):
        lines += ["", f"## {_t(lang, 'الثبات والصدق', 'Reliability and validity')}",
                  "", f"| {_t(lang, 'المتغير', 'Construct')} | alpha | CR | AVE |",
                  "| --- | --- | --- | --- |"]
        for factor in validity.per_factor:
            lines.append(f"| {factor.factor_name} | "
                         f"{factor.cronbach_alpha:.3f} | "
                         f"{factor.composite_reliability:.3f} | "
                         f"{factor.ave:.3f} |")
    return "\n".join(lines) + "\n"


__all__ = ["ReportBundle", "render_html", "write_html", "render_markdown"]

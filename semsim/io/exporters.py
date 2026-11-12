"""
semsim.io.exporters
===================
Everything that leaves the program as a file: the dataset itself (CSV, Excel,
SPSS ``.sav``) and ready-to-run syntax for the packages a social-science
researcher actually fits models in (lavaan, Mplus, AMOS).

Design notes
------------
*   **The dataset object is duck-typed.**  Every writer takes a
    :class:`~semsim.generation.engine.GeneratedDataset`-like object, i.e.
    anything exposing ``.data`` (a :class:`pandas.DataFrame`) and optionally
    ``.latent_scores``, ``.continuous_shadow``, ``.compiled``, ``.n``,
    ``.seed`` and ``.warnings``.  Nothing here imports the generation engine at
    runtime, so the IO layer stays loadable on its own.
*   **No failure reaches the user as a traceback.**  A missing directory, a
    read-only folder, a file still open in Excel, a name SPSS refuses -- all of
    them surface as :class:`~semsim.core.errors.ExportError` carrying a real
    Arabic message next to the English one.
*   **Syntax generators are pure functions of the specification.**  They never
    look at the data, so a researcher can read the model syntax before
    generating a single row, and the emitted model is exactly the model the
    data were generated from: declared cross-loadings and correlated residuals
    are included, because they are free parameters of the fitted model and
    leaving them out would change the degrees of freedom.
*   Names are sanitised per target package -- 64 bytes and no leading digit for
    SPSS, 8 characters for Mplus -- and every rename is reported rather than
    silently applied.

References
----------
Rosseel, Y. (2012). lavaan: An R package for structural equation modeling.
    *Journal of Statistical Software*, 48(2), 1-36.
Muthen, L. K., & Muthen, B. O. (1998-2017). *Mplus User's Guide* (8th ed.).
    Muthen & Muthen.
Arbuckle, J. L. (2019). *Amos 26 User's Guide*. IBM.
Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10), 1-23.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path as FsPath
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Mapping,
                    MutableMapping, Optional, Sequence, Tuple)

import numpy as np
import pandas as pd

from semsim.core.errors import ExportError
from semsim.core.spec import Item, ModelSpec, ScaleSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from semsim.generation.engine import GeneratedDataset

__all__ = [
    "PROGRAM_NAME", "PROGRAM_VERSION",
    "SPSS_MAX_NAME_BYTES", "MPLUS_MAX_NAME_CHARS", "MPLUS_MISSING_CODE",
    "export_csv", "export_xlsx", "export_sav", "export_rds_script",
    "export_mplus_dat",
    "lavaan_syntax", "mplus_syntax", "amos_notes", "export_all",
    "spss_variable_names", "mplus_variable_names",
]

PROGRAM_NAME = "SEM Data Simulation Engine (semsim)"

try:  # pragma: no cover - packaging detail
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        PROGRAM_VERSION = _pkg_version("semsim")
    except PackageNotFoundError:
        PROGRAM_VERSION = "1.2.2"
except Exception:  # pragma: no cover - very old interpreters
    PROGRAM_VERSION = "1.2.2"

#: SPSS variable names are limited to 64 *bytes* (not characters).
SPSS_MAX_NAME_BYTES = 64

#: Mplus truncates variable names to eight characters.
MPLUS_MAX_NAME_CHARS = 8

#: The numeric code Mplus reads as missing (``MISSING ARE ALL (-999)``).
MPLUS_MISSING_CODE = -999

#: Soft wrap width for generated syntax.  Mplus refuses lines over 90
#: characters; lavaan does not care but long lines are unreadable.
_WRAP_WIDTH = 76

#: Keywords SPSS reserves and will not accept as a variable name.
_SPSS_RESERVED = frozenset({
    "ALL", "AND", "BY", "EQ", "GE", "GT", "LE", "LT", "NE", "NOT", "OR",
    "TO", "WITH",
})

#: Default Likert anchors when the item's scale carries no labels.
_DEFAULT_MIN_LABEL = {"ar": "أدنى درجة على المقياس", "en": "Scale minimum"}
_DEFAULT_MAX_LABEL = {"ar": "أعلى درجة على المقياس", "en": "Scale maximum"}
_REVERSE_MARK = {"ar": "(عبارة عكسية)", "en": "(reverse-coded)"}


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _raise_write_error(path: Any, exc: BaseException, what_en: str,
                       what_ar: str) -> "ExportError":
    """Turn any writer failure into a bilingual :class:`ExportError`.

    Parameters
    ----------
    path
        The destination that could not be written.
    exc
        The original exception (kept as ``__cause__`` and in ``context``).
    what_en, what_ar
        Human name of the format, e.g. ``"CSV"`` / ``"ملف CSV"``.

    Raises
    ------
    ExportError
        Always.

    References
    ----------
    No published method; this implements the project rule that a user-caused
    failure never reaches the interface as a raw traceback.
    """
    raise ExportError(
        f"Could not write the {what_en} file to {path!s}: {_reason(exc)}. "
        f"Check that the folder exists, that you have permission to write "
        f"there, and that the file is not open in another program.",
        f"تعذّر كتابة {what_ar} في المسار «{path!s}»: {_reason(exc)}. "
        f"تأكّد من وجود المجلد ومن أنّ لديك صلاحية الكتابة فيه، "
        f"ومن أنّ الملف غير مفتوح في برنامج آخر.",
        path=str(path), original_error=repr(exc),
    ) from exc


def _as_fs_path(path: Any, what_en: str, what_ar: str) -> FsPath:
    """Coerce ``path`` to :class:`pathlib.Path`, reporting nonsense politely."""
    try:
        return FsPath(os.fspath(path))
    except TypeError as exc:
        raise ExportError(
            f"The destination for the {what_en} export is not a usable file "
            f"path: {path!r}.",
            f"المسار المحدَّد لتصدير {what_ar} ليس مسار ملف صالحًا: {path!r}.",
            path=repr(path),
        ) from exc


def _frame_of(ds: Any, *, what_en: str = "export",
              what_ar: str = "التصدير") -> pd.DataFrame:
    """Pull the observed-data frame out of a dataset-like object.

    Duck-typed on purpose: anything with a ``.data`` attribute that pandas can
    treat as a frame works, which keeps this module independent of
    :mod:`semsim.generation.engine`.

    Raises
    ------
    ExportError
        If the object carries no data, or the data are empty.

    References
    ----------
    Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10).
    """
    # Files carry the demographic columns in front of the item responses.
    data = getattr(ds, "full_frame", None)
    if data is None:
        data = getattr(ds, "data", None)
    if data is None:
        raise ExportError(
            f"The object passed to the {what_en} writer carries no dataset "
            f"(no `.data` attribute); generate the data first.",
            f"الكائن المُمرَّر إلى دالة {what_ar} لا يحتوي على جدول بيانات "
            f"(الخاصية `data` غير موجودة)؛ وَلِّد البيانات أولًا.",
        )
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as exc:  # pragma: no cover - defensive
            raise ExportError(
                f"The dataset passed to the {what_en} writer is not a table: "
                f"{_reason(exc)}.",
                f"البيانات المُمرَّرة إلى دالة {what_ar} ليست جدولًا: "
                f"{_reason(exc)}.",
            ) from exc
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ExportError(
            f"There is nothing to export: the dataset is empty "
            f"({data.shape[0]} rows x {data.shape[1]} columns).",
            f"لا توجد بيانات لتصديرها: الجدول فارغ "
            f"({data.shape[0]} صفًّا × {data.shape[1]} عمودًا).",
            shape=tuple(data.shape),
        )
    return data


def _spec_of(ds: Any, fallback: Optional[ModelSpec] = None) -> Optional[ModelSpec]:
    """Best-effort recovery of the specification behind a dataset."""
    if fallback is not None:
        return fallback
    compiled = getattr(ds, "compiled", None)
    return getattr(compiled, "spec", None)


# --------------------------------------------------------------------------- #
# Name hygiene
# --------------------------------------------------------------------------- #


def _ascii_name(raw: str, *, allowed: str = "_.", fallback: str = "V",
                prefix: str = "V") -> str:
    """ASCII-only identifier: ``fallback`` replaces an empty name, ``prefix``
    is prepended when the first character is not a legal leading one."""
    out: List[str] = []
    for ch in (raw or "").strip():
        if ch.isascii() and ch.isalnum():
            out.append(ch)
        elif ch in allowed:
            out.append(ch)
        elif ch.isspace() or ch in "-/\\":
            out.append("_")
    cleaned = "".join(out).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit() or cleaned[0] == ".":
        cleaned = prefix + cleaned
    return cleaned


def _truncate_bytes(name: str, limit: int) -> str:
    encoded = name.encode("utf-8")
    if len(encoded) <= limit:
        return name
    return encoded[:limit].decode("utf-8", errors="ignore")


def spss_variable_names(columns: Sequence[str]) -> Dict[str, str]:
    """Map dataset column names onto legal, unique SPSS variable names.

    SPSS accepts at most 64 *bytes* per name, forbids a leading digit, forbids
    a trailing period, reserves a handful of keywords (``ALL``, ``BY``,
    ``WITH``, ...) and treats names case-insensitively for uniqueness.  This
    function enforces all of that and never collapses two columns onto one
    name.

    Parameters
    ----------
    columns
        Column names in dataset order.

    Returns
    -------
    dict of str to str
        ``{original_name: spss_name}``, in the order given.  A name that is
        already legal maps to itself.

    References
    ----------
    IBM Corp. (2021). *IBM SPSS Statistics 28 Command Syntax Reference*,
        "Variable names" (rules on length, leading characters and reserved
        keywords).
    """
    mapping: Dict[str, str] = {}
    used: Dict[str, str] = {}      # upper-case name -> original column
    for position, original in enumerate(columns, start=1):
        candidate = _ascii_name(str(original), allowed="_.@#$",
                                fallback=f"V{position}")
        candidate = candidate.rstrip(".")
        if not candidate:
            candidate = f"V{position}"
        if candidate[0].isdigit():
            candidate = f"V{candidate}"
        candidate = _truncate_bytes(candidate, SPSS_MAX_NAME_BYTES)
        if candidate.upper() in _SPSS_RESERVED:
            candidate = _truncate_bytes(candidate + "_V", SPSS_MAX_NAME_BYTES)
        base = candidate
        suffix = 1
        while candidate.upper() in used:
            suffix += 1
            tag = f"_{suffix}"
            candidate = _truncate_bytes(
                base, SPSS_MAX_NAME_BYTES - len(tag.encode("utf-8"))) + tag
        used[candidate.upper()] = str(original)
        mapping[str(original)] = candidate
    return mapping


def mplus_variable_names(
    columns: Sequence[str],
    *,
    lang: str = "en",
) -> Tuple[Dict[str, str], List[str]]:
    """Map names onto Mplus's eight-character namespace, reporting renames.

    Parameters
    ----------
    columns
        Variable names in data-file order.
    lang
        ``"ar"`` or ``"en"``; selects the language of the returned warnings.

    Returns
    -------
    mapping : dict of str to str
        ``{original: mplus_name}`` for every column.
    warnings : list of str
        One message per forced rename, naming the original and the new name.

    Notes
    -----
    Truncation alone collides (``SERVICE_QUALITY_1`` and ``SERVICE_QUALITY_2``
    both become ``SERVICE_``), so a numeric tail replaces the last characters
    until the name is unique.  Mplus names are case-insensitive.

    References
    ----------
    Muthen, L. K., & Muthen, B. O. (1998-2017). *Mplus User's Guide* (8th ed.),
        chapter 14: variable names are limited to eight characters.
    """
    mapping: Dict[str, str] = {}
    warnings: List[str] = []
    used: Dict[str, str] = {}
    for position, original in enumerate(columns, start=1):
        clean = _ascii_name(str(original), allowed="_", fallback=f"V{position}")
        candidate = clean[:MPLUS_MAX_NAME_CHARS]
        if not candidate:
            candidate = f"V{position}"[:MPLUS_MAX_NAME_CHARS]
        suffix = 1
        while candidate.upper() in used:
            suffix += 1
            tag = str(suffix)
            keep = max(MPLUS_MAX_NAME_CHARS - len(tag), 1)
            candidate = (clean[:keep] or f"V{position}"[:keep]) + tag
            candidate = candidate[:MPLUS_MAX_NAME_CHARS]
        used[candidate.upper()] = str(original)
        mapping[str(original)] = candidate
        if candidate != str(original):
            if lang == "ar":
                warnings.append(
                    f"أعاد Mplus تسمية المتغير «{original}» إلى «{candidate}» "
                    f"لأنّ الحد الأقصى لطول اسم المتغير في Mplus هو "
                    f"{MPLUS_MAX_NAME_CHARS} حروف.")
            else:
                warnings.append(
                    f"Mplus renamed {original!r} to {candidate!r}: variable "
                    f"names are limited to {MPLUS_MAX_NAME_CHARS} characters.")
    return mapping, warnings


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def export_csv(ds: "GeneratedDataset", path: Any, *,
               decimal_places: int = 4) -> None:
    """Write the observed dataset as a comma-separated file.

    Integer-valued Likert columns stay integers; continuous columns are
    rounded to ``decimal_places`` so the file round-trips exactly at the
    precision it advertises.  Missing cells are written as empty fields, which
    both pandas and R read back as ``NA``.

    Parameters
    ----------
    ds
        A ``GeneratedDataset``-like object exposing ``.data``.
    path
        Destination file.
    decimal_places
        Digits kept after the decimal point for floating-point columns.

    Raises
    ------
    ExportError
        If the dataset is empty or the file cannot be written.

    References
    ----------
    Wickham, H. (2014). Tidy data. *Journal of Statistical Software*, 59(10),
        1-23. (One row per respondent, one column per item.)
    """
    data = _frame_of(ds, what_en="CSV", what_ar="تصدير CSV")
    target = _as_fs_path(path, "CSV", "ملف CSV")
    places = max(int(decimal_places), 0)
    frame = _round_for_text(data, places)
    try:
        frame.to_csv(target, index=False, na_rep="",
                     float_format=f"%.{places}f", encoding="utf-8-sig")
    except (OSError, UnicodeError, ValueError) as exc:
        _raise_write_error(target, exc, "CSV", "ملف CSV")


def _round_for_text(data: pd.DataFrame, places: int) -> pd.DataFrame:
    """Round float columns; leave integer and object columns untouched."""
    frame = data.copy()
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            rounded = series.round(places)
            # A float column holding only whole numbers (a Likert item that
            # became float because of missing cells) is written without a
            # decimal tail, so SPSS and R read it as the integer it is.
            finite = rounded.dropna()
            if len(finite) and np.all(np.equal(np.mod(finite.to_numpy(), 1), 0)):
                frame[column] = rounded.astype("Int64")
            else:
                frame[column] = rounded
    return frame


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #


def _sheet_title(raw: str, used: Iterable[str]) -> str:
    title = re.sub(r"[\[\]:*?/\\]", "_", str(raw)).strip() or "Sheet"
    title = title[:31]
    taken = {t.lower() for t in used}
    if title.lower() not in taken:
        return title
    for k in range(2, 100):
        tag = f"_{k}"
        candidate = title[:31 - len(tag)] + tag
        if candidate.lower() not in taken:
            return candidate
    return title[:28] + "_99"  # pragma: no cover - absurd input


def _cell_value(value: Any) -> Any:
    """Convert a pandas / numpy value into something openpyxl can store."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        scalar = value.item()
        if isinstance(scalar, float) and not np.isfinite(scalar):
            return None
        return scalar
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool, _dt.date, _dt.datetime)):
        return value
    return str(value)


def _write_sheet(worksheet: Any, frame: pd.DataFrame, *,
                 freeze_header: bool = True) -> None:
    """Write a frame to a worksheet, freeze the header and autosize columns."""
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    headers = [str(c) for c in frame.columns]
    worksheet.append(headers)
    bold = Font(bold=True)
    for cell in worksheet[1]:
        cell.font = bold
    widths = [len(h) for h in headers]
    for row in frame.itertuples(index=False, name=None):
        values = [_cell_value(v) for v in row]
        worksheet.append(values)
        for position, value in enumerate(values):
            if value is not None:
                text = (f"{value:.6g}" if isinstance(value, float)
                        else str(value))
                if len(text) > widths[position]:
                    widths[position] = len(text)
    for position, width in enumerate(widths, start=1):
        worksheet.column_dimensions[get_column_letter(position)].width = \
            min(max(width + 2, 8), 52)
    if freeze_header:
        worksheet.freeze_panes = "A2"


def _report_frame(value: Any) -> pd.DataFrame:
    """Coerce a supplied report object into a writable frame."""
    if isinstance(value, pd.DataFrame):
        # A meaningful index (statistic names, item names) becomes a column;
        # a plain positional index is dropped.
        if isinstance(value.index, pd.RangeIndex) and value.index.name is None:
            return value
        return value.reset_index()
    if isinstance(value, pd.Series):
        return value.rename_axis("key").reset_index(name="value")
    to_table = getattr(value, "to_table", None)
    if callable(to_table):
        try:
            table = to_table()
        except TypeError:  # pragma: no cover - to_table(lang=...) variants
            table = to_table("en")
        if isinstance(table, pd.DataFrame):
            return table
        value = table
    if isinstance(value, Mapping):
        return pd.DataFrame({"key": [str(k) for k in value],
                             "value": [_flat(v) for v in value.values()]})
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, Mapping) for v in value):
            return pd.DataFrame(list(value))
        return pd.DataFrame({"value": [_flat(v) for v in value]})
    return pd.DataFrame({"value": [_flat(value)]})


def _flat(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)


def export_xlsx(ds: "GeneratedDataset", path: Any, *,
                include_reports: Optional[Mapping[str, Any]] = None,
                lang: str = "ar") -> None:
    """Write an Excel workbook: the data plus metadata sheets.

    Sheets written, in order:

    ``Data``
        One row per respondent, one column per item; header row frozen.
    ``Specification``
        The model specification summary (title, seed, counts, degrees of
        freedom, targets) as property / value rows.
    ``Parameters``
        The population parameters actually used to generate the data --
        loadings, structural coefficients, latent correlations and residual
        (co)variances -- in long format.
    ``LatentScores`` / ``ContinuousShadow``
        Written only when the dataset carries them.
    One sheet per entry of ``include_reports``
        Accepts frames, mappings, sequences of mappings, or any object with a
        ``to_table()`` method (so the analysis layer's report objects drop in).

    Parameters
    ----------
    ds
        A ``GeneratedDataset``-like object.
    path
        Destination ``.xlsx`` file.
    include_reports
        Optional ``{sheet_name: report}`` mapping.
    lang
        Language of the metadata sheets' labels, ``"ar"`` (default, matching
        the interface) or ``"en"``.

    Raises
    ------
    ExportError
        If openpyxl is missing, the dataset is empty, or the file cannot be
        written.

    References
    ----------
    ECMA International (2016). *ECMA-376 Office Open XML File Formats*
        (the SpreadsheetML format openpyxl writes).
    """
    data = _frame_of(ds, what_en="Excel", what_ar="تصدير Excel")
    target = _as_fs_path(path, "Excel", "ملف Excel")
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - dependency present
        raise ExportError(
            "Excel export needs the openpyxl package, which is not available "
            "in this installation.",
            "تصدير ملفات Excel يحتاج إلى حزمة openpyxl غير المتوفّرة في هذه "
            "النسخة من البرنامج.",
        ) from exc

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    _write_sheet(sheet, data)

    titles = ["Data"]
    spec = _spec_of(ds)
    meta = workbook.create_sheet(_sheet_title("Specification", titles))
    titles.append(meta.title)
    _write_sheet(meta, specification_table(ds, spec, lang=lang))

    params = workbook.create_sheet(_sheet_title("Parameters", titles))
    titles.append(params.title)
    _write_sheet(params, parameter_table(ds, spec))

    if spec is not None and spec.demographics:
        codebook = workbook.create_sheet(_sheet_title("Codebook", titles))
        titles.append(codebook.title)
        _write_sheet(codebook, demographic_codebook(spec, lang=lang))

    for attribute, title in (("latent_scores", "LatentScores"),
                             ("continuous_shadow", "ContinuousShadow")):
        extra = getattr(ds, attribute, None)
        if isinstance(extra, pd.DataFrame) and not extra.empty:
            worksheet = workbook.create_sheet(_sheet_title(title, titles))
            titles.append(worksheet.title)
            _write_sheet(worksheet, extra)

    for name, report in (include_reports or {}).items():
        worksheet = workbook.create_sheet(_sheet_title(name, titles))
        titles.append(worksheet.title)
        try:
            frame = _report_frame(report)
        except Exception as exc:  # pragma: no cover - defensive
            frame = pd.DataFrame(
                {"value": [f"<unrenderable report: {_reason(exc)}>"]})
        _write_sheet(worksheet, frame)

    try:
        workbook.save(target)
    except (OSError, ValueError) as exc:
        _raise_write_error(target, exc, "Excel", "ملف Excel")
    finally:
        try:
            workbook.close()
        except Exception:  # pragma: no cover - openpyxl always closes
            pass


# --------------------------------------------------------------------------- #
# Metadata tables (shared by the Excel writer and the HTML report)
# --------------------------------------------------------------------------- #


def _label(lang: str, ar: str, en: str) -> str:
    return ar if lang == "ar" else en


def specification_table(ds: Any, spec: Optional[ModelSpec] = None, *,
                        lang: str = "ar") -> pd.DataFrame:
    """Build the model-specification summary written to the metadata sheet.

    Parameters
    ----------
    ds
        Dataset-like object (used for ``n``, ``seed`` and warnings).
    spec
        The specification; recovered from ``ds.compiled.spec`` when omitted.
    lang
        Language of the ``property`` column.

    Returns
    -------
    pandas.DataFrame
        Columns ``property`` and ``value``.

    References
    ----------
    Boomsma, A. (2000). Reporting analyses of covariance structures.
        *Structural Equation Modeling*, 7(3), 461-483. (What a simulation
        report has to state about the model.)
    """
    spec = _spec_of(ds, spec)
    rows: List[Tuple[str, Any]] = [
        (_label(lang, "البرنامج المُولِّد", "Generating program"),
         f"{PROGRAM_NAME} {PROGRAM_VERSION}"),
        (_label(lang, "تاريخ التصدير", "Exported on"),
         _dt.date.today().isoformat()),
    ]
    if spec is not None:
        rows += [
            (_label(lang, "عنوان النموذج", "Model title"),
             spec.title.get(lang)),
            (_label(lang, "الوصف", "Description"), spec.description),
            (_label(lang, "الباحث", "Author"), spec.author),
            (_label(lang, "عدد المتغيرات الكامنة", "Latent constructs"),
             spec.n_constructs),
            (_label(lang, "عدد العوامل (مع الأبعاد)", "Factors incl. dimensions"),
             spec.n_factors),
            (_label(lang, "عدد العبارات", "Observed items"), spec.n_items),
            (_label(lang, "عدد المسارات البنائية", "Structural paths"),
             len(spec.paths)),
            (_label(lang, "التشبعات المتقاطعة المعلنة", "Declared cross-loadings"),
             len(spec.cross_loadings)),
            (_label(lang, "تغايرات الأخطاء", "Error covariances"),
             len(spec.error_covariances)),
            (_label(lang, "عامل الطريقة المشترك", "Common method factor"),
             bool(spec.common_method_bias.enabled)),
            (_label(lang, "المعالم الحرة", "Free parameters"),
             spec.count_free_parameters(getattr(ds, "compiled", None))),
            (_label(lang, "عدد العزوم", "Observed moments"),
             spec.count_moments()),
            (_label(lang, "درجات الحرية", "Degrees of freedom"),
             spec.degrees_of_freedom(getattr(ds, "compiled", None))),
            (_label(lang, "مستوى المطابقة المستهدف", "Target fit level"),
             spec.fit_target.level.value),
            (_label(lang, "RMSEA المستهدف", "Target RMSEA"),
             (spec.fit_target.rmsea
              if spec.fit_target.level.value == "custom"
              else spec.fit_target.level.target_rmsea)),
            (_label(lang, "المقدِّر", "Estimator"), spec.analysis.estimator.value),
            (_label(lang, "محرّك التوليد", "Generation engine"),
             spec.resolved_engine().value),
            (_label(lang, "آلية البيانات المفقودة", "Missing-data mechanism"),
             spec.noise.missing_mechanism.value),
            (_label(lang, "نسبة الفقدان", "Missing proportion"),
             spec.noise.missing_proportion),
        ]
    rows += [
        (_label(lang, "حجم العينة المُولَّد", "Generated sample size"),
         getattr(ds, "n", None) if getattr(ds, "n", None) is not None
         else int(pd.DataFrame(ds.data).shape[0])),
        (_label(lang, "بذرة العشوائية", "Random seed"), getattr(ds, "seed", None)),
    ]
    engine_used = getattr(ds, "engine_used", None)
    if engine_used is not None:
        rows.append((_label(lang, "المحرّك المستخدم فعليًا", "Engine used"),
                     getattr(engine_used, "value", str(engine_used))))
    for index, warning in enumerate(getattr(ds, "warnings", None) or [], start=1):
        rows.append((_label(lang, f"تنبيه {index}", f"Warning {index}"),
                     str(warning)))
    return pd.DataFrame(
        {_label(lang, "الخاصية", "property"): [r[0] for r in rows],
         _label(lang, "القيمة", "value"): [_flat(r[1]) for r in rows]})


def demographic_codebook(spec: ModelSpec, *, lang: str = "ar") -> pd.DataFrame:
    """What each code of each demographic column means.

    Parameters
    ----------
    spec
        The specification holding the demographic variables.
    lang
        Language of the labels.

    Returns
    -------
    pandas.DataFrame
        One row per category: column, role, code, label and target percent.

    References
    ----------
    ICPSR (2020). *Guide to Social Science Data Preparation and Archiving*
        (6th ed.). (What a codebook must document.)
    """
    rows = []
    for variable in spec.demographics:
        for code, (category, share) in enumerate(
                zip(variable.categories, variable.normalised_proportions()),
                start=1):
            rows.append({
                _label(lang, "العمود", "column"): variable.name,
                _label(lang, "الدور", "role"): variable.role.value,
                _label(lang, "الكود", "code"): code,
                _label(lang, "الفئة", "category"): category.label.get(lang),
                _label(lang, "النسبة المطلوبة %", "target_percent"): share * 100.0,
            })
    return pd.DataFrame(rows)


def parameter_table(ds: Any, spec: Optional[ModelSpec] = None) -> pd.DataFrame:
    """Long-format table of the population parameters behind the data.

    Reads the compiled matrices when the dataset carries a ``CompiledModel``
    (so the table shows the values actually generated from, Heywood clamping
    and attenuation included) and falls back on the specification's declared
    values otherwise.

    Returns
    -------
    pandas.DataFrame
        Columns ``block``, ``from``, ``to``, ``value``, where ``block`` is one
        of ``lambda``, ``beta``, ``phi``, ``psi``, ``theta``.

    References
    ----------
    Joreskog, K. G. (1970). A general method for analysis of covariance
        structures. *Biometrika*, 57(2), 239-251.
    """
    spec = _spec_of(ds, spec)
    compiled = getattr(ds, "compiled", None)
    rows: List[Tuple[str, str, str, float]] = []
    if compiled is not None and getattr(compiled, "Lambda", None) is not None:
        factors = list(compiled.factor_names)
        items = list(compiled.item_names)
        Lambda = np.asarray(compiled.Lambda, dtype=float)
        for j, item in enumerate(items):
            for f, factor in enumerate(factors):
                if abs(Lambda[j, f]) > 0.0:
                    rows.append(("lambda", factor, item, float(Lambda[j, f])))
        B = np.asarray(compiled.B, dtype=float)
        for i, target in enumerate(factors):
            for j, source in enumerate(factors):
                if abs(B[i, j]) > 0.0:
                    rows.append(("beta", source, target, float(B[i, j])))
        Phi = np.asarray(compiled.Phi, dtype=float)
        for i in range(len(factors)):
            for j in range(i, len(factors)):
                if i == j or abs(Phi[i, j]) > 0.0:
                    rows.append(("phi", factors[i], factors[j],
                                 float(Phi[i, j])))
        Psi = np.asarray(compiled.Psi, dtype=float)
        for i in range(len(factors)):
            rows.append(("psi", factors[i], factors[i], float(Psi[i, i])))
        Theta = np.asarray(compiled.Theta, dtype=float)
        for i in range(len(items)):
            for j in range(i, len(items)):
                if i == j or abs(Theta[i, j]) > 0.0:
                    rows.append(("theta", items[i], items[j],
                                 float(Theta[i, j])))
    elif spec is not None:
        for item, construct, dimension in spec.iter_items_with_owner():
            owner = dimension.name if dimension is not None else construct.name
            rows.append(("lambda", owner, item.name, float(item.loading)))
            rows.append(("theta", item.name, item.name,
                         float(1.0 - item.loading ** 2
                               + item.extra_error_variance)))
        for construct in spec.constructs:
            for dimension in construct.dimensions:
                rows.append(("beta", construct.name, dimension.name,
                             float(dimension.second_order_loading)))
        for path in spec.paths:
            source = spec.construct(path.source)
            target = spec.construct(path.target)
            rows.append(("beta",
                         source.name if source else path.source,
                         target.name if target else path.target,
                         float(path.resolved_coefficient())))
        for cov in spec.latent_covariances:
            a = spec.construct(cov.construct_a)
            b = spec.construct(cov.construct_b)
            rows.append(("phi", a.name if a else cov.construct_a,
                         b.name if b else cov.construct_b,
                         float(cov.correlation)))
    if not rows:
        return pd.DataFrame(columns=["block", "from", "to", "value"])
    return pd.DataFrame(rows, columns=["block", "from", "to", "value"])


# --------------------------------------------------------------------------- #
# SPSS
# --------------------------------------------------------------------------- #


def _item_by_column(spec: Optional[ModelSpec]) -> Dict[str, Item]:
    if spec is None:
        return {}
    return {item.name: item for item in spec.all_items()}


def _variable_label(item: Item, lang: str) -> str:
    """Variable label for one item: its Arabic label when the interface is
    Arabic, else the English one."""
    label = item.label.get(lang) if item.label else ""
    text = label or item.name
    if item.reverse_coded:
        text = f"{text} {_REVERSE_MARK.get(lang, _REVERSE_MARK['en'])}"
    return text


def _value_labels(scale: ScaleSpec, lang: str) -> Dict[int, str]:
    """Value labels for a Likert item: anchors at the ends, numbers between."""
    if not scale.is_ordinal or scale.n_categories < 2:
        return {}
    low = (scale.min_label.get(lang) if scale.min_label else "") or \
        _DEFAULT_MIN_LABEL.get(lang, _DEFAULT_MIN_LABEL["en"])
    high = (scale.max_label.get(lang) if scale.max_label else "") or \
        _DEFAULT_MAX_LABEL.get(lang, _DEFAULT_MAX_LABEL["en"])
    labels: Dict[int, str] = {1: low}
    for category in range(2, scale.n_categories):
        labels[category] = str(category)
    labels[scale.n_categories] = high
    return labels


def export_sav(ds: "GeneratedDataset", path: Any, spec: Optional[ModelSpec] = None,
               *, lang: str = "ar") -> Dict[str, Any]:
    """Write an SPSS ``.sav`` file carrying variable and value labels.

    Variable labels are the item labels in the interface language (Arabic when
    ``lang == "ar"``, English otherwise).  Likert items also receive value
    labels: ``1`` gets the scale's minimum anchor, ``n_categories`` the maximum
    anchor, and the categories in between are labelled with their own number,
    which is how published questionnaires print a numbered rating scale.
    Ordinal items are flagged ``measure = ordinal`` so SPSS's own dialogs treat
    them correctly.

    SPSS variable names are limited to 64 bytes and may not begin with a digit,
    so column names are sanitised; the full mapping is returned rather than
    applied silently.

    Parameters
    ----------
    ds
        A ``GeneratedDataset``-like object.
    path
        Destination ``.sav`` file.
    spec
        The specification the labels come from.  Recovered from
        ``ds.compiled.spec`` when omitted.
    lang
        ``"ar"`` (default) or ``"en"``.

    Returns
    -------
    dict
        Export metadata: ``path``, ``n_rows``, ``n_columns``,
        ``column_names`` (``{original: spss_name}``), ``renamed`` (the subset
        that changed), ``column_labels``, ``variable_value_labels`` and
        ``warnings``.  The published contract returns ``None``; a caller that
        ignores the value is unaffected.

    Raises
    ------
    ExportError
        If pyreadstat is unavailable, the dataset is empty, or the file cannot
        be written.

    References
    ----------
    Likert, R. (1932). A technique for the measurement of attitudes.
        *Archives of Psychology*, 140, 1-55. (The numbered rating scale whose
        anchors become value labels.)
    IBM Corp. (2021). *IBM SPSS Statistics 28 Command Syntax Reference*
        ("VARIABLE LABELS", "VALUE LABELS", "VARIABLE LEVEL").
    """
    data = _frame_of(ds, what_en="SPSS", what_ar="تصدير SPSS")
    target = _as_fs_path(path, "SPSS", "ملف SPSS")
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - dependency present
        raise ExportError(
            "SPSS export needs the pyreadstat package, which is not available "
            "in this installation.",
            "تصدير ملفات SPSS يحتاج إلى حزمة pyreadstat غير المتوفّرة في هذه "
            "النسخة من البرنامج.",
        ) from exc

    spec = _spec_of(ds, spec)
    mapping = spss_variable_names([str(c) for c in data.columns])
    renamed = {k: v for k, v in mapping.items() if k != v}
    warnings: List[str] = []
    for original, new in renamed.items():
        warnings.append(
            f"أعاد SPSS تسمية المتغير «{original}» إلى «{new}»."
            if lang == "ar" else
            f"SPSS name for {original!r} is {new!r}.")

    items = _item_by_column(spec)
    demographics = {variable.name: variable
                    for variable in getattr(spec, "demographics", None) or []}
    frame = data.rename(columns=mapping)
    column_labels: List[str] = []
    value_labels: Dict[str, Dict[int, str]] = {}
    measures: Dict[str, str] = {}
    for original in (str(c) for c in data.columns):
        name = mapping[original]
        variable = demographics.get(original)
        if variable is not None:
            column_labels.append(variable.label.get(lang) or original)
            value_labels[name] = {code: category.label.get(lang)
                                  for code, category in enumerate(
                                      variable.categories, start=1)}
            measures[name] = ("ordinal" if variable.kind.value == "ordinal"
                              else "nominal")
            continue
        item = items.get(original)
        if item is None:
            column_labels.append(original)
            measures[name] = "scale"
            continue
        column_labels.append(_variable_label(item, lang))
        labels = _value_labels(item.scale, lang)
        if labels:
            value_labels[name] = labels
        measures[name] = ("ordinal" if item.scale.is_ordinal else "scale")

    file_label = ""
    if spec is not None:
        file_label = _truncate_bytes(spec.title.get(lang) or "", 64)
    note = (f"Generated by {PROGRAM_NAME} {PROGRAM_VERSION}; "
            f"seed = {getattr(ds, 'seed', 'NA')}")
    try:
        pyreadstat.write_sav(
            frame, str(target),
            file_label=file_label,
            column_labels=column_labels,
            variable_value_labels=value_labels or None,
            variable_measure=measures or None,
            note=note,
        )
    except Exception as exc:
        _raise_write_error(target, exc, "SPSS", "ملف SPSS")
    return {
        "path": str(target),
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "column_names": mapping,
        "renamed": renamed,
        "column_labels": dict(zip(frame.columns, column_labels)),
        "variable_value_labels": value_labels,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# R / Mplus companion data files
# --------------------------------------------------------------------------- #


def export_rds_script(ds: "GeneratedDataset", path: Any) -> None:
    """Write an R script that reads the exported CSV and saves it as ``.rds``.

    ``.rds`` is a serialised R object, so it cannot be produced from Python
    without an R process.  What a researcher needs is therefore a tiny script
    they run once: it reads the companion CSV, declares the Likert items as
    ordered factors, and calls ``saveRDS``.

    Parameters
    ----------
    ds
        A ``GeneratedDataset``-like object.
    path
        Destination ``.R`` file.  The companion CSV is assumed to sit next to
        it with the same stem (write it with :func:`export_csv`).

    Raises
    ------
    ExportError
        If the file cannot be written.

    References
    ----------
    R Core Team (2024). *R: A Language and Environment for Statistical
        Computing* ("readRDS"/"saveRDS", serialisation format 3).
    """
    data = _frame_of(ds, what_en="R script", what_ar="سكربت R")
    target = _as_fs_path(path, "R script", "سكربت R")
    spec = _spec_of(ds)
    csv_name = target.with_suffix(".csv").name
    rds_name = target.with_suffix(".rds").name
    ordinal = [item.name for item in (spec.all_items() if spec else [])
               if item.scale.is_ordinal and item.name in data.columns]
    lines = [
        f"# {PROGRAM_NAME} {PROGRAM_VERSION}",
        "# Converts the exported CSV into an R data object (.rds).",
        f"# Seed used to generate the data: {getattr(ds, 'seed', 'NA')}",
        f"# Run from the folder that holds {csv_name}:  Rscript {target.name}",
        "",
        f'df <- read.csv("{csv_name}", header = TRUE, na.strings = c("", "NA"),',
        '               fileEncoding = "UTF-8-BOM")',
        "",
    ]
    if ordinal:
        lines.append("ordinal_items <- c(")
        lines.extend(_r_vector_lines(ordinal))
        lines.append(")")
        lines.append("for (v in ordinal_items) {")
        lines.append("  df[[v]] <- factor(df[[v]], ordered = TRUE)")
        lines.append("}")
        lines.append("")
    lines += [
        f'saveRDS(df, file = "{rds_name}")',
        f'cat("wrote {rds_name}:", nrow(df), "rows x", ncol(df), "columns\\n")',
        "",
    ]
    _write_text(target, "\n".join(lines), "R script", "سكربت R")


def export_mplus_dat(ds: "GeneratedDataset", path: Any, *,
                     decimal_places: int = 4) -> None:
    """Write the free-format, header-less data file Mplus expects.

    Mplus reads numbers only: no header row, no quoted strings, and a numeric
    missing-value code declared by ``MISSING ARE ALL (-999)``.

    Parameters
    ----------
    ds
        A ``GeneratedDataset``-like object.
    path
        Destination ``.dat`` file.
    decimal_places
        Digits kept for continuous columns.

    Raises
    ------
    ExportError
        If the dataset is empty or the file cannot be written.

    References
    ----------
    Muthen, L. K., & Muthen, B. O. (1998-2017). *Mplus User's Guide* (8th ed.),
        chapter 16 ("DATA" and "MISSING ARE").
    """
    data = _frame_of(ds, what_en="Mplus data", what_ar="بيانات Mplus")
    target = _as_fs_path(path, "Mplus data", "بيانات Mplus")
    numeric = data.apply(pd.to_numeric, errors="coerce")
    filled = numeric.fillna(float(MPLUS_MISSING_CODE))
    try:
        filled.to_csv(target, sep=" ", index=False, header=False,
                      float_format=f"%.{max(int(decimal_places), 0)}f")
    except (OSError, ValueError) as exc:
        _raise_write_error(target, exc, "Mplus data", "بيانات Mplus")


def _r_comment_vector(prefix: str, names: Sequence[str],
                      close: str) -> List[str]:
    """Wrap ``prefix c("a", "b", ...)close`` inside a commented R call."""
    head = "#              " + prefix
    pad = "#" + " " * (len(head) - 1)
    lines: List[str] = []
    current = head
    for position, name in enumerate(names):
        piece = f'"{name}"' + ("," if position < len(names) - 1 else close)
        if current not in (head, pad) and len(current) + 1 + len(piece) > _WRAP_WIDTH:
            lines.append(current)
            current = pad
        current += (" " if current.endswith(",") else "") + piece
    lines.append(current + ")")
    return lines


def _r_vector_lines(names: Sequence[str], indent: str = "  ") -> List[str]:
    """Wrap a long R character vector over several lines."""
    out: List[str] = []
    current = indent
    for position, name in enumerate(names):
        piece = f'"{name}"' + ("," if position < len(names) - 1 else "")
        if len(current) + len(piece) + 1 > _WRAP_WIDTH and current.strip():
            out.append(current.rstrip())
            current = indent
        current += piece + " "
    if current.strip():
        out.append(current.rstrip())
    return out


def _write_text(target: FsPath, text: str, what_en: str, what_ar: str) -> None:
    try:
        target.write_text(text, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _raise_write_error(target, exc, what_en, what_ar)


# --------------------------------------------------------------------------- #
# lavaan
# --------------------------------------------------------------------------- #


def _syntax_name(raw: str, fallback: str) -> str:
    return _ascii_name(raw, allowed="_.", fallback=fallback)


def _factor_display_names(spec: ModelSpec) -> Dict[str, str]:
    """``{factor_id: syntax-safe name}`` for constructs and dimensions."""
    names: Dict[str, str] = {}
    used: set[str] = set()

    def register(fid: str, raw: str, fallback: str) -> None:
        name = _syntax_name(raw, fallback)
        base = name
        counter = 1
        while name in used:
            counter += 1
            name = f"{base}{counter}"
        used.add(name)
        names[fid] = name

    for index, construct in enumerate(spec.constructs, start=1):
        register(construct.id, construct.name, f"F{index}")
        for position, dimension in enumerate(construct.dimensions, start=1):
            register(dimension.id, dimension.name,
                     f"{names[construct.id]}_D{position}")
    return names


def _item_display_names(spec: ModelSpec) -> Dict[str, str]:
    names: Dict[str, str] = {}
    used: set[str] = set()
    for index, item in enumerate(spec.all_items(), start=1):
        name = _syntax_name(item.name, f"V{index}")
        base = name
        counter = 1
        while name in used:
            counter += 1
            name = f"{base}_{counter}"
        used.add(name)
        names[item.id] = name
    return names


def _letter_label(index: int) -> str:
    """0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa'."""
    label = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("a") + remainder) + label
    return label


def _path_labels(spec: ModelSpec, reserved: Iterable[str]) -> Dict[str, str]:
    """Assign a unique coefficient label to every structural path.

    A path's own ``label`` wins when it is a usable identifier; otherwise the
    paths are labelled ``a``, ``b``, ``c``, ... in declaration order, which
    reproduces the lettering of the mediation literature (``a`` for X -> M,
    ``b`` for M -> Y, ``c`` for the direct effect).

    References
    ----------
    Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable
        distinction in social psychological research. *Journal of Personality
        and Social Psychology*, 51(6), 1173-1182.
    """
    taken = {str(r) for r in reserved}
    labels: Dict[str, str] = {}
    cursor = 0
    for path in spec.paths:
        candidate = _syntax_name(path.label, "") if path.label else ""
        if not candidate or candidate in taken:
            while True:
                candidate = _letter_label(cursor)
                cursor += 1
                if candidate not in taken:
                    break
        taken.add(candidate)
        labels[path.id] = candidate
    return labels


def _wrap_operands(prefix: str, operands: Sequence[str], joiner: str = " + ",
                   indent: str = "    ") -> List[str]:
    """Lay out ``prefix op1 + op2 + ...`` over lines of at most ``_WRAP_WIDTH``.

    The break always comes *after* the ``+``, which is what lavaan needs to
    continue a formula onto the next line.
    """
    if not operands:
        return []
    lines: List[str] = []
    current = prefix + operands[0]
    for operand in operands[1:]:
        piece = joiner + operand
        if len(current) + len(piece) > _WRAP_WIDTH:
            lines.append(current + joiner.rstrip())
            current = indent + operand
        else:
            current += piece
    lines.append(current)
    return lines


def _comment_safe(text: str) -> str:
    """Strip characters that would end the surrounding R single-quoted string."""
    return str(text).replace("'", "").replace("\n", " ").strip()


def lavaan_syntax(spec: ModelSpec, *, lang: str = "en",
                  data_name: str = "df") -> str:
    """Emit lavaan model syntax for the specified model.

    The emitted model is the model the data were generated from:

    * ``=~`` measurement lines, exactly one per latent factor, with declared
      cross-loaded items appended to the factor they also load on;
    * a second-order construct gets its own ``=~`` over its *dimension* names,
      and each dimension keeps its own ``=~`` over its items;
    * ``~`` regressions with labelled coefficients (``Y ~ a*X + b*M``);
    * ``~~`` for correlated residuals between items and for free covariances
      between exogenous factors;
    * ``:=`` definitions of every declared indirect effect and of the
      corresponding total effect.

    The whole model is wrapped in ``model <- '...'`` so it can be pasted into R
    unchanged, and the header comment names the generating program, the seed,
    and -- when the model has ordinal items -- the ``sem(..., ordered = c(...))``
    call that fits them with WLSMV.

    Parameters
    ----------
    spec
        The model specification.
    lang
        Language of the explanatory comments, ``"en"`` (default) or ``"ar"``.
    data_name
        Name of the R data frame used in the example call.

    Returns
    -------
    str
        Ready-to-paste lavaan syntax, newline-terminated.

    References
    ----------
    Rosseel, Y. (2012). lavaan: An R package for structural equation modeling.
        *Journal of Statistical Software*, 48(2), 1-36.
    Chen, F. F., Sousa, K. H., & West, S. G. (2005). Testing measurement
        invariance of second-order factor models. *Structural Equation
        Modeling*, 12(3), 471-492. (Second-order specification.)
    Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling
        strategies for assessing and comparing indirect effects.
        *Behavior Research Methods*, 40(3), 879-891. (The ``:=`` definitions.)
    """
    spec = spec.expanded()
    factor_names = _factor_display_names(spec)
    item_names = _item_display_names(spec)
    # A control term is an observed column, named as it is in the data.
    for construct in spec.constructs:
        if construct.covariate:
            factor_names[construct.id] = item_names[construct.items[0].id]
    reserved = set(factor_names.values()) | set(item_names.values())
    labels = _path_labels(spec, reserved)

    ordinal = [item_names[it.id] for it in spec.all_items()
               if it.scale.is_ordinal]
    extra_indicators: Dict[str, List[str]] = {}
    for cross in spec.cross_loadings:
        if cross.item_id in item_names and cross.factor_id in factor_names:
            extra_indicators.setdefault(cross.factor_id, []).append(
                item_names[cross.item_id])

    def indicators(fid: str, base: Sequence[str]) -> List[str]:
        return list(base) + [n for n in extra_indicators.get(fid, [])
                             if n not in base]

    ar = lang == "ar"
    # The model frees the covariances of its control terms, so they are not
    # to be held at their sample values.
    fixed_x = (", fixed.x = FALSE" if any(c.covariate for c in spec.constructs)
               else "")
    head: List[str] = [
        "# " + "=" * (_WRAP_WIDTH - 2),
        f"# lavaan syntax generated by {PROGRAM_NAME} {PROGRAM_VERSION}",
        f"# Model : {_comment_safe(spec.title.get('en') or spec.title.get('ar'))}",
        f"# Seed  : {spec.seed}  (random-number seed of the generated data)",
        (f"# Items : {spec.n_items}   Factors: {spec.n_factors}   "
         f"df: {spec.degrees_of_freedom()}"),
        f"# Date  : {_dt.date.today().isoformat()}",
        "# " + "=" * (_WRAP_WIDTH - 2),
        "# Fit in R:",
        "#   library(lavaan)",
        f'#   {data_name} <- read.csv("data.csv")',
    ]
    if ordinal:
        head.append(f'#   fit <- sem(model, data = {data_name}, '
                    f'estimator = "WLSMV"{fixed_x},')
        head.extend(_r_comment_vector("ordered = c(", ordinal, ")"))
    else:
        head.append(f'#   fit <- sem(model, data = {data_name}, '
                    f'estimator = "ML"{fixed_x})')
    head.append("#   summary(fit, fit.measures = TRUE, standardized = TRUE, "
                "rsquare = TRUE)")
    for variable in spec.demographics:
        if variable.role.value == "mga":
            head.append(f'#   fit_{variable.name} <- sem(model, data = {data_name}, '
                        f'group = "{variable.name}")')
    head.append("")

    body: List[str] = ["model <- '"]
    body.append("  # " + ("--- أنموذج القياس " if ar
                          else "--- measurement model ") + "-" * 30)
    for construct in spec.constructs:
        if construct.covariate:
            continue
        if construct.is_second_order:
            for dimension in construct.dimensions:
                base = [item_names[it.id] for it in dimension.items]
                body += ["  " + line for line in _wrap_operands(
                    f"{factor_names[dimension.id]} =~ ",
                    indicators(dimension.id, base))]
        else:
            base = [item_names[it.id] for it in construct.items]
            body += ["  " + line for line in _wrap_operands(
                f"{factor_names[construct.id]} =~ ",
                indicators(construct.id, base))]

    second_order = [c for c in spec.constructs if c.is_second_order]
    if second_order:
        body.append("")
        body.append("  # " + ("--- البنية من الرتبة الثانية " if ar
                              else "--- second-order structure ") + "-" * 24)
        for construct in second_order:
            dims = [factor_names[d.id] for d in construct.dimensions]
            body += ["  " + line for line in _wrap_operands(
                f"{factor_names[construct.id]} =~ ", dims)]

    if spec.paths:
        body.append("")
        body.append("  # " + ("--- المسارات البنائية " if ar
                              else "--- structural paths ") + "-" * 30)
        for construct in spec.constructs:
            incoming = spec.incoming_paths(construct.id)
            if not incoming:
                continue
            operands = []
            for path in incoming:
                source = factor_names.get(path.source, path.source)
                operands.append(f"{labels[path.id]}*{source}")
            body += ["  " + line for line in _wrap_operands(
                f"{factor_names[construct.id]} ~ ", operands)]

    if spec.latent_covariances:
        body.append("")
        body.append("  # " + ("--- تغايرات العوامل الخارجية " if ar
                              else "--- exogenous covariances ") + "-" * 24)
        for cov in spec.latent_covariances:
            a = factor_names.get(cov.construct_a, cov.construct_a)
            b = factor_names.get(cov.construct_b, cov.construct_b)
            body.append(f"  {a} ~~ {b}")

    if spec.error_covariances:
        body.append("")
        body.append("  # " + ("--- تغايرات أخطاء القياس " if ar
                              else "--- residual covariances ") + "-" * 26)
        for cov in spec.error_covariances:
            a = item_names.get(cov.item_a, cov.item_a)
            b = item_names.get(cov.item_b, cov.item_b)
            body.append(f"  {a} ~~ {b}")

    definitions = _indirect_definitions(spec, labels, factor_names, ar)
    if definitions:
        body.append("")
        body.append("  # " + ("--- الآثار غير المباشرة والكلية " if ar
                              else "--- indirect and total effects ") + "-" * 20)
        body += definitions

    if spec.interactions:
        body.append("")
        body.append("  # " + ("--- تفاعلات كامنة " if ar
                              else "--- latent interactions ") + "-" * 28)
        for ix in spec.interactions:
            predictor = factor_names.get(ix.predictor, ix.predictor)
            moderator = factor_names.get(ix.moderator, ix.moderator)
            target = factor_names.get(ix.target, ix.target)
            body.append(
                f"  # {target} ~ {ix.resolved_coefficient():+.3f} * "
                f"({predictor} x {moderator})")
        body += _wrap_comment(
            "  # lavaan has no operator for the product of two latent "
            "variables: build product indicators (Marsh, Wen & Hau, 2004) or "
            "use the modsem package, then add the product factor as an extra "
            "predictor.", indent="  ")

    if spec.common_method_bias.enabled:
        affected = (spec.common_method_bias.item_ids
                    or [it.id for it in spec.all_items()])
        names = [item_names[i] for i in affected if i in item_names]
        body.append("")
        body += _wrap_comment(
            "  # The data carry a common-method factor. Uncomment to run the "
            "single-method-factor test (Podsakoff et al., 2003):", indent="  ")
        for line in _wrap_operands("CMF =~ ", names):
            body.append("  # " + line)

    body.append("'")
    body.append("")
    return "\n".join(head + body) + "\n"


def _indirect_definitions(spec: ModelSpec, labels: Mapping[str, str],
                          factor_names: Mapping[str, str],
                          ar: bool) -> List[str]:
    """``:=`` lines for every declared indirect effect, plus its total."""
    by_pair: Dict[Tuple[str, str], str] = {}
    for path in spec.paths:
        by_pair[(path.source, path.target)] = labels[path.id]
    single = len(spec.indirect_effects) == 1
    lines: List[str] = []
    for index, effect in enumerate(spec.indirect_effects, start=1):
        chain = list(effect.chain)
        human = " -> ".join(factor_names.get(c, c) for c in chain)
        if len(chain) < 3:
            lines.append(f"  # {human}: "
                         + ("سلسلة الوساطة تحتاج ثلاث حلقات على الأقل."
                            if ar else
                            "a mediation chain needs at least three links."))
            continue
        pieces = [by_pair.get((chain[k], chain[k + 1]))
                  for k in range(len(chain) - 1)]
        if any(p is None for p in pieces):
            lines.append(f"  # {human}: "
                         + ("أحد المسارات في السلسلة غير معرَّف في النموذج."
                            if ar else
                            "one link of this chain has no declared path."))
            continue
        ind_name = "ind" if single else f"ind{index}"
        total_name = "total" if single else f"total{index}"
        product = "*".join(pieces)
        lines.append(f"  # {human}")
        lines.append(f"  {ind_name} := {product}")
        direct = by_pair.get((chain[0], chain[-1]))
        if direct is not None:
            lines.append(f"  {total_name} := {direct} + {product}")
    return lines


def _wrap_comment(text: str, indent: str = "") -> List[str]:
    """Wrap an already-commented line over several lines, keeping the marker.

    Works for both comment syntaxes in play here: ``#`` (R / lavaan) and ``!``
    (Mplus).
    """
    stripped = text.strip()
    marker = "!" if stripped.startswith("!") else "#"
    words = stripped.lstrip("#!").strip().split()
    prefix = f"{indent}{marker}"
    lines: List[str] = []
    current = prefix
    for word in words:
        if current != prefix and len(current) + 1 + len(word) > _WRAP_WIDTH:
            lines.append(current)
            current = prefix
        current += " " + word
    if current != prefix:
        lines.append(current)
    return lines or [prefix]


# --------------------------------------------------------------------------- #
# Mplus
# --------------------------------------------------------------------------- #


def mplus_syntax(spec: ModelSpec, data_filename: str = "data.dat", *,
                 lang: str = "en",
                 renames: Optional[MutableMapping[str, str]] = None) -> str:
    """Emit a complete Mplus input file for the specified model.

    Sections, in Mplus's required order: ``TITLE``, ``DATA``, ``VARIABLE``
    (``NAMES``, ``USEVARIABLES``, ``CATEGORICAL``, ``MISSING ARE ALL (-999)``),
    ``ANALYSIS`` (``ESTIMATOR = WLSMV`` when any item is ordinal, ``MLR``
    otherwise, and numerical integration when the model has a latent
    interaction), ``MODEL`` (``BY`` for measurement, ``ON`` for regressions,
    ``WITH`` for covariances, ``XWITH`` for latent products), an optional
    ``MODEL CONSTRAINT`` defining the indirect and total effects, and
    ``OUTPUT: STDYX; MODINDICES;``.

    Mplus truncates variable names to eight characters, so names are shortened
    (uniquely) and every forced rename is both listed in a comment block at the
    top of the file and -- when ``renames`` is supplied -- written into that
    mapping.

    Parameters
    ----------
    spec
        The model specification.
    data_filename
        Name of the free-format data file the input refers to.
    lang
        Language of the comment block, ``"en"`` (default) or ``"ar"``.
    renames
        Optional mutable mapping that receives ``{original: mplus_name}`` for
        every variable whose name had to change.

    Returns
    -------
    str
        The Mplus input file, newline-terminated.

    References
    ----------
    Muthen, L. K., & Muthen, B. O. (1998-2017). *Mplus User's Guide* (8th ed.).
        Muthen & Muthen. (Command syntax, eight-character names, ``XWITH``.)
    Klein, A., & Moosbrugger, H. (2000). Maximum likelihood estimation of
        latent interaction effects with the LMS method. *Psychometrika*, 65(4),
        457-474. (What ``XWITH`` with numerical integration fits.)
    """
    spec = spec.expanded()
    covariate_ids = spec.covariate_item_ids()
    items = [it for it in spec.all_items() if it.id not in covariate_ids]
    item_order = [it.name for it in items]
    factor_raw = {}
    for construct in spec.constructs:
        if construct.covariate:
            continue
        factor_raw[construct.id] = construct.name
        for dimension in construct.dimensions:
            factor_raw[dimension.id] = dimension.name

    # One namespace: items and latent variables must not collide after
    # truncation, so they are mapped together.
    demographic_order = spec.demographic_columns()
    combined = demographic_order + item_order + list(factor_raw.values())
    mapping, warnings = mplus_variable_names(combined, lang=lang)
    item_names = {it.id: mapping[it.name] for it in items}
    factor_names = {fid: mapping[raw] for fid, raw in factor_raw.items()}
    for construct in spec.constructs:
        if construct.covariate:
            factor_names[construct.id] = mapping[construct.name]
    control_names = [factor_names[c.id] for c in spec.constructs if c.covariate]
    file_names = ([mapping[name] for name in demographic_order]
                  + list(item_names.values()))
    renamed = {k: v for k, v in mapping.items() if k != v}
    if renames is not None:
        renames.update(renamed)

    ordinal = [item_names[it.id] for it in items if it.scale.is_ordinal]
    ar = lang == "ar"

    lines: List[str] = [
        f"! {PROGRAM_NAME} {PROGRAM_VERSION}",
        "! Mplus input generated from the simulation specification.",
        f"! Seed of the generated data: {spec.seed}",
        (f"! Items: {spec.n_items}   Factors: {spec.n_factors}   "
         f"df: {spec.degrees_of_freedom()}"),
    ]
    if renamed:
        lines.append("!")
        lines += _wrap_comment(
            "! Mplus limits variable names to eight characters. "
            "These variables were renamed:" if not ar else
            "! يحدّد Mplus أسماء المتغيرات بثمانية حروف، "
            "ولذلك أعيدت تسمية المتغيرات الآتية:")
        for original, new in renamed.items():
            lines.append(f"!   {original}  ->  {new}")
    lines.append("")

    title = _comment_safe(spec.title.get("en") or spec.title.get("ar")
                          or "Simulated SEM data")
    lines += ["TITLE:", f"  {title[:80]};", ""]
    lines += ["DATA:", f"  FILE IS {data_filename};", ""]

    lines.append("VARIABLE:")
    lines += _mplus_statement("NAMES ARE", file_names)
    lines += _mplus_statement("USEVARIABLES ARE",
                              control_names + list(item_names.values()))
    if ordinal:
        lines += _mplus_statement("CATEGORICAL ARE", ordinal)
    lines.append(f"  MISSING ARE ALL ({MPLUS_MISSING_CODE});")
    lines.append("")

    lines.append("ANALYSIS:")
    if spec.interactions:
        lines.append("  TYPE = RANDOM;")
        lines.append("  ALGORITHM = INTEGRATION;")
        lines.append("  ESTIMATOR = ML;")
    else:
        lines.append(f"  ESTIMATOR = {'WLSMV' if ordinal else 'MLR'};")
    lines.append("")

    lines.append("MODEL:")
    for construct in spec.constructs:
        if construct.covariate:
            continue
        if construct.is_second_order:
            for dimension in construct.dimensions:
                lines += _mplus_statement(
                    f"{factor_names[dimension.id]} BY",
                    _mplus_indicators(spec, dimension.id,
                                      [item_names[i.id] for i in dimension.items],
                                      item_names))
        else:
            lines += _mplus_statement(
                f"{factor_names[construct.id]} BY",
                _mplus_indicators(spec, construct.id,
                                  [item_names[i.id] for i in construct.items],
                                  item_names))
    second_order = [c for c in spec.constructs if c.is_second_order]
    if second_order:
        lines.append("")
        for construct in second_order:
            lines += _mplus_statement(
                f"{factor_names[construct.id]} BY",
                [factor_names[d.id] for d in construct.dimensions])

    labels = _path_labels(spec, set(mapping.values()))
    if spec.paths:
        lines.append("")
        for construct in spec.constructs:
            incoming = spec.incoming_paths(construct.id)
            if not incoming:
                continue
            operands = []
            for path in incoming:
                source = factor_names.get(path.source, path.source)
                operands.append(f"{source} ({labels[path.id]})")
            lines += _mplus_statement(
                f"{factor_names[construct.id]} ON", operands)

    interaction_names: Dict[str, str] = {}
    if spec.interactions:
        lines.append("")
        used = set(mapping.values())
        for index, ix in enumerate(spec.interactions, start=1):
            name = f"XW{index}"
            while name.upper() in {u.upper() for u in used}:
                index += 1
                name = f"XW{index}"
            used.add(name)
            interaction_names[ix.id] = name
            predictor = factor_names.get(ix.predictor, ix.predictor)
            moderator = factor_names.get(ix.moderator, ix.moderator)
            lines.append(f"  {name} | {predictor} XWITH {moderator};")
        for ix in spec.interactions:
            target = factor_names.get(ix.target, ix.target)
            lines.append(f"  {target} ON {interaction_names[ix.id]};")

    covariance_lines: List[str] = []
    for cov in spec.latent_covariances:
        a = factor_names.get(cov.construct_a, cov.construct_a)
        b = factor_names.get(cov.construct_b, cov.construct_b)
        covariance_lines.append(f"  {a} WITH {b};")
    for cov in spec.error_covariances:
        a = item_names.get(cov.item_a, cov.item_a)
        b = item_names.get(cov.item_b, cov.item_b)
        covariance_lines.append(f"  {a} WITH {b};")
    if covariance_lines:
        lines.append("")
        lines += covariance_lines

    constraints = _mplus_constraints(spec, labels, factor_names)
    if constraints:
        lines.append("")
        lines.append("MODEL CONSTRAINT:")
        lines += constraints

    lines += ["", "OUTPUT:", "  STDYX;", "  MODINDICES;", ""]
    if warnings:
        lines.append("")
        for warning in warnings:
            lines += _wrap_comment("! " + warning)
    return "\n".join(lines) + "\n"


def _mplus_indicators(spec: ModelSpec, factor_id: str, base: List[str],
                      item_names: Mapping[str, str]) -> List[str]:
    extra = [item_names[c.item_id] for c in spec.cross_loadings
             if c.factor_id == factor_id and c.item_id in item_names]
    return base + [n for n in extra if n not in base]


def _mplus_statement(prefix: str, operands: Sequence[str],
                     indent: str = "  ") -> List[str]:
    """Lay out one Mplus statement, wrapped and terminated with ``;``."""
    if not operands:
        return []
    lines: List[str] = []
    current = f"{indent}{prefix} {operands[0]}"
    for operand in operands[1:]:
        if len(current) + len(operand) + 1 > _WRAP_WIDTH:
            lines.append(current)
            current = f"{indent}  {operand}"
        else:
            current += f" {operand}"
    lines.append(current + ";")
    return lines


def _mplus_constraints(spec: ModelSpec, labels: Mapping[str, str],
                       factor_names: Mapping[str, str]) -> List[str]:
    """``NEW(...)`` plus the arithmetic for every declared indirect effect."""
    by_pair = {(p.source, p.target): labels[p.id] for p in spec.paths}
    single = len(spec.indirect_effects) == 1
    new_names: List[str] = []
    body: List[str] = []
    for index, effect in enumerate(spec.indirect_effects, start=1):
        chain = list(effect.chain)
        if len(chain) < 3:
            continue
        pieces = [by_pair.get((chain[k], chain[k + 1]))
                  for k in range(len(chain) - 1)]
        if any(p is None for p in pieces):
            continue
        ind_name = "IND" if single else f"IND{index}"
        total_name = "TOTAL" if single else f"TOTAL{index}"
        product = "*".join(pieces)
        human = " -> ".join(factor_names.get(c, c) for c in chain)
        body.append(f"  ! {human}")
        body.append(f"  {ind_name} = {product};")
        new_names.append(ind_name)
        direct = by_pair.get((chain[0], chain[-1]))
        if direct is not None:
            body.append(f"  {total_name} = {direct} + {product};")
            new_names.append(total_name)
    if not new_names:
        return []
    return [f"  NEW({' '.join(new_names)});"] + body


# --------------------------------------------------------------------------- #
# AMOS
# --------------------------------------------------------------------------- #


def amos_notes(spec: ModelSpec, lang: str = "en") -> str:
    """Step-by-step instructions for rebuilding the model in AMOS.

    AMOS has no text model language a researcher pastes, so what helps is an
    ordered build sheet: which observed variables to drag in, which ellipses to
    draw, where the arrows go, which residual covariances to add, and which
    estimation options to switch on.

    Parameters
    ----------
    spec
        The model specification.
    lang
        ``"en"`` (default) or ``"ar"``.

    Returns
    -------
    str
        Plain text, newline-terminated.

    References
    ----------
    Arbuckle, J. L. (2019). *Amos 26 User's Guide*. IBM. (Graphics interface,
        "Analysis Properties", bootstrap for indirect effects.)
    Byrne, B. M. (2016). *Structural Equation Modeling with AMOS* (3rd ed.).
        Routledge.
    """
    ar = lang == "ar"
    factor_names = _factor_display_names(spec)
    item_names = _item_display_names(spec)
    out: List[str] = []
    add = out.append
    add((f"تعليمات بناء النموذج في AMOS — {spec.title.get('ar')}" if ar else
         f"AMOS build sheet — {spec.title.get('en')}"))
    add("=" * _WRAP_WIDTH)
    add(("البرنامج المُولِّد: " if ar else "Generated by: ")
        + f"{PROGRAM_NAME} {PROGRAM_VERSION}")
    add(("بذرة العشوائية: " if ar else "Seed: ") + str(spec.seed))
    add("")
    add(("1) استورد ملف البيانات (File > Data Files) واختر الملف المُصدَّر."
         if ar else
         "1) File > Data Files, and point AMOS at the exported file."))
    add(("2) ارسم العوامل الكامنة الآتية (بيضاوية) مع مؤشراتها (مستطيلات):"
         if ar else
         "2) Draw these latent factors (ellipses) with their indicators:"))
    for construct in spec.constructs:
        if construct.is_second_order:
            dims = ", ".join(factor_names[d.id] for d in construct.dimensions)
            add(f"   - {factor_names[construct.id]} "
                + ("(من الرتبة الثانية) <- " if ar else "(second order) <- ")
                + dims)
            for dimension in construct.dimensions:
                names = ", ".join(item_names[i.id] for i in dimension.items)
                add(f"     - {factor_names[dimension.id]} <- {names}")
        else:
            names = ", ".join(item_names[i.id] for i in construct.items)
            add(f"   - {factor_names[construct.id]} <- {names}")
    add(("3) أضف حدًّا للخطأ (e) لكل مؤشر، وحدًّا للبواقي (d) لكل عامل داخلي."
         if ar else
         "3) Add an error term (e) to every indicator and a disturbance (d) to "
         "every endogenous factor."))
    if spec.paths:
        add(("4) ارسم المسارات البنائية الآتية (أسهم أحادية الاتجاه):" if ar
             else "4) Draw these structural paths (single-headed arrows):"))
        for path in spec.paths:
            source = factor_names.get(path.source, path.source)
            target = factor_names.get(path.target, path.target)
            add(f"   - {source} -> {target}  "
                f"({'المعامل' if ar else 'coefficient'} "
                f"{path.resolved_coefficient():+.3f})")
    if spec.latent_covariances:
        add(("5) ارسم تغايرات (أسهم ثنائية الاتجاه) بين العوامل الخارجية:"
             if ar else
             "5) Draw covariances (double-headed arrows) between the "
             "exogenous factors:"))
        for cov in spec.latent_covariances:
            a = factor_names.get(cov.construct_a, cov.construct_a)
            b = factor_names.get(cov.construct_b, cov.construct_b)
            add(f"   - {a} <-> {b}  (r = {cov.correlation:+.3f})")
    if spec.error_covariances:
        add(("6) اربط أخطاء العبارات الآتية بسهم ثنائي الاتجاه:" if ar else
             "6) Connect the residuals of these item pairs:"))
        for cov in spec.error_covariances:
            a = item_names.get(cov.item_a, cov.item_a)
            b = item_names.get(cov.item_b, cov.item_b)
            add(f"   - e({a}) <-> e({b})  (r = {cov.correlation:+.3f})")
    if spec.indirect_effects:
        add(("7) من Analysis Properties فعّل Indirect, direct & total effects "
             "ضمن Output، وفعّل Bootstrap (2000 إعادة معاينة) مع "
             "Bias-corrected confidence intervals لاختبار الوساطة."
             if ar else
             "7) Analysis Properties > Output: tick 'Indirect, direct & total "
             "effects'; Bootstrap: 2000 resamples with bias-corrected "
             "confidence intervals, for the mediation tests."))
    if spec.interactions:
        add(("8) لا يتعامل AMOS مع تفاعل المتغيرات الكامنة مباشرة: أنشئ "
             "مؤشرات حاصل الضرب بعد توسيط المؤشرات (Marsh, Wen & Hau, 2004)، "
             "أو استخدم تحليل المجموعات المتعددة."
             if ar else
             "8) AMOS has no latent product term: build mean-centred product "
             "indicators (Marsh, Wen & Hau, 2004), or run a multi-group "
             "moderation analysis instead."))
    if any(it.scale.is_ordinal for it in spec.all_items()):
        add(("ملاحظة: البيانات رُتبية. يقدّر AMOS بطريقة ML على مصفوفة "
             "التغايرات؛ ولتقدير أدق استخدم Bayesian estimation أو برنامجًا "
             "يدعم WLSMV مثل Mplus أو lavaan."
             if ar else
             "Note: the items are ordinal. AMOS fits ML on the covariance "
             "matrix; for a categorical-data estimator use Bayesian estimation "
             "in AMOS, or WLSMV in Mplus or lavaan."))
    add("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# One-shot export
# --------------------------------------------------------------------------- #


def export_all(ds: "GeneratedDataset", directory: Any,
               spec: Optional[ModelSpec] = None,
               reports: Optional[Mapping[str, Any]] = None,
               *, stem: str = "simulated_data", lang: str = "ar") -> List[str]:
    """Write every output the specification asks for and list what was written.

    Honours ``spec.export``: the requested ``formats`` (``csv``, ``xlsx``,
    ``sav``, ``rds``, ``dat``), ``include_latent_scores``,
    ``include_continuous_shadow``, ``decimal_places`` and the three syntax
    switches (``emit_lavaan_syntax``, ``emit_mplus_syntax``,
    ``emit_amos_hint``).  A CSV is always written, because every other tool can
    read one.

    Parameters
    ----------
    ds
        A ``GeneratedDataset``-like object.
    directory
        Existing destination folder.
    spec
        The specification; recovered from ``ds.compiled.spec`` when omitted.
    reports
        Optional ``{sheet_name: report}`` mapping forwarded to the workbook.
    stem
        Base file name shared by every artefact.
    lang
        Interface language for labels and comments.

    Returns
    -------
    list of str
        Absolute paths of the files written, in the order written.

    Raises
    ------
    ExportError
        If the folder does not exist or any writer fails.

    References
    ----------
    Nosek, B. A., et al. (2015). Promoting an open research culture.
        *Science*, 348(6242), 1422-1425. (Shipping data, syntax and provenance
        together.)
    """
    _frame_of(ds, what_en="export", what_ar="التصدير")   # validate early
    folder = _as_fs_path(directory, "export", "التصدير")
    if not folder.is_dir():
        raise ExportError(
            f"The export folder does not exist: {folder}.",
            f"مجلد التصدير غير موجود: «{folder}». اختر مجلدًا موجودًا "
            f"ولديك صلاحية الكتابة فيه.",
            path=str(folder),
        )
    spec = _spec_of(ds, spec)
    export_spec = getattr(spec, "export", None)
    formats = [str(f).lower() for f in getattr(export_spec, "formats", None)
               or ["csv"]]
    places = int(getattr(export_spec, "decimal_places", 4) or 4)
    base = _ascii_name(stem, allowed="_.-", fallback="simulated_data")
    written: List[str] = []

    csv_path = folder / f"{base}.csv"
    export_csv(ds, csv_path, decimal_places=places)
    written.append(str(csv_path))

    if "xlsx" in formats or "excel" in formats:
        xlsx_path = folder / f"{base}.xlsx"
        export_xlsx(ds, xlsx_path, include_reports=reports, lang=lang)
        written.append(str(xlsx_path))
    if "sav" in formats or "spss" in formats:
        sav_path = folder / f"{base}.sav"
        export_sav(ds, sav_path, spec, lang=lang)
        written.append(str(sav_path))
    if "rds" in formats or "r" in formats:
        r_path = folder / f"{base}.R"
        export_rds_script(ds, r_path)
        written.append(str(r_path))

    if getattr(export_spec, "include_latent_scores", False):
        scores = getattr(ds, "latent_scores", None)
        if isinstance(scores, pd.DataFrame) and not scores.empty:
            scores_path = folder / f"{base}_latent_scores.csv"
            export_csv(_Frame(scores), scores_path, decimal_places=places)
            written.append(str(scores_path))
    if getattr(export_spec, "include_continuous_shadow", False):
        shadow = getattr(ds, "continuous_shadow", None)
        if isinstance(shadow, pd.DataFrame) and not shadow.empty:
            shadow_path = folder / f"{base}_continuous.csv"
            export_csv(_Frame(shadow), shadow_path, decimal_places=places)
            written.append(str(shadow_path))

    if spec is not None and getattr(export_spec, "emit_lavaan_syntax", True):
        lavaan_path = folder / f"{base}_lavaan.R"
        _write_text(lavaan_path, lavaan_syntax(spec, lang=lang),
                    "lavaan syntax", "صيغة lavaan")
        written.append(str(lavaan_path))
    if spec is not None and getattr(export_spec, "emit_mplus_syntax", False):
        dat_path = folder / f"{base}.dat"
        export_mplus_dat(ds, dat_path, decimal_places=places)
        written.append(str(dat_path))
        inp_path = folder / f"{base}.inp"
        _write_text(inp_path, mplus_syntax(spec, dat_path.name, lang=lang),
                    "Mplus syntax", "صيغة Mplus")
        written.append(str(inp_path))
    if spec is not None and getattr(export_spec, "emit_amos_hint", False):
        amos_path = folder / f"{base}_amos.txt"
        _write_text(amos_path, amos_notes(spec, lang=lang),
                    "AMOS notes", "تعليمات AMOS")
        written.append(str(amos_path))
    return written


class _Frame:
    """Minimal dataset-like wrapper so the CSV writer can take a bare frame."""

    __slots__ = ("data", "seed")

    def __init__(self, data: pd.DataFrame, seed: Any = None) -> None:
        self.data = data
        self.seed = seed

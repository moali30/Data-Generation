"""
Numerical and textual tests for :mod:`semsim.io.exporters`.

Every check is against something independently known, not merely "it ran":

* the CSV round-trip is compared cell by cell with the frame rounded to the
  exported precision, and the worst deviation is required to be no more than
  half of the last exported digit (``0.5e-4`` at four decimals);
* the workbook is reopened with openpyxl and its sheet names, header row,
  frozen pane and a sample cell are read back;
* the ``.sav`` file is reopened with pyreadstat and the variable labels and
  value labels that come back are compared with the Arabic strings that went
  in -- the observed metadata is printed;
* the population-parameter sheet is checked against the closed-form values of
  the minimal CFA (``lambda = 0.80``, ``theta = 1 - 0.80**2 = 0.36``,
  ``phi_12 = 0.40``);
* the lavaan syntax for ``second_order_hierarchy()`` is asserted on as text:
  exactly one ``=~`` line per factor, four of them belonging to the
  second-order construct, and ``ind := a*b`` / ``total := c + a*b``;
* an item name longer than Mplus's eight characters must be renamed, uniquely,
  and the rename must be recorded;
* every writer must raise :class:`~semsim.core.errors.ExportError` -- never a
  bare ``OSError`` -- for an unwritable path, with a non-empty Arabic message.

Run with::

    python3 -m pytest tests/test_exporters.py -q

The file also runs as a plain script and then prints PASS/FAIL per check.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semsim.core import examples
from semsim.core.compile import compile_model
from semsim.core.errors import ExportError
from semsim.core.spec import (Bilingual, ConstructRole, EffectSize,
                              IndirectEffect, ModelSpec, ScaleKind, ScaleSpec)
from semsim.io import exporters as X

ARABIC = re.compile(r"[؀-ۿ]")


def has_arabic(text: str) -> bool:
    return bool(ARABIC.search(text or ""))


# --------------------------------------------------------------------------- #
# A stand-in for GeneratedDataset (owned by semsim/generation/engine.py)
# --------------------------------------------------------------------------- #


class FakeDataset:
    """Duck-type of ``GeneratedDataset``: only the attributes exporters read."""

    def __init__(self, data: pd.DataFrame, *, compiled: Any = None,
                 latent_scores: Optional[pd.DataFrame] = None,
                 continuous_shadow: Optional[pd.DataFrame] = None,
                 n: Optional[int] = None, seed: int = 20240202,
                 warnings: Optional[List[str]] = None) -> None:
        self.data = data
        self.compiled = compiled
        self.latent_scores = latent_scores
        self.continuous_shadow = continuous_shadow
        self.n = n if n is not None else int(data.shape[0])
        self.seed = seed
        self.warnings = list(warnings or [])


def _likert_spec() -> ModelSpec:
    """Two five-point Likert items carrying Arabic labels and anchors."""
    spec = ModelSpec(title=Bilingual(ar="مقياس تجريبي", en="Test scale"),
                     seed=777)
    construct = spec.add_construct(
        "SAT", label_ar="الرضا الوظيفي", label_en="Job satisfaction",
        role=ConstructRole.INDEPENDENT, n_items=2, loading=0.8, n_categories=5)
    for item in construct.items:
        item.scale.min_label = Bilingual(ar="لا أوافق بشدة",
                                         en="Strongly disagree")
        item.scale.max_label = Bilingual(ar="أوافق بشدة", en="Strongly agree")
    construct.items[0].label = Bilingual(ar="أنا راضٍ عن عملي الحالي",
                                         en="I am satisfied with my job")
    construct.items[1].label = Bilingual(ar="أنصح الآخرين بالعمل هنا",
                                         en="I would recommend working here")
    construct.items[1].reverse_coded = True
    spec.ensure_unique_item_names()
    return spec


def _mixed_dataset(n: int = 40) -> FakeDataset:
    """Likert integers, a continuous column with long decimals, and a gap."""
    rng = np.random.default_rng(4321)
    frame = pd.DataFrame({
        "SAT1": rng.integers(1, 6, size=n),
        "SAT2": rng.integers(1, 6, size=n),
        "SCORE": rng.normal(size=n) * 3.14159265358979,
    })
    frame.loc[3, "SCORE"] = np.nan
    return FakeDataset(frame, seed=4321, warnings=["قُصَّ تشبع واحد (Heywood)."])


def _cfa_dataset(n: int = 25) -> FakeDataset:
    spec = examples.minimal_cfa()
    compiled = compile_model(spec)
    rng = np.random.default_rng(99)
    frame = pd.DataFrame(
        rng.normal(size=(n, compiled.n_items)).round(6),
        columns=compiled.item_names)
    scores = pd.DataFrame(rng.normal(size=(n, compiled.n_factors)).round(6),
                          columns=compiled.factor_names)
    return FakeDataset(frame, compiled=compiled, latent_scores=scores,
                       seed=12345)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def test_csv_round_trips_to_the_exported_precision() -> None:
    ds = _mixed_dataset()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "data.csv")
        X.export_csv(ds, path, decimal_places=4)
        back = pd.read_csv(path)

    assert list(back.columns) == list(ds.data.columns)
    assert back.shape == ds.data.shape

    expected = ds.data.round(4)
    pd.testing.assert_frame_equal(
        back.astype(float), expected.astype(float),
        check_dtype=False, atol=1e-12, rtol=0)

    # Likert columns stayed integral, the gap stayed a gap.
    assert back["SAT1"].dtype.kind in "iu", back["SAT1"].dtype
    assert bool(back["SCORE"].isna().iloc[3])
    assert int(back["SCORE"].isna().sum()) == 1

    # Worst deviation from the *unrounded* original, which is what "exported
    # precision" means: half of the last written digit.
    diff = (back["SCORE"] - ds.data["SCORE"]).abs().max()
    print(f"  CSV worst |read-back - original| = {diff:.3e}")
    assert diff <= 0.5e-4 + 1e-12, diff


def test_csv_decimal_places_are_honoured() -> None:
    ds = FakeDataset(pd.DataFrame({"x": [1.0 / 3.0, 2.0 / 3.0]}))
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "two.csv")
        X.export_csv(ds, path, decimal_places=2)
        text = open(path, encoding="utf-8-sig").read()
    assert "0.33" in text and "0.67" in text, text
    assert "0.3333" not in text, text


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #


def test_xlsx_sheets_header_and_freeze_pane() -> None:
    from openpyxl import load_workbook

    ds = _cfa_dataset()
    reports = {"Reliability": pd.DataFrame(
        {"factor": ["F1", "F2"], "alpha": [0.84, 0.86]})}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "book.xlsx")
        X.export_xlsx(ds, path, include_reports=reports)
        workbook = load_workbook(path)
        names = workbook.sheetnames
        data_sheet = workbook["Data"]
        header = [cell.value for cell in data_sheet[1]]
        freeze = data_sheet.freeze_panes
        first_value = data_sheet.cell(row=2, column=1).value
        spec_pairs = {row[0]: row[1] for row in
                      workbook["Specification"].iter_rows(min_row=2,
                                                          values_only=True)}
        report_header = [c.value for c in workbook["Reliability"][1]]
        workbook.close()

    print(f"  workbook sheets: {names}")
    assert names[:3] == ["Data", "Specification", "Parameters"], names
    assert "LatentScores" in names
    assert "Reliability" in names
    assert header == list(ds.data.columns), header
    assert freeze == "A2", freeze
    assert abs(float(first_value) - float(ds.data.iloc[0, 0])) < 1e-9
    assert report_header == ["factor", "alpha"]
    # The seed is part of the metadata a reader needs to reproduce the file.
    assert str(ds.seed) in {str(v) for v in spec_pairs.values()}


def test_xlsx_report_sheet_keeps_a_meaningful_index() -> None:
    from openpyxl import load_workbook

    ds = _cfa_dataset()
    described = ds.data.describe()           # index = count, mean, std, ...
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "rep.xlsx")
        X.export_xlsx(ds, path, include_reports={"Descriptives": described,
                                                 "Notes": {"engine": "stub"}})
        workbook = load_workbook(path)
        header = [c.value for c in workbook["Descriptives"][1]]
        first_column = [workbook["Descriptives"].cell(row=r, column=1).value
                        for r in range(2, 2 + len(described.index))]
        notes = [[c.value for c in row]
                 for row in workbook["Notes"].iter_rows(min_row=1)]
        workbook.close()
    print(f"  report header = {header}")
    assert header[0] == "index" and header[1:] == list(described.columns)
    assert first_column == list(described.index), first_column
    assert notes[0] == ["key", "value"] and notes[1] == ["engine", "stub"]


def test_xlsx_parameter_sheet_matches_the_closed_form() -> None:
    from openpyxl import load_workbook

    ds = _cfa_dataset()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "params.xlsx")
        X.export_xlsx(ds, path)
        workbook = load_workbook(path)
        rows = list(workbook["Parameters"].iter_rows(min_row=2,
                                                     values_only=True))
        workbook.close()

    frame = pd.DataFrame(rows, columns=["block", "from", "to", "value"])
    lam = frame[frame["block"] == "lambda"]["value"].to_numpy(dtype=float)
    theta = frame[(frame["block"] == "theta")
                  & (frame["from"] == frame["to"])]["value"].to_numpy(float)
    phi = frame[(frame["block"] == "phi")
                & (frame["from"] != frame["to"])]["value"].to_numpy(float)
    print(f"  lambda = {np.unique(np.round(lam, 10))}, "
          f"theta = {np.unique(np.round(theta, 10))}, phi_offdiag = {phi}")
    # minimal_cfa(): six items loading 0.80 on two factors correlated 0.40.
    assert len(lam) == 6 and np.allclose(lam, 0.80, atol=1e-12)
    assert len(theta) == 6 and np.allclose(theta, 1.0 - 0.80 ** 2, atol=1e-12)
    assert len(phi) == 1 and abs(phi[0] - 0.40) < 1e-12


# --------------------------------------------------------------------------- #
# SPSS
# --------------------------------------------------------------------------- #


def test_sav_preserves_names_variable_labels_and_value_labels() -> None:
    import pyreadstat

    spec = _likert_spec()
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({name: rng.integers(1, 6, size=30)
                          for name in spec.all_item_names()})
    ds = FakeDataset(frame, seed=777)

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "out.sav")
        meta_out = X.export_sav(ds, path, spec, lang="ar")
        back, meta = pyreadstat.read_sav(path)

    print(f"  sav column_names        = {meta.column_names}")
    print(f"  sav column_labels       = {meta.column_labels}")
    print(f"  sav value_labels        = {meta.variable_value_labels}")
    print(f"  sav measure             = {getattr(meta, 'variable_measure', {})}")

    assert list(back.columns) == list(frame.columns)
    assert meta.column_names == list(frame.columns)
    pd.testing.assert_frame_equal(back.astype(int), frame.astype(int),
                                  check_dtype=False)

    items = spec.all_items()
    expected_labels = [
        "أنا راضٍ عن عملي الحالي",
        "أنصح الآخرين بالعمل هنا (عبارة عكسية)",
    ]
    assert meta.column_labels == expected_labels, meta.column_labels
    assert all(has_arabic(label) for label in meta.column_labels)

    for item in items:
        got = meta.variable_value_labels[item.name]
        keys = sorted(int(k) for k in got)
        assert keys == [1, 2, 3, 4, 5], keys
        assert got[1.0] == "لا أوافق بشدة", got
        assert got[5.0] == "أوافق بشدة", got
        assert [got[float(k)] for k in (2, 3, 4)] == ["2", "3", "4"], got

    assert meta_out["renamed"] == {}
    assert meta_out["n_rows"] == 30 and meta_out["n_columns"] == 2


def test_sav_sanitises_illegal_variable_names_and_reports_the_mapping() -> None:
    import pyreadstat

    long_name = "service_quality_" + "x" * 60          # 76 characters
    frame = pd.DataFrame({
        "9lives": [1, 2, 3],
        "has space": [1.5, 2.5, 3.5],
        long_name: [4, 5, 6],
        "WITH": [1, 1, 1],
    })
    ds = FakeDataset(frame, seed=1)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "names.sav")
        meta_out = X.export_sav(ds, path, None, lang="ar")
        back, meta = pyreadstat.read_sav(path)

    mapping = meta_out["column_names"]
    print(f"  spss name mapping = {mapping}")
    assert mapping["9lives"] == "V9lives"
    assert mapping["has space"] == "has_space"
    assert mapping["WITH"] == "WITH_V"            # reserved keyword
    assert len(mapping[long_name].encode("utf-8")) == X.SPSS_MAX_NAME_BYTES
    assert meta.column_names == list(mapping.values())
    assert len(set(n.upper() for n in mapping.values())) == 4
    assert set(meta_out["renamed"]) == {"9lives", "has space", long_name,
                                        "WITH"}
    assert back.shape == (3, 4)


def test_value_labels_scale_with_the_number_of_categories() -> None:
    scale = ScaleSpec(n_categories=7)
    labels = X._value_labels(scale, "ar")
    assert sorted(labels) == list(range(1, 8))
    assert labels[1] == X._DEFAULT_MIN_LABEL["ar"]
    assert labels[7] == X._DEFAULT_MAX_LABEL["ar"]
    assert [labels[k] for k in range(2, 7)] == ["2", "3", "4", "5", "6"]
    # Continuous items get no value labels at all.
    assert X._value_labels(ScaleSpec(kind=ScaleKind.CONTINUOUS), "ar") == {}


# --------------------------------------------------------------------------- #
# lavaan
# --------------------------------------------------------------------------- #


def test_lavaan_second_order_has_one_measurement_line_per_factor() -> None:
    spec = examples.second_order_hierarchy()
    syntax = X.lavaan_syntax(spec)
    body = syntax.split("model <- '", 1)[1]
    measurement = [line.strip() for line in body.splitlines()
                   if "=~" in line and not line.strip().startswith("#")]
    print("  lavaan =~ lines:")
    for line in measurement:
        print(f"    {line}")

    factor_names = ["SQ", "SQ_D1", "SQ_D2", "SQ_D3", "CSAT", "LOY"]
    assert len(measurement) == len(factor_names) == 6, measurement
    heads = [line.split("=~")[0].strip() for line in measurement]
    assert sorted(heads) == sorted(factor_names), heads

    # The second-order construct contributes exactly four lines: its own, over
    # the dimension names, plus one per dimension over that dimension's items.
    sq_block = [line for line in measurement
                if line.split("=~")[0].strip() in
                ("SQ", "SQ_D1", "SQ_D2", "SQ_D3")]
    assert len(sq_block) == 4, sq_block
    assert "SQ =~ SQ_D1 + SQ_D2 + SQ_D3" in measurement
    assert "SQ_D1 =~ SQ11 + SQ12 + SQ13 + SQ14" in measurement
    assert "SQ_D2 =~ SQ21 + SQ22 + SQ23 + SQ24" in measurement
    assert "SQ_D3 =~ SQ31 + SQ32 + SQ33 + SQ34" in measurement
    assert "CSAT =~ CSAT1 + CSAT2 + CSAT3 + CSAT4" in measurement


def test_lavaan_labels_regressions_and_defines_the_indirect_effect() -> None:
    spec = examples.second_order_hierarchy()
    syntax = X.lavaan_syntax(spec)
    lines = [line.strip() for line in syntax.splitlines()]
    print("  lavaan structural / defined lines:")
    for line in lines:
        if line.startswith(("CSAT ~", "LOY ~", "ind ", "total ")):
            print(f"    {line}")

    assert "CSAT ~ a*SQ" in lines, [l for l in lines if l.startswith("CSAT ~")]
    assert "LOY ~ b*CSAT + c*SQ" in lines, [l for l in lines
                                            if l.startswith("LOY ~")]
    assert "ind := a*b" in lines
    assert "total := c + a*b" in lines
    assert sum(1 for line in lines if ":=" in line) == 2

    # Header provenance and the pasteable R wrapper.
    assert X.PROGRAM_NAME in syntax
    assert "Seed  : 20240202" in syntax
    assert syntax.count("model <- '") == 1
    assert syntax.rstrip().endswith("'")
    # Seven-point Likert items must be declared ordinal for WLSMV.
    assert 'estimator = "WLSMV"' in syntax
    assert 'ordered = c("SQ11",' in syntax
    assert '"LOY4"))' in syntax


def test_lavaan_covariances_cross_loadings_and_continuous_items() -> None:
    spec = examples.minimal_cfa()
    syntax = X.lavaan_syntax(spec)
    lines = [line.strip() for line in syntax.splitlines()]
    assert "F1 =~ F11 + F12 + F13" in lines
    assert "F1 ~~ F2" in lines               # exogenous covariance
    assert "ordered = c(" not in syntax      # continuous items
    assert 'estimator = "ML"' in syntax

    messy = examples.messy_reality()
    text = X.lavaan_syntax(messy)
    body = [line.strip() for line in text.split("model <- '", 1)[1].splitlines()]
    # Declared residual covariances.
    assert "ATT3 ~~ ATT4" in body
    assert "INT1 ~~ INT2" in body
    # A declared cross-loading joins the other factor's measurement line.
    norm_line = next(l for l in body if l.startswith("NORM =~"))
    print(f"  cross-loaded NORM line: {norm_line}")
    assert "ATT1" in norm_line, norm_line
    # One =~ line per factor, still.
    heads = [l.split("=~")[0].strip() for l in body
             if "=~" in l and not l.startswith("#")]
    assert sorted(heads) == ["ATT", "BEH", "INT", "NORM", "PBC"], heads
    # The method factor is offered as a commented-out test, not fitted.
    assert "# CMF =~" in text


def test_lavaan_indirect_effect_without_a_direct_path_defines_only_ind() -> None:
    spec = ModelSpec(seed=11)
    x = spec.add_construct("X", n_items=3)
    m = spec.add_construct("M", n_items=3, loading=0.7)
    y = spec.add_construct("Y", n_items=3, loading=0.7)
    spec.add_path("X", "M", effect_size=EffectSize.STRONG)
    spec.add_path("M", "Y", effect_size=EffectSize.MEDIUM)
    spec.indirect_effects.append(IndirectEffect(chain=[x.id, m.id, y.id]))
    spec.ensure_unique_item_names()
    lines = [line.strip() for line in X.lavaan_syntax(spec).splitlines()]
    assert "ind := a*b" in lines
    assert not any(line.startswith("total :=") for line in lines)


# --------------------------------------------------------------------------- #
# Mplus
# --------------------------------------------------------------------------- #


def _long_name_spec() -> ModelSpec:
    spec = ModelSpec(title=Bilingual(ar="أسماء طويلة", en="Long names"),
                     seed=31337)
    spec.add_construct("SATISFACTION", label_en="Satisfaction", n_items=3,
                       loading=0.8)
    spec.add_construct("PERFORMANCE", label_en="Performance", n_items=3,
                       loading=0.8)
    spec.add_path("SATISFACTION", "PERFORMANCE", effect_size=EffectSize.MEDIUM)
    spec.ensure_unique_item_names()
    return spec


def test_mplus_truncates_long_names_uniquely_and_records_the_rename() -> None:
    spec = _long_name_spec()
    original = [item.name for item in spec.all_items()]
    assert max(len(n) for n in original) > X.MPLUS_MAX_NAME_CHARS, original

    renames: Dict[str, str] = {}
    syntax = X.mplus_syntax(spec, "sim.dat", renames=renames)
    print(f"  original names = {original}")
    print(f"  mplus renames  = {renames}")

    mapping, warnings = X.mplus_variable_names(
        original + ["SATISFACTION", "PERFORMANCE"])
    short = [mapping[n] for n in original]
    assert all(len(n) <= X.MPLUS_MAX_NAME_CHARS for n in short), short
    assert len(set(n.upper() for n in short)) == len(short), short
    assert short[0] == "SATISFAC" and short[1] == "SATISFA2", short

    # Every original long name was renamed, and the rename is recorded.
    for name in original:
        assert name in renames, (name, renames)
        assert renames[name] == mapping[name]
    assert len(warnings) == len(renames)
    assert all("8 characters" in w for w in warnings), warnings

    # The syntax uses the short names, and the rename is visible in the file.
    names_block = syntax.split("NAMES ARE", 1)[1].split(";", 1)[0]
    for name in short:
        assert name in names_block, (name, names_block)
    assert "SATISFACTION1" not in names_block
    assert "SATISFACTION1  ->  SATISFAC" in syntax
    assert "renamed" in syntax

    # Arabic warnings on request.
    _, arabic = X.mplus_variable_names(original, lang="ar")
    assert arabic and all(has_arabic(w) for w in arabic), arabic


def test_mplus_sections_and_operators() -> None:
    spec = examples.second_order_hierarchy()
    syntax = X.mplus_syntax(spec, "sq.dat")
    print("  mplus MODEL block:")
    for line in syntax.split("MODEL:", 1)[1].splitlines()[:12]:
        print(f"    {line}")

    for section in ("TITLE:", "DATA:", "VARIABLE:", "ANALYSIS:", "MODEL:",
                    "OUTPUT:"):
        assert section in syntax, section
    assert syntax.index("TITLE:") < syntax.index("DATA:") < \
        syntax.index("VARIABLE:") < syntax.index("ANALYSIS:") < \
        syntax.index("MODEL:") < syntax.index("OUTPUT:")
    assert "FILE IS sq.dat;" in syntax
    assert "USEVARIABLES ARE" in syntax
    assert "CATEGORICAL ARE" in syntax
    assert f"MISSING ARE ALL ({X.MPLUS_MISSING_CODE});" in syntax
    assert "ESTIMATOR = WLSMV;" in syntax          # ordinal items
    assert "SQ BY SQ_D1 SQ_D2 SQ_D3;" in syntax    # second order, BY
    assert "SQ_D1 BY SQ11 SQ12 SQ13 SQ14;" in syntax
    assert "CSAT ON SQ (a);" in syntax             # regression, ON
    assert "LOY ON CSAT (b) SQ (c);" in syntax
    assert "NEW(IND TOTAL);" in syntax
    assert "IND = a*b;" in syntax
    assert "TOTAL = c + a*b;" in syntax
    assert "STDYX;" in syntax and "MODINDICES;" in syntax
    assert max(len(line) for line in syntax.splitlines()) <= 90

    # A continuous model uses MLR, and WITH carries the covariances.
    cfa = X.mplus_syntax(examples.minimal_cfa())
    assert "ESTIMATOR = MLR;" in cfa
    assert "CATEGORICAL ARE" not in cfa
    assert "F1 WITH F2;" in cfa

    # Residual covariances are WITH statements between items.
    messy = X.mplus_syntax(examples.messy_reality())
    assert "ATT3 WITH ATT4;" in messy

    # A latent interaction needs XWITH plus numerical integration.
    moderation = X.mplus_syntax(examples.moderation())
    assert "XWITH" in moderation
    assert "ALGORITHM = INTEGRATION;" in moderation


# --------------------------------------------------------------------------- #
# AMOS notes and the R script
# --------------------------------------------------------------------------- #


def test_amos_notes_are_bilingual_and_name_the_structure() -> None:
    spec = examples.second_order_hierarchy()
    english = X.amos_notes(spec, lang="en")
    arabic = X.amos_notes(spec, lang="ar")
    assert "SQ_D1" in english and "SQ -> CSAT" in english
    assert has_arabic(arabic)
    assert "SQ_D1" in arabic
    assert not has_arabic(english.split("\n", 1)[1].replace(
        spec.title.get("ar"), ""))


def test_rds_script_names_the_csv_and_the_ordinal_items() -> None:
    spec = examples.second_order_hierarchy()
    compiled = compile_model(spec)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.integers(1, 8, size=(10, compiled.n_items)),
                         columns=compiled.item_names)
    ds = FakeDataset(frame, compiled=compiled, seed=20240202)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sim.R")
        X.export_rds_script(ds, path)
        text = open(path, encoding="utf-8").read()
    assert 'read.csv("sim.csv"' in text
    assert 'saveRDS(df, file = "sim.rds")' in text
    assert '"SQ11"' in text and "ordered = TRUE" in text
    assert "20240202" in text


def test_mplus_dat_is_headerless_with_the_missing_code() -> None:
    ds = _mixed_dataset(n=5)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sim.dat")
        X.export_mplus_dat(ds, path, decimal_places=3)
        lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 5, lines
    assert "SAT1" not in lines[0]
    assert str(X.MPLUS_MISSING_CODE) in lines[3], lines[3]
    assert len(lines[0].split()) == 3


# --------------------------------------------------------------------------- #
# export_all
# --------------------------------------------------------------------------- #


def test_export_all_writes_everything_the_spec_asks_for() -> None:
    spec = examples.minimal_cfa()
    spec.export.formats = ["csv", "xlsx", "sav", "rds"]
    spec.export.emit_lavaan_syntax = True
    spec.export.emit_mplus_syntax = True
    spec.export.emit_amos_hint = True
    spec.export.include_latent_scores = True
    compiled = compile_model(spec)
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(12, compiled.n_items)),
                         columns=compiled.item_names)
    scores = pd.DataFrame(rng.normal(size=(12, compiled.n_factors)),
                          columns=compiled.factor_names)
    ds = FakeDataset(frame, compiled=compiled, latent_scores=scores, seed=12345)

    with tempfile.TemporaryDirectory() as folder:
        written = X.export_all(ds, folder, spec, stem="sim")
        names = sorted(os.path.basename(p) for p in written)
        sizes = {os.path.basename(p): os.path.getsize(p) for p in written}
    print(f"  export_all wrote: {names}")
    assert names == sorted([
        "sim.csv", "sim.xlsx", "sim.sav", "sim.R", "sim_latent_scores.csv",
        "sim_lavaan.R", "sim.dat", "sim.inp", "sim_amos.txt"]), names
    assert all(size > 0 for size in sizes.values()), sizes


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def _expect_export_error(call, label: str) -> None:
    try:
        call()
    except ExportError as exc:
        assert has_arabic(exc.message_ar), (label, exc.message_ar)
        assert exc.message_ar.strip(), label
        assert exc.message_en.strip(), label
        print(f"  {label}: {exc.message_ar[:70]}...")
    except OSError as exc:
        raise AssertionError(
            f"{label} leaked a raw OSError: {exc!r}") from exc
    else:
        raise AssertionError(f"{label} did not raise ExportError")


def test_every_writer_raises_export_error_for_an_unwritable_path() -> None:
    spec = _likert_spec()
    frame = pd.DataFrame({name: [1, 2, 3] for name in spec.all_item_names()})
    ds = FakeDataset(frame, seed=777)
    with tempfile.TemporaryDirectory() as folder:
        missing = os.path.join(folder, "no_such_folder", "out")
        _expect_export_error(lambda: X.export_csv(ds, missing + ".csv"), "csv")
        _expect_export_error(lambda: X.export_xlsx(ds, missing + ".xlsx"),
                             "xlsx")
        _expect_export_error(lambda: X.export_sav(ds, missing + ".sav", spec),
                             "sav")
        _expect_export_error(lambda: X.export_rds_script(ds, missing + ".R"),
                             "rds")
        _expect_export_error(lambda: X.export_mplus_dat(ds, missing + ".dat"),
                             "dat")
        _expect_export_error(
            lambda: X.export_all(ds, os.path.join(folder, "nope"), spec),
            "export_all")


def test_a_dataset_without_data_or_with_no_rows_is_refused() -> None:
    class Bare:
        pass

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "x.csv")
        _expect_export_error(lambda: X.export_csv(Bare(), path), "no .data")
        empty = FakeDataset(pd.DataFrame({"a": []}))
        _expect_export_error(lambda: X.export_csv(empty, path), "empty frame")
        _expect_export_error(lambda: X.export_xlsx(empty, path), "empty xlsx")


# --------------------------------------------------------------------------- #
# Script mode (pytest may be absent)
# --------------------------------------------------------------------------- #


def _main() -> int:
    checks = [(name, obj) for name, obj in sorted(globals().items())
              if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""
semsim.io.project
=================
Round-trip a :class:`~semsim.core.spec.ModelSpec` through JSON so researchers
can save, reopen and share a simulation design.

The codec is generic over the dataclasses in :mod:`semsim.core.spec`: it walks
``__dataclass_fields__`` and the annotated types, so adding a field to the spec
needs no change here.  Three properties are guaranteed:

*   **Exactness** -- ``from_dict(to_dict(spec))`` reproduces every value,
    including enum identity and tuple-vs-list distinctions.
*   **Forward tolerance** -- a file written by a newer build loads with its
    unknown keys dropped and recorded in :data:`LAST_LOAD_WARNINGS`, instead of
    raising.
*   **Backward tolerance** -- a file missing a field gets that field's current
    default.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin

from semsim.core import spec as spec_module
from semsim.core.errors import SemSimError
from semsim.core.spec import SPEC_VERSION, ModelSpec

PROJECT_SUFFIX = ".semsim"
FILE_MAGIC = "semsim-project"

#: Warnings produced by the most recent :func:`from_dict` call.  The UI reads
#: this to tell the user that a file was written by a different build.
LAST_LOAD_WARNINGS: List[str] = []


class ProjectFileError(SemSimError):
    """The file is not a readable semsim project."""


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, float):
        # JSON has no NaN/Infinity; keep the file strictly valid.
        if value != value:
            return {"__float__": "nan"}
        if value == float("inf"):
            return {"__float__": "inf"}
        if value == float("-inf"):
            return {"__float__": "-inf"}
        return value
    return value


def to_dict(spec: ModelSpec) -> Dict[str, Any]:
    """Serialise a spec to a JSON-safe dictionary."""
    return {
        "__format__": FILE_MAGIC,
        "spec_version": SPEC_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": _encode(spec),
    }


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def _unwrap_optional(tp: Any) -> Any:
    """``Optional[X]`` / ``X | None`` -> ``X``; leaves other unions alone."""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _resolve(tp: Any) -> Any:
    """Turn a string annotation into the real object (PEP 563 defers them)."""
    if isinstance(tp, str):
        ns = vars(spec_module)
        try:
            return eval(tp, {**vars(typing), **ns})  # noqa: S307 - our own module
        except Exception:
            return Any
    return tp


def _decode(value: Any, tp: Any, path: str) -> Any:
    tp = _unwrap_optional(_resolve(tp))

    if value is None:
        return None

    # Sentinel wrappers written by _encode.
    if isinstance(value, dict) and "__float__" in value:
        return float({"nan": "nan", "inf": "inf", "-inf": "-inf"}[value["__float__"]])
    if isinstance(value, dict) and "__tuple__" in value:
        inner = get_args(tp)
        items = value["__tuple__"]
        if inner and Ellipsis not in inner:
            return tuple(_decode(v, inner[min(i, len(inner) - 1)], f"{path}[{i}]")
                         for i, v in enumerate(items))
        return tuple(items)

    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(value)
        except ValueError:
            LAST_LOAD_WARNINGS.append(
                f"{path}: unknown value {value!r} for {tp.__name__}; using default")
            return list(tp)[0]

    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return _decode_dataclass(value, tp, path)

    origin = get_origin(tp)
    if origin in (list, List) and isinstance(value, list):
        (inner,) = get_args(tp) or (Any,)
        return [_decode(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (tuple, Tuple) and isinstance(value, list):
        inner = get_args(tp)
        if inner and Ellipsis not in inner:
            return tuple(_decode(v, inner[min(i, len(inner) - 1)], f"{path}[{i}]")
                         for i, v in enumerate(value))
        return tuple(value)
    if origin in (dict, Dict) and isinstance(value, dict):
        args = get_args(tp)
        vtp = args[1] if len(args) == 2 else Any
        return {k: _decode(v, vtp, f"{path}.{k}") for k, v in value.items()}

    if tp is float and isinstance(value, (int, float)):
        return float(value)
    if tp is int and isinstance(value, bool):
        return int(value)
    return value


def _decode_dataclass(payload: Dict[str, Any], cls: Any, path: str) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, raw in payload.items():
        field = fields.get(key)
        if field is None:
            LAST_LOAD_WARNINGS.append(
                f"{path}.{key}: not recognised by this version; ignored")
            continue
        kwargs[key] = _decode(raw, field.type, f"{path}.{key}")
    missing = [n for n in fields if n not in kwargs]
    if missing and path:  # defaults cover these; just note the drift
        LAST_LOAD_WARNINGS.append(
            f"{path}: using defaults for {', '.join(sorted(missing))}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ProjectFileError(
            f"Could not rebuild {cls.__name__} at {path or 'root'}: {exc}",
            f"تعذّر إعادة بناء {cls.__name__} في {path or 'الجذر'}: {exc}",
        ) from exc


def from_dict(payload: Dict[str, Any]) -> ModelSpec:
    """Rebuild a spec from :func:`to_dict` output.

    Unknown keys are dropped and recorded in :data:`LAST_LOAD_WARNINGS` rather
    than raising, so a project saved by a newer build still opens.
    """
    LAST_LOAD_WARNINGS.clear()
    if not isinstance(payload, dict):
        raise ProjectFileError(
            "Project file is not a JSON object.",
            "ملف المشروع ليس كائن JSON صالحًا.",
        )
    model = payload.get("model", payload)
    if not isinstance(model, dict):
        raise ProjectFileError(
            "Project file has no 'model' section.",
            "ملف المشروع لا يحتوي على قسم 'model'.",
        )
    file_version = str(payload.get("spec_version", SPEC_VERSION))
    if file_version != SPEC_VERSION:
        LAST_LOAD_WARNINGS.append(
            f"File was written for spec version {file_version}; "
            f"this build uses {SPEC_VERSION}.")
    spec = _decode_dataclass(model, ModelSpec, "")
    spec.spec_version = SPEC_VERSION
    return spec


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def save(spec: ModelSpec, path: "str | os.PathLike[str]") -> str:
    """Write ``spec`` to ``path`` as UTF-8 JSON (Arabic labels stay readable).

    Returns the path actually written, which may differ from the one given:
    the project suffix is appended when it is missing, and a caller that has
    to reopen or list the file would otherwise have to re-derive it.
    """
    target = str(path)
    if not target.endswith(PROJECT_SUFFIX):
        target += PROJECT_SUFFIX
    payload = to_dict(spec)
    tmp = target + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2,
                      allow_nan=False)
        os.replace(tmp, target)
    except OSError as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise ProjectFileError(
            f"Could not write project file: {exc}",
            f"تعذّر حفظ ملف المشروع: {exc}",
        ) from exc
    return target


def load(path: "str | os.PathLike[str]") -> ModelSpec:
    """Read a project file written by :func:`save`."""
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError as exc:
        raise ProjectFileError(
            f"Project file not found: {path}",
            f"ملف المشروع غير موجود: {path}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProjectFileError(
            f"Project file is not valid JSON (line {exc.lineno}): {exc.msg}",
            f"ملف المشروع ليس JSON صالحًا (السطر {exc.lineno}): {exc.msg}",
        ) from exc
    except OSError as exc:
        raise ProjectFileError(
            f"Could not read project file: {exc}",
            f"تعذّر قراءة ملف المشروع: {exc}",
        ) from exc
    return from_dict(payload)


def describe_project(path: "str | os.PathLike[str]") -> Dict[str, Any]:
    """Cheap header read for a "recent projects" list.

    Returns title, counts and timestamps without rebuilding the whole spec.
    """
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:  # a broken file must not break the recents list
        return {"path": str(path), "readable": False, "error": str(exc)}
    model = payload.get("model", {})
    title = model.get("title", {}) or {}
    constructs = model.get("constructs", []) or []
    n_items = 0
    for c in constructs:
        n_items += len(c.get("items", []) or [])
        for d in c.get("dimensions", []) or []:
            n_items += len(d.get("items", []) or [])
    return {
        "path": str(path),
        "readable": True,
        "title_ar": title.get("ar", ""),
        "title_en": title.get("en", ""),
        "spec_version": payload.get("spec_version", ""),
        "saved_at": payload.get("saved_at", ""),
        "n_constructs": len(constructs),
        "n_items": n_items,
        "n_paths": len(model.get("paths", []) or []),
    }


__all__ = [
    "PROJECT_SUFFIX", "FILE_MAGIC", "LAST_LOAD_WARNINGS", "ProjectFileError",
    "to_dict", "from_dict", "save", "load", "describe_project",
]

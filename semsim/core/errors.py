"""User-facing error hierarchy with bilingual messages."""

from __future__ import annotations

from typing import Any, Dict


class SemSimError(Exception):
    """Base class for every error the user is meant to read.

    Carries both interface languages so the UI can render whichever is active
    without a second lookup table.
    """

    def __init__(self, message_en: str, message_ar: str = "", **context: Any) -> None:
        self._message_en = message_en
        self._message_ar = message_ar or message_en
        self.context: Dict[str, Any] = context
        super().__init__(message_en)

    @property
    def message_en(self) -> str:
        return self._message_en

    @property
    def message_ar(self) -> str:
        return self._message_ar

    def message(self, lang: str = "en") -> str:
        return self._message_ar if lang == "ar" else self._message_en

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._message_en


class SpecificationError(SemSimError):
    """The model as described cannot be compiled."""


class IdentificationError(SemSimError):
    """Not enough information in the data to estimate the model."""


class HeywoodCaseError(SemSimError):
    """A variance implied by the specification is negative or zero."""


class ConvergenceError(SemSimError):
    """An iterative algorithm failed to settle."""


class InfeasibleTargetError(SemSimError):
    """The requested distribution / fit / significance target is unreachable."""


class EstimationError(SemSimError):
    """Model fitting failed."""


class ExportError(SemSimError):
    """Writing an output file failed."""


__all__ = [
    "SemSimError", "SpecificationError", "IdentificationError",
    "HeywoodCaseError", "ConvergenceError", "InfeasibleTargetError",
    "EstimationError", "ExportError",
]

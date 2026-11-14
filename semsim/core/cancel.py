"""
semsim.core.cancel
==================
Cooperative cancellation for long runs.

A run is stopped by setting an event that the heavy loops poll: the
estimator's objective function (evaluated every few milliseconds), each draw
of the calibration loop and each stage of data generation.  The token is
per thread, so the GUI's worker thread can be stopped while code called from
anywhere else -- tests, scripts -- never sees one and pays nothing.

:class:`Cancelled` derives from :class:`BaseException`, not
:class:`Exception`, on purpose.  The pipeline is full of ``except Exception``
blocks that turn a failed step into a warning and carry on; a stop request
caught by one of those would be reported as a soft error and the run would
continue, which is exactly the "Stop does nothing" bug this module exists to
prevent.  Like :class:`KeyboardInterrupt`, it passes through all of them.
"""

from __future__ import annotations

import threading
from typing import Optional

_local = threading.local()


class Cancelled(BaseException):
    """The user asked the current run to stop."""

    message_en = "Cancelled by the user."
    message_ar = "أُلغيت العملية."


def install(token: Optional[threading.Event]) -> None:
    """Make ``token`` the stop flag for the calling thread (``None`` clears)."""
    _local.token = token


def check() -> None:
    """Raise :class:`Cancelled` if the calling thread has been asked to stop."""
    token = getattr(_local, "token", None)
    if token is not None and token.is_set():
        raise Cancelled()


__all__ = ["Cancelled", "install", "check"]

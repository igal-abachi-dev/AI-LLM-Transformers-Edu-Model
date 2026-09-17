"""Catchable termination signals -> an immediate, resumable emergency checkpoint.

Beginner's map of this file
----------------------------
A multi-day training run is vulnerable *between* checkpoints: if something kills
the process mid-interval, every update since the last save is lost. This project
has already been hit by this for real -- an early 1B-token pre-work run lost
about three hours (~21,000 updates) to an external kill that landed between two
scheduled checkpoints (see ``tasks/backlog.md``'s MF-070 entry).

The fix here does not try to survive every possible kill -- it can't, and no
code anywhere can. What it *can* do is catch a **requested, graceful** stop
(Ctrl+C in the same console, or a real SIGTERM on a POSIX system) and force an
emergency checkpoint before the process actually exits, so a run stopped this
way loses at most the current update, not a whole interval's worth.

A hard force-kill (``Stop-Process -Force``, ``taskkill /F``, Task Manager's
"End task", or a real SIGKILL) can never be caught by any process, on any
platform -- that is what "force" means at the operating-system level. This
module narrows the real, previously-unprotected gap (a *graceful* termination
request landing mid-interval); it does not close it entirely, and never can.

Windows-specific caveat, worth knowing before relying on this: Python's own
``signal.SIGTERM`` handler registration is accepted on Windows, but
``os.kill(pid, signal.SIGTERM)`` and a forceful process kill both actually call
``TerminateProcess`` under the hood -- which gives a process no chance to run
any handler at all, regardless of which signal was "requested". The two
signals that are genuinely, reliably catchable for a Windows console process
are ``SIGINT`` (Ctrl+C in the same console) and ``SIGBREAK`` (Ctrl+Break) --
both are installed here alongside ``SIGTERM`` for portability, but ``SIGTERM``
itself is unlikely to ever actually fire on this project's own real Windows
training hardware.
"""

from __future__ import annotations

import signal
from types import FrameType


class GracefulTerminationRequested(Exception):
    """Raised after an emergency checkpoint completes, to unwind the training loop."""


class TerminationRequestTracker:
    """Sets a flag from a signal handler; the training loop checks it at a safe point.

    Signal handlers in CPython run on the main thread, between bytecode
    instructions -- never truly asynchronously the way a C signal handler
    does -- so setting a plain attribute here is safe. The actual checkpoint
    save happens later, from ordinary code in the training loop's own
    per-update callback, never from inside the handler itself: that is the one
    safe point where the model, optimizer, and data cursor are all in a
    consistent, saveable state.
    """

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._installed: dict[int, object] = {}

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        del frame
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def install(self) -> None:
        """Register handlers for every real, catchable termination signal.

        Raises ``ValueError`` unchanged if called off the main thread (Python's
        own ``signal.signal`` restriction) -- callers should treat that as "no
        safety net available here", not a fatal error.
        """

        if self._installed:
            raise RuntimeError("TerminationRequestTracker is already installed")
        signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGBREAK"):  # Windows-only: Ctrl+Break.
            signals.append(signal.SIGBREAK)
        installed: dict[int, object] = {}
        try:
            for sig in signals:
                installed[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
        except ValueError:
            for sig, previous in installed.items():
                signal.signal(sig, previous)
            raise
        self._installed = installed

    def restore(self) -> None:
        for sig, previous in self._installed.items():
            signal.signal(sig, previous)
        self._installed.clear()

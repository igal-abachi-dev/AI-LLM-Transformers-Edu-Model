"""Real signal-delivery tests for the emergency-checkpoint safety net (MF-146).

Uses `signal.raise_signal` (Python 3.8+), not `os.kill`, to fire signals at the
current process -- deliberately, since `os.kill(os.getpid(), signal.SIGTERM)`
on Windows calls `TerminateProcess` directly for any signal value that isn't
`CTRL_C_EVENT`/`CTRL_BREAK_EVENT`, bypassing any registered Python handler
entirely and killing the test process outright. `raise_signal` goes through
the C runtime's own `raise()`, which real registered handlers *do* intercept
on every platform, including Windows -- the correct, safe way to test this.
"""

from __future__ import annotations

import signal
import threading

import pytest

from minifrontier.termination import GracefulTerminationRequested, TerminationRequestTracker


@pytest.fixture
def tracker():
    instance = TerminationRequestTracker()
    yield instance
    # Unconditional: restore() on a never-installed (or already-restored)
    # tracker is a safe no-op, and this guarantees no handler leaks into a
    # later test in the same pytest process regardless of what this test did.
    instance.restore()


def test_starts_unrequested(tracker: TerminationRequestTracker) -> None:
    assert tracker.requested is False
    assert tracker.signal_name is None


def test_install_catches_a_real_sigterm(tracker: TerminationRequestTracker) -> None:
    tracker.install()
    signal.raise_signal(signal.SIGTERM)
    assert tracker.requested is True
    assert tracker.signal_name == "SIGTERM"


def test_install_catches_a_real_sigint(tracker: TerminationRequestTracker) -> None:
    tracker.install()
    signal.raise_signal(signal.SIGINT)
    assert tracker.requested is True
    assert tracker.signal_name == "SIGINT"


@pytest.mark.skipif(not hasattr(signal, "SIGBREAK"), reason="SIGBREAK is Windows-only")
def test_install_catches_a_real_sigbreak_on_windows(tracker: TerminationRequestTracker) -> None:
    tracker.install()
    signal.raise_signal(signal.SIGBREAK)
    assert tracker.requested is True
    assert tracker.signal_name == "SIGBREAK"


def test_restore_reverts_to_the_previous_handler(tracker: TerminationRequestTracker) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    tracker.install()
    assert signal.getsignal(signal.SIGTERM) is not previous
    tracker.restore()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_install_twice_without_restore_raises(tracker: TerminationRequestTracker) -> None:
    tracker.install()
    with pytest.raises(RuntimeError, match="already installed"):
        tracker.install()


def test_install_off_the_main_thread_raises_and_leaves_the_tracker_reusable(
    tracker: TerminationRequestTracker,
) -> None:
    """`signal.signal` itself refuses to run off the main thread -- confirms
    that failure propagates cleanly, and that install()'s own rollback leaves
    nothing partially registered behind (a fresh install from the main thread
    right afterward must succeed, not immediately raise "already installed")."""

    errors: list[BaseException] = []

    def attempt() -> None:
        try:
            tracker.install()
        except BaseException as error:  # captured for the main thread to inspect
            errors.append(error)

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    tracker.install()  # must succeed cleanly -- no partial state left behind


def test_graceful_termination_requested_is_a_real_exception() -> None:
    with pytest.raises(GracefulTerminationRequested, match="checkpoint"):
        raise GracefulTerminationRequested("emergency checkpoint saved")

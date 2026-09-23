from __future__ import annotations

import math

import pytest

from minifrontier.stability import (
    MIN_OBSERVATIONS_BEFORE_DETECTION,
    StabilityMonitor,
)


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="window_size"):
        StabilityMonitor(window_size=0)
    with pytest.raises(ValueError, match="sigma_factor"):
        StabilityMonitor(sigma_factor=0.0)


def test_never_flags_before_minimum_history_is_reached() -> None:
    monitor = StabilityMonitor(window_size=128, sigma_factor=3.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION - 1):
        observation = monitor.observe(loss=1.0, grad_norm=1.0)
        assert not observation.is_anomalous
        assert observation.loss_z_score is None
        assert observation.grad_norm_z_score is None
    # A wild value still isn't flagged -- there still isn't enough history.
    observation = monitor.observe(loss=1_000_000.0, grad_norm=1.0)
    assert not observation.is_anomalous


def test_flags_a_real_loss_spike_once_history_exists() -> None:
    monitor = StabilityMonitor(window_size=128, sigma_factor=3.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION + 20):
        assert not monitor.observe(loss=1.0, grad_norm=1.0).is_anomalous
    spike = monitor.observe(loss=1_000_000.0, grad_norm=1.0)
    assert spike.is_anomalous
    assert spike.loss_z_score is not None and spike.loss_z_score > 3.0
    assert spike.grad_norm_z_score == 0.0  # identical history, identical value


def test_flags_a_real_grad_norm_spike_independently_of_loss() -> None:
    monitor = StabilityMonitor(window_size=128, sigma_factor=3.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION + 20):
        assert not monitor.observe(loss=2.0, grad_norm=1.0).is_anomalous
    spike = monitor.observe(loss=2.0, grad_norm=1_000_000.0)
    assert spike.is_anomalous
    assert spike.loss_z_score == 0.0
    assert spike.grad_norm_z_score is not None and spike.grad_norm_z_score > 3.0


def test_does_not_tolerate_realistic_noise_as_a_false_positive() -> None:
    """A believable, mildly noisy loss curve should never trip a sigma_factor=6
    default -- the whole point is catching real, rare spikes, not ordinary
    step-to-step variance."""

    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    # A small, fixed pseudo-random-looking sequence, no external RNG needed.
    noisy_losses = [3.0 + 0.05 * ((i * 37) % 11 - 5) for i in range(200)]
    flags = [monitor.observe(loss=value, grad_norm=1.0).is_anomalous for value in noisy_losses]
    assert not any(flags)


def test_an_anomalous_observation_is_not_folded_into_future_history() -> None:
    """Confirms the real design decision stated in the module docstring: a
    flagged value must not corrupt what "normal" means for the next check."""

    monitor = StabilityMonitor(window_size=128, sigma_factor=3.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION + 5):
        monitor.observe(loss=1.0, grad_norm=1.0)
    spike = monitor.observe(loss=1_000_000.0, grad_norm=1.0)
    assert spike.is_anomalous
    # If the spike had been folded in, the mean/std would have shifted enough
    # that this next, perfectly ordinary value might not score as extreme.
    still_ordinary = monitor.observe(loss=1.0, grad_norm=1.0)
    assert not still_ordinary.is_anomalous


def test_window_size_bounds_how_much_history_is_kept() -> None:
    monitor = StabilityMonitor(window_size=10, sigma_factor=3.0)
    for _ in range(50):
        monitor.observe(loss=1.0, grad_norm=1.0)
    assert len(monitor._loss_history) == 10
    assert len(monitor._grad_norm_history) == 10


def test_nonfinite_loss_is_flagged_immediately_not_silently_accepted() -> None:
    """Before the fix, Python's `nan > sigma_factor` is always False, so a NaN
    loss was silently judged NOT anomalous on the call that introduced it."""

    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION):
        monitor.observe(loss=1.0, grad_norm=1.0)
    observation = monitor.observe(loss=float("nan"), grad_norm=1.0)
    assert observation.is_anomalous
    assert observation.loss_z_score == float("inf")


def test_nonfinite_grad_norm_is_flagged_immediately() -> None:
    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION):
        monitor.observe(loss=1.0, grad_norm=1.0)
    observation = monitor.observe(loss=1.0, grad_norm=float("inf"))
    assert observation.is_anomalous
    assert observation.grad_norm_z_score == float("inf")


def test_nonfinite_value_never_crashes_the_following_call() -> None:
    """Real reproduction of the actual bug: a NaN loss used to be silently
    appended to history (see the test above), and the *next* call would then
    crash inside statistics.pstdev with AttributeError -- verified by direct
    reproduction against Python 3.12's real `statistics` module before this fix
    (pstdev raises `AttributeError: 'float' object has no attribute
    'numerator'` once NaN is anywhere in its input). This must not crash before
    OR after MIN_OBSERVATIONS_BEFORE_DETECTION worth of history exists."""

    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION - 1):
        monitor.observe(loss=1.0, grad_norm=1.0)
    monitor.observe(loss=float("nan"), grad_norm=1.0)
    # This call used to crash with AttributeError once enough history existed.
    following = monitor.observe(loss=1.05, grad_norm=1.05)
    assert isinstance(following.is_anomalous, bool)


def test_nonfinite_value_is_not_folded_into_history() -> None:
    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    for _ in range(MIN_OBSERVATIONS_BEFORE_DETECTION + 5):
        monitor.observe(loss=1.0, grad_norm=1.0)
    monitor.observe(loss=float("nan"), grad_norm=float("inf"))
    assert float("nan") not in monitor._loss_history
    assert all(math.isfinite(value) for value in monitor._loss_history)
    assert all(math.isfinite(value) for value in monitor._grad_norm_history)

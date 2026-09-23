from __future__ import annotations

import math
import random

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


def test_a_spike_is_clipped_before_folding_so_it_cannot_inflate_normal() -> None:
    """Confirms the real design: a flagged value is folded into history, but
    clipped to the anomaly ceiling -- so it cannot corrupt what "normal" means
    for the next check the way folding it in at full value would."""

    monitor = StabilityMonitor(window_size=128, sigma_factor=3.0)
    rng_losses = [
        3.0 + 0.01 * ((i * 7) % 5 - 2) for i in range(MIN_OBSERVATIONS_BEFORE_DETECTION + 5)
    ]
    for value in rng_losses:
        monitor.observe(loss=value, grad_norm=1.0)
    spike = monitor.observe(loss=1_000_000.0, grad_norm=1.0)
    assert spike.is_anomalous
    # If the spike had been folded in at full value, the mean/std would have
    # shifted enough that this next, perfectly ordinary value might not score
    # as extreme -- or the history itself would visibly contain 1,000,000.
    still_ordinary = monitor.observe(loss=3.0, grad_norm=1.0)
    assert not still_ordinary.is_anomalous
    assert 1_000_000.0 not in monitor._loss_history


def test_stability_monitor_absorbs_a_level_shift_instead_of_locking_out() -> None:
    """Real, reproduced failure mode: under the old rule (never fold a flagged
    value in at all), a genuine, lasting level shift -- e.g. a decay-phase
    data-mixture switch raising the per-update loss -- flagged 300/300
    subsequent updates, forever, since the history could never catch up."""

    rng = random.Random(0)
    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    for _ in range(300):
        monitor.observe(3.0 + rng.gauss(0, 0.03), 0.3 + rng.gauss(0, 0.01))
    flags = [
        monitor.observe(3.4 + rng.gauss(0, 0.03), 0.3 + rng.gauss(0, 0.01)).is_anomalous
        for _ in range(300)
    ]
    assert flags[0]
    assert not any(flags[150:])  # used to be 300/300 flagged, forever


def test_stability_monitor_never_flags_an_improvement() -> None:
    """Detection is one-sided: a sudden drop in loss/grad_norm is good news,
    never an instability."""

    rng = random.Random(1)
    monitor = StabilityMonitor(window_size=128, sigma_factor=6.0)
    for _ in range(200):
        monitor.observe(3.0 + rng.gauss(0, 0.02), 0.3 + rng.gauss(0, 0.01))
    assert not monitor.observe(2.5, 0.3).is_anomalous
    assert not monitor.observe(3.0, 0.1).is_anomalous
    assert monitor.observe(4.0, 0.3).is_anomalous


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

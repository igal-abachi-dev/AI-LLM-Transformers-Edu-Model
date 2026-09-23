"""Rolling-window anomaly detection for a single training run's loss/grad norm (MF-148).

Beginner's map of this file
----------------------------
Gradient clipping (see ``training.py``) softens a huge gradient -- it shrinks the
step, but still takes it. This is a different, complementary idea, borrowed from
AI2's own OLMo-core training library (its ``SkipStepOptimizer`` and
``StabilityMonitorCallback``): keep a short rolling memory of "what a normal
loss/gradient-norm value has looked like recently," and flag -- optionally, refuse
outright -- a step whose value is a real statistical outlier against that memory,
not merely large in absolute terms.

Two knobs decide what happens once something is flagged, deliberately split apart
rather than bundled into one behavior:

- Every real training run gets the *diagnostic* for free -- ``StabilityMonitor``
  always tracks history and always counts how many updates were flagged
  (``TrainingState.anomalous_updates``), regardless of whether anything is done
  about it. This alone answers "how often would this have fired?" on a real run,
  cheaply, before anyone decides whether acting on it is worth the complexity --
  exactly OLMo-core's own two-tier split (a free observer callback, a separate,
  optional, stronger optimizer wrapper).
- ``TrainingConfig.skip_anomalous_steps`` (off by default) is the second tier:
  when a step is flagged, its gradients are discarded and the optimizer step is
  skipped entirely -- not shrunk, refused -- the same way `train_updates` already
  handles a genuinely non-finite gradient (see ``training.py``).

A flagged value is never folded into its own rolling history -- the monitor's
whole job is remembering what "normal" looks like, and a value it just judged
abnormal is not evidence of that.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

# Below this many prior (non-anomalous) observations, a rolling standard
# deviation is too noisy to trust -- a run's first few updates are exactly
# where loss/grad-norm are naturally still settling down from initialization,
# not evidence of a real anomaly. No observation is ever flagged before the
# window holds at least this many points.
MIN_OBSERVATIONS_BEFORE_DETECTION = 8


@dataclass(frozen=True, slots=True)
class StabilityObservation:
    """What one `StabilityMonitor.observe` call decided, and why."""

    is_anomalous: bool
    loss_z_score: float | None
    grad_norm_z_score: float | None


class StabilityMonitor:
    """Rolling z-score outlier detector over a run's own recent loss/grad-norm history.

    Two independent windows (loss, gradient norm) rather than one combined
    check, since either metric spiking on its own is real, useful signal --
    a loss spike with an ordinary gradient norm is still worth flagging, and
    vice versa.
    """

    def __init__(self, *, window_size: int = 128, sigma_factor: float = 6.0) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if sigma_factor <= 0:
            raise ValueError("sigma_factor must be positive")
        self.window_size = window_size
        self.sigma_factor = sigma_factor
        self._loss_history: deque[float] = deque(maxlen=window_size)
        self._grad_norm_history: deque[float] = deque(maxlen=window_size)

    @staticmethod
    def _z_score(history: deque[float], value: float) -> float | None:
        if len(history) < MIN_OBSERVATIONS_BEFORE_DETECTION:
            return None
        mean = statistics.fmean(history)
        deviation = statistics.pstdev(history, mean)
        if deviation == 0.0:
            # Every recent value was identical -- any real difference is an
            # infinite z-score in principle; report a large-but-finite number
            # so callers can still compare it against sigma_factor sensibly.
            return 0.0 if value == mean else float("inf")
        return abs(value - mean) / deviation

    def observe(self, loss: float, grad_norm: float) -> StabilityObservation:
        """Score one update's (loss, grad_norm) against this monitor's own history.

        Call exactly once per optimizer update actually attempted, with the
        real values that update produced -- not, e.g., a running average.

        A non-finite ``loss``/``grad_norm`` (real and routine under FP16+GradScaler,
        see ``training.py``) is flagged immediately, before it ever reaches
        ``statistics.pstdev`` -- that function raises ``AttributeError`` on a NaN
        input (CPython's `statistics` module tries an internal exact-ratio
        conversion that NaN has no `.numerator` for), and without this check the
        crash would not even happen on *this* call: Python's `nan > sigma_factor`
        is always `False`, so the broken value would be silently judged "not
        anomalous," folded into history by the existing code below, and only
        crash the *next* time `observe` runs and tries to compute statistics over
        a history that now contains it. Neither value is folded into history
        here, matching the same "a flagged value is never folded into future
        history" rule every other anomaly already follows.
        """

        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            return StabilityObservation(
                is_anomalous=True,
                loss_z_score=None if math.isfinite(loss) else float("inf"),
                grad_norm_z_score=None if math.isfinite(grad_norm) else float("inf"),
            )
        loss_z = self._z_score(self._loss_history, loss)
        grad_norm_z = self._z_score(self._grad_norm_history, grad_norm)
        is_anomalous = (loss_z is not None and loss_z > self.sigma_factor) or (
            grad_norm_z is not None and grad_norm_z > self.sigma_factor
        )
        if not is_anomalous:
            self._loss_history.append(loss)
            self._grad_norm_history.append(grad_norm)
        return StabilityObservation(is_anomalous, loss_z, grad_norm_z)

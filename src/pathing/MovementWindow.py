"""Bounded time window for distinguishing movement from a stationary player."""

from collections import deque
import math
from numbers import Real


_SAMPLE_INTERVAL = 0.5
_MAX_SAMPLE_GAP = 2.0


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class MovementWindow:
    """Track verified positions without inferring progress from target distance.

    Observations closer than half a second are coalesced. A gap over two seconds
    starts a fresh window, since missing observations are not evidence of staying
    still. ``displacement`` is the diagonal of the retained positions' bounding
    box, so a loop that returns near its first point still counts as movement.
    """

    def __init__(self, window=8.0, min_displacement=3.0):
        self.window = _finite_number(window, "window")
        self.min_displacement = _finite_number(min_displacement, "min_displacement")
        if self.window <= 0:
            raise ValueError("window must be greater than zero")
        if self.min_displacement < 0:
            raise ValueError("min_displacement must not be negative")
        self._max_samples = math.ceil(self.window / _SAMPLE_INTERVAL) + 3
        self.reset()

    def reset(self):
        self._samples = deque()
        self._last_time = None

    @property
    def displacement(self):
        if len(self._samples) < 2:
            return 0.0
        xs = [point[0] for _, point in self._samples]
        ys = [point[1] for _, point in self._samples]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def observe(self, xy, now):
        """Add a valid ``(x, y)`` position at monotonic time ``now``.

        Raises ``ValueError`` for malformed/non-finite coordinates or time, or
        for time moving backwards. Invalid observations do not alter the window.
        """
        now = _finite_number(now, "now")
        if not isinstance(xy, (tuple, list)) or len(xy) != 2:
            raise ValueError("xy must be an (x, y) pair")
        point = (_finite_number(xy[0], "x"), _finite_number(xy[1], "y"))
        if self._last_time is not None and now < self._last_time:
            raise ValueError("now must not move backwards")

        if self._samples and now - self._samples[-1][0] > _MAX_SAMPLE_GAP:
            self._samples.clear()
        self._last_time = now

        if self._samples and now - self._samples[-1][0] < _SAMPLE_INTERVAL:
            return False

        self._samples.append((now, point))
        cutoff = now - self.window
        # Keep one sample just before the window boundary to preserve its span.
        while len(self._samples) > 1 and self._samples[1][0] < cutoff:
            self._samples.popleft()
        while len(self._samples) > self._max_samples:
            self._samples.popleft()
        return True

    def is_stuck(self, now):
        """Return true only after a complete, sufficiently sampled still window."""
        now = _finite_number(now, "now")
        if self._last_time is not None and now < self._last_time:
            raise ValueError("now must not move backwards")
        if self._samples and now - self._samples[-1][0] > _MAX_SAMPLE_GAP:
            self.reset()
            self._last_time = now
            return False
        self._last_time = now
        if len(self._samples) < 3:
            return False
        return (now - self._samples[0][0] >= self.window
                and self.displacement <= self.min_displacement)

"""Debounce airborne observations and request one glide-opening jump per fall."""

import math


class FallGuard:
    """Track a fall using the existing airborne HUD signal.

    ``active`` becomes true on the first airborne observation, so callers can
    immediately pause ground movement while the signal is being confirmed. A
    jump is requested only after consecutive airborne frames span ``air_delay``.
    The request is latched until consecutive ground frames span ``ground_delay``.
    Unknown observations break debounce runs but never count as landing.
    """

    def __init__(self, air_delay=0.12, ground_delay=0.2, max_gap=0.6):
        self.air_delay = air_delay
        self.ground_delay = ground_delay
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        """Clear all state; call only when the character is known to be grounded."""
        self.active = False
        self.gliding = False
        self.airborne_confirmed = False
        self._last_at = None
        self._air_frames = 0
        self._air_since = None
        self._ground_frames = 0
        self._ground_since = None

    @property
    def airborne_or_unknown(self):
        """Whether ground-only route logic must remain paused."""
        return self.active

    def observe(self, airborne, now):
        """Return true exactly once when confirmed airborne and glide needs opening."""
        if airborne is not None and type(airborne) is not bool:
            raise TypeError('airborne must be bool or None')
        if not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('now must be a finite monotonic timestamp')

        now = float(now)
        if self._last_at is not None and (now <= self._last_at or now - self._last_at > self.max_gap):
            # A gap breaks "consecutive" evidence, but never releases a glide latch.
            self._clear_runs()
        self._last_at = now

        if airborne is None:
            self._clear_runs()
            return False

        if airborne:
            self.active = True
            self._ground_frames = 0
            self._ground_since = None
            if self._air_frames == 0:
                self._air_since = now
            self._air_frames += 1
            if (not self.airborne_confirmed and self._air_frames >= 2
                    and now - self._air_since >= self.air_delay):
                self.airborne_confirmed = True
                if not self.gliding:
                    self.gliding = True
                    return True
            return False

        # Ground is only actionable after a complete stable run. It also breaks
        # any unconfirmed airborne run, without treating one flicker as landing.
        self._air_frames = 0
        self._air_since = None
        if not self.active:
            return False
        if self._ground_frames == 0:
            self._ground_since = now
        self._ground_frames += 1
        if self._ground_frames >= 2 and now - self._ground_since >= self.ground_delay:
            self.active = False
            self.gliding = False
            self.airborne_confirmed = False
            self._ground_frames = 0
            self._ground_since = None
        return False

    def _clear_runs(self):
        self._air_frames = 0
        self._air_since = None
        self._ground_frames = 0
        self._ground_since = None

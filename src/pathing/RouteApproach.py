"""Slow the final approach using observed map motion, without skipping route points."""

import math


class RouteApproach:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_at = None
        self.speed = None

    def observe(self, xy, now):
        if self.previous is not None:
            elapsed = now - self.previous_at
            if 0.02 <= elapsed <= 2:
                observed = math.dist(xy, self.previous) / elapsed
                # Pauses between short pulses lower the average speed. Retain the
                # fastest measured rate until the next visual/control interruption.
                self.speed = max(self.speed or 0, observed)
            elif elapsed > 2:
                self.speed = None
        self.previous = tuple(xy)
        self.previous_at = now

    def pulse(self, distance, tolerance, interval):
        if self.speed is None or self.speed <= 0:
            return None
        if distance > max(tolerance * 3, self.speed * interval * 1.5):
            return None
        # Release input BEFORE the next expensive screenshot/match, so vision
        # latency cannot keep carrying the player through a narrow arrival zone.
        remaining = max(0, distance - tolerance * .5)
        return min(interval, .12, max(.005, remaining * .7 / self.speed))

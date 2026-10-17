from dataclasses import dataclass, field


# Clockwise angles relative to the camera; diagonal input preserves forward motion.
DIRECTIONS = (('w',), ('w', 'd'), ('d',), ('s', 'd'),
              ('s',), ('s', 'a'), ('a',), ('w', 'a'))


def angle_error(target, current):
    return (target - current + 180) % 360 - 180


@dataclass
class RouteSteering:
    camera_heading: float
    keys: tuple[str, ...] = field(default_factory=tuple)
    changed_at: float = 0

    def update(self, target, player_heading, now):
        if self.keys and now - self.changed_at >= 0.25:
            # The arrow faces the held movement direction, not necessarily the camera.
            relative = DIRECTIONS.index(self.keys) * 45
            observed_camera = (player_heading - relative) % 360
            correction = angle_error(observed_camera, self.camera_heading)
            self.camera_heading = (self.camera_heading + max(-20, min(20, correction)) * 0.6) % 360
        relative_target = (target - self.camera_heading) % 360
        if self.keys and abs(angle_error(relative_target, DIRECTIONS.index(self.keys) * 45)) <= 30:
            return self.keys  # Hysteresis avoids flickering between adjacent directions.
        keys = DIRECTIONS[int((relative_target + 22.5) // 45) % 8]
        if keys != self.keys:
            self.changed_at = now
            self.keys = keys
        return keys

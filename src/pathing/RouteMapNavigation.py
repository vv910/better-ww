"""Bounded visual map bookmarks. No game/world coordinates or blind teleport clicks."""

import math
import time

import cv2
import numpy as np
from ok import Box

from src.pathing.EchoRoute import read_map_image
from src.pathing.MapLocalizer import MapLocalizer


_MAP_STABLE_TIMEOUT = 3.0
_MAP_STABLE_POLL = 0.08
_MAP_STABLE_PAIRS = 2
_MAP_SIGNATURE_WIDTH = 160
_MAP_CHANGE_THRESHOLD = 8
_MAP_CHANGE_RATIO = 0.015
_MAP_MEAN_CHANGE = 1.5


def map_region(frame):
    height, width = frame.shape[:2]
    return (round(width * .28), round(height * .12), round(width * .87), round(height * .90))


def map_signature(frame):
    """Downsample only the world-map panel; reject blank/loading captures."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
        return None
    height, width = frame.shape[:2]
    left, top, right, bottom = map_region(frame)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    roi = frame[top:bottom, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    signature_height = max(1, round(gray.shape[0] * _MAP_SIGNATURE_WIDTH / gray.shape[1]))
    signature = cv2.resize(gray, (_MAP_SIGNATURE_WIDTH, signature_height),
                           interpolation=cv2.INTER_AREA)
    # A settled capture still needs a terrain match before it can locate anything.
    if (int(signature.max()) < 12 or np.count_nonzero(signature > 8) < signature.size * .05
            or float(signature.std()) < 1.5 or int(signature.max()) - int(signature.min()) < 8):
        return None
    return signature


def same_map_signature(first, second):
    if first.shape != second.shape:
        return False
    difference = cv2.absdiff(first, second)
    changed = np.count_nonzero(difference > _MAP_CHANGE_THRESHOLD) / difference.size
    return (changed <= _MAP_CHANGE_RATIO
            and float(difference.mean()) <= _MAP_MEAN_CHANGE)


def wait_for_map_stable(task, deadline):
    """Read fresh frames until two quiet pairs, within 3s and the caller's budget."""
    stable_deadline = min(deadline, time.monotonic() + _MAP_STABLE_TIMEOUT)
    previous = None
    stable_pairs = 0
    while time.monotonic() < stable_deadline:
        task.next_frame()
        current = map_signature(getattr(task, 'frame', None))
        if current is None:
            previous = None
            stable_pairs = 0
        elif previous is not None and same_map_signature(previous, current):
            stable_pairs += 1
            if stable_pairs >= _MAP_STABLE_PAIRS:
                return True
        else:
            stable_pairs = 0
        previous = current
        task.sleep(_MAP_STABLE_POLL)
    return False


def map_point_tolerance(frame):
    height = frame.shape[0] if isinstance(frame, np.ndarray) and frame.ndim >= 2 else 1080
    return max(4.0, height * .006)


def capture_map_views(task, frame, point, region, max_views=6):
    """Zoom out gradually and carry the beacon only across verified adjacent views."""
    views = []
    reference = MapLocalizer(frame, region)
    reference_point = point
    previous = frame
    for _ in range(max_views):
        left, top, right, bottom = region
        task.scroll(round(reference_point[0]), round(reference_point[1]), -2)
        task.sleep(.35)
        task.next_frame()
        current = task.frame.copy()
        if current.shape != frame.shape:
            task.screenshot(f'3c-map-overview-capture-size-changed-{len(views) + 1}', frame=current)
            raise RuntimeError('Capture size changed while recording map views')
        # At the zoom limit, do not save a stack of identical views.
        if abs(current[top:bottom, left:right].astype('float32')
               - previous[top:bottom, left:right].astype('float32')).mean() < 1:
            break
        projected = reference.project_map_point(current, reference_point, region)
        if projected is None:
            task.screenshot(f'3c-map-overview-capture-mismatch-{len(views) + 1}', frame=current)
            task.log_warning('Map overview capture stopped at an unverified zoom / 概览缩放未通过配准，停止继续缩小')
            break
        reference_point = projected.xy
        views.append((current, reference_point, region))
        reference = MapLocalizer(current, region)
        previous = current
    return views


def spiral_steps(radius=2):
    """Adjacent grid cells; callers drag one bounded step between observations."""
    x = y = 0
    yield x, y
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    distance = 1
    while distance <= radius * 2:
        for index, (dx, dy) in enumerate(directions):
            for _ in range(distance):
                x, y = x + dx, y + dy
                if abs(x) <= radius and abs(y) <= radius:
                    yield x, y
            if index % 2 == 1:
                distance += 1
    # The last edge is outside the two-segment loop above.
    while x < radius:
        x += 1
        if abs(x) <= radius and abs(y) <= radius:
            yield x, y


class RouteMapNavigator:
    def __init__(self, task, route):
        self.task = task
        self.references = []
        seen = set()

        def add_reference(image_path, region, point):
            key = (str(image_path.resolve()), float(point.x), float(point.y), tuple(region))
            if key in seen:
                return
            seen.add(key)
            self.references.append((MapLocalizer(read_map_image(image_path), region), point))

        if route.teleport.waypoint is not None:
            add_reference(route.map_image, route.map_region, route.teleport.waypoint)
        for view in route.teleport.views:
            add_reference(view.image, view.region, view.point)

    def project(self):
        # Fine view first; coarse views can still locate a beacon after opening a distant map.
        for localizer, point in self.references:
            projected = localizer.project_map_point(self.task.frame, (point.x, point.y), map_region(self.task.frame))
            if projected is not None:
                return projected
        return None

    _map_signature = staticmethod(map_signature)
    _same_map_signature = staticmethod(same_map_signature)
    _point_tolerance = staticmethod(map_point_tolerance)

    def _wait_for_map_stable(self, deadline):
        return wait_for_map_stable(self.task, deadline)

    def _stability_failure(self, stage):
        frame = getattr(self.task, 'frame', None)
        self.task.screenshot(f'3c-map-search-{stage}-unstable', frame=frame)
        log_warning = getattr(self.task, 'log_warning', None)
        if callable(log_warning):
            log_warning(f'Map view did not stabilize during {stage}; stopping beacon search')

    def _confirm_beacon(self, initial_projection, initial_beacon, deadline):
        """Reproject and re-detect the same beacon on two fresh settled frames."""
        tolerance = self._point_tolerance(self.task.frame)
        previous_projection = initial_projection
        previous_beacon = initial_beacon
        for _ in range(2):
            if time.monotonic() >= deadline or not self._wait_for_map_stable(deadline):
                self._stability_failure('beacon-confirmation')
                return None
            projection = self.project()
            if projection is None or math.dist(projection.xy, previous_projection.xy) > tolerance:
                self.task.screenshot('3c-map-search-beacon-projection-drift', frame=self.task.frame)
                return None
            beacon = self.beacon_at(projection)
            if (beacon is None
                    or math.dist(beacon.center(), previous_beacon.center()) > tolerance):
                self.task.screenshot('3c-map-search-beacon-drift', frame=self.task.frame)
                return None
            previous_projection = projection
            previous_beacon = beacon
        return previous_beacon

    def beacon_at(self, projected):
        height, width = self.task.frame.shape[:2]
        radius = max(20, round(width * .018))
        x1, y1 = max(0, round(projected.x) - radius), max(0, round(projected.y) - radius)
        x2, y2 = min(width, round(projected.x) + radius), min(height, round(projected.y) + radius)
        matches = self.task.find_feature(['map_way_point', 'map_way_point_big'],
                                          box=Box(x1, y1, x2 - x1, y2 - y1), threshold=.65)
        nearby = [box for box in matches if math.dist(box.center(), projected.xy) <= radius]
        return min(nearby, key=lambda box: math.dist(box.center(), projected.xy), default=None)

    def find(self, search=False):
        deadline = time.monotonic() + 40
        if not self._wait_for_map_stable(deadline):
            self._stability_failure('initial-view')
            return None
        projected = self.project()
        if projected is None and search:
            # Runtime zoom changes are deliberate; the recorded route image remains untouched.
            left, top, right, bottom = map_region(self.task.frame)
            self.task.scroll(round((left + right) / 2), round((top + bottom) / 2), -16)
            self.task.sleep(.5)
            if not self._wait_for_map_stable(deadline):
                self._stability_failure('overview-zoom')
                return None
            previous = (0, 0)
            for index, cell in enumerate(spiral_steps()):
                if time.monotonic() >= deadline:
                    break
                if index:
                    dx, dy = cell[0] - previous[0], cell[1] - previous[1]
                    cx, cy = (left + right) / 2, (top + bottom) / 2
                    try:
                        self.task.swipe(round(cx), round(cy), round(cx - dx * (right - left) * .42),
                                        round(cy - dy * (bottom - top) * .42), duration=.3, after_sleep=.25)
                    finally:
                        self.task.mouse_up()
                    if not self._wait_for_map_stable(deadline):
                        self._stability_failure('map-drag')
                        return None
                previous = cell
                self.task.info_set('Map Search Views', index + 1)
                projected = self.project()
                if projected is not None:
                    break
        if projected is None:
            return None
        for _ in range(7):
            beacon = self.beacon_at(projected)
            if beacon is not None:
                return self._confirm_beacon(projected, beacon, deadline)
            if time.monotonic() >= deadline:
                break
            # Zoom at the verified target, then re-align; never assume the cursor is the zoom centre.
            self.task.scroll(round(projected.x), round(projected.y), 2)
            self.task.sleep(.35)
            if not self._wait_for_map_stable(deadline):
                self._stability_failure('beacon-zoom')
                return None
            projected = self.project()
            if projected is None:
                break
        return None

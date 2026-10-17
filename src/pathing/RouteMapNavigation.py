"""Bounded visual map bookmarks. No game/world coordinates or blind teleport clicks."""

import math
import time

from ok import Box

from src.pathing.EchoRoute import read_map_image
from src.pathing.MapLocalizer import MapLocalizer


def map_region(frame):
    height, width = frame.shape[:2]
    return (round(width * .28), round(height * .12), round(width * .87), round(height * .90))


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
        projected = self.project()
        if projected is None and search:
            # Runtime zoom changes are deliberate; the recorded route image remains untouched.
            left, top, right, bottom = map_region(self.task.frame)
            self.task.scroll(round((left + right) / 2), round((top + bottom) / 2), -16)
            self.task.sleep(.5)
            self.task.next_frame()
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
                    self.task.next_frame()
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
                return beacon
            if time.monotonic() >= deadline:
                break
            # Zoom at the verified target, then re-align; never assume the cursor is the zoom centre.
            self.task.scroll(round(projected.x), round(projected.y), 2)
            self.task.sleep(.35)
            self.task.next_frame()
            projected = self.project()
            if projected is None:
                break
        return None

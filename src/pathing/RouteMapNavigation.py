"""Bounded visual map bookmarks. No game/world coordinates or blind teleport clicks."""

import math
import time
from copy import deepcopy

import cv2
import numpy as np
from ok import Box, TaskDisabledException

from src.pathing.EchoRoute import read_map_image
from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.RouteMapDrag import drag_route_map


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
        self.reference_names = []
        self.last_diagnostic = {}
        self.last_frame = None
        self._history = []
        self._popup_recoveries = 0
        seen = set()

        def add_reference(image_path, region, point):
            key = (str(image_path.resolve()), float(point.x), float(point.y), tuple(region))
            if key in seen:
                return
            seen.add(key)
            self.references.append((MapLocalizer(read_map_image(image_path), region), point))
            self.reference_names.append(str(image_path))

        if route.teleport.waypoint is not None:
            add_reference(route.map_image, route.map_region, route.teleport.waypoint)
        for view in route.teleport.views:
            add_reference(view.image, view.region, view.point)

    def project(self):
        """Recognize a view even when the target is offscreen, as in BetterGI TpTask.

        Share only this observation's SIFT extraction. A second matching reference
        must agree with the first; no prediction becomes permission to click.
        """
        frame = self.task.frame
        self.last_frame = frame.copy()
        region = map_region(frame)
        features = self.references[0][0].prepare_map_features(frame, region)
        accepted = []
        attempts = []
        for index, (localizer, point) in enumerate(self.references):
            projected = localizer.register_map_view(frame, (point.x, point.y), region, features=features)
            attempt = deepcopy(localizer.last_diagnostic)
            attempt['reference'] = self.reference_names[index]
            attempts.append(attempt)
            if projected is not None:
                if accepted and math.dist(accepted[0].xy, projected.xy) > max(12, map_point_tolerance(frame) * 2):
                    self._trace('registration', reason='reference_disagreement', attempts=attempts)
                    return None
                accepted.append(projected)
                if len(accepted) >= 2:
                    break
        result = max(accepted, key=lambda value: value.inliers * value.confidence, default=None)
        self._trace('registration', reason='matched' if result else 'registration_failed', attempts=attempts,
                    target=None if result is None else list(result.xy),
                    scale=None if result is None else result.scale)
        return result

    def _trace(self, stage, **details):
        entry = {'stage': stage, **details}
        self._history = (getattr(self, '_history', []) + [entry])[-48:]
        self.last_diagnostic = {**entry, 'history': deepcopy(self._history)}
        frame = getattr(self.task, '_last_map_ui_frame', None)
        if not isinstance(frame, np.ndarray):
            storage = getattr(getattr(self.task, '_executor', None), '__dict__', {})
            frame = storage.get('_frame', storage.get('frame'))
        if not isinstance(frame, np.ndarray):
            frame = getattr(self.task, '__dict__', {}).get('frame')
        if not isinstance(frame, np.ndarray):
            frame = getattr(self, 'last_frame', None)
        if isinstance(frame, np.ndarray):
            self.last_frame = frame.copy()
            record = getattr(self.task, '_record_navigation', None)
            if callable(record):
                record('record_map_ui', frame, 'bookmark:' + stage, diagnostic=entry)
        self.task.log_info(f'3C map search: {entry}')

    def _in_budget(self, deadline):
        if time.monotonic() < deadline:
            return True
        self._trace('stopped', reason='budget_exhausted')
        return False

    def _ready(self, stage, deadline):
        """Only dismiss a positively identified popup; a missed control is not one."""
        if not self._in_budget(deadline):
            return False
        button = self.task.wait_route_map_ui(time_out=min(.8, max(0, deadline - time.monotonic())))
        if not button:
            self._trace('map-ui', reason='map_controls_not_visible', operation=stage,
                        diagnostic=deepcopy(getattr(self.task, '_last_map_ui_diagnostic', {})))
            popup = self._known_popup_visible()
            if not self._in_budget(deadline):
                return False
            if popup:
                if getattr(self, '_popup_recoveries', 0) >= 1:
                    return False
                self._popup_recoveries = getattr(self, '_popup_recoveries', 0) + 1
                self._trace('map-ui', reason='known_popup_confirmed', operation=stage)
                if not self._in_budget(deadline):
                    return False
                self.task.send_key('esc', after_sleep=.2)
            button = self.task.wait_route_map_ui(time_out=min(1.5, max(0, deadline - time.monotonic())))
            if not button:
                self._trace('stopped', reason='map_ui_not_recovered' if popup else 'map_controls_not_confirmed',
                            operation=stage)
                return False
            self._trace('map-ui', reason='map_controls_recovered' if popup else 'map_controls_reconfirmed',
                        operation=stage)
        if not self._in_budget(deadline):
            return False
        if not self._wait_for_map_stable(deadline):
            self._stability_failure(stage)
            return False
        return self._in_budget(deadline)

    def _known_popup_visible(self):
        """Read the actual inspected frame; never infer a popup from low scores."""
        ocr = getattr(self.task, 'ocr', None)
        if not callable(ocr):
            return False
        frame = getattr(self.task, '_last_map_ui_frame', None)
        if not isinstance(frame, np.ndarray):
            return False
        try:
            hits = ocr(.68, .02, .97, .11, frame=frame, threshold=.8)
        except TaskDisabledException:
            raise
        except Exception as error:
            self._trace('map-ui', reason='popup_check_failed', error=str(error))
            return False
        if not isinstance(hits, (list, tuple)):
            return False
        labels = {'自定义标记', '自訂標記', '自定義標記', 'custom marker', 'custom mark'}
        for hit in hits:
            if not isinstance(getattr(hit, 'name', None), str) or hit.name.strip().casefold() not in labels:
                continue
            try:
                confidence = float(hit.confidence)
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(confidence) and confidence >= .8:
                return True
        return False

    def _safe_region(self):
        left, top, right, bottom = map_region(self.task.frame)
        margin = max(12, self.task.frame.shape[0] * .025)
        return left + margin, top + margin, right - margin, bottom - margin

    def _safe_target(self, projected):
        left, top, right, bottom = self._safe_region()
        return left <= projected.x < right and top <= projected.y < bottom

    def _measure_change(self, before, deadline):
        """Measure actual adjacent-view geometry, never infer it from wheel counts."""
        if not self._in_budget(deadline):
            return None
        region = map_region(before)
        point = ((region[0] + region[2]) / 2, (region[1] + region[3]) / 2)
        try:
            meter = MapLocalizer(before, region)
            changed = meter.register_map_view(self.task.frame, point, map_region(self.task.frame))
        except ValueError:
            changed = None
        if not self._in_budget(deadline):
            return None
        if changed is None:
            self._trace('stopped', reason='unverified_view_change')
        return changed

    def _zoom(self, amount, deadline, target=None):
        if not self._in_budget(deadline):
            return None, 'stopped'
        before = self.task.frame.copy()
        region = map_region(before)
        x, y = target.xy if target is not None else ((region[0] + region[2]) / 2,
                                                     (region[1] + region[3]) / 2)
        self._trace('zoom', wheel_notches=amount)
        if not self._in_budget(deadline):
            return None, 'stopped'
        self.task.scroll(round(x), round(y), amount)
        self.task.sleep(.25)
        if not self._ready('zoom-view', deadline):
            return None, 'stopped'
        signature_before, signature_after = map_signature(before), map_signature(self.task.frame)
        if (signature_before is not None and signature_after is not None
                and same_map_signature(signature_before, signature_after)):
            self._trace('zoom-result', reason='zoom_no_progress')
            return None, 'no_progress'
        change = self._measure_change(before, deadline)
        if change is None:
            return None, 'stopped'
        scale = change.scale
        if abs(scale - 1) < .005:
            self._trace('zoom-result', reason='zoom_no_progress', observed_scale=scale)
            return None, 'no_progress'
        if (amount < 0 and scale <= 1) or (amount > 0 and scale >= 1):
            self._trace('stopped', reason='unexpected_zoom_direction', observed_scale=scale)
            return None, 'stopped'
        self._trace('zoom-result', reason='zoom_verified', observed_scale=scale)
        result = self.project()
        if self.last_diagnostic.get('reason') == 'reference_disagreement':
            return None, 'stopped'
        return (result, 'changed') if self._in_budget(deadline) else (None, 'stopped')

    def _recenter(self, projected, deadline):
        """Move toward a recognized target, then measure the new view again."""
        if not self._in_budget(deadline):
            return None
        left, top, right, bottom = self._safe_region()
        center = np.array([(left + right) / 2, (top + bottom) / 2])
        delta = center - np.array(projected.xy)
        limit = min(1, (right - left) * .3 / max(abs(delta[0]), 1),
                    (bottom - top) * .3 / max(abs(delta[1]), 1))
        delta *= limit
        start, end = tuple(np.rint(center).astype(int)), tuple(np.rint(center + delta).astype(int))
        self._trace('directed-drag', target=list(projected.xy), delta=delta.tolist())
        if not self._in_budget(deadline):
            return None
        if deadline - time.monotonic() < .3:
            self._trace('stopped', reason='insufficient_drag_budget')
            return None
        drag_route_map(self.task, start, end, duration=.3)
        self.task.sleep(.2)
        if not self._ready('map-drag', deadline):
            return None
        result = self.project()
        if not self._in_budget(deadline):
            return None
        if result is None:
            self._trace('stopped', reason='registration_lost_after_drag')
            return None
        movement = np.array(result.xy) - np.array(projected.xy)
        if (np.dot(movement, delta) <= 0 or np.linalg.norm(movement) < 2
                or np.linalg.norm(np.array(result.xy) - center) >= np.linalg.norm(np.array(projected.xy) - center) - 2):
            self._trace('stopped', reason='drag_no_progress', observed_delta=movement.tolist())
            return None
        self._trace('drag-result', reason='drag_verified', observed_delta=movement.tolist())
        return result

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
        self._trace('stopped', reason='map_view_unstable', operation=stage)

    def _confirm_beacon(self, initial_projection, initial_beacon, deadline):
        """Reproject and re-detect the same beacon on two fresh settled frames."""
        tolerance = self._point_tolerance(self.task.frame)
        previous_projection = initial_projection
        previous_beacon = initial_beacon
        for _ in range(2):
            if time.monotonic() >= deadline or not self._ready('beacon-confirmation', deadline):
                self._stability_failure('beacon-confirmation')
                return None
            projection = self.project()
            if (not self._in_budget(deadline) or projection is None or not self._safe_target(projection)
                    or math.dist(projection.xy, previous_projection.xy) > tolerance):
                self.task.screenshot('3c-map-search-beacon-projection-drift', frame=self.task.frame)
                return None
            beacon = self.beacon_at(projection)
            if not self._in_budget(deadline):
                return None
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
        # The small and large templates may both detect the same icon.
        unique = []
        for box in sorted(nearby, key=lambda value: math.dist(value.center(), projected.xy)):
            if all(math.dist(box.center(), other.center()) > max(4, height * .003) for other in unique):
                unique.append(box)
        if len(unique) > 1:
            self._trace('beacon', reason='ambiguous_beacon_candidates', count=len(unique))
            return None
        return next(iter(unique), None)

    def find(self, search=False, *, current_view_only=False):
        deadline = time.monotonic() + (6 if current_view_only else 40)
        self._history = []
        self._popup_recoveries = 0
        if not self._ready('initial-view', deadline):
            return None
        projected = self.project()
        if not self._in_budget(deadline):
            return None
        if getattr(self, 'last_diagnostic', {}).get('reason') == 'reference_disagreement':
            return None
        if current_view_only and (projected is None or not self._safe_target(projected)):
            self._trace('current-view', reason='no_visible_route_bookmark')
            return None
        if projected is None and search:
            # BetterGI observes every zoom step and stops on lack of progress.
            # Without an atlas, an unregistered view cannot justify a drag direction.
            for direction, count in ((-2, 6), (2, 12)):
                stalled = 0
                for _ in range(count):
                    projected, state = self._zoom(direction, deadline)
                    if state == 'stopped':
                        return None
                    if projected is not None:
                        break
                    stalled = stalled + 1 if state == 'no_progress' else 0
                    if stalled >= 2:
                        break
                if projected is not None:
                    break
        if projected is None:
            self._trace('stopped', reason='registration_failed')
            return None
        for _ in range(10):
            if not self._in_budget(deadline):
                return None
            if not self._safe_target(projected):
                projected = self._recenter(projected, deadline)
                if projected is None:
                    return None
                continue
            beacon = self.beacon_at(projected)
            if not self._in_budget(deadline):
                return None
            if beacon is not None:
                return self._confirm_beacon(projected, beacon, deadline)
            if current_view_only:
                self._trace('current-view', reason='beacon_not_visible')
                return None
            projected, state = self._zoom(2, deadline, target=projected)
            if state != 'changed' or projected is None:
                self._trace('stopped', reason='beacon_not_visible')
                return None
        self._trace('stopped', reason='operation_limit')
        return None

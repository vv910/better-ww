"""Inspect one registered beacon without travelling or editing map markers."""

import math
import time
from copy import deepcopy
from numbers import Real

import numpy as np
from ok import Box, TaskDisabledException

from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.RouteBeaconLocation import read_beacon_location
from src.pathing.RouteMapNavigation import (map_region, map_signature,
                                            same_map_signature, wait_for_map_stable)
from src.pathing.RouteMapUI import RouteMapUI


class _Expired(Exception):
    pass


def _check(deadline):
    if time.monotonic() >= deadline:
        raise _Expired('beacon-location-budget-exhausted')


def _image(frame):
    return (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
            and frame.dtype == np.uint8 and min(frame.shape[:2]) > 0)


def _ordinary_map(task, frame, detector, deadline):
    control = detector.detect(frame)
    _check(deadline)
    if control is not None:
        return True
    try:
        return bool(task.find_one('map_my_location', frame=frame))
    except TaskDisabledException:
        raise
    except Exception:
        return False


def close_beacon_details(task, time_out=3):
    """Close a fresh, confirmed beacon detail and verify the ordinary map.

    A failed click may have left the ordinary map untouched. It needs no Esc.
    Unknown panels/overlapping-marker lists are never guessed or clicked.
    Captures have a short executor timeout and never use an implicit old frame.
    """
    if (isinstance(time_out, bool) or not isinstance(time_out, Real)
            or not math.isfinite(time_out) or not 0 < time_out <= 10):
        raise ValueError('Beacon cleanup timeout must be greater than zero and at most ten seconds')
    deadline = time.monotonic() + time_out
    detector = RouteMapUI()
    previous = None
    quiet_pairs = 0
    escaped = False
    diagnostic = {'status': 'unconfirmed', 'esc_sent': False, 'fresh_frames': 0}
    task._last_route_beacon_cleanup_diagnostic = diagnostic
    try:
        while time.monotonic() < deadline:
            frame = task.executor.next_frame(time_out=min(.5, deadline - time.monotonic()))
            if _image(frame):
                diagnostic['fresh_frames'] += 1
            _check(deadline)
            if not _image(frame):
                previous, quiet_pairs = None, 0
                continue
            ordinary = _ordinary_map(task, frame, detector, deadline)
            _check(deadline)
            if ordinary:
                signature = map_signature(frame)
                if signature is not None and previous is not None and same_map_signature(previous, signature):
                    quiet_pairs += 1
                    if quiet_pairs >= 2:
                        diagnostic.update(status='restored', reason='ordinary_map_confirmed')
                        return
                else:
                    quiet_pairs = 0
                previous = signature
            else:
                previous, quiet_pairs = None, 0
                details = {}
                diagnostic['detail'] = details
                read_beacon_location(task, frame, diagnostic=details)
                _check(deadline)
                if details.get('header_confirmed') and not escaped:
                    # This is the only permitted cleanup input. Never use a
                    # generic travel helper, which can also remove custom pins.
                    escaped = True
                    diagnostic['esc_sent'] = True
                    task.send_key('esc', after_sleep=0)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                task.executor.sleep(min(.05, remaining))
        raise _Expired('ordinary-map-not-confirmed-after-beacon-details')
    except TaskDisabledException as error:
        # Location OCR can be cancelled after this fresh frame's trusted header
        # was read. Close that confirmed detail once, without new business work.
        if diagnostic.get('detail', {}).get('header_confirmed') and not escaped:
            diagnostic['esc_sent'] = True
            try:
                task.send_key('esc', after_sleep=0)
            except BaseException as release_error:
                error.add_note(f'Beacon cleanup Esc failed: {release_error}')
        diagnostic.update(status='cancelled', reason='cleanup_cancelled')
        raise
    except Exception as error:
        diagnostic.update(status='unconfirmed', reason=str(error))
        raise RuntimeError('Cannot restore map after reading beacon location / '
                           '信标详情关闭后地图状态未确认，停止保存路线') from error


def _unique_beacon(task, frame, projected):
    height, width = frame.shape[:2]
    radius = max(20, round(width * .018))
    left, top = max(0, round(projected.x) - radius), max(0, round(projected.y) - radius)
    right, bottom = min(width, round(projected.x) + radius), min(height, round(projected.y) + radius)
    matches = task.find_feature(['map_way_point', 'map_way_point_big'], frame=frame,
                                box=Box(left, top, right - left, bottom - top), threshold=.65)
    nearby = [box for box in matches if math.dist(box.center(), projected.xy) <= radius]
    unique = []
    for box in sorted(nearby, key=lambda value: math.dist(value.center(), projected.xy)):
        if all(math.dist(box.center(), other.center()) > max(4, height * .003) for other in unique):
            unique.append(box)
    return unique[0] if len(unique) == 1 else None, len(unique)


def capture_beacon_location(task, reference_frame, point_xy, reference_region, time_out=10):
    """Read two fresh equal location chains after one geometrically verified click.

    The caller supplies the last successfully saved overview (or the original
    recording if there is no overview). No recorded pixels or points are changed.
    Unavailable evidence returns None; an unconfirmed restoration prevents saving.
    """
    if (isinstance(time_out, bool) or not isinstance(time_out, Real)
            or not math.isfinite(time_out) or not 0 < time_out <= 30):
        raise ValueError('Beacon capture timeout must be greater than zero and at most thirty seconds')
    diagnostic = {'status': 'unavailable', 'reason': 'no_observation', 'clicks': 0, 'observations': []}
    task._last_route_beacon_location_diagnostic = diagnostic
    deadline = time.monotonic() + time_out
    clicked = False
    last_details = {}
    primary_error = None
    try:
        _check(deadline)
        if not task.wait_route_map_ui(time_out=min(2, max(0, deadline - time.monotonic()))):
            diagnostic['reason'] = 'ordinary_map_controls_not_confirmed'
            return None
        _check(deadline)
        if not wait_for_map_stable(task, deadline):
            diagnostic['reason'] = 'map_view_not_stable'
            return None
        _check(deadline)
        current = task.frame.copy()
        if not _image(reference_frame) or current.shape != reference_frame.shape:
            diagnostic['reason'] = 'invalid_or_changed_capture_size'
            return None
        _check(deadline)
        localizer = MapLocalizer(reference_frame, reference_region)
        _check(deadline)
        projected = localizer.register_map_view(current, point_xy, map_region(current))
        diagnostic['registration'] = deepcopy(localizer.last_diagnostic)
        _check(deadline)
        if projected is None:
            diagnostic['reason'] = 'current_view_not_registered'
            return None
        left, top, right, bottom = map_region(current)
        margin = max(12, current.shape[0] * .025)
        if (not projected.inside_view or not left + margin <= projected.x < right - margin
                or not top + margin <= projected.y < bottom - margin):
            diagnostic['reason'] = 'projected_beacon_outside_safe_view'
            return None
        beacon, count = _unique_beacon(task, current, projected)
        diagnostic['beacon_candidates'] = count
        _check(deadline)
        if beacon is None:
            diagnostic['reason'] = 'beacon_not_unique'
            return None
        diagnostic['click_position'] = list(beacon.center())
        clicked = True  # The backend can send DOWN before raising an exception.
        diagnostic['clicks'] = 1
        task.click(beacon, after_sleep=0)
        previous = None
        while time.monotonic() < deadline:
            frame = task.next_frame()
            _check(deadline)
            if not _image(frame):
                diagnostic['reason'] = 'detail_capture_unavailable'
                return None
            last_details = {}
            location = read_beacon_location(task, frame, diagnostic=last_details)
            diagnostic['observations'] = (diagnostic['observations'] + [deepcopy(last_details)])[-6:]
            _check(deadline)
            if location is not None and previous is not None and location.parts == previous.parts:
                diagnostic.update(status='matched', reason='two_fresh_beacon_locations_confirmed',
                                  parts=list(location.parts), raw=location.raw)
                return location
            previous = location
            task.sleep(min(.1, max(0, deadline - time.monotonic())))
        raise _Expired('beacon-location-not-confirmed')
    except TaskDisabledException as error:
        diagnostic.update(status='cancelled', reason='beacon_capture_cancelled')
        primary_error = error
        raise
    except Exception as error:
        diagnostic.update(status='unavailable', reason=str(error))
        return None
    finally:
        if clicked:
            try:
                close_beacon_details(task, time_out=3)
                diagnostic['cleanup'] = deepcopy(task._last_route_beacon_cleanup_diagnostic)
            except BaseException as cleanup_error:
                diagnostic['cleanup'] = deepcopy(getattr(task, '_last_route_beacon_cleanup_diagnostic', {}))
                if primary_error is None:
                    raise
                # Cancellation is never replaced by a secondary restore error.
                # If cancellation prevented a fresh cleanup capture, only a
                # header from this click's actual inspected detail permits Esc.
                if (last_details.get('header_confirmed')
                        and not diagnostic['cleanup'].get('esc_sent')
                        and not diagnostic['cleanup'].get('fresh_frames')):
                    try:
                        task.send_key('esc', after_sleep=0)
                        diagnostic['cleanup']['esc_sent'] = True
                    except BaseException as release_error:
                        primary_error.add_note(f'Beacon cleanup Esc failed: {release_error}')
                primary_error.add_note(f'Beacon map restoration unconfirmed: {cleanup_error}')

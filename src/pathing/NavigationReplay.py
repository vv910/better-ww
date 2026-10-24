"""Bounded navigation evidence and offline replay. This module never sends input."""

import copy
import hashlib
import io
import json
import math
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MAX_RAW_BYTES = 24 * 1024 * 1024
MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
MAX_EVENTS_BYTES = 8 * 1024 * 1024


def _json_value(value, depth=0):
    """Keep diagnostics JSON-safe and bounded, including numpy scalar values."""
    if depth > 10:
        return '<depth limit>'
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:4000]
    if isinstance(value, dict):
        return {str(key)[:100]: _json_value(item, depth + 1)
                for key, item in list(value.items())[:256]}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, depth + 1) for item in value[:256]]
    return str(value)[:4000]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _check_image_header(content, max_pixels):
    # Check dimensions before OpenCV allocates decoded pixels, including tiny PNGs
    # that would otherwise expand beyond a replay's memory budget.
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
        if width <= 0 or height <= 0 or width * height > max_pixels:
            raise ValueError('Decoded replay image exceeds pixel budget')
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f'Cannot inspect replay image header: {error}') from error


def _read_image(path, max_raw_bytes=None):
    if path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError('Replay image exceeds 64 MiB')
    content = path.read_bytes()
    max_pixels = 40_000_000 if max_raw_bytes is None else max_raw_bytes // 3
    _check_image_header(content, max_pixels)
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode replay image: {path.name}')
    return image


def _bundle_path(root, relative):
    """Manifest paths are untrusted: reject absolute paths, traversal and symlinks out."""
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Replay paths must be non-empty relative POSIX paths')
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root):
        raise ValueError(f'Replay path leaves bundle: {relative}')
    return candidate


class NavigationReplay:
    """Retain recent visual inputs in RAM and write once after failure.

    Times accepted by record_* are absolute monotonic seconds. Images are copied,
    never resized or encoded while walking. Disk failures are exposed in
    ``last_error`` and never replace the original task exception.
    """

    def __init__(self, route_path, arrow_template=None, capture_size=None, *,
                 max_frames=120, max_bytes=MAX_RAW_BYTES, max_events=2000):
        if not 1 <= max_frames <= 120 or not 1 <= max_bytes <= MAX_RAW_BYTES:
            raise ValueError('Replay ring exceeds 120 frames or 24 MiB')
        if not 1 <= max_events <= 2000:
            raise ValueError('Replay event limit must be between 1 and 2000')
        self.route_path = Path(route_path).expanduser().resolve()
        self.origin = time.monotonic()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.capture_size = _json_value(capture_size)
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.events = deque(maxlen=max_events)
        self.frames = OrderedDict()
        self.raw_bytes = 0
        self.events_total = self.frames_total = self.frames_evicted = 0
        self.last_error = None
        self.errors = []
        self._flushed = False
        self._destination = None
        self._arrow = None
        if arrow_template is not None:
            self.set_arrow_template(arrow_template)
        try:
            if self.route_path.stat().st_size > 1024 * 1024:
                raise ValueError('Route JSON exceeds 1 MiB')
            self._route = json.loads(self.route_path.read_text(encoding='utf-8-sig'))
            if not isinstance(self._route, dict):
                raise ValueError('Route JSON must be an object')
        except (OSError, ValueError) as error:
            self._route = None
            self._error(f'Route snapshot failed: {error}')

    def _error(self, message):
        self.last_error = str(message)
        if len(self.errors) < 30:
            self.errors.append(self.last_error)

    def set_arrow_template(self, template):
        """Accept the lazily loaded arrow without retaining a feature-set object."""
        if (not isinstance(template, np.ndarray) or template.dtype != np.uint8
                or template.ndim != 3 or template.shape[2] != 3 or min(template.shape[:2]) < 1
                or template.nbytes > 1024 * 1024):
            self._error('Arrow template must be a BGR image within 1 MiB raw-memory budget')
            return False
        self._arrow = template.copy()
        return True

    def _record(self, kind, timestamp=None, **data):
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(stamp):
            self._error('Non-finite navigation timestamp was not recorded')
            return
        self.events_total += 1
        self.events.append({'kind': kind, 'seq': self.events_total,
                            't': max(0.0, stamp - self.origin), **_json_value(data)})

    def _frame(self, image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim not in (2, 3) or min(image.shape[:2]) < 1
                or (image.ndim == 3 and image.shape[2] != 3)):
            self._error('Invalid navigation image was not retained')
            return None
        if image.nbytes > self.max_bytes:
            self._error('Navigation image exceeds replay raw-memory budget')
            return None
        if self.frames:
            previous_id = next(reversed(self.frames))
            if np.array_equal(self.frames[previous_id], image):
                return previous_id
        self.frames_total += 1
        frame_id = f'{self.frames_total:06d}'
        while self.frames and (len(self.frames) >= self.max_frames
                               or self.raw_bytes + image.nbytes > self.max_bytes):
            _, old = self.frames.popitem(last=False)
            self.raw_bytes -= old.nbytes
            self.frames_evicted += 1
            del old
        stored = image.copy()
        self.frames[frame_id] = stored
        self.raw_bytes += stored.nbytes
        return frame_id

    def record_locate(self, minimap, position, diagnostic, *, started_at=None, elapsed=None, capture_size=None):
        self._record('locate', started_at, frame=self._frame(minimap), position=position,
                     diagnostic=diagnostic, elapsed_ms=None if elapsed is None else elapsed * 1000,
                     capture_size=capture_size or self.capture_size)

    def record_heading(self, minimap, estimate, *, timestamp=None, capture_size=None):
        self._record('heading', timestamp, frame=self._frame(minimap), estimate=estimate,
                     capture_size=capture_size or self.capture_size)

    def record_map_ui(self, frame, stage, diagnostic=None, timestamp=None):
        """Retain a failed map check's actual full frame within the same ring budget.

        A map UI observation is never a minimap localization input. Only error
        paths call this, after movement has stopped.
        """
        frame_id = self._frame(frame)
        if frame_id is not None:
            self.capture_size = [int(frame.shape[1]), int(frame.shape[0])]
        self._record('map_ui', timestamp, frame=frame_id, stage=stage,
                     diagnostic=diagnostic, capture_size=self.capture_size)

    def record_reset(self, position_hint=None, *, timestamp=None):
        self._record('reset', timestamp, hint=position_hint)

    def record_absolute_request(self, *, timestamp=None):
        self._record('absolute_request', timestamp)

    def record_keys(self, keys, *, timestamp=None):
        self._record('keys', timestamp, keys=keys)

    def record_target(self, point, *, timestamp=None):
        self._record('target', timestamp, target=point)

    def flush(self, reason, *, root='logs/3c-replays', metadata=None):
        """Write one self-contained bundle; repeat calls do no I/O, even after error."""
        if self._flushed:
            return self._destination
        self._flushed = True
        destination = None
        try:
            destination = Path(root).resolve() / (
                datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:10])
            destination.mkdir(parents=True, exist_ok=False)
            (destination / 'frames').mkdir()
            (destination / 'assets').mkdir()
            route = copy.deepcopy(self._route)
            assets = []
            used_bytes = 0
            if route is not None:
                references = [route]
                teleport = route.get('teleport')
                if isinstance(teleport, dict) and isinstance(teleport.get('views', []), list):
                    references.extend(view for view in teleport.get('views', []) if isinstance(view, dict))
                copied = {}
                for index, reference in enumerate(references):
                    original = reference.get('map_image')
                    if not isinstance(original, str) or not original:
                        self._error(f'Missing map_image in route reference {index}')
                        continue
                    source = (self.route_path.parent / original).resolve()
                    region = reference.get('map_region')
                    copy_key = source, tuple(region) if isinstance(region, list) else None
                    if copy_key in copied:
                        reference['map_image'] = copied[copy_key]
                        continue
                    relative = f'assets/map-{index:02d}.png'
                    reference['map_image'] = relative
                    copied[copy_key] = relative
                    try:
                        size = source.stat().st_size
                        if size > MAX_ASSET_BYTES:
                            raise ValueError('Route assets exceed 64 MiB budget')
                        content = source.read_bytes()
                        # Copy declared image references only, never arbitrary adjacent files.
                        _check_image_header(content, 40_000_000)
                        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError('Route map reference is not a decodable image')
                        height, width = image.shape[:2]
                        region = [0, 0, width, height] if region is None else region
                        if (not isinstance(region, list) or len(region) != 4
                                or any(type(value) is not int for value in region)):
                            raise ValueError('Route asset map_region must contain four integers')
                        left, top, right, bottom = region
                        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                            raise ValueError('Route asset map_region is outside its image')
                        # Preserve all coordinates while excluding account/UI pixels.
                        safe_image = np.zeros_like(image)
                        safe_image[top:bottom, left:right] = image[top:bottom, left:right]
                        encoded, pixels = cv2.imencode('.png', safe_image)
                        if not encoded or used_bytes + pixels.nbytes > MAX_ASSET_BYTES:
                            raise ValueError('Redacted route assets exceed encoding budget')
                        content = pixels.tobytes()
                        (destination / relative).write_bytes(content)
                        used_bytes += len(content)
                        assets.append({'path': relative, 'sha256': hashlib.sha256(content).hexdigest()})
                    except (OSError, ValueError, cv2.error) as error:
                        self._error(f'Route asset {index} unavailable: {error}')
                _write_json(destination / 'route.json', route)
            frame_files = {}
            for frame_id, frame in self.frames.items():
                relative = f'frames/{frame_id}.png'
                encoded, pixels = cv2.imencode('.png', frame)
                if not encoded:
                    self._error(f'Could not encode minimap {frame_id}')
                    continue
                used_bytes += pixels.nbytes
                if used_bytes > MAX_BUNDLE_BYTES:
                    self._error('Replay output exceeds bundle budget')
                    break
                (destination / relative).write_bytes(pixels.tobytes())
                frame_files[frame_id] = relative
            arrow_file = None
            if self._arrow is not None:
                encoded, pixels = cv2.imencode('.png', self._arrow)
                if encoded and pixels.nbytes <= 1024 * 1024:
                    arrow_file = 'assets/arrow.png'
                    (destination / arrow_file).write_bytes(pixels.tobytes())
                else:
                    self._error('Arrow template could not be saved within 1 MiB')
            event_lines = ''.join(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n'
                                  for event in self.events)
            if len(event_lines.encode('utf-8')) > MAX_EVENTS_BYTES:
                raise ValueError('Replay event metadata exceeds 8 MiB')
            (destination / 'events.jsonl').write_text(event_lines, encoding='utf-8')
            _write_json(destination / 'metadata.json', {
                'schema_version': 1, 'created_at_utc': self.created_at, 'reason': _json_value(reason),
                'route': 'route.json' if route is not None else None, 'events': 'events.jsonl',
                'arrow_template': arrow_file, 'frames': frame_files, 'assets': assets,
                'capture_size': self.capture_size, 'events_total': self.events_total,
                'events_retained': len(self.events), 'frames_total': self.frames_total,
                'frames_evicted': self.frames_evicted, 'raw_frame_bytes': self.raw_bytes,
                'truncated': self.frames_evicted > 0 or self.events_total > len(self.events),
                'errors': self.errors, 'extra': _json_value(metadata),
                'timing_note': 'Recorded monotonic event times preserve gaps and anchor/hint ages. '
                               'Replay uses each recorded locate duration as a conservative completion '
                               'time after its first clock read; intermediate clock readings and CPU '
                               'execution cannot be reproduced exactly. It cannot certify live freshness.',
            })
            self._destination = destination
            return destination
        except Exception as error:
            # Diagnostics must not mask cancellation, death, or the navigation exception.
            self._error(f'Replay bundle write failed: {error}')
            return None


def load_replay(path):
    """Read a bounded local bundle. No archives are extracted and no input is sent."""
    root = Path(path).resolve()
    meta_path = _bundle_path(root, 'metadata.json')
    if meta_path.stat().st_size > 1024 * 1024:
        raise ValueError('Replay metadata exceeds 1 MiB')
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or metadata.get('schema_version') != 1:
        raise ValueError('Unsupported navigation replay schema')
    events_path = _bundle_path(root, metadata.get('events'))
    if events_path.stat().st_size > MAX_EVENTS_BYTES:
        raise ValueError('Replay event metadata exceeds 8 MiB')
    events = []
    previous_time = -math.inf
    for line in events_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        stamp = event.get('t') if isinstance(event, dict) else None
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or stamp < 0 or stamp < previous_time):
            raise ValueError('Replay events require finite, nondecreasing monotonic times')
        if event.get('kind') not in ('locate', 'heading', 'reset', 'absolute_request', 'keys', 'target', 'map_ui'):
            raise ValueError('Replay event kind is unsupported')
        for field in ('position', 'diagnostic', 'estimate'):
            if event.get(field) is not None and not isinstance(event[field], dict):
                raise ValueError(f'Replay event {field} must be an object or null')
        previous_time = stamp
        events.append(event)
        if len(events) > 2000:
            raise ValueError('Replay contains more than 2000 retained events')
    frames = metadata.get('frames')
    if not isinstance(frames, dict) or len(frames) > 120:
        raise ValueError('Replay frame list exceeds 120 frames')
    for relative in frames.values():
        _bundle_path(root, relative)
    validation_errors = []
    total_bytes = meta_path.stat().st_size + events_path.stat().st_size
    referenced = list(frames.values())
    if metadata.get('arrow_template'):
        referenced.append(metadata['arrow_template'])
    if metadata.get('route'):
        route_path = _bundle_path(root, metadata['route'])
        if not route_path.is_file():
            validation_errors.append('Route JSON is missing')
        elif route_path.stat().st_size > 1024 * 1024:
            raise ValueError('Replay route exceeds 1 MiB')
        else:
            route = json.loads(route_path.read_text(encoding='utf-8'))
            if not isinstance(route, dict):
                raise ValueError('Replay route must be an object')
            references = [route]
            teleport = route.get('teleport')
            if isinstance(teleport, dict) and isinstance(teleport.get('views', []), list):
                references.extend(view for view in teleport.get('views', []) if isinstance(view, dict))
            for reference in references:
                if not isinstance(reference.get('map_image'), str):
                    validation_errors.append('Route map reference is missing')
                else:
                    referenced.append(reference['map_image'])
    else:
        validation_errors.append('Bundle has no route snapshot')
    for relative in set(referenced):
        file_path = _bundle_path(root, relative)
        if not file_path.is_file():
            validation_errors.append(f'Referenced image is missing: {relative}')
            continue
        size = file_path.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise ValueError('Replay image exceeds 64 MiB')
        total_bytes += size
    if total_bytes > MAX_BUNDLE_BYTES:
        raise ValueError('Replay files exceed 128 MiB')
    metadata['validation_errors'] = validation_errors
    return root, metadata, events


def replay_statistics(events):
    """Report recorded availability and discontinuities, not position accuracy."""
    observations = [event for event in events if event.get('kind') == 'locate']
    accepted = [event for event in observations if isinstance(event.get('position'), dict)]
    sources = Counter(event['position'].get('source', 'unknown') for event in accepted)
    reasons = Counter((event.get('diagnostic') or {}).get('reason', 'unknown')
                      for event in observations if not isinstance(event.get('position'), dict))
    longest_gap = 0.0
    failed_at = None
    last_position = None
    largest_step = 0.0
    jump_count = 0
    for event in events:
        if event.get('kind') == 'reset':
            last_position = None
            if failed_at is not None:
                longest_gap = max(longest_gap, event['t'] - failed_at)
                failed_at = None
        if event.get('kind') != 'locate':
            continue
        position = event.get('position')
        if not isinstance(position, dict):
            if failed_at is None:
                failed_at = event['t']
            longest_gap = max(longest_gap, event['t'] - failed_at)
            continue
        if failed_at is not None:
            longest_gap = max(longest_gap, event['t'] - failed_at)
            failed_at = None
        xy = position.get('x'), position.get('y')
        if all(isinstance(value, (int, float)) and math.isfinite(value) for value in xy):
            if last_position is not None:
                step = math.dist(xy, last_position)
                largest_step = max(largest_step, step)
                jump_count += step > 100
            last_position = xy
    elapsed = [event.get('elapsed_ms') for event in observations]
    elapsed = [value for value in elapsed if isinstance(value, (int, float)) and math.isfinite(value)]
    headings = [event for event in events if event.get('kind') == 'heading']
    map_checks = [event for event in events if event.get('kind') == 'map_ui']
    reliable = sum(isinstance(event.get('estimate'), dict)
                   and (event['estimate'].get('confidence') or 0) >= .72
                   and (event['estimate'].get('margin') or 0) >= .08 for event in headings)
    return {'scope': 'retained observations; accepted coordinates are not ground truth',
            'locate_calls': len(observations), 'located_calls': len(accepted),
            'availability': None if not observations else len(accepted) / len(observations),
            'position_sources': dict(sources), 'rejection_reasons': dict(reasons),
            'longest_unavailable_seconds': round(longest_gap, 3),
            'largest_accepted_step_map_pixels': round(largest_step, 3),
            'accepted_steps_over_100_map_pixels': jump_count,
            'heading_calls': len(headings), 'reliable_heading_calls': reliable,
            'map_ui_events': len(map_checks),
            'map_ui_stages': dict(Counter(event.get('stage', 'unknown') for event in map_checks)),
            'mean_locate_ms': None if not elapsed else round(sum(elapsed) / len(elapsed), 3)}


def relocalize_replay(path):
    """Re-run vision on retained frames with recorded time gaps, without game APIs.

    Missing/evicted inputs explicitly break tracking continuity. Virtual time is
    fixed at each recorded event; CPU timing inside that event is not reproduced.
    Recorded map hints keep their original age and are never renewed by replay.
    """
    from unittest.mock import patch

    from src.pathing.MapLocalizer import MapLocalizer, MapPosition
    from src.pathing.MinimapHeading import MinimapHeading

    root, metadata, events = load_replay(path)
    route_path = _bundle_path(root, metadata.get('route'))
    if route_path.stat().st_size > 1024 * 1024:
        raise ValueError('Replay route exceeds 1 MiB')
    route = json.loads(route_path.read_text(encoding='utf-8'))
    image_path = _bundle_path(root, route.get('map_image'))
    if image_path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError('Replay map exceeds asset budget')
    localizer = MapLocalizer(_read_image(image_path), route.get('map_region'))
    heading = None
    if metadata.get('arrow_template'):
        heading = MinimapHeading(_read_image(_bundle_path(root, metadata['arrow_template']), 1024 * 1024))
    output = []
    missing = 0
    raw_bytes = 0
    cache = {}
    clock = [0.0, 0.0, 0]

    def observed_clock():
        clock[2] += 1
        return clock[0] if clock[2] == 1 else clock[0] + clock[1]

    with patch('src.pathing.MapLocalizer.time.monotonic', side_effect=observed_clock):
        localizer.reset()
        for event in events:
            clock[0] = event['t']
            elapsed = event.get('elapsed_ms')
            clock[1] = (max(0.0, elapsed / 1000) if event.get('kind') == 'locate'
                        and isinstance(elapsed, (int, float)) and math.isfinite(elapsed) else 0.0)
            clock[2] = 0
            kind = event.get('kind')
            if kind == 'absolute_request':
                localizer.request_absolute()
                output.append(copy.deepcopy(event))
                continue
            if kind == 'reset':
                hint = event.get('hint')
                if isinstance(hint, dict) and hint.get('source') == 'map':
                    hint = MapPosition(hint.get('x'), hint.get('y'), hint.get('confidence', 0),
                                       hint.get('inliers', 0), source='map')
                elif isinstance(hint, list) and len(hint) == 2:
                    hint = tuple(hint)
                else:
                    hint = None
                localizer.reset(hint)
                output.append(copy.deepcopy(event))
                continue
            if kind not in ('locate', 'heading'):
                output.append(copy.deepcopy(event))
                continue
            frame_id = event.get('frame')
            relative = metadata['frames'].get(frame_id)
            if relative is None or not _bundle_path(root, relative).is_file():
                missing += 1
                localizer.reset()
                output.append({'kind': 'reset', 't': clock[0], 'hint': None, 'reason': 'missing_frame'})
                continue
            if frame_id not in cache:
                frame_path = _bundle_path(root, relative)
                if frame_path.stat().st_size > MAX_RAW_BYTES:
                    raise ValueError('Replay frame exceeds image budget')
                frame = _read_image(frame_path, MAX_RAW_BYTES)
                raw_bytes += frame.nbytes
                if raw_bytes > MAX_RAW_BYTES:
                    raise ValueError('Replay raw frames exceed 24 MiB')
                cache[frame_id] = frame
            frame = cache[frame_id]
            observed = copy.deepcopy(event)
            started = time.perf_counter()
            if kind == 'locate':
                position = localizer.locate(frame)
                observed.update(position=_json_value(position), diagnostic=_json_value(localizer.last_diagnostic),
                                elapsed_ms=(time.perf_counter() - started) * 1000)
            elif heading is not None:
                observed['estimate'] = _json_value(heading.estimate(frame))
            else:
                observed['estimate'] = None
            output.append(observed)
    return {'statistics': replay_statistics(output), 'recorded_statistics': replay_statistics(events),
            'missing_frame_observations': missing,
            'retained_events': len(events), 'replayed_events': len(output),
            'truncated': metadata.get('truncated'), 'frames_evicted': metadata.get('frames_evicted'),
            'timing_note': metadata.get('timing_note'),
            'errors': metadata.get('errors', []) + metadata.get('validation_errors', [])}

"""Read a beacon detail's own location line without operating the game UI."""

from dataclasses import dataclass
import math
from numbers import Real
import re
import unicodedata

import numpy as np


BEACON_HEADER_REGION = (.73, .025, .925, .095)
BEACON_LOCATION_REGION = (.685, .126, .985, .18)
MIN_OCR_CONFIDENCE = .6
# Exact detail titles, never arbitrary text containing "beacon" or "信标".
BEACON_HEADERS = frozenset({
    '小型信标', '中枢信标', '共鸣信标',
    '小型信標', '中樞信標', '共鳴信標',
    'small beacon', 'small resonance beacon', 'resonance beacon', 'resonance nexus',
})
_LOCATION_SEPARATOR = re.compile(r'[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]')


@dataclass(frozen=True)
class RouteBeaconLocation:
    world: str
    map_name: str
    area: str
    raw: str

    @property
    def parts(self):
        return self.world, self.map_name, self.area


def parse_beacon_location(raw):
    """Parse exactly three complete location fields, retaining the displayed line.

    These fields describe a beacon's location, not a guaranteed map-menu path.
    Missing fields, ambiguous additional separators and line/control characters
    are rejected rather than repaired using remembered scene or menu names.
    """
    if not isinstance(raw, str) or not raw or len(raw) > 240:
        return None
    if any(unicodedata.category(character) in ('Cc', 'Cf', 'Zl', 'Zp') for character in raw):
        return None
    text = raw.strip()
    parts = tuple(part.strip() for part in _LOCATION_SEPARATOR.split(text))
    if (len(parts) != 3 or any(not part or len(part) > 80 for part in parts)
            or any(not any(character.isalpha() for character in part) for part in parts)):
        return None
    return RouteBeaconLocation(*parts, raw=text)


def _header_name(box):
    name = getattr(box, 'name', None)
    if not isinstance(name, str):
        return None
    if any(unicodedata.category(character) in ('Cc', 'Cf', 'Zl', 'Zp') for character in name):
        return None
    return ' '.join(name.strip().split()).casefold()


def _trusted_box(box, region, frame):
    confidence = getattr(box, 'confidence', None)
    if isinstance(confidence, (bool, np.bool_)):
        return False
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(confidence) or not MIN_OCR_CONFIDENCE <= confidence <= 1:
        return False
    coordinates = tuple(getattr(box, key, None) for key in ('x', 'y', 'width', 'height'))
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           or not math.isfinite(value) for value in coordinates):
        return False
    x, y, width, height = coordinates
    if width <= 0 or height <= 0:
        return False
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = (region[0] * frame_width, region[1] * frame_height,
                               region[2] * frame_width, region[3] * frame_height)
    return left <= x and top <= y and x + width <= right and y + height <= bottom


def read_beacon_location(task, frame, *, diagnostic=None):
    """OCR only the inspected frame's fixed beacon header and location strip.

    Reading a position neither proves that the beacon is unlocked nor permits
    travel. Cross-frame stability, cancellation and time budgets belong to the
    caller. The optional diagnostic dict receives this call's result only.
    This function never reads task.frame, clicks, switches maps or sends
    keys, and deliberately does not join partial OCR boxes into guessed fields.
    """
    if diagnostic is not None:
        diagnostic.clear()
        diagnostic.update(status='rejected', header_confirmed=False)

    def reject(reason):
        if diagnostic is not None:
            diagnostic['reason'] = reason
        return None

    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3
            or min(frame.shape[:2]) < 360 or frame.dtype != np.uint8):
        return reject('invalid_frame')
    headers = list(task.ocr(*BEACON_HEADER_REGION, threshold=MIN_OCR_CONFIDENCE, frame=frame))
    candidates = [box for box in headers if _header_name(box) in BEACON_HEADERS]
    if diagnostic is not None:
        diagnostic['header_candidates'] = len(candidates)
    if len(candidates) != 1:
        return reject('ambiguous_header' if candidates else 'beacon_header_not_found')
    if not _trusted_box(candidates[0], BEACON_HEADER_REGION, frame):
        return reject('untrusted_header')
    if diagnostic is not None:
        diagnostic['header_confirmed'] = True
    lines = list(task.ocr(*BEACON_LOCATION_REGION, threshold=MIN_OCR_CONFIDENCE, frame=frame))
    locations = [(box, parse_beacon_location(getattr(box, 'name', None))) for box in lines]
    locations = [(box, location) for box, location in locations if location is not None]
    if diagnostic is not None:
        diagnostic['location_candidates'] = len(locations)
    if len(locations) != 1:
        return reject('ambiguous_location' if locations else 'complete_location_not_found')
    if not _trusted_box(locations[0][0], BEACON_LOCATION_REGION, frame):
        return reject('untrusted_location')
    if diagnostic is not None:
        diagnostic.update(status='matched', reason='beacon_detail_location_read',
                          raw=locations[0][1].raw, parts=list(locations[0][1].parts))
    return locations[0][1]

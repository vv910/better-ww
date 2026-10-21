"""Exact, bounded UI selection for a recorded world-map context.

Selecting labels is not proof that the game opened the intended map. Callers
must still verify the resulting map against a recorded visual bookmark.
"""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from ok import TaskDisabledException


_OPEN_LABELS = ("切换地图", "切換地圖", "Switch Map", "Change Map")
_OPEN_REGION = (.68, .84, .99, .99)
# The opening button and the opened picker's heading are different controls.
# Exclude the heading's icon: it can be recognized as a Chinese character.
_TITLE_REGION = (.572, .035, .705, .095)
_MAP_LIST_REGION = (.555, .12, .71, .85)
_AREA_LIST_REGION = (.76, .12, .96, .84)
_MAX_LEVELS = 4
_MAX_LABEL_LENGTH = 80
_POLL_SECONDS = 3.0
_POLL_INTERVAL = .12
_MIN_OCR_CONFIDENCE = .6


@dataclass(frozen=True)
class RouteMapContext:
    selection_path: tuple[str, ...]

    def __post_init__(self):
        path = self.selection_path
        if not isinstance(path, tuple) or not 1 <= len(path) <= _MAX_LEVELS:
            raise ValueError("selection_path must contain 1 to 4 exact UI labels")
        for label in path:
            if not isinstance(label, str) or not label.strip() or len(label) > _MAX_LABEL_LENGTH:
                raise ValueError("each selection label must be non-empty and at most 80 characters")
            if label != label.strip():
                raise ValueError("selection labels cannot have leading or trailing whitespace")
            if any(ord(char) < 32 or 127 <= ord(char) < 160 for char in label):
                raise ValueError("selection labels cannot contain control characters")

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"selection_path"}:
            raise ValueError("map context must contain only selection_path")
        path = data["selection_path"]
        if not isinstance(path, list):
            raise ValueError("selection_path must be a list of exact UI labels")
        return cls(tuple(path))


def parse_selection_path(text: str) -> tuple[str, ...]:
    """Parse one exact UI label per non-empty line."""
    if not isinstance(text, str):
        raise ValueError("selection path text must be a string")
    labels = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not labels:
        return ()
    return RouteMapContext(labels).selection_path


def menu_choice_selected(frame, box, pane):
    """Positive evidence from the picker row, never from background map text.

    Map rows use a white selection card; grey world cards are not selected.
    Area rows instead use a filled radio dot before the text. Unknown styles
    return False and must not be treated as permission to skip an ancestor.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        return False
    height, width = frame.shape[:2]
    _, cy = box.center()
    text_height = max(1, float(box.height))
    if pane == 'map':
        x0, x1 = int(width * .557), int(width * .705)
        y0, y1 = int(cy - text_height * .8), int(cy + text_height * .8)
    elif pane == 'area':
        x0, x1 = int(box.x - text_height * 2), int(box.x - text_height * .15)
        y0, y1 = int(cy - text_height * .6), int(cy + text_height * .6)
    else:
        raise ValueError('unknown map picker pane')
    crop = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    if crop.size == 0:
        return False
    neutral_white = ((crop.min(axis=2) >= 210) & (np.ptp(crop, axis=2) <= 45))
    if pane == 'map':
        return bool(neutral_white.mean() >= .65)

    # A card's thin white outline is not a selected radio dot. Require a
    # compact filled component with its centre aligned to the text baseline.
    count, _, stats, centers = cv2.connectedComponentsWithStats(neutral_white.astype(np.uint8))
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        if not text_height * .12 <= w <= text_height * .8:
            continue
        if .6 <= w / max(1, h) <= 1.6 and area / max(1, w * h) >= .55:
            if abs(centers[index][1] + max(0, y0) - cy) <= text_height * .35:
                return True
    return False


class RouteMapSelector:
    """Select exact visible labels from an already-open game map UI."""

    def __init__(self, task):
        self.task = task

    @staticmethod
    def _confidence(box):
        try:
            return float(box.confidence)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    @classmethod
    def _exact(cls, box, label):
        confidence = cls._confidence(box)
        return (math.isfinite(confidence) and confidence >= _MIN_OCR_CONFIDENCE
                and isinstance(getattr(box, "name", None), str)
                and box.name.strip() == label)

    def _position_tolerance(self):
        frame = getattr(self.task, "frame", None)
        try:
            height = int(frame.shape[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            height = 1080
        return max(1.0, 12.0 * height / 1080.0)

    @staticmethod
    def _xy(box):
        center = box.center()
        return float(center[0]), float(center[1])

    def _same_place(self, first, second):
        ax, ay = self._xy(first)
        bx, by = self._xy(second)
        tolerance = self._position_tolerance()
        return abs(ax - bx) <= tolerance and abs(ay - by) <= tolerance

    def _ocr(self, region):
        return list(self.task.ocr(*region))

    def _stable_unique(self, label, region, timeout=_POLL_SECONDS, excluded=()):
        """Require exactly one exact OCR hit at a stable location in two frames."""
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() <= deadline:
            boxes = [box for box in self._ocr(region)
                     if self._exact(box, label)
                     and all(not self._same_place(box, old) for old in excluded)]
            if len(boxes) > 1:
                return None
            current = boxes[0] if boxes else None
            if current is not None and previous is not None and self._same_place(current, previous):
                return current
            previous = current
            self.task.next_frame()
            self.task.sleep(_POLL_INTERVAL)
        return None

    def _failure(self, reason, label=None):
        self.task.screenshot(f"3c-map-context-{reason}")
        detail = '' if label is None else f', label={label!r}'
        raise RuntimeError(f"Map context selection failed / 地图选择失败，请核对菜单文字与截图: {reason}{detail}")

    def _has_switch_label(self, region):
        return any(self._exact(box, label) for box in self._ocr(region) for label in _OPEN_LABELS)

    def _menu_open(self):
        return self._has_switch_label(_TITLE_REGION)

    def _wait_state(self, predicate, timeout=_POLL_SECONDS):
        """Require two fresh observations; tolerate brief transitions/OCR misses."""
        deadline = time.monotonic() + timeout
        previous = False
        while time.monotonic() <= deadline:
            current = bool(predicate())
            if current and previous:
                return True
            previous = current
            self.task.next_frame()
            self.task.sleep(_POLL_INTERVAL)
        return False

    def _choices(self, label, include_areas):
        regions = [('map', _MAP_LIST_REGION)]
        if include_areas:
            regions.append(('area', _AREA_LIST_REGION))
        return [(box, pane) for pane, region in regions for box in self._ocr(region)
                if self._exact(box, label)]

    def _selected(self, choice):
        box, pane = choice
        return menu_choice_selected(getattr(self.task, 'frame', None), box, pane)

    def _stable_choice(self, label, include_areas):
        deadline = time.monotonic() + _POLL_SECONDS
        previous = None
        previous_selected = None
        while time.monotonic() <= deadline:
            if self._menu_open():
                choices = self._choices(label, include_areas)
                if len(choices) > 1:
                    self._failure('ambiguous-label', label)
                current = choices[0] if choices else None
                selected = self._selected(current) if current is not None else None
                if (current is not None and previous is not None and current[1] == previous[1]
                        and self._same_place(current[0], previous[0])
                        and selected == previous_selected):
                    return current
                previous = current
                previous_selected = selected
            else:
                previous = None
            self.task.next_frame()
            self.task.sleep(_POLL_INTERVAL)
        self._failure('label-not-uniquely-found', label)

    def _choice_applied(self, label, pane, last):
        if not self._menu_open():
            # A missing title alone can be an OCR miss or a closed game map.
            return last and self._has_switch_label(_OPEN_REGION)
        choices = self._choices(label, include_areas=pane == 'area')
        matches = [choice for choice in choices if choice[1] == pane]
        return len(matches) == 1 and self._selected(matches[0])

    def _close_picker(self):
        # The observed game UI centres the area but keeps the picker open.
        # Dismiss that picker before beacon search, and avoid an extra Esc if
        # it has already closed. The location button exists behind the panel.
        if self._menu_open():
            self.task.send_key('esc')
        if not self._wait_state(lambda: not self._menu_open() and self._has_switch_label(_OPEN_REGION)):
            self._failure('picker-not-closed')

    def _open_picker(self):
        if not self._menu_open():
            button_hits = [box for box in self._ocr(_OPEN_REGION)
                           if any(self._exact(box, label) for label in _OPEN_LABELS)]
            if len(button_hits) > 1:
                self._failure('ambiguous-switch-button')
            if not button_hits:
                self._failure('switch-button-not-found')
            button = self._stable_unique(button_hits[0].name.strip(), _OPEN_REGION)
            if button is None:
                self._failure('unstable-switch-button')
            self.task.click(button)
        if not self._wait_state(self._menu_open):
            self._failure('picker-not-confirmed')

    def _selected_context(self):
        selected_maps = []
        selected_areas = []
        for pane, region, selected in (('map', _MAP_LIST_REGION, selected_maps),
                                        ('area', _AREA_LIST_REGION, selected_areas)):
            for box in self._ocr(region):
                name = getattr(box, 'name', None)
                if isinstance(name, str) and self._exact(box, name.strip()) and self._selected((box, pane)):
                    selected.append(box)
        if not 1 <= len(selected_maps) <= 3 or len(selected_areas) != 1:
            return None
        ordered = sorted(selected_maps, key=lambda box: box.center()[1])
        if any(abs(first.center()[1] - second.center()[1]) <= self._position_tolerance()
               for first, second in zip(ordered, ordered[1:])):
            return None
        labels = tuple(box.name.strip() for box in (*ordered, selected_areas[0]))
        if len(set(labels)) != len(labels):
            return None
        try:
            return RouteMapContext(labels)
        except ValueError:
            return None

    def capture(self):
        """Read only selected menu rows; opening/closing never chooses an area.

        Called after stars and bookmarks have been captured, so UI changes
        cannot alter their pixel coordinates. Missing or ambiguous selection
        stops recording instead of silently saving a route for another map.
        """
        try:
            self._open_picker()
            deadline = time.monotonic() + _POLL_SECONDS
            previous = None
            while time.monotonic() <= deadline:
                current = self._selected_context() if self._menu_open() else None
                if current is not None and current == previous:
                    self._close_picker()
                    return current
                previous = current
                self.task.next_frame()
                self.task.sleep(_POLL_INTERVAL)
            self._failure('selected-context-not-confirmed')
        finally:
            self.task.mouse_up()

    def select(self, context: RouteMapContext):
        if not isinstance(context, RouteMapContext):
            raise TypeError("context must be a RouteMapContext")

        try:
            self._open_picker()

            for index, label in enumerate(context.selection_path):
                last = index == len(context.selection_path) - 1
                choice = self._stable_choice(label, include_areas=last)
                # Re-clicking an expanded parent can collapse its children.
                # Still click a final area to request centring even if its radio
                # is already selected. No unrelated OCR change counts as success.
                if not self._selected(choice) or choice[1] == 'area':
                    self.task.click(choice[0])
                if not self._wait_state(lambda: self._choice_applied(label, choice[1], last)):
                    self._failure('selection-state-not-confirmed', label)
            self._close_picker()
            # This means exact UI labels were selected, not that map identity was verified.
            return context.selection_path
        except TaskDisabledException:
            raise
        finally:
            self.task.mouse_up()

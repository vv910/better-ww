"""Exact, bounded UI selection for a recorded world-map context.

Selecting labels is not proof that the game opened the intended map. Callers
must still verify the resulting map against a recorded visual bookmark.
"""

from dataclasses import dataclass
import math
import time

from ok import TaskDisabledException


_OPEN_LABELS = ("切换地图", "切換地圖", "Switch Map", "Change Map")
_OPEN_REGION = (.68, .84, .99, .99)
_MENU_REGION = (.08, .08, .96, .94)
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

    def _failure(self, reason):
        self.task.screenshot(f"3c-map-context-{reason}")
        raise RuntimeError(f"Map context selection failed / 地图选择失败，请核对菜单文字与截图: {reason}")

    def _ui_signature(self, boxes):
        tolerance = self._position_tolerance()
        return frozenset((box.name.strip(), round(box.center()[0] / tolerance),
                          round(box.center()[1] / tolerance))
                         for box in boxes
                         if isinstance(getattr(box, "name", None), str)
                         and self._confidence(box) >= _MIN_OCR_CONFIDENCE)

    def _wait_for_stable_ui_change(self, before, region, timeout=_POLL_SECONDS):
        """Wait up to three seconds for two consecutive OCR-stable changed states."""
        old_signature = self._ui_signature(before)
        deadline = time.monotonic() + timeout
        previous = None
        while time.monotonic() <= deadline:
            boxes = self._ocr(region)
            current = self._ui_signature(boxes)
            if current and current != old_signature and current == previous:
                return boxes
            previous = current if current and current != old_signature else None
            self.task.next_frame()
            self.task.sleep(_POLL_INTERVAL)
        return None

    def select(self, context: RouteMapContext):
        if not isinstance(context, RouteMapContext):
            raise TypeError("context must be a RouteMapContext")

        try:
            # Baseline prevents clicking a same-named label already visible behind the picker.
            baseline = self._ocr(_MENU_REGION)
            button_hits = [(box, box.name.strip()) for box in self._ocr(_OPEN_REGION)
                           if any(self._exact(box, label) for label in _OPEN_LABELS)]
            if len(button_hits) > 1:
                self._failure("ambiguous-switch-button")
            if not button_hits:
                self._failure("switch-button-not-found")
            _, button_label = button_hits[0]
            button = self._stable_unique(button_label, _OPEN_REGION)
            if button is None:
                self._failure("unstable-switch-button")

            self.task.click(button)
            opened = self._wait_for_stable_ui_change(baseline, _MENU_REGION)
            if opened is None:
                self._failure("picker-not-confirmed")

            for label in context.selection_path:
                # The menu layout is unknown, so this prototype never scrolls or guesses at a list.
                # A target already present behind the picker at the same position is not evidence.
                background_matches = [box for box in baseline if self._exact(box, label)]
                selected = self._stable_unique(label, _MENU_REGION, excluded=background_matches)
                if selected is None:
                    self._failure("label-not-uniquely-found")

                before_click = self._ocr(_MENU_REGION)
                self.task.click(selected)
                if self._wait_for_stable_ui_change(before_click, _MENU_REGION) is None:
                    self._failure("selection-transition-not-confirmed")
            # This means exact UI labels were selected, not that map identity was verified.
            return context.selection_path
        except TaskDisabledException:
            raise
        finally:
            self.task.mouse_up()

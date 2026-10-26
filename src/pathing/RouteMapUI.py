"""Recognize the current world-map controls without relying on terrain markers."""

from pathlib import Path

import cv2
import numpy as np
from ok import Box, get_path_relative_to_exe


class RouteMapUI:
    """Confirm the current map layout using three separate right-side controls.

    The legacy yellow location pin is handled by the caller. This detector needs
    the current recenter icon and both zoom buttons, at a common height-derived
    UI scale and right-anchored positions. It does not interpret map stars or
    the player's world-map arrow as evidence that the map is open.
    """

    MIN_SCORE = 0.88
    FOREGROUND_LEVEL = 160
    REFERENCE_WIDTH = 2560
    REFERENCE_HEIGHT = 1440
    # Template top-left coordinates in the independently supplied 1440p UI.
    CONTROLS = {
        'center': ('map-center-current.png', 2246, 54),
        'zoom_in': ('map-zoom-in-current.png', 2395, 404),
        'zoom_out': ('map-zoom-out-current.png', 2395, 964),
    }

    def __init__(self):
        self.templates = {}
        self.last_diagnostic = {}
        for name, (filename, _, _) in self.CONTROLS.items():
            relative = Path('assets') / 'pathing' / filename
            path = Path(get_path_relative_to_exe(str(relative)))
            if not path.is_file():
                # Source imports also work when a test runner changes cwd.
                path = Path(__file__).resolve().parents[2] / relative
            self.templates[name] = cv2.imread(str(path)) if path.is_file() else None
        self._scaled_templates = {}
        self._zoom_mask = None

    def detect(self, frame):
        self.last_diagnostic = {'layout': 'current_right_controls', 'status': 'rejected', 'scores': {}}
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3
                or min(frame.shape[:2]) < 360):
            self.last_diagnostic['reason'] = 'invalid_frame'
            return None
        if any(template is None for template in self.templates.values()):
            self.last_diagnostic['reason'] = 'missing_template'
            return None
        height, width = frame.shape[:2]
        scale = height / self.REFERENCE_HEIGHT
        self.last_diagnostic['capture_size'] = [width, height]
        self.last_diagnostic['ui_scale'] = round(scale, 4)
        if height not in self._scaled_templates:
            # Retain one capture size, so changing resolution cannot grow a cache
            # indefinitely during repeated route runs.
            self._scaled_templates = {height: {
                name: cv2.resize(template,
                                 (max(1, round(template.shape[1] * scale)),
                                  max(1, round(template.shape[0] * scale))))
                for name, template in self.templates.items()
            }}
            zoom_templates = [self._scaled_templates[height][name] for name in ('zoom_in', 'zoom_out')]
            if zoom_templates[0].shape != zoom_templates[1].shape:
                self._scaled_templates = {}
                self._zoom_mask = None
                self.last_diagnostic['reason'] = 'incompatible_zoom_templates'
                return None
            # Include both symbols' foreground and its immediate edge contrast.
            # The union deliberately includes the vertical plus stroke, which
            # must instead be dark in a minus. A minus-only foreground mask
            # would also accept a plus. Ignore the moving terrain outside this
            # support: a beacon behind a button is not a missing map control.
            foreground = np.zeros(zoom_templates[0].shape[:2], np.uint8)
            for template in zoom_templates:
                foreground |= (cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                               >= self.FOREGROUND_LEVEL).astype(np.uint8)
            padding = max(1, round(scale))
            self._zoom_mask = cv2.dilate(foreground, np.ones((padding * 2 + 1,) * 2, np.uint8)) * 255
        controls = {}
        self.last_diagnostic['raw_scores'] = {}
        self.last_diagnostic['match_modes'] = {}
        horizontal_padding = max(3, round(height * 0.012))
        vertical_padding = max(3, round(height * 0.015))
        for name, (_, original_x, original_y) in self.CONTROLS.items():
            template = self._scaled_templates[height][name]
            th, tw = template.shape[:2]
            expected_x = width - (self.REFERENCE_WIDTH - original_x) * scale
            expected_y = original_y * scale
            left = max(0, round(expected_x) - horizontal_padding)
            top = max(0, round(expected_y) - vertical_padding)
            right = min(width, round(expected_x) + tw + horizontal_padding)
            bottom = min(height, round(expected_y) + th + vertical_padding)
            if right - left < tw or bottom - top < th:
                self.last_diagnostic['reason'] = 'controls_outside_frame'
                return None
            crop = frame[top:bottom, left:right, :3]
            raw_scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, raw_score, _, position = cv2.minMaxLoc(raw_scores)
            raw_score = float(raw_score) if np.isfinite(raw_score) else -1.0
            score = raw_score
            mode = 'color_rectangle'
            if name in ('zoom_in', 'zoom_out'):
                scores = cv2.matchTemplate(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                           cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                                           cv2.TM_CCOEFF_NORMED, mask=self._zoom_mask)
                # Constant/empty patches give undefined normalized correlation;
                # discard them before peak selection, not just after minMaxLoc.
                scores[~np.isfinite(scores)] = -1
                _, score, _, position = cv2.minMaxLoc(scores)
                score = float(score)
                mode = 'zoom_foreground_and_edge'
            self.last_diagnostic['raw_scores'][name] = round(raw_score, 4)
            self.last_diagnostic['match_modes'][name] = mode
            self.last_diagnostic['scores'][name] = round(score, 4)
            controls[name] = Box(left + position[0], top + position[1], tw, th,
                                 confidence=score, name='map_' + name + '_current')
        self.last_diagnostic['controls'] = {
            name: [box.x, box.y, box.width, box.height] for name, box in controls.items()
        }
        for name, control in controls.items():
            if control.confidence < self.MIN_SCORE:
                self.last_diagnostic['reason'] = name + '_not_matched'
                return None
        plus, minus = controls['zoom_in'], controls['zoom_out']
        if (abs(plus.center()[0] - minus.center()[0]) > height * 0.006
                or plus.center()[1] >= minus.center()[1]):
            self.last_diagnostic['reason'] = 'zoom_controls_not_in_same_column'
            return None
        self.last_diagnostic.update(status='matched', reason='three_map_controls_confirmed')
        center = controls['center']
        center.confidence = min(box.confidence for box in controls.values())
        return center

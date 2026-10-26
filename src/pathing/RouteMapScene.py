"""Recognize one confirmed independent map scene from its own visible control."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ok import get_path_relative_to_exe

from src.pathing.RouteMapContext import RouteMapContext
from src.pathing.RouteMarkers import overlap_scores, white_marker_mask


@dataclass(frozen=True)
class RouteMapSceneIdentity:
    name: str
    context: RouteMapContext
    confidence: float


class RouteMapSceneDetector:
    """Detect the confirmed Dreamweave scene cue without starting OCR.

    The mapping comes from the existing real scene/menu fixture pair. The
    player's top-left region label is deliberately ignored. A scene cue is
    evidence for a context candidate, never permission to click a beacon or to
    overwrite an explicit user context. Callers must still resolve conflicts,
    select the scene and register the resulting terrain against the route.
    Unknown scenes and redacted references return None.
    """

    MIN_SCORE = 0.90
    MIN_GLYPH_SCORE = 0.85
    REFERENCE_WIDTH = 2560
    REFERENCE_HEIGHT = 1440
    CUE_NAME = '万相空间'
    SCENE_NAME = '梦枢天罗'
    CONTEXT = RouteMapContext(('瑝珑', '梦州', '梦枢天罗'))

    @classmethod
    def from_beacon_location(cls, location):
        """Map a confirmed location to known menu labels, never its local area."""
        if (getattr(location, 'world', None), getattr(location, 'map_name', None)) == ('瑝珑', cls.SCENE_NAME):
            return RouteMapSceneIdentity(cls.SCENE_NAME, cls.CONTEXT, 1.0)
        return None

    def __init__(self):
        relative = Path('assets/pathing/map-scene-dreamweave-label.png')
        path = Path(get_path_relative_to_exe(str(relative)))
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / relative
        template = cv2.imread(str(path)) if path.is_file() else None
        self.template = None if template is None else white_marker_mask(template)
        self.last_diagnostic = {}

    def detect(self, frame):
        self.last_diagnostic = {'status': 'rejected', 'cue': self.CUE_NAME}

        def reject(reason):
            self.last_diagnostic['reason'] = reason
            return None

        if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3
                or min(frame.shape[:2]) < 360 or frame.dtype != np.uint8):
            return reject('invalid_frame')
        if self.template is None:
            return reject('missing_template')
        height, width = frame.shape[:2]
        scale = height / self.REFERENCE_HEIGHT
        canonical_width = round(width / scale)
        if canonical_width < 360:
            return reject('unsupported_capture_aspect')
        # Normalize the whole capture before cropping. Scaling only a cropped
        # text template introduces a half-pixel phase error at 1080p and can
        # change otherwise identical narrow Chinese character strokes.
        if height == self.REFERENCE_HEIGHT:
            normalized = frame[:, :, :3]
        else:
            normalized = cv2.resize(frame[:, :, :3],
                                    (canonical_width, self.REFERENCE_HEIGHT),
                                    interpolation=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR)
        left, right = canonical_width - 360, canonical_width - 185
        top, bottom = 1060, 1135
        self.last_diagnostic.update(capture_size=[width, height],
                                    canonical_region=[left, top, right, bottom])
        source = white_marker_mask(normalized[top:bottom, left:right])
        scores = overlap_scores(source, self.template)
        if scores is None:
            return reject('cue_region_too_small')
        _, score, _, pixel = cv2.minMaxLoc(scores)
        score = float(score) if np.isfinite(score) else 0.0
        self.last_diagnostic['score'] = round(score, 4)
        if score < self.MIN_SCORE:
            return reject('scene_cue_not_matched')
        th, tw = self.template.shape
        patch = source[pixel[1]:pixel[1] + th, pixel[0]:pixel[0] + tw]
        glyph_scores = []
        for index in range(4):
            x0, x1 = round(index * tw / 4), round((index + 1) * tw / 4)
            observed, expected = patch[:, x0:x1], self.template[:, x0:x1]
            glyph_score = float(2 * (observed * expected).sum()
                                / (observed.sum() + expected.sum() + 1e-6))
            glyph_scores.append(glyph_score)
        self.last_diagnostic['glyph_scores'] = [round(value, 4) for value in glyph_scores]
        # Three correct characters must not compensate for one missing or
        # replaced character. Button borders and the generic rotate icon are
        # absent from the template, so they cannot supply scene evidence.
        if min(glyph_scores) < self.MIN_GLYPH_SCORE:
            return reject('scene_cue_glyph_mismatch')
        self.last_diagnostic.update(status='matched', reason='scene_control_confirmed',
                                    name=self.SCENE_NAME,
                                    selection_path=list(self.CONTEXT.selection_path),
                                    cue_box=[round((left + pixel[0]) * scale),
                                             round((top + pixel[1]) * scale),
                                             round(tw * scale), round(th * scale)])
        return RouteMapSceneIdentity(self.SCENE_NAME, self.CONTEXT, score)

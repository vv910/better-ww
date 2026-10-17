from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerMatch:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    partial: bool = False

    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2


def white_marker_mask(image):
    low = image.min(axis=2)
    high = image.max(axis=2)
    # Ignore the yellow player arrow and blue beacons, while keeping the white pin glyphs.
    return ((low >= 170) & (high.astype(np.int16) - low <= 65)).astype(np.float32)


def overlap_scores(source, template):
    height, width = template.shape
    if height > source.shape[0] or width > source.shape[1]:
        return None
    overlap = cv2.matchTemplate(source, template, cv2.TM_CCORR)
    area = cv2.boxFilter(source, -1, (width, height), normalize=False, anchor=(0, 0),
                         borderType=cv2.BORDER_CONSTANT)[:overlap.shape[0], :overlap.shape[1]]
    return 2 * overlap / (float(template.sum()) + area + 1e-6)


class RouteMarkerDetector:
    """Find only custom gem/star pins; terrain colour and other icon types are excluded.

    Templates are the task's screen-scaled features. Normalize to at most 1920 pixels wide
    and search nearby sizes. The gem's distinct top can survive arrow/beacon overlap,
    but is accepted only with supporting foreground from the lower gem silhouette.
    """

    def __init__(self, diamond_template, star_template):
        self.diamond_template = diamond_template
        self.star_template = star_template
        self.best_scores = {}

    def detect(self, frame, region=None, capture_width=None):
        scale = min(1.0, 1920 / (capture_width or frame.shape[1]))
        left, top, right, bottom = region or (0, 0, frame.shape[1], frame.shape[0])
        image = frame[top:bottom, left:right]
        source = white_marker_mask(cv2.resize(image, None, fx=scale, fy=scale))
        self.best_scores = {}
        diamonds = self._find(source, self.diamond_template, 'diamond', scale, left, top)
        stars = self._find(source, self.star_template, 'star', scale, left, top)
        return diamonds, stars

    def _find(self, source, reference, name, normalization, left, top):
        candidates = []
        for size in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4):
            template = white_marker_mask(cv2.resize(reference, None,
                                                    fx=normalization * size, fy=normalization * size))
            height, width = template.shape
            full = overlap_scores(source, template)
            if full is None:
                continue
            self.best_scores[name] = max(self.best_scores.get(name, 0), float(full.max()))
            self._peaks(full, 0.85, width, height, normalization, left, top, candidates)
            if name == 'diamond':
                cut = round(height * 0.45)
                upper = overlap_scores(source, template[:cut])
                self.best_scores['diamond_top'] = max(self.best_scores.get('diamond_top', 0), float(upper.max()))
                # Avoid interpreting a small white patch or a different map icon as a start.
                upper = upper[:full.shape[0], :full.shape[1]]
                upper[full < 0.45] = 0
                lower = template[cut:]
                support = cv2.matchTemplate(source, lower, cv2.TM_CCORR)[cut:cut + full.shape[0], :full.shape[1]]
                upper[support < float(lower.sum()) * 0.15] = 0
                self._peaks(upper, 0.88, width, height, normalization, left, top, candidates, partial=True)
        accepted = []
        for candidate in sorted(candidates, key=lambda match: match.confidence, reverse=True):
            if all(np.linalg.norm(np.subtract(candidate.center(), other.center())) >
                   max(candidate.width, other.width) * 0.65 for other in accepted):
                accepted.append(candidate)
        return accepted

    @staticmethod
    def _peaks(scores, threshold, width, height, scale, left, top, output, partial=False):
        scores = scores.copy()
        for _ in range(100):
            _, confidence, _, (x, y) = cv2.minMaxLoc(scores)
            if confidence < threshold:
                break
            output.append(MarkerMatch(left + x / scale, top + y / scale, width / scale, height / scale,
                                      confidence, partial))
            scores[max(0, y - height // 2):y + height // 2 + 1,
                   max(0, x - width // 2):x + width // 2 + 1] = 0

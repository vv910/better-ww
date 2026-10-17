from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HeadingEstimate:
    angle: float
    confidence: float
    margin: float

    @property
    def reliable(self):
        return self.confidence >= 0.72 and self.margin >= 0.08


def arrow_mask(image):
    """Keep the player's gold/cream arrow, excluding terrain and blue/white map icons."""
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return ((hue >= 15) & (hue <= 45) & (saturation >= 45) & (value >= 160)).astype(np.uint8)


class MinimapHeading:
    """Fit the arrow silhouette independently of the minimap background and screen resolution.

    Angles are clockwise, with east at zero, like the project's original arrow template.
    A competing direction must score worse, so round yellow icons cannot supply a heading.
    """

    def __init__(self, template):
        # Both the feature template and minimap box scale with the current capture size.
        # Normalize them separately, then allow for arrow UI/glow differences with a scale search.
        size = 58
        self.templates = []
        mask = arrow_mask(cv2.resize(template, (size, size)))
        if np.count_nonzero(mask) < 30:
            raise ValueError('Player arrow template has no usable gold silhouette')
        for angle in range(0, 360, 3):
            for scale in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
                matrix = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), -angle, scale)
                rotated = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)
                rotated = rotated.astype(np.float32)
                self.templates.append((angle, rotated, float(rotated.sum())))

    def estimate(self, minimap):
        if minimap is None or min(minimap.shape[:2]) < 40:
            return None
        normalized = cv2.resize(minimap, (192, 192), interpolation=cv2.INTER_LINEAR)
        # Only search near the player at the minimap centre, never elsewhere on the map.
        source = arrow_mask(normalized[48:144, 48:144]).astype(np.float32)
        if source.sum() < 30:
            return None
        # F1 overlap penalizes both missing arrow pixels and extra gold pixels. Raw RGB
        # correlation was sensitive to whatever terrain happened to lie behind the arrow.
        area = cv2.boxFilter(source, -1, (58, 58), normalize=False, anchor=(0, 0),
                             borderType=cv2.BORDER_CONSTANT)[:39, :39]
        yy, xx = np.ogrid[:39, :39]
        central = (xx - 19) ** 2 + (yy - 19) ** 2 <= 12 ** 2
        scores = np.zeros(360, dtype=np.float32)
        for angle, template, template_area in self.templates:
            overlap = cv2.matchTemplate(source, template, cv2.TM_CCORR)
            score = 2 * overlap / (template_area + area + 1e-6)
            scores[angle] = max(scores[angle], float(np.max(score[central])))
        best = int(np.argmax(scores))
        difference = np.abs((np.arange(360) - best + 180) % 360 - 180)
        competitor = float(scores[difference >= 45].max())
        return HeadingEstimate(best, float(scores[best]), float(scores[best]) - competitor)

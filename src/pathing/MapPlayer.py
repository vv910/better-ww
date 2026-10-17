"""Find the player's gold arrow on a full-screen map using its silhouette."""

import math
from numbers import Real

import cv2
import numpy as np

from src.pathing.MinimapHeading import arrow_mask


_MIN_SCORE = 0.68
_MIN_MARGIN = 0.07


def _validate_region(frame, region):
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame must be a BGR image")
    if not isinstance(region, (tuple, list)) or len(region) != 4:
        raise ValueError("region must be (left, top, right, bottom)")
    values = []
    for value in region:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("region coordinates must be finite numbers")
        if int(value) != value:
            raise ValueError("region coordinates must be integers")
        values.append(int(value))
    left, top, right, bottom = values
    height, width = frame.shape[:2]
    if left < 0 or top < 0 or right > width or bottom > height or right <= left or bottom <= top:
        raise ValueError("region must be a non-empty rectangle inside the frame")
    return left, top, right, bottom


def _rotate_mask(mask, angle, scale):
    height, width = mask.shape
    resized = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    height, width = resized.shape
    radians = math.radians(angle)
    out_width = max(1, int(math.ceil(abs(width * math.cos(radians)) + abs(height * math.sin(radians)))))
    out_height = max(1, int(math.ceil(abs(height * math.cos(radians)) + abs(width * math.sin(radians)))))
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), -angle, 1.0)
    matrix[0, 2] += (out_width - width) / 2
    matrix[1, 2] += (out_height - height) / 2
    rotated = cv2.warpAffine(resized, matrix, (out_width, out_height), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rotated > 0)
    if len(xs) == 0:
        return None
    return rotated[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)


def _score_candidate(source, template, origin_x, origin_y, component_center):
    height, width = template.shape
    if height > source.shape[0] or width > source.shape[1]:
        return None
    overlaps = cv2.matchTemplate(source, template, cv2.TM_CCORR)
    source_area = cv2.boxFilter(source, -1, (width, height), normalize=False, anchor=(0, 0),
                                borderType=cv2.BORDER_CONSTANT)[:overlaps.shape[0], :overlaps.shape[1]]
    scores = 2 * overlaps / (float(template.sum()) + source_area + 1e-6)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    center = origin_x + x + width / 2, origin_y + y + height / 2
    if math.dist(center, component_center) > max(width, height) * 1.35:
        return None
    return float(score), center


def _component_matches(mask, template_mask, left, top, components):
    template_area = float(template_mask.sum())
    template_height, template_width = template_mask.shape
    matches = []
    height, width = mask.shape

    for x, y, component_width, component_height, area in components:
        ratio = component_width / max(component_height, 1)
        if not 0.42 <= ratio <= 2.4:
            continue
        if area < template_area * 0.16 or area > template_area * 4.0:
            continue
        # Estimate the rendered scale from gold pixels, then search a narrow range
        # to tolerate anti-aliasing, UI scaling and the arrow's outer glow.
        base_scale = math.sqrt(area / template_area)
        scales = sorted({round(min(2.0, max(0.28, base_scale * factor)), 3)
                         for factor in (0.78, 0.9, 1.0, 1.12, 1.28)})
        component_center = x + component_width / 2, y + component_height / 2
        for scale in scales:
            largest = int(math.ceil(max(template_width, template_height) * scale * 1.5))
            pad = max(12, largest)
            patch_left = max(0, x - pad)
            patch_top = max(0, y - pad)
            patch_right = min(width, x + component_width + pad)
            patch_bottom = min(height, y + component_height + pad)
            patch = mask[patch_top:patch_bottom, patch_left:patch_right].astype(np.float32)
            for angle in range(0, 360, 15):
                rotated = _rotate_mask(template_mask, angle, scale)
                if rotated is None:
                    continue
                result = _score_candidate(patch, rotated, patch_left, patch_top, component_center)
                if result is not None:
                    score, center = result
                    matches.append((score, left + center[0], top + center[1]))
    return matches


def find_map_player(frame, template, region, expected_center=None):
    """Return the detected player's map coordinate, or ``None`` when uncertain.

    ``region`` is a strict in-frame ``(left, top, right, bottom)`` rectangle. The
    optional ``expected_center`` only narrows the search; a matching arrow silhouette
    is still required before any coordinate is returned.
    """
    left, top, right, bottom = _validate_region(frame, region)
    if template is None or not isinstance(template, np.ndarray) or template.ndim != 3 or template.shape[2] < 3:
        return None
    if expected_center is not None:
        if not isinstance(expected_center, (tuple, list)) or len(expected_center) != 2:
            raise ValueError("expected_center must be an (x, y) pair")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in expected_center):
            raise ValueError("expected_center coordinates must be finite numbers")
        expected_x, expected_y = map(float, expected_center)
        if not (left <= expected_x < right and top <= expected_y < bottom):
            raise ValueError("expected_center must lie inside region")
    else:
        expected_x = expected_y = None

    template_gold = arrow_mask(template)
    ys, xs = np.where(template_gold > 0)
    if len(xs) < 30:
        return None
    template_gold = template_gold[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    map_mask = arrow_mask(frame[top:bottom, left:right])
    # A small close repairs thin anti-aliased gaps without making round markers look
    # like arrows; blue beacons and white icons never enter this gold-only mask.
    map_mask = cv2.morphologyEx(map_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(map_mask, connectivity=8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if width < template_gold.shape[1] * 0.22 or height < template_gold.shape[0] * 0.22:
            continue
        if expected_center is not None:
            center_x, center_y = left + centers[index][0], top + centers[index][1]
            radius = max(85.0, min(right - left, bottom - top) * 0.08)
            if math.dist((center_x, center_y), (expected_x, expected_y)) > radius:
                continue
        components.append((x, y, width, height, area))

    if not components:
        return None
    matches = _component_matches(map_mask, template_gold, left, top, components)
    if not matches:
        return None

    # Keep one best score for each physical candidate, merging matches produced by
    # neighboring scales/orientations of the same arrow.
    matches.sort(reverse=True)
    distinct = []
    for score, x, y in matches:
        if all(math.dist((x, y), (other_x, other_y)) > max(template_gold.shape) * 0.45
               for _, other_x, other_y in distinct):
            distinct.append((score, x, y))
    if not distinct or distinct[0][0] < _MIN_SCORE:
        return None
    if len(distinct) > 1 and distinct[0][0] - distinct[1][0] < _MIN_MARGIN:
        return None
    return round(distinct[0][1]), round(distinct[0][2])

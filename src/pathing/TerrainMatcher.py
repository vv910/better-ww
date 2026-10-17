"""Strict terrain-image verification for a geometrically plausible SIFT proposal."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.pathing.TerrainMask import terrain_feature_mask


@dataclass(frozen=True)
class TerrainMatch:
    transform: np.ndarray
    xy: tuple[float, float]
    score: float
    margin: float
    runner_up: float
    sector_scores: tuple[float, ...]


class TerrainMatcher:
    """Verify a weak SIFT proposal against distributed terrain appearance.

    SIFT supplies the scale, orientation and approximate location. This verifier
    keeps those geometric constraints and checks the whole minimap against the
    reference ROI using masked normalized correlation, a unique global peak and
    agreement across spatial sectors. It does not search arbitrary scales.
    """

    MIN_SCORE = 0.32
    MIN_PEAK_MARGIN = 0.045
    MIN_SECTOR_SCORE = 0.10

    def __init__(self):
        self.last_diagnostic = {}

    @staticmethod
    def _gray(image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def _masked_correlation_map(image, template, mask):
        """Compute zero-mean normalized correlation with a binary template mask."""
        mask = (mask > 0).astype(np.float32)
        count = float(mask.sum())
        if count < 64:
            return None
        template = template.astype(np.float32)
        image = image.astype(np.float32)
        masked_template = template * mask
        template_sum = float(masked_template.sum())
        template_energy = float((template * template * mask).sum())
        template_variance = template_energy - template_sum * template_sum / count
        if template_variance <= 1e-4:
            return None

        cross = cv2.matchTemplate(image, masked_template, cv2.TM_CCORR)
        image_sum = cv2.matchTemplate(image, mask, cv2.TM_CCORR)
        image_energy = cv2.matchTemplate(image * image, mask, cv2.TM_CCORR)
        covariance = cross - image_sum * template_sum / count
        image_variance = np.maximum(0, image_energy - image_sum * image_sum / count)
        denominator = np.sqrt(image_variance * template_variance)
        result = np.full(covariance.shape, -1, dtype=np.float32)
        np.divide(covariance, denominator, out=result, where=denominator > 1e-5)
        return np.clip(result, -1, 1)

    def verify(self, reference, query, query_mask, region, transform):
        """Return a refined map transform only when the proposal is unambiguous."""
        self.last_diagnostic = {'status': 'rejected'}
        query_image = query
        reference = self._gray(reference)
        query = self._gray(query)
        height, width = query.shape
        try:
            values = tuple(region)
        except TypeError:
            return self._reject('invalid_reference_region')
        if (len(values) != 4 or any(not isinstance(value, (int, np.integer))
                                    or isinstance(value, (bool, np.bool_)) for value in values)):
            return self._reject('invalid_reference_region')
        left, top, right, bottom = map(int, values)
        map_height, map_width = reference.shape
        if not (0 <= left < right <= map_width and 0 <= top < bottom <= map_height):
            return self._reject('invalid_reference_region')
        if query_mask.shape != query.shape:
            return self._reject('query_mask_size_mismatch')
        query_mask = cv2.bitwise_and(query_mask.astype(np.uint8), terrain_feature_mask(query_image))
        affine = np.asarray(transform, dtype=np.float64)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            return self._reject('invalid_sift_transform')
        scale = math.hypot(affine[0, 0], affine[1, 0])
        rotation = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
        self.last_diagnostic.update(scale=round(scale, 4), rotation=round(rotation, 3))
        if not 0.2 <= scale <= 8:
            return self._reject('scale_out_of_range')
        if abs(rotation) > 12:
            return self._reject('minimap_not_north_up')

        linear = affine[:, :2]
        corners = np.float64([[0, 0], [width, 0], [0, height], [width, height]]) @ linear.T
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        template_width, template_height = np.ceil(maximum - minimum).astype(int)
        if template_width < 24 or template_height < 24:
            return self._reject('template_too_small')
        if template_width > right - left or template_height > bottom - top:
            return self._reject('template_outside_reference_region')
        warp = np.column_stack((linear, -minimum))
        query_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(query)
        reference_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            reference[top:bottom, left:right])
        template = cv2.warpAffine(query_enhanced, warp, (template_width, template_height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        full_mask = cv2.warpAffine((query_mask > 0).astype(np.uint8) * 255, warp,
                                   (template_width, template_height), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT)
        expected_warped_area = np.count_nonzero(query_mask) * abs(float(np.linalg.det(linear)))
        if expected_warped_area < 64 or np.count_nonzero(full_mask) < 0.45 * expected_warped_area:
            return self._reject('insufficient_terrain_mask')

        scores = self._masked_correlation_map(reference_enhanced, template, full_mask)
        if scores is None:
            return self._reject('insufficient_template_texture')
        _, peak, _, peak_xy = cv2.minMaxLoc(scores)
        peak_x, peak_y = peak_xy
        # Suppress overlapping placements before measuring a competing location.
        separated = scores.copy()
        rx, ry = max(1, template_width // 2), max(1, template_height // 2)
        separated[max(0, peak_y - ry):min(scores.shape[0], peak_y + ry + 1),
                  max(0, peak_x - rx):min(scores.shape[1], peak_x + rx + 1)] = -1
        runner_up = float(separated.max()) if separated.size else -1.0
        margin = float(peak - runner_up)

        query_center = np.array([width / 2, height / 2, 1.0])
        proposed_center = affine @ query_center
        template_center = linear @ query_center[:2] - minimum
        proposed_top_left = proposed_center - np.array([left, top]) - template_center
        proposal_error = math.dist(proposed_top_left, peak_xy)
        self.last_diagnostic.update(score=round(float(peak), 4), runner_up=round(runner_up, 4),
                                    margin=round(margin, 4), proposal_error=round(proposal_error, 2),
                                    peak=[int(peak_x + left), int(peak_y + top)])
        if peak < self.MIN_SCORE:
            return self._reject('terrain_score_too_low')
        if margin < self.MIN_PEAK_MARGIN:
            return self._reject('terrain_peak_ambiguous')
        if proposal_error > max(6.0, min(template_width, template_height) * 0.10):
            return self._reject('terrain_peak_disagrees_with_sift')

        # Verify that more than one small patch supports the match. The central
        # arrow/camera cone is already masked by MapLocalizer; quadrant checks keep
        # a beacon or star icon from dominating the correlation score.
        quadrant_scores = []
        textured = 0
        for y0, y1 in ((0, height // 2), (height // 2, height)):
            for x0, x1 in ((0, width // 2), (width // 2, width)):
                quadrant = np.zeros((height, width), np.uint8)
                quadrant[y0:y1, x0:x1] = query_mask[y0:y1, x0:x1]
                sector_mask = cv2.warpAffine((quadrant > 0).astype(np.uint8) * 255, warp,
                                             (template_width, template_height),
                                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                sector_mask = sector_mask > 0
                if sector_mask.sum() < 64:
                    quadrant_scores.append(None)
                    continue
                q_values = template[sector_mask].astype(np.float32)
                r_patch = reference_enhanced[peak_y:peak_y + template_height,
                                             peak_x:peak_x + template_width][sector_mask].astype(np.float32)
                if q_values.std() < 5 or r_patch.std() < 5:
                    quadrant_scores.append(None)
                    continue
                textured += 1
                score = float(np.corrcoef(q_values, r_patch)[0, 1])
                quadrant_scores.append(score)
        supported = sum(score is not None and score >= self.MIN_SECTOR_SCORE
                        for score in quadrant_scores)
        self.last_diagnostic.update(sector_scores=[None if score is None else round(score, 3)
                                                    for score in quadrant_scores],
                                    textured_sectors=textured, supported_sectors=supported)
        if textured < 3:
            return self._reject('terrain_too_concentrated')
        if supported < 3:
            return self._reject('terrain_sector_mismatch')

        refined_center = np.array([peak_x + left, peak_y + top], dtype=np.float64) + template_center
        refined_transform = affine.copy()
        refined_transform[:, 2] = refined_center - linear @ query_center[:2]
        self.last_diagnostic.update(status='matched', reason='terrain_verified')
        return TerrainMatch(refined_transform.astype(np.float32),
                            (float(refined_center[0]), float(refined_center[1])),
                            float(peak), margin, runner_up,
                            tuple(float(score) if score is not None else float('nan')
                                  for score in quadrant_scores))

    def _reject(self, reason):
        self.last_diagnostic.update(status='rejected', reason=reason)
        return None

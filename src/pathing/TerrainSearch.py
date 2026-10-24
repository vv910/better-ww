"""Find independent minimap evidence near a freshly verified big-map position."""

import math
import time

import cv2
import numpy as np

from src.pathing.TerrainMask import terrain_feature_mask
from src.pathing.TerrainMatcher import TerrainMatcher


class TerrainSearch:
    """Bounded scale search used only after trusted big-map relocalization.

    The caller owns hint provenance and expiry. The hint limits where to look;
    it is never returned as a position. A Laplacian match must be distinctive
    across scales and then pass independent full-map appearance verification.
    Small coarse templates are deliberately unsupported: they make unrelated
    terrain fragments look convincing. This is not a world-map locator.
    """

    HINT_RADIUS = 15.0
    MIN_SCORE = 0.35
    MIN_MARGIN = 0.08
    MIN_TEMPLATE_SIZE = 48

    def __init__(self, reference, region):
        gray = TerrainMatcher._gray(reference)
        height, width = gray.shape
        try:
            values = tuple(region)
        except TypeError:
            raise ValueError('Terrain search region must contain four integer coordinates') from None
        if (len(values) != 4 or any(not isinstance(value, (int, np.integer))
                                    or isinstance(value, (bool, np.bool_)) for value in values)):
            raise ValueError('Terrain search region must contain four integer coordinates')
        left, top, right, bottom = map(int, values)
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError('Terrain search region must lie inside the reference image')
        self.reference = reference.copy()
        self.region = left, top, right, bottom
        roi = gray[top:bottom, left:right]
        self.ratio = min(1.0, 512.0 / max(roi.shape))
        reduced = cv2.resize(roi, None, fx=self.ratio, fy=self.ratio,
                             interpolation=cv2.INTER_AREA)
        self.laplacian = self._laplacian(reduced)
        self.verifier = TerrainMatcher()
        self.last_diagnostic = {}

    @staticmethod
    def _laplacian(image):
        return cv2.Laplacian(cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1),
                             cv2.CV_32F)

    def search(self, query, mask, hint):
        started = time.monotonic()
        self.last_diagnostic = {'algorithm': 'hinted_multiscale_laplacian', 'status': 'rejected'}

        def reject(reason):
            self.last_diagnostic.update(reason=reason,
                                        elapsed_ms=round((time.monotonic() - started) * 1000, 2))
            return None

        if hint is None:
            return reject('trusted_hint_required')
        try:
            hint_x, hint_y = map(float, hint)
        except (TypeError, ValueError):
            return reject('invalid_hint')
        left, top, right, bottom = self.region
        if (not math.isfinite(hint_x) or not math.isfinite(hint_y)
                or not (left <= hint_x < right and top <= hint_y < bottom)):
            return reject('hint_outside_reference_region')
        gray = TerrainMatcher._gray(query)
        height, width = gray.shape
        if min(height, width) < 40 or mask.shape != gray.shape:
            return reject('invalid_query_or_mask_size')
        mask = cv2.bitwise_and((mask > 0).astype(np.uint8) * 255,
                               terrain_feature_mask(query))
        usable = mask > 0
        if np.count_nonzero(usable) < max(256, gray.size * 0.10):
            return reject('insufficient_terrain_mask')
        sectors = [usable[y0:y1, x0:x1]
                   for y0, y1 in ((0, height // 2), (height // 2, height))
                   for x0, x1 in ((0, width // 2), (width // 2, width))]
        if sum(np.count_nonzero(sector) >= 64 for sector in sectors) < 3:
            return reject('terrain_mask_too_concentrated')
        if float(gray[usable].std()) < 3:
            return reject('insufficient_query_texture')

        ratio = self.ratio
        cx, cy = (hint_x - left) * ratio, (hint_y - top) * ratio
        radius = self.HINT_RADIUS * ratio
        candidates = []
        tested_sizes = set()

        def evaluate(scale):
            tw, th = round(width * scale * ratio), round(height * scale * ratio)
            if ((tw, th) in tested_sizes or min(tw, th) < self.MIN_TEMPLATE_SIZE
                    or tw > self.laplacian.shape[1] or th > self.laplacian.shape[0]):
                return
            tested_sizes.add((tw, th))
            x0, y0 = max(0, int(cx - tw / 2 - radius)), max(0, int(cy - th / 2 - radius))
            x1 = min(self.laplacian.shape[1], math.ceil(cx + tw / 2 + radius))
            y1 = min(self.laplacian.shape[0], math.ceil(cy + th / 2 + radius))
            if x1 - x0 < tw or y1 - y0 < th:
                return
            template = self._laplacian(cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA))
            template_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            scores = TerrainMatcher._masked_correlation_map(
                self.laplacian[y0:y1, x0:x1], template, template_mask)
            if scores is None:
                return
            yy, xx = np.ogrid[:scores.shape[0], :scores.shape[1]]
            inside = (xx + x0 + tw / 2 - cx) ** 2 + (yy + y0 + th / 2 - cy) ** 2 <= radius ** 2
            scores[~inside] = -1
            # Keep a second spatial candidate too; scale uniqueness alone cannot
            # distinguish repeated nearby terrain.
            for _ in range(2):
                _, score, _, pixel = cv2.minMaxLoc(scores)
                if score <= -1:
                    break
                center = ((pixel[0] + x0 + tw / 2) / ratio + left,
                          (pixel[1] + y0 + th / 2) / ratio + top)
                candidates.append((float(score), float(scale), center))
                rx, ry = max(1, round(6 * ratio)), max(1, round(6 * ratio))
                scores[max(0, pixel[1] - ry):pixel[1] + ry + 1,
                       max(0, pixel[0] - rx):pixel[0] + rx + 1] = -1

        for scale in np.geomspace(0.2, 8.0, 61):
            evaluate(float(scale))
        if not candidates:
            return reject('no_supported_template_scale')
        candidates.sort(reverse=True)
        fine_scales = []
        for _, scale, _ in candidates:
            if not any(abs(math.log(scale / old)) < 0.02 for old in fine_scales):
                fine_scales.append(scale)
            if len(fine_scales) == 5:
                break
        for scale in fine_scales:
            for fine in np.linspace(scale * 0.97, scale * 1.03, 7):
                evaluate(float(fine))
        candidates.sort(reverse=True)
        score, scale, center = candidates[0]
        competing = [candidate[0] for candidate in candidates
                     if math.dist(candidate[2], center) > 10
                     or max(candidate[1] / scale, scale / candidate[1]) > 1.15]
        runner_up = max(competing, default=-1.0)
        margin = score - runner_up
        self.last_diagnostic.update(score=round(score, 4), runner_up=round(runner_up, 4),
                                    margin=round(margin, 4), scale=round(scale, 4),
                                    candidate_position=center, tested_scales=len(tested_sizes))
        if score < self.MIN_SCORE:
            return reject('laplacian_score_too_low')
        if margin < self.MIN_MARGIN:
            return reject('laplacian_peak_ambiguous')
        transform = np.float64([[scale, 0, center[0] - scale * width / 2],
                               [0, scale, center[1] - scale * height / 2]])
        verified = self.verifier.verify(self.reference, query, mask, self.region, transform)
        self.last_diagnostic['verification'] = dict(self.verifier.last_diagnostic)
        if verified is None:
            return reject('global_terrain_verification_failed')
        if math.dist(verified.xy, (hint_x, hint_y)) > self.HINT_RADIUS:
            return reject('verified_position_outside_hint')
        self.last_diagnostic.update(status='matched', reason='hinted_terrain_verified',
                                    position=verified.xy,
                                    elapsed_ms=round((time.monotonic() - started) * 1000, 2))
        return verified

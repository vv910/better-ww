import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MapPosition:
    x: float
    y: float
    confidence: float
    inliers: int

    @property
    def xy(self):
        return self.x, self.y


class MapLocalizer:
    """Match the north-up minimap against one local map snapshot, without game memory."""

    def __init__(self, image, region=None, min_inliers=6, max_jump=100):
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        self.height, self.width = self.gray.shape
        self.region = region or (0, 0, self.width, self.height)
        mask = np.zeros_like(self.gray)
        x1, y1, x2, y2 = self.region
        mask[y1:y2, x1:x2] = 255
        self.sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.gray, mask)
        if self.descriptors is None or len(self.keypoints) < min_inliers:
            raise ValueError('Route map has too few visible terrain features')
        self.locations = np.float32([keypoint.pt for keypoint in self.keypoints])
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers = min_inliers
        self.max_jump = max_jump
        self.previous = None
        self.last_diagnostic = {}

    def reset(self):
        self.previous = None
        self.last_diagnostic = {}

    def _reject(self, reason):
        self.last_diagnostic.update(status='rejected', reason=reason)
        attempts = self.last_diagnostic.get('attempts', [])
        if attempts:
            attempts[-1]['reason'] = reason
        return None

    def locate(self, minimap):
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY) if minimap.ndim == 3 else minimap
        height, width = gray.shape
        self.last_diagnostic = {'mode': 'minimap', 'minimap_size': [width, height],
                                'map_features': len(self.locations), 'attempts': []}
        if min(height, width) < 40:
            return self._reject('minimap_too_small')
        mask = np.zeros_like(gray)
        center = (width // 2, height // 2)
        radius = int(min(width, height) * 0.46)
        cv2.circle(mask, center, radius, 255, -1)
        cv2.circle(mask, center, int(radius * 0.23), 0, -1)  # player arrow/camera cone
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        self.last_diagnostic['minimap_features'] = len(keypoints)
        if descriptors is None or len(keypoints) < self.min_inliers:
            return self._reject('too_few_minimap_features')
        candidates = np.arange(len(self.locations))
        result = None
        if self.previous is not None:
            distance = np.linalg.norm(self.locations - self.previous.xy, axis=1)
            local = candidates[distance < max(250, self.max_jump * 3)]
            if len(local) >= self.min_inliers:
                result = self._match(keypoints, descriptors, local, center, min(width, height))
        if result is None:
            result = self._match(keypoints, descriptors, candidates, center, min(width, height))
        if result is not None and self.previous is not None:
            if math.dist(result.xy, self.previous.xy) > self.max_jump:
                self.last_diagnostic.update(previous_position=self.previous.xy, candidate_position=result.xy,
                                            jump=round(math.dist(result.xy, self.previous.xy), 2))
                return self._reject('position_jump')
        if result is not None:
            self.previous = result
            self.last_diagnostic.update(status='matched', reason='matched', position=result.xy,
                                        inliers=result.inliers, confidence=round(result.confidence, 3))
        return result

    def _match(self, keypoints, descriptors, candidates, center, size):
        matched = self._transform(keypoints, descriptors, candidates, size)
        if matched is None:
            return None
        transform, confidence, inliers = matched
        x, y = transform @ np.array([*center, 1])
        x1, y1, x2, y2 = self.region
        if not (x1 <= x < x2 and y1 <= y < y2):
            self.last_diagnostic['candidate_position'] = [float(x), float(y)]
            return self._reject('position_outside_recorded_map')
        return MapPosition(float(x), float(y), confidence, inliers)

    def project_map_point(self, frame, point, screen_region):
        """Project a recorded beacon into the currently opened map, without a world atlas."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_diagnostic = {'mode': 'map_projection', 'attempts': []}
        mask = np.zeros_like(gray)
        left, top, right, bottom = screen_region
        mask[top:bottom, left:right] = 255
        keypoints, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None:
            return self._reject('too_few_map_features')
        matched = self._transform(keypoints, descriptors, np.arange(len(self.locations)),
                                  min(right - left, bottom - top))
        if matched is None:
            return None
        transform, confidence, inliers = matched
        if inliers < 12 or confidence < 0.6:
            return self._reject('weak_map_projection')
        x, y = cv2.invertAffineTransform(transform) @ np.array([*point, 1])
        if not (left <= x < right and top <= y < bottom):
            return self._reject('projected_beacon_outside_view')
        self.last_diagnostic.update(status='matched', reason='matched')
        return MapPosition(float(x), float(y), confidence, inliers)

    def _transform(self, keypoints, descriptors, candidates, size):
        diagnostic = {'candidate_features': len(candidates)}
        self.last_diagnostic.setdefault('attempts', []).append(diagnostic)
        pairs = self.matcher.knnMatch(descriptors, self.descriptors[candidates], k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        # Multiple query descriptors matching one landmark are not independent evidence.
        unique = {}
        for match in matches:
            if match.trainIdx not in unique or match.distance < unique[match.trainIdx].distance:
                unique[match.trainIdx] = match
        matches = list(unique.values())
        diagnostic['matches'] = len(matches)
        if len(matches) < self.min_inliers:
            return self._reject('too_few_matches')
        source = np.float32([keypoints[match.queryIdx].pt for match in matches])
        destination = self.locations[[candidates[match.trainIdx] for match in matches]]
        transform, inlier_mask = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                                          ransacReprojThreshold=3)
        if transform is None or inlier_mask is None:
            return self._reject('no_consistent_transform')
        accepted = inlier_mask.ravel().astype(bool)
        inliers = int(accepted.sum())
        confidence = inliers / len(matches)
        diagnostic.update(inliers=inliers, confidence=round(confidence, 3))
        if inliers < self.min_inliers or confidence < 0.5:
            return self._reject('too_few_geometric_inliers')
        # Reject icons or a tiny patch being mistaken for the whole minimap.
        spread = np.ptp(source[accepted], axis=0)
        diagnostic['spread'] = [round(float(value), 2) for value in spread]
        if min(spread) < size * 0.22:
            return self._reject('features_too_concentrated')
        scale = math.hypot(transform[0, 0], transform[1, 0])
        rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        diagnostic.update(scale=round(scale, 3), rotation=round(rotation, 2))
        if not 0.2 <= scale <= 8:
            return self._reject('scale_out_of_range')
        if abs(rotation) > 12:
            return self._reject('minimap_not_north_up')
        diagnostic['reason'] = 'matched'
        return transform, confidence, inliers

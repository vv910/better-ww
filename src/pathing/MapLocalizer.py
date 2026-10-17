import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.pathing.MinimapTracker import MinimapTracker
from src.pathing.TerrainMask import terrain_feature_mask
from src.pathing.TerrainMatcher import TerrainMatcher


@dataclass(frozen=True)
class MapPosition:
    x: float
    y: float
    confidence: float
    inliers: int
    source: str = 'sift'

    @property
    def xy(self):
        return self.x, self.y


class MapLocalizer:
    """Match the north-up minimap against one local map snapshot, without game memory."""

    def __init__(self, image, region=None, min_inliers=6, max_jump=100):
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        self.height, self.width = self.gray.shape
        self.region = region or (0, 0, self.width, self.height)
        self.sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
        self.keypoints, self.descriptors = self._region_features(self.gray, self.region, source_image=image)
        if self.descriptors is None or len(self.keypoints) < min_inliers:
            raise ValueError('Route map has too few visible terrain features')
        self.locations = np.float32([keypoint.pt for keypoint in self.keypoints])
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.terrain_matcher = TerrainMatcher()
        self.min_inliers = min_inliers
        self.max_jump = max_jump
        self.tracker = MinimapTracker()
        self.reset()

    def _region_features(self, gray, region, mask=None, source_image=None):
        # SIFT applies its feature quota BEFORE its mask. Detect on the ROI itself so
        # bright UI/out-of-route terrain cannot consume the budget, then restore coordinates.
        try:
            values = tuple(region)
        except TypeError:
            raise ValueError('Map feature region must contain four pixel coordinates') from None
        if (len(values) != 4 or any(not isinstance(value, (int, np.integer))
                                    or isinstance(value, (bool, np.bool_)) for value in values)):
            raise ValueError('Map feature region must contain four integer pixel coordinates')
        left, top, right, bottom = map(int, values)
        height, width = gray.shape
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError('Map feature region must be a non-empty rectangle inside the image')
        roi_mask = None
        if source_image is not None:
            if source_image.shape[:2] != gray.shape:
                raise ValueError('Map feature source image must match the grayscale image dimensions')
            roi_mask = terrain_feature_mask(source_image[top:bottom, left:right])
        elif mask is not None:
            if mask.shape == gray.shape:
                roi_mask = mask[top:bottom, left:right]
            elif mask.shape == (bottom - top, right - left):
                roi_mask = mask
            else:
                raise ValueError('Map feature mask must match the source image dimensions')
        keypoints, descriptors = self.sift.detectAndCompute(gray[top:bottom, left:right], roi_mask)
        for keypoint in keypoints:
            x, y = keypoint.pt
            keypoint.pt = x + left, y + top
        return keypoints, descriptors

    def reset(self, position_hint=None):
        """Clear visual tracking, optionally keeping a map-screen position hint.

        The hint only limits/validates the next absolute minimap match; it is never
        returned as a minimap localization result on its own.
        """
        if position_hint is None:
            self.previous = None
        elif isinstance(position_hint, MapPosition):
            if position_hint.source != 'map':
                raise ValueError('MapPosition hints must come from locate_map_player')
            x, y = position_hint.xy
            confidence, inliers = position_hint.confidence, position_hint.inliers
        else:
            try:
                x, y = position_hint
            except (TypeError, ValueError):
                raise ValueError('position_hint must be an (x, y) pair or MapPosition') from None
            confidence, inliers = 1.0, 0
        if position_hint is not None:
            x, y = float(x), float(y)
            x1, y1, x2, y2 = self.region
            if not math.isfinite(x) or not math.isfinite(y) or not (x1 <= x < x2 and y1 <= y < y2):
                raise ValueError('position_hint must be inside the recorded map region')
            self.previous = MapPosition(x, y, confidence, inliers, source='hint')
        self.last_diagnostic = {}
        self.tracker.reset()
        self._last_sift_at = -math.inf
        self._candidate_transform = None
        self._terrain_proposal = None

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
        mask = cv2.bitwise_and(mask, terrain_feature_mask(minimap))
        now = time.monotonic()
        tracked = self.tracker.track(gray, mask, now)
        self.last_diagnostic['tracking'] = dict(self.tracker.last_diagnostic)
        if tracked is not None:
            x1, y1, x2, y2 = self.region
            if not (x1 <= tracked.xy[0] < x2 and y1 <= tracked.xy[1] < y2):
                self.last_diagnostic['tracking']['reason'] = 'position_outside_recorded_map'
                tracked = None
        if tracked is not None and now - self._last_sift_at < 0.5:
            return self._accept_tracking(gray, mask, tracked, now)
        self._last_sift_at = now
        self._candidate_transform = None
        result = self._locate_absolute(gray, mask, center)
        if result is not None and tracked is not None and math.dist(result.xy, tracked.xy) > 12:
            # Conflicting visual evidence is not a temporary lack of landmarks.
            self.tracker.reset()
            return self._reject('absolute_disagrees_with_tracking')
        if result is not None:
            self.previous = result
            self.tracker.anchor(gray, mask, self._candidate_transform, result.xy, now)
            self.last_diagnostic.update(status='matched', reason='matched', method=result.source, position=result.xy,
                                        inliers=result.inliers, confidence=round(result.confidence, 3))
            return result
        if tracked is not None:
            if self.last_diagnostic.get('reason') in ('position_jump', 'minimap_not_north_up',
                                                      'scale_out_of_range', 'position_outside_recorded_map'):
                self.tracker.reset()
                return None
            self.last_diagnostic['absolute_reason'] = self.last_diagnostic.get('reason')
            return self._accept_tracking(gray, mask, tracked, now)
        return None

    def _accept_tracking(self, gray, mask, tracked, now):
        result = MapPosition(*tracked.xy, tracked.confidence, tracked.inliers, source='flow')
        self.tracker.commit(gray, mask, tracked, now)
        self.previous = result
        self.last_diagnostic.update(status='matched', reason='tracked', method='flow', position=result.xy,
                                    inliers=result.inliers, confidence=round(result.confidence, 3))
        return result

    def _locate_absolute(self, gray, mask, center):
        height, width = gray.shape
        self._terrain_proposal = None
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
        if result is None and self._terrain_proposal is not None:
            proposal = self._terrain_proposal
            verified = self.terrain_matcher.verify(self.gray, gray, mask, self.region,
                                                    proposal['transform'])
            self.last_diagnostic['terrain_verification'] = dict(self.terrain_matcher.last_diagnostic)
            if verified is not None:
                if self.previous is not None and math.dist(verified.xy, self.previous.xy) > self.max_jump:
                    self.last_diagnostic.update(previous_position=self.previous.xy,
                                                candidate_position=verified.xy,
                                                jump=round(math.dist(verified.xy, self.previous.xy), 2))
                    return self._reject('position_jump')
                self._candidate_transform = verified.transform
                self.last_diagnostic.update(status='matched', reason='matched', method='terrain',
                                            position=verified.xy, inliers=proposal['inliers'],
                                            confidence=round(proposal['confidence'], 3))
                return MapPosition(verified.xy[0], verified.xy[1], proposal['confidence'],
                                   proposal['inliers'], source='terrain')
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
        self._candidate_transform = transform
        return MapPosition(float(x), float(y), confidence, inliers)

    def project_map_point(self, frame, point, screen_region):
        """Project a recorded beacon into the currently opened map, without a world atlas."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_diagnostic = {'mode': 'map_projection', 'attempts': []}
        left, top, right, bottom = screen_region
        keypoints, descriptors = self._region_features(gray, screen_region, source_image=frame)
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

    def locate_map_player(self, frame, player_xy, screen_region):
        """Map a detected player marker from the open map view to recorded coords."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self.last_diagnostic = {'mode': 'map_player_localization', 'attempts': []}
        try:
            screen_region = tuple(screen_region)
            left, top, right, bottom = screen_region
            player_x, player_y = map(float, player_xy)
        except (TypeError, ValueError):
            return self._reject('invalid_player_or_screen_region')
        height, width = gray.shape
        if (any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                for value in screen_region)
                or not (0 <= left < right <= width and 0 <= top < bottom <= height)):
            return self._reject('invalid_screen_region')
        if not (math.isfinite(player_x) and math.isfinite(player_y)
                and left <= player_x < right and top <= player_y < bottom):
            return self._reject('player_marker_outside_screen_region')
        keypoints, descriptors = self._region_features(gray, screen_region, source_image=frame)
        self.last_diagnostic.update(screen_region=list(screen_region), map_features=len(self.locations),
                                    screen_features=len(keypoints))
        if descriptors is None:
            return self._reject('too_few_map_features')
        matched = self._transform(keypoints, descriptors, np.arange(len(self.locations)),
                                  min(right - left, bottom - top))
        if matched is None:
            return None
        transform, confidence, inliers = matched
        if inliers < 12 or confidence < 0.6:
            return self._reject('weak_map_projection')
        x, y = transform @ np.array([player_x, player_y, 1.0])
        x1, y1, x2, y2 = self.region
        if not (x1 <= x < x2 and y1 <= y < y2):
            self.last_diagnostic['candidate_position'] = [float(x), float(y)]
            return self._reject('player_outside_recorded_map')
        self.last_diagnostic.update(status='matched', reason='matched', method='map',
                                    position=[float(x), float(y)], inliers=inliers,
                                    confidence=round(confidence, 3))
        return MapPosition(float(x), float(y), confidence, inliers, source='map')

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
        scale = math.hypot(transform[0, 0], transform[1, 0])
        rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        diagnostic.update(scale=round(scale, 3), rotation=round(rotation, 2))
        if not 0.2 <= scale <= 8:
            return self._reject('scale_out_of_range')
        if abs(rotation) > 12:
            return self._reject('minimap_not_north_up')
        if min(spread) < size * 0.22:
            # Keep the original SIFT threshold. A separate full-terrain verifier
            # may corroborate this proposal, but weak SIFT alone remains rejected.
            proposal = {'transform': transform.copy(), 'confidence': confidence,
                        'inliers': inliers, 'spread': spread.copy()}
            if (self._terrain_proposal is None
                    or (inliers, confidence, float(np.prod(spread)))
                    > (self._terrain_proposal['inliers'], self._terrain_proposal['confidence'],
                       float(np.prod(self._terrain_proposal['spread'])))):
                self._terrain_proposal = proposal
            return self._reject('features_too_concentrated')
        diagnostic['reason'] = 'matched'
        return transform, confidence, inliers

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackedMinimap:
    transform: np.ndarray
    xy: tuple[float, float]
    confidence: float
    inliers: int


class MinimapTracker:
    """Brief visual odometry anchored to a verified map match; never extrapolate key input."""

    def __init__(self, max_age=3.0, max_distance=60.0):
        self.max_age = max_age
        self.max_distance = max_distance
        self.reset()

    def reset(self):
        self.gray = self.points = self.transform = None
        self.anchor_xy = None
        self.anchor_at = self.previous_at = -math.inf
        self.last_diagnostic = {}

    def anchor(self, gray, mask, transform, xy, now):
        self.anchor_xy = xy
        self.anchor_at = now
        self._remember(gray, mask, np.vstack((transform, [0, 0, 1])), now)

    def _remember(self, gray, mask, transform, now):
        self.gray = gray.copy()
        self.points = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.02,
                                              minDistance=7, mask=mask, blockSize=7)
        self.transform = transform.copy()
        self.previous_at = now

    def commit(self, gray, mask, tracked, now):
        # A relative observation must not renew the absolute-anchor time/distance budget.
        self._remember(gray, mask, tracked.transform, now)

    def track(self, gray, mask, now):
        self.last_diagnostic = {'age': None if self.anchor_xy is None else round(now - self.anchor_at, 3)}

        def reject(reason):
            self.last_diagnostic['reason'] = reason
            return None

        if self.gray is None or self.points is None or len(self.points) < 12:
            return reject('no_tracking_anchor')
        if gray.shape != self.gray.shape:
            return reject('capture_size_changed')
        if now - self.anchor_at > self.max_age or now - self.previous_at > 2.0:
            return reject('tracking_expired')
        # Coarser pyramid levels let the changing player arrow influence the entire small crop.
        options = dict(winSize=(21, 21), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        current, status, error = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        if current is None:
            return reject('no_optical_flow')
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, **options)
        if backward is None:
            return reject('no_backward_flow')
        old = self.points.reshape(-1, 2)
        new = current.reshape(-1, 2)
        height, width = gray.shape
        valid = (status.ravel() == 1) & (back_status.ravel() == 1)
        valid &= np.isfinite(new).all(axis=1) & np.isfinite(backward.reshape(-1, 2)).all(axis=1)
        valid &= (np.linalg.norm(backward.reshape(-1, 2) - old, axis=1) <= 1.0) & (error.ravel() <= 20)
        pixels = np.rint(np.clip(np.nan_to_num(new), -1, max(width, height))).astype(np.int32)
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        valid &= inside
        valid[inside] &= mask[pixels[inside, 1], pixels[inside, 0]] > 0
        self.last_diagnostic.update(corners=len(old), consistent_tracks=int(valid.sum()))
        if valid.sum() < max(12, len(old) * 0.5):
            return reject('too_few_consistent_tracks')
        # Map current pixel coordinates back into the preceding frame, then into the route map.
        affine, accepted = cv2.estimateAffinePartial2D(new[valid], old[valid], method=cv2.RANSAC,
                                                      ransacReprojThreshold=1.5)
        if affine is None or accepted is None:
            return reject('no_tracking_transform')
        inliers = accepted.ravel().astype(bool)
        confidence = float(inliers.mean())
        self.last_diagnostic.update(inliers=int(inliers.sum()), confidence=round(confidence, 3))
        if inliers.sum() < 12 or confidence < 0.75:
            return reject('weak_tracking_transform')
        landmarks = new[valid][inliers]
        if min(np.ptp(landmarks, axis=0)) < min(width, height) * 0.35:
            return reject('tracking_features_concentrated')
        sectors = (landmarks[:, 0] >= width / 2).astype(int) + 2 * (landmarks[:, 1] >= height / 2)
        if len(np.unique(sectors)) < 3:
            return reject('tracking_features_concentrated')
        scale = math.hypot(affine[0, 0], affine[1, 0])
        rotation = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
        if not 0.985 <= scale <= 1.015 or abs(rotation) > 3:
            return reject('tracking_view_changed')
        center = np.array([width // 2, height // 2, 1])
        displacement = np.linalg.norm(affine @ center - center[:2])
        if displacement > min(width, height) * 0.2:
            return reject('tracking_frame_jump')
        transform = self.transform @ np.vstack((affine, [0, 0, 1]))
        x, y, _ = transform @ center
        if math.dist((x, y), self.anchor_xy) > self.max_distance:
            return reject('tracking_distance_exceeded')
        self.last_diagnostic['reason'] = 'tracked'
        return TrackedMinimap(transform, (float(x), float(y)), confidence, int(inliers.sum()))

"""Independent absolute terrain reproof for an already anchored visual track."""

import math

import numpy as np

from src.pathing.MinimapTracker import TrackedMinimap
from src.pathing.TerrainMatcher import TerrainMatcher


class TrackingVerifier:
    """Revalidate an active track without treating optical flow as a new anchor.

    The caller must supply a non-expired track and the last absolute transform.
    The track proposes a location; the original absolute transform supplies the
    scale and orientation. Only a full-map, unique, distributed appearance match
    may become another absolute observation. A failed check never extends flow.
    """

    def __init__(self):
        self.matcher = TerrainMatcher()
        self.last_diagnostic = {}

    @staticmethod
    def _affine(value):
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if matrix.shape == (3, 3):
            if not np.allclose(matrix[2], (0, 0, 1), rtol=0, atol=1e-8):
                return None
            matrix = matrix[:2]
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            return None
        # Both SIFT and LK use estimateAffinePartial2D. Reject shear or unequal
        # axis scales rather than interpreting an arbitrary matrix as that model.
        linear = matrix[:, :2]
        if (not np.isclose(linear[0, 0], linear[1, 1], rtol=1e-4, atol=1e-6)
                or not np.isclose(linear[0, 1], -linear[1, 0], rtol=1e-4, atol=1e-6)):
            return None
        return matrix

    def verify(self, reference, query, mask, region, tracked, anchor_transform):
        self.last_diagnostic = {'algorithm': 'anchored_terrain_reproof', 'status': 'rejected'}

        def reject(reason):
            self.last_diagnostic['reason'] = reason
            return None

        if (not isinstance(tracked, TrackedMinimap)
                or not math.isfinite(tracked.confidence) or tracked.confidence < 0.75
                or tracked.inliers < 12):
            return reject('unreliable_tracking_candidate')
        anchor = self._affine(anchor_transform)
        flow = self._affine(tracked.transform)
        if anchor is None or flow is None:
            return reject('invalid_tracking_or_anchor_transform')
        anchor_scale = math.hypot(anchor[0, 0], anchor[1, 0])
        flow_scale = math.hypot(flow[0, 0], flow[1, 0])
        if not 0.2 <= anchor_scale <= 8:
            return reject('invalid_anchor_scale')
        anchor_rotation = math.degrees(math.atan2(anchor[1, 0], anchor[0, 0]))
        flow_rotation = math.degrees(math.atan2(flow[1, 0], flow[0, 0]))
        relative_rotation = (flow_rotation - anchor_rotation + 180) % 360 - 180
        relative_scale = flow_scale / anchor_scale
        self.last_diagnostic.update(anchor_scale=round(anchor_scale, 4),
                                    relative_scale=round(relative_scale, 5),
                                    relative_rotation=round(relative_rotation, 3))
        # Reuse the tracker's view-change limits, now relative to the absolute
        # anchor as well as its per-frame check. Gradual scale drift cannot renew
        # the anchor and quietly redefine the map coordinate system.
        if not 0.985 <= relative_scale <= 1.015 or abs(relative_rotation) > 3:
            return reject('tracking_view_changed_since_absolute_anchor')
        height, width = query.shape[:2]
        center = np.float64([width // 2, height // 2, 1])
        proposed = flow @ center
        try:
            candidate_xy = tuple(map(float, tracked.xy))
        except (TypeError, ValueError):
            return reject('invalid_tracking_position')
        if (len(candidate_xy) != 2 or not np.isfinite(candidate_xy).all()
                or not np.allclose(proposed, candidate_xy, rtol=0, atol=1e-4)):
            return reject('tracking_position_disagrees_with_transform')
        try:
            left, top, right, bottom = region
        except (TypeError, ValueError):
            return reject('invalid_reference_region')
        if not (left <= proposed[0] < right and top <= proposed[1] < bottom):
            return reject('tracking_position_outside_reference_region')
        transform = anchor.copy()
        transform[:, 2] = proposed - anchor[:, :2] @ center[:2]
        result = self.matcher.verify(reference, query, mask, region, transform)
        self.last_diagnostic['verification'] = dict(self.matcher.last_diagnostic)
        if result is None:
            return reject('absolute_terrain_reproof_failed')
        self.last_diagnostic.update(status='matched', reason='absolute_terrain_reproved',
                                    position=result.xy)
        return result

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer, MapPosition
from src.pathing.MinimapTracker import MinimapTracker


class TestMinimapTracker(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(260617)
        self.terrain = cv2.GaussianBlur(rng.integers(0, 256, (450, 500), np.uint8), (5, 5), 0.8)
        self.gray = self.terrain[100:300, 120:320]
        self.mask = np.zeros_like(self.gray)
        cv2.circle(self.mask, (100, 100), 92, 255, -1)
        cv2.circle(self.mask, (100, 100), 22, 0, -1)
        self.tracker = MinimapTracker()
        # A minimap pixel is two pixels in the recorded map.
        self.transform = np.float64([[2, 0, 200], [0, 2, 300]])
        self.tracker.anchor(self.gray, self.mask, self.transform, (400, 500), 0)

    def test_translated_terrain_maps_current_center_in_correct_direction_and_scale(self):
        moved = self.terrain[104:304, 128:328]
        result = self.tracker.track(moved, self.mask, 0.2)
        self.assertIsNotNone(result, self.tracker.last_diagnostic)
        np.testing.assert_allclose(result.xy, (416, 508), atol=0.5)
        self.assertGreaterEqual(result.inliers, 12)

    def test_subpixel_motion_and_center_arrow_change_are_tolerated(self):
        affine = np.float32([[1, 0, -1.4], [0, 1, 0.6]])
        moved = cv2.warpAffine(self.gray, affine, (200, 200))
        cv2.circle(moved, (100, 100), 12, 255, -1)
        result = self.tracker.track(moved, self.mask, 0.2)
        self.assertIsNotNone(result, self.tracker.last_diagnostic)
        np.testing.assert_allclose(result.xy, (402.8, 498.8), atol=0.5)

    def test_committing_relative_frames_does_not_renew_absolute_time_budget(self):
        for now in (0.5, 1, 1.5, 2, 2.5, 3):
            result = self.tracker.track(self.gray, self.mask, now)
            self.assertIsNotNone(result)
            self.tracker.commit(self.gray, self.mask, result, now)
        self.assertEqual(0, self.tracker.anchor_at)
        self.assertIsNone(self.tracker.track(self.gray, self.mask, 3.01))
        self.assertEqual('tracking_expired', self.tracker.last_diagnostic['reason'])

    def test_cumulative_distance_budget_is_measured_from_absolute_anchor(self):
        for index in (1, 2):
            frame = self.terrain[100:300, 120 + index * 12:320 + index * 12]
            result = self.tracker.track(frame, self.mask, index * 0.3)
            self.assertIsNotNone(result, self.tracker.last_diagnostic)
            self.tracker.commit(frame, self.mask, result, index * 0.3)
        frame = self.terrain[100:300, 156:356]
        self.assertIsNone(self.tracker.track(frame, self.mask, 0.9))
        self.assertEqual('tracking_distance_exceeded', self.tracker.last_diagnostic['reason'])

    def test_missing_anchor_size_change_and_long_frame_gap_reject(self):
        self.assertIsNone(self.tracker.track(self.gray[:190], self.mask[:190], 0.2))
        self.assertEqual('capture_size_changed', self.tracker.last_diagnostic['reason'])
        self.assertIsNone(self.tracker.track(self.gray, self.mask, 2.01))
        self.assertEqual('tracking_expired', self.tracker.last_diagnostic['reason'])
        self.tracker.reset()
        self.assertIsNone(self.tracker.track(self.gray, self.mask, 0.2))
        self.assertEqual('no_tracking_anchor', self.tracker.last_diagnostic['reason'])

    def test_blank_unrelated_zoom_and_rotation_do_not_continue_tracking(self):
        for frame in (np.zeros_like(self.gray), self.terrain[230:430, 290:490],
                      cv2.warpAffine(self.gray, cv2.getRotationMatrix2D((100, 100), 0, 1.1), (200, 200)),
                      cv2.warpAffine(self.gray, cv2.getRotationMatrix2D((100, 100), 8, 1), (200, 200))):
            with self.subTest():
                self.assertIsNone(self.tracker.track(frame, self.mask, 0.2), self.tracker.last_diagnostic)

    def test_landmarks_confined_to_one_corner_cannot_locate_the_player(self):
        corner = np.zeros_like(self.mask)
        corner[35:95, 35:95] = self.mask[35:95, 35:95]
        self.tracker.points = cv2.goodFeaturesToTrack(self.gray, 120, 0.02, 7, mask=corner, blockSize=7)
        self.assertGreaterEqual(len(self.tracker.points), 12)
        self.assertIsNone(self.tracker.track(self.gray, self.mask, 0.2))
        self.assertEqual('tracking_features_concentrated', self.tracker.last_diagnostic['reason'])


class TestLocalizerTracking(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.map = cv2.GaussianBlur(rng.integers(0, 256, (650, 850), np.uint8), (5, 5), 0.8)
        self.localizer = MapLocalizer(self.map)
        clock_patch = patch('src.pathing.MapLocalizer.time.monotonic', return_value=0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        result = self.localizer.locate(self.map[280:480, 380:580])
        self.assertIsNotNone(result)
        self.assertEqual('sift', result.source)

    def test_short_sift_failure_uses_visual_motion_then_returns_to_absolute_matching(self):
        with patch.object(self.localizer, '_locate_absolute', return_value=None) as absolute:
            self.clock.return_value = 0.2
            tracked = self.localizer.locate(self.map[283:483, 385:585])
            self.assertEqual('flow', tracked.source)
            np.testing.assert_allclose(tracked.xy, (485, 383), atol=1)
            absolute.assert_not_called()
            self.clock.return_value = 0.6
            self.assertEqual('flow', self.localizer.locate(self.map[285:485, 390:590]).source)
            absolute.assert_called_once()
            self.assertEqual(0, self.localizer.tracker.anchor_at)
        self.clock.return_value = 1.2
        absolute = self.localizer.locate(self.map[285:485, 390:590])
        self.assertEqual('sift', absolute.source)
        self.assertEqual(1.2, self.localizer.tracker.anchor_at)

    def test_persistent_absolute_failure_stops_relative_tracking_after_three_seconds(self):
        with patch.object(self.localizer, '_locate_absolute', return_value=None):
            for now in (0.6, 1.2, 1.8, 2.4, 3):
                self.clock.return_value = now
                self.assertIsNotNone(self.localizer.locate(self.map[280:480, 380:580]))
            self.clock.return_value = 3.1
            self.assertIsNone(self.localizer.locate(self.map[280:480, 380:580]))
            self.assertEqual('tracking_expired', self.localizer.last_diagnostic['tracking']['reason'])

    def test_reset_after_combat_requires_a_new_absolute_match(self):
        self.localizer.reset()
        self.clock.return_value = 0.2
        with patch.object(self.localizer, '_locate_absolute', return_value=None):
            self.assertIsNone(self.localizer.locate(self.map[280:480, 380:580]))
        self.assertIsNone(self.localizer.previous)

    def test_conflicting_absolute_match_stops_flow_until_a_new_absolute_anchor(self):
        self.clock.return_value = 0.6
        previous = self.localizer.previous
        with patch.object(self.localizer, '_locate_absolute', return_value=MapPosition(530, 380, 1, 20)):
            self.assertIsNone(self.localizer.locate(self.map[280:480, 380:580]))
        self.assertEqual('absolute_disagrees_with_tracking', self.localizer.last_diagnostic['reason'])
        self.assertEqual(previous, self.localizer.previous)
        self.assertIsNone(self.localizer.tracker.anchor_xy)
        self.clock.return_value = 0.7
        self.assertEqual('sift', self.localizer.locate(self.map[280:480, 380:580]).source)

    def test_explicit_absolute_jump_or_view_change_does_not_fall_back_to_flow(self):
        for reason in ('position_jump', 'minimap_not_north_up', 'scale_out_of_range',
                       'position_outside_recorded_map'):
            with self.subTest(reason=reason):
                self.clock.return_value = 0
                self.localizer.reset()
                self.localizer.locate(self.map[280:480, 380:580])
                self.clock.return_value = 0.6
                with patch.object(self.localizer, '_locate_absolute',
                                  side_effect=lambda *args: self.localizer._reject(reason)):
                    self.assertIsNone(self.localizer.locate(self.map[280:480, 380:580]))
                self.assertEqual(reason, self.localizer.last_diagnostic['reason'])
                self.assertIsNone(self.localizer.tracker.anchor_xy)

    def test_relative_position_outside_recorded_map_is_rejected(self):
        self.localizer.region = (0, 0, 482, 650)
        self.clock.return_value = 0.2
        with patch.object(self.localizer, '_locate_absolute', return_value=None):
            self.assertIsNone(self.localizer.locate(self.map[280:480, 385:585]))
        self.assertEqual('position_outside_recorded_map', self.localizer.last_diagnostic['tracking']['reason'])


if __name__ == '__main__':
    unittest.main()

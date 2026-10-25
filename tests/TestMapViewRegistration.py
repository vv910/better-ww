import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer


class TestMapViewRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(261025)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (650, 850), np.uint8), (5, 5), .8)
        cls.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_strong_offscreen_registration_is_kept_but_legacy_projection_rejects(self):
        localizer = MapLocalizer(self.image)
        frame = self.image[100:500, 300:800].copy()
        region = (0, 0, 500, 400)
        registered = localizer.register_map_view(frame, (50, 300), region)
        self.assertIsNotNone(registered, localizer.last_diagnostic)
        self.assertGreaterEqual(registered.inliers, 12)
        self.assertGreaterEqual(registered.confidence, .6)
        np.testing.assert_allclose(registered.xy, (-250, 200), atol=2)
        self.assertFalse(registered.inside_view)
        self.assertEqual('matched', localizer.last_diagnostic['status'])
        self.assertIsNone(localizer.project_map_point(frame, (50, 300), region))
        self.assertEqual('projected_beacon_outside_view', localizer.last_diagnostic['reason'])

    def test_known_transform_and_inverse_keep_full_image_roi_coordinates(self):
        localizer = MapLocalizer(self.image, (100, 60, 780, 590))
        expected = np.float64([[1.15, 0, 45], [0, 1.15, 25]])
        frame = cv2.warpAffine(self.image, expected, (1100, 850))
        region = (130, 110, 950, 740)
        with patch('src.pathing.MapLocalizer.time.monotonic', return_value=123.5):
            features = localizer.prepare_map_features(frame, region)
        registered = localizer.register_map_view(frame, (480, 380), region, features=features)
        self.assertIsNotNone(registered, localizer.last_diagnostic)
        np.testing.assert_allclose(registered.xy, (597, 462), atol=2)
        np.testing.assert_allclose(registered.reference_to_screen, expected, atol=.2)
        np.testing.assert_allclose(registered.screen_to_reference,
                                   cv2.invertAffineTransform(expected), atol=.2)
        np.testing.assert_allclose(registered.screen_to_reference @ [*registered.xy, 1],
                                   (480, 380), atol=.01)
        self.assertAlmostEqual(1 / 1.15, registered.scale, delta=.005)
        self.assertAlmostEqual(0, registered.rotation, delta=.1)
        self.assertEqual(region, registered.screen_region)
        self.assertEqual(123.5, registered.observed_at)
        self.assertTrue(registered.inside_view)
        with self.assertRaises(FrozenInstanceError):
            registered.inside_view = False
        for matrix in (registered.screen_to_reference, registered.reference_to_screen):
            self.assertFalse(matrix.flags.writeable)
            with self.assertRaises(ValueError):
                matrix[0, 0] = 99
            with self.assertRaises(ValueError):
                matrix.setflags(write=True)

    def test_shared_features_extract_once_for_different_references(self):
        first = MapLocalizer(self.image)
        second = MapLocalizer(self.image, (100, 60, 780, 590))
        frame = self.image[100:500, 300:800].copy()
        region = (0, 0, 500, 400)
        with patch.object(first, '_region_features', wraps=first._region_features) as extract_first, \
                patch.object(second, '_region_features', wraps=second._region_features) as extract_second:
            features = first.prepare_map_features(frame, region)
            first_result = first.register_map_view(frame, (480, 380), region, features=features)
            second_result = second.register_map_view(frame, (480, 380), region, features=features)
        self.assertEqual(1, extract_first.call_count)
        extract_second.assert_not_called()
        self.assertIsNotNone(first_result)
        self.assertIsNotNone(second_result)
        np.testing.assert_allclose(first_result.xy, second_result.xy, atol=1)
        self.assertEqual(first_result.observed_at, second_result.observed_at)

    def test_features_reject_other_frames_regions_shapes_and_rewritten_buffers(self):
        localizer = MapLocalizer(self.image)
        frame = self.image[100:500, 300:800].copy()
        region = (0, 0, 500, 400)
        features = localizer.prepare_map_features(frame, region)
        with self.assertRaisesRegex(ValueError, 'same frame'):
            localizer.register_map_view(frame.copy(), (480, 380), region, features=features)
        with self.assertRaisesRegex(ValueError, 'same frame'):
            localizer.register_map_view(frame, (480, 380), (10, 0, 500, 400), features=features)
        frame.resize((500, 400, 3), refcheck=False)
        with self.assertRaisesRegex(ValueError, 'same frame'):
            localizer.register_map_view(frame, (480, 380), (0, 0, 400, 500), features=features)
        frame.resize((400, 500, 3), refcheck=False)
        frame[30:60, 40:70] = 0
        with self.assertRaisesRegex(ValueError, 'changed'):
            localizer.register_map_view(frame, (480, 380), region, features=features)

    def test_low_quality_geometry_remains_rejected(self):
        localizer = MapLocalizer(self.image)
        region = (0, 0, 850, 650)
        blank = np.zeros_like(self.image)
        self.assertIsNone(localizer.register_map_view(blank, (480, 380), region))
        self.assertEqual('too_few_map_features', localizer.last_diagnostic['reason'])
        # Correct terrain only in a compact island is insufficient spatial support.
        patch_frame = np.zeros_like(self.image)
        patch_frame[270:360, 350:440] = self.image[270:360, 350:440]
        self.assertIsNone(localizer.register_map_view(patch_frame, (480, 380), region))
        self.assertEqual('features_too_concentrated', localizer.last_diagnostic['reason'])
        rotated = cv2.warpAffine(self.image, cv2.getRotationMatrix2D((425, 325), 30, 1), (850, 650))
        self.assertIsNone(localizer.register_map_view(rotated, (480, 380), region))
        self.assertEqual('minimap_not_north_up', localizer.last_diagnostic['reason'])

    def test_map_specific_inlier_and_confidence_thresholds_are_unchanged(self):
        localizer = MapLocalizer(self.image)
        region = (0, 0, 850, 650)
        features = localizer.prepare_map_features(self.image, region)
        identity = np.float64([[1, 0, 0], [0, 1, 0]])
        for confidence, inliers in ((.9, 11), (.59, 40)):
            with self.subTest(confidence=confidence, inliers=inliers), \
                    patch.object(localizer, '_transform', return_value=(identity, confidence, inliers)):
                self.assertIsNone(localizer.register_map_view(self.image, (480, 380), region, features))
                self.assertEqual('weak_map_projection', localizer.last_diagnostic['reason'])
        with patch.object(localizer, '_transform', return_value=(identity, .6, 12)):
            registered = localizer.register_map_view(self.image, (480, 380), region, features)
        self.assertIsNotNone(registered)
        identity[0, 2] = 100
        self.assertEqual(0, registered.screen_to_reference[0, 2])


if __name__ == '__main__':
    unittest.main()

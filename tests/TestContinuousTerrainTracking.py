import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.TerrainMatcher import TerrainMatch


class TestContinuousTerrainTracking(unittest.TestCase):
    def setUp(self):
        clock_patch = patch('src.pathing.MapLocalizer.time.monotonic', return_value=0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.reference = cv2.imread('tests/images/path.png')
        self.query = cv2.imread('tests/images/mini_map.png')[30:211, 35:218]
        self.localizer = MapLocalizer(self.reference, (538, 130, 1670, 972))
        self.start = self.localizer.locate(self.query)
        self.assertIsNotNone(self.start)
        self.initial_scale = self.localizer._absolute_transform[:, :2].copy()

    def shifted(self, dx):
        return cv2.warpAffine(self.query, np.float32([[1, 0, -dx], [0, 1, 0]]),
                              (self.query.shape[1], self.query.shape[0]))

    def no_sift(self, *args):
        return self.localizer._reject('too_few_matches')

    def test_independent_terrain_evidence_keeps_a_weak_sift_sequence_located(self):
        sources = []
        with patch.object(self.localizer, '_locate_absolute', side_effect=self.no_sift):
            for index in range(1, 11):
                self.clock.return_value = index * .6
                result = self.localizer.locate(self.shifted(index))
                self.assertIsNotNone(result, self.localizer.last_diagnostic)
                sources.append(result.source)
                np.testing.assert_allclose(result.xy, (self.start.x + index * .9825, self.start.y), atol=2)
        self.assertIn('terrain_track', sources)
        self.assertGreater(self.localizer.tracker.anchor_at, 3)
        np.testing.assert_allclose(self.initial_scale, self.localizer._absolute_transform[:, :2], rtol=0, atol=1e-7)

    def test_a_failed_full_map_reproof_never_renews_the_flow_anchor(self):
        with patch.object(self.localizer, '_locate_absolute', side_effect=self.no_sift), \
                patch.object(self.localizer.tracking_verifier, 'verify', return_value=None):
            for now in (.6, 1.2, 1.8, 2.4, 3):
                self.clock.return_value = now
                self.assertEqual('flow', self.localizer.locate(self.query).source)
                self.assertEqual(0, self.localizer.tracker.anchor_at)
            self.clock.return_value = 3.2
            self.assertIsNone(self.localizer.locate(self.query))

    def test_geometry_conflict_prevents_terrain_revalidation(self):
        self.clock.return_value = .6
        with patch.object(self.localizer, '_locate_absolute',
                          side_effect=lambda *args: self.localizer._reject('minimap_not_north_up')), \
                patch.object(self.localizer.tracking_verifier, 'verify') as verifier:
            self.assertIsNone(self.localizer.locate(self.query))
        verifier.assert_not_called()

    def test_slow_terrain_reproof_cannot_anchor_a_stale_image(self):
        self.clock.return_value = .6
        transform = self.localizer._absolute_transform.copy()

        def slow_reproof(*args):
            self.clock.return_value = 1.7
            return TerrainMatch(transform, self.start.xy, .8, .2, .6, (.8,) * 4)

        with patch.object(self.localizer, '_locate_absolute', side_effect=self.no_sift), \
                patch.object(self.localizer.tracking_verifier, 'verify', side_effect=slow_reproof):
            self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('observation_expired', self.localizer.last_diagnostic['reason'])
        self.assertIsNone(self.localizer.tracker.anchor_xy)


if __name__ == '__main__':
    unittest.main()

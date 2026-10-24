"""Map registration can guide image search, but cannot substitute for image evidence."""

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer, MapPosition
from src.pathing.TerrainMatcher import TerrainMatch


class TestMapHintSearch(unittest.TestCase):
    def setUp(self):
        clock_patch = patch('src.pathing.MapLocalizer.time.monotonic', return_value=0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        rng = np.random.default_rng(4811)
        self.image = cv2.GaussianBlur(rng.integers(20, 190, (500, 600), dtype=np.uint8), (5, 5), .8)
        self.localizer = MapLocalizer(self.image)
        self.query = self.image[150:350, 200:400]
        self.localizer.reset(MapPosition(300, 250, .9, 25, source='map'))
        self.localizer._locate_absolute = lambda *args: self.localizer._reject('too_few_matches')
        self.localizer.terrain_search.search = Mock(return_value=None)
        self.localizer.terrain_search.last_diagnostic = {'reason': 'no_image_match'}

    def test_registered_map_hint_cannot_return_a_position_without_minimap_evidence(self):
        self.assertIsNone(self.localizer.locate(self.query))
        self.localizer.terrain_search.search.assert_called_once()
        self.assertEqual('hint', self.localizer.previous.source)
        self.assertIsNone(self.localizer.tracker.anchor_xy)
        self.assertEqual('no_image_match', self.localizer.last_diagnostic['template_search']['reason'])

    def test_plain_coordinate_hint_and_route_start_cannot_enable_template_search(self):
        self.localizer.reset((300, 250))
        self.assertIsNone(self.localizer.locate(self.query))
        self.localizer.terrain_search.search.assert_not_called()

    def test_hint_expires_and_failed_searches_are_throttled(self):
        self.assertIsNone(self.localizer.locate(self.query))
        self.clock.return_value = .5
        self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('search_throttled', self.localizer.last_diagnostic['template_search']['reason'])
        self.clock.return_value = 1.1
        self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual(2, self.localizer.terrain_search.search.call_count)
        self.clock.return_value = 5.01
        self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('map_hint_expired', self.localizer.last_diagnostic['template_search']['reason'])
        self.assertEqual(2, self.localizer.terrain_search.search.call_count)

    def test_explicit_geometry_conflict_cannot_be_overridden(self):
        for reason in ('position_jump', 'minimap_not_north_up', 'scale_out_of_range',
                       'position_outside_recorded_map'):
            with self.subTest(reason=reason):
                self.localizer._locate_absolute = lambda *args: self.localizer._reject(reason)
                self.assertIsNone(self.localizer.locate(self.query))
                self.assertEqual(reason, self.localizer.last_diagnostic['reason'])
        self.localizer.terrain_search.search.assert_not_called()

    def test_sift_time_cannot_extend_hint_lifetime(self):
        def slow_sift(*args):
            self.clock.return_value = 5.1
            return self.localizer._reject('too_few_matches')

        self.localizer._locate_absolute = slow_sift
        self.assertIsNone(self.localizer.locate(self.query))
        self.localizer.terrain_search.search.assert_not_called()
        self.assertEqual('map_hint_expired', self.localizer.last_diagnostic['template_search']['reason'])

    def test_hint_that_expires_during_template_search_cannot_anchor_tracking(self):
        def slow_search(*args):
            self.clock.return_value = 5.1
            return TerrainMatch(np.float32([[1, 0, 200], [0, 1, 150]]),
                                (300, 250), .8, .2, .6, (.8,) * 4)

        self.localizer.terrain_search.search.side_effect = slow_search
        self.assertIsNone(self.localizer.locate(self.query))
        self.assertIsNone(self.localizer.tracker.anchor_xy)
        self.assertEqual('map_hint_expired', self.localizer.last_diagnostic['template_search']['reason'])

    def test_geometry_conflict_invalidates_hint_for_later_frames_too(self):
        self.localizer._locate_absolute = lambda *args: self.localizer._reject('minimap_not_north_up')
        self.assertIsNone(self.localizer.locate(self.query))
        self.clock.return_value = 1.1
        self.localizer._locate_absolute = lambda *args: self.localizer._reject('too_few_matches')
        self.assertIsNone(self.localizer.locate(self.query))
        self.localizer.terrain_search.search.assert_not_called()

    def test_geometry_conflict_in_local_attempt_is_not_hidden_by_global_failure(self):
        def failed_search(*args):
            self.localizer.last_diagnostic['attempts'] = [{'reason': 'minimap_not_north_up'},
                                                        {'reason': 'too_few_matches'}]
            return self.localizer._reject('too_few_matches')

        self.localizer._locate_absolute = failed_search
        self.assertIsNone(self.localizer.locate(self.query))
        self.localizer.terrain_search.search.assert_not_called()

    def test_template_match_anchors_flow_without_claiming_sift_inliers_and_consumes_hint(self):
        transform = np.float32([[1, 0, 200], [0, 1, 150]])
        match = TerrainMatch(transform, (300, 250), .8, .2, .6, (.8,) * 4)
        self.localizer.terrain_search.search.return_value = match
        result = self.localizer.locate(self.query)
        self.assertEqual('template', result.source)
        self.assertEqual(0, result.inliers)
        np.testing.assert_allclose(self.localizer.tracker.transform[:2], transform)
        self.localizer.tracker.reset()
        self.clock.return_value = 1.1
        self.assertIsNone(self.localizer.locate(self.query))
        self.localizer.terrain_search.search.assert_called_once()

    def test_template_search_keeps_the_existing_jump_limit(self):
        self.localizer.max_jump = 2
        match = TerrainMatch(np.float32([[1, 0, 205], [0, 1, 150]]),
                             (305, 250), .8, .2, .6, (.8,) * 4)
        self.localizer.terrain_search.search.return_value = match
        self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('position_jump', self.localizer.last_diagnostic['reason'])

    def test_real_minimap_can_bootstrap_after_forced_zero_sift_matches(self):
        image = cv2.imread('tests/images/3c-terrain-map.png')
        query = cv2.imread('tests/images/3c-terrain-minimap.png')
        localizer = MapLocalizer(image, (717, 173, 2227, 1296))
        localizer.reset(MapPosition(1322, 691, .9, 25, source='map'))
        with patch.object(localizer, '_locate_absolute',
                          side_effect=lambda *args: localizer._reject('too_few_matches')):
            result = localizer.locate(query)
        self.assertIsNotNone(result, localizer.last_diagnostic)
        self.assertEqual('template', result.source)
        np.testing.assert_allclose(result.xy, (1322, 691), atol=3)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer


class TestLocalizationFreshness(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1024)
        self.image = cv2.GaussianBlur(rng.integers(30, 190, (600, 800), np.uint8), (5, 5), .8)
        self.query = self.image[220:420, 310:510]
        clock_patch = patch('src.pathing.MapLocalizer.time.monotonic', return_value=0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.localizer = MapLocalizer(self.image)
        self.assertIsNotNone(self.localizer.locate(self.query))
        verifier_patch = patch.object(self.localizer.tracking_verifier, 'verify', return_value=None)
        verifier_patch.start()
        self.addCleanup(verifier_patch.stop)

    def test_slow_absolute_match_cannot_return_an_old_image_as_a_current_position(self):
        previous = self.localizer.previous
        absolute = self.localizer._locate_absolute

        def slow_match(*args):
            result = absolute(*args)
            self.clock.return_value = 2.1
            return result

        self.clock.return_value = 1
        with patch.object(self.localizer, '_locate_absolute', side_effect=slow_match):
            self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('observation_expired', self.localizer.last_diagnostic['reason'])
        self.assertEqual(previous, self.localizer.previous)
        self.assertIsNone(self.localizer.tracker.anchor_xy)

    def test_flow_anchor_expiring_during_sift_cannot_continue_motion(self):
        self.clock.return_value = 2.8
        # A recently observed flow frame does not renew the absolute anchor.
        self.localizer.tracker.previous_at = 2.6

        def slow_failure(*args):
            self.clock.return_value = 3.2
            return self.localizer._reject('too_few_matches')

        with patch.object(self.localizer, '_locate_absolute', side_effect=slow_failure):
            self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('tracking_expired_during_localization', self.localizer.last_diagnostic['reason'])
        self.assertIsNone(self.localizer.tracker.anchor_xy)

    def test_flow_previous_frame_gap_is_checked_after_recognition_too(self):
        self.clock.return_value = 1.8

        def slow_failure(*args):
            self.clock.return_value = 2.1
            return self.localizer._reject('too_few_matches')

        with patch.object(self.localizer, '_locate_absolute', side_effect=slow_failure):
            self.assertIsNone(self.localizer.locate(self.query))
        self.assertEqual('tracking_expired_during_localization', self.localizer.last_diagnostic['reason'])

    def test_local_geometry_conflict_is_not_hidden_by_a_later_missing_global_match(self):
        self.clock.return_value = .6

        def conflicting_failure(*args):
            self.localizer.last_diagnostic['attempts'] = [{'reason': 'minimap_not_north_up'},
                                                        {'reason': 'too_few_matches'}]
            return self.localizer._reject('too_few_matches')

        with patch.object(self.localizer, '_locate_absolute', side_effect=conflicting_failure):
            self.assertIsNone(self.localizer.locate(self.query))
        self.assertIsNone(self.localizer.tracker.anchor_xy)

    def test_new_absolute_image_can_replace_an_expiring_flow_anchor(self):
        self.clock.return_value = 2.8
        self.localizer.tracker.previous_at = 2.6
        absolute = self.localizer._locate_absolute

        def fresh_absolute(*args):
            result = absolute(*args)
            self.clock.return_value = 3.2
            return result

        with patch.object(self.localizer, '_locate_absolute', side_effect=fresh_absolute):
            result = self.localizer.locate(self.query)
        self.assertIsNotNone(result, self.localizer.last_diagnostic)
        self.assertEqual('sift', result.source)
        self.assertEqual(2.8, self.localizer.tracker.anchor_at)

    def test_request_absolute_bypasses_half_second_flow_shortcut(self):
        self.clock.return_value = .2
        self.localizer.request_absolute()
        with patch.object(self.localizer, '_locate_absolute', wraps=self.localizer._locate_absolute) as absolute:
            self.assertEqual('sift', self.localizer.locate(self.query).source)
        absolute.assert_called_once()


if __name__ == '__main__':
    unittest.main()

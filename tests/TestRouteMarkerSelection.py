import unittest
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from ok import Box

from src.pathing.RouteMarkers import MarkerMatch
from src.task.Farm3CEchoTask import Record3CEchoRouteTask


def marker(x, y):
    return MarkerMatch(x - 15, y - 15, 30, 30, 0.95)


class TestRouteMarkerSelection(unittest.TestCase):
    def setUp(self):
        self.task = Record3CEchoRouteTask.__new__(Record3CEchoRouteTask)
        self.task.config = {'Maximum Marker Gap': 200, 'Recording Radius': 500}
        self.task._executor = SimpleNamespace(frame=np.zeros((1080, 1920, 3), np.uint8))
        for name in ('next_frame', 'get_feature_by_name', 'log_info', 'log_warning', 'info_set', 'screenshot',
                     'find_feature',
                     'send_key', 'wait_in_team_and_world'):
            setattr(self.task, name, Mock())

    def load(self, stars, diamonds=None, mode='Diamond'):
        if mode is not None:
            self.task.config['Start Method'] = mode
        with patch('src.task.Farm3CEchoTask.RouteMarkerDetector') as detector:
            detected_diamonds = [marker(900, 500)] if diamonds is None else diamonds
            detector.return_value.detect.return_value = (detected_diamonds, stars)
            self.task.load_stars(wait_world=False)

    def test_nearby_route_excludes_a_separate_cluster(self):
        self.load([marker(1000, 500), marker(1100, 550), marker(1400, 500), marker(1450, 550)])
        self.assertEqual([(900, 500), (1000, 500), (1100, 550)], [box.center() for box in self.task.stars])
        self.assertEqual(2, len(self.task.excluded_stars))
        self.task.send_key.assert_called_once_with('esc')
        self.assertFalse(self.task._record_map_open)

    def test_radius_prevents_a_chain_of_stars_reaching_another_route(self):
        self.task.config['Recording Radius'] = 250
        self.load([marker(1000, 500), marker(1100, 500), marker(1200, 500), marker(1300, 500)])
        self.assertEqual(3, len(self.task.stars))
        self.assertEqual(2, len(self.task.excluded_stars))

    def test_distance_limits_scale_with_capture_resolution(self):
        self.task._executor.frame = np.zeros((1440, 2560, 3), np.uint8)
        self.load([marker(1250, 800), marker(1500, 800)], diamonds=[marker(1000, 800)])
        self.assertEqual(3, len(self.task.stars))  # 250 < 200 * 1440 / 1080

    def test_multiple_starts_fail_without_silently_selecting_one(self):
        with self.assertRaisesRegex(ValueError, 'Multiple start diamonds'):
            self.load([marker(1100, 500)], diamonds=[marker(900, 500), marker(1000, 600)])
        self.task.send_key.assert_not_called()
        self.task.screenshot.assert_called_once()

    def test_not_enough_connected_stars_fails_with_a_preview(self):
        with self.assertRaisesRegex(ValueError, 'at least two nearby stars'):
            self.load([marker(1000, 500), marker(1500, 700)])
        self.task.send_key.assert_not_called()
        self.assertEqual('3c-record-markers', self.task.screenshot.call_args.args[0])

    def test_nearest_beacon_default_ignores_diamonds_and_selects_central_star_cluster(self):
        self.task.find_feature.return_value = [Box(1065, 525, 30, 30), Box(1485, 485, 30, 30)]
        stars = [marker(1020, 500), marker(1100, 550), marker(1180, 600), marker(1500, 500)]
        self.load(stars, diamonds=[marker(800, 400), marker(900, 600)], mode=None)
        self.assertEqual([(1080, 540), (1100, 550), (1020, 500), (1180, 600)],
                         [star.center() for star in self.task.stars])
        self.assertEqual(1, len(self.task.excluded_stars))
        self.task.find_feature.assert_called_once()
        self.task.send_key.assert_not_called()  # Keep the map open for the route's overview capture.
        self.assertTrue(self.task._record_map_open)

    def test_nearest_beacon_without_visible_unlocked_beacon_fails(self):
        self.task.find_feature.return_value = []
        with self.assertRaisesRegex(ValueError, 'unlocked beacon'):
            self.load([marker(1020, 500), marker(1100, 550)], diamonds=[], mode=None)
        self.task.send_key.assert_not_called()

    def test_nearest_beacon_entry_leg_may_exceed_gap_but_stars_remain_gap_limited(self):
        beacon = Box(1385, 535, 30, 30)
        self.task.find_feature.return_value = [beacon]
        stars = [marker(1100, 550), marker(1100, 600), marker(1100, 650), marker(1350, 550)]
        self.load(stars, diamonds=[marker(800, 400), marker(900, 600)], mode=None)
        selected_centers = [star.center() for star in self.task.stars]
        self.assertEqual([(1400, 550), (1100, 550), (1100, 600), (1100, 650)], selected_centers)
        self.assertGreater(math.dist(selected_centers[0], selected_centers[1]), 200)
        self.assertLess(math.dist(selected_centers[0], selected_centers[1]), 500)
        self.assertEqual([(1350, 550)], [star.center() for star in self.task.excluded_stars])

    def test_nearest_beacon_can_record_one_star_as_one_combat_point(self):
        self.task.find_feature.return_value = [Box(1485, 535, 30, 30)]
        self.load([marker(1100, 550)], diamonds=[], mode=None)
        self.assertEqual([(1500, 550), (1100, 550)], [star.center() for star in self.task.stars])
        self.assertEqual([], self.task.excluded_stars)

    def test_nearest_beacon_rejects_a_first_star_beyond_recording_radius(self):
        self.task.config['Recording Radius'] = 300
        self.task.find_feature.return_value = [Box(1485, 535, 30, 30)]
        with self.assertRaisesRegex(ValueError, 'No reachable route stars'):
            self.load([marker(1100, 550)], diamonds=[], mode=None)
        self.assertEqual('3c-record-markers', self.task.screenshot.call_args.args[0])
        self.task.send_key.assert_not_called()

    def test_all_visible_stars_selects_entire_central_group_without_distance_filters(self):
        beacon = Box(1485, 785, 30, 30)
        self.task.find_feature.return_value = [beacon]
        stars = [marker(700, 250), marker(1150, 450), marker(1550, 850), marker(1800, 500)]
        self.task.config.update({'Star Selection': 'All Visible Stars'})
        self.load(stars, diamonds=[], mode=None)
        self.assertEqual([(1500, 800), (1550, 850), (1150, 450), (700, 250)],
                         [star.center() for star in self.task.stars])
        self.assertEqual([], self.task.excluded_stars)

    def test_all_visible_stars_also_works_with_legacy_diamond_start(self):
        self.task.config.update({'Star Selection': 'All Visible Stars', 'Recording Radius': 100})
        self.load([marker(1000, 500), marker(1500, 800)], diamonds=[marker(900, 500)], mode='Diamond')
        self.assertEqual([(900, 500), (1000, 500), (1500, 800)],
                         [star.center() for star in self.task.stars])
        self.assertEqual([], self.task.excluded_stars)

    def test_unknown_star_selection_fails_before_opening_or_closing_map(self):
        self.task.config['Star Selection'] = 'Everything'
        with self.assertRaisesRegex(ValueError, 'Unknown Star Selection'):
            self.load([marker(1000, 500)], mode='Nearest Beacon')
        self.task.send_key.assert_not_called()
        self.task.next_frame.assert_not_called()
        self.assertIsNone(self.task.validate_config('Star Selection', 'All Visible Stars'))
        self.assertTrue(self.task.validate_config('Star Selection', 'Everything'))

    def test_map_selection_path_validates_blank_and_bounded_exact_labels(self):
        self.assertIsNone(self.task.validate_config('Map Selection Path', ''))
        self.assertIsNone(self.task.validate_config('Map Selection Path', '区域\n梦境'))
        self.assertTrue(self.task.validate_config('Map Selection Path', 'a\nb\nc\nd\ne'))
        self.assertTrue(self.task.validate_config('Map Selection Path', 'a' * 81))
        self.assertTrue(self.task.validate_config('Map Selection Path', 'a\n\x01b'))

    def test_map_selection_path_without_teleport_fails_before_any_route_file_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            route_path = Path(directory) / 'route.json'
            self.task.config = {'Route File': str(route_path), 'Route Name': 'no teleport',
                                'Map Selection Path': '区域\n梦境'}
            self.task.big_map_frame = np.zeros((1080, 1920, 3), np.uint8)
            self.task.diamond = marker(900, 500)
            self.task.stars = [self.task.diamond, marker(1000, 500)]
            self.task._entry_beacon = None
            self.task._record_map_open = False
            self.task.load_stars = Mock()
            with patch('src.task.Farm3CEchoTask.get_path_relative_to_exe', return_value=str(route_path)), \
                    patch('src.task.Farm3CEchoTask.WWOneTimeTask.run'):
                with self.assertRaisesRegex(ValueError, 'Map Selection Path requires teleport metadata'):
                    self.task.run()
            self.assertFalse(route_path.exists())
            self.assertFalse(route_path.with_suffix('.map.png').exists())

    def test_invalid_map_selection_path_is_rejected_before_task_input(self):
        self.task.config = {'Map Selection Path': 'a\nb\nc\nd\ne'}
        with patch('src.task.Farm3CEchoTask.WWOneTimeTask.run') as base_run:
            with self.assertRaises(ValueError):
                self.task.run()
        base_run.assert_not_called()


if __name__ == '__main__':
    unittest.main()

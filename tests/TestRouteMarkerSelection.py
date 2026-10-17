import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.pathing.RouteMarkers import MarkerMatch
from src.task.Farm3CEchoTask import Record3CEchoRouteTask


def marker(x, y):
    return MarkerMatch(x - 15, y - 15, 30, 30, 0.95)


class TestRouteMarkerSelection(unittest.TestCase):
    def setUp(self):
        self.task = Record3CEchoRouteTask.__new__(Record3CEchoRouteTask)
        self.task.config = {'Maximum Marker Gap': 200, 'Recording Radius': 500}
        self.task._executor = SimpleNamespace(frame=np.zeros((1080, 1920, 3), np.uint8))
        for name in ('next_frame', 'get_feature_by_name', 'log_info', 'info_set', 'screenshot',
                     'send_key', 'wait_in_team_and_world'):
            setattr(self.task, name, Mock())

    def load(self, stars, diamonds=None):
        with patch('src.task.Farm3CEchoTask.RouteMarkerDetector') as detector:
            detector.return_value.detect.return_value = (diamonds or [marker(900, 500)], stars)
            self.task.load_stars(wait_world=False)

    def test_nearby_route_excludes_a_separate_cluster(self):
        self.load([marker(1000, 500), marker(1100, 550), marker(1400, 500), marker(1450, 550)])
        self.assertEqual([(900, 500), (1000, 500), (1100, 550)], [box.center() for box in self.task.stars])
        self.assertEqual(2, len(self.task.excluded_stars))
        self.task.send_key.assert_called_once_with('esc')

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


if __name__ == '__main__':
    unittest.main()

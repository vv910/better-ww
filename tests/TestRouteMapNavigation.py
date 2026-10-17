import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from ok import Box

from src.pathing.EchoRoute import RoutePoint
from src.pathing.MapLocalizer import MapPosition
from src.pathing.RouteMapNavigation import (RouteMapNavigator, capture_map_views, map_region,
                                            spiral_steps)


class TestRouteMapNavigation(unittest.TestCase):
    def test_spiral_is_25_unique_adjacent_cells(self):
        cells = list(spiral_steps())
        self.assertEqual(25, len(cells))
        self.assertEqual(25, len(set(cells)))
        self.assertEqual({(x, y) for x in range(-2, 3) for y in range(-2, 3)}, set(cells))
        for left, right in zip(cells, cells[1:]):
            self.assertEqual(1, abs(left[0] - right[0]) + abs(left[1] - right[1]), (left, right))

    def test_capture_projects_each_view_and_stops_at_first_mismatch_without_mutating_reference(self):
        original = np.full((90, 120, 3), 17, np.uint8)
        snapshot = original.copy()
        task = Mock()
        task.frame = original.copy()

        def next_frame():
            task.frame = np.full_like(original, 40 + task.next_frame.call_count)

        task.next_frame.side_effect = next_frame
        with patch('src.pathing.RouteMapNavigation.MapLocalizer') as localizer_type:
            localizer_type.return_value.project_map_point.side_effect = [
                MapPosition(31, 42, .9, 20), None]
            views = capture_map_views(task, original, (20, 25), (0, 0, 120, 90), max_views=5)

        self.assertEqual(1, len(views))
        np.testing.assert_array_equal(original, snapshot)
        self.assertEqual((31, 42), views[0][1])
        self.assertEqual((20, 25), localizer_type.return_value.project_map_point.call_args_list[0].args[1])
        self.assertEqual((31, 42), localizer_type.return_value.project_map_point.call_args_list[1].args[1])
        self.assertEqual(2, task.scroll.call_count)
        task.screenshot.assert_called_once()
        self.assertEqual('3c-map-overview-capture-mismatch-2', task.screenshot.call_args.args[0])
        np.testing.assert_array_equal(task.frame, task.screenshot.call_args.kwargs['frame'])
        task.log_warning.assert_called_once()

    def test_capture_rejects_changed_frame_size(self):
        task = Mock()
        task.frame = np.zeros((81, 120, 3), np.uint8)
        with patch('src.pathing.RouteMapNavigation.MapLocalizer'):
            with self.assertRaisesRegex(RuntimeError, 'size changed'):
                capture_map_views(task, np.zeros((80, 120, 3), np.uint8), (20, 20), (0, 0, 120, 80))
        task.screenshot.assert_called_once()
        self.assertEqual('3c-map-overview-capture-size-changed-1', task.screenshot.call_args.args[0])
        np.testing.assert_array_equal(task.frame, task.screenshot.call_args.kwargs['frame'])

    def test_identical_main_and_overview_reference_is_loaded_only_once(self):
        image_path = Path('tests/images/path.png').resolve()
        point = RoutePoint(1207, 469)
        same = SimpleNamespace(image=image_path, region=(538, 130, 1670, 972), point=point)
        distinct = SimpleNamespace(image=image_path, region=(538, 130, 1670, 972),
                                   point=RoutePoint(1210, 470))
        route = SimpleNamespace(map_image=image_path, map_region=same.region,
                                teleport=SimpleNamespace(waypoint=point, views=(same, same, distinct)))
        with patch('src.pathing.RouteMapNavigation.read_map_image', return_value=np.zeros((20, 20, 3), np.uint8)), \
                patch('src.pathing.RouteMapNavigation.MapLocalizer') as localizer_type:
            navigator = RouteMapNavigator(Mock(), route)
        self.assertEqual(2, localizer_type.call_count)
        self.assertEqual(2, len(navigator.references))

    def test_real_map_projection_handles_scale_translation_and_nonzero_roi(self):
        reference = cv2.imread('tests/images/path.png')
        self.assertIsNotNone(reference)
        roi = (538, 130, 1670, 972)
        live = cv2.warpAffine(reference, np.float32([[1.1, 0, -100], [0, 1.1, 10]]), (1920, 1080))
        route = SimpleNamespace(
            map_image=Path('tests/images/path.png'), map_region=roi,
            teleport=SimpleNamespace(waypoint=RoutePoint(1207, 469), views=()),
        )
        task = SimpleNamespace(frame=live)
        projected = RouteMapNavigator(task, route).project()
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(1227.7, projected.x, delta=3)
        self.assertAlmostEqual(525.9, projected.y, delta=3)

    def test_zoom_reprojects_before_testing_beacon_again(self):
        task = Mock()
        task.frame = np.zeros((1080, 1920, 3), np.uint8)
        navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        navigator.task = task
        first = MapPosition(700, 400, .9, 30)
        second = MapPosition(760, 430, .9, 30)
        beacon = Box(750, 420, 20, 20)
        with patch.object(navigator, 'project', side_effect=[first, second]) as project, \
                patch.object(navigator, 'beacon_at', side_effect=[None, beacon]):
            result = navigator.find()
        self.assertIs(result, beacon)
        task.scroll.assert_called_once_with(700, 400, 2)
        self.assertEqual(2, project.call_count)

    def test_beacon_icon_far_from_projected_waypoint_is_rejected(self):
        task = Mock()
        task.frame = np.zeros((1080, 1920, 3), np.uint8)
        task.find_feature.return_value = [Box(1000, 500, 24, 24)]
        navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        navigator.task = task
        self.assertIsNone(navigator.beacon_at(MapPosition(700, 400, .9, 30)))
        task.find_feature.assert_called_once()

    def test_search_stays_within_25_views_and_drag_steps_always_release_mouse(self):
        task = Mock()
        task.frame = np.zeros((1080, 1920, 3), np.uint8)
        navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        navigator.task = task
        with patch.object(navigator, 'project', return_value=None) as project, \
                patch('src.pathing.RouteMapNavigation.time.monotonic', return_value=0):
            self.assertIsNone(navigator.find(search=True))
        self.assertEqual(26, project.call_count)  # initial view plus the 25-cell grid
        self.assertEqual(25, task.info_set.call_count)
        self.assertEqual(24, task.swipe.call_count)
        self.assertEqual(24, task.mouse_up.call_count)

    def test_search_stops_at_40_second_budget(self):
        task = Mock()
        task.frame = np.zeros((1080, 1920, 3), np.uint8)
        navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        navigator.task = task
        elapsed = [0.0]

        def project_and_advance():
            # First call is the initial view; the first searched view consumes the budget.
            if task.info_set.call_count:
                elapsed[0] = 40.1
            return None

        with patch.object(navigator, 'project', side_effect=project_and_advance), \
                patch('src.pathing.RouteMapNavigation.time.monotonic', side_effect=lambda: elapsed[0]):
            self.assertIsNone(navigator.find(search=True))
        self.assertEqual(1, task.info_set.call_count)
        self.assertLessEqual(elapsed[0], 40.1)

    def test_drag_cancellation_still_releases_mouse(self):
        task = Mock()
        task.frame = np.zeros((1080, 1920, 3), np.uint8)
        task.swipe.side_effect = RuntimeError('cancelled')
        navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        navigator.task = task
        with patch.object(navigator, 'project', return_value=None), \
                patch('src.pathing.RouteMapNavigation.time.monotonic', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                navigator.find(search=True)
        task.mouse_up.assert_called_once()

if __name__ == '__main__':
    unittest.main()

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from ok import Box, TaskDisabledException

from src.pathing.EchoRoute import RoutePoint
from src.pathing.MapLocalizer import MapPosition
from src.pathing.RouteMapNavigation import (RouteMapNavigator, capture_map_views, map_region,
                                            spiral_steps)


class TestRouteMapNavigation(unittest.TestCase):
    @staticmethod
    def map_frame(shift=0):
        height, width = 240, 400
        y, x = np.indices((height, width))
        gray = ((x * 17 + y * 31) % 190 + 30).astype(np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        if shift:
            image = np.roll(image, shift, axis=1)
        return image

    def navigator_for(self, task):
        navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        navigator.task = task
        return navigator

    def test_real_maps_distinguish_motion_from_stationary_jpeg_noise(self):
        for path in ('tests/images/path.png', 'tests/images/3c-map-selector-closed.png'):
            with self.subTest(path=path):
                frame = cv2.imread(path)
                self.assertIsNotNone(frame)
                original = RouteMapNavigator._map_signature(frame)
                self.assertIsNotNone(original)
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                self.assertTrue(ok)
                compressed = RouteMapNavigator._map_signature(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
                self.assertTrue(RouteMapNavigator._same_map_signature(original, compressed))
                moved = cv2.warpAffine(frame, np.float32([[1, 0, frame.shape[1] * .04], [0, 1, 0]]),
                                      (frame.shape[1], frame.shape[0]))
                self.assertFalse(RouteMapNavigator._same_map_signature(
                    original, RouteMapNavigator._map_signature(moved)))

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
        task.frame = self.map_frame()
        navigator = self.navigator_for(task)
        first = MapPosition(700, 400, .9, 30)
        second = MapPosition(760, 430, .9, 30)
        beacon = Box(150, 100, 20, 20)
        latest_beacon = Box(151, 101, 20, 20)
        with patch.object(navigator, 'project', side_effect=[first, second, second, second]) as project, \
                patch.object(navigator, 'beacon_at', side_effect=[None, beacon, beacon, latest_beacon]):
            result = navigator.find()
        self.assertIs(result, latest_beacon)
        task.scroll.assert_called_once_with(700, 400, 2)
        self.assertEqual(4, project.call_count)

    def test_initial_motion_settles_before_projection_or_beacon_detection(self):
        task = Mock()
        stable = self.map_frame()
        frames = iter([self.map_frame(shift=5), self.map_frame(shift=11), stable, stable, stable,
                       stable, stable, stable, stable, stable, stable])
        task.frame = self.map_frame(shift=1)

        def advance():
            task.frame = next(frames, stable).copy()

        task.next_frame.side_effect = advance
        navigator = self.navigator_for(task)
        projected = MapPosition(150, 100, .9, 30)
        box = Box(150, 100, 20, 20)
        projected_frames = []

        def project():
            projected_frames.append(task.frame.copy())
            return projected

        with patch.object(navigator, 'project', side_effect=project), \
                patch.object(navigator, 'beacon_at', return_value=box) as beacon_at:
            result = navigator.find()

        self.assertIsNotNone(result)
        self.assertEqual(3, len(projected_frames))
        for frame in projected_frames:
            np.testing.assert_array_equal(stable, frame)
        self.assertEqual(3, beacon_at.call_count)

    def test_drag_animation_settles_before_reprojecting_or_accepting_beacon(self):
        task = Mock()
        stable = self.map_frame()
        moving = [self.map_frame(shift=5), self.map_frame(shift=12), stable, stable, stable]
        state = {'dragging': False, 'index': 0}
        task.frame = stable.copy()

        def advance():
            if state['dragging']:
                task.frame = moving[state['index']].copy()
                state['index'] += 1
                if state['index'] >= len(moving):
                    state['dragging'] = False
            else:
                task.frame = stable.copy()

        task.next_frame.side_effect = advance
        task.swipe.side_effect = lambda *args, **kwargs: state.__setitem__('dragging', True)
        navigator = self.navigator_for(task)
        projection = MapPosition(150, 100, .9, 30)
        observed = []

        def project():
            observed.append(task.frame.copy())
            # Only the first post-drag stable search view matches the bookmark.
            return projection if state['index'] >= len(moving) and task.swipe.call_count else None

        beacon = Box(120, 90, 20, 20)
        with patch.object(navigator, 'project', side_effect=project), \
                patch.object(navigator, 'beacon_at', return_value=beacon):
            result = navigator.find(search=True)

        self.assertIsNotNone(result)
        self.assertEqual(1, task.swipe.call_count)
        self.assertGreaterEqual(len(observed), 5)
        # The two animated frames are never used for a projection or beacon lookup.
        self.assertFalse(any(np.array_equal(frame, moving[0]) or np.array_equal(frame, moving[1])
                             for frame in observed))
        self.assertTrue(np.array_equal(stable, observed[-1]))

    def test_beacon_frame_drift_is_rejected(self):
        task = Mock()
        task.frame = self.map_frame()
        navigator = self.navigator_for(task)
        projected = MapPosition(150, 100, .9, 30)
        with patch.object(navigator, 'project', return_value=projected) as project, \
                patch.object(navigator, 'beacon_at', side_effect=[Box(120, 90, 20, 20),
                                                                   Box(135, 90, 20, 20)]):
            self.assertIsNone(navigator.find())
        self.assertEqual(2, project.call_count)
        task.scroll.assert_not_called()
        task.swipe.assert_not_called()
        task.screenshot.assert_called_once_with('3c-map-search-beacon-drift', frame=task.frame)

    def test_stable_beacon_returns_latest_frame_detection(self):
        task = Mock()
        task.frame = self.map_frame()
        navigator = self.navigator_for(task)
        projected = MapPosition(150, 100, .9, 30)
        first, second, latest = Box(120, 90, 20, 20), Box(121, 90, 20, 20), Box(122, 91, 20, 20)
        with patch.object(navigator, 'project', return_value=projected) as project, \
                patch.object(navigator, 'beacon_at', side_effect=[first, second, latest]) as beacon_at:
            result = navigator.find()
        self.assertIs(result, latest)
        self.assertEqual(3, project.call_count)
        self.assertEqual(3, beacon_at.call_count)

    def test_perpetual_map_motion_times_out_without_searching_or_guessing(self):
        task = Mock()
        task.frame = self.map_frame()
        elapsed = [0.0]
        shift = [1]

        def advance():
            shift[0] += 3
            task.frame = self.map_frame(shift[0])

        task.next_frame.side_effect = advance
        task.sleep.side_effect = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
        navigator = self.navigator_for(task)
        with patch.object(navigator, 'project') as project, \
                patch('src.pathing.RouteMapNavigation.time.monotonic', side_effect=lambda: elapsed[0]):
            self.assertIsNone(navigator.find(search=True))
        project.assert_not_called()
        task.scroll.assert_not_called()
        task.swipe.assert_not_called()
        task.screenshot.assert_called_once_with('3c-map-search-initial-view-unstable', frame=task.frame)
        self.assertGreaterEqual(elapsed[0], 3.0)
        self.assertLess(elapsed[0], 3.2)

    def test_blank_map_frame_never_counts_as_stable(self):
        task = Mock()
        task.frame = np.zeros((240, 400, 3), np.uint8)
        elapsed = [0.0]
        task.sleep.side_effect = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
        navigator = self.navigator_for(task)
        with patch.object(navigator, 'project') as project, \
                patch('src.pathing.RouteMapNavigation.time.monotonic', side_effect=lambda: elapsed[0]):
            self.assertIsNone(navigator.find())
        project.assert_not_called()
        task.screenshot.assert_called_once()


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
        task.frame = self.map_frame()
        navigator = self.navigator_for(task)
        with patch.object(navigator, 'project', return_value=None) as project:
            self.assertIsNone(navigator.find(search=True))
        self.assertEqual(26, project.call_count)  # initial view plus the 25-cell grid
        self.assertEqual(25, task.info_set.call_count)
        self.assertEqual(24, task.swipe.call_count)
        self.assertEqual(24, task.mouse_up.call_count)

    def test_search_stops_at_40_second_budget(self):
        task = Mock()
        task.frame = self.map_frame()
        navigator = self.navigator_for(task)
        elapsed = [0.0]
        task.sleep.side_effect = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)

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
        task.frame = self.map_frame()
        task.swipe.side_effect = TaskDisabledException()
        navigator = self.navigator_for(task)
        with patch.object(navigator, 'project', return_value=None):
            with self.assertRaises(TaskDisabledException):
                navigator.find(search=True)
        task.mouse_up.assert_called_once()

if __name__ == '__main__':
    unittest.main()

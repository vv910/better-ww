import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from ok import Box, TaskDisabledException

from tests.Test3CRouteRuntime import make_task


class Test3CMapEntry(unittest.TestCase):
    def setUp(self):
        self.task = make_task()
        self.clock = 0.0
        self.task.sleep.side_effect = lambda seconds: setattr(self, 'clock', self.clock + seconds)
        self.clock_patch = patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.detector = Mock()
        self.detector.detect.return_value = None
        self.detector.last_diagnostic = {'status': 'rejected', 'reason': 'controls_not_matched'}
        self.detector_patch = patch('src.task.Farm3CEchoTask.RouteMapUI', return_value=self.detector)
        self.detector_type = self.detector_patch.start()
        self.addCleanup(self.detector_patch.stop)

    def route(self, context=None):
        self.task.reset_to_false = Mock()
        self.task.find_legacy_route_beacon = Mock()
        self.task.travel_to_route_beacon = Mock()
        return SimpleNamespace(name='map-entry', map_context=context,
                               teleport=SimpleNamespace(anchor='', views=()))

    def test_legacy_pin_keeps_its_existing_threshold_and_does_not_load_new_templates(self):
        self.task.find_one.side_effect = [Box(10, 10, 20, 20), Box(11, 10, 20, 20)]
        result = self.task.wait_route_map_ui()
        self.assertEqual((21, 20), result.center())
        self.assertEqual(2, self.task.next_frame.call_count)
        self.detector_type.assert_not_called()
        for call in self.task.find_one.call_args_list:
            self.assertEqual(('map_my_location',), call.args)
            self.assertNotIn('threshold', call.kwargs)
            self.assertIn('frame', call.kwargs)
        self.assertIsNone(self.task._last_map_ui_frame)

    def test_new_layout_requires_two_new_frames_and_returns_the_latest_center(self):
        self.task.find_one.return_value = None
        first, latest = Box(10, 10, 20, 20), Box(11, 11, 20, 20)
        self.detector.detect.side_effect = [first, latest]
        inspected = []

        def frame():
            value = 40 + self.task.next_frame.call_count
            self.task._executor.frame = np.full((240, 400, 3), value, np.uint8)
            inspected.append(self.task._executor.frame.copy())

        self.task.next_frame.side_effect = frame
        self.assertIs(latest, self.task.wait_route_map_ui())
        self.assertEqual(2, self.detector.detect.call_count)
        self.detector_type.assert_called_once()
        for call, image in zip(self.detector.detect.call_args_list, inspected):
            np.testing.assert_array_equal(image, call.args[0])
        self.assertEqual('current', self.task._last_map_ui_diagnostic['layout'])
        self.assertIsNone(self.task._last_map_ui_frame)

    def test_one_frame_or_two_drifting_controls_cannot_confirm_map(self):
        for controls in ([Box(10, 10, 20, 20), None],
                         [Box(10, 10, 20, 20), Box(100, 10, 20, 20)]):
            with self.subTest(controls=controls):
                self.clock = 0
                self.task.find_one.side_effect = controls
                self.assertIsNone(self.task.wait_route_map_ui(time_out=.15))
                self.task.send_key.assert_not_called()
                self.assertIsNotNone(self.task._last_map_ui_frame)

    def test_old_and_new_evidence_cannot_be_combined_as_two_matching_frames(self):
        self.task.find_one.side_effect = [Box(10, 10, 20, 20), None]
        self.detector.detect.return_value = Box(10, 10, 20, 20)
        self.assertIsNone(self.task.wait_route_map_ui(time_out=.15))

    def test_late_capture_or_detection_cannot_confirm_a_second_frame_after_budget(self):
        for slow_phase in ('capture', 'detection'):
            with self.subTest(slow_phase=slow_phase):
                self.clock = 0
                self.task = make_task()
                self.task.sleep.side_effect = lambda seconds: setattr(self, 'clock', self.clock + seconds)

                def capture():
                    if slow_phase == 'capture' and self.task.next_frame.call_count == 2:
                        self.clock += 4

                def detect(*args, **kwargs):
                    if slow_phase == 'detection' and self.task.find_one.call_count == 2:
                        self.clock += 4
                    return Box(10, 10, 20, 20)

                self.task.next_frame.side_effect = capture
                self.task.find_one.side_effect = detect
                self.assertIsNone(self.task.wait_route_map_ui(time_out=3))
                self.assertEqual('ui_confirmation_budget_exhausted',
                                 self.task._last_map_ui_diagnostic['reason'])

    def test_failed_entry_saves_the_actual_last_frame_after_executor_storage_was_cleared(self):
        route = self.route()
        self.task.find_one.return_value = None
        self.task._navigation_replay = Mock()
        last = []

        def frame():
            image = np.full((240, 400, 3), self.task.next_frame.call_count, np.uint8)
            self.task._executor.frame = image
            last[:] = [image.copy()]

        def sleep(seconds):
            self.clock += seconds
            self.task._executor.frame = None

        self.task.next_frame.side_effect = frame
        self.task.sleep.side_effect = sleep
        with self.assertRaisesRegex(RuntimeError, 'Cannot open map for route entry'):
            self.task.teleport_to_route(route)
        self.task.send_key.assert_called_once_with('m', after_sleep=.5)
        self.task.travel_to_route_beacon.assert_not_called()
        self.task.screenshot.assert_called_once()
        self.assertEqual('3c-map-entry-failed', self.task.screenshot.call_args.args[0])
        np.testing.assert_array_equal(last[0], self.task.screenshot.call_args.kwargs['frame'])
        replay = self.task._navigation_replay.record_map_ui
        replay.assert_called_once()
        np.testing.assert_array_equal(last[0], replay.call_args.args[0])
        self.assertEqual('route-entry', replay.call_args.args[1])
        self.assertIsNone(self.task._last_map_ui_frame)

    def test_capture_or_diagnostic_write_failure_cannot_replace_the_entry_error(self):
        for failure in ('capture', 'screenshot', 'replay'):
            with self.subTest(failure=failure):
                self.clock = 0
                self.task = make_task()
                self.task.sleep.side_effect = lambda seconds: setattr(self, 'clock', self.clock + seconds)
                route = self.route()
                self.task.find_one.return_value = None
                self.task._navigation_replay = Mock()
                if failure == 'capture':
                    self.task.next_frame.side_effect = RuntimeError('capture missing')
                elif failure == 'screenshot':
                    self.task.screenshot.side_effect = OSError('disk full')
                else:
                    self.task._navigation_replay.record_map_ui.side_effect = OSError('disk full')
                with self.assertRaisesRegex(RuntimeError, 'Cannot open map for route entry'):
                    self.task.teleport_to_route(route)
                self.assertIsNone(self.task._last_map_ui_frame)

    def test_ui_wait_cancellation_releases_cached_image_without_retrying_map_key(self):
        route = self.route()
        self.task.next_frame.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            self.task.teleport_to_route(route)
        self.task.send_key.assert_called_once_with('m', after_sleep=.5)
        self.task.screenshot.assert_not_called()
        self.assertIsNone(self.task._last_map_ui_frame)

    def test_diagnostic_cancellation_is_not_swallowed(self):
        for component in ('screenshot', 'replay'):
            with self.subTest(component=component):
                self.clock = 0
                self.task = make_task()
                self.task.sleep.side_effect = lambda seconds: setattr(self, 'clock', self.clock + seconds)
                route = self.route()
                self.task.find_one.return_value = None
                self.task._navigation_replay = Mock()
                if component == 'screenshot':
                    self.task.screenshot.side_effect = TaskDisabledException()
                else:
                    self.task._navigation_replay.record_map_ui.side_effect = TaskDisabledException()
                with self.assertRaises(TaskDisabledException):
                    self.task.teleport_to_route(route)
                self.assertIsNone(self.task._last_map_ui_frame)

    def test_failed_post_selection_map_confirmation_is_recorded_before_teleport(self):
        context = object()
        route = self.route(context)
        self.task.wait_route_map_ui = Mock(side_effect=[Box(10, 10, 20, 20), None])
        self.task._navigation_replay = Mock()
        with patch('src.task.Farm3CEchoTask.RouteMapSelector') as selector, \
                self.assertRaisesRegex(RuntimeError, 'Map selector did not return'):
            self.task.teleport_to_route(route)
        selector.return_value.select.assert_called_once_with(context)
        self.task.travel_to_route_beacon.assert_not_called()
        self.task.screenshot.assert_called_once()
        self.assertEqual('3c-map-context-not-closed', self.task.screenshot.call_args.args[0])
        self.assertEqual('map-context', self.task._navigation_replay.record_map_ui.call_args.args[1])


if __name__ == '__main__':
    unittest.main()

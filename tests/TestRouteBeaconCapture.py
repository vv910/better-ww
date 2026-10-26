import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2
from ok import Box, TaskDisabledException

from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.RouteBeaconCapture import capture_beacon_location, close_beacon_details
from src.pathing.RouteBeaconLocation import RouteBeaconLocation


class BeaconTask:
    def __init__(self):
        self.clock = 0.
        self.stage = 1  # ordinary map=1, beacon details=2, unknown/world=3
        self.frame = self.make_frame(1)
        self.click_effective = True
        self.capture_error = None
        self.capture_none = False
        self.detail_reads = 0
        self.mouse_up = Mock()
        self.click = Mock(side_effect=self._click)
        self.send_key = Mock(side_effect=self._key)
        self.find_feature = Mock(return_value=[Box(370, 170, 20, 20)])
        self.find_one = Mock(return_value=None)
        self.sleep = Mock(side_effect=self._sleep)
        self.next_frame = Mock(side_effect=self._capture)
        self.wait_route_map_ui = Mock(side_effect=lambda **kwargs: self.stage == 1)
        self.executor = SimpleNamespace(next_frame=Mock(side_effect=self._capture),
                                        sleep=Mock(side_effect=self._sleep))
        # None of these operations belongs to reading a location.
        self.click_box = Mock(side_effect=AssertionError('travel/remove button clicked'))
        self.click_traval_button = Mock(side_effect=AssertionError('generic travel helper used'))
        self.travel_to_route_beacon = Mock(side_effect=AssertionError('travel started'))

    @staticmethod
    def make_frame(stage):
        y, x = np.indices((360, 640))
        gray = ((x * 17 + y * 31) % 190 + 30).astype(np.uint8)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        frame[0, 0] = stage
        return frame

    def _capture(self, **kwargs):
        self.clock += .04
        if self.capture_error:
            raise self.capture_error
        if self.capture_none:
            return None
        self.frame = self.make_frame(self.stage)
        return self.frame

    def _sleep(self, seconds):
        self.clock += seconds

    def _click(self, box, **kwargs):
        if self.click_effective:
            self.stage = 2

    def _key(self, key, **kwargs):
        if key != 'esc':
            raise AssertionError('unexpected input')
        self.stage = 1


class TestRouteBeaconCapture(unittest.TestCase):
    def setUp(self):
        self.task = BeaconTask()
        self.reference = self.task.frame.copy()
        self.reference_point = [300., 180.]
        self.region = (179, 43, 557, 324)
        self.location = RouteBeaconLocation('瑝珑', '梦枢天罗', '相心域', '瑝珑-梦枢天罗-相心域')
        self.projected = SimpleNamespace(x=380., y=180., xy=(380., 180.), inside_view=True)
        self.localizer = Mock()
        self.localizer.register_map_view.return_value = self.projected
        self.localizer.last_diagnostic = {'status': 'matched', 'reason': 'matched'}
        patches = [
            patch('src.pathing.RouteBeaconCapture.time.monotonic', side_effect=lambda: self.task.clock),
            patch('src.pathing.RouteBeaconCapture.MapLocalizer', return_value=self.localizer),
            patch('src.pathing.RouteBeaconCapture.RouteMapUI'),
            patch('src.pathing.RouteBeaconCapture.read_beacon_location', side_effect=self.read),
        ]
        self.patches = [value.start() for value in patches]
        for value in patches:
            self.addCleanup(value.stop)
        self.patches[2].return_value.detect.side_effect = lambda frame: object() if frame[0, 0, 0] == 1 else None

    def read(self, task, frame, *, diagnostic):
        diagnostic.clear()
        diagnostic['header_confirmed'] = frame[0, 0, 0] == 2
        if not diagnostic['header_confirmed']:
            diagnostic['reason'] = 'beacon_header_not_found'
            return None
        self.task.detail_reads += 1
        diagnostic.update(reason='beacon_detail_location_read', raw=self.location.raw)
        return self.location

    def capture(self, **kwargs):
        return capture_beacon_location(self.task, self.reference, self.reference_point, self.region, **kwargs)

    def assert_no_travel_or_marker_edits(self):
        self.task.click_box.assert_not_called()
        self.task.click_traval_button.assert_not_called()
        self.task.travel_to_route_beacon.assert_not_called()
        for call in self.task.find_feature.call_args_list:
            self.assertEqual(['map_way_point', 'map_way_point_big'], call.args[0])

    def test_live_projection_is_used_once_and_original_pixels_and_points_stay_unchanged(self):
        snapshot = self.reference.copy()
        result = self.capture()
        self.assertIs(self.location, result)
        self.assertEqual([300., 180.], self.reference_point)
        np.testing.assert_array_equal(snapshot, self.reference)
        self.localizer.register_map_view.assert_called_once()
        self.assertEqual(self.reference_point, self.localizer.register_map_view.call_args.args[1])
        self.task.click.assert_called_once()
        self.assertEqual((380, 180), self.task.click.call_args.args[0].center())
        self.assertNotEqual(tuple(self.reference_point), self.task.click.call_args.args[0].center())
        self.assertGreaterEqual(self.task.next_frame.call_count, 5)
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)
        self.assertEqual('matched', self.task._last_route_beacon_location_diagnostic['status'])
        self.assertEqual('restored', self.task._last_route_beacon_cleanup_diagnostic['status'])
        self.assert_no_travel_or_marker_edits()

    def test_matching_parts_allow_different_raw_separator_text(self):
        other = RouteBeaconLocation(*self.location.parts, raw='瑝珑—梦枢天罗—相心域')
        first = [True]

        def read(task, frame, *, diagnostic):
            result = self.read(task, frame, diagnostic=diagnostic)
            if result is not None and first[0]:
                first[0] = False
                return other
            return result

        self.patches[3].side_effect = read
        self.assertIs(self.location, self.capture())

    def test_real_terrain_registration_uses_the_moved_live_beacon(self):
        self.reference = cv2.imread('tests/images/path.png')
        self.assertIsNotNone(self.reference)
        self.reference_point = [1207., 469.]
        self.region = (538, 130, 1670, 972)
        height, width = self.reference.shape[:2]
        live = cv2.warpAffine(self.reference, np.float32([[1, 0, -200], [0, 1, 20]]), (width, height))

        def capture(**kwargs):
            self.task.clock += .04
            self.task.frame = live.copy()
            self.task.frame[0, 0] = self.task.stage
            return self.task.frame

        self.task.next_frame.side_effect = capture
        self.task.executor.next_frame.side_effect = capture
        capture()
        self.task.find_feature.return_value = [Box(997, 479, 20, 20)]
        self.patches[1].side_effect = MapLocalizer
        self.assertIs(self.location, self.capture())
        self.assertEqual((1007, 489), self.task.click.call_args.args[0].center())
        registration = self.task._last_route_beacon_location_diagnostic['registration']
        self.assertGreaterEqual(registration['inliers'], 12)
        self.assertGreaterEqual(registration['confidence'], .6)
        self.assert_no_travel_or_marker_edits()

    def test_unregistered_current_view_after_an_unsaved_overview_never_uses_old_point(self):
        self.localizer.register_map_view.return_value = None
        self.assertIsNone(self.capture())
        self.task.click.assert_not_called()
        self.task.send_key.assert_not_called()
        self.assertEqual('current_view_not_registered', self.task._last_route_beacon_location_diagnostic['reason'])

    def test_offscreen_or_unsafe_projection_never_authorizes_a_click(self):
        for x, inside in ((900, False), (180, True)):
            with self.subTest(x=x):
                self.localizer.register_map_view.return_value = SimpleNamespace(
                    x=x, y=180, xy=(x, 180), inside_view=inside)
                self.assertIsNone(self.capture())
                self.task.click.assert_not_called()
                self.task.send_key.assert_not_called()

    def test_missing_or_ambiguous_beacons_never_authorize_any_click(self):
        for matches in ([], [Box(370, 170, 20, 20), Box(380, 170, 20, 20)]):
            with self.subTest(matches=len(matches)):
                self.task.find_feature.return_value = matches
                self.assertIsNone(self.capture())
                self.task.click.assert_not_called()
                self.task.send_key.assert_not_called()

    def test_small_and_large_templates_for_the_same_beacon_are_deduplicated(self):
        self.task.find_feature.return_value = [Box(370, 170, 20, 20), Box(369, 169, 22, 22)]
        self.assertIs(self.location, self.capture())
        self.task.click.assert_called_once()

    def test_map_ui_or_motion_failure_stops_before_registration(self):
        self.task.wait_route_map_ui.return_value = None
        self.task.wait_route_map_ui.side_effect = None
        self.assertIsNone(self.capture())
        self.localizer.register_map_view.assert_not_called()
        self.task.click.assert_not_called()

    def test_animation_never_supplies_a_stable_frame_for_clicking(self):
        shift = [0]

        def moving_capture():
            self.task.clock += .04
            shift[0] += 5
            self.task.frame = np.roll(self.task.make_frame(1), shift[0], axis=1)
            return self.task.frame

        self.task.next_frame.side_effect = moving_capture
        self.assertIsNone(self.capture())
        self.localizer.register_map_view.assert_not_called()
        self.task.click.assert_not_called()
        self.task.send_key.assert_not_called()

    def test_alternating_location_parts_are_not_two_confirmed_observations(self):
        other = RouteBeaconLocation('瑝珑', 'Other Map', 'Other Area', '瑝珑-Other Map-Other Area')
        count = [0]

        def alternate(task, frame, *, diagnostic):
            result = self.read(task, frame, diagnostic=diagnostic)
            count[0] += 1
            return other if result is not None and count[0] % 2 else result

        self.patches[3].side_effect = alternate
        self.assertIsNone(self.capture(time_out=.8))
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)

    def test_late_registration_or_icon_detection_cannot_send_expired_input(self):
        for phase in ('registration', 'icon'):
            with self.subTest(phase=phase):
                self.task.clock = 0

                def slow(*args, **kwargs):
                    self.task.clock = 10.1
                    return self.projected if phase == 'registration' else [Box(370, 170, 20, 20)]

                self.localizer.register_map_view.side_effect = slow if phase == 'registration' else None
                self.task.find_feature.side_effect = slow if phase == 'icon' else None
                self.assertIsNone(self.capture())
                self.task.click.assert_not_called()
                self.task.send_key.assert_not_called()

    def test_late_second_ocr_result_is_not_returned_but_details_are_closed(self):
        calls = [0]

        def read(task, frame, *, diagnostic):
            result = self.read(task, frame, diagnostic=diagnostic)
            calls[0] += 1
            if calls[0] == 2:
                self.task.clock = 10.1
            return result

        self.patches[3].side_effect = read
        self.assertIsNone(self.capture())
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)
        self.assertEqual('restored', self.task._last_route_beacon_cleanup_diagnostic['status'])

    def test_incomplete_ocr_returns_none_after_budget_and_still_restores_map(self):
        def incomplete(task, frame, *, diagnostic):
            self.read(task, frame, diagnostic=diagnostic)
            diagnostic['reason'] = 'complete_location_not_found'
            return None

        self.patches[3].side_effect = incomplete
        self.assertIsNone(self.capture(time_out=.8))
        self.task.click.assert_called_once()
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)
        self.assertEqual('restored', self.task._last_route_beacon_cleanup_diagnostic['status'])

    def test_click_with_no_effect_is_not_followed_by_escape(self):
        self.task.click_effective = False
        self.assertIsNone(self.capture(time_out=.8))
        self.task.click.assert_called_once()
        self.task.send_key.assert_not_called()
        self.assertEqual('restored', self.task._last_route_beacon_cleanup_diagnostic['status'])

    def test_unknown_popup_or_world_does_not_authorize_escape_and_prevents_saving(self):
        self.task.click.side_effect = lambda *args, **kwargs: setattr(self.task, 'stage', 3)
        with self.assertRaisesRegex(RuntimeError, 'Cannot restore map'):
            self.capture(time_out=.8)
        self.task.send_key.assert_not_called()
        self.assertEqual('unconfirmed', self.task._last_route_beacon_cleanup_diagnostic['status'])
        self.assert_no_travel_or_marker_edits()

    def test_cancellation_after_trusted_header_restores_map_and_preserves_original_exception(self):
        cancelled = TaskDisabledException('cancel detail read')
        first = [True]

        def read(task, frame, *, diagnostic):
            result = self.read(task, frame, diagnostic=diagnostic)
            if result is not None and first[0]:
                first[0] = False
                raise cancelled
            return result

        self.patches[3].side_effect = read
        with self.assertRaises(TaskDisabledException) as caught:
            self.capture()
        self.assertIs(cancelled, caught.exception)
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)
        self.assert_no_travel_or_marker_edits()

    def test_cancellation_preventing_cleanup_capture_can_only_close_already_confirmed_details(self):
        cancelled = TaskDisabledException('disabled')

        def read(task, frame, *, diagnostic):
            self.read(task, frame, diagnostic=diagnostic)
            self.task.executor.next_frame.side_effect = cancelled
            raise cancelled

        self.patches[3].side_effect = read
        with self.assertRaises(TaskDisabledException) as caught:
            self.capture()
        self.assertIs(cancelled, caught.exception)
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)
        self.assertIn('restoration unconfirmed', '\n'.join(cancelled.__notes__))

    def test_cancellation_before_click_never_sends_cleanup_escape(self):
        cancelled = TaskDisabledException('before click')
        self.task.wait_route_map_ui.side_effect = cancelled
        with self.assertRaises(TaskDisabledException) as caught:
            self.capture()
        self.assertIs(cancelled, caught.exception)
        self.task.click.assert_not_called()
        self.task.send_key.assert_not_called()

    def test_cancelled_capture_cannot_use_an_old_header_after_fresh_unknown_cleanup_frame(self):
        cancelled = TaskDisabledException('cancelled')

        def read(task, frame, *, diagnostic):
            result = self.read(task, frame, diagnostic=diagnostic)
            if result is not None:
                self.task.stage = 3
                raise cancelled
            return result

        self.patches[3].side_effect = read
        with self.assertRaises(TaskDisabledException) as caught:
            self.capture()
        self.assertIs(cancelled, caught.exception)
        self.task.send_key.assert_not_called()
        self.assertGreater(self.task._last_route_beacon_cleanup_diagnostic['fresh_frames'], 0)

    def test_late_fresh_unknown_cleanup_frame_also_blocks_old_header_escape(self):
        cancelled = TaskDisabledException('cancelled')

        def read(task, frame, *, diagnostic):
            self.read(task, frame, diagnostic=diagnostic)

            def late_capture(**kwargs):
                self.task.clock += 3.1
                return self.task.make_frame(3)

            self.task.executor.next_frame.side_effect = late_capture
            raise cancelled

        self.patches[3].side_effect = read
        with self.assertRaises(TaskDisabledException) as caught:
            self.capture()
        self.assertIs(cancelled, caught.exception)
        self.task.send_key.assert_not_called()
        self.assertEqual(1, self.task._last_route_beacon_cleanup_diagnostic['fresh_frames'])

    def test_cleanup_already_ordinary_requires_three_fresh_settled_frames_and_no_escape(self):
        close_beacon_details(self.task)
        self.assertEqual(3, self.task.executor.next_frame.call_count)
        self.task.send_key.assert_not_called()
        self.patches[3].assert_not_called()
        self.assertTrue(all(call.kwargs['time_out'] <= .5
                            for call in self.task.executor.next_frame.call_args_list))

    def test_public_cleanup_cancellation_during_location_ocr_closes_its_fresh_trusted_header(self):
        self.task.stage = 2
        cancelled = TaskDisabledException('cancelled in cleanup location OCR')

        def cancelled_read(task, frame, *, diagnostic):
            diagnostic['header_confirmed'] = True
            raise cancelled

        self.patches[3].side_effect = cancelled_read
        with self.assertRaises(TaskDisabledException) as caught:
            close_beacon_details(self.task)
        self.assertIs(cancelled, caught.exception)
        self.task.send_key.assert_called_once_with('esc', after_sleep=0)
        self.assertEqual('cancelled', self.task._last_route_beacon_cleanup_diagnostic['status'])

    def test_public_cleanup_cancellation_before_header_does_not_guess_escape(self):
        self.task.stage = 2
        cancelled = TaskDisabledException('cancelled before header OCR')
        self.patches[3].side_effect = cancelled
        with self.assertRaises(TaskDisabledException) as caught:
            close_beacon_details(self.task)
        self.assertIs(cancelled, caught.exception)
        self.task.send_key.assert_not_called()

    def test_cleanup_unavailable_capture_or_unresponsive_escape_is_bounded(self):
        for phase in ('no_capture', 'escape_no_effect'):
            with self.subTest(phase=phase):
                self.task.clock = 0
                self.task.stage = 2
                self.task.send_key.reset_mock()
                self.task.capture_none = phase == 'no_capture'
                self.task.send_key.side_effect = lambda *args, **kwargs: None
                with self.assertRaisesRegex(RuntimeError, 'Cannot restore map'):
                    close_beacon_details(self.task, time_out=.5)
                self.assertLessEqual(self.task.clock, .6)
                self.assertEqual(0 if phase == 'no_capture' else 1, self.task.send_key.call_count)

    def test_slow_cleanup_detector_does_not_start_ocr_or_send_escape_after_budget(self):
        self.task.stage = 2

        def slow_detector(frame):
            self.task.clock += 3.1
            return None

        self.patches[2].return_value.detect.side_effect = slow_detector
        with self.assertRaisesRegex(RuntimeError, 'Cannot restore map'):
            close_beacon_details(self.task)
        self.patches[3].assert_not_called()
        self.task.find_one.assert_not_called()
        self.task.send_key.assert_not_called()

    def test_changed_capture_size_does_not_click_or_mutate_recorded_reference(self):
        self.reference = self.reference[:300]
        self.assertIsNone(self.capture())
        self.task.click.assert_not_called()
        self.task.send_key.assert_not_called()

    def test_invalid_budgets_fail_before_input(self):
        for budget in (0, -1, float('nan'), float('inf'), True, '10'):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    self.capture(time_out=budget)
                with self.assertRaises(ValueError):
                    close_beacon_details(self.task, time_out=budget)
        self.task.click.assert_not_called()
        self.task.send_key.assert_not_called()


if __name__ == '__main__':
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from ok import Box

from src.pathing.RouteMapNavigation import RouteMapNavigator


def registration(x=350, y=180, scale=1, inliers=40):
    return SimpleNamespace(x=float(x), y=float(y), xy=(float(x), float(y)),
                           scale=scale, inliers=inliers, confidence=.95)


class TestClosedMapSearch(unittest.TestCase):
    def setUp(self):
        self.clock = 0.
        self.clock_patch = patch('src.pathing.RouteMapNavigation.time.monotonic',
                                 side_effect=lambda: self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.task = Mock()
        y, x = np.indices((360, 640))
        gray = ((x * 17 + y * 31) % 190 + 30).astype(np.uint8)
        self.task.frame = np.repeat(gray[:, :, None], 3, axis=2)
        self.task.width, self.task.height = 640, 360
        self.task.wait_route_map_ui.return_value = Box(570, 10, 20, 20)
        self.task.sleep.side_effect = lambda value: setattr(self, 'clock', self.clock + value)
        self.navigator = RouteMapNavigator.__new__(RouteMapNavigator)
        self.navigator.task = self.task
        self.navigator.references = []
        self.navigator.reference_names = []
        self.navigator.last_diagnostic = {}
        self.navigator.last_frame = None
        self.navigator._history = []
        self.navigator._popup_recoveries = 0

    def change_frame_on_scroll(self):
        self.task.scroll.side_effect = lambda *args: setattr(
            self.task, 'frame', np.roll(self.task.frame, 9, axis=1))

    def test_popup_is_dismissed_once_then_normal_controls_and_fresh_stability_are_required(self):
        self.task.wait_route_map_ui.side_effect = [None, Box(570, 10, 20, 20)]
        self.navigator._known_popup_visible = Mock(return_value=True)
        self.assertTrue(self.navigator._ready('initial-view', 40))
        self.task.send_key.assert_called_once_with('esc', after_sleep=.2)
        self.assertGreaterEqual(self.task.next_frame.call_count, 3)
        self.assertEqual(2, self.task.wait_route_map_ui.call_count)
        # A second lost UI is not permission to press Esc indefinitely.
        self.task.wait_route_map_ui.side_effect = None
        self.task.wait_route_map_ui.return_value = None
        self.assertFalse(self.navigator._ready('zoom-view', 40))
        self.task.send_key.assert_called_once()
        self.task.scroll.assert_not_called()
        self.task.swipe.assert_not_called()

    def test_unrecoverable_popup_stops_before_projection_scroll_or_drag(self):
        self.task.wait_route_map_ui.return_value = None
        self.navigator._known_popup_visible = Mock(return_value=True)
        with patch.object(self.navigator, 'project') as project, \
                patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
            self.assertIsNone(self.navigator.find(search=True))
        project.assert_not_called()
        self.task.send_key.assert_called_once_with('esc', after_sleep=.2)
        self.task.scroll.assert_not_called()
        drag.assert_not_called()
        self.assertEqual('map_ui_not_recovered', self.navigator.last_diagnostic['reason'])

    def test_missing_controls_without_popup_never_closes_a_normal_map(self):
        self.task.wait_route_map_ui.return_value = None
        self.navigator._known_popup_visible = Mock(return_value=False)
        with patch.object(self.navigator, 'project') as project:
            self.assertIsNone(self.navigator.find(search=True))
        self.task.send_key.assert_not_called()
        self.task.scroll.assert_not_called()
        project.assert_not_called()
        self.assertEqual('map_controls_not_confirmed', self.navigator.last_diagnostic['reason'])

    def test_transient_control_miss_is_rechecked_without_escape(self):
        self.task.wait_route_map_ui.side_effect = [None, Box(570, 10, 20, 20)]
        self.navigator._known_popup_visible = Mock(return_value=False)
        self.assertTrue(self.navigator._ready('zoom-view', 40))
        self.task.send_key.assert_not_called()
        self.assertEqual('map_controls_reconfirmed', self.navigator.last_diagnostic['reason'])

    def test_current_view_probe_never_pans_or_zooms_unknown_offscreen_or_missing_beacons(self):
        for projected in (None, registration(900, 180), registration()):
            with self.subTest(projected=projected), \
                    patch.object(self.navigator, 'project', return_value=projected), \
                    patch.object(self.navigator, 'beacon_at', return_value=None), \
                    patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
                self.assertIsNone(self.navigator.find(current_view_only=True))
                self.task.scroll.assert_not_called()
                drag.assert_not_called()

    def test_current_view_probe_has_a_six_second_budget(self):
        def slow_project():
            self.clock = 6.1
            return registration()

        with patch.object(self.navigator, 'project', side_effect=slow_project), \
                patch.object(self.navigator, 'beacon_at') as beacon:
            self.assertIsNone(self.navigator.find(current_view_only=True))
        beacon.assert_not_called()
        self.task.scroll.assert_not_called()

    def test_popup_detection_uses_the_last_inspected_frame_and_exact_header(self):
        frame = self.task.frame.copy()
        self.task._last_map_ui_frame = frame
        for label, confidence, expected in [('自定义标记', .99, True), ('Custom Marker', .9, True),
                                             ('自定义标记说明', .99, False), ('自定义标记', .79, False),
                                             ('自定义标记', float('nan'), False)]:
            with self.subTest(label=label, confidence=confidence):
                self.task.ocr.return_value = [Box(450, 10, 90, 20, name=label, confidence=confidence)]
                self.assertEqual(expected, self.navigator._known_popup_visible())
                self.assertIs(frame, self.task.ocr.call_args.kwargs['frame'])

    def test_old_popup_frame_cannot_authorize_escape_after_a_new_capture_failure(self):
        self.navigator.last_frame = self.task.frame.copy()
        self.task._last_map_ui_frame = None
        self.task.ocr.return_value = [Box(450, 10, 90, 20, name='自定义标记', confidence=.99)]
        self.task.wait_route_map_ui.return_value = None
        self.assertFalse(self.navigator._ready('zoom-view', 40))
        self.task.ocr.assert_not_called()
        self.task.send_key.assert_not_called()

    def test_no_registration_never_causes_extreme_zoom_or_blind_drag(self):
        self.change_frame_on_scroll()
        with patch.object(self.navigator, 'project', return_value=None), \
                patch.object(self.navigator, '_measure_change', return_value=registration(scale=1.06)), \
                patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
            self.assertIsNone(self.navigator.find(search=True))
        # The first reversal is rejected because the observed scale did not
        # reverse. Every actual wheel command was still just two notches.
        amounts = [call.args[2] for call in self.task.scroll.call_args_list]
        self.assertEqual([-2] * 6 + [2], amounts)
        self.assertNotIn(-16, amounts)
        drag.assert_not_called()
        self.assertEqual('unexpected_zoom_direction', self.navigator.last_diagnostic['reason'])

    def test_two_unchanged_views_bound_each_zoom_direction(self):
        with patch.object(self.navigator, 'project', return_value=None), \
                patch.object(self.navigator, '_measure_change') as measure, \
                patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
            self.assertIsNone(self.navigator.find(search=True))
        self.assertEqual([-2, -2, 2, 2], [call.args[2] for call in self.task.scroll.call_args_list])
        measure.assert_not_called()
        drag.assert_not_called()

    def test_zoom_measures_actual_scale_and_rejects_the_wrong_direction(self):
        self.change_frame_on_scroll()
        with patch.object(self.navigator, '_measure_change', return_value=registration(scale=.95)), \
                patch.object(self.navigator, 'project') as project:
            result, state = self.navigator._zoom(-2, 40)
        self.assertIsNone(result)
        self.assertEqual('stopped', state)
        self.assertEqual('unexpected_zoom_direction', self.navigator.last_diagnostic['reason'])
        project.assert_not_called()

    def test_zoom_without_verified_adjacent_geometry_stops(self):
        self.change_frame_on_scroll()
        with patch.object(self.navigator, '_measure_change', return_value=None), \
                patch.object(self.navigator, 'project') as project:
            result, state = self.navigator._zoom(-2, 40)
        self.assertIsNone(result)
        self.assertEqual('stopped', state)
        project.assert_not_called()

    def test_strong_offscreen_registration_allows_only_directed_drag_and_rechecks_progress(self):
        old, new = registration(900, 180), registration(800, 180)
        with patch('src.pathing.RouteMapNavigation.drag_route_map') as drag, \
                patch.object(self.navigator, 'project', return_value=new) as project:
            result = self.navigator._recenter(old, 40)
        self.assertIs(result, new)
        start, end = drag.call_args.args[1:3]
        self.assertLess(end[0], start[0])
        self.assertAlmostEqual(end[1], start[1], delta=1)
        self.assertTrue(0 <= end[0] < self.task.width)
        project.assert_called_once()
        self.assertEqual('drag_verified', self.navigator.last_diagnostic['reason'])

    def test_drag_no_progress_wrong_direction_or_lost_registration_stops(self):
        for new in (registration(900, 180), registration(930, 180), None):
            with self.subTest(result=new), \
                    patch('src.pathing.RouteMapNavigation.drag_route_map') as drag, \
                    patch.object(self.navigator, 'project', return_value=new):
                self.assertIsNone(self.navigator._recenter(registration(900, 180), 40))
                drag.assert_called_once()
                self.assertIn(self.navigator.last_diagnostic['reason'],
                              ('drag_no_progress', 'registration_lost_after_drag'))

    def test_slow_projection_cannot_authorize_scroll_or_drag_after_deadline(self):
        def slow_project():
            self.clock = 40.1
            return registration(900, 180)
        with patch.object(self.navigator, 'project', side_effect=slow_project), \
                patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
            self.assertIsNone(self.navigator.find(search=True))
        self.task.scroll.assert_not_called()
        drag.assert_not_called()
        self.assertEqual('budget_exhausted', self.navigator.last_diagnostic['reason'])

    def test_slow_beacon_cv_cannot_send_an_expired_zoom(self):
        def slow_beacon(_):
            self.clock = 40.1
            return None
        with patch.object(self.navigator, 'project', return_value=registration()), \
                patch.object(self.navigator, 'beacon_at', side_effect=slow_beacon):
            self.assertIsNone(self.navigator.find())
        self.task.scroll.assert_not_called()
        self.task.swipe.assert_not_called()

    def test_slow_last_beacon_confirmation_is_not_accepted_after_deadline(self):
        beacon = Box(340, 170, 20, 20)
        calls = [0]
        def detect(_):
            calls[0] += 1
            if calls[0] == 3:
                self.clock = 40.1
            return beacon
        with patch.object(self.navigator, 'project', return_value=registration()), \
                patch.object(self.navigator, 'beacon_at', side_effect=detect):
            self.assertIsNone(self.navigator.find())
        self.assertEqual(3, calls[0])
        self.task.scroll.assert_not_called()

    def test_expired_operation_helpers_do_not_mutate_the_game(self):
        self.clock = 40.1
        self.assertEqual((None, 'stopped'), self.navigator._zoom(-2, 40))
        with patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
            self.assertIsNone(self.navigator._recenter(registration(900, 180), 40))
        self.task.scroll.assert_not_called()
        drag.assert_not_called()

    def test_projection_reuses_only_this_frames_features_and_rejects_reference_conflict(self):
        shared = object()
        first, second = Mock(), Mock()
        first.last_diagnostic = {'status': 'matched', 'reason': 'matched'}
        second.last_diagnostic = {'status': 'matched', 'reason': 'matched'}
        first.prepare_map_features.return_value = shared
        first.register_map_view.return_value = registration(350, 180, inliers=40)
        second.register_map_view.return_value = registration(410, 180, inliers=50)
        self.navigator.references = [(first, SimpleNamespace(x=100, y=100)),
                                     (second, SimpleNamespace(x=200, y=200))]
        self.navigator.reference_names = ['first', 'second']
        self.assertIsNone(self.navigator.project())
        first.prepare_map_features.assert_called_once()
        second.prepare_map_features.assert_not_called()
        self.assertIs(shared, first.register_map_view.call_args.kwargs['features'])
        self.assertIs(shared, second.register_map_view.call_args.kwargs['features'])
        self.assertEqual('reference_disagreement', self.navigator.last_diagnostic['reason'])
        shared_next_frame = object()
        first.prepare_map_features.return_value = shared_next_frame
        second.register_map_view.return_value = registration(351, 180, inliers=50)
        self.task.frame = self.task.frame.copy()
        self.assertIsNotNone(self.navigator.project())
        self.assertEqual(2, first.prepare_map_features.call_count)
        self.assertIs(shared_next_frame, second.register_map_view.call_args.kwargs['features'])

    def test_small_and_large_templates_for_one_beacon_are_deduplicated_but_two_beacons_are_not(self):
        projected = registration()
        self.task.find_feature.return_value = [Box(340, 170, 20, 20), Box(339, 169, 22, 22)]
        self.assertIsNotNone(self.navigator.beacon_at(projected))
        self.task.find_feature.return_value = [Box(340, 170, 20, 20), Box(350, 170, 20, 20)]
        self.assertIsNone(self.navigator.beacon_at(projected))
        self.assertEqual('ambiguous_beacon_candidates', self.navigator.last_diagnostic['reason'])

    def test_ui_observations_are_added_to_replay_diagnostics(self):
        with patch.object(self.navigator, 'project', return_value=None):
            self.assertIsNone(self.navigator.find())
        self.task._record_navigation.assert_called()
        event = self.task._record_navigation.call_args
        self.assertEqual('record_map_ui', event.args[0])
        self.assertEqual('bookmark:stopped', event.args[2])
        self.assertEqual('registration_failed', event.kwargs['diagnostic']['reason'])


    def test_trace_overhead_cannot_start_input_after_the_deadline(self):
        for action in ('zoom', 'drag'):
            with self.subTest(action=action):
                self.clock = 39.9
                with patch.object(self.navigator, '_trace',
                                  side_effect=lambda *args, **kwargs: setattr(self, 'clock', 40.1)), \
                        patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
                    if action == 'zoom':
                        self.assertEqual((None, 'stopped'), self.navigator._zoom(-2, 40))
                    else:
                        self.assertIsNone(self.navigator._recenter(registration(x=800), 40))
                drag.assert_not_called()
                self.task.scroll.assert_not_called()

    def test_drag_does_not_start_when_remaining_budget_is_shorter_than_the_gesture(self):
        self.clock = 39.8
        with patch('src.pathing.RouteMapNavigation.drag_route_map') as drag:
            self.assertIsNone(self.navigator._recenter(registration(x=800), 40))
        drag.assert_not_called()
        self.assertEqual('insufficient_drag_budget', self.navigator.last_diagnostic['reason'])

    def test_conflicting_references_stop_initial_search_or_the_next_zoom_iteration(self):
        for conflict_after_zoom in (False, True):
            with self.subTest(conflict_after_zoom=conflict_after_zoom):
                self.clock = 0
                self.task.scroll.reset_mock()
                self.change_frame_on_scroll()
                calls = []

                def project():
                    calls.append(1)
                    reason = ('reference_disagreement' if not conflict_after_zoom or len(calls) > 1
                              else 'registration_failed')
                    self.navigator.last_diagnostic = {'reason': reason}
                    return None

                with patch.object(self.navigator, '_ready', return_value=True), \
                        patch.object(self.navigator, '_measure_change', return_value=registration(scale=1.1)), \
                        patch.object(self.navigator, 'project', side_effect=project):
                    self.assertIsNone(self.navigator.find(search=True))
                self.assertEqual(int(conflict_after_zoom), self.task.scroll.call_count)
                self.task.swipe.assert_not_called()


if __name__ == '__main__':
    unittest.main()

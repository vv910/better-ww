import unittest
import tempfile
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
from ok import Box, TaskDisabledException
from src.pathing.EchoRoute import EchoRoute, RoutePoint, RouteTeleport
from src.pathing.MapLocalizer import MapPosition
from src.pathing.MinimapHeading import HeadingEstimate
from src.pathing.FallGuard import FallGuard
from src.pathing.RouteSteering import RouteSteering
from src.pathing.RouteMapContext import RouteMapContext
from src.task.BaseCombatTask import BaseCombatTask, CharDeadException
from src.combat.CombatCheck import CombatCheck
from src.task.Farm3CEchoTask import Farm3CEchoTask, Record3CEchoRouteTask
from src.task.WWOneTimeTask import WWOneTimeTask


def make_task():
    task = Farm3CEchoTask.__new__(Farm3CEchoTask)
    task.config = {'Route File': 'route.json', 'Combat Timeout': 10, 'Pickup Timeout': 10,
                   'Auto Glide when Airborne': False}
    task.key_config = {}
    task.info = {}
    task._combat_deadline = None
    task._in_combat = False
    task._steering = None
    task._camera_aligned = False
    task._camera_recenter_at = 0
    task._fall_protection_ready = False
    task._fall_guard = FallGuard()
    task._airborne_stage = False
    task._active_route = None
    task._route_keys = ()
    task.navigation_interval = 0.2
    task.localizer = Mock()
    task.combat_timeout = task.pickup_timeout = 10
    for name in ('wait_in_team_and_world', 'center_camera', 'next_frame', 'locate', 'in_combat',
                 'send_key', 'send_key_down', 'send_key_up', 'mouse_up', 'click', 'sleep', 'log_info', 'log_error', 'info_set',
                 'info_incr', 'pick_echo', 'yolo_find_echo', 'incr_drop', 'read_route_heading',
                 'log_debug', 'log_warning', 'find_one', 'save_heading_diagnostic', 'save_localization_diagnostic',
                 'save_navigation_diagnostic',
                 'prepare_route_runtime', 'route_enemy_visible', 'observe_route_airborne'):
        setattr(task, name, Mock())
    task.in_combat.return_value = False
    task.find_one.return_value = False
    task.route_enemy_visible.return_value = False
    task.observe_route_airborne.return_value = False
    task.pick_echo.return_value = False
    task.yolo_find_echo.return_value = (False, False)
    task.locate.return_value = MapPosition(10, 10, 1, 20)
    task.read_route_heading.return_value = HeadingEstimate(0, 0.9, 0.2)
    task.wait_until = lambda condition, **kwargs: condition()
    return task


class TestFarm3CEchoTask(unittest.TestCase):
    def setUp(self):
        self.task = make_task()
        self.clock = 0

    def tick(self):
        self.clock += 1

    def test_camera_recenter_on_a_straight_path_keeps_forward_key_held(self):
        self.task._route_keys = ('w',)
        self.task._steering = RouteSteering(0)
        self.assertEqual(20, self.task.maintain_route_camera(20, 2))
        self.task.click.assert_called_once_with(0.5, 0.5, key='middle', down_time=0.02)
        self.task.send_key_up.assert_not_called()
        self.task.sleep.assert_not_called()
        self.assertEqual(20, self.task._steering.camera_heading)
        self.task.maintain_route_camera(20, 2.2)
        self.task.click.assert_called_once()

    def test_camera_turn_releases_sideways_keys_and_rebases_heading(self):
        self.task._route_keys = ('w', 'd')
        self.task.read_route_heading.return_value = HeadingEstimate(45, 0.9, 0.2)
        self.assertEqual(45, self.task.maintain_route_camera(0, 2))
        self.task.send_key_up.assert_any_call('w')
        self.task.send_key_up.assert_any_call('d')
        self.assertEqual((), self.task._route_keys)
        self.assertEqual(('w',), self.task._steering.update(45, 45, 3))
        self.task.center_camera.assert_not_called()

    def test_airborne_glide_uses_configured_jump_key_once_until_stable_landing(self):
        self.task._fall_protection_ready = True
        self.task.key_config = {'Jump Key': 'j'}
        self.task.in_team = Mock(return_value=(True, None))
        self.task.get_box_by_name = Mock()
        self.task.calculate_color_percentage = Mock(side_effect=[0.01, 0.01, 0.01, 0.3, 0.3, 0.01, 0.01])
        for now in (0, 0.15, 0.3, 0.4, 0.65, 0.8, 0.95):
            with patch('src.task.Farm3CEchoTask.time.monotonic', return_value=now):
                Farm3CEchoTask.observe_route_airborne(self.task)
        self.assertEqual([call('j', down_time=0.05)] * 2, self.task.send_key.call_args_list)

    def test_airborne_at_route_start_does_not_turn_camera_or_move_blindly(self):
        self.task.observe_route_airborne.return_value = True
        self.task.next_frame.side_effect = self.tick
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.task.follow_point(RoutePoint(100, 100, timeout=5))
        self.task.send_key_down.assert_not_called()
        self.task.center_camera.assert_not_called()
        self.task.route_enemy_visible.assert_not_called()

    def test_preflight_uses_only_the_already_selected_ground_tool(self):
        self.task.config['Auto Glide when Airborne'] = True
        self.task.in_team = Mock(return_value=(True, None))
        self.task.get_box_by_name = Mock()
        self.task.calculate_color_percentage = Mock(return_value=0.4)
        self.task.find_one.return_value = True
        self.task.ensure_levitator = Mock()
        self.task.prepare_fall_protection()
        self.assertTrue(self.task._fall_protection_ready)
        self.task.ensure_levitator.assert_not_called()
        self.task.calculate_color_percentage.return_value = 0.01
        self.task.prepare_fall_protection()
        self.assertFalse(self.task._fall_protection_ready)

    def test_preflight_does_not_open_wheel_when_no_tool_is_selected_by_default(self):
        self.task.config['Auto Glide when Airborne'] = True
        self.task.ensure_levitator = Mock()
        self.task.prepare_fall_protection()
        self.task.ensure_levitator.assert_not_called()
        self.assertFalse(self.task._fall_protection_ready)

    def test_optional_tool_selection_failure_does_not_interrupt_route(self):
        self.task.config.update({'Auto Glide when Airborne': True, 'Select Levitator Automatically': True})
        self.task.stop_route_movement = Mock()
        self.task.ensure_levitator = Mock(return_value=False)
        route = EchoRoute('test', Path('tests/images/path.png'), RoutePoint(10, 10),
                          (RoutePoint(30, 30),), (0, 0, 1920, 1080))
        self.task.execute_route = Mock()
        with patch.object(EchoRoute, 'load', return_value=route), patch.object(WWOneTimeTask, 'run'), \
                patch('src.task.Farm3CEchoTask.MapLocalizer'):
            self.task.run()
        self.task.ensure_levitator.assert_called_once_with(force=True)
        self.task.log_warning.assert_called()
        self.task.execute_route.assert_called_once_with(route)

    def test_preflight_cancellation_during_optional_tool_selection_propagates(self):
        self.task.config.update({'Auto Glide when Airborne': True, 'Select Levitator Automatically': True})
        self.task.ensure_levitator = Mock(side_effect=TaskDisabledException())
        with self.assertRaises(TaskDisabledException):
            self.task.prepare_fall_protection()

    def test_cancelled_tool_selection_releases_wheel_key(self):
        self.task.key_config = {'Wheel Key': 'tab'}
        self.task.find_one = Mock(return_value=None)
        self.task.in_team = Mock(return_value=(True, None))

        def cancel_in_wheel(seconds):
            if seconds == 0.4:
                raise TaskDisabledException()

        self.task.sleep.side_effect = cancel_in_wheel
        with self.assertRaises(TaskDisabledException):
            CombatCheck.ensure_levitator(self.task, force=True)
        self.task.send_key_down.assert_called_once_with('tab')
        self.task.send_key_up.assert_any_call('tab')

    def test_camera_option_can_preserve_the_existing_view(self):
        self.task.config['Keep Camera Behind Player'] = False
        self.task._route_keys = ('w',)
        self.assertEqual(10, self.task.maintain_route_camera(10, 4))
        self.task.click.assert_not_called()

    def test_airborne_at_target_waits_for_ground_before_dispatching_combat(self):
        self.task._camera_aligned = True
        self.task.observe_route_airborne.side_effect = [True, True, False]
        self.task.locate.return_value = MapPosition(30, 0, 1, 20)
        self.task.follow_point(RoutePoint(30, 0, tolerance=2))
        self.assertEqual(3, self.task.locate.call_count)
        self.task.center_camera.assert_not_called()
        self.task.send_key_down.assert_not_called()

    def test_route_execution_preserves_point_order_and_dispatches_actions(self):
        points = (RoutePoint(70, 20), RoutePoint(30, 20, 'combat'), RoutePoint(60, 20, 'pickup'))
        route = EchoRoute('test', Path('tests/images/path.png'), RoutePoint(10, 10), points,
                          (0, 0, 1920, 1080))
        events = Mock()
        self.task.follow_point = Mock()
        self.task.pickup_echoes = Mock()
        events.attach_mock(self.task.follow_point, 'follow')
        events.attach_mock(self.task.pickup_echoes, 'pickup')
        with patch.object(EchoRoute, 'load', return_value=route), patch.object(WWOneTimeTask, 'run'):
            self.task.run()
        self.assertEqual([call.follow(points[0], keep_moving=True), call.follow(points[1], keep_moving=False),
                          call.pickup(), call.follow(points[2], keep_moving=False), call.pickup()], events.mock_calls)
        self.assertEqual(3, self.task.info_incr.call_args_list.count(call('Completed Route Points')))

    def test_invalid_route_fails_before_any_game_input(self):
        with patch.object(EchoRoute, 'load', side_effect=ValueError('invalid')), self.assertRaises(ValueError):
            self.task.run()
        self.task.send_key.assert_not_called()
        self.task.center_camera.assert_not_called()

    def test_wrong_start_fails_before_following_points(self):
        route = EchoRoute('test', Path('tests/images/path.png'), RoutePoint(900, 500),
                          (RoutePoint(930, 500),), (0, 0, 1920, 1080))
        self.task.follow_point = Mock()
        self.task.recover_route_position = Mock(return_value=None)
        with patch.object(EchoRoute, 'load', return_value=route), patch.object(WWOneTimeTask, 'run'), \
                self.assertRaisesRegex(RuntimeError, 'route start'):
            self.task.run()
        self.task.follow_point.assert_not_called()

    def test_mid_route_combat_finishes_before_movement_resumes(self):
        self.task.fight_and_pickup = Mock()
        self.task.route_enemy_visible.side_effect = [True, False, False, False]
        self.task.locate.side_effect = [MapPosition(0, 0, 1, 20), MapPosition(30, 0, 1, 20)]
        events = Mock()
        events.attach_mock(self.task.fight_and_pickup, 'battle')
        events.attach_mock(self.task.send_key_down, 'move')
        self.task.follow_point(RoutePoint(30, 0, tolerance=2))
        self.assertEqual('battle', events.mock_calls[0][0])
        self.assertEqual('move', events.mock_calls[1][0])
        self.task.send_key_up.assert_any_call('w')

    def test_lost_position_stops_without_blind_movement(self):
        self.task.locate.return_value = None
        self.task.next_frame.side_effect = self.tick
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'position lost'):
            self.task.follow_point(RoutePoint(100, 100))
        self.task.send_key.assert_not_called()
        self.task.send_key_up.assert_any_call('w')
        self.task.save_localization_diagnostic.assert_called_once_with('lost')

    def test_transient_heading_failure_retries_fresh_frames_before_moving(self):
        self.task.read_route_heading.side_effect = [HeadingEstimate(0, 0.55, 0.2), None,
                                                    HeadingEstimate(90, 0.9, 0.2)]
        self.task.locate.side_effect = [MapPosition(0, 0, 1, 20), MapPosition(30, 0, 1, 20)]
        self.task.follow_point(RoutePoint(30, 0, tolerance=2))
        self.task.send_key_down.assert_called_once_with('a')
        self.assertEqual(3, self.task.read_route_heading.call_count)
        self.assertEqual([call(0.2), call(0.2)], self.task.sleep.call_args_list[:2])
        self.task.save_heading_diagnostic.assert_not_called()

    def test_uncertain_heading_stops_after_three_frames_with_diagnostics(self):
        ambiguous = HeadingEstimate(90, 0.85, 0.01)
        self.task.read_route_heading.return_value = ambiguous
        with self.assertRaisesRegex(RuntimeError, 'heading after 3 frames'):
            self.task.follow_point(RoutePoint(100, 100))
        self.assertEqual(3, self.task.read_route_heading.call_count)
        self.task.save_heading_diagnostic.assert_called_once_with(ambiguous)
        self.task.send_key.assert_not_called()
        self.task.send_key_up.assert_any_call('w')

    def test_cancellation_during_heading_retry_releases_inputs(self):
        self.task.read_route_heading.return_value = None
        self.task.sleep.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            self.task.follow_point(RoutePoint(100, 100))
        self.task.send_key.assert_not_called()
        self.task.send_key_up.assert_any_call('d')
        self.task.save_heading_diagnostic.assert_not_called()

    def test_heading_retry_yields_to_combat(self):
        self.task.read_route_heading.return_value = None
        self.task.route_enemy_visible.return_value = True
        self.assertIsNone(self.task.confirm_route_heading())
        self.task.send_key.assert_not_called()
        self.task.save_heading_diagnostic.assert_not_called()

    def test_straight_path_holds_one_key_and_centers_camera_once(self):
        self.task.locate.side_effect = [MapPosition(x, 0, 1, 20) for x in (0, 10, 20, 30)]
        self.task.follow_point(RoutePoint(30, 0, tolerance=2))
        self.task.send_key_down.assert_called_once_with('w')
        self.task.center_camera.assert_called_once()
        self.task.in_combat.assert_not_called()
        self.task.send_key.assert_not_called()

    def test_walk_points_keep_input_held_between_points(self):
        self.task.locate.side_effect = [MapPosition(x, 0, 1, 20) for x in (0, 10, 20)]
        self.task.follow_point(RoutePoint(10, 0, tolerance=2), keep_moving=True)
        self.assertEqual(('w',), self.task._route_keys)
        self.task.send_key_up.assert_not_called()
        self.task.follow_point(RoutePoint(20, 0, tolerance=2))
        self.task.send_key_down.assert_called_once_with('w')
        self.task.center_camera.assert_called_once()
        self.task.send_key_up.assert_any_call('w')

    def test_lost_localization_releases_an_existing_held_key_immediately(self):
        self.task._route_keys = ('w',)
        self.task.locate.return_value = None
        self.task.next_frame.side_effect = self.tick
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'position lost'):
            self.task.follow_point(RoutePoint(100, 100))
        self.task.send_key_up.assert_any_call('w')
        self.task.send_key_down.assert_not_called()

    def test_short_localization_gap_releases_input_without_recentering_camera(self):
        self.task.locate.side_effect = [MapPosition(0, 0, 1, 20), None,
                                       MapPosition(10, 0, 1, 20), MapPosition(30, 0, 1, 20)]
        events = Mock()
        events.attach_mock(self.task.send_key_down, 'down')
        events.attach_mock(self.task.send_key_up, 'up')
        self.task.follow_point(RoutePoint(30, 0, tolerance=2))
        self.task.center_camera.assert_called_once()
        moves = [event for event in events.mock_calls if event in (call.down('w'), call.up('w'))]
        self.assertEqual([call.down('w'), call.up('w'), call.down('w'), call.up('w')], moves)

    def test_long_localization_pause_recalibrates_camera_before_moving_again(self):
        self.task.locate.side_effect = [MapPosition(0, 0, 1, 20), None, None,
                                       MapPosition(10, 0, 1, 20), MapPosition(30, 0, 1, 20)]
        self.task.sleep.side_effect = lambda seconds: setattr(self, 'clock', self.clock + 0.6)
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock):
            self.task.follow_point(RoutePoint(30, 0, tolerance=2))
        self.assertEqual(2, self.task.center_camera.call_count)

    def test_short_pause_keeps_camera_heading_when_player_faces_diagonally(self):
        self.task.locate.side_effect = [MapPosition(0, 0, 1, 20), None,
                                       MapPosition(10, 10, 1, 20), MapPosition(60, 60, 1, 20)]
        self.task.read_route_heading.side_effect = [HeadingEstimate(0, 0.9, 0.2),
                                                   HeadingEstimate(45, 0.9, 0.2)]
        self.task.follow_point(RoutePoint(60, 60, tolerance=2))
        self.assertEqual([call('w'), call('d'), call('w'), call('d')], self.task.send_key_down.call_args_list)
        self.task.center_camera.assert_called_once()

    def test_empty_pickup_sweep_checks_four_views_with_short_camera_waits(self):
        self.task.pickup_echoes()
        self.assertEqual(4, self.task.yolo_find_echo.call_count)
        self.assertEqual([call('a', down_time=0.05)] * 3, self.task.send_key.call_args_list)
        self.assertEqual([call(0.4)] * 4, self.task.sleep.call_args_list)
        self.task.center_camera.assert_not_called()
        self.task.localizer.reset.assert_called_once()

    def test_pickup_cancellation_resets_tracker_and_releases_inputs(self):
        self.task.yolo_find_echo.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            self.task.pickup_echoes()
        self.task.send_key_up.assert_any_call('w')
        self.task.localizer.reset.assert_called_once()

    def test_queue_teleports_switches_maps_and_runs_in_order(self):
        first = EchoRoute('first', Path('tests/images/path.png'), RoutePoint(10, 10),
                          (RoutePoint(30, 30),), (0, 0, 1920, 1080))
        second = EchoRoute('second', Path('tests/images/path.png'), RoutePoint(10, 10),
                           (RoutePoint(30, 30),), (0, 0, 1920, 1080),
                           RouteTeleport('Crownless', RoutePoint(10, 10)))
        self.task.config['Route Queue'] = 'first.json\nsecond.json'
        self.task.execute_route = Mock()
        self.task.teleport_to_route = Mock()
        self.task.prepare_fall_protection = Mock()
        events = Mock()
        events.attach_mock(self.task.execute_route, 'execute')
        events.attach_mock(self.task.teleport_to_route, 'teleport')
        events.attach_mock(self.task.prepare_fall_protection, 'preflight')
        with patch.object(EchoRoute, 'load', side_effect=[first, second]), \
                patch.object(WWOneTimeTask, 'run'), \
                patch('src.task.Farm3CEchoTask.MapLocalizer') as localizer:
            self.task.run()
        self.assertEqual([call.preflight(), call.execute(first), call.teleport(second), call.preflight(),
                          call.execute(second)], events.mock_calls)
        self.assertEqual(2, localizer.call_count)
        self.assertEqual(2, localizer.return_value.reset.call_count)

    def test_queue_missing_teleport_is_rejected_before_game_input(self):
        route = EchoRoute('test', Path('tests/images/path.png'), RoutePoint(10, 10),
                          (RoutePoint(30, 30),), (0, 0, 1920, 1080))
        self.task.config['Route Queue'] = 'a.json\nb.json'
        with patch.object(EchoRoute, 'load', return_value=route), patch.object(WWOneTimeTask, 'run') as start, \
                self.assertRaisesRegex(ValueError, 'teleport metadata'):
            self.task.run()
        start.assert_not_called()

    def test_opencv_thread_limit_is_restored_on_route_failure(self):
        route = EchoRoute('test', Path('tests/images/path.png'), RoutePoint(10, 10),
                          (RoutePoint(30, 30),), (0, 0, 1920, 1080))
        self.task.execute_route = Mock(side_effect=RuntimeError('blocked'))
        with patch.object(EchoRoute, 'load', return_value=route), patch.object(WWOneTimeTask, 'run'), \
                patch('cv2.getNumThreads', return_value=16), patch('cv2.setNumThreads') as threads, \
                self.assertRaisesRegex(RuntimeError, 'blocked'):
            self.task.run()
        self.assertEqual([call(2), call(16)], threads.call_args_list)

    def prepare_teleport_task(self):
        self.task._executor = SimpleNamespace(frame=np.zeros((1080, 1920, 3), np.uint8))
        self.task.localizer = Mock()
        self.task.localizer.project_map_point.return_value = MapPosition(600, 450, 0.95, 40)
        for name in ('reset_to_false', 'openF2Book', 'click', 'input_text', 'box_of_screen',
                     'find_feature', 'wait_feature', 'click_box', 'screenshot'):
            setattr(self.task, name, Mock())
        self.task.find_feature.return_value = [Box(590, 440, 20, 20)]
        self.task.wait_feature.return_value = Box(1700, 950, 50, 20)
        return EchoRoute('TP test', Path('tests/images/path.png'), RoutePoint(600, 450),
                         (RoutePoint(650, 470),), (0, 0, 1920, 1080),
                         RouteTeleport('无冠者', RoutePoint(600, 450)))

    def test_teleport_confirms_map_alignment_beacon_and_return_to_world(self):
        route = self.prepare_teleport_task()
        self.task.teleport_to_route(route)
        self.task.input_text.assert_called_once_with('无冠者')
        self.task.click_box.assert_called_once()
        self.task.wait_in_team_and_world.assert_called_once_with(time_out=30)
        self.task.localizer.reset.assert_called_once()
        self.task.send_key_down.assert_called_once_with('ctrl')
        self.task.send_key_up.assert_any_call('ctrl')

    def test_teleport_search_cancellation_releases_modifier_key(self):
        route = self.prepare_teleport_task()
        self.task.send_key.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            self.task.teleport_to_route(route)
        self.task.send_key_up.assert_any_call('ctrl')
        self.task.input_text.assert_not_called()

    def test_map_mismatch_prevents_teleport_click(self):
        route = self.prepare_teleport_task()
        self.task.localizer.project_map_point.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'align route anchor map'):
            self.task.teleport_to_route(route)
        self.task.click_box.assert_not_called()
        self.task.screenshot.assert_called_once_with('3c-teleport-map-mismatch')

    def test_saved_map_context_is_selected_before_beacon_and_teleport(self):
        route = replace(self.prepare_teleport_task(), map_context=RouteMapContext(('测试独立地图',)))
        events = Mock()
        events.attach_mock(self.task.click_box, 'teleport')
        with patch('src.task.Farm3CEchoTask.RouteMapSelector') as selector_type:
            events.attach_mock(selector_type.return_value.select, 'select_map')
            self.task.teleport_to_route(route)
        self.assertEqual(call.select_map(route.map_context), events.mock_calls[0])
        self.task.openF2Book.assert_not_called()
        self.task.localizer.project_map_point.assert_called_once()
        self.task.click_box.assert_called_once()
        self.task.wait_in_team_and_world.assert_called_once_with(time_out=30)

    def test_failed_context_selection_cannot_fall_back_to_clicking_a_beacon(self):
        route = replace(self.prepare_teleport_task(), map_context=RouteMapContext(('测试独立地图',)))
        with patch('src.task.Farm3CEchoTask.RouteMapSelector') as selector_type:
            selector_type.return_value.select.side_effect = RuntimeError('map label not found')
            with self.assertRaisesRegex(RuntimeError, 'map label not found'):
                self.task.teleport_to_route(route)
        self.task.localizer.project_map_point.assert_not_called()
        self.task.click_box.assert_not_called()
        self.task.wait_in_team_and_world.assert_not_called()

    def test_context_clicks_alone_are_not_proof_of_correct_map(self):
        route = replace(self.prepare_teleport_task(), map_context=RouteMapContext(('测试独立地图',)))
        self.task.localizer.project_map_point.return_value = None
        with patch('src.task.Farm3CEchoTask.RouteMapSelector'), \
                self.assertRaisesRegex(RuntimeError, 'align route anchor map'):
            self.task.teleport_to_route(route)
        self.task.click_box.assert_not_called()
        self.task.wait_in_team_and_world.assert_not_called()

    def test_locked_beacon_stops_without_assuming_teleport_succeeded(self):
        route = self.prepare_teleport_task()
        self.task.wait_feature.return_value = None
        self.task.find_feature.side_effect = [[Box(590, 440, 20, 20)], [Box(590, 440, 20, 20)], []]
        with self.assertRaisesRegex(RuntimeError, 'cannot be teleported'):
            self.task.teleport_to_route(route)
        self.task.click_box.assert_not_called()
        self.task.wait_in_team_and_world.assert_not_called()

    def test_stuck_recovery_is_bounded(self):
        self.task.next_frame.side_effect = self.tick
        self.task.recover_stuck = Mock()
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'two recovery'):
            self.task.follow_point(RoutePoint(100, 10, tolerance=2))
        self.assertEqual([call(0), call(1)], self.task.recover_stuck.call_args_list)

    def test_physical_movement_away_from_goal_is_not_a_stationary_trap(self):
        self.task.next_frame.side_effect = self.tick
        self.task.locate.side_effect = lambda: MapPosition(-self.clock * 5, 0, 1, 20)
        self.task.recover_stuck = Mock()
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'waypoint timed out'):
            self.task.follow_point(RoutePoint(100, 0, timeout=12))
        self.task.recover_stuck.assert_not_called()

    def test_moving_without_approaching_realigns_once_then_reports_a_different_failure(self):
        self.task.next_frame.side_effect = self.tick
        self.task.locate.side_effect = lambda: MapPosition(-self.clock * 5, 0, 1, 20)
        self.task.recover_stuck = Mock()
        self.task.recover_route_position = Mock(return_value=None)
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'Moving without approaching'):
            self.task.follow_point(RoutePoint(100, 0, timeout=70))
        self.task.recover_route_position.assert_called_once()
        self.task.recover_stuck.assert_not_called()
        self.assertEqual('not-approaching', self.task.save_navigation_diagnostic.call_args.args[2])

    def test_recovery_uses_jump_hotkey_and_stops_if_a_retreat_becomes_airborne(self):
        self.task.key_config = {'Jump Key': 'j'}
        self.task._fall_protection_ready = True
        self.task.in_team = Mock(return_value=(True, None))
        self.task.get_box_by_name = Mock()
        self.task.calculate_color_percentage = Mock(return_value=.3)
        self.task.recover_stuck(0)
        self.task.send_key.assert_has_calls([call('s', down_time=.3, after_sleep=.1),
                                             call('a', down_time=.4, after_sleep=.1),
                                             call('j', down_time=.05, after_sleep=.3)])
        self.task.send_key.reset_mock()
        self.task.observe_route_airborne.return_value = True
        self.task.recover_stuck(1)
        self.task.send_key.assert_called_once_with('s', down_time=.3, after_sleep=.1)

    def test_recovery_does_not_jump_when_ground_signal_is_unavailable(self):
        for ready, in_team, brightness in ((False, True, .3), (True, False, .3), (True, True, .01)):
            with self.subTest(ready=ready, in_team=in_team, brightness=brightness):
                self.task._fall_protection_ready = ready
                self.task.in_team = Mock(return_value=(in_team, None))
                self.task.get_box_by_name = Mock()
                self.task.calculate_color_percentage = Mock(return_value=brightness)
                self.task.send_key.reset_mock()
                self.task.recover_stuck(0)
                self.assertEqual([call('s', down_time=.3, after_sleep=.1),
                                  call('a', down_time=.4, after_sleep=.1)], self.task.send_key.call_args_list)

    def test_waypoint_timeout_and_cancellation_release_keys(self):
        self.task.next_frame.side_effect = self.tick
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: self.clock), \
                self.assertRaisesRegex(RuntimeError, 'waypoint timed out'):
            self.task.follow_point(RoutePoint(100, 10, timeout=5))
        self.task.next_frame.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            self.task.follow_point(RoutePoint(100, 10))
        self.task.mouse_up.assert_called_with(key='right')
        self.task.send_key_up.assert_any_call('d')

    def test_multiple_echoes_are_collected_until_search_finds_no_more(self):
        self.task.pick_echo.side_effect = [True, True, False, False, False, False]
        self.task.pickup_echoes()
        self.assertEqual([call(True), call(True)], self.task.incr_drop.call_args_list)
        self.assertEqual(4, self.task.yolo_find_echo.call_count)
        self.assertTrue(all(not call.kwargs['turn'] for call in self.task.yolo_find_echo.call_args_list))
        self.task.send_key_up.assert_any_call('w')

    def test_combat_timeout_is_enforced_outside_base_exception_handler(self):
        self.task._combat_deadline = -1
        with patch.object(BaseCombatTask, 'in_combat') as base, \
                self.assertRaisesRegex(RuntimeError, 'combat timed out'):
            Farm3CEchoTask.in_combat(self.task)
        base.assert_not_called()

    def test_death_releases_inputs_without_resuming_route(self):
        self.task.combat_once = Mock(side_effect=CharDeadException('dead'))
        self.task.pickup_echoes = Mock()
        with self.assertRaises(CharDeadException):
            self.task.fight_and_pickup()
        self.assertIsNone(self.task._combat_deadline)
        self.task.pickup_echoes.assert_not_called()
        self.task.send_key_up.assert_any_call('w')
        self.assertFalse(self.task.revive_action())
        self.assertTrue(self.task.is_open_world_auto_combat())

    def test_new_tasks_instantiate_and_are_registered(self):
        from config import config
        executor = Mock()
        executor.scene = None
        executor.global_config.get_config.return_value = {}
        for cls in (Farm3CEchoTask, Record3CEchoRouteTask):
            with self.subTest(cls=cls):
                task = cls(executor, Mock())
                self.assertIn('Route File', task.default_config)
                self.assertIn(['src.task.Farm3CEchoTask', cls.__name__], config['onetime_tasks'])

    def test_route_recorder_will_not_overwrite_existing_files(self):
        task = Record3CEchoRouteTask.__new__(Record3CEchoRouteTask)
        task.config = {'Route File': 'config.py'}
        with self.assertRaises(ValueError):
            task.run()
        task.config['Route File'] = 'assets/coco_annotations.json'
        with self.assertRaises(FileExistsError):
            task.run()

    def test_recorder_exports_a_loadable_route_and_matching_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            route_path = Path(directory) / '声骸路线' / 'route.json'
            task = Record3CEchoRouteTask.__new__(Record3CEchoRouteTask)
            task.config = {'Route File': str(route_path), 'Route Name': '3C test', 'Start Method': 'Diamond'}
            task.big_map_frame = cv2.imread('tests/images/path.png')
            task.diamond = Box(1190, 460, 20, 20)
            task.stars = [task.diamond, Box(1100, 540, 20, 20), Box(1030, 600, 20, 20)]
            task.load_stars = Mock()
            task.info_set = task.log_info = Mock()
            with patch.object(WWOneTimeTask, 'run'):
                task.run()
            route = EchoRoute.load(route_path)
            self.assertEqual('3C test', route.name)
            self.assertEqual([(1110, 550), (1040, 610)], [(point.x, point.y) for point in route.waypoints])
            self.assertTrue(route.map_image.is_file())
            task.load_stars.assert_called_once_with(wait_world=False)
            self.assertTrue(route_path.with_suffix('.preview.png').exists())

    def test_recorder_exports_nearby_beacon_and_anchor_for_automatic_teleport(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'route.json'
            task = Record3CEchoRouteTask.__new__(Record3CEchoRouteTask)
            task.config = {'Route File': str(path), 'Route Name': 'TP', 'Teleport Anchor': '无冠者',
                           'Start Method': 'Diamond'}
            task.big_map_frame = cv2.imread('tests/images/path.png')
            task.diamond = Box(1190, 460, 20, 20)
            task.stars = [task.diamond, Box(1100, 540, 20, 20), Box(1030, 600, 20, 20)]
            task.find_feature = Mock(return_value=[Box(1200, 460, 20, 20)])
            task.load_stars = task.info_set = task.log_info = Mock()
            with patch.object(WWOneTimeTask, 'run'):
                task.run()
            route = EchoRoute.load(path)
            self.assertEqual('无冠者', route.teleport.anchor)
            self.assertEqual((1210, 470), (route.teleport.waypoint.x, route.teleport.waypoint.y))
            task.find_feature.return_value = [Box(1500, 700, 20, 20)]
            task.config['Route File'] = str(Path(directory) / 'too-far.json')
            with patch.object(WWOneTimeTask, 'run'), self.assertRaisesRegex(ValueError, 'within 50'):
                task.run()
            self.assertFalse(Path(task.config['Route File']).exists())


if __name__ == '__main__':
    unittest.main()

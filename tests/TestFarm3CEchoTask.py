import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
from ok import Box, TaskDisabledException
from src.pathing.EchoRoute import EchoRoute, RoutePoint, RouteTeleport
from src.pathing.MapLocalizer import MapPosition
from src.pathing.MinimapHeading import HeadingEstimate
from src.task.BaseCombatTask import BaseCombatTask, CharDeadException
from src.task.Farm3CEchoTask import Farm3CEchoTask, Record3CEchoRouteTask
from src.task.WWOneTimeTask import WWOneTimeTask


def make_task():
    task = Farm3CEchoTask.__new__(Farm3CEchoTask)
    task.config = {'Route File': 'route.json', 'Combat Timeout': 10, 'Pickup Timeout': 10}
    task.info = {}
    task._combat_deadline = None
    task._in_combat = False
    task._steering = None
    task._route_keys = ()
    task.navigation_interval = 0.2
    task.combat_timeout = task.pickup_timeout = 10
    for name in ('wait_in_team_and_world', 'center_camera', 'next_frame', 'locate', 'in_combat',
                 'send_key', 'send_key_down', 'send_key_up', 'mouse_up', 'sleep', 'log_info', 'log_error', 'info_set',
                 'info_incr', 'pick_echo', 'yolo_find_echo', 'incr_drop', 'read_route_heading',
                 'log_debug', 'save_heading_diagnostic', 'save_localization_diagnostic',
                 'prepare_route_runtime', 'route_enemy_visible'):
        setattr(task, name, Mock())
    task.in_combat.return_value = False
    task.route_enemy_visible.return_value = False
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

    def test_queue_teleports_switches_maps_and_runs_in_order(self):
        first = EchoRoute('first', Path('tests/images/path.png'), RoutePoint(10, 10),
                          (RoutePoint(30, 30),), (0, 0, 1920, 1080))
        second = EchoRoute('second', Path('tests/images/path.png'), RoutePoint(10, 10),
                           (RoutePoint(30, 30),), (0, 0, 1920, 1080),
                           RouteTeleport('Crownless', RoutePoint(10, 10)))
        self.task.config['Route Queue'] = 'first.json\nsecond.json'
        self.task.execute_route = Mock()
        self.task.teleport_to_route = Mock()
        events = Mock()
        events.attach_mock(self.task.execute_route, 'execute')
        events.attach_mock(self.task.teleport_to_route, 'teleport')
        with patch.object(EchoRoute, 'load', side_effect=[first, second]), \
                patch.object(WWOneTimeTask, 'run'), \
                patch('src.task.Farm3CEchoTask.MapLocalizer') as localizer:
            self.task.run()
        self.assertEqual([call.execute(first), call.teleport(second), call.execute(second)], events.mock_calls)
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
        self.task.pick_echo.side_effect = [True, True, False]
        self.task.pickup_echoes()
        self.assertEqual([call(True), call(True)], self.task.incr_drop.call_args_list)
        self.task.yolo_find_echo.assert_called_once()
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
            task.config = {'Route File': str(route_path), 'Route Name': '3C test'}
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
            task.config = {'Route File': str(path), 'Route Name': 'TP', 'Teleport Anchor': '无冠者'}
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

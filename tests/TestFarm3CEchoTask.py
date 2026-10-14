import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
from ok import Box, TaskDisabledException
from src.pathing.EchoRoute import EchoRoute, RoutePoint
from src.pathing.MapLocalizer import MapPosition
from src.task.BaseCombatTask import BaseCombatTask, CharDeadException
from src.task.Farm3CEchoTask import Farm3CEchoTask, Record3CEchoRouteTask
from src.task.WWOneTimeTask import WWOneTimeTask


def make_task():
    task = Farm3CEchoTask.__new__(Farm3CEchoTask)
    task.config = {'Route File': 'route.json', 'Combat Timeout': 10, 'Pickup Timeout': 10}
    task.info = {}
    task._combat_deadline = None
    task.combat_timeout = task.pickup_timeout = 10
    for name in ('wait_in_team_and_world', 'center_camera', 'next_frame', 'locate', 'in_combat',
                 'send_key', 'send_key_up', 'mouse_up', 'sleep', 'log_info', 'log_error', 'info_set',
                 'info_incr', 'pick_echo', 'yolo_find_echo', 'incr_drop', 'rotate_arrow_and_find'):
        setattr(task, name, Mock())
    task.in_combat.return_value = False
    task.pick_echo.return_value = False
    task.yolo_find_echo.return_value = (False, False)
    task.locate.return_value = MapPosition(10, 10, 1, 20)
    task.rotate_arrow_and_find.return_value = (0, SimpleNamespace(confidence=0.9))
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
        self.assertEqual([call.follow(points[0]), call.follow(points[1]), call.pickup(),
                          call.follow(points[2]), call.pickup()], events.mock_calls)
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
        self.task.in_combat.side_effect = [True, False, False, False]
        self.task.locate.side_effect = [MapPosition(0, 0, 1, 20), MapPosition(30, 0, 1, 20)]
        events = Mock()
        events.attach_mock(self.task.fight_and_pickup, 'battle')
        events.attach_mock(self.task.send_key, 'move')
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
            task.load_stars.assert_called_once_with(wait_world=False, max_distance=0)


if __name__ == '__main__':
    unittest.main()

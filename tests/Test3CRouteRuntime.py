import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from ok import Box, TaskDisabledException

from src.pathing.FallGuard import FallGuard
from src.pathing.EchoRoute import EchoRoute, RoutePoint
from src.pathing.MapLocalizer import MapPosition
from src.task.Farm3CEchoTask import Farm3CEchoTask
from src.task.WWOneTimeTask import WWOneTimeTask


def make_task():
    task = Farm3CEchoTask.__new__(Farm3CEchoTask)
    task.config = {'Auto Glide when Airborne': False}
    task.key_config = {'Map Key': 'm'}
    task._active_route = object()
    task._fall_guard = FallGuard()
    task._executor = SimpleNamespace(frame=np.zeros((1080, 1920, 3), np.uint8))
    task.localizer = Mock()
    task.localizer.locate_map_player.return_value = MapPosition(500, 400, 0.9, 25, source='map')
    task.locate = Mock(return_value=MapPosition(501, 401, 0.9, 25))
    task.route_enemy_visible = Mock(return_value=False)
    task.in_team_and_world = Mock(return_value=True)
    task.in_team = Mock(return_value=(False, None))
    task.stop_route_movement = Mock()
    task.send_key = Mock()
    task.send_key_down = Mock()
    task.wait_feature = Mock(return_value=Box(1800, 1000, 40, 40))
    task.click = Mock()
    task.next_frame = Mock()
    task.get_feature_by_name = Mock(return_value=SimpleNamespace(mat=np.zeros((10, 10, 3), np.uint8)))
    task.screenshot = Mock()
    task.log_warning = Mock()
    task.log_info = Mock()
    task.info_set = Mock()
    task.info_incr = Mock()
    task.wait_in_team_and_world = Mock()
    task.wait_until = lambda condition, **kwargs: condition()
    return task


class Test3CRouteRuntime(unittest.TestCase):
    def test_map_recovery_opens_map_uses_player_hint_then_confirms_minimap(self):
        task = make_task()
        events = Mock()
        events.attach_mock(task.send_key, 'key')
        events.attach_mock(task.wait_feature, 'button')
        events.attach_mock(task.click, 'click')
        events.attach_mock(task.next_frame, 'frame')
        events.attach_mock(task.localizer.locate_map_player, 'map_player')
        events.attach_mock(task.localizer.reset, 'reset')
        events.attach_mock(task.locate, 'minimap')

        with patch('src.task.Farm3CEchoTask.map_region', return_value=(300, 100, 1700, 950)), \
                patch('src.task.Farm3CEchoTask.find_map_player', return_value=(500, 400)):
            result = task.recover_route_position()

        self.assertEqual(MapPosition(501, 401, 0.9, 25), result)
        self.assertLess(events.mock_calls.index(call.key('m', after_sleep=0.4)),
                        events.mock_calls.index(call.button('map_my_location', time_out=2,
                                                            raise_if_not_found=False)))
        self.assertLess(events.mock_calls.index(call.click(task.wait_feature.return_value, after_sleep=0.3)),
                        events.mock_calls.index(call.map_player(task.frame, (500, 400),
                                                                (300, 100, 1700, 950))))
        map_hint = MapPosition(500, 400, 0.9, 25, source='map')
        self.assertIn(call.reset(position_hint=map_hint), events.mock_calls)
        self.assertGreater(events.mock_calls.index(call.minimap()),
                           events.mock_calls.index(call.reset(position_hint=map_hint)))
        task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])

    def test_failed_map_match_returns_without_blind_movement_or_minimap_reset(self):
        task = make_task()
        with patch('src.task.Farm3CEchoTask.map_region', return_value=(300, 100, 1700, 950)), \
                patch('src.task.Farm3CEchoTask.find_map_player', return_value=None):
            result = task.recover_route_position()
        self.assertIsNone(result)
        task.localizer.locate_map_player.assert_not_called()
        task.localizer.reset.assert_not_called()
        task.send_key_down.assert_not_called()
        task.screenshot.assert_called_once_with('3c-map-relocalize-failed')
        task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])

    def test_cancellation_after_opening_map_closes_it_without_directional_input(self):
        task = make_task()
        task.wait_feature.side_effect = TaskDisabledException()
        with patch('src.task.Farm3CEchoTask.map_region', return_value=(300, 100, 1700, 950)), \
                self.assertRaises(TaskDisabledException):
            task.recover_route_position()
        task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])
        task.send_key_down.assert_not_called()
        task.localizer.locate_map_player.assert_not_called()

    def test_recovery_does_not_open_map_while_airborne_in_combat_or_out_of_world(self):
        for airborne, in_combat, in_world in ((True, False, True), (False, True, True), (False, False, False)):
            with self.subTest(airborne=airborne, in_combat=in_combat, in_world=in_world):
                task = make_task()
                task._fall_guard.active = airborne
                task.route_enemy_visible.return_value = in_combat
                task.in_team_and_world.return_value = in_world
                self.assertIsNone(task.recover_route_position())
                task.send_key.assert_not_called()
                task.stop_route_movement.assert_not_called()

    def test_route_with_unhandled_map_layer_is_rejected_before_game_input(self):
        task = make_task()
        route = EchoRoute('layered', 'map.png', RoutePoint(400, 300),
                          (RoutePoint(420, 320),), (300, 100, 1700, 950), map_layer='upper floor')
        with patch('src.task.Farm3CEchoTask.route_files', return_value=['route.json']), \
                patch.object(EchoRoute, 'load', return_value=route), \
                patch.object(WWOneTimeTask, 'run') as base_run, \
                self.assertRaisesRegex(ValueError, 'Automatic layer switching is not supported'):
            task.run()
        base_run.assert_not_called()
        task.send_key.assert_not_called()
        task.send_key_down.assert_not_called()


if __name__ == '__main__':
    unittest.main()

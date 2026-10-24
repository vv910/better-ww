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


def map_frame(shift=0):
    y, x = np.indices((240, 400))
    gray = ((x * 17 + y * 31) % 190 + 30).astype(np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)
    return np.roll(frame, shift, axis=1) if shift else frame


def make_task():
    task = Farm3CEchoTask.__new__(Farm3CEchoTask)
    task.config = {'Auto Glide when Airborne': False}
    task.key_config = {'Map Key': 'm'}
    task._active_route = object()
    task._fall_guard = FallGuard()
    task._executor = SimpleNamespace(frame=map_frame())
    task.localizer = Mock()
    task.localizer.locate_map_player.return_value = MapPosition(200, 120, 0.9, 25, source='map')
    task.locate = Mock(return_value=MapPosition(201, 121, 0.9, 25))
    task.route_enemy_visible = Mock(return_value=False)
    task.in_team_and_world = Mock(return_value=True)
    task.in_team = Mock(return_value=(False, None))
    task.stop_route_movement = Mock()
    task.send_key = Mock()
    task.send_key_down = Mock()
    task.wait_feature = Mock(return_value=Box(370, 215, 20, 20))
    task.click = Mock()
    task.next_frame = Mock()
    task.sleep = Mock()
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

        with patch('src.task.Farm3CEchoTask.find_map_player', return_value=(200, 120)) as player_detector:
            result = task.recover_route_position()

        self.assertEqual(MapPosition(201, 121, 0.9, 25), result)
        self.assertLess(events.mock_calls.index(call.key('m', after_sleep=0.4)),
                        events.mock_calls.index(call.button('map_my_location', time_out=2,
                                                            raise_if_not_found=False)))
        self.assertLess(events.mock_calls.index(call.click(task.wait_feature.return_value, after_sleep=0.3)),
                        events.mock_calls.index(call.map_player(task.frame, (200, 120),
                                                                (112, 29, 348, 216))))
        map_hint = MapPosition(200, 120, 0.9, 25, source='map')
        self.assertIn(call.reset(position_hint=map_hint), events.mock_calls)
        self.assertGreater(events.mock_calls.index(call.minimap()),
                           events.mock_calls.index(call.reset(position_hint=map_hint)))
        self.assertEqual(3, player_detector.call_count)
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
        task.screenshot.assert_called_once()
        self.assertEqual('3c-map-relocalize-failed', task.screenshot.call_args.args[0])
        np.testing.assert_array_equal(task.frame, task.screenshot.call_args.kwargs['frame'])
        task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])

    def test_opening_and_centering_animation_settle_before_player_detection(self):
        task = make_task()
        open_stable = map_frame()
        centered_stable = map_frame(29)
        opening = iter([map_frame(5), map_frame(11), open_stable, open_stable, open_stable])
        centering = iter([map_frame(41), map_frame(36), centered_stable, centered_stable, centered_stable])
        observed = []

        def advance():
            source = centering if task.click.call_count else opening
            task._executor.frame = next(source, centered_stable if task.click.call_count else open_stable).copy()

        def click(*args, **kwargs):
            np.testing.assert_array_equal(open_stable, task.frame)

        def detect(*args, **kwargs):
            observed.append(task.frame.copy())
            return (200, 120)

        task.next_frame.side_effect = advance
        task.click.side_effect = click
        with patch('src.task.Farm3CEchoTask.find_map_player', side_effect=detect):
            result = task.recover_route_position()
        self.assertIsNotNone(result)
        self.assertEqual(3, len(observed))
        for frame in observed:
            np.testing.assert_array_equal(centered_stable, frame)
        self.assertEqual(2, task.localizer.locate_map_player.call_count)
        matched_frame, player, region = task.localizer.locate_map_player.call_args.args
        np.testing.assert_array_equal(centered_stable, matched_frame)
        self.assertEqual((200, 120), player)
        self.assertEqual((112, 29, 348, 216), region)

    def test_unsettled_or_blank_map_times_out_without_player_or_hint(self):
        for stage, blank in (('opening', False), ('centering', False), ('opening', True)):
            with self.subTest(stage=stage, blank=blank):
                task = make_task()
                elapsed = [0.0]
                count = [0]

                def advance():
                    count[0] += 1
                    if stage == 'centering' and not task.click.call_count:
                        task._executor.frame = map_frame()
                    else:
                        task._executor.frame = (np.zeros_like(task.frame) if blank
                                                else map_frame(count[0] * 3))

                task.next_frame.side_effect = advance
                task.sleep.side_effect = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
                with patch('src.pathing.RouteMapNavigation.time.monotonic', side_effect=lambda: elapsed[0]), \
                        patch('src.task.Farm3CEchoTask.find_map_player') as detector:
                    self.assertIsNone(task.recover_route_position())
                detector.assert_not_called()
                task.localizer.locate_map_player.assert_not_called()
                task.localizer.reset.assert_not_called()
                task.locate.assert_not_called()
                task.send_key_down.assert_not_called()
                task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])
                self.assertLess(elapsed[0], 3.5)
                if stage == 'opening':
                    task.click.assert_not_called()
                else:
                    task.click.assert_called_once()

    def test_player_disappearance_or_drift_on_new_settled_frame_rejects_hint(self):
        for markers in ([(200, 120), None], [(200, 120), (220, 120)],
                        [(200, 120), (203, 120), (206, 120)]):
            with self.subTest(markers=markers):
                task = make_task()
                with patch('src.task.Farm3CEchoTask.find_map_player', side_effect=markers):
                    self.assertIsNone(task.recover_route_position())
                self.assertEqual(1, task.localizer.locate_map_player.call_count)
                task.localizer.reset.assert_not_called()
                task.locate.assert_not_called()
                task.send_key_down.assert_not_called()
                task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])

    def test_unsettled_confirmation_frame_does_not_reuse_initial_player(self):
        task = make_task()
        elapsed = [0.0]
        calls = [0]
        detections = []

        def advance():
            calls[0] += 1
            task._executor.frame = map_frame(calls[0] * 3 if detections else 0)

        def detect(*args, **kwargs):
            detections.append(1)
            return (200, 120)

        task.next_frame.side_effect = advance
        task.sleep.side_effect = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
        with patch('src.pathing.RouteMapNavigation.time.monotonic', side_effect=lambda: elapsed[0]), \
                patch('src.task.Farm3CEchoTask.find_map_player', side_effect=detect):
            self.assertIsNone(task.recover_route_position())
        self.assertEqual(1, len(detections))
        task.localizer.locate_map_player.assert_called_once()
        task.localizer.reset.assert_not_called()
        task.locate.assert_not_called()
        self.assertLess(elapsed[0], 3.6)

    def test_stable_marker_still_requires_terrain_and_minimap_confirmation(self):
        for terrain_confirmed in (False, True):
            with self.subTest(terrain_confirmed=terrain_confirmed):
                task = make_task()
                if terrain_confirmed:
                    task.locate.return_value = None
                else:
                    task.localizer.locate_map_player.return_value = None
                with patch('src.task.Farm3CEchoTask.find_map_player', return_value=(200, 120)):
                    self.assertIsNone(task.recover_route_position())
                if terrain_confirmed:
                    task.localizer.reset.assert_called_once()
                    task.locate.assert_called_once()
                else:
                    task.localizer.reset.assert_not_called()
                    task.locate.assert_not_called()
                task.send_key_down.assert_not_called()

    def test_stable_screen_arrow_requires_two_consistent_terrain_registrations(self):
        for second_position in (None, MapPosition(220, 120, .9, 25, source='map')):
            with self.subTest(second_position=second_position):
                task = make_task()
                task.localizer.locate_map_player.side_effect = [
                    MapPosition(200, 120, .9, 25, source='map'), second_position]
                with patch('src.task.Farm3CEchoTask.find_map_player', return_value=(200, 120)):
                    self.assertIsNone(task.recover_route_position())
                self.assertEqual(2, task.localizer.locate_map_player.call_count)
                task.localizer.reset.assert_not_called()
                task.locate.assert_not_called()
                task.send_key_down.assert_not_called()

    def test_cancellation_while_waiting_for_stable_map_closes_map(self):
        task = make_task()
        task.next_frame.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            task.recover_route_position()
        task.stop_route_movement.assert_called_once()
        task.send_key.assert_has_calls([call('m', after_sleep=0.4), call('esc')])
        task.send_key_down.assert_not_called()
        task.localizer.reset.assert_not_called()

    def test_map_registration_exhausting_recovery_budget_cannot_supply_hint(self):
        for slow_registration in (1, 2):
            with self.subTest(slow_registration=slow_registration):
                task = make_task()
                elapsed = [0.0]
                registrations = [0]

                def register(*args):
                    registrations[0] += 1
                    if registrations[0] == slow_registration:
                        elapsed[0] += 11
                    return MapPosition(200, 120, .9, 25, source='map')

                task.localizer.locate_map_player.side_effect = register
                task.sleep.side_effect = lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
                with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: elapsed[0]), \
                        patch('src.task.Farm3CEchoTask.find_map_player', return_value=(200, 120)):
                    self.assertIsNone(task.recover_route_position())
                self.assertEqual(slow_registration, registrations[0])
                task.localizer.reset.assert_not_called()
                task.locate.assert_not_called()
                task.send_key_down.assert_not_called()
                task.send_key.assert_has_calls([call('m', after_sleep=.4), call('esc')])
                task.log_warning.assert_any_call(
                    '3C map relocalization failed: stage=recovery-budget-exhausted')

    def test_map_registration_cannot_start_after_recovery_budget(self):
        task = make_task()
        elapsed = [0.0]
        waits = [0]

        def settle(*args):
            waits[0] += 1
            if waits[0] == 2:
                elapsed[0] = 10
            return True

        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=lambda: elapsed[0]), \
                patch('src.task.Farm3CEchoTask.wait_for_map_stable', side_effect=settle), \
                patch('src.task.Farm3CEchoTask.find_map_player', return_value=(200, 120)):
            self.assertIsNone(task.recover_route_position())
        task.localizer.locate_map_player.assert_not_called()
        task.localizer.reset.assert_not_called()
        task.locate.assert_not_called()
        task.send_key.assert_has_calls([call('m', after_sleep=.4), call('esc')])
        task.log_warning.assert_any_call(
            '3C map relocalization failed: stage=recovery-budget-exhausted')

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

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from ok import TaskDisabledException

from src.pathing.EchoRoute import EchoRoute, RoutePoint
from src.pathing.MapLocalizer import MapPosition
from src.task.Farm3CEchoTask import Farm3CEchoTask
from src.task.WWOneTimeTask import WWOneTimeTask
from tests.TestFarm3CEchoTask import make_task


class TestNavigationReplayTask(unittest.TestCase):
    def setUp(self):
        self.task = make_task()
        self.replay = Mock()
        self.replay.flush.return_value = Path('logs/3c-replays/test')
        self.replay.last_error = None
        self.task._navigation_replay = self.replay

    def test_locate_records_the_actual_input_result_and_diagnostic(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        minimap = frame[:180, :180]
        self.task._executor = SimpleNamespace(frame=frame)
        self.task.get_box_by_name = Mock(return_value=SimpleNamespace(crop_frame=lambda _: minimap))
        position = MapPosition(300, 200, .9, 18)
        self.task.localizer.locate.return_value = position
        self.task.localizer.last_diagnostic = {'status': 'matched', 'method': 'sift'}
        self.assertEqual(position, Farm3CEchoTask.locate(self.task))
        args, kwargs = self.replay.record_locate.call_args
        np.testing.assert_array_equal(args[0], minimap)
        self.assertEqual(position, args[1])
        self.assertEqual('sift', args[2]['method'])
        self.assertEqual((1920, 1080), kwargs['capture_size'])
        self.assertGreaterEqual(kwargs['elapsed'], 0)

    def test_diagnostic_recording_failure_does_not_replace_a_visual_result(self):
        self.replay.record_keys.side_effect = OSError('recording unavailable')
        self.task.set_route_keys(('w',))
        self.assertEqual(('w',), self.task._route_keys)
        self.task.send_key_down.assert_called_once_with('w')
        self.assertIsNone(self.task._navigation_replay)

    def test_reset_and_command_events_preserve_map_hint_and_release(self):
        hint = MapPosition(300, 200, .9, 25, source='map')
        self.task.reset_route_localizer(hint)
        self.task.localizer.reset.assert_called_once_with(position_hint=hint)
        self.replay.record_reset.assert_called_once_with(hint)
        self.task.set_route_keys(('w', 'd'))
        self.task.stop_route_movement()
        self.assertEqual([call(('w', 'd')), call(())], self.replay.record_keys.call_args_list)

    def run_failure(self, error):
        route = EchoRoute('test', Path('tests/images/path.png'), RoutePoint(10, 10),
                          (RoutePoint(30, 30),), (0, 0, 1920, 1080))
        self.task.execute_route = Mock(side_effect=error)
        self.task.stop_route_movement = Mock()
        events = Mock()
        events.attach_mock(self.task.stop_route_movement, 'stop')
        events.attach_mock(self.replay.flush, 'flush')
        with patch.object(EchoRoute, 'load', return_value=route), patch.object(WWOneTimeTask, 'run'), \
                patch('src.task.Farm3CEchoTask.NavigationReplay', return_value=self.replay), \
                self.assertRaises(type(error)) as raised:
            self.task.run()
        self.assertIs(error, raised.exception)
        self.assertIsNone(self.task._navigation_replay)
        return events.mock_calls

    def test_failure_flushes_once_after_inputs_are_released(self):
        events = self.run_failure(RuntimeError('navigation failed'))
        self.replay.flush.assert_called_once()
        index = next(index for index, event in enumerate(events) if event[0] == 'flush')
        self.assertEqual('stop', events[index - 1][0])

    def test_user_cancellation_releases_inputs_without_writing_a_failure_bundle(self):
        self.run_failure(TaskDisabledException())
        self.replay.flush.assert_not_called()
        self.task.stop_route_movement.assert_called()

    def test_disk_failure_cannot_hide_the_original_navigation_error(self):
        self.replay.flush.side_effect = OSError('disk unavailable')
        self.run_failure(RuntimeError('original failure'))
        self.task.log_debug.assert_called()

    def test_partial_bundle_warns_without_hiding_the_original_navigation_error(self):
        self.replay.last_error = 'Missing map asset'
        self.run_failure(RuntimeError('original failure'))
        warnings = [event.args[0] for event in self.task.log_warning.call_args_list]
        self.assertEqual(1, sum('Missing map asset' in message for message in warnings))


if __name__ == '__main__':
    unittest.main()

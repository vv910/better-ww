import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.task.Farm3CEchoTask import Farm3CEchoTask
from src.task.BaseWWTask import BaseWWTask
from ok import TaskDisabledException
from tests.TestFarm3CEchoTask import make_task


class TestRouteCapture(unittest.TestCase):
    def make_capture_task(self):
        task = make_task()
        task._active_route = object()
        task._route_keys = ('w',)
        task._executor = SimpleNamespace(next_frame=Mock())
        return task

    def test_missing_capture_releases_input_and_clears_position_without_reading_old_frame(self):
        task = self.make_capture_task()
        task.executor.next_frame.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'capture unavailable'):
            Farm3CEchoTask.next_frame(task)
        task.executor.next_frame.assert_called_once_with(time_out=.5)
        self.assertEqual((), task._route_keys)
        task.send_key_up.assert_any_call('w')
        task.localizer.reset.assert_called_once()

    def test_delayed_capture_cannot_leave_movement_held_even_if_it_returns_an_image(self):
        task = self.make_capture_task()
        task.executor.next_frame.return_value = np.zeros((100, 100, 3), np.uint8)
        with patch('src.task.Farm3CEchoTask.time.monotonic', side_effect=[0, 1.1]), \
                self.assertRaisesRegex(RuntimeError, 'capture unavailable'):
            Farm3CEchoTask.next_frame(task)
        self.assertEqual((), task._route_keys)
        task.localizer.reset.assert_called_once()

    def test_fresh_capture_keeps_continuous_input(self):
        task = self.make_capture_task()
        image = np.zeros((100, 100, 3), np.uint8)
        task.executor.next_frame.return_value = image
        self.assertIs(image, Farm3CEchoTask.next_frame(task))
        self.assertEqual(('w',), task._route_keys)
        task.send_key_up.assert_not_called()

    def test_missing_initial_capture_cannot_reuse_an_old_image_to_start_moving(self):
        task = self.make_capture_task()
        task._route_keys = ()
        task.executor.next_frame.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'capture unavailable'):
            Farm3CEchoTask.next_frame(task)
        task.send_key_down.assert_not_called()
        task.localizer.reset.assert_called_once()

    def test_held_key_sleep_temporarily_disables_hidden_executor_capture_and_restores_on_cancel(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                task = self.make_capture_task()
                task.sleep_check_interval = .4

                def sleep(_):
                    self.assertEqual(-1, task.sleep_check_interval)
                    if cancelled:
                        raise TaskDisabledException()

                task.executor.sleep = Mock(side_effect=sleep)
                with patch.object(BaseWWTask, 'check_for_monthly_card', return_value=0):
                    if cancelled:
                        with self.assertRaises(TaskDisabledException):
                            Farm3CEchoTask.sleep(task, .2)
                    else:
                        Farm3CEchoTask.sleep(task, .2)
                self.assertEqual(.4, task.sleep_check_interval)

    def test_capture_poll_disables_nested_sleep_capture_until_returning(self):
        task = self.make_capture_task()
        task.sleep_check_interval = .4

        def capture(**kwargs):
            self.assertEqual(-1, task.sleep_check_interval)
            return np.zeros((100, 100, 3), np.uint8)

        task.executor.next_frame.side_effect = capture
        Farm3CEchoTask.next_frame(task)
        self.assertEqual(.4, task.sleep_check_interval)

    def test_combat_or_stationary_sleep_keeps_original_background_checks(self):
        task = self.make_capture_task()
        task._route_keys = ()
        task.sleep_check_interval = .4
        task.executor.sleep = Mock(side_effect=lambda _: self.assertEqual(.4, task.sleep_check_interval))
        with patch.object(BaseWWTask, 'check_for_monthly_card', return_value=0):
            Farm3CEchoTask.sleep(task, .2)
        self.assertEqual(.4, task.sleep_check_interval)


if __name__ == '__main__':
    unittest.main()

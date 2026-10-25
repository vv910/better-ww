import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import win32con
from ok import TaskDisabledException
from ok.device.interaction_methods.post_message import PostMessageInteraction
from ok.task.TaskExecutor import TaskExecutor

from src.pathing.RouteMapDrag import drag_route_map


class FakeClock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakePostMessage(PostMessageInteraction):
    """Record messages without creating a window or issuing Windows input."""

    def __init__(self, clock):
        self.clock = clock
        self.moves = []
        self.messages = []
        self.converted = []
        self.move_error_at = None
        self.release_error = None
        self.move_cost = 0.

    def move(self, x, y, down_btn=0):
        self.moves.append((x, y, down_btn, self.clock.now))
        self.clock.now += self.move_cost
        if len(self.moves) == self.move_error_at:
            raise RuntimeError('move failed')

    def update_mouse_pos(self, x, y, activate=True):
        self.converted.append((x, y))
        # Deliberately nonidentity capture -> client conversion.
        return (y + 19) * 65536 + x + 37

    def post(self, message, wParam=0, lParam=0, hwnd=None):
        self.messages.append((message, wParam, lParam))
        if self.release_error:
            raise self.release_error


class TestRouteMapDrag(unittest.TestCase):
    def make_task(self, background=True):
        clock = FakeClock()
        interaction = FakePostMessage(clock) if background else SimpleNamespace()
        task = SimpleNamespace(width=2560, height=1440,
                               executor=SimpleNamespace(interaction=interaction, reset_scene=Mock(),
                                                        check_enabled=Mock(), sleep=Mock(side_effect=clock.sleep)),
                               mouse_down=Mock(), mouse_up=Mock(), swipe=Mock(), sleep=Mock(),
                               sleep_check_interval=.4)
        task.executor.current_task = task
        return task, interaction, clock

    def run_drag(self, task, clock, start=(1500, 700), end=(900, 1100), duration=.3):
        with patch('src.pathing.RouteMapDrag.time.monotonic', clock.monotonic):
            drag_route_map(task, start, end, duration)

    def assert_released_at(self, task, interaction, point):
        self.assertEqual(point, interaction.converted[-1])
        x, y = point
        self.assertEqual((win32con.WM_LBUTTONUP, 0, (y + 19) * 65536 + x + 37),
                         interaction.messages[-1])
        task.executor.reset_scene.assert_called_once_with()

    def test_background_drag_uses_seconds_held_moves_and_exact_converted_endpoint(self):
        task, interaction, clock = self.make_task()
        self.run_drag(task, clock)
        task.mouse_down.assert_called_once_with(1500, 700)
        self.assertEqual(19, len(interaction.moves))
        self.assertEqual((1500, 700, 0, 0.), interaction.moves[0])
        self.assertEqual((900, 1100), interaction.moves[-1][:2])
        self.assertTrue(all(move[2] == win32con.MK_LBUTTON for move in interaction.moves[1:]))
        self.assertAlmostEqual(.3, clock.now, places=9)
        for first, second in zip(interaction.moves, interaction.moves[1:]):
            self.assertGreater(second[3], first[3])
            self.assertLessEqual(second[3] - first[3], 1 / 60 + 1e-9)
        self.assert_released_at(task, interaction, (900, 1100))
        task.mouse_up.assert_not_called()
        task.swipe.assert_not_called()
        task.sleep.assert_not_called()
        self.assertEqual(.4, task.sleep_check_interval)

    def test_all_directions_include_exact_endpoint(self):
        for end in ((2100, 700), (900, 700), (1500, 300), (1500, 1200), (2100, 300)):
            with self.subTest(end=end):
                task, interaction, clock = self.make_task()
                self.run_drag(task, clock, end=end)
                self.assertEqual(end, interaction.moves[-1][:2])
                self.assert_released_at(task, interaction, end)

    def test_scheduling_accounts_for_input_cost_without_adding_it_to_every_sleep(self):
        task, interaction, clock = self.make_task()
        interaction.move_cost = .004
        self.run_drag(task, clock)
        # The initial positioning and final input cost surround the .3s drag.
        self.assertAlmostEqual(.308, clock.now, places=9)
        self.assertLess(sum(call.args[0] for call in task.executor.sleep.call_args_list), .3)
        self.assert_released_at(task, interaction, (900, 1100))

    def test_cancelled_sleep_releases_last_cursor_and_preserves_cancellation(self):
        task, interaction, clock = self.make_task()
        cancelled = TaskDisabledException('cancelled')

        def sleep(duration):
            if task.executor.sleep.call_count == 4:
                raise cancelled
            clock.sleep(duration)

        task.executor.sleep.side_effect = sleep
        with self.assertRaises(TaskDisabledException) as caught:
            self.run_drag(task, clock)
        self.assertIs(cancelled, caught.exception)
        self.assert_released_at(task, interaction, interaction.moves[-1][:2])
        self.assertEqual(.4, task.sleep_check_interval)

    def test_failed_move_still_releases_at_attempted_cursor(self):
        task, interaction, clock = self.make_task()
        interaction.move_error_at = 5
        with self.assertRaisesRegex(RuntimeError, 'move failed'):
            self.run_drag(task, clock)
        self.assert_released_at(task, interaction, interaction.moves[-1][:2])

    def test_failed_initial_move_or_mouse_down_still_releases(self):
        for phase in ('move', 'down'):
            with self.subTest(phase=phase):
                task, interaction, clock = self.make_task()
                if phase == 'move':
                    interaction.move_error_at = 1
                else:
                    task.mouse_down.side_effect = RuntimeError('down failed')
                with self.assertRaises(RuntimeError):
                    self.run_drag(task, clock)
                self.assert_released_at(task, interaction, (1500, 700))

    def test_failed_exact_release_attempts_fallback_and_reports_failure(self):
        task, interaction, clock = self.make_task()
        failure = RuntimeError('exact release failed')
        interaction.release_error = failure
        with self.assertRaises(RuntimeError) as caught:
            self.run_drag(task, clock)
        self.assertIs(failure, caught.exception)
        task.mouse_up.assert_called_once_with()
        task.executor.reset_scene.assert_called_once_with()

    def test_release_failure_preserves_original_operation_error(self):
        task, interaction, clock = self.make_task()
        interaction.move_error_at = 3
        interaction.release_error = RuntimeError('release failed')
        with self.assertRaisesRegex(RuntimeError, 'move failed') as caught:
            self.run_drag(task, clock)
        self.assertIn('release failed', '\n'.join(caught.exception.__notes__))
        task.mouse_up.assert_called_once_with()
        task.executor.reset_scene.assert_called_once_with()

    def test_cancelled_release_is_not_swallowed_by_prior_operation_failure(self):
        task, interaction, clock = self.make_task()
        interaction.move_error_at = 3
        cancelled = TaskDisabledException('release cancelled')
        interaction.release_error = cancelled
        with self.assertRaises(TaskDisabledException) as caught:
            self.run_drag(task, clock)
        self.assertIs(cancelled, caught.exception)
        self.assertIn('move failed', str(caught.exception.__cause__))
        task.mouse_up.assert_called_once_with()

    def test_original_cancellation_survives_both_release_failures(self):
        task, interaction, clock = self.make_task()
        cancelled = TaskDisabledException('original cancellation')
        task.executor.sleep.side_effect = cancelled
        interaction.release_error = RuntimeError('exact release failed')
        task.mouse_up.side_effect = TaskDisabledException('fallback cancellation')
        with self.assertRaises(TaskDisabledException) as caught:
            self.run_drag(task, clock)
        self.assertIs(cancelled, caught.exception)
        task.executor.reset_scene.assert_called_once_with()
        self.assertEqual(.4, task.sleep_check_interval)

    def test_scene_reset_failure_does_not_hide_input_failure(self):
        task, interaction, clock = self.make_task()
        interaction.move_error_at = 3
        task.executor.reset_scene.side_effect = RuntimeError('reset failed')
        with self.assertRaisesRegex(RuntimeError, 'move failed') as caught:
            self.run_drag(task, clock)
        self.assertIn('reset failed', '\n'.join(caught.exception.__notes__))
        self.assertEqual(win32con.WM_LBUTTONUP, interaction.messages[-1][0])

    def test_other_backend_keeps_its_swipe_and_always_releases(self):
        task, interaction, clock = self.make_task(background=False)
        self.run_drag(task, clock)
        task.swipe.assert_called_once_with(1500, 700, 900, 1100, duration=.3, after_sleep=0)
        task.mouse_up.assert_called_once_with()
        task.mouse_down.assert_not_called()
        task.executor.reset_scene.assert_called_once_with()

    def test_other_backend_cancellation_still_releases(self):
        task, interaction, clock = self.make_task(background=False)
        cancelled = TaskDisabledException('swipe cancelled')
        task.swipe.side_effect = cancelled
        with self.assertRaises(TaskDisabledException) as caught:
            self.run_drag(task, clock)
        self.assertIs(cancelled, caught.exception)
        task.mouse_up.assert_called_once_with()

    def test_invalid_duration_is_rejected_before_input(self):
        for duration in (0, -.1, 2.001, float('nan'), float('inf'), '0.3', True, None):
            with self.subTest(duration=duration):
                task, interaction, clock = self.make_task()
                with self.assertRaises(ValueError):
                    self.run_drag(task, clock, duration=duration)
                self.assertEqual([], interaction.moves)
                self.assertEqual([], interaction.messages)

    def test_invalid_or_stationary_endpoints_are_rejected_before_input(self):
        for start, end in (((-1, 10), (50, 50)), ((10, 10), (2560, 10)),
                           ((10, 10), (50, 1440)), ((10, 10), (float('nan'), 20)),
                           ((10, 10), (True, 20)), ((10, 10), (10.1, 10.1)),
                           ((10, 10), (50,)), (None, (50, 50)), ((10, 10), (2559.9, 20))):
            with self.subTest(start=start, end=end):
                task, interaction, clock = self.make_task()
                with self.assertRaises(ValueError):
                    self.run_drag(task, clock, start, end)
                self.assertEqual([], interaction.moves)
                self.assertEqual([], interaction.messages)

    def test_invalid_capture_size_is_rejected_before_input(self):
        for width, height in ((0, 1440), (2560, float('inf')), ('2560', 1440), (True, 1440)):
            with self.subTest(width=width, height=height):
                task, interaction, clock = self.make_task()
                task.width, task.height = width, height
                with self.assertRaises(ValueError):
                    self.run_drag(task, clock)
                self.assertEqual([], interaction.moves)

    def test_direct_executor_wait_suppresses_real_sleep_check_capture_and_game_hooks(self):
        task, interaction, clock = self.make_task()
        task._enabled = True
        task.paused = False
        task.in_sleep_check = False
        task.last_sleep_check_time = -100
        task.sleep_check = Mock(side_effect=AssertionError('visual task hook ran while dragging'))
        task.check_for_monthly_card = Mock(side_effect=AssertionError('game UI hook ran while dragging'))
        task.sleep.side_effect = AssertionError('game-level sleep used while dragging')
        executor = task.executor
        executor.debug_mode = False
        executor.paused = False
        executor.exit_event = SimpleNamespace(is_set=lambda: False)
        executor.next_frame = Mock(side_effect=AssertionError('capture ran while dragging'))
        executor.check_enabled = TaskExecutor.check_enabled.__get__(executor)
        interaction.should_capture = lambda: True
        intervals = []

        def wait(duration):
            intervals.append(task.sleep_check_interval)
            clock.sleep(duration)

        executor._wait_for_activity = wait
        executor.sleep = TaskExecutor.sleep.__get__(executor)
        with patch('src.pathing.RouteMapDrag.time.monotonic', clock.monotonic), \
                patch('ok.task.TaskExecutor.time.time', clock.monotonic):
            drag_route_map(task, (1500, 700), (900, 1100))
        self.assertEqual([-1] * 18, intervals)
        self.assertEqual(.4, task.sleep_check_interval)
        self.assertAlmostEqual(.3, clock.now, places=9)
        executor.next_frame.assert_not_called()
        task.sleep_check.assert_not_called()
        task.check_for_monthly_card.assert_not_called()
        task.sleep.assert_not_called()
        self.assertEqual(win32con.WM_LBUTTONUP, interaction.messages[-1][0])

    def test_enable_check_cancellation_in_debug_style_wait_still_releases(self):
        task, interaction, clock = self.make_task()
        cancelled = TaskDisabledException('cancel after debug wait')
        task.executor.check_enabled.side_effect = [None, None, cancelled]
        with self.assertRaises(TaskDisabledException) as caught:
            self.run_drag(task, clock)
        self.assertIs(cancelled, caught.exception)
        self.assertEqual(2, len(interaction.moves))
        self.assert_released_at(task, interaction, interaction.moves[-1][:2])
        self.assertEqual(.4, task.sleep_check_interval)

    def test_interval_restored_after_release_failure_and_when_originally_absent(self):
        task, interaction, clock = self.make_task()
        interaction.release_error = RuntimeError('release failed')
        with self.assertRaises(RuntimeError):
            self.run_drag(task, clock)
        self.assertEqual(.4, task.sleep_check_interval)
        task, interaction, clock = self.make_task()
        del task.sleep_check_interval
        self.run_drag(task, clock)
        self.assertFalse(hasattr(task, 'sleep_check_interval'))

    def test_other_backend_scope_disables_checks_and_restores_previous_value(self):
        task, interaction, clock = self.make_task(background=False)
        task.sleep_check_interval = .8

        def swipe(*args, **kwargs):
            self.assertEqual(-1, task.sleep_check_interval)
            raise TaskDisabledException('backend cancelled')

        task.swipe.side_effect = swipe
        with self.assertRaises(TaskDisabledException):
            self.run_drag(task, clock)
        self.assertEqual(.8, task.sleep_check_interval)
        task.mouse_up.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()

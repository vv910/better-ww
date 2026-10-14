import unittest
from unittest.mock import Mock, PropertyMock, patch

from src.task.BaseCombatTask import BaseCombatTask
from src.task.BaseWWTask import BaseWWTask
from tests.TestFarmEchoPickup import make_task


class TestEchoFarmingRecovery(unittest.TestCase):
    def test_4c_exception_before_round_count_has_bounded_reteleport_attempts(self):
        task = make_task({'Repeat Farm Count': 1})
        task.teleport_to_boss_enabled.return_value = True
        task.should_reteleport_after_farm_exception = Mock(return_value=True)
        task.manage_boss_interactions.side_effect = RuntimeError('bad interaction')
        task.log_error = Mock()
        with self.assertRaisesRegex(RuntimeError, 'exceeded recovery retries'):
            task.do_run()
        self.assertEqual(3, task.manage_boss_interactions.call_count)
        self.assertEqual(3, task.teleport_to_configured_boss_and_prepare.call_count)

    def test_4c_failed_combat_round_is_retried_without_consuming_repeat_count(self):
        task = make_task({'Repeat Farm Count': 1})
        task.teleport_to_boss_enabled.return_value = True
        task.should_reteleport_after_farm_exception = Mock(return_value=True)
        task.combat_once.side_effect = [RuntimeError('failed combat'), True]
        task.pick_echo.return_value = True
        task.log_error = Mock()
        task.do_run()
        self.assertEqual(2, task.combat_once.call_count)
        task.incr_drop.assert_called_once_with(True)

    def test_yolo_pickup_releases_movement_if_detector_or_task_raises(self):
        task = BaseWWTask.__new__(BaseWWTask)
        for name in ('next_frame', 'pick_echo', 'pick_f', 'in_combat', 'find_echos', 'send_key_down',
                     'send_key_up', 'sleep', 'log_debug'):
            setattr(task, name, Mock())
        task.pick_echo.return_value = False
        task.pick_f.return_value = True  # an unrelated F prompt must not be treated as an Echo
        task.in_combat.return_value = False
        task.find_echos.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'detector'):
            task.walk_to_yolo_echo(update_function=Mock(side_effect=RuntimeError('detector')))
        task.send_key_down.assert_called_once_with('w')
        task.send_key_up.assert_called_once_with('w')
        task.pick_f.assert_not_called()

    def test_combat_without_wait_has_initialized_result(self):
        task = BaseCombatTask.__new__(BaseCombatTask)
        task.in_combat = Mock(return_value=False)
        self.assertFalse(task.combat_once(wait_combat_time=0))

    def test_missing_enemy_does_not_increment_combat_count(self):
        task = BaseCombatTask.__new__(BaseCombatTask)
        task.wait_combat = Mock(return_value=None)
        task.info = {}
        self.assertFalse(task.combat_once(wait_combat_time=1, raise_if_not_found=False))
        self.assertNotIn('Combat Count', task.info)

    def test_unknown_window_language_does_not_accept_generic_f_interactions(self):
        task = BaseWWTask.__new__(BaseWWTask)
        task.send_key = Mock()
        task.find_f_with_text = Mock(return_value=None)
        with patch.object(BaseWWTask, 'game_lang', new_callable=PropertyMock, return_value='unknown_lang'):
            task.pick_echo()
        pattern = task.find_f_with_text.call_args.kwargs['target_text']
        self.assertIsNotNone(pattern)
        self.assertIsNotNone(pattern.search('吸收'))
        self.assertIsNotNone(pattern.search('Absorb'))
        self.assertIsNone(pattern.search('领取奖励'))
        task.send_key.assert_not_called()


if __name__ == '__main__':
    unittest.main()

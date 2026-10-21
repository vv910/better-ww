import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from ok import Box, TaskDisabledException

from src.pathing.RouteMapContext import (RouteMapContext, RouteMapSelector,
                                         parse_selection_path)


class PickerTask:
    """A picker that expands/collapses parents and can dismiss a chosen leaf."""

    def __init__(self, active=(), opened=False, auto_close=True):
        self.active = set(active)
        self.opened = opened
        self.auto_close = auto_close
        self.world = False
        self.extra_map_rows = []
        self.switch = Box(1450, 990, 180, 35, name='Switch Map')
        self.title = Box(1100, 50, 170, 35, name='Switch Map')
        self.parent = Box(1080, 160, 160, 32, name='Parent')
        self.child = Box(1090, 260, 140, 32, name='Child')
        self.area = Box(1490, 175, 150, 32, name='Area')
        self.other = Box(1080, 600, 150, 32, name='Other World')
        self.ocr = Mock(side_effect=self.read)
        self.click = Mock(side_effect=self.choose)
        self.send_key = Mock(side_effect=self.key)
        self.next_frame = Mock(side_effect=self.render)
        self.sleep = Mock()
        self.screenshot = Mock()
        self.mouse_up = Mock()
        self.render()

    def map_rows(self):
        return [self.parent, self.other, *self.extra_map_rows] + (
            [self.child] if 'Parent' in self.active else [])

    def read(self, *region):
        if region == (.572, .035, .705, .095):
            return [self.title] if self.opened else []
        if region == (.68, .84, .99, .99):
            return [self.switch] if not self.opened and not self.world else []
        if region == (.555, .12, .71, .85):
            return self.map_rows() if self.opened else []
        if region == (.76, .12, .96, .84):
            return [self.area] if self.opened and 'Child' in self.active else []
        raise AssertionError(f'Unexpected broad/background OCR region: {region}')

    def render(self):
        self.frame = np.zeros((1080, 1920, 3), np.uint8)
        for row in self.map_rows():
            _, y = row.center()
            color = 245 if row.name in self.active else 188
            self.frame[int(y - 26):int(y + 26), 1070:1354] = color
        if 'Area' in self.active:
            cv2.circle(self.frame, (self.area.x - 32, int(self.area.center()[1])), 7,
                       (255, 255, 255), -1)

    def choose(self, box):
        if box is self.switch:
            self.opened = True
        elif box is self.parent:
            # Clicking an already expanded parent is a real regression here.
            if 'Parent' in self.active:
                self.active.difference_update({'Parent', 'Child', 'Area'})
            else:
                self.active.add('Parent')
        elif box is self.child:
            self.active.add('Child')
        elif box is self.area:
            self.active.add('Area')
            if self.auto_close:
                self.opened = False
        self.render()

    def key(self, key):
        if key == 'esc':
            self.opened = False


class TestRouteMapContext(unittest.TestCase):
    def _clocked(self, task, callback):
        clock = [0.0]
        task.sleep.side_effect = lambda duration: clock.__setitem__(0, clock[0] + duration)
        with patch('src.pathing.RouteMapContext.time.monotonic', side_effect=lambda: clock[0]):
            return callback()

    def test_context_validation_and_multiline_parser(self):
        self.assertEqual(('Jinzhou', 'Mt. Firmament'),
                         parse_selection_path(' Jinzhou \n\n Mt. Firmament '))
        self.assertEqual((), parse_selection_path(' \n\t '))
        self.assertEqual(('Jinzhou',), RouteMapContext.from_dict(
            {'selection_path': ['Jinzhou']}).selection_path)
        for bad in ({}, {'selection_path': []}, {'selection_path': ['']},
                    {'selection_path': ['x' * 81]}, {'selection_path': ['a\nb']},
                    {'selection_path': ['a'] * 5}, {'selection_path': ['a'], 'layer': 'b'},
                    {'selection_path': [' Jinzhou']}, {'selection_path': ['Jinzhou ']}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RouteMapContext.from_dict(bad)

    def test_parent_child_area_selection_uses_local_states_then_auto_dismisses(self):
        task = PickerTask()
        path = ('Parent', 'Child', 'Area')
        self.assertEqual(path, self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(path))))
        self.assertEqual([task.switch, task.parent, task.child, task.area],
                         [call.args[0] for call in task.click.call_args_list])
        task.send_key.assert_not_called()
        task.mouse_up.assert_called_once_with()
        task.screenshot.assert_not_called()

    def test_selected_ancestors_are_not_clicked_into_a_collapsed_state(self):
        task = PickerTask(active=('Parent', 'Child'), opened=True)
        self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Parent', 'Child', 'Area'))))
        self.assertEqual([task.area], [call.args[0] for call in task.click.call_args_list])
        self.assertTrue({'Parent', 'Child'} <= task.active)

    def test_selected_parent_with_hidden_child_does_not_get_blindly_toggled(self):
        task = PickerTask(active=('Parent',), opened=True)
        task.map_rows = lambda: [task.parent, task.other]
        with self.assertRaisesRegex(RuntimeError, "label-not-uniquely-found, label='Child'"):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Parent', 'Child'))))
        task.click.assert_not_called()
        task.screenshot.assert_called_once_with('3c-map-context-label-not-uniquely-found')

    def test_selected_leaf_is_still_clicked_to_request_centering_and_open_picker_is_closed(self):
        task = PickerTask(active=('Parent', 'Child', 'Area'), auto_close=False)
        self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Parent', 'Child', 'Area'))))
        self.assertEqual([task.switch, task.area], [call.args[0] for call in task.click.call_args_list])
        task.send_key.assert_called_once_with('esc')
        self.assertFalse(task.opened)

    def test_new_area_selection_is_followed_by_escape_before_returning_to_map(self):
        task = PickerTask(auto_close=False)
        events = Mock()
        events.attach_mock(task.click, 'click')
        events.attach_mock(task.send_key, 'key')
        path = ('Parent', 'Child', 'Area')
        self.assertEqual(path, self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(path))))
        actions = [(event[0], event.args[0]) for event in events.mock_calls]
        self.assertEqual([('click', task.switch), ('click', task.parent), ('click', task.child),
                          ('click', task.area), ('key', 'esc')], actions)
        self.assertFalse(task.opened)
        self.assertIn('Area', task.active)
        task.screenshot.assert_not_called()

    def test_path_can_finish_at_a_selected_map_without_an_area_label(self):
        task = PickerTask(active=('Parent', 'Child'), opened=True)
        self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Parent', 'Child'))))
        task.click.assert_not_called()
        task.send_key.assert_called_once_with('esc')

    def test_unrelated_text_changes_do_not_confirm_selection(self):
        task = PickerTask(opened=True)

        def unrelated_change(box):
            task.extra_map_rows.append(Box(1080, 700, 130, 32, name='Unrelated new row'))
            task.render()

        task.click.side_effect = unrelated_change
        with self.assertRaisesRegex(RuntimeError, 'selection-state-not-confirmed'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Parent', 'Child'))))
        self.assertEqual([task.parent], [call.args[0] for call in task.click.call_args_list])
        task.mouse_up.assert_called_once_with()

    def test_area_list_cannot_supply_an_ancestor(self):
        task = PickerTask(active=('Parent', 'Child'), opened=True)
        with self.assertRaisesRegex(RuntimeError, 'label-not-uniquely-found'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Area', 'Child'))))
        task.click.assert_not_called()

    def test_same_name_in_both_menu_columns_is_ambiguous(self):
        task = PickerTask(active=('Parent', 'Child'), opened=True)
        task.extra_map_rows = [Box(1080, 700, 130, 32, name='Area')]
        with self.assertRaisesRegex(RuntimeError, 'ambiguous-label'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Parent', 'Child', 'Area'))))
        task.click.assert_not_called()

    def test_missing_label_never_searches_the_map_background_or_scrolls_it(self):
        task = PickerTask(opened=True)
        with self.assertRaisesRegex(RuntimeError, 'label-not-uniquely-found'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Background landmark',))))
        task.click.assert_not_called()
        task.send_key.assert_not_called()

    def test_visible_location_feature_is_not_proof_picker_is_closed(self):
        task = PickerTask(active=('Parent', 'Child', 'Area'), opened=True, auto_close=False)
        task.send_key.side_effect = None
        task.wait_feature = Mock(return_value=True)
        with self.assertRaisesRegex(RuntimeError, 'picker-not-closed'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Area',))))
        task.send_key.assert_called_once_with('esc')
        task.wait_feature.assert_not_called()

    def test_absent_title_does_not_mean_returned_to_map(self):
        task = PickerTask(active=('Parent', 'Child'), opened=True)

        def leave_to_world(box):
            task.opened = False
            task.world = True

        task.click.side_effect = leave_to_world
        with self.assertRaisesRegex(RuntimeError, 'selection-state-not-confirmed'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Area',))))
        task.send_key.assert_not_called()

    def test_substring_or_ambiguous_candidates_fail_without_clicking_target(self):
        task = Mock()
        selector = RouteMapSelector(task)
        for boxes in ([], [Box(100, 100, 120, 20, name='Jinzhou City')],
                      [Box(100, 100, 70, 20, name='Jinzhou'), Box(300, 100, 70, 20, name='Jinzhou')]):
            with self.subTest(boxes=boxes):
                task.ocr.return_value = boxes
                self.assertIsNone(self._clocked(task, lambda: selector._stable_unique(
                    'Jinzhou', (0, 0, 1, 1), timeout=.25)))
        task.click.assert_not_called()

    def test_low_confidence_or_nonfinite_text_is_not_accepted(self):
        for confidence in (.59, float('inf'), float('nan')):
            with self.subTest(confidence=confidence):
                task = Mock()
                task.ocr.return_value = [Box(100, 100, 70, 20, confidence=confidence, name='Jinzhou')]
                self.assertIsNone(self._clocked(task, lambda: RouteMapSelector(task)._stable_unique(
                    'Jinzhou', (0, 0, 1, 1), timeout=.25)))
                task.click.assert_not_called()

    def test_slow_ocr_has_time_for_two_fresh_stable_frames(self):
        task = Mock()
        box = Box(100, 100, 70, 20, name='Jinzhou')
        clock = [0.0]

        def slow_ocr(*args):
            clock[0] += .4
            return [box]

        task.ocr.side_effect = slow_ocr
        task.sleep.side_effect = lambda duration: clock.__setitem__(0, clock[0] + duration)
        with patch('src.pathing.RouteMapContext.time.monotonic', side_effect=lambda: clock[0]):
            result = RouteMapSelector(task)._stable_unique('Jinzhou', (0, 0, 1, 1))
        self.assertIs(result, box)
        self.assertEqual(2, task.ocr.call_count)

    def test_cancellation_during_poll_sleep_propagates(self):
        task = PickerTask()
        task.sleep.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            RouteMapSelector(task).select(RouteMapContext(('Parent',)))
        task.mouse_up.assert_called_once_with()

    def test_task_disabled_propagates_and_mouse_is_released(self):
        task = PickerTask()
        task.ocr.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            RouteMapSelector(task).select(RouteMapContext(('Parent',)))
        task.mouse_up.assert_called_once_with()
        task.screenshot.assert_not_called()


if __name__ == '__main__':
    unittest.main()

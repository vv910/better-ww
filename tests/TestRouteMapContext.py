import unittest
from unittest.mock import Mock, patch

from ok import Box, TaskDisabledException

from src.pathing.RouteMapContext import (RouteMapContext, RouteMapSelector,
                                         parse_selection_path)


class TestRouteMapContext(unittest.TestCase):
    def _clocked(self, task, callback):
        clock = [0.0]
        task.sleep.side_effect = lambda duration: clock.__setitem__(0, clock[0] + duration)
        with patch("src.pathing.RouteMapContext.time.monotonic", side_effect=lambda: clock[0]):
            return callback()

    def test_context_validation_and_multiline_parser(self):
        self.assertEqual(("Jinzhou", "Mt. Firmament"),
                         parse_selection_path(" Jinzhou \n\n Mt. Firmament "))
        self.assertEqual((), parse_selection_path(" \n\t "))
        self.assertEqual(("Jinzhou",), RouteMapContext.from_dict(
            {"selection_path": ["Jinzhou"]}).selection_path)
        for bad in ({}, {"selection_path": []}, {"selection_path": [""]},
                    {"selection_path": ["x" * 81]}, {"selection_path": ["a\nb"]},
                    {"selection_path": ["a"] * 5}, {"selection_path": ["a"], "layer": "b"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RouteMapContext.from_dict(bad)
        for bad in ({"selection_path": [" Jinzhou"]}, {"selection_path": ["Jinzhou "]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RouteMapContext.from_dict(bad)

    def test_unique_exact_stable_label_is_clicked_and_mouse_released(self):
        task = Mock()
        switch = Box(900, 900, 100, 30, name="Switch Map")
        background = Box(110, 110, 100, 30, name="Other")
        picker_title = Box(500, 200, 130, 30, name="Jinzhou")
        changed_title = Box(520, 220, 150, 30, name="Jinzhou City")
        outputs = iter([
            [background],                         # pre-open baseline
            [switch],                            # exact switch-label search
            [switch], [switch],                  # stable switch button
            [picker_title], [picker_title],      # delayed but stable picker UI
            [picker_title], [picker_title],      # exact stable target
            [picker_title],                      # before click
            [picker_title], [changed_title], [changed_title],  # transition settles
        ])
        task.ocr.side_effect = lambda *args: next(outputs, [changed_title])
        selector = RouteMapSelector(task)
        result = self._clocked(task, lambda: selector.select(RouteMapContext(("Jinzhou",))))
        self.assertEqual(("Jinzhou",), result)
        self.assertEqual([switch, picker_title], [call.args[0] for call in task.click.call_args_list])
        task.mouse_up.assert_called_once_with()
        task.screenshot.assert_not_called()

    def test_substring_or_ambiguous_candidates_fail_without_clicking_target(self):
        task = Mock()
        selector = RouteMapSelector(task)
        exact_a = Box(100, 100, 70, 20, name="Jinzhou")
        exact_b = Box(300, 100, 70, 20, name="Jinzhou")
        task.ocr.return_value = []
        self.assertIsNone(self._clocked(task, lambda: selector._stable_unique(
            "Jinzhou", (0, 0, 1, 1), timeout=.25)))
        self.assertGreaterEqual(task.ocr.call_count, 2)
        task.ocr.reset_mock()
        task.ocr.return_value = [Box(100, 100, 120, 20, name="Jinzhou City")]
        self.assertIsNone(self._clocked(task, lambda: selector._stable_unique(
            "Jinzhou", (0, 0, 1, 1), timeout=.25)))
        self.assertGreaterEqual(task.ocr.call_count, 2)
        task.ocr.reset_mock()
        task.ocr.return_value = [exact_a, exact_b]
        self.assertIsNone(self._clocked(task, lambda: selector._stable_unique(
            "Jinzhou", (0, 0, 1, 1), timeout=.25)))
        task.ocr.assert_called_once()
        task.click.assert_not_called()

    def test_low_confidence_exact_text_is_not_accepted(self):
        for confidence in (.59, float('inf'), float('nan')):
            with self.subTest(confidence=confidence):
                task = Mock()
                task.ocr.return_value = [Box(100, 100, 70, 20, confidence=confidence, name="Jinzhou")]
                selector = RouteMapSelector(task)
                self.assertIsNone(self._clocked(task, lambda: selector._stable_unique(
                    "Jinzhou", (0, 0, 1, 1), timeout=.25)))
                task.click.assert_not_called()

    def test_click_without_ui_transition_fails_instead_of_reporting_a_selected_map(self):
        task = Mock()
        switch = Box(900, 900, 100, 30, name="Switch Map")
        target = Box(500, 200, 100, 30, name="Jinzhou")
        outputs = iter([[], [switch], [switch], [switch], [target], [target],
                        [target], [target], [target]])
        task.ocr.side_effect = lambda *args: next(outputs, [target])
        with self.assertRaisesRegex(RuntimeError, 'selection-transition-not-confirmed'):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(('Jinzhou',))))
        self.assertEqual(2, task.click.call_count)
        task.screenshot.assert_called_once_with('3c-map-context-selection-transition-not-confirmed')
        task.mouse_up.assert_called_once_with()

    def test_slow_ocr_has_time_for_two_fresh_stable_frames(self):
        task = Mock()
        box = Box(100, 100, 70, 20, name="Jinzhou")
        clock = [0.0]

        def slow_ocr(*args):
            clock[0] += .4
            return [box]

        task.ocr.side_effect = slow_ocr
        task.sleep.side_effect = lambda duration: clock.__setitem__(0, clock[0] + duration)
        with patch("src.pathing.RouteMapContext.time.monotonic", side_effect=lambda: clock[0]):
            result = RouteMapSelector(task)._stable_unique("Jinzhou", (0, 0, 1, 1))
        self.assertIs(result, box)
        self.assertEqual(2, task.ocr.call_count)
        self.assertGreaterEqual(task.sleep.call_args.args[0], .1)

    def test_same_name_background_at_same_position_is_excluded(self):
        task = Mock()
        switch = Box(900, 900, 100, 30, name="Switch Map")
        background = Box(500, 200, 100, 30, name="Jinzhou")
        outputs = [
            [background], [switch], [switch], [switch],
            [Box(500, 200, 100, 30, name="Jinzhou"), Box(650, 240, 100, 30, name="Other")],
            [Box(500, 200, 100, 30, name="Jinzhou"), Box(650, 240, 100, 30, name="Other")],
            [Box(500, 200, 100, 30, name="Jinzhou")],
        ]
        output_iter = iter(outputs)
        task.ocr.side_effect = lambda *args: next(output_iter, outputs[-1])
        with self.assertRaisesRegex(RuntimeError, "label-not-uniquely-found"):
            self._clocked(task, lambda: RouteMapSelector(task).select(RouteMapContext(("Jinzhou",))))
        self.assertEqual(1, task.click.call_count)  # only opened the switch control
        task.screenshot.assert_called_once()
        task.mouse_up.assert_called_once_with()

    def test_each_hierarchy_label_requires_its_own_exact_stable_click(self):
        task = Mock()
        switch = Box(900, 900, 100, 30, name="Change Map")
        first = Box(500, 200, 100, 30, name="Huanglong")
        child = Box(540, 260, 100, 30, name="Jinzhou")
        outputs = iter([
            [], [switch], [switch], [switch],
            [Box(500, 150, 120, 30, name="Map list")],
            [Box(500, 150, 120, 30, name="Map list")],
            [first], [first], [first], [first],  # baseline + stable option + pre-click frame
            [first], [child], [child],          # old UI lingers, then new child UI stabilizes
            [child], [child], [child],
            [Box(550, 250, 110, 30, name="Jinzhou City")],
            [Box(550, 250, 110, 30, name="Jinzhou City")],
        ])
        task.ocr.side_effect = lambda *args: next(outputs, [Box(550, 250, 110, 30, name="Jinzhou City")])
        result = self._clocked(task, lambda: RouteMapSelector(task).select(
            RouteMapContext(("Huanglong", "Jinzhou"))))
        self.assertEqual(("Huanglong", "Jinzhou"), result)
        self.assertEqual([switch, first, child], [call.args[0] for call in task.click.call_args_list])
        task.mouse_up.assert_called_once_with()

    def test_cancellation_during_poll_sleep_propagates(self):
        task = Mock()
        switch = Box(900, 900, 100, 30, name="Switch Map")
        task.ocr.side_effect = [[], [switch], [switch], [switch]]
        task.sleep.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            RouteMapSelector(task).select(RouteMapContext(("Jinzhou",)))
        task.mouse_up.assert_called_once_with()

    def test_task_disabled_propagates_and_mouse_is_released(self):
        task = Mock()
        task.ocr.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            RouteMapSelector(task).select(RouteMapContext(("Jinzhou",)))
        task.mouse_up.assert_called_once_with()
        task.screenshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()

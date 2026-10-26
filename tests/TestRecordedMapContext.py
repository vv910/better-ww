import unittest
from unittest.mock import patch

import cv2
import numpy as np
from ok import TaskDisabledException

from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.RouteMapContext import RouteMapContext, RouteMapSelector
from tests.TestRouteMapContext import PickerTask


class RecordedContextTask(PickerTask):
    def __init__(self, selected_frame, original_frame=None):
        self.selected_frame = selected_frame
        self.original_frame = selected_frame if original_frame is None else original_frame
        self.applied = False
        self.events = []
        super().__init__(active=('Parent', 'Child', 'Area'), auto_close=False)

    def render(self):
        super().render()
        if not self.opened:
            self.frame = (self.selected_frame if self.applied else self.original_frame).copy()
        self.events.append('frame')

    def choose(self, box):
        self.events.append('click:' + box.name)
        if box is self.area:
            self.applied = True
        super().choose(box)

    def key(self, key):
        self.events.append('key:' + key)
        super().key(key)


class TestRecordedMapContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = cv2.imread('tests/images/path.png')
        cls.other_map = cv2.imread('tests/images/big_map.png')
        assert cls.reference is not None and cls.other_map is not None
        cls.context = RouteMapContext(('Parent', 'Child', 'Area'))
        cls.point = (1207, 469)
        cls.region = (538, 130, 1670, 972)

    def setUp(self):
        self.clock = [0.0]
        self.monotonic = patch('src.pathing.RouteMapContext.time.monotonic', side_effect=lambda: self.clock[0])
        self.monotonic.start()
        self.addCleanup(self.monotonic.stop)

    def task(self, image, original=None):
        task = RecordedContextTask(image, original)
        task.sleep.side_effect = lambda duration: self.clock.__setitem__(0, self.clock[0] + duration)
        return task

    def references(self):
        return [(self.reference, self.point, self.region)]

    def test_real_map_is_checked_after_reselecting_even_an_already_selected_leaf(self):
        task = self.task(self.reference, self.other_map)
        selector = RouteMapSelector(task)
        original = self.reference.copy()
        register = MapLocalizer.register_map_view

        def observe(localizer, *args, **kwargs):
            self.assertTrue(task.applied)
            self.assertFalse(task.opened)
            self.assertIn('key:esc', task.events)
            task.events.append('registration')
            return register(localizer, *args, **kwargs)

        with patch.object(MapLocalizer, 'register_map_view', new=observe):
            result = selector.verify_context(self.context, self.references())
        self.assertIs(self.context, result)
        self.assertEqual([task.switch, task.area], [call.args[0] for call in task.click.call_args_list])
        self.assertLess(task.events.index('click:Area'), task.events.index('registration'))
        self.assertGreater(task.events[task.events.index('key:esc'):].count('frame'), 2)
        np.testing.assert_array_equal(original, self.reference)
        self.assertEqual('matched', selector.last_verification_diagnostic['status'])
        task.screenshot.assert_not_called()

    def test_real_other_map_is_unverified_and_keeps_reference_and_checked_frame(self):
        task = self.task(self.other_map, self.reference)
        selector = RouteMapSelector(task)
        with self.assertRaisesRegex(RuntimeError, '入口视图未验证') as caught:
            selector.verify_context(self.context, self.references())
        self.assertNotIn('错误地图', str(caught.exception))
        self.assertIsNotNone(caught.exception.__cause__)
        saved = {call.args[0]: call.kwargs['frame'] for call in task.screenshot.call_args_list}
        np.testing.assert_array_equal(self.reference, saved['3c-map-context-unverified-reference'])
        np.testing.assert_array_equal(self.other_map, saved['3c-map-context-unverified-actual'])
        self.assertEqual('unverified', selector.last_verification_diagnostic['status'])

    def test_any_recorded_view_can_verify_with_current_features_extracted_once(self):
        task = self.task(self.reference)
        selector = RouteMapSelector(task)
        prepare = MapLocalizer.prepare_map_features
        extractions = []

        def observe(localizer, *args, **kwargs):
            extractions.append(args[0])
            return prepare(localizer, *args, **kwargs)

        references = [(self.other_map, self.point, self.region), *self.references()]
        with patch.object(MapLocalizer, 'prepare_map_features', new=observe):
            self.assertIs(self.context, selector.verify_context(self.context, references))
        self.assertEqual(1, len(extractions))
        self.assertEqual(1, selector.last_verification_diagnostic['reference'])

    def test_verified_geometry_may_have_an_offscreen_beacon(self):
        live = cv2.warpAffine(self.reference, np.float32([[1, 0, -400], [0, 1, 25]]), (1920, 1080))
        task = self.task(live)
        selector = RouteMapSelector(task)
        references = [(self.reference, (650, 469), self.region)]
        self.assertIs(self.context, selector.verify_context(self.context, references))
        self.assertFalse(selector.last_verification_diagnostic['inside_view'])

    def test_slow_registration_cannot_pass_after_total_budget(self):
        task = self.task(self.reference)
        selector = RouteMapSelector(task)
        register = MapLocalizer.register_map_view

        def slow(localizer, *args, **kwargs):
            result = register(localizer, *args, **kwargs)
            self.assertIsNotNone(result)
            self.clock[0] = 21
            return result

        with patch.object(MapLocalizer, 'register_map_view', new=slow):
            with self.assertRaisesRegex(RuntimeError, 'budget-exhausted'):
                selector.verify_context(self.context, self.references())
        self.assertIsNone(selector._verification_deadline)

    def test_menu_work_consumes_the_same_total_budget_and_stops_input(self):
        task = self.task(self.reference)
        selector = RouteMapSelector(task)
        with patch('src.pathing.RouteMapContext._VERIFY_SECONDS', .2):
            with self.assertRaisesRegex(RuntimeError, 'budget-exhausted'):
                selector.verify_context(self.context, self.references())
        self.assertLessEqual(self.clock[0], .2)
        self.assertNotIn('click:Area', task.events)
        self.assertFalse(task.applied)
        task.mouse_up.assert_called_once_with()

    def test_slow_ocr_cannot_authorize_input_after_the_total_budget(self):
        task = self.task(self.reference)
        selector = RouteMapSelector(task)

        def slow(*region):
            self.clock[0] = 21
            return task.read(*region)

        task.ocr.side_effect = slow
        with self.assertRaisesRegex(RuntimeError, 'budget-exhausted'):
            selector.verify_context(self.context, self.references())
        task.click.assert_not_called()
        task.send_key.assert_not_called()

    def test_cancellation_propagates_without_failure_screenshots(self):
        task = self.task(self.reference)
        selector = RouteMapSelector(task)
        task.ocr.side_effect = TaskDisabledException()
        with self.assertRaises(TaskDisabledException):
            selector.verify_context(self.context, self.references())
        task.screenshot.assert_not_called()
        task.mouse_up.assert_called_once_with()
        self.assertIsNone(selector._verification_deadline)

    def test_diagnostic_io_failure_does_not_hide_the_verification_failure(self):
        task = self.task(self.other_map)
        task.screenshot.side_effect = OSError('disk unavailable')
        selector = RouteMapSelector(task)
        with self.assertRaisesRegex(RuntimeError, 'no-strong-reference-match') as caught:
            selector.verify_context(self.context, self.references())
        self.assertEqual('no-strong-reference-match-after-context-selection', str(caught.exception.__cause__))
        self.assertEqual(2, task.screenshot.call_count)

    def test_invalid_reference_inputs_fail_before_any_map_selection(self):
        for references in ([], [None], [(None, self.point, self.region)], self.references() * 10):
            with self.subTest(references_type=type(references).__name__):
                task = self.task(self.reference)
                with self.assertRaises(ValueError):
                    RouteMapSelector(task).verify_context(self.context, references)
                task.click.assert_not_called()
                task.ocr.assert_not_called()


if __name__ == '__main__':
    unittest.main()

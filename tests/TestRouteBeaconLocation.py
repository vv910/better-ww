import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.pathing.RouteBeaconLocation import (
    BEACON_HEADER_REGION, BEACON_LOCATION_REGION, RouteBeaconLocation,
    parse_beacon_location, read_beacon_location,
)


def box(text, header=False, confidence=.99, **changes):
    values = dict(name=text, confidence=confidence,
                  x=1876 if header else 1768, y=48 if header else 192,
                  width=200 if header else 345, height=61 if header else 37)
    values.update(changes)
    return SimpleNamespace(**values)


class TestRouteBeaconLocation(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1440, 2560, 3), np.uint8)
        self.raw = '瑝珑-梦枢天罗-相心域'

    def reader(self, headers=None, locations=None):
        task = Mock()
        task.ocr.side_effect = [headers if headers is not None else [box('小型信标', header=True)],
                                locations if locations is not None else [box(self.raw)]]
        return task

    def test_three_complete_parts_preserve_raw_text_and_do_not_invent_menu_context(self):
        raw = '  瑝珑 － 梦枢天罗 — 相心域  '
        result = parse_beacon_location(raw)
        self.assertIsInstance(result, RouteBeaconLocation)
        self.assertEqual(('瑝珑', '梦枢天罗', '相心域'), result.parts)
        self.assertEqual(raw.strip(), result.raw)
        self.assertFalse(hasattr(result, 'context'))

    def test_common_chinese_and_english_dash_forms_are_parsed(self):
        for separator in ('-', '‐', '‑', '‒', '–', '—', '―', '−', '﹘', '﹣', '－'):
            with self.subTest(separator=separator):
                result = parse_beacon_location(separator.join(('Rinascita', 'Ragunna', 'Whisperwind Haven')))
                self.assertEqual(('Rinascita', 'Ragunna', 'Whisperwind Haven'), result.parts)

    def test_missing_extra_empty_or_control_contaminated_parts_are_rejected(self):
        for raw in (None, 123, {}, [], '', ' ', '瑝珑', '瑝珑-相心域',
                    '瑝珑--相心域', '瑝珑-梦枢天罗-', '-梦枢天罗-相心域',
                    '瑝珑-梦枢天罗-相心域-其他', '瑝珑/梦枢天罗/相心域',
                    '瑝珑>梦枢天罗>相心域', '1-2-3',
                    '瑝珑-梦枢天罗-相心域\n', '瑝珑-梦枢\t天罗-相心域',
                    '瑝珑-梦枢天罗-相心域\u2028说明',
                    '瑝珑-梦枢\u200b天罗-相心域', 'A' * 81 + '-Map-Area'):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_beacon_location(raw))

    def test_ocr_reads_only_the_explicit_inspected_frame_and_two_fixed_regions(self):
        task = self.reader()
        task.frame = np.full_like(self.frame, 255)
        result = read_beacon_location(task, self.frame)
        self.assertEqual(('瑝珑', '梦枢天罗', '相心域'), result.parts)
        self.assertEqual(2, task.ocr.call_count)
        self.assertEqual(BEACON_HEADER_REGION, task.ocr.call_args_list[0].args)
        self.assertEqual(BEACON_LOCATION_REGION, task.ocr.call_args_list[1].args)
        for call in task.ocr.call_args_list:
            self.assertIs(self.frame, call.kwargs['frame'])
            self.assertEqual(.6, call.kwargs['threshold'])
        task.click.assert_not_called()
        task.send_key.assert_not_called()
        task.travel.assert_not_called()

    def test_only_exact_observed_beacon_titles_are_accepted(self):
        for title in ('小型信标', '中枢信标', '共鸣信标', '小型信標', '中樞信標', '共鳴信標',
                      'Small Beacon', 'Small Resonance Beacon', 'Resonance Beacon', 'Resonance Nexus'):
            with self.subTest(title=title):
                self.assertIsNotNone(read_beacon_location(self.reader(headers=[box(title, header=True)]),
                                                          self.frame))
        for title in ('自定义标记', '切换地图', '相心域', '信标', '信标说明',
                      '小型信标附近', '附近的小型信标', 'Resonance Beacon Location', 'Beacon Information'):
            with self.subTest(title=title):
                task = self.reader(headers=[box(title, header=True)])
                self.assertIsNone(read_beacon_location(task, self.frame))
                self.assertEqual(1, task.ocr.call_count)

    def test_low_nonfinite_or_out_of_range_confidence_is_rejected_for_both_lines(self):
        for confidence in (.59, float('nan'), float('inf'), -1, 100, True, None, 'bad'):
            with self.subTest(confidence=confidence):
                task = self.reader(headers=[box('小型信标', header=True, confidence=confidence)])
                self.assertIsNone(read_beacon_location(task, self.frame))
                task = self.reader(locations=[box(self.raw, confidence=confidence)])
                self.assertIsNone(read_beacon_location(task, self.frame))

    def test_multiple_candidate_headers_or_location_lines_are_rejected(self):
        self.assertIsNone(read_beacon_location(self.reader(headers=[box('小型信标', header=True),
                                                                   box('共鸣信标', header=True)]), self.frame))
        self.assertIsNone(read_beacon_location(self.reader(locations=[box(self.raw), box('瑝珑-梦州-相心域')]),
                                               self.frame))
        self.assertIsNone(read_beacon_location(self.reader(locations=[box(self.raw),
                                                                      box(self.raw, confidence=.1)]), self.frame))

    def test_split_or_partial_ocr_results_are_not_reconstructed(self):
        self.assertIsNone(read_beacon_location(self.reader(locations=[box('瑝珑'), box('梦枢天罗'), box('相心域')]),
                                               self.frame))
        self.assertIsNone(read_beacon_location(self.reader(locations=[box('瑝珑-梦枢天罗')]), self.frame))

    def test_text_outside_the_detail_panel_or_invalid_geometry_is_rejected(self):
        for changes in ({'x': 100}, {'y': 500}, {'width': -1}, {'height': 0},
                        {'x': float('nan')}, {'y': float('inf')}, {'width': True}):
            with self.subTest(changes=changes):
                self.assertIsNone(read_beacon_location(self.reader(locations=[box(self.raw, **changes)]),
                                                       self.frame))
        self.assertIsNone(read_beacon_location(self.reader(headers=[box('小型信标', header=True, x=100)]),
                                               self.frame))

    def test_invalid_frames_do_not_start_ocr(self):
        task = self.reader()
        for frame in (None, np.zeros((1440, 2560), np.uint8), np.zeros((100, 100, 3), np.uint8),
                      np.zeros((1440, 2560, 3), np.float32)):
            self.assertIsNone(read_beacon_location(task, frame))
        task.ocr.assert_not_called()

    def test_optional_diagnostics_are_per_call_and_do_not_carry_stale_success(self):
        diagnostic = {'stale': True}
        result = read_beacon_location(self.reader(), self.frame, diagnostic=diagnostic)
        self.assertEqual(result.raw, diagnostic['raw'])
        self.assertEqual('matched', diagnostic['status'])
        self.assertTrue(diagnostic['header_confirmed'])
        self.assertNotIn('stale', diagnostic)
        self.assertIsNone(read_beacon_location(self.reader(headers=[]), self.frame,
                                               diagnostic=diagnostic))
        self.assertEqual('rejected', diagnostic['status'])
        self.assertEqual('beacon_header_not_found', diagnostic['reason'])
        self.assertFalse(diagnostic['header_confirmed'])
        self.assertNotIn('raw', diagnostic)
        self.assertIsNone(read_beacon_location(self.reader(locations=[]), self.frame,
                                               diagnostic=diagnostic))
        self.assertTrue(diagnostic['header_confirmed'])
        self.assertEqual('complete_location_not_found', diagnostic['reason'])


if __name__ == '__main__':
    unittest.main()

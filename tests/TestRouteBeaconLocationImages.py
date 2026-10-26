import unittest

import cv2
from ok import BaseTask
from ok.test.TaskTestCase import TaskTestCase

from config import config
from src.pathing.RouteBeaconLocation import read_beacon_location


class TestRouteBeaconLocationImages(TaskTestCase):
    task_class = BaseTask
    config = dict(config)
    config.update(onetime_tasks=[], trigger_tasks=[], custom_tasks=False)

    def test_actual_beacon_header_and_complete_location_at_multiple_resolutions(self):
        path = 'tests/images/3c-beacon-location-panel.png'
        original = cv2.imread(path)
        self.assertIsNotNone(original)
        self.set_image(path)
        for factor in (.75, 1, 1.5):
            for quality in (None, 78):
                with self.subTest(factor=factor, jpeg=quality):
                    frame = cv2.resize(original, None, fx=factor, fy=factor)
                    if quality is not None:
                        encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
                        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    diagnostic = {}
                    result = read_beacon_location(self.task, frame, diagnostic=diagnostic)
                    self.assertIsNotNone(result, diagnostic)
                    self.assertEqual(('瑝珑', '梦枢天罗', '相心域'), result.parts)
                    self.assertTrue(diagnostic['header_confirmed'])

    def test_world_map_menu_and_custom_marker_details_do_not_identify_a_beacon(self):
        self.set_image('tests/images/3c-beacon-location-panel.png')
        for path in ('tests/images/mini_map.png', 'tests/images/3c-map-selector-selected.png',
                     'tests/images/3c-custom-marker-detail-panel.png'):
            with self.subTest(path=path):
                frame = cv2.imread(path)
                self.assertIsNotNone(frame)
                self.assertIsNone(read_beacon_location(self.task, frame))

    def test_location_line_alone_or_beacon_header_alone_is_insufficient(self):
        path = 'tests/images/3c-beacon-location-panel.png'
        original = cv2.imread(path)
        self.set_image(path)
        for removed in ('header', 'location'):
            frame = original.copy()
            if removed == 'header':
                frame[:170] = 0
            else:
                frame[170:270] = 0
            diagnostic = {}
            self.assertIsNone(read_beacon_location(self.task, frame, diagnostic=diagnostic))
            self.assertEqual(removed == 'location', diagnostic['header_confirmed'])


if __name__ == '__main__':
    unittest.main()

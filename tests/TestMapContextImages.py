import unittest

import cv2
from ok.test.TaskTestCase import TaskTestCase

from config import config
from src.pathing.RouteMapContext import RouteMapSelector
from src.task.Farm3CEchoTask import Farm3CEchoTask


class TestMapContextImages(TaskTestCase):
    task_class = Farm3CEchoTask
    config = dict(config)

    def test_actual_game_switch_button_is_readable_at_multiple_capture_sizes(self):
        # Only the bottom-right switch control is retained; account UI is removed.
        source = cv2.imread('tests/images/3c-map-switch-button.png')
        self.assertIsNotNone(source)
        self.set_image('tests/images/3c-map-switch-button.png')
        for scale in (.75, 1, 1.5):
            with self.subTest(scale=scale):
                frame = cv2.resize(source, None, fx=scale, fy=scale)
                hits = self.task.ocr(.68, .84, .99, .99, threshold=.6, frame=frame)
                switches = [hit for hit in hits if RouteMapSelector._exact(hit, '切换地图')]
                self.assertEqual(1, len(switches), [(hit.name, hit.confidence) for hit in hits])


if __name__ == '__main__':
    unittest.main()

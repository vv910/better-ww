import unittest

import cv2
from ok.test.TaskTestCase import TaskTestCase

from config import config
from src.pathing.RouteMapContext import RouteMapSelector, menu_choice_selected
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

    def test_actual_map_picker_columns_and_title_at_multiple_sizes(self):
        path = 'tests/images/3c-map-selector.png'
        source = cv2.imread(path)
        self.assertIsNotNone(source)
        self.set_image(path)
        left_names = {'罗伊冰原', '黯原', '冰原地表', '拉海洛', '黎那汐塔', '黑海岸', '瑝珑'}
        right_names = {'落日堤屿', '封存地', '寂静断崖', '恒黯之原'}

        for scale in (.75, 1, 1.5):
            with self.subTest(scale=scale):
                frame = cv2.resize(source, None, fx=scale, fy=scale)
                title_hits = self.task.ocr(.572, .035, .705, .095, threshold=.6, frame=frame)
                left_hits = self.task.ocr(.555, .12, .71, .85, threshold=.6, frame=frame)
                right_hits = self.task.ocr(.76, .12, .96, .84, threshold=.6, frame=frame)
                self.assertEqual({'切换地图'}, {hit.name for hit in title_hits})
                self.assertEqual(left_names, {hit.name for hit in left_hits})
                self.assertEqual(right_names, {hit.name for hit in right_hits})
                self.assertFalse(set(hit.name for hit in left_hits) & right_names)
                self.assertFalse(set(hit.name for hit in right_hits) & left_names)

                # The heading belongs to the top text-only ROI, not the bottom
                # right switch control ROI used to open the picker.
                bottom = self.task.ocr(.68, .84, .99, .99, threshold=.6, frame=frame)
                self.assertFalse([hit for hit in bottom
                                  if RouteMapSelector._exact(hit, '切换地图')],
                                 [(hit.name, hit.confidence) for hit in bottom])

    def test_actual_picker_selection_state_helper_at_multiple_sizes(self):
        source = cv2.imread('tests/images/3c-map-selector.png')
        self.assertIsNotNone(source)
        self.set_image('tests/images/3c-map-selector.png')
        map_states = {
            '罗伊冰原': True, '黯原': True, '冰原地表': False, '拉海洛': False,
            '黎那汐塔': False, '黑海岸': False, '瑝珑': False,
        }
        area_states = {
            '落日堤屿': True, '封存地': False, '寂静断崖': False, '恒黯之原': False,
        }

        for scale in (.75, 1, 1.5):
            for jpeg_quality in (None, 78):
                with self.subTest(scale=scale, jpeg_quality=jpeg_quality):
                    frame = cv2.resize(source, None, fx=scale, fy=scale)
                    if jpeg_quality is not None:
                        ok, encoded = cv2.imencode('.jpg', frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                        self.assertTrue(ok)
                        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

                    map_boxes = self.task.ocr(.555, .12, .71, .85, threshold=.6, frame=frame)
                    area_boxes = self.task.ocr(.76, .12, .96, .84, threshold=.6, frame=frame)
                    self.assertEqual(set(map_states), {box.name for box in map_boxes})
                    self.assertEqual(set(area_states), {box.name for box in area_boxes})
                    for box in map_boxes:
                        self.assertEqual(map_states[box.name],
                                         menu_choice_selected(frame, box, 'map'),
                                         (scale, jpeg_quality, box.name, box.center(), box.height))
                    for box in area_boxes:
                        self.assertEqual(area_states[box.name],
                                         menu_choice_selected(frame, box, 'area'),
                                         (scale, jpeg_quality, box.name, box.center(), box.height))


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import Mock

import cv2
from ok.test.TaskTestCase import TaskTestCase

from config import config
from src.pathing.RouteMapContext import RouteMapContext, RouteMapSelector, menu_choice_selected
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

    def test_actual_selected_picker_context_and_labels(self):
        path = 'tests/images/3c-map-selector-selected.png'
        source = cv2.imread(path)
        self.assertIsNotNone(source)
        self.set_image(path)
        map_states = {
            '罗伊冰原': False, '黎那汐塔': False, '黑海岸': False,
            '瑝珑': True, '梦州': True, '今州': False,
        }
        area_states = {
            '玄方城': False, '方擎西峰': False, '落渊南丘': False,
            '玄幽东岳': False, '梦枢天罗': True,
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
                    title_hits = self.task.ocr(.572, .035, .705, .095, threshold=.6, frame=frame)
                    map_boxes = self.task.ocr(.555, .12, .71, .85, threshold=.6, frame=frame)
                    area_boxes = self.task.ocr(.76, .12, .96, .84, threshold=.6, frame=frame)
                    self.assertEqual({'切换地图'}, {box.name for box in title_hits})
                    self.assertLessEqual(set(map_states), {box.name for box in map_boxes})
                    self.assertLessEqual(set(area_states), {box.name for box in area_boxes})
                    for box in map_boxes:
                        selected = menu_choice_selected(frame, box, 'map')
                        self.assertEqual(map_states.get(box.name, False), selected,
                                         (scale, jpeg_quality, box.name, box.center(), box.height))
                    for box in area_boxes:
                        selected = menu_choice_selected(frame, box, 'area')
                        self.assertEqual(area_states.get(box.name, False), selected,
                                         (scale, jpeg_quality, box.name, box.center(), box.height))

    def test_actual_closed_map_has_switch_control_but_no_picker_title(self):
        path = 'tests/images/3c-map-selector-closed.png'
        source = cv2.imread(path)
        self.assertIsNotNone(source)
        self.set_image(path)
        for scale in (.75, 1, 1.5):
            with self.subTest(scale=scale):
                frame = cv2.resize(source, None, fx=scale, fy=scale)
                title = self.task.ocr(.572, .035, .705, .095, threshold=.6, frame=frame)
                bottom = self.task.ocr(.68, .84, .99, .99, threshold=.6, frame=frame)
                self.assertFalse([box for box in title
                                  if RouteMapSelector._exact(box, '切换地图')],
                                 [(box.name, box.confidence) for box in title])
                self.assertEqual(1, len([box for box in bottom
                                         if RouteMapSelector._exact(box, '切换地图')]),
                                 [(box.name, box.confidence) for box in bottom])

    def test_selector_uses_real_menu_ocr_then_closes_to_real_map(self):
        menu_path = 'tests/images/3c-map-selector-selected.png'
        closed_path = 'tests/images/3c-map-selector-closed.png'
        self.assertIsNotNone(cv2.imread(menu_path))
        self.assertIsNotNone(cv2.imread(closed_path))
        self.set_image(menu_path)
        clicks = []
        self.task.click = Mock(side_effect=lambda box: clicks.append(box.name))
        self.task.send_key = Mock(side_effect=lambda key: self.set_image(closed_path)
                                  if key == 'esc' else None)
        self.task.mouse_up = Mock()
        self.task.sleep = Mock(return_value=None)

        path = ('瑝珑', '梦州', '梦枢天罗')
        self.assertEqual(path, RouteMapSelector(self.task).select(RouteMapContext(path)))
        self.assertEqual(['梦枢天罗'], clicks)
        self.task.send_key.assert_called_once_with('esc')
        # Selection completes only after the ordinary map and bottom switch
        # control are recognized in the post-Esc fixture.
        self.assertEqual(1, len([box for box in self.task.ocr(.68, .84, .99, .99, threshold=.6)
                                if RouteMapSelector._exact(box, '切换地图')]))

    def test_capture_reads_selected_context_and_preserves_terrain_frame(self):
        menu_path = 'tests/images/3c-map-selector-selected.png'
        closed_path = 'tests/images/3c-map-selector-closed.png'
        terrain = cv2.imread(closed_path)
        self.assertIsNotNone(terrain)
        self.set_image(menu_path)
        background_labels = [box for box in self.task.ocr(.555, .12, .71, .85, threshold=.6)
                             if box.name == '梦玄溪山']
        self.assertTrue(background_labels)
        self.assertTrue(all(not menu_choice_selected(self.task.frame, box, 'map')
                            for box in background_labels))
        self.task.big_map_frame = terrain.copy()
        original_terrain = self.task.big_map_frame.copy()
        self.task.click = Mock()
        self.task.send_key = Mock(side_effect=lambda key: self.set_image(closed_path)
                                  if key == 'esc' else None)
        self.task.mouse_up = Mock()
        self.task.sleep = Mock(return_value=None)

        selected = RouteMapSelector(self.task).capture()
        self.assertEqual(('瑝珑', '梦州', '梦枢天罗'), selected.selection_path)
        self.task.click.assert_not_called()
        self.task.send_key.assert_called_once_with('esc')
        after = cv2.absdiff(original_terrain, self.task.big_map_frame)
        self.assertEqual(0, cv2.countNonZero(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)))


if __name__ == '__main__':
    unittest.main()

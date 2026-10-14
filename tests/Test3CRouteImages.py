import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
from ok.test.TaskTestCase import TaskTestCase

from config import config
from src.pathing.EchoRoute import EchoRoute
from src.pathing.MapLocalizer import MapLocalizer
from src.task.Farm3CEchoTask import Farm3CEchoTask, Record3CEchoRouteTask
from src.task.WWOneTimeTask import WWOneTimeTask


class Test3CRouteImages(TaskTestCase):
    task_class = Farm3CEchoTask
    config = dict(config)

    def test_native_minimap_box_localizes_player_on_reference_map(self):
        self.set_image('tests/images/mini_map.png')
        self.task.localizer = MapLocalizer(cv2.imread('tests/images/path.png'), (538, 130, 1670, 972))
        result = self.task.locate()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(1207, result.x, delta=5)
        self.assertAlmostEqual(469, result.y, delta=5)

    def test_native_arrow_template_passes_route_heading_confidence_gate(self):
        self.set_image('tests/images/mini_map.png')
        angle, arrow = self.task.rotate_arrow_and_find()
        self.assertIsNotNone(arrow)
        self.assertGreaterEqual(arrow.confidence, 0.6)
        self.assertTrue(0 <= angle < 360)

    def test_record_real_map_markers_to_loadable_route(self):
        self.set_image('tests/images/path.png')
        recorder = Record3CEchoRouteTask(self.task.executor, None)
        recorder.feature_set = self.task.feature_set
        recorder.after_init(executor=self.task.executor, scene=self.task.scene)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recorded.json'
            recorder.config = {'Route File': str(path), 'Route Name': 'marker test'}
            with patch.object(WWOneTimeTask, 'run'):
                recorder.run()
            route = EchoRoute.load(path)
            self.assertEqual(3, len(route.waypoints))
            self.assertTrue(all(point.action == 'combat' for point in route.waypoints))
            self.assertAlmostEqual(1140, route.waypoints[0].x, delta=15)
            self.assertAlmostEqual(550, route.waypoints[0].y, delta=15)
            # A hand-placed marker has an offset from the player arrow in this real fixture.
            self.assertLessEqual(route.start.distance_to((1207, 469)), route.start.tolerance)


if __name__ == '__main__':
    unittest.main()

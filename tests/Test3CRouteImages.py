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

    def test_route_heading_on_real_minimap_at_1080p_1440p_and_2160p(self):
        original = cv2.imread('tests/images/mini_map.png')
        with tempfile.TemporaryDirectory() as directory:
            for width, height in ((1920, 1080), (2560, 1440), (3840, 2160)):
                with self.subTest(resolution=(width, height)):
                    path = str(Path(directory) / f'minimap-{height}.png')
                    cv2.imwrite(path, cv2.resize(original, (width, height)))
                    self.set_image(path)
                    self.task._heading_detector = None
                    result = self.task.read_route_heading()
                    self.assertIsNotNone(result)
                    self.assertTrue(result.reliable, result)
                    self.assertAlmostEqual(24, result.angle, delta=6)

    def test_heading_failure_saves_the_actual_minimap_and_logs_scores(self):
        self.set_image('tests/images/mini_map.png')
        result = self.task.read_route_heading()
        with patch.object(self.task, 'screenshot') as screenshot, \
                patch.object(self.task, 'log_warning') as warning:
            self.task.save_heading_diagnostic(result)
        self.assertEqual('3c-heading-minimap', screenshot.call_args.args[0])
        self.assertEqual((185, 186, 3), screenshot.call_args.kwargs['frame'].shape)
        self.assertIn('capture=1920x1080', warning.call_args.args[0])
        self.assertIn('confidence=', warning.call_args.args[0])

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

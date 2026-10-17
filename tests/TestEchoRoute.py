import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.pathing.EchoRoute import EchoRoute, RoutePoint, movement_direction, route_files, target_angle
from src.pathing.MapLocalizer import MapLocalizer, MapPosition


class TestEchoRoute(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        cv2.imwrite(str(self.root / 'map.png'), np.zeros((100, 120, 3), np.uint8))
        self.path = self.root / 'route.json'
        self.data = {
            'schema_version': 1, 'echo_cost': 3, 'name': 'test', 'map_image': 'map.png',
            'start': {'x': 50, 'y': 50},
            'waypoints': [{'x': 80, 'y': 60, 'action': 'combat'}, {'x': 20, 'y': 70, 'action': 'pickup'}],
        }

    def load(self):
        self.path.write_text(json.dumps(self.data), encoding='utf-8')
        return EchoRoute.load(self.path)

    def test_route_keeps_author_order_and_resolves_image_relative_to_json(self):
        route = self.load()
        self.assertEqual([80, 20], [point.x for point in route.waypoints])
        self.assertEqual(['combat', 'pickup'], [point.action for point in route.waypoints])
        self.assertEqual(self.root / 'map.png', route.map_image)

    def test_rejects_bad_version_missing_resource_and_wrong_cost(self):
        for key, value in [('schema_version', True), ('schema_version', 2), ('echo_cost', 4),
                           ('map_image', 'missing.png'), ('name', '')]:
            with self.subTest(key=key, value=value):
                original = self.data[key]
                self.data[key] = value
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load()
                self.data[key] = original

    def test_rejects_unsafe_or_unsupported_waypoint_data(self):
        for change in ({'x': float('nan')}, {'y': float('inf')}, {'x': True}, {'x': 120},
                       {'x': -1}, {'action': 'teleport'}, {'timeout': 0}, {'tolerance': 0}):
            with self.subTest(change=change):
                self.data['waypoints'] = [{'x': 80, 'y': 60} | change]
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_points_outside_visible_region_and_empty_routes(self):
        self.data['map_region'] = [40, 30, 100, 90]
        with self.assertRaises(ValueError):
            self.load()
        self.data.pop('map_region')
        self.data['waypoints'] = []
        with self.assertRaises(ValueError):
            self.load()

    def test_screen_coordinate_directions_and_angle_wrap(self):
        for (x, y), angle, key in [((10, 0), 0, 'w'), ((0, 10), 90, 'd'),
                                   ((-10, 0), 180, 's'), ((0, -10), 270, 'a')]:
            with self.subTest(point=(x, y)):
                actual = target_angle((0, 0), RoutePoint(x, y))
                self.assertEqual(angle, actual)
                self.assertEqual(key, movement_direction(0, actual))
        self.assertEqual('w', movement_direction(350, 10))
        self.assertEqual('w', movement_direction(10, 350))

    def test_multiline_queue_preserves_order_and_single_route_compatibility(self):
        self.assertEqual(['a.json', 'b.json', 'a.json'], route_files('old.json', 'a.json\n b.json\n\n a.json'))
        self.assertEqual(['old.json'], route_files('old.json', ' \n'))
        for single, queue in [('', ''), ('a', []), ('a', '\n'.join(['a'] * 101))]:
            with self.assertRaises(ValueError):
                route_files(single, queue)

    def test_teleport_metadata_requires_a_beacon_near_start(self):
        self.data['teleport'] = {'anchor': '无冠者', 'x': 55, 'y': 50}
        route = self.load()
        self.assertEqual('无冠者', route.teleport.anchor)
        self.assertEqual(55, route.teleport.waypoint.x)
        for change in ({'anchor': ''}, {'x': 110}, {'x': float('nan')}, {'y': True}):
            self.data['teleport'] = {'anchor': '无冠者', 'x': 55, 'y': 50} | change
            with self.assertRaises(ValueError):
                self.load()


class TestMapLocalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        gray = rng.integers(0, 256, (650, 850), dtype=np.uint8)
        cls.map = cv2.cvtColor(cv2.GaussianBlur(gray, (5, 5), 0.8), cv2.COLOR_GRAY2BGR)

    def test_localizes_scaled_terrain_crops(self):
        for scale in (0.8, 1.0, 1.4):
            with self.subTest(scale=scale):
                minimap = cv2.resize(self.map[280:480, 380:580], None, fx=scale, fy=scale)
                result = MapLocalizer(self.map).locate(minimap)
                self.assertIsNotNone(result)
                self.assertAlmostEqual(480, result.x, delta=2)
                self.assertAlmostEqual(380, result.y, delta=2)

    def test_projects_recorded_beacon_onto_translated_and_scaled_map(self):
        localizer = MapLocalizer(self.map)
        transform = np.float32([[1.15, 0, 45], [0, 1.15, 25]])
        frame = cv2.warpAffine(self.map, transform, (1100, 850))
        result = localizer.project_map_point(frame, (480, 380), (0, 0, 1100, 850))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(597, result.x, delta=2)
        self.assertAlmostEqual(462, result.y, delta=2)
        self.assertIsNone(localizer.project_map_point(np.zeros_like(frame), (480, 380), (0, 0, 1100, 850)))

    def test_projection_rejects_beacon_outside_current_map_view(self):
        localizer = MapLocalizer(self.map)
        crop = self.map[100:500, 300:800].copy()
        self.assertIsNone(localizer.project_map_point(crop, (50, 300), (0, 0, 500, 400)))

    def test_real_mingchao_minimap_matches_supplied_map_snapshot(self):
        image = cv2.imread('tests/images/path.png')
        minimap = cv2.imread('tests/images/mini_map.png')[30:211, 35:218]
        result = MapLocalizer(image, (538, 130, 1670, 972)).locate(minimap)
        self.assertIsNotNone(result)
        # Player arrow location independently visible on tests/images/path.png.
        self.assertAlmostEqual(1207, result.x, delta=5)
        self.assertAlmostEqual(469, result.y, delta=5)

    def test_rejects_blank_unrelated_and_rotating_minimap(self):
        localizer = MapLocalizer(self.map)
        self.assertIsNone(localizer.locate(np.zeros((180, 180, 3), np.uint8)))
        unrelated = np.random.default_rng(1).integers(0, 256, (180, 180, 3), dtype=np.uint8)
        self.assertIsNone(localizer.locate(unrelated))
        crop = self.map[280:480, 380:580]
        rotated = cv2.warpAffine(crop, cv2.getRotationMatrix2D((100, 100), 45, 1), (200, 200))
        self.assertIsNone(localizer.locate(rotated))

    def test_rejects_position_jumps_without_poisoning_previous_position(self):
        localizer = MapLocalizer(self.map, max_jump=30)
        first = localizer.locate(self.map[280:480, 380:580])
        self.assertIsNotNone(first)
        self.assertIsNone(localizer.locate(self.map[80:280, 80:280]))
        self.assertEqual(first, localizer.previous)
        self.assertIsNotNone(localizer.locate(self.map[285:485, 385:585]))


if __name__ == '__main__':
    unittest.main()

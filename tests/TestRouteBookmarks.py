import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.pathing.EchoRoute import EchoRoute, RouteMapView


class TestRouteBookmarks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.route_path = self.root / 'route.json'
        self.data = {
            'schema_version': 1,
            'echo_cost': 3,
            'name': 'bookmark route',
            'map_image': 'map.png',
            'start': {'x': 50, 'y': 50},
            'waypoints': [{'x': 80, 'y': 60, 'action': 'combat'}],
        }
        self.write_image(self.root / 'map.png')

    @staticmethod
    def write_image(path):
        encoded, image = cv2.imencode('.png', np.zeros((100, 120, 3), np.uint8))
        if not encoded:
            raise AssertionError('Could not encode fixture image')
        path.write_bytes(image.tobytes())

    def load(self):
        self.route_path.write_text(json.dumps(self.data), encoding='utf-8')
        return EchoRoute.load(self.route_path)

    def test_legacy_anchor_metadata_and_schema_remain_compatible(self):
        self.data['teleport'] = {'anchor': 'Boss Anchor', 'x': 50, 'y': 50}
        route = self.load()
        self.assertEqual('Boss Anchor', route.teleport.anchor)
        self.assertEqual(50, route.teleport.waypoint.x)
        self.assertEqual((), route.teleport.views)
        self.assertEqual('', route.map_layer)

    def test_views_can_replace_anchor_and_support_unicode_relative_paths(self):
        view_name = '地图.view-1.png'
        self.write_image(self.root / view_name)
        self.data['map_layer'] = '地下层 A'
        self.data['teleport'] = {
            'views': [{
                'map_image': view_name,
                'x': 70,
                'y': 40,
                'map_region': [10, 10, 110, 90],
            }],
        }

        route = self.load()

        self.assertEqual('地下层 A', route.map_layer)
        self.assertEqual('', route.teleport.anchor)
        self.assertIsNone(route.teleport.waypoint)
        self.assertEqual(1, len(route.teleport.views))
        view = route.teleport.views[0]
        self.assertIsInstance(view, RouteMapView)
        self.assertEqual(self.root / view_name, view.image)
        self.assertEqual((70, 40), (view.point.x, view.point.y))
        self.assertEqual((10, 10, 110, 90), view.region)

    def test_anchor_and_views_may_be_combined(self):
        view_name = 'map.view-1.png'
        self.write_image(self.root / view_name)
        self.data['teleport'] = {
            'anchor': 'Boss Anchor',
            'x': 50,
            'y': 50,
            'views': [{'map_image': view_name, 'x': 70, 'y': 40}],
        }
        route = self.load()
        self.assertEqual('Boss Anchor', route.teleport.anchor)
        self.assertIsNotNone(route.teleport.waypoint)
        self.assertEqual(1, len(route.teleport.views))

    def test_rejects_invalid_anchor_views_and_bookmark_geometry(self):
        view_name = 'map.view-1.png'
        self.write_image(self.root / view_name)
        invalid_specs = [
            {'views': []},
            {'anchor': None, 'views': [{'map_image': view_name, 'x': 50, 'y': 50}]},
            {'anchor': 42, 'views': []},
            {'anchor': '', 'x': 50, 'y': 50, 'views': []},
            {'views': [{'map_image': 'missing.png', 'x': 50, 'y': 50}]},
            {'views': [{'map_image': view_name, 'x': 120, 'y': 50}]},
            {'views': [{'map_image': view_name, 'x': 50, 'y': 50,
                        'map_region': [0, 0, 120, 101]}]},
            {'views': [{'map_image': view_name, 'x': 50, 'y': 50,
                        'map_region': [0, 0, 120, 100.0]}]},
            {'views': [{'map_image': view_name, 'x': True, 'y': 50}]},
            {'views': [None]},
            {'views': [{'map_image': view_name, 'x': 50, 'y': 50}] * 9},
        ]
        for spec in invalid_specs:
            with self.subTest(spec=spec):
                self.data['teleport'] = spec
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load()

    def test_rejects_invalid_map_layer_labels(self):
        for label in (None, 1, 'x' * 81, 'upper\nlower', 'upper\rlower'):
            with self.subTest(label=label):
                self.data['map_layer'] = label
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_legacy_anchor_without_teleport_waypoint(self):
        self.data['teleport'] = {'anchor': 'Boss Anchor'}
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()

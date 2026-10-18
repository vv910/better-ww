import tempfile
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from ok.test.TaskTestCase import TaskTestCase
from ok import Box

from config import config
from src.pathing.EchoRoute import EchoRoute
from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.RouteMarkers import MarkerMatch
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

    def test_low_contrast_user_minimap_reacquires_with_unchanged_geometry_thresholds(self):
        # Full-size reference preserves the original feature-budget failure; account UI is removed.
        image = cv2.imread('tests/images/3c-terrain-map.png')
        original = cv2.imread('tests/images/3c-terrain-minimap.png')
        localizer = MapLocalizer(image, (717, 173, 2227, 1296))
        for scale in (0.75, 1, 1.5):
            with self.subTest(scale=scale):
                localizer.reset()
                minimap = cv2.resize(original, None, fx=scale, fy=scale)
                result = localizer.locate(minimap)
                self.assertIsNotNone(result, localizer.last_diagnostic)
                self.assertEqual('sift', result.source)
                self.assertGreaterEqual(result.inliers, 6)
                # Approximate manual alignment from the beacon/star and nearby terrain.
                np.testing.assert_allclose(result.xy, (1321, 692), atol=8)

    def test_user_minimap_handles_contrast_and_jpeg_variation(self):
        original = cv2.imread('tests/images/3c-terrain-minimap.png')
        changed = np.clip(original.astype(np.float32) * 0.8 + 20, 0, 255).astype(np.uint8)
        encoded = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 65])[1]
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        localizer = MapLocalizer(cv2.imread('tests/images/3c-terrain-map.png'), (717, 173, 2227, 1296))
        for minimap in (changed, compressed):
            with self.subTest():
                localizer.reset()
                result = localizer.locate(minimap)
                self.assertIsNotNone(result, localizer.last_diagnostic)
                np.testing.assert_allclose(result.xy, (1321, 692), atol=8)

    def test_user_and_existing_map_pairs_reject_each_others_minimaps(self):
        new_map = cv2.imread('tests/images/3c-terrain-map.png')
        new_mini = cv2.imread('tests/images/3c-terrain-minimap.png')
        old_mini = cv2.imread('tests/images/mini_map.png')[30:211, 35:218]
        self.assertIsNone(MapLocalizer(new_map, (717, 173, 2227, 1296)).locate(old_mini))
        self.assertIsNone(MapLocalizer(cv2.imread('tests/images/path.png'),
                                       (538, 130, 1670, 972)).locate(new_mini))

    def test_real_minimap_visual_motion_bridges_a_temporary_absolute_failure(self):
        self.set_image('tests/images/mini_map.png')
        original = self.task.get_box_by_name('box_minimap').crop_frame(self.task.frame)
        for scale in (1, 4 / 3, 2):
            with self.subTest(scale=scale), patch('src.pathing.MapLocalizer.time.monotonic') as clock:
                clock.return_value = 0
                minimap = cv2.resize(original, None, fx=scale, fy=scale)
                localizer = MapLocalizer(cv2.imread('tests/images/path.png'), (538, 130, 1670, 972))
                start = localizer.locate(minimap)
                self.assertIsNotNone(start)
                transform = localizer.tracker.transform.copy()
                height, width = minimap.shape[:2]
                moved = cv2.warpAffine(minimap, np.float32([[1, 0, -4], [0, 1, 2]]), (width, height))
                expected = transform @ np.array([width // 2 + 4, height // 2 - 2, 1])
                clock.return_value = 0.6
                with patch.object(localizer, '_locate_absolute', return_value=None):
                    result = localizer.locate(moved)
                self.assertIsNotNone(result, localizer.last_diagnostic)
                self.assertEqual('flow', result.source)
                np.testing.assert_allclose(result.xy, expected[:2], atol=1)
                self.assertEqual(0, localizer.tracker.anchor_at)

    def test_route_template_warmup_uses_existing_features_before_moving(self):
        self.set_image('tests/images/mini_map.png')
        with patch.object(self.task, 'load_chars'), patch.object(self.task, 'prepare_fall_protection'), \
                patch('src.task.Farm3CEchoTask.og.my_app.yolo_detect') as detect:
            self.task.prepare_route_runtime()
        detect.assert_called_once()
        self.assertTrue(self.task.read_route_heading().reliable)

    def test_passive_route_enemy_detection_avoids_character_reload(self):
        self.task._in_combat = False
        with patch.object(self.task, 'load_chars') as load_chars:
            self.set_image('tests/images/mini_map.png')
            self.assertFalse(self.task.route_enemy_visible())
            self.set_image('tests/images/in_combat.png')
            self.assertTrue(self.task.route_enemy_visible())
        load_chars.assert_not_called()

    def test_reference_point_projects_on_real_map_after_zoom_and_pan(self):
        image = cv2.imread('tests/images/path.png')
        localizer = MapLocalizer(image, (538, 130, 1670, 972))
        transformed = cv2.warpAffine(image, np.float32([[1.1, 0, -100], [0, 1.1, 10]]), (1920, 1080))
        projected = localizer.project_map_point(transformed, (1207, 469), (538, 130, 1670, 972))
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(1227.7, projected.x, delta=3)
        self.assertAlmostEqual(525.9, projected.y, delta=3)

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

    def test_location_failure_preserves_last_good_minimap_and_logs_rejection_reason(self):
        self.set_image('tests/images/mini_map.png')
        self.task.localizer = MapLocalizer(cv2.imread('tests/images/path.png'), (538, 130, 1670, 972))
        self.assertIsNotNone(self.task.locate())
        last_good = self.task._last_good_minimap.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'blank.png')
            cv2.imwrite(path, np.zeros((1080, 1920, 3), np.uint8))
            self.set_image(path)
            self.assertIsNone(self.task.locate())
            with patch.object(self.task, 'screenshot') as screenshot, \
                    patch.object(self.task, 'log_warning') as warning:
                self.task.save_localization_diagnostic('lost')
        names = [call.args[0] for call in screenshot.call_args_list]
        self.assertEqual(['3c-location-lost-full', '3c-location-lost-minimap',
                          '3c-location-lost-last-good'], names)
        self.assertIn('too_few_minimap_features', warning.call_args.args[0])
        np.testing.assert_array_equal(last_good, screenshot.call_args_list[-1].kwargs['frame'])

    def test_record_real_map_markers_to_loadable_route(self):
        self.set_image('tests/images/path.png')
        recorder = Record3CEchoRouteTask(self.task.executor, None)
        recorder.feature_set = self.task.feature_set
        recorder.after_init(executor=self.task.executor, scene=self.task.scene)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'recorded.json'
            recorder.config = {'Route File': str(path), 'Route Name': 'marker test', 'Start Method': 'Diamond'}
            with patch.object(WWOneTimeTask, 'run'):
                recorder.run()
            route = EchoRoute.load(path)
            self.assertEqual(3, len(route.waypoints))
            self.assertTrue(all(point.action == 'combat' for point in route.waypoints))
            self.assertAlmostEqual(1140, route.waypoints[0].x, delta=15)
            self.assertAlmostEqual(550, route.waypoints[0].y, delta=15)
            # A hand-placed marker has an offset from the player arrow in this real fixture.
            self.assertLessEqual(route.start.distance_to((1207, 469)), route.start.tolerance)

    def test_record_user_overlapping_start_ignores_other_icons_and_distant_stars(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        crop = cv2.imread('tests/images/3c-map-overlap.png')
        frame[400:1120, 600:2250] = crop
        reference = cv2.imread('assets/images/0.png')[1022:1078, 764:824]
        star = cv2.resize(reference, (40, 37))
        white = star.min(axis=2) >= 170
        for x, y in ((1940, 930), (2090, 600)):
            region = frame[y:y + 37, x:x + 40]
            region[white] = star[white]
        with tempfile.TemporaryDirectory() as directory:
            capture = str(Path(directory) / 'capture.png')
            cv2.imwrite(capture, frame)
            self.set_image(capture)
            recorder = Record3CEchoRouteTask(self.task.executor, None)
            recorder.feature_set = self.task.feature_set
            recorder.after_init(executor=self.task.executor, scene=self.task.scene)
            path = Path(directory) / 'user-route.json'
            recorder.config = {'Route File': str(path), 'Route Name': 'overlap regression',
                               'Start Method': 'Diamond'}
            with patch.object(WWOneTimeTask, 'run'), patch.object(recorder, 'click_relative') as click:
                recorder.run()
            click.assert_not_called()  # Preserve the user's zoom and selected map view.
            route = EchoRoute.load(path)
            self.assertEqual(3, len(route.waypoints))
            self.assertEqual(2, len(recorder.excluded_stars))
            self.assertAlmostEqual(1431, route.start.x, delta=3)
            self.assertAlmostEqual(908, route.start.y, delta=3)
            self.assertTrue(path.with_suffix('.preview.png').exists())
            self.assertAlmostEqual(1477, route.waypoints[0].x, delta=3)

    def test_default_nearest_beacon_connects_the_central_group_to_its_entry(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        frame[400:1120, 600:2250] = cv2.imread('tests/images/3c-map-overlap.png')
        reference = cv2.imread('assets/images/0.png')[1022:1078, 764:824]
        star = cv2.resize(reference, (40, 37))
        white = star.min(axis=2) >= 170
        for x, y in ((1940, 930), (2090, 600)):
            region = frame[y:y + 37, x:x + 40]
            region[white] = star[white]
        with tempfile.TemporaryDirectory() as directory:
            capture_path = str(Path(directory) / 'nearest-default.png')
            cv2.imwrite(capture_path, frame)
            self.set_image(capture_path)
            recorder = Record3CEchoRouteTask(self.task.executor, None)
            recorder.feature_set = self.task.feature_set
            recorder.after_init(executor=self.task.executor, scene=self.task.scene)
            recorder.config = {'Route File': str(Path(directory) / 'nearest-default.json')}
            # No Start Method is configured: this exercises the public default mode.
            recorder.load_stars(wait_world=False)
            self.assertGreaterEqual(len(recorder.stars), 3)
            self.assertIsNotNone(recorder._entry_beacon)

    def test_all_visible_stars_records_every_real_star_in_the_central_area(self):
        with tempfile.TemporaryDirectory() as directory:
            self.set_image('tests/images/3c-map-dream-stars.png')
            results = {}
            for selection, expected in (('All Visible Stars', (3, 0)), ('Connected Group', (1, 2))):
                with self.subTest(selection=selection):
                    recorder = Record3CEchoRouteTask(self.task.executor, None)
                    recorder.feature_set = self.task.feature_set
                    recorder.after_init(executor=self.task.executor, scene=self.task.scene)
                    recorder.config = {'Route File': str(Path(directory) / f'{selection}.json'),
                                       'Star Selection': selection}
                    recorder.load_stars(wait_world=False)
                    results[selection] = (len(recorder.stars) - 1, len(recorder.excluded_stars))
                    self.assertEqual(expected, results[selection])
                    self.assertIsNotNone(recorder._entry_beacon)
            self.assertEqual({'All Visible Stars': (3, 0), 'Connected Group': (1, 2)}, results)

    def test_nearest_beacon_records_real_beacon_and_ignores_multiple_diamonds(self):
        frame = np.zeros((1440, 2560, 3), np.uint8)
        crop = cv2.imread('tests/images/3c-map-overlap.png')
        frame[400:1120, 600:2250] = crop
        with tempfile.TemporaryDirectory() as directory:
            capture_path = str(Path(directory) / 'map.png')
            cv2.imwrite(capture_path, frame)
            self.set_image(capture_path)
            recorder = Record3CEchoRouteTask(self.task.executor, None)
            recorder.feature_set = self.task.feature_set
            recorder.after_init(executor=self.task.executor, scene=self.task.scene)
            region = recorder.recording_region(frame)
            left, top, right, bottom = region
            beacons = recorder.find_feature(['map_way_point', 'map_way_point_big'], frame=frame,
                                            box=Box(left, top, right - left, bottom - top), threshold=.65)
            self.assertTrue(beacons, 'real blue waypoint beacon should match a map template')
            beacon = min(beacons, key=lambda item: math.dist(item.center(), (1431, 908)))
            bx, by = beacon.center()
            stars = [MarkerMatch(bx - 45, by - 35, 30, 30, .95),
                     MarkerMatch(bx - 15, by - 10, 30, 30, .95),
                     MarkerMatch(bx + 15, by + 15, 30, 30, .95)]
            diamonds = [MarkerMatch(700, 500, 30, 30, .95), MarkerMatch(800, 650, 30, 30, .95)]
            path = Path(directory) / 'nearest-route.json'
            recorder.config = {'Route File': str(path), 'Route Name': 'nearest beacon',
                               'Capture Teleport Overview': True,
                               'Map Selection Path': '区域\n梦境'}

            def fake_overviews(task, map_frame, point, map_region):
                return [(map_frame.copy(), point, map_region)]

            with patch.object(WWOneTimeTask, 'run'), \
                    patch('src.task.Farm3CEchoTask.RouteMarkerDetector') as detector, \
                    patch('src.task.Farm3CEchoTask.capture_map_views', side_effect=fake_overviews) as capture:
                detector.return_value.detect.return_value = (diamonds, stars)
                recorder.run()

            route = EchoRoute.load(path)
            import json
            self.assertEqual({'selection_path': ['区域', '梦境']}, json.loads(path.read_text(encoding='utf-8'))['map_context'])
            self.assertEqual((bx, by), (route.start.x, route.start.y))
            self.assertEqual(2, len(route.teleport.views))
            self.assertEqual(1, capture.call_count)
            self.assertTrue(path.with_suffix('.view-1.png').exists())


if __name__ == '__main__':
    unittest.main()

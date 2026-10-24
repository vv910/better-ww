import unittest

import cv2
import numpy as np

from src.pathing.RouteMapUI import RouteMapUI


class TestRouteMapUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Three small untouched control regions from the real recording frame.
        # Everything else, including account UI and terrain, has been removed.
        cls.frame = cv2.imread('tests/images/3c-map-current-ui.png')
        assert cls.frame is not None

    def setUp(self):
        self.detector = RouteMapUI()

    def test_real_controls_at_1080p_1440p_and_2160p(self):
        for scale in (0.75, 1, 1.5):
            for quality in (None, 78):
                with self.subTest(scale=scale, jpeg=quality):
                    frame = cv2.resize(self.frame, None, fx=scale, fy=scale)
                    if quality is not None:
                        encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
                        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    result = self.detector.detect(frame)
                    self.assertIsNotNone(result, self.detector.last_diagnostic)
                    np.testing.assert_allclose(result.center(), (2278 * scale, 86 * scale), atol=2)
                    self.assertEqual('three_map_controls_confirmed', self.detector.last_diagnostic['reason'])

    def test_height_scaled_controls_remain_anchored_to_the_right_edge(self):
        wider = np.zeros((1440, 3440, 3), np.uint8)
        wider[:, 880:] = self.frame
        result = self.detector.detect(wider)
        self.assertIsNotNone(result, self.detector.last_diagnostic)
        np.testing.assert_allclose(result.center(), (3158, 86), atol=1)

    def test_each_control_is_required(self):
        for name, (_, x, y) in self.detector.CONTROLS.items():
            changed = self.frame.copy()
            template = self.detector.templates[name]
            changed[y - 25:y + template.shape[0] + 25,
                    x - 25:x + template.shape[1] + 25] = 0
            with self.subTest(missing=name):
                self.assertIsNone(self.detector.detect(changed), self.detector.last_diagnostic)
                self.assertEqual(name + '_not_matched', self.detector.last_diagnostic['reason'])

    def test_zoom_buttons_must_be_in_the_same_column(self):
        changed = self.frame.copy()
        _, x, y = self.detector.CONTROLS['zoom_out']
        template = self.detector.templates['zoom_out']
        changed[y - 25:y + template.shape[0] + 25,
                x - 25:x + template.shape[1] + 25] = 0
        changed[y:y + template.shape[0], x + 12:x + 12 + template.shape[1]] = template
        self.assertIsNone(self.detector.detect(changed), self.detector.last_diagnostic)
        self.assertEqual('zoom_controls_not_in_same_column', self.detector.last_diagnostic['reason'])

    def test_swapping_plus_and_minus_is_rejected(self):
        changed = self.frame.copy()
        plus = self.detector.templates['zoom_in']
        minus = self.detector.templates['zoom_out']
        for name, replacement in (('zoom_in', minus), ('zoom_out', plus)):
            _, x, y = self.detector.CONTROLS[name]
            changed[y:y + replacement.shape[0], x:x + replacement.shape[1]] = replacement
        self.assertIsNone(self.detector.detect(changed), self.detector.last_diagnostic)

    def test_terrain_stars_and_player_triangles_are_not_recenter_controls(self):
        for shape in ('star', 'triangle'):
            changed = self.frame.copy()
            changed[25:145, 2220:2340] = 0
            if shape == 'triangle':
                points = np.int32([[2278, 57], [2250, 112], [2306, 112]])
            else:
                angles = np.arange(10) * np.pi / 5 - np.pi / 2
                radii = np.tile([29, 12], 5)
                points = np.rint(np.column_stack((2278 + np.cos(angles) * radii,
                                                  86 + np.sin(angles) * radii))).astype(np.int32)
            cv2.fillPoly(changed, [points], (245, 245, 245))
            self.assertIsNone(self.detector.detect(changed), self.detector.last_diagnostic)

    def test_world_combat_and_redacted_map_only_frames_are_rejected(self):
        for path in ('tests/images/mini_map.png', 'tests/images/in_combat.png',
                     'tests/images/in_combat3.png', 'tests/images/cloud_game_combat.png'):
            frame = cv2.imread(path)
            self.assertIsNotNone(frame, path)
            with self.subTest(path=path):
                self.assertIsNone(self.detector.detect(frame), self.detector.last_diagnostic)
        terrain = cv2.resize(cv2.imread('tests/images/big_map.png'), (2560, 1440))
        redacted = np.zeros_like(terrain)
        redacted[173:1296, 717:2227] = terrain[173:1296, 717:2227]
        self.assertIsNone(self.detector.detect(redacted), self.detector.last_diagnostic)

    def test_invalid_frame_and_missing_runtime_assets_fail_closed(self):
        for frame in (None, np.zeros((80, 80), np.uint8), np.zeros((100, 100, 3), np.uint8)):
            self.assertIsNone(self.detector.detect(frame))
        self.detector.templates['center'] = None
        self.assertIsNone(self.detector.detect(self.frame))
        self.assertEqual('missing_template', self.detector.last_diagnostic['reason'])


if __name__ == '__main__':
    unittest.main()

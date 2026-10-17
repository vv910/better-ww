import unittest

import cv2
import numpy as np

from src.pathing.MapPlayer import find_map_player


class TestMapPlayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = cv2.imread('assets/images/1.png')[188:301, 200:313]

    @staticmethod
    def render_player(frame_size, center, angle, scale, template):
        height, width = frame_size
        frame = np.full((height, width, 3), (55, 74, 67), np.uint8)
        # Map-like texture and irrelevant white/blue points must not count as players.
        cv2.line(frame, (0, height // 3), (width, height // 2), (180, 180, 180), 9)
        cv2.circle(frame, (width // 4, height // 4), 28, (255, 255, 255), 4)
        cv2.rectangle(frame, (width // 2, height // 3), (width // 2 + 24, height // 3 + 55),
                      (220, 170, 80), 5)
        from src.pathing.MinimapHeading import arrow_mask

        mask = arrow_mask(template)
        ys, xs = np.where(mask > 0)
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        rendered = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        h, w = rendered.shape
        diagonal = int(np.ceil(np.hypot(w, h)))
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), -angle, 1)
        matrix[0, 2] += (diagonal - w) / 2
        matrix[1, 2] += (diagonal - h) / 2
        rendered = cv2.warpAffine(rendered, matrix, (diagonal, diagonal), flags=cv2.INTER_NEAREST)
        ys, xs = np.where(rendered > 0)
        rendered = rendered[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        x, y = center[0] - rendered.shape[1] // 2, center[1] - rendered.shape[0] // 2
        gold = np.zeros((height, width), np.uint8)
        gold[y:y + rendered.shape[0], x:x + rendered.shape[1]] = rendered
        hsv = np.zeros((height, width, 3), np.uint8)
        hsv[:, :, 0] = 28
        hsv[:, :, 1] = 230
        hsv[:, :, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame[gold > 0] = bgr[gold > 0]
        return frame

    def test_finds_real_arrow_in_map_screenshots(self):
        image = cv2.imread('tests/images/path.png')
        result = find_map_player(image, self.template, (650, 250, 1600, 750))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(1207, result[0], delta=8)
        self.assertAlmostEqual(469, result[1], delta=8)

        image = cv2.imread('tests/images/3c-terrain-map.png')
        result = find_map_player(image, self.template, (850, 450, 1700, 1000), expected_center=(1280, 715))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(1279, result[0], delta=10)
        self.assertAlmostEqual(714, result[1], delta=10)

    def test_handles_capture_resolutions_scales_and_arrow_directions(self):
        for width, height, scale in ((1920, 1080, 0.72), (2560, 1440, 0.95), (3840, 2160, 1.25)):
            for angle in (0, 55, 135, 225, 315):
                with self.subTest(width=width, angle=angle):
                    center = (width * 3 // 5, height * 2 // 5)
                    frame = self.render_player((height, width), center, angle, scale, self.template)
                    result = find_map_player(frame, self.template, (width // 4, height // 6,
                                                                     width * 5 // 6, height * 5 // 6))
                    self.assertIsNotNone(result)
                    self.assertAlmostEqual(center[0], result[0], delta=5)
                    self.assertAlmostEqual(center[1], result[1], delta=5)

    def test_round_gold_shapes_and_white_blue_icons_are_rejected(self):
        frame = np.full((500, 700, 3), (55, 74, 67), np.uint8)
        cv2.circle(frame, (180, 180), 32, (0, 220, 255), -1)
        cv2.rectangle(frame, (330, 145), (520, 215), (0, 220, 255), -1)  # a yellow label
        cv2.circle(frame, (590, 350), 28, (255, 255, 255), -1)
        cv2.rectangle(frame, (70, 320), (105, 390), (255, 120, 20), -1)
        self.assertIsNone(find_map_player(frame, self.template, (0, 0, 700, 500)))

    def test_multiple_arrows_are_ambiguous_unless_expected_center_narrows_search(self):
        frame = self.render_player((800, 1200), (350, 380), 20, 0.8, self.template)
        second = self.render_player((800, 1200), (850, 380), 180, 0.8, self.template)
        # Merge only the gold foreground from the second rendered map into the first.
        from src.pathing.MinimapHeading import arrow_mask

        mask = arrow_mask(second)
        frame[mask > 0] = second[mask > 0]
        self.assertIsNone(find_map_player(frame, self.template, (0, 0, 1200, 800)))
        result = find_map_player(frame, self.template, (0, 0, 1200, 800), expected_center=(365, 385))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(350, result[0], delta=5)
        self.assertAlmostEqual(380, result[1], delta=5)

    def test_region_and_expected_center_are_strictly_validated(self):
        frame = np.zeros((200, 300, 3), np.uint8)
        for region in ((-1, 0, 200, 100), (0, 0, 301, 100), (20, 20, 20, 100), (0, 0, 100.5, 100)):
            with self.subTest(region=region), self.assertRaises(ValueError):
                find_map_player(frame, self.template, region)
        with self.assertRaisesRegex(ValueError, 'inside region'):
            find_map_player(frame, self.template, (10, 10, 200, 150), expected_center=(250, 100))


if __name__ == '__main__':
    unittest.main()

import unittest

import cv2
import numpy as np

from src.pathing.RouteMarkers import RouteMarkerDetector


class TestRouteMarkers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image = cv2.imread('assets/images/0.png')
        cls.diamond = image[1046:1118, 1878:1962]
        cls.star = image[1022:1078, 764:824]
        # Cropped user regression image: no UID, account UI or Windows watermark.
        cls.map = cv2.imread('tests/images/3c-map-overlap.png')

    def detector(self, capture_width=2560):
        scale = capture_width / 3840
        return RouteMarkerDetector(cv2.resize(self.diamond, None, fx=scale, fy=scale),
                                   cv2.resize(self.star, None, fx=scale, fy=scale))

    def test_overlapping_arrow_and_beacon_hide_start_from_legacy_rgb_matcher(self):
        reference = cv2.resize(self.diamond, (56, 48))
        _, confidence, _, _ = cv2.minMaxLoc(cv2.matchTemplate(self.map, reference, cv2.TM_CCOEFF_NORMED))
        self.assertLess(confidence, 0.7)
        diamonds, stars = self.detector().detect(self.map, capture_width=2560)
        self.assertEqual(1, len(diamonds))
        self.assertTrue(diamonds[0].partial)
        self.assertGreater(diamonds[0].confidence, 0.9)
        self.assertAlmostEqual(831.7, diamonds[0].center()[0], delta=3)
        self.assertAlmostEqual(507.7, diamonds[0].center()[1], delta=3)
        self.assertEqual(3, len(stars))  # Skulls, Boss badges and blue markers are ignored.

    def test_resizing_user_capture_preserves_marker_types_and_count(self):
        for width in (1920, 2560, 3840):
            with self.subTest(capture_width=width):
                scale = width / 2560
                frame = cv2.resize(self.map, None, fx=scale, fy=scale)
                diamonds, stars = self.detector(width).detect(frame, capture_width=width)
                self.assertEqual(1, len(diamonds))
                self.assertEqual(3, len(stars))
                self.assertAlmostEqual(831.7 * scale, diamonds[0].center()[0], delta=4)

    def test_blank_and_missing_gem_do_not_produce_an_invented_start(self):
        for frame in (np.zeros_like(self.map), self.map[:350].copy()):
            diamonds, _ = self.detector().detect(frame, capture_width=2560)
            self.assertEqual([], diamonds)

    def test_multiple_gems_remain_distinct_instead_of_picking_highest_confidence(self):
        frame = self.map.copy()
        template = cv2.resize(self.diamond, (56, 48))
        region = frame[30:78, 50:106]
        white = template.min(axis=2) > 170
        region[white] = template[white]
        diamonds, _ = self.detector().detect(frame, capture_width=2560)
        self.assertEqual(2, len(diamonds))

    def test_a_white_gem_top_without_lower_support_is_not_a_start(self):
        template = cv2.resize(self.diamond, (56, 48))
        frame = np.zeros((300, 400, 3), np.uint8)
        frame[50:72, 100:156] = template[:22]
        diamonds, _ = self.detector().detect(frame, capture_width=2560)
        self.assertEqual([], diamonds)


if __name__ == '__main__':
    unittest.main()

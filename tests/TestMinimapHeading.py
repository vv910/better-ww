import unittest

import cv2
import numpy as np

from src.pathing.MinimapHeading import MinimapHeading


class TestMinimapHeading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The shipped east-facing feature, before capture-resolution scaling.
        cls.arrow = cv2.imread('assets/images/1.png')[188:301, 200:313]
        cls.detector = MinimapHeading(cls.arrow)

    def render_arrow(self, angle, size=372, scale=1, bright_terrain=False):
        image = np.full((372, 372, 3), 60, np.uint8)
        if bright_terrain:
            # White streets, blue map markers and green terrain behind the player.
            cv2.rectangle(image, (100, 100), (280, 280), (90, 150, 90), -1)
            cv2.line(image, (80, 186), (300, 186), (220, 220, 220), 15)
            cv2.circle(image, (230, 185), 15, (240, 210, 120), -1)
        transform = cv2.getRotationMatrix2D((56, 56), -angle, scale)
        arrow = cv2.warpAffine(self.arrow, transform, (113, 113))
        region = image[130:243, 130:243]
        visible = np.any(arrow > 20, axis=2)
        region[visible] = arrow[visible]
        return cv2.resize(image, (size, size))

    def test_direction_is_preserved_across_resolution_scale_and_background(self):
        for angle in (0, 27, 90, 137, 180, 234, 270, 345):
            for size, scale in ((124, 0.85), (186, 1), (248, 1.2), (372, 1)):
                with self.subTest(angle=angle, minimap_size=size, scale=scale):
                    result = self.detector.estimate(self.render_arrow(angle, size, scale, True))
                    self.assertIsNotNone(result)
                    self.assertTrue(result.reliable, result)
                    error = abs((result.angle - angle + 180) % 360 - 180)
                    self.assertLessEqual(error, 6)

    def test_blank_and_gold_icons_cannot_supply_a_heading(self):
        for shape in ('blank', 'circle', 'square', 'diamond'):
            with self.subTest(shape=shape):
                image = np.full((186, 186, 3), 60, np.uint8)
                gold = (60, 220, 250)
                if shape == 'circle':
                    cv2.circle(image, (93, 93), 18, gold, -1)
                elif shape == 'square':
                    cv2.rectangle(image, (76, 76), (110, 110), gold, -1)
                elif shape == 'diamond':
                    cv2.fillPoly(image, [np.array([[93, 70], [116, 93], [93, 116], [70, 93]])], gold)
                result = self.detector.estimate(image)
                self.assertTrue(result is None or not result.reliable, result)

    def test_arrow_away_from_player_center_is_rejected(self):
        image = self.render_arrow(90, size=186)
        image = cv2.warpAffine(image, np.float32([[1, 0, 40], [0, 1, 30]]), (186, 186))
        result = self.detector.estimate(image)
        self.assertTrue(result is None or not result.reliable, result)

    def test_unusable_input_is_rejected(self):
        self.assertIsNone(self.detector.estimate(None))
        self.assertIsNone(self.detector.estimate(np.zeros((20, 20, 3), np.uint8)))
        with self.assertRaisesRegex(ValueError, 'no usable gold'):
            MinimapHeading(np.zeros((57, 57, 3), np.uint8))


if __name__ == '__main__':
    unittest.main()

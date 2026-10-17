import unittest

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.TerrainMask import terrain_feature_mask


class TestTerrainMask(unittest.TestCase):
    def test_gray_icon_only_reference_is_rejected_without_shared_icon_localization(self):
        reference = np.full((650, 850), 65, np.uint8)
        query = np.full((200, 200), 85, np.uint8)
        for x, y in ((45, 45), (155, 45), (45, 155), (155, 155)):
            cv2.circle(query, (x, y), 13, 255, -1)
            cv2.circle(reference, (x + 380, y + 280), 13, 255, -1)
        localizer = MapLocalizer(reference)
        self.assertIsNone(localizer.locate(query), localizer.last_diagnostic)
        self.assertEqual('too_few_minimap_features', localizer.last_diagnostic['reason'])

    def test_mask_excludes_padded_white_blue_and_gold_markers_but_keeps_terrain(self):
        image = np.full((240, 320, 3), 80, np.uint8)
        cv2.circle(image, (60, 80), 8, (255, 255, 255), -1)
        cv2.circle(image, (150, 80), 8, (255, 110, 20), -1)  # blue/cyan marker in BGR
        cv2.circle(image, (240, 80), 8, (0, 210, 255), -1)  # gold marker in BGR
        mask = terrain_feature_mask(image)
        for x in (60, 150, 240):
            self.assertEqual(0, mask[80, x])
            self.assertEqual(0, mask[80, x + 12])
        self.assertEqual(255, mask[180, 160])

    def test_grayscale_textured_terrain_still_localizes(self):
        rng = np.random.default_rng(7112)
        noise = rng.integers(0, 256, (850, 1100), dtype=np.uint8)
        reference = cv2.GaussianBlur(noise, (7, 7), 1.1)
        reference = np.clip(reference.astype(np.float32) * .45 + 35, 0, 255).astype(np.uint8)
        query = reference[300:500, 400:600].copy()
        localizer = MapLocalizer(reference)
        result = localizer.locate(query)
        self.assertIsNotNone(result, localizer.last_diagnostic)
        np.testing.assert_allclose(result.xy, (500, 400), atol=2)


if __name__ == '__main__':
    unittest.main()

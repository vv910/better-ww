import unittest

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer


class TestMapFeatureRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(261017)
        # Simulated HUD has stronger corners than the dim terrain inside the requested region.
        cls.busy = rng.integers(0, 256, (1000, 1500), np.uint8)
        terrain = cv2.GaussianBlur(rng.integers(0, 256, (500, 500), np.uint8), (5, 5), 0.8)
        cls.terrain = (terrain.astype(np.float32) * 0.35 + 55).astype(np.uint8)
        cls.region = (450, 250, 950, 750)
        cls.busy[250:750, 450:950] = cls.terrain
        cls.quiet = np.zeros_like(cls.busy)
        cls.quiet[250:750, 450:950] = cls.terrain

    def test_strong_features_outside_region_cannot_starve_dim_terrain(self):
        localizer = MapLocalizer(self.busy, self.region)
        for x, y in ((600, 400), (800, 550), (700, 600)):
            with self.subTest(position=(x, y)):
                localizer.reset()
                result = localizer.locate(self.busy[y - 100:y + 100, x - 100:x + 100])
                self.assertIsNotNone(result, localizer.last_diagnostic)
                self.assertAlmostEqual(x, result.x, delta=2)
                self.assertAlmostEqual(y, result.y, delta=2)

    def test_changing_outside_ui_does_not_change_reference_features(self):
        busy = MapLocalizer(self.busy, self.region)
        quiet = MapLocalizer(self.quiet, self.region)
        np.testing.assert_array_equal(busy.locations, quiet.locations)
        np.testing.assert_array_equal(busy.descriptors, quiet.descriptors)

    def test_terrain_inside_region_is_not_confused_with_outside_content(self):
        localizer = MapLocalizer(self.busy, self.region)
        self.assertIsNone(localizer.locate(self.busy[20:220, 20:220]))

    def test_map_projection_restores_the_nonzero_screen_region_offset(self):
        localizer = MapLocalizer(self.quiet, self.region)
        projection = localizer.project_map_point(cv2.cvtColor(self.busy, cv2.COLOR_GRAY2BGR),
                                                 (680, 490), self.region)
        self.assertIsNotNone(projection)
        self.assertAlmostEqual(680, projection.x, delta=1)
        self.assertAlmostEqual(490, projection.y, delta=1)

    def test_full_image_default_and_explicit_regions_are_equivalent(self):
        implicit = MapLocalizer(self.terrain)
        explicit = MapLocalizer(self.terrain, (0, 0, 500, 500))
        minimap = self.terrain[150:350, 180:380]
        a, b = implicit.locate(minimap), explicit.locate(minimap)
        self.assertIsNotNone(a)
        np.testing.assert_allclose(a.xy, b.xy, atol=0.1)
        np.testing.assert_allclose(a.xy, (280, 250), atol=2)

    def test_invalid_regions_reject_before_opencv_cropping(self):
        for region in ((-1, 0, 500, 500), (0, -1, 500, 500), (0, 0, 501, 500),
                       (0, 0, 500, 501), (20, 0, 10, 500), (0, 10, 500, 10)):
            with self.subTest(region=region), self.assertRaisesRegex(ValueError, 'region'):
                MapLocalizer(self.terrain, region)
        localizer = MapLocalizer(self.terrain)
        with self.assertRaisesRegex(ValueError, 'region'):
            localizer.project_map_point(cv2.cvtColor(self.terrain, cv2.COLOR_GRAY2BGR),
                                        (100, 100), (-2, 0, 500, 500))

    def test_other_map_regions_hold_out_across_scale_and_contrast(self):
        # Independent pre-existing sea/cliff map; none of these patches supplied the fix.
        image = cv2.imread('tests/images/big_map.png')
        localizer = MapLocalizer(image, (538, 130, 1670, 972))
        for x, y in ((650, 650), (1100, 620), (750, 350)):
            for scale in (0.75, 1.0, 1.4):
                with self.subTest(position=(x, y), scale=scale):
                    minimap = cv2.resize(image[y - 100:y + 100, x - 100:x + 100], None,
                                         fx=scale, fy=scale)
                    minimap = np.clip(minimap.astype(np.float32) * 0.8 + 15, 0, 255).astype(np.uint8)
                    localizer.reset()
                    result = localizer.locate(minimap)
                    self.assertIsNotNone(result, localizer.last_diagnostic)
                    np.testing.assert_allclose(result.xy, (x, y), atol=3)


if __name__ == '__main__':
    unittest.main()

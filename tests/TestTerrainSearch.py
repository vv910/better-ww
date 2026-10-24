import unittest

import cv2
import numpy as np

from src.pathing.TerrainSearch import TerrainSearch


def minimap_mask(shape):
    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)
    radius = int(min(height, width) * 0.46)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, -1)
    cv2.circle(mask, (width // 2, height // 2), int(radius * 0.23), 0, -1)
    return mask


class TestTerrainSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = cv2.getNumThreads()
        cv2.setNumThreads(2)
        cls.pairs = (
            (cv2.imread('tests/images/3c-terrain-map.png'),
             cv2.imread('tests/images/3c-terrain-minimap.png'),
             (717, 173, 2227, 1296), (1322.187, 691.153)),
            (cv2.imread('tests/images/path.png'),
             cv2.imread('tests/images/mini_map.png')[30:211, 35:218],
             (538, 130, 1670, 972), (1206.196, 468.964)),
        )

    @classmethod
    def tearDownClass(cls):
        cv2.setNumThreads(cls.original_threads)

    def test_independent_real_maps_and_capture_scales(self):
        # Coordinates are assertions/hints from independent big-map registration;
        # no SIFT descriptors or SIFT transform are passed to the matcher.
        for reference, query, region, expected in self.pairs:
            search = TerrainSearch(reference, region)
            for factor in (0.75, 1, 1.4):
                with self.subTest(map=reference.shape, factor=factor):
                    resized = cv2.resize(query, None, fx=factor, fy=factor)
                    result = search.search(resized, minimap_mask(resized.shape), expected)
                    self.assertIsNotNone(result, search.last_diagnostic)
                    np.testing.assert_allclose(result.xy, expected, atol=3)
                    self.assertGreaterEqual(search.last_diagnostic['margin'], search.MIN_MARGIN)

    def test_missing_invalid_and_displaced_hints_cannot_supply_a_position(self):
        reference, query, region, expected = self.pairs[0]
        search = TerrainSearch(reference, region)
        for hint in (None, (float('nan'), 200), (1,), ('bad', 300),
                     (-1, -1), (expected[0] + 40, expected[1])):
            with self.subTest(hint=hint):
                self.assertIsNone(search.search(query, minimap_mask(query.shape), hint),
                                  search.last_diagnostic)

    def test_image_evidence_refines_an_imprecise_hint(self):
        for reference, query, region, expected in self.pairs:
            search = TerrainSearch(reference, region)
            hint = (expected[0] + 8, expected[1] - 5)
            result = search.search(query, minimap_mask(query.shape), hint)
            self.assertIsNotNone(result, search.last_diagnostic)
            np.testing.assert_allclose(result.xy, expected, atol=3)
            self.assertGreater(np.linalg.norm(np.subtract(result.xy, hint)), 5)

    def test_wrong_map_pair_is_rejected_in_both_directions(self):
        for index, (reference, _, region, hint) in enumerate(self.pairs):
            search = TerrainSearch(reference, region)
            query = self.pairs[1 - index][1]
            self.assertIsNone(search.search(query, minimap_mask(query.shape), hint),
                              search.last_diagnostic)

    def test_rotated_map_cannot_bootstrap_tracking(self):
        for reference, query, region, hint in self.pairs:
            search = TerrainSearch(reference, region)
            height, width = query.shape[:2]
            for degrees in (8, 20):
                with self.subTest(map=reference.shape, rotation=degrees):
                    rotated = cv2.warpAffine(query, cv2.getRotationMatrix2D(
                        (width / 2, height / 2), degrees, 1), (width, height))
                    self.assertIsNone(search.search(rotated, minimap_mask(rotated.shape), hint),
                                      search.last_diagnostic)

    def test_repeated_terrain_fails_global_uniqueness(self):
        rng = np.random.default_rng(41)
        tile = cv2.GaussianBlur(rng.integers(40, 190, (72, 72), np.uint8), (5, 5), 1)
        reference = np.tile(tile, (8, 8))
        query = reference[180:360, 180:360].copy()
        search = TerrainSearch(reference, (0, 0, 576, 576))
        self.assertIsNone(search.search(query, minimap_mask(query.shape), (270, 270)),
                          search.last_diagnostic)

    def test_blank_icon_only_and_sparse_or_concentrated_masks_are_rejected(self):
        reference, query, region, hint = self.pairs[0]
        search = TerrainSearch(reference, region)
        blank = np.full_like(query, 80)
        cv2.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 18, (255, 255, 255), -1)
        self.assertIsNone(search.search(blank, minimap_mask(blank.shape), hint))
        sparse = np.zeros(query.shape[:2], np.uint8)
        sparse[30:40, 30:40] = 255
        self.assertIsNone(search.search(query, sparse, hint))
        concentrated = np.zeros(query.shape[:2], np.uint8)
        concentrated[10:115, 10:115] = 255
        self.assertIsNone(search.search(query, concentrated, hint))

    def test_invalid_regions_reject_before_preprocessing(self):
        reference = self.pairs[0][0]
        for region in ((0, 0, 0, 100), (-1, 0, 50, 50), (0, 0, 9999, 50), (0, 0, 50.1, 50)):
            with self.subTest(region=region), self.assertRaises(ValueError):
                TerrainSearch(reference, region)


if __name__ == '__main__':
    unittest.main()

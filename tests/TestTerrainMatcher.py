import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer, MapPosition
from src.pathing.TerrainMatcher import TerrainMatcher


def minimap_mask(shape):
    height, width = shape
    mask = np.zeros((height, width), np.uint8)
    radius = int(min(width, height) * 0.46)
    cv2.circle(mask, (width // 2, height // 2), radius, 255, -1)
    cv2.circle(mask, (width // 2, height // 2), int(radius * 0.23), 0, -1)
    return mask


class TestTerrainMatcher(unittest.TestCase):
    @staticmethod
    def load_pair(family):
        if family == 'user':
            reference = cv2.imread('tests/images/3c-terrain-map.png')
            query = cv2.imread('tests/images/3c-terrain-minimap.png')
            region = (717, 173, 2227, 1296)
        else:
            reference = cv2.imread('tests/images/path.png')
            query = cv2.imread('tests/images/mini_map.png')[30:211, 35:218]
            region = (538, 130, 1670, 972)
        return reference, query, region

    def verify_pair(self, family):
        reference, query, region = self.load_pair(family)
        localizer = MapLocalizer(reference, region)
        position = localizer.locate(query)
        self.assertIsNotNone(position, localizer.last_diagnostic)
        matcher = TerrainMatcher()
        result = matcher.verify(localizer.gray, query, minimap_mask(query.shape[:2]),
                                region, localizer.tracker.transform[:2])
        self.assertIsNotNone(result, matcher.last_diagnostic)
        self.assertLessEqual(np.linalg.norm(np.subtract(result.xy, position.xy)), 3)
        self.assertGreaterEqual(matcher.last_diagnostic['supported_sectors'], 3)
        self.assertGreaterEqual(matcher.last_diagnostic['margin'], TerrainMatcher.MIN_PEAK_MARGIN)
        return reference, query, region, localizer, result

    def test_verifies_two_independent_real_map_families(self):
        self.verify_pair('user')
        self.verify_pair('reference')

    def test_cross_map_pair_is_rejected(self):
        map_a, query_a, region_a, localizer_a, _ = self.verify_pair('user')
        map_b, query_b, region_b, localizer_b, _ = self.verify_pair('reference')
        cases = ((map_b, query_a, region_b, localizer_a.tracker.transform[:2]),
                 (map_a, query_b, region_a, localizer_b.tracker.transform[:2]))
        for reference, query, region, transform in cases:
            with self.subTest(query_size=query.shape):
                matcher = TerrainMatcher()
                result = matcher.verify(reference, query, minimap_mask(query.shape[:2]),
                                        region, transform)
                self.assertIsNone(result, matcher.last_diagnostic)

    def test_repeated_terrain_texture_is_rejected_as_ambiguous(self):
        rng = np.random.default_rng(41)
        tile = rng.integers(0, 256, (72, 72), dtype=np.uint8)
        reference = np.tile(tile, (6, 6))
        query = reference[144:288, 144:288].copy()
        transform = np.float32([[1, 0, 144], [0, 1, 144]])
        matcher = TerrainMatcher()
        result = matcher.verify(reference, query, minimap_mask(query.shape),
                                (0, 0, reference.shape[1], reference.shape[0]), transform)
        self.assertIsNone(result)
        self.assertEqual('terrain_peak_ambiguous', matcher.last_diagnostic['reason'])

    def test_blank_or_icon_only_query_is_rejected(self):
        rng = np.random.default_rng(42)
        reference = cv2.GaussianBlur(rng.integers(0, 256, (500, 600), dtype=np.uint8), (5, 5), 1)
        query = np.zeros((180, 180), np.uint8)
        cv2.circle(query, (90, 90), 24, 255, -1)  # lies in the intentionally masked player center
        matcher = TerrainMatcher()
        result = matcher.verify(reference, query, minimap_mask(query.shape),
                                (0, 0, 600, 500), np.float32([[1, 0, 210], [0, 1, 160]]))
        self.assertIsNone(result)

    def test_shared_white_icons_cannot_dominate_terrain_verification(self):
        reference = np.full((650, 850), 65, np.uint8)
        query = np.full((200, 200), 85, np.uint8)
        for x, y in ((45, 45), (155, 45), (45, 155), (155, 155)):
            cv2.circle(query, (x, y), 13, 255, -1)
            cv2.circle(reference, (x + 380, y + 280), 13, 255, -1)
        matcher = TerrainMatcher()
        verified_mask_sizes = []
        original_correlation = matcher._masked_correlation_map

        def record_mask(image, template, mask):
            verified_mask_sizes.append(int(np.count_nonzero(mask)))
            return original_correlation(image, template, mask)

        with patch.object(matcher, '_masked_correlation_map', side_effect=record_mask):
            result = matcher.verify(reference, query, np.full(query.shape, 255, np.uint8),
                                    (0, 0, 850, 650), np.float32([[1, 0, 380], [0, 1, 280]]))
        self.assertIsNone(result)
        self.assertTrue(verified_mask_sizes)
        self.assertLess(verified_mask_sizes[0], query.size)

    def test_direct_bgr_verification_masks_blue_and_gold_icon_neighborhoods(self):
        reference = np.full((650, 850), 65, np.uint8)
        query = np.full((200, 200, 3), (85, 85, 85), np.uint8)
        for (x, y), color in zip(((45, 45), (155, 45), (45, 155), (155, 155)),
                                 ((255, 110, 20), (0, 210, 255),
                                  (255, 110, 20), (0, 210, 255))):
            cv2.circle(query, (x, y), 13, color, -1)
            cv2.circle(reference, (x + 380, y + 280), 13, 255, -1)
        matcher = TerrainMatcher()
        verified_mask_sizes = []
        original_correlation = matcher._masked_correlation_map

        def record_mask(image, template, mask):
            verified_mask_sizes.append(int(np.count_nonzero(mask)))
            return original_correlation(image, template, mask)

        with patch.object(matcher, '_masked_correlation_map', side_effect=record_mask):
            result = matcher.verify(reference, query, np.full(query.shape[:2], 255, np.uint8),
                                    (0, 0, 850, 650), np.float32([[1, 0, 380], [0, 1, 280]]))
        self.assertIsNone(result)
        self.assertTrue(verified_mask_sizes)
        self.assertLess(verified_mask_sizes[0], query.shape[0] * query.shape[1] * .9)

    def test_masked_correlation_accepts_subunit_and_superunit_map_scales(self):
        rng = np.random.default_rng(9017)
        reference = cv2.GaussianBlur(rng.integers(45, 165, (800, 900), dtype=np.uint8), (5, 5), 0.8)
        patch = reference[250:410, 300:460]
        for scale in (0.5, 1.4):
            with self.subTest(scale=scale):
                query = cv2.resize(patch, None, fx=1 / scale, fy=1 / scale,
                                   interpolation=cv2.INTER_LINEAR)
                transform = np.float32([[scale, 0, 300], [0, scale, 250]])
                matcher = TerrainMatcher()
                result = matcher.verify(reference, query, np.full(query.shape, 255, np.uint8),
                                        (0, 0, 900, 800), transform)
                self.assertIsNotNone(result, matcher.last_diagnostic)
                np.testing.assert_allclose(result.xy, (380, 330), atol=2)

    def test_masked_correlation_rejects_insufficient_terrain_coverage(self):
        rng = np.random.default_rng(219)
        reference = cv2.GaussianBlur(rng.integers(0, 256, (500, 600), dtype=np.uint8), (5, 5), 1)
        query = reference[150:350, 200:400].copy()
        mask = np.zeros(query.shape, np.uint8)
        cv2.rectangle(mask, (95, 95), (105, 105), 255, -1)
        matcher = TerrainMatcher()
        result = matcher.verify(reference, query, mask, (0, 0, 600, 500),
                                np.float32([[0.5, 0, 250], [0, 0.5, 200]]))
        self.assertIsNone(result)
        self.assertEqual('insufficient_terrain_mask', matcher.last_diagnostic['reason'])

    def test_rejects_rotation_and_invalid_region(self):
        reference, query, region, localizer, _ = self.verify_pair('user')
        matcher = TerrainMatcher()
        center = np.array([query.shape[1] / 2, query.shape[0] / 2])
        theta = np.deg2rad(20)
        linear = np.float32([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        rotated = np.column_stack((linear, np.array([1321, 692]) - linear @ center))
        self.assertIsNone(matcher.verify(localizer.gray, query, minimap_mask(query.shape[:2]),
                                         region, rotated))
        self.assertEqual('minimap_not_north_up', matcher.last_diagnostic['reason'])
        self.assertIsNone(matcher.verify(localizer.gray, query, minimap_mask(query.shape[:2]),
                                         (100, 100, 100, 200), localizer.tracker.transform[:2]))
        self.assertEqual('invalid_reference_region', matcher.last_diagnostic['reason'])

    def test_only_concentrated_sift_proposal_uses_terrain_fallback(self):
        reference, query, region = self.load_pair('user')
        localizer = MapLocalizer(reference, region)
        center = (query.shape[1] // 2, query.shape[0] // 2)
        self.assertIsNotNone(localizer.locate(query))
        transform = localizer.tracker.transform[:2].copy()
        localizer.reset()

        def concentrated(*args, **kwargs):
            localizer._terrain_proposal = {'transform': transform, 'confidence': 0.5,
                                           'inliers': 6, 'spread': np.array([40.0, 60.0])}
            return localizer._reject('features_too_concentrated')

        with patch.object(localizer, '_match', side_effect=concentrated):
            result = localizer.locate(query)
        self.assertIsNotNone(result, localizer.last_diagnostic)
        self.assertEqual('terrain', result.source)
        self.assertEqual('terrain', localizer.last_diagnostic['method'])
        self.assertAlmostEqual(1322, result.x, delta=5)
        self.assertAlmostEqual(691, result.y, delta=5)
        self.assertGreaterEqual(localizer.last_diagnostic['terrain_verification']['supported_sectors'], 3)

    def test_locates_player_from_scaled_map_view_inside_nonzero_screen_roi(self):
        reference = cv2.imread('tests/images/path.png')
        route_region = (538, 130, 1670, 972)
        localizer = MapLocalizer(reference, route_region)
        warp = np.float32([[1.1, 0, -100], [0, 1.1, 10]])
        map_view = cv2.warpAffine(reference, warp, (1920, 1080))
        frame = np.zeros((1200, 2200, 3), np.uint8)
        frame[50:1130, 100:2020] = map_view
        screen_region = (100, 50, 2020, 1130)
        player_in_view = warp @ np.array([1207, 469, 1], dtype=np.float32)
        player_in_frame = player_in_view + np.array([100, 50], dtype=np.float32)
        result = localizer.locate_map_player(frame, player_in_frame, screen_region)
        self.assertIsNotNone(result, localizer.last_diagnostic)
        self.assertEqual('map', result.source)
        np.testing.assert_allclose(result.xy, (1207, 469), atol=3)

    def test_map_player_locator_rejects_wrong_icon_bounds_and_cross_map(self):
        reference = cv2.imread('tests/images/path.png')
        localizer = MapLocalizer(reference, (538, 130, 1670, 972))
        warp = np.float32([[1.1, 0, -100], [0, 1.1, 10]])
        map_view = cv2.warpAffine(reference, warp, (1920, 1080))
        self.assertIsNone(localizer.locate_map_player(map_view, (20, 20), (100, 50, 1900, 1000)))
        self.assertEqual('player_marker_outside_screen_region', localizer.last_diagnostic['reason'])

        rng = np.random.default_rng(810)
        unrelated = cv2.GaussianBlur(rng.integers(0, 256, map_view.shape, dtype=np.uint8),
                                     (7, 7), 1.2)
        result = localizer.locate_map_player(unrelated, (1227, 525), (0, 0, 1920, 1080))
        self.assertIsNone(result, localizer.last_diagnostic)

    def test_reset_map_hint_guides_but_does_not_replace_minimap_match(self):
        reference = cv2.imread('tests/images/path.png')
        minimap = cv2.imread('tests/images/mini_map.png')[30:211, 35:218]
        localizer = MapLocalizer(reference, (538, 130, 1670, 972))
        localizer.reset(position_hint=MapPosition(1207, 469, 0.8, 20, source='map'))
        self.assertEqual('hint', localizer.previous.source)
        result = localizer.locate(minimap)
        self.assertIsNotNone(result)
        self.assertEqual('sift', result.source)
        self.assertAlmostEqual(1207, result.x, delta=5)
        self.assertEqual('matched', localizer.last_diagnostic['reason'])
        with self.assertRaisesRegex(ValueError, 'locate_map_player'):
            localizer.reset(position_hint=MapPosition(1207, 469, 0.8, 20, source='sift'))


if __name__ == '__main__':
    unittest.main()

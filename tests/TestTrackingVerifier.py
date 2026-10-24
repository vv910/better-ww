import unittest

import cv2
import numpy as np

from src.pathing.MapLocalizer import MapLocalizer
from src.pathing.MinimapTracker import MinimapTracker, TrackedMinimap
from src.pathing.TerrainMask import terrain_feature_mask
from src.pathing.TrackingVerifier import TrackingVerifier


def minimap_mask(shape):
    height, width = shape[:2]
    result = np.zeros((height, width), np.uint8)
    radius = int(min(height, width) * 0.46)
    cv2.circle(result, (width // 2, height // 2), radius, 255, -1)
    cv2.circle(result, (width // 2, height // 2), int(radius * 0.23), 0, -1)
    return result


def candidate(transform, shape, confidence=0.95, inliers=30):
    affine = np.asarray(transform, dtype=np.float64)
    if affine.shape == (2, 3):
        affine = np.vstack((affine, (0, 0, 1)))
    center = np.float64([shape[1] // 2, shape[0] // 2, 1])
    xy = affine @ center
    return TrackedMinimap(affine, (float(xy[0]), float(xy[1])), confidence, inliers)


class TestTrackingVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = cv2.getNumThreads()
        cv2.setNumThreads(2)
        cls.pairs = []
        for map_path, query_path, region in (
                ('tests/images/3c-terrain-map.png', 'tests/images/3c-terrain-minimap.png',
                 (717, 173, 2227, 1296)),
                ('tests/images/path.png', 'tests/images/mini_map.png', (538, 130, 1670, 972))):
            reference, query = cv2.imread(map_path), cv2.imread(query_path)
            if 'mini_map' in query_path:
                query = query[30:211, 35:218]
            locator = MapLocalizer(reference, region)
            position = locator.locate(query)
            assert position is not None, locator.last_diagnostic
            cls.pairs.append((reference, query, region, locator.tracker.transform[:2].copy()))

    @classmethod
    def tearDownClass(cls):
        cv2.setNumThreads(cls.original_threads)

    def test_two_real_maps_at_independently_proposed_translations(self):
        for reference, query, region, anchor in self.pairs:
            verifier = TrackingVerifier()
            for pixels in (1, 5, 12):
                with self.subTest(map=reference.shape, pixels=pixels):
                    moved = cv2.warpAffine(query, np.float32([[1, 0, -pixels], [0, 1, 0]]),
                                           (query.shape[1], query.shape[0]))
                    flow = anchor.copy()
                    flow[:, 2] += anchor[:, 0] * pixels
                    tracked = candidate(flow, moved.shape)
                    result = verifier.verify(reference, moved, minimap_mask(moved.shape),
                                             region, tracked, anchor)
                    self.assertIsNotNone(result, verifier.last_diagnostic)
                    np.testing.assert_allclose(result.xy, tracked.xy, atol=2)
                    np.testing.assert_array_equal(result.transform[:, :2], anchor[:, :2].astype(np.float32))

    def test_ambiguous_user_terrain_does_not_extend_tracking(self):
        reference, query, region, anchor = self.pairs[0]
        moved = cv2.warpAffine(query, np.float32([[1, 0, -17], [0, 1, 0]]),
                               (query.shape[1], query.shape[0]))
        flow = anchor.copy()
        flow[:, 2] += anchor[:, 0] * 17
        verifier = TrackingVerifier()
        self.assertIsNone(verifier.verify(reference, moved, minimap_mask(moved.shape), region,
                                         candidate(flow, moved.shape), anchor), verifier.last_diagnostic)
        self.assertEqual('terrain_peak_ambiguous', verifier.last_diagnostic['verification']['reason'])

    def test_actual_lk_track_on_real_images_can_be_independently_reproved(self):
        for reference, query, region, anchor in self.pairs:
            tracker = MinimapTracker()
            initial = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
            mask = minimap_mask(query.shape) & terrain_feature_mask(query)
            tracker.anchor(initial, mask, anchor, candidate(anchor, query.shape).xy, 0)
            verifier = TrackingVerifier()
            for step in range(1, 7):
                pixels = step * 2
                moved = cv2.warpAffine(query, np.float32([[1, 0, -pixels], [0, 1, 0]]),
                                       (query.shape[1], query.shape[0]))
                gray = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
                mask = minimap_mask(moved.shape) & terrain_feature_mask(moved)
                tracked = tracker.track(gray, mask, step * 0.2)
                self.assertIsNotNone(tracked, tracker.last_diagnostic)
                result = verifier.verify(reference, moved, mask, region, tracked, anchor)
                self.assertIsNotNone(result, verifier.last_diagnostic)
                np.testing.assert_allclose(result.xy, tracked.xy, atol=2)
                tracker.commit(gray, mask, tracked, step * 0.2)
            self.assertEqual(0, tracker.anchor_at)

    def test_wrong_maps_and_rotated_queries_do_not_extend_tracking(self):
        for index, (reference, query, region, anchor) in enumerate(self.pairs):
            cases = [self.pairs[1 - index][1]]
            height, width = query.shape[:2]
            for rotation in (8, 20):
                cases.append(cv2.warpAffine(query, cv2.getRotationMatrix2D(
                    (width / 2, height / 2), rotation, 1), (width, height)))
            for changed in cases:
                with self.subTest(map=reference.shape, query=changed.shape):
                    verifier = TrackingVerifier()
                    self.assertIsNone(verifier.verify(reference, changed, minimap_mask(changed.shape),
                                                     region, candidate(anchor, changed.shape), anchor),
                                      verifier.last_diagnostic)

    def test_central_player_icon_changes_cannot_supply_terrain_evidence(self):
        for reference, query, region, anchor in self.pairs:
            changed = query.copy()
            center = (query.shape[1] // 2, query.shape[0] // 2)
            cv2.circle(changed, center, 12, (255, 255, 255), -1)
            verifier = TrackingVerifier()
            result = verifier.verify(reference, changed, minimap_mask(changed.shape), region,
                                     candidate(anchor, changed.shape), anchor)
            self.assertIsNotNone(result, verifier.last_diagnostic)
            np.testing.assert_allclose(result.xy, candidate(anchor, changed.shape).xy, atol=2)

    def test_repeated_terrain_and_single_textured_patch_cannot_extend_tracking(self):
        rng = np.random.default_rng(41)
        tile = cv2.GaussianBlur(rng.integers(40, 190, (72, 72), np.uint8), (5, 5), 1)
        reference = np.tile(tile, (8, 8))
        query = reference[180:360, 180:360].copy()
        anchor = np.float64([[1, 0, 180], [0, 1, 180]])
        verifier = TrackingVerifier()
        self.assertIsNone(verifier.verify(reference, query, minimap_mask(query.shape),
                                         (0, 0, 576, 576), candidate(anchor, query.shape), anchor),
                          verifier.last_diagnostic)

        reference = np.full((600, 700), 80, np.uint8)
        query = np.full((200, 200), 80, np.uint8)
        patch = cv2.GaussianBlur(rng.integers(40, 190, (55, 55), np.uint8), (5, 5), 1)
        query[30:85, 30:85] = patch
        reference[230:285, 330:385] = patch
        anchor = np.float64([[1, 0, 300], [0, 1, 200]])
        self.assertIsNone(verifier.verify(reference, query, minimap_mask(query.shape),
                                         (0, 0, 700, 600), candidate(anchor, query.shape), anchor),
                          verifier.last_diagnostic)

    def test_low_quality_or_inconsistent_flow_and_changed_scale_are_rejected(self):
        reference, query, region, anchor = self.pairs[0]
        verifier = TrackingVerifier()
        for tracked in (None, candidate(anchor, query.shape, confidence=0.74),
                        candidate(anchor, query.shape, confidence=float('nan')),
                        candidate(anchor, query.shape, inliers=11)):
            self.assertIsNone(verifier.verify(reference, query, minimap_mask(query.shape),
                                             region, tracked, anchor))
        tracked = candidate(anchor, query.shape)
        inconsistent = TrackedMinimap(tracked.transform, (tracked.xy[0] + 10, tracked.xy[1]),
                                      tracked.confidence, tracked.inliers)
        self.assertIsNone(verifier.verify(reference, query, minimap_mask(query.shape),
                                         region, inconsistent, anchor))
        for factor in (0, 0.97, 1.02, 4):
            flow = anchor.copy()
            flow[:, :2] *= factor
            self.assertIsNone(verifier.verify(reference, query, minimap_mask(query.shape),
                                             region, candidate(flow, query.shape), anchor),
                              verifier.last_diagnostic)
        for invalid in (None, np.zeros((2, 2)), np.full((2, 3), float('nan'))):
            self.assertIsNone(verifier.verify(reference, query, minimap_mask(query.shape),
                                             region, tracked, invalid))


if __name__ == '__main__':
    unittest.main()

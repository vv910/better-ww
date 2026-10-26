import unittest

import cv2
import numpy as np

from src.pathing.RouteMapContext import RouteMapContext
from src.pathing.RouteBeaconLocation import parse_beacon_location
from src.pathing.RouteMapScene import RouteMapSceneDetector
from src.pathing.RouteMarkers import white_marker_mask


class TestRouteMapScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closed = cv2.imread('tests/images/3c-map-selector-closed.png')
        # Independently recorded scene, including its original preview line.
        # Only the right-side cue region is retained; all account UI is removed.
        cls.recorded = cv2.imread('tests/images/3c-map-dream-scene-cue.png')
        assert cls.closed is not None and cls.recorded is not None

    def setUp(self):
        self.detector = RouteMapSceneDetector()

    def test_real_closed_and_independent_recording_at_three_resolutions_and_jpeg(self):
        for source in (self.closed, self.recorded):
            for scale in (.75, 1, 1.5):
                for quality in (None, 78):
                    with self.subTest(source=source.shape, scale=scale, quality=quality):
                        frame = cv2.resize(source, None, fx=scale, fy=scale)
                        if quality is not None:
                            encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
                            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        result = self.detector.detect(frame)
                        self.assertIsNotNone(result, self.detector.last_diagnostic)
                        self.assertEqual('梦枢天罗', result.name)
                        self.assertEqual(('瑝珑', '梦州', '梦枢天罗'), result.context.selection_path)
                        self.assertGreaterEqual(result.confidence, RouteMapSceneDetector.MIN_SCORE)

    def test_identity_is_a_context_candidate_and_does_not_hide_explicit_context_conflicts(self):
        result = self.detector.detect(self.recorded)
        self.assertIsNotNone(result, self.detector.last_diagnostic)
        configured = RouteMapContext(('瑝珑', '梦州', '落渊南丘'))
        self.assertNotEqual(configured, result.context)
        self.assertEqual(('瑝珑', '梦州', '落渊南丘'), configured.selection_path)

    def test_confirmed_beacon_location_maps_scene_without_using_local_area_as_menu_label(self):
        location = parse_beacon_location('瑝珑-梦枢天罗-相心域')
        result = self.detector.from_beacon_location(location)
        self.assertEqual('梦枢天罗', result.name)
        self.assertEqual(('瑝珑', '梦州', '梦枢天罗'), result.context.selection_path)
        for raw in ('黎那汐塔-梦枢天罗-相心域', '瑝珑-其他地图-相心域'):
            self.assertIsNone(self.detector.from_beacon_location(parse_beacon_location(raw)))
        self.assertIsNone(self.detector.from_beacon_location(None))

    def test_cue_is_right_anchored_and_does_not_depend_on_the_players_region_label(self):
        wider = np.zeros((1440, 3440, 3), np.uint8)
        wider[:, 880:] = self.recorded
        wider[:300, :700] = self.closed[:300, :700]
        result = self.detector.detect(wider)
        self.assertIsNotNone(result, self.detector.last_diagnostic)
        self.assertEqual(('瑝珑', '梦州', '梦枢天罗'), result.context.selection_path)
        without_cue = wider.copy()
        without_cue[1060:1135, 3080:3330] = 0
        self.assertIsNone(self.detector.detect(without_cue))

    def test_world_islands_and_ordinary_map_are_unknown_scenes(self):
        for path in ('tests/images/mini_map.png', 'tests/images/in_combat.png',
                     'tests/images/path.png', 'tests/images/big_map.png',
                     'tests/images/3c-map-island-scene-cue.png',
                     'tests/images/3c-map-ordinary-scene-cue.png'):
            frame = cv2.imread(path)
            self.assertIsNotNone(frame, path)
            with self.subTest(path=path):
                self.assertIsNone(self.detector.detect(frame), self.detector.last_diagnostic)

    def test_redacted_route_reference_without_scene_control_has_no_identity(self):
        redacted = np.zeros_like(self.closed)
        redacted[173:1296, 717:2227] = self.closed[173:1296, 717:2227]
        self.assertIsNone(self.detector.detect(redacted), self.detector.last_diagnostic)

    def test_every_character_is_required_and_cannot_be_replaced_by_another_glyph(self):
        self.assertIsNotNone(self.detector.detect(self.recorded))
        left, top, width, height = self.detector.last_diagnostic['cue_box']
        for index in range(4):
            for replacement in ('empty', 'triangle', 'other_character'):
                frame = self.recorded.copy()
                x0, x1 = left + round(index * width / 4), left + round((index + 1) * width / 4)
                frame[top:top + height, x0:x1] = 0
                if replacement == 'triangle':
                    cv2.fillPoly(frame, [np.int32([[x0 + 3, top + height - 4],
                                                   [x1 - 4, top + height - 4],
                                                   [(x0 + x1) // 2, top + 4]])], (245, 245, 245))
                elif replacement == 'other_character':
                    source_index = (index + 1) % 4
                    a0 = left + round(source_index * width / 4)
                    a1 = left + round((source_index + 1) * width / 4)
                    frame[top:top + height, x0:x1] = cv2.resize(
                        self.recorded[top:top + height, a0:a1], (x1 - x0, height))
                with self.subTest(character=index, replacement=replacement):
                    self.assertIsNone(self.detector.detect(frame), self.detector.last_diagnostic)

    def test_control_background_changes_do_not_supply_or_erase_text_evidence(self):
        frame = self.recorded.copy()
        crop = frame[1060:1135, 2200:2450]
        foreground = white_marker_mask(crop) > 0
        crop[~foreground] = (55, 80, 100)
        self.assertIsNotNone(self.detector.detect(frame), self.detector.last_diagnostic)
        # Retaining the common border/rotate icon while erasing the label must
        # not identify a scene.
        frame[1075:1118, 2220:2368] = (55, 80, 100)
        self.assertIsNone(self.detector.detect(frame), self.detector.last_diagnostic)

    def test_invalid_frame_and_missing_asset_are_rejected(self):
        for frame in (None, np.zeros((200, 400, 3), np.uint8),
                      np.zeros((1080, 1920), np.uint8), np.zeros((1080, 1920, 3), np.float32)):
            self.assertIsNone(self.detector.detect(frame))
        self.detector.template = None
        self.assertIsNone(self.detector.detect(self.closed))
        self.assertEqual('missing_template', self.detector.last_diagnostic['reason'])


if __name__ == '__main__':
    unittest.main()

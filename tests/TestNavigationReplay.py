import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts.replay_3c_navigation import main
from src.pathing.MapLocalizer import MapLocalizer, MapPosition
from src.pathing.MinimapHeading import HeadingEstimate
from src.pathing.NavigationReplay import (NavigationReplay, load_replay, relocalize_replay,
                                         replay_statistics)


class TestNavigationReplay(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(87151)
        self.reference = cv2.GaussianBlur(rng.integers(20, 190, (500, 600), np.uint8), (5, 5), .8)
        self.query = self.reference[150:350, 200:400].copy()
        self.map_path = self.root / '地图.png'
        self.write_image(self.map_path, self.reference)
        self.route_path = self.root / 'route.json'
        self.route_path.write_text(json.dumps({'map_image': self.map_path.name,
                                              'map_region': [0, 0, 600, 500]}), encoding='utf-8')
        self.clock_patch = patch('src.pathing.NavigationReplay.time.monotonic', return_value=100)
        self.clock = self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    @staticmethod
    def write_image(path, image):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.png', image)[1].tofile(path)

    def writer(self, **kwargs):
        return NavigationReplay(self.route_path, capture_size=(1920, 1080), **kwargs)

    def flush(self, writer):
        path = writer.flush('localization-lost', root=self.root / 'bundles')
        self.assertIsNotNone(path, writer.last_error)
        return path

    def test_raw_ring_is_bounded_copies_inputs_and_deduplicates_same_observation(self):
        writer = self.writer(max_frames=2, max_bytes=self.query.nbytes * 2)
        frame = self.query.copy()
        writer.record_locate(frame, None, {'reason': 'lost'}, started_at=100)
        writer.record_heading(frame, HeadingEstimate(30, .9, .2), timestamp=100)
        self.assertEqual(1, len(writer.frames))
        frame[:] = 0
        self.assertFalse(np.array_equal(frame, next(iter(writer.frames.values()))))
        for index in (1, 2, 3):
            writer.record_locate(np.full_like(frame, index), None, {}, started_at=100 + index)
            self.assertLessEqual(writer.raw_bytes, writer.max_bytes)
            self.assertLessEqual(len(writer.frames), 2)
        self.assertEqual(2, writer.frames_evicted)
        self.assertFalse((self.root / 'bundles').exists())

    def test_bundle_rewrites_colliding_map_names_and_captures_route_snapshot(self):
        first = self.root / 'first' / 'map.png'
        second = self.root / 'second' / 'map.png'
        self.write_image(first, self.reference)
        self.write_image(second, self.reference[:300, :400])
        original = {'map_image': 'first/map.png', 'map_region': [20, 30, 550, 470],
                    'teleport': {'views': [{'map_image': 'second/map.png', 'map_region': [10, 20, 350, 270]},
                                           {'map_image': 'first/map.png', 'map_region': [20, 30, 550, 470]}]}}
        self.route_path.write_text(json.dumps(original), encoding='utf-8')
        arrow = np.full((32, 32, 3), 120, np.uint8)
        writer = self.writer(arrow_template=arrow)
        writer.record_reset(MapPosition(300, 250, .9, 25, source='map'), timestamp=100)
        writer.record_locate(self.query, MapPosition(300, 250, .9, 20),
                             {'scale': np.float32(1), 'value': float('nan')}, started_at=100.1,
                             elapsed=.04)
        writer.record_heading(self.query, HeadingEstimate(30, .9, .2), timestamp=100.2)
        writer.record_keys(('w', 'd'), timestamp=100.2)
        writer.record_target({'x': 330, 'y': 260}, timestamp=100.2)
        self.route_path.write_text('{}', encoding='utf-8')  # Snapshot survives later route edits.
        bundle = self.flush(writer)
        root, metadata, events = load_replay(bundle)
        copied = json.loads((root / 'route.json').read_text(encoding='utf-8'))
        primary = copied['map_image']
        overview = copied['teleport']['views'][0]['map_image']
        self.assertNotEqual(primary, overview)
        self.assertEqual(primary, copied['teleport']['views'][1]['map_image'])
        self.assertTrue((root / primary).is_file())
        self.assertTrue((root / overview).is_file())
        redacted = cv2.imdecode(np.fromfile(root / primary, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((500, 600, 3), redacted.shape)
        self.assertFalse(redacted[:30].any())
        self.assertFalse(redacted[:, :20].any())
        self.assertTrue(redacted[30:470, 20:550].any())
        np.testing.assert_array_equal(self.reference,
                                      cv2.imdecode(np.fromfile(first, np.uint8), cv2.IMREAD_GRAYSCALE))
        self.assertEqual([], metadata['validation_errors'])
        self.assertEqual(['reset', 'locate', 'heading', 'keys', 'target'], [event['kind'] for event in events])
        self.assertAlmostEqual(.1, events[1]['t'])
        self.assertEqual('sift', events[1]['position']['source'])
        self.assertIsNone(events[1]['diagnostic']['value'])
        self.assertEqual([1920, 1080], events[1]['capture_size'])
        self.assertTrue((root / metadata['arrow_template']).is_file())
        self.assertEqual(bundle, writer.flush('again', root=self.root / 'ignored'))
        self.assertFalse((self.root / 'ignored').exists())

    def test_failure_to_flush_never_masks_original_error_or_retries_io(self):
        writer = self.writer()
        blocked = self.root / 'not-a-directory'
        blocked.write_text('occupied', encoding='utf-8')
        self.assertIsNone(writer.flush('failure', root=blocked))
        self.assertIn('Replay bundle write failed', writer.last_error)
        self.assertIsNone(writer.flush('again', root=self.root / 'would-succeed'))
        self.assertFalse((self.root / 'would-succeed').exists())

    def test_arrow_template_can_be_supplied_lazily_without_aliasing_or_exceeding_budget(self):
        writer = self.writer()
        arrow = np.full((32, 32, 3), 120, np.uint8)
        self.assertTrue(writer.set_arrow_template(arrow))
        arrow[:] = 0
        self.assertTrue(writer._arrow.any())
        self.assertFalse(writer.set_arrow_template(np.zeros((600, 600, 3), np.uint8)))
        self.assertTrue(writer._arrow.any())
        bundle = self.flush(writer)
        _, metadata, _ = load_replay(bundle)
        self.assertTrue((bundle / metadata['arrow_template']).is_file())

    def test_statistics_distinguish_flow_rejections_jumps_and_reset_boundaries(self):
        position = lambda x, source='sift': {'x': x, 'y': 0, 'source': source}
        events = [
            {'kind': 'locate', 't': 0, 'position': position(0), 'elapsed_ms': 40},
            {'kind': 'locate', 't': 1, 'position': position(5, 'flow')},
            {'kind': 'locate', 't': 2, 'position': None, 'diagnostic': {'reason': 'too_few_matches'}},
            {'kind': 'locate', 't': 3, 'position': None, 'diagnostic': {'reason': 'tracking_expired'}},
            {'kind': 'locate', 't': 4, 'position': position(150), 'elapsed_ms': 80},
            {'kind': 'reset', 't': 5},
            {'kind': 'locate', 't': 6, 'position': position(500)},
            {'kind': 'heading', 't': 6, 'estimate': {'confidence': .8, 'margin': .2}},
        ]
        summary = replay_statistics(events)
        self.assertAlmostEqual(4 / 6, summary['availability'])
        self.assertEqual({'sift': 3, 'flow': 1}, summary['position_sources'])
        self.assertEqual(2, summary['longest_unavailable_seconds'])
        self.assertEqual(145, summary['largest_accepted_step_map_pixels'])
        self.assertEqual(1, summary['accepted_steps_over_100_map_pixels'])
        self.assertEqual(1, summary['reliable_heading_calls'])
        self.assertEqual(60, summary['mean_locate_ms'])

    def test_default_cli_only_summarizes_without_loading_vision(self):
        writer = self.writer()
        writer.record_locate(self.query, MapPosition(300, 250, .9, 25), {}, started_at=100)
        bundle = self.flush(writer)
        with patch('scripts.replay_3c_navigation.relocalize_replay') as recompute, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, main([str(bundle)]))
        recompute.assert_not_called()
        self.assertEqual(1, json.loads(output.getvalue())['statistics']['located_calls'])

    def test_missing_images_are_reported_and_do_not_silently_continue_flow(self):
        writer = self.writer()
        writer.record_locate(self.query, MapPosition(300, 250, .9, 25), {}, started_at=100)
        bundle = self.flush(writer)
        _, metadata, _ = load_replay(bundle)
        (bundle / next(iter(metadata['frames'].values()))).unlink()
        _, validated, _ = load_replay(bundle)
        self.assertTrue(any('missing' in error for error in validated['validation_errors']))
        report = relocalize_replay(bundle)
        self.assertEqual(1, report['missing_frame_observations'])
        self.assertEqual(0, report['statistics']['locate_calls'])

    def test_replay_rejects_traversal_and_nonmonotonic_event_times(self):
        writer = self.writer()
        writer.record_locate(self.query, None, {}, started_at=100)
        bundle = self.flush(writer)
        manifest = json.loads((bundle / 'metadata.json').read_text(encoding='utf-8'))
        saved = json.dumps(manifest)
        manifest['frames'] = {'bad': '../outside.png'}
        (bundle / 'metadata.json').write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'leaves bundle'):
            load_replay(bundle)
        (bundle / 'metadata.json').write_text(saved, encoding='utf-8')
        (bundle / 'events.jsonl').write_text('\n'.join(json.dumps(event) for event in (
            {'kind': 'reset', 't': 2}, {'kind': 'reset', 't': 1})), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            load_replay(bundle)

    def test_recorded_frame_gaps_expire_flow_after_three_seconds(self):
        writer = self.writer()
        writer.record_reset(timestamp=100)
        for relative in (0, .6, 1.2, 1.8, 2.4, 3.1):
            writer.record_locate(self.query, None, {}, started_at=100 + relative)
        bundle = self.flush(writer)
        original = MapLocalizer._locate_absolute
        calls = [0]

        def first_anchor_only(localizer, *args):
            calls[0] += 1
            return (original(localizer, *args) if calls[0] == 1
                    else localizer._reject('too_few_matches'))

        with patch.object(MapLocalizer, '_locate_absolute', first_anchor_only), \
                patch('src.pathing.TrackingVerifier.TrackingVerifier.verify', return_value=None):
            report = relocalize_replay(bundle)
        summary = report['statistics']
        self.assertEqual({'sift': 1, 'flow': 4}, summary['position_sources'])
        self.assertEqual(6, summary['locate_calls'])
        self.assertEqual(5, summary['located_calls'])

    def test_replay_does_not_refresh_an_old_registered_map_hint(self):
        writer = self.writer()
        writer.record_reset(MapPosition(300, 250, .9, 25, source='map'), timestamp=100)
        writer.record_locate(self.query, None, {}, started_at=105.1)
        bundle = self.flush(writer)
        with patch.object(MapLocalizer, '_locate_absolute',
                          lambda localizer, *args: localizer._reject('too_few_matches')), \
                patch('src.pathing.TerrainSearch.TerrainSearch.search') as search:
            report = relocalize_replay(bundle)
        search.assert_not_called()
        self.assertEqual(0, report['statistics']['located_calls'])

    def test_absolute_request_is_replayed_and_slow_observation_is_not_refreshed(self):
        writer = self.writer()
        writer.record_reset(timestamp=100)
        writer.record_absolute_request(timestamp=100)
        writer.record_locate(self.query, None, {}, started_at=100, elapsed=1.1)
        bundle = self.flush(writer)
        original = MapLocalizer.request_absolute
        requested = Mock()

        def request(localizer):
            requested()
            original(localizer)

        with patch.object(MapLocalizer, 'request_absolute', request):
            report = relocalize_replay(bundle)
        requested.assert_called_once()
        self.assertEqual(0, report['statistics']['located_calls'])
        self.assertEqual({'observation_expired': 1}, report['statistics']['rejection_reasons'])

    def test_missing_route_assets_make_incomplete_bundle_explicit(self):
        writer = self.writer()
        self.map_path.unlink()
        bundle = self.flush(writer)
        _, metadata, _ = load_replay(bundle)
        self.assertTrue(metadata['errors'])
        self.assertTrue(metadata['validation_errors'])
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(2, main([str(bundle), '--relocalize']))
        self.assertIn('unavailable', error.getvalue())


if __name__ == '__main__':
    unittest.main()

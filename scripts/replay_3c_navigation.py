"""Summarize a navigation bundle; opt in to offline vision with --relocalize."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pathing.NavigationReplay import load_replay, relocalize_replay, replay_statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path, help='Extracted logs/3c-replays/<run-id> directory')
    parser.add_argument('--relocalize', action='store_true', help='Recompute vision only; never sends game input')
    arguments = parser.parse_args(argv)
    try:
        if arguments.relocalize:
            report = relocalize_replay(arguments.bundle)
        else:
            _, metadata, events = load_replay(arguments.bundle)
            report = {'statistics': replay_statistics(events), 'truncated': metadata.get('truncated'),
                      'frames_retained': len(metadata['frames']), 'frames_evicted': metadata.get('frames_evicted'),
                      'events_total': metadata.get('events_total'),
                      'errors': metadata.get('errors', []) + metadata.get('validation_errors', []),
                      'timing_note': metadata.get('timing_note')}
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f'Navigation replay unavailable: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

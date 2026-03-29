from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run_step(step_main, argv: list[str]) -> int:
    previous_argv = sys.argv[:]
    try:
        sys.argv = argv
        return int(step_main())
    finally:
        sys.argv = previous_argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the travel-sceneflow pipeline.")
    parser.add_argument('--root', default='.', help='Root directory to scan for media')
    parser.add_argument('--run-dir', default=None, help='Shared run directory under outputs/')
    parser.add_argument('--with-meanings', action='store_true', help='Also build scene_meanings.json')
    parser.add_argument('--with-structure', action='store_true', help='Also build edit_structure.json')
    args = parser.parse_args()

    from sceneflow.pipeline import candidates, meanings, representatives, scan, sceneify, structure, tagging

    root = Path(args.root).resolve()
    if args.run_dir is None:
        run_dir = Path('outputs') / 'manual-run'
    else:
        run_dir = Path(args.run_dir)

    run_dir_str = str(run_dir)
    scan_out = run_dir / 'scan' / 'media_info.csv'
    sceneify_out = run_dir / 'sceneify' / 'media_scene.csv'
    reps_out = run_dir / 'representatives' / 'scene_representatives.csv'
    tagging_out = run_dir / 'tagging' / 'scene_representatives_tagged.csv'
    candidates_out = run_dir / 'candidates' / 'scene_edit_candidates.json'
    meanings_out = run_dir / 'meaning' / 'scene_meanings.json'

    steps = [
        ('scan', scan.main, ['scan', '--root', str(root), '--run-dir', run_dir_str]),
        ('sceneify', sceneify.main, ['sceneify', '--input', str(scan_out), '--run-dir', run_dir_str]),
        ('representatives', representatives.main, ['representatives', '--input', str(sceneify_out), '--run-dir', run_dir_str]),
        ('tagging', tagging.main, ['tagging', '--input', str(reps_out), '--root', str(root), '--run-dir', run_dir_str]),
        ('candidates', candidates.main, ['candidates', '--media-scene', str(sceneify_out), '--representatives', str(tagging_out), '--root', str(root), '--run-dir', run_dir_str]),
    ]

    if args.with_meanings or args.with_structure:
        steps.append(('meanings', meanings.main, ['meanings', '--input', str(candidates_out), '--run-dir', run_dir_str]))
    if args.with_structure:
        steps.append(('structure', structure.main, ['structure', '--input', str(meanings_out), '--run-dir', run_dir_str]))

    for _, step_main, argv in steps:
        code = _run_step(step_main, argv)
        if code != 0:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

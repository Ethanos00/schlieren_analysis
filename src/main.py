"""Single entry point for the schlieren project.

Place as project/src/main.py.
From project/src: python main.py analysis | calibration | all
From project root: python src/main.py analysis
"""
from pathlib import Path
import argparse
import subprocess
import sys


def run(command, folder):
    print('\nRunning:', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, cwd=folder, check=True)


def main():
    parser = argparse.ArgumentParser(description='Schlieren project runner')
    parser.add_argument('stage', nargs='?', choices=('analysis', 'calibration', 'all'), default='analysis')
    parser.add_argument('--config', type=Path, default=None,
                        help='Optional analysis config path; defaults to src/config.yaml')
    args = parser.parse_args()
    src = Path(__file__).resolve().parent
    cfg = args.config.resolve() if args.config else src / 'config.yaml'
    if not cfg.is_file():
        parser.error(f'Config not found: {cfg}')

    if args.stage in ('calibration', 'all'):
        # Existing calibrate.py reads src/config.yaml from utils.py.
        # It also interprets image paths relative to the working directory.
        if cfg != src / 'config.yaml':
            parser.error('Existing calibrate.py uses src/config.yaml; --config is analysis-only.')
        print('=== Calibration ===', flush=True)
        run([sys.executable, str(src / 'calibrate.py')], src)

    if args.stage in ('analysis', 'all'):
        print('=== Analysis ===', flush=True)
        run([sys.executable, str(src / 'analysis.py'), '--config', str(cfg)], src)

    print('\nAll requested stages finished.')


if __name__ == '__main__':
    main()

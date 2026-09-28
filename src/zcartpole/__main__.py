"""Command line entry: python -m zcartpole config.json --output run_dir."""
import argparse

from .runner import run_system


def main():
    parser = argparse.ArgumentParser(description='Run a configured cart-pole system')
    parser.add_argument('config', help='System description as JSON')
    parser.add_argument('--output', '-o', required=True,
                        help='New output directory for this run')
    args = parser.parse_args()
    try:
        result = run_system(args.config, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'{result.status}: {result.output_dir / "summary.json"}')
    return 0 if result.status == 'caught' else 1


if __name__ == '__main__':
    raise SystemExit(main())

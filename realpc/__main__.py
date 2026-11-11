"""Two public presets: python -m realpc {accurate,fast} {convert,live}."""
import argparse
import importlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ('accurate', 'fast')


def preset_config(name):
    if name not in PRESETS:
        raise ValueError(f'Unknown pipeline: {name}')
    return ROOT / 'configs' / f'pipeline_{name}.json'


def run_live(args, config):
    import numpy as np
    from .live import SceneSetup, build_live_pipeline
    module, separator, function = args.source.partition(':')
    if not separator:
        raise ValueError('--source must be module:function (a zero-argument packet iterator factory)')
    factory = getattr(importlib.import_module(module), function)
    setup = SceneSetup(args.setup, args.task)
    pipeline = build_live_pipeline(setup, config, cloth=args.task == 'greycloth')
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    for packet in factory():
        tick = packet['tick']
        row = pipeline.step(tick, packet['frames'], packet['q'])
        points = pipeline.points(row)
        values = dict(points=points, valid=np.isfinite(points).all(-1), tick=tick,
                      sam_called=row['sam_called'], max_observed_tick=row['max_observed_tick'])
        for key in ('cloth_valid', 'cloth_approximate', 'cloth_observed'):
            if key in row:
                values[key] = row[key]
        temporary = destination / f'{tick:06d}.tmp.npz'
        np.savez_compressed(temporary, **values)
        temporary.replace(destination / f'{tick:06d}.npz')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('pipeline', choices=PRESETS)
    parser.add_argument('command', choices=('convert', 'live'))
    if argv in (['--help'], ['-h']):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args(argv)
    config = preset_config(args.pipeline)
    if args.command == 'convert':
        if any(x == '--config' or x.startswith('--config=') for x in remaining):
            parser.error('The selected pipeline owns --config; edit its config file to tune it.')
        script = ROOT / 'scripts/convert_scene.py'
        old_argv = sys.argv
        try:
            sys.argv = [str(script), '--config', str(config), *remaining]
            runpy.run_path(str(script), run_name='__main__', init_globals={'PUBLIC_PRESET': args.pipeline})
        finally:
            sys.argv = old_argv
    else:
        live = argparse.ArgumentParser(prog=f'python -m realpc {args.pipeline} live')
        live.add_argument('--setup', required=True, help='Static camera/robot/workspace setup directory')
        live.add_argument('--task', required=True, choices=('sweep', 'hang', 'greycloth'))
        live.add_argument('--source', required=True, help='module:function returning current packet dictionaries')
        live.add_argument('--out', required=True, help='Streaming per-frame point NPZ output directory')
        run_live(live.parse_args(remaining), config)


if __name__ == '__main__':
    main()

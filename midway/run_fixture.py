"""Ground-truth fixture run (SPEC §6.3, `gt` setting only).

Runs upstream's generator with --generation_method ground_truth: every "generated" build is the prompt's
ground-truth build. CPU only, no model. Output has exactly upstream's format, so it exercises the whole
eval path. BrickGPT generation settings live in Megha's runner, not here.

Writes runs/gt_{prompts}_s{seed}/generated_dataset.json (+ generations/, run_meta.json).

Run:  make fixture
"""
import argparse
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from brickgpt_experiments.generate import BrickGenerator, BrickGeneratorConfig, _ground_truth_fn

from midway.config import BRICKGPT_SHA, UPSTREAM_DIR
from midway.make_subset import prompts_path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompts', default='full250', help='full250 or subsetN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--out', type=Path, default=None, help='default: runs/gt_{prompts}_s{seed}')
    args = parser.parse_args()
    out = args.out or Path('runs') / f'gt_{args.prompts}_s{args.seed}'

    meta = {
        'setting': 'gt', 'flags': '--generation_method ground_truth', 'seed': args.seed, 'prompts': args.prompts,
        'upstream_sha': subprocess.run(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True).stdout.strip(),
        'brickgpt_sha': BRICKGPT_SHA, 'hostname': socket.gethostname(), 'gpu': None,
        'n_workers': args.n_workers, 'start_time': now(),
    }
    cfg = BrickGeneratorConfig(output_dir=str(out), eval_dataset_path=str(prompts_path(args.prompts)),
                               generation_method='ground_truth', n_workers=args.n_workers, seed=args.seed)
    try:
        BrickGenerator(_ground_truth_fn, cfg)()
        meta['exit_status'] = 0
    except Exception:
        meta['exit_status'] = 1
        raise
    finally:
        meta['end_time'] = now()
        out.mkdir(parents=True, exist_ok=True)
        (out / 'run_meta.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

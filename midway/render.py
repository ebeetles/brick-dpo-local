"""Render every build of a run with upstream's renderer, using a configurable number of Blender processes (P2).

Upstream `evaluate --render_generated_bricks` always runs 8 Blender processes in parallel. On a 36 GB Mac that
filled swap and stalled (1 of 250 images in 30 min), so this calls the same upstream function,
`brickgpt_experiments.evaluate._render_bricks`, with `--workers` processes (default 1). Output is identical:
runs/<name>/renders/<object_id>.png (+ .ldr). Already-rendered images are skipped, so reruns resume.

Run:  make render RUN=runs/<name>
"""
import argparse
import json
import multiprocessing as mp
import subprocess
import time
from functools import partial
from pathlib import Path

from midway import upstream_eval  # noqa: F401  (installs the chamferdist stub and uv env vars first)
from brickgpt_experiments.evaluate import _render_bricks  # noqa: E402

from midway.shapes import parse_build  # noqa: E402


def render_one(row: dict, render_dir: str) -> tuple[str, str]:
    """Render one row's build; returns (object_id, 'ok' | error message). Unparseable/empty builds are skipped."""
    build = parse_build(row['output'])
    if build is None or len(build) == 0:
        return row['object_id'], 'skipped (unparseable or empty)'
    try:
        _render_bricks(build, render_dir, row['object_id'])
        return row['object_id'], 'ok'
    except subprocess.CalledProcessError as err:
        return row['object_id'], f'render failed: {err.stderr.decode(errors="replace")[-300:]}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    rows = json.loads((args.run / 'generated_dataset.json').read_text())
    render_dir = str(args.run / 'renders')
    start = time.time()
    with mp.Pool(args.workers) as pool:
        for i, (object_id, status) in enumerate(pool.imap_unordered(partial(render_one, render_dir=render_dir), rows), 1):
            print(f'[{i}/{len(rows)} {time.time() - start:.0f}s] {object_id}: {status}', flush=True)


if __name__ == '__main__':
    main()

"""Eyeball check without Blender: 3D voxel plots of N random builds of a run, each beside its ground truth.

Rows are picked with random.Random(seed) (default 0), so the same builds come up every time.
Left: the generated build (blue; voxels covered by two or more bricks, i.e. collisions, in red).
Right: the prompt's ground-truth build (gray). Each row is titled with the caption and the eval verdict.
Voxels are drawn 1.2x taller than wide, like real bricks (9.6 mm tall, 8 mm stud pitch).

Output: runs/<name>/preview_builds.png
Run:    make preview-builds RUN=runs/<name> N=5
"""
import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from midway.config import GROUND_TRUTH_COLOR, MODELS, WORLD_DIM  # noqa: E402
from midway.per_sample_eval import load_generated  # noqa: E402
from midway.shapes import parse_build  # noqa: E402

BUILD_COLOR = MODELS['BrickGPT']['color']
COLLISION_COLOR = '#e34948'


def draw_build(ax, build, color: str, title: str) -> None:
    """Voxel plot of one build on the 20^3 grid; colliding voxels in red. Handles unparseable builds."""
    ax.set_title(title, fontsize=8)
    if build is None or len(build) == 0:
        ax.text2D(0.5, 0.5, 'unparseable' if build is None else 'empty', transform=ax.transAxes, ha='center')
        ax.set_axis_off()
        return
    occupancy = build.voxel_occupancy
    filled = occupancy > 0
    colors = np.empty(filled.shape, dtype=object)
    colors[filled] = color
    colors[occupancy > 1] = COLLISION_COLOR
    ax.voxels(filled, facecolors=colors, edgecolor='#333333', linewidth=0.1)
    ax.set_xlim(0, WORLD_DIM)
    ax.set_ylim(0, WORLD_DIM)
    ax.set_zlim(0, WORLD_DIM)
    ax.set_box_aspect((1, 1, 1.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=25, azim=-60)


def verdict(record: dict | None) -> str:
    """Short eval verdict from per_sample.jsonl, e.g. 'valid, stable' or 'INVALID (collision)'."""
    if record is None:
        return 'not evaluated'
    if not record['valid']:
        return f'INVALID ({record["invalid_reason"]})'
    return f'valid, {"stable" if record["stable"] else "UNSTABLE"}, min stab {record["min_stab_raw"]:.2f}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rows, _ = load_generated(args.run)
    per_sample = args.run / 'per_sample.jsonl'
    records = ({r['object_id']: r for r in map(json.loads, per_sample.read_text().splitlines())}
               if per_sample.exists() else {})
    picked = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    fig = plt.figure(figsize=(7, 3.6 * len(picked)), layout='constrained')
    fig.suptitle(f'{args.run.name}: {len(picked)} random builds (seed {args.seed}); red = colliding voxels', fontsize=10)
    for subfig, row in zip(np.atleast_1d(fig.subfigures(len(picked), 1)), picked):
        build = parse_build(row.get('output'))
        gt = parse_build(row['messages'][-1]['content']) if row.get('messages') else None
        caption = '\n'.join(textwrap.wrap(row.get('caption', ''), 80))
        subfig.suptitle(f'{row["object_id"][:8]}: "{caption}"', fontsize=9)
        left, right = subfig.subplots(1, 2, subplot_kw={'projection': '3d'})
        n_bricks = len(build) if build is not None else 'n/a'
        draw_build(left, build, BUILD_COLOR, f'generated: {n_bricks} bricks, {verdict(records.get(row["object_id"]))}')
        draw_build(right, gt, GROUND_TRUTH_COLOR, f'ground truth: {len(gt) if gt is not None else "n/a"} bricks')
    out = args.run / 'preview_builds.png'
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'Wrote {out}: {", ".join(r["object_id"][:8] for r in picked)}')


if __name__ == '__main__':
    main()

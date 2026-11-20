"""Per-sample evaluation of one run directory (SPEC §6.4).

Input:  runs/<name>/generated_dataset.json in upstream's format (each row = prompt fields + `output`,
        `generation_time`, optionally `n_regenerations`, `rejection_reasons`), plus run_meta.json if present.
        If generated_dataset.json is missing (run still in progress), the per-object files in
        runs/<name>/generations/ are used instead, as upstream `evaluate` does for a directory.
Output: runs/<name>/per_sample.jsonl (one record per prompt) and runs/<name>/cross_check.json.

Missing or undefined values are None (null in JSON), never 0. A row whose output cannot be parsed
(see shapes.parse_build_checked) never crashes the evaluation: it gets valid=False and a `parse_error` reason.

Cross-check: upstream `evaluate` is run on the parseable rows only (it would crash on the others), in
runs/<name>/cross_check/. cross_check.json records how many rows were excluded from it and why.

Run:  make eval RUN=runs/<name>
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import re
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np

from midway.config import REPO_ROOT, SETTINGS
from midway.shapes import footprint, frac_1x1, height, parse_build, parse_build_checked, voxel_iou


# ---------------------------------------------------------------- stability (upstream physics)

def stability_fields(build) -> dict:
    """Validity and stability of one parsed build, using upstream BrickStructure only.

    valid         build parsed, is non-empty, has no colliding bricks and no out-of-bounds bricks
                  (same rule as upstream Evaluator._analyze_stability).
    max_stress    max over voxels of upstream `stability_scores()` (a per-voxel stress map in [0, 1];
                  exactly 1 where the solver finds unbalanced force/torque). None if not valid.
    mean_stress   mean of `stability_scores()` over occupied voxels. None if not valid.
    stable        valid and max_stress < 1.0 (upstream's rule).
    min_stab_raw  1 - max_stress;  mean_stab_raw = 1 - mean_stress  (paper's stability s = 1 - stress).
    *_clip        the same, clipped to [0, 1]. Kept for the convention check (SPEC §6.4).
    """
    if build is None or len(build) == 0 or build.has_collisions() or build.has_out_of_bounds_bricks():
        return {'valid': False, 'stable': False, 'max_stress': None, 'mean_stress': None,
                'min_stab_raw': None, 'mean_stab_raw': None, 'min_stab_clip': None, 'mean_stab_clip': None}

    with contextlib.redirect_stdout(io.StringIO()):  # upstream/Gurobi print license banners
        scores = build.stability_scores()
    max_stress = float(scores.max())
    mean_stress = float(scores[build.voxel_occupancy > 0].mean())
    return {
        'valid': True,
        'stable': max_stress < 1.0,
        'max_stress': max_stress,
        'mean_stress': mean_stress,
        'min_stab_raw': 1 - max_stress,
        'mean_stab_raw': 1 - mean_stress,
        'min_stab_clip': float(np.clip(1 - max_stress, 0, 1)),
        'mean_stab_clip': float(np.clip(1 - mean_stress, 0, 1)),
    }


# ---------------------------------------------------------------- one record per prompt

def evaluate_row(row: dict, setting: str | None, seed: int | None, max_regenerations: int | None) -> dict:
    """Build the per-sample record for one row of generated_dataset.json.

    n_bricks         number of bricks parsed (None if the output could not be parsed).
    parse_error      None, or why the output could not be parsed (then valid=False and all physics fields None).
    height           top layer index + 1.   footprint: occupied (x, y) cells, top-down.
    frac_1x1         share of bricks that are 1x1.
    voxel_iou_gt     IoU of 20^3 occupancy vs the prompt's ground-truth build (messages[-1]['content']).
    generation_time  seconds, copied from the run.
    n_regenerations  rollbacks used, copied (None if the run did not record it).
    n_rejections     total count over `rejection_reasons` (None if not recorded).
    hit_rb_limit     n_regenerations == max_regenerations and not stable. None when rollback is off
                     (max_regenerations 0 or unknown), since then every unstable build would count.
    """
    build, parse_error = parse_build_checked(row.get('output'))
    gt_build = parse_build(row['messages'][-1]['content']) if row.get('messages') else None
    record = {
        'object_id': row['object_id'],
        'category_id': row.get('category_id'),
        'caption': row.get('caption'),
        'seed': seed,
        'setting': setting,
        'n_bricks': len(build) if build is not None else None,
        'parse_error': parse_error,
        **stability_fields(build),
        'height': height(build) if build is not None else None,
        'footprint': footprint(build) if build is not None else None,
        'frac_1x1': frac_1x1(build) if build is not None else None,
        'voxel_iou_gt': voxel_iou(build, gt_build) if build is not None and gt_build is not None else None,
        'generation_time': row.get('generation_time'),
        'n_regenerations': row.get('n_regenerations'),
        'n_rejections': sum(row['rejection_reasons'].values()) if row.get('rejection_reasons') is not None else None,
    }
    if record['n_regenerations'] is None or not max_regenerations:
        record['hit_rb_limit'] = None
    else:
        record['hit_rb_limit'] = record['n_regenerations'] == max_regenerations and not record['stable']
    return record


def attach_clip_dino(records: list[dict], run_dir: Path) -> None:
    """Add per-sample `clip` and `dino` from upstream's outputs, if `make render` has produced them.

    clip  CLIP text-image score of the build's render vs "A LEGO model of: <caption>"
          (upstream clip_analysis.json, keyed by '<object_id>.png').
    dino  DINOv2 image similarity of the build's render vs the ground-truth render, (cos + 1) / 2
          (upstream img_similarity_analysis.json['dino'], same keys).
    Set only for valid builds (paper convention; upstream also scores out-of-bounds builds). Otherwise None.
    """
    clip_path, sim_path = run_dir / 'clip_analysis.json', run_dir / 'img_similarity_analysis.json'
    clip = json.loads(clip_path.read_text()) if clip_path.exists() else {}
    dino = json.loads(sim_path.read_text()).get('dino', {}) if sim_path.exists() else {}
    for r in records:
        key = f'{r["object_id"]}.png'
        r['clip'] = clip.get(key) if r['valid'] else None
        r['dino'] = dino.get(key) if r['valid'] else None


# ---------------------------------------------------------------- run-level inputs

def load_generated(run_dir: Path) -> tuple[list[dict], Path]:
    """Rows of the run, from generated_dataset.json, or from generations/*.json if the run is unfinished."""
    dataset_path = run_dir / 'generated_dataset.json'
    if dataset_path.exists():
        return json.loads(dataset_path.read_text()), dataset_path
    generations = run_dir / 'generations'
    if generations.is_dir():
        files = sorted(generations.glob('**/*.json'))
        print(f'WARNING: {dataset_path} missing; using {len(files)} files in {generations} (partial run).')
        return [json.loads(f.read_text()) for f in files], generations
    raise FileNotFoundError(f'No generated_dataset.json or generations/ in {run_dir}')


def run_info(run_dir: Path) -> dict:
    """Setting, prompt set, seed, max_regenerations for a run.

    Read from run_meta.json when present; otherwise parsed from the run name `{setting}_{prompts}_s{seed}`.
    max_regenerations: run_meta field, else `--max_regenerations N` in its flags, else the setting's default.
    """
    meta_path = run_dir / 'run_meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    match = re.fullmatch(r'(?P<setting>[a-z]+)_(?P<prompts>full250|subset\d+)_s(?P<seed>\d+)', run_dir.name)
    from_name = match.groupdict() if match else {}

    setting = meta.get('setting', from_name.get('setting'))
    seed = meta.get('seed', from_name.get('seed'))
    max_regen = meta.get('max_regenerations')
    if max_regen is None:
        flags = ' '.join(meta['flags']) if isinstance(meta.get('flags'), list) else str(meta.get('flags', ''))
        flag_match = re.search(r'--max_regenerations[ =](\d+)', flags)
        if flag_match:
            max_regen = int(flag_match.group(1))
        elif setting in SETTINGS:
            max_regen = SETTINGS[setting]['max_regenerations']
    return {
        'setting': setting,
        'prompts': meta.get('prompts', meta.get('prompt_set', from_name.get('prompts'))),
        'seed': int(seed) if seed is not None else None,
        'max_regenerations': max_regen,
    }


# ---------------------------------------------------------------- cross-check against upstream evaluate

def run_upstream_evaluate(check_dir: Path, rows: list[dict]) -> tuple[dict, dict]:
    """Run upstream `evaluate --analyze_stability --analyze_time` on `rows`; return its two JSON outputs.

    The rows are written to check_dir/generated_dataset.json, so upstream writes its outputs into check_dir
    and never touches the run's own files. Raises CalledProcessError if upstream crashes.
    """
    check_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = check_dir / 'generated_dataset.json'
    dataset_path.write_text(json.dumps(rows))
    cmd = [sys.executable, '-m', 'midway.upstream_eval', '--generated_dataset_path', str(dataset_path),
           '--analyze_stability', '--analyze_time']
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return (json.loads((check_dir / 'stability_analysis.json').read_text()),
            json.loads((check_dir / 'time_analysis.json').read_text()))


def compare_with_upstream(records: list[dict], upstream_stability: dict, upstream_time: dict) -> dict:
    """Compare our counts to upstream's on the same rows. Counts and mean brick count must match exactly.

    Upstream's `stable`/`valid` object_id lists are NOT compared: upstream zips results from
    `imap_unordered` with the dataset in input order, so those lists can be misaligned. Counts are unaffected.
    Upstream's mean_max_stability / mean_mean_stability (mean stress over valid builds) are compared to 1e-9,
    because the two sides add the same numbers in a different order.
    """
    valid = [r for r in records if r['valid']]
    ours = {
        'n_total': len(records),
        'n_valid': len(valid),
        'n_stable': sum(r['stable'] for r in records),
        'mean_bricks': float(np.mean([r['n_bricks'] for r in records])),
        'mean_max_stability': float(np.mean([r['max_stress'] for r in valid])) if valid else 0,
        'mean_mean_stability': float(np.mean([r['mean_stress'] for r in valid])) if valid else 0,
    }
    theirs = {key: upstream_stability[key] for key in ('n_total', 'n_valid', 'n_stable',
                                                        'mean_max_stability', 'mean_mean_stability')}
    theirs['mean_bricks'] = upstream_time['mean_bricks']
    exact = all(ours[k] == theirs[k] for k in ('n_total', 'n_valid', 'n_stable', 'mean_bricks'))
    close = all(abs(ours[k] - theirs[k]) < 1e-9 for k in ('mean_max_stability', 'mean_mean_stability'))
    return {'ours': ours, 'upstream': theirs, 'match': exact and close}


def cross_check(run_dir: Path, rows: list[dict], records: list[dict], source: str) -> dict:
    """Cross-check our records against upstream `evaluate`, on the parseable rows only.

    Returns the cross_check.json content:
      match        True only if upstream ran and agreed on every compared number.
      n_rows       rows in the run;  n_checked: rows given to upstream;  n_excluded / excluded: unparseable rows
                   (object_id + parse_error), which upstream would crash on. They still count as invalid in our metrics.
      source       which file the rows came from (generated_dataset.json, or generations/ for a partial run).
      error        None, or why upstream could not be run.
    """
    excluded = [{'object_id': r['object_id'], 'parse_error': r['parse_error']} for r in records if r['parse_error']]
    kept = [(row, r) for row, r in zip(rows, records) if not r['parse_error']]
    result = {'match': False, 'n_rows': len(rows), 'n_checked': len(kept), 'n_excluded': len(excluded),
              'excluded': excluded, 'source': source, 'error': None, 'ours': None, 'upstream': None}
    if not kept:
        result['error'] = 'no parseable rows to cross-check'
        return result
    try:
        upstream = run_upstream_evaluate(run_dir / 'cross_check', [row for row, _ in kept])
    except subprocess.CalledProcessError as err:
        result['error'] = f'upstream evaluate crashed: {(err.stderr or "")[-500:]}'
        return result
    return result | compare_with_upstream([r for _, r in kept], *upstream)


# ---------------------------------------------------------------- printing

def print_summary(records: list[dict]) -> None:
    """% valid, % stable, and the min_stab_raw distribution (stability-convention check, SPEC §6.4)."""
    n = len(records)
    n_valid = sum(r['valid'] for r in records)
    n_stable = sum(r['stable'] for r in records)
    print(f'n={n}  valid={n_valid} ({100 * n_valid / n:.1f}%)  stable={n_stable} ({100 * n_stable / n:.1f}%)')
    raw = np.array([r['min_stab_raw'] for r in records if r['min_stab_raw'] is not None])
    if len(raw):
        q = np.percentile(raw, [0, 1, 5, 25, 50, 75, 95, 100])
        print('min_stab_raw percentiles [0,1,5,25,50,75,95,100]:', ' '.join(f'{v:.4f}' for v in q))
        print(f'min_stab_raw < 0: {(raw < 0).sum()}   == 0: {(raw == 0).sum()}   '
              f'raw != clipped: {sum(r["min_stab_raw"] != r["min_stab_clip"] for r in records if r["valid"])}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, type=Path, help='run directory, e.g. runs/gt_full250_s42')
    parser.add_argument('--workers', type=int, default=None, help='process pool size (default: all cores)')
    parser.add_argument('--skip_cross_check', action='store_true',
                        help='debugging only: the run is then excluded from tables until it is cross-checked')
    args = parser.parse_args()

    rows, dataset_path = load_generated(args.run)
    info = run_info(args.run)
    print(f'{args.run}: {len(rows)} rows, {info}')

    evaluate = partial(evaluate_row, setting=info['setting'], seed=info['seed'],
                       max_regenerations=info['max_regenerations'])
    with mp.Pool(args.workers) as pool:  # Gurobi is CPU-bound; one build per task, results kept in order
        records = pool.map(evaluate, rows, chunksize=1)
    attach_clip_dino(records, args.run)

    out_path = args.run / 'per_sample.jsonl'
    out_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(f'Wrote {out_path}')
    print_summary(records)

    if args.skip_cross_check:
        (args.run / 'cross_check.json').unlink(missing_ok=True)  # an old check must not vouch for new results
        print('Cross-check SKIPPED: this run is excluded from tables until `make eval` cross-checks it.')
        return 0
    source = 'generated_dataset.json' if dataset_path.name == 'generated_dataset.json' else 'generations/ (partial run)'
    result = cross_check(args.run, rows, records, source)
    (args.run / 'cross_check.json').write_text(json.dumps(result, indent=2))
    print_cross_check(result)
    return 0 if result['match'] else 1


def print_cross_check(result: dict) -> None:
    """Human-readable cross-check report; loud when rows were excluded or the check failed."""
    if result['n_excluded']:
        print(f'NOTE: {result["n_excluded"]} of {result["n_rows"]} rows could not be parsed. They count as invalid in our '
              f'metrics and were left out of the upstream cross-check (upstream would crash on them):')
        for row in result['excluded'][:10]:
            print(f'  {row["object_id"]}: {row["parse_error"]}')
    if result['error']:
        print(f'CROSS-CHECK COULD NOT RUN: {result["error"]}', file=sys.stderr)
        return
    for key in result['ours']:
        ours, theirs = result['ours'][key], result['upstream'][key]
        flag = 'ok' if ours == theirs or abs(ours - theirs) < 1e-9 else 'MISMATCH'
        print(f'  {key:22} ours={ours!r:<24} upstream={theirs!r:<24} {flag}')
    if result['match']:
        print(f'Cross-check vs upstream evaluate on {result["n_checked"]}/{result["n_rows"]} rows: MATCH')
    else:
        print('CROSS-CHECK FAILED: per_sample.jsonl disagrees with upstream evaluate. Run is excluded from tables.',
              file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())

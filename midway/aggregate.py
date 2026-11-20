"""Aggregate every evaluated run into runs/aggregate.json (SPEC §6.5).

A *run set* is all runs with the same setting and prompt set, e.g. rs_full250 = runs/rs_full250_s42 (+ s43, ...).
Each metric is a mean (or median) with a 95% cluster-bootstrap CI over prompts (seeds of one prompt stay together).

A run set is *reportable* (allowed into tables/figures) only if every run in it passed the cross-check against
upstream evaluate (on its parseable rows) and per_sample.jsonl is up to date. Partial runs (fewer prompts than the
prompt set, e.g. a Colab run that stopped early) are reportable; tables show their actual n and CIs use that n.
Every run set that is NOT reportable, and every run dir that was never evaluated, is printed loudly here and listed
at the top of RESULTS.md, so no run silently disappears from the tables.

Run:  make report   (this is its first step)
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from midway.config import REPO_ROOT, RUNS_DIR, STABILITY_CONVENTION
from midway.per_sample_eval import attach_clip_dino, run_info
from midway.stats import cluster_bootstrap_ci

# (aggregate name, per-sample field, statistic). Over-valid-only metrics need no filter here:
# per_sample_eval stores None for them on invalid builds, and None values are ignored.
METRICS = [
    ('valid', 'valid', 'mean'),                                 # % valid = mean of 0/1 over all prompts
    ('stable', 'stable', 'mean'),                               # % stable = mean of 0/1 over all prompts
    ('mean_stab', f'mean_stab_{STABILITY_CONVENTION}', 'mean'),  # mean over valid builds of 1 - mean stress
    ('min_stab', f'min_stab_{STABILITY_CONVENTION}', 'mean'),    # mean over valid builds of 1 - max stress
    ('clip', 'clip', 'mean'),                                   # over valid builds with a render
    ('dino', 'dino', 'mean'),                                   # over valid builds with a render
    ('rollbacks_median', 'n_regenerations', 'median'),
    ('sec_per_build_median', 'generation_time', 'median'),
    ('n_bricks', 'n_bricks', 'mean'),                           # shape stats: over all parsed builds
    ('iou', 'voxel_iou_gt', 'mean'),
    ('height', 'height', 'mean'),
    ('footprint', 'footprint', 'mean'),
    ('frac_1x1', 'frac_1x1', 'mean'),
    ('hit_rb_limit', 'hit_rb_limit', 'mean'),
]


def load_records(run_dir: Path) -> list[dict]:
    """per_sample.jsonl of one run, with CLIP/DINO joined if upstream has produced them."""
    records = [json.loads(line) for line in (run_dir / 'per_sample.jsonl').read_text().splitlines()]
    attach_clip_dino(records, run_dir)
    return records


def run_source(run_dir: Path) -> Path | None:
    """The run's generation output: generated_dataset.json, or generations/ for an unfinished run."""
    for candidate in (run_dir / 'generated_dataset.json', run_dir / 'generations'):
        if candidate.exists():
            return candidate
    return None


def run_meta(run_dir: Path) -> dict:
    path = run_dir / 'run_meta.json'
    return json.loads(path.read_text()) if path.exists() else {}


def run_status(run_dir: Path) -> tuple[bool, str]:
    """(reportable, reason). Reportable = up-to-date per_sample.jsonl whose cross-check against upstream matched,
    and not excluded on purpose via `exclude_from_tables` in run_meta.json."""
    source = run_source(run_dir)
    per_sample = run_dir / 'per_sample.jsonl'
    if source is not None and source.stat().st_mtime > per_sample.stat().st_mtime:
        return False, f'per_sample.jsonl is older than {source.name}; rerun make eval'
    check_path = run_dir / 'cross_check.json'
    if not check_path.exists():
        return False, 'not cross-checked against upstream (rerun make eval)'
    check = json.loads(check_path.read_text())
    if check.get('error'):
        return False, f'cross-check could not run: {check["error"][:200]}'
    if not check['match']:
        return False, 'cross-check MISMATCH with upstream evaluate'
    note = f'cross-check matched on {check.get("n_checked", "all")}/{check.get("n_rows", "all")} rows'
    if check.get('n_excluded'):
        note += f' ({check["n_excluded"]} rows upstream would crash on were left out of the check; counted as invalid)'
    excluded_on_purpose = run_meta(run_dir).get('exclude_from_tables')
    if excluded_on_purpose:
        return False, f'excluded on purpose (run_meta.json exclude_from_tables: {excluded_on_purpose}); {note}'
    return True, note


def expected_n(prompts: str | None) -> int | None:
    """Number of prompts in a prompt set: full250 -> 250, subsetN -> N."""
    match = re.fullmatch(r'(?:full|subset|first)(\d+)', prompts or '')
    return int(match.group(1)) if match else None


def summarize(records: list[dict]) -> dict:
    """All METRICS for a list of per-sample records, each as {mean, lo, hi, n} (None where undefined)."""
    result = {}
    for name, key, statistic in METRICS:
        values = [{'object_id': r['object_id'], key: None if r.get(key) is None else float(r[key])}
                  for r in records]
        result[name] = cluster_bootstrap_ci(values, key, statistic=statistic)._asdict()
    return result


def find_runs(runs_dir: Path = RUNS_DIR) -> tuple[dict[str, list[Path]], list[str]]:
    """Group evaluated run dirs into run sets keyed '{setting}_{prompts}'.

    Also returns the evaluated dirs whose setting/prompt set cannot be identified (reported loudly, never dropped silently).
    """
    run_sets, unrecognized = defaultdict(list), []
    for per_sample in sorted(runs_dir.glob('*/per_sample.jsonl')):
        info = run_info(per_sample.parent)
        if info['setting'] is None or info['prompts'] is None:
            unrecognized.append(f'runs/{per_sample.parent.name}')
            continue
        run_sets[f'{info["setting"]}_{info["prompts"]}'].append(per_sample.parent)
    return run_sets, unrecognized


def aggregate_run_set(run_dirs: list[Path]) -> dict:
    """Summary of one run set: identity, provenance, reportable flag, metrics."""
    info = run_info(run_dirs[0])
    records = [r for d in run_dirs for r in load_records(d)]
    statuses = [run_status(d) for d in run_dirs]
    metas = [run_meta(d) for d in run_dirs]
    checks = [json.loads((d / 'cross_check.json').read_text()) if (d / 'cross_check.json').exists() else {}
              for d in run_dirs]
    gpus = sorted({m.get('gpu') or m.get('gpu_name') for m in metas} - {None})
    n_prompts = len({r['object_id'] for r in records})
    return {
        'setting': info['setting'],
        'prompts': info['prompts'],
        'seeds': sorted({run_info(d)['seed'] for d in run_dirs} - {None}),
        'runs': [str(d.relative_to(REPO_ROOT)) if d.is_relative_to(REPO_ROOT) else str(d) for d in run_dirs],
        'n_prompts': n_prompts,
        'expected_n': expected_n(info['prompts']),
        # partial: fewer prompts than the prompt set, or marked partial in run_meta.json (e.g. first100 of a 250 run)
        'partial': (expected_n(info['prompts']) is not None and n_prompts < expected_n(info['prompts']))
                   or any(m.get('partial') for m in metas),
        'n_records': len(records),
        'n_parse_errors': sum(r.get('parse_error') is not None for r in records),
        'invalid_reasons': dict(Counter(r['invalid_reason'] for r in records if r.get('invalid_reason'))),
        'n_cross_check_excluded': sum(c.get('n_excluded', 0) for c in checks),
        'gpu': ', '.join(gpus) if gpus else None,
        'reportable': all(ok for ok, _ in statuses),
        'status': '; '.join(f'{d.name}: {why}' for d, (_, why) in zip(run_dirs, statuses)),
        'metrics': summarize(records),
    }


def primary_run_set(agg: dict, setting: str) -> dict | None:
    """The reportable run set used for `setting` in tables/figures: full250 if present, else the largest subset."""
    candidates = [rs for rs in agg['run_sets'].values() if rs['setting'] == setting and rs['reportable']]
    if not candidates:
        return None
    return max(candidates, key=lambda rs: (rs['prompts'] == 'full250', rs['n_prompts']))


def load_aggregate(path: Path = RUNS_DIR / 'aggregate.json') -> dict:
    return json.loads(path.read_text())


def unevaluated_run_dirs(runs_dir: Path = RUNS_DIR) -> list[str]:
    """Run dirs with generation output but no per_sample.jsonl: never evaluated (or eval crashed)."""
    return sorted(f'runs/{d.name}' for d in runs_dir.iterdir()
                  if d.is_dir() and run_source(d) is not None and not (d / 'per_sample.jsonl').exists())


def print_loud_warnings(agg: dict) -> None:
    """Make every exclusion impossible to miss in `make report` output."""
    problems = [f'{name}: {rs["status"]}' for name, rs in agg['run_sets'].items() if not rs['reportable']]
    problems += [f'{d}: has generation output but was never evaluated (make eval RUN={d})' for d in agg['unevaluated']]
    problems += [f'{d}: evaluated, but setting/prompt set unknown (name it {{setting}}_{{prompts}}_s{{seed}})'
                 for d in agg['unrecognized']]
    if problems:
        bar = '!' * 100
        print(f'{bar}\nEXCLUDED FROM TABLES AND FIGURES:')
        for line in problems:
            print(f'  {line}')
        print(bar)
    for name, rs in agg['run_sets'].items():
        if rs['partial']:
            print(f'NOTE {name}: partial run, n={rs["n_prompts"]} prompts (tables show this n).')
        if rs['n_cross_check_excluded']:
            print(f'NOTE {name}: {rs["n_cross_check_excluded"]} rows left out of the upstream cross-check '
                  f'(upstream would crash on them); counted as invalid.')


def main() -> None:
    run_sets, unrecognized = find_runs()
    agg = {'stability_convention': STABILITY_CONVENTION,
           'run_sets': {name: aggregate_run_set(dirs) for name, dirs in run_sets.items()},
           'unevaluated': unevaluated_run_dirs(),
           'unrecognized': unrecognized}
    out = RUNS_DIR / 'aggregate.json'
    out.write_text(json.dumps(agg, indent=2))
    print(f'Wrote {out}')
    for name, rs in agg['run_sets'].items():
        flag = 'reportable' if rs['reportable'] else 'NOT reportable'
        print(f'  {name}: n={rs["n_prompts"]}/{rs["expected_n"]} seeds={rs["seeds"]} {flag} ({rs["status"]})')
    print_loud_warnings(agg)


if __name__ == '__main__':
    main()

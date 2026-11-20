"""runs/RESULTS.md: plain-text digest of every evaluated run set (SPEC §7.4).

For each run set: provenance, whether it is reportable, and every metric with its 95% CI, n, the paper value
(where the paper has one) and ours minus paper. Lets the team write Section 5 and check table cells quickly.

Run:  make report   (last step)
"""
from datetime import datetime

from midway.aggregate import load_aggregate, primary_run_set
from midway.config import BOOTSTRAP_N, PAPER_TABLE1, RUNS_DIR, SETTINGS

# (label, metric name, scale, decimals)
ROWS = [
    ('% valid', 'valid', 100, 1),
    ('% valid by upstream rule (ignores brick library)', 'valid_upstream', 100, 1),
    ('% stable', 'stable', 100, 1),
    ('Mean stab. (1 - mean stress, valid builds)', 'mean_stab', 1, 3),
    ('Min. stab. (1 - max stress, valid builds)', 'min_stab', 1, 3),
    ('CLIP (valid builds)', 'clip', 1, 3),
    ('DINO (valid builds)', 'dino', 1, 3),
    ('Rollbacks (median)', 'rollbacks_median', 1, 1),
    ('Seconds per build (median)', 'sec_per_build_median', 1, 2),
    ('Bricks per build (mean)', 'n_bricks', 1, 1),
    ('Voxel IoU vs ground truth (mean)', 'iou', 1, 3),
    ('Height in layers (mean)', 'height', 1, 2),
    ('Footprint in cells (mean)', 'footprint', 1, 1),
    ('% 1x1 bricks (mean)', 'frac_1x1', 100, 1),
    ('% hit rollback limit and unstable', 'hit_rb_limit', 100, 1),
]


def fmt(x, scale: float, decimals: int) -> str:
    return 'pending' if x is None else f'{scale * x:.{decimals}f}'


def metric_row(label: str, ci: dict, paper, scale: float, decimals: int) -> str:
    """One markdown table row: ours, CI, n, paper, ours - paper (paper values are already scaled)."""
    ours = fmt(ci['mean'], scale, decimals)
    interval = 'n/a' if ci['mean'] is None else f'[{fmt(ci["lo"], scale, decimals)}, {fmt(ci["hi"], scale, decimals)}]'
    paper_txt = 'n/r' if paper is None else f'{paper:.{decimals}f}'
    gap = 'n/a' if paper is None or ci['mean'] is None else f'{scale * ci["mean"] - paper:+.{decimals}f}'
    return f'| {label} | {ours} | {interval} | {ci["n"]} | {paper_txt} | {gap} |'


def table_use(name: str, rs: dict, agg: dict) -> str:
    """Whether this run set feeds the tables/figures, and if not, why."""
    if not rs['reportable']:
        return f'**NO: excluded from tables and figures.** Reason: {rs["status"]}'
    if rs['setting'] not in SETTINGS or rs['setting'] == 'gt':
        return 'reference only (ground truth: Figure 3 and this digest)'
    chosen = primary_run_set(agg, rs['setting'])
    if chosen['runs'] != rs['runs']:
        return (f'**NO: not used in tables**; {chosen["prompts"]} is used for this setting '
                f'(tables prefer full250, else the largest subset)')
    return f'yes, as the "{SETTINGS[rs["setting"]]["label"]}" setting'


def run_set_section(name: str, rs: dict, agg: dict) -> list[str]:
    label = SETTINGS.get(rs['setting'], {}).get('label', rs['setting'])
    n = f'{rs["n_prompts"]} prompts' + (' (**partial run**)' if rs['partial'] else '')
    reasons = ', '.join(f'{k}: {v}' for k, v in sorted(rs.get('invalid_reasons', {}).items())) or 'none'
    lines = [f'## {name}: {label}', '',
             f'- In tables: {table_use(name, rs, agg)}',
             f'- Cross-check: {rs["status"]}',
             f'- Runs: {", ".join(rs["runs"])}; prompt set {rs["prompts"]}; n = {n}; {rs["n_records"]} records; '
             f'seeds {rs["seeds"]}; GPU {rs["gpu"] or "n/a"}',
             f'- Invalid builds by reason: {reasons}',
             f'- Builds valid by upstream\'s rule but invalid only because of a brick size not in the library: '
             f'{rs.get("n_brick_type_only", 0)}',
             f'- Rows left out of the upstream cross-check (upstream would crash; counted invalid): '
             f'{rs.get("n_cross_check_excluded", 0)}', '',
             '| Metric | Ours | 95% CI | n | Paper | Ours - paper |', '|---|---|---|---|---|---|']
    paper = PAPER_TABLE1.get(rs['setting'], {})
    rollback_off = SETTINGS.get(rs['setting'], {}).get('max_regenerations') == 0
    for row_label, metric, scale, decimals in ROWS:
        if metric in ('rollbacks_median', 'hit_rb_limit') and rollback_off:
            lines.append(f'| {row_label} | n/a (RB off) | n/a | n/a | n/a | n/a |')
            continue
        lines.append(metric_row(row_label, rs['metrics'][metric], paper.get(metric), scale, decimals))
    return lines + ['']


def attention_lines(agg: dict) -> list[str]:
    """Everything a reader must not miss: exclusions, unevaluated runs, partial runs, parse errors."""
    items = [f'- **EXCLUDED from tables: {name}**: {rs["status"]}' for name, rs in agg['run_sets'].items()
             if not rs['reportable']]
    items += [f'- **NOT EVALUATED: {d}** has generation output but no per_sample.jsonl (run `make eval RUN={d}`)'
              for d in agg.get('unevaluated', [])]
    items += [f'- **UNRECOGNIZED: {d}** was evaluated but its setting/prompt set is unknown '
              f'(name it {{setting}}_{{prompts}}_s{{seed}})' for d in agg.get('unrecognized', [])]
    items += [f'- {name}: partial run, n = {rs["n_prompts"]} prompts; tables show this n'
              for name, rs in agg['run_sets'].items() if rs['partial']]
    items += [f'- {name}: {rs["n_cross_check_excluded"]} rows left out of the upstream cross-check (upstream would '
              f'crash on them); counted as invalid' for name, rs in agg['run_sets'].items()
              if rs.get('n_cross_check_excluded')]
    items += [f'- {name}: {table_use(name, rs, agg)}' for name, rs in agg['run_sets'].items()
              if rs['reportable'] and table_use(name, rs, agg).startswith('**NO')]
    return ['## Attention', ''] + (items or ['- Nothing excluded; no partial runs; no unparseable builds.']) + ['']


def main() -> None:
    agg = load_aggregate()
    lines = ['# Results digest', '',
             f'Auto-generated by `midway/results_md.py` on {datetime.now():%Y-%m-%d %H:%M}; do not edit by hand. '
             'Stability: raw (1 - stress); upstream caps stress at 1, so raw equals clipped. '
             f'CIs: 95% percentile bootstrap, {BOOTSTRAP_N:,} resamples over the n prompts actually evaluated '
             '(seeds of a prompt kept together). Rates are over all evaluated prompts; stability/CLIP/DINO over valid '
             'builds; shape stats over all parsed builds. Paper values: BrickGPT paper Table 1 (n/r = not reported). '
             '"pending" = no data. CLIP/DINO are not computed for the midway (see runs/NOTES.md).', '']
    lines += attention_lines(agg)
    for name, rs in sorted(agg['run_sets'].items()):
        lines += run_set_section(name, rs, agg)
    out = RUNS_DIR / 'RESULTS.md'
    out.write_text('\n'.join(lines))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

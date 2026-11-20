"""Unparseable builds, cross-check gating, partial runs, and loud exclusions.

Unparseable rows must never crash evaluation: they get valid=False and a parse_error. The cross-check runs on
the parseable rows only and records what it excluded. Partial runs (fewer prompts than the prompt set) stay
reportable with their actual n. Excluded runs are listed in RESULTS.md, never silently dropped.
"""
import json
import os
import subprocess
import sys

import pytest

from midway.aggregate import aggregate_run_set, run_status
from midway.config import REPO_ROOT
from midway.per_sample_eval import evaluate_row
from midway.results_md import attention_lines
from midway.tables import provenance

GT = [{'role': 'assistant', 'content': '2x4 (0,0,0)\n'}]


def row(object_id: str, output) -> dict:
    return {'object_id': object_id, 'category_id': 'c', 'caption': 'a box', 'messages': GT,
            'output': output, 'generation_time': 1.0, 'n_regenerations': 0, 'rejection_reasons': {}}


@pytest.mark.parametrize('output, reason', [
    ('2x4 (0,0,0)\n2x4 (0,0', 'ill-formatted'),        # ill-formatted line
    ('2x4 (0,0,0)\n2x4 (0,0,25)', 'outside the 20x20x20 grid'),  # out-of-range coordinate (z >= 20)
    ('3x3 (0,0,0)', 'not in the brick library'),        # parses, but upstream stability code would raise
    (None, 'missing or not text'),                      # no output at all
])
def test_unparseable_row_is_invalid_with_parse_error(output, reason):
    record = evaluate_row(row('o1', output), setting='rs', seed=42, max_regenerations=0)
    assert reason in record['parse_error']
    assert record['valid'] is False and record['stable'] is False
    assert record['n_bricks'] is None and record['max_stress'] is None and record['min_stab_raw'] is None


def test_parseable_row_has_no_parse_error():
    record = evaluate_row(row('o1', '2x4 (0,0,0)\n'), setting='rs', seed=42, max_regenerations=0)
    assert record['parse_error'] is None and record['valid'] and record['stable']


def test_out_of_bounds_x_is_invalid_not_unparseable():
    record = evaluate_row(row('o1', '2x4 (19,0,0)\n'), setting='rs', seed=42, max_regenerations=0)
    assert record['parse_error'] is None and record['valid'] is False


def test_partial_run_with_unparseable_row_end_to_end(tmp_path):
    """3 of 5 prompts finished (generations/ only, no generated_dataset.json), one of them unparseable.

    Runs the real per_sample_eval, including upstream evaluate on the 2 parseable rows.
    """
    run = tmp_path / 'rs_subset5_s42'
    (run / 'generations').mkdir(parents=True)
    rows = [row('a', '2x4 (0,0,0)\n'), row('b', '2x4 (0,0,0)\n2x4 (0,0,1)\n'), row('c', '2x4 (0,0,')]
    for r in rows:
        (run / 'generations' / f'{r["object_id"]}.json').write_text(json.dumps(r))

    env = os.environ | {'PYTHONPATH': str(REPO_ROOT)}
    proc = subprocess.run([sys.executable, '-m', 'midway.per_sample_eval', '--run', str(run), '--workers', '2'],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    records = {r['object_id']: r for r in map(json.loads, (run / 'per_sample.jsonl').read_text().splitlines())}
    assert 'ill-formatted' in records['c']['parse_error'] and records['c']['valid'] is False

    check = json.loads((run / 'cross_check.json').read_text())
    assert check['match'] is True
    assert (check['n_rows'], check['n_checked'], check['n_excluded']) == (3, 2, 1)
    assert check['excluded'][0]['object_id'] == 'c'
    assert check['source'].startswith('generations/')

    reportable, note = run_status(run)
    assert reportable and '1 unparseable' in note

    rs = aggregate_run_set([run])
    assert rs['reportable'] and rs['partial'] and rs['n_prompts'] == 3 and rs['expected_n'] == 5
    assert rs['metrics']['valid']['n'] == 3 and rs['metrics']['valid']['mean'] == pytest.approx(2 / 3)
    assert provenance(rs) == 'subset5, n=3 of 5, s42'


def test_mismatch_or_missing_check_excludes_run_loudly(tmp_path):
    run = tmp_path / 'rs_full250_s42'
    run.mkdir()
    (run / 'generated_dataset.json').write_text('[]')
    (run / 'per_sample.jsonl').write_text('')
    assert run_status(run) == (False, 'not cross-checked against upstream (rerun make eval)')
    (run / 'cross_check.json').write_text(json.dumps({'match': False, 'error': None}))
    assert run_status(run)[0] is False and 'MISMATCH' in run_status(run)[1]


def test_results_md_attention_lists_exclusions_and_unevaluated():
    agg = {'run_sets': {'rs_full250': {'reportable': False, 'status': 'cross-check MISMATCH with upstream evaluate',
                                       'partial': False, 'n_parse_errors': 0, 'setting': 'rs'}},
           'unevaluated': ['runs/full_subset50_s42']}
    text = '\n'.join(attention_lines(agg))
    assert '**EXCLUDED from tables: rs_full250**' in text and 'MISMATCH' in text
    assert '**NOT EVALUATED: runs/full_subset50_s42**' in text

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
    (None, 'missing or not text'),                      # no output at all
])
def test_unparseable_row_is_invalid_with_parse_error(output, reason):
    record = evaluate_row(row('o1', output), setting='rs', seed=42, max_regenerations=0)
    assert reason in record['parse_error']
    assert record['valid'] is False and record['stable'] is False and record['invalid_reason'] == 'unparseable'
    assert record['n_bricks'] is None and record['max_stress'] is None and record['min_stab_raw'] is None


@pytest.mark.parametrize('output, reason', [
    ('3x1 (0,0,0)', 'brick_not_in_library'),                      # parses; upstream solver would raise
    ('2x4 (0,0,0)\n3x1 (0,0,0)', 'collision+brick_not_in_library'),  # the case seen in norsnorb_first100_s42
    ('2x4 (0,0,0)\n2x4 (1,0,0)', 'collision'),
    ('', 'empty'),
])
def test_invalid_reason(output, reason):
    record = evaluate_row(row('o1', output), setting='rs', seed=42, max_regenerations=0)
    assert record['parse_error'] is None and record['valid'] is False and record['invalid_reason'] == reason
    # upstream's rule ignores the brick library: only a build whose sole problem is brick type is upstream-valid
    assert record['valid_upstream'] is (reason == 'brick_not_in_library')
    assert record['n_bricks'] == len([line for line in output.splitlines() if line.strip()])  # still counted


def test_parseable_row_has_no_parse_error():
    record = evaluate_row(row('o1', '2x4 (0,0,0)\n'), setting='rs', seed=42, max_regenerations=0)
    assert record['parse_error'] is None and record['valid'] and record['stable']


def test_out_of_bounds_x_is_invalid_not_unparseable():
    record = evaluate_row(row('o1', '2x4 (19,0,0)\n'), setting='rs', seed=42, max_regenerations=0)
    assert record['parse_error'] is None and record['valid'] is False and record['invalid_reason'] == 'out_of_bounds'


def test_partial_run_with_unparseable_row_end_to_end(tmp_path):
    """5 of 7 prompts finished (generations/ only, no generated_dataset.json). Row c is unparseable and row d is an
    otherwise-valid build with a 3x1 brick: upstream evaluate would crash on both, so they are left out of the
    cross-check. Row e has a 3x1 AND a collision: upstream calls it invalid without crashing, so it stays in.

    Runs the real per_sample_eval, including upstream evaluate on rows a, b, e.
    """
    run = tmp_path / 'rs_subset7_s42'
    (run / 'generations').mkdir(parents=True)
    rows = [row('a', '2x4 (0,0,0)\n'), row('b', '2x4 (0,0,0)\n2x4 (0,0,1)\n'), row('c', '2x4 (0,0,'),
            row('d', '3x1 (0,0,0)\n'), row('e', '2x4 (0,0,0)\n3x1 (0,0,0)\n')]
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
    assert (check['n_rows'], check['n_checked'], check['n_excluded']) == (5, 3, 2)
    assert [x['object_id'] for x in check['excluded']] == ['c', 'd']
    assert check['n_brick_type_only'] == 1  # row d: a non-colliding 3x1, valid by upstream's rule only
    assert check['ours']['n_valid'] == check['upstream']['n_valid'] == 2  # rows a, b (e collides)
    assert check['source'].startswith('generations/')

    reportable, note = run_status(run)
    assert reportable and '2 rows upstream would crash on' in note

    rs = aggregate_run_set([run])
    assert rs['reportable'] and rs['partial'] and rs['n_prompts'] == 5 and rs['expected_n'] == 7
    assert rs['metrics']['valid']['n'] == 5 and rs['metrics']['valid']['mean'] == pytest.approx(2 / 5)
    assert rs['invalid_reasons'] == {'unparseable': 1, 'brick_not_in_library': 1, 'collision+brick_not_in_library': 1}
    assert rs['n_brick_type_only'] == 1 and rs['metrics']['valid_upstream']['mean'] == pytest.approx(3 / 5)
    assert provenance(rs) == 'subset7, n=5 of 7, s42'

    # Excluding the run on purpose keeps it out of the tables, with the reason in its status.
    (run / 'run_meta.json').write_text(json.dumps({'exclude_from_tables': 'pipeline test', 'partial': True}))
    reportable, note = run_status(run)
    assert not reportable and note.startswith('excluded on purpose') and 'pipeline test' in note


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
                                       'partial': False, 'n_cross_check_excluded': 0, 'setting': 'rs'}},
           'unevaluated': ['runs/full_subset50_s42'], 'unrecognized': ['runs/my_test_run']}
    text = '\n'.join(attention_lines(agg))
    assert '**EXCLUDED from tables: rs_full250**' in text and 'MISMATCH' in text
    assert '**NOT EVALUATED: runs/full_subset50_s42**' in text
    assert '**UNRECOGNIZED: runs/my_test_run**' in text


def test_first_n_prompt_set_and_partial_label():
    from midway.make_subset import load_prompts
    first = load_prompts('first100')
    assert [p['object_id'] for p in first] == [p['object_id'] for p in load_prompts('full250')[:100]]
    rs = {'prompts': 'first100', 'n_prompts': 100, 'expected_n': 100, 'partial': True, 'seeds': [42]}
    assert provenance(rs) == 'first100, n=100, partial, s42'

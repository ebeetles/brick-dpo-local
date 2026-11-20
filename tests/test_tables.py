"""SPEC §9 item 10: a missing run renders \\tbd, never 0 or nan. Also the run-dir interface with Megha's runs.

All aggregates here are synthetic test inputs; nothing in this file is ever written to report/.
"""
import json
import re

from midway.aggregate import summarize
from midway.per_sample_eval import evaluate_row, run_info
from midway.tables import fmt_ci, table1, table2, table3, table4

TBD = r'\tbd'


def ci(mean, lo=None, hi=None, n=250):
    return {'mean': mean, 'lo': mean if lo is None else lo, 'hi': mean if hi is None else hi, 'n': n}


def fake_run_set(setting, metrics):
    return {'setting': setting, 'prompts': 'full250', 'seeds': [42], 'runs': [f'runs/{setting}_full250_s42'],
            'n_prompts': 250, 'gpu': 'NVIDIA A10G',
            'reportable': True, 'metrics': metrics}


def cells(tex: str) -> list[str]:
    """Body cells of every tabular row (split on & and \\\\)."""
    body = tex.split(r'\midrule', 1)[1].split(r'\bottomrule')[0]
    return [c.strip() for row in body.split(r'\\') for c in row.split('&') if c.strip()]


def assert_no_zero_or_nan(tex: str):
    assert 'nan' not in tex.lower()
    for cell in cells(tex):
        assert not re.fullmatch(r'0(\.0+)?', cell), f'bare zero cell: {cell!r}'


EMPTY = {'stability_convention': 'raw', 'run_sets': {}}


def test_fmt_missing_is_tbd():
    assert fmt_ci(None, 'pct') == TBD
    assert fmt_ci(ci(None, n=0), 'pct') == TBD
    assert fmt_ci(ci(float('nan')), 'stab') == TBD


def test_fmt_with_ci():
    assert fmt_ci(ci(0.228, 0.176, 0.280), 'pct') == r'22.8\textsubscript{\,$\pm$5.2}'


def test_table1_no_runs_all_ours_tbd():
    tex = table1(EMPTY)
    assert_no_zero_or_nan(tex)
    body = tex.split(r'\midrule', 1)[1].split(r'\bottomrule')[0]
    for row in filter(str.strip, body.split(r'\\')):
        for cell in row.split('&')[1:]:  # skip the row label; every other cell is "paper / ours"
            if cell.strip() == 'n/a (RB off)':
                continue
            assert cell.rsplit('/', 1)[1].strip() == TBD


def test_table1_rollbacks_na_when_rollback_off():
    rollbacks = next(line for line in table1(EMPTY).splitlines() if line.startswith('Rollbacks'))
    norsnorb, rs, full = [c.strip() for c in rollbacks.rstrip('\\ ').split('&')[1:]]
    assert norsnorb == rs == 'n/a (RB off)'
    assert full == f'2.0 / {TBD}'


def test_table1_only_rs_run_fills_only_rs_column():
    metrics = {k: ci(None, n=0) for k in ('clip', 'dino', 'rollbacks_median', 'sec_per_build_median',
                                          'mean_stab', 'min_stab')}
    metrics |= {'valid': ci(1.0), 'stable': ci(0.228, 0.176, 0.280)}
    agg = {'stability_convention': 'raw', 'run_sets': {'rs_full250': fake_run_set('rs', metrics)}}
    tex = table1(agg)
    assert_no_zero_or_nan(tex)
    stable_row = next(line for line in tex.splitlines() if line.startswith(r'\% stable'))
    norsnorb, rs, full = [c.strip() for c in stable_row.rstrip('\\ ').split('&')[1:]]
    assert norsnorb == f'12.8 / {TBD}'
    assert rs.startswith('24.0 / 22.8')
    assert full == f'98.8 / {TBD}'
    clip_row = next(line for line in tex.splitlines() if line.startswith('CLIP'))
    assert clip_row.count(TBD) == 3  # metric exists but undefined -> tbd, not 0


def test_table1_caption_names_partial_column():
    full = fake_run_set('full', {'stable': ci(0.958, 0.875, 1.0, n=24)}) | {
        'prompts': 'first30', 'n_prompts': 24, 'expected_n': 30, 'partial': True}
    tex = table1({'stability_convention': 'raw', 'run_sets': {'full_first30': full}})
    assert 'RS + RB: partial run, $n=24$ of 30 (first30).' in tex
    assert 'first30, n=24 of 30, s42' in tex  # column header
    assert 'No RS, no RB: partial' not in tex


def test_table1_gpu_caption_names_columns_without_gpu():
    rs = fake_run_set('rs', {'stable': ci(0.23)})
    full = fake_run_set('full', {'stable': ci(0.96)}) | {'gpu': None}
    tex = table1({'stability_convention': 'raw', 'run_sets': {'rs_full250': rs, 'full_full250': full}})
    assert 'GPU: NVIDIA A10G (RS + RB: not recorded).' in tex


def test_unreportable_run_set_is_ignored():
    rs = fake_run_set('rs', {'stable': ci(0.5)}) | {'reportable': False}
    tex = table1({'stability_convention': 'raw', 'run_sets': {'rs_full250': rs}})
    assert '50.0' not in tex


def test_skeleton_tables_dpo_rows_all_tbd():
    for make in (table2, table3, table4):
        tex = make(EMPTY)
        assert_no_zero_or_nan(tex)
        for line in tex.splitlines():
            if line.startswith(('RFT', 'DPO')):
                assert all(c.strip() == TBD for c in line.rstrip('\\ ').split('&')[1:]), line


def test_missing_optional_fields_become_null_not_zero():
    # A BrickGPT-format row without n_regenerations / rejection_reasons (e.g. a different generator).
    row = {'object_id': 'o1', 'category_id': 'c', 'caption': 'x', 'output': '2x4 (0,0,0)\n',
           'messages': [{'content': '2x4 (0,0,0)\n'}], 'generation_time': 1.5}
    record = evaluate_row(row, setting='rs', seed=42, max_regenerations=0)
    assert record['n_regenerations'] is None and record['n_rejections'] is None and record['hit_rb_limit'] is None
    summary = summarize([record])
    assert summary['rollbacks_median']['mean'] is None
    assert summary['sec_per_build_median']['mean'] == 1.5


def test_brickgpt_row_fields():
    row = {'object_id': 'o1', 'output': '2x4 (0,0,2)\n', 'messages': [{'content': '2x4 (0,0,0)\n'}],
           'generation_time': 3.0, 'n_regenerations': 100,
           'rejection_reasons': {'collision': 3, 'out_of_bounds': 2}}
    record = evaluate_row(row, setting='full', seed=42, max_regenerations=100)
    assert record['n_rejections'] == 5
    assert record['stable'] is False and record['hit_rb_limit'] is True
    assert record['voxel_iou_gt'] == 0.0


def test_run_info_from_name_and_meta(tmp_path):
    run = tmp_path / 'full_subset50_s43'
    run.mkdir()
    assert run_info(run) == {'setting': 'full', 'prompts': 'subset50', 'seed': 43, 'max_regenerations': 100}
    (run / 'run_meta.json').write_text(json.dumps({'flags': '--max_regenerations 7'}))
    assert run_info(run)['max_regenerations'] == 7


def test_clip_dino_join_valid_builds_only(tmp_path):
    from midway.per_sample_eval import attach_clip_dino
    (tmp_path / 'clip_analysis.json').write_text(json.dumps({'a.png': 0.3, 'b.png': 0.4, 'mean_clip_score': 0.35}))
    (tmp_path / 'img_similarity_analysis.json').write_text(json.dumps({'dino': {'a.png': 0.9, 'b.png': 0.8}}))
    records = [{'object_id': 'a', 'valid': True}, {'object_id': 'b', 'valid': False}, {'object_id': 'c', 'valid': True}]
    attach_clip_dino(records, tmp_path)
    assert (records[0]['clip'], records[0]['dino']) == (0.3, 0.9)
    assert (records[1]['clip'], records[1]['dino']) == (None, None)  # invalid build: excluded, paper convention
    assert (records[2]['clip'], records[2]['dino']) == (None, None)  # no render: null, not 0

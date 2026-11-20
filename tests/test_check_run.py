"""`make check-run`: format validation of a run folder without evaluating it."""
import json

from midway.check_run import ERROR, WARN, check_run
from midway.make_subset import load_prompts

PROMPTS = load_prompts('full250')


def make_run(tmp_path, name, rows, meta=None):
    run = tmp_path / name
    run.mkdir()
    (run / 'generated_dataset.json').write_text(json.dumps(rows))
    if meta is not None:
        (run / 'run_meta.json').write_text(json.dumps(meta))
    return run


def brickgpt_row(prompt, n_regenerations=0):
    return prompt | {'output': '2x4 (0,0,0)\n', 'generation_time': 12.3,
                     'n_regenerations': n_regenerations, 'rejection_reasons': {'collision': 2}}


def levels(findings, level):
    return [message for lvl, message in findings if lvl == level]


def test_good_full_run_has_no_errors_or_warnings(tmp_path):
    meta = {'setting': 'rs', 'flags': '--max_regenerations 0', 'seed': 42, 'prompts': 'full250', 'upstream_sha': 'x',
            'brickgpt_sha': 'y', 'hostname': 'h', 'gpu': 'NVIDIA A10G', 'n_workers': 4, 'start_time': 't0',
            'end_time': 't1', 'exit_status': 0}
    findings = check_run(make_run(tmp_path, 'rs_full250_s42', [brickgpt_row(p) for p in PROMPTS], meta))
    assert levels(findings, ERROR) == [] and levels(findings, WARN) == []


def test_partial_run_warns_but_passes(tmp_path):
    findings = check_run(make_run(tmp_path, 'norsnorb_full250_s42', [brickgpt_row(p) for p in PROMPTS[:180]]))
    assert levels(findings, ERROR) == []
    assert any('partial run: 180 of 250' in m for m in levels(findings, WARN))


def test_missing_output_and_bad_name_are_errors(tmp_path):
    rows = [brickgpt_row(p) for p in PROMPTS[:3]]
    del rows[0]['output']
    assert any("lack a valid 'output'" in m for m in levels(check_run(make_run(tmp_path, 'rs_full250_s42', rows)), ERROR))
    assert any('folder name' in m for m in levels(check_run(make_run(tmp_path, 'my_run', rows)), ERROR))


def test_rollbacks_in_rs_only_run_is_error(tmp_path):
    rows = [brickgpt_row(p, n_regenerations=3) for p in PROMPTS[:3]]
    assert any('rollback off' in m for m in levels(check_run(make_run(tmp_path, 'rs_full250_s42', rows)), ERROR))


def test_meta_disagreeing_with_folder_name_is_error(tmp_path):
    rows = [brickgpt_row(p) for p in PROMPTS[:3]]
    findings = check_run(make_run(tmp_path, 'rs_full250_s42', rows, {'setting': 'full', 'seed': 42}))
    assert any("setting='full'" in m for m in levels(findings, ERROR))


def test_unknown_object_id_is_error_and_unparseable_is_warning(tmp_path):
    rows = [brickgpt_row(p) for p in PROMPTS[:2]]
    rows[0] = rows[0] | {'object_id': 'not-a-real-id'}
    rows[1] = rows[1] | {'output': '2x4 (0,0'}
    findings = check_run(make_run(tmp_path, 'rs_full250_s42', rows))
    assert any('not in prompt set' in m for m in levels(findings, ERROR))
    assert any('cannot be parsed' in m for m in levels(findings, WARN))

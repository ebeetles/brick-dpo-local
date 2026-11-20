"""Validate a run folder's format WITHOUT evaluating it (no Gurobi, no upstream evaluate; takes seconds).

Checks the folder name, generated_dataset.json (or generations/ for an unfinished run), required row fields,
object_ids against the prompt set, BrickGPT-only fields, flag consistency, unparseable outputs, and run_meta.json.
Prints one line per finding: ERROR (must fix before `make eval`), WARN (eval works; something will be null,
partial, or invalid), OK. Exits 1 if there is any ERROR.

Run:  make check-run RUN=runs/<name>
"""
import argparse
import json
import re
import sys
from pathlib import Path

from midway.config import PROMPT_SET_PATTERN, RUN_NAME_PATTERN, SETTINGS
from midway.make_subset import load_prompts
from midway.shapes import parse_build_checked, sizes_not_in_library

ERROR, WARN, OK = 'ERROR', 'WARN', 'OK'
META_FIELDS = ['setting', 'flags', 'seed', 'prompts', 'upstream_sha', 'brickgpt_sha', 'hostname', 'gpu',
               'n_workers', 'start_time', 'end_time', 'exit_status']
BRICKGPT_SETTINGS = ('norsnorb', 'rs', 'full')


def examples(ids: list) -> str:
    return ', '.join(map(str, ids[:3])) + (' ...' if len(ids) > 3 else '')


def check_name(run_dir: Path, meta: dict) -> tuple[list, dict]:
    """Folder name must be {setting}_{prompts}_s{seed}; run_meta.json (if any) must agree with it."""
    match = re.fullmatch(RUN_NAME_PATTERN, run_dir.name)
    if not match:
        return [(ERROR, f'folder name {run_dir.name!r} is not {{setting}}_{{prompts}}_s{{seed}}, e.g. rs_full250_s42; '
                        f'setting is one of {", ".join(SETTINGS)}; prompts must match {PROMPT_SET_PATTERN}')], {}
    info = match.groupdict() | {'seed': int(match['seed'])}
    findings = []
    if info['setting'] not in SETTINGS:
        findings.append((ERROR, f'unknown setting {info["setting"]!r}; expected one of {", ".join(SETTINGS)}'))
    for key in ('setting', 'prompts', 'seed'):
        if key in meta and str(meta[key]) != str(info[key]):
            findings.append((ERROR, f'run_meta.json says {key}={meta[key]!r} but the folder name says {info[key]!r}'))
    return findings or [(OK, f'folder name: setting={info["setting"]}, prompts={info["prompts"]}, seed={info["seed"]}')], info


def load_rows(run_dir: Path) -> tuple[list, list | None]:
    """Rows from generated_dataset.json, else from generations/*.json (partial run)."""
    dataset = run_dir / 'generated_dataset.json'
    if dataset.exists():
        try:
            rows = json.loads(dataset.read_text())
        except json.JSONDecodeError as err:
            return [(ERROR, f'generated_dataset.json is not valid JSON: {err}')], None
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            return [(ERROR, 'generated_dataset.json must be a JSON list of objects (one per prompt)')], None
        return [(OK, f'generated_dataset.json: {len(rows)} rows')], rows
    generations = run_dir / 'generations'
    if generations.is_dir():
        rows, bad = [], []
        for path in sorted(generations.glob('*.json')):
            try:
                rows.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                bad.append(path.name)
        findings = [(WARN, f'no generated_dataset.json: unfinished run, using {len(rows)} files in generations/')]
        if bad:
            findings.append((WARN, f'{len(bad)} unreadable files in generations/ (ignored; eval will fail on them): {examples(bad)}'))
        return findings, rows
    return [(ERROR, 'neither generated_dataset.json nor generations/ found')], None


def check_rows(rows: list, info: dict) -> list:
    """Required fields, unique object_ids from the prompt set, BrickGPT fields, flag consistency, parseability."""
    findings = []
    for field, kind in (('object_id', str), ('output', str), ('generation_time', (int, float))):
        missing = [r.get('object_id', f'row {i}') for i, r in enumerate(rows) if not isinstance(r.get(field), kind)]
        if missing:
            findings.append((ERROR, f'{len(missing)} rows lack a valid {field!r}: {examples(missing)}'))
    ids = [r.get('object_id') for r in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        findings.append((ERROR, f'duplicate object_ids: {examples(duplicates)}'))

    try:
        prompt_ids = {p['object_id'] for p in load_prompts(info['prompts'])}
    except (FileNotFoundError, ValueError):
        return findings + [(ERROR, f'prompt set {info["prompts"]!r} not found (subsets live in runs/prompts/)')]
    unknown = [i for i in ids if i not in prompt_ids]
    if unknown:
        findings.append((ERROR, f'{len(unknown)} object_ids are not in prompt set {info["prompts"]}: {examples(unknown)}'))
    if len(set(ids) & prompt_ids) < len(prompt_ids):
        findings.append((WARN, f'partial run: {len(set(ids) & prompt_ids)} of {len(prompt_ids)} prompts '
                               f'(allowed; tables will show this n)'))

    no_messages = [r.get('object_id') for r in rows if not r.get('messages')]
    if no_messages:
        findings.append((WARN, f'{len(no_messages)} rows lack prompt field "messages" (ground truth): voxel_iou_gt will be null'))

    if info['setting'] in BRICKGPT_SETTINGS:
        findings += check_brickgpt_fields(rows, info['setting'])

    unparseable = [(r.get('object_id'), err) for r in rows for _, err in [parse_build_checked(r.get('output'))] if err]
    if unparseable:
        findings.append((WARN, f'{len(unparseable)} outputs cannot be parsed (eval counts them invalid and leaves them '
                               f'out of the upstream cross-check), e.g. {unparseable[0][0]}: {unparseable[0][1]}'))
    builds = [(r.get('object_id'), parse_build_checked(r.get('output'))[0]) for r in rows]
    off_library = [(oid, b) for oid, b in builds if b is not None and sizes_not_in_library(b)]
    if off_library:
        would_crash = [oid for oid, b in off_library if not b.has_collisions() and not b.has_out_of_bounds_bricks()]
        sizes = sorted({s for _, b in off_library for s in sizes_not_in_library(b)})
        findings.append((WARN, f'{len(off_library)} builds use brick sizes not in the library ({", ".join(sizes)}): '
                               f'counted invalid; {len(would_crash)} of them are otherwise valid, would crash upstream '
                               f'evaluate, and are left out of the cross-check'))
    empty = [r.get('object_id') for r in rows if isinstance(r.get('output'), str) and not r['output'].strip()]
    if empty:
        findings.append((WARN, f'{len(empty)} empty outputs (counted invalid): {examples(empty)}'))
    return findings


def check_brickgpt_fields(rows: list, setting: str) -> list:
    """BrickGPT runs should carry n_regenerations and rejection_reasons, consistent with the setting's flags."""
    findings = []
    for field in ('n_regenerations', 'rejection_reasons'):
        missing = [r.get('object_id') for r in rows if field not in r]
        if missing:
            findings.append((WARN, f'{len(missing)} rows lack {field!r}: that metric will be null (not 0) for them'))
    bad_reasons = [r.get('object_id') for r in rows if 'rejection_reasons' in r and not isinstance(r['rejection_reasons'], dict)]
    if bad_reasons:
        findings.append((ERROR, f'rejection_reasons must be an object of counts, e.g. {{"collision": 3}}: {examples(bad_reasons)}'))
    rollbacks = [r['n_regenerations'] for r in rows if isinstance(r.get('n_regenerations'), int)]
    if setting in ('norsnorb', 'rs') and any(n > 0 for n in rollbacks):
        findings.append((ERROR, f'setting {setting} has rollback off (--max_regenerations 0) but some rows have '
                                f'n_regenerations > 0 (max {max(rollbacks)}): the run used the wrong flags or folder name'))
    if setting == 'full' and rollbacks and max(rollbacks) > SETTINGS['full']['max_regenerations']:
        findings.append((WARN, f'n_regenerations up to {max(rollbacks)} exceeds the default limit 100; '
                               f'record max_regenerations in run_meta.json'))
    return findings


def check_meta(run_dir: Path) -> tuple[list, dict]:
    """run_meta.json is optional but feeds GPU/time in captions; missing fields are warnings."""
    path = run_dir / 'run_meta.json'
    if not path.exists():
        return [(WARN, 'no run_meta.json: setting/seed come from the folder name; GPU and timing will show n/a')], {}
    try:
        meta = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        return [(ERROR, f'run_meta.json is not valid JSON: {err}')], {}
    findings = []
    missing = [f for f in META_FIELDS if f not in meta]
    if missing:
        findings.append((WARN, f'run_meta.json lacks recommended fields: {", ".join(missing)}'))
    unknown = [f for f in META_FIELDS if f in meta and meta[f] is None]
    if unknown:
        findings.append((WARN, f'run_meta.json has unknown (null) fields: {", ".join(unknown)} (shown as n/a in the report)'))
    if meta.get('partial'):
        findings.append((WARN, 'run_meta.json marks this run as partial (tables would label its n as partial)'))
    if meta.get('exclude_from_tables'):
        findings.append((WARN, f'run_meta.json excludes this run from tables on purpose: {meta["exclude_from_tables"]}'))
    if meta.get('exit_status') not in (None, 0):
        findings.append((WARN, f'run_meta.json exit_status={meta["exit_status"]}: the run did not finish cleanly'))
    return findings or [(OK, 'run_meta.json has all recommended fields')], meta


def check_run(run_dir: Path) -> list:
    """All findings for one run folder, in reading order."""
    if not run_dir.is_dir():
        return [(ERROR, f'{run_dir} is not a directory')]
    meta_findings, meta = check_meta(run_dir)
    name_findings, info = check_name(run_dir, meta)
    findings = name_findings + meta_findings
    load_findings, rows = load_rows(run_dir)
    findings += load_findings
    if rows is not None and info:
        findings += check_rows(rows, info)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    findings = check_run(args.run)
    for level, message in findings:
        print(f'{level:5}  {message}')
    n_errors = sum(level == ERROR for level, _ in findings)
    print(f'\n{args.run}: {"FAIL, " + str(n_errors) + " error(s)" if n_errors else "format OK"} '
          f'({sum(level == WARN for level, _ in findings)} warning(s))')
    return 1 if n_errors else 0


if __name__ == '__main__':
    sys.exit(main())

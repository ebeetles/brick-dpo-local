"""Environment check (SPEC §5). Prints a PASS/FAIL/WARN table; exits non-zero on any FAIL.

Scope (Elwin, evaluation side): Python/pins, Gurobi size, bpy (WARN), disk.
HF access and GPU are reported as WARN only, since evaluation does not need them.

Run:  make env-check
"""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

from midway.config import BRICKGPT_SHA, UPSTREAM_DIR, UPSTREAM_SHA

PASS, FAIL, WARN = 'PASS', 'FAIL', 'WARN'

GUROBI_LICENSE_HELP = (
    'The pip Gurobi license is size-limited (2,000 vars / 2,000 constraints), too small for real builds. '
    'Get a free Academic WLS license at https://www.gurobi.com/academia/academic-program-and-licenses/ , '
    'download gurobi.lic, and save it as ~/gurobi.lic (or set GRB_LICENSE_FILE=/path/to/gurobi.lic).'
)


def check_python() -> tuple[str, str]:
    """Python must be 3.11.x (bpy 4.4 requires it)."""
    version = '.'.join(map(str, sys.version_info[:3]))
    return (PASS if sys.version_info[:2] == (3, 11) else FAIL), version


def check_upstream_sha() -> tuple[str, str]:
    """third_party/BrickGPT-experiments must be checked out at the pinned commit."""
    try:
        sha = subprocess.run(['git', '-C', str(UPSTREAM_DIR), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        return FAIL, f'cannot read upstream SHA: {err}'
    return (PASS if sha == UPSTREAM_SHA else FAIL), sha


def check_brickgpt_sha() -> tuple[str, str]:
    """The installed brickgpt package must come from the commit locked in upstream uv.lock."""
    try:
        direct_url = json.loads(importlib.metadata.distribution('brickgpt').read_text('direct_url.json'))
    except (importlib.metadata.PackageNotFoundError, TypeError) as err:
        return FAIL, f'brickgpt not installed: {err}'
    sha = direct_url.get('vcs_info', {}).get('commit_id', '?')
    return (PASS if sha == BRICKGPT_SHA else FAIL), sha


def check_hf_token() -> tuple[str, str]:
    """HF_TOKEN is needed only for generation (Megha's runs), so this is WARN-only here."""
    return (PASS, 'HF_TOKEN set (access not tested)') if os.environ.get('HF_TOKEN') else \
        (WARN, 'HF_TOKEN not set; only needed for generation')


def check_gurobi(n: int = 3000) -> tuple[str, str]:
    """Build and solve a dummy LP with n variables and n constraints (pip license caps at 2,000)."""
    try:
        import gurobipy as gp
    except ImportError as err:
        return FAIL, f'gurobipy not importable: {err}'
    try:
        with gp.Env(params={'OutputFlag': 0}) as env, gp.Model(env=env) as model:
            x = model.addVars(n, lb=0, ub=1)
            model.addConstrs(x[i] + x[(i + 1) % n] >= 0.5 for i in range(n))
            model.setObjective(x.sum())
            model.optimize()
            ok = model.Status == gp.GRB.OPTIMAL
            return (PASS if ok else FAIL), f'{n}x{n} LP status={model.Status}'
    except gp.GurobiError as err:
        return FAIL, f'{err}. {GUROBI_LICENSE_HELP}'


def check_gpu() -> tuple[str, str]:
    """GPU is not needed for the fixture or per-sample eval, so absence is WARN."""
    try:
        import torch
    except ImportError as err:
        return WARN, f'torch not importable: {err}'
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return PASS, f'{torch.cuda.get_device_name(0)}, {free / 2**30:.1f}/{total / 2**30:.1f} GiB free'
    if torch.backends.mps.is_available():
        return WARN, 'no CUDA (Apple MPS available); upstream CLIP eval hard-codes cuda'
    return WARN, 'no CUDA GPU'


def check_bpy() -> tuple[str, str]:
    """Rendering (P2) needs `import bpy` to work. WARN only."""
    try:
        import bpy
    except Exception as err:  # bpy can fail with non-ImportError errors on bad installs
        return WARN, f'import bpy failed: {err}'
    return PASS, f'bpy {bpy.app.version_string}'


def check_render_assets() -> tuple[str, str]:
    """ImportLDraw files must be real files (not Git LFS stubs) and an LDraw parts library must exist."""
    import_ldraw = UPSTREAM_DIR / 'ImportLDraw'
    if not (import_ldraw / '__init__.py').exists():
        return WARN, 'ImportLDraw submodule missing (git submodule update)'
    stubs = [p for p in import_ldraw.rglob('*') if p.is_file() and p.stat().st_size < 200
             and p.read_bytes().startswith(b'version https://git-lfs')]
    if stubs:
        return WARN, f'{len(stubs)} Git LFS pointer stubs in ImportLDraw (run git lfs pull)'
    ldraw = os.environ.get('LDRAW_LIBRARY_PATH') or os.path.expanduser('~/ldraw')
    if not os.path.isdir(os.path.join(ldraw, 'parts')):
        return WARN, f'no LDraw parts library at {ldraw} (needed only for rendering)'
    return PASS, f'ImportLDraw OK, LDraw library at {ldraw}'


def check_disk(min_gb: float = 20) -> tuple[str, str]:
    """At least min_gb GB free on the repo's disk."""
    free_gb = shutil.disk_usage(UPSTREAM_DIR.parent).free / 1e9
    return (PASS if free_gb >= min_gb else FAIL), f'{free_gb:.0f} GB free'


CHECKS = [
    ('Python 3.11', check_python),
    ('Upstream SHA', check_upstream_sha),
    ('brickgpt SHA', check_brickgpt_sha),
    ('HF token', check_hf_token),
    ('Gurobi 3000x3000', check_gurobi),
    ('GPU', check_gpu),
    ('bpy import', check_bpy),
    ('Render assets', check_render_assets),
    ('Disk >= 20 GB', check_disk),
]


def main() -> int:
    results = [(name, *fn()) for name, fn in CHECKS]
    width = max(len(name) for name, _, _ in results)
    for name, status, detail in results:
        print(f'{status:4}  {name:<{width}}  {detail}')
    n_fail = sum(status == FAIL for _, status, _ in results)
    print(f'\n{n_fail} FAIL(s).' if n_fail else '\nAll required checks passed.')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())

"""`make preview-builds`: voxel previews render, including unparseable and empty builds."""
import json
import os
import subprocess
import sys

from midway.config import REPO_ROOT


def test_preview_builds_writes_png(tmp_path):
    run = tmp_path / 'rs_subset3_s42'
    run.mkdir()
    gt = [{'role': 'assistant', 'content': '2x4 (0,0,0)\n2x4 (0,0,1)\n'}]
    rows = [{'object_id': oid, 'caption': f'a box {oid}', 'messages': gt, 'output': out, 'generation_time': 1.0}
            for oid, out in [('a', '2x4 (0,0,0)\n2x4 (1,0,0)\n'), ('b', '2x4 (0,0'), ('c', '')]]
    (run / 'generated_dataset.json').write_text(json.dumps(rows))
    proc = subprocess.run([sys.executable, '-m', 'midway.preview_builds', '--run', str(run), '--n', '3'],
                          cwd=REPO_ROOT, env=os.environ | {'PYTHONPATH': str(REPO_ROOT)}, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (run / 'preview_builds.png').stat().st_size > 10_000

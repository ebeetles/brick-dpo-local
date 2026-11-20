"""Run upstream `evaluate` (brickgpt_experiments.evaluate.main) unchanged.

Same CLI as `uv run evaluate`. Three machine-compatibility shims, none of which edits upstream files:
1. If `chamferdist` cannot be imported (it does not compile on macOS), a stub module is installed first.
   Upstream imports chamferdist at module load but uses it only for `--analyze_nearest_neighbors`,
   which we never run. The stub raises if called.
2. Upstream renders each build by spawning `uv run render_bricks.py` from the current directory. We set
   UV_PROJECT / UV_FROZEN / UV_NO_SYNC so that subprocess uses the pinned upstream environment as-is.
3. Upstream calls CLIP with device 'cuda' hard-coded. Without CUDA we pass 'cpu' instead (same model,
   float32 instead of float16; scores can differ in the last digits).

Example:  python -m midway.upstream_eval --generated_dataset_path runs/X/generated_dataset.json --analyze_stability
"""
import os
import sys
import types

from midway.config import UPSTREAM_DIR

try:
    import chamferdist  # noqa: F401
except ImportError:
    class _ChamferDistanceUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('chamferdist is not installed on this machine; nearest-neighbor analysis is disabled.')

    stub = types.ModuleType('chamferdist')
    stub.ChamferDistance = _ChamferDistanceUnavailable
    sys.modules['chamferdist'] = stub

os.environ.setdefault('UV_PROJECT', str(UPSTREAM_DIR))
os.environ.setdefault('UV_FROZEN', '1')
os.environ.setdefault('UV_NO_SYNC', '1')

import torch  # noqa: E402
import brickgpt_experiments.evaluate as upstream_evaluate  # noqa: E402

if not torch.cuda.is_available():
    _upstream_clipeval = upstream_evaluate.clipeval
    upstream_evaluate.clipeval = lambda image_paths, captions, device: _upstream_clipeval(image_paths, captions, 'cpu')

if __name__ == '__main__':
    upstream_evaluate.main()

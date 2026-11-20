# Handing over generation runs (for Megha and Ivan)

Elwin's evaluation picks up any run folder in the format below. Before sending a run, validate it in seconds
(no Gurobi, no GPU):

```bash
make check-run RUN=runs/rs_full250_s42
```

`ERROR` lines must be fixed; `WARN` lines are fine but tell you what will be null, partial or invalid.
Then send the whole folder (zip it, including `generations/`). Elwin drops it into `runs/` and runs `make preview`.

## Folder layout

```
runs/
  {setting}_{prompts}_s{seed}/          e.g. rs_full250_s42, norsnorb_full250_s42, full_subset50_s42
    generated_dataset.json              upstream's output; written only when every prompt has finished
    generations/{object_id}.json        upstream's per-prompt cache; ALWAYS include it (lets us read unfinished runs)
    run_meta.json                       recommended (GPU, timing, flags)
```

The folder name is how eval knows what the run is. Use exactly these values:

| `setting` | Table 1 column | upstream flags |
|---|---|---|
| `norsnorb` | No RS, no RB | `--max_brick_rejections 0 --max_regenerations 0` |
| `rs` | RS only (Stable@1) | `--max_regenerations 0` |
| `full` | RS + RB | defaults (`max_brick_rejections=500`, `max_regenerations=100`) |
| `gt` | (fixture) | `--generation_method ground_truth` |

| `prompts` | file to pass as upstream `--eval_dataset_path` |
|---|---|
| `full250` | default (`src/brickgpt_experiments/data/eval_dataset.json` in upstream) |
| `subset50` | `runs/prompts/subset50.json` (committed; never regenerate it) |

`seed`: upstream `--seed`, default 42. Later seeds: 43, 44.

**Unfinished runs are fine.** If a Colab session dies after 180 prompts, send the folder with just `generations/`.
Do not hand-write a `generated_dataset.json`. Eval reads `generations/`, and tables show `n=180 of 250` for that
column, with CIs computed from those 180 prompts. Upstream resumes from `generations/` if you rerun into the same folder.

## What each row must contain

Upstream writes these automatically; just don't strip anything.

| field | required | notes |
|---|---|---|
| prompt fields (`object_id`, `category_id`, `caption`, `messages`, ...) | yes | `messages[-1].content` is the ground truth, used for voxel IoU |
| `output` | yes | build text, one `HxW (x,y,z)` brick per line |
| `generation_time` | yes | seconds |
| `n_regenerations` | BrickGPT runs | rollbacks used; must be 0 for `norsnorb` and `rs` (check-run errors otherwise) |
| `rejection_reasons` | BrickGPT runs | object of counts, e.g. `{"collision": 3}` |

Missing optional fields become null (shown as `\tbd`), never 0. Unparseable outputs never crash eval: they are
counted invalid and listed in RESULTS.md.

## run_meta.json

All fields optional, but these appear in the report (GPU in the Table 1 caption, timing in the compute section):

```json
{
  "setting": "rs",
  "flags": "--max_regenerations 0",
  "seed": 42,
  "prompts": "full250",
  "upstream_sha": "7c3cce4b17b21d0cff75b750ce2227f32586b00b",
  "brickgpt_sha": "59cca62b49363f3a18e01528a327f152df91bfea",
  "hostname": "ip-10-0-0-1",
  "gpu": "NVIDIA A10G",
  "instance_type": "g5.2xlarge",
  "n_workers": 4,
  "start_time": "2026-11-20T18:00:00+00:00",
  "end_time": "2026-11-20T19:30:00+00:00",
  "exit_status": 0
}
```

`setting`, `prompts` and `seed` must agree with the folder name (check-run errors if not). If you change
`--max_regenerations` from the default, add `"max_regenerations": N` (used for the "hit rollback limit" metric).

## Setup gotchas (upstream `uv sync` does not work as written)

`make setup` handles all of this. To do it by hand in `third_party/BrickGPT-experiments`:

1. **chamferdist (every platform).** Its `setup.py` imports torch without declaring it, so `uv sync` fails while
   building it. Install everything else first, then build it against the installed torch:
   ```bash
   uv sync --frozen --no-install-package chamferdist
   ```
   ```bash
   uv pip install --no-build-isolation --no-deps chamferdist==1.0.3
   ```
   It is only used for upstream's nearest-neighbor analysis. If it still fails to build, generation is unaffected.
2. **CUDA-only packages (macOS only).** `onnxruntime-gpu`, `xformers`, `nvdiffrast` and `hy3dgen` have no macOS wheels.
   Add `--no-install-package X` for each to the `uv sync` line. They are only used by the mesh baselines.
3. Always use `--frozen` (never `uv lock --upgrade`) so the pinned versions stay pinned. Python is 3.11 (bpy needs it).
4. `HF_TOKEN` with approved access to `meta-llama/Llama-3.2-1B-Instruct` is needed for BrickGPT generation.
5. **Gurobi:** generation calls it only when rollback is on (the `full` setting); `norsnorb` and `rs` never do
   (`brickgpt/models/brickgpt.py:144`). The free pip license is too small for any real build (0 of the 250
   ground-truth builds fit), so `full` runs need an Academic WLS license (`~/gurobi.lic` or `GRB_LICENSE_FILE`).

## The 'lego' / 'bricks' issue (blocks BrickGPT generation)

Found by reading the code at the pinned commits; nobody has run it yet. Upstream `generate.py` does
`y['output'] = y['lego'].to_txt()`, but `BrickGPT.__call__` returns the structure under the key `'bricks'`. So
`uv run generate --generation_method brickgpt` should fail with `KeyError: 'lego'` on the first prompt.
The ground-truth path is unaffected.

Fix without editing upstream (Megha owns this, SPEC §6.1): build upstream's `BrickGenerator` yourself with a
generator function that renames the key and passes everything else (`rejection_reasons`, `n_regenerations`)
through. Untested sketch:

```python
class RenameBricksToLego:  # top-level class, so upstream's worker processes can pickle it
    def __init__(self, brickgpt_cfg):
        from brickgpt_experiments.models import BrickGPTWrapper
        self.inner = BrickGPTWrapper(brickgpt_cfg)

    def __call__(self, x: dict) -> dict:
        y = self.inner(x)
        y['lego'] = y.pop('bricks')
        return y

# BrickGenerator(RenameBricksToLego(brickgpt_cfg), generator_cfg)()   # from brickgpt_experiments.generate
```

The output format is unchanged, so eval needs nothing extra. Please note in the run's `run_meta.json`
(e.g. `"shim": "bricks->lego rename"`) that the shim was used.

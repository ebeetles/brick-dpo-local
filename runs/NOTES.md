# Run notes and deviations (feeds the report)

## Decisions for the midway report (Elwin, 2026-11-20)

- **Stability convention: raw** (1 - stress). Upstream's stress map is capped at 1 by construction, so raw equals
  clipped and an unstable build has min. stability exactly 0. Stated in the Table 1 caption and RESULTS.md.
- **CLIP and DINO stay `\tbd` for the midway.** Why (for the report): upstream renders every build with Blender
  Cycles at 512 samples. On Elwin's laptop (Apple M3 Max) Cycles fell back to the CPU and large
  builds took 4-13 min each, so one 250-prompt run needs most of a day; upstream's fixed 8-process render pool also
  exhausted memory. DINO additionally needs ground-truth renders, which the authors did not release, so it would
  use our own GT renders and not be directly comparable to the paper. Details under "Rendering, CLIP, DINO" below.

## Environment (2026-11-20, Elwin's laptop: Apple M3 Max, macOS, no CUDA)

- Upstream `BrickGPT-experiments` @ `7c3cce4b`, `brickgpt` @ `59cca62b` (verified by `make env-check`).
- `uv sync --frozen` (pinned lockfile) **cannot complete as-is**:
  - `onnxruntime-gpu`, `xformers`, `nvdiffrast`, `hy3dgen` have no macOS wheels. They are used only by the
    mesh baselines (rembg / LGM / Hunyuan3D), not by generation-from-text or evaluation. On macOS `make setup`
    skips them with `--no-install-package`; every other package is installed at its locked version.
  - `chamferdist==1.0.3` imports torch in its `setup.py` without declaring it as a build dependency, so it fails
    under build isolation on **every** platform. `make setup` installs the rest first, then builds chamferdist with
    `--no-build-isolation`. On macOS that build also fails (C++/CUDA extension). chamferdist is only used by
    upstream's nearest-neighbor analysis, but `evaluate.py` imports it at module load, so our wrapper around
    upstream `evaluate` stubs the module only when it fails to import. No upstream file is edited.
- ImportLDraw @ `026a8c0d` has no LFS-tracked files, so `git lfs pull` is not needed. Rendering will need an
  LDraw parts library at `~/ldraw` (or `LDRAW_LIBRARY_PATH`); not installed yet.

## Gurobi (was a blocker at checkpoint §10.1; RESOLVED 2026-11-20 with an Academic WLS license at ~/gurobi.lic)

- The pip `gurobipy` ships a size-limited license (2,000 vars / 2,000 constraints, expires 2026-11-23).
- Measured: **0 of 250** ground-truth builds fit under it (smallest failing build: 11 bricks).
  Stability evaluation is impossible without an Academic WLS license (`~/gurobi.lic` or `GRB_LICENSE_FILE`).
- Correction (verified in `brickgpt/models/brickgpt.py:144`): generation calls Gurobi **only when rollback is on**
  (`max_regenerations > 0`, i.e. the RS+RB setting). No-RS/no-RB and RS-only generation never call it; rejection
  sampling only checks collisions and bounds. Whoever *evaluates* (Elwin) always needs the license.

## Ground-truth fixture `runs/gt_full250_s42` (2026-11-20)

- Upstream `generate --generation_method ground_truth` via `midway/run_fixture.py` (named so it does not clash
  with Megha's `run_generate.py`). 250 rows, ~1 min on 14 CPU cores including per-sample eval.
- 250/250 valid, 250/250 stable. Cross-check vs upstream `evaluate --analyze_stability --analyze_time`: n_total,
  n_valid, n_stable, mean_bricks (119.524) equal exactly; mean stress over valid builds equal to 1e-16.
- `min_stab_raw` (= 1 - max stress): min 0.3165, median 0.989, mean 0.9458; none negative; raw == clipped for
  every build. Upstream's stress map is in [0, 1] by construction, so 1 - stress is never negative and an
  unstable build gets min stability exactly 0. Convention choice: **[human, pending]**.

## Evaluation conventions and upstream quirks found while reading the code

- Upstream `analyze_stability` zips `imap_unordered` results with the dataset in input order, so its
  `valid`/`stable`/`invalid` object_id lists can be misaligned. Counts and means are unaffected; our cross-check
  compares only those.
- Upstream CLIP/DINO inputs treat a build as valid if non-empty and collision-free; they do not check
  out-of-bounds bricks, unlike the stability metric.
- Outputs upstream cannot parse (ill-formatted line, or z >= 20) would crash upstream `evaluate`. Our
  per-sample eval records them as invalid with `n_bricks = null`, and the cross-check then fails loudly.
- `hit_rb_limit` is null when rollback is off (max_regenerations 0), since otherwise every unstable RS-only build
  would count as hitting the limit.
- SPEC §9.4's simple cantilever (2x6 on a 2x2 support) is **stable** per upstream (stress 0.006, stud friction
  holds it, even with 4 bricks loaded on the overhang). The unstable test case is a 1x8 on a single 1x1 stud
  with an 18-high tower on the overhang (stress 1.0; 0.96 with 17).
- Upstream `analyze_time` prints a numpy "Mean of empty slice" warning when no row has `mesh_generation_time`.
  Harmless.

## Report artifacts (2026-11-20)

- Stability convention: **raw** (Elwin, checkpoint §10.2).
- Tables 1, 2 and 4 are `table*` (span both columns). Table 1 with three "paper / ours±CI" cells is ~0.6 in wider than a
  3.3 in column at 8 pt; Tables 2 and 4 have 8 metric columns. Table 3 fits one column.
- Only run sets whose runs are complete (`generated_dataset.json` present) and passed the upstream cross-check are used
  in tables/figures. Others are listed in RESULTS.md as not reportable.
- Table 2/4 BrickGPT rows: Stable@1 from the RS-only run; all other columns from the RS+RB run.
- Shape statistics (#bricks, IoU, height, footprint, % 1x1) are over all parsed builds, not only valid ones.
- If a setting has several prompt sets, tables use full250 if present, else the largest subset; the column header says which.

## Rendering, CLIP, DINO (P2, started 2026-11-20 13:19, timebox ~1 h)

- LDraw parts library: `complete.zip` (147 MB, library.ldraw.org, last-modified 2026-11-09) extracted to `~/ldraw`.
- **Upstream rendering is broken at the pinned ImportLDraw commit.** `026a8c0d` is literally "Delete loadldraw/background.exr",
  but `loadldraw.py` still loads that file for the realistic look. We restored it from the submodule's own history
  (`git show 78d110b:loadldraw/background.exr`); its sha256 `83f78d60…` and size 61,727,090 bytes match the LFS pointer
  in `f13e97a`. It sits in `third_party/.../ImportLDraw/loadldraw/` as an **untracked** file; no tracked upstream file is modified.
  The authors presumably had it left over in their working copy.
- `midway/upstream_eval.py` sets `UV_PROJECT/UV_FROZEN/UV_NO_SYNC` so upstream's per-image `uv run render_bricks.py`
  subprocess uses the pinned env, and passes `cpu` to CLIP when there is no CUDA (upstream hard-codes `cuda`).
  CLIP on CPU is float32 vs float16 on CUDA, so scores may differ from a CUDA run in the last digits.
- Render cost on the M3 Max (Cycles, Metal GPU, 512 samples, 512x512, upstream settings unchanged): ~48 s for one
  92-brick build alone; with upstream's 8 parallel workers sharing one GPU see throughput below.
- DINO uses **our own** ground-truth renders (`runs/gt_full250_s42/renders`) because upstream's default
  `--ground_truth_render_dir` is on the authors' machine. Our DINO may therefore not match the paper's exactly.
- Pipeline check on one build (scratch run): CLIP 0.358; DINO of a render against itself 1.000 (sanity).
- **Throughput:** upstream's fixed 8-process render pool filled swap on this 36 GB Mac (20.3 of 21.5 GB used) and
  stalled: 1 image in 30 min. Killed it at 13:53. `midway/render.py` calls the same upstream `_render_bricks` with a
  configurable worker count (default 1): ~45-50 s per image, so 250 images take ~3.3 h per run.
- **Timebox reached: CLIP and DINO are `\tbd`.** After the first few images, renders slowed to 4-13 min each:
  Blender ran at ~1200% CPU, i.e. Cycles fell back to the **CPU** even though upstream requests Metal (cause not
  investigated). 250 images would take most of a day, so the render was stopped at 14:20 with 9/250 GT images done
  (kept; `make render` resumes). Rendering each BrickGPT run on this Mac would
  take another ~3.3 h per run; a CUDA machine (Megha's) is likely much faster. Once the GT renders and a run's renders
  exist, `make render RUN=runs/<name>` followed by `make report` fills CLIP/DINO with per-sample CIs.

## Robustness changes before teammates' runs (2026-11-20)

- Unparseable outputs (ill-formatted line, brick with z >= 20, brick size not in the brick library, missing output)
  get `valid=False` and a `parse_error` reason in per_sample.jsonl; evaluation never crashes on them. Upstream
  `evaluate` would crash on all four, so the cross-check runs upstream on the **parseable rows only** (in
  `runs/<name>/cross_check/`) and `cross_check.json` records `n_excluded` and each excluded object_id + reason.
  A run whose parseable subset matches stays in the tables; the exclusions are listed at the top of RESULTS.md.
- Partial runs (only `generations/`, fewer prompts than the prompt set) are evaluated, cross-checked and allowed
  into tables. Column headers show the actual n ("n=180 of 250"); CIs bootstrap over those n prompts.
- A run is excluded from tables only if its cross-check mismatched, could not run, was skipped, or its
  per_sample.jsonl is stale. Every exclusion and every never-evaluated run dir is printed in a `!!!` block by
  `make report` and listed under "Attention" in RESULTS.md. A failed `make eval` no longer stops `make report`.

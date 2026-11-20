# Run notes and deviations (feeds the report)

## Decisions for the midway report (Elwin, 2026-11-20)

- **Stability convention: raw** (1 - stress). Upstream's stress map is capped at 1 by construction, so raw equals
  clipped and an unstable build has min. stability exactly 0. Stated in the Table 1 caption and RESULTS.md.
- **CLIP and DINO stay `\tbd` for the midway.** Why (for the report): upstream renders every build with Blender
  Cycles at 512 samples. On Elwin's laptop (Apple M3 Max) Cycles fell back to the CPU and large
  builds took 4-13 min each, so one 250-prompt run needs most of a day; upstream's fixed 8-process render pool also
  exhausted memory. DINO additionally needs ground-truth renders, which the authors did not release, so it would
  use our own GT renders and not be directly comparable to the paper. Details under "Rendering, CLIP, DINO" below.

## Seeding: upstream resets the seed before every prompt (for the report)

- Upstream `generate.py` (`BrickGenerator._generate_bricks`, line 104) calls `transformers.set_seed(seed)` before
  generating **each** prompt, with the same seed (default 42) for every prompt. So every prompt starts sampling from
  the same RNG state, rather than one random stream running through the whole prompt set.
- Consequences for the report:
  - Near-identical prompts can produce identical builds. Observed in `norsnorb_first100_s42`: four paraphrases of
    "rectangular table with a flat top and four legs" gave the same 61-brick build.
  - Samples across prompts are not fully independent. Our bootstrap CIs resample prompts as if independent, so
    they may be slightly too narrow when many prompts are near-duplicates.
  - A "seed" is a property of the whole run: seeds 42/43/44 give three samples per prompt, and our cluster
    bootstrap keeps a prompt's seeds together.
  - Our runs use upstream's generator unchanged, so this matches how the paper's numbers were produced (assuming
    the authors used this code).

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

## norsnorb_first100_s42 (Ivan's Colab No RS/no RB run, first 100 prompts; evaluated 2026-11-20)

- Pipeline test, **kept out of tables** (`exclude_from_tables` in its run_meta.json); Ivan's full 250-prompt run
  replaces it. run_meta.json written by the eval side with only known facts: seed 42 is assumed (not recorded),
  GPU unconfirmed (probably a Colab T4), all other provenance fields null.
- Prompts are the first 100 of eval_dataset.json in order (verified), **not a random sample**. Category mix is close
  to full250 (tables 28% vs 32%, chairs 18% vs 20%), but these prompts' ground-truth builds are smaller on average
  (110.8 vs 119.5 bricks), and validity falls with build size (GT >= 140 bricks: 4/20 valid, 0 stable), so this run
  may slightly overstate validity relative to the full 250.
- Results: valid 34.0% [25.0, 43.0], stable 13.0% [7.0, 20.0] (n=100); paper 37.2% / 12.8%, both inside our CIs.
  Mean stab. 0.968 [0.953, 0.980] vs 0.956; min. stab. 0.344 [0.197, 0.504] vs 0.325 (n=34 valid builds).
  All 66 invalid builds collide; 3 of them also use a brick size not in the library (3x1, 3x1, 4x6). No
  out-of-bounds or empty builds. Rollbacks and rejections are 0 everywhere, consistent with the flags.
- Time: median 26.4 s/build [23.0, 29.5], mean 34.5 s (6.3-119.7 s); correlation with brick count 0.998, i.e.
  time is token generation. Not comparable to the paper's 40.8 s (different setting and hardware).
- Cross-check vs upstream evaluate: MATCH on 100/100 rows (the 3 off-library builds also collide, so upstream
  rejects them as invalid before its stability solver and does not crash).
- Identical outputs for 4 paraphrased "rectangular table with four legs" prompts: upstream re-seeds (seed 42)
  before every prompt, so near-identical prompts follow the same sampling path. Not a bug, but samples are not
  independent across similar prompts.
- What this run changed in the pipeline: `firstN` prompt sets; `invalid_reason` per build; a brick size not in the
  library is now an invalid reason, not a parse error (rows are left out of the cross-check only if upstream would
  really crash); `partial` / `exclude_from_tables` in run_meta.json; check-run warns on null provenance fields;
  run dirs with an unrecognized name are listed loudly instead of being skipped with one line.

## Validity: our rule vs upstream's rule (2026-11-20)

- Every per-sample record has `valid_upstream` (upstream's rule: parsed, non-empty, no collisions, no out-of-bounds;
  ignores the brick library) and `valid` (ours: the same AND every brick size is in the library). All reported
  metrics use `valid`. The cross-check compares upstream's counts with `valid_upstream`.
- Builds that differ only by brick type (`valid_upstream` but not `valid`) make upstream's stability solver raise,
  so they are left out of the cross-check; their count is `n_brick_type_only` in cross_check.json and RESULTS.md.
  `norsnorb_first100_s42` has 0 (its 3 off-library builds also collide).
- Rollbacks print as "n/a (RB off)" in Table 1 and RESULTS.md for the No RS/no RB and RS-only settings.
- `make preview-builds RUN=... N=5` saves voxel plots of N random builds (seed 0) beside their ground truth
  (runs/<name>/preview_builds.png), a quick visual check while Blender rendering is unavailable. Only surface
  collisions are visible (red); interior collisions are hidden.

## norsnorb_full250_s42: the No RS/no RB column of Table 1 (Ivan, evaluated 2026-11-20)

- Source: Ivan's outputs.zip (`norsnorb_250_s42`), moved to `runs/norsnorb_full250_s42/`; Ivan's own metadata kept as
  `run_meta_ivan.json`, his log as `generation.log`. **In Table 1.** `norsnorb_first100_s42` is superseded and excluded.
- **Provenance differs from SPEC; state it in the report's replication paragraph:**
  - Generated with Ivan's `brickrl/run_baseline.py` calling `brickgpt.models.BrickGPT` directly on **Kaggle**
    (Python 3.13, T4), not with upstream `generate.py` in the pinned environment. `brickgpt`/`transformers` versions
    are unknown; the log shows two crashes from a newer `transformers` before the completed session.
  - Base model loaded from the ungated mirror `unsloth/Llama-3.2-1B-Instruct` (not `meta-llama/Llama-3.2-1B-Instruct`);
    the model was loaded from a local merged checkpoint `/kaggle/working/brickgpt_merged`.
  - `transformers` warned that this checkpoint's tokenizer has an incorrect regex pattern. **To confirm with Ivan**
    whether tokenization matches the original tokenizer.
  - Seed 42 per prompt, flags `--max_brick_rejections 0 --max_regenerations 0` (from his metadata). 20 of 250 builds
    came from an earlier session's cache.
- Results (n=250): valid 36.4% [30.4, 42.4], stable 12.8% [8.8, 17.2], mean stab. 0.960 [0.948, 0.970], min. stab.
  0.322 [0.230, 0.416] (n=91 valid). Paper: 37.2 / 12.8 / 0.956 / 0.325; all inside our CIs.
  Invalid: collision 152, collision + brick not in library 4, collision + out of bounds 1, out of bounds 2.
  Time: median 26.0 s/build [23.8, 29.5], mean 35.7 s (6.2-291.4 s), corr. with brick count 0.992.
  4 builds exceed the largest ground truth (367 bricks), up to 701 bricks (all collide).
- Cross-check vs upstream evaluate: MATCH on 250/250 rows.
- Overlap with first100: all 100 outputs identical, **and so are all 100 generation_times** (to full precision), so
  first100 was cut from the same records, not re-generated. This confirms consistency, not reproducibility.

## DPO smoke test (Ivan; `runs/dpo_smoke_test/`, excluded)

- Pipeline check only: 30 steps on synthetic pairs, lr 5e-5, not a result. Contents: three `log_history.json` (DPO-seq,
  DPO-prefix, DPO-local m=5) and a loss/reward-margin plot. No generated builds, so nothing for the eval pipeline.
  Not a run folder (no generated_dataset.json, name not a run name), so it cannot reach any table, figure or RESULTS.md.
- **For Ivan:** DPO-seq and DPO-prefix have identical losses and reward margins at every logged step (train_loss 0.5856
  both); their log-probs differ by the same 28.947 for chosen and rejected. With pairs that share a prefix, the
  prefix's log-probs cancel in the DPO loss, so the two variants are mathematically identical unless DPO-prefix
  changes something else (pairs, weighting, or reference model). Worth resolving before Table 2 has both rows.

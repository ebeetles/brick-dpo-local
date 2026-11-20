# SPEC: Midway pipeline for "Bricks That Don't Fall Over" (10-423/623/723, Fall 2026)

**Owner:** the team (Megha: generation runs · Elwin: evaluation, tables, plots · Ivan: training, later)
**Deadline:** Midway Executive Summary due **Fri Nov 20, 11:59pm ET**. Everything here exists to get real baseline numbers and skeleton tables/plots into that PDF.
**Builder:** Claude Code, working in the team repo. Humans review and run everything that matters (see §10).

---

## 1. Goal (read this first)

Build a small, reproducible pipeline that:

1. Generates text-to-brick builds with the **released BrickGPT** weights under three decoding settings.
2. Evaluates every build **per sample** (validity, stability, cost, shape/shortcut stats, alignment).
3. Aggregates with **bootstrap 95% confidence intervals**.
4. Emits **LaTeX tables and PDF plots** for the midway report: one filled baseline table, and labeled skeleton tables/plots for the experiments that don't exist yet.

The report needs (handout §2.3): baseline results on the task, skeleton tables/plots for future experiments, and actual compute used (GPU type, hours, dollars).

### Non-goals for tonight
- No DPO / RFT training, no preference-pair construction (Ivan's later milestone). One optional hook in §6.8.
- No retraining BrickGPT SFT.
- No writing of report prose (Sections 1–4, 6). Only the artifacts in §7.
- Do not modify upstream repos' files. Wrap or adapt instead.

### Hard rule: never fabricate results
Any number not produced by an actual run in `runs/` must render as `\tbd` (a gray dash) in tables and as an empty/"pending" element in plots. No placeholder numbers, no "expected" values in result cells. Paper numbers appear only in clearly labeled "paper" cells.

---

## 2. Priorities (do in this order; stop when time runs out)

| Pri | Item | Why |
|---|---|---|
| **P0** | Env check (§5) | Surfaces blockers (Llama access, Gurobi, GPU) immediately |
| **P0** | Ground-truth fixture run + per-sample eval (§6.2–6.4) | Tests the whole eval path without a GPU or model |
| **P0** | RS-only generation on 250 prompts + eval | This is **Stable@1**, the key baseline number (paper: 24.0%) |
| **P0** | Table 1 (replication) with whatever runs exist (§7.1) | Required baseline result |
| **P0** | Skeleton Tables 2–4 + Figures 2–4 (§7.2–7.3) | Required skeletons |
| **P1** | No-RS/no-RB generation on 250 + eval | Cheap second column |
| **P1** | Bootstrap CIs everywhere (§6.5) | Every number gets an interval |
| **P1** | Compute log → Section 7 numbers (§6.7) | Required compute section |
| **P2** | Full RS+RB on a **subset** (e.g. 50 prompts) | Slow; mean time per build may be far above the paper's 40.8s median |
| **P2** | Rendering → CLIP and DINO (§6.6) | Needs Blender-as-module; most likely to eat time |
| **P3** | Rollback logging hook (§6.8) | Free pair data for later, only if it provably changes nothing |

---

## 3. Background the builder needs

**Task.** Caption in, LEGO build out. A build is text, one brick per line, `HxW (x,y,z)`, e.g. `6x2 (12,18,0)`, on a 20×20×20 grid, bottom-to-top.

**Model.** BrickGPT = Llama-3.2-1B-Instruct + LoRA, released at `AvaLovelace/BrickGPT`. The base model `meta-llama/Llama-3.2-1B-Instruct` is **gated** on Hugging Face (each runner needs approved access + `HF_TOKEN`).

**Decoding knobs** (`brickgpt.models.BrickGPTConfig`; defaults read from code at the pinned commit):
- `use_logit_masking` (default True): grammar-constrains output format. Keep on in all settings.
- `max_brick_rejections` (default 500): rejection sampling (RS) of invalid bricks. 0 = off.
- `max_regenerations` (default 100): physics-aware rollback (RB). 0 = off.
- `use_gurobi` (default True): stability via Gurobi force-equilibrium solver. Must stay True.

| Setting (Table 1 column) | Flags |
|---|---|
| No RS, no RB | `--max_brick_rejections 0 --max_regenerations 0` |
| RS only (**Stable@1**) | `--max_regenerations 0` |
| RS + RB (full BrickGPT) | defaults |

Confirm these against the paper's ablation description and the code before running; note any doubt in `runs/<run>/NOTES.md`.

**Stability.** `BrickStructure.stability_scores()` returns a per-voxel **stress** map; a build is stable iff `max stress < 1`. The paper reports stability `s = 1 − stress`. See §6.3 for the convention check.

**Paper targets** (Pun et al., BrickGPT, Table 1; 250 prompts; stability/CLIP/DINO over valid builds):

| | No RS, no RB | RS only | RS + RB |
|---|---|---|---|
| % valid | 37.2 | 100 | 100 |
| % stable | 12.8 | 24.0 | 98.8 |
| Mean stab. | 0.956 | 0.947 | 0.996 |
| Min. stab. | 0.325 | 0.228 | 0.915 |
| CLIP | 0.329 | 0.322 | 0.324 |
| DINO | 0.888 | 0.882 | 0.880 |

Paper also reports a median of 2 rollbacks and 40.8s per build for the full setting (on their hardware).

---

## 4. Repo layout

```
<team-repo>/
  SPEC.md                         # this file
  AI_USE.md                       # running log of AI assistance (see §11)
  Makefile                        # targets in §8
  third_party/
    BrickGPT-experiments/         # git clone, pinned (below), with submodules + LFS
  midway/                         # our code (Python 3.11 package)
    __init__.py
    config.py                     # paths, run definitions, prices, seeds
    env_check.py
    make_subset.py
    run_generate.py
    adapters.py                   # compatibility shim, see §6.1
    shapes.py                     # shortcut stats + voxel IoU
    per_sample_eval.py
    stats.py                      # bootstrap CIs
    aggregate.py
    compute_log.py
    tables.py
    plots.py
  tests/
    test_shapes.py
    test_stability_fixtures.py
    test_stats.py
    test_tables.py
  runs/                           # gitignored; one dir per run
  report/
    tables/*.tex                  # \input{} into main.tex
    figures/*.pdf
```

**Pinned upstream** (record SHAs in `config.py` and in the report):
- `AvaLovelace1/BrickGPT-experiments` @ `7c3cce4b17b21d0cff75b750ce2227f32586b00b`
- Its `uv.lock` pins `brickgpt` @ `59cca62b49363f3a18e01528a327f152df91bfea`. Use the lockfile (`uv sync`, not `uv lock --upgrade`).

**Environment.** One environment: the upstream uv project (Python 3.11; `bpy` requires it). Run our code inside it, e.g. `uv run --project third_party/BrickGPT-experiments python -m midway.<module>` with the repo root on `PYTHONPATH` (wrap this in the Makefile). Setup:

```bash
git clone https://github.com/AvaLovelace1/BrickGPT-experiments third_party/BrickGPT-experiments
cd third_party/BrickGPT-experiments && git checkout 7c3cce4b17b21d0cff75b750ce2227f32586b00b
git submodule init && git submodule update
(cd ImportLDraw && git lfs pull)     # needed only for rendering
uv sync
```

---

## 5. `env_check.py` (P0)

Prints a PASS/FAIL/WARN table and exits non-zero on any P0 failure. Checks:

1. Python is 3.11.x; upstream repo is at the pinned SHA; `brickgpt` resolves to the locked commit.
2. **HF access:** `HF_TOKEN` set; can download `config.json` from `meta-llama/Llama-3.2-1B-Instruct` and from `AvaLovelace/BrickGPT`. On 401/403, print: "request access on the model page".
3. **Gurobi:** builds and solves a dummy LP with **3,000 variables and 3,000 constraints**. The free pip license is capped at 2,000/2,000, which is too small for real builds; on a size-limit error, print the Academic WLS instructions (`GRB_LICENSE_FILE` / `~/gurobi.lic`).
4. **GPU:** `torch.cuda.is_available()`, device name, free memory. WARN (not FAIL) if absent, since the fixture run and per-sample eval don't need it.
5. **Rendering (WARN only):** `import bpy` succeeds; ImportLDraw LFS files are real files, not pointer stubs.
6. Disk space ≥ 20 GB free.

---

## 6. Components

### 6.1 Compatibility shim (`adapters.py`) — read before running generation

As read from the code (not yet run): upstream `generate.py` does `y['lego'].to_txt()` on the generator's output, but `BrickGPT.__call__` at the locked commit returns the structure under the key **`'bricks'`**, not `'lego'`. If confirmed, `uv run generate --generation_method brickgpt` will fail with `KeyError: 'lego'`.

Fix **without editing upstream files**: in `run_generate.py`, construct upstream's `BrickGenerator` directly with a wrapped generator function that renames `'bricks'` → `'lego'` and passes everything else through (`rejection_reasons`, `n_regenerations`). The ground-truth path already uses `'lego'` and needs no shim. Add a test that runs the wrapper on a stub generator. Document this in `runs/<run>/NOTES.md` and the report's replication paragraph.

### 6.2 Prompt sets (`make_subset.py`)

- `full250`: upstream `src/brickgpt_experiments/data/eval_dataset.json` (250 prompts; fields `design_name, category_id, object_id, caption, is_stable, messages`; `messages[-1]['content']` is the ground-truth build; all 250 have `is_stable == 1`).
- `subsetN`: N prompts sampled from `full250` with `random.Random(0)`, written to `runs/prompts/subset{N}.json` with the same schema. Print the category counts. Record the object_ids.
- Never resample a subset after results exist; subsets are part of the experiment definition.

### 6.3 Generation runner (`run_generate.py`)

CLI: `--setting {gt,norsnorb,rs,full} --prompts {full250,subsetN} --seed INT --n_workers INT --out runs/<name>`.

- `gt`: upstream `--generation_method ground_truth` (CPU only; no model). This is the **fixture** for the whole eval pipeline.
- Others: BrickGPT with the flags in §3, via the §6.1 shim. Default seed 42 (upstream default). Seeds for later: 42, 43, 44.
- Run name convention: `runs/{setting}_{prompts}_s{seed}/`.
- Upstream caches one JSON per object under `generations/`, so reruns **resume**. Keep that. Long runs go under `nohup`/`tmux`; print progress.
- Write `runs/<name>/run_meta.json`: setting, flags, seed, prompt set, upstream SHAs, hostname, GPU name, instance type (if `INSTANCE_TYPE` env var set), `n_workers`, start/end wall-clock timestamps, exit status.
- Output is upstream's `generated_dataset.json`: each row = prompt fields + `output` (build text), `generation_time` (seconds), and for BrickGPT `n_regenerations`, `rejection_reasons`.
- Start with `n_workers=1` on a 20-prompt pilot, measure seconds/build and GPU memory, then raise workers (several 1B copies fit on a 24 GB GPU). Log the pilot timings in `runs/pilot/NOTES.md`.

### 6.4 Per-sample evaluation (`per_sample_eval.py`) — core of Elwin's part

Input: a run dir. Output: `runs/<name>/per_sample.jsonl`, one record per prompt, plus a run-level cross-check.

Per record (missing values are `null`, never 0):

| Field | Definition |
|---|---|
| `object_id`, `category_id`, `caption`, `seed`, `setting` | copied |
| `n_bricks` | number of bricks parsed |
| `valid` | non-empty AND no collisions AND no out-of-bounds bricks (same rule as upstream `Evaluator._analyze_stability`) |
| `max_stress` | `stability_scores().max()` if valid |
| `mean_stress` | mean of `stability_scores()` over occupied voxels if valid |
| `stable` | valid AND `max_stress < 1.0` |
| `min_stab_raw`, `mean_stab_raw` | `1 − max_stress`, `1 − mean_stress` |
| `min_stab_clip`, `mean_stab_clip` | same, clipped to [0, 1] |
| `height` | top layer index + 1 |
| `footprint` | number of occupied (x, y) cells in the top-down projection |
| `frac_1x1` | share of bricks that are 1×1 |
| `voxel_iou_gt` | IoU of 20³ occupancy grids vs the prompt's ground-truth build (`messages[-1]['content']`) |
| `generation_time` | seconds, from the run |
| `n_regenerations` | rollbacks used (BrickGPT runs only) |
| `n_rejections` | total of `rejection_reasons` |
| `hit_rb_limit` | `n_regenerations == max_regenerations` AND not stable |

Use `brickgpt.data.BrickStructure` (`from_txt`, `has_collisions`, `has_out_of_bounds_bricks`, `stability_scores`, `voxel_occupancy`) rather than reimplementing physics. Parallelize with a process pool (Gurobi is CPU-bound).

**Mandatory cross-check:** also run upstream `uv run evaluate --generated_dataset_path ... --analyze_stability --analyze_time` and assert our `n_valid`, `n_stable`, and mean brick counts equal upstream's `stability_analysis.json` / `time_analysis.json` **exactly**. Fail loudly on mismatch.

**Stability-convention check (flag for humans, don't decide silently):** on the ground-truth fixture, expect 100% valid and 100% stable (all 250 are labeled stable). Print the distribution of `min_stab_raw`. Paper columns are in [0, 1] even for unstable builds, which suggests clipping, but this is unverified. Report both raw and clipped in the aggregate; the team picks one for the report and states it.

### 6.5 Statistics (`stats.py`)

- `bootstrap_ci(values, n=10_000, alpha=0.05, rng_seed=0)` → (mean, lo, hi), percentile method. Ignores `null`s; reports `n` used.
- **Cluster bootstrap by prompt** when a run set has multiple seeds: resample prompts, carry all of a prompt's seeds together.
- `paired_diff_ci(a, b)`: paired by `object_id` (+ seed), for model-vs-model comparisons later.
- Rates (valid, stable) are means of 0/1. Mean/min stability, CLIP, DINO are over **valid** builds only (paper convention).
- Tests: CI of a constant vector is degenerate; CI of Bernoulli(0.24), n=250, has half-width ≈ 5.3 points (±1).

### 6.6 Rendering, CLIP, DINO (P2)

- Use upstream: `uv run evaluate --render_generated_bricks --analyze_clip --analyze_img_similarity` (CLIP runs on CUDA).
- **DINO needs ground-truth renders.** Upstream's default `--ground_truth_render_dir` is a path on the authors' machine and isn't released. Workaround: render the `gt` fixture run (`runs/gt_full250_s42/renders/{object_id}.png`) and pass that directory as `--ground_truth_render_dir`. Renders are named by `object_id`, which is how upstream matches them. Note in the report that our DINO uses our own ground-truth renders, so it may not match the paper's exactly.
- Join upstream's CLIP/DINO outputs into `per_sample.jsonl` per object if upstream exposes per-sample scores; otherwise record run-level means only and mark CIs as unavailable for those columns.
- If `bpy` won't install or render headless within ~1 hour of effort, stop, leave CLIP/DINO as `\tbd`, and write why in `runs/NOTES.md`.

### 6.7 Compute log (`compute_log.py`)

- From every `run_meta.json`: wall-clock hours × GPUs = GPU-hours, per run and total.
- Cost = GPU-hours × hourly price from `config.py` (`PRICE_PER_HOUR = {"g5.2xlarge": 1.212}`; **verify on the AWS on-demand pricing page** and record the date).
- Emit `report/tables/compute.tex` (run, setting, prompts, GPU, hours, $) and `runs/compute.json`. Include pilot and failed runs: the handout asks for actual use.

### 6.8 Optional rollback logging hook (P3, only if time remains)

Goal: when full RS+RB runs, also save each rollback as `(prefix P, discarded continuation y_l)` for later pair construction, at no extra solver cost.

- Implement as a subclass/monkeypatch of `BrickGPT._remove_all_bricks_after_first_unstable_brick` that records the input and output structures, then calls the original. **No behavior change.**
- Write to `runs/<name>/rollbacks/{object_id}.jsonl`.
- Acceptance: with a fixed seed on 5 prompts, outputs with and without the hook are byte-identical. If not, disable the hook and note it.

---

## 7. Report artifacts

All `.tex` files go in `report/tables/`, compile inside the course template (`twocolumn`, `\footnotesize`, `booktabs`), use the `\tbd` macro already defined in `main.tex`, and fit one column unless noted. Regenerating must be one command (`make report`). Each artifact gets a one-line `% generated by midway/tables.py from runs/...` header comment.

### 7.1 Table 1: replication (filled with real runs)
Rows: % valid, % stable, mean stab., min stab., CLIP, DINO, plus **rollbacks (median)** and **s/build (median)**. Columns: the three settings. Cell format `paper / ours`, where ours is the value with a short CI, e.g. `24.0 / 22.8\textsubscript{\,±5.2}`. Missing runs → `\tbd`. Footnote-able caption fields: prompt set (full250 or subsetN), seeds, GPU, stability convention.

### 7.2 Skeleton tables (all cells `\tbd` except baseline rows filled from Table 1 runs where they apply)
- **Table 2, main comparison (RQ2/RQ3):** rows BrickGPT (ours), RFT, DPO-seq, DPO-prefix (BrickDPO's method), DPO-local (ours). Columns: Stable@1, final % stable (RB on), rollbacks, s/build, CLIP, DINO, IoU, #bricks. Fill the BrickGPT row from real runs.
- **Table 3, window sweep (RQ3):** rows m ∈ {1, 5, 20} for DPO-local at one pair budget. Columns: Stable@1, rollbacks, #bricks, IoU.
- **Table 4, shape filter ablation (RQ4):** rows DPO-local with / without IoU filter, plus BrickGPT reference. Columns: Stable@1, CLIP, DINO, IoU, #bricks, height, footprint, % 1×1. Fill the BrickGPT reference row.

### 7.3 Figures (`report/figures/*.pdf`, matplotlib, single-column 3.3 in wide, ≥8 pt fonts, colorblind-safe palette, consistent model colors across figures)
- **Fig 2, stability vs. cost (RQ5):** x = median seconds/build, y = final % stable (RB on), one point per model with CI bars. Plot real BrickGPT points; other models appear in the legend as hollow markers labeled "pending", with no fake positions.
- **Fig 3, brick-count distributions:** one violin/box per model. Real for BrickGPT runs and the ground truth; empty slots labeled "pending" for the rest. This is the direct test of BrickDPO's build inflation.
- **Fig 4, Stable@1 vs. pair budget (RQ3):** x = pairs ∈ {500, 2k, 5k}, lines for DPO-seq, DPO-prefix, DPO-local; BrickGPT as a horizontal reference line if Stable@1 exists. All DPO lines pending.

### 7.4 Results digest
`runs/RESULTS.md`: auto-generated summary of every completed run (n, valid, stable, CIs, medians, gaps vs paper), so the team can write Section 5 prose quickly and check numbers against tables.

---

## 8. Makefile targets

```
make setup           # clone/pin upstream, submodules, uv sync
make env-check
make fixture         # gt run on full250 + per-sample eval + cross-check
make pilot           # rs on 20 prompts, n_workers=1, timing notes
make gen SETTING=rs PROMPTS=full250 SEED=42 WORKERS=?
make eval RUN=runs/<name>
make render RUN=runs/<name>      # P2
make report          # aggregate + tables + figures + compute + RESULTS.md
make test
```

---

## 9. Tests (`make test`, must pass before any number goes in the report)

Hand-built structures, each a few lines of brick text:
1. One 2×4 brick on the ground → valid, stable.
2. Two stacked, aligned 2×4 bricks → valid, stable.
3. A brick floating at z=2 with nothing below → upstream's verdict recorded (expect unstable).
4. A long cantilever overhang (e.g. 2×6 hanging mostly off a 2×2 support) → expect unstable; record stress.
5. Two overlapping bricks → invalid (collision).
6. A brick at x=19 with width 4 → invalid (out of bounds).
7. IoU: identical builds → 1.0; disjoint → 0.0; hand-computed partial overlap → exact value.
8. `frac_1x1`, `height`, `footprint` on a hand-computed build.
9. Shim test (§6.1) and bootstrap tests (§6.5).
10. Table generator: a run set with a missing run renders `\tbd`, never `0` or `nan`.

---

## 10. Human checkpoints (the builder must stop and ask at each)

1. **After `env-check`:** humans resolve any FAIL (access requests, license, GPU quota). Do not work around a gated model or license.
2. **After the fixture:** a human reads `per_sample_eval.py` end to end and confirms 100% valid/stable on ground truth and the stability convention (§6.4).
3. **After the pilot:** humans decide prompt counts and whether full RS+RB runs on full250 or a subset, based on measured seconds/build and remaining hours.
4. **After each real run:** a human opens 5 random builds (renders if available, otherwise text) and checks they look like the captions; compares numbers to the paper and writes 1–2 sentences on any gap.
5. **Before `make report` output goes into the PDF:** a human spot-checks 3 table cells against `per_sample.jsonl` by hand.

---

## 11. AI-use logging

The project handout allows AI assistance but requires disclosure of what used AI, which tools, how, what humans did, and how it was checked. Append to `AI_USE.md` as you go: date, component, what the assistant wrote, who reviewed it, which tests/checks verified it. Mark human-only work explicitly (decisions in §10, interpretation of results). This file feeds Section 6 of the report.

---

## 12. Definition of done for the midway

- [ ] `make test` passes.
- [ ] Fixture run: 250/250 valid and stable, cross-check matches upstream.
- [ ] At least the RS-only run on full250 evaluated, with CIs, in Table 1.
- [ ] Every other Table 1 cell is either a real number or `\tbd`, with the prompt set stated.
- [ ] Tables 2–4 and Figures 2–4 compile in the course template, baseline rows/points filled where runs exist.
- [ ] `compute.tex` lists every run, including pilot and failures, with GPU-hours and dollars.
- [ ] `RESULTS.md` and `AI_USE.md` are current.
- [ ] Any deviation (shim, subset, own GT renders, stability convention) is written in `runs/NOTES.md` for the report.

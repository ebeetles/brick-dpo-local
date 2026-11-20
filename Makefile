# Midway pipeline, evaluation side (SPEC §8). Generation targets (pilot, gen) are Megha's.
UPSTREAM := third_party/BrickGPT-experiments
UPSTREAM_SHA := 7c3cce4b17b21d0cff75b750ce2227f32586b00b
# --no-sync: never let uv re-resolve or reinstall; the env is built once by `make setup` from uv.lock.
PY := PYTHONPATH=$(CURDIR) uv run --frozen --no-sync --project $(UPSTREAM) python

# Packages in upstream uv.lock that have no macOS wheels and are not used by evaluation
# (mesh baselines / LGM / Hunyuan3D). Skipped on macOS only.
ifeq ($(shell uname -s),Darwin)
SKIP_PKGS := --no-install-package onnxruntime-gpu --no-install-package xformers \
             --no-install-package nvdiffrast --no-install-package hy3dgen
endif

RUN ?= runs/gt_full250_s42
SEED ?= 42

.PHONY: setup env-check prompts fixture check-run eval eval-new render report preview test

setup:
	test -d $(UPSTREAM) || git clone https://github.com/AvaLovelace1/BrickGPT-experiments $(UPSTREAM)
	cd $(UPSTREAM) && git checkout $(UPSTREAM_SHA) && git submodule init && git submodule update
	# chamferdist needs torch at build time but does not declare it, so install it after the rest.
	cd $(UPSTREAM) && uv sync --frozen $(SKIP_PKGS) --no-install-package chamferdist
	cd $(UPSTREAM) && (uv pip install --no-build-isolation --no-deps chamferdist==1.0.3 \
	  || echo "WARN: chamferdist failed to build; midway.upstream_eval stubs it (only used for nearest neighbors)")
	cd $(UPSTREAM) && uv pip install pytest

env-check:
	$(PY) -m midway.env_check

prompts:
	$(PY) -m midway.make_subset --n 50

fixture:
	$(PY) -m midway.run_fixture --prompts full250 --seed $(SEED) --out runs/gt_full250_s$(SEED)
	$(MAKE) eval RUN=runs/gt_full250_s$(SEED)

# Validate a run folder's format without evaluating it (seconds; no Gurobi needed).
check-run:
	$(PY) -m midway.check_run --run $(RUN)

eval:
	$(PY) -m midway.per_sample_eval --run $(RUN)

# P2: render with upstream (Blender, resumable), then upstream CLIP + DINO. DINO compares against our own
# ground-truth renders (upstream's default GT render dir is not released), so render the gt fixture first.
GT_RENDERS ?= runs/gt_full250_s42/renders
RENDER_WORKERS ?= 1
render:
	$(PY) -m midway.render --run $(RUN) --workers $(RENDER_WORKERS)
	$(PY) -m midway.upstream_eval --generated_dataset_path $(RUN)/generated_dataset.json \
	  --analyze_clip --analyze_img_similarity --ground_truth_render_dir $(GT_RENDERS)

# Evaluate every run dir whose generation output (generated_dataset.json, or generations/ for an unfinished run)
# is newer than its per_sample.jsonl. A failed eval does not stop the report: the run is listed as EXCLUDED instead.
eval-new:
	@failed=""; for d in runs/*/; do d=$${d%/}; \
	  src=$$d/generated_dataset.json; [ -f $$src ] || src=$$d/generations; \
	  if [ -e $$src ] && [ ! $$d/per_sample.jsonl -nt $$src ]; then \
	    $(MAKE) --no-print-directory eval RUN=$$d || failed="$$failed $$d"; fi; done; \
	if [ -n "$$failed" ]; then echo "!!! make eval FAILED for:$$failed -- these runs are EXCLUDED from tables (see RESULTS.md)"; fi

report: eval-new
	$(PY) -m midway.aggregate
	$(PY) -m midway.tables
	$(PY) -m midway.plots
	$(PY) -m midway.results_md

preview: report
	cd report && latexmk -pdf -interaction=nonstopmode -quiet preview.tex

test:
	$(PY) -m pytest -q tests

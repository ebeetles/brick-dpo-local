"""Paths, pinned upstream versions, and run definitions shared by the midway pipeline."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / 'third_party' / 'BrickGPT-experiments'
EVAL_DATASET = UPSTREAM_DIR / 'src' / 'brickgpt_experiments' / 'data' / 'eval_dataset.json'
RUNS_DIR = REPO_ROOT / 'runs'
PROMPTS_DIR = RUNS_DIR / 'prompts'
REPORT_DIR = REPO_ROOT / 'report'
TABLES_DIR = REPORT_DIR / 'tables'
FIGURES_DIR = REPORT_DIR / 'figures'

# Pinned upstream commits (SPEC §4). Also reported in the paper's replication paragraph.
UPSTREAM_SHA = '7c3cce4b17b21d0cff75b750ce2227f32586b00b'  # AvaLovelace1/BrickGPT-experiments
BRICKGPT_SHA = '59cca62b49363f3a18e01528a327f152df91bfea'  # brickgpt, as locked in upstream uv.lock

# Prompt sets (SPEC §6.2): full250 = upstream eval_dataset.json; subsetN = runs/prompts/subsetN.json (random, seed 0);
# firstN = the first N prompts of full250 in eval_dataset order (e.g. a pipeline test that stopped early).
PROMPT_SET_PATTERN = r'full250|subset\d+|first\d+'
RUN_NAME_PATTERN = rf'(?P<setting>[a-z]+)_(?P<prompts>{PROMPT_SET_PATTERN})_s(?P<seed>\d+)'

WORLD_DIM = 20
DEFAULT_SEED = 42

# Decoding settings (SPEC §3). Generation itself is Megha's; these are here so eval/tables can
# label runs and know each setting's max_regenerations (needed for `hit_rb_limit`).
# Upstream BrickGPTConfig defaults: max_brick_rejections=500, max_regenerations=100.
SETTINGS = {
    'gt': {'label': 'Ground truth', 'flags': '--generation_method ground_truth', 'max_regenerations': None},
    'norsnorb': {'label': 'No RS, no RB', 'flags': '--max_brick_rejections 0 --max_regenerations 0',
                 'max_regenerations': 0},
    'rs': {'label': 'RS only', 'flags': '--max_regenerations 0', 'max_regenerations': 0},
    'full': {'label': 'RS + RB', 'flags': '(upstream defaults)', 'max_regenerations': 100},
}

# Paper numbers (Pun et al., BrickGPT, Table 1). Only ever rendered in cells labeled "paper".
PAPER_TABLE1 = {
    'norsnorb': {'valid': 37.2, 'stable': 12.8, 'mean_stab': 0.956, 'min_stab': 0.325, 'clip': 0.329, 'dino': 0.888},
    'rs': {'valid': 100.0, 'stable': 24.0, 'mean_stab': 0.947, 'min_stab': 0.228, 'clip': 0.322, 'dino': 0.882},
    'full': {'valid': 100.0, 'stable': 98.8, 'mean_stab': 0.996, 'min_stab': 0.915, 'clip': 0.324, 'dino': 0.880,
             'rollbacks_median': 2, 'sec_per_build_median': 40.8},
}

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 0

# Stability convention for the report (decided by Elwin at checkpoint §10.2, 2026-11-20):
# raw = 1 - stress, unclipped. On upstream's stress map (always in [0, 1]) raw and clipped are identical.
STABILITY_CONVENTION = 'raw'

# Models in Tables 2-4 / Figures 2-4, in fixed order. Colors: validated colorblind-safe categorical palette
# (dataviz reference palette slots 1-5); each model also gets its own marker so color is never the only cue.
MODELS = {
    'BrickGPT': {'color': '#2a78d6', 'marker': 'o'},
    'RFT': {'color': '#eb6834', 'marker': 's'},
    'DPO-seq': {'color': '#1baf7a', 'marker': '^'},
    'DPO-prefix': {'color': '#eda100', 'marker': 'D'},
    'DPO-local': {'color': '#e87ba4', 'marker': 'v'},
}
GROUND_TRUTH_COLOR = '#6b6b6b'

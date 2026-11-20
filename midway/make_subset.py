"""Prompt sets (SPEC §6.2).

full250   upstream src/brickgpt_experiments/data/eval_dataset.json (250 prompts, all labeled is_stable == 1).
firstN    the first N prompts of full250, in eval_dataset order (no file; read from full250).
subsetN   N prompts drawn from full250 with random.Random(0), kept in full250 order, written to
          runs/prompts/subset{N}.json with the same schema. Subsets are part of the experiment definition:
          an existing subset file is never overwritten.

Run:  make prompts   (= python -m midway.make_subset --n 50)
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from midway.config import EVAL_DATASET, PROMPTS_DIR


def prompts_path(name: str) -> Path:
    """Path of a prompt-set file: 'full250' or 'subsetN' (firstN has no file of its own; see load_prompts)."""
    if name == 'full250':
        return EVAL_DATASET
    if re.fullmatch(r'subset\d+', name):
        return PROMPTS_DIR / f'{name}.json'
    raise ValueError(f'Prompt set {name!r} has no file; expected full250 or subsetN')


def load_prompts(name: str) -> list[dict]:
    """Prompts of a prompt set: full250, subsetN, or firstN (first N prompts of full250 in order)."""
    first = re.fullmatch(r'first(\d+)', name)
    if first:
        return json.loads(EVAL_DATASET.read_text())[:int(first.group(1))]
    return json.loads(prompts_path(name).read_text())


def make_subset(n: int, seed: int = 0) -> list[dict]:
    """Sample n prompts from full250 with random.Random(seed); keep them in full250 order."""
    full = load_prompts('full250')
    indices = sorted(random.Random(seed).sample(range(len(full)), n))
    return [full[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()

    path = prompts_path(f'subset{args.n}')
    subset = make_subset(args.n)
    if path.exists():
        existing = json.loads(path.read_text())
        same = [x['object_id'] for x in existing] == [x['object_id'] for x in subset]
        print(f'{path} already exists ({"identical" if same else "DIFFERENT from a fresh sample"}); not overwriting.')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset))
    print(f'Wrote {path} ({len(subset)} prompts). object_ids are in the file.')
    for category, count in Counter(x['category_id'] for x in subset).most_common():
        print(f'  {category}: {count}')


if __name__ == '__main__':
    main()

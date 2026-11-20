"""Parsing and shape statistics for brick builds (SPEC §6.4 shape fields).

All geometry comes from upstream `brickgpt.data.BrickStructure`; this file only counts things.
A build is a 20x20x20 grid. `BrickStructure.voxel_occupancy[x, y, z]` is the number of bricks
covering that voxel (2+ means a collision).
"""
import warnings

import numpy as np
from brickgpt.data import BrickStructure


def parse_build_checked(text) -> tuple[BrickStructure | None, str | None]:
    """Parse build text (one `HxW (x,y,z)` brick per line) with upstream's parser.

    Returns (build, None) on success, or (None, reason) for anything upstream `evaluate` would crash on:
    - the output is missing or not text;
    - an ill-formatted line (upstream raises ValueError);
    - a brick with z >= 20 (upstream indexes past the 20^3 grid: IndexError);
    - a brick size not in the brick library, e.g. 3x3 (parses, but upstream's stability code raises).
    Bricks with x or y outside the grid parse fine; they make the build invalid (out of bounds), not unparseable.
    """
    if not isinstance(text, str):
        return None, f'output is missing or not text ({type(text).__name__})'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # upstream warns when the lowest brick is not at z=0
            build = BrickStructure.from_txt(text)
    except ValueError as err:
        return None, f'ill-formatted line ({err})'
    except IndexError:
        return None, 'brick coordinate outside the 20x20x20 grid (z >= 20)'
    not_in_library = sorted({f'{b.h}x{b.w}' for b in build.bricks if not in_brick_library(b)})
    if not_in_library:
        return None, f'brick size not in the brick library ({", ".join(not_in_library)})'
    return build, None


def parse_build(text) -> BrickStructure | None:
    """Like parse_build_checked, but returns only the build (None if it cannot be parsed)."""
    return parse_build_checked(text)[0]


def in_brick_library(brick) -> bool:
    """True if upstream's brick library has a brick of this size (in either orientation)."""
    try:
        brick.brick_id
        return True
    except ValueError:
        return False


def occupied(build: BrickStructure) -> np.ndarray:
    """Boolean 20^3 grid: True where at least one brick covers the voxel."""
    return build.voxel_occupancy > 0


def height(build: BrickStructure) -> int | None:
    """Top layer index + 1 (number of layers from the ground to the highest brick). None if empty."""
    if len(build) == 0:
        return None
    return max(brick.z for brick in build.bricks) + 1


def footprint(build: BrickStructure) -> int:
    """Number of occupied (x, y) cells when the build is projected top-down onto the ground."""
    return int(occupied(build).any(axis=2).sum())


def frac_1x1(build: BrickStructure) -> float | None:
    """Share of bricks that are 1x1. None if the build has no bricks."""
    if len(build) == 0:
        return None
    return sum(brick.h == 1 and brick.w == 1 for brick in build.bricks) / len(build)


def voxel_iou(a: BrickStructure, b: BrickStructure) -> float | None:
    """Intersection-over-union of the two builds' 20^3 occupancy sets.

    |occupied(a) AND occupied(b)| / |occupied(a) OR occupied(b)|. None if both builds are empty.
    Bricks sticking out of the grid are clipped to the grid by numpy slicing, as in upstream.
    """
    a_occ, b_occ = occupied(a), occupied(b)
    union = np.logical_or(a_occ, b_occ).sum()
    if union == 0:
        return None
    return float(np.logical_and(a_occ, b_occ).sum() / union)

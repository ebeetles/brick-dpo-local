"""SPEC §9 items 1–6: validity and stability verdicts from upstream BrickStructure on hand-built builds.

These call upstream's Gurobi solver. The structures are small enough for the size-limited pip license.
Expected stress values were recorded from upstream at brickgpt@59cca62b on 2026-11-20.
"""
import pytest

from midway.per_sample_eval import stability_fields
from midway.shapes import parse_build


def fields(text: str) -> dict:
    return stability_fields(parse_build(text))


def test_single_brick_on_ground_valid_stable():
    f = fields('2x4 (0,0,0)')
    assert f['valid'] and f['stable']
    assert f['max_stress'] == pytest.approx(0.0)


def test_two_stacked_aligned_bricks_valid_stable():
    f = fields('2x4 (0,0,0)\n2x4 (0,0,1)')
    assert f['valid'] and f['stable']


def test_floating_brick_unstable():
    # Upstream verdict: a lone brick at z=2 is valid (no collision / out of bounds) but has stress 1 -> unstable.
    f = fields('2x4 (0,0,2)')
    assert f['valid'] and not f['stable']
    assert f['max_stress'] == pytest.approx(1.0)
    assert f['min_stab_raw'] == pytest.approx(0.0)


def test_simple_cantilever_is_stable_per_upstream():
    # SPEC §9.4 suggested a 2x6 hanging off a 2x2 support as "expect unstable". Upstream disagrees:
    # stud friction holds it with stress ~0.006. Recorded here so the verdict is documented.
    f = fields('2x2 (0,0,0)\n2x6 (0,0,1)')
    assert f['valid'] and f['stable']
    assert f['max_stress'] == pytest.approx(0.0059, abs=1e-3)


def test_heavily_loaded_cantilever_unstable():
    # A 1x8 balanced on one 1x1 stud, with an 18-high tower of 1x2 bricks on its far end.
    # Upstream: stress 0.96 with a 17-brick tower (stable), 1.0 with 18 (unstable).
    tower = ''.join(f'1x2 (0,6,{z})\n' for z in range(2, 20))
    f = fields('1x1 (0,0,0)\n1x8 (0,0,1)\n' + tower)
    assert f['valid'] and not f['stable']
    assert f['max_stress'] == pytest.approx(1.0)


def test_overlapping_bricks_invalid():
    f = fields('2x4 (0,0,0)\n2x4 (1,0,0)')
    assert not f['valid'] and not f['stable']
    assert f['max_stress'] is None  # undefined for invalid builds: null, never 0


def test_out_of_bounds_brick_invalid():
    f = fields('4x1 (19,0,0)')  # spans x = 19..22 on a 20-wide grid
    assert not f['valid'] and not f['stable']
    assert f['min_stab_raw'] is None


def test_empty_and_unparseable_builds_invalid():
    assert not stability_fields(parse_build(''))['valid']
    assert not stability_fields(None)['valid']

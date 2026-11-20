"""SPEC §9 items 7–8: voxel IoU and shortcut statistics on hand-built structures."""
import pytest

from midway.shapes import footprint, frac_1x1, height, parse_build, voxel_iou


def test_iou_identical_is_one():
    a = parse_build('2x4 (0,0,0)\n2x2 (0,0,1)')
    assert voxel_iou(a, a) == 1.0


def test_iou_disjoint_is_zero():
    a = parse_build('2x4 (0,0,0)')
    b = parse_build('2x4 (10,10,0)')
    assert voxel_iou(a, b) == 0.0


def test_iou_partial_overlap_exact():
    # a: 2x4 at (0,0,0) covers x 0-1, y 0-3 -> 8 voxels.
    # b: 2x4 at (0,2,0) covers x 0-1, y 2-5 -> 8 voxels.
    # Overlap: x 0-1, y 2-3 -> 4 voxels. Union: 8 + 8 - 4 = 12. IoU = 4/12.
    a = parse_build('2x4 (0,0,0)')
    b = parse_build('2x4 (0,2,0)')
    assert voxel_iou(a, b) == pytest.approx(4 / 12)


def test_iou_counts_colliding_voxels_once():
    # Collisions make voxel_occupancy == 2; IoU must treat occupancy as a 0/1 set.
    a = parse_build('2x4 (0,0,0)\n2x4 (0,0,0)')
    b = parse_build('2x4 (0,0,0)')
    assert voxel_iou(a, b) == 1.0


def test_shape_stats_hand_computed():
    # Layer 0: 2x4 at (0,0) and 1x1 at (5,5). Layer 1: 1x1 at (0,0), sits on the 2x4.
    # Layer 2: 1x2 at (0,0).
    build = parse_build('2x4 (0,0,0)\n1x1 (5,5,0)\n1x1 (0,0,1)\n1x2 (0,0,2)')
    assert height(build) == 3                      # top layer z=2 -> 2 + 1
    assert footprint(build) == 8 + 1               # 2x4 cells + the separate 1x1 cell; upper bricks add none
    assert frac_1x1(build) == pytest.approx(2 / 4)  # two of four bricks are 1x1


def test_shape_stats_empty_build_are_none():
    build = parse_build('')
    assert height(build) is None
    assert footprint(build) == 0
    assert frac_1x1(build) is None


def test_parse_build_bad_text_returns_none():
    assert parse_build('2x4 (0,0') is None        # ill-formatted line
    assert parse_build('2x4 (0,0,20)') is None    # z outside the 20^3 grid: upstream raises IndexError

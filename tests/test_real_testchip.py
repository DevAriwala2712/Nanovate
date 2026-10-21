"""Track C Rung 2 test chip: every block's drawn topology matches its intended class, at every preset scale."""
import dataclasses

import gdstk
import numpy as np
import pytest

from nanovate.real.testchip import (EXPECTED, LAYER_DOC, LAYER_NAV, LAYER_PATTERN, PRESETS, ChipCfg, block_set, build_chip,
                                    defect_rows, design_delta_beta, rule_verdict)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_defect_has_its_exact_design_delta_beta(preset):
    c = dataclasses.replace(ChipCfg(), **PRESETS[preset])
    for b in block_set(c):
        assert design_delta_beta(b, c) == EXPECTED[b.defect][1], (b.defect, b.size_frac)


def test_rule_verdicts_follow_module4_ladder():
    assert rule_verdict(-1, 0) == "KILLER_BRIDGE"          # bridges, via bridges, and a missing via (lost component)
    assert rule_verdict(1, 0) == "KILLER_OPEN/PARTICLE"
    assert rule_verdict(0, 0) == "NUISANCE or NEAR_MISS"


def test_defect_map_and_gds_roundtrip(tmp_path):
    c = ChipCfg()
    lib, blocks = build_chip(c)
    rows = defect_rows(blocks, c)
    assert len(rows) == 3 * len(block_set(c)) + 3
    # defect bbox inside its block; blocks do not overlap
    for r in rows:
        if r["defect_bbox_um"]:
            x0, y0, x1, y1 = map(float, r["defect_bbox_um"].split())
            assert r["block_x0_um"] <= x0 <= x1 <= r["block_x1_um"] and r["block_y0_um"] <= y0 <= y1 <= r["block_y1_um"]
    boxes = [(r["block_x0_um"], r["block_y0_um"], r["block_x1_um"], r["block_y1_um"]) for r in rows]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    p = tmp_path / "chip.gds"
    lib.write_gds(p)
    back = gdstk.read_gds(p)
    assert back.unit == pytest.approx(1e-9)
    polys = back.top_level()[0].polygons
    n_pattern = sum(len(b.polys) for b in blocks)
    assert sum(q.layer == LAYER_PATTERN for q in polys) == n_pattern
    assert sum(q.layer == LAYER_DOC for q in polys) == len(blocks)
    # navigation shapes keep clear of every block by >= 0.3 of the block gap
    for q in polys:
        if q.layer != LAYER_NAV:
            continue
        (qx0, qy0), (qx1, qy1) = q.bounding_box()
        for bx0, by0, bx1, by1 in boxes:
            dx = max(bx0 * 1e3 - qx1, qx0 - bx1 * 1e3, 0); dy = max(by0 * 1e3 - qy1, qy0 - by1 * 1e3, 0)
            assert max(dx, dy) >= 0.3 * c.gap - 1e-6


def test_same_seed_same_map_and_shuffle_differs_by_seed():
    a = [r["defect_type"] for r in defect_rows(build_chip(ChipCfg())[1], ChipCfg())]
    b = [r["defect_type"] for r in defect_rows(build_chip(ChipCfg())[1], ChipCfg())]
    c = [r["defect_type"] for r in defect_rows(build_chip(ChipCfg(seed=1))[1], ChipCfg(seed=1))]
    assert a == b and a != c and sorted(a) == sorted(c)


def test_sizes_scale_with_line_width():
    rows = defect_rows(build_chip(ChipCfg(w=50.0))[1], ChipCfg(w=50.0))
    br = sorted({r["size_nm"] for r in rows if r["defect_type"] == "bridge"})
    assert np.allclose(br, [12.5, 25.0, 37.5, 50.0])

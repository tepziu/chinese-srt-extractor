"""Regression coverage for the default subtitle removal policy."""
import ast
from pathlib import Path

import pytest

from services.pipeline_options import validate_pipeline_options
from services.batch_pipeline import validate_batch_options

ROOT = Path(__file__).resolve().parents[1]


def test_studio_populates_clean_plate_default_without_mutating_input():
    supplied = {"steps": {"burn_sub": True}, "burn_options": {}}
    result = validate_pipeline_options(supplied)
    assert result["burn_options"]["render_mode"] == "inpaint_burn"
    assert supplied["burn_options"] == {}


@pytest.mark.parametrize("mode", ["blur", "pure_burn", "inpaint_burn"])
def test_studio_preserves_explicit_render_choice(mode):
    result = validate_pipeline_options({"steps": {"burn_sub": True},
                                        "burn_options": {"render_mode": mode}})
    assert result["burn_options"]["render_mode"] == mode


@pytest.mark.parametrize("mode", ["blur", "opaque_band", "inpaint_burn"])
def test_batch_preserves_explicit_render_choice(mode):
    assert validate_batch_options({"old_subtitle_removal": mode})["old_subtitle_removal"] == mode


def test_direct_burn_entrypoints_default_to_clean_plate():
    tree = ast.parse((ROOT / "services/burn_sub.py").read_text(encoding="utf-8"))
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"burn_sub_video", "burnsub_worker"}:
            defaults = dict(zip([arg.arg for arg in node.args.args][-len(node.args.defaults):], node.args.defaults))
            assert ast.literal_eval(defaults["render_mode"]) == "inpaint_burn"
            found.add(node.name)
    assert found == {"burn_sub_video", "burnsub_worker"}


def test_ui_defaults_and_one_time_preference_migration():
    html = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    assert '<option value="inpaint_burn" selected>' in html
    assert "localStorage.setItem('preferred_render_mode', 'inpaint_burn')" in html
    assert "localStorage.getItem('clean_plate_default_version') !== '1'" in html
    assert "render_mode: currentRenderMode || 'inpaint_burn'" in html

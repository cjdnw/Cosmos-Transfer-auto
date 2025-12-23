import ast
import re
from pathlib import Path

import pytest


def load_cosmos_style_prompt() -> str:
    root = Path(__file__).resolve().parents[1]
    source = root / "robot_video_to_prompt.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "COSMOS_STYLE_PROMPT":
                    return ast.literal_eval(node.value)
    raise RuntimeError("COSMOS_STYLE_PROMPT not found in robot_video_to_prompt.py")


def test_visibility_change_restrictions_present():
    COSMOS_STYLE_PROMPT = load_cosmos_style_prompt()
    verbs = [
        "appears",
        "emerges",
        "shows up",
        "comes into view",
        "enters the frame/scene",
        "is introduced",
        "materializes",
        "pops up",
        "is revealed",
        "becomes visible",
        "is seen for the first time",
        "disappears",
        "vanishes",
        "leaves the frame/scene",
        "exits",
        "is removed",
        "is no longer visible",
    ]
    lower_prompt = COSMOS_STYLE_PROMPT.lower()
    for verb in verbs:
        assert verb in lower_prompt


def test_tabletop_soft_item_is_not_called_base():
    COSMOS_STYLE_PROMPT = load_cosmos_style_prompt()
    assert "STRUCTURE VS. TABLETOP OBJECT DISAMBIGUATION" in COSMOS_STYLE_PROMPT
    assert (
        "Any colored item on the table surface (especially red/dark red) should NOT be labeled as a robot base."
        in COSMOS_STYLE_PROMPT
    )
    for soft_item in ["cloth", "rag", "pad", "towel"]:
        assert soft_item in COSMOS_STYLE_PROMPT
    assert "dark red base" not in COSMOS_STYLE_PROMPT.lower()


def test_base_color_guidance_is_neutral():
    COSMOS_STYLE_PROMPT = load_cosmos_style_prompt()
    assert "Do not specify non-neutral base colors" in COSMOS_STYLE_PROMPT
    assert "light-colored base" in COSMOS_STYLE_PROMPT
    assert "dark base" in COSMOS_STYLE_PROMPT
    # Spot-check that direct color claims like "red base" are discouraged by only appearing inside a "do not" rule.
    assert re.search(r"\\bred base\\b", COSMOS_STYLE_PROMPT.lower()) is None

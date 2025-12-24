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


def test_temporal_consistency_rules_present():
    COSMOS_STYLE_PROMPT = load_cosmos_style_prompt()
    assert "TEMPORAL CONSISTENCY" in COSMOS_STYLE_PROMPT
    assert "If the manipulated object is visible at the beginning" in COSMOS_STYLE_PROMPT
    assert "If it becomes visible later" in COSMOS_STYLE_PROMPT
    assert "Do not describe minor surface marks" in COSMOS_STYLE_PROMPT


def test_robot_description_is_minimal():
    COSMOS_STYLE_PROMPT = load_cosmos_style_prompt()
    assert "Mention the robot arms only for position" in COSMOS_STYLE_PROMPT
    assert "end-effector type (gripper)" in COSMOS_STYLE_PROMPT


def test_style_rules_are_present():
    COSMOS_STYLE_PROMPT = load_cosmos_style_prompt()
    assert "Avoid rare/poetic color words" in COSMOS_STYLE_PROMPT
    assert "Do not add extra objects" in COSMOS_STYLE_PROMPT
    assert re.search(r"rare/poetic", COSMOS_STYLE_PROMPT)

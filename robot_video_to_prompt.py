import os
import re
import json
import random
import argparse
from typing import Dict, List, Any, Set

import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# =======================
# 1) Prompts
# =======================
COSMOS_STYLE_PROMPT = """# You are a video prompt generator. Write a single English paragraph in the same style as the example: detailed, factual, step-by-step, and descriptive.
#
# CRITICAL TEMPORAL CONSTRAINT:
# - Do not assume the presence of any manipulated object at the beginning unless it is explicitly visible at the start.
# - If an object appears later in the video, describe its entry as a distinct event before any interaction.
#
# Requirements:
# - Describe static background elements first from an overhead viewpoint.
# - Describe each static object in a separate, standalone sentence.
# - Identify the robot arm(s), their materials, and their positions relative to the scene.
# - Identify any container and describe its initial state (empty or occupied).
#
# Action description rules:
# - Use temporal phrasing (At the beginning…, As the video progresses…, Then…, Finally…).
# - Introduce manipulated objects only when they first appear.
# - Do not imply grasping, lifting, or containment before the object’s appearance is described.
#
# Do not mention models, frames, timestamps, or reasoning steps.
# Output only the paragraph.
#
# Example (Correct – object presence and manipulation are temporally consistent):
# The video is a demonstration of robotic manipulation in a controlled indoor environment. It features two robotic arms interacting with a piece of red fabric. The setting includes a beige couch in the background, providing a neutral backdrop for the activity. A yellow cushion is placed in front of the couch, and the red fabric lies flat on top of the cushion. The left robotic arm is white and equipped with a black gripper. The right robotic arm is black and fitted with a more articulated gripper. At the beginning, the fabric rests motionless on the cushion. As the video progresses, the left arm moves toward the fabric and grasps one side. The fabric is lifted slightly off the cushion. The right arm then moves in and grasps the opposite side of the fabric. Both arms coordinate to lift and hold the fabric between them, demonstrating controlled and precise manipulation.
#
# Counterexample (Incorrect – violates object placement and temporal introduction):
# The video shows two robotic arms manipulating a piece of red fabric. At the beginning, the fabric is already being held between the two robotic grippers above a cushion. The arms then adjust their grip and continue lifting the fabric while the background couch remains visible.
#
#
# Now generate the paragraph for the given video.
"""

BRACKET_PROMPT_TEMPLATE = """You are a formatting tool.
Your task is ONLY to insert Chinese corner brackets 【】 around existing spans in the paragraph.

STRICT RULES:
- DO NOT translate, paraphrase, rewrite, reorder, or delete ANY characters.
- The output must be IDENTICAL to the input paragraph after removing all 【】.
- The text inside each 【】 MUST be an exact, verbatim substring copied from the paragraph.

WHAT TO BRACKET:
- Bracket ALL non-robot COLOR / MATERIAL / TEXTURE / SURFACE-FINISH descriptors, including for major static objects
  (e.g., floor, table, sofa, rug, basket, cloth, objects).
- Treat the following as texture/material/finish words and bracket them when they modify an object:
  smooth, matte, glossy, textured, woven, rough, soft, hard, metallic, wooden, plastic, rubber, ceramic, glass.
- Include finer-grained surface descriptors when present (e.g., striped, patterned, ribbed, knitted, wrinkled).
- Prefer bracketing the full noun phrase when possible (descriptor + noun), e.g., "orange cushion" not just "orange".

WHAT NOT TO BRACKET:
- NEVER bracket anything describing the robot system, including: robot/robotic, arm, gripper, end-effector,
  finger(s), base, mount, joint(s), cable(s), wire(s), sensor(s).
  Skip only the robot descriptors themselves; other non-robot descriptors in the same sentence can still be bracketed.

PARAGRAPH:
{base_prompt}
"""

DESCRIPTOR_EXTRACT_PROMPT_TEMPLATE = """Extract all non-robot COLOR / MATERIAL / TEXTURE / SURFACE-FINISH descriptors from the paragraph.

STRICT RULES:
- Output STRICT JSON only: {{"descriptors": ["..."]}}.
- Each descriptor must be an exact, verbatim substring copied from the paragraph.
- Prefer full noun phrases (descriptor + noun), e.g., "orange cushion" not just "orange".
- Do NOT include any robot-system descriptors (robot/robotic/arm/gripper/end-effector/finger/base/mount/joint/cable/wire/sensor).
- If a descriptor applies to a robot component, exclude it.
- Do NOT add explanations.

PARAGRAPH:
{base_prompt}
"""

BRACKET_REFLECTION_PROMPT_TEMPLATE = """You are a formatting tool.
Review the bracketed paragraph and fix any missing or extra brackets.

STRICT RULES:
- DO NOT translate, paraphrase, reorder, or delete ANY characters.
- You may ONLY add or remove Chinese corner brackets 【】.
- The output must be IDENTICAL to the input paragraph after removing all 【】.
- The text inside each 【】 MUST be an exact, verbatim substring copied from the paragraph.
- Bracket non-robot COLOR / MATERIAL / TEXTURE / SHAPE descriptors tied to objects.
- Do NOT bracket framing phrases like "background consists of", "the setting includes", or "the scene features".
  If such a phrase appears, move brackets inward to cover ONLY the object phrase (e.g., background consists of 【smooth, light-gray walls】).
- Never start a bracket span with copula/aux verbs (is/are/was/were/...). If needed, bracket ONLY the noun phrase after it (e.g., is a 【white box】).
- If replacing a bracketed span with a single noun would make the sentence ungrammatical
  or change the meaning, consider it over-bracketed and remove it.
- If a descriptor phrase is missing brackets, add them.
- Output STRICT JSON only: {{"paragraph": "<result>"}}

PARAGRAPH:
{base_prompt}

CURRENT:
{bracketed_prompt}
"""

PRE2_SPLIT_PROMPT_TEMPLATE = """You are a formatting tool.
Insert commas to separate robot-related clauses from non-robot clauses when they appear in the same sentence.

STRICT RULES:
- DO NOT translate, paraphrase, reorder, or delete ANY characters.
- You may ONLY insert commas and spaces.
- Output must be identical to the input paragraph after removing the commas you inserted.
- If no change is needed, return the original paragraph.
- Output STRICT JSON only: {{"paragraph": "<result>"}}

PARAGRAPH:
{base_prompt}
"""

REPLACEMENT_JSON_PROMPT_TEMPLATE = """Output protocol (must follow exactly):
- Only output a single JSON object.
- No markdown. No code fences. No extra text.
- The top-level must be an object with exactly one key: "replacements".
- "replacements" must be an object mapping string -> array[string].
- Output must be valid JSON that can be parsed by json.loads() directly.

Valid example:
{"replacements":{
  "red":["scarlet","crimson","ruby"],
  "plastic":["polymer","synthetic","resin"]
}}

Invalid examples: [...], {{"data": ...}}, Here is the JSON: {{...}}

Task:
Generate prompt variations. For each phrase below, propose {num_alternatives} short alternatives
that keep the same object/category and stay grounded in indoor robotics scenes.

Rules:
- Do NOT repeat the original.
- Keep 1-4 words (or match the original length when needed).
- Use common, simple descriptors suitable for indoor robotics scenes.
- For COLOR terms, alternatives MUST change the color (not just texture) and should vary the palette.
- Preserve any MATERIAL words from the original phrase (e.g., keep "plastic" if present).
- Keep the core object noun; you may vary shape/size adjectives to increase diversity.
- If multiple phrases share the same object noun (e.g., "table" and "table surface"),
  keep their color/material/shape changes consistent across those phrases.
- Avoid rare or poetic color words.
- Avoid vague/qualifier words (e.g., "pale", "light", "dark", "soft", "muted", "faint", "slight", "subtle", "smooth"), unless the source phrase already contains them (keeping them is acceptable to preserve fidelity).
- Do NOT include robot-related terms (robot/robotic/arm/gripper/end-effector/finger/base/mount/joint/cable/wire/sensor).

DESCRIPTORS:
{descriptor_block}
"""

REPLACEMENT_RETRY_PROMPT_TEMPLATE = """Regenerate replacements for the phrase below.

Issues to fix:
{issues}

Rules:
- Provide {num_alternatives} alternatives.
- Do NOT repeat the original.
- Keep 1-4 words (or match the original length when needed).
- Use common, simple descriptors suitable for indoor robotics scenes.
- For COLOR terms, alternatives MUST change the color (not just texture) and should vary the palette.
- Preserve any MATERIAL words from the original phrase (e.g., keep "plastic" if present).
- Keep the core object noun; you may vary shape/size adjectives to increase diversity.
- Avoid vague/qualifier words (e.g., "pale", "light", "dark", "soft", "muted", "faint", "slight", "subtle", "smooth"), unless the source phrase already contains them (keeping them is acceptable to preserve fidelity).
- Do NOT include robot-related terms (robot/robotic/arm/gripper/end-effector/finger/base/mount/joint/cable/wire/sensor).
- Output STRICT JSON only: {{"alternatives": ["alt1", ...]}}.

PHRASE:
{term}
"""

ROBOT_KEYWORDS = {
    "robot",
    "robotic",
    "arm",
    "gripper",
    "end-effector",
    "endeffector",
    "finger",
    "fingers",
    "base",
    "mount",
    "joint",
    "cable",
    "cables",
    "wire",
    "wires",
    "sensor",
    "sensors",
}

ROBOT_CANONICAL_SENTENCE = "Two robotic arms with articulated grippers are positioned on the left and right sides of the container."
ROBOT_SENTENCE_PATTERN = re.compile(r"\\b(" + "|".join(sorted(ROBOT_KEYWORDS, key=len, reverse=True)) + r")\\b", re.IGNORECASE)
CONTAINER_KEYWORDS = {
    "basket",
    "box",
    "bin",
    "bowl",
    "container",
    "tray",
}
VAGUE_MODIFIERS = [
    "faint",
    "slight",
    "subtle",
    "somewhat",
    "vaguely",
]
VAGUE_PHRASES = [
    "a bit",
    "kind of",
]
COLORISH_BASES = [
    "brown",
    "gray",
    "grey",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "black",
    "white",
    "beige",
    "tan",
    "silver",
    "gold",

]
COLOR_BASES = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "beige",
    "tan",
    "silver",
    "gold",

]
MATERIAL_BASES = [
    "plastic",
    "metal",
    "metallic",
    "wooden",
    "glass",
    "ceramic",
    "rubber",
]
TEXTURE_BASES = [
    "smooth",
    "matte",
    "glossy",
    "satin",
    "polished",
    "brushed",
    "grainy",
    "speckled",
    "pebbled",
    "bumpy",
    "textured",
    "woven",
    "felted",
    "velvety",
    "rough",
    "soft",
    "hard",
    "striped",
    "patterned",
    "ribbed",
    "knitted",
    "wrinkled",
    "transparent",
]
SHAPE_BASES = [
    "rectangular",
    "square",
    "round",
    "circular",
    "oval",
    "triangular",
    "long",
    "short",
    "wide",
    "narrow",
    "large",
    "small",
    "big",
]
VAGUE_COLOR_MODIFIERS = [
    "pale",
    "light",
    "dark",
    "soft",
    "muted",
    "faint",
    "slight",
    "subtle",
]
VAGUE_TEXTURE_MODIFIERS = [
    "smooth",
    "soft",
    "hard",
    "textured",
    "matte",
    "glossy",
]

# Step3 cleaning behavior
# If True, Step3 allows vague modifiers (e.g., light/smooth) *when they already exist in the source phrase*.
STEP3_ALLOW_SOURCE_VAGUE_MODIFIERS = True
# If True, Step3 fills missing alternatives using deterministic, rule-based generators (colors/materials/textures).
STEP3_ENABLE_RULE_BACKFILL = True

# =======================
# 2) Model Options
# =======================
QWEN25_VL_MODELS = {
    "3B": {"instruct": "Qwen/Qwen2.5-VL-3B-Instruct", "awq": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"},
    "7B": {"instruct": "Qwen/Qwen2.5-VL-7B-Instruct", "awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
    "32B": {"instruct": "Qwen/Qwen2.5-VL-32B-Instruct", "awq": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"},
    "72B": {"instruct": "Qwen/Qwen2.5-VL-72B-Instruct", "awq": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"},
}

QWEN3_VL_MODELS = {
    "2B": {"instruct": "Qwen/Qwen3-VL-2B-Instruct", "awq": "Qwen/Qwen3-VL-2B-Instruct-AWQ"},
    "4B": {"instruct": "Qwen/Qwen3-VL-4B-Instruct", "awq": "Qwen/Qwen3-VL-4B-Instruct-AWQ"},
    "8B": {"instruct": "Qwen/Qwen3-VL-8B-Instruct", "awq": "Qwen/Qwen3-VL-8B-Instruct-AWQ"},
    "32B": {"instruct": "Qwen/Qwen3-VL-32B-Instruct", "awq": "Qwen/Qwen3-VL-32B-Instruct-AWQ"},
}

# =======================
# 3) Helpers
# =======================
def pick_model_id(size: str, variant: str, model_type: str) -> str:
    size = size.upper()
    variant = variant.lower()
    model_dict = QWEN3_VL_MODELS if model_type == "Qwen3" else QWEN25_VL_MODELS
    if size not in model_dict:
        raise ValueError(f"Unsupported --size {size}")
    if variant not in model_dict[size]:
        raise ValueError(f"Unsupported --variant {variant}")
    return model_dict[size][variant]


def safe_parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()
    if "```" in cleaned:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[idx:])
                return obj
            except json.JSONDecodeError:
                continue
        return {}


def extract_bracket_terms(text: str) -> List[str]:
    raw_terms = [t.strip() for t in re.findall(r"【([\s\S]*?)】", text)]
    unique_terms = list(dict.fromkeys(t for t in raw_terms if t))
    lowered = [term.lower() for term in unique_terms]
    filtered = []
    for idx, term in enumerate(unique_terms):
        if any(
            idx != other_idx and lowered[idx] in lowered[other_idx]
            for other_idx in range(len(unique_terms))
        ):
            continue
        filtered.append(term)
    return filtered


def normalize_term_key(term: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", term.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _descriptor_token(token: str) -> bool:
    lower = token.lower().strip(",.;:")
    if not lower:
        return False
    if lower in ROBOT_KEYWORDS:
        return False
    if lower in COLOR_BASES or lower in MATERIAL_BASES or lower in TEXTURE_BASES or lower in SHAPE_BASES:
        return True
    if "-" in lower:
        parts = [p for p in lower.split("-") if p]
        if any(part in COLOR_BASES for part in parts):
            return True
    if any(base in lower for base in COLOR_BASES):
        return True
    return False


def _noun_modifier_token(token: str) -> bool:
    return bool(re.match(r"^[A-Za-z-]+$", token))


GENERIC_OBJECT_TOKENS = {
    "surface",
    "top",
    "side",
    "edge",
    "front",
    "back",
    "center",
    "middle",
    "bottom",
    "upper",
    "lower",
}


_TERM_STOPWORDS = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # copula / auxiliaries (common LLM bracketing mistakes)
    "am", "is", "are", "was", "were", "be", "being", "been",
    # adverbs / discourse markers that often appear in mistaken spans
    "also", "still", "then", "there", "here", "just", "really", "quite", "rather",
    # common prepositions / conjunctions
    "and", "or", "but", "of", "with", "without", "in", "on", "at", "from", "to", "into", "onto",
    "near", "above", "below", "over", "under", "around", "between", "within", "across",
}


def _term_object_tokens(term: str) -> List[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z-]+", term)]
    filtered = []
    for token in tokens:
        if token in _TERM_STOPWORDS:
            continue
        if token in ROBOT_KEYWORDS or token in GENERIC_OBJECT_TOKENS:
            continue
        if _descriptor_token(token):
            continue
        filtered.append(token)
    return filtered



def _shared_object_groups(terms: List[str]) -> Dict[str, str]:
    freq: Dict[str, int] = {}
    term_tokens: Dict[str, List[str]] = {}
    for term in terms:
        tokens = _term_object_tokens(term)
        term_tokens[term] = tokens
        for token in set(tokens):
            freq[token] = freq.get(token, 0) + 1

    mapping: Dict[str, str] = {}
    for term, tokens in term_tokens.items():
        shared = [t for t in tokens if freq.get(t, 0) >= 2]
        if not shared:
            continue
        shared.sort(key=lambda t: (freq[t], len(t)), reverse=True)
        mapping[term] = shared[0]
    return mapping


def _required_materials(term: str) -> List[str]:
    materials = []
    for material in MATERIAL_BASES:
        if re.search(rf"\\b{re.escape(material)}\\b", term, re.IGNORECASE):
            materials.append(material)
    return materials


def _ensure_materials(term: str, alt: str) -> str:
    materials = _required_materials(term)
    if not materials:
        return alt
    missing = [m for m in materials if not re.search(rf"\\b{re.escape(m)}\\b", alt, re.IGNORECASE)]
    if not missing:
        return alt
    tokens = alt.split()
    if not tokens:
        return alt
    insert_at = max(len(tokens) - 1, 0)
    for material in missing:
        tokens.insert(insert_at, material)
        insert_at += 1
    return " ".join(tokens)


def _align_term_with_anchor(term: str, anchor_alt: str, shared_token: str) -> str:
    term_tokens = term.split()
    anchor_tokens = anchor_alt.split()
    if not term_tokens or not anchor_tokens:
        return anchor_alt
    if shared_token not in [t.lower() for t in term_tokens]:
        return anchor_alt
    aligned = []
    for token in term_tokens:
        if token.lower() == shared_token:
            aligned.extend(anchor_tokens)
        else:
            aligned.append(token)
    return " ".join(aligned)

def expand_bracketed_prompt(paragraph: str) -> str:
    base = paragraph.replace("【", "").replace("】", "")
    terms = extract_bracket_terms(paragraph)
    if not terms:
        return paragraph

    token_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\b[\w-]+\b", base)]
    stopwords = {"a", "an", "the", "and", "or", "with", "to", "from", "on", "in", "at", "by", "near", "toward"}
    verbs = {
        "move", "moves", "moving",
        "grasp", "grasps", "grasping",
        "lift", "lifts", "lifting",
        "hold", "holds", "holding",
        "place", "places", "placed", "placing",
        "extend", "extends", "extended", "extending",
        "reach", "reaches", "reached", "reaching",
    }
    spans = []

    for term in terms:
        for match in re.finditer(re.escape(term), base):
            span_start, span_end = match.span()
            token_index = next(
                (idx for idx, (_, start, _) in enumerate(token_spans) if start == span_start),
                None,
            )
            if token_index is None:
                spans.append((span_start, span_end))
                continue
            expanded_start = span_start
            idx = token_index - 1
            while idx >= 0:
                token, start, end = token_spans[idx]
                gap = base[end:expanded_start]
                if any(ch in gap for ch in ",;:.!?"):
                    break
                lower = token.lower()
                if lower in stopwords or lower in VAGUE_MODIFIERS or lower in verbs:
                    break
                if any(keyword in lower for keyword in ROBOT_KEYWORDS):
                    break
                if not _descriptor_token(token) and not _noun_modifier_token(token):
                    break
                expanded_start = start
                idx -= 1
            spans.append((expanded_start, span_end))

    merged = _merge_spans(spans)
    return _insert_brackets(base, merged)


def _split_on_commas_excluding_ranges(sentence: str, ranges: List[tuple]) -> List[str]:
    if not ranges:
        return [c.strip() for c in sentence.split(",") if c.strip()]
    clauses = []
    last = 0
    for idx, ch in enumerate(sentence):
        if ch != ",":
            continue
        if any(start <= idx < end for start, end in ranges):
            continue
        clause = sentence[last:idx].strip()
        if clause:
            clauses.append(clause)
        last = idx + 1
    tail = sentence[last:].strip()
    if tail:
        clauses.append(tail)
    return clauses


def _in_robot_context(span: str, paragraph: str, window: int = 40) -> bool:
    """Heuristic: treat a span as 'robot-context' only if a robot keyword appears *near* the span.

    This avoids over-filtering phrases like 'white rectangular table' that co-occur in the same sentence
    with robot mentions (e.g., 'robotic arms ... on either side of a white rectangular table').
    """
    if not span:
        return False

    lowered_span = span.lower()
    if any(keyword in lowered_span for keyword in ROBOT_KEYWORDS):
        return True

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s]
    for sentence in sentences:
        lowered_sentence = sentence.lower()
        if lowered_span not in lowered_sentence:
            continue

        span_ranges = [match.span() for match in re.finditer(re.escape(span), sentence)]
        clauses = _split_on_commas_excluding_ranges(sentence, span_ranges)

        for clause in clauses:
            lowered_clause = clause.lower()
            if lowered_span not in lowered_clause:
                continue

            # Find all robot keyword occurrences in this clause.
            robot_hits = []
            for kw in ROBOT_KEYWORDS:
                for m in re.finditer(rf"\b{re.escape(kw)}\b", lowered_clause):
                    robot_hits.append(m.span())
            if not robot_hits:
                continue

            # For each occurrence of the span in the clause, check proximity to a robot keyword.
            for span_m in re.finditer(re.escape(lowered_span), lowered_clause):
                s0, s1 = span_m.span()

                # direct proximity window
                for r0, r1 in robot_hits:
                    if r0 <= s1 + window and r1 >= s0 - window:
                        return True

                # copula predicate case: 'robotic arm is silver' (span='is silver' etc.)
                if re.match(r"^(is|are|was|were|seems|seem|appears|appear)\b", lowered_span):
                    for r0, r1 in robot_hits:
                        if r1 <= s0 and (s0 - r1) <= max(window * 2, 80):
                            return True

    return False


def _collect_existing_spans(paragraph: str, terms: List[str]) -> List[tuple]:
    spans = []
    for term in terms:
        if not term:
            continue
        if re.search(r"\bbackground\s+consists\s+of\b", term, re.IGNORECASE):
            continue
        for match in re.finditer(re.escape(term), paragraph):
            if _in_robot_context(term, paragraph):
                continue
            spans.append(match.span())
    return spans


def _merge_spans(spans: List[tuple]) -> List[tuple]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    merged = []
    for span in spans:
        overlap = False
        for kept in merged:
            if not (span[1] <= kept[0] or span[0] >= kept[1]):
                overlap = True
                break
        if not overlap:
            merged.append(span)
    return merged


def _insert_brackets(paragraph: str, spans: List[tuple]) -> str:
    if not spans:
        return paragraph
    for start, end in sorted(spans, key=lambda s: s[0], reverse=True):
        paragraph = paragraph[:end] + "】" + paragraph[end:]
        paragraph = paragraph[:start] + "【" + paragraph[start:]
    return paragraph


def _should_keep_bracket(term: str) -> bool:
    # Reject framing phrases as *whole* spans; we may re-bracket their object phrase in cleanup.
    if re.search(r"\bbackground\s+consists\s+of\b", term, re.IGNORECASE):
        return False
    if any(keyword in term.lower() for keyword in ROBOT_KEYWORDS):
        return False
    if not _term_object_tokens(term):
        return False
    return True


_BRACKET_FRAMING_PATTERNS = [
    re.compile(r"^((?:the\s+)?background\s+consists\s+of\s+)(.+)$", re.IGNORECASE),
    re.compile(r"^((?:the\s+)?background\s+(?:includes|contains|features|shows)\s+)(.+)$", re.IGNORECASE),
    re.compile(r"^((?:the\s+)?setting\s+(?:includes|contains|features)\s+)(.+)$", re.IGNORECASE),
    re.compile(r"^((?:the\s+)?scene\s+(?:includes|contains|features)\s+)(.+)$", re.IGNORECASE),
    re.compile(r"^((?:the\s+)?environment\s+(?:includes|contains|features)\s+)(.+)$", re.IGNORECASE),
]

# Pre-rewrite for cases where the model brackets the whole framing phrase.
# Example: 【background consists of smooth, light-gray walls】  ->  background consists of 【smooth, light-gray walls】
_FRAMING_BRACKET_REWRITE = re.compile(
    r"【\s*((?:the\s+)?background\s+(?:consists\s+of|includes|contains|features|shows|is\s+(?:made|composed)\s+of)\s+)([\s\S]*?)\s*】",
    re.IGNORECASE,
)


_BRACKET_COPULA_PATTERN = re.compile(
    r"^((?:is|are|was|were|seems|seem|appears|appear)\b(?:\s+(?:a|an|the)\b)?\s+)(.+)$",
    re.IGNORECASE,
)


def _rewrite_bracket_span(term: str) -> str:
    """Rewrite a single bracket span while preserving original text.

    Goal: prevent *framing phrases* (e.g., 'background consists of ...') and
    *predicate-only spans* (e.g., 'is silver') from being treated as replaceable units.

    Cases handled:
    - Framing: 'background consists of X'  ->  'background consists of 【X】'
    - Copula:  'is a wooden box'          ->  'is a 【wooden box】'
              (but drops brackets entirely if the remainder has no object token, e.g., 'is silver')
    - Multiline spans: normalize whitespace and retry; if still not keepable, drop brackets.
    """
    raw = term
    term = term.strip()
    if not term:
        return raw

    # Normalize internal whitespace for matching (handles hidden newlines/spaces from LLM output).
    term_norm = re.sub(r"\s+", " ", term).strip()

    # Helper: keep trailing punctuation outside the inner bracket.
    def _split_trailing_punct(s: str) -> tuple[str, str]:
        m = re.match(r"^(.*?)([\.,;:!?]+)?$", s.strip())
        if not m:
            return s.strip(), ""
        core = (m.group(1) or "").rstrip()
        punct = m.group(2) or ""
        return core, punct

    # 1) Framing phrases: move brackets inward to cover only the object phrase.
    for pat in _BRACKET_FRAMING_PATTERNS:
        m = pat.match(term_norm)
        if not m:
            continue
        prefix, rest = m.group(1), m.group(2)
        rest_core, rest_punct = _split_trailing_punct(rest)
        if _should_keep_bracket(rest_core):
            return f"{prefix}【{rest_core}】{rest_punct}"
        return prefix + rest

    # 2) Copula/aux prefixes: avoid bracket terms like 'is silver' / 'is also silver'.
    m = _BRACKET_COPULA_PATTERN.match(term_norm)
    if m:
        prefix, rest = m.group(1), m.group(2)
        rest_core, rest_punct = _split_trailing_punct(rest)
        if _should_keep_bracket(rest_core):
            return f"{prefix}【{rest_core}】{rest_punct}"
        return prefix + rest

    # 3) Default: keep or drop as-is (use normalized text to avoid multiline leakage).
    return f"【{term_norm}】" if _should_keep_bracket(term_norm) else term_norm

def _cleanup_bracketed_prompt(bracketed_prompt: str) -> str:
    # First, rewrite whole-framing brackets into inner-object brackets.
    bracketed_prompt = _FRAMING_BRACKET_REWRITE.sub(
        lambda m: f"{m.group(1)}【{m.group(2).strip()}】",
        bracketed_prompt,
    )

    def repl(match: re.Match) -> str:
        term = match.group(1)
        return _rewrite_bracket_span(term)

    return re.sub(r"【([\s\S]*?)】", repl, bracketed_prompt)

def _reflect_brackets(model, processor, base_prompt: str, bracketed_prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": BRACKET_REFLECTION_PROMPT_TEMPLATE.format(
                base_prompt=base_prompt,
                bracketed_prompt=bracketed_prompt,
            ),
        }],
    }]
    raw = run_generation(model, processor, messages, 500, False, 1.0, 1.0)
    parsed = safe_parse_json(raw)
    paragraph = parsed.get("paragraph") if isinstance(parsed, dict) else None
    if not isinstance(paragraph, str):
        return bracketed_prompt
    normalized = paragraph.replace("【", "").replace("】", "").replace(" ", "")
    original = base_prompt.replace(" ", "")
    if normalized != original:
        return bracketed_prompt
    return _cleanup_bracketed_prompt(paragraph)


def _expand_shared_noun_phrases(paragraph: str, bracketed_prompt: str) -> str:
    base = paragraph.replace("【", "").replace("】", "")
    terms = extract_bracket_terms(bracketed_prompt)
    if not terms:
        return bracketed_prompt
    noun_tokens = sorted({token for term in terms for token in _term_object_tokens(term)})
    if not noun_tokens:
        return bracketed_prompt

    spans = []
    for term in terms:
        for match in re.finditer(re.escape(term), base):
            spans.append(match.span())
    for noun in noun_tokens:
        pattern = re.compile(rf"\\b([A-Za-z-]+(?:\\s+[A-Za-z-]+){{0,4}})\\s+{re.escape(noun)}\\b")
        for match in pattern.finditer(base):
            full_span = match.span()
            phrase = match.group(0)
            tokens = [t for t in re.findall(r"[A-Za-z-]+", match.group(1))]
            if not any(_descriptor_token(token) for token in tokens):
                continue
            if re.search(r"\bbackground\s+consists\s+of\b", phrase, re.IGNORECASE):
                continue
            spans.append(full_span)

    merged = _merge_spans(spans)
    if not merged:
        return bracketed_prompt
    expanded = _insert_brackets(base, merged)
    return _cleanup_bracketed_prompt(expanded)


def canonicalize_robot_sentences(paragraph: str) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\\s+", paragraph.strip()) if s]
    if not sentences:
        return paragraph.strip()

    output = []
    robot_added = False
    for sentence in sentences:
        if ROBOT_SENTENCE_PATTERN.search(sentence):
            if not robot_added:
                output.append(ROBOT_CANONICAL_SENTENCE)
                robot_added = True
            continue
        output.append(sentence)

    if not output:
        return ROBOT_CANONICAL_SENTENCE
    return " ".join(output).strip()


def _append_initially_empty(sentence: str) -> str:
    match = re.search(r"([.!?])$", sentence.strip())
    if match:
        punctuation = match.group(1)
        return sentence.strip()[:-1] + ", initially empty" + punctuation
    return sentence.strip() + ", initially empty"


def enforce_container_initial_state(paragraph: str) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\\s+", paragraph.strip()) if s]
    if not sentences:
        return paragraph.strip()

    seen = set()
    initial_phrase = re.compile(r"\\binitially\\s+(empty|containing)\\b", re.IGNORECASE)
    output = []
    for sentence in sentences:
        lowered = sentence.lower()
        found = {kw for kw in CONTAINER_KEYWORDS if re.search(rf"\\b{re.escape(kw)}\\b", lowered)}
        first_mentions = found - seen
        if first_mentions:
            seen |= first_mentions
            if not initial_phrase.search(sentence):
                sentence = _append_initially_empty(sentence)
        output.append(sentence)
    return " ".join(output).strip()


def cleanup_vague_terms(paragraph: str) -> str:
    warnings = []
    color_pattern = re.compile(
        r"\\b(" + "|".join(COLORISH_BASES) + r")ish\\b",
        re.IGNORECASE,
    )

    def color_repl(match: re.Match) -> str:
        original = match.group(0)
        base = match.group(1)
        replacement = base
        if original.isupper():
            replacement = base.upper()
        elif original[0].isupper():
            replacement = base.capitalize()
        warnings.append(f"vague color '{original}' -> '{replacement}'")
        return replacement

    cleaned = color_pattern.sub(color_repl, paragraph)

    def remove_modifier(match: re.Match) -> str:
        original = match.group(0).strip()
        warnings.append(f"vague modifier removed '{original}'")
        return ""

    if VAGUE_MODIFIERS:
        cleaned = re.sub(
            r"\\b(" + "|".join(VAGUE_MODIFIERS) + r")\\b\\s+",
            remove_modifier,
            cleaned,
            flags=re.IGNORECASE,
        )
    if VAGUE_PHRASES:
        cleaned = re.sub(
            r"\\b(" + "|".join(map(re.escape, VAGUE_PHRASES)) + r")\\b\\s+",
            remove_modifier,
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = re.sub(r"\\s{2,}", " ", cleaned).strip()
    for warning in warnings:
        print(f"[WARN] Step 1.5: {warning}")
    return cleaned


def _apply_color_case(color: str, original: str) -> str:
    if original.isupper():
        return color.upper()
    if original[0].isupper():
        return color.capitalize()
    return color


def build_color_variations(term: str, max_alternatives: int) -> List[str]:
    pattern = re.compile(r"\\b(" + "|".join(COLOR_BASES) + r")\\b", re.IGNORECASE)
    match = pattern.search(term)
    if not match:
        return []
    original_color = match.group(1)
    original_lower = original_color.lower()
    replacements = []
    for color in COLOR_BASES:
        if color == original_lower:
            continue
        replacement_color = _apply_color_case(color, original_color)
        candidate = pattern.sub(replacement_color, term, count=1)
        if candidate != term:
            replacements.append(candidate)
        if len(replacements) >= max_alternatives:
            break
    return replacements


def _replace_descriptor_word(term: str, pattern: re.Pattern, replacement: str) -> str:
    return pattern.sub(replacement, term, count=1)


def build_material_variations(term: str, max_alternatives: int) -> List[str]:
    pattern = re.compile(r"\\b(" + "|".join(MATERIAL_BASES) + r")\\b", re.IGNORECASE)
    match = pattern.search(term)
    if not match:
        return []
    original = match.group(1)
    original_lower = original.lower()
    replacements = []
    for material in MATERIAL_BASES:
        if material == original_lower:
            continue
        replacement_material = _apply_color_case(material, original)
        candidate = _replace_descriptor_word(term, pattern, replacement_material)
        if candidate != term:
            replacements.append(candidate)
        if len(replacements) >= max_alternatives:
            break
    return replacements


def build_texture_variations(term: str, max_alternatives: int) -> List[str]:
    pattern = re.compile(r"\\b(" + "|".join(TEXTURE_BASES) + r")\\b", re.IGNORECASE)
    match = pattern.search(term)
    if not match:
        return []
    original = match.group(1)
    original_lower = original.lower()
    replacements = []
    for texture in TEXTURE_BASES:
        if texture == original_lower:
            continue
        replacement_texture = _apply_color_case(texture, original)
        candidate = _replace_descriptor_word(term, pattern, replacement_texture)
        if candidate != term:
            replacements.append(candidate)
        if len(replacements) >= max_alternatives:
            break
    return replacements


def build_color_material_combinations(term: str, max_alternatives: int) -> List[str]:
    color_pattern = re.compile(r"\\b(" + "|".join(COLOR_BASES) + r")\\b", re.IGNORECASE)
    material_pattern = re.compile(r"\\b(" + "|".join(MATERIAL_BASES) + r")\\b", re.IGNORECASE)
    color_match = color_pattern.search(term)
    material_match = material_pattern.search(term)
    if not color_match and not material_match:
        return []

    replacements = []
    original_color = color_match.group(1) if color_match else None
    original_material = material_match.group(1) if material_match else None
    colors = [c for c in COLOR_BASES if c != (original_color.lower() if original_color else None)]
    materials = [m for m in MATERIAL_BASES if m != (original_material.lower() if original_material else None)]
    if not colors:
        colors = COLOR_BASES[:]
    if not materials:
        materials = MATERIAL_BASES[:]

    for color in colors:
        for material in materials:
            if len(replacements) >= max_alternatives:
                break
            candidate = term
            if material_match:
                candidate = _replace_descriptor_word(
                    candidate,
                    material_pattern,
                    _apply_color_case(material, original_material),
                )
            else:
                candidate = f"{material} {candidate}".strip()
            if color_match:
                candidate = _replace_descriptor_word(
                    candidate,
                    color_pattern,
                    _apply_color_case(color, original_color),
                )
            else:
                candidate = f"{color} {candidate}".strip()
            if candidate != term and candidate not in replacements:
                replacements.append(candidate)
        if len(replacements) >= max_alternatives:
            break
    return replacements


def build_texture_material_combinations(term: str, max_alternatives: int) -> List[str]:
    texture_pattern = re.compile(r"\\b(" + "|".join(TEXTURE_BASES) + r")\\b", re.IGNORECASE)
    material_pattern = re.compile(r"\\b(" + "|".join(MATERIAL_BASES) + r")\\b", re.IGNORECASE)
    texture_match = texture_pattern.search(term)
    material_match = material_pattern.search(term)
    if not texture_match or not material_match:
        return []

    replacements = []
    original_texture = texture_match.group(1)
    original_material = material_match.group(1)
    textures = [t for t in TEXTURE_BASES if t != original_texture.lower()]
    materials = [m for m in MATERIAL_BASES if m != original_material.lower()]
    if not textures:
        textures = TEXTURE_BASES[:]
    if not materials:
        materials = MATERIAL_BASES[:]

    for texture in textures:
        for material in materials:
            if len(replacements) >= max_alternatives:
                break
            candidate = term
            candidate = _replace_descriptor_word(
                candidate,
                material_pattern,
                _apply_color_case(material, original_material),
            )
            candidate = _replace_descriptor_word(
                candidate,
                texture_pattern,
                _apply_color_case(texture, original_texture),
            )
            if candidate != term and candidate not in replacements:
                replacements.append(candidate)
        if len(replacements) >= max_alternatives:
            break
    return replacements


def build_multicolor_variations(term: str, max_alternatives: int) -> List[str]:
    pattern = re.compile(r"\b(" + "|".join(COLOR_BASES) + r")\b", re.IGNORECASE)
    match = pattern.search(term)
    replacements = []
    if match:
        original = match.group(1)
        original_lower = original.lower()
        color_pairs = []
        for color in COLOR_BASES:
            if color == original_lower:
                continue
            pair = (original_lower, color)
            if pair not in color_pairs:
                color_pairs.append(pair)
        for first, second in color_pairs:
            if len(replacements) >= max_alternatives:
                break
            first_color = _apply_color_case(first, original)
            candidate = pattern.sub(f"{first_color} and {second}", term, count=1)
            if candidate != term and candidate not in replacements:
                replacements.append(candidate)
    else:
        seen_pairs = set()
        for i, first in enumerate(COLOR_BASES):
            for second in COLOR_BASES[i + 1:]:
                if len(replacements) >= max_alternatives:
                    break
                pair = (first, second)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                candidate = f"{first} and {second} {term}".strip()
                if candidate != term and candidate not in replacements:
                    replacements.append(candidate)
            if len(replacements) >= max_alternatives:
                break
    return replacements


def build_color_pair_variations(term: str, max_alternatives: int) -> List[str]:
    pattern = re.compile(r"\b(" + "|".join(COLOR_BASES) + r")\b", re.IGNORECASE)
    match = pattern.search(term)
    replacements = []
    seen = set()
    if match:
        original = match.group(1)
        for first in COLOR_BASES:
            for second in COLOR_BASES:
                if first == second:
                    continue
                pair = f"{_apply_color_case(first, original)} and {second}"
                candidate = pattern.sub(pair, term, count=1)
                if candidate != term and candidate not in seen:
                    replacements.append(candidate)
                    seen.add(candidate)
                if len(replacements) >= max_alternatives:
                    return replacements
    else:
        for first in COLOR_BASES:
            for second in COLOR_BASES:
                if first == second:
                    continue
                candidate = f"{first} and {second} {term}".strip()
                if candidate != term and candidate not in seen:
                    replacements.append(candidate)
                    seen.add(candidate)
                if len(replacements) >= max_alternatives:
                    return replacements
    return replacements


def is_vague_color_phrase(text: str) -> bool:
    pattern = re.compile(r"\\b(" + "|".join(VAGUE_COLOR_MODIFIERS) + r")\\b", re.IGNORECASE)
    return bool(pattern.search(text))


def _normalize_replacement_phrase(text: str) -> str:
    if not text:
        return text
    cleaned = text
    colorish_pattern = re.compile(
        r"\\b(" + "|".join(COLORISH_BASES) + r")ish\\b",
        re.IGNORECASE,
    )

    def colorish_repl(match: re.Match) -> str:
        base = match.group(1)
        if match.group(0).isupper():
            return base.upper()
        if match.group(0)[0].isupper():
            return base.capitalize()
        return base

    cleaned = colorish_pattern.sub(colorish_repl, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s+", ", ", cleaned)
    return cleaned.strip(" ,")


def _contains_vague_modifier(text: str) -> bool:
    pattern = re.compile(
        r"\\b(" + "|".join(VAGUE_MODIFIERS + VAGUE_COLOR_MODIFIERS + VAGUE_TEXTURE_MODIFIERS) + r")\\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))

def _collect_word_matches(text: str, vocab: List[str]) -> Set[str]:
    if not text:
        return set()
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, vocab)) + r")\b", re.IGNORECASE)
    return {m.group(1).lower() for m in pattern.finditer(text)}


def _newly_introduced_modifiers(source: str, candidate: str, vocab: List[str]) -> Set[str]:
    """Return modifiers that appear in candidate but not in source (case-insensitive, word-boundary match)."""
    src = _collect_word_matches(source, vocab)
    cand = _collect_word_matches(candidate, vocab)
    return cand - src


def _has_new_vague_color_modifiers(source: str, candidate: str) -> bool:
    if not STEP3_ALLOW_SOURCE_VAGUE_MODIFIERS:
        return is_vague_color_phrase(candidate)
    return bool(_newly_introduced_modifiers(source, candidate, VAGUE_COLOR_MODIFIERS))


def _has_new_vague_modifiers(source: str, candidate: str) -> bool:
    vocab = VAGUE_MODIFIERS + VAGUE_COLOR_MODIFIERS + VAGUE_TEXTURE_MODIFIERS
    if not STEP3_ALLOW_SOURCE_VAGUE_MODIFIERS:
        return _contains_vague_modifier(candidate)
    return bool(_newly_introduced_modifiers(source, candidate, vocab))


def build_hyphen_and_color_variations(term: str, max_alternatives: int) -> List[str]:
    """Handle patterns like 'yellow-and-black' (hyphenated) in a controlled way."""
    colors = "|".join(map(re.escape, COLOR_BASES))
    pattern = re.compile(rf"\b({colors})-and-({colors})\b", re.IGNORECASE)
    m = pattern.search(term)
    if not m:
        return []
    c1_raw, c2_raw = m.group(1), m.group(2)
    c1, c2 = c1_raw.lower(), c2_raw.lower()
    out: List[str] = []
    for first in COLOR_BASES:
        for second in COLOR_BASES:
            if first == c1 and second == c2:
                continue
            if len(out) >= max_alternatives:
                return out
            first_c = _apply_color_case(first, c1_raw)
            second_c = _apply_color_case(second, c2_raw)
            repl = f"{first_c}-and-{second_c}"
            cand = term[:m.start()] + repl + term[m.end():]
            if cand != term and cand not in out:
                out.append(cand)
    return out


def _step3_rule_backfill_replacements(term: str, max_needed: int) -> List[str]:
    """Deterministic fallback: generate color/material/texture variations when LLM yields too few usable options."""
    if max_needed <= 0:
        return []
    max_gen = max(16, max_needed * 4)
    pool: List[str] = []
    # Prefer structured variations first.
    pool.extend(build_hyphen_and_color_variations(term, max_gen))
    pool.extend(build_multicolor_variations(term, max_gen))
    pool.extend(build_color_pair_variations(term, max_gen))
    pool.extend(build_color_variations(term, max_gen))
    pool.extend(build_material_variations(term, max_gen))
    pool.extend(build_texture_variations(term, max_gen))
    pool.extend(build_color_material_combinations(term, max_gen))
    pool.extend(build_texture_material_combinations(term, max_gen))

    out: List[str] = []
    for cand in pool:
        normalized = _normalize_replacement_phrase(str(cand).strip())
        normalized = _ensure_materials(term, normalized)
        if not normalized or normalized == term or normalized in out:
            continue
        if _has_new_vague_color_modifiers(term, normalized):
            continue
        if _has_new_vague_modifiers(term, normalized):
            continue
        if any(keyword in normalized.lower() for keyword in ROBOT_KEYWORDS):
            continue
        out.append(normalized)
        if len(out) >= max_needed:
            break
    return out


def _retry_replacements_for_term(
    model,
    processor,
    term: str,
    needed: int,
    issues: List[str],
) -> List[str]:
    issue_block = "\n".join(f"- {issue}" for issue in issues) if issues else "- Prior output did not meet constraints."
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": REPLACEMENT_RETRY_PROMPT_TEMPLATE.format(
                term=term,
                num_alternatives=needed,
                issues=issue_block,
            ),
        }],
    }]
    raw = run_generation(model, processor, messages, 500, True, 0.9, 0.95)
    parsed = safe_parse_json(raw)
    alts = parsed.get("alternatives", []) if isinstance(parsed, dict) else []
    return [alt for alt in alts if isinstance(alt, str)]


def _split_color_bias(term: str, candidates: List[str]) -> List[str]:
    pattern = re.compile(r"\\b(" + "|".join(COLOR_BASES) + r")\\b", re.IGNORECASE)
    match = pattern.search(term)
    if not match:
        return candidates
    original_color = match.group(1).lower()
    without = [c for c in candidates if original_color not in c.lower()]
    with_original = [c for c in candidates if original_color in c.lower()]
    return without + with_original


def step1_5_postprocess(paragraph: str) -> str:
    updated = enforce_container_initial_state(paragraph)
    updated = cleanup_vague_terms(updated)
    return updated


def apply_random_replacements(
    bracketed_prompt: str,
    replacements: Dict[str, List[str]],
    seed: int,
    min_terms: int = 1,
    max_terms: int = 3,
    replace_all_terms: bool = False,
) -> str:
    rng = random.Random(seed)
    terms = extract_bracket_terms(bracketed_prompt)

    if not terms:
        return bracketed_prompt.replace("【", "").replace("】", "")

    if replace_all_terms:
        selected = set(terms)
    else:
        max_allowed = min(max_terms, len(terms))
        min_allowed = min(min_terms, max_allowed)
        k = rng.randint(min_allowed, max_allowed) if max_allowed > 0 else 0
        selected = set(rng.sample(terms, k)) if k > 0 else set()

    # --- Consistency across same-object terms (e.g., 'white rectangular table' vs 'table surface') ---
    # If multiple selected terms share a salient object token, choose ONE variant index for that group
    # (anchored by the earliest mention in the paragraph) and apply aligned variants consistently.
    base_no_brackets = bracketed_prompt.replace("【", "").replace("】", "")
    group_map = _shared_object_groups(terms)  # term -> shared object token

    chosen_replacements: Dict[str, str] = {}

    group_to_terms: Dict[str, List[str]] = {}
    for term in selected:
        group_key = group_map.get(term)
        if group_key:
            group_to_terms.setdefault(group_key, []).append(term)

    for group_key, group_terms in group_to_terms.items():
        if len(group_terms) < 2:
            continue

        # Anchor: earliest occurrence among the *selected* terms (no extra replacements introduced).
        def first_pos(t: str) -> int:
            pos = base_no_brackets.find(t)
            return pos if pos >= 0 else 10**9

        anchor_term = min(group_terms, key=first_pos)

        anchor_choices = replacements.get(anchor_term, []) or []
        if not anchor_choices:
            continue

        idx = rng.randrange(len(anchor_choices))
        anchor_choice = anchor_choices[idx]
        chosen_replacements[anchor_term] = _ensure_materials(anchor_term, anchor_choice)

        for term in group_terms:
            if term == anchor_term:
                continue

            term_choices = replacements.get(term, []) or []
            if term_choices:
                chosen = term_choices[idx] if idx < len(term_choices) else rng.choice(term_choices)
            else:
                chosen = _align_term_with_anchor(term, anchor_choice, group_key)

            chosen_replacements[term] = _ensure_materials(term, chosen)

    def repl(match: re.Match) -> str:
        src = match.group(1).strip()
        if src not in selected:
            return src
        if src in chosen_replacements:
            return chosen_replacements[src]
        choices = replacements.get(src, [])
        chosen = rng.choice(choices) if choices else src
        chosen_replacements[src] = chosen
        return chosen

    out = re.sub(r"【([\s\S]*?)】", repl, bracketed_prompt)
    return out.replace("【", "").replace("】", "")



@torch.no_grad()


@torch.no_grad()
def run_generation(model, processor, messages, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs

    inputs = processor(**kwargs).to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p})

    outputs = model.generate(**inputs, **gen_kwargs)
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def load_model_and_processor(model_type: str, size: str, variant: str):
    model_id = pick_model_id(size, variant, model_type)
    print(f"[INFO] Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if model_type == "Qwen3":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", dtype=dtype, trust_remote_code=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", dtype=dtype, trust_remote_code=True
        )
    model.eval()
    return model, processor


# =======================
# 4) Step APIs
# =======================
def step1_generate_base(model, processor, video_path: str, fps: float) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": fps},
            {"type": "text", "text": COSMOS_STYLE_PROMPT},
        ],
    }]
    base_prompt = run_generation(model, processor, messages, 500, False, 1.0, 1.0)
    return canonicalize_robot_sentences(base_prompt)


def step2_prepare_clause_context(model, processor, base_prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": PRE2_SPLIT_PROMPT_TEMPLATE.format(base_prompt=base_prompt)}],
    }]
    raw = run_generation(model, processor, messages, 400, False, 1.0, 1.0)
    parsed = safe_parse_json(raw)
    paragraph = parsed.get("paragraph") if isinstance(parsed, dict) else None
    if not isinstance(paragraph, str):
        return base_prompt
    normalized = paragraph.replace(",", "").replace(" ", "")
    original = base_prompt.replace(",", "").replace(" ", "")
    if normalized != original:
        return base_prompt
    return paragraph


def step2_bracket_base(model, processor, base_prompt: str) -> str:
    clause_context = step2_prepare_clause_context(model, processor, base_prompt)
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": DESCRIPTOR_EXTRACT_PROMPT_TEMPLATE.format(base_prompt=base_prompt)}],
    }]
    raw = run_generation(model, processor, messages, 500, False, 1.0, 1.0)
    parsed = safe_parse_json(raw)
    descriptors = parsed.get("descriptors", []) if isinstance(parsed, dict) else []
    cleaned = []
    for descriptor in descriptors or []:
        if not isinstance(descriptor, str):
            continue
        term = descriptor.strip()
        if not term or term not in base_prompt:
            continue
        if _in_robot_context(term, clause_context):
            continue
        cleaned.append(term)
    cleaned = list(dict.fromkeys(cleaned))

    spans = _collect_existing_spans(base_prompt, cleaned)
    merged = _merge_spans(spans)
    bracketed = _insert_brackets(base_prompt, merged)
    bracketed = _reflect_brackets(model, processor, base_prompt, bracketed)
    return _expand_shared_noun_phrases(base_prompt, bracketed)


def step3_generate_replacements_json(
    model,
    processor,
    bracketed_prompt: str,
    alternatives_per_term: int = 64,
) -> Dict[str, List[str]]:
    """Generate a per-descriptor replacement list (Step3).
    """
    def _normalize_replacements(parsed_json) -> Dict[str, List[str]]:
        # Case 1: model returns {"replacements": {...}} or {term: [cand...], ...}
        if isinstance(parsed_json, dict):
            if "replacements" in parsed_json:
                rr = parsed_json["replacements"] or {}
                return rr if isinstance(rr, dict) else {}
            if all(isinstance(v, list) for v in parsed_json.values()):
                return parsed_json
            raise ValueError(f"Unexpected dict schema keys={list(parsed_json.keys())[:20]}")

        # Case 2: model returns a top-level list
        if isinstance(parsed_json, list):
            repl_map: Dict[str, List[str]] = {}
            for item in parsed_json:
                if isinstance(item, dict):
                    term = item.get("term") or item.get("source") or item.get("src") or item.get("key")
                    cands = (
                        item.get("replacements")
                        or item.get("candidates")
                        or item.get("alts")
                        or item.get("options")
                        or item.get("values")
                    )
                    if term and isinstance(cands, list):
                        repl_map[str(term)] = cands
                        continue
                    if "replacements" in item and isinstance(item["replacements"], dict):
                        for k, v in item["replacements"].items():
                            if isinstance(v, list):
                                repl_map[str(k)] = v
                        continue
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    term = str(item[0])
                    v = item[1]
                    repl_map[term] = v if isinstance(v, list) else [str(v)]
                    continue
            if repl_map:
                return repl_map
            raise ValueError("Parsed JSON is a list, but cannot normalize to replacements mapping.")

        raise TypeError(f"Unexpected parsed type: {type(parsed_json)}")

    bracketed_prompt = expand_bracketed_prompt(bracketed_prompt)
    terms = extract_bracket_terms(bracketed_prompt)
    terms = [t for t in terms if not any(keyword in t.lower() for keyword in ROBOT_KEYWORDS)]
    if not terms:
        return {}

    normalized_terms = {normalize_term_key(t): t for t in terms}
    canonical_terms = list(normalized_terms.values())
    shared_groups = _shared_object_groups(canonical_terms)

    block = "\n".join([f"- {t}" for t in canonical_terms])
    # -----------------------
    # Produce raw LLM replacements
    # -----------------------
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": REPLACEMENT_JSON_PROMPT_TEMPLATE.format(
                descriptor_block=block,
                num_alternatives=alternatives_per_term,
            ),
        }],
    }]
    raw = run_generation(model, processor, messages, 800, True, 0.9, 0.95)
    parsed = safe_parse_json(raw)
    raw_replacements = _normalize_replacements(parsed)

    # Normalize keys to reduce "zero replacements" caused by punctuation/hyphen differences.
    raw_key_map: Dict[str, str] = {}
    for k in raw_replacements.keys():
        if isinstance(k, str):
            raw_key_map[normalize_term_key(k)] = k

    # -----------------------
    # Parent/group alignment setup
    # -----------------------
    ordered_keys = sorted(normalized_terms.keys(), key=len, reverse=True)

    # Map child -> parent by normalized key inclusion (e.g., "blue cap" inside "blue plastic bottle cap")
    parent_map: Dict[str, str] = {}
    for key in ordered_keys:
        for other_key in ordered_keys:
            if key == other_key:
                continue
            if key in other_key:
                parent_map[key] = other_key
                break

    group_anchor: Dict[str, str] = {}
    for term in canonical_terms:
        group_key = shared_groups.get(term)
        if not group_key:
            continue
        current = group_anchor.get(group_key)
        if current is None or len(term.split()) > len(current.split()):
            group_anchor[group_key] = term

    # -----------------------
    # Clean + backfill per term
    # -----------------------
    merged: Dict[str, List[str]] = {}
    for key in ordered_keys:
        term = normalized_terms[key]
        raw_key = raw_key_map.get(key)
        model_alts = raw_replacements.get(raw_key, []) if raw_key else raw_replacements.get(term, []) or []

        cleaned_alts: List[str] = []
        issues = set()

        def _accept_candidate(src_term: str, candidate: str) -> bool:
            if not candidate or candidate == src_term or candidate in cleaned_alts:
                return False
            if _has_new_vague_color_modifiers(src_term, candidate):
                issues.add("Some outputs contained newly-introduced vague color modifiers.")
                return False
            if _has_new_vague_modifiers(src_term, candidate):
                issues.add("Some outputs contained newly-introduced vague texture/modifier terms.")
                return False
            if any(keyword in candidate.lower() for keyword in ROBOT_KEYWORDS):
                issues.add("Some outputs included robot-related terms.")
                return False
            return True

        for alt in model_alts:
            if not isinstance(alt, str):
                continue
            normalized = _normalize_replacement_phrase(alt.strip())
            normalized = _ensure_materials(term, normalized)
            if not normalized:
                continue
            if normalized == term:
                issues.add("Some outputs repeated the original phrase.")
                continue
            if _accept_candidate(term, normalized):
                cleaned_alts.append(normalized)
            if len(cleaned_alts) >= alternatives_per_term:
                break

        cleaned_alts = _split_color_bias(term, cleaned_alts)

        if len(cleaned_alts) < alternatives_per_term and issues:
            needed = alternatives_per_term - len(cleaned_alts)
            retry_alts = _retry_replacements_for_term(model, processor, term, needed, sorted(issues))
            for alt in retry_alts:
                if not isinstance(alt, str):
                    continue
                normalized = _normalize_replacement_phrase(alt.strip())
                normalized = _ensure_materials(term, normalized)
                if _accept_candidate(term, normalized):
                    cleaned_alts.append(normalized)
                if len(cleaned_alts) >= alternatives_per_term:
                    break

        if STEP3_ENABLE_RULE_BACKFILL and len(cleaned_alts) < alternatives_per_term:
            needed = alternatives_per_term - len(cleaned_alts)
            for bf in _step3_rule_backfill_replacements(term, needed):
                if _accept_candidate(term, bf):
                    cleaned_alts.append(bf)
                if len(cleaned_alts) >= alternatives_per_term:
                    break

        if not cleaned_alts:
            cleaned_alts = [_normalize_replacement_phrase(term) or term]

        # Parent alignment: if a shorter phrase is a substring of a longer phrase, align by suffix tokens.
        parent_norm = parent_map.get(key)
        parent_term = normalized_terms.get(parent_norm) if parent_norm else None
        if parent_term and parent_term in merged:
            parent_alts = merged[parent_term]
            child_tokens = term.split()
            aligned: List[str] = []
            for alt in parent_alts:
                alt_tokens = str(alt).split()
                if len(alt_tokens) >= len(child_tokens):
                    aligned.append(" ".join(alt_tokens[-len(child_tokens):]))
                else:
                    aligned.append(str(alt))
            cleaned_alts = aligned
        else:
            # Shared-object grouping: enforce consistent color/material across grouped phrases.
            group_key = shared_groups.get(term)
            anchor_term = group_anchor.get(group_key) if group_key else None
            if anchor_term and anchor_term != term and anchor_term in merged:
                anchor_alts = merged[anchor_term]
                cleaned_alts = [
                    _ensure_materials(term, _align_term_with_anchor(term, alt, group_key))
                    for alt in anchor_alts
                ]

        merged[term] = cleaned_alts[:alternatives_per_term]

    return merged

def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =======================
# 5) CLI
# =======================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_model_args(p):
        p.add_argument("--size", default="4B", choices=["2B", "4B", "8B", "32B"])
        p.add_argument("--variant", default="instruct", choices=["instruct", "awq"])
        p.add_argument("--model_type", default="Qwen3", choices=["Qwen2.5", "Qwen3"])

    # step1
    p1 = sub.add_parser("step1_base")
    add_model_args(p1)
    p1.add_argument("--video", required=True)
    p1.add_argument("--fps", type=float, default=4.0)
    p1.add_argument("--out", required=True)

    # step2
    p2 = sub.add_parser("step2_bracket")
    add_model_args(p2)
    p2.add_argument("--base_in", required=True)
    p2.add_argument("--out", required=True)

    # step2+3 combined
    p23 = sub.add_parser("step23")
    add_model_args(p23)
    p23.add_argument("--base_in", required=True)
    p23.add_argument("--out_bracket", required=True)
    p23.add_argument("--out_repl_json", required=True)
    p23.add_argument("--replacements_per_term", type=int, default=64)

    # step3a
    p3 = sub.add_parser("step3_replacements")
    add_model_args(p3)
    p3.add_argument("--bracket_in", required=True)
    p3.add_argument("--out_json", required=True)
    p3.add_argument("--replacements_per_term", type=int, default=64)

    # step3b apply
    p4 = sub.add_parser("step3_apply")
    p4.add_argument("--bracket_in", required=True)
    p4.add_argument("--repl_json", required=False, help="If omitted, will apply without changes (only remove brackets).")
    p4.add_argument("--seed", type=int, required=True)
    p4.add_argument("--min_replace_terms", type=int, default=1)
    p4.add_argument("--max_replace_terms", type=int, default=3)
    p4.add_argument("--replace_all_terms", action="store_true", help="Replace all bracketed terms instead of sampling a subset.")
    p4.add_argument("--out", required=True)

    # full pipeline
    p5 = sub.add_parser("all")
    add_model_args(p5)
    p5.add_argument("--video", required=True)
    p5.add_argument("--fps", type=float, default=4.0)
    p5.add_argument("--seed", type=int, required=False)
    p5.add_argument("--min_replace_terms", type=int, default=1)
    p5.add_argument("--max_replace_terms", type=int, default=3)
    p5.add_argument("--replace_all_terms", action="store_true", help="Replace all bracketed terms instead of sampling a subset.")
    p5.add_argument("--replacements_per_term", type=int, default=64)
    p5.add_argument("--out_base", required=True)
    p5.add_argument("--out_bracket", required=True)
    p5.add_argument("--out_repl_json", required=True)
    p5.add_argument("--out_final", required=True)

    args = parser.parse_args()

    if args.cmd == "step3_apply":
        bracketed = read_text(args.bracket_in)
        replacements = read_json(args.repl_json).get("replacements", {}) if args.repl_json else {}
        final = apply_random_replacements(
            bracketed,
            replacements,
            seed=args.seed,
            min_terms=args.min_replace_terms,
            max_terms=args.max_replace_terms,
            replace_all_terms=args.replace_all_terms,
        )
        write_text(args.out, final)
        print("[INFO] Wrote:", args.out)
        return

    # other commands need model
    model, processor = load_model_and_processor(args.model_type, args.size, args.variant)

    if args.cmd == "step1_base":
        base = step1_generate_base(model, processor, args.video, args.fps)
        print("[INFO] Step 1.5: post-processing base prompt")
        base = step1_5_postprocess(base)
        write_text(args.out, base)
        print("[INFO] Wrote:", args.out)
        return

    if args.cmd == "step2_bracket":
        base = read_text(args.base_in)
        bracketed = step2_bracket_base(model, processor, base)
        write_text(args.out, bracketed)
        print("[INFO] Wrote:", args.out)
        return

    if args.cmd == "step23":
        base = read_text(args.base_in)
        bracketed = step2_bracket_base(model, processor, base)
        bracketed = expand_bracketed_prompt(bracketed)
        write_text(args.out_bracket, bracketed)

        repl = step3_generate_replacements_json(
            model,
            processor,
            bracketed,
            alternatives_per_term=args.replacements_per_term,
        )
        write_json(args.out_repl_json, {"replacements": repl})

        print("[INFO] Wrote:", args.out_bracket, args.out_repl_json)
        return

    if args.cmd == "step3_replacements":
        bracketed = read_text(args.bracket_in)
        repl = step3_generate_replacements_json(
            model,
            processor,
            bracketed,
            alternatives_per_term=args.replacements_per_term,
        )
        write_json(args.out_json, {"replacements": repl})
        print("[INFO] Wrote:", args.out_json)
        return

    if args.cmd == "all":
        if args.seed is None:
            args.seed = random.randrange(2**32)

        base = step1_generate_base(model, processor, args.video, args.fps)
        print("[INFO] Step 1.5: post-processing base prompt")
        base = step1_5_postprocess(base)
        write_text(args.out_base, base)

        bracketed = step2_bracket_base(model, processor, base)
        bracketed = expand_bracketed_prompt(bracketed)
        write_text(args.out_bracket, bracketed)

        repl = step3_generate_replacements_json(
            model,
            processor,
            bracketed,
            alternatives_per_term=args.replacements_per_term,
        )
        write_json(args.out_repl_json, {"replacements": repl})

        final = apply_random_replacements(
            bracketed,
            repl,
            seed=args.seed,
            min_terms=args.min_replace_terms,
            max_terms=args.max_replace_terms,
            replace_all_terms=args.replace_all_terms,
        )
        write_text(args.out_final, final)

        print("[INFO] Wrote:", args.out_base, args.out_bracket, args.out_repl_json, args.out_final)
        return


if __name__ == "__main__":
    main()

import os
import re
import json
import random
import argparse
from typing import Dict, List, Any

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

REPLACEMENT_JSON_PROMPT_TEMPLATE = """Generate prompt variations.
For each descriptor below (COLOR or TEXTURE), propose {num_alternatives} short alternatives.

Rules:
- Do NOT repeat the original.
- Keep 1-3 words.
- Use common, simple descriptors suitable for indoor robotics scenes.
- For COLOR terms, alternatives MUST change the color (not just texture) and should vary the palette.
- For material/texture phrases, include swaps and combinations that mix color + material or texture + material when reasonable.
- Avoid rare or poetic color words.
- Do NOT use vague or qualifier words like "pale", "light", "dark", "soft", "muted", "faint", "slight", "subtle".
- Output STRICT JSON only:
  {{"replacements": {{"<src>": ["alt1", ...], ...}}}}

DESCRIPTORS:
{descriptor_block}
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
    "textured",
    "woven",
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
DESCRIPTOR_WORDS = sorted({
    *COLOR_BASES,
    "smooth",
    "matte",
    "glossy",
    "textured",
    "woven",
    "rough",
    "soft",
    "hard",
    "metallic",
    "metal",
    "wooden",
    "plastic",
    "rubber",
    "ceramic",
    "glass",
    "striped",
    "patterned",
    "ribbed",
    "knitted",
    "wrinkled",
    "transparent",
})
SPAN_STOPWORDS = {
    "and",
    "or",
    "with",
    "into",
    "onto",
    "over",
    "above",
    "below",
    "under",
    "in",
    "on",
    "at",
    "to",
    "of",
    "from",
    "by",
    "for",
    "as",
    "while",
    "then",
}

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
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def extract_bracket_terms(text: str) -> List[str]:
    raw_terms = [t.strip() for t in re.findall(r"【(.*?)】", text)]
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


def _in_robot_context(span: str, paragraph: str, window: int = 200) -> bool:
    for match in re.finditer(re.escape(span), paragraph):
        start, end = match.span()
        ctx_start = max(0, start - window)
        ctx_end = min(len(paragraph), end + window)
        context = paragraph[ctx_start:ctx_end].lower()
        if any(keyword in context for keyword in ROBOT_KEYWORDS):
            return True
    return False


def filter_robot_descriptors(bracketed_prompt: str, window: int = 200) -> List[str]:
    base_paragraph = bracketed_prompt.replace("【", "").replace("】", "")
    descriptors = extract_bracket_terms(bracketed_prompt)
    filtered = []
    for descriptor in descriptors:
        if _in_robot_context(descriptor, base_paragraph, window):
            continue
        filtered.append(descriptor)
    return filtered


def strip_robot_brackets(bracketed_prompt: str) -> str:
    def repl(match: re.Match) -> str:
        term = match.group(1)
        lowered = term.lower()
        if any(keyword in lowered for keyword in ROBOT_KEYWORDS):
            return term
        return match.group(0)

    return re.sub(r"【(.*?)】", repl, bracketed_prompt)


def _trim_descriptor_span(span: str) -> str:
    tokens = span.split()
    if not tokens:
        return span
    kept = [tokens[0]]
    for tok in tokens[1:]:
        clean = tok.strip(",.;:")
        lower = clean.lower()
        if lower in SPAN_STOPWORDS:
            break
        if lower in ROBOT_KEYWORDS:
            break
        kept.append(clean)
        if tok != clean:
            break
    return " ".join(kept).strip()


def _collect_descriptor_spans(paragraph: str, max_follow_tokens: int = 3) -> List[tuple]:
    if not DESCRIPTOR_WORDS:
        return []
    pattern = re.compile(
        r"\\b(" + "|".join(map(re.escape, DESCRIPTOR_WORDS)) + r")\\b(?:\\s+[\\w-]+){0," + str(max_follow_tokens) + r"}",
        re.IGNORECASE,
    )
    spans = []
    for match in pattern.finditer(paragraph):
        raw = match.group(0)
        trimmed = _trim_descriptor_span(raw)
        if not trimmed:
            continue
        if _in_robot_context(trimmed, paragraph):
            continue
        if any(keyword in trimmed.lower().split() for keyword in ROBOT_KEYWORDS):
            continue
        start = match.start()
        end = start + len(trimmed)
        spans.append((start, end))
    return spans


def _collect_existing_spans(paragraph: str, terms: List[str]) -> List[tuple]:
    spans = []
    for term in terms:
        if not term:
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


def auto_bracket_descriptors(base_paragraph: str, bracketed_prompt: str) -> str:
    normalized = bracketed_prompt.replace("【", "").replace("】", "")
    if normalized != base_paragraph:
        print("[WARN] Step 2: bracket output mismatch; falling back to base prompt for bracketing.")
        bracketed_prompt = base_paragraph
    existing_terms = extract_bracket_terms(bracketed_prompt)
    spans = _collect_existing_spans(base_paragraph, existing_terms)
    spans += _collect_descriptor_spans(base_paragraph)
    merged = _merge_spans(spans)
    return _insert_brackets(base_paragraph, merged)


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


def is_vague_color_phrase(text: str) -> bool:
    pattern = re.compile(r"\\b(" + "|".join(VAGUE_COLOR_MODIFIERS) + r")\\b", re.IGNORECASE)
    return bool(pattern.search(text))


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

    chosen_replacements: Dict[str, str] = {}

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

    out = re.sub(r"【(.*?)】", repl, bracketed_prompt)
    return out.replace("【", "").replace("】", "")


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


def step2_bracket_base(model, processor, base_prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": BRACKET_PROMPT_TEMPLATE.format(base_prompt=base_prompt)}],
    }]
    bracketed = run_generation(model, processor, messages, 700, False, 1.0, 1.0)
    bracketed = strip_robot_brackets(bracketed)
    return auto_bracket_descriptors(base_prompt, bracketed)


def step3_generate_replacements_json(
    model,
    processor,
    bracketed_prompt: str,
    alternatives_per_term: int = 64,
) -> Dict[str, List[str]]:
    terms = extract_bracket_terms(bracketed_prompt)
    if not terms:
        return {}

    block = "\n".join([f"- {t}" for t in terms])
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
    replacements = safe_parse_json(raw).get("replacements", {}) or {}

    merged: Dict[str, List[str]] = {}
    for term in terms:
        model_alts = replacements.get(term, []) or []
        color_alts = build_color_variations(term, alternatives_per_term)
        material_alts = build_material_variations(term, alternatives_per_term)
        texture_alts = build_texture_variations(term, alternatives_per_term)
        combo_alts = build_color_material_combinations(term, alternatives_per_term)
        texture_combo_alts = build_texture_material_combinations(term, alternatives_per_term)
        multicolor_alts = build_multicolor_variations(term, alternatives_per_term)
        combined = []
        for alt in (
            model_alts
            + color_alts
            + material_alts
            + texture_alts
            + combo_alts
            + texture_combo_alts
            + multicolor_alts
        ):
            if not alt or alt == term or alt in combined or is_vague_color_phrase(alt):
                continue
            combined.append(alt)
            if len(combined) >= alternatives_per_term:
                break
        merged[term] = combined

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

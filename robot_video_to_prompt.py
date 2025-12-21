import os
import re
import json
import random
import argparse
from typing import Dict, List, Any

import torch
from transformers import AutoProcessor

# Qwen2.5‑VL 模型类
from transformers import Qwen2_5_VLForConditionalGeneration

# Qwen3‑VL 模型类 （官方 transformers 支持）
from transformers import Qwen3VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

# =======================
# 1) Prompts
# =======================
COSMOS_STYLE_PROMPT = """# You are a video prompt generator.
# Write a single English paragraph describing the video in a natural, fluent, demonstration-style narrative.
# The writing style should resemble an official robotics demonstration description:
# factual, neutral in tone, continuous, and focused on clearly describing the scene, setup, and manipulation process.
# The paragraph should read smoothly rather than like a checklist, while still remaining precise and unambiguous.
#
# CRITICAL TEMPORAL CONSTRAINT:
# - Do not assume the presence of any manipulated object at the beginning unless it is explicitly visible at the start.
# - If an object appears later in the video, describe its entry as a distinct event before any interaction.
#
# Scene description rules:
# - Begin with a brief overview of the setting, as if documenting a laboratory demonstration.
# - Describe static background elements first from an overhead or observer viewpoint.
# - Describe each static object in a separate, standalone sentence.
# - Identify the robot arm(s), including materials, colors, and their positions relative to the scene.
# - Identify any container and clearly state its initial state (empty or occupied).
#
# Action description rules:
# - Use natural temporal phrasing (At the beginning…, As the video progresses…, Then…, Finally…).
# - Introduce manipulated objects only at the moment they first appear.
# - Do not imply grasping, lifting, or containment before the object’s appearance is explicitly described.
# - Describe motions and interactions in a clear, observational manner, without speculation or intent.
#
# Do not mention models, frames, timestamps, or reasoning steps.
# Output only the paragraph.
#
# Now generate the paragraph for the given video.
"""

BRACKET_PROMPT_TEMPLATE = """You are a formatting tool.
Your task is ONLY to insert Chinese corner brackets 【】 around existing spans in the paragraph.

STRICT RULES:
- DO NOT translate, paraphrase, rewrite, reorder, or delete ANY characters.
- The output must be IDENTICAL to the input paragraph after removing all 【】.
- The text inside each 【】 MUST be an exact, verbatim substring copied from the paragraph.
- If unsure whether a span describes color/material/texture/finish/surface, leave it unbracketed.

Example:
Input:  "a light-colored sofa with an orange cushion"
Output: "a 【light-colored】 sofa with an 【orange】 cushion"

PARAGRAPH:
{base_prompt}
"""


REPLACEMENT_JSON_PROMPT_TEMPLATE = """You are helping create prompt variations.
For each descriptor in the list below (each is a word or short phrase describing color/material/texture/finish),
propose 6 alternative descriptors of the SAME TYPE that are plausible in a robotics manipulation scene description.

Rules:
- Do NOT repeat the original descriptor.
- Keep each alternative short (1-5 words).
- Output STRICT JSON only, exactly in this schema:
  {{"replacements": {{"<src>": ["alt1","alt2","alt3","alt4","alt5","alt6"], ...}}}}

DESCRIPTORS:
{descriptor_block}
"""


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
def add_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return root + suffix + (ext or "")


def pick_model_id(size: str, variant: str, model_type: str) -> str:
    size = size.upper()
    variant = variant.lower()

    if model_type == "Qwen3":
        model_dict = QWEN3_VL_MODELS
    else:
        model_dict = QWEN25_VL_MODELS

    if size not in model_dict:
        raise ValueError(f"Unsupported --size {size}")
    if variant not in model_dict[size]:
        raise ValueError(f"Unsupported --variant {variant}")

    return model_dict[size][variant]


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def extract_bracket_terms(text: str) -> List[str]:
    return list({t.strip() for t in re.findall(r"【(.*?)】", text)})


def apply_random_replacements(bracketed_prompt: str, replacements: Dict[str, List[str]], seed: int) -> str:
    rng = random.Random(seed)

    def repl(match: re.Match) -> str:
        src = match.group(1).strip()
        choices = replacements.get(src, [])
        return rng.choice(choices) if choices else src

    return re.sub(r"【(.*?)】", repl, bracketed_prompt).replace("【", "").replace("】", "")


@torch.no_grad()
def run_generation(model, processor, messages, max_new_tokens: int, do_sample: bool, temperature: float,
                   top_p: float) -> str:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", default="4B", choices=["2B", "4B", "8B", "32B"])
    parser.add_argument("--variant", default="instruct", choices=["instruct", "awq"])
    parser.add_argument("--model_type", default="Qwen3", choices=["Qwen2.5", "Qwen3"])
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randrange(2 ** 32)

    model_id = pick_model_id(args.size, args.variant, args.model_type)
    print(f"[INFO] Loading model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if args.model_type == "Qwen3":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

    model.eval()

    # STEP1: Base prompt
    messages1 = [{"role": "user", "content": [{"type": "video", "video": f"file://{args.video}", "fps": args.fps},
                                              {"type": "text", "text": COSMOS_STYLE_PROMPT}]}]
    base = run_generation(model, processor, messages1, 500, False, 1.0, 1.0)
    print("STEP1 BASE:", base)

    # STEP2: Bracketed prompt
    messages2 = [
        {"role": "user", "content": [{"type": "text", "text": BRACKET_PROMPT_TEMPLATE.format(base_prompt=base)}]}]
    bracketed = run_generation(model, processor, messages2, 700, False, 1.0, 1.0)
    print("STEP2 BRACKETED:", bracketed)

    # Extract bracket terms
    terms = extract_bracket_terms(bracketed)

    replacements = {}
    if terms:
        block = "\n".join([f"- {t}" for t in terms])
        messages3 = [{"role": "user", "content": [
            {"type": "text", "text": REPLACEMENT_JSON_PROMPT_TEMPLATE.format(descriptor_block=block)}]}]
        raw = run_generation(model, processor, messages3, 800, True, 0.9, 0.95)
        replacements = safe_parse_json(raw).get("replacements", {})

    final = apply_random_replacements(bracketed, replacements, args.seed)
    print("STEP3 FINAL:", final)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(final)


if __name__ == "__main__":
    main()

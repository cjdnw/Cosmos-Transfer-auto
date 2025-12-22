import os
import re
import json
import random
import argparse
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
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
# - Constrain each sentence to a SINGLE texture/material/finish descriptor; if multiple objects have textures, split them into separate sentences to keep one texture description per sentence.
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
    return list({t.strip() for t in re.findall(r"【(.*?)】", text)})


def apply_random_replacements(
    bracketed_prompt: str,
    replacements: Dict[str, List[str]],
    seed: int,
    min_terms: int = 1,
    max_terms: int = 3,
) -> str:
    rng = random.Random(seed)
    terms = list({t.strip() for t in re.findall(r"【(.*?)】", bracketed_prompt)})

    if not terms:
        return bracketed_prompt.replace("【", "").replace("】", "")

    max_allowed = min(max_terms, len(terms))
    min_allowed = min(min_terms, max_allowed)
    k = rng.randint(min_allowed, max_allowed) if max_allowed > 0 else 0
    selected = set(rng.sample(terms, k)) if k > 0 else set()

    def repl(match: re.Match) -> str:
        src = match.group(1).strip()
        if src not in selected:
            return src
        choices = replacements.get(src, [])
        return rng.choice(choices) if choices else src

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
    return run_generation(model, processor, messages, 500, False, 1.0, 1.0)


def step2_bracket_base(model, processor, base_prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": BRACKET_PROMPT_TEMPLATE.format(base_prompt=base_prompt)}],
    }]
    return run_generation(model, processor, messages, 700, False, 1.0, 1.0)


def step3_generate_replacements_json(model, processor, bracketed_prompt: str) -> Dict[str, List[str]]:
    terms = extract_bracket_terms(bracketed_prompt)
    if not terms:
        return {}

    block = "\n".join([f"- {t}" for t in terms])
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": REPLACEMENT_JSON_PROMPT_TEMPLATE.format(descriptor_block=block)}],
    }]

    raw = run_generation(model, processor, messages, 800, True, 0.9, 0.95)
    return safe_parse_json(raw).get("replacements", {}) or {}


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

    # step3a
    p3 = sub.add_parser("step3_replacements")
    add_model_args(p3)
    p3.add_argument("--bracket_in", required=True)
    p3.add_argument("--out_json", required=True)

    # step3b apply
    p4 = sub.add_parser("step3_apply")
    p4.add_argument("--bracket_in", required=True)
    p4.add_argument("--repl_json", required=False, help="If omitted, will apply without changes (only remove brackets).")
    p4.add_argument("--seed", type=int, required=True)
    p4.add_argument("--min_replace_terms", type=int, default=1)
    p4.add_argument("--max_replace_terms", type=int, default=3)
    p4.add_argument("--out", required=True)

    # full pipeline
    p5 = sub.add_parser("all")
    add_model_args(p5)
    p5.add_argument("--video", required=True)
    p5.add_argument("--fps", type=float, default=4.0)
    p5.add_argument("--seed", type=int, required=False)
    p5.add_argument("--min_replace_terms", type=int, default=1)
    p5.add_argument("--max_replace_terms", type=int, default=3)
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
        )
        write_text(args.out, final)
        print("[INFO] Wrote:", args.out)
        return

    # other commands need model
    model, processor = load_model_and_processor(args.model_type, args.size, args.variant)

    if args.cmd == "step1_base":
        base = step1_generate_base(model, processor, args.video, args.fps)
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

        repl = step3_generate_replacements_json(model, processor, bracketed)
        write_json(args.out_repl_json, {"replacements": repl})

        print("[INFO] Wrote:", args.out_bracket, args.out_repl_json)
        return

    if args.cmd == "step3_replacements":
        bracketed = read_text(args.bracket_in)
        repl = step3_generate_replacements_json(model, processor, bracketed)
        write_json(args.out_json, {"replacements": repl})
        print("[INFO] Wrote:", args.out_json)
        return

    if args.cmd == "all":
        if args.seed is None:
            args.seed = random.randrange(2**32)

        base = step1_generate_base(model, processor, args.video, args.fps)
        write_text(args.out_base, base)

        bracketed = step2_bracket_base(model, processor, base)
        write_text(args.out_bracket, bracketed)

        repl = step3_generate_replacements_json(model, processor, bracketed)
        write_json(args.out_repl_json, {"replacements": repl})

        final = apply_random_replacements(
            bracketed,
            repl,
            seed=args.seed,
            min_terms=args.min_replace_terms,
            max_terms=args.max_replace_terms,
        )
        write_text(args.out_final, final)

        print("[INFO] Wrote:", args.out_base, args.out_bracket, args.out_repl_json, args.out_final)
        return


if __name__ == "__main__":
    main()

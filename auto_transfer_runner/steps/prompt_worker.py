from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

BRACKET_PROMPT_TEMPLATE = """You are a prompt editor.
Wrap every word or short phrase that describes color, material, texture, finish, or surface appearance in Chinese brackets 【】.
Keep all other words, punctuation, and the paragraph structure unchanged.
Do not add or remove any information.
Output ONLY the modified paragraph.

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
  {"replacements": {"<src>": ["alt1","alt2","alt3","alt4","alt5","alt6"], ...}}

DESCRIPTORS:
{descriptor_block}
"""

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


def family_to_model_type(model_family: str) -> str:
    return {"qwen25vl": "Qwen2.5", "qwen3vl": "Qwen3"}.get(model_family.lower(), model_family)


def pick_model_id(size: str, variant: str, model_type: str = "Qwen2.5") -> str:
    size = size.upper()
    variant = variant.lower()
    if model_type == "Qwen3":
        model_dict = QWEN3_VL_MODELS
    elif model_type == "Qwen2.5":
        model_dict = QWEN25_VL_MODELS
    else:
        raise ValueError("Unsupported model type. Choose from: Qwen3, Qwen2.5.")

    if size not in model_dict:
        raise ValueError(f"Unsupported --size {size}.")
    if variant not in model_dict[size]:
        raise ValueError(f"Unsupported --variant {variant}. Choose from: instruct/awq.")
    return model_dict[size][variant]


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    return {}


def extract_bracket_terms(text: str) -> List[str]:
    seen = set()
    terms: List[str] = []
    for match in text.split("【")[1:]:
        if "】" not in match:
            continue
        candidate = match.split("】", 1)[0].strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            terms.append(candidate)
    return terms


def apply_random_replacements(bracketed_prompt: str, replacements: Dict[str, List[str]], seed: int) -> str:
    rng = random.Random(seed)

    def repl_fn(segment: str) -> str:
        src = segment.strip()
        alts = replacements.get(src, [])
        if not alts:
            return src
        choice = rng.choice(alts).strip()
        return choice or src

    parts = bracketed_prompt.split("【")
    output_parts = [parts[0]]
    for part in parts[1:]:
        if "】" not in part:
            output_parts.append(part)
            continue
        word, rest = part.split("】", 1)
        output_parts.append(repl_fn(word))
        output_parts.append(rest)
    return "".join(output_parts)


@torch.no_grad()
def run_generation(model, processor, messages, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs = dict(text=[text], padding=True, return_tensors="pt")
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs
    inputs = processor(**kwargs).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    outputs = model.generate(**inputs, **gen_kwargs)
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()


def load_qwen(model_id: str, model_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if model_type == "Qwen3":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    model.eval()
    return model, processor


def run_step1(args) -> None:
    model_type = family_to_model_type(args.model_family)
    model_id = pick_model_id(args.size, args.variant, model_type=model_type)
    model, processor = load_qwen(model_id, model_type=model_type)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{args.video}", "fps": args.fps},
                {"type": "text", "text": COSMOS_STYLE_PROMPT},
            ],
        }
    ]

    base_prompt = run_generation(
        model,
        processor,
        messages,
        max_new_tokens=500,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    Path(args.base_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.base_out).write_text(base_prompt + "\n", encoding="utf-8")


def run_step23(args) -> None:
    model_type = family_to_model_type(args.model_family)
    model_id = pick_model_id(args.size, args.variant, model_type=model_type)
    model, processor = load_qwen(model_id, model_type=model_type)

    base_prompt = Path(args.base_in).read_text(encoding="utf-8").strip()

    if args.reuse_bracket and Path(args.bracket_out).exists():
        bracketed_prompt = Path(args.bracket_out).read_text(encoding="utf-8").strip()
    else:
        step2_text = BRACKET_PROMPT_TEMPLATE.format(base_prompt=base_prompt)
        messages_step2 = [{"role": "user", "content": [{"type": "text", "text": step2_text}]}]
        bracketed_prompt = run_generation(
            model,
            processor,
            messages_step2,
            max_new_tokens=700,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        Path(args.bracket_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.bracket_out).write_text(bracketed_prompt + "\n", encoding="utf-8")

    terms = extract_bracket_terms(bracketed_prompt)
    replacements: Dict[str, List[str]] = {}
    if terms:
        descriptor_block = "\n".join([f"- {t}" for t in terms])
        step3_text = REPLACEMENT_JSON_PROMPT_TEMPLATE.format(descriptor_block=descriptor_block)
        messages_step3 = [{"role": "user", "content": [{"type": "text", "text": step3_text}]}]
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        raw_json = run_generation(
            model,
            processor,
            messages_step3,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        parsed = safe_parse_json(raw_json)
        if isinstance(parsed, dict):
            replacements = parsed.get("replacements", {}) or {}

    Path(args.replacements_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.replacements_out).write_text(json.dumps({"replacements": replacements, "seed": args.seed}, ensure_ascii=False, indent=2), encoding="utf-8")

    if replacements:
        final_prompt = apply_random_replacements(bracketed_prompt, replacements, seed=args.seed)
    else:
        final_prompt = bracketed_prompt.replace("【", "").replace("】", "").strip()

    Path(args.final_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.final_out).write_text(final_prompt + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split AutoPrompt runner for step1 and step2/3.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("step1", help="Run step1: video -> base prompt")
    p1.add_argument("--video", required=True)
    p1.add_argument("--base_out", required=True)
    p1.add_argument("--size", default="7B")
    p1.add_argument("--variant", default="instruct")
    p1.add_argument("--model_family", default="qwen25vl")
    p1.add_argument("--fps", type=float, default=4.0)

    p23 = sub.add_parser("step23", help="Run step2+3: text refinement")
    p23.add_argument("--base_in", required=True)
    p23.add_argument("--bracket_out", required=True)
    p23.add_argument("--replacements_out", required=True)
    p23.add_argument("--final_out", required=True)
    p23.add_argument("--size", default="7B")
    p23.add_argument("--variant", default="instruct")
    p23.add_argument("--model_family", default="qwen25vl")
    p23.add_argument("--seed", type=int, default=None)
    p23.add_argument("--reuse_bracket", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "step1":
        run_step1(args)
        return

    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "big")
    run_step23(args)


if __name__ == "__main__":
    main()

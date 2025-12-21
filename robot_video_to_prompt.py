import os
import re
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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
  {{"replacements": {{"<src>": ["alt1","alt2","alt3","alt4","alt5","alt6"], ...}}}}

DESCRIPTORS:
{descriptor_block}
"""

# =======================
# 2) Model Options (Qwen3 and Qwen2.5)
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
    # MoE 型号如 30B‑A3B / 235B‑A22B 都也可以进一步加进去
}


# =======================
# 3) Helpers
# =======================
def add_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if not ext:
        return path + suffix
    return root + suffix + ext


def pick_model_id(size: str, variant: str, model_type: str = "Qwen2.5") -> str:
    size = size.upper()
    variant = variant.lower()

    # Check for Qwen3 or Qwen2.5
    if model_type == "Qwen3":
        model_dict = QWEN3_VL_MODELS
    elif model_type == "Qwen2.5":
        model_dict = QWEN25_VL_MODELS
    else:
        raise ValueError("Unsupported model type. Choose from: Qwen3, Qwen2.5.")

    if size not in model_dict:
        raise ValueError(f"Unsupported --size {size}. Choose from: 2B/4B/8B/32B.")
    if variant not in model_dict[size]:
        raise ValueError(f"Unsupported --variant {variant}. Choose from: instruct/awq.")

    return model_dict[size][variant]


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def extract_bracket_terms(text: str) -> List[str]:
    terms = re.findall(r"【(.*?)】", text)
    seen = set()
    uniq = []
    for t in terms:
        t2 = t.strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            uniq.append(t2)
    return uniq


def apply_random_replacements(bracketed_prompt: str, replacements: Dict[str, List[str]], seed: int) -> str:
    rng = random.Random(seed)

    def repl_fn(match: re.Match) -> str:
        src = match.group(1).strip()
        alts = replacements.get(src, [])
        if not alts:
            return src
        choice = rng.choice(alts).strip()
        return choice if choice else src

    out = re.sub(r"【(.*?)】", repl_fn, bracketed_prompt)
    out = out.replace("【", "").replace("】", "")
    return out.strip()


@torch.no_grad()
def run_generation(model, processor, messages, max_new_tokens: int,
                   do_sample: bool, temperature: float, top_p: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    kwargs = dict(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs

    inputs = processor(**kwargs).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

    outputs = model.generate(**inputs, **gen_kwargs)

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    result = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", required=True, help="Output path for FINAL prompt (.txt recommended)")
    parser.add_argument("--size", default="4B", choices=["2B", "4B", "8B", "32B"], help="Model size")
    parser.add_argument("--variant", default="instruct", choices=["instruct", "awq"], help="Model variant")
    parser.add_argument("--model_type", default="Qwen3", choices=["Qwen2.5", "Qwen3"],
                        help="Model type (Qwen3 or Qwen2.5)")

    parser.add_argument("--fps", type=float, default=4.0, help="Video sampling FPS")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for step-3 replacement (omit for random).")

    args = parser.parse_args()
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "big")

    model_id = pick_model_id(args.size, args.variant, model_type=args.model_type)
    print(f"[INFO] Using model: {model_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    messages_step1 = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{args.video}", "fps": args.fps},
            {"type": "text", "text": COSMOS_STYLE_PROMPT},
        ],
    }]

    base_prompt = run_generation(
        model, processor, messages_step1,
        max_new_tokens=500,
        do_sample=False, temperature=1.0, top_p=1.0
    )
    print("\n===== STEP 1: BASE PROMPT =====\n")
    print(base_prompt)

    step2_text = BRACKET_PROMPT_TEMPLATE.format(base_prompt=base_prompt)
    messages_step2 = [{
        "role": "user",
        "content": [{"type": "text", "text": step2_text}],
    }]

    bracketed_prompt = run_generation(
        model, processor, messages_step2,
        max_new_tokens=700,
        do_sample=False, temperature=1.0, top_p=1.0
    )
    print("\n===== STEP 2: BRACKETED PROMPT =====\n")
    print(bracketed_prompt)

    terms = extract_bracket_terms(bracketed_prompt)

    replacements: Dict[str, List[str]] = {}
    if terms:
        descriptor_block = "\n".join([f"- {t}" for t in terms])
        step3_text = REPLACEMENT_JSON_PROMPT_TEMPLATE.format(descriptor_block=descriptor_block)
        messages_step3 = [{
            "role": "user",
            "content": [{"type": "text", "text": step3_text}],
        }]

        raw_json = run_generation(
            model, processor, messages_step3,
            max_new_tokens=800,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )

        parsed = safe_parse_json(raw_json)
        if isinstance(parsed, dict):
            replacements = parsed.get("replacements", {}) or {}

    if replacements:
        final_prompt = apply_random_replacements(bracketed_prompt, replacements, seed=args.seed)
    else:
        final_prompt = bracketed_prompt.replace("【", "").replace("】", "").strip()

    print("\n===== STEP 3: FINAL PROMPT =====\n")
    print(final_prompt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    out_base = add_suffix(args.out, "_base")
    out_bracketed = add_suffix(args.out, "_bracketed")
    out_repl = add_suffix(args.out, "_replacements")

    with open(out_base, "w", encoding="utf-8") as f:
        f.write(base_prompt + "\n")
    with open(out_bracketed, "w", encoding="utf-8") as f:
        f.write(bracketed_prompt + "\n")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(final_prompt + "\n")
    with open(out_repl, "w", encoding="utf-8") as f:
        json.dump({"replacements": replacements, "seed": args.seed}, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Saved:")
    print(f"  Step1 base:       {out_base}")
    print(f"  Step2 bracketed:  {out_bracketed}")
    print(f"  Step3 final:      {args.out}")
    print(f"  Replacement json: {out_repl}")


if __name__ == "__main__":
    main()

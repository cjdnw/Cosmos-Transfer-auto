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
# 2) Model Options (Qwen2.5-VL only)
# =======================
QWEN25_VL_MODELS = {
    "3B":  {"instruct": "Qwen/Qwen2.5-VL-3B-Instruct",  "awq": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"},
    "7B":  {"instruct": "Qwen/Qwen2.5-VL-7B-Instruct",  "awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
    "32B": {"instruct": "Qwen/Qwen2.5-VL-32B-Instruct", "awq": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"},
    "72B": {"instruct": "Qwen/Qwen2.5-VL-72B-Instruct", "awq": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"},
}


# =======================
# 3) Helpers
# =======================
def add_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if not ext:
        return path + suffix
    return root + suffix + ext


def pick_model_id(size: str, variant: str) -> str:
    size = size.upper()
    variant = variant.lower()
    if size not in QWEN25_VL_MODELS:
        raise ValueError(f"Unsupported --size {size}. Choose from: 3B/7B/32B/72B.")
    if variant not in QWEN25_VL_MODELS[size]:
        raise ValueError(f"Unsupported --variant {variant}. Choose from: instruct/awq.")
    return QWEN25_VL_MODELS[size][variant]


def safe_parse_json(text: str) -> Dict[str, Any]:
    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) extract first JSON object block
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
            return src  # no replacement candidate -> keep original
        choice = rng.choice(alts).strip()
        return choice if choice else src

    # replace bracket segments and implicitly remove brackets
    out = re.sub(r"【(.*?)】", repl_fn, bracketed_prompt)
    # safety cleanup
    out = out.replace("【", "").replace("】", "")
    return out.strip()


@torch.no_grad()
def run_generation(model, processor, messages, max_new_tokens: int,
                   do_sample: bool, temperature: float, top_p: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # process_vision_info should return (image_inputs, video_inputs)
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


# =======================
# 4) Main
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", required=True, help="Output path for FINAL prompt (.txt recommended)")
    parser.add_argument("--size", default="7B", choices=["3B", "7B", "32B", "72B"], help="Qwen2.5-VL model size")
    parser.add_argument("--variant", default="instruct", choices=["instruct", "awq"], help="Model variant")
    parser.add_argument("--fps", type=float, default=4.0, help="Video sampling FPS")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for step-3 replacement (omit for random).")

    parser.add_argument("--max_new_tokens_step1", type=int, default=500)
    parser.add_argument("--max_new_tokens_step2", type=int, default=700)
    parser.add_argument("--max_new_tokens_step3", type=int, default=800)

    # step3 sampling controls (used only when generating replacement candidates)
    parser.add_argument("--temperature_step3", type=float, default=0.9)
    parser.add_argument("--top_p_step3", type=float, default=0.95)

    args = parser.parse_args()
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "big")  # 0~2^32-1

    # reproducibility (best-effort)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_id = pick_model_id(args.size, args.variant)
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

    # -----------------------
    # Step 1: video -> base prompt
    # -----------------------
    messages_step1 = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{args.video}", "fps": args.fps},
            {"type": "text", "text": COSMOS_STYLE_PROMPT},
        ],
    }]

    base_prompt = run_generation(
        model, processor, messages_step1,
        max_new_tokens=args.max_new_tokens_step1,
        do_sample=False, temperature=1.0, top_p=1.0
    )
    print("\n===== STEP 1: BASE PROMPT =====\n")
    print(base_prompt)

    # -----------------------
    # Step 2: bracket descriptors inside paragraph
    # -----------------------
    step2_text = BRACKET_PROMPT_TEMPLATE.format(base_prompt=base_prompt)
    messages_step2 = [{
        "role": "user",
        "content": [{"type": "text", "text": step2_text}],
    }]

    bracketed_prompt = run_generation(
        model, processor, messages_step2,
        max_new_tokens=args.max_new_tokens_step2,
        do_sample=False, temperature=1.0, top_p=1.0
    )
    print("\n===== STEP 2: BRACKETED PROMPT =====\n")
    print(bracketed_prompt)

    terms = extract_bracket_terms(bracketed_prompt)

    # -----------------------
    # Step 3: generate replacement candidates (JSON) -> random replace -> remove brackets
    # -----------------------
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
            max_new_tokens=args.max_new_tokens_step3,
            do_sample=True,  # diversity
            temperature=args.temperature_step3,
            top_p=args.top_p_step3
        )

        parsed = safe_parse_json(raw_json)
        if isinstance(parsed, dict):
            replacements = parsed.get("replacements", {}) or {}

    if replacements:
        final_prompt = apply_random_replacements(bracketed_prompt, replacements, seed=args.seed)
    else:
        # fallback: no replacements available -> just remove brackets
        final_prompt = bracketed_prompt.replace("【", "").replace("】", "").strip()

    print("\n===== STEP 3: FINAL PROMPT =====\n")
    print(final_prompt)

    # -----------------------
    # Save outputs
    # -----------------------
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

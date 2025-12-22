import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Replace only a subset of unique descriptors, controlled by seed and range."""
    rng = random.Random(seed)
    terms = list({t.strip() for t in re.findall(r"【(.*?)】", bracketed_prompt)})

    if not terms:
        return bracketed_prompt.replace("【", "").replace("】", "")

    k = rng.randint(min_terms, max_terms)
    k = max(0, min(k, len(terms)))
    selected = set(rng.sample(terms, k)) if k > 0 else set()

    def repl(match: re.Match) -> str:
        src = match.group(1).strip()
        if src not in selected:
            return src
        choices = replacements.get(src, [])
        return rng.choice(choices) if choices else src

    out = re.sub(r"【(.*?)】", repl, bracketed_prompt)
    return out.replace("【", "").replace("】", "")


def dynamic_max_new_tokens(base_prompt: str, processor: Any) -> int:
    if processor and getattr(processor, "tokenizer", None):
        try:
            token_len = len(processor.tokenizer.encode(base_prompt))
            return min(512, token_len + 64)
        except Exception:
            pass
    # Fallback based on text length
    approx_tokens = max(1, len(base_prompt.split()))
    return min(512, approx_tokens + 64)


def dynamic_max_new_tokens(base_prompt: str, processor: Any) -> int:
    if processor and getattr(processor, "tokenizer", None):
        try:
            token_len = len(processor.tokenizer.encode(base_prompt))
            return min(512, token_len + 64)
        except Exception:
            pass
    # Fallback based on text length
    approx_tokens = max(1, len(base_prompt.split()))
    return min(512, approx_tokens + 64)


@torch.no_grad()
def run_generation(model, processor, messages, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> Tuple[str, int]:
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
    return processor.decode(gen_ids, skip_special_tokens=True).strip(), gen_ids.numel()


def load_model_and_processor(model_type: str, size: str, variant: str) -> Tuple[Any, Any, float]:
    start = time.perf_counter()
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
    load_time = time.perf_counter() - start
    return model, processor, load_time


# =======================
# 4) Step APIs
# =======================
def step1_generate_base(model, processor, video_path: str, fps: float) -> Tuple[str, int]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": fps},
            {"type": "text", "text": COSMOS_STYLE_PROMPT},
        ],
    }]
    return run_generation(model, processor, messages, 500, False, 1.0, 1.0)


def step2_bracket_base(model, processor, base_prompt: str, max_new_tokens: int) -> Tuple[str, int]:
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": BRACKET_PROMPT_TEMPLATE.format(base_prompt=base_prompt)}],
    }]
    return run_generation(model, processor, messages, max_new_tokens, False, 1.0, 1.0)


def step3_generate_replacements_json(model, processor, bracketed_prompt: str, num_alts: int) -> Dict[str, List[str]]:
    terms = extract_bracket_terms(bracketed_prompt)
    if not terms:
        return {}

    block = "\n".join([f"- {t}" for t in terms])
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": REPLACEMENT_JSON_PROMPT_TEMPLATE.format(descriptor_block=block)}],
    }]

    raw, gen_tokens = run_generation(model, processor, messages, 800, True, 0.9, 0.95)
    repl = safe_parse_json(raw).get("replacements", {}) or {}
    cleaned: Dict[str, List[str]] = {}
    for k, vals in repl.items():
        uniq = []
        seen = set()
        for v in vals:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
            if len(uniq) >= num_alts:
                break
        cleaned[k] = uniq
    return cleaned, gen_tokens


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
# 5) Utils: caching & IO
# =======================
def video_fingerprint(video_path: str, keep: int = 24) -> str:
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:keep]


def cache_paths(cache_dir: Path, fingerprint: str) -> Dict[str, Path]:
    base = cache_dir / fingerprint
    return {
        "base": base / "base.txt",
        "bracket": base / "bracket.txt",
        "repl": base / "repl.json",
    }


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def log_time(label: str, duration: float):
    print(f"[TIME] {label}={duration:.2f}s")


class FileLock:
    def __init__(self, path: Path, timeout: int = 6 * 3600, stale: int = 3 * 3600):
        self.path = path
        self.timeout = timeout
        self.stale = stale
        self.acquired = False

    def acquire(self) -> None:
        # Ensure lock directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()} t={time.time()}\n".encode("utf-8"))
                os.close(fd)
                self.acquired = True
                return
            except FileExistsError:
                if self.path.exists() and (time.time() - self.path.stat().st_mtime > self.stale):
                    try:
                        self.path.unlink()
                        continue
                    except Exception:
                        pass
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Lock timeout: {self.path}")
                time.sleep(2)

    def release(self) -> None:
        if self.acquired and self.path.exists():
            try:
                self.path.unlink()
            finally:
                self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


# =======================
# 6) CLI
# =======================
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_model_args(p):
        p.add_argument("--size", default="4B", choices=["2B", "4B", "8B", "32B"])
        p.add_argument("--variant", default="instruct", choices=["instruct", "awq"])
        p.add_argument("--model_type", default="Qwen3", choices=["Qwen2.5", "Qwen3"])

    def add_replace_args(p):
        p.add_argument("--min_replace_terms", type=int, default=1)
        p.add_argument("--max_replace_terms", type=int, default=3)

    # step1
    p1 = sub.add_parser("step1_base")
    add_model_args(p1)
    p1.add_argument("--video", required=True)
    p1.add_argument("--fps", type=float, default=4.0)
    p1.add_argument("--out_base", required=True)
    p1.add_argument("--cache_dir", type=str, default=None)

    # step2
    p2 = sub.add_parser("step2_bracket")
    add_model_args(p2)
    p2.add_argument("--base_in", required=True)
    p2.add_argument("--out_bracket", required=True)

    # step3a
    p3 = sub.add_parser("step3_replacements")
    add_model_args(p3)
    p3.add_argument("--bracket_in", required=True)
    p3.add_argument("--out_repl_json", required=True)
    p3.add_argument("--num_alts", type=int, default=6)

    # step3b apply
    p4 = sub.add_parser("step3_apply")
    p4.add_argument("--bracket_in", required=True)
    p4.add_argument("--repl_json", required=False, help="If omitted, will apply without changes (only remove brackets).")
    p4.add_argument("--seed", type=int, default=None, help="If omitted, a random seed will be generated")
    p4.add_argument("--out_final", required=True)
    add_replace_args(p4)

    # step23 combined
    p23 = sub.add_parser("step23")
    add_model_args(p23)
    p23.add_argument("--base_in", required=True)
    p23.add_argument("--seed", type=int, default=None, help="If omitted, a random seed will be generated")
    p23.add_argument("--out_final", required=True)
    p23.add_argument("--out_bracket", type=str, default=None)
    p23.add_argument("--out_repl_json", type=str, default=None)
    p23.add_argument("--repl_json_in", type=str, default=None)
    p23.add_argument("--num_alts", type=int, default=6)
    p23.add_argument("--cache_dir", type=str, default=None)
    p23.add_argument("--video", type=str, default=None, help="Used for cache lookup (fingerprint).")
    add_replace_args(p23)

    # cache_build
    p_cache = sub.add_parser("cache_build")
    add_model_args(p_cache)
    p_cache.add_argument("--video", required=True)
    p_cache.add_argument("--fps", type=float, default=4.0)
    p_cache.add_argument("--cache_dir", type=str, default="./cache")
    p_cache.add_argument("--num_alts", type=int, default=6)

    # skin_batch
    p_skin = sub.add_parser("skin_batch")
    p_skin.add_argument("--video", required=True)
    p_skin.add_argument("--cache_dir", type=str, default="./cache")
    p_skin.add_argument("--seeds", type=int, nargs="*", default=None)
    p_skin.add_argument("--num", type=int, default=0, help="Generate N random seeds if --seeds not provided")
    p_skin.add_argument("--out_dir", required=True)
    add_replace_args(p_skin)

    args = parser.parse_args()

    if getattr(args, "min_replace_terms", 1) > getattr(args, "max_replace_terms", 1):
        raise ValueError("--min_replace_terms must be <= --max_replace_terms")
    if hasattr(args, "num_alts") and args.num_alts < 1:
        raise ValueError("--num_alts must be >= 1")

    if args.cmd == "step3_apply":
        print("[INFO] step3_apply: no model load (cache-only apply)")
        bracketed = read_text(args.bracket_in)
        replacements = read_json(args.repl_json).get("replacements", {}) if args.repl_json else {}
        if args.seed is None:
            args.seed = random.SystemRandom().randrange(2**32)
            print(f"[INFO] Generated random seed for apply: {args.seed}")
        final = apply_random_replacements(
            bracketed,
            replacements,
            seed=args.seed,
            min_terms=args.min_replace_terms,
            max_terms=args.max_replace_terms,
        )
        ensure_parent(Path(args.out_final))
        write_text(args.out_final, final)
        print("[INFO] Wrote:", args.out_final)
        return

    if args.cmd == "skin_batch":
        fp = video_fingerprint(args.video)
        paths = cache_paths(Path(args.cache_dir), fp)
        if not paths["bracket"].exists() or not paths["repl"].exists():
            raise FileNotFoundError(f"Cache missing for {args.video}. Run cache_build first. Expected {paths}")
        bracketed = read_text(paths["bracket"])
        replacements = read_json(paths["repl"]).get("replacements", {})
        seeds = args.seeds if args.seeds else []
        if not seeds and args.num > 0:
            rng = random.Random(0)
            seeds = [rng.randrange(2**32) for _ in range(args.num)]
        if not seeds:
            raise ValueError("Provide --seeds or --num for skin_batch")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        print("[INFO] skin_batch: no model load (cache-only apply)")
        start = time.perf_counter()
        for s in seeds:
            final = apply_random_replacements(
                bracketed,
                replacements,
                seed=int(s),
                min_terms=args.min_replace_terms,
                max_terms=args.max_replace_terms,
            )
            out = Path(args.out_dir) / f"final_seed{s}.txt"
            write_text(str(out), final)
        log_time("APPLY_TIME", time.perf_counter() - start)
        print(f"[INFO] Generated {len(seeds)} files to {args.out_dir}")
        return

    if args.cmd == "cache_build":
        cache_root = Path(args.cache_dir)
        fp = video_fingerprint(args.video)
        paths = cache_paths(cache_root, fp)
        paths["base"].parent.mkdir(parents=True, exist_ok=True)

        lock = FileLock(paths["base"].parent / ".lock")
        with lock:
            missing_base = not paths["base"].exists()
            missing_bracket = not paths["bracket"].exists()
            missing_repl = not paths["repl"].exists()

            model = processor = None
            if missing_base or missing_bracket or missing_repl:
                model, processor, load_time = load_model_and_processor(args.model_type, args.size, args.variant)
                log_time("MODEL_LOAD_TIME", load_time)

            if missing_base:
                start = time.perf_counter()
                base, gen_tokens = step1_generate_base(model, processor, args.video, args.fps)
                log_time("GENERATE_TIME_STEP1", time.perf_counter() - start)
                print(f"[TOKENS] STEP1={gen_tokens}")
                write_text(str(paths["base"]), base)
            else:
                print("[CACHE] base exists, skip")

            if missing_bracket:
                base = read_text(paths["base"])
                max_new = dynamic_max_new_tokens(base, processor)
                start = time.perf_counter()
                bracket, gen_tokens = step2_bracket_base(model, processor, base, max_new)
                log_time("GENERATE_TIME_STEP2", time.perf_counter() - start)
                print(f"[TOKENS] STEP2={gen_tokens}")
                write_text(str(paths["bracket"]), bracket)
            else:
                print("[CACHE] bracket exists, skip")

            if missing_repl:
                bracket = read_text(paths["bracket"])
                start = time.perf_counter()
                repl, gen_tokens = step3_generate_replacements_json(model, processor, bracket, args.num_alts)
                log_time("GENERATE_TIME_STEP3", time.perf_counter() - start)
                print(f"[TOKENS] STEP3={gen_tokens}")
                write_json(str(paths["repl"]), {"replacements": repl})
            else:
                print("[CACHE] repl exists, skip")

        return

    if args.cmd == "step1_base" and args.cache_dir:
        cache_root = Path(args.cache_dir)
        fp = video_fingerprint(args.video)
        paths = cache_paths(cache_root, fp)
        lock = FileLock(paths["base"].parent / ".lock")
        with lock:
            if paths["base"].exists():
                print("[CACHE] base exists, skip model load")
                ensure_parent(Path(args.out_base))
                write_text(args.out_base, read_text(paths["base"]))
                return

    if args.cmd == "step23":
        cache_root = Path(args.cache_dir) if args.cache_dir else None
        cache_paths_map = None
        bracket_path_cached = None
        repl_data_cached = None
        if cache_root and args.video:
            fp = video_fingerprint(args.video)
            cache_paths_map = cache_paths(cache_root, fp)
            if cache_paths_map["bracket"].exists():
                bracket_path_cached = cache_paths_map["bracket"]
            if args.repl_json_in:
                repl_data_cached = read_json(args.repl_json_in).get("replacements", {})
            elif cache_paths_map["repl"].exists():
                repl_data_cached = read_json(cache_paths_map["repl"]).get("replacements", {})
        if bracket_path_cached is not None and repl_data_cached is not None:
            print("[INFO] step23: cache hit, no model load")
            bracket_prompt = read_text(str(bracket_path_cached))
            apply_start = time.perf_counter()
            final = apply_random_replacements(
                bracket_prompt,
                repl_data_cached,
                seed=args.seed if args.seed is not None else random.SystemRandom().randrange(2**32),
                min_terms=args.min_replace_terms,
                max_terms=args.max_replace_terms,
            )
            apply_duration = time.perf_counter() - apply_start
            ensure_parent(Path(args.out_final))
            write_text(args.out_final, final)
            if args.out_bracket:
                write_text(args.out_bracket, bracket_prompt)
            if args.out_repl_json:
                write_json(args.out_repl_json, {"replacements": repl_data_cached})
            log_time("MODEL_LOAD_TIME", 0.0)
            log_time("GENERATE_TIME_STEP2", 0.0)
            log_time("GENERATE_TIME_STEP3", 0.0)
            log_time("APPLY_TIME", apply_duration)
            return
    # Commands below need model (step1/2/3/23)
    model, processor, load_time = load_model_and_processor(args.model_type, args.size, args.variant)
    log_time("MODEL_LOAD_TIME", load_time)

    if args.cmd == "step1_base":
        cache_root = Path(args.cache_dir) if args.cache_dir else None
        out_base = Path(args.out_base)
        if cache_root:
            fp = video_fingerprint(args.video)
            paths = cache_paths(cache_root, fp)
            lock = FileLock(paths["base"].parent / ".lock")
            with lock:
                if paths["base"].exists():
                    print("[CACHE] base exists, skip generation")
                    ensure_parent(out_base)
                    write_text(str(out_base), read_text(paths["base"]))
                    return
                out_base = paths["base"]
                start = time.perf_counter()
                base, gen_tokens = step1_generate_base(model, processor, args.video, args.fps)
                log_time("GENERATE_TIME_STEP1", time.perf_counter() - start)
                print(f"[TOKENS] STEP1={gen_tokens}")
                write_text(str(out_base), base)
        else:
            start = time.perf_counter()
            base, gen_tokens = step1_generate_base(model, processor, args.video, args.fps)
            log_time("GENERATE_TIME_STEP1", time.perf_counter() - start)
            print(f"[TOKENS] STEP1={gen_tokens}")
            write_text(str(out_base), base)
        print("[INFO] Wrote:", out_base)
        return

    if args.cmd == "step2_bracket":
        base = read_text(args.base_in)
        max_new = dynamic_max_new_tokens(base, processor)
        start = time.perf_counter()
        bracketed, gen_tokens = step2_bracket_base(model, processor, base, max_new)
        log_time("GENERATE_TIME_STEP2", time.perf_counter() - start)
        print(f"[TOKENS] STEP2={gen_tokens}")
        write_text(args.out_bracket, bracketed)
        print("[INFO] Wrote:", args.out_bracket)
        return

    if args.cmd == "step3_replacements":
        bracketed = read_text(args.bracket_in)
        start = time.perf_counter()
        repl, gen_tokens = step3_generate_replacements_json(model, processor, bracketed, args.num_alts)
        log_time("GENERATE_TIME_STEP3", time.perf_counter() - start)
        print(f"[TOKENS] STEP3={gen_tokens}")
        write_json(args.out_repl_json, {"replacements": repl})
        print("[INFO] Wrote:", args.out_repl_json)
        return

    if args.cmd == "step23":
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
        cache_root = Path(args.cache_dir) if args.cache_dir else None
        bracket_path_out = Path(args.out_bracket) if args.out_bracket else None
        repl_path_out = Path(args.out_repl_json) if args.out_repl_json else None

        base_path = Path(args.base_in)
        cache_paths_map: Optional[Dict[str, Path]] = None
        lock: Optional[FileLock] = None

        if cache_root and args.video:
            fp = video_fingerprint(args.video)
            cache_paths_map = cache_paths(cache_root, fp)
            if cache_paths_map["base"].exists():
                base_path = cache_paths_map["base"]
            lock = FileLock(cache_paths_map["base"].parent / ".lock")

        base_prompt = read_text(str(base_path))
        max_new = dynamic_max_new_tokens(base_prompt, processor)

        bracket_path = cache_paths_map["bracket"] if cache_paths_map else bracket_path_out
        repl_path = cache_paths_map["repl"] if cache_paths_map else None

        if lock:
            lock.__enter__()
        try:
            if cache_paths_map and not base_path.exists():
                start = time.perf_counter()
                base_prompt, gen_tokens = step1_generate_base(model, processor, args.video, args.fps)
                log_time("GENERATE_TIME_STEP1", time.perf_counter() - start)
                print(f"[TOKENS] STEP1={gen_tokens}")
                ensure_parent(base_path)
                write_text(str(base_path), base_prompt)
            else:
                base_prompt = base_prompt

            need_step2 = not (bracket_path and bracket_path.exists())
            need_step3 = (args.repl_json_in is None) and not (repl_path and repl_path.exists())

            start = time.perf_counter()
            if need_step2:
                bracket_prompt, gen_tokens = step2_bracket_base(model, processor, base_prompt, max_new)
                log_time("GENERATE_TIME_STEP2", time.perf_counter() - start)
                print(f"[TOKENS] STEP2={gen_tokens}")
                target_path = bracket_path_out or bracket_path or Path(args.out_final).with_suffix(".bracket.txt")
                ensure_parent(target_path)
                write_text(str(target_path), bracket_prompt)
                bracket_path = target_path
            else:
                bracket_prompt = read_text(str(bracket_path))
                log_time("GENERATE_TIME_STEP2", time.perf_counter() - start)

            if args.repl_json_in:
                repl_data = read_json(args.repl_json_in).get("replacements", {})
            elif repl_path and repl_path.exists():
                repl_data = read_json(str(repl_path)).get("replacements", {})
                log_time("GENERATE_TIME_STEP3", 0.0)
            else:
                step3_start = time.perf_counter()
                repl_data, gen_tokens = step3_generate_replacements_json(model, processor, bracket_prompt, args.num_alts)
                log_time("GENERATE_TIME_STEP3", time.perf_counter() - step3_start)
                print(f"[TOKENS] STEP3={gen_tokens}")
                target_repl = repl_path_out or repl_path or Path(args.out_final).with_suffix(".repl.json")
                ensure_parent(target_repl)
                write_json(str(target_repl), {"replacements": repl_data})
                repl_path = target_repl
        finally:
            if lock:
                lock.__exit__(None, None, None)

        apply_start = time.perf_counter()
        final = apply_random_replacements(
            bracket_prompt,
            repl_data,
            seed=seed,
            min_terms=args.min_replace_terms,
            max_terms=args.max_replace_terms,
        )
        log_time("APPLY_TIME", time.perf_counter() - apply_start)
        ensure_parent(Path(args.out_final))
        write_text(args.out_final, final)

        if bracket_path_out and bracket_path_out != bracket_path:
            write_text(str(bracket_path_out), bracket_prompt)
        if repl_path_out and repl_path_out != repl_path:
            write_json(str(repl_path_out), {"replacements": repl_data})

        print("[INFO] Wrote:", args.out_final)
        return


if __name__ == "__main__":
    main()

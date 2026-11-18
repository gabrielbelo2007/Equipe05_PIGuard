"""
Official Synthetic Data Generator for Prompt Injection Guard (RP_PIGuard)

Faithful reproduction of the data generation process from the original paper:
- Uses gpt-4o-mini-2024-07-18 via OpenRouter API.
- Generates 435 synthetic samples across 17 categories according to Table 6 (BASE_DISTRIBUTION).
- Uses the exact original System and User Prompts without selection filters.
- All 435 generated samples are assigned 'label': 1, 'source': 'LLM Augmented set'.
- Generates in batches of up to 10 samples (randomizing 3 examples per batch if > 3 available).
- Deterministic random seed for reproducible example selection.
- Outputs Augmented/LLM_augmented.json with only keys: 'prompt', 'label', 'source'.
- Directly appends clean generated text to 'prompt' without surrounding brackets '[' or ']'.
- Concatenates with datasets/train_cleaned_oficial.json to produce datasets/train_augmented.json.
"""

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

CACHE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CACHE_DIR.parents[2]


def resolve_repo_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    return str(path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve())

# Model snapshot
GENERATION_MODEL = "openai/gpt-4o-mini-2024-07-18"

# Table 6 - 435 total samples across 17 categories
BASE_DISTRIBUTION: Dict[str, int] = {
    "Email Injection": 48,
    "Document Injection": 25,
    "Chat Conversation Injection": 25,
    "JSON Injection": 23,
    "Code Injection": 23,
    "Markdown Injection": 23,
    "HTML Injection": 23,
    "URL Injection": 23,
    "Base64 Injection": 23,
    "Table Injection": 23,
    "XML Injection": 23,
    "CSV Injection": 23,
    "Config File Injection": 23,
    "Log File Injection": 23,
    "Image Link Injection": 23,
    "Translation Injection": 27,
    "Website Injection": 34,
}

# Original paper prompts
SYSTEM_PROMPT = (
    "You are a professional data generator specializing in LLM security, known\n"
    "for creating benign and safe content with diverse templates."
)

USER_PROMPT_TEMPLATE = (
    "Below are some examples of indirect hint injections: {demonstrations}\n"
    "Please generate {number} benign samples on the topic {key}, ensuring that each sample is safe "
    "and does not contain any form of harmful or misleading content. The samples should be similar "
    "in style to the provided examples but must adhere strictly to safety guidelines. Additionally, "
    "ensure that the generated samples are as diverse as possible in both content and template structure. "
    "Each sample should be no more than 50 words and presented in the followin format: \n"
    "[INPUT]"
)


def load_api_key() -> str:
    """Load OpenRouter/OpenAI API key from environment or .env file."""
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_var)
        if val:
            return val.strip().strip("\"'")

    for env_path in (REPO_ROOT / ".env", Path(".env")):
        if env_path.is_file():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip().lower() in ("openrouter_api_key", "openai_api_key"):
                            return v.strip().strip("\"'")

    raise RuntimeError("API key not found in environment or .env file.")


def call_openrouter(
    payload: dict,
    api_key: str,
    max_retries: int = 5,
    timeout: int = 120,
) -> dict:
    """Execute a request to OpenRouter API with exponential backoff on errors."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_exc = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            status = e.code
            if status in (429, 500, 502, 503, 504):
                sleep_sec = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_sec)
                continue
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {status} error from OpenRouter: {err_body}") from e
        except Exception as e:
            last_exc = e
            sleep_sec = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(sleep_sec)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_exc}")


def call_generation_model(
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    temperature: float = 1.0,
) -> str:
    """Call OpenRouter with normalized model identifier."""
    model_id = model if "/" in model else f"openai/{model}"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    resp = call_openrouter(payload, api_key)
    return resp["choices"][0]["message"].get("content") or ""


def clean_chunk(chunk: str, fmt: str) -> str:
    """
    Clean formatting artifacts from an extracted candidate chunk.
    Strips markers like [INPUT], numbers, wrappers, and surrounding brackets [ ... ].
    """
    chunk = chunk.strip()

    # Strip trailing standalone number line if split occurred at marker
    chunk = re.sub(r"\n\s*\d+[\.\)]\s*$", "", chunk).strip()

    # Strip **[INPUT]**: or [INPUT]: or **[Input]**
    chunk = re.sub(r"^\*{0,2}\[input\]\*{0,2}:?\s*", "", chunk, flags=re.IGNORECASE).strip()
    chunk = re.sub(r"^\*\*Sample\s*\d+:?\*\*\s*", "", chunk, flags=re.IGNORECASE).strip()

    # Strip **[ ... ]** wrapper
    if chunk.startswith("**[") and chunk.endswith("]**"):
        chunk = chunk[3:-3].strip()

    # Strip outer code block if the model wrapped the entire item in ```...```
    if chunk.startswith("```") and chunk.endswith("```") and chunk.count("```") == 2:
        lines = chunk.splitlines()
        if len(lines) >= 2:
            chunk = "\n".join(lines[1:-1]).strip()

    # Unescape literal \n and \" that came from JSON representation
    chunk = chunk.replace("\\n", "\n").replace('\\"', '"')

    # Strip outer quotes if wrapped in "..."
    if len(chunk) >= 2 and chunk.startswith('"') and chunk.endswith('"'):
        chunk = chunk[1:-1].strip()

    # Strip outer [ ... ] or dangling [ if it was used as an [INPUT] bracket wrapper
    if chunk.startswith("[") and chunk.endswith("]"):
        if (
            fmt not in ("Log File", "Image Link")
            and not chunk.startswith("[INFO")
            and not chunk.startswith("[WARN")
            and not chunk.startswith("[ERROR")
            and not chunk.startswith("[Start")
            and not chunk.startswith("[ignore")
        ):
            chunk = chunk[1:-1].strip()
    elif chunk.startswith("[") and not chunk.endswith("]"):
        if (
            fmt not in ("Log File", "Image Link")
            and not chunk.startswith("[INFO")
            and not chunk.startswith("[WARN")
            and not chunk.startswith("[ERROR")
            and not chunk.startswith("[Start")
            and not chunk.startswith("[ignore")
        ):
            chunk = chunk[1:].strip()

    # Strip Prompt: "..." wrapper if present
    m_prompt = re.match(r'^Prompt:\s*"(.*)"$', chunk, flags=re.DOTALL)
    if m_prompt:
        chunk = m_prompt.group(1).strip()

    # Strip outer quotes again if inside brackets
    if len(chunk) >= 2 and chunk.startswith('"') and chunk.endswith('"'):
        chunk = chunk[1:-1].strip()
    elif chunk.startswith('"') and not chunk.endswith('"'):
        chunk = chunk[1:].strip()

    chunk = chunk.replace('\\"', '"')

    # Dedent lines if all lines after first have leading spaces from numbered list indentation
    lines = chunk.splitlines()
    if len(lines) > 1 and fmt != "Config File":
        lines = [lines[0].strip()] + [re.sub(r"^[ \t]{2,4}", "", ln) for ln in lines[1:]]
        chunk = "\n".join(lines).strip()

    return chunk


def strip_closing_remark(chunk: str) -> str:
    """Strip conversational closing remarks from the last sample."""
    parts = chunk.split("\n\n")
    if len(parts) > 1 and any(
        parts[-1].strip().startswith(prefix)
        for prefix in (
            "These samples",
            "Each sample",
            "Feel free",
            "Let me know",
            "All samples",
            "I hope",
            "Note:",
            "Here are",
        )
    ):
        return "\n\n".join(parts[:-1]).strip()
    return chunk


def parse_samples(raw_text: str, fmt: str) -> List[str]:
    """Extract individual samples from raw model response without any validation filtering."""
    text = raw_text.strip()
    items: List[str] = []

    # Strip markdown code block wrapper if entire text is in ```...```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    # 1. Check if [INPUT] markers are used
    input_pattern = re.compile(
        r"(?:^|\n)\s*(?:\d+[\.\)]\s*)?\*{0,2}\[INPUT\]\*{0,2}:?\s*",
        re.IGNORECASE,
    )
    input_matches = list(input_pattern.finditer(text))

    if len(input_matches) >= 1:
        for i, m in enumerate(input_matches):
            start = m.end()
            end = input_matches[i + 1].start() if i + 1 < len(input_matches) else len(text)
            chunk = text[start:end].strip()
            if i == len(input_matches) - 1:
                chunk = strip_closing_remark(chunk)
            chunk = clean_chunk(chunk, fmt)
            if chunk:
                items.append(chunk)
        if items:
            return items

    # 2. Try JSON list parsing if the model returned a valid JSON array
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                for elem in parsed_json:
                    if isinstance(elem, str):
                        c = clean_chunk(elem, fmt)
                        if c:
                            items.append(c)
                    elif isinstance(elem, dict):
                        if "text" in elem and isinstance(elem["text"], str):
                            c = clean_chunk(elem["text"], fmt)
                            if c:
                                items.append(c)
                        elif "prompt" in elem and isinstance(elem["prompt"], str):
                            c = clean_chunk(elem["prompt"], fmt)
                            if c:
                                items.append(c)
                        else:
                            items.append(json.dumps(elem, ensure_ascii=False))
                if items:
                    return items
        except Exception:
            json_text_matches = re.findall(
                r'\{\s*"(?:text|prompt)"\s*:\s*"(.*?)(?="\s*\}\s*,\s*\{\s*"(?:text|prompt)"|"\s*\}\s*\]|"\s*\}$)',
                text,
                re.DOTALL,
            )
            if json_text_matches:
                for match in json_text_matches:
                    c = clean_chunk(match, fmt)
                    if c:
                        items.append(c)
                if items:
                    return items

    # 3. Otherwise split by top-level numbered items: ^\s*\d+[\.\)]\s+
    num_pattern = re.compile(r"(?:^|\n)\s*(\d+)[\.\)]\s+", re.MULTILINE)
    matches = list(num_pattern.finditer(text))
    if matches:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if i == len(matches) - 1:
                chunk = strip_closing_remark(chunk)
            chunk = clean_chunk(chunk, fmt)
            if chunk:
                items.append(chunk)
        if items:
            return items

    # 4. Fallback: lines wrapped in [...]
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            c = clean_chunk(line, fmt)
            if c:
                items.append(c)

    # 5. Final fallback for single unnumbered response
    if not items and text:
        c = clean_chunk(strip_closing_remark(text), fmt)
        if c:
            items.append(c)

    return items


def generate_category_samples(
    key: str,
    base_count: int,
    demos: List[str],
    model: str,
    api_key: str,
    rng: random.Random,
    temperature: float = 1.0,
) -> List[str]:
    """
    Generate exactly base_count samples for a category in batches of up to 10.
    Re-randomizes 3 examples at each batch using rng (if > 3 demos available).
    No filtering is applied: every generated sample is accepted unconditionally.
    """
    fmt = key.replace(" Injection", "")
    samples: List[str] = []
    attempt = 0

    while len(samples) < base_count:
        attempt += 1
        missing = base_count - len(samples)
        batch_size = min(10, missing)

        if len(demos) > 3:
            selected_demos = rng.sample(demos, 3)
        else:
            selected_demos = list(demos)

        demo_str = json.dumps(selected_demos, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            demonstrations=demo_str,
            number=batch_size,
            key=key,
        )

        try:
            raw_text = call_generation_model(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            print(f"  [ERROR] {key} batch attempt {attempt} failed: {e}")
            time.sleep(2)
            continue

        extracted = parse_samples(raw_text, fmt)
        if not extracted:
            print(f"  [WARN] {key} batch attempt {attempt}: no samples parsed, retrying...")
            continue

        for cand in extracted:
            if len(samples) < base_count and cand.strip():
                samples.append(cand.strip())

        print(f"  [{fmt}] Batch {attempt}: {len(samples)}/{base_count} collected")

    return samples[:base_count]


def run_generation(
    model: str = GENERATION_MODEL,
    seed: int = 42,
    curated_path: str = "datasets/Augmented/Cache/curated_prompt_injections.json",
    output_augmented_json: str = "datasets/Augmented/Cache/LLM_augmented_rp.json",
    train_cleaned_path: str = "datasets/Augmented/train_cleaned_original.json",
    train_augmented_path: str = "datasets/Augmented/train_augmented_rp.json",
    temperature: float = 1.0,
):
    """Main routine to generate LLM_augmented_rp.json and train_augmented_rp.json."""
    api_key = load_api_key()

    curated_path = resolve_repo_path(curated_path)
    output_augmented_json = resolve_repo_path(output_augmented_json)
    train_cleaned_path = resolve_repo_path(train_cleaned_path)
    train_augmented_path = resolve_repo_path(train_augmented_path)
    if not os.path.exists(curated_path):
        raise FileNotFoundError(f"Curated file not found: {curated_path}")

    with open(curated_path, "r", encoding="utf-8") as f:
        curated_data = json.load(f)

    demos_by_format: Dict[str, List[str]] = {}
    for item in curated_data:
        fmt = item["format"]
        demos_by_format.setdefault(fmt, []).append(item["prompt"])

    rng = random.Random(seed)

    print("=================================================================")
    print("Starting Official Generation Process (Faithful Paper Replication)")
    print(f"Model: {model}")
    print(f"Seed: {seed}")
    print(f"Total categories: {len(BASE_DISTRIBUTION)} | Total target samples: {sum(BASE_DISTRIBUTION.values())}")
    print("=================================================================")

    start_time = time.time()
    all_augmented_records = []

    for key, count in BASE_DISTRIBUTION.items():
        fmt = key.replace(" Injection", "")
        demos = demos_by_format.get(fmt, [])
        if not demos:
            raise ValueError(f"No demonstrations found for format '{fmt}' ({key})")

        print(f"\n>>> Generating category: {key} (target: {count}, available demos: {len(demos)})")
        cat_samples = generate_category_samples(
            key=key,
            base_count=count,
            demos=demos,
            model=model,
            api_key=api_key,
            rng=rng,
            temperature=temperature,
        )

        for s in cat_samples:
            all_augmented_records.append({
                "prompt": s,
                "label": 1,
                "source": "LLM Augmented set",
            })

    # Save Augmented/LLM_augmented.json
    os.makedirs(os.path.dirname(output_augmented_json) or ".", exist_ok=True)
    with open(output_augmented_json, "w", encoding="utf-8") as f:
        json.dump(all_augmented_records, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Saved {len(all_augmented_records)} synthetic samples to {output_augmented_json}")

    # Concatenate with datasets/train_cleaned_oficial.json -> datasets/train_augmented.json
    if not os.path.exists(train_cleaned_path):
        raise FileNotFoundError(f"Cleaned training dataset not found: {train_cleaned_path}")

    with open(train_cleaned_path, "r", encoding="utf-8") as f:
        cleaned_dataset = json.load(f)

    print(f"Loaded {len(cleaned_dataset)} baseline samples from {train_cleaned_path}")
    train_augmented_data = cleaned_dataset + all_augmented_records

    os.makedirs(os.path.dirname(train_augmented_path) or ".", exist_ok=True)
    with open(train_augmented_path, "w", encoding="utf-8") as f:
        json.dump(train_augmented_data, f, indent=4, ensure_ascii=False)

    elapsed = round(time.time() - start_time, 2)
    print(f"[OK] Saved {len(train_augmented_data)} total samples to {train_augmented_path}")
    print(f"Completed in {elapsed}s.")


def main():
    parser = argparse.ArgumentParser(description="Official Synthetic Data Generator without filters")
    parser.add_argument("--model", default=GENERATION_MODEL, help=f"Generation model (default: {GENERATION_MODEL})")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--curated-path", default="datasets/Augmented/Cache/curated_prompt_injections.json", help="Path to curated demos")
    parser.add_argument("--output-augmented", default="datasets/Augmented/Cache/LLM_augmented_rp.json", help="Path to save synthetic set")
    parser.add_argument("--train-cleaned", default="datasets/Augmented/train_cleaned_original.json", help="Path to cleaned dataset")
    parser.add_argument("--train-augmented", default="datasets/Augmented/train_augmented_rp.json", help="Path to output train_augmented")
    args = parser.parse_args()

    run_generation(
        model=args.model,
        seed=args.seed,
        curated_path=args.curated_path,
        output_augmented_json=args.output_augmented,
        train_cleaned_path=args.train_cleaned,
        train_augmented_path=args.train_augmented,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

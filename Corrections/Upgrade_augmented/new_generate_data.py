"""
Gerador Unificado de Dados Sintéticos para Proteção contra Injeção de Prompt (RP_PIGuard)

Gera conjuntos de dados sintéticos maliciosos (injeções de prompt indiretas) em 17 categorias,
utilizando o GPT-4o-mini por meio da API do OpenRouter.

Emprega um funil de validação com 4 filtros, importado de `filters.py`:
- Filtro 1: Descarte de amostras vazias ou com mais de 80 palavras.
- Filtro 2: Verificação se a sintaxe gerada corresponde à categoria solicitada (heurística leve/regex).
- Filtro 3: Prompt de detecção de injeção em LLM (adotando a persona de Eliezer Yudkowsky), exigindo a resposta "YES".
- Filtro 4: Verificação de similaridade em relação a amostras já validadas, descartando aquelas excessivamente semelhantes.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple


# ==============================================================================
# Ancoragem dinâmica na raiz do repositório
# ==============================================================================
def find_repo_root(start: Path) -> Path:
    """Sobe a partir de ``start`` até achar o diretório com ``datasets`` e ``Original``."""
    for candidate in (start, *start.parents):
        if (candidate / "datasets").is_dir() and (candidate / "Original").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Raiz do repositório não encontrada a partir de {start}: "
        "esperado um diretório contendo 'datasets' e 'Original'."
    )


ROOT_DIR = find_repo_root(Path(__file__).resolve().parent)

for _local_path in (ROOT_DIR, Path(__file__).resolve().parent):
    _entry = str(_local_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Import validation funnel and filters. O primeiro caminho vale para execução
# como módulo (python -m ...); o segundo, para execução direta do arquivo.
try:
    from .filters import ValidationFunnel
except ImportError:
    from filters import ValidationFunnel


def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve caminhos relativos contra ``ROOT_DIR`` (nunca contra o cwd)."""
    if value is None or str(value) == "":
        return Path(default).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()

# Model configuration
GENERATION_MODEL = "openai/gpt-4o-mini-2024-07-18"

# 1. Target distribution (Table 6 - 435 total samples)
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

# 2. Generation Prompts
SYSTEM_PROMPT = (
    "You are a professional data generator specializing in LLM security, "
    "known for creating indirect prompt injection samples with diverse templates."
)

USER_PROMPT_TEMPLATE = (
    "Below are some examples of indirect hint injections: {demonstrations} "
    "Please generate {number} prompt injection samples on the topic {key}, ensuring that each sample contains a realistic indirect prompt injection attack hidden within the context. "
    "The samples should be similar in style to the provided examples. "
    "Additionally, ensure that the generated samples are as diverse as possible in both content and template structure. "
    "Each sample should be no more than 50 words and presented in the following format: [INPUT]"
)


def load_api_key() -> str:
    """Load OpenRouter/OpenAI API key from environment or .env file."""
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_var)
        if val:
            return val.strip().strip("\"'")

    env_path = ROOT_DIR / ".env"
    if env_path.is_file():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip().lower() in ("openrouter_api_key", "openai_api_key"):
                        return v.strip().strip("\"'")

    raise RuntimeError(
        f"API key not found in environment or {env_path}."
    )


class FatalAPIError(RuntimeError):
    """Non-recoverable API error (e.g. HTTP 401/402/403 expired key or insufficient credits)."""


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
            if status in (401, 402, 403):
                raise FatalAPIError(f"HTTP {status} fatal error from OpenRouter: {err_body}") from e
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
    """Call the specified LLM to generate candidate samples."""
    payload = {
        "model": model if "/" in model else f"openai/{model}",
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
    """Clean and strip formatting artifacts from an extracted candidate chunk."""
    chunk = chunk.strip()

    # Strip trailing standalone number line if next marker was split awkwardly
    chunk = re.sub(r"\n\s*\d+[\.\)]\s*$", "", chunk).strip()

    # Strip **[INPUT]**: or [INPUT]: or **[Input]** etc.
    chunk = re.sub(r"^\*{0,2}\[input\]\*{0,2}:?\s*", "", chunk, flags=re.IGNORECASE).strip()
    chunk = re.sub(r"^\*\*Sample\s*\d+:?\*\*\s*", "", chunk, flags=re.IGNORECASE).strip()

    # Strip **[ ... ]** wrapper
    if chunk.startswith("**[") and chunk.endswith("]**"):
        chunk = chunk[3:-3].strip()

    # Strip **<!-- ... -->** bold around HTML comment
    chunk = re.sub(r"\*\*(<!--.*?-->)\*\*", r"\1", chunk)

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

    # Strip outer [ ... ] or dangling [ if it was used as the [INPUT] wrapper
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

    # Unescape double-escaped quotes
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
    """Extract individual samples from raw model response."""
    text = raw_text.strip()
    items: List[str] = []

    # Strip markdown code block wrapper if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    # 1. Check if [INPUT] markers are used (supports 1 or more samples)
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

    # 2. Try JSON list parsing if the model returned valid JSON array
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
            # Fallback regex extraction for {"text": "..."} when internal unescaped quotes break json.loads
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


# =====================================================================
# Category Generation & Validation Funnel Workflow
# =====================================================================

def _load_existing_checkpoint(
    output_dir: str,
    tag: str,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict], float]:
    """Load previously validated samples and category details from output_dir if available."""
    samples_by_key: Dict[str, List[str]] = {k: [] for k in BASE_DISTRIBUTION}
    details_by_key: Dict[str, Dict] = {}
    prev_elapsed = 0.0

    # Candidate sample files in priority order
    candidate_files = [
        os.path.join(output_dir, "LLM_augmented.json"),
        os.path.join(output_dir, f"generated_malicious_samples_{tag}.json"),
        os.path.join(output_dir, "LLM_augmented.partial.json"),
        os.path.join(output_dir, f"generated_malicious_samples_{tag}.partial.json"),
    ]
    for path in candidate_files:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list) or not records:
                continue
            temp_by_key: Dict[str, List[str]] = {k: [] for k in BASE_DISTRIBUTION}
            for row in records:
                if not isinstance(row, dict):
                    continue
                prompt = row.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                topic = row.get("topic")
                fmt = row.get("format")
                if not topic and isinstance(fmt, str):
                    topic = f"{fmt} Injection"
                if topic in temp_by_key:
                    base_c = BASE_DISTRIBUTION[topic]
                    if len(temp_by_key[topic]) < base_c:
                        temp_by_key[topic].append(prompt)
            if sum(len(v) for v in temp_by_key.values()) > 0:
                samples_by_key = temp_by_key
                print(
                    f"[RESUME] Loaded {sum(len(v) for v in samples_by_key.values())}/435 "
                    f"validated samples from: {path}"
                )
                break
        except Exception as e:
            print(f"[WARNING] Could not read checkpoint {path}: {e}")

    raw_path = os.path.join(output_dir, f"generated_malicious_raw_responses_{tag}.json")
    if os.path.isfile(raw_path):
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict):
                prev_elapsed = float(raw_data.get("elapsed_seconds", 0.0) or 0.0)
                cat_details = raw_data.get("category_details", {})
                if isinstance(cat_details, dict):
                    details_by_key = cat_details
        except Exception as e:
            print(f"[WARNING] Could not read raw responses log {raw_path}: {e}")

    return samples_by_key, details_by_key, prev_elapsed


def generate_and_filter_category(
    key: str,
    base_count: int,
    demos: List[str],
    model: str,
    api_key: str,
    temperature: float = 1.0,
    seed: int = 42,
    max_attempts: int = 300,
    initial_samples: List[str] = None,
    initial_detail: Dict = None,
) -> Tuple[str, str, int, List[str], Dict]:
    """
    Para cada um dos 17 tipos (X) com meta N_X definida na Tabela 6 (BASE_DISTRIBUTION):
    1. Inicializa valid_samples_X (reaproveitando amostras já validadas se houver checkpoint).
    2. Enquanto len(valid_samples_X) < N_X:
       - Calcula quantas faltam: missing = N_X - len(valid_samples_X).
       - Seleciona 3 exemplos aleatórios dentre os disponíveis para a categoria
         (para os tipos que só tiverem 3 exemplos utiliza todos).
       - Pede ao modelo (com temperature no máximo) em lote.
       - Passa cada amostra gerada pelo funil de validação (Filtros 1, 2, 3 e 4 via ValidationFunnel).
       - Adiciona apenas as aprovadas em valid_samples_X.
    3. Assim que atingir exatamente N_X, encerra o laço daquele formato.
    """
    fmt = key.replace(" Injection", "")
    rng = random.Random(seed + list(BASE_DISTRIBUTION).index(key))

    valid_samples_X: List[str] = list((initial_samples or [])[:base_count])
    seen_exact_set: Set[str] = {s.strip().lower() for s in valid_samples_X}

    prev_stats = (initial_detail or {}).get("stats", {}) if isinstance(initial_detail, dict) else {}
    prev_raw = (initial_detail or {}).get("raw_responses", []) if isinstance(initial_detail, dict) else []

    stats = {
        "attempts": int(prev_stats.get("attempts", 0)),
        "candidates_parsed": int(prev_stats.get("candidates_parsed", 0)),
        "rejected_filter1_words": int(prev_stats.get("rejected_filter1_words", 0)),
        "rejected_filter2_syntax": int(prev_stats.get("rejected_filter2_syntax", 0)),
        "rejected_filter3_judge": int(prev_stats.get("rejected_filter3_judge", 0)),
        "rejected_filter4_similarity": int(prev_stats.get("rejected_filter4_similarity", 0)),
        "accepted_total": len(valid_samples_X),
    }
    raw_responses: List[str] = list(prev_raw) if isinstance(prev_raw, list) else []

    # Advance RNG past already-executed attempts so resumed runs sample new demo combinations
    for _ in range(stats["attempts"]):
        if len(demos) > 3:
            rng.sample(demos, 3)

    if len(valid_samples_X) >= base_count:
        stats["accepted_total"] = base_count
        stats["complete"] = True
        return key, fmt, base_count, valid_samples_X[:base_count], {
            "stats": stats,
            "raw_responses": raw_responses,
        }

    # Initialize ValidationFunnel with dependency-injected OpenRouter caller
    funnel = ValidationFunnel(
        api_key=api_key,
        call_fn=lambda payload, key_val: call_openrouter(payload, key_val, max_retries=3, timeout=45),
    )

    new_attempts = 0
    while len(valid_samples_X) < base_count and new_attempts < max_attempts:
        new_attempts += 1
        stats["attempts"] += 1
        attempt = stats["attempts"]

        missing = base_count - len(valid_samples_X)
        ask_num = min(max(missing + 3, 6), 10)

        # Select 3 random examples from available category demos (or all if <= 3)
        sampled_demos = rng.sample(demos, 3) if len(demos) > 3 else list(demos)
        demo_str = json.dumps(sampled_demos, ensure_ascii=False)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            demonstrations=demo_str,
            number=ask_num,
            key=key,
        )

        try:
            raw = call_generation_model(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                api_key=api_key,
                temperature=temperature,
            )
            raw_responses.append(raw)
        except FatalAPIError:
            raise
        except Exception as e:
            print(f"  [ERROR] {key} attempt {attempt} API call failed: {e}")
            continue

        if ("I'm sorry" in raw or "I cannot" in raw) and len(raw) < 200:
            continue

        candidates = parse_samples(raw, fmt)
        stats["candidates_parsed"] += len(candidates)

        for cand in candidates:
            if len(valid_samples_X) >= base_count:
                break

            # Process candidate through the 4-filter validation funnel
            is_valid, failed_filter, _ = funnel.validate_sample(
                sample=cand,
                fmt=fmt,
                valid_samples=valid_samples_X,
                seen_exact_set=seen_exact_set,
            )

            if not is_valid:
                if failed_filter == "filter1":
                    stats["rejected_filter1_words"] += 1
                elif failed_filter == "filter2":
                    stats["rejected_filter2_syntax"] += 1
                elif failed_filter == "filter3":
                    stats["rejected_filter3_judge"] += 1
                elif failed_filter == "filter4":
                    stats["rejected_filter4_similarity"] += 1
                continue

            valid_samples_X.append(cand)
            seen_exact_set.add(cand.strip().lower())

        print(
            f"  [{fmt}] Attempt {attempt}: {len(valid_samples_X)}/{base_count} validated "
            f"(missing={base_count - len(valid_samples_X)})"
        )

    final_samples = valid_samples_X[:base_count]
    stats["accepted_total"] = len(final_samples)
    stats["complete"] = len(final_samples) == base_count

    return key, fmt, base_count, final_samples, {
        "stats": stats,
        "raw_responses": raw_responses,
    }


def run_pipeline_for_model(
    model_name: str = GENERATION_MODEL,
    api_key: str = "",
    demos_by_format: Dict[str, List[str]] = None,
    workers: int = 1,
    output_dir: str = str(ROOT_DIR / "Corrections" / "outputs" / "augmented"),
    temperature: float = 1.0,
    seed: int = 42,
    max_attempts: int = 300,
    resume: bool = True,
) -> Tuple[str, int]:
    """Execute the data generation and 4-filter validation funnel across all 17 categories."""
    tag = "gpt4o_mini"
    print("\n=================================================================")
    print(f"Starting Generation & 4-Filter Funnel for Model: {model_name} ({tag})")
    print(f"Temperature: {temperature} | Workers: {workers} | Max attempts/run: {max_attempts}")
    print(f"Target count: {sum(BASE_DISTRIBUTION.values())} across {len(BASE_DISTRIBUTION)} categories")
    print("=================================================================")

    start_time = time.time()
    results_by_key = {}
    details_log = {}
    existing_samples_by_key: Dict[str, List[str]] = {k: [] for k in BASE_DISTRIBUTION}
    existing_details_by_key: Dict[str, Dict] = {}
    prev_elapsed = 0.0

    if resume:
        existing_samples_by_key, existing_details_by_key, prev_elapsed = _load_existing_checkpoint(
            output_dir=output_dir,
            tag=tag,
        )

    pending_keys = []
    for key, base_count in BASE_DISTRIBUTION.items():
        fmt = key.replace(" Injection", "")
        prev_samples = existing_samples_by_key.get(key, [])[:base_count]
        prev_detail = existing_details_by_key.get(key)
        if len(prev_samples) >= base_count:
            if not isinstance(prev_detail, dict) or "stats" not in prev_detail:
                prev_detail = {
                    "stats": {
                        "attempts": 0,
                        "candidates_parsed": len(prev_samples),
                        "rejected_filter1_words": 0,
                        "rejected_filter2_syntax": 0,
                        "rejected_filter3_judge": 0,
                        "rejected_filter4_similarity": 0,
                        "accepted_total": len(prev_samples),
                        "complete": True,
                    },
                    "raw_responses": [],
                }
            else:
                prev_detail["stats"]["accepted_total"] = len(prev_samples)
                prev_detail["stats"]["complete"] = True
            results_by_key[key] = (fmt, base_count, prev_samples)
            details_log[key] = prev_detail
            print(f"  [RESUME/DONE] {key} ({fmt}): {len(prev_samples)}/{base_count} already validated (skipping)")
        else:
            if prev_samples:
                print(
                    f"  [RESUME/PARTIAL] {key} ({fmt}): {len(prev_samples)}/{base_count} validated "
                    f"(generating remaining {base_count - len(prev_samples)})"
                )
            pending_keys.append((key, base_count, prev_samples, prev_detail))

    if not pending_keys:
        print("\nAll 17 categories (435/435 samples) are already validated!")
    elif workers <= 1 or len(pending_keys) == 1:
        for key, base_count, prev_samples, prev_detail in pending_keys:
            fmt = key.replace(" Injection", "")
            print(
                f"\n>>> Generating category: {key} "
                f"(current = {len(prev_samples)}/{base_count}, available demos = {len(demos_by_format[fmt])})"
            )
            cat_key, fmt, base_c, samples, detail = generate_and_filter_category(
                key=key,
                base_count=base_count,
                demos=demos_by_format[fmt],
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                seed=seed,
                max_attempts=max_attempts,
                initial_samples=prev_samples,
                initial_detail=prev_detail,
            )
            results_by_key[cat_key] = (fmt, base_c, samples)
            details_log[cat_key] = detail
            st = detail["stats"]
            print(
                f"  [DONE] {cat_key} ({fmt}): "
                f"target={base_c}, final={len(samples)}, "
                f"attempts={st['attempts']}, parsed={st['candidates_parsed']}, "
                f"rej_F1={st['rejected_filter1_words']}, "
                f"rej_F2={st['rejected_filter2_syntax']}, "
                f"rej_F3={st['rejected_filter3_judge']}, "
                f"rej_F4={st['rejected_filter4_similarity']}"
            )
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending_keys))) as executor:
            futures = {
                executor.submit(
                    generate_and_filter_category,
                    key,
                    base_count,
                    demos_by_format[key.replace(" Injection", "")],
                    model_name,
                    api_key,
                    temperature,
                    seed,
                    max_attempts,
                    prev_samples,
                    prev_detail,
                ): key
                for key, base_count, prev_samples, prev_detail in pending_keys
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    cat_key, fmt, base_c, samples, detail = future.result()
                    results_by_key[cat_key] = (fmt, base_c, samples)
                    details_log[cat_key] = detail
                    st = detail["stats"]
                    print(
                        f"  [DONE] {cat_key} ({fmt}): "
                        f"target={base_c}, final={len(samples)}, "
                        f"attempts={st['attempts']}, parsed={st['candidates_parsed']}, "
                        f"rej_F1={st['rejected_filter1_words']}, "
                        f"rej_F2={st['rejected_filter2_syntax']}, "
                        f"rej_F3={st['rejected_filter3_judge']}, "
                        f"rej_F4={st['rejected_filter4_similarity']}"
                    )
                except FatalAPIError:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    print(f"  [FAILED] {key}: {e}")

    # Build final records in exact BASE_DISTRIBUTION order
    all_records = []
    summary_report = []

    for key, base_count in BASE_DISTRIBUTION.items():
        if key in results_by_key:
            fmt, base_c, samples = results_by_key[key]
        else:
            fmt = key.replace(" Injection", "")
            base_c = base_count
            samples = existing_samples_by_key.get(key, [])[:base_count]

        summary_report.append({
            "topic": key,
            "format": fmt,
            "model": model_name,
            "base_count": base_c,
            "generated_and_validated_count": len(samples),
            "stats": details_log.get(key, {}).get("stats", {}),
        })

        for s in samples:
            all_records.append({
                "prompt": s,
                "label": 1,  # Malicious (indirect prompt injection)
                "source": "LLM Augmented set",
                "format": fmt,
                "topic": key,
                "model": model_name,
            })

    os.makedirs(output_dir, exist_ok=True)
    complete = len(all_records) == sum(BASE_DISTRIBUTION.values())
    suffix = ".json" if complete else ".partial.json"
    out_path = os.path.join(output_dir, f"generated_malicious_samples_{tag}{suffix}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)

    llm_aug_path = os.path.join(output_dir, f"LLM_augmented{suffix}")
    with open(llm_aug_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=4, ensure_ascii=False)

    total_elapsed = round(prev_elapsed + (time.time() - start_time), 2)
    raw_path = os.path.join(output_dir, f"generated_malicious_raw_responses_{tag}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "seed": seed,
            "status": "complete" if complete else "incomplete",
            "total_samples": len(all_records),
            "elapsed_seconds": total_elapsed,
            "summary": summary_report,
            "category_details": details_log,
        }, f, indent=4, ensure_ascii=False)

    elapsed = round(time.time() - start_time, 1)
    print(f"\nModel {model_name} finished in {elapsed}s (cumulative {total_elapsed}s).")
    print(f"Saved {len(all_records)} verified samples (label=1) to: {out_path}")
    print(f"Saved raw responses and pipeline log to: {raw_path}")

    if not complete:
        raise RuntimeError(f"Geração incompleta: {len(all_records)}/435. Consulte {raw_path}.")
    return out_path, len(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator with 4-Filter Validation Funnel"
    )
    parser.add_argument(
        "--model",
        default=GENERATION_MODEL,
        help=f"Model to generate data with (default: {GENERATION_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent category worker threads (default: 5)",
    )
    parser.add_argument(
        "--curated-path",
        default=str(
            ROOT_DIR
            / "datasets"
            / "Augmented"
            / "Cache"
            / "curated_prompt_injections.json"
        ),
        help="Path to curated demonstrations JSON (padrão: <repo>/datasets/Augmented/Cache).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "Corrections" / "outputs" / "augmented"),
        help="Output directory for generated files (padrão: <repo>/Corrections/outputs/augmented).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=300, help="Máximo de lotes por categoria por execução.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora checkpoints (.partial.json / .json) existentes em --output-dir e gera tudo do zero.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1:
        parser.error("workers e max-attempts devem ser positivos")

    api_key = load_api_key()

    curated_path = resolve_path(
        args.curated_path,
        ROOT_DIR / "datasets/Augmented/Cache/curated_prompt_injections.json",
    )
    if not curated_path.is_file():
        local_fallback = Path(__file__).resolve().parent / "curated_prompt_injections.json"
        if local_fallback.is_file():
            curated_path = local_fallback
        else:
            raise FileNotFoundError(f"Curated file not found: {curated_path}")

    with open(curated_path, "r", encoding="utf-8") as f:
        curated_data = json.load(f)

    demos_by_format: Dict[str, List[str]] = {}
    for item in curated_data:
        fmt = item["format"]
        demos_by_format.setdefault(fmt, []).append(item["prompt"])

    for fmt, demos in demos_by_format.items():
        if len(demos) < 3:
            print(f"[WARNING] Format '{fmt}' has fewer than 3 demos ({len(demos)})")
    missing = [key for key in BASE_DISTRIBUTION if not demos_by_format.get(key.replace(" Injection", ""))]
    if missing:
        raise ValueError(f"Categorias sem demonstrações: {missing}")

    output_dir = resolve_path(
        args.output_dir, ROOT_DIR / "Corrections/outputs/augmented"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline_for_model(
        model_name=args.model,
        api_key=api_key,
        demos_by_format=demos_by_format,
        workers=args.workers,
        output_dir=str(output_dir),
        temperature=args.temperature,
        seed=args.seed,
        max_attempts=args.max_attempts,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()

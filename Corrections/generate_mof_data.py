"""Gera MOF a partir de gatilhos sanitizados, reutilizando os prompts do Cache."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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

for _local_path in (ROOT_DIR, ROOT_DIR / "Corrections"):
    _entry = str(_local_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

try:  # execução como módulo (python -m Corrections.generate_mof_data)
    from Corrections.Upgrade_augmented.filters import validate_filter4_similarity
except ImportError:  # execução direta (python Corrections/generate_mof_data.py)
    from Upgrade_augmented.filters import validate_filter4_similarity

REFERENCE_PATH = ROOT_DIR / "datasets/MOF/Cache/generate_mof_data.py"
spec = importlib.util.spec_from_file_location("mof_reference", REFERENCE_PATH)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve caminhos relativos contra ``ROOT_DIR`` (nunca contra o cwd)."""
    if value is None or str(value) == "":
        return Path(default).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def load_api_key() -> str:
    """Carrega a chave do OpenRouter do ambiente ou de ``<raiz>/.env``."""
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(env_var)
        if value:
            return value.strip().strip("\"'")
    env_path = ROOT_DIR / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip().lower() in ("openrouter_api_key", "openai_api_key"):
                    return value.strip().strip("\"'")
    raise RuntimeError(
        f"API key não encontrada em OPENROUTER_API_KEY/OPENAI_API_KEY ou {env_path}."
    )


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def contains_chosen_words(text, chosen_words):
    return all(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None for word in chosen_words)


def generate_candidate(sample_id, words, api_key, seed):
    # O sorteio ocorre por ID, independentemente da ordem de conclusão das chamadas.
    rng = random.Random(seed + sample_id)
    chosen_words = rng.sample(words, min(rng.randint(1, 3), len(words)))
    candidate = {"id": sample_id, "chosen_words": chosen_words}
    try:
        response = reference.call_openrouter({
            "model": reference.MODEL_NAME,
            "messages": [
                {"role": "system", "content": reference.GEN_SYSTEM_PROMPT},
                {"role": "user", "content": reference.GEN_USER_PROMPT_TEMPLATE.format(word_list=", ".join(chosen_words))},
            ],
            "temperature": 0.8,
            "max_tokens": 256,
        }, api_key)
        raw = response["choices"][0]["message"].get("content") or ""
        text = reference.extract_instruction(raw)
        candidate.update(raw_response=raw, instruction=text)
        if not text or not contains_chosen_words(text, chosen_words):
            candidate.update(approved=False, reason="missing_chosen_words")
            return candidate
        judged = reference.judge_single_sample(candidate, api_key)
        answer = judged["judge_response"]
        approved = re.fullmatch(r"NO[.!]?", answer.strip(), re.IGNORECASE) is not None
        candidate.update(judge_response=answer, approved=approved,
                         reason="accepted_by_judge" if approved else "judge_rejected_or_ambiguous")
    except Exception as exc:
        candidate.update(approved=False, reason="api_error", error=str(exc))
    return candidate


def select_approved(candidates, target):
    """Aplica a checagem de similaridade existente em ordem de ID."""
    selected = []
    texts = []
    exact = set()
    counts = Counter()
    for candidate in sorted(candidates, key=lambda item: item["id"]):
        if not candidate["approved"]:
            counts[candidate["reason"]] += 1
            continue
        text = candidate["instruction"]
        valid, reason = validate_filter4_similarity(text, texts, exact)
        if not valid:
            counts["similarity"] += 1
            continue
        selected.append(candidate)
        texts.append(text)
        exact.add(text.strip().lower())
        if len(selected) == target:
            break
    return selected, dict(counts)


def run(trigger_file, output, *, initial_target=1000, final_target=762, workers=5,
        seed=42, max_candidates=10000, api_key=None):
    trigger_file = resolve_path(trigger_file, ROOT_DIR / "Corrections/outputs/trigger_words.json")
    output = resolve_path(output, ROOT_DIR / "Corrections/outputs/MOF.json")
    words = json.loads(trigger_file.read_text(encoding="utf-8"))
    if not isinstance(words, list) or any(not isinstance(word, str) or not word.strip() for word in words):
        raise ValueError("A lista de gatilhos deve conter apenas strings não vazias")
    words = list(dict.fromkeys(words))
    report_path = output.with_suffix(".report.json")
    if not words:
        save_json(output, [])
        save_json(report_path, {"status": "no_triggers", "n": 0, "trigger_file": str(trigger_file)})
        print("Lista vazia: MOF não aplicável a este checkpoint.")
        return []
    if min(initial_target, final_target, workers, max_candidates) < 1 or max_candidates < initial_target:
        raise ValueError("Metas e workers devem ser positivos; max_candidates deve cobrir initial_target")

    manifest = {
        "trigger_sha256": hashlib.sha256(trigger_file.read_bytes()).hexdigest(),
        "seed": seed, "model": reference.MODEL_NAME,
        "initial_target": initial_target, "final_target": final_target,
        "reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "filters_sha256": hashlib.sha256((ROOT_DIR / "Corrections/Upgrade_augmented/filters.py").read_bytes()).hexdigest(),
    }
    recheck_report = trigger_file.with_suffix(".report.json")
    if recheck_report.is_file():
        manifest["recheck_report_sha256"] = hashlib.sha256(recheck_report.read_bytes()).hexdigest()
    cache_path = output.with_suffix(".candidates.json")
    candidates = []
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["manifest"] != manifest:
            raise ValueError(f"Cache de outra configuração: use outro arquivo de saída ({cache_path})")
        candidates = cached["candidates"]
    selected, counts = select_approved(candidates, final_target)
    if len(selected) < final_target:
        api_key = api_key or load_api_key()

    while len(selected) < final_target and len(candidates) < max_candidates:
        remaining = max(initial_target - len(candidates), 50)
        batch_count = min(remaining, max_candidates - len(candidates))
        start_id = max((item["id"] for item in candidates), default=-1) + 1
        # Salvar a cada microlote limita a repetição de chamadas após uma interrupção.
        for offset in range(0, batch_count, workers):
            ids = range(start_id + offset, start_id + min(offset + workers, batch_count))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(lambda idx: generate_candidate(idx, words, api_key, seed), ids))
            candidates.extend(results)
            save_json(cache_path, {"manifest": manifest, "candidates": candidates})
            if all(item["reason"] == "api_error" for item in results):
                raise RuntimeError(f"Lote falhou na API; detalhes preservados em {cache_path}")
        selected, counts = select_approved(candidates, final_target)
        print(f"MOF: {len(selected)}/{final_target} aprovadas entre {len(candidates)} candidatos.")

    status = "complete" if len(selected) == final_target else "incomplete"
    records = [{"prompt": item["instruction"], "source": "over-defense", "label": 0} for item in selected]
    save_json(report_path, {
        "manifest": manifest, "status": status, "n": len(records),
        "candidate_count": len(candidates), "rejections": counts,
        "words_per_sample": dict(Counter(len(item["chosen_words"]) for item in selected)),
        "selected_ids": [item["id"] for item in selected],
    })
    destination = output if status == "complete" else output.with_suffix(".partial.json")
    save_json(destination, records)
    if status != "complete":
        raise RuntimeError(f"MOF incompleto: {len(records)}/{final_target}. Consulte {report_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trigger_words",
        required=True,
        help="JSON com a lista de gatilhos sanitizados.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Arquivo JSON de saída do MOF.",
    )
    parser.add_argument("--initial_target", type=int, default=1000)
    parser.add_argument("--final_target", type=int, default=762)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_candidates", type=int, default=10000)
    args = parser.parse_args()
    run(args.trigger_words, args.output, initial_target=args.initial_target,
        final_target=args.final_target, workers=args.workers, seed=args.seed,
        max_candidates=args.max_candidates)


if __name__ == "__main__":
    main()

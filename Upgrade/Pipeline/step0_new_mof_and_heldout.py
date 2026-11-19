"""Etapa 0 do Pipeline Upgrade: Geração do Pool do Novo MOF e Separação dos Excedentes Held-Out.

Premissas metodológicas implementadas:
1. Nenhuma amostra é removida de `train_cleaned_original.json` ou `train_augmented_corrected.json`;
2. Os arquivos antigos `MOF.json` e `MOF.candidates.json` não são utilizados;
3. `datasets/valid.json` não é utilizado para calibração (por conter sobreposição com benchmarks);
4. A partir dos 253 gatilhos de `Corrections/outputs/augmented_corrected/trigger_words.report.json`, gera-se
   um pool novo de candidatos benignos aprovados (amostragem ponderada 20% uniforme / 80% severidade,
   4 domínios críticos, <= 256 tokens DeBERTa, presença estrita dos k gatilhos distintos, resposta NO
   do juiz LLM e filtro de similaridade + disjunção total contra benchmarks e dados de treino);
5. Avaliam-se os candidatos aprovados com `Corrections/outputs/augmented_corrected/best_model.pth` e
   separa-se de forma 100% disjunta:
   - `calibration_heldout.json`: 238 excedentes benignos estratificados (48/71/119) + 102 ataques
     gerados pelo funil de 4 filtros (6 por formato, fora dos 435 de `LLM_augmented.json`);
   - `MOF_v2_train.json`: 762 exemplos benignos (152/229/381) via Seleção Contextual Hard-Negative
     (80% maiores scores P_PI(inj | x) + 20% aleatórios).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Permite executar este arquivo diretamente de qualquer diretório.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import (
    ALL_BENCHMARK_SETS,
    benchmark_file_path,
    default_corrections_dir,
    load_api_key,
    normalize_text,
    read_json,
    resolve_path,
    sha256_file,
    write_json,
)

# Importações do funil existente em Corrections/ sem duplicar prompts ou heurísticas.
from Corrections.Upgrade_augmented.filters import (
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE,
    ValidationFunnel,
    validate_filter4_similarity,
)
from Corrections.Upgrade_augmented.new_generate_data import (
    BASE_DISTRIBUTION,
    GENERATION_MODEL,
    SYSTEM_PROMPT as AUG_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE as AUG_USER_PROMPT_TEMPLATE,
    FatalAPIError,
    call_generation_model,
    call_openrouter,
    parse_samples,
)

MOF_REF_PATH = REPO_ROOT / "datasets" / "MOF" / "Cache" / "generate_mof_data.py"
DEFAULT_UPGRADE_OUTPUT = REPO_ROOT / "Upgrade" / "outputs"


def _load_mof_reference():
    spec = importlib.util.spec_from_file_location("upgrade_mof_reference", MOF_REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Cotas exatas do Novo MOF (762) e dos excedentes held-out (238) sobre o pool base de 1.000.
POOL_QUOTAS: Dict[int, int] = {1: 200, 2: 300, 3: 500}
DEFAULT_POOL_BUFFER: Dict[int, int] = {1: 20, 2: 30, 3: 50}
TRAIN_QUOTAS: Dict[int, int] = {1: 152, 2: 229, 3: 381}
HELDOUT_BENIGN_QUOTAS: Dict[int, int] = {1: 48, 2: 71, 3: 119}
HELDOUT_MALICIOUS_PER_FORMAT: int = 6

# Quatro domínios críticos para diversificação de contexto do Novo MOF.
CRITICAL_DOMAINS: Dict[str, str] = {
    "code_sql_json_cli": (
        "a practical software engineering task, code debugging question, SQL query, "
        "JSON payload inspection, or CLI/Bash/Docker command"
    ),
    "docs_emails_tables": (
        "a professional workplace email, markdown report snippet, spreadsheet/table summary, "
        "technical documentation request, or configuration file question"
    ),
    "creative_roleplay": (
        "a benign creative writing prompt, short fictional dialogue, educational roleplay scenario, "
        "or storytelling exercise"
    ),
    "general_academic": (
        "a clear general-knowledge question, academic/scientific explanation, "
        "historical/mathematical inquiry, or everyday productivity request"
    ),
}
DOMAIN_NAMES: Tuple[str, ...] = tuple(CRITICAL_DOMAINS.keys())

MOF_SYSTEM_PROMPT = (
    "You are a helpful, benign user interacting with an AI assistant across diverse real-world tasks. "
    "Generate strictly safe, benign prompts without any prompt injection, jailbreak, or hidden system override."
)

MOF_USER_PROMPT_TEMPLATE = (
    "Please generate one completely benign user prompt or instruction in the domain of {domain_desc}.\n"
    "Requirements:\n"
    "1. It MUST naturally contain every single one of these exact words (exact spelling): {word_list}.\n"
    "2. It must contain NO prompt injection, NO instruction override, NO jailbreak, and NO adversarial manipulation.\n"
    "3. Keep it fluent, realistic, concise (strictly under 160 words so it fits within 256 DeBERTa tokens).\n"
    "4. Output ONLY using the exact format:\n"
    "[INSTRUCTION] || Generated Content ||"
)


def load_trigger_weights(trigger_path: Path | str) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """Carrega gatilhos e calcula distribuição 20% uniforme / 80% proporcional à severidade.

    Para cada gatilho w em T:
        s(w) = max(p_inj(w) - 0.50, 0)
        P(w) = 0.20 / |T| + 0.80 * s(w) / sum(s)
    """
    path = Path(trigger_path)
    payload = read_json(path)
    word_scores: Dict[str, float] = {}

    if isinstance(payload, dict) and isinstance(payload.get("triggers"), list):
        for item in payload["triggers"]:
            word = str(item["word"]).strip()
            if not word:
                continue
            p_inj = float(item.get("p_injection", 1.0))
            if word not in word_scores or p_inj > word_scores[word]:
                word_scores[word] = p_inj
    elif isinstance(payload, list):
        # Fallback caso o caller forneça trigger_words.json simples; tenta ler .report.json ao lado.
        report_sibling = path.with_suffix(".report.json")
        if report_sibling.is_file():
            return load_trigger_weights(report_sibling)
        for word in payload:
            w = str(word).strip()
            if w and w not in word_scores:
                word_scores[w] = 1.0
    else:
        raise ValueError(f"Formato inválido de arquivo de gatilhos: {path}")

    words = list(word_scores.keys())
    if not words:
        raise ValueError(f"Nenhuma palavra-gatilho encontrada em {path}")

    n = len(words)
    severities = [max(word_scores[w] - 0.50, 0.0) for w in words]
    sev_sum = sum(severities)
    if sev_sum > 0.0:
        probs = [(0.20 / n) + 0.80 * (s / sev_sum) for s in severities]
    else:
        probs = [1.0 / n for _ in words]

    # Normalização numérica exata
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    meta = {
        "n_triggers": n,
        "severity_sum": sev_sum,
        "min_p_inj": min(word_scores.values()),
        "max_p_inj": max(word_scores.values()),
        "min_weight": min(probs),
        "max_weight": max(probs),
    }
    return words, probs, meta


def sample_weighted_triggers_without_replacement(
    words: Sequence[str],
    probs: Sequence[float],
    k: int,
    rng: random.Random,
) -> List[str]:
    """Amostra k gatilhos distintos sem reposição de acordo com os pesos P(w)."""
    if k < 1:
        raise ValueError("k deve ser >= 1")
    if k > len(words):
        raise ValueError(f"k={k} excede o número de gatilhos disponíveis ({len(words)})")

    pool_words = list(words)
    pool_weights = list(probs)
    chosen: List[str] = []
    for _ in range(k):
        total = sum(pool_weights)
        if total <= 0.0:
            idx = rng.randrange(len(pool_words))
        else:
            r = rng.random() * total
            acc = 0.0
            idx = len(pool_words) - 1
            for j, w in enumerate(pool_weights):
                acc += w
                if r <= acc:
                    idx = j
                    break
        chosen.append(pool_words.pop(idx))
        pool_weights.pop(idx)
    return chosen


def contains_chosen_words(text: str, chosen_words: Sequence[str]) -> bool:
    """Verifica presença estrita de todos os gatilhos via fronteira de palavra."""
    return all(
        re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None
        for word in chosen_words
    )


def _extract_texts_from_json(path: Path) -> List[str]:
    data = read_json(path)
    if isinstance(data, list):
        out = []
        for row in data:
            if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                out.append(row["prompt"])
            elif isinstance(row, str):
                out.append(row)
        return out
    if isinstance(data, dict):
        out = []
        for val in data.values():
            if isinstance(val, list):
                out.extend(str(item) for item in val)
        return out
    return []


def load_forbidden_texts(
    repo_root: Path | str = REPO_ROOT,
    *,
    extra_paths: Optional[Iterable[Path | str]] = None,
) -> Tuple[Set[str], Set[str], Dict[str, int]]:
    """Carrega conjuntos exatos e normalizados proibidos (valid.json, benchmarks e treino)."""
    repo = Path(repo_root).expanduser().resolve()
    candidate_files: List[Tuple[str, Path]] = [
        ("valid.json", repo / "datasets" / "valid.json"),
        ("train_cleaned_original.json", repo / "datasets" / "Augmented" / "train_cleaned_original.json"),
    ]
    for name in ALL_BENCHMARK_SETS:
        candidate_files.append((name, benchmark_file_path(repo / "datasets", name)))

    if extra_paths:
        for p in extra_paths:
            path_obj = Path(p)
            candidate_files.append((path_obj.name, path_obj))

    exact_lower: Set[str] = set()
    normalized: Set[str] = set()
    counts: Dict[str, int] = {}

    for label, path in candidate_files:
        if not path.is_file():
            continue
        texts = _extract_texts_from_json(path)
        counts[label] = len(texts)
        for text in texts:
            exact_lower.add(text.strip().lower())
            normalized.add(normalize_text(text))
    return exact_lower, normalized, counts


def _judge_benign_no(
    text: str,
    api_key: str,
    *,
    judge_model: str = JUDGE_MODEL,
    call_fn: Optional[Callable[[dict, str], dict]] = None,
) -> Tuple[bool, str]:
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_PROMPT_TEMPLATE.format(instruction=text)},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    resp = call_fn(payload, api_key) if call_fn is not None else call_openrouter(payload, api_key, max_retries=4, timeout=45)
    answer = (resp["choices"][0]["message"].get("content") or "").strip()
    is_no = re.fullmatch(r"NO[.!]?", answer, re.IGNORECASE) is not None
    return is_no, answer


def validate_mof_candidate(
    text: str,
    chosen_words: Sequence[str],
    tokenizer: Any,
    accepted_texts: List[str],
    forbidden_exact_set: Set[str],
    api_key: str,
    *,
    forbidden_norm_set: Optional[Set[str]] = None,
    call_fn: Optional[Callable[[dict, str], dict]] = None,
    max_deberta_tokens: int = 256,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Valida um candidato benigno do Novo MOF (gatilhos, comprimento, disjunção, similaridade e juiz NO)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False, "empty_text", {}

    if not contains_chosen_words(cleaned, chosen_words):
        return False, "missing_chosen_words", {}

    token_ids = tokenizer.encode(cleaned, add_special_tokens=True)
    n_tokens = len(token_ids)
    if n_tokens > max_deberta_tokens:
        return False, f"token_limit_exceeded ({n_tokens} > {max_deberta_tokens})", {"n_tokens": n_tokens}

    lowered = cleaned.lower()
    normed = normalize_text(cleaned)
    if lowered in forbidden_exact_set or (forbidden_norm_set is not None and normed in forbidden_norm_set):
        return False, "forbidden_overlap", {"n_tokens": n_tokens}

    seen_exact = set(forbidden_exact_set) | {t.strip().lower() for t in accepted_texts}
    sim_ok, sim_reason = validate_filter4_similarity(cleaned, accepted_texts, seen_exact)
    if not sim_ok:
        return False, f"similarity:{sim_reason}", {"n_tokens": n_tokens}

    try:
        is_no, judge_answer = _judge_benign_no(cleaned, api_key, call_fn=call_fn)
    except Exception as exc:
        return False, f"judge_error:{exc}", {"n_tokens": n_tokens}

    if not is_no:
        return False, "judge_rejected_or_ambiguous", {"n_tokens": n_tokens, "judge_response": judge_answer}

    return True, "accepted_by_judge", {"n_tokens": n_tokens, "judge_response": judge_answer}


def _generate_single_mof_candidate(
    sample_id: int,
    k: int,
    domain: str,
    words: Sequence[str],
    probs: Sequence[float],
    tokenizer: Any,
    forbidden_exact_set: Set[str],
    forbidden_norm_set: Set[str],
    api_key: str,
    seed: int,
    *,
    max_deberta_tokens: int = 256,
    call_fn: Optional[Callable[[dict, str], dict]] = None,
) -> Dict[str, Any]:
    ref = _load_mof_reference()
    rng = random.Random(seed + sample_id * 131 + k * 17)
    chosen_words = sample_weighted_triggers_without_replacement(words, probs, k, rng)
    candidate: Dict[str, Any] = {
        "id": sample_id,
        "k": k,
        "domain": domain,
        "chosen_words": chosen_words,
    }
    try:
        payload = {
            "model": GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": MOF_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": MOF_USER_PROMPT_TEMPLATE.format(
                        domain_desc=CRITICAL_DOMAINS[domain],
                        word_list=", ".join(chosen_words),
                    ),
                },
            ],
            "temperature": 0.85,
            "max_tokens": 300,
        }
        resp = call_fn(payload, api_key) if call_fn is not None else call_openrouter(payload, api_key, max_retries=4, timeout=45)
        raw = resp["choices"][0]["message"].get("content") or ""
        text = ref.extract_instruction(raw).strip()
        candidate["raw_response"] = raw
        candidate["instruction"] = text

        if not text or not contains_chosen_words(text, chosen_words):
            candidate.update(approved=False, reason="missing_chosen_words")
            return candidate

        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        candidate["n_tokens"] = n_tokens
        if n_tokens > max_deberta_tokens:
            candidate.update(approved=False, reason="token_limit_exceeded")
            return candidate

        if text.lower() in forbidden_exact_set or normalize_text(text) in forbidden_norm_set:
            candidate.update(approved=False, reason="forbidden_overlap")
            return candidate

        is_no, judge_answer = _judge_benign_no(text, api_key, call_fn=call_fn)
        candidate["judge_response"] = judge_answer
        candidate["approved"] = bool(is_no)
        candidate["reason"] = "accepted_by_judge" if is_no else "judge_rejected_or_ambiguous"
    except Exception as exc:
        candidate.update(approved=False, reason="api_error", error=str(exc))
    return candidate


def _accept_batch_into_pool(
    batch: Sequence[Dict[str, Any]],
    target_by_k: Dict[int, int],
    accepted_by_k: Dict[int, List[Dict[str, Any]]],
    accepted_texts: List[str],
    seen_exact: Set[str],
    rejections: Counter,
) -> None:
    """Processa novos candidatos em ordem determinística de ID contra o filtro de similaridade."""
    for cand in sorted(batch, key=lambda item: item["id"]):
        k = int(cand["k"])
        if k not in target_by_k:
            continue
        if len(accepted_by_k[k]) >= target_by_k[k]:
            continue
        if not cand.get("approved"):
            rejections[cand.get("reason", "unknown")] += 1
            continue
        text = str(cand.get("instruction", "")).strip()
        ok, _ = validate_filter4_similarity(text, accepted_texts, seen_exact)
        if not ok:
            rejections["similarity"] += 1
            continue
        accepted_by_k[k].append(cand)
        accepted_texts.append(text)
        seen_exact.add(text.lower())


def _collect_approved_pool(
    candidates: Sequence[Dict[str, Any]],
    target_by_k: Dict[int, int],
    forbidden_exact_set: Set[str],
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    """Aplica o filtro de similaridade sequencial em ordem determinística de ID por estrato k."""
    accepted_by_k: Dict[int, List[Dict[str, Any]]] = {k: [] for k in sorted(target_by_k)}
    accepted_texts: List[str] = []
    seen_exact: Set[str] = set(forbidden_exact_set)
    rejections: Counter = Counter()
    _accept_batch_into_pool(
        candidates, target_by_k, accepted_by_k, accepted_texts, seen_exact, rejections
    )
    return accepted_by_k, dict(rejections)


def generate_new_mof_pool(
    trigger_path: Path | str,
    cache_path: Path | str,
    forbidden_exact_set: Set[str],
    forbidden_norm_set: Set[str],
    *,
    tokenizer: Any = None,
    pool_quotas: Optional[Dict[int, int]] = None,
    pool_buffer: Optional[Dict[int, int]] = None,
    workers: int = 5,
    seed: int = 42,
    max_candidates: int = 5000,
    max_deberta_tokens: int = 256,
    api_key: Optional[str] = None,
    call_fn: Optional[Callable[[dict, str], dict]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gera o pool aprovado do Novo MOF estratificado por k in {1, 2, 3} e pelos 4 domínios."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    base_quotas = dict(pool_quotas or POOL_QUOTAS)
    buffer = dict(pool_buffer if pool_buffer is not None else DEFAULT_POOL_BUFFER)
    target_by_k = {k: int(base_quotas[k]) + int(buffer.get(k, 0)) for k in sorted(base_quotas)}

    words, probs, trigger_meta = load_trigger_weights(trigger_path)
    cache_file = Path(cache_path)
    candidates: List[Dict[str, Any]] = []
    if cache_file.is_file():
        cached = read_json(cache_file)
        if isinstance(cached, dict) and isinstance(cached.get("candidates"), list):
            candidates = cached["candidates"]

    accepted_by_k: Dict[int, List[Dict[str, Any]]] = {k: [] for k in sorted(target_by_k)}
    accepted_texts: List[str] = []
    seen_exact: Set[str] = set(forbidden_exact_set)
    rejections_counter: Counter = Counter()
    if candidates:
        _accept_batch_into_pool(
            candidates, target_by_k, accepted_by_k, accepted_texts, seen_exact, rejections_counter
        )
    needs_generation = any(len(accepted_by_k[k]) < target_by_k[k] for k in target_by_k)

    if needs_generation and call_fn is None:
        if not api_key:
            api_key = load_api_key(repo_root=REPO_ROOT)
        if not api_key:
            ref = _load_mof_reference()
            api_key = ref.load_api_key()
    api_key = api_key or ""

    next_id = max((int(c["id"]) for c in candidates), default=-1) + 1
    while any(len(accepted_by_k[k]) < target_by_k[k] for k in target_by_k) and len(candidates) < max_candidates:
        jobs: List[Tuple[int, int, str]] = []
        # Planeja lotes proporcionais ao déficit por estrato k e balanceados entre os 4 domínios.
        for k in sorted(target_by_k):
            deficit = target_by_k[k] - len(accepted_by_k[k])
            if deficit <= 0:
                continue
            domain_counts = Counter(item["domain"] for item in accepted_by_k[k])
            batch_for_k = min(max(deficit + 4, 8), max_candidates - len(candidates) - len(jobs))
            for _ in range(batch_for_k):
                chosen_domain = min(DOMAIN_NAMES, key=lambda d: (domain_counts[d], DOMAIN_NAMES.index(d)))
                domain_counts[chosen_domain] += 1
                jobs.append((next_id, k, chosen_domain))
                next_id += 1
        if not jobs:
            break

        for start in range(0, len(jobs), max(1, workers)):
            chunk = jobs[start : start + max(1, workers)]
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                results = list(
                    executor.map(
                        lambda spec: _generate_single_mof_candidate(
                            sample_id=spec[0],
                            k=spec[1],
                            domain=spec[2],
                            words=words,
                            probs=probs,
                            tokenizer=tokenizer,
                            forbidden_exact_set=forbidden_exact_set,
                            forbidden_norm_set=forbidden_norm_set,
                            api_key=api_key,
                            seed=seed,
                            max_deberta_tokens=max_deberta_tokens,
                            call_fn=call_fn,
                        ),
                        chunk,
                    )
                )
            candidates.extend(results)
            _accept_batch_into_pool(
                results, target_by_k, accepted_by_k, accepted_texts, seen_exact, rejections_counter
            )
            if results and all(item.get("reason") == "api_error" for item in results):
                write_json(
                    cache_file,
                    {
                        "trigger_meta": trigger_meta,
                        "target_by_k": target_by_k,
                        "candidates": candidates,
                    },
                )
                raise RuntimeError(f"Todas as chamadas de geração falharam; cache salvo em {cache_file}")

        write_json(
            cache_file,
            {
                "trigger_meta": trigger_meta,
                "target_by_k": target_by_k,
                "candidates": candidates,
            },
        )
        progress = ", ".join(f"k={k}:{len(accepted_by_k[k])}/{target_by_k[k]}" for k in sorted(target_by_k))
        print(f"[Etapa 0 / Pool Novo MOF] {progress} (total candidatos={len(candidates)})", flush=True)

    rejections = dict(rejections_counter)

    missing = {k: target_by_k[k] - len(accepted_by_k[k]) for k in target_by_k if len(accepted_by_k[k]) < target_by_k[k]}
    if missing:
        raise RuntimeError(
            f"Pool do Novo MOF incompleto após {len(candidates)} candidatos. Déficit: {missing}"
        )

    approved_pool = [item for k in sorted(accepted_by_k) for item in accepted_by_k[k]]
    stats = {
        "trigger_meta": trigger_meta,
        "target_by_k": target_by_k,
        "approved_counts_by_k": {str(k): len(accepted_by_k[k]) for k in sorted(accepted_by_k)},
        "approved_counts_by_domain": dict(Counter(item["domain"] for item in approved_pool)),
        "total_candidates_generated": len(candidates),
        "rejections": rejections,
    }
    return approved_pool, stats


def score_candidates_with_piguard(
    candidates: List[Dict[str, Any]],
    checkpoint_path: Path | str,
    *,
    device: Optional[str] = None,
    batch_size: int = 32,
    scorer_fn: Optional[Callable[[List[str]], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Avalia o score contextual P_PI(inj | x) de cada candidato usando o checkpoint base."""
    prompts = [str(item["instruction"]) for item in candidates]
    if scorer_fn is not None:
        scores = list(scorer_fn(prompts))
    else:
        import torch
        from transformers import AutoTokenizer
        from Corrections.renew_token_recheck import load_checkpoint

        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        _, model = load_checkpoint(str(checkpoint_path), resolved_device)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        classifier_dtype = next(model.classifier.parameters()).dtype

        scores = []
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(resolved_device)
                hidden = model.deberta(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_hidden_states=False,
                )["last_hidden_state"]
                pooled = hidden[:, 0, :].to(dtype=classifier_dtype)
                logits = model.classifier(pooled)
                probs = logits.softmax(dim=-1)[:, 1].cpu().tolist()
                scores.extend(float(p) for p in probs)

    if len(scores) != len(candidates):
        raise ValueError(f"Quantidade de scores ({len(scores)}) difere dos candidatos ({len(candidates)})")

    scored: List[Dict[str, Any]] = []
    for item, score in zip(candidates, scores):
        copy_item = dict(item)
        copy_item["p_inj_score"] = float(score)
        scored.append(copy_item)
    return scored


def _allocate_domain_quotas(items: Sequence[Dict[str, Any]], total_take: int) -> Dict[str, int]:
    by_dom = defaultdict(list)
    for it in items:
        by_dom[it["domain"]].append(it)
    quotas = {d: 0 for d in DOMAIN_NAMES}
    remaining = total_take
    while remaining > 0:
        progressed = False
        for d in DOMAIN_NAMES:
            if remaining == 0:
                break
            if quotas[d] < len(by_dom[d]):
                quotas[d] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def partition_pool_into_train_and_heldout(
    scored_candidates: Sequence[Dict[str, Any]],
    *,
    train_quotas: Optional[Dict[int, int]] = None,
    heldout_quotas: Optional[Dict[int, int]] = None,
    top_fraction: float = 0.80,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separa de forma 100% disjunta:
    1) Os excedentes benignos para `calibration_heldout.json` (estratificados por k, domínio e quantil de score);
    2) Os 762 exemplos de `MOF_v2_train.json` via Seleção Contextual Hard-Negative (80% maiores scores + 20% aleatórios).
    """
    t_quotas = dict(train_quotas or TRAIN_QUOTAS)
    h_quotas = dict(heldout_quotas or HELDOUT_BENIGN_QUOTAS)
    rng = random.Random(seed)

    by_k: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in scored_candidates:
        by_k[int(item["k"])].append(dict(item))

    heldout_benign: List[Dict[str, Any]] = []
    train_selected: List[Dict[str, Any]] = []
    unused_overflow: List[Dict[str, Any]] = []

    for k in sorted(t_quotas):
        stratum = by_k.get(k, [])
        need_train = int(t_quotas[k])
        need_heldout = int(h_quotas.get(k, 0))
        if len(stratum) < need_train + need_heldout:
            raise ValueError(
                f"Estrato k={k} possui {len(stratum)} amostras; mínimo necessário={need_train + need_heldout}"
            )

        # 1. Seleção estratificada por domínio e quantil de P_PI(inj | x) para o held-out benigno.
        dom_buckets: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAIN_NAMES}
        for it in sorted(stratum, key=lambda x: (x["domain"], float(x["p_inj_score"]), int(x["id"]))):
            dom_buckets.setdefault(it["domain"], []).append(it)

        dom_take = _allocate_domain_quotas(stratum, need_heldout)
        heldout_ids: Set[int] = set()
        for dom, take_n in dom_take.items():
            if take_n <= 0:
                continue
            bucket = dom_buckets[dom]
            # Amostragem estratificada por quantis ao longo do score ordenado no domínio.
            picked_indices = sorted(
                {
                    min(len(bucket) - 1, int((i + 0.5) * len(bucket) / take_n))
                    for i in range(take_n)
                }
            )
            # Caso haja colisão de arredondamento (não ocorre quando take_n <= len(bucket)), completa determinístico.
            if len(picked_indices) < take_n:
                used_set = set(picked_indices)
                for idx in range(len(bucket)):
                    if idx not in used_set:
                        picked_indices.append(idx)
                        used_set.add(idx)
                        if len(picked_indices) == take_n:
                            break
            for idx in picked_indices:
                chosen = dict(bucket[idx])
                chosen["partition"] = "heldout_benign"
                heldout_benign.append(chosen)
                heldout_ids.add(int(chosen["id"]))

        # 2. Entre os restantes do estrato k, aplica Seleção Contextual Hard-Negative 80/20.
        remaining = [dict(it) for it in stratum if int(it["id"]) not in heldout_ids]
        remaining_sorted = sorted(
            remaining,
            key=lambda x: (-float(x["p_inj_score"]), int(x["id"])),
        )
        n_top = int(round(need_train * top_fraction))
        n_rand = need_train - n_top

        top_slice = remaining_sorted[:n_top]
        tail_pool = remaining_sorted[n_top:]
        rand_slice = rng.sample(tail_pool, n_rand) if n_rand > 0 else []
        rand_ids = {int(it["id"]) for it in rand_slice}

        for it in top_slice:
            it["partition"] = "train_mof_v2"
            it["selection_mode"] = "hard_negative_top80"
            train_selected.append(it)
        for it in rand_slice:
            it["partition"] = "train_mof_v2"
            it["selection_mode"] = "random_20"
            train_selected.append(it)
        for it in tail_pool:
            if int(it["id"]) not in rand_ids:
                it["partition"] = "unused_overflow"
                unused_overflow.append(it)

    return train_selected, heldout_benign, unused_overflow


def generate_heldout_malicious_samples(
    curated_path: Path | str,
    cache_path: Path | str,
    forbidden_exact_set: Set[str],
    *,
    forbidden_norm_set: Optional[Set[str]] = None,
    per_format: int = HELDOUT_MALICIOUS_PER_FORMAT,
    seed: int = 1042,
    max_attempts: int = 120,
    api_key: Optional[str] = None,
    call_fn: Optional[Callable[[dict, str], dict]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gera amostras maliciosas held-out (6 por formato = 102) via funil de 4 filtros.

    Não toca nos 435 exemplos de `LLM_augmented.json` nem em `train_cleaned_original.json`.
    """
    curated_data = read_json(curated_path)
    demos_by_format: Dict[str, List[str]] = {}
    for row in curated_data:
        demos_by_format.setdefault(str(row["format"]), []).append(str(row["prompt"]))

    cache_file = Path(cache_path)
    existing_by_key: Dict[str, List[str]] = {k: [] for k in BASE_DISTRIBUTION}
    if cache_file.is_file():
        cached = read_json(cache_file)
        if isinstance(cached, list):
            for row in cached:
                topic = row.get("topic")
                if topic in existing_by_key and len(existing_by_key[topic]) < per_format:
                    existing_by_key[topic].append(str(row["prompt"]))

    need_generation = any(len(existing_by_key[k]) < per_format for k in BASE_DISTRIBUTION)
    if need_generation and call_fn is None and not api_key:
        api_key = load_api_key(repo_root=REPO_ROOT)
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY necessária para gerar amostras maliciosas held-out.")
    api_key = api_key or ""

    custom_call = call_fn or (lambda payload, k: call_openrouter(payload, k, max_retries=3, timeout=45))
    funnel = ValidationFunnel(api_key=api_key, call_fn=custom_call)
    norm_forbidden = forbidden_norm_set or set()

    all_records: List[Dict[str, Any]] = []
    stats_by_topic: Dict[str, Any] = {}

    for idx, topic in enumerate(BASE_DISTRIBUTION):
        fmt = topic.replace(" Injection", "")
        demos = demos_by_format.get(fmt, [])
        if not demos:
            raise ValueError(f"Sem demonstrações para o formato '{fmt}'")

        valid_samples: List[str] = list(existing_by_key[topic][:per_format])
        seen_exact: Set[str] = set(forbidden_exact_set) | {s.strip().lower() for s in valid_samples}
        rng = random.Random(seed + idx * 31)
        attempts = 0

        while len(valid_samples) < per_format and attempts < max_attempts:
            attempts += 1
            missing = per_format - len(valid_samples)
            ask_num = min(max(missing + 3, 6), 10)
            sampled_demos = rng.sample(demos, 3) if len(demos) > 3 else list(demos)
            user_prompt = AUG_USER_PROMPT_TEMPLATE.format(
                demonstrations=json.dumps(sampled_demos, ensure_ascii=False),
                number=ask_num,
                key=topic,
            )
            try:
                if call_fn is not None:
                    payload = {
                        "model": GENERATION_MODEL,
                        "messages": [
                            {"role": "system", "content": AUG_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 1.0,
                        "max_tokens": 2048,
                    }
                    raw = call_fn(payload, api_key)["choices"][0]["message"].get("content") or ""
                else:
                    raw = call_generation_model(
                        model=GENERATION_MODEL,
                        system_prompt=AUG_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        api_key=api_key,
                        temperature=1.0,
                    )
            except FatalAPIError:
                raise
            except Exception as exc:
                print(f"  [HeldOut Malicious WARN] {fmt} tentativa {attempts}: {exc}", flush=True)
                continue

            for cand in parse_samples(raw, fmt):
                if len(valid_samples) >= per_format:
                    break
                if normalize_text(cand) in norm_forbidden:
                    continue
                ok, _, _ = funnel.validate_sample(
                    sample=cand,
                    fmt=fmt,
                    valid_samples=valid_samples,
                    seen_exact_set=seen_exact,
                )
                if ok:
                    valid_samples.append(cand)
                    seen_exact.add(cand.strip().lower())
                    norm_forbidden.add(normalize_text(cand))

        if len(valid_samples) < per_format:
            raise RuntimeError(
                f"Held-out malicioso incompleto em {topic}: {len(valid_samples)}/{per_format}"
            )

        stats_by_topic[topic] = {"format": fmt, "target": per_format, "validated": len(valid_samples), "attempts": attempts}
        for s in valid_samples[:per_format]:
            all_records.append(
                {
                    "prompt": s,
                    "label": 1,
                    "source": "heldout_malicious_4filters",
                    "format": fmt,
                    "topic": topic,
                }
            )
        write_json(cache_file, all_records)

    return all_records, stats_by_topic


def verify_disjointness(
    sets_by_name: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    """Verifica interseção zero (texto exato e normalizado) entre todos os pares de conjuntos."""
    norm_sets = {name: {normalize_text(t) for t in texts} for name, texts in sets_by_name.items()}
    names = list(norm_sets.keys())
    overlaps: Dict[str, int] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = norm_sets[a].intersection(norm_sets[b])
            overlaps[f"{a}__vs__{b}"] = len(inter)
    has_leakage = any(v > 0 for v in overlaps.values())
    return {"disjoint": not has_leakage, "pair_overlaps": overlaps}


def run_step0(
    *,
    trigger_path: Path | str,
    checkpoint_path: Path | str,
    output_dir: Path | str,
    repo_root: Path | str = REPO_ROOT,
    corrections_dir: Optional[Path | str] = None,
    pool_quotas: Optional[Dict[int, int]] = None,
    pool_buffer: Optional[Dict[int, int]] = None,
    train_quotas: Optional[Dict[int, int]] = None,
    heldout_benign_quotas: Optional[Dict[int, int]] = None,
    heldout_malicious_per_format: int = HELDOUT_MALICIOUS_PER_FORMAT,
    workers: int = 5,
    seed: int = 42,
    force: bool = False,
    api_key: Optional[str] = None,
    tokenizer: Any = None,
    call_fn: Optional[Callable[[dict, str], dict]] = None,
    scorer_fn: Optional[Callable[[List[str]], List[float]]] = None,
) -> Dict[str, Any]:
    """Executa toda a Etapa 0 e persiste MOF_v2_train.json + calibration_heldout.json + relatório."""
    repo = Path(repo_root).expanduser().resolve()
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    trigger_path = Path(trigger_path).expanduser()
    if not trigger_path.is_absolute():
        trigger_path = (repo / trigger_path).resolve()
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo / checkpoint_path).resolve()
    if corrections_dir is not None:
        corrections_dir = Path(corrections_dir).expanduser()
        if not corrections_dir.is_absolute():
            corrections_dir = (repo / corrections_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    mof_train_path = out / "MOF_v2_train.json"
    mof_detailed_path = out / "MOF_v2_train.detailed.json"
    heldout_path = out / "calibration_heldout.json"
    report_path = out / "step0_report.json"

    if not force and mof_train_path.is_file() and heldout_path.is_file() and report_path.is_file():
        report = read_json(report_path)
        artifacts = {"mof_v2_train": mof_train_path, "calibration_heldout": heldout_path}
        if all(
            not report.get(f"{name}_sha256") or report[f"{name}_sha256"] == sha256_file(path)
            for name, path in artifacts.items()
        ):
            for name, path in artifacts.items():
                report[f"{name}_path"] = str(path)
            print(f"[Etapa 0] Reutilizando artefatos existentes em {out}", flush=True)
            return report

    extra_forbidden: List[Path] = []
    if corrections_dir is not None:
        cdir = Path(corrections_dir)
        for rel in (
            Path("augmented/LLM_augmented.json"),
            Path("data/train_augmented_corrected.json"),
        ):
            cand = cdir / rel
            if cand.is_file():
                extra_forbidden.append(cand)

    forbidden_exact, forbidden_norm, forbidden_counts = load_forbidden_texts(
        repo_root=repo,
        extra_paths=extra_forbidden,
    )

    # 1. Pool do Novo MOF (benigno com 1, 2 e 3 gatilhos distintos nos 4 domínios)
    pool_cache = out / "mof_v2_pool.candidates.json"
    approved_pool, pool_stats = generate_new_mof_pool(
        trigger_path=trigger_path,
        cache_path=pool_cache,
        forbidden_exact_set=forbidden_exact,
        forbidden_norm_set=forbidden_norm,
        tokenizer=tokenizer,
        pool_quotas=pool_quotas,
        pool_buffer=pool_buffer,
        workers=workers,
        seed=seed,
        api_key=api_key,
        call_fn=call_fn,
    )

    # 2. Score contextual no checkpoint `augmented_corrected/best_model.pth`
    scored_pool = score_candidates_with_piguard(
        approved_pool,
        checkpoint_path=checkpoint_path,
        scorer_fn=scorer_fn,
    )

    # 3. Particionamento disjunto em MOF_v2_train (762) e held-out benigno (238)
    t_quotas = dict(train_quotas or TRAIN_QUOTAS)
    h_quotas = dict(heldout_benign_quotas or HELDOUT_BENIGN_QUOTAS)
    train_selected, heldout_benign, unused_overflow = partition_pool_into_train_and_heldout(
        scored_pool,
        train_quotas=t_quotas,
        heldout_quotas=h_quotas,
        top_fraction=0.80,
        seed=seed,
    )

    # Garante que textos benignos do pool não colidam com o lote malicioso held-out
    pool_exact = {str(x["instruction"]).strip().lower() for x in approved_pool}
    pool_norm = {normalize_text(x["instruction"]) for x in approved_pool}

    # 4. Lote malicioso held-out (102 amostras pelos 4 filtros, 100% fora do LLM_augmented.json)
    curated_path = repo / "datasets" / "Augmented" / "Cache" / "curated_prompt_injections.json"
    malicious_cache = out / "heldout_malicious.cache.json"
    heldout_malicious, mal_stats = generate_heldout_malicious_samples(
        curated_path=curated_path,
        cache_path=malicious_cache,
        forbidden_exact_set=forbidden_exact | pool_exact,
        forbidden_norm_set=forbidden_norm | pool_norm,
        per_format=heldout_malicious_per_format,
        seed=seed + 1000,
        api_key=api_key,
        call_fn=call_fn,
    )

    # 5. Serialização no formato padrão do train.py e formato completo para auditoria
    mof_records = [
        {"prompt": item["instruction"], "source": "over-defense", "label": 0}
        for item in train_selected
    ]
    heldout_records: List[Dict[str, Any]] = []
    for item in heldout_benign:
        heldout_records.append(
            {
                "prompt": item["instruction"],
                "label": 0,
                "source": "heldout_new_mof_overflow",
                "k": int(item["k"]),
                "domain": item["domain"],
                "chosen_words": item["chosen_words"],
                "p_inj_score": float(item["p_inj_score"]),
            }
        )
    heldout_records.extend(heldout_malicious)

    write_json(mof_train_path, mof_records)
    write_json(mof_detailed_path, train_selected)
    write_json(heldout_path, heldout_records)

    # 6. Auditoria de disjunção total
    disjoint_check = verify_disjointness(
        {
            "MOF_v2_train": [r["prompt"] for r in mof_records],
            "calibration_heldout_benign": [r["prompt"] for r in heldout_benign_items(heldout_records)],
            "calibration_heldout_malicious": [r["prompt"] for r in heldout_malicious],
            "forbidden_train_and_benchmarks": list(forbidden_norm),
        }
    )
    if not disjoint_check["disjoint"]:
        raise RuntimeError(f"Vazamento detectado na Etapa 0: {disjoint_check}")

    report = {
        "status": "complete",
        "seed": seed,
        "mof_v2_train_path": str(mof_train_path),
        "mof_v2_train_sha256": sha256_file(mof_train_path),
        "calibration_heldout_path": str(heldout_path),
        "calibration_heldout_sha256": sha256_file(heldout_path),
        "mof_v2_count": len(mof_records),
        "mof_v2_k_distribution": dict(Counter(int(x["k"]) for x in train_selected)),
        "mof_v2_domain_distribution": dict(Counter(x["domain"] for x in train_selected)),
        "mof_v2_selection_modes": dict(Counter(x["selection_mode"] for x in train_selected)),
        "heldout_total_count": len(heldout_records),
        "heldout_benign_count": len(heldout_benign),
        "heldout_benign_k_distribution": dict(Counter(int(x["k"]) for x in heldout_benign)),
        "heldout_benign_domain_distribution": dict(Counter(x["domain"] for x in heldout_benign)),
        "heldout_malicious_count": len(heldout_malicious),
        "unused_overflow_count": len(unused_overflow),
        "forbidden_sources_counts": forbidden_counts,
        "pool_stats": pool_stats,
        "heldout_malicious_stats": mal_stats,
        "disjoint_check": disjoint_check,
    }
    write_json(report_path, report)
    return report


def heldout_benign_items(heldout_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in heldout_records if int(row["label"]) == 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trigger-report",
        default=None,
        help=(
            "Caminho para trigger_words.report.json "
            "(padrão: <Corrections>/augmented_corrected/trigger_words.report.json)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Caminho para augmented_corrected/best_model.pth "
            "(padrão: <Corrections>/augmented_corrected/best_model.pth)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de saída da Etapa 0 (padrão: <Upgrade>/outputs/step0).",
    )
    parser.add_argument(
        "--corrections-dir",
        default=None,
        help="Diretório de artefatos do Corrections (padrão: Corrections/outputs).",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    api_key = load_api_key(args.api_key, repo_root=repo_root)
    corrections_dir = resolve_path(
        args.corrections_dir,
        default_corrections_dir(repo_root),
        repo_root=repo_root,
    )
    upgrade_output = repo_root / "Upgrade" / "outputs"
    output_dir = resolve_path(args.output_dir, upgrade_output / "step0", repo_root=repo_root)
    trigger_report = resolve_path(
        args.trigger_report,
        corrections_dir / "augmented_corrected" / "trigger_words.report.json",
        repo_root=repo_root,
    )
    if not trigger_report.is_file():
        fallback = trigger_report.with_name("trigger_words.json")
        if fallback.is_file():
            trigger_report = fallback
    checkpoint_path = resolve_path(
        args.checkpoint,
        corrections_dir / "augmented_corrected" / "best_model.pth",
        repo_root=repo_root,
    )

    report = run_step0(
        trigger_path=trigger_report,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        repo_root=repo_root,
        corrections_dir=corrections_dir,
        workers=args.workers,
        seed=args.seed,
        force=args.force,
        api_key=api_key,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "pool_stats"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

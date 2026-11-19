"""Pacote Upgrade do PIGuard.

Contém o pipeline completo em scripts Python executáveis:
- Etapa 0: geração do pool do Novo MOF (v2) e separação disjunta dos excedentes held-out;
- Laya English: cascata seletiva com agregação em dois níveis, limiar T_L, calibração e Paired Bootstrap;
- Tópico 1: avaliação dos 5 modelos `rp` de `outputs` com Laya English;
- Tópico 2: retreinamento scratch sobre `train_augmented_corrected.json` intacto + `MOF_v2_train.json`;
- Tópico 3: consolidação geral de métricas, intervalos de confiança, latência e gráficos.

Entradas e saídas são resolvidas a partir da raiz do repositório; consulte
``Upgrade/upgrade_pipeline.py`` para a execução completa.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__version__ = "2.0.0"

PUBLIC_SETS: Tuple[str, ...] = (
    "NotInject_one",
    "NotInject_two",
    "NotInject_three",
    "WildGuard",
    "BIPIA_text",
    "BIPIA_code",
)

ALL_BENCHMARK_SETS: Tuple[str, ...] = PUBLIC_SETS + (
    "tensor_trust_hijacking",
    "tensor_trust_extraction",
)

BENCHMARK_LABELS: Dict[str, int] = {
    "NotInject_one": 0,
    "NotInject_two": 0,
    "NotInject_three": 0,
    "WildGuard": 0,
    "BIPIA_text": 1,
    "BIPIA_code": 1,
    "tensor_trust_hijacking": 1,
    "tensor_trust_extraction": 1,
}

BENCHMARK_FILENAMES: Dict[str, str] = {
    "WildGuard": "wildguard.json",
    "tensor_trust_hijacking": "Extra_dataset/tensor_trust_hijacking.json",
    "tensor_trust_extraction": "Extra_dataset/tensor_trust_extraction.json",
}


def resolve_path(
    value: Path | str | None,
    default: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    """Resolve um caminho relativo contra a raiz do repositório.

    O padrão é ancorado em ``<repo>`` (dois níveis acima deste pacote), não no
    diretório de trabalho. Caminhos absolutos são preservados.
    """
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    if value is None or str(value).strip() == "":
        return Path(default).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def default_corrections_dir(repo_root: Path | str | None = None) -> Path:
    """Retorna a pasta de artefatos do pipeline Corrections.

    Prioriza ``Corrections/outputs`` (padrão de ``Corrections/corrections.py``)
    e usa ``Corrections/`` como fallback para layouts antigos.
    """
    repo = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    candidates = (repo / "Corrections" / "outputs", repo / "Corrections")
    for candidate in candidates:
        if (candidate / "data").is_dir() or (candidate / "augmented_corrected").is_dir():
            return candidate
    return candidates[0]


def load_api_key(
    explicit_key: str | None = None,
    *,
    repo_root: Path | str | None = None,
    prompt: bool = False,
) -> str | None:
    """Carrega a chave do OpenRouter do argumento, ambiente ou ``<repo>/.env``."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip().strip("\"'")
    for env_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip().strip("\"'")

    repo = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    env_path = repo / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip().lower() in ("openrouter_api_key", "openai_api_key"):
                value = value.strip().strip("\"'")
                if value:
                    return value

    if prompt:
        import getpass

        value = getpass.getpass("Cole sua OPENROUTER_API_KEY: ").strip().strip("\"'")
        return value or None
    return None


def read_json(path: Path | str) -> Any:
    """Lê e desserializa um arquivo JSON UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, data: Any, *, indent: int = 2) -> Path:
    """Escreve JSON de forma atômica via arquivo temporário."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def write_csv(path: Path | str, rows: List[Dict[str, Any]]) -> Path:
    """Escreve uma lista de dicionários em CSV mantendo a ordem das colunas."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) or ["status"]
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)
    return target


def sha256_file(path: Path | str) -> str:
    """Calcula o digest SHA-256 em streaming para arquivos de qualquer tamanho."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_text(text: str) -> str:
    """Normaliza texto para verificação exata de duplicatas e disjunção."""
    return " ".join(str(text).strip().lower().split())


def benchmark_file_path(dataset_root: Path | str, name: str) -> Path:
    """Resolve o caminho do arquivo JSON de um benchmark a partir de ``dataset_root``."""
    relpath = BENCHMARK_FILENAMES.get(name, f"{name}.json")
    return Path(dataset_root) / relpath


def load_benchmarks(
    dataset_root: Path | str,
    *,
    include_tensortrust: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Tuple[List[str], int]]:
    """Carrega os benchmarks públicos e opcionalmente TensorTrust com rótulos binários."""
    root = Path(dataset_root)
    names = ALL_BENCHMARK_SETS if include_tensortrust else PUBLIC_SETS
    benchmarks: Dict[str, Tuple[List[str], int]] = {}
    for name in names:
        path = benchmark_file_path(root, name)
        data = read_json(path)
        if isinstance(data, list):
            texts = [str(row["prompt"]) for row in data]
        elif isinstance(data, dict):
            texts = [str(item) for group in data.values() for item in group]
        else:
            raise ValueError(f"Formato inesperado em benchmark: {path}")
        if max_samples is not None:
            texts = texts[:max_samples]
        if not texts:
            raise ValueError(f"Benchmark vazio: {path}")
        benchmarks[name] = (texts, BENCHMARK_LABELS[name])
    return benchmarks


def aggregate_benchmark_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Calcula métricas consolidadas (Over-defense, Benign, Malicious, Average e TensorTrust)."""
    overdefense = (
        scores["NotInject_one"] + scores["NotInject_two"] + scores["NotInject_three"]
    ) / 3.0
    benign = scores["WildGuard"]
    malicious = (scores["BIPIA_text"] + scores["BIPIA_code"]) / 2.0
    average = (overdefense + benign + malicious) / 3.0
    metrics: Dict[str, float] = {
        "overdefense": overdefense,
        "benign": benign,
        "malicious": malicious,
        "average": average,
    }
    if "tensor_trust_hijacking" in scores and "tensor_trust_extraction" in scores:
        metrics["tensor_trust_hijacking"] = scores["tensor_trust_hijacking"]
        metrics["tensor_trust_extraction"] = scores["tensor_trust_extraction"]
        metrics["tensor_trust_average"] = (
            scores["tensor_trust_hijacking"] + scores["tensor_trust_extraction"]
        ) / 2.0
    return metrics


__all__ = [
    "__version__",
    "PUBLIC_SETS",
    "ALL_BENCHMARK_SETS",
    "BENCHMARK_LABELS",
    "BENCHMARK_FILENAMES",
    "resolve_path",
    "default_corrections_dir",
    "load_api_key",
    "read_json",
    "write_json",
    "write_csv",
    "sha256_file",
    "normalize_text",
    "benchmark_file_path",
    "load_benchmarks",
    "aggregate_benchmark_scores",
]

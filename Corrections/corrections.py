"""
O fluxo executa, em sequência:

1. Geração do aumento corrigido (435 exemplos, 17 categorias) com os quatro
   filtros de ``Corrections/Upgrade_augmented/``.
2. Treinamento da base limpa, do aumento publicado e do aumento corrigido.
3. Token Recheck sanitizado dos três checkpoints.
4. Geração MOF e retreinamento da variante corrigida com MOF.
5. Comparação final contra ``leolee99/PIGuard``.

Uso:
    python Corrections/corrections.py
    python Corrections/corrections.py --skip-generation --force-retrain
    python Corrections/corrections.py --output-dir /caminho/para/saida
    python Corrections/corrections.py --ntfy   # notificações opcionais

Observações:
- A geração e o MOF exigem ``OPENROUTER_API_KEY`` (ou ``OPENAI_API_KEY``) no
  ambiente ou em ``<raiz>/.env``.
- A comparação final baixa ``microsoft/deberta-v3-base`` e, opcionalmente,
  ``leolee99/PIGuard`` do Hugging Face.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence


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

# Permite imports como ``Corrections.*`` e módulos de ``Original/`` em qualquer cwd.
for _local_path in (ROOT_DIR, ROOT_DIR / "Original", ROOT_DIR / "Corrections"):
    _entry = str(_local_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)



# Notificações opcionais (ntfy.sh)
NTFY_URL = "https://ntfy.sh/"
NTFY_ENABLED = False
NTFY_TOPIC = os.environ.get("PIGUARD_NTFY_TOPIC", "piguard_gabriel_treinos")


def notify_ntfy(
    title: str,
    message: str,
    *,
    priority: int = 3,
    tags: Sequence[str] | None = None,
    timeout: int = 15,
) -> None:
    """Envia uma notificação opcional; falhas nunca interrompem o pipeline."""
    if not NTFY_ENABLED:
        return
    payload = {
        "topic": NTFY_TOPIC,
        "title": str(title)[:250],
        "message": str(message)[:3800],
        "priority": int(priority),
    }
    if tags:
        payload["tags"] = list(tags)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        NTFY_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except Exception as exc:  # noqa: BLE001 - notificação é best-effort
        print(f"[ntfy] Aviso ao enviar notificação ({title}): {exc}", flush=True)


@contextmanager
def timed_stage(name: str):
    """Mede a etapa, imprime início/fim e notifica sucesso ou falha."""
    started = time.monotonic()
    notes: list[str] = []
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}", flush=True)
    try:
        yield notes
    except BaseException as exc:
        elapsed = time.monotonic() - started
        summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        message = f"Etapa '{name}' falhou após {elapsed:.1f}s.\nErro: {summary}"
        if notes:
            message += "\nContexto:\n" + "\n".join(str(note) for note in notes)
        notify_ntfy(
            title=f"[PIGuard Corrections] Erro: {name}",
            message=message,
            priority=4,
            tags=["x", "warning"],
        )
        raise
    else:
        elapsed = time.monotonic() - started
        message = f"Etapa '{name}' concluída em {elapsed:.1f}s."
        if notes:
            message += "\n" + "\n".join(str(note) for note in notes)
        print(f"[OK] {message}", flush=True)
        notify_ntfy(
            title=f"[PIGuard Corrections] Concluída: {name}",
            message=message,
            priority=3,
            tags=["white_check_mark"],
        )


# Utilitários de caminho, execução e JSON
def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve caminhos relativos contra ``ROOT_DIR`` (nunca contra o cwd)."""
    if value is None or str(value) == "":
        return Path(default).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def write_json(path: Path, data) -> None:
    """Grava JSON de forma atômica, criando os diretórios necessários."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def subprocess_env() -> dict[str, str]:
    """Ambiente que torna ``Corrections`` e ``Original`` importáveis."""
    env = os.environ.copy()
    entries = [str(ROOT_DIR), str(ROOT_DIR / "Original")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_streamed(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Executa um comando transmitindo a saída e guardando as últimas linhas."""
    tail: deque[str] = deque(maxlen=25)
    returncode = -1
    log_stream = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Subprocesso sem stdout capturado")
            for line in process.stdout:
                print(line, end="", flush=True)
                tail.append(line.rstrip("\n"))
                if log_stream is not None:
                    log_stream.write(line)
                    log_stream.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    finally:
        if log_stream is not None:
            log_stream.close()
    if returncode != 0:
        snippet = "\n".join(tail)
        raise RuntimeError(
            f"Comando falhou (código {returncode}): {' '.join(str(p) for p in command)}\n"
            f"{snippet}"
        )


def has_api_key() -> bool:
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return True
    return (ROOT_DIR / ".env").is_file()


def runtime_info() -> dict[str, object]:
    """Importa torch/transformers sob demanda e descreve o ambiente."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(
            f"[ERRO] Dependências ausentes ({exc}). "
            "Instale com: python -m pip install -r requirements.txt"
        ) from exc

    cuda = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
        "python": platform.python_version(),
    }


# Etapa 2: geração corrigida + quatro filtros
def stage_generate(cfg: argparse.Namespace, paths: dict[str, Path]) -> Path:
    """Gera as 17 categorias (435 exemplos) e aplica os quatro filtros.

    O prompt pede no máximo 50 palavras e o filtro 1 aceita até 80. O arquivo
    ``Upgrade_augmented/LLM_augmented_new.json`` é um artefato histórico e não
    é usado como entrada.
    """
    from Corrections.Upgrade_augmented.new_generate_data import BASE_DISTRIBUTION

    augmented_path = paths["augmented"]
    out_dir = augmented_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    curated_path = ROOT_DIR / "datasets/Augmented/Cache/curated_prompt_injections.json"
    if not curated_path.is_file():
        raise FileNotFoundError(f"Arquivo de demonstrações ausente: {curated_path}")

    run_streamed(
        [
            sys.executable,
            "-u",
            "-m",
            "Corrections.Upgrade_augmented.new_generate_data",
            "--curated-path",
            str(curated_path),
            "--output-dir",
            str(out_dir),
            "--seed",
            str(cfg.seed),
            "--workers",
            str(cfg.workers),
            "--max-attempts",
            str(cfg.max_attempts),
        ],
        cwd=ROOT_DIR,
        log_path=paths["logs"] / "generation.log",
        env=subprocess_env(),
    )

    if not augmented_path.is_file():
        raise FileNotFoundError(
            f"Aumento não encontrado em {augmented_path}. "
            "Use --skip-generation apenas se o arquivo já existir."
        )

    augmented = json.loads(augmented_path.read_text(encoding="utf-8"))
    expected = {
        key.replace(" Injection", ""): value
        for key, value in BASE_DISTRIBUTION.items()
    }
    if len(augmented) != sum(BASE_DISTRIBUTION.values()):
        raise ValueError(
            f"Aumento com {len(augmented)} amostras; "
            f"esperado {sum(BASE_DISTRIBUTION.values())}."
        )
    if any(row.get("label") != 1 for row in augmented):
        raise ValueError("Todas as amostras do aumento devem ter label=1.")
    if dict(Counter(row.get("format") for row in augmented)) != expected:
        raise ValueError("Distribuição por categoria diferente da esperada.")
    if any(row.get("source") != "LLM Augmented set" for row in augmented):
        raise ValueError("Campo 'source' inesperado no aumento.")

    clean_path = ROOT_DIR / "datasets/Augmented/train_cleaned_original.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    normalized = [
        {"prompt": row["prompt"], "label": row["label"], "source": row["source"]}
        for row in augmented
    ]
    corrected_path = paths["data"] / "train_augmented_corrected.json"
    write_json(corrected_path, clean + normalized)
    print(
        f"Base limpa: {len(clean)} | Aumento corrigido: {len(normalized)}",
        flush=True,
    )

    report_path = out_dir / "generated_malicious_raw_responses_gpt4o_mini.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report.get("summary", []):
            print(row["format"], row["stats"], flush=True)

    return corrected_path


# Etapa 3: treinamento
def run_train(
    name: str,
    dataset: Path,
    *,
    cfg: argparse.Namespace,
    paths: dict[str, Path],
    reference_name: str | None = None,
) -> Path:
    target = paths["output"] / name
    target.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset)
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset de treino ausente: {dataset}")

    train_script = ROOT_DIR / "Original/train.py"
    spec = {
        "seed": cfg.seed,
        "max_length": cfg.max_length,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "lr": cfg.lr,
        "warmup": cfg.warmup,
        "save_step": cfg.save_step,
        "dataset_sha256": digest(dataset),
        "valid_sha256": digest(ROOT_DIR / "datasets/valid.json"),
        "train_sha256": digest(train_script),
        "model_sha256": digest(ROOT_DIR / "Original/PIGuard.py"),
        "params_sha256": digest(ROOT_DIR / "Original/params.py"),
        "torch": cfg.runtime["torch"],
        "transformers": cfg.runtime["transformers"],
        "python": platform.python_version(),
    }

    manifest = target / "manifest.json"
    best_model = target / "best_model.pth"
    if not cfg.force_retrain and manifest.is_file() and best_model.is_file():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")) == spec:
                print(f"Reutilizando: {name}", flush=True)
                return best_model
        except Exception:  # noqa: BLE001 - manifesto inválido força retreino
            pass

    if (
        reference_name
        and cfg.reference_output_root is not None
        and not cfg.force_retrain
    ):
        reference = Path(cfg.reference_output_root) / reference_name
        if (reference / "best_model.pth").is_file() and (
            reference / "manifest.json"
        ).is_file():
            try:
                old = json.loads(
                    (reference / "manifest.json").read_text(encoding="utf-8")
                )
            except Exception:  # noqa: BLE001
                old = {}
            keys = (
                "seed",
                "max_length",
                "epochs",
                "batch_size",
                "lr",
                "warmup",
                "save_step",
                "dataset_sha256",
                "valid_sha256",
                "torch",
                "transformers",
            )
            if all(old.get(key) == spec[key] for key in keys):
                shutil.copy2(reference / "best_model.pth", best_model)
                write_json(target / "reference_manifest.json", old)
                write_json(manifest, spec)
                print(f"Referência reutilizada: {reference}", flush=True)
                return best_model

    work = paths["output"] / "work" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    local_log = work / "stdout.log"

    command = [
        sys.executable,
        "-u",
        str(train_script),
        "--name",
        name,
        "--train_set",
        str(dataset),
        "--valid_set",
        str(ROOT_DIR / "datasets/valid.json"),
        "--dataset_root",
        str(ROOT_DIR / "datasets"),
        "--checkpoint_path",
        str(work),
        "--logs",
        str(target),
        "--seed",
        str(cfg.seed),
        "--max_length",
        str(cfg.max_length),
        "--epochs",
        str(cfg.epochs),
        "--batch_size",
        str(cfg.batch_size),
        "--eval_batch_size",
        str(cfg.eval_batch_size),
        "--lr",
        str(cfg.lr),
        "--warmup",
        str(cfg.warmup),
        "--save_step",
        str(cfg.save_step),
    ]
    print(f"\nINICIANDO TREINAMENTO: {name}\n", flush=True)
    try:
        run_streamed(
            command,
            cwd=ROOT_DIR,
            log_path=local_log,
            env=subprocess_env(),
        )
    finally:
        if local_log.is_file():
            shutil.copy2(local_log, target / "stdout.log")

    best = work / "best_model.pth"
    if not best.is_file():
        raise RuntimeError(f"Checkpoint ausente após treino: {best}")
    shutil.copy2(best, best_model)
    write_json(manifest, spec)
    shutil.rmtree(work, ignore_errors=True)
    return best_model


def stage_train(
    cfg: argparse.Namespace, paths: dict[str, Path], corrected_path: Path
) -> dict[str, Path]:
    """Treina base limpa, aumento publicado e aumento corrigido.

    Usa seed 42, ``datasets/valid.json`` e os mesmos hiperparâmetros do
    treinamento existente. Checkpoints de referência compatíveis podem ser
    reaproveitados via ``--reference-output-root``.
    """
    models: dict[str, Path] = {}
    for name, dataset_path, reference_name in (
        (
            "clean_original",
            ROOT_DIR / "datasets/Augmented/train_cleaned_original.json",
            "train_cleaned_original",
        ),
        (
            "augmented_original",
            ROOT_DIR / "datasets/Augmented/train_augmented_original.json",
            "train_augmented_original",
        ),
        ("augmented_corrected", corrected_path, None),
    ):
        models[name] = run_train(
            name,
            dataset_path,
            cfg=cfg,
            paths=paths,
            reference_name=reference_name,
        )
    return models


# Etapa 4: Token Recheck sanitizado
def stage_recheck(
    cfg: argparse.Namespace, paths: dict[str, Path], models: dict[str, Path]
) -> dict[str, Path]:
    """Executa o Token Recheck sanitizado de cada checkpoint.

    Cada candidato deve iniciar com ``▁``, ter ao menos três letras ASCII e
    retornar ao mesmo ID após tokenização. O teste é ``[CLS, token_id, SEP]``
    com argmax para a classe 1. Lista vazia com status ``no_triggers`` é um
    resultado válido.
    """
    trigger_files: dict[str, Path] = {}
    for name in ("clean_original", "augmented_original", "augmented_corrected"):
        output = paths["output"] / name / "trigger_words.json"
        run_streamed(
            [
                sys.executable,
                "-u",
                "-m",
                "Corrections.renew_token_recheck",
                "--checkpoint",
                str(models[name]),
                "--output",
                str(output),
                "--batch_size",
                "256",
                "--no_dict_filter",
            ],
            cwd=ROOT_DIR,
            log_path=paths["logs"] / f"recheck_{name}.log",
            env=subprocess_env(),
        )
        trigger_files[name] = output
        result = json.loads(
            output.with_suffix(".report.json").read_text(encoding="utf-8")
        )
        summary = (
            f"{name}: {result['status']} | gatilhos={result['trigger_count']} | "
            f"{result['counts']}"
        )
        print(summary, flush=True)
    return trigger_files


# Etapa 5: MOF e retreinamento
def stage_mof(
    cfg: argparse.Namespace,
    paths: dict[str, Path],
    models: dict[str, Path],
    trigger_files: dict[str, Path],
    corrected_path: Path,
) -> None:
    """Gera MOF (1.000 candidatos, 762 aprovados) e retreina a variante corrigida.

    Cada amostra usa de uma a três palavras-gatilho, precisa conter as palavras
    escolhidas e receber ``NO`` do juiz. A checagem de similaridade existente é
    aplicada. Sem gatilhos, a variante é avaliada sem MOF.
    """
    models["augmented_mof_original"] = run_train(
        "augmented_mof_original",
        ROOT_DIR / "datasets/train.json",
        cfg=cfg,
        paths=paths,
        reference_name="train_augmented_mof_original",
    )

    mof_path = paths["output"] / "augmented_corrected" / "MOF.json"
    run_streamed(
        [
            sys.executable,
            "-u",
            "-m",
            "Corrections.generate_mof_data",
            "--trigger_words",
            str(trigger_files["augmented_corrected"]),
            "--output",
            str(mof_path),
            "--initial_target",
            "1000",
            "--final_target",
            "762",
            "--seed",
            str(cfg.seed),
            "--workers",
            str(cfg.workers),
        ],
        cwd=ROOT_DIR,
        log_path=paths["logs"] / "mof.log",
        env=subprocess_env(),
    )

    mof = json.loads(mof_path.read_text(encoding="utf-8"))
    report = json.loads(
        mof_path.with_suffix(".report.json").read_text(encoding="utf-8")
    )
    print(report, flush=True)
    if not mof:
        models.pop("augmented_mof_corrected", None)
        print(
            "Sem gatilhos (no_triggers): avaliado sem augmented_mof_corrected.",
            flush=True,
        )
        return

    if len(mof) != 762 or any(row.get("label") != 0 for row in mof):
        raise ValueError("MOF gerado fora do formato esperado (762 amostras label=0).")

    normalized = [
        {"prompt": row["prompt"], "label": row["label"], "source": row["source"]}
        for row in mof
    ]
    merged = json.loads(corrected_path.read_text(encoding="utf-8")) + normalized
    combined_path = paths["data"] / "train_augmented_mof_corrected.json"
    write_json(combined_path, merged)
    models["augmented_mof_corrected"] = run_train(
        "augmented_mof_corrected",
        combined_path,
        cfg=cfg,
        paths=paths,
    )


# Etapa 6: comparação final
def stage_compare(
    cfg: argparse.Namespace, paths: dict[str, Path], models: dict[str, Path]
):
    """Compara os modelos com argmax, ``eval()`` e 512 tokens.

    Sem ``PINT.json``, Benign = WildGuard e Malicious = média de BIPIA
    texto/código. O relatório inclui diferenças em pontos percentuais e as
    contagens de erros por conjunto.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from Corrections.renew_token_recheck import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmarks: dict[str, tuple[list[str], int]] = {}
    for name, filename, label in (
        ("NotInject_one", "NotInject_one.json", 0),
        ("NotInject_two", "NotInject_two.json", 0),
        ("NotInject_three", "NotInject_three.json", 0),
        ("WildGuard", "wildguard.json", 0),
        ("BIPIA_text", "BIPIA_text.json", 1),
        ("BIPIA_code", "BIPIA_code.json", 1),
    ):
        data = json.loads((ROOT_DIR / "datasets" / filename).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            texts = [text for group in data.values() for text in group]
        else:
            texts = [row["prompt"] for row in data]
        benchmarks[name] = (texts, label)

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    evaluation_models: dict[str, Path | str] = dict(models)
    if not cfg.skip_hf:
        evaluation_models["PIGuard_HF"] = "leolee99/PIGuard"

    for name, checkpoint in evaluation_models.items():
        if name == "PIGuard_HF":
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, trust_remote_code=True
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint, trust_remote_code=True
            )
        else:
            _, model = load_checkpoint(checkpoint, device)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        model.eval()

        scores: dict[str, float] = {}
        counts: dict[str, dict[str, object]] = {}
        with torch.inference_mode():
            for dataset, (texts, label) in benchmarks.items():
                predictions: list[int] = []
                for start in range(0, len(texts), cfg.batch_size):
                    inputs = tokenizer(
                        texts[start : start + cfg.batch_size],
                        padding=True,
                        truncation=True,
                        max_length=cfg.max_length,
                        return_tensors="pt",
                    ).to(device)
                    if name == "PIGuard_HF":
                        outputs = model(**inputs)
                        logits = (
                            outputs.logits
                            if hasattr(outputs, "logits")
                            else outputs[0]
                        )
                    else:
                        hidden = model.deberta(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                        )["last_hidden_state"]
                        logits = model.classifier(
                            hidden[:, 0, :].to(model.classifier.weight.dtype)
                        )
                    predictions.extend(logits.argmax(-1).cpu().tolist())
                correct = sum(pred == label for pred in predictions)
                scores[dataset] = 100 * correct / len(texts)
                counts[dataset] = {
                    "correct": correct,
                    "n": len(texts),
                    "predictions": predictions,
                }

        over = (
            scores["NotInject_one"]
            + scores["NotInject_two"]
            + scores["NotInject_three"]
        ) / 3
        malicious = (scores["BIPIA_text"] + scores["BIPIA_code"]) / 2
        row = {
            "model": name,
            **scores,
            "overdefense": over,
            "benign": scores["WildGuard"],
            "malicious": malicious,
            "average": (over + scores["WildGuard"] + malicious) / 3,
            "FNR_text": 100 - scores["BIPIA_text"],
            "FNR_code": 100 - scores["BIPIA_code"],
            "FPR_NotInject": 100 - over,
            "FPR_WildGuard": 100 - scores["WildGuard"],
        }
        rows.append(row)
        details[name] = {
            "checkpoint": str(checkpoint),
            "scores": scores,
            "counts": counts,
        }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(paths["output"] / "comparison.csv")
    write_json(
        paths["output"] / "evaluation_raw.json",
        {
            "seed": cfg.seed,
            "max_length": cfg.max_length,
            "pint_included": False,
            "models": details,
        },
    )
    print("\nTabela comparativa (seed 42 / 512 tokens):")
    print(table.round(2).to_string())

    references = {
        "augmented_corrected": "augmented_original",
        "augmented_mof_corrected": "augmented_mof_original",
    }
    deltas: list[dict[str, object]] = []
    for model, reference in references.items():
        if model in table.index and reference in table.index:
            deltas.append(
                {
                    "model": model,
                    "reference": reference,
                    **(table.loc[model] - table.loc[reference]).to_dict(),
                }
            )
    final_name = (
        "augmented_mof_corrected"
        if "augmented_mof_corrected" in table.index
        else "augmented_corrected"
    )
    if "PIGuard_HF" in table.index and final_name in table.index:
        deltas.append(
            {
                "model": final_name,
                "reference": "PIGuard_HF",
                **(table.loc[final_name] - table.loc["PIGuard_HF"]).to_dict(),
            }
        )
    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(paths["output"] / "deltas_pp.csv", index=False)
    print("\nDeltas (pontos percentuais):")
    print(deltas_df.round(2).to_string())

    axis = table[["overdefense", "benign", "malicious"]].plot.bar(
        figsize=(11, 5), ylim=(0, 100), rot=30
    )
    axis.set_ylabel("Acurácia (%)")
    axis.figure.tight_layout()
    axis.figure.savefig(paths["output"] / "comparison.png", dpi=150)
    plt.close(axis.figure)
    print(
        "Interpretar os deltas por conjunto. Uma melhora da média pode "
        "acompanhar perda na detecção de texto.",
        flush=True,
    )
    return table


# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Correções pontuais do PIGuard (seed 42): geração, treino, "
            "Token Recheck, MOF e comparação final."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de saída (padrão: <repo>/Corrections/outputs).",
    )
    parser.add_argument(
        "--reference-output-root",
        default=None,
        help="Diretório com checkpoints de referência (opcional).",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Reutiliza o LLM_augmented.json existente em vez de gerar novamente.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignora manifestos/caches e retreina todos os modelos.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Não avalia leolee99/PIGuard do Hugging Face.",
    )
    parser.add_argument(
        "--ntfy",
        action="store_true",
        help="Envia notificações de início/fim/falha via ntfy.sh (desligado por padrão).",
    )
    parser.add_argument(
        "--ntfy-topic",
        default=os.environ.get("PIGUARD_NTFY_TOPIC", NTFY_TOPIC),
        help="Tópico do ntfy.sh.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--save-step", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global NTFY_ENABLED, NTFY_TOPIC

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_attempts < 1:
        parser.error("--workers e --max-attempts devem ser positivos")

    NTFY_ENABLED = bool(args.ntfy)
    NTFY_TOPIC = args.ntfy_topic

    output = resolve_path(
        args.output_dir,
        ROOT_DIR / "Corrections/outputs",
    )
    paths = {
        "output": output,
        "data": output / "data",
        "logs": output / "logs",
        "augmented": output / "augmented" / "LLM_augmented.json",
    }
    for key in ("output", "data", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)

    args.runtime = runtime_info()
    args.reference_output_root = (
        resolve_path(args.reference_output_root, output)
        if args.reference_output_root
        else None
    )
    print(
        {
            "seed": args.seed,
            "output": str(output),
            **args.runtime,
            "api_key_present": has_api_key(),
        },
        flush=True,
    )

    with timed_stage("1. Geração corrigida e quatro filtros") as notes:
        if args.skip_generation:
            if not paths["augmented"].is_file():
                raise FileNotFoundError(
                    f"--skip-generation informado, mas {paths['augmented']} não existe."
                )
            from Corrections.Upgrade_augmented.new_generate_data import (
                BASE_DISTRIBUTION,
            )

            augmented = json.loads(paths["augmented"].read_text(encoding="utf-8"))
            expected = {
                key.replace(" Injection", ""): value
                for key, value in BASE_DISTRIBUTION.items()
            }
            if len(augmented) != sum(BASE_DISTRIBUTION.values()):
                raise ValueError("Aumento existente não tem 435 amostras.")
            if dict(Counter(row.get("format") for row in augmented)) != expected:
                raise ValueError("Distribuição por categoria diferente da esperada.")
            if any(row.get("label") != 1 for row in augmented):
                raise ValueError("Todas as amostras do aumento devem ter label=1.")
            if any(row.get("source") != "LLM Augmented set" for row in augmented):
                raise ValueError("Campo 'source' inesperado no aumento.")
            clean = json.loads(
                (
                    ROOT_DIR / "datasets/Augmented/train_cleaned_original.json"
                ).read_text(encoding="utf-8")
            )
            normalized = [
                {
                    "prompt": row["prompt"],
                    "label": row["label"],
                    "source": row["source"],
                }
                for row in augmented
            ]
            corrected_path = paths["data"] / "train_augmented_corrected.json"
            write_json(corrected_path, clean + normalized)
        else:
            corrected_path = stage_generate(args, paths)
        notes.append(f"corrigido: {corrected_path}")

    with timed_stage("2. Treinamento e referências") as notes:
        models = stage_train(args, paths, corrected_path)
        notes.extend(f"{name}: {path}" for name, path in models.items())

    with timed_stage("3. Token Recheck sanitizado") as notes:
        trigger_files = stage_recheck(args, paths, models)
        notes.extend(f"{name}: {path}" for name, path in trigger_files.items())

    with timed_stage("4. MOF e retreinamento") as notes:
        stage_mof(args, paths, models, trigger_files, corrected_path)
        if "augmented_mof_corrected" in models:
            notes.append(f"augmented_mof_corrected: {models['augmented_mof_corrected']}")
        else:
            notes.append("sem gatilhos; avaliado sem MOF")

    with timed_stage("5. Comparação final") as notes:
        table = stage_compare(args, paths, models)
        for model_name, row in table.iterrows():
            notes.append(
                f"{model_name}: over={row['overdefense']:.2f}% | "
                f"benign={row['benign']:.2f}% | mal={row['malicious']:.2f}% | "
                f"avg={row['average']:.2f}%"
            )
        notes.append(f"resultados: {paths['output']}")

    print(f"\nConcluído. Artefatos em: {paths['output']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

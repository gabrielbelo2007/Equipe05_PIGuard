"""Tópico 2 do Pipeline Upgrade: Retreinamento Scratch com Base Corrigida Intacta + Novo MOF.

Etapas:
1. Une `Corrections/outputs/data/train_augmented_corrected.json` (75.973 exemplos intactos)
   com `MOF_v2_train.json` (762 exemplos da Etapa 0), totalizando 76.735 exemplos;
2. Valida esquema uniforme (`prompt`, `label`, `source`) e interseção zero com `calibration_heldout.json`;
3. Treina `train_augmented_corrected_new_mof` do zero (a partir de `microsoft/deberta-v3-base`)
   com `seed=42`, `max_length=512`, `epochs=3`, `batch_size=32`, `lr=2e-5`, `warmup=100`, `save_step=200`;
4. Avalia nas 3 modalidades (`PIGuard Puro`, `+ Laya English Não Calibrado`, `+ Laya English Calibrado`).
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import (
    default_corrections_dir,
    load_benchmarks,
    normalize_text,
    read_json,
    resolve_path,
    sha256_file,
    write_csv,
    write_json,
)
from Upgrade.topic1_rp_laya_evaluation import (
    _rows_from_evaluation_results,
    evaluate_piguard_with_laya,
    evaluation_inputs,
    extract_or_load_laya_cache,
    file_signature,
)

if TYPE_CHECKING:
    from Upgrade.Pipeline.laya_english_cascade import LayaEnglishEngine

DEFAULT_UPGRADE_OUTPUT = REPO_ROOT / "Upgrade" / "outputs"


def build_combined_training_dataset(
    *,
    mof_v2_path: Path | str,
    output_path: Path | str,
    corrected_train_path: Optional[Path | str] = None,
    llm_augmented_path: Optional[Path | str] = None,
    heldout_path: Optional[Path | str] = None,
    repo_root: Path | str = REPO_ROOT,
) -> Tuple[Path, Dict[str, Any]]:
    """Constrói `train_augmented_corrected_new_mof.json` preservando a base corrigida 100% intacta."""
    repo = Path(repo_root).expanduser().resolve()

    def _resolve(value: Path | str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (repo / path).resolve()

    corrected_train_path = _resolve(corrected_train_path)
    llm_augmented_path = _resolve(llm_augmented_path)
    heldout_path = _resolve(heldout_path)
    mof_v2_path = _resolve(mof_v2_path)
    output_path = _resolve(output_path)
    assert mof_v2_path is not None and output_path is not None

    if corrected_train_path is not None and corrected_train_path.is_file():
        base_records = read_json(corrected_train_path)
    else:
        clean_path = repo / "datasets" / "Augmented" / "train_cleaned_original.json"
        if llm_augmented_path is None or not llm_augmented_path.is_file():
            raise FileNotFoundError(
                "Necessário fornecer `train_augmented_corrected.json` ou `LLM_augmented.json`."
            )
        clean_records = read_json(clean_path)
        aug_records = read_json(llm_augmented_path)
        base_records = list(clean_records) + list(aug_records)

    mof_records = read_json(mof_v2_path)
    normalized_base = [
        {"prompt": str(r["prompt"]), "label": int(r["label"]), "source": str(r["source"])}
        for r in base_records
    ]
    normalized_mof = [
        {"prompt": str(r["prompt"]), "label": int(r["label"]), "source": str(r["source"])}
        for r in mof_records
    ]
    combined = normalized_base + normalized_mof

    if heldout_path is not None and Path(heldout_path).is_file():
        heldout = read_json(heldout_path)
        train_norm = {normalize_text(r["prompt"]) for r in combined}
        held_norm = {normalize_text(r["prompt"]) for r in heldout}
        overlap = train_norm.intersection(held_norm)
        if overlap:
            raise RuntimeError(
                f"Vazamento detectado: {len(overlap)} amostras de calibration_heldout presentes no treino!"
            )

    dest = Path(output_path)
    write_json(dest, combined)
    stats = {
        "base_count": len(normalized_base),
        "mof_v2_count": len(normalized_mof),
        "total_count": len(combined),
        "output_path": str(dest),
        "dataset_sha256": sha256_file(dest),
    }
    return dest, stats


def _run_streamed(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    tail: deque[str] = deque(maxlen=30)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                tail.append(line.rstrip("\n"))
                log_stream.write(line)
                log_stream.flush()
            rc = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if rc != 0:
        raise RuntimeError(f"Treinamento falhou (código {rc}):\n" + "\n".join(tail))


def run_train_corrected_new_mof(
    *,
    dataset_path: Path | str,
    model_dir: Path | str,
    repo_root: Path | str = REPO_ROOT,
    model_name: str = "train_augmented_corrected_new_mof",
    force_retrain: bool = False,
) -> Path:
    """Executa o treinamento scratch via ``Original/train.py`` com checkpointing seguro."""
    import torch
    import transformers

    repo = Path(repo_root).expanduser().resolve()
    target = Path(model_dir).expanduser()
    if not target.is_absolute():
        target = (repo / target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_path).expanduser()
    if not dataset.is_absolute():
        dataset = (repo / dataset).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset de treino ausente: {dataset}")

    spec = {
        "seed": 42,
        "max_length": 512,
        "epochs": 3,
        "batch_size": 32,
        "eval_batch_size": 32,
        "lr": 2e-5,
        "warmup": 100,
        "save_step": 200,
        "dataset_sha256": sha256_file(dataset),
        "valid_sha256": sha256_file(repo / "datasets" / "valid.json"),
        "train_sha256": sha256_file(repo / "Original" / "train.py"),
        "eval_sha256": sha256_file(repo / "Original" / "eval.py"),
        "model_sha256": sha256_file(repo / "Original" / "PIGuard.py"),
        "params_sha256": sha256_file(repo / "Original" / "params.py"),
        "util_sha256": sha256_file(repo / "Original" / "util.py"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }

    manifest_path = target / "manifest.json"
    best_ckpt = target / "best_model.pth"
    if not force_retrain and manifest_path.is_file() and best_ckpt.is_file():
        try:
            if read_json(manifest_path) == spec:
                print(f"[Tópico 2] Reutilizando checkpoint treinado em {best_ckpt}", flush=True)
                return best_ckpt
        except Exception:
            pass

    work_dir = repo / "Upgrade" / "work" / model_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    local_log = work_dir / "stdout.log"

    command = [
        sys.executable,
        "-u",
        str(repo / "Original" / "train.py"),
        "--name",
        model_name,
        "--train_set",
        str(dataset),
        "--valid_set",
        "datasets/valid.json",
        "--dataset_root",
        "datasets",
        "--checkpoint_path",
        str(work_dir),
        "--logs",
        str(target),
        "--seed",
        "42",
        "--max_length",
        "512",
        "--epochs",
        "3",
        "--batch_size",
        "32",
        "--eval_batch_size",
        "32",
        "--lr",
        "2e-5",
        "--warmup",
        "100",
        "--save_step",
        "200",
    ]
    print(f"\n{'=' * 70}\n[Tópico 2] INICIANDO TREINAMENTO SCRATCH: {model_name}\n{'=' * 70}", flush=True)
    try:
        _run_streamed(command, cwd=repo, log_path=local_log)
    finally:
        if local_log.is_file():
            shutil.copy2(local_log, target / "stdout.log")

    work_best = work_dir / "best_model.pth"
    if not work_best.is_file():
        raise RuntimeError(f"Checkpoint ausente após treino: {work_best}")
    shutil.copy2(work_best, best_ckpt)
    write_json(manifest_path, spec)
    shutil.rmtree(work_dir, ignore_errors=True)
    return best_ckpt


def run_topic2(
    *,
    mof_v2_path: Path | str,
    heldout_path: Path | str,
    output_dir: Path | str,
    corrected_train_path: Optional[Path | str] = None,
    llm_augmented_path: Optional[Path | str] = None,
    repo_root: Path | str = REPO_ROOT,
    model_name: str = "train_augmented_corrected_new_mof",
    laya_engine: Optional[LayaEnglishEngine] = None,
    fpr_budget: float = 0.01,
    n_boot: int = 2000,
    seed: int = 42,
    batch_size: int = 32,
    laya_batch_size: int = 16,
    force_retrain: bool = False,
    force_eval: bool = False,
    checkpoint_override: Optional[Path | str] = None,
    predict_fn: Optional[Callable[[Sequence[str]], Tuple[List[int], List[float]]]] = None,
) -> Dict[str, Any]:
    """Executa todo o Tópico 2: construção do dataset, treino scratch e avaliação Puro vs Laya."""
    repo = Path(repo_root).expanduser().resolve()
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    mof_v2_path = Path(mof_v2_path).expanduser()
    if not mof_v2_path.is_absolute():
        mof_v2_path = (repo / mof_v2_path).resolve()
    heldout_path = Path(heldout_path).expanduser()
    if not heldout_path.is_absolute():
        heldout_path = (repo / heldout_path).resolve()
    if corrected_train_path is not None:
        corrected_train_path = Path(corrected_train_path).expanduser()
        if not corrected_train_path.is_absolute():
            corrected_train_path = (repo / corrected_train_path).resolve()
    if llm_augmented_path is not None:
        llm_augmented_path = Path(llm_augmented_path).expanduser()
        if not llm_augmented_path.is_absolute():
            llm_augmented_path = (repo / llm_augmented_path).resolve()
    if checkpoint_override is not None:
        checkpoint_override = Path(checkpoint_override).expanduser()
        if not checkpoint_override.is_absolute():
            checkpoint_override = (repo / checkpoint_override).resolve()
    data_dir = out / "data"
    topic2_dir = out / "topic2"
    cache_dir = out / "cache"
    model_dir = out / model_name
    for d in (data_dir, topic2_dir, cache_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    result_json_path = topic2_dir / "topic2_results.json"
    benchmarks = load_benchmarks(repo / "datasets", include_tensortrust=True)
    heldout_records = read_json(heldout_path)
    expected_checkpoint = checkpoint_override or (model_dir / "best_model.pth")
    if checkpoint_override is not None and predict_fn is None and not expected_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint solicitado ausente: {expected_checkpoint}")
    base = file_signature(corrected_train_path)
    if base is None:
        base = [
            file_signature(repo / "datasets/Augmented/train_cleaned_original.json"),
            file_signature(llm_augmented_path),
        ]
    inputs = {
        "evaluation": evaluation_inputs(
            benchmarks, heldout_path, fpr_budget=fpr_budget, n_boot=n_boot, seed=seed,
            batch_size=batch_size, laya_batch_size=laya_batch_size,
        ),
        "mof_v2": sha256_file(mof_v2_path),
        "base": base,
        "model": model_name,
        "checkpoint": file_signature(expected_checkpoint),
    }
    if not force_eval and not force_retrain and predict_fn is None and result_json_path.is_file():
        cached = read_json(result_json_path)
        if inputs["checkpoint"] and cached.get("inputs") == inputs and not cached.get("simulated_predictions"):
            print(f"[Tópico 2] Reutilizando resultados salvos em {result_json_path}", flush=True)
            return cached

    combined_path, dataset_stats = build_combined_training_dataset(
        mof_v2_path=mof_v2_path,
        output_path=data_dir / f"{model_name}.json",
        corrected_train_path=corrected_train_path,
        llm_augmented_path=llm_augmented_path,
        heldout_path=heldout_path,
        repo_root=repo_root,
    )

    if checkpoint_override is not None:
        ckpt_path = Path(checkpoint_override)
    elif predict_fn is not None:
        ckpt_path = model_dir / "best_model.pth"
    else:
        ckpt_path = run_train_corrected_new_mof(
            dataset_path=combined_path,
            model_dir=model_dir,
            repo_root=repo_root,
            model_name=model_name,
            force_retrain=force_retrain,
        )

    inputs["checkpoint"] = file_signature(ckpt_path)

    bench_texts = {ds_name: texts for ds_name, (texts, _) in benchmarks.items()}
    bench_laya_cache = extract_or_load_laya_cache(
        bench_texts,
        cache_dir / "laya_english_benchmark_logits.json",
        laya_engine=laya_engine,
        batch_size=laya_batch_size,
        force=False,
    )
    heldout_laya_cache = extract_or_load_laya_cache(
        {"heldout": [str(row["prompt"]) for row in heldout_records]},
        cache_dir / "laya_english_heldout_logits.json",
        laya_engine=laya_engine,
        batch_size=laya_batch_size,
        force=False,
    )

    eval_result = evaluate_piguard_with_laya(
        model_name=model_name,
        checkpoint_path=ckpt_path,
        benchmarks=benchmarks,
        heldout_records=heldout_records,
        benchmark_laya_cache=bench_laya_cache,
        heldout_laya_cache=heldout_laya_cache,
        fpr_budget=fpr_budget,
        n_boot=n_boot,
        seed=seed,
        batch_size=batch_size,
        predict_fn=predict_fn,
    )

    summary_rows, bootstrap_rows = _rows_from_evaluation_results([eval_result])
    write_csv(topic2_dir / "topic2_summary.csv", summary_rows)
    write_csv(topic2_dir / "topic2_bootstrap_ci.csv", bootstrap_rows)

    payload = {
        "status": "complete",
        "inputs": inputs,
        "simulated_predictions": predict_fn is not None,
        "model_name": model_name,
        "dataset_stats": dataset_stats,
        "result": eval_result,
        "summary_rows": summary_rows,
    }
    write_json(result_json_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mof-v2",
        default=None,
        help="Caminho para MOF_v2_train.json (padrão: <output>/step0/MOF_v2_train.json).",
    )
    parser.add_argument(
        "--heldout",
        default=None,
        help="Caminho para calibration_heldout.json (padrão: <output>/step0/...).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório raiz dos artefatos Upgrade (padrão: <repo>/Upgrade/outputs).",
    )
    parser.add_argument(
        "--corrections-dir",
        default=None,
        help="Diretório de artefatos do Corrections (padrão: Corrections/outputs).",
    )
    parser.add_argument(
        "--corrected-train",
        default=None,
        help="Caminho para train_augmented_corrected.json (padrão: <Corrections>/data/...).",
    )
    parser.add_argument(
        "--llm-augmented",
        default=None,
        help="Fallback para LLM_augmented.json (padrão: <Corrections>/augmented/...).",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    output_dir = resolve_path(
        args.output_dir, repo_root / "Upgrade" / "outputs", repo_root=repo_root
    )
    corrections_dir = resolve_path(
        args.corrections_dir,
        default_corrections_dir(repo_root),
        repo_root=repo_root,
    )
    mof_v2_path = resolve_path(
        args.mof_v2, output_dir / "step0" / "MOF_v2_train.json", repo_root=repo_root
    )
    heldout_path = resolve_path(
        args.heldout,
        output_dir / "step0" / "calibration_heldout.json",
        repo_root=repo_root,
    )
    corrected_train = resolve_path(
        args.corrected_train,
        corrections_dir / "data" / "train_augmented_corrected.json",
        repo_root=repo_root,
    )
    llm_augmented = resolve_path(
        args.llm_augmented,
        corrections_dir / "augmented" / "LLM_augmented.json",
        repo_root=repo_root,
    )

    payload = run_topic2(
        mof_v2_path=mof_v2_path,
        heldout_path=heldout_path,
        output_dir=output_dir,
        corrected_train_path=corrected_train,
        llm_augmented_path=llm_augmented,
        repo_root=repo_root,
        force_retrain=args.force_retrain,
        force_eval=args.force_eval,
    )
    print(f"Tópico 2 concluído: {payload['dataset_stats']['total_count']} amostras de treino.")


if __name__ == "__main__":
    main()

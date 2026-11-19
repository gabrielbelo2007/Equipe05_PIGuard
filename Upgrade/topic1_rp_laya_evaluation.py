"""Tópico 1 do Pipeline Upgrade: Avaliação dos 5 Modelos `rp` de RESULTADOS_SEED42_MAX512 com Laya English.

Reutiliza diretamente os 5 checkpoints `.pth` marcados como `rp` (em
`outputs/` ou em um diretório indicado por `--drive-rp-root`), sem
retreiná-los, e compara cada modelo nas 3 configurações:
1. `pure`: PIGuard Puro (referência de `RESULTADOS_SEED42_MAX512`);
2. `laya_uncalibrated`: PIGuard + Laya English Não Calibrado (T=1.0, T_L=0.50);
3. `laya_calibrated`: PIGuard + Laya English Calibrado (T*, T_L* ajustados em `calibration_heldout.json`).

Inclui pré-extração com cache em disco dos logits do Laya English, verificação contra os caches
originais de `RESULTADOS_SEED42_MAX512/Gráficos/_cache/`, deltas em pontos percentuais, IC 95% via
Paired Bootstrap (2.000 réplicas) e métricas completas de latência e Call Rate (%).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

# Permite executar este arquivo diretamente de qualquer diretório.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import (
    ALL_BENCHMARK_SETS,
    BENCHMARK_LABELS,
    PUBLIC_SETS,
    aggregate_benchmark_scores,
    load_benchmarks,
    read_json,
    resolve_path,
    sha256_file,
    write_csv,
    write_json,
)

if TYPE_CHECKING:
    from Upgrade.laya_english_cascade import LayaEnglishEngine

RP_MODELS: Tuple[str, ...] = (
    "train_augmented_rp",
    "train_clean_mof_rp",
    "train_augmented_mof_finetuning_rp",
    "train_augmented_mof_rp",
    "train_augmented_shortcut_rp",
)

DEFAULT_RP_ROOT = REPO_ROOT / "outputs"
DEFAULT_UPGRADE_OUTPUT = REPO_ROOT / "Upgrade" / "outputs"


def _texts_digest(texts: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8", errors="replace")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
    return hasher.hexdigest()


def evaluation_inputs(
    benchmarks: Dict[str, Tuple[List[str], int]],
    heldout_path: Path,
    *,
    fpr_budget: float,
    n_boot: int,
    seed: int,
    batch_size: int,
    laya_batch_size: int,
) -> Dict[str, Any]:
    return {
        "benchmarks": {name: [_texts_digest(texts), label] for name, (texts, label) in benchmarks.items()},
        "heldout_sha256": sha256_file(heldout_path),
        "fpr_budget": fpr_budget,
        "n_boot": n_boot,
        "seed": seed,
        "batch_size": batch_size,
        "laya_batch_size": laya_batch_size,
    }


def file_signature(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None or not path.is_file():
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def extract_or_load_laya_cache(
    texts_by_key: Dict[str, Sequence[str]],
    cache_path: Path | str,
    *,
    laya_engine: Optional[LayaEnglishEngine] = None,
    window_max_len: int = 2048,
    overlap_tokens: int = 512,
    batch_size: int = 16,
    force: bool = False,
) -> Dict[str, Any]:
    """Extrai (ou reaproveita do cache em disco) os logits por janela do Laya English."""
    path = Path(cache_path)
    expected_hashes = {k: _texts_digest(v) for k, v in texts_by_key.items()}
    if not force and path.is_file():
        cached = read_json(path)
        if (
            isinstance(cached, dict)
            and cached.get("hashes") == expected_hashes
            and int(cached.get("window_max_len", -1)) == int(window_max_len)
            and int(cached.get("overlap_tokens", -1)) == int(overlap_tokens)
        ):
            print(f"[Laya Cache] Reaproveitando logits salvos em {path}", flush=True)
            return cached

    if laya_engine is None:
        from Upgrade.laya_english_cascade import LayaEnglishEngine

        engine = LayaEnglishEngine(
            window_max_len=window_max_len,
            overlap_tokens=overlap_tokens,
        )
    else:
        engine = laya_engine
    entries: Dict[str, Any] = {}
    for key, texts in texts_by_key.items():
        print(f"[Laya English] Extraindo logits por janela para '{key}' ({len(texts)} amostras)...", flush=True)
        win_logits, sample_ms = engine.extract_window_logits(list(texts), batch_size=batch_size)
        entries[key] = {
            "n": len(texts),
            "window_logits": win_logits,
            "laya_ms": sample_ms,
        }

    payload = {
        "model": "convaiinnovations/laya (English)",
        "window_max_len": int(window_max_len),
        "overlap_tokens": int(overlap_tokens),
        "hashes": expected_hashes,
        "entries": entries,
    }
    write_json(path, payload)
    return payload


def run_piguard_inference(
    checkpoint_path: Optional[Path | str],
    texts_by_key: Dict[str, Sequence[str]],
    *,
    hf_model_id: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    predict_fn: Optional[Callable[[Sequence[str]], Tuple[List[int], List[float]]]] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    """Executa inferência do PIGuard (.pth local ou Hugging Face) e mede latência por amostra (ms)."""
    preds_by_key: Dict[str, List[int]] = {}
    ms_by_key: Dict[str, List[float]] = {}

    if predict_fn is not None:
        for key, texts in texts_by_key.items():
            preds, ms_list = predict_fn(texts)
            preds_by_key[key] = [int(p) for p in preds]
            ms_by_key[key] = [float(m) for m in ms_list]
        return preds_by_key, ms_by_key

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from Corrections.renew_token_recheck import load_checkpoint

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    if hf_model_id:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_id, trust_remote_code=True
        ).to(resolved_device)
        model.eval()
        classifier_dtype = None
    else:
        if checkpoint_path is None:
            raise ValueError("Necessário informar `checkpoint_path` ou `hf_model_id`.")
        _, model = load_checkpoint(str(checkpoint_path), resolved_device)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        classifier_dtype = next(model.classifier.parameters()).dtype

    try:
        with torch.inference_mode():
            for key, texts in texts_by_key.items():
                key_preds: List[int] = []
                key_ms: List[float] = []
                for start in range(0, len(texts), batch_size):
                    batch = list(texts[start : start + batch_size])
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(resolved_device)

                    if resolved_device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if hf_model_id:
                        logits = model(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                        ).logits
                    else:
                        hidden = model.deberta(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                            output_hidden_states=False,
                        )["last_hidden_state"]
                        pooled = hidden[:, 0, :].to(dtype=classifier_dtype)
                        logits = model.classifier(pooled)
                    batch_preds = logits.argmax(dim=-1).cpu().tolist()
                    if resolved_device.type == "cuda":
                        torch.cuda.synchronize()
                    per_item_ms = ((time.perf_counter() - t0) * 1000.0) / len(batch)

                    key_preds.extend(int(p) for p in batch_preds)
                    key_ms.extend([per_item_ms] * len(batch))
                preds_by_key[key] = key_preds
                ms_by_key[key] = key_ms
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return preds_by_key, ms_by_key


def _compute_scores_from_preds(
    preds_by_ds: Dict[str, Sequence[int]],
    labels_by_ds: Dict[str, int],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    scores: Dict[str, float] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for ds_name, preds in preds_by_ds.items():
        target = int(labels_by_ds[ds_name])
        correct = sum(1 for p in preds if int(p) == target)
        n = len(preds)
        scores[ds_name] = 100.0 * correct / n if n else 0.0
        counts[ds_name] = {"correct": int(correct), "n": int(n)}
    return scores, counts


def evaluate_piguard_with_laya(
    model_name: str,
    checkpoint_path: Optional[Path | str],
    *,
    benchmarks: Dict[str, Tuple[List[str], int]],
    heldout_records: Sequence[Dict[str, Any]],
    benchmark_laya_cache: Dict[str, Any],
    heldout_laya_cache: Dict[str, Any],
    hf_model_id: Optional[str] = None,
    drive_cache_reference: Optional[Dict[str, Any]] = None,
    fpr_budget: float = 0.01,
    n_boot: int = 2000,
    seed: int = 42,
    batch_size: int = 32,
    predict_fn: Optional[Callable[[Sequence[str]], Tuple[List[int], List[float]]]] = None,
) -> Dict[str, Any]:
    """Avalia um modelo PIGuard em 3 modos: Puro, + Laya Não Calibrado e + Laya Calibrado."""
    from Upgrade.laya_english_cascade import (
        calibrate_laya_on_heldout,
        cascade_predict,
        compute_cascade_latency_metrics,
        paired_bootstrap_ci,
        score_dataset_from_window_logits,
    )

    texts_by_key: Dict[str, Sequence[str]] = {
        ds_name: texts for ds_name, (texts, _) in benchmarks.items()
    }
    texts_by_key["heldout"] = [str(row["prompt"]) for row in heldout_records]

    pi_preds_all, pi_ms_all = run_piguard_inference(
        checkpoint_path=checkpoint_path,
        texts_by_key=texts_by_key,
        hf_model_id=hf_model_id,
        batch_size=batch_size,
        predict_fn=predict_fn,
    )

    heldout_pi_preds = pi_preds_all["heldout"]
    heldout_labels = [int(row["label"]) for row in heldout_records]
    heldout_win_logits = heldout_laya_cache["entries"]["heldout"]["window_logits"]

    # Calibração pós-hoc em `calibration_heldout.json` (T* por NLL + T_L* por orçamento FPR)
    calibration = calibrate_laya_on_heldout(
        heldout_window_logits=heldout_win_logits,
        heldout_labels=heldout_labels,
        heldout_piguard_preds=heldout_pi_preds,
        fpr_budget=fpr_budget,
    )

    labels_by_ds = {ds_name: label for ds_name, (_, label) in benchmarks.items()}
    pi_preds_ds = {ds_name: pi_preds_all[ds_name] for ds_name in benchmarks}
    pi_ms_ds = {ds_name: pi_ms_all[ds_name] for ds_name in benchmarks}

    # Scores Laya Não Calibrado (T=1.0, T_L=0.50) e Calibrado (T*, T_L*)
    uncal_preds_ds: Dict[str, List[int]] = {}
    cal_preds_ds: Dict[str, List[int]] = {}
    laya_ms_ds: Dict[str, List[float]] = {}

    for ds_name in benchmarks:
        entry = benchmark_laya_cache["entries"][ds_name]
        win_logits = entry["window_logits"]
        laya_ms_ds[ds_name] = [float(x) for x in entry["laya_ms"]]

        scores_uncal = score_dataset_from_window_logits(win_logits, temperature=1.0)
        scores_cal = score_dataset_from_window_logits(
            win_logits, temperature=calibration["temperature_star"]
        )
        uncal_preds_ds[ds_name] = cascade_predict(
            pi_preds_ds[ds_name], scores_uncal, threshold_tl=0.50
        )
        cal_preds_ds[ds_name] = cascade_predict(
            pi_preds_ds[ds_name], scores_cal, threshold_tl=calibration["threshold_tl"]
        )

    modes: Dict[str, Any] = {}
    mode_specs = [
        ("pure", pi_preds_ds, None, None, None),
        ("laya_uncalibrated", uncal_preds_ds, laya_ms_ds, 1.0, 0.50),
        (
            "laya_calibrated",
            cal_preds_ds,
            laya_ms_ds,
            calibration["temperature_star"],
            calibration["threshold_tl"],
        ),
    ]

    for mode_name, preds_ds, laya_ms_arg, temp_val, tl_val in mode_specs:
        scores, counts = _compute_scores_from_preds(preds_ds, labels_by_ds)
        metrics = aggregate_benchmark_scores(scores)
        latency_public = compute_cascade_latency_metrics(
            pi_preds_ds,
            pi_ms_ds,
            laya_ms_arg,
            average_accuracy=metrics["average"],
            datasets_for_latency=PUBLIC_SETS,
        )
        latency_all = compute_cascade_latency_metrics(
            pi_preds_ds,
            pi_ms_ds,
            laya_ms_arg,
            average_accuracy=metrics["average"],
            datasets_for_latency=list(benchmarks.keys()),
        )
        bootstrap = (
            paired_bootstrap_ci(
                pi_preds_ds, preds_ds, labels_by_ds, n_boot=n_boot, seed=seed
            )
            if mode_name != "pure"
            else None
        )
        modes[mode_name] = {
            "temperature": temp_val,
            "threshold_tl": tl_val,
            "scores": scores,
            "counts": counts,
            "metrics": metrics,
            "latency_public": latency_public,
            "latency_all": latency_all,
            "bootstrap_vs_pure": bootstrap,
        }

    # Bootstrap pareado Calibrado vs Não Calibrado
    bootstrap_cal_vs_uncal = paired_bootstrap_ci(
        uncal_preds_ds, cal_preds_ds, labels_by_ds, n_boot=n_boot, seed=seed
    )

    return {
        "model": model_name,
        "checkpoint": str(hf_model_id or checkpoint_path),
        "calibration": calibration,
        "drive_cache_reference": drive_cache_reference,
        "modes": modes,
        "bootstrap_calibrated_vs_uncalibrated": bootstrap_cal_vs_uncal,
    }


def _rows_from_evaluation_results(results: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []

    for res in results:
        model_name = res["model"]
        pure_avg = res["modes"]["pure"]["metrics"]["average"]
        for mode_key, mode_label in (
            ("pure", "PIGuard Puro"),
            ("laya_uncalibrated", "+ Laya English (Não Calibrado)"),
            ("laya_calibrated", "+ Laya English (Calibrado)"),
        ):
            m = res["modes"][mode_key]
            met = m["metrics"]
            sc = m["scores"]
            lat = m["latency_public"]
            boot = m["bootstrap_vs_pure"] or {}
            avg_boot = boot.get("average", {})
            summary_rows.append(
                {
                    "model": model_name,
                    "mode": mode_key,
                    "mode_label": mode_label,
                    "temperature": m["temperature"],
                    "threshold_tl": m["threshold_tl"],
                    "NotInject_one": sc.get("NotInject_one"),
                    "NotInject_two": sc.get("NotInject_two"),
                    "NotInject_three": sc.get("NotInject_three"),
                    "overdefense": met["overdefense"],
                    "benign": met["benign"],
                    "BIPIA_text": sc.get("BIPIA_text"),
                    "BIPIA_code": sc.get("BIPIA_code"),
                    "malicious": met["malicious"],
                    "average": met["average"],
                    "delta_average_pp": met["average"] - pure_avg,
                    "ci95_average_low": avg_boot.get("ci95_low"),
                    "ci95_average_high": avg_boot.get("ci95_high"),
                    "tensor_trust_hijacking": met.get("tensor_trust_hijacking"),
                    "tensor_trust_extraction": met.get("tensor_trust_extraction"),
                    "tensor_trust_average": met.get("tensor_trust_average"),
                    "call_rate_pct": lat["call_rate_pct"],
                    "inference_ms": lat["inference_ms"],
                    "p50_ms": lat["p50_ms"],
                    "p95_ms": lat["p95_ms"],
                    "efficiency": lat["efficiency"],
                }
            )
            if boot:
                for metric_name, bvals in boot.items():
                    bootstrap_rows.append(
                        {
                            "model": model_name,
                            "comparison": f"{mode_key}_vs_pure",
                            "metric": metric_name,
                            **bvals,
                        }
                    )

        for metric_name, bvals in res["bootstrap_calibrated_vs_uncalibrated"].items():
            bootstrap_rows.append(
                {
                    "model": model_name,
                    "comparison": "calibrated_vs_uncalibrated",
                    "metric": metric_name,
                    **bvals,
                }
            )

    return summary_rows, bootstrap_rows


RP_CHECKPOINT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train_clean_mof_rp": ("train_clean_mof",),
    "train_augmented_mof_finetuning_rp": ("train_augmented_mof_finetuning",),
    "train_augmented_mof_rp": ("train_augmented_mof",),
}


def resolve_rp_checkpoint(rp_root: Path, model_name: str) -> Path:
    candidates = [model_name, *RP_CHECKPOINT_ALIASES.get(model_name, ())]
    for cand in candidates:
        ckpt = rp_root / cand / "best_model.pth"
        if ckpt.is_file():
            return ckpt
    return rp_root / model_name / "best_model.pth"


def run_topic1(
    *,
    drive_rp_root: Path | str,
    heldout_path: Path | str,
    output_dir: Path | str,
    repo_root: Path | str = REPO_ROOT,
    models: Sequence[str] = RP_MODELS,
    laya_engine: Optional[LayaEnglishEngine] = None,
    fpr_budget: float = 0.01,
    n_boot: int = 2000,
    seed: int = 42,
    batch_size: int = 32,
    laya_batch_size: int = 16,
    force: bool = False,
    predict_fn_factory: Optional[Callable[[str, Path], Callable[[Sequence[str]], Tuple[List[int], List[float]]]]] = None,
) -> Dict[str, Any]:
    """Executa o Tópico 1 nos 5 modelos `rp` de RESULTADOS_SEED42_MAX512 com Laya English."""
    repo = Path(repo_root).expanduser().resolve()
    rp_root = Path(drive_rp_root).expanduser()
    if not rp_root.is_absolute():
        rp_root = (repo / rp_root).resolve()
    heldout_path = Path(heldout_path).expanduser()
    if not heldout_path.is_absolute():
        heldout_path = (repo / heldout_path).resolve()
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    topic1_dir = out / "topic1"
    cache_dir = out / "cache"
    topic1_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = topic1_dir / "topic1_rp_laya_results.json"
    benchmarks = load_benchmarks(repo / "datasets", include_tensortrust=True)
    heldout_records = read_json(heldout_path)
    checkpoints = {name: resolve_rp_checkpoint(rp_root, name) for name in models}
    references = {
        name: rp_root / "Gráficos" / "_cache" / f"{name}.json" for name in models
    }
    if predict_fn_factory is None:
        for ckpt in checkpoints.values():
            if not ckpt.is_file():
                raise FileNotFoundError(f"Checkpoint `rp` ausente: {ckpt}")
    inputs = {
        "evaluation": evaluation_inputs(
            benchmarks, heldout_path, fpr_budget=fpr_budget, n_boot=n_boot, seed=seed,
            batch_size=batch_size, laya_batch_size=laya_batch_size,
        ),
        "models": list(models),
        "checkpoints": {name: file_signature(path) for name, path in checkpoints.items()},
        "references": {name: file_signature(path) for name, path in references.items()},
    }
    if not force and predict_fn_factory is None and result_json_path.is_file():
        cached = read_json(result_json_path)
        if cached.get("inputs") == inputs and not cached.get("simulated_predictions"):
            print(f"[Tópico 1] Reutilizando resultados salvos em {result_json_path}", flush=True)
            return cached

    # Pré-extração compartilhada do Laya English para todos os benchmarks e held-out
    bench_texts = {ds_name: texts for ds_name, (texts, _) in benchmarks.items()}
    bench_laya_cache = extract_or_load_laya_cache(
        bench_texts,
        cache_dir / "laya_english_benchmark_logits.json",
        laya_engine=laya_engine,
        batch_size=laya_batch_size,
        force=force,
    )
    heldout_laya_cache = extract_or_load_laya_cache(
        {"heldout": [str(row["prompt"]) for row in heldout_records]},
        cache_dir / "laya_english_heldout_logits.json",
        laya_engine=laya_engine,
        batch_size=laya_batch_size,
        force=force,
    )

    model_results: List[Dict[str, Any]] = []
    for model_name in models:
        ckpt = checkpoints[model_name]
        drive_cache_file = references[model_name]
        drive_ref = read_json(drive_cache_file) if drive_cache_file.is_file() else None
        if isinstance(drive_ref, dict):
            drive_ref = {
                "scores": drive_ref.get("scores"),
                "metrics": drive_ref.get("metrics"),
                "inference_ms": drive_ref.get("inference_ms"),
            }

        print(f"\n[Tópico 1] Avaliando modelo rp: {model_name}", flush=True)
        p_fn = predict_fn_factory(model_name, ckpt) if predict_fn_factory is not None else None
        res = evaluate_piguard_with_laya(
            model_name=model_name,
            checkpoint_path=ckpt,
            benchmarks=benchmarks,
            heldout_records=heldout_records,
            benchmark_laya_cache=bench_laya_cache,
            heldout_laya_cache=heldout_laya_cache,
            drive_cache_reference=drive_ref,
            fpr_budget=fpr_budget,
            n_boot=n_boot,
            seed=seed,
            batch_size=batch_size,
            predict_fn=p_fn,
        )
        model_results.append(res)

    summary_rows, bootstrap_rows = _rows_from_evaluation_results(model_results)
    write_csv(topic1_dir / "topic1_rp_laya_summary.csv", summary_rows)
    write_csv(topic1_dir / "topic1_rp_laya_bootstrap_ci.csv", bootstrap_rows)

    payload = {
        "status": "complete",
        "inputs": inputs,
        "simulated_predictions": predict_fn_factory is not None,
        "fpr_budget": fpr_budget,
        "n_boot": n_boot,
        "seed": seed,
        "models": list(models),
        "results": model_results,
        "summary_rows": summary_rows,
    }
    write_json(result_json_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-rp-root",
        default=None,
        help=(
            "Pasta com os checkpoints `*_rp` e `Gráficos/_cache` "
            "(padrão: <repo>/outputs)."
        ),
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
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--fpr-budget", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    output_dir = resolve_path(
        args.output_dir, repo_root / "Upgrade" / "outputs", repo_root=repo_root
    )
    rp_root = resolve_path(
        args.drive_rp_root, repo_root / "outputs", repo_root=repo_root
    )
    heldout_path = resolve_path(
        args.heldout,
        output_dir / "step0" / "calibration_heldout.json",
        repo_root=repo_root,
    )

    payload = run_topic1(
        drive_rp_root=rp_root,
        heldout_path=heldout_path,
        output_dir=output_dir,
        repo_root=repo_root,
        fpr_budget=args.fpr_budget,
        n_boot=args.n_boot,
        seed=args.seed,
        force=args.force,
    )
    print(f"Tópico 1 concluído para {len(payload['results'])} modelos rp.")


if __name__ == "__main__":
    main()

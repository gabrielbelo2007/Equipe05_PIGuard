"""Comparação focada na linhagem `corrected` do PIGuard sob protocolo unificado.

Avalia e compara sob o mesmo protocolo completo (PIGuard Puro, + Laya English
Não Calibrado, + Laya English Calibrado, TensorTrust, Latência CUDA e IC 95%
via Paired Bootstrap) os 5 modelos do escopo `corrected`:
1. `augmented_corrected` (Corrections: base corrigida 75.973, sem MOF);
2. `augmented_mof_corrected` (Corrections: base corrigida + MOF antigo);
3. `train_augmented_corrected_new_mof` (Upgrade Tópico 2: base corrigida + Novo MOF v2);
4. `augmented_mof_original` (Âncora Corrections: reprodução original antes da correção);
5. `PIGuard_HF` (Âncora oficial Hugging Face: `leolee99/PIGuard`).

Reutiliza os caches de logits do Laya English em `Upgrade/cache/` e o resultado
completo de `Upgrade/topic2/topic2_results.json` quando disponíveis, gravando
todos os novos artefatos isoladamente em `Upgrade/corrected_only/`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import (
    PUBLIC_SETS,
    default_corrections_dir,
    load_benchmarks,
    read_json,
    resolve_path,
    write_csv,
    write_json,
)
from Upgrade.laya_english_cascade import LayaEnglishEngine
from Upgrade.topic1_rp_laya_evaluation import (
    _rows_from_evaluation_results,
    evaluate_piguard_with_laya,
    extract_or_load_laya_cache,
)

CORRECTED_COMPARISON_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "model": "augmented_corrected",
        "display_name": "augmented_corrected (Sem MOF)",
        "short_name": "aug_corrected\n(Sem MOF)",
        "group": "corrected",
        "mof_note": "Base corrigida intacta (75.973), sem MOF",
        "source_tag": "Corrections",
        "ckpt_root": "corrections",
        "ckpt_rel": "augmented_corrected/best_model.pth",
        "hf_model_id": None,
        "ref_key": "augmented_corrected",
        "reuse_topic2": False,
    },
    {
        "model": "augmented_mof_corrected",
        "display_name": "augmented_mof_corrected (MOF Antigo)",
        "short_name": "aug_mof_corrected\n(MOF Antigo)",
        "group": "corrected",
        "mof_note": "Base corrigida + MOF antigo (337/237/188)",
        "source_tag": "Corrections",
        "ckpt_root": "corrections",
        "ckpt_rel": "augmented_mof_corrected/best_model.pth",
        "hf_model_id": None,
        "ref_key": "augmented_mof_corrected",
        "reuse_topic2": False,
    },
    {
        "model": "train_augmented_corrected_new_mof",
        "display_name": "train_augmented_corrected_new_mof (MOF v2)",
        "short_name": "corrected_new_mof\n(MOF v2)",
        "group": "corrected",
        "mof_note": "Base corrigida + Novo MOF v2 (152/229/381)",
        "source_tag": "Upgrade Tópico 2",
        "ckpt_root": "upgrade",
        "ckpt_rel": "train_augmented_corrected_new_mof/best_model.pth",
        "hf_model_id": None,
        "ref_key": None,
        "reuse_topic2": True,
    },
    {
        "model": "augmented_mof_original",
        "display_name": "augmented_mof_original (Âncora)",
        "short_name": "aug_mof_original\n(Âncora)",
        "group": "anchor",
        "mof_note": "Âncora: reprodução original antes da correção",
        "source_tag": "Corrections (Âncora)",
        "ckpt_root": "corrections",
        "ckpt_rel": "augmented_mof_original/best_model.pth",
        "hf_model_id": None,
        "ref_key": "augmented_mof_original",
        "reuse_topic2": False,
    },
    {
        "model": "PIGuard_HF",
        "display_name": "PIGuard_HF (Oficial)",
        "short_name": "PIGuard_HF\n(Oficial)",
        "group": "anchor",
        "mof_note": "Âncora oficial HF (leolee99/PIGuard)",
        "source_tag": "HuggingFace (Âncora)",
        "ckpt_root": "hf",
        "ckpt_rel": None,
        "hf_model_id": "leolee99/PIGuard",
        "ref_key": "PIGuard_HF",
        "reuse_topic2": False,
    },
)


def _load_corrections_eval_raw(corrections_dir: Path) -> Dict[str, Dict[str, float]]:
    raw_path = corrections_dir / "evaluation_raw.json"
    if not raw_path.is_file():
        return {}
    raw = read_json(raw_path)
    out: Dict[str, Dict[str, float]] = {}
    for name, info in (raw.get("models") or {}).items():
        scores = info.get("scores") or {}
        if scores:
            out[name] = {k: float(v) for k, v in scores.items()}
    return out


def _is_valid_topic2_result(payload: Any, expected_model: str) -> bool:
    if not isinstance(payload, dict):
        return False
    res = payload.get("result")
    if not isinstance(res, dict) or res.get("model") != expected_model:
        return False
    modes = res.get("modes") or {}
    for mode_key in ("pure", "laya_uncalibrated", "laya_calibrated"):
        m = modes.get(mode_key)
        if not isinstance(m, dict) or "scores" not in m or "metrics" not in m:
            return False
        if "tensor_trust_average" not in m["metrics"]:
            return False
    return True


def _compare_with_corrections_ref(
    pure_scores: Dict[str, float],
    ref_scores: Optional[Dict[str, float]],
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    if not ref_scores:
        return None, None
    diffs: Dict[str, float] = {}
    for ds_name in PUBLIC_SETS:
        if ds_name in pure_scores and ds_name in ref_scores:
            diffs[ds_name] = float(pure_scores[ds_name]) - float(ref_scores[ds_name])
    max_abs = max((abs(v) for v in diffs.values()), default=0.0) if diffs else None
    return diffs, max_abs


def _build_summary_rows(
    results: Sequence[Dict[str, Any]],
    spec_by_model: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_rows, _ = _rows_from_evaluation_results(results)
    rows_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["model"], r["mode"]): r for r in base_rows
    }

    pure_metrics_by_model: Dict[str, Dict[str, float]] = {
        res["model"]: res["modes"]["pure"]["metrics"] for res in results
    }
    refs = {
        "hf": pure_metrics_by_model.get("PIGuard_HF"),
        "mof_orig": pure_metrics_by_model.get("augmented_mof_original"),
        "aug_corr": pure_metrics_by_model.get("augmented_corrected"),
    }

    enriched: List[Dict[str, Any]] = []
    for res in results:
        model_name = res["model"]
        spec = spec_by_model[model_name]
        cal = res["calibration"]
        for mode_key in ("pure", "laya_uncalibrated", "laya_calibrated"):
            row = dict(rows_by_key[(model_name, mode_key)])
            mode_obj = res["modes"][mode_key]
            lat_all = mode_obj.get("latency_all") or {}
            boot = mode_obj.get("bootstrap_vs_pure") or {}
            row["display_name"] = f"{spec['display_name']} [{row['mode_label']}]" if mode_key != "pure" else spec["display_name"]
            row["group"] = spec["group"]
            row["source_tag"] = spec["source_tag"]
            row["mof_note"] = spec["mof_note"]
            row["ci95_overdef_low"] = (boot.get("overdefense") or {}).get("ci95_low")
            row["ci95_overdef_high"] = (boot.get("overdefense") or {}).get("ci95_high")
            row["ci95_malicious_low"] = (boot.get("malicious") or {}).get("ci95_low")
            row["ci95_malicious_high"] = (boot.get("malicious") or {}).get("ci95_high")
            row["ci95_tt_low"] = (boot.get("tensor_trust_average") or {}).get("ci95_low")
            row["ci95_tt_high"] = (boot.get("tensor_trust_average") or {}).get("ci95_high")
            row["call_rate_all_pct"] = lat_all.get("call_rate_pct")
            row["inference_all_ms"] = lat_all.get("inference_ms")
            row["heldout_delta_fpr_pp"] = (
                float(cal["delta_fpr"]) * 100.0 if mode_key == "laya_calibrated" else None
            )
            row["corrections_ref_max_abs_diff"] = (
                res.get("corrections_ref_max_abs_diff") if mode_key == "pure" else None
            )

            for ref_tag, ref_metrics in refs.items():
                if not ref_metrics:
                    continue
                for metric in ("overdefense", "benign", "malicious", "average", "tensor_trust_average"):
                    if metric in row and row[metric] is not None and metric in ref_metrics:
                        row[f"delta_{ref_tag}_{metric}_pp"] = float(row[metric]) - float(ref_metrics[metric])
            enriched.append(row)
    return enriched


def _plot_corrected_evolution_bars(
    results: Sequence[Dict[str, Any]],
    spec_by_model: Dict[str, Dict[str, Any]],
    dest: Path,
) -> Optional[Path]:
    if not results:
        return None
    metrics = ("overdefense", "benign", "malicious", "average", "tensor_trust_average")
    labels = ("Over-defense", "Benign (WG)", "Malicious (BIPIA)", "Average (Macro)", "TensorTrust")
    colors = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3")

    x = np.arange(len(results))
    width = 0.16
    x_labels = [spec_by_model[r["model"]]["short_name"] for r in results]

    fig, ax = plt.subplots(figsize=(max(11.5, len(results) * 2.2), 5.8), layout="constrained")
    for j, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [float(r["modes"]["pure"]["metrics"][m]) for r in results]
        bars = ax.bar(x + (j - 2.0) * width, vals, width, label=lbl, color=col)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=7.3,
                xytext=(0, 2),
                textcoords="offset points",
            )

    n_corr = sum(1 for r in results if spec_by_model[r["model"]]["group"] == "corrected")
    if 0 < n_corr < len(results):
        ax.axvline(n_corr - 0.5, color="#555555", linestyle="--", linewidth=1.1)

    ax.set_xticks(x, x_labels, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Linhagem `corrected` vs. Âncoras — Avaliação Unificada (Modo Puro)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_corrected_laya_effect(
    results: Sequence[Dict[str, Any]],
    spec_by_model: Dict[str, Dict[str, Any]],
    dest: Path,
) -> Optional[Path]:
    if not results:
        return None
    metrics = ("overdefense", "benign", "malicious", "average")
    titles = ("Over-defense (NotInject)", "Benign (WildGuard)", "Malicious (BIPIA)", "Average (Macro)")
    modes = (
        ("pure", "PIGuard Puro", "#4C72B0"),
        ("laya_uncalibrated", "+ Laya Não Cal.", "#DD8452"),
        ("laya_calibrated", "+ Laya Cal.", "#55A868"),
    )

    x = np.arange(len(results))
    width = 0.25
    x_labels = [spec_by_model[r["model"]]["short_name"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2), layout="constrained")
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for j, (mkey, mlabel, color) in enumerate(modes):
            vals = [float(r["modes"][mkey]["metrics"][metric]) for r in results]
            bars = ax.bar(x + (j - 1) * width, vals, width, label=mlabel, color=color)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    (bar.get_x() + bar.get_width() / 2, h),
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        ax.set_xticks(x, x_labels, fontsize=8.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Acurácia (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)

    fig.suptitle("Linhagem `corrected` + Âncoras — Puro vs. Laya English (Não Calibrado e Calibrado)", fontsize=12.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_corrected_bootstrap_forest(
    results: Sequence[Dict[str, Any]],
    spec_by_model: Dict[str, Dict[str, Any]],
    dest: Path,
) -> Optional[Path]:
    if not results:
        return None
    metrics = ("overdefense", "malicious", "average", "tensor_trust_average")
    titles = ("Δ Over-defense (pp)", "Δ Malicious (pp)", "Δ Average (pp)", "Δ TensorTrust (pp)")

    fig, axes = plt.subplots(
        1, 4, figsize=(18, max(4.6, len(results) * 0.95)), sharey=True, layout="constrained"
    )
    y_pos = np.arange(len(results))

    for ax, metric, title in zip(axes, metrics, titles):
        for offset, mkey, label, color in (
            (-0.15, "laya_uncalibrated", "Laya Não Calibrado", "#DD8452"),
            (0.15, "laya_calibrated", "Laya Calibrado", "#55A868"),
        ):
            deltas, err_lo, err_hi = [], [], []
            for r in results:
                b = r["modes"][mkey]["bootstrap_vs_pure"][metric]
                d = float(b["delta_pp"])
                deltas.append(d)
                err_lo.append(max(0.0, d - float(b["ci95_low"])))
                err_hi.append(max(0.0, float(b["ci95_high"]) - d))
            ax.errorbar(
                deltas,
                y_pos + offset,
                xerr=[err_lo, err_hi],
                fmt="o",
                color=color,
                label=label,
                capsize=4,
                linewidth=1.5,
            )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.legend(fontsize=7.8, loc="best")

    axes[0].set_yticks(y_pos, [spec_by_model[r["model"]]["display_name"] for r in results], fontsize=8.5)
    fig.suptitle("Paired Bootstrap (B=2.000, IC 95%) — Impacto do Laya English na Linhagem `corrected`", fontsize=12.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_corrected_latency_tradeoff(
    summary_rows: Sequence[Dict[str, Any]],
    dest: Path,
) -> Optional[Path]:
    valid = [r for r in summary_rows if isinstance(r.get("inference_ms"), (int, float)) and r["inference_ms"] > 0]
    if not valid:
        return None

    fig, ax = plt.subplots(figsize=(12.2, 6.8), layout="constrained")
    mode_colors = {"pure": "#4C72B0", "laya_uncalibrated": "#DD8452", "laya_calibrated": "#55A868"}
    mode_labels = {"pure": "PIGuard Puro", "laya_uncalibrated": "+ Laya Não Cal.", "laya_calibrated": "+ Laya Cal."}
    seen_modes = set()

    for r in valid:
        mk = r["mode"]
        lbl = mode_labels[mk] if mk not in seen_modes else None
        seen_modes.add(mk)
        ax.scatter(
            r["inference_ms"],
            r["average"],
            color=mode_colors.get(mk, "#7F7F7F"),
            s=95,
            alpha=0.90,
            edgecolors="black",
            linewidths=0.6,
            label=lbl,
            zorder=3,
        )

    # Anotação escalonada por cluster de latência para eliminar sobreposição de rótulos
    xs = [float(r["inference_ms"]) for r in valid]
    ys = [float(r["average"]) for r in valid]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(1.0, x_max - x_min)
    y_span = max(2.0, y_max - y_min)

    low_lat = sorted([r for r in valid if float(r["inference_ms"]) < (x_min + 0.45 * x_span)], key=lambda r: float(r["average"]))
    high_lat = sorted([r for r in valid if float(r["inference_ms"]) >= (x_min + 0.45 * x_span)], key=lambda r: float(r["average"]))

    def _place_labels(cluster: List[Dict[str, Any]], target_x: float, ha: str) -> None:
        if not cluster:
            return
        slots = np.linspace(y_min - 0.05 * y_span, y_max + 0.06 * y_span, num=max(2, len(cluster)))
        if len(cluster) == 1:
            slots = [float(cluster[0]["average"])]
        for row, ty in zip(cluster, slots):
            call_txt = f" ({row['call_rate_pct']:.0f}%)" if row.get("call_rate_pct", 0.0) > 0 else ""
            ax.annotate(
                f"{row['display_name']}{call_txt}",
                xy=(float(row["inference_ms"]), float(row["average"])),
                xytext=(target_x, float(ty)),
                ha=ha,
                va="center",
                fontsize=7.6,
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.7, "shrinkA": 2, "shrinkB": 3},
                zorder=4,
            )

    _place_labels(low_lat, x_min + 0.14 * x_span, "left")
    _place_labels(high_lat, x_max + 0.07 * x_span, "left")

    ax.set_xlim(x_min - 0.12 * x_span, x_max + 0.88 * x_span)
    ax.set_ylim(y_min - 0.14 * y_span, y_max + 0.14 * y_span)
    ax.set_xlabel("Latência Média Pública por Amostra (ms)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Linhagem `corrected` + Âncoras — Trade-off Latência × Acurácia")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower left", fontsize=8.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _render_markdown_report(
    summary_rows: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    spec_by_model: Dict[str, Dict[str, Any]],
) -> str:
    lines = [
        "# Comparação Focada na Linhagem `corrected` (Protocolo Unificado + Laya English)",
        "",
        "## 1. Visão Consolidada (Puro, + Laya Não Calibrado e + Laya Calibrado)",
        "",
        "| Modelo / Modo | Over-def (%) | Benign (%) | Malicious (%) | Average (%) | TensorTrust (%) | Δ vs Puro (pp) | IC 95% Avg | Δ vs `aug_corrected` (pp) | Δ vs `PIGuard_HF` (pp) | Call Rate (%) | Latência (ms) | p95 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_rows:
        ci_txt = (
            f"[{r['ci95_average_low']:+.2f}, {r['ci95_average_high']:+.2f}]"
            if r.get("ci95_average_low") is not None
            else "-"
        )
        d_corr = f"{r['delta_aug_corr_average_pp']:+.2f}" if r.get("delta_aug_corr_average_pp") is not None else "-"
        d_hf = f"{r['delta_hf_average_pp']:+.2f}" if r.get("delta_hf_average_pp") is not None else "-"
        tt_txt = f"{r['tensor_trust_average']:.2f}" if r.get("tensor_trust_average") is not None else "-"
        lat_txt = f"{r['inference_ms']:.2f}" if r.get("inference_ms") is not None else "-"
        p95_txt = f"{r['p95_ms']:.2f}" if r.get("p95_ms") is not None else "-"
        lines.append(
            f"| {r['display_name']} | {r['overdefense']:.2f} | {r['benign']:.2f} | "
            f"{r['malicious']:.2f} | {r['average']:.2f} | {tt_txt} | {r['delta_average_pp']:+.2f} | "
            f"{ci_txt} | {d_corr} | {d_hf} | {r['call_rate_pct']:.1f}% | {lat_txt} | {p95_txt} |"
        )

    lines.extend([
        "",
        "## 2. Parâmetros de Calibração (Laya English em `calibration_heldout.json`) e Verificação de Referência",
        "",
        "| Modelo | T* | T_L* | NLL (T=1 -> T*) | Held-out Base FPR (%) | Held-out ΔFPR (pp) | Held-out Mal. TPR (%) | Max Diff vs `evaluation_raw.json` (pp) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for res in results:
        cal = res["calibration"]
        diff_val = res.get("corrections_ref_max_abs_diff")
        diff_txt = f"{diff_val:.6f}" if diff_val is not None else "N/A (modelo novo)"
        lines.append(
            f"| {spec_by_model[res['model']]['display_name']} | {cal['temperature_star']:.4f} | "
            f"{cal['threshold_tl']:.3f} | {cal['nll_uncalibrated_t1']:.4f} -> {cal['nll_calibrated_tstar']:.4f} | "
            f"{cal['base_fpr']*100:.2f} | {cal['delta_fpr']*100:+.2f} | "
            f"{(cal.get('heldout_malicious_tpr') or 0.0)*100:.2f} | {diff_txt} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orquestrador Principal
# ---------------------------------------------------------------------------

def run_corrected_comparison(
    *,
    corrections_dir: Path | str,
    heldout_path: Path | str,
    output_dir: Path | str,
    repo_root: Path | str = REPO_ROOT,
    specs: Sequence[Dict[str, Any]] = CORRECTED_COMPARISON_SPECS,
    laya_engine: Optional[LayaEnglishEngine] = None,
    fpr_budget: float = 0.01,
    n_boot: int = 2000,
    seed: int = 42,
    batch_size: int = 32,
    laya_batch_size: int = 16,
    force: bool = False,
    force_reval_new_mof: bool = False,
    predict_fn_factory: Optional[Callable[[Dict[str, Any], Optional[Path]], Callable[[Sequence[str]], Tuple[List[int], List[float]]]]] = None,
) -> Dict[str, Any]:
    """Executa a comparação unificada focada nos modelos da linhagem `corrected` e âncoras."""
    repo = Path(repo_root).expanduser().resolve()
    corr_dir = Path(corrections_dir).expanduser()
    if not corr_dir.is_absolute():
        corr_dir = (repo / corr_dir).resolve()
    heldout_path = Path(heldout_path).expanduser()
    if not heldout_path.is_absolute():
        heldout_path = (repo / heldout_path).resolve()
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    comp_dir = out / "corrected_only"
    charts_dir = comp_dir / "charts"
    cache_dir = out / "cache"
    for d in (comp_dir, charts_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    result_json_path = comp_dir / "corrected_comparison_results.json"
    spec_by_model = {s["model"]: dict(s) for s in specs}

    if not force and not force_reval_new_mof and result_json_path.is_file():
        cached_bundle = read_json(result_json_path)
        expected_models = [s["model"] for s in specs]
        existing_models = [r["model"] for r in (cached_bundle.get("results") or [])]
        if existing_models == expected_models:
            print(f"[Corrected Comparison] Reutilizando resultados salvos em {result_json_path}", flush=True)
            return cached_bundle

    benchmarks = load_benchmarks(repo / "datasets", include_tensortrust=True)
    heldout_records = read_json(heldout_path)
    corr_refs = _load_corrections_eval_raw(corr_dir)

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

    topic2_path = out / "topic2" / "topic2_results.json"
    topic2_payload = read_json(topic2_path) if topic2_path.is_file() else None

    results: List[Dict[str, Any]] = []
    for spec in specs:
        model_name = spec["model"]
        hf_model_id = spec.get("hf_model_id")
        ref_key = spec.get("ref_key")
        ref_scores = corr_refs.get(ref_key) if ref_key else None

        if (
            spec.get("reuse_topic2")
            and not force
            and not force_reval_new_mof
            and _is_valid_topic2_result(topic2_payload, model_name)
        ):
            print(f"[Corrected Comparison] Reutilizando {model_name} de {topic2_path}", flush=True)
            res = dict(topic2_payload["result"])
        else:
            ckpt: Optional[Path] = None
            if spec["ckpt_root"] == "corrections":
                ckpt = corr_dir / spec["ckpt_rel"]
            elif spec["ckpt_root"] == "upgrade":
                ckpt = out / spec["ckpt_rel"]

            if ckpt is not None and not ckpt.is_file() and predict_fn_factory is None:
                raise FileNotFoundError(f"Checkpoint ausente para '{model_name}': {ckpt}")

            p_fn = predict_fn_factory(spec, ckpt) if predict_fn_factory is not None else None
            print(f"\n[Corrected Comparison] Avaliando: {spec['display_name']}", flush=True)
            res = evaluate_piguard_with_laya(
                model_name=model_name,
                checkpoint_path=ckpt,
                hf_model_id=hf_model_id,
                benchmarks=benchmarks,
                heldout_records=heldout_records,
                benchmark_laya_cache=bench_laya_cache,
                heldout_laya_cache=heldout_laya_cache,
                drive_cache_reference={"scores": ref_scores} if ref_scores else None,
                fpr_budget=fpr_budget,
                n_boot=n_boot,
                seed=seed,
                batch_size=batch_size,
                predict_fn=p_fn,
            )

        diffs, max_diff = _compare_with_corrections_ref(
            res["modes"]["pure"]["scores"], ref_scores
        )
        res["corrections_ref_diffs"] = diffs
        res["corrections_ref_max_abs_diff"] = max_diff
        results.append(res)

    summary_rows = _build_summary_rows(results, spec_by_model)
    _, bootstrap_rows = _rows_from_evaluation_results(results)

    write_csv(comp_dir / "corrected_comparison_summary.csv", summary_rows)
    write_csv(comp_dir / "corrected_bootstrap_ci.csv", bootstrap_rows)

    chart_paths: Dict[str, str] = {}
    p1 = _plot_corrected_evolution_bars(results, spec_by_model, charts_dir / "corrected_evolution_bars.png")
    if p1:
        chart_paths["evolution_bars"] = str(p1)
    p2 = _plot_corrected_laya_effect(results, spec_by_model, charts_dir / "corrected_laya_effect_bars.png")
    if p2:
        chart_paths["laya_effect_bars"] = str(p2)
    p3 = _plot_corrected_bootstrap_forest(results, spec_by_model, charts_dir / "corrected_bootstrap_forest.png")
    if p3:
        chart_paths["bootstrap_forest"] = str(p3)
    p4 = _plot_corrected_latency_tradeoff(summary_rows, charts_dir / "corrected_latency_tradeoff.png")
    if p4:
        chart_paths["latency_tradeoff"] = str(p4)

    md_text = _render_markdown_report(summary_rows, results, spec_by_model)
    md_path = comp_dir / "RELATORIO_CORRECTED_ONLY.md"
    md_path.write_text(md_text, encoding="utf-8")

    bundle = {
        "status": "complete",
        "fpr_budget": fpr_budget,
        "n_boot": n_boot,
        "seed": seed,
        "models": [s["model"] for s in specs],
        "results": results,
        "summary_rows": summary_rows,
        "charts": chart_paths,
        "markdown_report": str(md_path),
    }
    write_json(result_json_path, bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corrections-dir",
        default=None,
        help="Diretório de artefatos do Corrections (padrão: Corrections/outputs).",
    )
    parser.add_argument(
        "--heldout",
        default=None,
        help="Caminho para calibration_heldout.json (padrão: <output>/step0/...).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório raiz dos artefatos Upgrade (padrão: PIGUARD_UPGRADE_OUTPUT ou <repo>/Upgrade/outputs).",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--fpr-budget", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-reval-new-mof", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    upgrade_default = resolve_path(
        os.environ.get("PIGUARD_UPGRADE_OUTPUT"),
        repo_root / "Upgrade" / "outputs",
        repo_root=repo_root,
    )
    output_dir = resolve_path(args.output_dir, upgrade_default, repo_root=repo_root)
    corrections_dir = resolve_path(
        args.corrections_dir,
        default_corrections_dir(repo_root),
        repo_root=repo_root,
    )
    heldout_path = resolve_path(
        args.heldout,
        output_dir / "step0" / "calibration_heldout.json",
        repo_root=repo_root,
    )

    bundle = run_corrected_comparison(
        corrections_dir=corrections_dir,
        heldout_path=heldout_path,
        output_dir=output_dir,
        repo_root=repo_root,
        fpr_budget=args.fpr_budget,
        n_boot=args.n_boot,
        seed=args.seed,
        force=args.force,
        force_reval_new_mof=args.force_reval_new_mof,
    )
    print(f"Comparação `corrected` concluída. Relatório salvo em {bundle['markdown_report']}")


if __name__ == "__main__":
    main()

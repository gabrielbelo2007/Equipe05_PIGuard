"""Tópico 3 do Pipeline Upgrade: Consolidação Geral de Resultados, Tabelas, ICs e Gráficos.

Unifica em um único relatório:
1. Tópico 1: Os 5 modelos `rp` de RESULTADOS_SEED42_MAX512 (Puro vs + Laya Não Calibrado vs + Laya Calibrado);
2. Tópico 2: Evolução do MOF na base corrigida (`augmented_corrected` -> `augmented_mof_corrected` ->
   `train_augmented_corrected_new_mof` Puro e + Laya English);
3. Tópico 3: Tabela Mestra e Gráficos Comparativos frente às referências oficiais (`PIGuard HF` e
   `train_augmented_mof_original`), incluindo IC 95% via Paired Bootstrap e análise de latência.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import (
    aggregate_benchmark_scores,
    default_corrections_dir,
    read_json,
    resolve_path,
    write_csv,
    write_json,
)

DEFAULT_UPGRADE_OUTPUT = REPO_ROOT / "Upgrade" / "outputs"


def _load_rp_cache_reference(drive_rp_root: Optional[Path | str], name: str) -> Optional[Dict[str, Any]]:
    if drive_rp_root is None:
        return None
    path = Path(drive_rp_root) / "Gráficos" / "_cache" / f"{name}.json"
    if not path.is_file():
        return None
    data = read_json(path)
    if not isinstance(data, dict) or "scores" not in data:
        return None
    scores = data["scores"]
    metrics = data.get("metrics") or aggregate_benchmark_scores(scores)
    return {
        "model": name,
        "source": "RESULTADOS_SEED42_MAX512",
        "scores": scores,
        "metrics": metrics,
        "inference_ms": data.get("inference_ms") or metrics.get("inference_ms"),
        "efficiency": metrics.get("efficiency"),
    }


def _load_corrections_models(corrections_dir: Optional[Path | str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if corrections_dir is None:
        return out
    cdir = Path(corrections_dir)

    raw_path = cdir / "evaluation_raw.json"
    if raw_path.is_file():
        raw = read_json(raw_path)
        for name, info in (raw.get("models") or {}).items():
            scores = info.get("scores") or {}
            if all(
                k in scores
                for k in ("NotInject_one", "NotInject_two", "NotInject_three", "WildGuard", "BIPIA_text", "BIPIA_code")
            ):
                metrics = aggregate_benchmark_scores(scores)
                out[name] = {
                    "model": name,
                    "source": "Corrections",
                    "scores": scores,
                    "metrics": metrics,
                    "inference_ms": None,
                    "efficiency": None,
                }
        return out

    csv_path = cdir / "comparison.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = row.get("model")
                if not name:
                    continue
                scores = {
                    k: float(row[k])
                    for k in ("NotInject_one", "NotInject_two", "NotInject_three", "WildGuard", "BIPIA_text", "BIPIA_code")
                    if row.get(k) not in (None, "")
                }
                if len(scores) == 6:
                    metrics = aggregate_benchmark_scores(scores)
                    out[name] = {
                        "model": name,
                        "source": "Corrections",
                        "scores": scores,
                        "metrics": metrics,
                        "inference_ms": None,
                        "efficiency": None,
                    }
    return out


def _flatten_eval_mode_row(
    res: Dict[str, Any],
    mode_key: str,
    *,
    display_name: str,
    source_tag: str,
    mof_note: str = "",
) -> Dict[str, Any]:
    m = res["modes"][mode_key]
    sc = m["scores"]
    met = m["metrics"]
    lat = m["latency_public"]
    pure_avg = res["modes"]["pure"]["metrics"]["average"]
    boot = m.get("bootstrap_vs_pure") or {}
    avg_boot = boot.get("average") or {}

    return {
        "display_name": display_name,
        "model": res["model"],
        "mode": mode_key,
        "source": source_tag,
        "mof_note": mof_note,
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
        "delta_vs_pure_pp": met["average"] - pure_avg,
        "ci95_low": avg_boot.get("ci95_low"),
        "ci95_high": avg_boot.get("ci95_high"),
        "tensor_trust_hijacking": met.get("tensor_trust_hijacking"),
        "tensor_trust_extraction": met.get("tensor_trust_extraction"),
        "tensor_trust_average": met.get("tensor_trust_average"),
        "call_rate_pct": lat["call_rate_pct"],
        "inference_ms": lat["inference_ms"],
        "p50_ms": lat["p50_ms"],
        "p95_ms": lat["p95_ms"],
        "efficiency": lat["efficiency"],
    }


def _flatten_static_row(
    entry: Dict[str, Any],
    *,
    display_name: str,
    mof_note: str = "",
) -> Dict[str, Any]:
    sc = entry["scores"]
    met = entry["metrics"]
    return {
        "display_name": display_name,
        "model": entry["model"],
        "mode": "pure",
        "source": entry["source"],
        "mof_note": mof_note,
        "temperature": None,
        "threshold_tl": None,
        "NotInject_one": sc.get("NotInject_one"),
        "NotInject_two": sc.get("NotInject_two"),
        "NotInject_three": sc.get("NotInject_three"),
        "overdefense": met["overdefense"],
        "benign": met["benign"],
        "BIPIA_text": sc.get("BIPIA_text"),
        "BIPIA_code": sc.get("BIPIA_code"),
        "malicious": met["malicious"],
        "average": met["average"],
        "delta_vs_pure_pp": 0.0,
        "ci95_low": None,
        "ci95_high": None,
        "tensor_trust_hijacking": sc.get("tensor_trust_hijacking"),
        "tensor_trust_extraction": sc.get("tensor_trust_extraction"),
        "tensor_trust_average": (
            (sc["tensor_trust_hijacking"] + sc["tensor_trust_extraction"]) / 2.0
            if "tensor_trust_hijacking" in sc and "tensor_trust_extraction" in sc
            else None
        ),
        "call_rate_pct": 0.0,
        "inference_ms": entry.get("inference_ms"),
        "p50_ms": None,
        "p95_ms": None,
        "efficiency": entry.get("efficiency"),
    }


def _add_reference_deltas(
    rows: List[Dict[str, Any]],
    refs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for ref_tag, ref_row in refs.items():
            if ref_row is None:
                continue
            for metric in ("overdefense", "benign", "malicious", "average"):
                if isinstance(item.get(metric), (int, float)) and isinstance(ref_row.get(metric), (int, float)):
                    item[f"delta_{ref_tag}_{metric}_pp"] = float(item[metric]) - float(ref_row[metric])
        enriched.append(item)
    return enriched


# ---------------------------------------------------------------------------
# Geração de Gráficos
# ---------------------------------------------------------------------------

def _plot_topic1_rp_bars(topic1_results: Sequence[Dict[str, Any]], dest: Path) -> Optional[Path]:
    if not topic1_results:
        return None
    models = [r["model"] for r in topic1_results]
    metrics = ("overdefense", "benign", "malicious", "average")
    titles = ("Over-defense (NotInject)", "Benign (WildGuard)", "Malicious (BIPIA)", "Average (Macro)")
    modes = (
        ("pure", "PIGuard Puro", "#4C72B0"),
        ("laya_uncalibrated", "+ Laya Não Calibrado", "#DD8452"),
        ("laya_calibrated", "+ Laya Calibrado", "#55A868"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    x = np.arange(len(models))
    width = 0.25

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for j, (mkey, label, color) in enumerate(modes):
            vals = [r["modes"][mkey]["metrics"][metric] for r in topic1_results]
            ax.bar(x + (j - 1) * width, vals, width, label=label, color=color)
        ax.set_xticks(x, [m.replace("train_", "") for m in models], rotation=22, ha="right", fontsize=8.5)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Acurácia (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)

    fig.suptitle("Tópico 1 — Modelos `rp` (RESULTADOS_SEED42_MAX512): Puro vs Laya English", fontsize=12.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_topic2_evolution(rows: Sequence[Dict[str, Any]], dest: Path) -> Optional[Path]:
    if not rows:
        return None
    metrics = ("overdefense", "benign", "malicious", "average")
    labels = [r["display_name"] for r in rows]
    x = np.arange(len(rows))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.9), 5.4), layout="constrained")
    for j, m in enumerate(metrics):
        vals = [float(r[m]) for r in rows]
        bars = ax.bar(x + (j - 1.5) * width, vals, width, label=m)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7.5, xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Tópico 2 — Evolução do MOF na Base Corrigida + Laya English")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_bootstrap_forest(eval_results: Sequence[Dict[str, Any]], dest: Path) -> Optional[Path]:
    if not eval_results:
        return None
    metrics = ("overdefense", "malicious", "average")
    titles = ("Δ Over-defense (pp)", "Δ Malicious (pp)", "Δ Average (pp)")

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, len(eval_results) * 0.9)), sharey=True, layout="constrained")
    y_pos = np.arange(len(eval_results))

    for ax, metric, title in zip(axes, metrics, titles):
        for offset, mkey, label, color in (
            (-0.14, "laya_uncalibrated", "Laya Não Calibrado", "#DD8452"),
            (0.14, "laya_calibrated", "Laya Calibrado", "#55A868"),
        ):
            deltas = []
            err_lo = []
            err_hi = []
            for r in eval_results:
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
        ax.legend(fontsize=8, loc="best")

    axes[0].set_yticks(y_pos, [r["model"] for r in eval_results], fontsize=8.5)
    fig.suptitle("Paired Bootstrap (B=2.000, IC 95%) — Ganho sobre PIGuard Puro", fontsize=12.5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_latency_tradeoff(rows: Sequence[Dict[str, Any]], dest: Path) -> Optional[Path]:
    valid = [r for r in rows if isinstance(r.get("inference_ms"), (int, float)) and r["inference_ms"] > 0]
    if not valid:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    mode_colors = {"pure": "#4C72B0", "laya_uncalibrated": "#DD8452", "laya_calibrated": "#55A868"}

    for r in valid:
        c = mode_colors.get(r["mode"], "#7F7F7F")
        ax.scatter(r["inference_ms"], r["average"], color=c, s=90, alpha=0.88, edgecolors="black", linewidths=0.6)
        call_txt = f" ({r['call_rate_pct']:.0f}% Laya)" if r.get("call_rate_pct", 0.0) > 0 else ""
        ax.annotate(
            f"{r['display_name']}{call_txt}",
            (r["inference_ms"], r["average"]),
            fontsize=7.5,
            xytext=(5, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Latência Média por Amostra (ms)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Trade-off Acurácia vs Latência e Taxa de Acionamento do Laya English")
    ax.grid(True, alpha=0.25, linestyle="--")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def _plot_master_comparison(master_rows: Sequence[Dict[str, Any]], dest: Path) -> Optional[Path]:
    if not master_rows:
        return None
    metrics = ("overdefense", "benign", "malicious", "average")
    labels = [r["display_name"] for r in master_rows]
    x = np.arange(len(master_rows))
    width = 0.19

    fig, ax = plt.subplots(figsize=(max(11, len(master_rows) * 1.7), 5.8), layout="constrained")
    for j, m in enumerate(metrics):
        vals = [float(r[m]) for r in master_rows]
        ax.bar(x + (j - 1.5) * width, vals, width, label=m)

    ax.set_xticks(x, labels, rotation=22, ha="right", fontsize=8.8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Acurácia (%)")
    ax.set_title("Tópico 3 — Consolidação Geral (Baselines, RP + Laya e Novo MOF Corrigido)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


# ---------------------------------------------------------------------------
# Orquestrador do Tópico 3
# ---------------------------------------------------------------------------

def run_topic3(
    *,
    output_dir: Path | str,
    drive_rp_root: Optional[Path | str] = None,
    corrections_dir: Optional[Path | str] = None,
    repo_root: Path | str = REPO_ROOT,
) -> Dict[str, Any]:
    """Consolida resultados dos Tópicos 1 e 2 com as referências oficiais e do Corrections."""
    repo = Path(repo_root).expanduser().resolve()
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = (repo / out).resolve()
    if drive_rp_root is not None:
        drive_rp_root = Path(drive_rp_root).expanduser()
        if not drive_rp_root.is_absolute():
            drive_rp_root = (repo / drive_rp_root).resolve()
    if corrections_dir is not None:
        corrections_dir = Path(corrections_dir).expanduser()
        if not corrections_dir.is_absolute():
            corrections_dir = (repo / corrections_dir).resolve()
    topic1_path = out / "topic1" / "topic1_rp_laya_results.json"
    topic2_path = out / "topic2" / "topic2_results.json"
    topic3_dir = out / "topic3"
    charts_dir = topic3_dir / "charts"
    topic3_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    topic1_data = read_json(topic1_path) if topic1_path.is_file() else {"results": []}
    topic2_data = read_json(topic2_path) if topic2_path.is_file() else {}
    topic1_results: List[Dict[str, Any]] = list(topic1_data.get("results") or [])
    topic2_result: Optional[Dict[str, Any]] = topic2_data.get("result")

    # Carrega referências históricas (sem reexecutar modelos)
    hf_ref = _load_rp_cache_reference(drive_rp_root, "PIGuard HF")
    mof_orig_ref = _load_rp_cache_reference(drive_rp_root, "train_augmented_mof_original")
    corr_models = _load_corrections_models(corrections_dir)

    refs_for_deltas = {
        "hf": hf_ref["metrics"] if hf_ref else None,
        "mof_orig": mof_orig_ref["metrics"] if mof_orig_ref else None,
    }

    # 1. Tabela Tópico 1 (5 modelos `rp` x 3 modos)
    t1_rows: List[Dict[str, Any]] = []
    for res in topic1_results:
        for mode_key, suffix in (
            ("pure", "Puro"),
            ("laya_uncalibrated", "+ Laya Não Cal."),
            ("laya_calibrated", "+ Laya Cal."),
        ):
            t1_rows.append(
                _flatten_eval_mode_row(
                    res,
                    mode_key,
                    display_name=f"{res['model']} ({suffix})",
                    source_tag="Tópico 1 (RESULTADOS_SEED42_MAX512)",
                )
            )
    t1_rows = _add_reference_deltas(t1_rows, refs_for_deltas)
    write_csv(topic3_dir / "table_topic1_rp_laya_comparison.csv", t1_rows)

    # 2. Tabela Tópico 2 (Evolução do MOF na Base Corrigida)
    t2_rows: List[Dict[str, Any]] = []
    if "augmented_corrected" in corr_models:
        t2_rows.append(
            _flatten_static_row(
                corr_models["augmented_corrected"],
                display_name="augmented_corrected (Sem MOF)",
                mof_note="Sem MOF (75.973 exemplos)",
            )
        )
    if "augmented_mof_corrected" in corr_models:
        t2_rows.append(
            _flatten_static_row(
                corr_models["augmented_mof_corrected"],
                display_name="augmented_mof_corrected (MOF Antigo)",
                mof_note="MOF antigo (337/237/188 gatilhos)",
            )
        )
    if topic2_result is not None:
        for mode_key, suffix in (
            ("pure", "Puro"),
            ("laya_uncalibrated", "+ Laya Não Cal."),
            ("laya_calibrated", "+ Laya Cal."),
        ):
            t2_rows.append(
                _flatten_eval_mode_row(
                    topic2_result,
                    mode_key,
                    display_name=f"corrected_new_mof ({suffix})",
                    source_tag="Tópico 2 (Upgrade)",
                    mof_note="MOF v2 (152/229/381 + 4 domínios + HN 80/20)",
                )
            )
    t2_rows = _add_reference_deltas(t2_rows, refs_for_deltas)
    write_csv(topic3_dir / "table_topic2_mof_evolution_corrected.csv", t2_rows)

    # 3. Tabela Mestra Tópico 3
    master_rows: List[Dict[str, Any]] = []
    if hf_ref is not None:
        master_rows.append(_flatten_static_row(hf_ref, display_name="PIGuard HF (Oficial)", mof_note="Checkpoint HF"))
    if mof_orig_ref is not None:
        master_rows.append(
            _flatten_static_row(
                mof_orig_ref,
                display_name="train_augmented_mof_original",
                mof_note="MOF original publicado",
            )
        )
    for res in topic1_results:
        if res["model"] == "train_augmented_mof_rp":
            master_rows.append(
                _flatten_eval_mode_row(
                    res,
                    "pure",
                    display_name="train_augmented_mof_rp (Puro)",
                    source_tag="Tópico 1",
                    mof_note="Reprodução rp pura",
                )
            )
            master_rows.append(
                _flatten_eval_mode_row(
                    res,
                    "laya_calibrated",
                    display_name="train_augmented_mof_rp (+ Laya Cal.)",
                    source_tag="Tópico 1",
                    mof_note="Reprodução rp + Laya English Calibrado",
                )
            )
    if "augmented_mof_corrected" in corr_models:
        master_rows.append(
            _flatten_static_row(
                corr_models["augmented_mof_corrected"],
                display_name="augmented_mof_corrected (MOF Antigo)",
                mof_note="Base corrigida + MOF antigo",
            )
        )
    if topic2_result is not None:
        for mode_key, suffix in (
            ("pure", "Puro"),
            ("laya_uncalibrated", "+ Laya Não Cal."),
            ("laya_calibrated", "+ Laya Cal."),
        ):
            master_rows.append(
                _flatten_eval_mode_row(
                    topic2_result,
                    mode_key,
                    display_name=f"corrected_new_mof ({suffix})",
                    source_tag="Tópico 2",
                    mof_note="Base corrigida intacta + Novo MOF",
                )
            )
    master_rows = _add_reference_deltas(master_rows, refs_for_deltas)
    write_csv(topic3_dir / "table_topic3_master_consolidation.csv", master_rows)

    # 4. Gráficos PNG
    all_eval_results = list(topic1_results) + ([topic2_result] if topic2_result else [])
    chart_paths = {}
    p1 = _plot_topic1_rp_bars(topic1_results, charts_dir / "topic1_rp_models_laya_bars.png")
    if p1:
        chart_paths["topic1_rp_bars"] = str(p1)
    p2 = _plot_topic2_evolution(t2_rows, charts_dir / "topic2_mof_evolution_bars.png")
    if p2:
        chart_paths["topic2_evolution"] = str(p2)
    p3 = _plot_master_comparison(master_rows, charts_dir / "topic3_master_comparison.png")
    if p3:
        chart_paths["topic3_master"] = str(p3)
    p4 = _plot_bootstrap_forest(all_eval_results, charts_dir / "topic3_paired_bootstrap_forest.png")
    if p4:
        chart_paths["topic3_bootstrap_forest"] = str(p4)
    p5 = _plot_latency_tradeoff(t1_rows + [r for r in t2_rows if r.get("source") == "Tópico 2 (Upgrade)"], charts_dir / "topic3_latency_accuracy_tradeoff.png")
    if p5:
        chart_paths["topic3_latency_tradeoff"] = str(p5)

    # 5. Relatório Markdown consolidado
    md_lines = [
        "# Relatório Consolidado — Pipeline Upgrade (PIGuard + Novo MOF + Laya English)",
        "",
        "## Resumo Tópico 1: Modelos `rp` (RESULTADOS_SEED42_MAX512) + Laya English",
        "",
        "| Modelo / Modo | Over-def (%) | Benign (%) | Malicious (%) | Average (%) | Δ vs Puro (pp) | IC 95% (Bootstrap) | Call Rate (%) | Latência (ms) | p95 (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in t1_rows:
        ci_txt = (
            f"[{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
            if r.get("ci95_low") is not None
            else "-"
        )
        lat_txt = f"{r['inference_ms']:.2f}" if r.get("inference_ms") is not None else "-"
        p95_txt = f"{r['p95_ms']:.2f}" if r.get("p95_ms") is not None else "-"
        md_lines.append(
            f"| {r['display_name']} | {r['overdefense']:.2f} | {r['benign']:.2f} | "
            f"{r['malicious']:.2f} | {r['average']:.2f} | {r['delta_vs_pure_pp']:+.2f} | "
            f"{ci_txt} | {r['call_rate_pct']:.1f}% | {lat_txt} | {p95_txt} |"
        )

    md_lines.extend(
        [
            "",
            "## Resumo Tópico 2 e 3: Consolidação Geral e Evolução na Base Corrigida",
            "",
            "| Modelo / Configuração | Over-def (%) | Benign (%) | Malicious (%) | Average (%) | Latência (ms) | Nota Metodológica |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for r in master_rows:
        lat_txt = f"{r['inference_ms']:.2f}" if r.get("inference_ms") is not None else "-"
        md_lines.append(
            f"| {r['display_name']} | {r['overdefense']:.2f} | {r['benign']:.2f} | "
            f"{r['malicious']:.2f} | {r['average']:.2f} | {lat_txt} | {r['mof_note']} |"
        )

    report_md = topic3_dir / "RELATORIO_CONSOLIDADO_UPGRADE.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    bundle = {
        "status": "complete",
        "topic1_rows": t1_rows,
        "topic2_rows": t2_rows,
        "master_rows": master_rows,
        "charts": chart_paths,
        "markdown_report": str(report_md),
    }
    write_json(topic3_dir / "topic3_consolidated.json", bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório raiz dos artefatos Upgrade (padrão: <repo>/Upgrade/outputs).",
    )
    parser.add_argument(
        "--drive-rp-root",
        default=None,
        help="Caminho para a pasta com os checkpoints rp (padrão: <repo>/outputs).",
    )
    parser.add_argument(
        "--corrections-dir",
        default=None,
        help="Diretório de artefatos do Corrections (padrão: Corrections/outputs).",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    output_dir = resolve_path(
        args.output_dir, repo_root / "Upgrade" / "outputs", repo_root=repo_root
    )
    drive_rp_root = resolve_path(
        args.drive_rp_root, repo_root / "outputs", repo_root=repo_root
    )
    corrections_dir = resolve_path(
        args.corrections_dir,
        default_corrections_dir(repo_root),
        repo_root=repo_root,
    )

    bundle = run_topic3(
        output_dir=output_dir,
        drive_rp_root=drive_rp_root,
        corrections_dir=corrections_dir,
        repo_root=repo_root,
    )
    print(f"Tópico 3 concluído. Relatório salvo em {bundle['markdown_report']}")


if __name__ == "__main__":
    main()

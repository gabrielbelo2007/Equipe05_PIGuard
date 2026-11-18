"""PIGuard - Validação Comparativa Geral (Tabela 1) + variantes MOF.

Conversão do notebook ``Notebooks/second_train/eval_table1.ipynb`` para script
Python autônomo, sem comandos mágicos, ``display(Image)`` ou Google Colab.

Uso:
    python Notebooks/second_train/eval_table1.py
    python Notebooks/second_train/eval_table1.py --skip-openrouter
    python Notebooks/second_train/eval_table1.py --force-rerun --max-samples 20

A inferência dos modelos é delegada a ``Eval/validate_models_table1.py``. O
script lê os resultados brutos, monta a tabela de deltas (Artigo vs Reproduzido)
e gera os gráficos individuais e o mosaico consolidado.

Chave do OpenRouter: ``--openrouter-key``, ``OPENROUTER_API_KEY``,
``OPENAI_API_KEY`` ou arquivo ``.env`` na raiz do projeto. Sem chave, os LLMs via
OpenRouter são pulados. Resultados são gravados em ``logs/Table1`` e replicados
no diretório de saída.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    plt = None
    np = None
    pd = None
    _PLOT_IMPORT_ERROR: ImportError | None = exc
else:
    _PLOT_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PIGUARD_OUTPUT_DIR", REPO_ROOT / "outputs")
).expanduser()

# Valores oficiais do artigo (Tabela 1).
PAPER_REF = {
    "Fmops": {"overdefense": 28.32, "benign": 34.63, "malicious": 93.50, "average": 52.15},
    "Deepset": {"overdefense": 29.50, "benign": 34.06, "malicious": 91.50, "average": 51.68},
    "PromptGuard": {"overdefense": 0.29, "benign": 26.82, "malicious": 97.10, "average": 41.40},
    "ProtectAIv2": {"overdefense": 57.23, "benign": 86.20, "malicious": 48.60, "average": 64.01},
    "GPT-4o": {"overdefense": 86.73, "benign": 90.78, "malicious": 79.10, "average": 85.53},
    "Llama-2-chat": {"overdefense": 76.40, "benign": 61.03, "malicious": 31.09, "average": 56.17},
    "LlamaGuard3": {"overdefense": 99.71, "benign": 95.18, "malicious": 28.28, "average": 74.39},
    "PIGuard": {"overdefense": 87.32, "benign": 85.74, "malicious": 77.39, "average": 83.48},
}

MODEL_MAPPING = [
    ("Fmops (DistilBERT)", "Fmops (fmops, 2024)", "Fmops"),
    ("Deepset (DeBERTa-v3)", "Deepset (Deepset, 2024b)", "Deepset"),
    ("PromptGuard (Meta 86M)", "PromptGuard (Meta, 2024)", "PromptGuard"),
    ("ProtectAIv2 (DeBERTa-v3)", "ProtectAIv2 (ProtectAI.com, 2024)", "ProtectAIv2"),
    ("GPT-4o (OpenAI)", "GPT-4o (OpenAI, 2024a)", "GPT-4o"),
    ("Llama-2-chat (Touvron et al.)", "Llama-2-chat (Touvron et al., 2023)", "Llama-2-chat"),
    ("LlamaGuard3 (Dubey et al.)", "LlamaGuard3 (Dubey et al., 2024)", "LlamaGuard3"),
    ("PIGuard (Oficial HF - leolee99)", "PIGuard (Oficial HF - leolee99)", "PIGuard"),
    ("PIGuard (train_augmented_mof_original)", "PIGuard (train_augmented_mof_original)", "PIGuard"),
    ("PIGuard (train_augmented_mof_semi_rp)", "PIGuard (train_augmented_mof_semi_rp)", "PIGuard"),
    ("PIGuard (train_augmented_mof_rp)", "PIGuard (train_augmented_mof_rp)", "PIGuard"),
]

SYNC_FILES = [
    "table1_validation_raw.json",
    "table1_validation_summary.csv",
    "table1_validation_report.md",
    "table1_detailed_deltas.csv",
    "table1_comparison.png",
    "table1_all_models_comparison_grid.png",
    "openrouter_eval_cache.json",
]


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def find_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@contextmanager
def subprocess_environment(extra_paths: Sequence[Path] = ()) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    paths = [str(path) for path in extra_paths]
    paths.extend([str(REPO_ROOT), str(REPO_ROOT / "Original"), str(REPO_ROOT / "Eval")])
    if len(REPO_ROOT.parents) >= 2:
        paths.append(str(REPO_ROOT.parents[1]))

    with tempfile.TemporaryDirectory(prefix="piguard_compat_") as tmp:
        try:
            package_dir = Path(tmp) / "envio_gith"
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "Equipe05_PIGuard").symlink_to(
                REPO_ROOT, target_is_directory=True
            )
            paths.insert(0, tmp)
        except OSError:
            pass

        current = env.get("PYTHONPATH")
        if current:
            paths.append(current)
        env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
        yield env


def find_validator() -> Path | None:
    return find_existing(
        [
            REPO_ROOT / "Eval/validate_models_table1.py",
            REPO_ROOT / "validate_models_table1.py",
            REPO_ROOT / "logs/validate_models_table1.py",
        ]
    )


def run_validator(
    validator: Path,
    dataset_root: Path,
    logs_dir: Path,
    *,
    checkpoint_root: Path,
    openrouter_key: str | None,
    skip_openrouter: bool,
    force_rerun: bool,
    batch_size: int,
    max_samples: int | None,
) -> None:
    cmd = [
        sys.executable,
        str(validator),
        "--dataset_root", str(dataset_root),
        "--output_dir", str(logs_dir),
        "--batch_size", str(batch_size),
    ]
    if skip_openrouter:
        cmd.append("--skip_openrouter")
    if force_rerun:
        cmd.append("--force_rerun")
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])

    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    with subprocess_environment([validator.parent]) as env:
        env["PIGUARD_OUTPUT_DIR"] = str(checkpoint_root)
        if openrouter_key:
            env["OPENROUTER_API_KEY"] = openrouter_key
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"validate_models_table1.py terminou com código {result.returncode}."
        )


def sync_results(logs_dir: Path, output_dir: Path) -> None:
    """Publica uma cópia dos relatórios da rodada atual em logs/Table1."""
    for file_name in SYNC_FILES:
        local = logs_dir / file_name
        output = output_dir / file_name
        if output.is_file() and not (local.exists() and local.samefile(output)):
            shutil.copy2(output, local)


def load_raw_results(logs_dir: Path) -> dict[str, dict[str, float]]:
    raw_path = logs_dir / "table1_validation_raw.json"
    if not raw_path.is_file():
        return {}
    try:
        with raw_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[AVISO] Não foi possível ler {raw_path}: {exc}")
        return {}


def load_ablation_results(output_dir: Path) -> dict[str, dict[str, float]]:
    candidates = [output_dir / "ablation_results_raw.json"]
    if output_dir == REPO_ROOT / "outputs":
        candidates.extend([
            REPO_ROOT / "logs/MOFs/ablation_results_raw.json",
            REPO_ROOT / "Eval/MOFs/ablation_results_raw.json",
        ])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("evaluated_experiments", data)
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def load_results_with_ablation_fallback(
    raw_key: str,
    results_raw: dict[str, dict[str, float]],
    ablation_results: dict[str, dict[str, float]],
) -> dict[str, float]:
    data = results_raw.get(raw_key, {})
    if data or "train_augmented_mof" not in raw_key:
        return data

    short_key = raw_key.replace("PIGuard (", "").rstrip(")")
    ablation = ablation_results.get(short_key)
    if not ablation:
        return {}
    return {
        "overdefense": (ablation.get("NotInject_overall_acc") or 0.0) * 100.0,
        "benign": (ablation.get("WildGuard_acc") or 0.0) * 100.0,
        "malicious": (ablation.get("BIPIA_overall_acc") or 0.0) * 100.0,
        "average": (ablation.get("Overall_acc") or 0.0) * 100.0,
    }


def build_detailed_table(
    results_raw: dict[str, dict[str, float]],
    ablation_results: dict[str, dict[str, float]],
) -> "pd.DataFrame":
    rows = []
    for display_name, raw_key, paper_key in MODEL_MAPPING:
        paper = PAPER_REF[paper_key]
        reproduced = load_results_with_ablation_fallback(
            raw_key, results_raw, ablation_results
        )
        values = {
            "overdefense": reproduced.get("overdefense"),
            "benign": reproduced.get("benign"),
            "malicious": reproduced.get("malicious"),
            "average": reproduced.get("average"),
        }
        row = {
            "Modelo": display_name,
            "Over-def (Artigo)": f"{paper['overdefense']:.2f}%",
            "Over-def (Reprod)": f"{values['overdefense']:.2f}%" if values["overdefense"] is not None else "N/D",
            "Δ Over-def": f"{(values['overdefense'] - paper['overdefense']):+.2f}%" if values["overdefense"] is not None else "-",
            "Benign (Artigo)": f"{paper['benign']:.2f}%",
            "Benign (Reprod)": f"{values['benign']:.2f}%" if values["benign"] is not None else "N/D",
            "Δ Benign": f"{(values['benign'] - paper['benign']):+.2f}%" if values["benign"] is not None else "-",
            "Malicious (Artigo)": f"{paper['malicious']:.2f}%",
            "Malicious (Reprod)": f"{values['malicious']:.2f}%" if values["malicious"] is not None else "N/D",
            "Δ Malicious": f"{(values['malicious'] - paper['malicious']):+.2f}%" if values["malicious"] is not None else "-",
            "Average (Artigo)": f"{paper['average']:.2f}%",
            "Average (Reprod)": f"{values['average']:.2f}%" if values["average"] is not None else "N/D",
            "Δ Average": f"{(values['average'] - paper['average']):+.2f}%" if values["average"] is not None else "-",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def reproduced_values(
    display_name: str,
    raw_key: str,
    results_raw: dict[str, dict[str, float]],
    ablation_results: dict[str, dict[str, float]],
) -> list[float]:
    data = load_results_with_ablation_fallback(raw_key, results_raw, ablation_results)
    values = []
    for key in ("overdefense", "benign", "malicious", "average"):
        value = data.get(key)
        values.append(float(value) if isinstance(value, (int, float)) else 0.0)
    return values


def plot_individual_charts(
    results_raw: dict[str, dict[str, float]],
    ablation_results: dict[str, dict[str, float]],
    logs_dir: Path,
    output_dir: Path,
) -> None:
    charts_dir = logs_dir / "individual_charts"
    output_charts_dir = output_dir / "individual_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    output_charts_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["Over-defense", "Benign", "Malicious", "Average"]
    width = 0.35

    for display_name, raw_key, paper_key in MODEL_MAPPING:
        paper = PAPER_REF[paper_key]
        paper_values = [
            paper["overdefense"], paper["benign"], paper["malicious"], paper["average"]
        ]
        values = reproduced_values(
            display_name, raw_key, results_raw, ablation_results
        )
        if not any(value > 0 for value in values):
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        x = np.arange(len(metrics))
        rects_paper = ax.bar(
            x - width / 2, paper_values, width,
            label="Artigo (Original)", color="#1D3557",
            edgecolor="black", linewidth=0.6,
        )
        rects_reproduced = ax.bar(
            x + width / 2, values, width,
            label="Reproduzido (Obtido)", color="#E63946",
            edgecolor="black", linewidth=0.6,
        )

        average_delta = values[3] - paper_values[3]
        delta_symbol = "▲" if average_delta >= 0 else "▼"
        ax.set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{display_name}\nArtigo (Original) vs Reproduzido (Obtido) | "
            f"Δ Médio: {average_delta:+.2f}% {delta_symbol}",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.legend(
            frameon=True, facecolor="#F8F9FA", edgecolor="#ADB5BD",
            fontsize=10, loc="upper right",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects_paper:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#1D3557",
            )
        for index, rect in enumerate(rects_reproduced):
            height = rect.get_height()
            difference = height - paper_values[index]
            difference_text = f"({difference:+.1f}%)" if abs(difference) >= 0.05 else ""
            ax.annotate(
                f"{height:.1f}%\n{difference_text}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#E63946",
            )

        fig.tight_layout()
        slug = (
            display_name.lower()
            .replace(" ", "_").replace("(", "").replace(")", "")
            .replace("-", "_").replace(".", "_")
        )
        file_name = f"compare_{slug}.png"
        for directory in (charts_dir, output_charts_dir):
            fig.savefig(directory / file_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Gráficos individuais salvos em: {charts_dir}")


def plot_mosaic(
    results_raw: dict[str, dict[str, float]],
    ablation_results: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    valid_models = []
    for display_name, raw_key, paper_key in MODEL_MAPPING:
        paper = PAPER_REF[paper_key]
        values = reproduced_values(
            display_name, raw_key, results_raw, ablation_results
        )
        if any(value > 0 for value in values):
            paper_values = [
                paper["overdefense"], paper["benign"],
                paper["malicious"], paper["average"],
            ]
            valid_models.append((display_name, paper_values, values))

    if not valid_models:
        print("[AVISO] Nenhum resultado reproduzido disponível para o mosaico.")
        return

    number_models = len(valid_models)
    columns = 2
    rows = math.ceil(number_models / columns)
    fig_grid, axes_grid = plt.subplots(
        rows, columns, figsize=(16, 4.2 * rows), dpi=200
    )
    axes_grid = np.atleast_1d(axes_grid).flatten()
    metrics_abbr = ["Overdef", "Benign", "Malicious", "Average"]
    width = 0.35

    for index, (display_name, paper_values, values) in enumerate(valid_models):
        ax = axes_grid[index]
        x = np.arange(len(metrics_abbr))
        ax.bar(
            x - width / 2, paper_values, width, label="Artigo",
            color="#1D3557", edgecolor="black", linewidth=0.5,
        )
        ax.bar(
            x + width / 2, values, width, label="Reproduzido",
            color="#E63946", edgecolor="black", linewidth=0.5,
        )
        average_delta = values[3] - paper_values[3]
        ax.set_title(
            f"{display_name} (Δ Avg: {average_delta:+.1f}%)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_abbr, fontsize=10)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if index == 0:
            ax.legend(loc="upper right", fontsize=9)
        for metric_index in range(len(metrics_abbr)):
            ax.annotate(
                f"{paper_values[metric_index]:.0f}%",
                xy=(x[metric_index] - width / 2, paper_values[metric_index]),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                color="#1D3557", fontweight="bold",
            )
            ax.annotate(
                f"{values[metric_index]:.0f}%",
                xy=(x[metric_index] + width / 2, values[metric_index]),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
                color="#E63946", fontweight="bold",
            )

    for index in range(number_models, len(axes_grid)):
        fig_grid.delaxes(axes_grid[index])

    fig_grid.suptitle(
        "Dashboard Consolidado: Artigo (Original) vs Reproduzido em Todos os Modelos",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig_grid.tight_layout()

    logs_path = REPO_ROOT / "logs/Table1/table1_all_models_comparison_grid.png"
    output_path = output_dir / "table1_all_models_comparison_grid.png"
    for path in (logs_path, output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig_grid.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig_grid)
    print(f"Mosaico consolidado salvo em: {logs_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia a Tabela 1 e gera tabelas de deltas/gráficos.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de artefatos. Padrão: PIGUARD_OUTPUT_DIR ou <repo>/outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Pasta dos benchmarks (padrão: <repo>/datasets).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do lote.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita amostras por dataset (útil para smoke tests).",
    )
    parser.add_argument(
        "--openrouter-key",
        default=None,
        help="Chave da API OpenRouter (também aceita OPENROUTER_API_KEY/OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--skip-openrouter",
        action="store_true",
        help="Pula os LLMs via OpenRouter.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Força nova inferência e ignora caches de resultados.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if _PLOT_IMPORT_ERROR is not None:
        raise SystemExit(
            f"[ERRO] Dependências ausentes ({_PLOT_IMPORT_ERROR}). "
            "Instale com: python -m pip install -r requirements.txt"
        )

    output_dir = repo_path(args.output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = (
        repo_path(args.dataset_root)
        if args.dataset_root
        else (REPO_ROOT / "datasets")
    ).resolve()
    logs_dir = REPO_ROOT / "logs/Table1"
    logs_dir.mkdir(parents=True, exist_ok=True)

    validator = find_validator()
    if validator is None:
        parser.error("Eval/validate_models_table1.py não encontrado.")
    try:
        run_validator(
            validator, dataset_root, output_dir, checkpoint_root=output_dir,
            openrouter_key=args.openrouter_key, skip_openrouter=args.skip_openrouter,
            force_rerun=args.force_rerun, batch_size=args.batch_size, max_samples=args.max_samples,
        )
    except RuntimeError as exc:
        raise SystemExit(f"[ERRO] {exc}") from exc

    results_raw = load_raw_results(output_dir)
    ablation_results = (
        load_ablation_results(output_dir)
        if dataset_root == REPO_ROOT / "datasets" and args.max_samples is None
        else {}
    )
    detailed = build_detailed_table(results_raw, ablation_results)

    print("\n📋 Comparação Lado a Lado: Artigo vs Reproduzido com Deltas (Δ)")
    print(detailed.to_string(index=False))

    detailed.to_csv(output_dir / "table1_detailed_deltas.csv", index=False)

    if results_raw:
        # Tabela resumo (usa o CSV do validador quando disponível).
        csv_path = output_dir / "table1_validation_summary.csv"
        if csv_path.is_file():
            summary = pd.read_csv(csv_path)
            print("\n📊 Tabela 1: Avaliação Comparativa de Performance e Overhead")
            print(summary.to_string(index=False))
        chart_path = output_dir / "table1_comparison.png"
        if chart_path.is_file():
            print(f"\nGráfico consolidado: {chart_path}")
    else:
        print("\n[AVISO] Nenhum resultado bruto disponível; apenas a tabela de referência foi gerada.")

    plot_individual_charts(results_raw, ablation_results, logs_dir, output_dir)
    plot_mosaic(results_raw, ablation_results, output_dir)
    sync_results(logs_dir, output_dir)

    print(f"\nResultados em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

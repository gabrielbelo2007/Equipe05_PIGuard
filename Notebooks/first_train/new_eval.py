"""PIGuard - Validação Comparativa (variante histórica) e Análise de Over-Defense.

Conversão do notebook ``Notebooks/first_train/new_eval.ipynb`` para script Python

Uso:
    python Notebooks/first_train/new_eval.py
    python Notebooks/first_train/new_eval.py --batch-size 64 --device cpu

Modelos avaliados:
1. ``train_augmented``: base aumentada original do artigo.
2. ``train_new_augmented``: novas injeções geradas pelo pipeline reproduzido.
3. ``train_cleaned``: baseline limpo, sem aumento de dados.

O script gera a tabela comparativa, o JSON bruto, um diagnóstico científico e o
gráfico de barras no diretório de saída.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

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

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as exc:
    torch = None
    AutoTokenizer = None
    _RUNTIME_IMPORT_ERROR: ImportError | None = exc
else:
    _RUNTIME_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PIGUARD_OUTPUT_DIR", REPO_ROOT / "outputs")
).expanduser()

PAPER_CLEAN = {
    "NotInject_overall_acc": 0.7522,
    "WildGuard_acc": 0.7853,
    "BIPIA_overall_acc": 0.7017,
    "Overall_acc": 0.7464,
    "trigger_tokens_count": "Referência Limpa",
}

PAPER_AUGMENTED = {
    "NotInject_overall_acc": 0.6431,
    "WildGuard_acc": 0.8136,
    "BIPIA_overall_acc": 0.7595,
    "Overall_acc": 0.7387,
    "trigger_tokens_count": "Alto Viés (Queda Over-defense)",
}


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def find_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_existing(candidates: Sequence[Path]) -> Path:
    return find_existing(candidates) or candidates[0]


def build_experiments(output_dir: Path) -> dict[str, dict[str, object]]:
    checkpoints = REPO_ROOT / "checkpoints"

    def ckpt_candidates(name: str) -> list[Path]:
        return [output_dir / name / "best_model.pth", checkpoints / name / "best_model.pth"]

    return {
        "Modelo 1 (train_augmented)": {
            "candidates": ckpt_candidates("train_augmented"),
            "trigger_candidates": [
                output_dir / "train_augmented" / "trigger_words_train_augmented.json",
                REPO_ROOT / "MOF" / "trigger_words_train_augmented.json",
            ],
            "description": "Baseline + LLM Augmented Original (PIGuard oficial)",
        },
        "Modelo 2 (train_new_augmented)": {
            "candidates": ckpt_candidates("train_new_augmented"),
            "trigger_candidates": [
                output_dir / "train_new_augmented" / "trigger_words_train_new_augmented.json",
                REPO_ROOT / "MOF" / "trigger_words_train_new_augmented.json",
            ],
            "description": "Baseline + Novas Gerações GPT-4o-mini (17 formatos estruturados)",
        },
        "Modelo 3 (train_cleaned)": {
            "candidates": ckpt_candidates("train_cleaned"),
            "trigger_candidates": [
                output_dir / "train_cleaned" / "trigger_words_train_cleaned.json",
                REPO_ROOT / "MOF" / "trigger_words_train_cleaned.json",
            ],
            "description": "Baseline Limpo sem cauda longa (sem aumento de dados)",
        },
    }


def load_piguard_class():
    original_dir = str(REPO_ROOT / "Original")
    if original_dir not in sys.path:
        sys.path.insert(0, original_dir)
    try:
        from PIGuard import PIGuard

        print("PIGuard importado com sucesso a partir de Original/PIGuard.py")
        return PIGuard
    except ImportError:
        print(
            "Aviso: PIGuard.py não encontrado no path. "
            "Usando definição autossuficiente da arquitetura PIGuard..."
        )
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel

        class PIGuard(nn.Module):  # type: ignore[no-redef]
            def __init__(self, model_name: str, num_labels: int, device: str):
                super().__init__()
                self.device = device
                self.config = AutoConfig.from_pretrained(model_name, output_attentions=False)
                self.deberta = AutoModel.from_pretrained(model_name, config=self.config).to(device)
                self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels).to(device)
                self.loss_fct = nn.CrossEntropyLoss().to(device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            def forward(self, input_ids, attention_mask, labels=None, mode="train"):
                outputs = self.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                pooled_output = outputs["last_hidden_state"][:, 0, :].to(
                    self.classifier.weight.dtype
                )
                logits = self.classifier(pooled_output)
                loss = 0
                if labels is not None:
                    loss = self.loss_fct(
                        logits.view(-1, self.classifier.out_features), labels.view(-1)
                    )
                return logits, loss

            def classify(self, input_text: str):
                tokenized = self.tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=512
                )
                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)
                outputs = self.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )["last_hidden_state"]
                pooled_output = outputs[:, 0, :].to(self.classifier.weight.dtype)
                return self.classifier(pooled_output)

        return PIGuard


def evaluate_dataset_batch(
    model,
    tokenizer,
    text_samples: Sequence[str],
    target_label: int,
    batch_size: int = 32,
    device: str = "cuda",
) -> float:
    if not text_samples:
        return 0.0

    correct = 0
    total = len(text_samples)
    classifier_dtype = next(model.classifier.parameters()).dtype
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = list(text_samples[start : start + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model.deberta(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
            )["last_hidden_state"]
            pooled = outputs[:, 0, :].to(dtype=classifier_dtype)
            logits = model.classifier(pooled)
            predictions = logits.argmax(dim=-1).cpu().tolist()
            correct += sum(1 for prediction in predictions if prediction == target_label)
    return correct / total


def find_dataset_root(custom: str | None = None) -> Path:
    candidates: list[Path] = []
    if custom:
        candidates.append(repo_path(custom))
    candidates.extend(
        [REPO_ROOT / "datasets", SCRIPT_DIR / "datasets", Path.cwd() / "datasets"]
    )
    for candidate in candidates:
        if (candidate / "wildguard.json").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def run_full_evaluation(
    model_path: Path,
    dataset_root: Path,
    batch_size: int,
    device: str,
) -> dict[str, float | None]:
    print(f"\nCarregando modelo a partir de: {model_path}")
    print(f"Pasta de datasets resolvida em: {dataset_root}")

    PIGuard = load_piguard_class()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = PIGuard("microsoft/deberta-v3-base", num_labels=2, device=device)

    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    results: dict[str, float | None] = {}

    notinject_accs: list[float] = []
    for subset in ("one", "two", "three"):
        path = dataset_root / f"NotInject_{subset}.json"
        if not path.is_file():
            results[f"NotInject_{subset}_acc"] = None
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts = [item["prompt"] for item in data]
        accuracy = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=0,
            batch_size=batch_size, device=device,
        )
        results[f"NotInject_{subset}_acc"] = accuracy
        notinject_accs.append(accuracy)
    results["NotInject_overall_acc"] = (
        sum(notinject_accs) / len(notinject_accs) if notinject_accs else 0.0
    )

    wildguard_path = dataset_root / "wildguard.json"
    if wildguard_path.is_file():
        with wildguard_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts = [item["prompt"] for item in data]
        results["WildGuard_acc"] = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=0,
            batch_size=batch_size, device=device,
        )
    else:
        results["WildGuard_acc"] = 0.0

    bipia_accs: list[float] = []
    for kind in ("text", "code"):
        path = dataset_root / f"BIPIA_{kind}.json"
        if not path.is_file():
            results[f"BIPIA_{kind}_acc"] = 0.0
            bipia_accs.append(0.0)
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts = [item for sublist in data.values() for item in sublist]
        accuracy = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=1,
            batch_size=batch_size, device=device,
        )
        results[f"BIPIA_{kind}_acc"] = accuracy
        bipia_accs.append(accuracy)
    results["BIPIA_overall_acc"] = sum(bipia_accs) / len(bipia_accs)
    results["Overall_acc"] = (
        results["NotInject_overall_acc"]
        + results["WildGuard_acc"]
        + results["BIPIA_overall_acc"]
    ) / 3
    return results


def read_trigger_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return len(data) if isinstance(data, list) else len(data.keys())
    except (OSError, json.JSONDecodeError):
        return 0


def build_comparison_table(
    experiments: dict[str, dict[str, object]],
    all_results: dict[str, dict[str, float | None]],
) -> "pd.DataFrame":
    metrics_rows = [
        ("Over-Defense ACC (NotInject Total)", "NotInject_overall_acc", True),
        ("  └ NotInject 1 (Subwords/Sentences)", "NotInject_one_acc", True),
        ("  └ NotInject 2 (Phrases/Variations)", "NotInject_two_acc", True),
        ("  └ NotInject 3 (Complex/Real-world)", "NotInject_three_acc", True),
        ("Benign ACC (WildGuard)", "WildGuard_acc", True),
        ("Injection / Malicioso ACC (BIPIA Total)", "BIPIA_overall_acc", True),
        ("  └ BIPIA Text (Indirect Injections)", "BIPIA_text_acc", True),
        ("  └ BIPIA Code (Code Injections)", "BIPIA_code_acc", True),
        ("Overall ACC (Média Geral)", "Overall_acc", True),
        ("Trigger Tokens Extraídos (Viés de Atalho)", "trigger_tokens_count", False),
    ]

    table_data: list[dict[str, object]] = []
    for label, key, is_percentage in metrics_rows:
        row: dict[str, object] = {"Métrica de Avaliação": label}
        value_clean = PAPER_CLEAN.get(key)
        row["Modelo Limpo (Artigo)"] = (
            f"{value_clean:.2%}"
            if is_percentage and isinstance(value_clean, float)
            else str(value_clean) if value_clean is not None else "-"
        )
        value_augmented = PAPER_AUGMENTED.get(key)
        row["Modelo Augmented (Artigo)"] = (
            f"{value_augmented:.2%}"
            if is_percentage and isinstance(value_augmented, float)
            else str(value_augmented) if value_augmented is not None else "-"
        )

        for experiment_name in experiments:
            if experiment_name not in all_results:
                row[experiment_name] = "Pendente"
                continue
            value = all_results[experiment_name].get(key)
            if value is None:
                row[experiment_name] = "-"
            elif is_percentage and isinstance(value, (int, float)):
                row[experiment_name] = f"{value:.2%}"
            else:
                row[experiment_name] = str(value)
        table_data.append(row)

    return pd.DataFrame(table_data)


def print_diagnostics(all_results: dict[str, dict[str, float | None]]) -> None:
    print("\n=== Diagnóstico Científico e Comparativo ===")
    model_2_key = "Modelo 2 (train_new_augmented)"
    if model_2_key in all_results:
        result = all_results[model_2_key]
        injection = result.get("BIPIA_overall_acc") or 0.0
        overdefense = result.get("NotInject_overall_acc") or 0.0
        overall = result.get("Overall_acc") or 0.0
        trigger_count = result.get("trigger_tokens_count") or 0
        print(f"• Análise do Modelo 2 (train_new_augmented - {trigger_count} trigger words):")
        print(
            f"  - Over-Defense ACC: {overdefense:.2%} "
            f"(vs {PAPER_AUGMENTED['NotInject_overall_acc']:.2%} do Augmented Artigo "
            f"e {PAPER_CLEAN['NotInject_overall_acc']:.2%} do Limpo Artigo)"
        )
        print(
            f"  - Injection ACC:    {injection:.2%} "
            f"(vs {PAPER_AUGMENTED['BIPIA_overall_acc']:.2%} do Augmented Artigo "
            f"e {PAPER_CLEAN['BIPIA_overall_acc']:.2%} do Limpo Artigo)"
        )
        print(
            f"  - Média Geral:      {overall:.2%} "
            f"(vs {PAPER_AUGMENTED['Overall_acc']:.2%} do Augmented Artigo "
            f"e {PAPER_CLEAN['Overall_acc']:.2%} do Limpo Artigo)"
        )
        if overdefense > PAPER_AUGMENTED["NotInject_overall_acc"] and injection >= 0.70:
            print(
                "  ✅ RESULTADO POSITIVO: o Modelo 2 superou o viés do modelo "
                "augmented original, protegendo contra injeções sem a queda severa "
                "de over-defense observada no artigo."
            )
        elif injection < 0.65:
            print(
                f"  ⚠️ ATENÇÃO: a detecção de injeções caiu para {injection:.2%}. "
                "O modelo se tornou conservador demais."
            )

    model_3_key = "Modelo 3 (train_cleaned)"
    if model_3_key in all_results:
        result = all_results[model_3_key]
        print("• Comparativo do Modelo 3 (train_cleaned) com o Limpo do Artigo:")
        print(
            f"  - Over-Defense: {(result.get('NotInject_overall_acc') or 0.0):.2%} "
            f"(Artigo: {PAPER_CLEAN['NotInject_overall_acc']:.2%})"
        )
        print(
            f"  - Injection:    {(result.get('BIPIA_overall_acc') or 0.0):.2%} "
            f"(Artigo: {PAPER_CLEAN['BIPIA_overall_acc']:.2%})"
        )


def plot_comparison(
    all_results: dict[str, dict[str, float | None]],
    chart_path: Path,
) -> None:
    metrics = ["Overdefense (NotInject)", "Benign (WildGuard)", "Malicioso (BIPIA)", "Média Geral"]
    chart_series = [
        {
            "name": "Modelo Limpo (Artigo)",
            "vals": [75.22, 78.53, 70.17, 74.64],
            "color": "#6C757D",
            "hatch": "//",
        },
        {
            "name": "Modelo Augmented (Artigo)",
            "vals": [64.31, 81.36, 75.95, 73.87],
            "color": "#E76F51",
            "hatch": "\\\\",
        },
    ]

    trained_colors = {
        "Modelo 1 (train_augmented)": "#F4A261",
        "Modelo 2 (train_new_augmented)": "#2A9D8F",
        "Modelo 3 (train_cleaned)": "#264653",
    }
    for experiment_name, color in trained_colors.items():
        if experiment_name not in all_results:
            continue
        result = all_results[experiment_name]
        chart_series.append(
            {
                "name": experiment_name,
                "vals": [
                    (result.get("NotInject_overall_acc") or 0.0) * 100,
                    (result.get("WildGuard_acc") or 0.0) * 100,
                    (result.get("BIPIA_overall_acc") or 0.0) * 100,
                    (result.get("Overall_acc") or 0.0) * 100,
                ],
                "color": color,
                "hatch": None,
            }
        )

    num_models = len(chart_series)
    x = np.arange(len(metrics))
    width = 0.82 / num_models

    fig, ax = plt.subplots(figsize=(14, 7))
    for index, item in enumerate(chart_series):
        offset = (index - num_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            item["vals"],
            width,
            label=item["name"],
            color=item["color"],
            edgecolor="black",
            linewidth=0.8,
            hatch=item["hatch"],
            alpha=0.92,
        )
        for bar in bars:
            value = bar.get_height()
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 1.2,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_ylabel("Acurácia (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Comparação Direta: Modelos de Treinamento do Artigo vs Nossos Modelos Treinados",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(75, color="gray", linestyle="--", alpha=0.5, label="Linha Base 75%")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Gráfico comparativo salvo em: {chart_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia os modelos da variante histórica contra os benchmarks oficiais.",
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
        "--device",
        default=None,
        help="cuda, cpu, mps etc. Padrão: detecta automaticamente.",
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
    if _RUNTIME_IMPORT_ERROR is not None:
        raise SystemExit(
            f"[ERRO] Dependências ausentes ({_RUNTIME_IMPORT_ERROR}). "
            "Instale PyTorch e Transformers conforme requirements.txt."
        )

    output_dir = (
        repo_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = find_dataset_root(args.dataset_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Diretório de saída: {output_dir}")
    print(f"Datasets: {dataset_root}")
    print(f"Device em uso para avaliação: {device}")

    experiments = build_experiments(output_dir)

    status_rows = []
    for name, info in experiments.items():
        checkpoint_path = resolve_existing(info["candidates"])  # type: ignore[arg-type]
        trigger_path = resolve_existing(info["trigger_candidates"])  # type: ignore[arg-type]
        status_rows.append(
            {
                "Experimento": name,
                "Checkpoint": "Encontrado" if checkpoint_path.is_file() else "Ausente",
                "Caminho Resolvido": str(checkpoint_path),
                "Trigger Words Salvas": (
                    f"{read_trigger_count(trigger_path)} tokens"
                    if trigger_path.is_file() else "Ausente"
                ),
            }
        )
    print("\n=== Status de Presença dos Modelos e Metadados ===")
    print(pd.DataFrame(status_rows).to_string(index=False))

    all_results: dict[str, dict[str, float | None]] = {}
    for name, info in experiments.items():
        checkpoint_path = resolve_existing(info["candidates"])  # type: ignore[arg-type]
        trigger_path = resolve_existing(info["trigger_candidates"])  # type: ignore[arg-type]

        if not checkpoint_path.is_file():
            print(f"\n[AVISO] Checkpoint de '{name}' não encontrado em {checkpoint_path}. Pulando...")
            continue

        print(f"\n================ Avaliando {name} ================")
        result = run_full_evaluation(
            checkpoint_path, dataset_root, args.batch_size, device
        )
        result["trigger_tokens_count"] = read_trigger_count(trigger_path)
        all_results[name] = result

        print(f"Resultados parciais para {name}:")
        print(f"  - Over-Defense ACC (NotInject): {result['NotInject_overall_acc']:.2%}")
        print(f"  - Benign ACC (WildGuard):        {result['WildGuard_acc']:.2%}")
        print(f"  - Injection ACC (BIPIA):         {result['BIPIA_overall_acc']:.2%}")
        print(f"  - Overall ACC:                   {result['Overall_acc']:.2%}")
        print(f"  - Trigger Tokens Extraídos:      {result['trigger_tokens_count']}")

    comparison = build_comparison_table(experiments, all_results)
    print("\n📊 Tabela Comparativa: Modelos do Artigo vs Modelos Treinados")
    print(comparison.to_string(index=False))
    print_diagnostics(all_results)

    csv_path = output_dir / "evaluation_comparison_table.csv"
    json_path = output_dir / "evaluation_comparison_raw.json"
    comparison.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "paper_clean": PAPER_CLEAN,
                "paper_augmented": PAPER_AUGMENTED,
                "experiments": all_results,
            },
            handle,
            indent=4,
            ensure_ascii=False,
        )
    print(f"\nTabela comparativa salva em: {csv_path}")
    print(f"Resultados brutos salvos em: {json_path}")

    plot_comparison(all_results, output_dir / "benchmark_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

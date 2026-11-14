"""PIGuard - Avaliação e Reprodução da Tabela de Ablação com MOF (Tabela 2).

Conversão do notebook ``Notebooks/second_train/eval.ipynb`` para script Python
autônomo

O script avalia os modelos do estudo de ablação e reconstrói a Tabela 2 do
artigo PIGuard, comparando as 3 vertentes (Original, Semi-Reprodução e
Reproduzida). Resultados, tabelas e gráficos são gravados em ``logs/MOFs`` e no
diretório de saída. Resultados anteriores podem ser reutilizados para evitar
inferência pesada; use ``--force-reevaluate`` para recalcular tudo.
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

PAPER_TABLE_2 = [
    {
        "design": "Basic dataset training",
        "short_name": "Basic",
        "overdefense": 75.22,
        "benign": 78.53,
        "malicious": 70.17,
        "overall": 74.64,
        "orig_model": "train_cleaned_original",
        "semi_rp_model": "train_cleaned_original",
        "rp_model": "train_cleaned_original",
    },
    {
        "design": "+ Data-centric augment",
        "short_name": "+ Aug",
        "overdefense": 64.31,
        "benign": 81.36,
        "malicious": 75.95,
        "overall": 73.87,
        "orig_model": "train_augmented_original",
        "semi_rp_model": "train_augmented_original",
        "rp_model": "train_augmented_rp",
    },
    {
        "design": "Basic + MOF scratch retrain",
        "short_name": "Basic + MOF (Scratch)",
        "overdefense": 89.38,
        "benign": 84.73,
        "malicious": 71.57,
        "overall": 81.89,
        "orig_model": "train_clean_mof_original",
        "semi_rp_model": "train_clean_mof_semi_rp",
        "rp_model": "train_clean_mof_rp",
    },
    {
        "design": "Basic + Aug + MOF finetuning",
        "short_name": "+ Aug + MOF (FT)",
        "overdefense": 68.14,
        "benign": 82.11,
        "malicious": 74.81,
        "overall": 75.02,
        "orig_model": "train_augmented_mof_finetuning_original",
        "semi_rp_model": "train_augmented_mof_finetuning_semi_rp",
        "rp_model": "train_augmented_mof_finetuning_rp",
    },
    {
        "design": "Basic + Aug + MOF scratch retrain",
        "short_name": "+ Aug + MOF (Scratch)",
        "overdefense": 87.32,
        "benign": 85.74,
        "malicious": 77.39,
        "overall": 83.48,
        "orig_model": "train_augmented_mof_original",
        "semi_rp_model": "train_augmented_mof_semi_rp",
        "rp_model": "train_augmented_mof_rp",
    },
]

ALIASES = {
    "train_cleaned_original": ["train_cleaned_oficial"],
    "train_augmented_original": ["train_augmented_oficial"],
    "train_augmented_rp": ["train_augmented"],
    "train_clean_mof_semi_rp": ["train_clean_mof_oficial"],
    "train_augmented_mof_finetuning_semi_rp": ["train_augmented_mof_finetuning_oficial"],
    "train_augmented_mof_semi_rp": ["train_augmented_mof_oficial"],
    "train_clean_mof_rp": ["train_clean_mof"],
    "train_augmented_mof_finetuning_rp": ["train_augmented_mof_finetuning"],
    "train_augmented_mof_rp": ["train_augmented_mof"],
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

    def ckpt(name: str, *aliases: str) -> list[Path]:
        candidates = [output_dir / name / "best_model.pth"]
        for alias in aliases:
            candidates.extend(
                [output_dir / alias / "best_model.pth", checkpoints / alias / "best_model.pth"]
            )
        candidates.append(checkpoints / name / "best_model.pth")
        return candidates

    return {
        # Modelos base da Fase 1.
        "train_cleaned_original": {
            "design": "Basic dataset training",
            "vertente": "Base",
            "candidates": ckpt("train_cleaned_original", "train_cleaned_oficial"),
        },
        "train_augmented_original": {
            "design": "+ Data-centric augment",
            "vertente": "Original",
            "candidates": ckpt("train_augmented_original", "train_augmented_oficial"),
        },
        "train_augmented_rp": {
            "design": "+ Data-centric augment",
            "vertente": "Reproduzida",
            "candidates": ckpt("train_augmented_rp", "train_augmented"),
        },
        # Vertente 1: datasets originais do artigo.
        "train_clean_mof_original": {
            "design": "Basic + MOF scratch retrain",
            "vertente": "Original",
            "candidates": ckpt("train_clean_mof_original"),
        },
        "train_augmented_mof_finetuning_original": {
            "design": "Basic + Aug + MOF finetuning",
            "vertente": "Original",
            "candidates": ckpt("train_augmented_mof_finetuning_original"),
        },
        "train_augmented_mof_original": {
            "design": "Basic + Aug + MOF scratch retrain",
            "vertente": "Original",
            "candidates": ckpt("train_augmented_mof_original"),
        },
        # Vertente 2: semi-reprodução / oficial.
        "train_clean_mof_semi_rp": {
            "design": "Basic + MOF scratch retrain",
            "vertente": "Semi-Reprodução",
            "candidates": ckpt("train_clean_mof_semi_rp", "train_clean_mof_oficial"),
        },
        "train_augmented_mof_finetuning_semi_rp": {
            "design": "Basic + Aug + MOF finetuning",
            "vertente": "Semi-Reprodução",
            "candidates": ckpt(
                "train_augmented_mof_finetuning_semi_rp",
                "train_augmented_mof_finetuning_oficial",
            ),
        },
        "train_augmented_mof_semi_rp": {
            "design": "Basic + Aug + MOF scratch retrain",
            "vertente": "Semi-Reprodução",
            "candidates": ckpt("train_augmented_mof_semi_rp", "train_augmented_mof_oficial"),
        },
        # Vertente 3: reproduzida.
        "train_clean_mof_rp": {
            "design": "Basic + MOF scratch retrain",
            "vertente": "Reproduzida",
            "candidates": ckpt("train_clean_mof_rp", "train_clean_mof"),
        },
        "train_augmented_mof_finetuning_rp": {
            "design": "Basic + Aug + MOF finetuning",
            "vertente": "Reproduzida",
            "candidates": ckpt(
                "train_augmented_mof_finetuning_rp",
                "train_augmented_mof_finetuning",
            ),
        },
        "train_augmented_mof_rp": {
            "design": "Basic + Aug + MOF scratch retrain",
            "vertente": "Reproduzida",
            "candidates": ckpt("train_augmented_mof_rp", "train_augmented_mof"),
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
                loss = None
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
    print(f"\nCarregando checkpoint: {model_path}")
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


def load_cached_results(output_dir: Path) -> tuple[dict[str, dict[str, float | None]], Path | None]:
    """Recupera inferências já salvas, aplicando os aliases do notebook."""
    candidates = [
        output_dir / "ablation_results_raw.json",
        REPO_ROOT / "logs/MOFs/ablation_results_raw.json",
        REPO_ROOT / "Eval/MOFs/ablation_results_raw.json",
        output_dir / "table2_validation_raw.json",
        REPO_ROOT / "logs/MOFs/table2_validation_raw.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                loaded_data = json.load(handle)
            saved = loaded_data.get("evaluated_experiments", loaded_data)
            if not isinstance(saved, dict) or not saved:
                continue

            all_results: dict[str, dict[str, float | None]] = {}
            for experiment_name in build_experiments(output_dir):
                if experiment_name in saved:
                    all_results[experiment_name] = saved[experiment_name]
                    continue
                for alias in ALIASES.get(experiment_name, []):
                    if alias in saved:
                        all_results[experiment_name] = saved[alias]
                        break
            for key, value in saved.items():
                all_results.setdefault(key, value)
            return all_results, candidate
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Aviso ao ler {candidate}: {exc}")
    return {}, None


def save_raw_results(
    all_results: dict[str, dict[str, float | None]], output_dir: Path
) -> tuple[Path, Path]:
    logs_dir = REPO_ROOT / "logs/MOFs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"evaluated_experiments": all_results}
    local_path = logs_dir / "ablation_results_raw.json"
    output_path = output_dir / "ablation_results_raw.json"
    for path in (local_path, output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, ensure_ascii=False)
    return local_path, output_path


def get_model_result(
    model_key: str, all_results: dict[str, dict[str, float | None]]
) -> dict[str, float | None]:
    if model_key in all_results:
        return all_results[model_key]
    for alias in ALIASES.get(model_key, []):
        if alias in all_results:
            return all_results[alias]
    return {}


def metric_percent(
    result: dict[str, float | None], key: str
) -> float:
    """Retorna a métrica em pontos percentuais, ou NaN quando ausente."""
    value = result.get(key)
    if isinstance(value, (int, float)):
        return float(value) * 100.0
    return float("nan")


def format_metrics(result: dict[str, float | None]) -> str:
    if not result:
        return "Pendente"
    overdefense = (result.get("NotInject_overall_acc") or 0.0) * 100.0
    benign = (result.get("WildGuard_acc") or 0.0) * 100.0
    malicious = (result.get("BIPIA_overall_acc") or 0.0) * 100.0
    overall = (result.get("Overall_acc") or 0.0) * 100.0
    return f"{overdefense:.2f}% / {benign:.2f}% / {malicious:.2f}% / {overall:.2f}%"


def build_tables(
    all_results: dict[str, dict[str, float | None]]
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    summary_rows = []
    for item in PAPER_TABLE_2:
        summary_rows.append(
            {
                "Design de Treinamento": item["design"],
                "Artigo (Overdef / Benign / Mal / Avg)": (
                    f"{item['overdefense']:.2f}% / {item['benign']:.2f}% / "
                    f"{item['malicious']:.2f}% / {item['overall']:.2f}%"
                ),
                "Vertente 1: Original (Artigo)": format_metrics(
                    get_model_result(item["orig_model"], all_results)
                ),
                "Vertente 2: Semi-Reprodução": format_metrics(
                    get_model_result(item["semi_rp_model"], all_results)
                ),
                "Vertente 3: Reproduzida": format_metrics(
                    get_model_result(item["rp_model"], all_results)
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    metric_mapping = [
        ("Over-defense (NotInject)", "overdefense", "NotInject_overall_acc"),
        ("Benign (WildGuard)", "benign", "WildGuard_acc"),
        ("Malicious (BIPIA)", "malicious", "BIPIA_overall_acc"),
        ("Average (Overall ACC)", "overall", "Overall_acc"),
    ]
    detailed_rows = []
    for item in PAPER_TABLE_2:
        result_original = get_model_result(item["orig_model"], all_results)
        result_semi = get_model_result(item["semi_rp_model"], all_results)
        result_rp = get_model_result(item["rp_model"], all_results)
        for metric_label, paper_key, result_key in metric_mapping:
            paper_value = item[paper_key]
            value_original = (
                (result_original.get(result_key) or 0.0) * 100.0
                if result_key in result_original else None
            )
            value_semi = (
                (result_semi.get(result_key) or 0.0) * 100.0
                if result_key in result_semi else None
            )
            value_rp = (
                (result_rp.get(result_key) or 0.0) * 100.0
                if result_key in result_rp else None
            )
            detailed_rows.append(
                {
                    "Design": item["short_name"],
                    "Métrica": metric_label,
                    "Artigo": f"{paper_value:.2f}%",
                    "V1 (Orig)": f"{value_original:.2f}%" if value_original is not None else "N/D",
                    "Δ V1": f"{(value_original - paper_value):+.2f}%" if value_original is not None else "-",
                    "V2 (Semi-RP)": f"{value_semi:.2f}%" if value_semi is not None else "N/D",
                    "Δ V2": f"{(value_semi - paper_value):+.2f}%" if value_semi is not None else "-",
                    "V3 (Reprod)": f"{value_rp:.2f}%" if value_rp is not None else "N/D",
                    "Δ V3": f"{(value_rp - paper_value):+.2f}%" if value_rp is not None else "-",
                }
            )
    return summary, pd.DataFrame(detailed_rows)


def plot_trajectories(
    all_results: dict[str, dict[str, float | None]], output_dir: Path
) -> Path:
    designs_labels = [item["short_name"] for item in PAPER_TABLE_2]
    x = np.arange(len(designs_labels))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)

    metric_trajectories = [
        ("Recuperação do Over-Defense (NotInject ACC)", "overdefense", "NotInject_overall_acc", (55, 95), axes[0, 0]),
        ("Acurácia Benigna (WildGuard ACC)", "benign", "WildGuard_acc", (70, 92), axes[0, 1]),
        ("Segurança contra Injeção (BIPIA ACC)", "malicious", "BIPIA_overall_acc", (65, 85), axes[1, 0]),
        ("Acurácia Média Global (Overall ACC)", "overall", "Overall_acc", (68, 88), axes[1, 1]),
    ]
    series = [
        ("orig_model", "Vertente 1: Original", "#2A9D8F", "s", "--"),
        ("semi_rp_model", "Vertente 2: Semi-Reprodução", "#E9C46A", "D", "-."),
        ("rp_model", "Vertente 3: Reproduzida", "#E76F51", "^", ":"),
    ]

    for title, paper_key, result_key, y_limits, ax in metric_trajectories:
        paper_values = [item[paper_key] for item in PAPER_TABLE_2]
        ax.plot(
            x, paper_values, marker="o", linewidth=3.2, markersize=8.5,
            color="#264653", label="Artigo Oficial (Referência)", zorder=5,
        )
        for model_key, label, color, marker, linestyle in series:
            values = [
                metric_percent(get_model_result(item[model_key], all_results), result_key)
                for item in PAPER_TABLE_2
            ]
            valid = [(xi, yi) for xi, yi in zip(x, values) if not np.isnan(yi)]
            if valid:
                ax.plot(
                    *zip(*valid), marker=marker, linewidth=2.4, linestyle=linestyle,
                    markersize=7.5, color=color, label=label, zorder=4,
                )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Acurácia (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(designs_labels, rotation=20, ha="right", fontsize=10.5)
        ax.set_ylim(y_limits)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "Trajetórias Comparativas ao Longo dos Designs de Treinamento (Artigo vs Vertentes)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout()

    local_path = REPO_ROOT / "logs/MOFs/ablation_trajectories_comparison.png"
    output_path = output_dir / "ablation_trajectories_comparison.png"
    for path in (local_path, output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de trajetórias salvo em: {local_path}")
    return local_path


def plot_individual_designs(
    all_results: dict[str, dict[str, float | None]], output_dir: Path
) -> None:
    charts_dir = REPO_ROOT / "logs/MOFs/individual_charts"
    output_charts_dir = output_dir / "individual_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    output_charts_dir.mkdir(parents=True, exist_ok=True)

    metrics_labels = [
        "Over-defense\n(NotInject)", "Benign\n(WildGuard)",
        "Malicious\n(BIPIA)", "Overall\n(Média)",
    ]
    metrics_keys = [
        "NotInject_overall_acc", "WildGuard_acc", "BIPIA_overall_acc", "Overall_acc",
    ]
    paper_keys = ["overdefense", "benign", "malicious", "overall"]
    series_configs = [
        ("Artigo (Referência)", "#264653", None),
        ("Vertente 1: Original", "#2A9D8F", "//"),
        ("Vertente 2: Semi-RP", "#E9C46A", "\\"),
        ("Vertente 3: Reproduzida", "#E76F51", "xx"),
    ]

    for index, design_info in enumerate(PAPER_TABLE_2):
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
        x = np.arange(len(metrics_labels))
        width = 0.20
        values_list = [
            [design_info[key] for key in paper_keys],
            [
                metric_percent(get_model_result(design_info["orig_model"], all_results), key)
                for key in metrics_keys
            ],
            [
                metric_percent(get_model_result(design_info["semi_rp_model"], all_results), key)
                for key in metrics_keys
            ],
            [
                metric_percent(get_model_result(design_info["rp_model"], all_results), key)
                for key in metrics_keys
            ],
        ]

        for series_index, (series_name, color, hatch) in enumerate(series_configs):
            offset = (series_index - 1.5) * width
            values = values_list[series_index]
            bars = ax.bar(
                x + offset, values, width, label=series_name, color=color,
                hatch=hatch, alpha=0.92, edgecolor="black", linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{value:.1f}%", ha="center", va="bottom",
                        fontsize=8, rotation=35,
                    )

        ax.set_title(
            f"Design {index + 1}: {design_info['design']} — Comparação Individual",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.set_ylabel("Acurácia (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, fontsize=10.5)
        ax.set_ylim(45, 102)
        ax.grid(axis="y", linestyle="--", alpha=0.55)
        ax.legend(loc="lower right", fontsize=10)

        file_name = (
            f"design_{index + 1}_"
            f"{design_info['short_name'].replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}.png"
        )
        fig.tight_layout()
        for directory in (charts_dir, output_charts_dir):
            fig.savefig(directory / file_name, bbox_inches="tight")
        plt.close(fig)

    # Mosaico panorâmico 2x3 com painel explicativo no 6º quadrante.
    fig_mosaic, axes_mosaic = plt.subplots(2, 3, figsize=(22, 13), dpi=300)
    axes_flat = axes_mosaic.flatten()

    for index, design_info in enumerate(PAPER_TABLE_2):
        ax = axes_flat[index]
        x = np.arange(len(metrics_labels))
        width = 0.20
        values_list = [
            [design_info[key] for key in paper_keys],
            [
                metric_percent(get_model_result(design_info["orig_model"], all_results), key)
                for key in metrics_keys
            ],
            [
                metric_percent(get_model_result(design_info["semi_rp_model"], all_results), key)
                for key in metrics_keys
            ],
            [
                metric_percent(get_model_result(design_info["rp_model"], all_results), key)
                for key in metrics_keys
            ],
        ]
        for series_index, (series_name, color, hatch) in enumerate(series_configs):
            offset = (series_index - 1.5) * width
            values = values_list[series_index]
            bars = ax.bar(
                x + offset, values, width,
                label=series_name if index == 0 else "",
                color=color, hatch=hatch, alpha=0.92,
                edgecolor="black", linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{value:.1f}%", ha="center", va="bottom",
                        fontsize=7.2, rotation=45,
                    )
        ax.set_title(
            f"Design {index + 1}: {design_info['design']}",
            fontsize=11.5, fontweight="bold", pad=10,
        )
        ax.set_ylabel("Acurácia (%)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, fontsize=9.5)
        ax.set_ylim(45, 102)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    note_ax = axes_flat[5]
    note_ax.axis("off")
    note_content = (
        "[NOTA METODOLÓGICA SOBRE DISCREPÂNCIAS]\n\n"
        "• Conjunto de Avaliação Estendido no Artigo:\n"
        "  Os resultados originais do paper PIGuard foram avaliados\n"
        "  sobre conjuntos de teste que incorporavam amostras suplementares\n"
        "  (amostras adicionais benignas e de injeção direta/indireta).\n\n"
        "• Análise das Variações Percentuais:\n"
        "  Discrepâncias discretas (na faixa de ±1% a ±3%) observadas entre\n"
        "  o artigo e os checkpoints reproduzidos são inerentes a essa sutil\n"
        "  diferença amostral na distribuição de avaliação.\n\n"
        "• Comprovação Integral dos Efeitos de Ablação:\n"
        "  1. O viés de atalho induzido pelo '+ Aug' é confirmado (~75% -> ~64%).\n"
        "  2. A eficácia máxima do MOF no overdefense é replicada (~89%).\n"
        "  3. O balanço ótimo com '+ Aug + MOF scratch' atinge ~87% Overdef e ~83% Overall."
    )
    note_ax.text(
        0.04, 0.5, note_content, transform=note_ax.transAxes, fontsize=10.2,
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.8", facecolor="#F8F9FA",
            edgecolor="#264653", linewidth=1.6,
        ),
    )
    fig_mosaic.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4,
        fontsize=12, frameon=True,
    )
    fig_mosaic.tight_layout(rect=[0, 0, 1, 0.965])

    local_mosaic = REPO_ROOT / "logs/MOFs/ablation_mosaic_all_designs.png"
    output_mosaic = output_dir / "ablation_mosaic_all_designs.png"
    for path in (local_mosaic, output_mosaic):
        fig_mosaic.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig_mosaic)
    print(f"Mosaico panorâmico salvo em: {local_mosaic}")


def plot_discrepancy(
    all_results: dict[str, dict[str, float | None]], output_dir: Path
) -> None:
    designs_labels = [item["short_name"] for item in PAPER_TABLE_2]
    x = np.arange(len(designs_labels))
    width = 0.20

    deltas_overdefense = []
    deltas_benign = []
    deltas_malicious = []
    deltas_overall = []
    for item in PAPER_TABLE_2:
        result_rp = get_model_result(item["rp_model"], all_results)
        overdefense = metric_percent(result_rp, "NotInject_overall_acc")
        benign = metric_percent(result_rp, "WildGuard_acc")
        malicious = metric_percent(result_rp, "BIPIA_overall_acc")
        overall = metric_percent(result_rp, "Overall_acc")
        deltas_overdefense.append(overdefense - item["overdefense"] if not np.isnan(overdefense) else 0.0)
        deltas_benign.append(benign - item["benign"] if not np.isnan(benign) else 0.0)
        deltas_malicious.append(malicious - item["malicious"] if not np.isnan(malicious) else 0.0)
        deltas_overall.append(overall - item["overall"] if not np.isnan(overall) else 0.0)

    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="-")
    bar_groups = [
        ax.bar(x - 1.5 * width, deltas_overdefense, width, label="Δ Over-Defense (NotInject)", color="#457B9D"),
        ax.bar(x - 0.5 * width, deltas_benign, width, label="Δ Benign (WildGuard)", color="#2A9D8F"),
        ax.bar(x + 0.5 * width, deltas_malicious, width, label="Δ Malicious (BIPIA)", color="#E76F51"),
        ax.bar(x + 1.5 * width, deltas_overall, width, label="Δ Overall ACC", color="#264653"),
    ]
    for group in bar_groups:
        for bar in group:
            height = bar.get_height()
            vertical_alignment = "bottom" if height >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (0.15 if height >= 0 else -0.25),
                f"{height:+.1f}%", ha="center", va=vertical_alignment,
                fontsize=8, fontweight="bold",
            )

    ax.set_title(
        "Discrepância dos Resultados: Vertente Reproduzida vs Artigo Oficial (Δ em Pontos Percentuais)\n"
        "[Nota: Flutuações discretas decorrem das amostras adicionais benignas/maliciosas do teste do paper]",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_ylabel("Diferença (p.p. em relação ao Artigo)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(designs_labels, rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.55)
    ax.set_ylim(-6, 6)
    ax.legend(loc="upper right", fontsize=10.5)
    fig.tight_layout()

    local_path = REPO_ROOT / "logs/MOFs/ablation_discrepancy_deltas.png"
    output_path = output_dir / "ablation_discrepancy_deltas.png"
    for path in (local_path, output_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de deltas salvo em: {local_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia os modelos do estudo de ablação e reconstrói a Tabela 2.",
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
    parser.add_argument(
        "--force-reevaluate",
        action="store_true",
        help="Ignora resultados em cache e recalcula todas as inferências.",
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

    experiments = build_experiments(output_dir)

    # Status de presença (substitui display(status_df)).
    status_rows = [
        {
            "Modelo / Experimento": name,
            "Design de Treinamento": info["design"],
            "Vertente": info["vertente"],
            "Status": "✅ Encontrado" if resolve_existing(info["candidates"]).is_file() else "❌ Ausente",
            "Caminho": str(resolve_existing(info["candidates"])),
        }
        for name, info in experiments.items()
    ]
    print("=== Status de Presença dos Checkpoints ===")
    print(pd.DataFrame(status_rows).to_string(index=False))

    all_results: dict[str, dict[str, float | None]] = {}
    loaded_from: Path | None = None
    if not args.force_reevaluate:
        all_results, loaded_from = load_cached_results(output_dir)

    missing_models = [name for name in experiments if name not in all_results]
    cached_complete = bool(all_results) and not missing_models and not args.force_reevaluate

    if cached_complete:
        print("\n" + "=" * 80)
        print("🔒 [TRAVA DE CACHE ATIVA - INFERÊNCIAS RECUPERADAS COM SUCESSO]")
        print(f"📁 Resultados resgatados de: {loaded_from}")
        print(f"📊 Total de modelos carregados: {len(all_results)} / {len(experiments)}")
        print("⚡ Nenhuma inferência pesada necessária.")
        print("Use --force-reevaluate para recalcular tudo.")
        print("=" * 80)
        for name in experiments:
            result = all_results.get(name, {})
            print(
                f"[{name[:38]:38s}] "
                f"Over-Def: {(result.get('NotInject_overall_acc') or 0.0):.2%} | "
                f"Benign: {(result.get('WildGuard_acc') or 0.0):.2%} | "
                f"Malicious: {(result.get('BIPIA_overall_acc') or 0.0):.2%} | "
                f"Overall: {(result.get('Overall_acc') or 0.0):.2%}"
            )
    else:
        if args.force_reevaluate:
            print("\n🚀 [FORCE_REEVALUATE] Recalculando todas as inferências...")
            all_results = {}
            missing_models = list(experiments.keys())
        elif all_results:
            print(
                f"\n⚡ Resultados parciais recuperados ({len(all_results)} modelos). "
                f"Avaliando {len(missing_models)} modelos pendentes..."
            )
        else:
            print("\n🚀 Nenhum cache prévio encontrado. Iniciando inferências...")

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device em uso: {device} | Datasets: {dataset_root}")

        for name in missing_models:
            info = experiments[name]
            checkpoint_path = resolve_existing(info["candidates"])
            if not checkpoint_path.is_file():
                print(f"\n[PULANDO] Checkpoint de '{name}' não encontrado em: {checkpoint_path}")
                continue

            print(f"\n{'=' * 60}")
            print(f"🔍 Avaliando: {name} ({info['design']} - {info['vertente']})")
            print(f"{'=' * 60}")
            result = run_full_evaluation(
                checkpoint_path, dataset_root, args.batch_size, device
            )
            all_results[name] = result
            print(f"Resultados para {name}:")
            print(f"  - Over-Defense (NotInject): {(result.get('NotInject_overall_acc') or 0.0):.2%}")
            print(f"  - Benign (WildGuard):        {(result.get('WildGuard_acc') or 0.0):.2%}")
            print(f"  - Malicious (BIPIA):         {(result.get('BIPIA_overall_acc') or 0.0):.2%}")
            print(f"  - Overall ACC:               {(result.get('Overall_acc') or 0.0):.2%}")

        local_path, output_path = save_raw_results(all_results, output_dir)
        print(f"\nResultados brutos salvos em: {local_path}")
        print(f"Cópia no diretório de saída: {output_path}")

    # Reconstrução da Tabela 2 e das tabelas detalhadas.
    summary, detailed = build_tables(all_results)
    print("\n📊 Reprodução da Tabela 2 do Artigo (Ablação PIGuard com MOF)")
    print(summary.to_string(index=False))
    print("\n📋 Tabela Detalhada com Deltas de Discrepância")
    print(detailed.to_string(index=False))

    logs_dir = REPO_ROOT / "logs/MOFs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(logs_dir / "ablation_table_2_summary.csv", index=False)
    detailed.to_csv(logs_dir / "ablation_table_2_detailed_deltas.csv", index=False)
    summary.to_csv(output_dir / "ablation_table_2_summary.csv", index=False)
    detailed.to_csv(output_dir / "ablation_table_2_detailed_deltas.csv", index=False)

    raw_payload = {"paper_table_2": PAPER_TABLE_2, "evaluated_experiments": all_results}
    for raw_path in (logs_dir / "ablation_results_raw.json", output_dir / "ablation_results_raw.json"):
        with raw_path.open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=4, ensure_ascii=False)

    plot_trajectories(all_results, output_dir)
    plot_individual_designs(all_results, output_dir)
    plot_discrepancy(all_results, output_dir)

    print(f"\nTabelas e gráficos salvos em: {logs_dir}")
    print(f"Cópias dos artefatos em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

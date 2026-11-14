"""PIGuard - Avaliação do Estudo de Ablação: Shortcut Mitigation vs. MOF.

Conversão do notebook ``Notebooks/train_extras/eval.ipynb`` para script Python

Uso:
    python Notebooks/train_extras/eval.py
    python Notebooks/train_extras/eval.py --benchmark shortcut
    python Notebooks/train_extras/eval.py --benchmark tensor-trust
    python Notebooks/train_extras/eval.py --benchmark all --force-reevaluate

O script cobre duas frentes:
1. Avaliação dos modelos de Shortcut Mitigation (NotInject + BIPIA) e comparação
   com as referências do artigo e com os modelos MOF.
2. Avaliação no benchmark Tensor Trust (``qxcv/tensor-trust``): prompt hijacking
   e prompt extraction, comparando o PIGuard oficial (Hugging Face) com as três
   vertentes MOF treinadas localmente.

"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
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
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
except ImportError as exc:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _RUNTIME_IMPORT_ERROR: ImportError | None = exc
else:
    _RUNTIME_IMPORT_ERROR = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PIGUARD_OUTPUT_DIR", REPO_ROOT / "outputs")
).expanduser()

PAPER_SHORTCUT = {"overdefense": 86.73, "malicious": 65.63, "average": 76.18}
PAPER_UNAUGMENTED = {"overdefense": 64.31, "malicious": 75.95, "average": 70.13}
PAPER_MOF = {"overdefense": 87.32, "malicious": 77.39, "average": 82.36}

TT_URLS = {
    "hijacking": (
        "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/main/"
        "benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
    ),
    "extraction": (
        "https://huggingface.co/datasets/qxcv/tensor-trust/resolve/main/"
        "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl"
    ),
}

TT_MODELS = {
    "PIGuard (Hugging Face - leolee99)": {
        "type": "huggingface",
        "repo_id": "leolee99/PIGuard",
        "name": "PIGuard Oficial (Hugging Face / ACL 2025)",
        "vertente": "Referência Oficial (HF)",
        "candidates": [],
    },
    "train_augmented_mof_original": {
        "type": "local",
        "repo_id": None,
        "name": "PIGuard MOF Scratch Retrain (Original)",
        "vertente": "Vertente 1: Original",
        "candidates": [
            "train_augmented_mof_original",
            "train_augmented_mof_oficial",
        ],
    },
    "train_augmented_mof_semi_rp": {
        "type": "local",
        "repo_id": None,
        "name": "PIGuard MOF Scratch Retrain (Semi-Reprodução)",
        "vertente": "Vertente 2: Semi-Reprodução",
        "candidates": [
            "train_augmented_mof_semi_rp",
            "train_augmented_mof_oficial",
        ],
    },
    "train_augmented_mof_rp": {
        "type": "local",
        "repo_id": None,
        "name": "PIGuard MOF Scratch Retrain (Reproduzida)",
        "vertente": "Vertente 3: Reproduzida",
        "candidates": [
            "train_augmented_mof_rp",
            "train_augmented_mof",
        ],
    },
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


def build_shortcut_models(output_dir: Path) -> dict[str, dict[str, object]]:
    checkpoints = REPO_ROOT / "checkpoints"
    return {
        "train_augmented_shortcut_original": {
            "name": "Shortcut Mitigation (Original/Oficial)",
            "candidates": [
                output_dir / "train_augmented_shortcut_original" / "best_model.pth",
                checkpoints / "train_augmented_shortcut_original" / "best_model.pth",
            ],
        },
        "train_augmented_shortcut_rp": {
            "name": "Shortcut Mitigation (Reproduzida)",
            "candidates": [
                output_dir / "train_augmented_shortcut_rp" / "best_model.pth",
                checkpoints / "train_augmented_shortcut_rp" / "best_model.pth",
            ],
        },
    }


def build_tensor_trust_models(output_dir: Path) -> dict[str, dict[str, object]]:
    checkpoints = REPO_ROOT / "checkpoints"
    models: dict[str, dict[str, object]] = {}
    for key, info in TT_MODELS.items():
        item = dict(info)
        if item["type"] == "local":
            candidates = []
            for name in item["candidates"]:  # type: ignore[union-attr]
                candidates.extend(
                    [
                        output_dir / name / "best_model.pth",
                        checkpoints / name / "best_model.pth",
                    ]
                )
            item["candidates"] = candidates
        models[key] = item
    return models


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
        if (candidate / "NotInject_one.json").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def run_shortcut_evaluation(
    model_path: Path,
    dataset_root: Path,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    print(f"\nCarregando modelo de: {model_path}")
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

    # 1. Over-defense (NotInject 1, 2, 3).
    notinject_accs: list[float] = []
    details: dict[str, float] = {}
    for subset in ("one", "two", "three"):
        path = dataset_root / f"NotInject_{subset}.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts = [item["prompt"] for item in data]
        accuracy = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=0,
            batch_size=batch_size, device=device,
        )
        notinject_accs.append(accuracy)
        details[f"NotInject_{subset}_acc"] = accuracy
    overdefense_acc = (
        sum(notinject_accs) / len(notinject_accs) if notinject_accs else 0.0
    )

    # 2. Malicious (BIPIA Text/Code).
    bipia_text_acc = 0.0
    bipia_text_path = dataset_root / "BIPIA_text.json"
    if bipia_text_path.is_file():
        with bipia_text_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts = [item for sublist in data.values() for item in sublist]
        bipia_text_acc = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=1,
            batch_size=batch_size, device=device,
        )

    bipia_code_acc = 0.0
    bipia_code_path = dataset_root / "BIPIA_code.json"
    if bipia_code_path.is_file():
        with bipia_code_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        texts = [item for sublist in data.values() for item in sublist]
        bipia_code_acc = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=1,
            batch_size=batch_size, device=device,
        )

    malicious_acc = (bipia_text_acc + bipia_code_acc) / 2.0
    average_acc = (overdefense_acc + malicious_acc) / 2.0
    details["BIPIA_text_acc"] = bipia_text_acc
    details["BIPIA_code_acc"] = bipia_code_acc
    return {
        "overdefense_acc": overdefense_acc,
        "malicious_acc": malicious_acc,
        "average_acc": average_acc,
        "details": details,
    }


def load_ablation_results(output_dir: Path) -> dict[str, dict[str, float]]:
    candidates = [
        REPO_ROOT / "Eval/MOFs/ablation_results_raw.json",
        REPO_ROOT / "logs/MOFs/ablation_results_raw.json",
        output_dir / "ablation_results_raw.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            loaded = data.get("evaluated_experiments", data)
            if isinstance(loaded, dict):
                return loaded
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def extract_metrics(result: dict[str, float] | None) -> tuple[float | None, float | None, float | None]:
    if not result:
        return None, None, None
    overdefense = result.get("overdefense_acc", result.get("NotInject_overall_acc"))
    malicious = result.get("malicious_acc", result.get("BIPIA_overall_acc"))
    if overdefense is None or malicious is None:
        return None, None, None
    return overdefense * 100.0, malicious * 100.0, ((overdefense + malicious) / 2.0) * 100.0


def run_shortcut_section(
    output_dir: Path,
    dataset_root: Path,
    batch_size: int,
    device: str,
    force_reevaluate: bool,
) -> dict[str, dict[str, object]]:
    eval_dir = REPO_ROOT / "Eval/ShortcutMitigation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "shortcut_mitigation_results.json"
    output_raw_path = output_dir / "shortcut_mitigation_results.json"

    # Sincroniza cache do diretório de saída, se necessário.
    if output_raw_path.is_file() and not raw_path.is_file():
        shutil.copy2(output_raw_path, raw_path)

    results_data: dict[str, object] = {}
    if raw_path.is_file():
        try:
            with raw_path.open("r", encoding="utf-8") as handle:
                results_data = json.load(handle)
            print("\n🔒 [CACHE ATIVO] Resultados de Shortcut Mitigation carregados.")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Aviso ao ler cache: {exc}")
            results_data = {}

    evaluated_models: dict[str, dict[str, object]] = results_data.get(
        "evaluated_models", {}
    )  # type: ignore[assignment]
    models = build_shortcut_models(output_dir)

    status_rows = []
    for key, info in models.items():
        checkpoint_path = resolve_existing(info["candidates"])  # type: ignore[arg-type]
        status_rows.append(
            {
                "Modelo": key,
                "Descrição": info["name"],
                "Status": "✅ Encontrado" if checkpoint_path.is_file() else "❌ Ausente",
                "Caminho Resolvido": str(checkpoint_path),
            }
        )
    print("\n=== Status dos Checkpoints de Shortcut Mitigation ===")
    print(pd.DataFrame(status_rows).to_string(index=False))

    for key, info in models.items():
        cached_entry = evaluated_models.get(key)
        is_valid_cache = (
            isinstance(cached_entry, dict)
            and cached_entry.get("overdefense_acc") is not None
            and cached_entry.get("malicious_acc") is not None
        )
        if is_valid_cache and not force_reevaluate:
            print(f"[CACHE] Modelo '{key}' já avaliado. Pulando inferência.")
            continue

        checkpoint_path = resolve_existing(info["candidates"])  # type: ignore[arg-type]
        if not checkpoint_path.is_file():
            print(f"\n[AVISO] Checkpoint de '{key}' não encontrado em: {checkpoint_path}. Pulando...")
            continue

        print("\n" + "=" * 60)
        print(f"🔍 Avaliando: {key} ({info['name']})")
        print("=" * 60)
        result = run_shortcut_evaluation(
            checkpoint_path, dataset_root, batch_size, device
        )
        evaluated_models[key] = result  # type: ignore[assignment]
        print(f"Resultados para {key}:")
        print(f"  - Over-defense (NotInject): {result['overdefense_acc']:.2%}")
        print(f"  - Malicious (BIPIA):         {result['malicious_acc']:.2%}")
        print(f"  - Média ((Overdef + Mal)/2): {result['average_acc']:.2%}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_payload = {
        "experiment": "Ablação de Mitigação de Atalhos (Shortcut Mitigation) vs. MOF",
        "metric_formula": "average_acc = (overdefense_acc + malicious_acc) / 2",
        "paper_reference": {
            "shortcut_mitigation": {
                "overdefense_acc": 0.8673,
                "malicious_acc": 0.6563,
                "average_acc": 0.7618,
            },
            "unmitigated_augmented": {
                "overdefense_acc": 0.6431,
                "malicious_acc": 0.7595,
                "average_acc": 0.7013,
            },
            "mof_scratch_retrain": {
                "overdefense_acc": 0.8732,
                "malicious_acc": 0.7739,
                "average_acc": 0.82355,
            },
        },
        "evaluated_models": evaluated_models,
    }
    for path in (raw_path, output_raw_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=4, ensure_ascii=False)
    print(f"\n💾 JSON final salvo em: {raw_path}")

    # Tabela 1: artigo vs modelos treinados.
    table1_rows = [
        {
            "Modelo / Configuração": "Shortcut Mitigation (Artigo Oficial)",
            "Over-defense (%)": f"{PAPER_SHORTCUT['overdefense']:.2f}%",
            "Δ Over-def": "-",
            "Malicious (%)": f"{PAPER_SHORTCUT['malicious']:.2f}%",
            "Δ Malicious": "-",
            "Média (%)": f"{PAPER_SHORTCUT['average']:.2f}%",
            "Δ Média": "-",
        }
    ]
    for key, display_name in [
        ("train_augmented_shortcut_original", "train_augmented_shortcut_original (Oficial/Original)"),
        ("train_augmented_shortcut_rp", "train_augmented_shortcut_rp (Reproduzida)"),
    ]:
        result = evaluated_models.get(key, {})
        overdefense = result.get("overdefense_acc")
        malicious = result.get("malicious_acc")
        average = result.get("average_acc")
        overdefense_pct = overdefense * 100.0 if overdefense is not None else None
        malicious_pct = malicious * 100.0 if malicious is not None else None
        average_pct = average * 100.0 if average is not None else None
        table1_rows.append(
            {
                "Modelo / Configuração": display_name,
                "Over-defense (%)": f"{overdefense_pct:.2f}%" if overdefense_pct is not None else "Pendente",
                "Δ Over-def": f"{(overdefense_pct - PAPER_SHORTCUT['overdefense']):+.2f}%" if overdefense_pct is not None else "-",
                "Malicious (%)": f"{malicious_pct:.2f}%" if malicious_pct is not None else "Pendente",
                "Δ Malicious": f"{(malicious_pct - PAPER_SHORTCUT['malicious']):+.2f}%" if malicious_pct is not None else "-",
                "Média (%)": f"{average_pct:.2f}%" if average_pct is not None else "Pendente",
                "Δ Média": f"{(average_pct - PAPER_SHORTCUT['average']):+.2f}%" if average_pct is not None else "-",
            }
        )
    table1 = pd.DataFrame(table1_rows)
    print("\n📊 Tabela 1: Shortcut Mitigation (Artigo vs Modelos sem MOF)")
    print(table1.to_string(index=False))
    table1.to_csv(eval_dir / "shortcut_mitigation_table.csv", index=False)
    table1.to_csv(output_dir / "shortcut_mitigation_table.csv", index=False)

    # Tabela 2: ablação completa (sem mitigação vs Shortcut vs MOF).
    ablation = load_ablation_results(output_dir)
    ablation_designs = [
        ("Artigo Oficial", "+ Aug (Sem Mitigação)", PAPER_UNAUGMENTED["overdefense"], PAPER_UNAUGMENTED["malicious"], PAPER_UNAUGMENTED["average"]),
        ("Artigo Oficial", "+ Aug + Shortcut Mitigation (Sem MOF)", PAPER_SHORTCUT["overdefense"], PAPER_SHORTCUT["malicious"], PAPER_SHORTCUT["average"]),
        ("Artigo Oficial", "+ Aug + MOF (Scratch Retrain)", PAPER_MOF["overdefense"], PAPER_MOF["malicious"], PAPER_MOF["average"]),
        ("Vertente Oficial/Original", "+ Aug (Sem Mitigação)", *extract_metrics(ablation.get("train_augmented_original"))),
        ("Vertente Oficial/Original", "+ Aug + Shortcut Mitigation (Sem MOF)", *extract_metrics(evaluated_models.get("train_augmented_shortcut_original"))),
        ("Vertente Oficial/Original", "+ Aug + MOF (Scratch Retrain)", *extract_metrics(ablation.get("train_augmented_mof_original"))),
        ("Vertente Reproduzida", "+ Aug (Sem Mitigação)", *extract_metrics(ablation.get("train_augmented_rp"))),
        ("Vertente Reproduzida", "+ Aug + Shortcut Mitigation (Sem MOF)", *extract_metrics(evaluated_models.get("train_augmented_shortcut_rp"))),
        ("Vertente Reproduzida", "+ Aug + MOF (Scratch Retrain)", *extract_metrics(ablation.get("train_augmented_mof_rp"))),
    ]
    table2_rows = []
    for vertente, design, overdefense, malicious, average in ablation_designs:
        table2_rows.append(
            {
                "Vertente": vertente,
                "Design de Treinamento": design,
                "Over-defense (%)": f"{overdefense:.2f}%" if overdefense is not None else "Pendente",
                "Malicious (%)": f"{malicious:.2f}%" if malicious is not None else "Pendente",
                "Média ((Overdef + Mal)/2)": f"{average:.2f}%" if average is not None else "Pendente",
            }
        )
    table2 = pd.DataFrame(table2_rows)
    print("\n📋 Tabela 2: Ablação Comparativa (Sem Mitigação vs Shortcut vs MOF)")
    print(table2.to_string(index=False))
    table2.to_csv(eval_dir / "shortcut_vs_mof_ablation.csv", index=False)
    table2.to_csv(output_dir / "shortcut_vs_mof_ablation.csv", index=False)

    # Gráfico comparativo.
    metrics_names = ["Over-defense (NotInject)", "Malicious (BIPIA)", "Média ((Overdef + Mal)/2)"]
    x = np.arange(len(metrics_names))
    series = [
        {"name": "Artigo: Shortcut Mitigation", "vals": [86.73, 65.63, 76.18], "color": "#264653"},
        {"name": "Artigo: MOF (PIGuard)", "vals": [87.32, 77.39, 82.36], "color": "#2A9D8F"},
    ]
    result_original = evaluated_models.get("train_augmented_shortcut_original", {})
    if result_original.get("overdefense_acc") is not None:
        series.append(
            {
                "name": "Shortcut Mitigation (Original)",
                "vals": [
                    result_original["overdefense_acc"] * 100,
                    result_original["malicious_acc"] * 100,
                    result_original["average_acc"] * 100,
                ],
                "color": "#E9C46A",
            }
        )
    result_rp = evaluated_models.get("train_augmented_shortcut_rp", {})
    if result_rp.get("overdefense_acc") is not None:
        series.append(
            {
                "name": "Shortcut Mitigation (Reproduzida)",
                "vals": [
                    result_rp["overdefense_acc"] * 100,
                    result_rp["malicious_acc"] * 100,
                    result_rp["average_acc"] * 100,
                ],
                "color": "#E76F51",
            }
        )
    if len(series) == 2:
        print(
            "\n⚠️ Aviso: as barras dos modelos de Shortcut Mitigation não foram "
            "exibidas porque suas métricas constam como Pendente. Execute o "
            "treinamento/avaliação correspondente."
        )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    number_series = len(series)
    width = 0.80 / number_series
    for index, item in enumerate(series):
        offset = (index - number_series / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, item["vals"], width, label=item["name"],
            color=item["color"], edgecolor="black", linewidth=0.7,
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                )
    ax.set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Ablação: Shortcut Mitigation (Reweighting) vs. MOF (Over-defense & Malicious)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    for path in (
        eval_dir / "shortcut_vs_mof_comparison.png",
        output_dir / "shortcut_vs_mof_comparison.png",
    ):
        fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico de Shortcut Mitigation salvo em: {eval_dir / 'shortcut_vs_mof_comparison.png'}")

    return evaluated_models


def download_tensor_trust(dataset_root: Path) -> dict[str, list[dict[str, object]]]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_files = {
        "hijacking": dataset_root / "tensor_trust_hijacking.json",
        "extraction": dataset_root / "tensor_trust_extraction.json",
    }
    combined_path = dataset_root / "tensor_trust.json"
    converted: dict[str, list[dict[str, object]]] = {}

    print("\n📥 Verificando benchmarks do Tensor Trust (qxcv/tensor-trust)...")
    for subset, url in TT_URLS.items():
        output_path = output_files[subset]
        if output_path.is_file():
            print(f"[CACHE] '{output_path}' já existe. Carregando...")
            with output_path.open("r", encoding="utf-8") as handle:
                converted[subset] = json.load(handle)
            continue

        print(f"Baixando subset '{subset}' de: {url}")
        samples: list[dict[str, object]] = []
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request) as response:
                for line in response:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    item = json.loads(line_str)
                    attack_prompt = str(item.get("attack", "")).strip()
                    if attack_prompt:
                        samples.append(
                            {
                                "prompt": attack_prompt,
                                "label": 1,
                                "source": f"tensor_trust_{subset}",
                            }
                        )
        except (urllib.error.URLError, TimeoutError) as exc:
            print(
                f"[AVISO] Falha ao baixar Tensor Trust ({exc}). "
                "A avaliação desse benchmark será pulada."
            )
            return {}

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, indent=4, ensure_ascii=False)
        converted[subset] = samples
        print(f"Salvo '{subset}': {len(samples)} amostras em {output_path}")

    all_samples = [sample for subset in ("hijacking", "extraction") for sample in converted.get(subset, [])]
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(all_samples, handle, indent=4, ensure_ascii=False)

    summary = pd.DataFrame(
        [
            {
                "Subconjunto": "Prompt Hijacking (v1)",
                "Amostras": len(converted.get("hijacking", [])),
                "Label": "1 (Ataque)",
                "Arquivo": str(output_files["hijacking"]),
            },
            {
                "Subconjunto": "Prompt Extraction (v1)",
                "Amostras": len(converted.get("extraction", [])),
                "Label": "1 (Ataque)",
                "Arquivo": str(output_files["extraction"]),
            },
            {
                "Subconjunto": "Tensor Trust Consolidado",
                "Amostras": len(all_samples),
                "Label": "1 (Ataque)",
                "Arquivo": str(combined_path),
            },
        ]
    )
    print(summary.to_string(index=False))
    return converted


def evaluate_predictions_batch(
    model,
    tokenizer,
    texts: Sequence[str],
    target_label: int = 1,
    batch_size: int = 32,
    is_hf_model: bool = False,
    device: str = "cuda",
) -> float:
    if not texts:
        return 0.0
    correct = 0
    total = len(texts)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            if is_hf_model:
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            else:
                classifier_dtype = next(model.classifier.parameters()).dtype
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


def run_tensor_trust_section(
    output_dir: Path,
    dataset_root: Path,
    batch_size: int,
    device: str,
    force_reevaluate: bool,
    tt_subsets: dict[str, list[dict[str, object]]],
) -> None:
    if not tt_subsets:
        print("[AVISO] Tensor Trust indisponível; seção ignorada.")
        return

    eval_dir = REPO_ROOT / "Eval/TensorTrust"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "tensor_trust_results.json"
    output_results_path = output_dir / "tensor_trust_results.json"
    if output_results_path.is_file() and not results_path.is_file():
        shutil.copy2(output_results_path, results_path)

    tt_results_data: dict[str, object] = {}
    if results_path.is_file() and not force_reevaluate:
        try:
            with results_path.open("r", encoding="utf-8") as handle:
                tt_results_data = json.load(handle)
            print("\n🔒 [CACHE ATIVO] Resultados do Tensor Trust carregados.")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Aviso ao ler cache do Tensor Trust: {exc}")

    evaluated_models: dict[str, dict[str, object]] = tt_results_data.get(
        "evaluated_models", {}
    )  # type: ignore[assignment]
    hjacking_prompts = [str(item["prompt"]) for item in tt_subsets.get("hijacking", [])]
    extraction_prompts = [str(item["prompt"]) for item in tt_subsets.get("extraction", [])]
    models = build_tensor_trust_models(output_dir)

    tt_status_rows = []
    for key, info in models.items():
        if info["type"] == "huggingface":
            status = "🌐 Hugging Face Hub"
            resolved_path = str(info["repo_id"])
        else:
            resolved_path_obj = resolve_existing(info["candidates"])  # type: ignore[arg-type]
            resolved_path = str(resolved_path_obj)
            status = "✅ Encontrado" if resolved_path_obj.is_file() else "❌ Ausente"
        tt_status_rows.append(
            {
                "Identificador": key,
                "Descrição": info["name"],
                "Vertente": info["vertente"],
                "Status": status,
                "Caminho / Origem": resolved_path,
            }
        )
    print("\n=== Status dos Modelos Avaliados no Tensor Trust ===")
    print(pd.DataFrame(tt_status_rows).to_string(index=False))

    for key, info in models.items():
        if key in evaluated_models and not force_reevaluate:
            print(f"[CACHE] Modelo '{key}' já avaliado no Tensor Trust. Pulando.")
            continue

        print("\n" + "=" * 65)
        print(f"🛡️ Avaliando no Tensor Trust: {key} ({info['name']})")
        print("=" * 65)

        if info["type"] == "huggingface":
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    info["repo_id"], trust_remote_code=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    info["repo_id"], trust_remote_code=True
                ).to(device)
                model.eval()
                is_hf = True
            except Exception as exc:  # rede, acesso restrito ou modelo incompatível
                print(f"[ERRO] Falha ao carregar modelo HF '{info['repo_id']}': {exc}")
                continue
        else:
            checkpoint_path = resolve_existing(info["candidates"])  # type: ignore[arg-type]
            if not checkpoint_path.is_file():
                print(f"[AVISO] Checkpoint de '{key}' não encontrado em: {checkpoint_path}. Pulando...")
                continue
            try:
                PIGuard = load_piguard_class()
                tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
                model = PIGuard("microsoft/deberta-v3-base", num_labels=2, device=device)
                state_dict = torch.load(checkpoint_path, map_location=device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                state_dict = {
                    state_key.replace("module.", ""): value
                    for state_key, value in state_dict.items()
                }
                model.load_state_dict(state_dict, strict=False)
                model = model.to(device=device, dtype=torch.float32)
                model.eval()
                is_hf = False
            except Exception as exc:
                print(f"[ERRO] Falha ao carregar checkpoint '{checkpoint_path}': {exc}")
                continue

        hijacking_acc = evaluate_predictions_batch(
            model, tokenizer, hjacking_prompts,
            target_label=1, batch_size=batch_size,
            is_hf_model=is_hf, device=device,
        )
        extraction_acc = evaluate_predictions_batch(
            model, tokenizer, extraction_prompts,
            target_label=1, batch_size=batch_size,
            is_hf_model=is_hf, device=device,
        )
        average_acc = (hijacking_acc + extraction_acc) / 2.0
        evaluated_models[key] = {
            "hijacking_acc": hijacking_acc,
            "extraction_acc": extraction_acc,
            "average_acc": average_acc,
            "samples_count": {
                "hijacking": len(hjacking_prompts),
                "extraction": len(extraction_prompts),
                "total": len(hjacking_prompts) + len(extraction_prompts),
            },
        }
        print(f"Resultados para {key}:")
        print(f"  - Prompt Hijacking ACC:  {hijacking_acc:.2%}")
        print(f"  - Prompt Extraction ACC: {extraction_acc:.2%}")
        print(f"  - Média dos Subconjuntos: {average_acc:.2%}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_payload = {
        "dataset": "qxcv/tensor-trust (benchmarks v1)",
        "description": "Detecção de ataques de hijacking e extraction (label=1)",
        "metric_formula": "average_acc = (hijacking_acc + extraction_acc) / 2",
        "evaluated_models": evaluated_models,
    }
    for path in (results_path, output_results_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=4, ensure_ascii=False)
    print(f"\n💾 Resultados do Tensor Trust salvos em: {results_path}")

    # Tabela comparativa.
    reference_key = "PIGuard (Hugging Face - leolee99)"
    reference_result = evaluated_models.get(reference_key, {})
    reference_average = reference_result.get("average_acc")
    rows = []
    for key, info in models.items():
        result = evaluated_models.get(key, {})
        hijacking = result.get("hijacking_acc")
        extraction = result.get("extraction_acc")
        average = result.get("average_acc")
        if average is not None and reference_average is not None and key != reference_key:
            delta = f"{(average - reference_average) * 100.0:+.2f}%"
        elif key == reference_key and average is not None:
            delta = "Referência (0.00%)"
        else:
            delta = "-"
        rows.append(
            {
                "Modelo": key,
                "Vertente / Descrição": info["vertente"],
                "Prompt Hijacking (%)": f"{hijacking * 100.0:.2f}%" if hijacking is not None else "Pendente",
                "Prompt Extraction (%)": f"{extraction * 100.0:.2f}%" if extraction is not None else "Pendente",
                "Média dos Subconjuntos (%)": f"{average * 100.0:.2f}%" if average is not None else "Pendente",
                "Δ vs leolee99/PIGuard": delta,
            }
        )
    table = pd.DataFrame(rows)
    print("\n📊 Tabela: Avaliação Comparativa no Tensor Trust")
    print(table.to_string(index=False))
    table.to_csv(eval_dir / "tensor_trust_comparison.csv", index=False)
    table.to_csv(output_dir / "tensor_trust_comparison.csv", index=False)

    # Gráfico comparativo.
    chart_models: list[str] = []
    hijacking_values: list[float] = []
    extraction_values: list[float] = []
    average_values: list[float] = []
    for key, info in models.items():
        result = evaluated_models.get(key, {})
        if result.get("average_acc") is None:
            continue
        chart_models.append(
            key.replace("PIGuard (Hugging Face - leolee99)", "PIGuard (Oficial HF)")
            .replace("train_augmented_mof_", "MOF ")
        )
        hijacking_values.append(result["hijacking_acc"] * 100.0)  # type: ignore[operator]
        extraction_values.append(result["extraction_acc"] * 100.0)  # type: ignore[operator]
        average_values.append(result["average_acc"] * 100.0)  # type: ignore[operator]

    if chart_models:
        x = np.arange(len(chart_models))
        width = 0.25
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
        bars_hijacking = ax.bar(
            x - width, hijacking_values, width,
            label="Prompt Hijacking", color="#457B9D",
            edgecolor="black", linewidth=0.6,
        )
        bars_extraction = ax.bar(
            x, extraction_values, width,
            label="Prompt Extraction", color="#E76F51",
            edgecolor="black", linewidth=0.6,
        )
        bars_average = ax.bar(
            x + width, average_values, width,
            label="Média dos Subconjuntos", color="#2A9D8F",
            edgecolor="black", linewidth=0.8,
        )
        for bars in (bars_hijacking, bars_extraction, bars_average):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold",
                    )
        ax.set_ylabel("Acurácia de Detecção de Injeção (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Desempenho no Benchmark Tensor Trust: Modelos MOF vs. leolee99/PIGuard",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(chart_models, fontsize=10, fontweight="bold", rotation=10, ha="right")
        ax.set_ylim(0, 115)
        ax.legend(frameon=True, fontsize=10, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        for path in (
            eval_dir / "tensor_trust_comparison.png",
            output_dir / "tensor_trust_comparison.png",
        ):
            fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico do Tensor Trust salvo em: {eval_dir / 'tensor_trust_comparison.png'}")
    else:
        print("[AVISO] Nenhum modelo teve métricas no Tensor Trust; gráfico não gerado.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Avalia Shortcut Mitigation e/ou Tensor Trust.",
    )
    parser.add_argument(
        "--benchmark",
        choices=("all", "shortcut", "tensor-trust"),
        default="all",
        help="Benchmark a executar. Padrão: all.",
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
        help="Ignora caches e recalcula as inferências.",
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
    print(f"Device em uso: {device}")
    print(f"Benchmark: {args.benchmark}")

    if args.benchmark in ("all", "shortcut"):
        run_shortcut_section(
            output_dir=output_dir,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            device=device,
            force_reevaluate=args.force_reevaluate,
        )

    if args.benchmark in ("all", "tensor-trust"):
        tt_subsets = download_tensor_trust(dataset_root)
        run_tensor_trust_section(
            output_dir=output_dir,
            dataset_root=dataset_root,
            batch_size=args.batch_size,
            device=device,
            force_reevaluate=args.force_reevaluate,
            tt_subsets=tt_subsets,
        )

    print("\n" + "=" * 75)
    print("Avaliação concluída.")
    print(f"Artefatos em: {output_dir}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

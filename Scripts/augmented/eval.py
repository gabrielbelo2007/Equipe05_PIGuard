"""PIGuard - Validação Comparativa dos Modelos e Análise de Over-Defense.

Equivalente ao notebook ``Notebooks/first_train/eval.ipynb``, adaptado para
execução direta como arquivo ``.py`` e sem qualquer dependência do Google Colab.

Uso:
    python Notebooks/first_train/eval.py

Avalia os 3 modelos treinados contra os benchmarks oficiais do artigo
(NotInject, WildGuard e BIPIA) e salva tabelas, gráficos e resultados brutos em
``./outputs`` (configurável pela variável ``PIGUARD_OUTPUT_DIR``).
"""

import json
import os
import sys

# Resolução de caminhos 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

OUTPUT_DIR = os.environ.get("PIGUARD_OUTPUT_DIR", os.path.join(REPO_DIR, "outputs"))

import pandas as pd  

# Backend não interativo: permite gerar gráficos em servidores sem display.
import matplotlib  

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

EXPERIMENTS = {
    "Modelo 1 (train_augmented_original)": {
        "candidates": [
            os.path.join(OUTPUT_DIR, "train_augmented_original", "best_model.pth"),
            os.path.join(REPO_DIR, "checkpoints", "train_augmented_original", "best_model.pth"),
            "checkpoints/train_augmented_original/best_model.pth",
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
            "checkpoints/train_augmented_rp/best_model.pth",
        ],
        "trigger_candidates": [
            os.path.join(OUTPUT_DIR, "train_augmented_original", "trigger_words_train_augmented_oficial.json"),
            os.path.join(REPO_DIR, "MOF", "trigger_words_train_augmented_oficial.json"),
            "MOF/trigger_words_train_augmented_oficial.json",
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "trigger_words_train_augmented.json"),
            "MOF/trigger_words_train_augmented.json",
        ],
        "description": "Baseline + LLM Augmented Oficial do Artigo (PIGuard oficial)",
    },
    "Modelo 2 (train_augmented_rp)": {
        "candidates": [
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
            os.path.join(REPO_DIR, "checkpoints", "train_augmented_rp", "best_model.pth"),
            "checkpoints/train_augmented_rp/best_model.pth",
        ],
        "trigger_candidates": [
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "trigger_words_train_augmented.json"),
            os.path.join(REPO_DIR, "MOF", "trigger_words_train_augmented.json"),
            "MOF/trigger_words_train_augmented.json",
        ],
        "description": "Baseline + LLM Augmented Reproduzido (generate_data_oficial.py - sem filtros)",
    },
    "Modelo 3 (train_cleaned_original)": {
        "candidates": [
            os.path.join(OUTPUT_DIR, "train_cleaned_original", "best_model.pth"),
            os.path.join(REPO_DIR, "checkpoints", "train_cleaned_original", "best_model.pth"),
            "checkpoints/train_cleaned_original/best_model.pth",
            os.path.join(OUTPUT_DIR, "train_cleaned", "best_model.pth"),
            "checkpoints/train_cleaned/best_model.pth",
        ],
        "trigger_candidates": [
            os.path.join(OUTPUT_DIR, "train_cleaned_original", "trigger_words_train_cleaned_oficial.json"),
            os.path.join(REPO_DIR, "MOF", "trigger_words_train_cleaned_oficial.json"),
            "MOF/trigger_words_train_cleaned_oficial.json",
            os.path.join(OUTPUT_DIR, "train_cleaned", "trigger_words_train_cleaned.json"),
            "MOF/trigger_words_train_cleaned.json",
        ],
        "description": "Baseline Limpo Oficial sem cauda longa (sem aumento de dados)",
    },
}


def resolve_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def load_piguard_class():
    """Importa a arquitetura PIGuard com fallback autossuficiente."""
    try:
        from original.PIGuard import PIGuard

        print("PIGuard importado com sucesso a partir de PIGuard.py")
        return PIGuard
    except ImportError:
        print(
            "Aviso: PIGuard.py não encontrado no path. "
            "Usando definição autossuficiente da arquitetura PIGuard..."
        )

        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        class PIGuard(nn.Module):
            def __init__(self, model_name, num_labels, device):
                super(PIGuard, self).__init__()
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
                last_hidden_state = outputs["last_hidden_state"]
                pooled_output = last_hidden_state[:, 0, :].to(self.classifier.weight.dtype)
                logits = self.classifier(pooled_output)
                loss = 0
                if labels is not None:
                    loss = self.loss_fct(
                        logits.view(-1, self.classifier.out_features), labels.view(-1)
                    )
                return logits, loss

            def classify(self, input_text):
                tokenzied_text = self.tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=256
                )
                input_ids = tokenzied_text["input_ids"].to(self.device)
                attention_mask = tokenzied_text["attention_mask"].to(self.device)
                outputs = self.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )["last_hidden_state"]
                pooled_output = outputs[:, 0, :].to(self.classifier.weight.dtype)
                return self.classifier(pooled_output)

        return PIGuard


def evaluate_dataset_batch(model, tokenizer, text_samples, target_label, batch_size=32, device="cuda"):
    """Avalia um conjunto de textos em lotes, com atenção mascarada e truncamento."""
    import torch

    if not text_samples:
        return 0.0
    correct = 0
    total = len(text_samples)
    classifier_dtype = next(model.classifier.parameters()).dtype
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = text_samples[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            outputs = model.deberta(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
            )["last_hidden_state"]
            pooled = outputs[:, 0, :].to(dtype=classifier_dtype)
            logits = model.classifier(pooled)
            preds = logits.argmax(dim=-1).cpu().tolist()
            correct += sum(1 for p in preds if p == target_label)
    return correct / total


def find_dataset_root(dataset_root="datasets"):
    """Localiza a pasta datasets de forma robusta em qualquer ambiente local."""
    candidates = [
        dataset_root,
        "datasets",
        os.path.join(REPO_DIR, "datasets"),
        os.path.join(os.getcwd(), "datasets"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.exists(os.path.join(candidate, "wildguard.json")):
            return candidate
    return dataset_root


def run_full_evaluation(model_path, dataset_root="datasets", batch_size=32, device="cuda"):
    """Executa a bateria completa de testes oficiais do PIGuard.

    - NotInject 1, 2, 3 (Over-Defense Benchmark: target_label = 0 [benign])
    - WildGuard (Benign Benchmark: target_label = 0 [benign])
    - BIPIA Text & BIPIA Code (Injection Benchmark: target_label = 1 [injection])
    """
    import torch
    from transformers import AutoTokenizer

    PIGuard = load_piguard_class()

    dataset_root = find_dataset_root(dataset_root)
    print("\nCarregando modelo a partir de: {}".format(model_path))
    print("Pasta de datasets resolvida em: {}".format(dataset_root))

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = PIGuard("microsoft/deberta-v3-base", num_labels=2, device=device)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    results = {}

    # 1. NotInject Benchmark
    notinject_accs = []
    for sub in ["one", "two", "three"]:
        path = os.path.join(dataset_root, "NotInject_{}.json".format(sub))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = [x["prompt"] for x in data]
            acc = evaluate_dataset_batch(
                model, tokenizer, texts, target_label=0, batch_size=batch_size, device=device
            )
            results["NotInject_{}_acc".format(sub)] = acc
            notinject_accs.append(acc)
        else:
            results["NotInject_{}_acc".format(sub)] = None
    results["NotInject_overall_acc"] = (
        sum(notinject_accs) / len(notinject_accs) if notinject_accs else 0.0
    )

    # 2. WildGuard Benchmark
    wild_path = os.path.join(dataset_root, "wildguard.json")
    if os.path.exists(wild_path):
        with open(wild_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [x["prompt"] for x in data]
        results["WildGuard_acc"] = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=0, batch_size=batch_size, device=device
        )
    else:
        results["WildGuard_acc"] = 0.0

    # 3. BIPIA Benchmark
    bipia_text_path = os.path.join(dataset_root, "BIPIA_text.json")
    if os.path.exists(bipia_text_path):
        with open(bipia_text_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [item for sublist in data.values() for item in sublist]
        results["BIPIA_text_acc"] = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=1, batch_size=batch_size, device=device
        )
    else:
        results["BIPIA_text_acc"] = 0.0

    bipia_code_path = os.path.join(dataset_root, "BIPIA_code.json")
    if os.path.exists(bipia_code_path):
        with open(bipia_code_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [item for sublist in data.values() for item in sublist]
        results["BIPIA_code_acc"] = evaluate_dataset_batch(
            model, tokenizer, texts, target_label=1, batch_size=batch_size, device=device
        )
    else:
        results["BIPIA_code_acc"] = 0.0

    results["BIPIA_overall_acc"] = (
        results["BIPIA_text_acc"] + results["BIPIA_code_acc"]
    ) / 2

    # 4. Overall Accuracy
    results["Overall_acc"] = (
        results["NotInject_overall_acc"]
        + results["WildGuard_acc"]
        + results["BIPIA_overall_acc"]
    ) / 3

    return results


def main():
    import torch

    # 1. Verificação de hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device em uso para avaliação: {}".format(device))

    # 2. Status de presença dos modelos e metadados
    print("=== Status de Presença dos Modelos e Metadados ===")
    status_list = []
    for name, info in EXPERIMENTS.items():
        ckpt_path = resolve_existing(info["candidates"])
        trig_path = resolve_existing(info["trigger_candidates"])
        ckpt_found = os.path.exists(ckpt_path)
        trig_found = os.path.exists(trig_path)
        num_trigs = "N/A"
        if trig_found:
            try:
                with open(trig_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                num_trigs = len(data) if isinstance(data, list) else len(data.keys())
            except Exception:
                pass
        status_list.append(
            {
                "Experimento": name,
                "Checkpoint": "Encontrado" if ckpt_found else "Ausente",
                "Caminho Resolvido": ckpt_path if ckpt_found else info["candidates"][0],
                "Trigger Words Salvas": "{} tokens".format(num_trigs) if trig_found else "Ausente",
            }
        )

    status_df = pd.DataFrame(status_list)
    print(status_df.to_string(index=False))

    # 3. Execução da avaliação de todos os modelos
    all_results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name, info in EXPERIMENTS.items():
        ckpt_path = resolve_existing(info["candidates"])
        trig_path = resolve_existing(info["trigger_candidates"])
        if not os.path.exists(ckpt_path):
            print("\n[AVISO] Checkpoint para '{}' não encontrado em {}. Pulando...".format(name, ckpt_path))
            continue
        print("\n================ Avaliando {} ================".format(name))
        res = run_full_evaluation(ckpt_path, dataset_root="datasets", batch_size=32, device=device)
        num_trigs = 0
        if os.path.exists(trig_path):
            try:
                with open(trig_path, "r", encoding="utf-8") as f:
                    trig_data = json.load(f)
                num_trigs = len(trig_data) if isinstance(trig_data, list) else len(trig_data.keys())
            except Exception:
                pass
        res["trigger_tokens_count"] = num_trigs
        all_results[name] = res
        print("Resultados parciais para {}:".format(name))
        print("  - Over-Defense ACC (NotInject): {:.2%}".format(res["NotInject_overall_acc"]))
        print("  - Benign ACC (WildGuard):        {:.2%}".format(res["WildGuard_acc"]))
        print("  - Injection ACC (BIPIA):         {:.2%}".format(res["BIPIA_overall_acc"]))
        print("  - Overall ACC:                   {:.2%}".format(res["Overall_acc"]))
        print("  - Trigger Tokens Extraídos:      {}".format(num_trigs))

    # 4. Tabela comparativa com os modelos do artigo e diagnóstico
    PAPER_CLEAN = {
        "NotInject_overall_acc": 0.7522,
        "WildGuard_acc": 0.7853,
        "BIPIA_overall_acc": 0.7017,
        "Overall_acc": 0.7464,
        "trigger_tokens_count": "Referência Limpa",
    }

    PAPER_AUGMENTED = {
        "NotInject_overall_acc": 0.6331,
        "WildGuard_acc": 0.8136,
        "BIPIA_overall_acc": 0.7595,
        "Overall_acc": 0.7387,
        "trigger_tokens_count": "Alto Viés (Queda Over-defense)",
    }

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

    table_data = []
    for label, key, is_pct in metrics_rows:
        row = {"Métrica de Avaliação": label}
        val_clean = PAPER_CLEAN.get(key)
        if val_clean is not None:
            row["Modelo Limpo (Artigo)"] = (
                "{:.2%}".format(val_clean) if (is_pct and isinstance(val_clean, float)) else str(val_clean)
            )
        else:
            row["Modelo Limpo (Artigo)"] = "-"
        val_aug = PAPER_AUGMENTED.get(key)
        if val_aug is not None:
            row["Modelo Augmented (Artigo)"] = (
                "{:.2%}".format(val_aug) if (is_pct and isinstance(val_aug, float)) else str(val_aug)
            )
        else:
            row["Modelo Augmented (Artigo)"] = "-"
        for exp_name in EXPERIMENTS.keys():
            if exp_name in all_results:
                val = all_results[exp_name].get(key)
                if val is None:
                    row[exp_name] = "-"
                elif is_pct and isinstance(val, (int, float)):
                    row[exp_name] = "{:.2%}".format(val)
                else:
                    row[exp_name] = str(val)
            else:
                row[exp_name] = "Pendente"
        table_data.append(row)

    df_comparison = pd.DataFrame(table_data)
    print("\n📊 Tabela Comparativa: Modelos do Artigo vs Modelos Treinados")
    print(df_comparison.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_comparison_table.csv")
    json_path = os.path.join(OUTPUT_DIR, "evaluation_comparison_raw.json")
    df_comparison.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "paper_clean": PAPER_CLEAN,
                "paper_augmented": PAPER_AUGMENTED,
                "experiments": all_results,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    print("\nTabela comparativa salva em: {}".format(csv_path))

    # 5. Criação do gráfico
    metrics = [
        "Overdefense (NotInject)",
        "Benign (WildGuard)",
        "Malicioso (BIPIA)",
        "Média Geral",
    ]
    chart_series = [
        {
            "name": "Modelo Limpo (Artigo)",
            "vals": [75.22, 78.53, 70.17, 74.64],
            "color": "#6C757D",
            "hatch": "//",
        },
        {
            "name": "Modelo Augmented (Artigo)",
            "vals": [63.31, 81.36, 75.95, 73.87],
            "color": "#E76F51",
            "hatch": "\\\\",
        },
    ]

    trained_colors = {
        "Modelo 1 (train_augmented_original)": "#F4A261",
        "Modelo 2 (train_augmented_rp)": "#2A9D8F",
        "Modelo 3 (train_cleaned_original)": "#264653",
    }

    for exp_name, color in trained_colors.items():
        if exp_name in all_results:
            res = all_results[exp_name]
            vals = [
                res.get("NotInject_overall_acc", 0.0) * 100,
                res.get("WildGuard_acc", 0.0) * 100,
                res.get("BIPIA_overall_acc", 0.0) * 100,
                res.get("Overall_acc", 0.0) * 100,
            ]
            chart_series.append(
                {"name": exp_name, "vals": vals, "color": color, "hatch": None}
            )

    num_models = len(chart_series)
    x = np.arange(len(metrics))
    width = 0.82 / num_models

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, item in enumerate(chart_series):
        offset = (idx - num_models / 2 + 0.5) * width
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
            yval = bar.get_height()
            if yval > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yval + 1.2,
                    "{:.1f}%".format(yval),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    rotation=0,
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
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close(fig)
    print("Gráfico comparativo salvo com sucesso em: {}".format(chart_path))

    print("\nAvaliação concluída.")


if __name__ == "__main__":
    main()

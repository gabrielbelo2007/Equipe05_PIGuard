"""PIGuard - Validação Comparativa Geral (Tabela 1 do Artigo) + variantes MOF.

Equivalente ao notebook ``Notebooks/second_train/eval_table1.ipynb``, adaptado
para execução direta como arquivo ``.py`` e sem qualquer dependência do Google
Colab.

Uso:
    python Notebooks/second_train/eval_table1.py

A chave do OpenRouter é lida das variáveis de ambiente ``OPENROUTER_API_KEY`` ou
``OPENAI_API_KEY`` (ou de um arquivo ``.env`` na raiz do repositório). Sem chave,
os LLMs via OpenRouter são pulados automaticamente.

Resultados, tabelas e gráficos são gravados em ``./logs/Table1`` e replicados em
``./outputs`` (configurável pela variável ``PIGUARD_OUTPUT_DIR``).
"""

import json
import math
import os
import shutil
import subprocess
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

# Valores oficiais do artigo (Tabela 1)
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

# Mapeamento dos modelos avaliados para a referência correspondente no artigo
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


def run_command(cmd):
    """Executa um comando externo a partir da raiz do repositório."""
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
    print("\n$ {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, cwd=REPO_DIR, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Comando falhou (código de saída {}): {}".format(
                result.returncode, " ".join(cmd)
            )
        )
    return result.returncode


def load_results_with_ablation_fallback(raw_key, results_raw):
    """Obtém os resultados de um modelo, com fallback para a ablação MOF."""
    r_data = results_raw.get(raw_key, {})
    if not r_data and "train_augmented_mof" in raw_key:
        ablation_path = "logs/MOFs/ablation_results_raw.json"
        try:
            with open(ablation_path, "r", encoding="utf-8") as f_abl:
                abl_data = json.load(f_abl).get("evaluated_experiments", {})
            short_k = raw_key.replace("PIGuard (", "").rstrip(")")
            if short_k in abl_data:
                c = abl_data[short_k]
                r_data = {
                    "overdefense": c.get("NotInject_overall_acc", 0.0) * 100.0,
                    "benign": c.get("WildGuard_acc", 0.0) * 100.0,
                    "malicious": c.get("BIPIA_overall_acc", 0.0) * 100.0,
                    "average": c.get("Overall_acc", 0.0) * 100.0,
                }
        except Exception:
            pass
    return r_data


def main():
    import torch

    # 1. Verificação de hardware
    print("\n--- Informações de Hardware e PyTorch ---")
    print("CUDA Disponível: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Dispositivo: {}".format(torch.cuda.get_device_name(0)))
        print(
            "Total VRAM: {:.2f} GB".format(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
        )

    # 2. Diretório de saída e recuperação de resultados prévios
    print("\nDiretório de checkpoints/resultados: {}".format(OUTPUT_DIR))
    os.makedirs("logs/Table1", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SYNC_FILES = [
        "table1_validation_raw.json",
        "table1_validation_summary.csv",
        "table1_detailed_deltas.csv",
        "table1_comparison.png",
        "table1_all_models_comparison_grid.png",
        "openrouter_eval_cache.json",
    ]
    for s_file in SYNC_FILES:
        output_path = os.path.join(OUTPUT_DIR, s_file)
        local_path = os.path.join("logs/Table1", s_file)
        if os.path.exists(output_path) and not os.path.exists(local_path):
            shutil.copy(output_path, local_path)
            print("[SYNC OUTPUT -> LOCAL] Recuperado com sucesso: {}".format(s_file))

    # 3. Credenciais do OpenRouter e verificação dos datasets
    openrouter_key = None
    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_var)
        if val:
            openrouter_key = val.strip().strip("\"'")
            break

    if not openrouter_key and os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip().lower() in ("openrouter_api_key", "openai_api_key"):
                        openrouter_key = v.strip().strip("\"'")
                        break

    print("\n=== Status de Configuração ===")
    print(
        "OpenRouter API Key configurada: {}".format(
            "Sim" if openrouter_key else "Não (LLMs via OpenRouter serão pulados se vazia)"
        )
    )

    print("\n--- Verificação dos Datasets de Avaliação ---")
    for ds in [
        "NotInject_one.json",
        "NotInject_two.json",
        "NotInject_three.json",
        "wildguard.json",
        "BIPIA_text.json",
        "BIPIA_code.json",
    ]:
        p = os.path.join("datasets", ds)
        print("[{}] {}".format("OK" if os.path.exists(p) else "AUSENTE", p))

    # 4. TRAVA DE SEGURANÇA CONTRA RE-EXECUÇÃO E GASTOS INDEVIDOS DE API

    # Se FORCE_RERUN = False:
    #   Verifica se os resultados já foram gerados e salvos (em OUTPUT_DIR ou em
    #   logs/Table1/). Se encontrados, PULA totalmente a inferência, garantindo
    #   zero chamadas de API (GPT-4o).
    # Se FORCE_RERUN = True:
    #   Força uma nova inferência de todos os modelos e sobrescreve os resultados.
    FORCE_RERUN = False

    raw_result_path = "logs/Table1/table1_validation_raw.json"
    output_raw_result = os.path.join(OUTPUT_DIR, "table1_validation_raw.json")

    # Recupera de OUTPUT_DIR se ainda não estiver no disco local.
    if os.path.exists(output_raw_result) and not os.path.exists(raw_result_path):
        os.makedirs("logs/Table1", exist_ok=True)
        shutil.copy(output_raw_result, raw_result_path)
        print("Resultados prévios recuperados de: {}".format(output_raw_result))

    if os.path.exists(raw_result_path) and not FORCE_RERUN:
        print("\n" + "=" * 80)
        print("[TRAVA DE SEGURANÇA ATIVA - INFERÊNCIA PULADA COM SUCESSO]")
        print("Resultados de inferência prévios encontrados em: {}".format(raw_result_path))
        print("Zero chamadas de API realizadas. Nenhum crédito de API foi gasto.")
        print("As etapas seguintes irão gerar todas as tabelas e gráficos comparativos")
        print("diretamente a partir dos dados salvos.")
        print("(Dica: Se quiser reexecutar uma nova inferência completa, altere 'FORCE_RERUN = True').")
        print("=" * 80 + "\n")
    else:
        print("\n[INICIANDO INFERÊNCIA DOS MODELOS]...")
        validate_script = os.path.join("Eval", "validate_models_table1.py")
        if not os.path.exists(validate_script):
            raise FileNotFoundError(
                "Script de validação não encontrado: {}".format(
                    os.path.join(REPO_DIR, validate_script)
                )
            )

        cmd = [sys.executable, validate_script]
        if openrouter_key:
            cmd.extend(["--openrouter_key", openrouter_key])
        else:
            cmd.append("--skip_openrouter")
        if FORCE_RERUN:
            cmd.append("--force_rerun")
        cmd.extend(["--output_dir", "logs/Table1"])
        run_command(cmd)

    # 5. Visualização dos resultados e tabela resumo
    raw_path = "logs/Table1/table1_validation_raw.json"
    csv_path = "logs/Table1/table1_validation_summary.csv"

    # Se o CSV não existir mas o JSON bruto estiver presente, reconstrói o resumo.
    if not os.path.exists(csv_path) and os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_d = json.load(f)
        rows = []
        for m_name, m in raw_d.items():
            rows.append(
                {
                    "Category": m.get("category", ""),
                    "Model": m_name,
                    "Over-defense (%)": round(m.get("overdefense", 0.0), 2),
                    "Benign (%)": round(m.get("benign", 0.0), 2),
                    "Malicious (%)": round(m.get("malicious", 0.0), 2),
                    "Average (%)": round(m.get("average", 0.0), 2),
                    "GFLOPs": m.get("gflops", ""),
                    "Inference (ms)": round(m.get("inference_ms", 0.0), 2),
                    "Efficiency": round(m.get("efficiency", 0.0), 2),
                }
            )
        df_gen = pd.DataFrame(rows)
        df_gen.to_csv(csv_path, index=False)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\n📊 Tabela 1: Avaliação Comparativa de Performance e Overhead")
        print(df.to_string(index=False))

        # Salva cópia no diretório de saída
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shutil.copy(csv_path, os.path.join(OUTPUT_DIR, "table1_validation_summary.csv"))
        if os.path.exists("logs/Table1/table1_validation_raw.json"):
            shutil.copy(
                "logs/Table1/table1_validation_raw.json",
                os.path.join(OUTPUT_DIR, "table1_validation_raw.json"),
            )
        if os.path.exists("logs/Table1/table1_comparison.png"):
            shutil.copy(
                "logs/Table1/table1_comparison.png",
                os.path.join(OUTPUT_DIR, "table1_comparison.png"),
            )
        print("\nArquivos de resultado sincronizados em: {}".format(OUTPUT_DIR))

    chart_path = "logs/Table1/table1_comparison.png"
    if os.path.exists(chart_path):
        print("\n📈 Visão Geral: Comparação Global de Modelos")
        print("Gráfico: {}".format(os.path.abspath(chart_path)))

    # 6. Tabela detalhada de comparação com deltas (Artigo vs Reproduzido)
    results_raw = {}
    raw_json_path = "logs/Table1/table1_validation_raw.json"
    if os.path.exists(raw_json_path):
        with open(raw_json_path, "r", encoding="utf-8") as f:
            results_raw = json.load(f)

    table_rows = []
    for disp_name, raw_key, paper_key in MODEL_MAPPING:
        p_data = PAPER_REF[paper_key]
        r_data = load_results_with_ablation_fallback(raw_key, results_raw)

        o_p = p_data["overdefense"]
        b_p = p_data["benign"]
        m_p = p_data["malicious"]
        a_p = p_data["average"]

        o_r = r_data.get("overdefense", None)
        b_r = r_data.get("benign", None)
        m_r = r_data.get("malicious", None)
        a_r = r_data.get("average", None)

        row = {
            "Modelo": disp_name,
            "Over-def (Artigo)": "{:.2f}%".format(o_p),
            "Over-def (Reprod)": "{:.2f}%".format(o_r) if o_r is not None else "N/D",
            "Δ Over-def": "{:+.2f}%".format(o_r - o_p) if o_r is not None else "-",
            "Benign (Artigo)": "{:.2f}%".format(b_p),
            "Benign (Reprod)": "{:.2f}%".format(b_r) if b_r is not None else "N/D",
            "Δ Benign": "{:+.2f}%".format(b_r - b_p) if b_r is not None else "-",
            "Malicious (Artigo)": "{:.2f}%".format(m_p),
            "Malicious (Reprod)": "{:.2f}%".format(m_r) if m_r is not None else "N/D",
            "Δ Malicious": "{:+.2f}%".format(m_r - m_p) if m_r is not None else "-",
            "Average (Artigo)": "{:.2f}%".format(a_p),
            "Average (Reprod)": "{:.2f}%".format(a_r) if a_r is not None else "N/D",
            "Δ Average": "{:+.2f}%".format(a_r - a_p) if a_r is not None else "-",
        }
        table_rows.append(row)

    df_deltas = pd.DataFrame(table_rows)
    print("\n📋 Comparação Lado a Lado: Artigo vs Reproduzido com Deltas (Δ)")
    print(df_deltas.to_string(index=False))

    os.makedirs("logs/Table1", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_deltas.to_csv("logs/Table1/table1_detailed_deltas.csv", index=False)
    df_deltas.to_csv(os.path.join(OUTPUT_DIR, "table1_detailed_deltas.csv"), index=False)
    print(
        "Tabela com deltas sincronizada em: {}".format(
            os.path.join(OUTPUT_DIR, "table1_detailed_deltas.csv")
        )
    )

    # 7. Gráficos individuais por modelo (Artigo vs Reproduzido)
    CHARTS_DIR = "logs/Table1/individual_charts"
    os.makedirs(CHARTS_DIR, exist_ok=True)
    OUTPUT_CHARTS_DIR = os.path.join(OUTPUT_DIR, "individual_charts")
    os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)

    metrics = ["Over-defense", "Benign", "Malicious", "Average"]
    width = 0.35

    print(
        "\n📊 Gráficos Comparativos Individuais por Modelo\n"
        "Para cada modelo avaliado, as colunas exibem lado a lado os resultados do "
        "Artigo (Original) e os obtidos na Reprodução."
    )

    for disp_name, raw_key, paper_key in MODEL_MAPPING:
        p_data = PAPER_REF[paper_key]
        r_data = load_results_with_ablation_fallback(raw_key, results_raw)

        paper_vals = [
            p_data["overdefense"],
            p_data["benign"],
            p_data["malicious"],
            p_data["average"],
        ]

        reprod_vals = [
            r_data.get("overdefense", 0.0),
            r_data.get("benign", 0.0),
            r_data.get("malicious", 0.0),
            r_data.get("average", 0.0),
        ]

        has_reprod = any(v > 0 for v in reprod_vals)
        if not has_reprod:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
        x = np.arange(len(metrics))

        rects1 = ax.bar(
            x - width / 2,
            paper_vals,
            width,
            label="Artigo (Original)",
            color="#1D3557",
            edgecolor="black",
            linewidth=0.6,
        )
        rects2 = ax.bar(
            x + width / 2,
            reprod_vals,
            width,
            label="Reproduzido (Obtido)",
            color="#E63946",
            edgecolor="black",
            linewidth=0.6,
        )

        avg_delta = reprod_vals[3] - paper_vals[3]
        delta_symbol = "▲" if avg_delta >= 0 else "▼"

        ax.set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            "{}\nArtigo (Original) vs Reproduzido (Obtido) | Δ Médio: {:+.2f}% {}".format(
                disp_name, avg_delta, delta_symbol
            ),
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 118)
        ax.legend(
            frameon=True,
            facecolor="#F8F9FA",
            edgecolor="#ADB5BD",
            fontsize=10,
            loc="upper right",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(
                "{:.1f}%".format(h),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#1D3557",
            )

        for i, rect in enumerate(rects2):
            h = rect.get_height()
            diff = h - paper_vals[i]
            diff_str = "({:+.1f}%)".format(diff) if abs(diff) >= 0.05 else ""
            ax.annotate(
                "{:.1f}%\n{}".format(h, diff_str),
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#E63946",
            )

        plt.tight_layout()

        slug = (
            disp_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace(".", "_")
        )
        chart_file = os.path.join(CHARTS_DIR, "compare_{}.png".format(slug))
        plt.savefig(chart_file, dpi=200, bbox_inches="tight")
        shutil.copy(
            chart_file, os.path.join(OUTPUT_CHARTS_DIR, "compare_{}.png".format(slug))
        )
        plt.close(fig)

    print("\nTodos os gráficos individuais foram salvos em: {}".format(CHARTS_DIR))
    print("Gráficos individuais sincronizados em: {}".format(OUTPUT_CHARTS_DIR))

    # 8. Mosaico panorâmico (grid comparativo de todos os modelos)
    valid_models = []
    for disp_name, raw_key, paper_key in MODEL_MAPPING:
        p_data = PAPER_REF[paper_key]
        r_data = load_results_with_ablation_fallback(raw_key, results_raw)

        reprod_vals = [
            r_data.get("overdefense", 0.0),
            r_data.get("benign", 0.0),
            r_data.get("malicious", 0.0),
            r_data.get("average", 0.0),
        ]
        if any(v > 0 for v in reprod_vals):
            paper_vals = [
                p_data["overdefense"],
                p_data["benign"],
                p_data["malicious"],
                p_data["average"],
            ]
            valid_models.append((disp_name, paper_vals, reprod_vals))

    n_models = len(valid_models)
    ncols = 2
    nrows = math.ceil(n_models / ncols) if n_models else 1

    fig_grid, axes_grid = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), dpi=200)
    axes_grid = np.atleast_1d(axes_grid).flatten()

    metrics_abbr = ["Overdef", "Benign", "Malicious", "Average"]
    width = 0.35

    for idx, (disp_name, p_vals, r_vals) in enumerate(valid_models):
        ax = axes_grid[idx]
        x_g = np.arange(len(metrics_abbr))

        ax.bar(
            x_g - width / 2,
            p_vals,
            width,
            label="Artigo",
            color="#1D3557",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x_g + width / 2,
            r_vals,
            width,
            label="Reproduzido",
            color="#E63946",
            edgecolor="black",
            linewidth=0.5,
        )

        delta_avg = r_vals[3] - p_vals[3]
        ax.set_title(
            "{} (Δ Avg: {:+.1f}%)".format(disp_name, delta_avg),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xticks(x_g)
        ax.set_xticklabels(metrics_abbr, fontsize=10)
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

        for i_g in range(len(metrics_abbr)):
            ax.annotate(
                "{:.0f}%".format(p_vals[i_g]),
                xy=(x_g[i_g] - width / 2, p_vals[i_g]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1D3557",
                fontweight="bold",
            )
            ax.annotate(
                "{:.0f}%".format(r_vals[i_g]),
                xy=(x_g[i_g] + width / 2, r_vals[i_g]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#E63946",
                fontweight="bold",
            )

    # Oculta eixos sobressalentes, se houver.
    for idx in range(n_models, len(axes_grid)):
        fig_grid.delaxes(axes_grid[idx])

    plt.suptitle(
        "Dashboard Consolidado: Artigo (Original) vs Reproduzido em Todos os Modelos",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    grid_chart_path = "logs/Table1/table1_all_models_comparison_grid.png"
    plt.savefig(grid_chart_path, dpi=250, bbox_inches="tight")
    shutil.copy(
        grid_chart_path,
        os.path.join(OUTPUT_DIR, "table1_all_models_comparison_grid.png"),
    )
    print(
        "\nMosaico consolidado sincronizado em: {}".format(
            os.path.join(OUTPUT_DIR, "table1_all_models_comparison_grid.png")
        )
    )
    plt.close(fig_grid)

    print("\nAvaliação da Tabela 1 concluída.")


if __name__ == "__main__":
    main()

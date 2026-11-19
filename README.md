# Reprodução Científica: PIGuard (ACL 2025)

Este repositório contém o projeto de reprodução experimental do artigo **"PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free"**, aceito na *ACL 2025* (Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics).

> **Referência original:**  
> Hao Li, Xiaogeng Liu, Ning Zhang, Chaowei Xiao. *PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free*. ACL 2025.  
> Repositório oficial dos autores: [leolee99/PIGuard](https://github.com/leolee99/PIGuard)

---

## 📌 1. Visão Geral e Objetivos da Reprodução

Os modelos de salvaguarda (*guardrails*) contra ataques de injeção de prompt (ex.: ProtectAIv2, PromptGuard, Fmops) frequentemente sofrem de **sobredefesa (*over-defense*)**: classificam entradas legítimas e benignas como ataques devido à dependência excessiva de atalhos (*shortcut learning*) associados a palavras de gatilho (*trigger words*, como *"ignore"*, *"cancel"*, *"system"* ou *"execute"*).

O artigo propõe duas contribuições centrais:

1. **Benchmark NotInject**: dataset de avaliação com 339 amostras benignas desafiadoras contendo palavras de gatilho frequentes em ataques, divididas em 3 níveis de dificuldade (1, 2 e 3 palavras-gatilho).
2. **MOF (*Mitigating Over-defense for Free*)**: estratégia de treinamento sem datasets manuais de sobredefesa. Consiste em:
   - treinar uma base supervisionada inicial;
   - realizar uma varredura token por token (*token-wise recheck*) para detectar quais vocábulos isolados ativam o classificador como injeção;
   - sintetizar frases benignas contendo essas palavras com o auxílio de um LLM;
   - retreinar o modelo do zero (*retrain from scratch*) com essa ampliação de dados.

### Objetivos deste Trabalho

* [x] Replicar o pipeline de dados de treino básico e a ampliação de formatos em cauda longa (XML, CSV, JSON, Markdown).
* [x] Reproduzir o procedimento *token-wise recheck* para identificar tokens enviesados.
* [x] Implementar o treinamento do classificador baseado em `microsoft/deberta-v3-base`.
* [x] Implementar a avaliação e comparação contra baselines abertos no dataset **NotInject**, medindo acurácia nas 3 dimensões: **Benigna**, **Maliciosa** e **Sobredefesa**.

---

## 🏗️ 2. Estrutura do Projeto

```text
Equipe05_PIGuard/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Original/                         # Implementação de referência do PIGuard
│   ├── PIGuard.py                    # Arquitetura DeBERTa-v3 + classificador
│   ├── train.py                      # Treino base (Original/train.py)
│   ├── train_shortcut.py             # Treino com mitigação de atalhos
│   ├── eval.py                       # Avaliação local (NotInject, WildGuard, BIPIA, PINT opcional)
│   ├── eval_hf.py                    # Avaliação de modelos Hugging Face
│   ├── params.py                     # Argumentos de treino
│   └── util.py                       # Seed, logger, acurácia e conversão PINT
│
├── Notebooks/                        # Scripts das etapas de treino e avaliação
│   ├── first_train/
│   │   ├── train.py                  # 3 treinos da primeira fase
│   │   └── eval.py                   # Avaliação dos 3 modelos da primeira fase
│   ├── second_train/
│   │   ├── train.py                  # 9 treinos do estudo de ablação com MOF
│   │   ├── eval.py                   # Reconstrução da Tabela 2
│   │   └── eval_table1.py            # Tabela 1 + variantes MOF
│   └── train_extras/
│       ├── train.py                  # 2 treinos de Shortcut Mitigation
│       └── eval.py                   # Shortcut Mitigation + Tensor Trust
│
├── Eval/                             # Avaliação consolidada e validadores
│   ├── validate_models_table1.py     # Validador da Tabela 1 e baselines
│   └── reproduction_validation.py    # 5 blocos de validação da reprodução
│
├── Corrections/                      # Pipeline independente de correções/verificação
│   ├── corrections.py                # Orquestrador: geração, treino, recheck, MOF e comparação
│   ├── generate_mof_data.py          # Geração do MOF a partir de trigger words
│   ├── renew_token_recheck.py        # Token-wise recheck sanitizado
│   └── Upgrade_augmented/
│       ├── new_generate_data.py      # Geração de 435 amostras em 17 categorias
│       ├── filters.py                # Quatro filtros de validação (biblioteca)
│       ├── filter_data.py            # Alias de compatibilidade para filters.py
│       └── LLM_augmented_new.json    # Artefato histórico; não é lido pelo fluxo novo
│
├── Upgrade/                          # Pipeline avançado: Novo MOF v2 + Laya English
│   ├── __init__.py                   # Constantes, utilitários e carregamento dos benchmarks
│   ├── Pipeline/
│   │   ├── upgrade_pipeline.py       # Orquestrador CLI completo (Etapa 0 + Tópicos 1–3)
│   │   ├── step0_new_mof_and_heldout.py  # Pool do Novo MOF (v2) e separação held-out disjunta
│   │   └── laya_english_cascade.py   # Cascata seletiva Laya English + calibração + Bootstrap
│   ├── topic1_rp_laya_evaluation.py  # Tópico 1: avaliação dos 5 modelos rp com Laya
│   ├── topic2_corrected_new_mof.py   # Tópico 2: treino scratch base corrigida + MOF v2
│   ├── topic3_consolidated_report.py # Tópico 3: consolidação de tabelas, ICs e gráficos
│   ├── corrected_comparison.py       # Comparação unificada da linhagem corrected
│   └── run_corrected_comparison.py   # Runner CLI para corrected_comparison.py
│
└── datasets/                         # Dados versionados
    ├── train.json                    # Base de treino combinada (76.735 amostras)
    ├── valid.json                    # Validação (144 amostras)
    ├── wildguard.json                # Benignos para avaliação (971 amostras)
    ├── NotInject_one.json            # NotInject 1 palavra-gatilho (113 amostras)
    ├── NotInject_two.json            # NotInject 2 palavras-gatilho (113 amostras)
    ├── NotInject_three.json          # NotInject 3 palavras-gatilho (113 amostras)
    ├── BIPIA_text.json               # BIPIA texto: 15 grupos × 5 (75 amostras)
    ├── BIPIA_code.json               # BIPIA código: 10 grupos × 5 (50 amostras)
    ├── tensor_trust.json             # Tensor Trust combinado (1.346 amostras)
    ├── Extra_dataset/
    │   ├── tensor_trust_hijacking.json   # Tensor Trust hijacking (776 amostras)
    │   └── tensor_trust_extraction.json  # Tensor Trust extraction (570 amostras)
    ├── Augmented/
    │   ├── __init__.py               # Marca Augmented como pacote Python
    │   ├── train_cleaned_original.json    # Base limpa oficial (75.538 amostras)
    │   ├── train_augmented_original.json  # Base + aumento oficial (75.973 amostras)
    │   ├── train_augmented_rp.json        # Base + aumento reproduzido (75.973 amostras)
    │   └── Cache/
    │       ├── build_curated.py           # Seleciona demonstrações do aumento
    │       ├── clean_train_dataset.py     # Gera a base limpa a partir de train.json
    │       ├── generate_data_rp.py        # Gerador de dados aumentados
    │       ├── LLM_augmented_rp.json      # 435 amostras aumentadas reproduzidas
    │       └── curated_prompt_injections.json  # 110 demonstrações
    └── MOF/
        ├── MOF_original.json              # 762 amostras MOF oficiais
        ├── MOF_semi_rp.json               # 762 amostras MOF semi-reproduzidas
        ├── MOF_rp.json                    # 762 amostras MOF reproduzidas
        ├── train_clean_mof_original.json  # Base limpa + MOF (76.300)
        ├── train_clean_mof_semi_rp.json   # Base limpa + MOF (76.300)
        ├── train_clean_mof_rp.json        # Base limpa + MOF (76.300)
        ├── train_augmented_mof_semi_rp.json  # Base aumentada + MOF (76.735)
        ├── train_augmented_mof_rp.json       # Base aumentada + MOF (76.735)
        └── Cache/
            ├── generate_mof_data.py       # Gerador de MOF
            ├── token_recheck.py           # Token-wise recheck de referência
            ├── trigger_words_train_augmented_rp.json      # 629 tokens
            └── trigger_words_train_augmented_semi_rp.json # 427 tokens
```

> **Artefatos gerados:** `outputs/`, `Corrections/outputs/`, `Upgrade/outputs/`, `checkpoints/`, `logs/`, `Eval/ShortcutMitigation/` e `Eval/TensorTrust/` são criados em tempo de execução. Checkpoints e modelos não são versionados (`.gitignore`); os resultados numéricos da reprodução também não acompanham este checkout.

---

## ⚙️ 3. Ambiente e Pré-requisitos

```text
Hardware usado como referência nos experimentos: NVIDIA A100 (80 GB de VRAM), Python 3.12
```

### Requisitos de Hardware

* **GPU**: mínimo de 16 GB de VRAM (NVIDIA T4, RTX 3090, A100 ou equivalente) para treinar o `DeBERTa-v3-base` com `batch_size=32` e `max_length=512`. Os scripts também executam em CPU, mas o treinamento completo é muito mais lento.
* **Espaço em disco**: cerca de 20 GB livres para datasets, cache do Hugging Face, checkpoints e resultados. Os dados versionados ocupam aproximadamente 370 MB; cada `best_model.pth` do DeBERTa-v3-base ocupa cerca de 440 MB.
* **Rede**: o primeiro uso baixa `microsoft/deberta-v3-base`, os baselines da Tabela 1 e, no módulo `Upgrade/`, `convaiinnovations/laya` do Hugging Face (o Tensor Trust já vem salvo em `datasets/Extra_dataset/`, sendo baixado apenas se os arquivos estiverem ausentes). Modelos restritos, como o Prompt Guard, exigem acesso concedido na conta do Hugging Face.

### Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/gabrielbelo2007/Equipe05_PIGuard.git
   cd Equipe05_PIGuard
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   ```

3. Instale as dependências (único arquivo para todo o repositório, incluindo `Upgrade/` com `ModernBERT-large` via `transformers==4.48.3`):
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

O `requirements.txt` inclui todas as dependências de todos os fluxos (`Original/`, `Notebooks/`, `Eval/`, `Corrections/` e `Upgrade/`). Para instalar manualmente apenas o conjunto mínimo em Python 3.10 a 3.12:

```bash
python -m pip install torch==2.4.0 transformers==4.48.3 datasets==2.21.0 pyarrow==17.0.0 \
    sentencepiece==0.2.0 protobuf==4.25.4 tiktoken==0.7.0 PyYAML==6.0.2 \
    numpy==1.26.3 pandas==2.2.2 matplotlib==3.9.2 tqdm==4.66.5 \
    scikit-learn==1.5.1
```

Para GPU NVIDIA, instale um driver compatível com a distribuição CUDA do PyTorch. Consulte as [instruções do PyTorch](https://pytorch.org/get-started/previous-versions/#v240).

### Credenciais

A geração de dados aumentados, a geração de MOF e parte da Tabela 1 usam a API do OpenRouter (modelo `openai/gpt-4o-mini-2024-07-18` por padrão). Informe a chave de uma destas formas:

```bash
export OPENROUTER_API_KEY="sua-chave"
# ou
export OPENAI_API_KEY="sua-chave"
```

Também é aceito um arquivo `.env` na raiz do projeto:

```dotenv
OPENROUTER_API_KEY=sua-chave
```

O `.env` é ignorado pelo Git. As avaliações locais e o token recheck não fazem chamadas de API.

---

## 🚀 4. Guia de Execução Passo a Passo

Para chamar as CLIs de outro diretório, use o caminho absoluto do script. Caminhos locais relativos passados como argumentos são resolvidos a partir da raiz do repositório. Use `--help` para consultar os argumentos de cada CLI.

### 4.1 Primeira fase: treino base

A primeira fase treina três checkpoints:

| Experimento | Dataset |
|---|---|
| `train_cleaned_original` | `datasets/Augmented/train_cleaned_original.json` |
| `train_augmented_original` | `datasets/Augmented/train_augmented_original.json` |
| `train_augmented_rp` | `datasets/Augmented/train_augmented_rp.json` |

```bash
python Notebooks/first_train/train.py
python Notebooks/first_train/train.py --experiment train_augmented_original
python Notebooks/first_train/train.py --force-retrain
```

Os checkpoints, logs e trigger words são gravados em `outputs/<experimento>/` (ou no diretório informado via `--output-dir`). Ao final, o token-wise recheck é executado para os modelos aumentados; use `--skip-token-recheck` para omiti-lo.

Os scripts `datasets/Augmented/Cache/clean_train_dataset.py`, `datasets/Augmented/Cache/build_curated.py` e `datasets/Augmented/Cache/generate_data_rp.py` compõem a preparação/geração da base aumentada. O gerador MOF (`datasets/MOF/Cache/generate_mof_data.py`) lê as trigger words da rodada reproduzida em `<saída>/train_augmented_rp/`. Se usou `--output-dir` no treino, passe o mesmo caminho ao gerador via `--training-output-dir`. A variante semi-reproduzida usa as trigger words oficiais de `datasets/MOF/Cache/`. Use `--skip_gen` para consolidar os conjuntos MOF já disponíveis localmente, sem credenciais.

### 4.2 Segunda fase: ablação com MOF

A segunda fase executa os nove treinamentos da Tabela 2, em três vertentes:

| Vertente | Significado |
|---|---|
| Original | datasets oficiais do artigo |
| Semi-Reprodução (`semi_rp`) | trigger words oficiais e geração parcialmente reproduzida |
| Reproduzida (`rp`) | pipeline inteiramente reproduzido |

```bash
python Notebooks/second_train/train.py
python Notebooks/second_train/train.py --experiment train_augmented_mof_original
python Notebooks/second_train/train.py --experiment train_clean_mof_rp --force-retrain
```

Os experimentos de fine-tuning (`train_augmented_mof_finetuning_*`) dependem dos checkpoints base da primeira fase. Os demais usam os datasets já consolidados em `datasets/MOF/`.

### 4.3 Shortcut Mitigation e benchmarks extras

```bash
python Notebooks/train_extras/train.py
python Notebooks/train_extras/train.py --experiment train_augmented_shortcut_rp
python Notebooks/train_extras/eval.py --benchmark shortcut
python Notebooks/train_extras/eval.py --benchmark tensor-trust
python Notebooks/train_extras/eval.py --benchmark all --force-reevaluate
```

O treino de shortcut mitigation usa `Original/train_shortcut.py` e procura as trigger words da primeira fase no mesmo diretório de saída. As listas versionadas servem como fallback. A avaliação extra carrega o Tensor Trust local quando disponível e grava resultados em `Eval/ShortcutMitigation/` e `Eval/TensorTrust/`.

### 4.4 Avaliações

```bash
# Modelos da primeira fase: comparação entre limpo, aumentado oficial e aumentado reproduzido
python Notebooks/first_train/eval.py

# Tabela 2: ablação completa com MOF
python Notebooks/second_train/eval.py

# Tabela 1 e variantes MOF (sem chamadas OpenRouter)
python Notebooks/second_train/eval_table1.py --skip-openrouter

# Validador direto da Tabela 1 (baselines Hugging Face + variantes PIGuard)
python Eval/validate_models_table1.py --skip_openrouter

# Cinco blocos de validação da reprodução (sem chamadas de API)
python Eval/reproduction_validation.py
```

* `Notebooks/first_train/eval.py` grava `evaluation_comparison_table.csv`, `evaluation_comparison_raw.json` e o gráfico comparativo no diretório de saída.
* `Notebooks/second_train/eval.py` grava `logs/MOFs/ablation_table_2_summary.csv`, `logs/MOFs/ablation_table_2_detailed_deltas.csv`, JSONs brutos e gráficos. Use `--force-reevaluate` para ignorar o cache.
* `Notebooks/second_train/eval_table1.py` busca os checkpoints e grava os resultados no diretório de saída da rodada, com cópias dos relatórios principais em `logs/Table1/`. Resultados com as mesmas entradas podem ser reaproveitados. Use `--force-rerun` para recalcular.
* `Eval/reproduction_validation.py` organiza os resultados em `outputs/Gráficos/` com cinco blocos: Tabelas 1 e 7, Tabela 2 (desenhos 1 e 2), Tabela 2 completa, Tabela 3 (shortcut) e Tensor Trust. Use `--skip-recheck` para não regerar trigger words e `--models` para restringir os modelos.

### 4.5 Pipeline de correções

`Corrections/` é um fluxo independente de correção e verificação. Ele:

1. gera 435 amostras aumentadas em 17 categorias e aplica quatro filtros;
2. treina base limpa, aumento publicado e aumento corrigido;
3. executa token recheck nos três checkpoints;
4. gera MOF e retreina a variante corrigida com MOF;
5. compara os modelos com a versão oficial do Hugging Face (`leolee99/PIGuard`).

```bash
# Pipeline completo
python Corrections/corrections.py

# Reutiliza o LLM_augmented.json já gerado
python Corrections/corrections.py --skip-generation

# Sem baixar/avaliar o PIGuard oficial do Hugging Face
python Corrections/corrections.py --skip-hf

# Diretório de saída alternativo
python Corrections/corrections.py --output-dir /caminho/para/saida
```

Por padrão, os artefatos ficam em `Corrections/outputs/` (alterável via `--output-dir`). Também é possível usar `--reference-output-root` para reaproveitar checkpoints de uma execução anterior.

Scripts auxiliares:

```bash
# Geração de MOF a partir de uma lista de trigger words
python Corrections/generate_mof_data.py \
    --trigger_words Corrections/outputs/augmented_corrected/trigger_words.json \
    --output Corrections/outputs/augmented_corrected/MOF.json

# Token recheck de um checkpoint
python Corrections/renew_token_recheck.py \
    --checkpoint outputs/train_augmented_original/best_model.pth \
    --output outputs/train_augmented_original/trigger_words.json

# Geração de dados aumentados com os quatro filtros
python Corrections/Upgrade_augmented/new_generate_data.py \
    --output-dir Corrections/outputs/augmented
```

`filters.py` e `filter_data.py` são bibliotecas: o filtro 3 usa OpenRouter quando chamado, mas não têm execução principal. O arquivo `Corrections/Upgrade_augmented/LLM_augmented_new.json` é histórico e não é usado pelo fluxo novo.

### 4.6 Pipeline Upgrade (Novo MOF v2 + Laya English)

O módulo `Upgrade/` estende a reprodução com a geração estratificada do Novo MOF (`MOF_v2_train.json`), conjunto de calibração disjunto (`calibration_heldout.json`) e cascata seletiva com Laya English (`convaiinnovations/laya`, calibrada via temperatura $T^*$ e limiar $T_L^*$):

```bash
# Dry-run dos caminhos resolvidos
python Upgrade/Pipeline/upgrade_pipeline.py --dry-run
python Upgrade/run_corrected_comparison.py --dry-run

# Pipeline completo (requer OPENROUTER_API_KEY na Etapa 0 e ambiente de requirements.txt)
python Upgrade/Pipeline/upgrade_pipeline.py

# Comparação focada na linhagem corrected
python Upgrade/run_corrected_comparison.py
```

Os artefatos são gravados por padrão em `Upgrade/outputs/`. As etapas também podem ser executadas individualmente via `Upgrade/Pipeline/step0_new_mof_and_heldout.py`, `Upgrade/topic1_rp_laya_evaluation.py`, `Upgrade/topic2_corrected_new_mof.py` e `Upgrade/topic3_consolidated_report.py`.

Use `--rp-root` (ou `--drive-rp-root` nos scripts individuais do módulo) para indicar outra pasta com os checkpoints reproduzidos (`outputs/` por padrão), `--corrections-dir` se executou o `Corrections/` fora de `Corrections/outputs/` e `--output-dir` para alterar o destino de saída.

### 4.7 Diretórios de saída e cache

* `outputs/` (na raiz do projeto) é o destino padrão dos checkpoints, trigger words e relatórios de `Notebooks/` e de `Eval/reproduction_validation.py`.
* `Corrections/outputs/` e `Upgrade/outputs/` são os destinos padrão dos pipelines `Corrections/` e `Upgrade/`, respectivamente.
* `logs/Table1/` e `logs/MOFs/` guardam resultados e gráficos das avaliações de Tabela 1 e Tabela 2.
* `Eval/ShortcutMitigation/` e `Eval/TensorTrust/` guardam as avaliações extras.
* Todos os scripts permitem sobrescrever o destino padrão diretamente via CLI (`--output-dir`, `--output_dir` em `Eval/validate_models_table1.py` ou `--output` em `Eval/reproduction_validation.py`).
* As avaliações reutilizam resultados compatíveis. Para forçar recálculo, use `--force-reevaluate` ou `--force-rerun`; para retreinar, use `--force-retrain`.
* Os manifestos incluem seed, `max_length`, hash dos datasets, hash do checkpoint e versões do ambiente, permitindo verificar a compatibilidade do cache.

---

## 📊 5. Resultados e Comparação

Métricas usadas nos scripts:

* **Sobredefesa (Over-defense)**: média da acurácia benigna em `NotInject_one`, `NotInject_two` e `NotInject_three`.
* **Benigna**: acurácia em `wildguard.json` (quando `datasets/PINT.json` existe, a avaliação original também incorpora PINT).
* **Maliciosa**: média da acurácia em `BIPIA_text` e `BIPIA_code`.
* **Média Global**: média aritmética das três métricas.

O `datasets/PINT.json` **não** acompanha este repositório. Por isso, as avaliações atuais usam `WildGuard` para benignos e `BIPIA` para maliciosos; `Original/eval.py` e `Original/eval_hf.py` detectam a ausência de PINT e seguem sem ele.

### 5.1 Tabela 1 — referência do artigo

| Modelo | Sobredefesa (NotInject) | Benigna | Maliciosa | Média |
| :--- | :---: | :---: | :---: | :---: |
| **Fmops** (DistilBERT) | 28,32% | 34,63% | 93,50% | 52,15% |
| **Deepset** (DeBERTa-v3) | 29,50% | 34,06% | 91,50% | 51,68% |
| **PromptGuard** (Meta) | 0,29% | 26,82% | **97,10%** | 41,40% |
| **ProtectAIv2** | 57,23% | 86,20% | 48,60% | 64,01% |
| **GPT-4o** | 86,73% | 90,78% | 79,10% | 85,53% |
| **Llama-2-chat** | 76,40% | 61,03% | 31,09% | 56,17% |
| **LlamaGuard3** | 99,71% | 95,18% | 28,28% | 74,39% |
| **PIGuard (artigo)** | **87,32%** | **85,74%** | 77,39% | **83,48%** |
| **PIGuard (`train_augmented_mof_rp` — melhor média)** | **96,17%** | **96,60%** | 54,67% | **82,48%** |
| **PIGuard (`augmented_mof_corrected` — melhor equilíbrio)** | 93,51% | 79,81% | 70,67% | 81,33% |

¹ Resultados obtidos na rodada oficial do repositório (`RESULTADOS_SEED42_MAX512`, com `seed=42` e `max_length=512`, para `train_augmented_mof_rp`) e na linhagem corrigida (`Corrections` / `Upgrade`, para `augmented_mof_corrected`), avaliados **sem `PINT.json`** (`WildGuard` para benignos e `BIPIA` para maliciosos). Sob exatamente esse mesmo protocolo sem PINT, os pesos oficiais publicados pelos autores (`PIGuard HF` / `leolee99/PIGuard`) alcançam **88,50%** (Sobredefesa), **76,11%** (Benigna), **68,33%** (Maliciosa) e **77,65%** (Média) — portanto nossos melhores modelos superam o checkpoint oficial em **+4,83 pp** (`train_augmented_mof_rp`) e **+3,69 pp** (`augmented_mof_corrected`), respectivamente. Os checkpoints não são versionados neste repositório; para gerar os relatórios localmente, execute `python Notebooks/second_train/eval_table1.py` (ou `Eval/validate_models_table1.py`).

### 5.2 Melhores resultados da reprodução (mesmo protocolo sem PINT)

Comparação direta das nossas melhores variantes frente ao checkpoint oficial publicado (`PIGuard HF`) e ao controle interno (`train_augmented_mof_original`) sob exatamente os mesmos datasets disponíveis publicamente (`NotInject_1/2/3`, `WildGuard`, `BIPIA` e benchmark externo `TensorTrust`):

| Modelo | Origem / Rodada | Sobredefesa | Benigna | Maliciosa | Média Global | Δ vs. `PIGuard HF` | TensorTrust (OOD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `PIGuard HF` (`leolee99/PIGuard`) | Referência oficial (sem PINT) | 88,50% | 76,11% | 68,33% | 77,65% | — | 84,93% |
| `train_augmented_mof_original` | Controle (`SEED42_MAX512`) | 82,60% | 75,59% | **72,00%** | 76,73% | -0,92 pp | 89,92% |
| `train_augmented_mof_semi_rp` | Semi-reprodução (`SEED42_MAX512`) | 91,45% | 85,38% | 66,67% | 81,16% | +3,52 pp | 88,76% |
| `augmented_mof_corrected` | Base corrigida + MOF (`Corrections`) | 93,51% | 79,81% | 70,67% | 81,33% | +3,69 pp | **90,58%** |
| **`train_augmented_mof_rp`** | **Reprodução pura (`SEED42_MAX512`)** | **96,17%** | **96,60%** | 54,67% | **82,48%** | **+4,83 pp** | 67,00% |

* **Maior Média Global (82,48%) e menor sobredefesa (96,17%)**: `train_augmented_mof_rp` na rodada `RESULTADOS_SEED42_MAX512` (`+4,83 pp` acima do `PIGuard HF` e `+5,75 pp` acima do `train_augmented_mof_original`).
* **Melhor equilíbrio entre detecção maliciosa, sobredefesa e generalização OOD**: `augmented_mof_corrected` (`81,33%` de Média Global, preservando `70,67%` em `BIPIA` e liderando o benchmark externo `TensorTrust` com `90,58%`), seguido de perto por `train_augmented_mof_semi_rp` (`81,16%` de média, `66,67%` em `BIPIA` e `88,76%` em `TensorTrust`).

### 5.3 Tabela 2 — referência do artigo

| Desenho | Sobredefesa | Benigna | Maliciosa | Média | Experimentos locais |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Basic dataset training | 75,22% | 78,53% | 70,17% | 74,64% | `train_cleaned_original` |
| + Data-centric augment | 64,31% | 81,36% | 75,95% | 73,87% | `train_augmented_original` / `train_augmented_rp` |
| Basic + MOF scratch retrain | 89,38% | 84,73% | 71,57% | 81,89% | `train_clean_mof_original` / `train_clean_mof_semi_rp` / `train_clean_mof_rp` |
| + Aug + MOF finetuning | 68,14% | 82,11% | 74,81% | 75,02% | `train_augmented_mof_finetuning_original` / `..._semi_rp` / `..._rp` |
| + Aug + MOF scratch retrain | 87,32% | 85,74% | 77,39% | 83,48% | `train_augmented_mof_original` / `train_augmented_mof_semi_rp` / `train_augmented_mof_rp` |

Os avaliadores também mantêm aliases para checkpoints de rodadas anteriores (`*_oficial` para `*_semi_rp` e nomes sem sufixo para `*_rp`). Os valores medidos localmente são gravados em `logs/MOFs/ablation_table_2_summary.csv` e `logs/MOFs/ablation_table_2_detailed_deltas.csv` ao executar `python Notebooks/second_train/eval.py`.

### 5.4 Protocolo de avaliação

A reprodução usa:

* modelo base `microsoft/deberta-v3-base`;
* seed `42`;
* `max_length=512`;
* `batch_size=32` para treino e avaliação;
* 3 épocas, `learning_rate=2e-5`, `warmup=100`;
* `argmax` da classe 1 para considerar uma entrada maliciosa;
* benchmarks `NotInject_one`, `NotInject_two`, `NotInject_three`, `WildGuard`, `BIPIA_text`, `BIPIA_code`;
* Tensor Trust como benchmark extra de hijacking e extraction.

O validador `Eval/reproduction_validation.py` grava um manifesto com hashes dos datasets, checkpoint, implementação e versões de `torch`/`transformers`, além de seed e `max_length`, para que resultados antigos não sejam reutilizados silenciosamente.

---

## 🔍 6. Principais Desafios e Notas de Reprodutibilidade

1. **Estabilidade de treinamento no DeBERTa-v3**: modelos da família DeBERTa com *Electra-style pre-training* podem apresentar sensibilidade a sementes aleatórias e taxa de aprendizado. O repositório fixa seed 42 e parâmetros conservadores, mas pequenas variações entre execuções são esperadas.
2. **Dependência de LLM na síntese MOF**: o artigo usa `GPT-4o-mini` para produzir as amostras com palavras de gatilho. Variações estocásticas nas respostas podem alterar a composição do conjunto de mitigação. O pipeline usa temperatura 1.0 e registra respostas brutas/candidatos em cache.
3. **Equilíbrio entre injeção e sobredefesa**: como apontado na ablação (Tabela 2 do artigo), a mitigação excessiva de atalhos pode afetar a detecção maliciosa se o número de amostras MOF passar de 1.000. O repositório usa 762 amostras MOF aprovadas, seguindo o protocolo do artigo.
4. **Ausência de PINT neste checkout**: `datasets/PINT.json` não está versionado. Sem ele, as métricas de benignidade usam `WildGuard`; para comparar diretamente com os agregados do artigo, é necessário fornecer o PINT no formato esperado por `Original/eval.py`.
5. **Custo e tempo**: a segunda fase executa nove treinamentos completos; a primeira fase executa três. Em uma A100, o fluxo completo leva horas e consome vários GB de checkpoints. Use `--experiment` para executar um cenário por vez e `--output-dir` para separar execuções.
6. **Cache e retomada**: avaliações e token rechecks podem ser retomados a partir de resultados salvos. Se um dataset, checkpoint ou ambiente mudar, o cache é invalidado pelo manifesto; para forçar, use as flags `--force-*`.
7. **Modelos remotos**: o primeiro uso de `microsoft/deberta-v3-base`, dos baselines e do Tensor Trust requer rede e cache do Hugging Face. O Prompt Guard exige aceite de acesso na conta. As linhas `Llama-2-chat` e `LlamaGuard3` da Tabela 1 usam identificadores OpenRouter históricos definidos em `Eval/validate_models_table1.py`.

---

## 🧪 7. Solução de Problemas

| Sintoma | Causa provável | Ação |
| :--- | :--- | :--- |
| `ModuleNotFoundError: torch` / `transformers` | dependências ausentes no Python usado pelo comando. Um alias pode fazer `python` apontar para o sistema mesmo com `(.venv)` no prompt | No Linux/macOS, use `./.venv/bin/python -m pip install -r requirements.txt` e execute o script com `./.venv/bin/python`. No Windows, use `.venv\Scripts\python.exe`. |
| `ImportError: pyarrow requires NumPy 2.0 or newer` | PyArrow incompatível com o NumPy do ambiente principal | Reinstale as dependências de `requirements.txt` no mesmo ambiente do treino. |
| `checkpoint ausente` | treino não executado ou `--output-dir` diferente | rode a fase de treino correspondente ou informe `--output-dir` |
| `API key not found` | geração/MOF sem credencial | exporte `OPENROUTER_API_KEY` ou crie `.env` na raiz |
| `CUDA out of memory` | lote grande para a GPU | reduza `--batch-size` / `--eval-batch-size` |
| resultados antigos reaparecem | cache compatível | use `--force-reevaluate`, `--force-rerun` ou `--force-retrain` |
| avaliação pula OpenRouter | `--skip-openrouter` ou ausência de chave | informe a chave ou remova a flag |
| PINT ausente | arquivo não versionado | as avaliações seguem com WildGuard/BIPIA; forneça `datasets/PINT.json` se quiser PINT |

---

## 📄 8. Citação

```bibtex
@inproceedings{li2025piguard,
  title     = {PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free},
  author    = {Li, Hao and Liu, Xiaogeng and Zhang, Ning and Xiao, Chaowei},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {30420--30437},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```

---

## 👥 Equipe

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/allan-f0101">
        <img src="https://avatars.githubusercontent.com/u/244098995?v=4" width="80px;" alt="Allan"/><br>
        <sub><b>Allan Fernandes</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/arthlz">
        <img src="https://avatars.githubusercontent.com/u/173482833?v=4" width="80px;" alt="Arthur"/><br>
        <sub><b>Arthur Luz</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/gabrielbelo2007">
        <img src="https://avatars.githubusercontent.com/u/91219977?v=4" width="80px;" alt="Gabriel"/><br>
        <sub><b>Gabriel Belo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/LHFalcao">
        <img src="https://avatars.githubusercontent.com/u/211526897?v=4" width="80px;" alt="Luiz"/><br>
        <sub><b>Luiz Falcão</b></sub>
      </a>
    </td>
  </tr>
</table>

# Reprodução Científica: PIGuard (ACL 2025)

Este repositório contém o projeto de reprodução experimental do artigo **"PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free"**, aceito na *ACL 2025* (Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics).

> **Referência original:**  
> Hao Li, Xiaogeng Liu, Ning Zhang, Chaowei Xiao. *PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free*. ACL 2025.  
> Repositório oficial dos autores: [leolee99/PIGuard](https://github.com/leolee99/PIGuard)

---

## 📌 1. Visão Geral e Objetivos da Reprodução

Os modelos de salvaguarda (*guardrails*) contra ataques de injeção de prompt (ex.: ProtectAIv2, PromptGuard, Fmops) frequentemente sofrem de **sobredefesa (*over-defense*)**: classificam entradas legítimas e benignas como ataques devido à dependência excessiva de atalhos (*shortcut learning*) associados a palavras de gatilho (*trigger words*, como *"ignore"*, *"cancel"*, *"system"* ou *"execute"*).

O artigo propõe duas contribuições centrais:
1. **Benchmark NotInject**: Dataset de avaliação composto por 339 amostras benignas desafiadoras contendo palavras de gatilho frequentes em ataques, divididas em 3 níveis de dificuldade (1, 2 e 3 palavras-gatilho).
2. **MOF (*Mitigating Over-defense for Free*)**: Estratégia de treinamento sem necessidade de datasets manuais de sobredefesa. Consiste em:
   - Treinar uma base supervisionada inicial.
   - Realizar uma varredura token por token (*token-wise recheck*) para detectar quais vocábulos isolados ativam o classificador como injeção.
   - Sintetizar frases benignas contendo essas palavras com o auxílio de um LLM.
   - Retreinar o modelo do zero (*retrain from scratch*) com essa ampliação de dados.

### Objetivos deste Trabalho
* [ ] Replicar o pipeline de dados de treino básico e a ampliação de formatos em cauda longa (XML, CSV, JSON, Markdown).
* [ ] Reproduzir o procedimento *token-wise recheck* para identificar tokens enviesados.
* [ ] Executar o treinamento do classificador baseado em `microsoft/deberta-v3-base`.
* [ ] Avaliar e comparar o modelo contra baselines abertos no dataset **NotInject**, medindo acurácia nas 3 dimensões: **Benigna**, **Maliciosa** e **Sobredefesa**.

---

## 🏗️ 2. Estrutura do Projeto

```text
ainda a se colocar
```

---

## ⚙️ 3. Ambiente e Pré-requisitos
```
  Rodamos o modelo no google colab em uma A100(80GB de VRAM)
```
### Requisitos de Hardware
* **GPU**: Mínimo de 16 GB de VRAM (NVIDIA T4, RTX 3090, A100 ou equivalente) para treinamento do `DeBERTa-v3-base` com tamanho de lote 32 e context length de 512.
* **Espaço em disco**: ~20 GB livres para datasets, modelos base e checkpoints.

### Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/piguard-reproduction.git
   cd piguard-reproduction
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Executar os notebooks convertidos no Linux

Os nove scripts em `Scripts/first_train/`, `Scripts/second_train/` e
`Scripts/train_extras/` podem ser chamados individualmente com `python3`.
Use o caminho absoluto do script para chamá-lo de outro diretório. Os caminhos
relativos passados como argumentos são interpretados a partir da raiz deste
projeto. Cada script aceita `--help`.

Para instalar apenas as dependências usadas pelos scripts, use Python 3.10 a
3.12 em um ambiente virtual. Os pacotes abaixo já constam no
`requirements.txt`. O arquivo completo também inclui dependências de outros
fluxos do projeto.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.0 transformers==4.44.0 datasets==2.21.0 \
    sentencepiece==0.2.0 protobuf==4.25.4 tiktoken==0.7.0 PyYAML==6.0.2 \
    numpy==1.26.3 pandas==2.2.2 matplotlib==3.9.2 tqdm==4.66.5
```

Para usar uma GPU NVIDIA, instale um driver compatível com a distribuição
CUDA do PyTorch. Os scripts também executam em CPU.
[Instruções do PyTorch](https://pytorch.org/get-started/previous-versions/#v240)

Os treinos chamam os programas Python do checkout local de `RP_PIGuard`. Os
scripts procuram esse checkout dentro do projeto ou nos dois diretórios
superiores. Use `RP_PIGUARD_DIR` para indicar outro caminho até um checkout
com o nome `RP_PIGuard`. Os datasets de treino são procurados primeiro em
`datasets/` neste projeto e depois no checkout de reprodução.

Exemplos a partir da raiz do projeto

```bash
python3 Scripts/first_train/train.py --experiment train_augmented_original
python3 Scripts/second_train/train.py --experiment train_augmented_mof_finetuning_original
python3 Scripts/second_train/eval.py
python3 Scripts/second_train/eval_table1.py --skip-openrouter
python3 Scripts/train_extras/eval.py --benchmark tensor-trust
```

- Os scripts de treino executam todos os experimentos por padrão. Use
  `--experiment` para selecionar um deles. Use `--train-set` com um único
  experimento para apontar a versão correta do dataset. Se já houver um
  checkpoint e você quiser substituí-lo, use também `--force-retrain`.
- Os pesos e as trigger words ficam em `checkpoints/<experimento>/` ou no
  diretório definido por `PIGUARD_OUTPUT_DIR`. Os logs e relatórios ficam em
  `logs/`. Nas avaliações, use `--output-dir` para indicar outro destino.
- O fine-tuning MOF requer os pesos de base produzidos na primeira etapa. As
  avaliações locais precisam dos checkpoints dos experimentos correspondentes.
  A Tabela 1, a Tabela 2 e as avaliações extras podem reutilizar métricas
  salvas quando disponíveis.
- A variante `first_train/new_train.py` espera os datasets legados
  `train_augmented.json`, `train_new_augmented.json` e `train_cleaned.json`.
  Forneça esses arquivos em `datasets/` ou use `--train-set` por experimento.
- O primeiro uso dos modelos e do Tensor Trust requer acesso ao Hugging Face
  quando os arquivos ainda estão fora do cache local. Modelos restritos, como
  Prompt Guard na Tabela 1, exigem acesso concedido à conta do Hugging Face.
  As chamadas ao OpenRouter na Tabela 1 usam `OPENROUTER_API_KEY` ou
  `OPENAI_API_KEY` do ambiente ou do arquivo `.env` na raiz.
- O validador preserva os identificadores OpenRouter usados pela reprodução.
  A linha histórica `Llama-2-chat` executa `meta-llama/llama-3.1-8b-instruct`
  e a linha `LlamaGuard3` executa `meta-llama/llama-guard-4-12b`.

---

## 📊 5. Resultados e Comparação

Tabela comparativa entre os resultados reportados no artigo original (ACL 2025) e os valores obtidos nesta reprodução:

| Modelo | Acurácia Sobredefesa (NotInject) | Acurácia Benigna (WildGuard/PINT) | Acurácia Maliciosa (PINT/BIPIA) | Média Global |
| :--- | :---: | :---: | :---: | :---: |
| **PromptGuard** (Meta) | 0.29% | 26.82% | **97.10%** | 41.40% |
| **ProtectAIv2** | 57.23% | 86.20% | 48.60% | 64.01% |
| **PIGuard (Paper Original)** | **87.32%** | **85.74%** | 77.39% | **83.48%** |
| **PIGuard (Nossa Reprodução)** | *%* | *%* | *%* | *%* |

---

## 🔍 6. Principais Desafios e Notas de Reprodutibilidade

Durante o processo de reprodução, atente-se aos seguintes pontos críticos:
1. **Estabilidade de Treinamento no DeBERTa-v3**: Modelos da família DeBERTa com *Electra-style pre-training* podem apresentar sensibilidade a sementes aleatórias (*random seeds*) e taxa de aprendizado. 
2. **Dependência de LLM na Síntese MOF**: No paper, o `GPT-4o-mini` foi utilizado para produzir as 1.000 amostras com palavras de gatilho. Pequenas variações estocásticas nas respostas podem alterar sutilmente a composição do conjunto de mitigação.
3. **Equilíbrio entre Injeção e Sobredefesa**: Como apontado no estudo de ablação (Tabela 2 do artigo), a mitigação excessiva de atalhos pode afetar a taxa de detecção maliciosa se o número de amostras MOF passar de 1.000.

---

## 📄 7. Citação

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

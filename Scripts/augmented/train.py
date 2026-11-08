"""PIGuard - Treinamento de Modelos (executável como script Python).

Equivalente ao notebook ``Notebooks/first_train/train.ipynb``, adaptado para
execução direta como arquivo ``.py`` e sem qualquer dependência do Google Colab.

Uso:
    python Notebooks/first_train/train.py

Os artefatos (checkpoints, logs e métricas) são gravados em ``./outputs``.
Esse diretório pode ser alterado pela variável de ambiente ``PIGUARD_OUTPUT_DIR``.

Três cenários de treinamento e ablação comparativa:
1. ``train_augmented_original``: dataset aumentado oficial do artigo.
2. ``train_augmented_rp``: dataset aumentado reproduzido.
3. ``train_cleaned_original``: baseline limpo, sem aumento de dados.

Ao final de cada treinamento, os melhores pesos (``best_model.pth``) e logs são
copiados para o diretório de saída, seguidos pelo Token-wise Recheck dos modelos
aumentados.
"""

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

# Se False, pula treinamentos cujos modelos (best_model.pth) já existam em OUTPUT_DIR.
FORCE_RETRAIN = False

EXPERIMENT_NAMES = [
    "train_augmented_original",
    "train_augmented_rp",
    "train_cleaned_original",
]


def run_command(cmd, check=True):
    """Executa um comando externo a partir da raiz do repositório."""
    print("\n$ {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, cwd=REPO_DIR)
    if check and result.returncode != 0:
        raise RuntimeError(
            "Comando falhou (código de saída {}): {}".format(
                result.returncode, " ".join(cmd)
            )
        )
    return result.returncode


def print_hardware_info():
    """Exibe informações de GPU/PyTorch, substituindo o antigo ``!nvidia-smi``."""
    try:
        import torch
    except ImportError:
        print("[AVISO] PyTorch não está instalado; pulando verificação de hardware.")
        return

    print("\n--- Informações de Hardware e PyTorch ---")
    print("CUDA Disponível: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Dispositivo: {}".format(torch.cuda.get_device_name(0)))
        print(
            "Total VRAM: {:.2f} GB".format(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
        )


def train_experiment(name, train_set, log_name):
    """Executa um treinamento e persiste o melhor checkpoint/log em OUTPUT_DIR."""
    target_dir = os.path.join(OUTPUT_DIR, name)
    target_best = os.path.join(target_dir, "best_model.pth")
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_best) and not FORCE_RETRAIN:
        print("[SKIP] Modelo '{}' já existe em: {}".format(name, target_best))
        print(
            "Pulando treinamento para economizar recursos. "
            "(Defina FORCE_RETRAIN=True para reexecutar)."
        )
        return

    checkpoint_dir = os.path.join("checkpoints", name)
    logs_dir = os.path.join("logs", name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    run_command(
        [
            sys.executable,
            "train.py",
            "--name", name,
            "--train_set", train_set,
            "--valid_set", "datasets/valid.json",
            "--dataset_root", "datasets",
            "--checkpoint_path", checkpoint_dir,
            "--logs", logs_dir,
            "--batch_size", "32",
            "--eval_batch_size", "32",
            "--save_step", "100",
            "--epochs", "3",
        ]
    )

    local_best = os.path.join(checkpoint_dir, "best_model.pth")
    local_log = os.path.join(logs_dir, log_name)
    if not os.path.exists(local_best):
        print("Aviso: {} não encontrado.".format(local_best))
        return

    print("\nCopiando {} para {}...".format(local_best, target_dir))
    shutil.copy(local_best, target_best)
    if os.path.exists(local_log):
        shutil.copy(local_log, os.path.join(target_dir, log_name))
    print("Treinamento '{}' salvo com sucesso em: {}".format(name, target_best))

    # Remove checkpoints intermediários de época para poupar espaço em disco.
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("epoch_") and fname.endswith(".pth"):
            try:
                os.remove(os.path.join(checkpoint_dir, fname))
            except OSError:
                pass


def run_token_recheck(model_name, output_name):
    """Executa o Token-wise Recheck e copia o resultado para ``MOF/``."""
    model_path = os.path.join("checkpoints", model_name, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(OUTPUT_DIR, model_name, "best_model.pth")
    if model_name == "train_augmented_original" and not os.path.exists(model_path):
        model_path = os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth")

    output_path = os.path.join(OUTPUT_DIR, model_name, output_name)

    if not os.path.exists(model_path):
        print("Aviso: Checkpoint {} não encontrado.".format(model_path))
        return

    recheck_script = os.path.join(REPO_DIR, "token_recheck.py")
    if not os.path.exists(recheck_script):
        print(
            "[AVISO] token_recheck.py não encontrado em {}. "
            "Pulando o Token-wise Recheck de {}.".format(REPO_DIR, model_name)
        )
        return

    print("Disparando Token-wise Recheck para {}: {}".format(model_name, model_path))
    run_command(
        [
            sys.executable,
            "token_recheck.py",
            "--checkpoint", model_path,
            "--output", output_path,
            "--batch_size", "256",
        ]
    )

    os.makedirs("MOF", exist_ok=True)
    try:
        shutil.copy(output_path, os.path.join("MOF", output_name))
    except OSError as exc:
        print("Aviso ao copiar trigger words para MOF/: {}".format(exc))


def main():
    # 1. Verificação de hardware
    print_hardware_info()

    # 2. Configuração do diretório de saída e verificação do ambiente
    for exp in EXPERIMENT_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, exp), exist_ok=True)
    print("Diretório de persistência configurado em: {}".format(OUTPUT_DIR))

    print("\nFORCE_RETRAIN configurado como: {}".format(FORCE_RETRAIN))

    # 3. Treinamento 1: datasets/Augmented/train_augmented_original.json
    target_1 = os.path.join(OUTPUT_DIR, "train_augmented_original")
    best_1 = os.path.join(target_1, "best_model.pth")
    legacy_best = os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth")

    # Reaproveita checkpoint legado se o dataset reproduzido não estiver presente.
    if (
        not os.path.exists(best_1)
        and os.path.exists(legacy_best)
        and not os.path.exists("datasets/Augmented/train_augmented_rp.json")
    ):
        print("Reaproveitando checkpoint legado oficial para {}...".format(target_1))
        os.makedirs(target_1, exist_ok=True)
        shutil.copy(legacy_best, best_1)

    train_experiment(
        name="train_augmented_original",
        train_set="datasets/Augmented/train_augmented_original.json",
        log_name="log_train_augmented_original.txt",
    )

    # 4. Treinamento 2: datasets/Augmented/train_augmented_rp.json
    train_experiment(
        name="train_augmented_rp",
        train_set="datasets/Augmented/train_augmented_rp.json",
        log_name="log_train_augmented.txt",
    )

    # 5. Treinamento 3: datasets/Augmented/train_cleaned_original.json
    target_3 = os.path.join(OUTPUT_DIR, "train_cleaned_original")
    best_3 = os.path.join(target_3, "best_model.pth")
    legacy_cleaned = os.path.join(OUTPUT_DIR, "train_cleaned", "best_model.pth")

    if not os.path.exists(best_3) and os.path.exists(legacy_cleaned):
        print("Reaproveitando checkpoint train_cleaned existente para {}...".format(target_3))
        os.makedirs(target_3, exist_ok=True)
        shutil.copy(legacy_cleaned, best_3)

    train_experiment(
        name="train_cleaned_original",
        train_set="datasets/Augmented/train_cleaned_original.json",
        log_name="log_train_cleaned_oficial.txt",
    )

    # 6. Token-wise Recheck (Seção 4.2 do artigo PIGuard)
    run_token_recheck(
        model_name="train_augmented_original",
        output_name="trigger_words_train_augmented_original.json",
    )
    run_token_recheck(
        model_name="train_augmented_rp",
        output_name="trigger_words_train_augmented.json",
    )

    # 7. Verificação final
    print("\n" + "=" * 70)
    print("Processamento concluído com sucesso!")
    print("Artefatos persistidos em: {}".format(OUTPUT_DIR))
    for exp in EXPERIMENT_NAMES:
        path = os.path.join(OUTPUT_DIR, exp, "best_model.pth")
        status = "OK (Presente)" if os.path.exists(path) else "Pendente/Ausente"
        print("   - [{}] {}".format(status, path))
    print("=" * 70)


if __name__ == "__main__":
    main()

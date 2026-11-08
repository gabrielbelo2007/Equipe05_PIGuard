"""PIGuard - Estudo de Ablação com MOF (Fase 2: Scratch Retrain & Finetuning).

Equivalente ao notebook ``Notebooks/second_train/train.ipynb``, adaptado para
execução direta como arquivo ``.py`` e sem qualquer dependência do Google Colab.

Uso:
    python Notebooks/second_train/train.py

Executa os 9 treinamentos da Tabela de Ablação (Tabela 2) do artigo PIGuard,
distribuídos em 3 vertentes (Original, Oficial/Semi-Reprodução e Reproduzida).
Os checkpoints e logs são persistidos em ``./outputs`` (configurável pela
variável de ambiente ``PIGUARD_OUTPUT_DIR``).

Proteções de engenharia preservadas:
1. Prevenção de colapso de classe no fine-tuning (``--save_step 2`` e ``--warmup 2``).
2. Persistência dos checkpoints no diretório de saída.
3. Pulo inteligente (``FORCE_RETRAIN = False``) para modelos já treinados.
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
    # Vertente 1: Datasets Originais do Artigo
    "train_clean_mof_original",
    "train_augmented_mof_finetuning_original",
    "train_augmented_mof_original",
    # Vertente 2: Vertente Oficial (Re-gerada)
    "train_clean_mof_oficial",
    "train_augmented_mof_finetuning_oficial",
    "train_augmented_mof_oficial",
    # Vertente 3: Vertente Reproduzida
    "train_clean_mof",
    "train_augmented_mof_finetuning",
    "train_augmented_mof",
]


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


def find_checkpoint(candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def run_mof_experiment(name, train_set, save_step=100, warmup=100, epochs=3, resume_ckpt=None):
    """Executa um treinamento padronizado, com log e persistência em OUTPUT_DIR."""
    output_target = os.path.join(OUTPUT_DIR, name)
    output_best = os.path.join(output_target, "best_model.pth")

    if os.path.exists(output_best) and not FORCE_RETRAIN:
        print("\n[SKIP] Modelo '{}' já existe em: {}".format(name, output_best))
        print(
            "Pulando para economizar tempo e recursos de GPU. "
            "(Defina FORCE_RETRAIN=True para reexecutar)."
        )
        return

    print("\n" + "=" * 75)
    print("INICIANDO EXPERIMENTO: {}".format(name))
    print("Dataset: {}".format(train_set))
    print(
        "Configuração: save_step={}, warmup={}, epochs={}".format(
            save_step, warmup, epochs
        )
    )
    if resume_ckpt:
        print("Resumindo pesos de: {}".format(resume_ckpt))
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(
                "Checkpoint base não encontrado: {}. "
                "Execute o treinamento base primeiro!".format(resume_ckpt)
            )
    print("=" * 75 + "\n")

    checkpoint_dir = os.path.join("checkpoints", name)
    logs_dir = os.path.join("logs", name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(output_target, exist_ok=True)

    cmd = [
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
        "--save_step", str(save_step),
        "--warmup", str(warmup),
        "--epochs", str(epochs),
    ]
    if resume_ckpt:
        cmd.extend(["--resume", resume_ckpt])

    print("$ {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, cwd=REPO_DIR)
    if result.returncode != 0:
        raise RuntimeError(
            "Erro durante o treinamento de {} (código de saída: {})".format(
                name, result.returncode
            )
        )

    local_best = os.path.join(checkpoint_dir, "best_model.pth")
    local_log = os.path.join(logs_dir, "log_{}.txt".format(name))
    if os.path.exists(local_best):
        print("\nCopiando {} para {}...".format(local_best, output_target))
        shutil.copy(local_best, output_best)
        if os.path.exists(local_log):
            shutil.copy(local_log, os.path.join(output_target, "log_{}.txt".format(name)))
        print("Checkpoint salvo com sucesso em: {}".format(output_best))

        # Limpa checkpoints intermediários de época para poupar espaço em disco.
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith("epoch_") and fname.endswith(".pth"):
                try:
                    os.remove(os.path.join(checkpoint_dir, fname))
                except OSError:
                    pass
    else:
        print("Aviso: {} não foi gerado.".format(local_best))


def main():
    # 1. Verificação de hardware
    print_hardware_info()

    # 2. Configuração do diretório de saída
    for exp in EXPERIMENT_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, exp), exist_ok=True)
    print("Diretórios de persistência configurados em: {}".format(OUTPUT_DIR))
    print("\nFORCE_RETRAIN configurado como: {}".format(FORCE_RETRAIN))

    # 3. Verificação dos datasets e checkpoints base
    print("\n--- Verificação de Datasets e Checkpoints Base ---")

    def check_file(path, desc):
        exists = os.path.exists(path)
        status = "OK" if exists else "AUSENTE"
        print("[{}] {}: {}".format(status, desc, path))
        return exists

    print("Datasets MOF:")
    for ds in [
        "MOF/train_clean_mof_original.json",
        "MOF/MOF_original.json",
        "datasets/train.json",
        "MOF/train_clean_mof_semi_rp.json",
        "MOF/MOF_semi_rp.json",
        "MOF/train_augmented_mof_semi_rp.json",
        "MOF/train_clean_mof_rp.json",
        "MOF/MOF_rp.json",
        "MOF/train_augmented_mof_rp.json",
    ]:
        check_file(ds, ds.split("/")[-1])

    print("\nCheckpoints Base para Fine-Tuning (Fase 1):")
    base_oficial_candidates = [
        os.path.join(OUTPUT_DIR, "train_augmented_original", "best_model.pth"),
        os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
        "checkpoints/train_augmented_original/best_model.pth",
    ]
    base_reprod_candidates = [
        os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
        "checkpoints/train_augmented_rp/best_model.pth",
    ]

    base_oficial = find_checkpoint(base_oficial_candidates)
    base_reprod = find_checkpoint(base_reprod_candidates)
    print("Checkpoint Base Oficial: {}".format(base_oficial))
    print("Checkpoint Base Reproduzido: {}".format(base_reprod))

    # Vertente 1: Datasets Originais do Artigo

    # Experimento 1: Basic + MOF scratch retrain
    run_mof_experiment(
        name="train_clean_mof_original",
        train_set="MOF/train_clean_mof_original.json",
        save_step=100,
        warmup=100,
        epochs=3,
        resume_ckpt=None,
    )

    # Experimento 2: Basic + Aug + MOF finetuning
    base_ckpt = find_checkpoint(
        [
            os.path.join(OUTPUT_DIR, "train_augmented_original", "best_model.pth"),
            "checkpoints/train_augmented_original/best_model.pth",
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
        ]
    )
    print("Utilizando checkpoint base: {}".format(base_ckpt))
    run_mof_experiment(
        name="train_augmented_mof_finetuning_original",
        train_set="MOF/MOF_original.json",
        save_step=2,
        warmup=2,
        epochs=3,
        resume_ckpt=base_ckpt,
    )

    # Experimento 3: Basic + Aug + MOF scratch retrain
    run_mof_experiment(
        name="train_augmented_mof_original",
        train_set="datasets/train.json",
        save_step=100,
        warmup=100,
        epochs=3,
        resume_ckpt=None,
    )

    # Vertente 2: Vertente Oficial (Re-gerada via Trigger Words Oficiais)

    # Experimento 4: Basic + MOF scratch retrain
    run_mof_experiment(
        name="train_clean_mof_oficial",
        train_set="MOF/train_clean_mof_semi_rp.json",
        save_step=100,
        warmup=100,
        epochs=3,
        resume_ckpt=None,
    )

    # Experimento 5: Basic + Aug + MOF finetuning
    base_ckpt = find_checkpoint(
        [
            os.path.join(OUTPUT_DIR, "train_augmented_original", "best_model.pth"),
            "checkpoints/train_augmented_original/best_model.pth",
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
        ]
    )
    print("Utilizando checkpoint base: {}".format(base_ckpt))
    run_mof_experiment(
        name="train_augmented_mof_finetuning_oficial",
        train_set="MOF/MOF_semi_rp.json",
        save_step=2,
        warmup=2,
        epochs=3,
        resume_ckpt=base_ckpt,
    )

    # Experimento 6: Basic + Aug + MOF scratch retrain
    run_mof_experiment(
        name="train_augmented_mof_oficial",
        train_set="MOF/train_augmented_mof_semi_rp.json",
        save_step=100,
        warmup=100,
        epochs=3,
        resume_ckpt=None,
    )

    # Vertente 3: Vertente Reproduzida (Trigger Words Reproduzidas)

    # Experimento 7: Basic + MOF scratch retrain
    run_mof_experiment(
        name="train_clean_mof",
        train_set="MOF/train_clean_mof_rp.json",
        save_step=100,
        warmup=100,
        epochs=3,
        resume_ckpt=None,
    )

    # Experimento 8: Basic + Aug + MOF finetuning
    base_ckpt = find_checkpoint(
        [
            os.path.join(OUTPUT_DIR, "train_augmented_rp", "best_model.pth"),
            "checkpoints/train_augmented_rp/best_model.pth",
        ]
    )
    print("Utilizando checkpoint base: {}".format(base_ckpt))
    run_mof_experiment(
        name="train_augmented_mof_finetuning",
        train_set="MOF/MOF_rp.json",
        save_step=2,
        warmup=2,
        epochs=3,
        resume_ckpt=base_ckpt,
    )

    # Experimento 9: Basic + Aug + MOF scratch retrain
    run_mof_experiment(
        name="train_augmented_mof",
        train_set="MOF/train_augmented_mof_rp.json",
        save_step=100,
        warmup=100,
        epochs=3,
        resume_ckpt=None,
    )

    # 4. Verificação final
    print("\n" + "=" * 75)
    print("Todos os treinamentos da Fase 2 (MOF) foram processados!")
    print("Status dos 9 modelos em {}:".format(OUTPUT_DIR))

    for exp in EXPERIMENT_NAMES:
        path = os.path.join(OUTPUT_DIR, exp, "best_model.pth")
        status = "OK (Salvo)" if os.path.exists(path) else "PENDENTE/AUSENTE"
        print("[{:15}] {}".format(status, exp))

    print("=" * 75)


if __name__ == "__main__":
    main()

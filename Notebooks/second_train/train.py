"""PIGuard - Estudo de Ablação com MOF (Fase 2: Scratch Retrain & Finetuning).

Conversão do notebook ``Notebooks/second_train/train.ipynb`` para script Python
autônomo, sem comandos mágicos e sem dependência do Google Colab.

Uso:
    python Notebooks/second_train/train.py
    python Notebooks/second_train/train.py --experiment train_clean_mof_rp
    python Notebooks/second_train/train.py --force-retrain

Executa os 9 treinamentos da Tabela 2 do artigo, em 3 vertentes:
- Original (datasets do artigo);
- Oficial / Semi-Reprodução (trigger words oficiais);
- Reproduzida (pipeline inteiramente reproduzido).

"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PIGUARD_OUTPUT_DIR", REPO_ROOT / "outputs")
).expanduser()


@dataclass(frozen=True)
class Experiment:
    name: str
    train_set: str
    save_step: int = 200
    warmup: int = 100
    epochs: int = 3
    resume: str | None = None  # "original", "reproduced" ou None


EXPERIMENTS = (
    # Vertente 1: datasets originais do artigo.
    Experiment(
        "train_clean_mof_original",
        "datasets/MOF/train_clean_mof_original.json",
    ),
    Experiment(
        "train_augmented_mof_finetuning_original",
        "datasets/MOF/MOF_original.json",
        save_step=2,
        warmup=2,
        resume="original",
    ),
    Experiment(
        "train_augmented_mof_original",
        "datasets/train.json",
    ),
    # Vertente 2: semi-reprodução (trigger words oficiais).
    Experiment(
        "train_clean_mof_semi_rp",
        "datasets/MOF/train_clean_mof_semi_rp.json",
    ),
    Experiment(
        "train_augmented_mof_finetuning_semi_rp",
        "datasets/MOF/MOF_semi_rp.json",
        save_step=2,
        warmup=2,
        resume="original",
    ),
    Experiment(
        "train_augmented_mof_semi_rp",
        "datasets/MOF/train_augmented_mof_semi_rp.json",
    ),
    # Vertente 3: vertente reproduzida.
    Experiment(
        "train_clean_mof_rp",
        "datasets/MOF/train_clean_mof_rp.json",
    ),
    Experiment(
        "train_augmented_mof_finetuning_rp",
        "datasets/MOF/MOF_rp.json",
        save_step=2,
        warmup=2,
        resume="reproduced",
    ),
    Experiment(
        "train_augmented_mof_rp",
        "datasets/MOF/train_augmented_mof_rp.json",
    ),
)

EXPERIMENT_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}
LEGACY_NAME_ALIASES = {
    "train_clean_mof_semi_rp": ("train_clean_mof_oficial",),
    "train_augmented_mof_finetuning_semi_rp": ("train_augmented_mof_finetuning_oficial",),
    "train_augmented_mof_semi_rp": ("train_augmented_mof_oficial",),
    "train_clean_mof_rp": ("train_clean_mof",),
    "train_augmented_mof_finetuning_rp": ("train_augmented_mof_finetuning",),
    "train_augmented_mof_rp": ("train_augmented_mof",),
}


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def find_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def find_repo_script(filename: str, extra_dirs: Sequence[Path] = ()) -> Path | None:
    for directory in (
        REPO_ROOT,
        REPO_ROOT / "Original",
        REPO_ROOT / "Eval",
        *extra_dirs,
    ):
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


@contextmanager
def subprocess_environment(extra_paths: Sequence[Path] = ()) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    paths = [str(path) for path in extra_paths]
    paths.extend([str(REPO_ROOT), str(REPO_ROOT / "Original")])
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


def run_python_script(script: Path, arguments: Sequence[str]) -> None:
    cmd = [sys.executable, str(script), *arguments]
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    with subprocess_environment([script.parent]) as env:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def print_hardware_info() -> None:
    try:
        import torch
    except ImportError:
        print("[AVISO] PyTorch não está instalado; pulando verificação de hardware.")
        return

    print("\n--- Informações de Hardware e PyTorch ---")
    print(f"CUDA Disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
        print(
            f"Total VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )


def resolve_base_checkpoint(kind: str, output_dir: Path) -> Path | None:
    checkpoints = REPO_ROOT / "checkpoints"
    if kind == "original":
        return find_existing(
            [
                output_dir / "train_augmented_original" / "best_model.pth",
                checkpoints / "train_augmented_original" / "best_model.pth",
            ]
        )
    return find_existing(
        [
            output_dir / "train_augmented_rp" / "best_model.pth",
            checkpoints / "train_augmented_rp" / "best_model.pth",
        ]
    )


def run_mof_experiment(
    experiment: Experiment,
    output_dir: Path,
    force_retrain: bool,
    train_script: Path,
    valid_set: Path,
    dataset_root: Path,
) -> bool:
    """Executa um treinamento, salva log/checkpoint e limpa épocas intermediárias."""
    output_target = output_dir / experiment.name
    output_best = output_target / "best_model.pth"
    existing_best = find_existing(
        [
            output_best,
            *(output_dir / alias / "best_model.pth" for alias in LEGACY_NAME_ALIASES.get(experiment.name, ())),
        ]
    )

    if existing_best is not None and not force_retrain:
        print(f"\n[SKIP] Modelo '{experiment.name}' já existe em: {existing_best}")
        print("Use --force-retrain para reexecutar.")
        return False

    train_set = repo_path(experiment.train_set)
    if not train_set.is_file():
        raise SystemExit(
            f"[ERRO] Dataset não encontrado: {train_set}\n"
            "Verifique se os arquivos MOF/datasets foram baixados corretamente."
        )

    resume_checkpoint = None
    if experiment.resume:
        resume_checkpoint = resolve_base_checkpoint(experiment.resume, output_dir)
        if resume_checkpoint is None:
            raise SystemExit(
                "[ERRO] Checkpoint base para fine-tuning não encontrado. "
                "Execute primeiro ``Notebooks/first_train/train.py`` para gerar "
                "train_augmented_original ou train_augmented_rp."
            )

    print("\n" + "=" * 75)
    print(f"INICIANDO EXPERIMENTO: {experiment.name}")
    print(f"Dataset: {train_set}")
    print(
        f"Configuração: save_step={experiment.save_step}, "
        f"warmup={experiment.warmup}, epochs={experiment.epochs}"
    )
    if resume_checkpoint:
        print(f"Resumindo pesos de: {resume_checkpoint}")
    print("=" * 75)

    checkpoint_dir = REPO_ROOT / "checkpoints" / experiment.name
    logs_dir = REPO_ROOT / "logs" / experiment.name
    output_target.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    arguments = [
        "--name", experiment.name,
        "--train_set", str(train_set),
        "--valid_set", str(valid_set),
        "--dataset_root", str(dataset_root),
        "--checkpoint_path", str(checkpoint_dir),
        "--logs", str(logs_dir),
        "--batch_size", "32",
        "--eval_batch_size", "32",
        "--save_step", str(experiment.save_step),
        "--warmup", str(experiment.warmup),
        "--seed", "42",
        "--max_length", "512",
        "--epochs", str(experiment.epochs),
    ]
    if resume_checkpoint is not None:
        arguments.extend(["--resume", str(resume_checkpoint)])

    run_python_script(train_script, arguments)

    local_best = checkpoint_dir / "best_model.pth"
    local_log = logs_dir / f"log_{experiment.name}.txt"
    if not local_best.is_file():
        print(f"[AVISO] {local_best} não foi gerado.")
        return False

    print(f"\nCopiando {local_best} para {output_target}...")
    shutil.copy2(local_best, output_best)
    if local_log.is_file():
        shutil.copy2(local_log, output_target / f"log_{experiment.name}.txt")
    print(f"Checkpoint salvo com sucesso em: {output_best}")

    for filename in checkpoint_dir.glob("epoch_*.pth"):
        try:
            filename.unlink()
        except OSError:
            pass
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina os 9 modelos do estudo de ablação com MOF (Tabela 2).",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=tuple(EXPERIMENT_BY_NAME),
        help="Executa apenas o experimento indicado (pode ser repetido).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório de artefatos. Padrão: PIGUARD_OUTPUT_DIR ou <repo>/outputs.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Reexecuta mesmo que best_model.pth já exista.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help="Caminho do train.py (padrão: Original/train.py).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = repo_path(args.output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = tuple(dict.fromkeys(args.experiment or EXPERIMENT_BY_NAME))
    selected = tuple(EXPERIMENT_BY_NAME[name] for name in selected_names)

    train_script = (
        repo_path(args.train_script)
        if args.train_script
        else find_repo_script("train.py")
    )
    if train_script is None or not train_script.is_file():
        raise SystemExit(
            "[ERRO] train.py não encontrado. Informe o caminho com --train-script."
        )

    valid_set = REPO_ROOT / "datasets/valid.json"
    dataset_root = REPO_ROOT / "datasets"

    print_hardware_info()
    print(f"\nDiretório de saída: {output_dir}")
    print(f"Script de treino: {train_script}")
    print(f"FORCE_RETRAIN: {args.force_retrain}")

    print("\n--- Verificação de Datasets e Checkpoints Base ---")
    for experiment in EXPERIMENTS:
        path = repo_path(experiment.train_set)
        status = "OK" if path.is_file() else "AUSENTE"
        print(f"[{status}] {experiment.name}: {path}")

    base_oficial = resolve_base_checkpoint("original", output_dir)
    base_reproduzido = resolve_base_checkpoint("reproduced", output_dir)
    print(f"Checkpoint base oficial: {base_oficial}")
    print(f"Checkpoint base reproduzido: {base_reproduzido}")

    for experiment in selected:
        run_mof_experiment(
            experiment=experiment,
            output_dir=output_dir,
            force_retrain=args.force_retrain,
            train_script=train_script,
            valid_set=valid_set,
            dataset_root=dataset_root,
        )

    print("\n" + "=" * 75)
    print("Todos os experimentos selecionados foram processados.")
    print(f"Artefatos persistidos em: {output_dir}")
    for experiment in EXPERIMENTS:
        path = output_dir / experiment.name / "best_model.pth"
        status = "OK (Salvo)" if path.is_file() else "PENDENTE/AUSENTE"
        print(f"   [{status:15}] {experiment.name}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

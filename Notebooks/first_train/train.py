"""PIGuard - Treinamento de Modelos (GPU/CPU).

Conversão do notebook ``Notebooks/first_train/train.ipynb`` para script Python

Uso:
    python Notebooks/first_train/train.py
    python Notebooks/first_train/train.py --experiment train_augmented_rp
    python Notebooks/first_train/train.py --force-retrain

Os artefatos (checkpoints, logs e trigger words) são gravados em ``outputs/`` na
raiz do projeto. Esse destino pode ser alterado com ``--output-dir`` ou com a
variável de ambiente ``PIGUARD_OUTPUT_DIR``.

Três cenários de treinamento e ablação comparativa:
1. ``train_augmented_original``: dataset aumentado oficial do artigo.
2. ``train_augmented_rp``: novo dataset aumentado gerado pelo pipeline reproduzido.
3. ``train_cleaned_original``: base limpa oficial, sem aumento de dados.

Ao final, o Token-wise Recheck (Seção 4.2) é executado para os dois modelos
aumentados. O treinamento em si é delegado ao ``train.py`` do checkout local
(``Original/train.py`` por padrão).
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
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
# Notebooks/<subpasta>/<script>.py -> raiz do repositório.
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PIGUARD_OUTPUT_DIR", REPO_ROOT / "outputs")
).expanduser()

EXPERIMENT_NAMES = (
    "train_augmented_original",
    "train_augmented_rp",
    "train_cleaned_original",
)

DEFAULT_TRAIN_SETS = {
    "train_augmented_original": REPO_ROOT / "datasets/Augmented/train_augmented_original.json",
    "train_augmented_rp": REPO_ROOT / "datasets/Augmented/train_augmented_rp.json",
    "train_cleaned_original": REPO_ROOT / "datasets/Augmented/train_cleaned_original.json",
}

LOG_NAMES = {
    "train_augmented_original": "log_train_augmented_original.txt",
    "train_augmented_rp": "log_train_augmented.txt",
    "train_cleaned_original": "log_train_cleaned_oficial.txt",
}


def repo_path(value: str | os.PathLike[str]) -> Path:
    """Resolve um caminho relativo a partir da raiz do repositório."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def find_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def find_repo_script(filename: str, extra_dirs: Sequence[Path] = ()) -> Path | None:
    """Localiza um script auxiliar do repositório sem depender do diretório atual."""
    directories = (
        REPO_ROOT,
        REPO_ROOT / "Original",
        REPO_ROOT / "Eval",
        REPO_ROOT / "datasets" / "MOF" / "Cache",
        *extra_dirs,
    )
    for directory in directories:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


@contextmanager
def subprocess_environment(extra_paths: Sequence[Path] = ()) -> Iterator[dict[str, str]]:
    """Monta PYTHONPATH com compatibilidade para os imports absolutos legados.

    Alguns arquivos do repositório importam ``envio_gith.Equipe05_PIGuard...``.
    Um diretório temporário com o symlink esperado permite executá-los mesmo
    quando o checkout local tem outro caminho/nome.
    """
    env = os.environ.copy()
    paths = [str(path) for path in extra_paths]
    paths.extend([str(REPO_ROOT), str(REPO_ROOT / "Original")])
    if len(REPO_ROOT.parents) >= 2:
        paths.append(str(REPO_ROOT.parents[1]))

    with tempfile.TemporaryDirectory(prefix="piguard_compat_") as tmp:
        try:
            package_dir = Path(tmp) / "envio_gith"
            package_dir.mkdir(parents=True, exist_ok=True)
            link = package_dir / "Equipe05_PIGuard"
            link.symlink_to(REPO_ROOT, target_is_directory=True)
            paths.insert(0, tmp)
        except OSError:
            # Symlinks podem não estar disponíveis; os caminhos normais continuam válidos.
            pass

        current = env.get("PYTHONPATH")
        if current:
            paths.append(current)
        env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
        yield env


def run_python_script(script: Path, arguments: Sequence[str]) -> None:
    """Executa outro script Python do projeto a partir da raiz do repositório."""
    cmd = [sys.executable, str(script), *arguments]
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    with subprocess_environment([script.parent]) as env:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def print_hardware_info() -> None:
    """Substitui ``!nvidia-smi`` por uma verificação opcional via PyTorch."""
    try:
        import torch  # importação tardia: o script continua útil sem GPU/PyTorch
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


def train_experiment(
    name: str,
    train_set: Path,
    output_dir: Path,
    force_retrain: bool,
    train_script: Path,
    valid_set: Path,
    dataset_root: Path,
) -> bool:
    """Treina um experimento e persiste o melhor checkpoint/log no diretório de saída."""
    target_dir = output_dir / name
    target_best = target_dir / "best_model.pth"

    if target_best.is_file() and not force_retrain:
        print(f"\n[SKIP] Modelo '{name}' já existe em: {target_best}")
        print(
            "Pulando treinamento para economizar recursos. "
            "Use --force-retrain para reexecutar."
        )
        return False

    print("\n" + "=" * 75)
    print(f"INICIANDO EXPERIMENTO: {name}")
    print(f"Dataset de treino: {train_set}")
    print(f"Dataset de validação: {valid_set}")
    print("=" * 75)

    if not train_set.is_file():
        raise SystemExit(
            f"[ERRO] Dataset não encontrado: {train_set}\n"
            "Ajuste o caminho com --train-set NOME=CAMINHO."
        )

    checkpoint_dir = REPO_ROOT / "checkpoints" / name
    logs_dir = REPO_ROOT / "logs" / name
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_python_script(
        train_script,
        [
            "--name", name,
            "--train_set", str(train_set),
            "--valid_set", str(valid_set),
            "--dataset_root", str(dataset_root),
            "--checkpoint_path", str(checkpoint_dir),
            "--logs", str(logs_dir),
            "--batch_size", "32",
            "--eval_batch_size", "32",
            "--save_step", "200",
            "--seed", "42",
            "--max_length", "512",
            "--epochs", "3",
        ],
    )

    local_best = checkpoint_dir / "best_model.pth"
    local_log = logs_dir / LOG_NAMES[name]
    if not local_best.is_file():
        print(f"[AVISO] {local_best} não foi gerado.")
        return False

    print(f"\nCopiando {local_best} para {target_dir}...")
    shutil.copy2(local_best, target_best)
    if local_log.is_file():
        shutil.copy2(local_log, target_dir / LOG_NAMES[name])
    print(f"Treinamento '{name}' salvo com sucesso em: {target_best}")

    # Remove checkpoints intermediários de época para poupar espaço em disco.
    for filename in checkpoint_dir.glob("epoch_*.pth"):
        try:
            filename.unlink()
        except OSError:
            pass
    return True


def run_token_recheck(
    model_name: str,
    output_name: str,
    output_dir: Path,
    recheck_script: Path,
    batch_size: int = 256,
) -> Path | None:
    """Executa o Token-wise Recheck e copia o JSON de trigger words para ``MOF/``."""
    model_path = find_existing(
        [
            REPO_ROOT / "checkpoints" / model_name / "best_model.pth",
            output_dir / model_name / "best_model.pth",
        ]
    )
    if model_path is None and model_name == "train_augmented_original":
        model_path = find_existing([output_dir / "train_augmented_rp" / "best_model.pth"])

    if model_path is None:
        print(f"[AVISO] Checkpoint de '{model_name}' não encontrado; recheck ignorado.")
        return None

    output_path = output_dir / model_name / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nDisparando Token-wise Recheck para {model_name}: {model_path}")
    run_python_script(
        recheck_script,
        [
            "--checkpoint", str(model_path),
            "--output", str(output_path),
            "--batch_size", str(batch_size),
        ],
    )

    mof_dir = REPO_ROOT / "MOF"
    mof_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, mof_dir / output_name)
    return output_path


def parse_train_set_override(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(
            "Use o formato NOME=CAMINHO, por exemplo: "
            "train_augmented_rp=datasets/outro.json"
        )
    if name not in EXPERIMENT_NAMES:
        raise argparse.ArgumentTypeError(
            f"Experimento desconhecido: {name}. Opções: {', '.join(EXPERIMENT_NAMES)}"
        )
    return name, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina os 3 modelos da primeira fase do PIGuard.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        choices=EXPERIMENT_NAMES,
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
        "--train-set",
        action="append",
        type=parse_train_set_override,
        default=[],
        metavar="NOME=CAMINHO",
        help="Substitui o dataset de treino de um experimento.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help="Caminho do train.py (padrão: Original/train.py).",
    )
    parser.add_argument(
        "--token-recheck-script",
        default=None,
        help="Caminho do token_recheck.py (padrão: busca em datasets/MOF/Cache).",
    )
    parser.add_argument(
        "--skip-token-recheck",
        action="store_true",
        help="Não executa o Token-wise Recheck ao final.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = (
        repo_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = tuple(dict.fromkeys(args.experiment or EXPERIMENT_NAMES))
    train_sets = dict(DEFAULT_TRAIN_SETS)
    for name, path in args.train_set:
        train_sets[name] = repo_path(path)

    train_script = (
        repo_path(args.train_script)
        if args.train_script
        else find_repo_script("train.py")
    )
    if train_script is None or not train_script.is_file():
        raise SystemExit(
            "[ERRO] train.py não encontrado. Informe o caminho com --train-script."
        )

    recheck_script = None
    if not args.skip_token_recheck:
        recheck_script = (
            repo_path(args.token_recheck_script)
            if args.token_recheck_script
            else find_repo_script("token_recheck.py")
        )
        if recheck_script is None or not recheck_script.is_file():
            print(
                "[AVISO] token_recheck.py não encontrado; o recheck será ignorado. "
                "Use --token-recheck-script para indicar o caminho."
            )

    valid_set = REPO_ROOT / "datasets/valid.json"
    dataset_root = REPO_ROOT / "datasets"

    print_hardware_info()
    print(f"\nDiretório de saída: {output_dir}")
    print(f"Script de treino: {train_script}")
    print(f"FORCE_RETRAIN: {args.force_retrain}")

    # Reaproveitamento de checkpoints legados, como no notebook original.
    if "train_augmented_original" in selected:
        best_1 = output_dir / "train_augmented_original/best_model.pth"
        legacy_1 = output_dir / "train_augmented_rp/best_model.pth"
        if (
            not best_1.is_file()
            and legacy_1.is_file()
            and not (REPO_ROOT / "datasets/Augmented/train_augmented_rp.json").is_file()
        ):
            print(f"Reaproveitando checkpoint legado oficial para {best_1.parent}...")
            best_1.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_1, best_1)

    if "train_cleaned_original" in selected:
        best_3 = output_dir / "train_cleaned_original/best_model.pth"
        legacy_3 = output_dir / "train_cleaned/best_model.pth"
        if not best_3.is_file() and legacy_3.is_file():
            print(f"Reaproveitando checkpoint legado para {best_3.parent}...")
            best_3.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_3, best_3)

    for name in selected:
        train_experiment(
            name=name,
            train_set=train_sets[name],
            output_dir=output_dir,
            force_retrain=args.force_retrain,
            train_script=train_script,
            valid_set=valid_set,
            dataset_root=dataset_root,
        )

    if recheck_script is not None:
        if "train_augmented_original" in selected:
            run_token_recheck(
                "train_augmented_original",
                "trigger_words_train_augmented_original.json",
                output_dir,
                recheck_script,
            )
        if "train_augmented_rp" in selected:
            run_token_recheck(
                "train_augmented_rp",
                "trigger_words_train_augmented.json",
                output_dir,
                recheck_script,
            )

    print("\n" + "=" * 75)
    print("Processamento concluído.")
    print(f"Artefatos persistidos em: {output_dir}")
    for name in EXPERIMENT_NAMES:
        best = output_dir / name / "best_model.pth"
        status = "OK (Presente)" if best.is_file() else "Pendente/Ausente"
        print(f"   - [{status}] {best}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

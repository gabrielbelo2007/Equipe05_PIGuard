"""PIGuard - Estudo de Ablação: Shortcut Mitigation vs. MOF.

Uso:
    python Notebooks/train_extras/train.py
    python Notebooks/train_extras/train.py --experiment train_augmented_shortcut_rp
    python Notebooks/train_extras/train.py --force-retrain

Treina os dois modelos de mitigação de atalhos (reponderação da perda pelos
grupos (z, y), sem MOF):
1. ``train_augmented_shortcut_original``: dataset aumentado original + trigger
   words extraídas do modelo aumentado original.
2. ``train_augmented_shortcut_rp``: dataset aumentado reproduzido + trigger
   words extraídas do modelo aumentado reproduzido.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT_DIR = (REPO_ROOT / "outputs").resolve()


@dataclass(frozen=True)
class ShortcutExperiment:
    name: str
    train_set: str
    trigger_words: str
    base_experiment: str
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup: int = 100
    save_step: int = 200
    save_threshold: float = 0.8


EXPERIMENTS = (
    ShortcutExperiment(
        name="train_augmented_shortcut_original",
        train_set="datasets/Augmented/train_augmented_original.json",
        trigger_words="datasets/MOF/Cache/trigger_words_train_augmented_semi_rp.json",
        base_experiment="train_augmented_original",
    ),
    ShortcutExperiment(
        name="train_augmented_shortcut_rp",
        train_set="datasets/Augmented/train_augmented_rp.json",
        trigger_words="datasets/MOF/Cache/trigger_words_train_augmented_rp.json",
        base_experiment="train_augmented_rp",
    ),
)

EXPERIMENT_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_trigger_words(experiment: ShortcutExperiment, output_dir: Path) -> Path:
    base_dir = output_dir / experiment.base_experiment
    candidates = [base_dir / f"trigger_words_{experiment.base_experiment}.json"]
    if experiment.base_experiment == "train_augmented_rp":
        candidates.append(base_dir / "trigger_words_train_augmented.json")
    return find_existing(candidates) or repo_path(experiment.trigger_words)


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
        REPO_ROOT / "datasets" / "MOF" / "Cache",
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


def build_trigger_token_ids(tokenizer, trigger_words_path: Path) -> set[int]:
    """Mapeia trigger words para token IDs do vocabulário do DeBERTa."""
    with trigger_words_path.open("r", encoding="utf-8") as handle:
        raw_words = json.load(handle)
    trigger_set = {
        word.strip() for word in raw_words if isinstance(word, str) and word.strip()
    }
    special_ids = set(tokenizer.all_special_ids)
    trigger_ids: set[int] = set()
    for token_string, token_id in tokenizer.get_vocab().items():
        if token_id in special_ids:
            continue
        if token_string.lstrip("▁") in trigger_set:
            trigger_ids.add(token_id)
    return trigger_ids


def compute_group_weights(
    z_list: Sequence[int], y_list: Sequence[int], mode: str = "cond_prob"
) -> dict[str, object]:
    """Calcula pesos por grupo (z, y), equivalentes aos de train_shortcut.py."""
    total = len(y_list)
    counts = {(z, y): 0 for z in (0, 1) for y in (0, 1)}
    for z_value, y_value in zip(z_list, y_list):
        counts[(int(z_value), int(y_value))] += 1
    z_totals = {z: counts[(z, 0)] + counts[(z, 1)] for z in (0, 1)}

    raw_weights: dict[tuple[int, int], float] = {}
    conditional_probabilities: dict[tuple[int, int], float] = {}
    for z in (0, 1):
        for y in (0, 1):
            count_zy = counts[(z, y)]
            count_z = z_totals[z]
            probability = (count_zy / count_z) if count_z > 0 else 0.0
            conditional_probabilities[(z, y)] = probability
            raw_weights[(z, y)] = (1.0 / probability) if probability > 0 else 0.0

    if total > 0:
        mean_weight = sum(
            raw_weights[(int(z), int(y))] for z, y in zip(z_list, y_list)
        ) / total
    else:
        mean_weight = 0.0
    normalized = {
        key: (value / mean_weight if mean_weight > 0 else 1.0)
        for key, value in raw_weights.items()
    }
    return {
        "total_samples": total,
        "mode": mode,
        "group_counts": {f"z={z},y={y}": counts[(z, y)] for z in (0, 1) for y in (0, 1)},
        "group_frequencies": {
            f"z={z},y={y}": counts[(z, y)] / total if total else 0.0
            for z in (0, 1) for y in (0, 1)
        },
        "cond_probs": {
            f"P(y={y}|z={z})": conditional_probabilities[(z, y)]
            for z in (0, 1) for y in (0, 1)
        },
        "normalized_weights": {
            f"z={z},y={y}": normalized[(z, y)] for z in (0, 1) for y in (0, 1)
        },
    }


def preview_group_distribution(
    experiment: ShortcutExperiment, batch_size_tokens: int = 4096, *,
    trigger_path: Path | None = None,
) -> None:
    """Pré-visualiza distribuições (z, y) e pesos, como no notebook original."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[AVISO] Transformers não instalado; pré-visualização ignorada.")
        return

    dataset_path = repo_path(experiment.train_set)
    if trigger_path is None:
        trigger_path = resolve_trigger_words(experiment, repo_path(DEFAULT_OUTPUT_DIR))
    if not dataset_path.is_file() or not trigger_path.is_file():
        print(f"\n[AVISO] Arquivos ausentes para pré-visualizar {experiment.name}.")
        return

    print(f"\n--- Distribuição e Pesos para {experiment.name} ---")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    trigger_ids = build_trigger_token_ids(tokenizer, trigger_path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    prompts = [item["prompt"] for item in data]
    labels = [int(item["label"]) for item in data]
    z_flags: list[int] = []
    for start in range(0, len(prompts), batch_size_tokens):
        batch = prompts[start : start + batch_size_tokens]
        encoded = tokenizer(
            batch, truncation=True, max_length=512, add_special_tokens=False
        )["input_ids"]
        z_flags.extend(
            1 if any(token_id in trigger_ids for token_id in sequence) else 0
            for sequence in encoded
        )

    summary = compute_group_weights(z_flags, labels, mode="cond_prob")
    print(
        f"{'Grupo':<18}{'Contagem':>12}{'Frequência':>15}"
        f"{'P(y|z)':>12}{'Peso w_i':>12}"
    )
    for z in (0, 1):
        for y in (0, 1):
            key = f"z={z},y={y}"
            print(
                f"{key:<18}"
                f"{summary['group_counts'][key]:>12}"  # type: ignore[index]
                f"{summary['group_frequencies'][key]:>14.2%}"  # type: ignore[index]
                f"{summary['cond_probs'][f'P(y={y}|z={z})']:>12.4f}"  # type: ignore[index]
                f"{summary['normalized_weights'][key]:>12.4f}"  # type: ignore[index]
            )


def run_shortcut_experiment(
    experiment: ShortcutExperiment,
    output_dir: Path,
    force_retrain: bool,
    train_script: Path,
    valid_set: Path,
    dataset_root: Path,
) -> bool:
    output_target = output_dir / experiment.name
    output_best = output_target / "best_model.pth"

    if output_best.is_file() and not force_retrain:
        print(f"\n[SKIP] Modelo '{experiment.name}' já existe em: {output_best}")
        print("Use --force-retrain para reexecutar.")
        return False

    train_set = repo_path(experiment.train_set)
    trigger_words = resolve_trigger_words(experiment, output_dir)
    if not train_set.is_file():
        raise SystemExit(f"[ERRO] Dataset não encontrado: {train_set}")
    if not trigger_words.is_file():
        raise SystemExit(f"[ERRO] Trigger words não encontradas: {trigger_words}")

    print("\n" + "=" * 75)
    print(f"INICIANDO SHORTCUT MITIGATION: {experiment.name}")
    print(f"Dataset: {train_set}")
    print(f"Trigger words: {trigger_words}")
    print(
        f"Configuração: epochs={experiment.epochs}, batch_size={experiment.batch_size}, "
        f"lr={experiment.learning_rate}, warmup={experiment.warmup}, "
        f"save_step={experiment.save_step}, save_thres={experiment.save_threshold}"
    )
    print("=" * 75)

    checkpoint_dir = REPO_ROOT / "checkpoints" / experiment.name
    logs_dir = REPO_ROOT / "logs" / experiment.name
    output_target.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", str(train_script),
        "--name", experiment.name,
        "--train_set", str(train_set),
        "--trigger_words", str(trigger_words),
        "--valid_set", str(valid_set),
        "--dataset_root", str(dataset_root),
        "--checkpoint_path", str(checkpoint_dir),
        "--logs", str(logs_dir),
        "--batch_size", str(experiment.batch_size),
        "--eval_batch_size", str(experiment.batch_size),
        "--seed", "42",
        "--epochs", str(experiment.epochs),
        "--lr", str(experiment.learning_rate),
        "--warmup", str(experiment.warmup),
        "--max_length", "512",
        "--save_step", str(experiment.save_step),
        "--save_thres", str(experiment.save_threshold),
        "--weight_mode", "cond_prob",
    ]

    print("\n$ " + " ".join(shlex.quote(part) for part in cmd) + "\n", flush=True)
    with subprocess_environment([train_script.parent]) as env:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
        process.stdout.close()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Erro no treinamento de {experiment.name} "
            f"(código de saída: {return_code})"
        )

    local_best = checkpoint_dir / "best_model.pth"
    local_log = logs_dir / f"log_{experiment.name}.txt"
    local_summary = find_existing(
        [
            checkpoint_dir / "group_weights_summary.json",
            logs_dir / "group_weights_summary.json",
        ]
    )
    if not local_best.is_file():
        print(f"[AVISO] {local_best} não foi gerado.")
        return False

    print(f"\nCopiando artefatos para {output_target}...")
    shutil.copy2(local_best, output_best)
    if local_log.is_file():
        shutil.copy2(local_log, output_target / f"log_{experiment.name}.txt")
    if local_summary is not None:
        shutil.copy2(local_summary, output_target / "group_weights_summary.json")
    print(f"Modelo salvo com sucesso em: {output_best}")

    for filename in checkpoint_dir.glob("epoch_*.pth"):
        try:
            filename.unlink()
        except OSError:
            pass
    return True


def run_residual_recheck(
    experiment_name: str,
    output_dir: Path,
    recheck_script: Path,
    batch_size: int = 512,
) -> Path | None:
    checkpoint_path = find_existing(
        [
            output_dir / experiment_name / "best_model.pth",
            REPO_ROOT / "checkpoints" / experiment_name / "best_model.pth",
        ]
    )
    if checkpoint_path is None:
        print(f"[AVISO] Checkpoint de '{experiment_name}' não encontrado; recheck ignorado.")
        return None

    output_path = output_dir / experiment_name / f"trigger_words_residual_{experiment_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExecutando Token-wise Recheck residual para {experiment_name}...")
    cmd = [
        sys.executable,
        str(recheck_script),
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_path),
        "--batch_size", str(batch_size),
    ]
    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    with subprocess_environment([recheck_script.parent]) as env:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

    local_checkpoint_dir = REPO_ROOT / "checkpoints" / experiment_name
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        output_path,
        local_checkpoint_dir / f"trigger_words_residual_{experiment_name}.json",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina os modelos de Shortcut Mitigation (sem MOF).",
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
        help="Diretório de artefatos (padrão: <repo>/outputs).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Reexecuta mesmo que best_model.pth já exista.",
    )
    parser.add_argument(
        "--train-script",
        default=None,
        help="Caminho do train_shortcut.py (padrão: Original/train_shortcut.py).",
    )
    parser.add_argument(
        "--token-recheck-script",
        default=None,
        help="Caminho do token_recheck.py (padrão: datasets/MOF/Cache/token_recheck.py).",
    )
    parser.add_argument(
        "--skip-token-recheck",
        action="store_true",
        help="Não executa o Token-wise Recheck residual.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Não calcula a distribuição dos grupos (z, y) antes do treino.",
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
        else find_repo_script("train_shortcut.py")
    )
    if train_script is None or not train_script.is_file():
        raise SystemExit(
            "[ERRO] train_shortcut.py não encontrado. "
            "Informe o caminho com --train-script."
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
                "[AVISO] token_recheck.py não encontrado; recheck residual será "
                "ignorado. Use --token-recheck-script para indicar o caminho."
            )

    valid_set = REPO_ROOT / "datasets/valid.json"
    dataset_root = REPO_ROOT / "datasets"

    print_hardware_info()
    print(f"\nDiretório de saída: {output_dir}")
    print(f"Script de treino: {train_script}")
    print(f"FORCE_RETRAIN: {args.force_retrain}")

    print("\n--- Verificação de Datasets e Trigger Words ---")
    for experiment in EXPERIMENTS:
        dataset = repo_path(experiment.train_set)
        triggers = resolve_trigger_words(experiment, output_dir)
        print(
            f"[{'OK' if dataset.is_file() else 'AUSENTE'}] {experiment.name} dataset: {dataset}"
        )
        print(
            f"[{'OK' if triggers.is_file() else 'AUSENTE'}] {experiment.name} triggers: {triggers}"
        )

    if not args.skip_preview:
        for experiment in selected:
            preview_group_distribution(
                experiment, trigger_path=resolve_trigger_words(experiment, output_dir)
            )

    for experiment in selected:
        run_shortcut_experiment(
            experiment=experiment,
            output_dir=output_dir,
            force_retrain=args.force_retrain,
            train_script=train_script,
            valid_set=valid_set,
            dataset_root=dataset_root,
        )

    if recheck_script is not None:
        for experiment in selected:
            run_residual_recheck(
                experiment.name, output_dir, recheck_script
            )

    print("\n" + "=" * 75)
    print("Treinamentos de Shortcut Mitigation concluídos.")
    print(f"Artefatos persistidos em: {output_dir}")
    for experiment in EXPERIMENTS:
        path = output_dir / experiment.name / "best_model.pth"
        status = "OK (Salvo)" if path.is_file() else "PENDENTE"
        print(f"   [{status}] {experiment.name}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

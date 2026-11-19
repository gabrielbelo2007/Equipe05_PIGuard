#!/usr/bin/env python3
"""Pipeline Upgrade — PIGuard + Novo MOF + Laya English.

Executa, em um único processo Python, as etapas:

1. Etapa 0: gera o pool do Novo MOF (v2), separa ``MOF_v2_train.json`` (762)
   e ``calibration_heldout.json`` (238 benignos + 102 ataques);
2. Tópico 1: avalia os cinco checkpoints ``*_rp`` com PIGuard puro, Laya
   English não calibrado e Laya English calibrado;
3. Tópico 2: retreina do zero ``train_augmented_corrected_new_mof`` com a base
   corrigida intacta + Novo MOF e avalia nas três modalidades;
4. Tópico 3: consolida tabelas, intervalos de confiança, latência e gráficos.

Entradas padrão:

* Corrections: ``Corrections/outputs``;
* checkpoints ``rp``: ``outputs``;
* saída: ``Upgrade/outputs``.

A Etapa 0 e a geração de amostras exigem ``OPENROUTER_API_KEY`` ou
``OPENAI_API_KEY`` no ambiente ou em ``<repo>/.env``. As demais etapas usam
apenas os artefatos locais.

Requisitos deste fluxo: ``transformers==4.48.3`` (ModernBERT),
``safetensors``, ``huggingface_hub`` e ``pandas``/``matplotlib`` para os
relatórios — todos já incluídos no ``requirements.txt`` principal na raiz do
repositório.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def find_repo_root(start: Path) -> Path:
    """Sobe até encontrar o diretório que contém ``datasets`` e ``Original``."""
    for candidate in (start, *start.parents):
        if (candidate / "datasets").is_dir() and (candidate / "Original").is_dir():
            return candidate
    raise RuntimeError(
        f"Não foi possível localizar a raiz do repositório a partir de {start}."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import default_corrections_dir, load_api_key, resolve_path

NTFY_URL = "https://ntfy.sh/"
NTFY_ENABLED = False
NTFY_TOPIC = os.environ.get("PIGUARD_NTFY_TOPIC", "piguard_gabriel_treinos")


def notify_ntfy(
    title: str,
    message: str,
    *,
    priority: int = 3,
    tags: Sequence[str] | None = None,
    topic: str | None = None,
    timeout: int = 15,
) -> None:
    """Envia uma notificação opcional para o ntfy.sh."""
    if not NTFY_ENABLED:
        return
    payload = {
        "topic": topic or NTFY_TOPIC,
        "title": str(title)[:250],
        "message": str(message)[:3800],
        "priority": int(priority),
    }
    if tags:
        payload["tags"] = list(tags)
    request = urllib.request.Request(
        NTFY_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except Exception as exc:  # notificação nunca deve interromper o pipeline
        print(f"[ntfy] Aviso ao enviar notificação ({title}): {exc}", flush=True)


@contextmanager
def pipeline_stage(name: str) -> Iterator[list[str]]:
    """Executa uma etapa e notifica início/fim/falha quando ``--ntfy`` está ativo."""
    started = time.monotonic()
    notes: list[str] = []
    try:
        yield notes
    except BaseException as exc:
        elapsed = time.monotonic() - started
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        message = f"Etapa '{name}' falhou após {elapsed:.1f}s.\nErro: {detail}"
        if notes:
            message += "\nContexto:\n" + "\n".join(str(note) for note in notes)
        notify_ntfy(
            f"[PIGuard Upgrade] Erro: {name}",
            message,
            priority=4,
            tags=["x", "warning"],
        )
        raise
    else:
        elapsed = time.monotonic() - started
        message = f"Etapa '{name}' concluída em {elapsed:.1f}s."
        if notes:
            message += "\n" + "\n".join(str(note) for note in notes)
        notify_ntfy(
            f"[PIGuard Upgrade] Concluída: {name}",
            message,
            tags=["white_check_mark"],
        )


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Path | str | None]:
    """Resolve todos os caminhos contra a raiz, independentemente do ``cwd``."""
    repo = resolve_path(args.repo_root, REPO_ROOT)
    output_dir = resolve_path(args.output_dir, repo / "Upgrade" / "outputs", repo_root=repo)
    rp_root = resolve_path(args.rp_root, repo / "outputs", repo_root=repo)
    corrections_dir = resolve_path(
        args.corrections_dir,
        default_corrections_dir(repo),
        repo_root=repo,
    )

    trigger_report = resolve_path(
        args.trigger_report,
        corrections_dir / "augmented_corrected" / "trigger_words.report.json",
        repo_root=repo,
    )
    if not trigger_report.is_file():
        fallback = trigger_report.with_name("trigger_words.json")
        if fallback.is_file():
            trigger_report = fallback

    return {
        "repo": repo,
        "output_dir": output_dir,
        "rp_root": rp_root,
        "corrections_dir": corrections_dir,
        "trigger_report": trigger_report,
        "corrected_checkpoint": resolve_path(
            args.corrected_checkpoint,
            corrections_dir / "augmented_corrected" / "best_model.pth",
            repo_root=repo,
        ),
        "mof_v2": resolve_path(
            args.mof_v2,
            output_dir / "step0" / "MOF_v2_train.json",
            repo_root=repo,
        ),
        "heldout": resolve_path(
            args.heldout,
            output_dir / "step0" / "calibration_heldout.json",
            repo_root=repo,
        ),
        "corrected_train": resolve_path(
            args.corrected_train,
            corrections_dir / "data" / "train_augmented_corrected.json",
            repo_root=repo,
        ),
        "llm_augmented": resolve_path(
            args.llm_augmented,
            corrections_dir / "augmented" / "LLM_augmented.json",
            repo_root=repo,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o pipeline Upgrade completo (Etapa 0 + Tópicos 1-3) "
            "sem depender do Google Colab."
        )
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--corrections-dir", default=None)
    parser.add_argument("--rp-root", default=None)
    parser.add_argument("--trigger-report", default=None)
    parser.add_argument("--corrected-checkpoint", default=None)
    parser.add_argument("--mof-v2", default=None, help="Arquivo MOF v2 usado pelo Tópico 2.")
    parser.add_argument("--heldout", default=None, help="Arquivo de calibração usado pelos Tópicos 1 e 2.")
    parser.add_argument("--corrected-train", default=None)
    parser.add_argument("--llm-augmented", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--skip-step0", action="store_true")
    parser.add_argument("--skip-topic1", action="store_true")
    parser.add_argument("--skip-topic2", action="store_true")
    parser.add_argument("--skip-topic3", action="store_true")
    parser.add_argument("--force-step0", action="store_true")
    parser.add_argument("--force-topic1", action="store_true")
    parser.add_argument("--force-retrain-topic2", action="store_true")
    parser.add_argument("--force-eval-topic2", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fpr-budget", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--laya-batch-size", type=int, default=16)
    parser.add_argument("--ntfy", action="store_true")
    parser.add_argument("--ntfy-topic", default=NTFY_TOPIC)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global NTFY_ENABLED, NTFY_TOPIC

    parser = build_parser()
    args = parser.parse_args(argv)
    NTFY_ENABLED = bool(args.ntfy)
    NTFY_TOPIC = args.ntfy_topic

    paths = _resolve_inputs(args)
    repo = paths["repo"]
    output_dir = paths["output_dir"]
    corrections_dir = paths["corrections_dir"]
    rp_root = paths["rp_root"]
    trigger_report = paths["trigger_report"]
    corrected_checkpoint = paths["corrected_checkpoint"]
    mof_v2 = paths["mof_v2"]
    heldout = paths["heldout"]
    corrected_train = paths["corrected_train"]
    llm_augmented = paths["llm_augmented"]

    print(
        json.dumps(
            {
                "repo": str(repo),
                "corrections_dir": str(corrections_dir),
                "rp_root": str(rp_root),
                "output_dir": str(output_dir),
                "trigger_report": str(trigger_report),
                "corrected_checkpoint": str(corrected_checkpoint),
                "mof_v2": str(mof_v2),
                "heldout": str(heldout),
                "skip_step0": args.skip_step0,
                "skip_topic1": args.skip_topic1,
                "skip_topic2": args.skip_topic2,
                "skip_topic3": args.skip_topic3,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.dry_run:
        print("\n[dry-run] Nenhuma etapa foi executada.", flush=True)
        return 0

    api_key: str | None = None
    if not args.skip_step0:
        api_key = load_api_key(args.api_key, repo_root=repo, prompt=args.prompt_api_key)
        if not api_key:
            raise RuntimeError(
                "A Etapa 0 exige OPENROUTER_API_KEY/OPENAI_API_KEY. "
                "Defina a variável, crie <repo>/.env ou use --api-key."
            )
    if not args.skip_step0:
        with pipeline_stage("Etapa 0: Novo MOF + held-out") as notes:
            from Upgrade.Pipeline.step0_new_mof_and_heldout import run_step0

            report = run_step0(
                trigger_path=trigger_report,
                checkpoint_path=corrected_checkpoint,
                output_dir=output_dir / "step0",
                repo_root=repo,
                corrections_dir=corrections_dir,
                workers=args.workers,
                seed=args.seed,
                force=args.force_step0,
                api_key=api_key,
            )
            if args.mof_v2 is None:
                mof_v2 = Path(report["mof_v2_train_path"])
            if args.heldout is None:
                heldout = Path(report["calibration_heldout_path"])
            notes.append(
                f"MOF_v2={report['mof_v2_count']} | "
                f"heldout={report['heldout_total_count']} | "
                f"disjunto={report['disjoint_check']['disjoint']}"
            )
            print(
                f"[Etapa 0] MOF_v2={report['mof_v2_count']} | "
                f"heldout={report['heldout_total_count']} | "
                f"disjunto={report['disjoint_check']['disjoint']}",
                flush=True,
            )

    if (not args.skip_topic1 or not args.skip_topic2) and not Path(heldout).is_file():
        raise FileNotFoundError(f"Conjunto de calibração ausente: {heldout}")
    if not args.skip_topic2 and not Path(mof_v2).is_file():
        raise FileNotFoundError(f"MOF v2 de treino ausente: {mof_v2}")

    if not args.skip_topic1:
        with pipeline_stage("Tópico 1: modelos rp + Laya English") as notes:
            from Upgrade.topic1_rp_laya_evaluation import RP_MODELS, run_topic1

            payload = run_topic1(
                drive_rp_root=rp_root,
                heldout_path=heldout,
                output_dir=output_dir,
                repo_root=repo,
                models=RP_MODELS,
                fpr_budget=args.fpr_budget,
                n_boot=args.n_boot,
                seed=args.seed,
                batch_size=args.batch_size,
                laya_batch_size=args.laya_batch_size,
                force=args.force_topic1,
            )
            notes.append(f"modelos avaliados: {len(payload['results'])}")
            print(f"[Tópico 1] {len(payload['results'])} modelos avaliados.", flush=True)

    if not args.skip_topic2:
        with pipeline_stage("Tópico 2: base corrigida + Novo MOF") as notes:
            from Upgrade.topic2_corrected_new_mof import run_topic2

            payload = run_topic2(
                mof_v2_path=mof_v2,
                heldout_path=heldout,
                output_dir=output_dir,
                corrected_train_path=corrected_train,
                llm_augmented_path=llm_augmented,
                repo_root=repo,
                fpr_budget=args.fpr_budget,
                n_boot=args.n_boot,
                seed=args.seed,
                batch_size=args.batch_size,
                laya_batch_size=args.laya_batch_size,
                force_retrain=args.force_retrain_topic2,
                force_eval=args.force_eval_topic2,
            )
            stats = payload["dataset_stats"]
            notes.append(f"treino={stats['total_count']} amostras")
            print(
                f"[Tópico 2] treino concluído com {stats['total_count']} amostras.",
                flush=True,
            )

    if not args.skip_topic3:
        with pipeline_stage("Tópico 3: consolidação geral") as notes:
            from Upgrade.topic3_consolidated_report import run_topic3

            bundle = run_topic3(
                output_dir=output_dir,
                drive_rp_root=rp_root,
                corrections_dir=corrections_dir,
                repo_root=repo,
            )
            notes.append(f"relatório={bundle['markdown_report']}")
            print(f"[Tópico 3] relatório: {bundle['markdown_report']}", flush=True)

    print(f"\nPipeline concluído. Artefatos em: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

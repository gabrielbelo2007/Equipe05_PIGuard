#!/usr/bin/env python3
"""Comparação unificada da linhagem ``corrected`` do PIGuard.

Este script é a versão executável do notebook ``corrected_comparison.ipynb``.
Ele compara, sob o mesmo protocolo (PIGuard puro, Laya English não calibrado,
Laya English calibrado, Tensor Trust, latência e IC 95% por Paired Bootstrap):

1. ``augmented_corrected`` — base corrigida sem MOF;
2. ``augmented_mof_corrected`` — base corrigida + MOF antigo;
3. ``train_augmented_corrected_new_mof`` — base corrigida + Novo MOF v2;
4. ``augmented_mof_original`` — âncora da reprodução original;
5. ``PIGuard_HF`` — checkpoint oficial ``leolee99/PIGuard``.

Uso::

    python Upgrade/run_corrected_comparison.py
    python Upgrade/run_corrected_comparison.py --dry-run
    python Upgrade/run_corrected_comparison.py --heldout Upgrade/outputs/step0/calibration_heldout.json

Por padrão, os artefatos são lidos de ``Corrections/outputs`` e
``Upgrade/outputs/step0`` e gravados em ``Upgrade/outputs/corrected_only``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


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

from Upgrade import default_corrections_dir, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara a linhagem corrected (augmented_corrected, "
            "augmented_mof_corrected e train_augmented_corrected_new_mof) "
            "com as âncoras oficiais, usando Laya English."
        )
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--corrections-dir",
        default=None,
        help="Diretório de artefatos do Corrections (padrão: Corrections/outputs).",
    )
    parser.add_argument(
        "--heldout",
        default=None,
        help="Caminho para calibration_heldout.json (padrão: <output>/step0/...).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório raiz dos artefatos Upgrade (padrão: PIGUARD_UPGRADE_OUTPUT ou <repo>/Upgrade/outputs).",
    )
    parser.add_argument("--fpr-budget", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--laya-batch-size", type=int, default=16)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignora o cache de resultados e reavalia todos os modelos.",
    )
    parser.add_argument(
        "--force-reval-new-mof",
        action="store_true",
        help="Reavalia especificamente o modelo train_augmented_corrected_new_mof.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra os caminhos resolvidos e encerra sem avaliar.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = resolve_path(args.repo_root, REPO_ROOT)
    upgrade_default = resolve_path(
        os.environ.get("PIGUARD_UPGRADE_OUTPUT"),
        repo_root / "Upgrade" / "outputs",
        repo_root=repo_root,
    )
    output_dir = resolve_path(args.output_dir, upgrade_default, repo_root=repo_root)
    corrections_dir = resolve_path(
        args.corrections_dir,
        default_corrections_dir(repo_root),
        repo_root=repo_root,
    )
    heldout_path = resolve_path(
        args.heldout,
        output_dir / "step0" / "calibration_heldout.json",
        repo_root=repo_root,
    )

    print(
        json.dumps(
            {
                "repo": str(repo_root),
                "corrections_dir": str(corrections_dir),
                "heldout": str(heldout_path),
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        print("\n[dry-run] Nenhuma avaliação foi executada.", flush=True)
        return 0

    from Upgrade.corrected_comparison import run_corrected_comparison

    bundle = run_corrected_comparison(
        corrections_dir=corrections_dir,
        heldout_path=heldout_path,
        output_dir=output_dir,
        repo_root=repo_root,
        fpr_budget=args.fpr_budget,
        n_boot=args.n_boot,
        seed=args.seed,
        batch_size=args.batch_size,
        laya_batch_size=args.laya_batch_size,
        force=args.force,
        force_reval_new_mof=args.force_reval_new_mof,
    )
    print(
        f"\nComparação concluída para {len(bundle['results'])} modelos.\n"
        f"Relatório: {bundle['markdown_report']}\n"
        f"Resumo CSV: {output_dir / 'corrected_only' / 'corrected_comparison_summary.csv'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

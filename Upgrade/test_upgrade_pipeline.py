"""Testes unitários e de integração do pipeline Upgrade."""

from __future__ import annotations

import ast
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

# Permite executar os testes diretamente de qualquer diretório.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Upgrade import ALL_BENCHMARK_SETS, PUBLIC_SETS, read_json, write_json
from Upgrade.laya_english_cascade import (
    DEFAULT_PI_QUESTION,
    calibrate_laya_on_heldout,
    cascade_predict,
    compute_binary_nll,
    compute_cascade_latency_metrics,
    compute_laya_score_from_window_logits,
    fit_temperature_nll,
    fit_threshold_tl,
    paired_bootstrap_ci,
    score_dataset_from_window_logits,
    split_text_into_window_sequences,
)
from Upgrade.step0_new_mof_and_heldout import (
    DOMAIN_NAMES,
    HELDOUT_BENIGN_QUOTAS,
    POOL_QUOTAS,
    TRAIN_QUOTAS,
    contains_chosen_words,
    load_forbidden_texts,
    load_trigger_weights,
    partition_pool_into_train_and_heldout,
    run_step0,
    sample_weighted_triggers_without_replacement,
    validate_mof_candidate,
    verify_disjointness,
)
from Upgrade.topic1_rp_laya_evaluation import RP_MODELS, run_topic1
from Upgrade.topic2_corrected_new_mof import build_combined_training_dataset, run_topic2
from Upgrade.topic3_consolidated_report import run_topic3


class _DummyWordTokenizer:
    """Tokenizador determinístico leve para testes sem download de pesos."""

    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103
    pad_token_id = 0
    mask_token = "[MASK]"

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = [abs(hash(tok)) % 10000 + 10 for tok in text.split()]
        if add_special_tokens:
            return [self.cls_token_id] + tokens + [self.sep_token_id]
        return tokens

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


class _FakeLayaEngine:
    """Motor Laya simulado para validar cache, janelamento e integração Tópicos 1-3."""

    def extract_window_logits(self, texts, *, batch_size: int = 16):
        logits_out = []
        ms_out = []
        for idx, text in enumerate(texts):
            # Textos com palavras-chave maliciosas recebem logits altos na classe 1 (direct_override)
            lower = text.lower()
            is_mal = any(
                kw in lower
                for kw in (
                    "ignore",
                    "system",
                    "reveal",
                    "hack",
                    "override",
                    "password",
                    "injection",
                    "exfiltrate",
                    "jailbreak",
                )
            )
            if is_mal:
                win = [[-1.5, 2.2, 0.8, 0.5]]
            else:
                win = [[2.0, -1.2, -1.0, -1.4]]
            # Simula segunda janela quando texto é longo
            if len(text.split()) > 120:
                win.append([1.8, -1.1, -1.0, -1.3])
            logits_out.append(win)
            ms_out.append(28.0 + (idx % 5))
        return logits_out, ms_out


class TestStep0NewMofAndHeldout(unittest.TestCase):
    def test_trigger_weights_20_80_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "trigger_words.report.json"
            write_json(
                report_path,
                {
                    "triggers": [
                        {"word": "alpha", "token_id": 1, "p_injection": 0.99},
                        {"word": "beta", "token_id": 2, "p_injection": 0.60},
                        {"word": "gamma", "token_id": 3, "p_injection": 0.50},
                    ]
                },
            )
            words, probs, meta = load_trigger_weights(report_path)
            self.assertEqual(words, ["alpha", "beta", "gamma"])
            self.assertAlmostEqual(sum(probs), 1.0, places=9)
            # gamma tem s=0, então recebe apenas a parcela uniforme 0.20 / 3
            self.assertAlmostEqual(probs[2], 0.20 / 3.0, places=6)
            # alpha tem maior severidade (0.49 vs 0.10)
            self.assertGreater(probs[0], probs[1])
            self.assertGreater(probs[1], probs[2])

            rng = random.Random(42)
            for k in (1, 2, 3):
                picked = sample_weighted_triggers_without_replacement(words, probs, k, rng)
                self.assertEqual(len(picked), k)
                self.assertEqual(len(set(picked)), k)

    def test_candidate_validation_and_forbidden_overlap(self):
        tok = _DummyWordTokenizer()
        forbidden_exact, forbidden_norm, _ = load_forbidden_texts(REPO_ROOT)
        # Garante que valid.json e benchmarks estão presentes
        valid_sample = read_json(REPO_ROOT / "datasets" / "valid.json")[0]["prompt"]
        self.assertIn(valid_sample.strip().lower(), forbidden_exact)

        def ok_judge(payload, api_key):
            return {"choices": [{"message": {"content": "NO"}}]}

        def yes_judge(payload, api_key):
            return {"choices": [{"message": {"content": "YES"}}]}

        # 1) Falha por fronteira de palavra (substring "alphabet" não casa "alpha")
        ok, reason, _ = validate_mof_candidate(
            "Explain the Greek alphabet clearly.",
            ["alpha"],
            tok,
            [],
            forbidden_exact,
            "",
            forbidden_norm_set=forbidden_norm,
            call_fn=ok_judge,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_chosen_words")
        self.assertTrue(contains_chosen_words("Use alpha and beta here.", ["alpha", "beta"]))

        # 2) Falha por exceder 256 tokens
        long_text = "alpha " + " ".join(["word"] * 260)
        ok, reason, _ = validate_mof_candidate(
            long_text,
            ["alpha"],
            tok,
            [],
            forbidden_exact,
            "",
            forbidden_norm_set=forbidden_norm,
            call_fn=ok_judge,
        )
        self.assertFalse(ok)
        self.assertIn("token_limit_exceeded", reason)

        # 3) Falha por resposta YES do juiz
        ok, reason, _ = validate_mof_candidate(
            "Please explain how alpha works in Python.",
            ["alpha"],
            tok,
            [],
            forbidden_exact,
            "",
            forbidden_norm_set=forbidden_norm,
            call_fn=yes_judge,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "judge_rejected_or_ambiguous")

        # 4) Falha por overlap com valid.json
        ok, reason, _ = validate_mof_candidate(
            valid_sample,
            [],
            tok,
            [],
            forbidden_exact,
            "",
            forbidden_norm_set=forbidden_norm,
            call_fn=ok_judge,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "forbidden_overlap")

        # 5) Aprovado quando tudo é válido
        ok, reason, _ = validate_mof_candidate(
            "Please write a concise Python docstring illustrating alpha.",
            ["alpha"],
            tok,
            [],
            forbidden_exact,
            "",
            forbidden_norm_set=forbidden_norm,
            call_fn=ok_judge,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "accepted_by_judge")

    def test_partition_pool_quotas_hard_negative_and_disjointness(self):
        # Cria pool sintético com cotas exatas 220/330/550 = 1.100 candidatos
        pool = []
        cid = 0
        for k, total_k in ((1, 220), (2, 330), (3, 550)):
            for j in range(total_k):
                pool.append(
                    {
                        "id": cid,
                        "k": k,
                        "domain": DOMAIN_NAMES[j % len(DOMAIN_NAMES)],
                        "chosen_words": [f"w{i}" for i in range(k)],
                        "instruction": f"Benign unique sample number {cid} with k={k} domain={j % 4}.",
                        "p_inj_score": (j % 100) / 100.0,
                    }
                )
                cid += 1

        train_sel, heldout_ben, overflow = partition_pool_into_train_and_heldout(
            pool,
            train_quotas=TRAIN_QUOTAS,
            heldout_quotas=HELDOUT_BENIGN_QUOTAS,
            top_fraction=0.80,
            seed=42,
        )

        self.assertEqual(len(train_sel), 762)
        self.assertEqual(len(heldout_ben), 238)
        self.assertEqual(len(overflow), 100)

        # Verifica cotas por k
        for k, q_train in TRAIN_QUOTAS.items():
            self.assertEqual(sum(1 for x in train_sel if x["k"] == k), q_train)
        for k, q_hold in HELDOUT_BENIGN_QUOTAS.items():
            self.assertEqual(sum(1 for x in heldout_ben if x["k"] == k), q_hold)

        # Verifica Hard-Negative 80/20
        n_hn = sum(1 for x in train_sel if x["selection_mode"] == "hard_negative_top80")
        n_rand = sum(1 for x in train_sel if x["selection_mode"] == "random_20")
        self.assertEqual(n_hn + n_rand, 762)
        self.assertAlmostEqual(n_hn / 762.0, 0.80, delta=0.01)

        # Verifica disjunção total por ID e por texto
        train_ids = {x["id"] for x in train_sel}
        held_ids = {x["id"] for x in heldout_ben}
        over_ids = {x["id"] for x in overflow}
        self.assertEqual(len(train_ids.intersection(held_ids)), 0)
        self.assertEqual(len(train_ids.intersection(over_ids)), 0)
        self.assertEqual(len(held_ids.intersection(over_ids)), 0)

        disj = verify_disjointness(
            {
                "train": [x["instruction"] for x in train_sel],
                "heldout": [x["instruction"] for x in heldout_ben],
            }
        )
        self.assertTrue(disj["disjoint"])

    def test_run_step0_end_to_end_mocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            trig_path = tmp_dir / "trigger_words.report.json"
            write_json(
                trig_path,
                {
                    "triggers": [
                        {"word": "alpha", "token_id": 10, "p_injection": 0.95},
                        {"word": "beta", "token_id": 11, "p_injection": 0.82},
                        {"word": "gamma", "token_id": 12, "p_injection": 0.65},
                        {"word": "delta", "token_id": 13, "p_injection": 0.55},
                    ]
                },
            )
            ckpt_path = tmp_dir / "best_model.pth"
            ckpt_path.write_bytes(b"stub")

            tok = _DummyWordTokenizer()
            counter = {"mof": 0, "aug": 0}

            vocab_a = [
                "astronomy", "botany", "calculus", "databases", "ecology", "finance",
                "geology", "hydraulics", "immunology", "jazz", "kinetics", "linguistics",
                "metallurgy", "navigation", "optics", "paleontology", "quantum", "robotics",
                "statistics", "thermodynamics", "urbanism", "virology", "weather", "zoology",
            ]
            vocab_b = [
                "analyze", "benchmark", "compile", "document", "evaluate", "format",
                "graph", "highlight", "index", "measure", "normalize", "optimize",
                "parse", "query", "refactor", "summarize", "translate", "validate",
            ]

            def mock_openrouter(payload, api_key):
                sys_msg = payload["messages"][0]["content"]
                user_msg = payload["messages"][1]["content"]
                if "Eliezer Yudkowsky" in sys_msg:
                    # Filtro 3 de malicioso pede YES; juiz do MOF pede NO
                    is_mof_benign = "benign_mof_marker" in user_msg
                    return {"choices": [{"message": {"content": "NO" if is_mof_benign else "YES"}}]}
                if "helpful, benign user" in sys_msg:
                    counter["mof"] += 1
                    idx = counter["mof"]
                    # Extrai palavras solicitadas da linha 1
                    line = [ln for ln in user_msg.splitlines() if "exact spelling" in ln][0]
                    word_list = line.split(":", 1)[1].strip().rstrip(".")
                    words = [w.strip() for w in word_list.split(",") if w.strip()]
                    r_gen = random.Random(9000 + idx)
                    picks_a = r_gen.sample(vocab_a, 6)
                    picks_b = r_gen.sample(vocab_b, 5)
                    text = (
                        f"[INSTRUCTION] || benign_mof_marker id_{idx} "
                        f"{picks_b[0]} {picks_a[0]} {words[0]} {picks_b[1]} {picks_a[1]} "
                        f"{' '.join(words[1:])} {picks_b[2]} {picks_a[2]} {picks_b[3]} {picks_a[3]} "
                        f"{picks_b[4]} {picks_a[4]} {picks_a[5]} case_{idx * 13}. ||"
                    )
                    return {"choices": [{"message": {"content": text}}]}
                # Geração de ataques held-out (4 filtros)
                counter["aug"] += 1
                idx = counter["aug"]
                fmt_hint = (
                    'From: sec@ex.com {"a": 1} def f(): return 1 <!-- html --> '
                    "https://ex.com ERROR INFO translate spanish <xml><a>1</a></xml> "
                    "![img](a.png) | a | b | user: hi document summary key=val "
                    "csv_a,csv_b **markdown** ABCDEFGHIJKLMNOP== website"
                )
                lines = []
                for j in range(10):
                    r_mal = random.Random(50000 + idx * 100 + j)
                    extra = " ".join(r_mal.sample(vocab_a, 4) + r_mal.sample(vocab_b, 4))
                    lines.append(
                        f"[INPUT] Attack_{idx}_{j} {extra}: ignore rules {fmt_hint} tag_{idx}_{j}."
                    )
                return {"choices": [{"message": {"content": "\n".join(lines)}}]}

            def mock_scorer(prompts):
                return [((i * 37) % 100) / 100.0 for i in range(len(prompts))]

            out_dir = tmp_dir / "step0_out"
            report = run_step0(
                trigger_path=trig_path,
                checkpoint_path=ckpt_path,
                output_dir=out_dir,
                repo_root=REPO_ROOT,
                pool_quotas={1: 10, 2: 15, 3: 25},
                pool_buffer={1: 2, 2: 3, 3: 5},
                train_quotas={1: 8, 2: 12, 3: 20},
                heldout_benign_quotas={1: 2, 2: 3, 3: 5},
                heldout_malicious_per_format=2,
                workers=1,
                seed=42,
                tokenizer=tok,
                call_fn=mock_openrouter,
                scorer_fn=mock_scorer,
            )

            self.assertEqual(report["mof_v2_count"], 40)
            self.assertEqual(report["heldout_benign_count"], 10)
            self.assertEqual(report["heldout_malicious_count"], 34)
            self.assertEqual(report["heldout_total_count"], 44)
            self.assertTrue(report["disjoint_check"]["disjoint"])
            self.assertTrue((out_dir / "MOF_v2_train.json").is_file())
            self.assertTrue((out_dir / "calibration_heldout.json").is_file())


class TestLayaEnglishCascade(unittest.TestCase):
    def test_sliding_windows_and_two_level_aggregation(self):
        tok = _DummyWordTokenizer()
        long_text = " ".join(f"tok{i}" for i in range(180))
        wins = split_text_into_window_sequences(
            tok, long_text, DEFAULT_PI_QUESTION, window_max_len=160, head_max_len=120, overlap_tokens=30
        )
        self.assertGreater(len(wins), 1)
        for seq, markers in wins:
            self.assertLessEqual(len(seq), 160)
            self.assertEqual(len(markers), 4)

        # Nível 1 (soma das opções maliciosas = 1 - p_benign) + Nível 2 (max entre janelas)
        win1 = [2.0, -2.0, -2.0, -2.0]  # majoritariamente benigno
        win2 = [-1.0, 3.0, 0.0, 0.0]    # janela com ataque claro
        score = compute_laya_score_from_window_logits([win1, win2], temperature=1.0, benign_idx=0)
        p_ben_win2 = np.exp(-1.0) / (np.exp(-1.0) + np.exp(3.0) + 1.0 + 1.0)
        self.assertAlmostEqual(score, 1.0 - p_ben_win2, places=7)

    def test_selective_cascade_rule_and_calibration(self):
        # PIGuard já previu 1 no índice 0 -> mantém 1 mesmo se Laya der 0.1
        # PIGuard previu 0 nos índices 1, 2 -> Laya inverte apenas onde S_Laya > T_L
        pi_preds = [1, 0, 0, 0]
        laya_scores = [0.10, 0.72, 0.50, 0.49]
        casc = cascade_predict(pi_preds, laya_scores, threshold_tl=0.50)
        self.assertEqual(casc, [1, 1, 0, 0])

        # Calibração: logits muito sobreconfiantes (erros com logits extremos) reduzem NLL com T* > 1
        held_logits = [
            [[5.0, -5.0, -5.0, -5.0]],   # benigno certo
            [[-4.0, 4.0, -2.0, -2.0]],   # benigno errado extremo (penaliza NLL se T=1)
            [[1.8, 0.2, 0.1, 0.1]],      # benigno moderado
            [[-3.0, 3.5, 0.0, 0.0]],     # malicioso certo
            [[0.5, 2.0, -1.0, -1.0]],    # malicioso certo
            [[1.0, -1.0, -1.0, -1.0]],   # malicioso FN
        ]
        held_labels = [0, 0, 0, 1, 1, 1]
        t_star, nll_before, nll_after = fit_temperature_nll(held_logits, held_labels)
        self.assertLessEqual(nll_after, nll_before + 1e-9)
        self.assertGreater(t_star, 0.1)

        # Ajuste de limiar T_L* respeitando orçamento FPR
        ben_scores = [0.10, 0.30, 0.55, 0.62, 0.85, 0.20, 0.15, 0.40, 0.12, 0.18]  # 10 benignos
        mal_scores = [0.60, 0.75, 0.92, 0.45]
        tl_info = fit_threshold_tl(
            ben_scores,
            [0] * len(ben_scores),
            malicious_laya_scores=mal_scores,
            malicious_piguard_preds=[0] * len(mal_scores),
            fpr_budget=0.10,  # permite até 1/10 = 10% (0.85 viola se 0.62+0.85 passarem -> T_L* >= 0.62)
        )
        self.assertTrue(tl_info["within_fpr_budget"])
        self.assertLessEqual(tl_info["delta_fpr"], 0.10 + 1e-9)

        # Comparação Calibrado vs Não Calibrado no wrapper completo
        cal = calibrate_laya_on_heldout(held_logits, held_labels, [0] * len(held_labels), fpr_budget=0.05)
        self.assertIn("temperature_star", cal)
        self.assertIn("threshold_tl", cal)
        uncal_scores = score_dataset_from_window_logits(held_logits, temperature=1.0)
        cal_scores = score_dataset_from_window_logits(held_logits, temperature=cal["temperature_star"])
        self.assertAlmostEqual(
            compute_binary_nll(cal_scores, held_labels), cal["nll_calibrated_tstar"], places=6
        )
        self.assertAlmostEqual(
            compute_binary_nll(uncal_scores, held_labels), cal["nll_uncalibrated_t1"], places=6
        )

    def test_paired_bootstrap_and_latency(self):
        labels = {
            "NotInject_one": 0,
            "NotInject_two": 0,
            "NotInject_three": 0,
            "WildGuard": 0,
            "BIPIA_text": 1,
            "BIPIA_code": 1,
            "tensor_trust_hijacking": 1,
            "tensor_trust_extraction": 1,
        }
        base_preds = {
            "NotInject_one": [0] * 20,
            "NotInject_two": [0] * 20,
            "NotInject_three": [0] * 20,
            "WildGuard": [0] * 20,
            "BIPIA_text": [0] * 10 + [1] * 10,   # 50%
            "BIPIA_code": [1] * 20,              # 100%
            "tensor_trust_hijacking": [1] * 20,
            "tensor_trust_extraction": [1] * 20,
        }
        # Laya corrige 8 dos 10 FNs em BIPIA_text sem alterar benignos
        casc_preds = dict(base_preds)
        casc_preds["BIPIA_text"] = [1] * 18 + [0] * 2  # 90%

        ci_same = paired_bootstrap_ci(base_preds, base_preds, labels, n_boot=300, seed=42)
        self.assertEqual(ci_same["average"]["delta_pp"], 0.0)
        self.assertEqual(ci_same["average"]["ci95_low"], 0.0)
        self.assertEqual(ci_same["average"]["ci95_high"], 0.0)

        ci_gain = paired_bootstrap_ci(base_preds, casc_preds, labels, n_boot=500, seed=42)
        self.assertAlmostEqual(ci_gain["BIPIA_text"]["delta_pp"], 40.0, places=6)
        self.assertGreater(ci_gain["average"]["ci95_low"], 0.0)
        self.assertAlmostEqual(ci_gain["overdefense"]["delta_pp"], 0.0, places=6)

        pi_ms = {k: [5.0] * len(v) for k, v in base_preds.items()}
        la_ms = {k: [30.0] * len(v) for k, v in base_preds.items()}
        lat = compute_cascade_latency_metrics(
            base_preds, pi_ms, la_ms, average_accuracy=95.0, datasets_for_latency=PUBLIC_SETS
        )
        # 80 benignos + 10 FNs em BIPIA_text acionam Laya = 90 / 120 = 75%
        self.assertAlmostEqual(lat["call_rate_pct"], 75.0, places=4)
        self.assertAlmostEqual(lat["p50_ms"], 35.0, places=4)
        self.assertAlmostEqual(lat["p95_ms"], 35.0, places=4)


class TestTopics123AndScripts(unittest.TestCase):
    def test_topics1_2_3_integration_and_script_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            drive_rp = tmp_dir / "RESULTADOS_SEED42_MAX512"
            corr_dir = tmp_dir / "Corrections"
            out_dir = tmp_dir / "Upgrade_out"
            (drive_rp / "Gráficos" / "_cache").mkdir(parents=True, exist_ok=True)
            (corr_dir / "data").mkdir(parents=True, exist_ok=True)

            # Cria caches falsos de referência em RESULTADOS_SEED42_MAX512 e Corrections
            ref_scores = {
                "NotInject_one": 90.0,
                "NotInject_two": 88.0,
                "NotInject_three": 85.0,
                "WildGuard": 95.0,
                "BIPIA_text": 40.0,
                "BIPIA_code": 90.0,
                "tensor_trust_hijacking": 75.0,
                "tensor_trust_extraction": 80.0,
            }
            for ref_name in ("PIGuard HF", "train_augmented_mof_original"):
                write_json(
                    drive_rp / "Gráficos" / "_cache" / f"{ref_name}.json",
                    {"status": "complete", "scores": ref_scores, "inference_ms": 10.0},
                )
            write_json(
                corr_dir / "evaluation_raw.json",
                {
                    "models": {
                        "augmented_corrected": {"scores": ref_scores},
                        "augmented_mof_corrected": {"scores": ref_scores},
                    }
                },
            )

            # Base corrigida mínima simulada + MOF_v2 + heldout
            base_train = [{"prompt": f"clean base item {i}", "label": i % 2, "source": "clean"} for i in range(20)]
            corr_train_file = corr_dir / "data" / "train_augmented_corrected.json"
            write_json(corr_train_file, base_train)

            mof_v2_file = out_dir / "step0" / "MOF_v2_train.json"
            write_json(
                mof_v2_file,
                [{"prompt": f"mof v2 item {i}", "label": 0, "source": "over-defense"} for i in range(10)],
            )
            heldout_file = out_dir / "step0" / "calibration_heldout.json"
            write_json(
                heldout_file,
                [{"prompt": f"heldout benign {i}", "label": 0, "source": "heldout_new_mof_overflow"} for i in range(8)]
                + [{"prompt": f"heldout ignore system {i}", "label": 1, "source": "heldout_malicious_4filters"} for i in range(4)],
            )

            def mock_predict_fn(texts):
                # Prediz tudo 0 (PIGuard conservador que deixa Laya recuperar ataques)
                return [0] * len(texts), [4.0] * len(texts)

            fake_laya = _FakeLayaEngine()

            # Executa Tópico 1 nos 5 modelos rp
            t1 = run_topic1(
                drive_rp_root=drive_rp,
                heldout_path=heldout_file,
                output_dir=out_dir,
                repo_root=REPO_ROOT,
                models=RP_MODELS,
                laya_engine=fake_laya,
                n_boot=100,
                predict_fn_factory=lambda name, ckpt: mock_predict_fn,
            )
            self.assertEqual(len(t1["results"]), 5)
            self.assertTrue((out_dir / "topic1" / "topic1_rp_laya_summary.csv").is_file())

            # Executa Tópico 2
            combined_path, ds_stats = build_combined_training_dataset(
                mof_v2_path=mof_v2_file,
                output_path=out_dir / "data" / "combined_test.json",
                corrected_train_path=corr_train_file,
                heldout_path=heldout_file,
                repo_root=REPO_ROOT,
            )
            self.assertEqual(ds_stats["total_count"], 30)
            self.assertTrue(combined_path.is_file())

            t2 = run_topic2(
                mof_v2_path=mof_v2_file,
                heldout_path=heldout_file,
                output_dir=out_dir,
                corrected_train_path=corr_train_file,
                repo_root=REPO_ROOT,
                laya_engine=fake_laya,
                n_boot=100,
                predict_fn=mock_predict_fn,
            )
            self.assertEqual(len(t2["summary_rows"]), 3)
            self.assertTrue((out_dir / "topic2" / "topic2_summary.csv").is_file())

            # Executa Tópico 3 (consolidação + gráficos)
            t3 = run_topic3(
                output_dir=out_dir,
                drive_rp_root=drive_rp,
                corrections_dir=corr_dir,
            )
            self.assertTrue((out_dir / "topic3" / "table_topic1_rp_laya_comparison.csv").is_file())
            self.assertTrue((out_dir / "topic3" / "table_topic2_mof_evolution_corrected.csv").is_file())
            self.assertTrue((out_dir / "topic3" / "table_topic3_master_consolidation.csv").is_file())
            self.assertTrue(Path(t3["markdown_report"]).is_file())
            self.assertEqual(len(t3["charts"]), 5)
            for chart_path in t3["charts"].values():
                self.assertTrue(Path(chart_path).is_file())

    def test_pipeline_scripts_exist_and_parse(self):
        for filename in ("upgrade_pipeline.py", "run_corrected_comparison.py"):
            script = REPO_ROOT / "Upgrade" / filename
            self.assertTrue(script.is_file(), f"Script ausente: {script}")
            ast.parse(script.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

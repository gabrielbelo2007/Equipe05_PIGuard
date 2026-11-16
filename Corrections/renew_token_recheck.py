"""Token-wise recheck com sanitização lexical, mantendo o teste por ID isolado."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


# ==============================================================================
# Ancoragem dinâmica na raiz do repositório
# ==============================================================================
def find_repo_root(start: Path) -> Path:
    """Sobe a partir de ``start`` até achar o diretório com ``datasets`` e ``Original``."""
    for candidate in (start, *start.parents):
        if (candidate / "datasets").is_dir() and (candidate / "Original").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Raiz do repositório não encontrada a partir de {start}: "
        "esperado um diretório contendo 'datasets' e 'Original'."
    )


ROOT_DIR = find_repo_root(Path(__file__).resolve().parent)

for _local_path in (ROOT_DIR, ROOT_DIR / "Original"):
    _entry = str(_local_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from PIGuard import PIGuard  # noqa: E402  (depende do sys.path acima)


def resolve_path(value: str | os.PathLike[str] | None, default: Path) -> Path:
    """Resolve caminhos relativos contra ``ROOT_DIR`` (nunca contra o cwd)."""
    if value is None or str(value) == "":
        return Path(default).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def load_english_dictionary():
    """Filtro opcional já existente para descartar prefixos que constam no vocabulário."""
    import nltk
    from nltk.corpus import words, wordnet

    for resource in ("words", "wordnet"):
        nltk.download(resource, quiet=True, raise_on_error=True)
    dictionary = {word.lower() for word in words.words()}
    dictionary.update(
        lemma.lower()
        for synset in wordnet.all_synsets()
        for lemma in synset.lemma_names()
        if "_" not in lemma and "-" not in lemma
    )
    return dictionary


def is_valid_complete_word(token, min_len=3, dictionary=None):
    """Seleciona formas alfabéticas ASCII iniciadas em uma fronteira SentencePiece.

    O marcador ▁ indica espaço. A completude lexical só é checada quando o
    dicionário opcional está habilitado.
    """
    if not token.startswith("▁"):
        return False, ""
    word = token[1:]
    if len(word) < min_len or re.fullmatch(r"[A-Za-z]+", word) is None:
        return False, ""
    if dictionary is not None and word.lower() not in dictionary:
        return False, ""
    return True, word


def select_candidates(tokenizer, min_len=3, dictionary=None):
    candidates = []
    counts = Counter()
    special_ids = set(tokenizer.all_special_ids)
    for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        if token_id in special_ids:
            counts["special"] += 1
            continue
        valid, word = is_valid_complete_word(token, min_len, dictionary)
        if not valid:
            counts["lexical_filter"] += 1
            continue
        # Checar a identidade do ID também evita testar outro token por engano.
        if tokenizer.encode(word, add_special_tokens=False) != [token_id]:
            counts["round_trip_mismatch"] += 1
            continue
        candidates.append((word, token_id))
    counts["selected"] = len(candidates)
    return candidates, dict(counts)


def load_checkpoint(checkpoint, device):
    checkpoint = resolve_path(checkpoint, ROOT_DIR / "outputs/best_model.pth")
    if checkpoint.is_dir():
        checkpoint = checkpoint / "best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint solicitado ausente: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = PIGuard("microsoft/deberta-v3-base", num_labels=2, device=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return checkpoint, model


def recheck(model, tokenizer, candidates, device, batch_size=256):
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Tokenizer sem CLS/SEP")
    triggers = []
    details = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(candidates), batch_size), desc="Token-wise Recheck"):
            batch = candidates[start:start + batch_size]
            inputs = torch.tensor(
                [[tokenizer.cls_token_id, token_id, tokenizer.sep_token_id] for _, token_id in batch],
                dtype=torch.long,
                device=device,
            )
            outputs = model.deberta(input_ids=inputs, attention_mask=torch.ones_like(inputs))
            pooled = outputs["last_hidden_state"][:, 0, :].to(model.classifier.weight.dtype)
            logits = model.classifier(pooled)
            if logits.shape != (len(batch), 2) or not torch.isfinite(logits).all():
                raise ValueError("Saída inválida do classificador")
            probabilities = logits.softmax(dim=-1)[:, 1].cpu().tolist()
            predictions = logits.argmax(dim=-1).cpu().tolist()
            for (word, token_id), prediction, probability in zip(batch, predictions, probabilities):
                if prediction == 1:
                    triggers.append(word)
                    details.append({"word": word, "token_id": token_id, "p_injection": probability})
    return triggers, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint .pth ou diretório que contenha best_model.pth.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "Corrections/outputs/trigger_words.json"),
        help="Saída JSON (padrão: <repo>/Corrections/outputs/trigger_words.json).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_len", type=int, default=3)
    dictionary_flags = parser.add_mutually_exclusive_group()
    dictionary_flags.add_argument("--dictionary_filter", action="store_true", help="Habilita NLTK/WordNet.")
    dictionary_flags.add_argument("--no_dict_filter", action="store_true", help="Compatibilidade: usa apenas sanitização lexical.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.min_len < 1:
        parser.error("batch_size e min_len devem ser positivos")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint, model = load_checkpoint(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    dictionary = load_english_dictionary() if args.dictionary_filter else None
    candidates, counts = select_candidates(tokenizer, args.min_len, dictionary)
    if not candidates:
        raise ValueError("Nenhum candidato passou pela sanitização; confira tokenizer e filtros.")
    triggers, details = recheck(model, tokenizer, candidates, device, args.batch_size)

    output = resolve_path(args.output, ROOT_DIR / "Corrections/outputs/trigger_words.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    checkpoint_sha256 = hasher.hexdigest()
    report = {
        "status": "complete" if triggers else "no_triggers",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer": tokenizer.name_or_path,
        "classes": {"benign": 0, "injection": 1},
        "device": str(device),
        "min_len": args.min_len,
        "dictionary_filter": args.dictionary_filter,
        "counts": counts,
        "trigger_count": len(triggers),
        "triggers": details,
    }
    report_path = output.with_suffix(".report.json")
    for path, data in ((output, triggers), (report_path, report)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    print(f"{len(triggers)} trigger words entre {len(candidates)} tokens sanitizados. Saída: {output}")


if __name__ == "__main__":
    main()

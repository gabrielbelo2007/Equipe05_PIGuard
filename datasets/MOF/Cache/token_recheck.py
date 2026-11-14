import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

class FallbackPIGuard(nn.Module):
    def __init__(self, base_model_name: str = "microsoft/deberta-v3-base", num_labels: int = 2):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)

    def forward_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]
        return self.classifier(outputs[:, 0, :])

def load_guard_model(model_path: str, base_arch: str, device: torch.device):
    if os.path.isdir(model_path):
        best_candidate = os.path.join(model_path, "best_model.pth")
        if os.path.exists(best_candidate):
            model_path = best_candidate

    if os.path.isfile(model_path) and model_path.endswith((".pth", ".pt")):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        try:
            from PIGuard import PIGuard
            model = PIGuard(base_arch, num_labels=2, device=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device=device, dtype=torch.float32)
            model.eval()
            return model, "piguard_custom"
        except ImportError:
            model = FallbackPIGuard(base_arch, num_labels=2).to(device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device=device, dtype=torch.float32)
            model.eval()
            return model, "fallback_piguard"

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, "hf"


def main():
    parser = argparse.ArgumentParser(
        description="Reprodução fiel do Token-wise Recheck (PIGuard - Seção 4.2)"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho do checkpoint .pth ou modelo HF.")
    parser.add_argument("--base_arch", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--output", type=str, default="trigger_words_reproduction.json")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_arch)
    model, mode = load_guard_model(args.checkpoint, args.base_arch, device)

    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    candidate_entries = []
    for token_str, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        if token_id in special_ids:
            continue
        clean_str = token_str.lstrip("▁")
        if not clean_str:
            continue
        candidate_entries.append((clean_str, token_id))

    biased_tokens = []

    with torch.no_grad():
        for i in tqdm(range(0, len(candidate_entries), args.batch_size), desc="Token-wise Recheck"):
            batch = candidate_entries[i : i + args.batch_size]
            batch_strs = [item[0] for item in batch]
            batch_ids = [item[1] for item in batch]

            input_ids = torch.tensor(
                [[cls_id, tid, sep_id] for tid in batch_ids], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

            if mode == "piguard_custom":
                outputs = model.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["last_hidden_state"]
                pooled = outputs[:, 0, :].to(model.classifier.weight.dtype)
                logits = model.classifier(pooled)
            elif mode == "fallback_piguard":
                logits = model.forward_logits(input_ids, attention_mask)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            for token_text, pred in zip(batch_strs, preds):
                if pred == 1:
                    biased_tokens.append(token_text)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(biased_tokens, f, indent=4, ensure_ascii=False)
    print(f"Total de trigger tokens extraídos: {len(biased_tokens)} | Salvo em: {args.output}")


if __name__ == "__main__":
    main()
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from PIGuard import PIGuard
from datasets import load_dataset
from util import set_seed, get_logger, compute_accuracy
from params import build_parser, repo_path
from eval import evaluate


def parse_shortcut_args():
    parser = build_parser()
    parser.description = "PIGuard - Shortcut Mitigation via Group Loss Reweighting"
    parser.add_argument(
        "--trigger_words", type=repo_path, required=True,
        help="JSON file with trigger words extracted from the augmented checkpoint.",
    )
    parser.add_argument(
        "--weight_mode", choices=["cond_prob", "joint_prob"], default="cond_prob",
        help="Group weighting: inverse P(y|z) or inverse P(z,y), normalized to mean 1.",
    )
    return parser.parse_args()


def build_trigger_token_ids(tokenizer, trigger_words_path: str) -> set[int]:
    """
    Mapeia a lista de trigger words para os token IDs correspondentes no vocabulário
    do DeBERTa-v3, replicando fielmente a identificação de tokens enviesados.
    """
    with open(trigger_words_path, "r", encoding="utf-8") as f:
        raw_words = json.load(f)

    trigger_set = {w.strip() for w in raw_words if isinstance(w, str) and w.strip()}
    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)

    trigger_token_ids = set()
    for tok_str, tid in vocab.items():
        if tid in special_ids:
            continue
        clean_tok = tok_str.lstrip("▁")
        if clean_tok in trigger_set:
            trigger_token_ids.add(tid)

    return trigger_token_ids


def compute_group_weights(z_list: list[int], y_list: list[int], mode: str = "cond_prob"):
    """
    Calcula os pesos das instâncias baseando-se nos 4 grupos (z in {0, 1}, y in {0, 1}).
    Modo 'cond_prob' (Recomendado): w_raw = 1 / P(y | z), normalizado para média 1.0.
    Modo 'joint_prob': w_raw = 1 / P(z, y), normalizado para média 1.0.
    """
    total_n = len(y_list)
    counts = {(z, y): 0 for z in (0, 1) for y in (0, 1)}
    for z_val, y_val in zip(z_list, y_list):
        counts[(int(z_val), int(y_val))] += 1

    z_totals = {z: counts[(z, 0)] + counts[(z, 1)] for z in (0, 1)}
    raw_weights = {}
    cond_probs = {}
    joint_probs = {}

    for z in (0, 1):
        for y in (0, 1):
            n_zy = counts[(z, y)]
            n_z = z_totals[z]
            joint_probs[(z, y)] = n_zy / total_n if total_n > 0 else 0.0
            p_y_given_z = (n_zy / n_z) if n_z > 0 else 0.0
            cond_probs[(z, y)] = p_y_given_z

            if mode == "cond_prob":
                raw_weights[(z, y)] = (1.0 / p_y_given_z) if p_y_given_z > 0 else 0.0
            else:
                p_zy = joint_probs[(z, y)]
                raw_weights[(z, y)] = (1.0 / p_zy) if p_zy > 0 else 0.0

    mean_w = sum(raw_weights[(int(z), int(y))] for z, y in zip(z_list, y_list)) / total_n
    norm_weights = {k: (v / mean_w if mean_w > 0 else 1.0) for k, v in raw_weights.items()}
    sample_weights = [norm_weights[(int(z), int(y))] for z, y in zip(z_list, y_list)]

    summary = {
        "total_samples": total_n,
        "weight_mode": mode,
        "group_counts": {f"z={z},y={y}": counts[(z, y)] for z in (0, 1) for y in (0, 1)},
        "group_frequencies": {f"z={z},y={y}": counts[(z, y)] / total_n for z in (0, 1) for y in (0, 1)},
        "z_marginals": {f"z={z}": z_totals[z] for z in (0, 1)},
        "cond_probs_y_given_z": {f"P(y={y}|z={z})": cond_probs[(z, y)] for z in (0, 1) for y in (0, 1)},
        "raw_weights": {f"z={z},y={y}": raw_weights[(z, y)] for z in (0, 1) for y in (0, 1)},
        "normalized_weights": {f"z={z},y={y}": norm_weights[(z, y)] for z in (0, 1) for y in (0, 1)},
        "mean_weight": sum(sample_weights) / len(sample_weights),
    }

    return sample_weights, summary


def train_shortcut():
    global logger
    args = parse_shortcut_args()

    set_seed(args)
    logger = get_logger(os.path.join(args.logs, "log_{}.txt".format(args.name)))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # init train setting
    epochs = args.epochs
    save_path = args.checkpoint_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    # tokenizer initial
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    trigger_token_ids = build_trigger_token_ids(tokenizer, args.trigger_words)

    def preprocess_function(examples):
        encoding_text = tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=args.max_length)

        return {
            'input_ids': encoding_text['input_ids'],
            'z_flag': [int(any(t in trigger_token_ids for t in ids)) for ids in encoding_text['input_ids']],
            'attention_mask': encoding_text['attention_mask'],
        }

    # Prepare dataset
    data_files = {
        "train": args.train_set,
        "valid": args.valid_set,
    }

    dataset = load_dataset('json', data_files=data_files)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.map(lambda examples: {'labels': [label for label in examples['label']]}, batched=True)

    # Only training samples receive weights; validation keeps the reference loss.
    sample_weights, weight_info = compute_group_weights(
        encoded_dataset['train']['z_flag'], encoded_dataset['train']['labels'],
        mode=args.weight_mode,
    )
    encoded_train = encoded_dataset['train'].add_column('weights', sample_weights)
    encoded_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'weights'])
    with open(os.path.join(save_path, 'group_weights_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(weight_info, f, indent=4, ensure_ascii=False)

    train_loader = DataLoader(encoded_train, batch_size=args.batch_size, shuffle=True)
    encoded_dataset['valid'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_loader = DataLoader(encoded_dataset['valid'], batch_size=args.eval_batch_size, shuffle=False)

    model = PIGuard('microsoft/deberta-v3-base', num_labels=2, device=device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=epochs * len(train_loader)
    )

    per_sample_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    best_accuracy = float('-inf')
    best_model_path = None
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, _ = model(input_ids, attention_mask, labels, mode="train")
            weights = batch['weights'].to(device).float()
            per_sample_loss = per_sample_loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = (weights * per_sample_loss).mean()

            loss.backward()

            optimizer.step()
            scheduler.step()

            if step % args.display == 0:
                logger.info(f"Época {epoch + 1}/{epochs} | Step {step + 1}/{len(train_loader)} | Loss: {loss:.3f}")

            if ((step % args.save_step == 0) and (step != 0)) or (step == (len(train_loader) - 1)):
                model.eval()
                loss_list, logits_list, labels_list = [], [], []

                with torch.no_grad():
                    for eval_step, batch in enumerate(validation_loader):
                        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                        optimizer.zero_grad()
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)

                        logits, loss = model(input_ids, attention_mask, labels)

                        loss_list.append(loss.cpu().item())
                        logits_list.append(logits.cpu())
                        labels_list.append(labels.cpu())

                        if eval_step % args.display == 0:
                            logger.info(f"Step: {eval_step} / {len(validation_loader)}.")
                            logger.info(f"Loss: {loss:.3f}")

                    combined_logits = torch.cat(logits_list, dim=0)
                    combined_labels = torch.cat(labels_list, dim=0)

                    pred = combined_logits.argmax(1)

                    benign_accuracy, injection_accuracy, total_accuracy = compute_accuracy(pred, combined_labels)

                    print(f"total accuracy: {total_accuracy}")
                    print(f"benign accuracy: {benign_accuracy}")
                    print(f"injection accuracy: {injection_accuracy}")
                    print(f"loss: {np.mean(loss_list)}")

                    # eval on valid set
                    if total_accuracy > args.save_thres:
                        model_path = f'{save_path}/epoch_{epoch}_{step}_model.pth'
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved to {model_path}.")

                    # Sempre guardar o melhor candidato, inclusive quando nenhum
                    # checkpoint ultrapassa o limiar. Empates preservam o primeiro.
                    if total_accuracy > best_accuracy:
                        best_accuracy = total_accuracy
                        best_model_path = f'{save_path}/best_model.pth'
                        torch.save(model.state_dict(), best_model_path)
                        logger.info(f"Melhor validação até agora: {best_accuracy:.4f} (época {epoch + 1}, step {step + 1}).")

                model.train()

    # evaluate on overall test set
    logger.info("Evaluate Best Model on Test Sets.")
    if best_model_path is None:
        raise RuntimeError("Nenhuma validação foi executada; confira os dados e a configuração do treinamento.")
    if best_accuracy <= args.save_thres:
        logger.warning(f"Nenhuma validação superou {args.save_thres:.2%}; usando o melhor checkpoint observado ({best_accuracy:.2%}).")
    else:
        logger.info(f"Melhor acurácia de validação: {best_accuracy:.2%}.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    logger.info(f"Loaded model from {best_model_path}.")
    model.eval()
    evaluate(model, args.dataset_root)

if __name__ == "__main__":
    train_shortcut()

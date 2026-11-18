import os
import torch
import yaml
import json
import random
import numpy as np
import logging
from pathlib import Path

def compute_accuracy(predictions, labels):
    correct = predictions == labels

    label_0_indices = labels == 0
    label_1_indices = labels == 1

    correct_0 = correct[label_0_indices].sum().item()
    correct_1 = correct[label_1_indices].sum().item()

    total_0 = label_0_indices.sum().item()
    total_1 = label_1_indices.sum().item()

    total_samples = labels.size(0)
    total_correct = correct.sum().item()

    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

    total_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return accuracy_0, accuracy_1, total_accuracy

def set_seed(args):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args.seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


def yaml_to_json(yaml_file, json_file):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        yaml_content = yaml.safe_load(file)  # load YAML
    
    # write to json file
    with open(json_file, 'w', encoding='utf-8') as json_output:
        json.dump(yaml_content, json_output, indent=4)


if __name__ == "__main__":
    import argparse

    from params import repo_path

    parser = argparse.ArgumentParser(description="Converte o dataset opcional PINT de YAML para JSON.")
    parser.add_argument("--input", type=repo_path, default="datasets/pint-v1.0.1-679dd3c2.yaml")
    parser.add_argument("--output", type=repo_path, default="datasets/PINT.json")
    args = parser.parse_args()
    if not Path(args.input).is_file():
        parser.error(f"Arquivo PINT não encontrado: {args.input}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    yaml_to_json(args.input, args.output)

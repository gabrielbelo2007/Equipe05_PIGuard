import json

input_path = "./datasets/train.json"
output_path = "./datasets/train_cleaned.json"

baseline_samples = []
removed_longtail = 0
removed_mof = 0

with open(input_path, "r", encoding="utf-8") as f:
    try:
        content = json.load(f)
        items = content if isinstance(content, list) else [content]
    except json.JSONDecodeError:
        f.seek(0)
        items = [json.loads(line) for line in f if line.strip()]

for item in items:
    source = item.get("source") or item.get("dataset") or item.get("type")
    
    # Remove apenas a Cauda Longa
    if source == "LLM Augmented set":
        removed_longtail += 1
    elif source == "over-defense":
        removed_mof += 1
    else:
        baseline_samples.append(item)

print(f"Amostras de Cauda Longa removidas: {removed_longtail}")
print(f"Amostras de MOF removidas: {removed_mof}")
print(f"Total de amostras mantidas no Dataset Baseline: {len(baseline_samples)}")

# Salva a base original limpa
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(baseline_samples, f, indent=4, ensure_ascii=False)
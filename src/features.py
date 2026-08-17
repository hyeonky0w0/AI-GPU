import random


def build_dataset(config):
    """Replace this deterministic sample with the real baseball data loader."""
    random.seed(config.get("seed", 42))
    rows = config["data"].get("sample_rows", 200)
    features = [[random.random(), random.random()] for _ in range(rows)]
    targets = [2.5 * row[0] - 1.2 * row[1] + random.gauss(0, 0.05) for row in features]
    split = int(rows * config["data"].get("train_ratio", 0.8))
    return features[:split], targets[:split], features[split:], targets[split:]

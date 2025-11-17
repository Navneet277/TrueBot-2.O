"""Quick accuracy evaluation utility."""

import json

from config import DATA_PATH, MODEL_DIR
from modules.train_model import main as train_main


def evaluate() -> None:
    metrics = train_main(DATA_PATH, MODEL_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate()



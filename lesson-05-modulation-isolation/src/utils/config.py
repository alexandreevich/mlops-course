import json
from typing import Dict


def load_config(path: str = "config.json") -> Dict:
    with open(path, "r") as f:
        return json.load(f)

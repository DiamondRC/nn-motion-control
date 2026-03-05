import json
import os

TESTING = "src/deltabot_nn_controller/model_zoo/basic_mlp.json"


def load_config(file_path=TESTING):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file {file_path} not found")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)  # Returns dict directly

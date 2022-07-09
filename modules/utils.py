import json


def load_json_dict(json_path):
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    return json_dict
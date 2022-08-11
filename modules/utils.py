import json


def load_json(json_path):
    with open(json_path, "r") as f:
        file = json.load(f)
    return file


def save_json(file, json_path):
    with open(json_path, "w") as f:
        json.dump(file, f,  indent=4)


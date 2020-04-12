import json

def get_params(filepath: str) -> dict:
    with open(filepath) as f:
        params = json.load(f)

    return params

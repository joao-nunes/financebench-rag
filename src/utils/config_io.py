import json
from pathlib import Path


def save_config(config_module, output_path: str | Path) -> None:
    """
    Save all uppercase configuration variables from a config module to JSON.
    """

    config = {}

    for key, value in vars(config_module).items():
        if not key.isupper():
            continue

        # Convert Path objects to strings
        if isinstance(value, Path):
            value = str(value)

        config[key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
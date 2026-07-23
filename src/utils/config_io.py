import json
from pathlib import Path


import json
from pathlib import Path


def save_experiment_config(
    config_module,
    output_path: str | Path,
    **experiment_kwargs,
) -> None:
    """
    Save the project configuration together with
    experiment-specific parameters.

    Parameters
    ----------
    config_module
        Python module containing the global configuration.
    output_path
        Destination JSON file.
    experiment_kwargs
        Extra experiment parameters.
    """

    config = {}

    # Save all uppercase variables from config.py
    for key, value in vars(config_module).items():

        if not key.isupper():
            continue

        if isinstance(value, Path):
            value = str(value)

        config[key] = value

    # Add experiment-specific settings
    config.update(experiment_kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
"""Preparation CLI, deliberately without a real-data training command."""
import argparse
import json
from pathlib import Path
import tomllib
from . import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "encode", "baseline", "pairs", "verify"])
    parser.add_argument("--config", type=Path, default=Path("configs/setup.toml"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    root = config_path.parent.parent
    result = getattr(prepare, args.command)(config, root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

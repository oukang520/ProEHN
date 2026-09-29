"""Train the ProEHN kinetic gatekeeper from an analysis-ready feature table."""

from __future__ import annotations

import argparse
import json

from proehn.config import load_config
from proehn.training import train_kinetic_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ProEHN kinetic gatekeeper.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--data", required=True, help="Input analysis-ready cohort CSV.")
    parser.add_argument(
        "--params-out",
        default=None,
        help="Output .msgpack parameters. Defaults to config artifacts.kinetic_params.",
    )
    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Output metadata pickle. Defaults to config artifacts.kinetic_metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    params_output = args.params_out or config["artifacts"]["kinetic_params"]
    metadata_output = args.metadata_out or config["artifacts"]["kinetic_metadata"]
    summary = train_kinetic_from_csv(
        args.data,
        params_output,
        metadata_output,
        config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

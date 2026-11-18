"""Train the ProEHN kinetic gatekeeper from an analysis-ready feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proehn.config import load_config
from proehn.training import train_kinetic_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ProEHN kinetic gatekeeper.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--training", required=True, help="Raw training-patient CSV.")
    parser.add_argument(
        "--validation", required=True, help="Disjoint raw validation-patient CSV."
    )
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
    for output in (params_output, metadata_output):
        path = Path(output).resolve()
        if path.is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError("Choose output artifacts outside the source repository")
        if path.exists():
            raise FileExistsError(
                "A fitted artifact already exists; choose new output paths"
            )
    summary = train_kinetic_from_csv(
        args.training,
        params_output,
        metadata_output,
        config,
        validation_csv_path=args.validation,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

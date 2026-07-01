"""Train the ProEHN topology engine from a cohort feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proehn.config import load_config
from proehn.training import train_topology_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ProEHN feature-modulated topology engine.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--data", required=True, help="Input cohort CSV.")
    parser.add_argument("--out", default=None, help="Output .npz artifact. Defaults to config artifacts.topology_model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = args.out or config["artifacts"]["topology_model"]
    summary = train_topology_from_csv(args.data, output, config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Run ProEHN inference for one patient represented as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proehn.config import load_config
from proehn.engine import ProEHNEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ProEHN conditional event scores for one patient.")
    parser.add_argument("--config", required=True, help="YAML configuration file.")
    parser.add_argument("--patient", required=True, help="Patient JSON file.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top conditional event scores to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    with Path(args.patient).open("r", encoding="utf-8") as handle:
        patient = json.load(handle)

    engine = ProEHNEngine(
        kinetic_params_path=config["artifacts"]["kinetic_params"],
        kinetic_metadata_path=config["artifacts"]["kinetic_metadata"],
        topology_model_path=config["artifacts"]["topology_model"],
        stop_threshold=config["kinetic"]["stop_threshold"],
        fallback_go_probability=config["kinetic"]["fallback_go_probability"],
        regularization_strength=config["topology"]["regularization_strength"],
        log_rate_clip_min=config["topology"]["log_rate_clip_min"],
        log_rate_clip_max=config["topology"]["log_rate_clip_max"],
    )
    result = engine.predict(patient)
    result["integrated_event_scores"] = result["integrated_event_scores"][: args.top_k]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

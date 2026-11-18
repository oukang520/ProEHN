"""Run the GENIE 0..50% training-label-noise experiment in one-percent steps."""

import argparse
from pathlib import Path
import pandas as pd
from proehn.config import load_config
from proehn.label_noise import run_noise, SEEDS
from proehn.training import prepare_kinetic_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=SEEDS,
        help="Optional worker seed; default runs all five.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("cohort") != "GENIE":
        parser.error("The published dense label-noise experiment uses GENIE")
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.is_relative_to(repo):
        parser.error("Choose an output directory outside the source repository")
    train, validation, cfg, schema = prepare_kinetic_frames(
        pd.read_csv(args.training), pd.read_csv(args.validation), config
    )
    test, _, _, _ = prepare_kinetic_frames(pd.read_csv(args.test), validation, config)
    run_noise(
        train,
        validation,
        test,
        model_config=cfg,
        label_schema=schema,
        output=output,
        seeds=(args.seed,) if args.seed else SEEDS,
    )


if __name__ == "__main__":
    main()

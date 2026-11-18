"""Export Fig.5 numeric path, state, branch and landscape results without drawing."""

import argparse
import json
from pathlib import Path
import pandas as pd
from proehn.research.paths import export_path_data
from proehn.research.protocols import GENES, require_external_output, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=GENES)
    parser.add_argument(
        "--model", type=Path, required=True, help="Native four-training-fold model JSON"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Prepared native cohort CSV with fold"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    if artifact.get("selected_candidate", "legacy") != "legacy":
        raise ValueError(
            "Use the frozen native model, rather than a selected snapshot fit"
        )
    frame = pd.read_csv(args.data)
    training = frame[frame.fold.ne(args.test_fold)]
    tables, mass = export_path_data(artifact, training, GENES[args.cohort])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / (name + ".csv"), index=False, float_format="%.17g")
    save_json(
        args.output / "path_config.json",
        dict(
            cohort=args.cohort,
            training_folds=[f for f in range(5) if f != args.test_fold],
            n_paths=50000,
            seed=20261111,
            raw_profile_quantiles=[0.25, 0.5, 0.75],
            rank_quantiles=[0.1, 0.9],
            maximum_branch_depth=3,
            retained_mutations_per_node=2,
            seeding_always_retained=True,
            branch_probabilities_renormalized=False,
            **mass,
        ),
    )


if __name__ == "__main__":
    main()

"""Export numeric fixed-model context analyses for all three cohorts, without drawing."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proehn.research.context import context_results
from proehn.research.protocols import GENES, require_external_output, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in ["genie", "mela", "paca"]:
        parser.add_argument("--" + cohort + "-model", type=Path, required=True)
        parser.add_argument("--" + cohort + "-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    # Preserve the executed RNG stream: one shared path-uniform block preceded
    # the three sequential context-bootstrap analyses in the original source.
    rng = np.random.default_rng(20261111)
    rng.random((50000, 13))
    summaries = {}
    for cohort in ["GENIE", "MELA", "PACA"]:
        artifact = json.loads(
            getattr(args, cohort.lower() + "_model").read_text(encoding="utf-8")
        )
        if artifact.get("selected_candidate", "legacy") != "legacy":
            raise ValueError("Context descriptions require native complete-record fits")
        frame = pd.read_csv(getattr(args, cohort.lower() + "_data"))
        tables, metadata = context_results(artifact, frame, GENES[cohort], rng=rng)
        folder = args.output / cohort
        folder.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(folder / (name + ".csv"), index=False, float_format="%.17g")
        summaries[cohort] = metadata
    save_json(
        args.output / "completion.json",
        dict(
            status="COMPLETE",
            seed=20261111,
            cohorts=summaries,
            cluster_bootstrap_replicates=20,
            model_retraining=False,
        ),
    )


if __name__ == "__main__":
    main()

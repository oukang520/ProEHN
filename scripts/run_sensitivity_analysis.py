"""Evaluate the frozen generating-point sparse-observation sensitivity design."""

import argparse
from pathlib import Path
import json
import jax
import pandas as pd
from proehn.research.identifiability import rank_witness
from proehn.research.protocols import require_external_output, load_protocol, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    config = load_protocol(args.config)
    results = []
    for scenario in config["scenarios"]:
        path = args.output / (scenario + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = rank_witness(scenario, config["identifiability"])
            save_json(path, result)
        results.append(result)
    spectra = []
    for result in results:
        for i, (raw, scaled) in enumerate(
            zip(
                result["raw_singular_values"],
                result["column_normalized_singular_values"],
            )
        ):
            spectra.append(
                dict(
                    scenario=result["scenario"],
                    index=i + 1,
                    raw_singular=raw,
                    column_normalized_singular=scaled,
                )
            )
    pd.DataFrame(spectra).to_csv(args.output / "sensitivity_spectrum.csv", index=False)
    save_json(
        args.output / "completion.json",
        dict(
            status="COMPLETE",
            mechanisms=len(results),
            parameters=390,
            model_training=False,
            config=config["identifiability"],
        ),
    )


if __name__ == "__main__":
    main()

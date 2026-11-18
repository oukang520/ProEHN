"""Apply a saved calibrated gate to a prepared patient table."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from proehn.kinetic import ProEHNKineticGatekeeper
from proehn.label_noise import gate_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", required=True, help="Output CSV outside the source checkout"
    )
    parser.add_argument("--partition", choices=["training", "validation", "test"])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Choose an output CSV outside the source repository")
    if output.exists():
        parser.error("Output already exists; choose a new CSV path")
    gate = ProEHNKineticGatekeeper(args.params, args.metadata)
    if not gate.ready:
        parser.error("Both saved gate files are required")
    frame = pd.read_csv(args.input)
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        parser.error("One row per patient is required")
    rename = {
        f"{p}.{g}": f"{p}.{g} (M)"
        for p in ["P", "M"]
        for g in gate.model_config.get("gene_names", [])
    }
    frame = frame.rename(columns={a: b for a, b in rename.items() if b not in frame})
    go, raw = gate_probabilities(gate, frame)
    result = pd.DataFrame(dict(patient_id=frame.patient_id, p_Go=go, raw_p_Go=raw))
    for column in ["fold", "Patient_Label"]:
        if column in frame:
            result[column] = frame[column].to_numpy()
    if args.partition:
        result["partition"] = args.partition
    if not np.isfinite(go).all():
        raise FloatingPointError("Nonfinite calibrated predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


if __name__ == "__main__":
    main()

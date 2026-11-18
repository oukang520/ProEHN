"""Evaluate calibrated clinical predictions without refitting the evolutionary model."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from proehn.clinical_analysis import (
    fixed_capacity_triage,
    gate_performance,
    genie_cox,
    icgc_cox,
    source_components,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=[
            "gate",
            "triage",
            "genie-cox",
            "icgc-cox",
            "components",
            "decision-curves",
        ],
        required=True,
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Patient CSV: patient_id,p_Go,Patient_Label for gate/triage; duration (positive source OS days),"
            "event (1=death),age,burden (log1p of mean background count),vaf (mean background VAF) for Cox;"
            "metastatic (specimen available=1) additionally for GENIE Cox; fold (0..4) for ICGC/components;"
            "q_Seeding for components. GENIE uses sequencing-landmark OS, ICGC diagnosis-origin OS."
            "Decision curves use long patient_id,model,label,p_Go rows, label 0=Go,1=Stop,-1=unknown."
        ),
    )
    parser.add_argument(
        "--validation",
        help="Separate fold-1 validation CSV; test --input is fold 0 for GENIE Cox.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New output directory outside the source checkout.",
    )
    args = parser.parse_args()
    if args.task in ["triage", "genie-cox"] and not args.validation:
        parser.error("--validation is required for this task")
    frame = pd.read_csv(args.input)
    out = Path(args.output).resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Choose an output directory outside the source repository")
    summary = {}
    if args.task in ["triage", "genie-cox"]:
        validation = pd.read_csv(args.validation)
    out.mkdir(parents=True, exist_ok=False)
    if args.task == "gate":
        if "Patient_Label" not in frame and "label" in frame:
            frame = frame.rename(columns={"label": "Patient_Label"})
        if "model" in frame:
            pd.concat(
                [
                    gate_performance(part).assign(model=name)
                    for name, part in frame.groupby("model", sort=False)
                ],
                ignore_index=True,
            ).to_csv(out / "metrics.csv", index=False)
        else:
            gate_performance(frame).to_csv(out / "metrics.csv", index=False)
    elif args.task == "decision-curves":
        from proehn.decision_analysis import decision_curves, reliability_bins

        for name, table in decision_curves(frame).items():
            table.to_csv(out / (name + ".csv"), index=False)
        reliability_bins(frame).to_csv(out / "reliability.csv", index=False)
    elif args.task == "triage":
        fixed_capacity_triage(validation, frame).to_csv(out / "triage.csv", index=False)
    elif args.task == "genie-cox":
        summary, coefficients, predictions, groups = genie_cox(validation, frame)
        coefficients.to_csv(out / "coefficients.csv", index=False)
        groups.to_csv(out / "group_hazard_ratio.csv", index=False)
        predictions.to_csv(out / "predictions.csv", index=False)
    elif args.task == "icgc-cox":
        coefficients, summary = icgc_cox(frame)
        coefficients.to_csv(out / "coefficients.csv", index=False)
    else:
        table, summary = source_components(frame)
        table.to_csv(out / "components.csv", index=False)
    if summary:
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

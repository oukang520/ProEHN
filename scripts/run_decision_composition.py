"""Evaluate the target-masked shared-gate three-class clinical/genotype endpoint."""

import argparse
import hashlib
from pathlib import Path
import pandas as pd
from proehn.config import load_config
from proehn.research.composition import fit_target_queries, summarize_composition
from proehn.research.protocols import require_external_output, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--clinical",
        required=True,
        type=Path,
        help="Prepared clinical CSV with raw 180-day source fields and fold.",
    )
    parser.add_argument(
        "--genotypes",
        required=True,
        type=Path,
        help="Native long CSV: patient_id,compartment,gene,model,label,prediction. All models share targets.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gate-directory",
        type=Path,
        help="Reuse completed target-masked gate artifacts; new gates are saved here.",
    )
    parser.add_argument("--primary-model", default="ProEHN context")
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument(
        "--no-forced-go",
        action="store_true",
        help="Omit the separately identified Go=1 ablations.",
    )
    args = parser.parse_args()
    out = require_external_output(args.output)
    gates = (
        require_external_output(args.gate_directory) if args.gate_directory else None
    )
    config = load_config(args.config)
    queries, eligibility = fit_target_queries(
        pd.read_csv(args.clinical),
        pd.read_csv(args.genotypes),
        config,
        out,
        folds=args.fold,
        gate_directory=gates,
        primary_model=args.primary_model,
        include_forced_go=not args.no_forced_go,
    )
    tables, summary = summarize_composition(queries, primary_model=args.primary_model)
    out.mkdir(parents=True, exist_ok=True)
    queries.to_csv(out / "query_scores_private.csv", index=False, float_format="%.17g")
    eligibility.to_csv(out / "target_eligibility.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.17g")
    summary["source_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [
            ("config", args.config),
            ("clinical", args.clinical),
            ("genotypes", args.genotypes),
        ]
    }
    save_json(out / "summary.json", summary)
    print(
        "COMPLETE",
        summary["n_patients"],
        "patients",
        summary["target_patient_queries"],
        "queries",
    )


if __name__ == "__main__":
    main()

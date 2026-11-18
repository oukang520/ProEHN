"""Reconstruct GENIE or ICGC records from locally supplied official source files."""

import argparse
from pathlib import Path
from proehn.research.source_data import prepare_genie, prepare_icgc
from proehn.research.protocols import require_external_output, save_json, GENES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cohort", required=True)
    genie = sub.add_parser("GENIE")
    for name in ["samples", "variants", "panels", "patient-universe"]:
        genie.add_argument("--" + name, type=Path, required=True)
    for cohort in ["MELA", "PACA"]:
        icgc = sub.add_parser(cohort)
        for name in ["source", "bindings", "hgnc"]:
            icgc.add_argument("--" + name, type=Path, required=True)
    for child in sub.choices.values():
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = require_external_output(args.output)
    if args.cohort == "GENIE":
        frame, samples = prepare_genie(
            sample_path=args.samples,
            variant_path=args.variants,
            panel_path=args.panels,
            universe_path=args.patient_universe,
        )
    else:
        frame, samples = prepare_icgc(
            cohort=args.cohort,
            source=args.source,
            bindings_path=args.bindings,
            hgnc_path=args.hgnc,
        )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cohort.csv", index=False)
    samples.to_csv(output / "samples.csv", index=False)
    save_json(
        output / "source_config.json",
        dict(
            cohort=args.cohort,
            genes=list(GENES[args.cohort]),
            patients=len(frame),
            samples=len(samples),
            gene_vocabulary_reselected=False,
            background_exclusion="Whole allele excluded when any annotation belongs to any modeled gene",
            negative_call_assumption="All 12 genes included in GENIE panel; WGS/WXS broad-gene profile for ICGC",
        ),
    )


if __name__ == "__main__":
    main()

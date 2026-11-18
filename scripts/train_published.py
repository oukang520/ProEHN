"""Fit frozen-vocabulary native or common-snapshot models from prepared records."""

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import jax
from proehn.research.native import fit_native, select_snapshot_candidate, predict_masked
from proehn.research.protocols import (
    GENES,
    partition,
    snapshot_frame,
    patient_digest,
    save_json,
    require_external_output,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=GENES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=["native", "snapshot"], default="native")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--constant", action="store_true")
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    jax.config.update("jax_enable_x64", True)
    frame = pd.read_csv(args.input)
    genes = GENES[args.cohort]
    if frame.patient_id.astype(str).duplicated().any():
        raise ValueError("One prepared row per patient is required")
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if args.task == "snapshot":
        frame, _ = snapshot_frame(frame, genes)
    outer = range(5) if args.fold is None else [args.fold]
    for fold in outer:
        if args.task == "native":
            train = frame[frame.fold.ne(fold)].copy()
            validation = None
            test = frame[frame.fold.eq(fold)].copy()
        else:
            train, validation, test = partition(frame, fold)
        folder = args.output / f"fold{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "model.json"
        binding = dict(
            cohort=args.cohort,
            task=args.task,
            fold=fold,
            constant=args.constant,
            input_sha256=source_sha,
            training_patient_sha256=patient_digest(train),
            genes=list(genes),
        )
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("run_binding") != binding:
                raise ValueError(
                    "Existing output belongs to a different run; choose another --output"
                )
            print("REUSED", args.cohort, args.task, fold, flush=True)
            artifact = existing
        else:
            artifact = fit_native(train, genes, args.cohort, constant=args.constant)
            diagnostics = []
            if args.task == "snapshot" and not args.constant:
                baseline = fit_native(train, genes, args.cohort, constant=True)
                artifact, diagnostics = select_snapshot_candidate(
                    train, validation, genes, args.cohort, baseline, artifact
                )
            artifact["run_binding"] = binding
            save_json(path, artifact)
            if diagnostics:
                pd.DataFrame(diagnostics).to_csv(
                    folder / "inner_candidates.csv", index=False
                )
        for name, subset in [("validation", validation), ("test", test)]:
            if subset is not None:
                for compartment in ["PT"] if args.task == "snapshot" else ["PT", "MT"]:
                    output = folder / f"{name}_{compartment}_conditional.csv"
                    if not output.exists():
                        predict_masked(artifact, subset, genes, compartment).to_csv(
                            output, index=False, float_format="%.17g"
                        )
        print("COMPLETE", args.cohort, args.task, fold, artifact["nf"], flush=True)


if __name__ == "__main__":
    main()

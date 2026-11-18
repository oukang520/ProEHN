"""Evaluate seven unique fixed-model future stresses after complete core validation."""

import argparse
import json
import hashlib
from pathlib import Path
from proehn.research import stress, stress_generators
from proehn.research.protocols import load_protocol, require_external_output, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stress.yaml"))
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--reference-fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(args.config)
    datasets = [
        (scenario, seed)
        for scenario in protocol["scenarios"]
        for seed in protocol["training_seeds"]
    ]
    allowed = {f"{scenario}_s{seed}" for scenario, seed in datasets}
    if args.dataset and args.dataset not in allowed:
        raise ValueError("Unknown dataset")
    inputs = {}
    proof = next(
        (
            path
            for path in [args.core / "completion.json", args.core / "verification.json"]
            if path.exists()
        ),
        None,
    )
    if proof is None or json.loads(proof.read_text(encoding="utf-8")).get(
        "status"
    ) not in ["COMPLETE", "PASS"]:
        raise ValueError("All core datasets must pass before any stress evaluation")
    core_protocol_sha = stress.sha(args.core / "protocol.json")
    for scenario, seed in datasets:
        folder = args.core / "datasets" / f"{scenario}_s{seed}"
        completion = json.loads(
            (folder / "completion.json").read_text(encoding="utf-8")
        )
        if completion["scenario"] != scenario or completion["training_seed"] != seed:
            raise ValueError("Core validation is incomplete or mismatched")
        if completion["protocol_sha256"] != core_protocol_sha:
            raise ValueError(
                "Core dataset protocol differs from the complete core experiment"
            )
        for filename, digest in completion["sha256"].items():
            if stress.sha(folder / filename) != digest:
                raise ValueError("A completed core output changed: " + filename)
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        if not selection["selection_completed_before_test"]:
            raise ValueError("A committed development-selected model is required")
        paths = [
            folder / "selection.json",
            folder / "fresh_test_truth.jsonl.gz",
            folder / "completion.json",
        ]
        paths += [
            args.reference_fits / f"{scenario}_n4000_s{seed}_i11_{nf}.json"
            for nf in [0, 1]
        ]
        inputs.update(
            {
                str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            }
        )
    for name, value in [
        ("protocol.json", protocol),
        ("input_manifest.json", dict(inputs=inputs)),
    ]:
        path = args.output / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(
                "Existing stress output belongs to different inputs or settings"
            )
        if not path.exists():
            save_json(path, value)
    manifest = dict(
        inputs=inputs, protocol_sha256=stress.sha(args.output / "protocol.json")
    )
    conditions = stress_generators.conditions(protocol)
    save_json(args.output / "generator_validation.json", stress_generators.validate())
    save_json(args.output / "conditions.json", conditions)
    for scenario, seed in datasets:
        if args.dataset and args.dataset != f"{scenario}_s{seed}":
            continue
        stress.run_dataset(
            protocol,
            stress_generators,
            scenario,
            seed,
            conditions,
            manifest,
            output=args.output,
            core=args.core,
            reference_fits=args.reference_fits,
        )
    if all(
        (args.output / "datasets" / name / "completion.json").exists()
        for name in allowed
    ):
        stress.summarize(protocol, conditions, manifest, output=args.output)


if __name__ == "__main__":
    main()

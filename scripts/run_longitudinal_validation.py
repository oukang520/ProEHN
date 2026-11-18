"""Select on independent observed records, then evaluate true simulated futures."""

import argparse
import hashlib
import json
from pathlib import Path
import jax
from proehn.research.protocols import load_protocol, save_json, require_external_output
from proehn.research.longitudinal import evaluate_dataset, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--reference-fits", type=Path, required=True)
    parser.add_argument("--additional-fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        help="Optional scenario_sSEED; aggregate only when all are available",
    )
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    jax.config.update("jax_enable_x64", True)
    cfg = load_protocol(args.config)
    protocol = dict(
        scenarios=cfg["scenarios"],
        training_seeds=cfg["training_seeds"],
        candidate_weights=cfg["development"]["candidate_weights"],
        development_patients_per_context=cfg["development"]["patients_per_context"],
        development_seed_offset=cfg["development"]["seed_offset"],
        test_patients_per_context=cfg["test"]["patients_per_context"],
        test_seed_offset=cfg["test"]["seed_offset"],
        continuation_seed_namespace=cfg["test"]["continuation_seed_namespace"],
        prefix_lengths=cfg["test"]["prefix_lengths"],
        future_lengths=cfg["test"]["future_lengths"],
        bootstrap_replicates=cfg["test"]["bootstrap_replicates"],
        bootstrap_seed=cfg["test"]["bootstrap_seed"],
        uncertainty="Paired patients within six context-by-observation-design strata; equal weights across three training datasets; conditional on fixed fits and fixed generators.",
    )
    sources = {}
    for scenario in protocol["scenarios"]:
        for seed in protocol["training_seeds"]:
            for init, nf in [(11, 0), (11, 1), (29, 1)]:
                filename = f"{scenario}_n4000_s{seed}_i{init}_{nf}.json"
                source = (
                    args.reference_fits if init == 11 else args.additional_fits
                ) / filename
                sources[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol["source_fit_sha256"] = sources
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError(
                "Output belongs to different fits or settings; choose a new --output"
            )
    else:
        save_json(path, protocol)
    for scenario in protocol["scenarios"]:
        for seed in protocol["training_seeds"]:
            if args.dataset and args.dataset != f"{scenario}_s{seed}":
                continue
            evaluate_dataset(
                protocol,
                scenario,
                seed,
                output=args.output,
                reference_fits=args.reference_fits,
                additional_fits=args.additional_fits,
            )
    summarize(protocol, output=args.output)


if __name__ == "__main__":
    main()

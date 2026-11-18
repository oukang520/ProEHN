"""Export all390 coefficients, all78 precedence pairs and independent-start stability."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from proehn.research.simulation import (
    truth_parameters,
    project,
    embed_constant,
    coefficient_groups,
    precedence_probabilities,
    state_probes,
    event_rates,
)
from proehn.research.protocols import require_external_output, load_protocol, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--reference-fits", type=Path, required=True)
    parser.add_argument("--additional-fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_protocol(args.config)
    order = cfg["recovery"]
    uniforms = np.random.default_rng(order["seed"]).random(
        (order["paths_per_context"], 13)
    )
    block, context = coefficient_groups()
    metrics = []
    coefficients = []
    pairs = []
    fits = {}
    for scenario in cfg["scenarios"]:
        truth = truth_parameters(scenario)
        true_order = {
            z: precedence_probabilities(project(truth, z)[0], uniforms)[2]
            for z in order["contexts"]
        }
        for seed in cfg["training_seeds"]:
            for init, nf in [(11, 0), (11, 1), (29, 1)]:
                directory = args.reference_fits if init == 11 else args.additional_fits
                path = directory / f"{scenario}_n4000_s{seed}_i{init}_{nf}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("status") != "COMPLETE":
                    raise ValueError(f"Completed frozen fit required: {path.name}")
                fits[scenario, seed, init, nf] = result
                full = np.asarray(result["parameters"])
                embedded = full if nf else embed_constant(full)
                error = embedded - truth
                common = dict(
                    scenario=scenario,
                    simulation_seed=seed,
                    initialization_seed=init,
                    model="ProEHN topology" if nf else "Covariate-free coupled",
                    n_train=4000,
                )
                for group in [
                    "all",
                    "evolution",
                    "primary_observation",
                    "metastasis_observation",
                ]:
                    for kind in ["all", "intercept", "context"]:
                        mask = (
                            (block == group) if group != "all" else np.ones(390, bool)
                        ) & ((context == kind) if kind != "all" else np.ones(390, bool))
                        metrics.append(
                            dict(
                                **common,
                                group=group,
                                coefficient_type=kind,
                                parameters=int(mask.sum()),
                                RMSE=float(np.sqrt(np.mean(error[mask] ** 2))),
                                bias=float(error[mask].mean()),
                                converged=result["fit"]["converged"],
                            )
                        )
                coefficients.extend(
                    dict(
                        **common,
                        index=i,
                        group=block[i],
                        coefficient_type=context[i],
                        truth=float(t),
                        estimate=float(v),
                        error=float(e),
                    )
                    for i, (t, v, e) in enumerate(zip(truth, embedded, error))
                )
                for z in order["contexts"]:
                    a, b, p = precedence_probabilities(
                        project(full, z, features=nf)[0], uniforms
                    )
                    pairs.extend(
                        dict(
                            **common,
                            context=z,
                            event_i="seeding" if i == 12 else f"gene{i+1}",
                            event_j="seeding" if j == 12 else f"gene{j+1}",
                            truth_precedence=float(t),
                            estimated_precedence=float(v),
                            absolute_error=float(abs(v - t)),
                            squared_error=float((v - t) ** 2),
                        )
                        for i, j, t, v in zip(a, b, true_order[z], p)
                    )
    stability = []
    for scenario in cfg["scenarios"]:
        for seed in cfg["training_seeds"]:
            reference = fits[scenario, seed, 11, 1]
            comparison = fits[scenario, seed, 29, 1]
            a, b = np.asarray(reference["parameters"]), np.asarray(
                comparison["parameters"]
            )
            tv = []
            for state, _ in state_probes(truth_parameters(scenario), seed + 20000):
                z, pt, mt, s, pd_, md, kind = state
                ra = event_rates(*project(a, z), pt, mt, s, pd_, md, kind)
                rb = event_rates(*project(b, z), pt, mt, s, pd_, md, kind)
                tv.append(0.5 * np.abs(ra / ra.sum() - rb / rb.sum()).sum())
            stability.append(
                dict(
                    scenario=scenario,
                    simulation_seed=seed,
                    reference_seed=11,
                    comparison_seed=29,
                    parameter_RMS_difference=float(np.sqrt(np.mean((a - b) ** 2))),
                    heldout_NLL_difference=float(
                        comparison["heldout_nll"] - reference["heldout_nll"]
                    ),
                    competing_event_TV_between_starts=float(np.mean(tv)),
                    both_converged=bool(
                        reference["fit"]["converged"] and comparison["fit"]["converged"]
                    ),
                )
            )
    for name, rows in [
        ("parameter_recovery", metrics),
        ("all_coefficients", coefficients),
        ("all_precedence_pairs", pairs),
        ("initialization_stability", stability),
    ]:
        pd.DataFrame(rows).to_csv(
            args.output / (name + ".csv"), index=False, float_format="%.17g"
        )
    pd.DataFrame(pairs).groupby(
        ["scenario", "simulation_seed", "initialization_seed", "model"], as_index=False
    )[["absolute_error", "squared_error"]].mean().to_csv(
        args.output / "order_recovery_summary.csv", index=False
    )
    save_json(
        args.output / "completion.json",
        dict(
            status="COMPLETE",
            fits=len(fits),
            stability_pairs=len(stability),
            parameters_per_fit=390,
            precedence_pairs_per_context=78,
            paths_per_context=order["paths_per_context"],
            seed=order["seed"],
            model_fits=0,
        ),
    )


if __name__ == "__main__":
    main()

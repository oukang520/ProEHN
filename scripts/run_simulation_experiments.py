"""Run the 24 reference fits and 12 independent context initializations."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
from proehn.training import fit_topology_model
from proehn.preprocessing import calculate_marginal_rates
from proehn.topology import ProEHNTopologyModel
from proehn.research.simulation import (
    truth_parameters,
    sample_observations,
    buckets,
    state_probes,
    evaluate,
)
from proehn.research.protocols import load_protocol, save_json, require_external_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scenario",
        choices=[
            "no_context",
            "evolution_context",
            "observation_context",
            "joint_context",
        ],
    )
    parser.add_argument("--seed", type=int, choices=[101, 202, 303])
    args = parser.parse_args()
    args.output = require_external_output(args.output)
    jax.config.update("jax_enable_x64", True)
    config = load_protocol(args.config)
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = [args.scenario] if args.scenario else config["scenarios"]
    seeds = [args.seed] if args.seed else config["training_seeds"]
    for scenario in scenarios:
        truth = truth_parameters(scenario)
        for seed in seeds:
            data = sample_observations(
                truth, config["train_patients_per_context"], seed
            )
            heldout = sample_observations(
                truth, config["heldout_patients_per_context"], seed + 10000
            )
            probes = state_probes(truth, seed + 20000)
            for init in config["initialization_seeds"]:
                for nf in [0, 1] if init == 11 else [1]:
                    path = (
                        args.output
                        / f"{scenario}_n{len(data)}_s{seed}_i{init}_{nf}.json"
                    )
                    binding = dict(
                        config_sha256=config_hash,
                        scenario=scenario,
                        simulation_seed=seed,
                        initialization_seed=init,
                        nf=nf,
                    )
                    if path.exists():
                        old = json.loads(path.read_text(encoding="utf-8"))
                        if old.get("run_binding") != binding:
                            raise ValueError(
                                "Existing fit has a different run binding; choose a new output"
                            )
                        if old.get("status") == "COMPLETE":
                            print("REUSED", path.name, flush=True)
                            continue
                    grouped, weights = buckets(data, features=nf)
                    checkpoint = path.with_suffix(".checkpoint.json")
                    initial = None
                    if checkpoint.exists():
                        state = json.loads(checkpoint.read_text(encoding="utf-8"))
                        if state["run_binding"] != binding:
                            raise ValueError("Checkpoint protocol mismatch")
                        initial = np.asarray(state["parameters"])
                    elif init != 11:
                        marginal = calculate_marginal_rates(grouped, 12, weights)
                        base = ProEHNTopologyModel(
                            12, nf, log_rate_clip_min=-5.0, log_rate_clip_max=5.0
                        ).init_params(jax.random.PRNGKey(init), base_rates=marginal)
                        bound = config["optimizer"]["optimizer_bound"]
                        initial = np.clip(
                            np.asarray(base)
                            + np.random.default_rng(init).normal(
                                0, config["additional_start_sd"], 390
                            ),
                            -bound,
                            bound,
                        )
                    calls = [0]

                    def progress(parameters):
                        calls[0] += 1
                        if calls[0] % 25 == 0:
                            save_json(
                                checkpoint,
                                dict(
                                    parameters=np.asarray(parameters).tolist(),
                                    iteration_this_invocation=calls[0],
                                    run_binding=binding,
                                ),
                            )

                    model, parameters, _ = fit_topology_model(
                        grouped,
                        12,
                        nf,
                        len(data),
                        bucket_sample_weights=weights,
                        random_seed=init,
                        initial_parameters=initial,
                        progress_callback=progress,
                        **config["optimizer"],
                    )
                    result = dict(
                        status="COMPLETE",
                        run_binding=binding,
                        parameters=parameters.tolist(),
                        fit=model.fit_diagnostics,
                        n_train=len(data),
                        n_test=len(heldout),
                        cumulative_optimizer_iterations=None,
                        **evaluate(parameters, heldout, nf, probes),
                    )
                    save_json(path, result)
                    print("COMPLETE", path.name, flush=True)


if __name__ == "__main__":
    main()

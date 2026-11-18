"""Fresh-branch stress evaluation of the committed development-selected fits."""

from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json
import numpy as np
import pandas as pd
from .simulation import truth_parameters, project, embed_constant
from .future import prefix_arrays, forecast_metrics, ordered_log_probability
from .protocols import save_json as dump

MODELS = ["No-covariate", "Original init11", "Development-selected"]
COMPARISONS = [
    ("selected_vs_baseline", "No-covariate", "Development-selected"),
    ("original_vs_baseline", "No-covariate", "Original init11"),
    ("selected_vs_original", "Original init11", "Development-selected"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def case_strata(cases):
    groups = []
    for case in cases:
        record = case["original_record"]
        design = 0 if case["kind"] == 4 else (2 if record[3][-1] == 0 else 1)
        groups.append(3 * int(case["context_index"]) + design)
    groups = np.asarray(groups, dtype=int)
    contexts = np.asarray([case["context"] for case in cases])
    assert set(np.unique(groups)) == set(range(6))
    assert set(np.unique(contexts)) == {-1.0, 1.0}
    for context in [-1.0, 1.0]:
        indices = np.flatnonzero(contexts == context)
        counts = [np.sum(groups[indices] % 3 == design) for design in [0, 1, 2]]
        assert counts == [len(indices) // 2, len(indices) // 4, len(indices) // 4]
    return contexts, groups


def bootstrap_six_strata(gains, contexts, groups, weights, replicates, seed):
    """gains axes: comparison, condition, patient, metric.

    One common draw is used across every comparison/condition/metric. Group
    means receive q_context * n_group/n_context, matching the fixed design.
    """
    comparisons, conditions, patients, metrics = gains.shape
    assert len(contexts) == len(groups) == patients
    weights = np.asarray(weights, dtype=float)
    result = np.zeros((replicates, comparisons, conditions, metrics))
    rng = np.random.default_rng(seed)
    for group in sorted(np.unique(groups)):
        ids = np.flatnonzero(groups == group)
        context = contexts[ids[0]]
        assert np.all(contexts[ids] == context)
        design_weight = len(ids) / np.sum(contexts == context)
        population_weight = weights if context == 1 else 1 - weights
        values = gains[:, :, ids, :].transpose(2, 0, 1, 3).reshape(len(ids), -1)
        probability = np.full(len(ids), 1.0 / len(ids))
        for start in range(0, replicates, 32):
            batch = min(32, replicates - start)
            multiplicity = rng.multinomial(len(ids), probability, size=batch)
            means = (multiplicity @ values / len(ids)).reshape(
                batch, comparisons, conditions, metrics
            )
            result[start : start + batch] += (
                means * population_weight[None, None, :, None] * design_weight
            )
    return result


def run_dataset(
    protocol,
    legacy,
    scenario,
    seed,
    conditions,
    manifest,
    *,
    output,
    core,
    reference_fits,
):
    ROOT = Path(output)
    CORE = Path(core)
    REFERENCE = Path(reference_fits)
    out = ROOT / "datasets" / f"{scenario}_s{seed}"
    completion_path = out / "completion.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        assert completion["protocol_sha256"] == sha(ROOT / "protocol.json")
        assert completion["input_manifest_sha256"] == sha(ROOT / "input_manifest.json")
        assert all(
            sha(out / name) == digest for name, digest in completion["outputs"].items()
        )
        log("REUSE " + out.name)
        return
    out.mkdir(parents=True, exist_ok=True)
    core_directory = CORE / "datasets" / out.name
    with gzip.open(
        core_directory / "fresh_test_truth.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        cases = [json.loads(line) for line in stream]
    assert len(cases) == protocol["patients_per_dataset"] == 8000
    contexts, groups = case_strata(cases)
    ids = np.asarray(
        [
            f"{scenario}_s{seed}_c{c['context_index']}_p{c['within_context_patient']:05d}"
            for c in cases
        ]
    )
    assert len(np.unique(ids)) == len(cases)
    states, _ = prefix_arrays(
        cases, protocol["latent_prefix_length"], protocol["future_length"]
    )
    scenario_index = protocol["scenarios"].index(scenario)
    truth = truth_parameters(scenario)
    truth_thetas = {z: project(truth, z)[0] for z in [-1.0, 1.0]}
    future_seed = np.random.SeedSequence(
        [protocol["future_uniform_namespace"], scenario_index, seed]
    )
    mask_seed = np.random.SeedSequence(
        [protocol["measurement_namespace"], scenario_index, seed]
    )
    future_uniforms = np.random.default_rng(future_seed).random((len(cases), 3))
    call_uniforms = np.random.default_rng(mask_seed).random((len(cases), 24))
    amplitudes = protocol["families"]["higher_order_future_generator"]["levels"]
    dropout_levels = protocol["families"]["prefix_false_negative"]["levels"]
    targets = {
        amplitude: legacy.draw_future(
            states, contexts, truth_thetas, amplitude, future_uniforms
        )
        for amplitude in amplitudes
    }
    measured = {
        level: legacy.drop_calls(states, level, call_uniforms)
        for level in dropout_levels
    }
    np.savez_compressed(
        out / "case_inputs_and_targets.npz",
        patient_id=ids,
        contexts=contexts,
        strata=groups,
        true_prefix=states,
        dropout_levels=np.asarray(dropout_levels),
        measured_prefix=np.stack([measured[level] for level in dropout_levels]),
        amplitudes=np.asarray(amplitudes),
        true_future3=np.stack([targets[a] for a in amplitudes]),
        future_uniforms=future_uniforms,
        measurement_uniforms=call_uniforms,
    )
    baseline = read_json(REFERENCE / f"{scenario}_n4000_s{seed}_i11_0.json")
    original = read_json(REFERENCE / f"{scenario}_n4000_s{seed}_i11_1.json")
    selected = read_json(core_directory / "selection.json")
    assert baseline["status"] == original["status"] == "COMPLETE"
    parameters = {
        "No-covariate": embed_constant(np.asarray(baseline["parameters"])),
        "Original init11": np.asarray(original["parameters"]),
        "Development-selected": np.asarray(selected["parameters"]),
    }
    fitted = {
        model: {context: project(value, context)[0] for context in [-1.0, 1.0]}
        for model, value in parameters.items()
    }
    metric_names = protocol["metrics"]
    cache = {}
    support_failures = []
    metric_values = np.empty((3, len(conditions), len(cases), len(metric_names)))
    for condition_index, condition in enumerate(conditions):
        dropout = condition["dropout_rate"]
        amplitude = condition["higher_order_amplitude"]
        key = (dropout, amplitude)
        if key not in cache:
            cache[key] = {}
            for model in MODELS:
                values = np.empty((len(cases), len(metric_names)))
                for context in [-1.0, 1.0]:
                    chosen = np.flatnonzero(contexts == context)
                    state = measured[dropout][chosen]
                    target = targets[amplitude][chosen]
                    support = ordered_log_probability(
                        fitted[model][context], state, target
                    )
                    bad = ~np.isfinite(support)
                    if bad.any():
                        support_failures.append(
                            {
                                "model": model,
                                "dropout_rate": dropout,
                                "higher_order_amplitude": amplitude,
                                "context": context,
                                "patient_ids": ids[chosen[bad]].tolist(),
                            }
                        )
                        dump(
                            out / "support_failure.json",
                            {
                                "status": "FAIL",
                                "no_case_excluded": True,
                                "failures": support_failures,
                                "all_inputs_retained": "case_inputs_and_targets.npz",
                            },
                        )
                        raise RuntimeError(
                            "Unsupported true stress target. Evidence retained; no completed result declared."
                        )
                    metrics = forecast_metrics(fitted[model][context], state, target)
                    values[chosen] = np.column_stack(
                        [metrics[name] for name in metric_names]
                    )
                assert np.isfinite(values).all()
                cache[key][model] = values
            log(f"SCORED {out.name} dropout={dropout:g} amplitude={amplitude:g}")
        for model_index, model in enumerate(MODELS):
            metric_values[model_index, condition_index] = cache[key][model]
    analysis_weights = np.asarray(
        [
            np.where(
                contexts == 1,
                c["weight_context_plus"] / np.sum(contexts == 1),
                (1 - c["weight_context_plus"]) / np.sum(contexts == -1),
            )
            for c in conditions
        ]
    )
    assert np.allclose(analysis_weights.sum(axis=1), 1)
    signs = np.asarray(
        [
            1.0 if name.endswith("nll") or name.endswith("brier") else -1.0
            for name in metric_names
        ]
    )
    gains = np.stack(
        [
            (metric_values[MODELS.index(left)] - metric_values[MODELS.index(right)])
            * signs
            for _, left, right in COMPARISONS
        ]
    )
    model_means = np.einsum("kn,mknv->mkv", analysis_weights, metric_values)
    gain_means = np.einsum("kn,cknv->ckv", analysis_weights, gains)
    weights = [c["weight_context_plus"] for c in conditions]
    bootstrap_seed = int(
        np.random.SeedSequence(
            [
                protocol["bootstrap_namespace"],
                scenario_index,
                seed,
            ]
        ).generate_state(1)[0]
    )
    log("BOOTSTRAP " + out.name + " shared six-stratum patient draws")
    boot = bootstrap_six_strata(
        gains,
        contexts,
        groups,
        weights,
        protocol["bootstrap"]["replicates"],
        bootstrap_seed,
    )
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    zero_aliases = [
        i
        for i, c in enumerate(conditions)
        if c["dropout_rate"] == 0
        and c["higher_order_amplitude"] == 0
        and c["weight_context_plus"] == 0.5
    ]
    assert len(zero_aliases) == 3
    for index in zero_aliases[1:]:
        assert np.array_equal(
            metric_values[:, index], metric_values[:, zero_aliases[0]]
        )
        assert np.allclose(
            boot[:, :, index], boot[:, :, zero_aliases[0]], rtol=0, atol=1e-14
        )
    np.savez_compressed(
        out / "per_patient_model_metrics.npz",
        models=np.asarray(MODELS),
        conditions=np.asarray([c["condition"] for c in conditions]),
        metrics=np.asarray(metric_names),
        patient_id=ids,
        values=metric_values,
        analysis_weights=analysis_weights,
        contexts=contexts,
        strata=groups,
    )
    np.savez_compressed(
        out / "per_patient_paired_gains.npz",
        comparisons=np.asarray([c[0] for c in COMPARISONS]),
        conditions=np.asarray([c["condition"] for c in conditions]),
        metrics=np.asarray(metric_names),
        patient_id=ids,
        values=gains,
        analysis_weights=analysis_weights,
    )
    np.savez_compressed(
        out / "paired_bootstrap_draws.npz",
        conditions=np.asarray([c["condition"] for c in conditions]),
        metrics=np.asarray(metric_names),
        **{name: boot[:, index] for index, (name, _, _) in enumerate(COMPARISONS)},
    )
    model_rows, gain_rows = [], []
    for condition_index, condition in enumerate(conditions):
        common = dict(
            scenario=scenario,
            training_seed=seed,
            **condition,
            n_actual_patients=len(cases),
            zero_target_support_cases=0,
        )
        for model_index, model in enumerate(MODELS):
            for metric_index, metric in enumerate(metric_names):
                model_rows.append(
                    dict(
                        common,
                        model=model,
                        endpoint=metric,
                        weighted_mean=float(
                            model_means[model_index, condition_index, metric_index]
                        ),
                    )
                )
        for comparison_index, (comparison, _, right) in enumerate(COMPARISONS):
            for metric_index, metric in enumerate(metric_names):
                gain_rows.append(
                    dict(
                        common,
                        comparison=comparison,
                        endpoint=metric,
                        gain=float(
                            gain_means[comparison_index, condition_index, metric_index]
                        ),
                        low=float(low[comparison_index, condition_index, metric_index]),
                        high=float(
                            high[comparison_index, condition_index, metric_index]
                        ),
                        positive_favors=right,
                    )
                )
    pd.DataFrame(model_rows).to_csv(
        out / "model_metric_summary.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(gain_rows).to_csv(
        out / "paired_gain_summary.csv", index=False, float_format="%.17g"
    )
    outputs = {
        path.name: sha(path)
        for path in out.iterdir()
        if path.is_file() and path != completion_path
    }
    dump(
        completion_path,
        {
            "status": "COMPLETE",
            "scenario": scenario,
            "training_seed": seed,
            "protocol_sha256": manifest["protocol_sha256"],
            "input_manifest_sha256": sha(ROOT / "input_manifest.json"),
            "selected_candidate": selected["id"],
            "selection_sha256": sha(core_directory / "selection.json"),
            "actual_patients": len(cases),
            "fixed_strata": 6,
            "condition_labels": 9,
            "unique_conditions": 7,
            "prediction_input_conditions": len(cache),
            "models": 3,
            "metrics": 8,
            "no_case_excluded": True,
            "zero_target_support_cases": 0,
            "baseline_aliases_match": True,
            "shared_patient_draws_all_comparisons_conditions_metrics": True,
            "outputs": outputs,
        },
    )
    log("COMPLETE " + out.name)


def summarize(protocol, conditions, manifest, *, output):
    ROOT = Path(output)
    paired, model = [], []
    for scenario in protocol["scenarios"]:
        for seed in protocol["training_seeds"]:
            directory = ROOT / "datasets" / f"{scenario}_s{seed}"
            completion = read_json(directory / "completion.json")
            assert completion["status"] == "COMPLETE"
            assert completion["actual_patients"] == protocol["patients_per_dataset"]
            paired.append(
                pd.read_csv(
                    directory / "paired_gain_summary.csv", float_precision="round_trip"
                )
            )
            model.append(
                pd.read_csv(
                    directory / "model_metric_summary.csv", float_precision="round_trip"
                )
            )
    gains = pd.concat(paired, ignore_index=True)
    models = pd.concat(model, ignore_index=True)
    gains.to_csv(ROOT / "all_dataset_gains.csv", index=False, float_format="%.17g")
    models.to_csv(ROOT / "all_model_metrics.csv", index=False, float_format="%.17g")
    rows = []
    for scenario in protocol["scenarios"]:
        for comparison, _, right in COMPARISONS:
            distributions = []
            for seed in protocol["training_seeds"]:
                path = (
                    ROOT
                    / "datasets"
                    / f"{scenario}_s{seed}"
                    / "paired_bootstrap_draws.npz"
                )
                with np.load(path) as archive:
                    assert archive["conditions"].tolist() == [
                        c["condition"] for c in conditions
                    ]
                    assert archive["metrics"].tolist() == protocol["metrics"]
                    distributions.append(archive[comparison].copy())
            low, high = np.quantile(
                np.mean(distributions, axis=0), [0.025, 0.975], axis=0
            )
            for condition_index, condition in enumerate(conditions):
                for metric_index, metric in enumerate(protocol["metrics"]):
                    selected = gains[
                        gains.scenario.eq(scenario)
                        & gains.comparison.eq(comparison)
                        & gains.condition.eq(condition["condition"])
                        & gains.endpoint.eq(metric)
                    ]
                    assert len(selected) == 3
                    rows.append(
                        dict(
                            scenario=scenario,
                            **condition,
                            comparison=comparison,
                            endpoint=metric,
                            gain=float(selected.gain.mean()),
                            low=float(low[condition_index, metric_index]),
                            high=float(high[condition_index, metric_index]),
                            n_fixed_training_datasets=3,
                            n_actual_patient_cases=3 * protocol["patients_per_dataset"],
                            min_dataset_gain=float(selected.gain.min()),
                            max_dataset_gain=float(selected.gain.max()),
                            positive_point_datasets=int((selected.gain > 0).sum()),
                            CI_positive_datasets=int((selected.low > 0).sum()),
                            CI_cross_zero_datasets=int(
                                ((selected.low <= 0) & (selected.high >= 0)).sum()
                            ),
                            CI_negative_datasets=int((selected.high < 0).sum()),
                            positive_favors=right,
                            true_biological_context_effect=scenario
                            in ["evolution_context", "joint_context"],
                            pointwise_sign=(
                                "positive"
                                if low[condition_index, metric_index] > 0
                                else (
                                    "negative"
                                    if high[condition_index, metric_index] < 0
                                    else "inconclusive"
                                )
                            ),
                        )
                    )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(ROOT / "aggregate_gains.csv", index=False, float_format="%.17g")
    aggregate[aggregate.endpoint.eq(protocol["primary_metric"])].to_csv(
        ROOT / "positive_negative_inventory.csv",
        index=False,
        float_format="%.17g",
    )
    assert all(sha(path) == digest for path, digest in manifest["inputs"].items())
    summary = {
        "status": "COMPLETE",
        "datasets": 12,
        "patients_per_dataset": 8000,
        "patient_mechanism_cases": 96000,
        "fixed_true_mechanisms": 4,
        "training_datasets_per_mechanism": 3,
        "condition_labels": 9,
        "unique_conditions": 7,
        "models": MODELS,
        "comparisons": [c[0] for c in COMPARISONS],
        "metrics": protocol["metrics"],
        "all_12_core_complete_before_stress": True,
        "selected_model_fixed_across_all_stresses": True,
        "new_training": False,
        "all_cases_retained": True,
        "zero_target_support_cases": 0,
        "scope": protocol["scope"],
        "mechanism_scope": protocol["mechanism_scope"],
        "zero_reference": "Three labels are aliases. Their new true branch differs from the core original suffix. Neither aliases nor family levels are independent patient replications.",
        "context_weights": "Analytical population reweighting of the same cases, not new cohorts.",
        "uncertainty": protocol["bootstrap"],
        "calendar_time_claim": False,
        "inputs_unchanged": True,
    }
    dump(ROOT / "summary.json", summary)
    dump(
        ROOT / "verification.json",
        {
            "status": "PASS",
            "complete_datasets": 12,
            "core_input_hashes_unchanged": True,
            "no_stress_dependent_selection": True,
            "shared_six_stratum_patient_bootstrap": True,
            "no_case_excluded": True,
            "zero_target_support_cases": 0,
            "all_original_scenarios_conditions_retained": True,
            "protocol_sha256": sha(ROOT / "protocol.json"),
            "code_sha256": sha(__file__),
            "aggregate_sha256": sha(ROOT / "aggregate_gains.csv"),
        },
    )
    log("ALL STRESS COMPLETE")

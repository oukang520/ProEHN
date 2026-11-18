"""Independent-development validation of frozen fits, without retraining."""

from pathlib import Path
from datetime import datetime, timezone
import gzip, hashlib, json, time
import numpy as np
import pandas as pd
from . import simulation
from .simulation import (
    truth_parameters,
    sample_observations,
    project,
    embed_constant,
    state_probes,
    event_rates,
)
from .future import (
    replay_observation_histories,
    complete_truth_histories,
    prefix_arrays,
    forecast_metrics,
)
from .development import (
    strata,
    candidate_set,
    compress_records,
    losses,
    choose,
    bootstrap,
    order_losses,
)
from .protocols import save_json as dump

METRICS = [
    "unordered_future_set_nll",
    "ordered_future_sequence_nll",
    "next_event_nll",
    "next_event_brier",
    "next_event_mrr",
    "next_event_recall_at_1",
    "next_event_recall_at_3",
    "next_event_recall_at_5",
]
MODELS = ["No-covariate", "Original init11", "Development-selected", "Oracle"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def evaluate_dataset(
    protocol, scenario, seed, *, output, reference_fits, additional_fits
):
    ROOT = Path(output)

    def fit_paths(scenario, seed):
        return {
            (init, nf): Path(reference_fits if init == 11 else additional_fits)
            / f"{scenario}_n4000_s{seed}_i{init}_{nf}.json"
            for init, nf in [(11, 0), (11, 1), (29, 1)]
        }

    dest = ROOT / "datasets" / f"{scenario}_s{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / "completion.json"
    if done.exists():
        for f, h in load(done)["sha256"].items():
            assert sha(dest / f) == h
        log("REUSE " + dest.name)
        return
    started = time.time()
    fit = {k: load(v) for k, v in fit_paths(scenario, seed).items()}
    assert all(
        f["status"] == "COMPLETE" and f["fit"]["converged"] for f in fit.values()
    )
    base = embed_constant(np.array(fit[11, 0]["parameters"]))
    full = {init: np.array(fit[init, 1]["parameters"]) for init in [11, 29]}
    candidates = candidate_set(base, full, protocol["candidate_weights"])
    truth = truth_parameters(scenario)
    selection_path = dest / "selection.json"
    if selection_path.exists():
        selected = load(selection_path)
        parameters = np.array(selected["parameters"])
        assert selected["protocol_sha256"] == sha(ROOT / "protocol.json")
    else:
        dev = sample_observations(
            truth,
            protocol["development_patients_per_context"],
            seed + protocol["development_seed_offset"],
        )
        groups = strata(protocol["development_patients_per_context"])
        compressed = compress_records(dev)
        log(
            f"DEV {dest.name} patients={len(dev)} unique={len(compressed[0])} candidates={len(candidates)}"
        )
        all_losses = []
        for i, c in enumerate(candidates):
            all_losses.append(losses(c["parameters"], compressed))
            if i % 5 == 0:
                log(
                    f"DEV {dest.name} candidate={i+1}/{len(candidates)} elapsed={time.time()-started:.1f}s"
                )
        values = np.array(all_losses)
        index, scores = choose(candidates, values, groups)
        pd.DataFrame(scores).to_csv(
            dest / "development_candidate_scores.csv", index=False, float_format="%.17g"
        )
        np.savez_compressed(
            dest / "development_losses.npz",
            losses=values,
            strata=groups,
            candidate_ids=np.array([c["id"] for c in candidates]),
        )
        parameters = candidates[index]["parameters"]
        selected = {
            k: v for k, v in candidates[index].items() if k != "parameters"
        } | dict(
            parameters=parameters.tolist(),
            protocol_sha256=sha(ROOT / "protocol.json"),
            selection_completed_before_test=True,
            dev_patients=len(dev),
            dev_mean_nll=scores[index]["dev_mean_nll"],
            n_candidates=len(candidates),
        )
        dump(selection_path, selected)
        log(f'SELECT {dest.name} {selected["id"]}')
    # No test data are generated or scored until selection has been committed.
    source = simulation
    cases = replay_observation_histories(
        source,
        truth,
        protocol["test_patients_per_context"],
        seed + protocol["test_seed_offset"],
    )
    complete_truth_histories(
        cases,
        {z: project(truth, z)[0] for z in [-1.0, 1.0]},
        seed,
        protocol["continuation_seed_namespace"],
    )
    with gzip.open(dest / "fresh_test_truth.jsonl.gz", "wt", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    contexts = np.array([c["context"] for c in cases])
    groups = strata(protocol["test_patients_per_context"])
    params = {
        "No-covariate": base,
        "Original init11": full[11],
        "Development-selected": parameters,
        "Oracle": truth,
    }
    records = [tuple(c["original_record"]) for c in cases]
    compressed = compress_records(records)
    model_values = {}
    endpoint_groups = {}
    endpoint_meta = []
    arrays = {m: [] for m in MODELS}
    keys = []

    def add(key, vals, group, metadata):
        keys.append(key)
        endpoint_groups[key] = group
        endpoint_meta.append(dict(endpoint=key, **metadata))
        for model in MODELS:
            arrays[model].append(vals[model])

    add(
        "record_nll",
        {m: losses(p, compressed) for m, p in params.items()},
        groups,
        dict(domain="observed_record", metric="nll", horizon=0, prefix=0),
    )
    # Keep diagnosis endpoints separate because only paired patients contribute.
    order = {m: order_losses(p, cases) for m, p in params.items()}
    order_groups = groups[order["Oracle"][2]]
    order_values = {m: np.c_[v[0], v[1]] for m, v in order.items()}
    for prefix in protocol["prefix_lengths"]:
        states, suffix = prefix_arrays(cases, prefix, 3)
        for horizon in protocol["future_lengths"]:
            vals = {m: {k: np.empty(len(cases)) for k in METRICS} for m in MODELS}
            for m, p in params.items():
                for z in [-1.0, 1.0]:
                    ids = np.flatnonzero(contexts == z)
                    res = forecast_metrics(
                        project(p, z)[0], states[ids], suffix[ids, :horizon]
                    )
                    for metric in METRICS:
                        vals[m][metric][ids] = res[metric]
            for metric in METRICS:
                add(
                    f"p{prefix}_h{horizon}_{metric}",
                    {m: vals[m][metric] for m in MODELS},
                    groups,
                    dict(
                        domain="biological_future",
                        metric=metric,
                        horizon=horizon,
                        prefix=prefix,
                    ),
                )
    models = {m: np.column_stack(v) for m, v in arrays.items()}
    np.savez_compressed(
        dest / "test_patient_metrics.npz",
        **{m.replace(" ", "_"): v for m, v in models.items()},
        endpoint=np.array(keys),
        strata=groups,
        contexts=contexts,
        **{m.replace(" ", "_") + "_order": v for m, v in order_values.items()},
        order_strata=order_groups,
    )
    # Exact, patient-independent transition errors on fresh truth-state probes.
    probes = state_probes(truth, seed + 310000, npatients=2000)
    probe_rows = []
    for model, p in params.items():
        tv = []
        bio = []
        for (z, pt, mt, s, pd_, md, kind), target in probes:
            q = event_rates(*project(p, z), pt, mt, s, pd_, md, kind)
            tv.append(0.5 * np.abs(q / q.sum() - target / target.sum()).sum())
            if target[:-2].sum() > 0 and q[:-2].sum() > 0:
                bio.append(
                    0.5
                    * np.abs(
                        q[:-2] / q[:-2].sum() - target[:-2] / target[:-2].sum()
                    ).sum()
                )
        probe_rows.append(
            dict(
                model=model,
                competing_event_tv=float(np.mean(tv)),
                biological_transition_tv=float(np.mean(bio)),
                n_probes=len(probes),
            )
        )
    pd.DataFrame(probe_rows).to_csv(
        dest / "transition_recovery.csv", index=False, float_format="%.17g"
    )
    # Positive signs favor selected/original as appropriate; proper losses and rankings differ.
    signs = np.array(
        [-1 if k.endswith("mrr") or "recall_at_" in k else 1 for k in keys]
    )
    comparisons = [
        ("selected_vs_baseline", "No-covariate", "Development-selected"),
        ("original_vs_baseline", "No-covariate", "Original init11"),
        ("selected_vs_original", "Original init11", "Development-selected"),
    ]
    rows = []
    boot_arrays = {}
    metric_rows = []
    for m in MODELS:
        for i, key in enumerate(keys):
            metric_rows.append(
                dict(
                    model=m,
                    endpoint=key,
                    mean=float(models[m][:, i].mean()),
                    n_patients=len(cases),
                )
            )
        for i, key in enumerate(["diagnosis_order_nll", "diagnosis_order_brier"]):
            metric_rows.append(
                dict(
                    model=m,
                    endpoint=key,
                    mean=float(order_values[m][:, i].mean()),
                    n_patients=len(order_groups),
                )
            )
    for ci, (name, left, right) in enumerate(comparisons):
        delta = (models[left] - models[right]) * signs
        order_delta = order_values[left] - order_values[right]
        bseed = int(
            np.random.SeedSequence(
                [protocol["bootstrap_seed"], seed, ci]
            ).generate_state(1)[0]
        )
        boots = bootstrap(delta, groups, protocol["bootstrap_replicates"], bseed)
        oboot = bootstrap(
            order_delta, order_groups, protocol["bootstrap_replicates"], bseed + 1
        )
        combined = np.c_[boots, oboot]
        means = np.r_[delta.mean(axis=0), order_delta.mean(axis=0)]
        names = keys + ["diagnosis_order_nll", "diagnosis_order_brier"]
        low, high = np.quantile(combined, [0.025, 0.975], axis=0)
        boot_arrays[name] = combined
        for i, key in enumerate(names):
            rows.append(
                dict(
                    comparison=name,
                    endpoint=key,
                    gain=float(means[i]),
                    low=float(low[i]),
                    high=float(high[i]),
                )
            )
    np.savez_compressed(
        dest / "paired_bootstrap.npz",
        endpoint=np.array(keys + ["diagnosis_order_nll", "diagnosis_order_brier"]),
        **boot_arrays,
    )
    pd.DataFrame(rows).to_csv(
        dest / "paired_gains.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(metric_rows).to_csv(
        dest / "test_model_metrics.csv", index=False, float_format="%.17g"
    )
    dump(dest / "endpoint_metadata.json", endpoint_meta)
    outputs = [p for p in dest.iterdir() if p.is_file() and p != done]
    dump(
        done,
        dict(
            scenario=scenario,
            training_seed=seed,
            seconds=time.time() - started,
            protocol_sha256=sha(ROOT / "protocol.json"),
            selected_candidate=selected["id"],
            sha256={p.name: sha(p) for p in outputs},
        ),
    )
    log(f"COMPLETE {dest.name} elapsed={time.time()-started:.1f}s")


def summarize(protocol, *, output):
    ROOT = Path(output)
    gains = []
    metrics = []
    selections = []
    recovery = []
    aggregate = []
    for scenario in protocol["scenarios"]:
        boots_by_comp = {}
        means_by_comp = {}
        for seed in protocol["training_seeds"]:
            dest = ROOT / "datasets" / f"{scenario}_s{seed}"
            if not (dest / "completion.json").exists():
                return
            gains.append(
                pd.read_csv(dest / "paired_gains.csv").assign(
                    scenario=scenario, training_seed=seed
                )
            )
            metrics.append(
                pd.read_csv(dest / "test_model_metrics.csv").assign(
                    scenario=scenario, training_seed=seed
                )
            )
            recovery.append(
                pd.read_csv(dest / "transition_recovery.csv").assign(
                    scenario=scenario, training_seed=seed
                )
            )
            sel = load(dest / "selection.json")
            selections.append(
                {k: v for k, v in sel.items() if k != "parameters"}
                | dict(scenario=scenario, training_seed=seed)
            )
            b = np.load(dest / "paired_bootstrap.npz")
            endpoints = b["endpoint"].tolist()
            frame = pd.read_csv(dest / "paired_gains.csv")
            for name in [
                "selected_vs_baseline",
                "original_vs_baseline",
                "selected_vs_original",
            ]:
                boots_by_comp.setdefault(name, []).append(b[name])
                means_by_comp.setdefault(name, []).append(
                    frame[frame.comparison == name]
                    .set_index("endpoint")
                    .loc[endpoints, "gain"]
                    .to_numpy()
                )
        for comp, parts in boots_by_comp.items():
            combined = np.mean(parts, axis=0)
            low, high = np.quantile(combined, [0.025, 0.975], axis=0)
            seeds = np.array(means_by_comp[comp])
            means = seeds.mean(axis=0)
            for i, key in enumerate(endpoints):
                aggregate.append(
                    dict(
                        scenario=scenario,
                        comparison=comp,
                        endpoint=key,
                        gain=float(means[i]),
                        low=float(low[i]),
                        high=float(high[i]),
                        positive_training_datasets=int((seeds[:, i] > 1e-12).sum()),
                        min_dataset_gain=float(seeds[:, i].min()),
                        max_dataset_gain=float(seeds[:, i].max()),
                    )
                )
    pd.concat(gains).to_csv(
        ROOT / "all_dataset_gains.csv", index=False, float_format="%.17g"
    )
    pd.concat(metrics).to_csv(
        ROOT / "all_model_metrics.csv", index=False, float_format="%.17g"
    )
    pd.concat(recovery).to_csv(
        ROOT / "all_transition_recovery.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(selections).to_csv(
        ROOT / "all_selections.csv", index=False, float_format="%.17g"
    )
    agg = pd.DataFrame(aggregate)
    agg.to_csv(ROOT / "aggregate_gains.csv", index=False, float_format="%.17g")
    all_gains = pd.concat(gains)
    all_metrics = pd.concat(metrics)
    primary = []
    ni = []
    margin = float(np.log(1.01))
    null_keys = ["record_nll"] + [
        f"p{p}_h{h}_unordered_future_set_nll"
        for p in protocol["prefix_lengths"]
        for h in protocol["future_lengths"]
    ]
    for scope, frame in [("aggregate", agg)] + [
        (f"seed{seed}", all_gains[all_gains.training_seed == seed])
        for seed in protocol["training_seeds"]
    ]:
        f = frame[
            (frame.scenario == "no_context")
            & (frame.comparison == "selected_vs_baseline")
        ].set_index("endpoint")
        for key in null_keys:
            h = 1 if key == "record_nll" else int(key.split("_")[1][1:])
            row = f.loc[key]
            ni.append(
                dict(
                    scope=scope,
                    endpoint=key,
                    excess_nll_per_unit=float(-row["gain"] / h),
                    upper95_excess_nll_per_unit=float(-row["low"] / h),
                    margin=margin,
                    noninferior=bool(-row["low"] / h <= margin),
                )
            )
    primary.append(
        dict(
            scenario="no_context",
            claim="noninferiority, not superiority",
            aggregate_all_seven_endpoints_noninferior=all(
                r["noninferior"] for r in ni if r["scope"] == "aggregate"
            ),
            every_dataset_all_seven_noninferior=all(
                r["noninferior"] for r in ni if r["scope"] != "aggregate"
            ),
            geometric_predictive_probability_margin=1.01,
        )
    )
    for scenario in ["evolution_context", "observation_context", "joint_context"]:
        key = (
            "diagnosis_order_nll"
            if scenario == "observation_context"
            else "p5_h3_unordered_future_set_nll"
        )
        row = agg[
            (agg.scenario == scenario)
            & (agg.comparison == "selected_vs_baseline")
            & (agg.endpoint == key)
        ].iloc[0]
        m = (
            all_metrics[
                (all_metrics.scenario == scenario) & (all_metrics.endpoint == key)
            ]
            .groupby("model")["mean"]
            .mean()
        )
        primary.append(
            dict(
                scenario=scenario,
                endpoint=key,
                gain=float(row["gain"]),
                low=float(row["low"]),
                high=float(row["high"]),
                conditional95_interval_positive=bool(row["low"] > 0),
                positive_training_datasets=int(row.positive_training_datasets),
                baseline_nll=float(m["No-covariate"]),
                selected_nll=float(m["Development-selected"]),
                relative_loss_reduction=float(row["gain"] / m["No-covariate"]),
                practically_strong_not_defined_by_significance_alone=True,
            )
        )
    dump(
        ROOT / "primary_endpoint_results.json",
        dict(primary=primary, uncertainty=protocol["uncertainty"]),
    )
    pd.DataFrame(ni).to_csv(
        ROOT / "null_noninferiority.csv", index=False, float_format="%.17g"
    )
    dump(
        ROOT / "completion.json",
        dict(
            status="COMPLETE",
            datasets=len(protocol["scenarios"]) * len(protocol["training_seeds"]),
            new_training=False,
            selection_before_test=True,
            all_scenarios_retained=True,
            interval_scope=protocol["uncertainty"],
        ),
    )
    log("ALL COMPLETE")

"""Single positive-deletion recovery with common predecessor/candidate sets."""

import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from .paths import project_transition as theta_for
from .protocols import save_json as write

MODELS = ["Training prevalence", "Coupled baseline", "Context model", "Uniform random"]


def rank_metrics(scores, target_index):
    scores = np.asarray(scores, float)
    target = scores[target_index]
    ties = np.abs(scores - target) <= 1e-12
    greater = int(np.sum(scores > target + 1e-12))
    n = int(ties.sum())
    ranks = np.arange(greater + 1, greater + n + 1)
    p = np.exp(scores - logsumexp(scores))
    return {
        **{f"Top{k}": float(np.clip((k - greater) / n, 0, 1)) for k in [1, 3, 5]},
        "MRR": float(np.mean(1 / ranks)),
        "target_log_score": float(np.log(p[target_index])),
    }, p


def score_fold(cohort, fold, frame, genes, models, folder):
    frame = frame.copy()
    aliases = {
        0: "primary_only",
        1: "primary_only",
        2: "metastasis_only",
        3: "paired",
        4: "primary_only",
        "primary_no_dissemination": "primary_only",
        "primary_disseminated": "primary_only",
        "primary_seeding_unknown": "primary_only",
    }
    frame["observation_type"] = frame.observation_type.map(
        lambda value: aliases.get(value, value)
    )
    if not frame.observation_type.isin(
        ["primary_only", "metastasis_only", "paired"]
    ).all():
        raise ValueError("Invalid observation type")
    train = frame[frame.fold.ne(fold)]
    test = frame[frame.fold.eq(fold)].reset_index(drop=True)
    items = []
    eligibility = []
    tensors = {name: theta_for(a, test) for name, a in models.items()}
    for comp, prefix, absent in [
        ("PT", "P.", "metastasis_only"),
        ("MT", "M.", "primary_only"),
    ]:
        observed = train[train.observation_type.ne(absent)]
        freq = (observed[[prefix + g for g in genes]].sum().to_numpy(float) + 0.5) / (
            len(observed) + 1
        )
        for i, row in test.iterrows():
            if row.observation_type == absent:
                continue
            observed_state = row[[prefix + g for g in genes]].to_numpy(int)
            targets = np.flatnonzero(observed_state)
            eligibility.append(
                dict(
                    patient_id=row.patient_id,
                    cohort=cohort,
                    fold=fold,
                    compartment=comp,
                    positive_mutations=len(targets),
                    status=(
                        "ELIGIBLE" if len(targets) else "NO_POSITIVE_DELETION_TARGET"
                    ),
                )
            )
            for target in targets:
                state = observed_state.copy()
                state[target] = 0
                active = np.flatnonzero(state)
                candidates = np.flatnonzero(state == 0)
                target_index = int(np.flatnonzero(candidates == target)[0])
                C = len(candidates)
                scores = {
                    "Training prevalence": np.log(freq[candidates]),
                    "Uniform random": np.zeros(C),
                }
                for name, theta in tensors.items():
                    t = theta[i]
                    scores[name] = (
                        np.diag(t)[candidates]
                        + t[np.ix_(candidates, active)].sum(axis=1)
                        + (t[candidates, 12] if comp == "MT" else 0)
                    )
                for name, values in scores.items():
                    metrics, p = rank_metrics(values, target_index)
                    items.append(
                        dict(
                            patient_id=row.patient_id,
                            cohort=cohort,
                            fold=fold,
                            compartment=comp,
                            target=genes[target],
                            model=name,
                            candidate_count=C,
                            candidates="|".join(genes[j] for j in candidates),
                            candidate_probabilities=json.dumps(
                                p.tolist(), separators=(",", ":")
                            ),
                            **metrics,
                        )
                    )
    result = pd.DataFrame(items)
    result.to_csv(folder / "queries_private.csv", index=False)
    pd.DataFrame(eligibility).to_csv(folder / "eligibility_private.csv", index=False)
    metrics = ["Top1", "Top3", "Top5", "MRR", "target_log_score"]
    rows = []
    for model in MODELS:
        for comp in ["PT", "MT"]:
            for gene in genes:
                q = result[
                    result.model.eq(model)
                    & result.compartment.eq(comp)
                    & result.target.eq(gene)
                ]
                rows.append(
                    dict(
                        cohort=cohort,
                        fold=fold,
                        model=model,
                        compartment=comp,
                        target=gene,
                        n_queries=len(q),
                        n_patients=q.patient_id.nunique(),
                        status="EVALUATED" if len(q) else "NOT_EVALUABLE",
                        reason="" if len(q) else "no_observed_positive_target",
                        **{k: float(q[k].mean()) if len(q) else None for k in metrics},
                    )
                )
    pd.DataFrame(rows).to_csv(folder / "all24_target_metrics.csv", index=False)
    pt = result.groupby(["model", "patient_id"])[metrics].mean().reset_index()
    pt.to_csv(folder / "patient_metrics_private.csv", index=False)
    summary = dict(
        status="COMPLETE",
        cohort=cohort,
        fold=fold,
        n_train=len(train),
        n_test=len(test),
        n_eligible_patients=pt.patient_id.nunique(),
        n_queries_per_model=len(result) // 4,
        patient_macro=pt.groupby("model")[metrics].mean().to_dict("index"),
        query_micro=result.groupby("model")[metrics].mean().to_dict("index"),
    )
    write(folder / "summary.json", summary)
    return summary

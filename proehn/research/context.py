"""Fixed-model context representations, feature contrasts and exact Shapley values."""

from itertools import permutations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from .paths import project_transition, primary_rates

FEATURES = ["age", "background_nMut", "background_VAF_mean"]


def context_results(artifact, frame, genes, *, rng, seed=20261111):
    training = frame[frame.fold.ne(0)]
    test = frame[frame.fold.eq(0)]
    median = training[FEATURES].median().to_frame().T
    baseline = project_transition(artifact, median)[0]
    effects = []
    edges = []
    for feature in FEATURES:
        low = median.copy()
        high = median.copy()
        low[feature] = training[feature].quantile(0.25)
        high[feature] = training[feature].quantile(0.75)
        lt, ht = (
            project_transition(artifact, low)[0],
            project_transition(artifact, high)[0],
        )
        lr, hr = primary_rates(lt, np.zeros(13)), primary_rates(ht, np.zeros(13))
        for target, a, b in [
            ("Seeding", lr[-1], hr[-1]),
            ("Primary mutations", lr[:12].sum(), hr[:12].sum()),
        ]:
            effects.append(
                dict(
                    feature=feature,
                    target=target,
                    low=float(low[feature].iloc[0]),
                    high=float(high[feature].iloc[0]),
                    log2_rate_ratio=float(np.log2(b / a)),
                )
            )
        for i, target in enumerate([*genes, "Seeding"]):
            for j, regulator in enumerate([*genes, "Seeding"]):
                edges.append(
                    dict(
                        feature=feature,
                        target=target,
                        regulator=regulator,
                        kind="Basal" if i == j else "Interaction",
                        median_parameter=float(baseline[i, j]),
                        low_parameter=float(lt[i, j]),
                        high_parameter=float(ht[i, j]),
                        log2_modulation=float((ht[i, j] - lt[i, j]) / np.log(2)),
                        effective_primary_edge=not (i < 12 and j == 12),
                    )
                )
    theta = project_transition(artifact, frame)
    flattened = theta.reshape(len(frame), -1)
    is_training = frame.fold.ne(0).to_numpy()
    scaler = StandardScaler().fit(flattened[is_training])
    scaled = scaler.transform(flattened)
    pca = PCA(n_components=0.95, svd_solver="full").fit(scaled[is_training])
    reduced = pca.transform(scaled)
    model = KMeans(n_clusters=3, n_init=20, random_state=seed).fit(reduced[is_training])
    labels = model.predict(reduced[~is_training])
    ari = []
    for replicate in range(20):
        indices = rng.integers(0, len(training), len(training))
        fitted = KMeans(n_clusters=3, n_init=10, random_state=seed + replicate).fit(
            reduced[is_training][indices]
        )
        ari.append(
            float(adjusted_rand_score(labels, fitted.predict(reduced[~is_training])))
        )
    embedding = np.zeros((len(test), 2))
    embedding[:, : min(2, reduced.shape[1])] = reduced[~is_training, :2]
    representations = pd.DataFrame(
        dict(
            patient_id=test.patient_id.to_numpy(),
            cluster=labels,
            PC1=embedding[:, 0],
            PC2=embedding[:, 1],
            WT_log_seeding_rate=theta[~is_training, 12, 12],
        )
    )
    values = {}
    for mask in range(8):
        raw = pd.concat([median] * len(test), ignore_index=True)
        for j, column in enumerate(FEATURES):
            if mask & (1 << j):
                raw[column] = test[column].to_numpy()
        projected = project_transition(artifact, raw)
        values[mask] = np.c_[
            projected[:, 12, 12],
            np.log(
                np.exp(np.diagonal(projected, axis1=1, axis2=2)[:, :12]).sum(axis=1)
            ),
        ] / np.log(2)
    contribution = np.zeros((len(test), 3, 2))
    for order in permutations(range(3)):
        mask = 0
        for j in order:
            changed = mask | (1 << j)
            contribution[:, j, :] += (values[changed] - values[mask]) / 6
            mask = changed
    residual = float(np.max(np.abs(contribution.sum(axis=1) - (values[7] - values[0]))))
    if residual > 1e-10:
        raise AssertionError("Shapley contributions do not sum to the fitted change")
    rows = []
    for i in range(len(test)):
        for j, feature in enumerate(FEATURES):
            for k, target in enumerate(["Seeding", "Primary accumulation"]):
                rows.append(
                    dict(
                        observation_index=i,
                        feature=feature,
                        target=target,
                        log2_contribution=float(contribution[i, j, k]),
                    )
                )
    response = []
    for age_quantile in [0.25, 0.5, 0.75]:
        for burden_quantile in np.linspace(0.01, 0.99, 99):
            raw = median.copy()
            raw["age"] = training.age.quantile(age_quantile)
            raw["background_nMut"] = training.background_nMut.quantile(burden_quantile)
            projected = project_transition(artifact, raw)[0]
            response.append(
                dict(
                    age_quantile=age_quantile,
                    age=float(raw.age.iloc[0]),
                    burden_percentile=100 * burden_quantile,
                    burden=float(raw.background_nMut.iloc[0]),
                    log2_seeding_rate=float(projected[12, 12] / np.log(2)),
                    log2_rate_ratio=float(
                        (projected[12, 12] - baseline[12, 12]) / np.log(2)
                    ),
                )
            )
    tables = dict(
        context_effects=pd.DataFrame(effects),
        all_context_edge_effects=pd.DataFrame(edges),
        representations=representations,
        patient_feature_contributions=pd.DataFrame(rows),
        age_burden_response=pd.DataFrame(response),
    )
    metadata = dict(
        n_training=len(training),
        n_test=len(test),
        pca_dimensions=reduced.shape[1],
        explained_variance=pca.explained_variance_ratio_.tolist(),
        bootstrap_ARI=ari,
        bootstrap_ARI_median=float(np.median(ari)),
        cluster_counts=np.bincount(labels, minlength=3).tolist(),
        shapley_max_residual=residual,
        model_fits=0,
    )
    return tables, metadata

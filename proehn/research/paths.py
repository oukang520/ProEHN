"""Primary-lineage projection, complete path orders and mass-accounted branches."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .native import feature_matrix


def project_transition(artifact, frame):
    nf = artifact["nf"]
    weights = np.asarray(artifact["parameters"])[: (nf + 1) * 169].reshape(
        nf + 1, 13, 13
    )
    return np.clip(
        np.einsum("nk,kij->nij", feature_matrix(artifact, frame), weights), -5.0, 5.0
    )


def primary_rates(theta, state):
    theta = np.asarray(theta).copy()
    theta[:12, 12] = 0.0
    active = np.flatnonzero(state)
    log = np.diag(theta) + (theta[:, active].sum(axis=1) if len(active) else 0)
    rate = np.exp(log)
    rate[np.asarray(state, dtype=bool)] = 0.0
    return rate


def path_positions(theta, uniforms):
    theta = np.asarray(theta).copy()
    theta[:12, 12] = 0.0
    rate = np.broadcast_to(np.exp(np.diag(theta)), uniforms.shape).copy()
    positions = np.empty(uniforms.shape, int)
    index = np.arange(len(rate))
    for step in range(13):
        cumulative = np.cumsum(rate, axis=1)
        event = np.minimum(
            (cumulative <= uniforms[:, step, None] * cumulative[:, -1, None]).sum(
                axis=1
            ),
            12,
        )
        if not np.all(rate[index, event] > 0):
            raise ValueError("Sampled an unavailable event")
        positions[index, event] = step + 1
        rate *= np.exp(theta[:, event].T)
        rate[index, event] = 0.0
    return positions


def branch_tree(theta, genes, *, max_depth=3):
    """Keep two mutations and seeding per node, without renormalization."""
    nodes = []
    edges = []
    omitted_mass = [0.0]

    def walk(state, path, parent, conditional, depth, cumulative):
        node = len(nodes)
        terminal = "Seeding" in path or depth == max_depth
        item = dict(
            node=node,
            parent=parent,
            depth=depth,
            path=" > ".join(path) if path else "WT",
            event=path[-1] if path else "WT",
            conditional_probability=float(conditional),
            path_probability=float(cumulative),
            terminal=terminal,
            omitted_probability=0.0,
        )
        nodes.append(item)
        if parent >= 0:
            edges.append(
                dict(
                    parent=parent,
                    child=node,
                    conditional_probability=float(conditional),
                    path_probability=float(cumulative),
                )
            )
        if terminal:
            return
        rate = primary_rates(theta, state)
        probability = rate / rate.sum()
        available = np.flatnonzero(rate[:12] > 0)
        selected = sorted(available, key=lambda j: (-probability[j], j))[:2] + [12]
        selected = sorted(selected, key=lambda j: (-probability[j], j))
        omitted = 1 - float(probability[selected].sum())
        item["omitted_probability"] = omitted
        omitted_mass[0] += cumulative * omitted
        for j in selected:
            next_state = state.copy()
            next_state[j] = 1
            walk(
                next_state,
                path + [genes[j] if j < 12 else "Seeding"],
                node,
                probability[j],
                depth + 1,
                cumulative * probability[j],
            )

    walk(np.zeros(13), [], -1, 1.0, 0, 1.0)
    retained = sum(node["path_probability"] for node in nodes if node["terminal"])
    if not np.isclose(retained + omitted_mass[0], 1.0, atol=1e-10):
        raise AssertionError("Branch probability mass did not close")
    return (
        pd.DataFrame(nodes),
        pd.DataFrame(edges),
        dict(retained_path_mass=retained, omitted_path_mass=omitted_mass[0]),
    )


def representative_states(training, genes):
    p = training[["P." + g for g in genes]].to_numpy(float)
    m = training[["M." + g for g in genes]].to_numpy(float)
    mt_only = training.observation_type.isin(["metastasis_only", 2]).to_numpy()
    observed = np.where(mt_only[:, None], m, p)
    if not np.isin(observed, [0, 1]).all():
        raise ValueError("Callable training genotypes are required")
    order = np.argsort(-observed.mean(axis=0), kind="stable")
    pairs = [
        (int(np.sum(observed[:, i] * observed[:, j])), i, j)
        for i in range(12)
        for j in range(i + 1, 12)
    ]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = [(), (int(order[0]),), (int(order[1]),), (pairs[0][1], pairs[0][2])]
    for allowed in [{"KMT2D", "KMT2C", "ARID1A", "SMARCA4", "SETD2"}, {"ATM"}]:
        candidates = [
            int(j) for j in order if genes[j] in allowed and (int(j),) not in chosen
        ]
        if not candidates:
            candidates = [int(j) for j in order if (int(j),) not in chosen]
        chosen.append((candidates[0],))
    return chosen, observed


def export_path_data(artifact, training, genes, *, n_paths=50000, seed=20261111):
    """Return numeric Fig.5 data; Q10/Q90 describe paths, not parameter CIs."""
    features = ["age", "background_nMut", "background_VAF_mean"]
    median = training[features].median().to_frame().T
    uniforms = np.random.default_rng(seed).random((n_paths, 13))
    profiles = []
    positions = []
    next_event = []
    for quantile in [0.25, 0.5, 0.75]:
        raw = median.copy()
        raw["background_nMut"] = training.background_nMut.quantile(
            quantile, interpolation="linear"
        )
        theta = project_transition(artifact, raw)[0]
        pos = path_positions(theta, uniforms)
        rate = primary_rates(theta, np.zeros(13))
        probability = rate / rate.sum()
        profiles.append(
            dict(
                quantile=quantile,
                age=float(raw.age.iloc[0]),
                background_nMut=float(raw.background_nMut.iloc[0]),
                background_VAF_mean=float(raw.background_VAF_mean.iloc[0]),
            )
        )
        for j, event in enumerate([*genes, "Seeding"]):
            positions.append(
                dict(
                    quantile=quantile,
                    event=event,
                    mean_position=float(pos[:, j].mean()),
                    q10=float(np.quantile(pos[:, j], 0.1)),
                    q90=float(np.quantile(pos[:, j], 0.9)),
                    MC_SE=float(pos[:, j].std(ddof=1) / np.sqrt(n_paths)),
                )
            )
            next_event.append(
                dict(
                    state="WT",
                    quantile=quantile,
                    event=event,
                    probability=float(probability[j]),
                )
            )
    theta = project_transition(artifact, median)[0]
    chosen, observed = representative_states(training, genes)
    for active in chosen[1:]:
        state = np.zeros(13)
        state[list(active)] = 1.0
        rate = primary_rates(theta, state)
        for j, event in enumerate([*genes, "Seeding"]):
            next_event.append(
                dict(
                    state="+".join(genes[k] for k in active),
                    quantile=0.5,
                    event=event,
                    probability=float(rate[j] / rate.sum()),
                )
            )
    nodes, edges, mass = branch_tree(theta, genes)
    pca = PCA(n_components=2, svd_solver="full").fit(observed)
    states = ((np.arange(4096)[:, None] >> np.arange(12)) & 1).astype(float)
    coordinates = pca.transform(states)
    rate = np.array([primary_rates(theta, np.r_[state, 0]) for state in states])
    probability = rate / rate.sum(axis=1, keepdims=True)
    displacement = probability[:, :12] @ pca.components_.T
    landscape = pd.DataFrame(
        dict(
            state_code=np.arange(4096),
            n_mutations=states.sum(axis=1).astype(int),
            PC1=coordinates[:, 0],
            PC2=coordinates[:, 1],
            dPC1=displacement[:, 0],
            dPC2=displacement[:, 1],
            seeding_probability=probability[:, 12],
        )
    )
    return (
        dict(
            profiles=pd.DataFrame(profiles),
            path_positions=pd.DataFrame(positions),
            next_event=pd.DataFrame(next_event),
            branch_nodes=nodes,
            branch_edges=edges,
            landscape=landscape,
        ),
        mass,
    )

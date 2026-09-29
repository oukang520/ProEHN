"""Patient specific transition representation; projections are separate summaries."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class PatientTransitionRepresentation:
    theta: np.ndarray
    log_diagnosis_primary: np.ndarray
    log_diagnosis_metastasis: np.ndarray
    accessible_events: tuple[tuple[str, str], ...]
    conditional_probabilities: np.ndarray

    def integrated_scores(self, progression_propensity):
        if not np.isfinite(progression_propensity) or not 0 <= progression_propensity <= 1:
            raise ValueError('Progression propensity must lie in [0,1]')
        return self.conditional_probabilities * progression_propensity


def accessible_transition_log_rates(theta, genes, primary, metastasis, seeded):
    """Same compartment transition definition as the metMHN CTMC generator.

    Prior to seeding the MT copy is latent/inherited, not an independent event.
    Seeding copies the primary genotype. After seeding the two compartments
    have separate accessible mutation events; only MT hazards use seed effects.
    """
    pt, mt = np.asarray(primary), np.asarray(metastasis)
    n = len(genes)
    theta = np.asarray(theta, float)
    if theta.shape != (n+1, n+1) or not np.isfinite(theta).all():
        raise ValueError('Invalid transition parameters')
    if pt.shape != (n,) or mt.shape != (n,) or not np.isin(pt, [0, 1]).all() or not np.isin(mt, [0, 1]).all() or seeded not in (0, 1):
        raise ValueError('Known binary observation-time genotype and seeding required')
    if not seeded and mt.any():
        raise ValueError('Observed MT conflicts with absence of seeding')
    rows = []
    active_p = np.flatnonzero(pt)
    for j, gene in enumerate(genes):
        if not pt[j]:
            rows.append(dict(compartment='primary', event=gene, log_rate=float(theta[j, j]+theta[j, active_p].sum())))
    if not seeded:
        rows.append(dict(compartment='primary', event='Metastatic seeding', log_rate=float(theta[n, n]+theta[n, active_p].sum())))
    else:
        active_m = np.r_[np.flatnonzero(mt), n]
        for j, gene in enumerate(genes):
            if not mt[j]:
                rows.append(dict(compartment='metastasis', event=gene, log_rate=float(theta[j, j]+theta[j, active_m].sum())))
    return rows


def patient_transition_representation(model, params, features_with_bias, gene_names, primary, metastasis, seeded):
    theta, dp, dm = model.compute_patient_params(*model.parse_params(params), features_with_bias,
                                               model.log_rate_clip_min, model.log_rate_clip_max)
    rows = accessible_transition_log_rates(theta, gene_names, primary, metastasis, seeded)
    probabilities = softmax([r['log_rate'] for r in rows]) if rows else np.empty(0)
    return PatientTransitionRepresentation(np.asarray(theta), np.asarray(dp), np.asarray(dm),
        tuple((r['compartment'], r['event']) for r in rows), probabilities)


def model_implied_transition_path(theta, genes, primary, metastasis, seeded, *, max_steps=3):
    """Greedy model-implied path at fixed covariates; no time or future-risk claim."""
    pt, mt, seed = list(primary), list(metastasis), int(seeded)
    path = []
    for _ in range(max_steps):
        rows = accessible_transition_log_rates(theta, genes, pt, mt, seed)
        if not rows:
            break
        probabilities = softmax([r['log_rate'] for r in rows])
        index = int(np.argmax(probabilities))
        row = rows[index]
        path.append({**row, 'conditional_probability': float(probabilities[index])})
        if row['event'] == 'Metastatic seeding':
            seed, mt = 1, pt.copy()
        else:
            (pt if row['compartment'] == 'primary' else mt)[genes.index(row['event'])] = 1
    return path

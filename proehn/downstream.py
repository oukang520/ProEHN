"""Representation, projection, visualization and clustering are distinct objects.

These APIs do not run analyses at import. Estimators and selection/stability
protocols must be explicitly supplied; t-SNE outputs are never cluster inputs.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RepresentationMatrix:
    patient_ids: tuple[str,...]
    values: np.ndarray
    kind: str
    event_schema_version: str

    def validate(self):
        x=np.asarray(self.values,float)
        if self.kind not in ('vectorized_theta','conditional_transition') or not self.event_schema_version:
            raise ValueError('Clustering requires theta or conditional transition representation, never t-SNE coordinates')
        if x.ndim!=2 or len(x)!=len(self.patient_ids) or len(set(self.patient_ids))!=len(x) or not np.isfinite(x).all() or not len(x):
            raise ValueError('Aligned independent patients and finite representation required')
        if self.kind=='conditional_transition' and (np.any(x<0) or not np.allclose(x.sum(axis=1),1)):
            raise ValueError('Conditional event support must be aligned and normalized, including any terminal atom')
        return self


@dataclass(frozen=True)
class Projection:
    patient_ids: tuple[str,...]
    coordinates: np.ndarray
    method: str


@dataclass
class FittedPCA:
    estimator: object
    training_patient_ids: tuple[str,...]
    representation_kind: str
    event_schema_version: str

    def transform(self, representation):
        representation.validate()
        if (representation.kind,representation.event_schema_version)!=(self.representation_kind,self.event_schema_version):
            raise ValueError('Projection representation contract mismatch')
        return Projection(representation.patient_ids,np.asarray(self.estimator.transform(representation.values)),'PCA')


def fit_pca(training_representation, estimator):
    training_representation.validate()
    estimator.fit(np.array(training_representation.values,copy=True))
    return FittedPCA(estimator,training_representation.patient_ids,training_representation.kind,training_representation.event_schema_version)


def tsne_visualization(representation, estimator):
    representation.validate()
    return Projection(representation.patient_ids,np.asarray(estimator.fit_transform(np.array(representation.values,copy=True))),'t-SNE visualization only')


@dataclass
class FittedClusters:
    estimator: object
    representation_kind: str
    event_schema_version: str
    training_patient_ids: tuple[str,...]

    def assign(self, representation):
        if not isinstance(representation,RepresentationMatrix):
            raise ValueError('Cluster assignment cannot use PCA/t-SNE coordinates')
        representation.validate()
        if (representation.kind,representation.event_schema_version)!=(self.representation_kind,self.event_schema_version):
            raise ValueError('Clustering representation contract mismatch')
        labels=np.asarray(self.estimator.predict(representation.values))
        if labels.shape!=(len(representation.patient_ids),):
            raise ValueError('Cluster labels must align with patients')
        return labels


def fit_clusters(training_representation, estimator):
    if not isinstance(training_representation,RepresentationMatrix):
        raise ValueError('Cluster directly on theta or conditional transitions, not t-SNE')
    training_representation.validate()
    estimator.fit(np.array(training_representation.values,copy=True))
    return FittedClusters(estimator,training_representation.kind,training_representation.event_schema_version,training_representation.patient_ids)


def select_cluster_number(training_representation, candidates, fit_and_score):
    """Prespecified training-only unsupervised criterion, larger is better."""
    training_representation.validate(); candidates=tuple(candidates)
    if not candidates or any(k<2 or k>=len(training_representation.patient_ids) for k in candidates):
        raise ValueError('Freeze admissible cluster-number candidates')
    scores=[float(fit_and_score(np.array(training_representation.values,copy=True),k)) for k in candidates]
    if not np.isfinite(scores).all():
        raise ValueError('Invalid training-only clustering criterion')
    return candidates[int(np.argmax(scores))]


def bootstrap_cluster_stability(training_representation, *, n_clusters, repetitions, random_seed, fit_and_assign):
    """Bootstrap independent training patients; compare all-training assignments.

    fit_and_assign(bootstrap_values, reference_values, k) must fit only the first
    matrix. ARI avoids arbitrary label-number alignment. No outcome argument.
    """
    from sklearn.metrics import adjusted_rand_score
    training_representation.validate(); x=np.asarray(training_representation.values)
    if repetitions<2 or not 2<=n_clusters<len(x):
        raise ValueError('Freeze valid bootstrap repetitions and cluster count')
    rng=np.random.default_rng(random_seed); assignments=[]
    for _ in range(repetitions):
        labels=np.asarray(fit_and_assign(x[rng.integers(0,len(x),len(x))].copy(),x.copy(),n_clusters))
        if labels.shape!=(len(x),):
            raise ValueError('Stability assignments must share reference-patient alignment')
        assignments.append(labels)
    return np.array([adjusted_rand_score(assignments[i],assignments[j]) for i in range(repetitions) for j in range(i)])

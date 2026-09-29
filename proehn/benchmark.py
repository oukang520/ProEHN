"""Formal target-event recovery through the shared ProEHN components.

Cross-sectional target recovery tests ranking of masked observed events. It is
not validation of longitudinal next-event probability. No surrogate default.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .features import TargetMaskingProtocol, build_leave_target_out_features, FrozenCovariateProtocol, TopologyCovariateSchema
from .preprocessing import TopologyTrainingPreprocessor, build_topology_training_data, GenePair
from .training import fit_topology_model, fit_kinetic_model
from .topology import ProEHNTopologyModel
from .kinetic import ProEHNKineticGatekeeper
from .labels import ProgressionLabelSchema, build_progression_label, require_frozen_label_protocol
from .representation import patient_transition_representation_observed
from .observations import observation_types


@dataclass(frozen=True)
class TargetRecoverySpec:
    target_gene: str
    compartment: str
    masking_protocols: tuple[TargetMaskingProtocol, ...]
    # Frozen event definitions (indicator name -> complete raw variant loci).
    event_loci: tuple[tuple[str, tuple[str, ...]], ...]
    patient_id_column: str = 'patient_id'
    stop_label_column: str = 'Patient_Label'

    def __post_init__(self):
        if self.compartment not in ('primary', 'metastasis') or not self.masking_protocols:
            raise ValueError('Target compartment and raw masking provenance required')
        columns = dict(self.event_loci)
        if len(columns) != len(self.event_loci) or not all(self.target_column(c) in columns for c in ('primary', 'metastasis')):
            raise ValueError('Target requires explicit event definitions for both compartments')
        raw = {col for p in self.masking_protocols for col in p.mutation_columns+p.cna_columns}
        target = {col for p in self.masking_protocols for col in p.target_mutation_columns+p.target_cna_columns}
        if any(not loci or not set(loci) <= raw for _, loci in self.event_loci):
            raise ValueError('Every event must have complete raw locus provenance')
        expected_target = set(columns[self.target_column('primary')]) | set(columns[self.target_column('metastasis')])
        if target != expected_target:
            raise ValueError('Masking must cover exactly the target event loci in both compartments')
        for name, loci in self.event_loci:
            if name not in (self.target_column('primary'), self.target_column('metastasis')) and set(loci) & target:
                raise ValueError('Target locus cannot remain in another event definition')

    def target_column(self, compartment=None):
        return f"{'P' if (compartment or self.compartment) == 'primary' else 'M'}.{self.target_gene} (M)"

    def features(self, raw):
        # Rebuild from closed provenance; never copy arbitrary dataframe columns.
        out = pd.DataFrame(index=raw.index)
        metadata = {}
        kinds = observation_types(raw)
        for protocol in self.masking_protocols:
            observed = np.isin(kinds, [0, 1, 3, 4] if protocol.compartment == 'Primary' else [2, 3])
            part = build_leave_target_out_features(raw.loc[observed], protocol).reindex(raw.index)
            metadata.update(part.attrs['feature_metadata'])
            for name in part:
                if name in out and not out[name].equals(part[name]):
                    raise ValueError('Conflicting compartment feature provenance')
                out[name] = part[name]
        target_columns = {self.target_column('primary'), self.target_column('metastasis')}
        # Keep only named baseline/summaries and event indicators, not raw loci.
        allowed = {c for p in self.masking_protocols for c in p.baseline_columns}
        allowed |= {f'{f}_{p.compartment}' for p in self.masking_protocols for f in ('nMut', 'VAF_mean', 'CNA')}
        out = out[[c for c in out if c in allowed]]
        for name, loci in self.event_loci:
            present = kinds != 2 if name.startswith('P.') else np.isin(kinds, [2, 3])
            out[name] = 0 if name in target_columns else raw[list(loci)].max(axis=1, skipna=False).where(present)
        out['observation_type'] = kinds
        if 'observed_seeding' in raw:
            seed = raw['observed_seeding'].to_numpy()
            for i, kind in enumerate(kinds):
                if not pd.isna(seed[i]) and (seed[i] not in (0, 1) or (kind != 4 and seed[i] != (0 if kind == 0 else 1))):
                    raise ValueError('Observation type conflicts with independent seeding annotation')
                if kind == 4 and not pd.isna(seed[i]):
                    raise ValueError('Known seeding requires the corresponding explicit observation type')
        for name in ('diag_order', 'seeding_at_first_observation', 'joint_snapshot'):
            if name in raw:
                out[name] = raw[name].to_numpy()
        out.attrs['feature_metadata'] = metadata
        return out

    def training_frame(self, raw):
        out = self.features(raw)
        for name, loci in self.event_loci:
            out[name] = raw[list(loci)].max(axis=1, skipna=False)
        for name in ('observation_type', 'type', 'diag_order', 'seeding_at_first_observation'):
            if name in raw:
                out[name] = raw[name].to_numpy()
        return out


@dataclass
class FormalProEHNPredictor:
    topology: ProEHNTopologyModel
    topology_params: np.ndarray
    topology_preprocessor: TopologyTrainingPreprocessor
    kinetic: ProEHNKineticGatekeeper
    spec: TargetRecoverySpec

    def __post_init__(self):
        if not isinstance(self.topology, ProEHNTopologyModel) or not isinstance(self.kinetic, ProEHNKineticGatekeeper) or not self.kinetic.ready:
            raise TypeError('Formal fitted topology and kinetic components required')

    def predict(self, features):
        """Return [P(Go), conditional target score, integrated target score]."""
        for compartment in ('primary', 'metastasis'):
            if not (features[self.spec.target_column(compartment)] == 0).all():
                raise ValueError('Target event must be masked before prediction')
        z = self.topology_preprocessor.transform(features)
        pairs = self.topology_preprocessor.gene_pairs
        genes = [g.name for g in pairs]
        if self.spec.target_gene not in genes:
            raise ValueError('Target absent from fitted topology')
        out = []
        for i, (_, row) in enumerate(features.iterrows()):
            go = self.kinetic.predict_go_probability(row.to_dict())
            kind = int(row['observation_type'])
            representation = patient_transition_representation_observed(
                self.topology, self.topology_params, np.r_[1., z[i]], genes,
                [row[g.primary] for g in pairs] if kind != 2 else None,
                [row[g.metastasis] for g in pairs] if kind in (2, 3) else None, kind,
                diagnosis_order=row.get('diag_order'), first_seeding=row.get('seeding_at_first_observation', -1),
                joint_snapshot=bool(row.get('joint_snapshot', False)))
            key = (self.spec.compartment, self.spec.target_gene)
            if key not in representation.accessible_events:
                raise ValueError('Target must be accessible for every evaluation patient')
            j = representation.accessible_events.index(key)
            out.append([go, representation.conditional_probabilities[j], representation.integrated_scores(go)[j]])
        return np.asarray(out).reshape(len(features), 3)


@dataclass
class FormalProEHNTrainer:
    spec: TargetRecoverySpec
    label_schema: ProgressionLabelSchema
    top_n_genes: int = 20
    kinetic_config: dict | None = None
    topology_config: dict | None = None
    covariate_protocol: FrozenCovariateProtocol | None = None

    def fit(self, training, validation):
        require_frozen_label_protocol(self.label_schema)
        ids = self.spec.patient_id_column
        if training[ids].isna().any() or validation[ids].isna().any() or set(training[ids]) & set(validation[ids]):
            raise ValueError('Training/validation patients must be disjoint and known')
        train_features = self.spec.features(training)
        val_features = self.spec.features(validation)
        if self.covariate_protocol is None or self.covariate_protocol.cohort != self.label_schema.cohort:
            raise ValueError('Matching frozen cohort covariate protocol required')
        self.covariate_protocol.validate(train_features).validate(val_features)
        for raw, features in ((training, train_features), (validation, val_features)):
            features[ids] = raw[ids].to_numpy()
            features[self.spec.stop_label_column] = build_progression_label(raw, self.label_schema)
        gate = fit_kinetic_model(train_features, val_features, patient_id_column=ids,
                               label_column=self.spec.stop_label_column, model_config=dict(self.kinetic_config or {}, covariates=self.covariate_protocol.kinetic))
        topology_data = self.spec.training_frame(training)
        prep = TopologyTrainingPreprocessor(self.top_n_genes, schema=TopologyCovariateSchema(self.covariate_protocol.topology)).fit(topology_data)
        if self.spec.target_gene not in [g.name for g in prep.gene_pairs]:
            # A prespecified target is always retained; context ranking uses train only.
            prep.gene_pairs = prep.gene_pairs[:max(0, self.top_n_genes-1)] + [GenePair(
                self.spec.target_gene, self.spec.target_column('primary'), self.spec.target_column('metastasis'))]
        buckets, ne, nf, ns, _, _ = build_topology_training_data(topology_data, preprocessor=prep)
        model, params, _ = fit_topology_model(buckets, ne, nf, ns, **dict(self.topology_config or {}))
        return FormalProEHNPredictor(model, params, prep, gate, self.spec)

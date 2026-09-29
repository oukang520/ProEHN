"""Formal target-event recovery through the shared ProEHN components.

Cross-sectional target recovery tests ranking of masked observed events. It is
not validation of longitudinal next-event probability. No surrogate default.
"""
from dataclasses import dataclass, replace, field, asdict
import numpy as np
import pandas as pd

from .features import TargetMaskingProtocol, build_leave_target_out_features, FrozenCovariateProtocol, TopologyCovariateSchema
from .preprocessing import TopologyTrainingPreprocessor, build_topology_training_data, GenePair
from .training import fit_topology_model, fit_kinetic_model
from .topology import ProEHNTopologyModel
from .kinetic import ProEHNKineticGatekeeper
from .labels import ProgressionLabelSchema, build_progression_label, require_frozen_label_protocol, label_provenance
from .representation import posterior_transition_representation, PosteriorInferenceProtocol
from .observations import observation_types
from .events import EventSelectionProtocol, select_event_schema


@dataclass(frozen=True)
class TargetRecoverySpec:
    target_gene: str
    compartment: str
    masking_protocols: tuple[TargetMaskingProtocol, ...]
    # Frozen event definitions (indicator name -> complete raw variant loci).
    event_loci: tuple[tuple[str, tuple[str, ...]], ...]
    patient_id_column: str = 'patient_id'
    stop_label_column: str = 'Patient_Label'
    event_schema_version: str = 'explicit-loci-v1'
    event_selection_protocol: EventSelectionProtocol | None = None

    def __post_init__(self):
        if self.event_selection_protocol is not None and self.event_schema_version != self.event_selection_protocol.schema_version:
            raise ValueError('Event schema version differs from shared selection protocol')
        if not self.event_schema_version:
            raise ValueError('Frozen event schema version required')
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
            required = protocol.mutation_columns + protocol.vaf_columns + protocol.cna_columns + protocol.baseline_columns + protocol.cna_segment_length_columns
            source = raw.loc[observed] if observed.any() else pd.DataFrame(columns=required, index=raw.index[:0])
            part = build_leave_target_out_features(source, protocol).reindex(raw.index)
            for baseline in protocol.baseline_columns:
                part[baseline] = raw[baseline]
            metadata.update(part.attrs['feature_metadata'])
            for name in part:
                if name in out and not out[name].equals(part[name]):
                    raise ValueError('Conflicting compartment feature provenance')
                out[name] = part[name]
        target_columns = {self.target_column('primary'), self.target_column('metastasis')}
        # Keep only named baseline/summaries and event indicators, not raw loci.
        allowed = {c for p in self.masking_protocols for c in p.baseline_columns}
        allowed |= {f'{f}_{p.compartment}' for p in self.masking_protocols for f in ('nMut', 'VAF_mean', 'CNA', 'TMB', 'FGA', 'CNA_adjusted')}
        out = out[[c for c in out if c in allowed]]
        for name, loci in self.event_loci:
            present = kinds != 2 if name.startswith('P.') else np.isin(kinds, [2, 3])
            out[name] = 0 if name in target_columns else self._event_calls(raw, loci, present)
        out['observation_type'] = kinds
        if 'observed_seeding' in raw:
            seed = raw['observed_seeding'].to_numpy()
            for i, kind in enumerate(kinds):
                if not pd.isna(seed[i]) and (seed[i] not in (0, 1) or (kind != 4 and seed[i] != (0 if kind == 0 else 1))):
                    raise ValueError('Observation type conflicts with independent seeding annotation')
                if kind == 4 and not pd.isna(seed[i]):
                    raise ValueError('Known seeding requires the corresponding explicit observation type')
        # Missing history annotations mean unknown, never a seeded/synchronous plug-in.
        defaults = {'diag_order': 0, 'seeding_at_first_observation': -1, 'joint_snapshot': False}
        allowed_annotations = {'diag_order': (0,1,2), 'seeding_at_first_observation': (-1,0,1), 'joint_snapshot': (False,True)}
        for name, default in defaults.items():
            values = raw[name].fillna(default) if name in raw else pd.Series(default,index=raw.index)
            if not np.isin(values.to_numpy(),allowed_annotations[name]).all():
                raise ValueError('Invalid observation history annotation')
            out[name] = values.to_numpy()
        out.attrs['feature_metadata'] = metadata
        return out

    @staticmethod
    def _event_calls(raw, loci, observed):
        calls = pd.Series(np.nan, index=raw.index)
        if np.any(observed):
            values = raw.loc[observed, list(loci)]
            if not np.isin(values.to_numpy(), [0, 1]).all():
                raise ValueError('Observed event loci require complete binary calls')
            calls.loc[observed] = values.max(axis=1, skipna=False)
        return calls

    def outer_folds(self, raw, label_schema, policy):
        """One frozen target-specific split; ineligible labels are an explicit error."""
        from .evaluation import FormalFoldProtocol, TargetNotEvaluable
        require_frozen_label_protocol(label_schema)
        kinds = observation_types(raw)
        eligible = kinds != 2 if self.compartment == 'primary' else np.isin(kinds, [2, 3])
        if not eligible.all():
            raise TargetNotEvaluable('Target compartment is unobserved: freeze the eligible patient set before generating folds')
        target = raw[list(dict(self.event_loci)[self.target_column()])].max(axis=1, skipna=False)
        return FormalFoldProtocol.create(raw[self.patient_id_column], target, build_progression_label(raw, label_schema),
            target=self.target_column(), policy=policy)

    def training_frame(self, raw):
        out = self.features(raw)
        kinds = observation_types(raw)
        for name, loci in self.event_loci:
            observed = kinds != 2 if name.startswith('P.') else np.isin(kinds, [2, 3])
            out[name] = self._event_calls(raw, loci, observed)
        for name in ('observation_type', 'type'):
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
    inference_protocol: PosteriorInferenceProtocol = field(default_factory=PosteriorInferenceProtocol)

    def __post_init__(self):
        if not isinstance(self.topology, ProEHNTopologyModel) or not isinstance(self.kinetic, ProEHNKineticGatekeeper) or not self.kinetic.ready:
            raise TypeError('Formal fitted topology and kinetic components required')

    def save_topology(self, path):
        from .artifacts import ScientificArtifactMetadata, save_topology_artifact
        protocol = dict(getattr(self.kinetic, 'scientific_training_protocol', {}))
        if not protocol or not hasattr(self, 'selection_metadata'):
            raise ValueError('Formal artifact requires completed training/selection provenance')
        protocol['selection'] = self.selection_metadata
        protocol['posterior_inference'] = asdict(self.inference_protocol)
        prep = self.topology_preprocessor.covariates
        model = self.topology
        meta = ScientificArtifactMetadata(tuple(g.name for g in self.topology_preprocessor.gene_pairs),
            tuple(prep.feature_names), tuple(prep.columns), prep.metadata(), model.log_rate_clip_min,
            model.log_rate_clip_max, {k:getattr(model,k) for k in ('regularization_strength','l1_ratio','l2_floor')},
            self.selected_event_schema.schema_version, protocol['training_feature_provenance_version'], training_protocol=protocol)
        save_topology_artifact(path, self.topology_params, meta)

    def predict(self, features):
        return self._predict(features)

    def _prediction_rows(self, features, *, progression_override=None):
        for compartment in ('primary', 'metastasis'):
            if not (features[self.spec.target_column(compartment)] == 0).all():
                raise ValueError('Target event must be masked before prediction')
        z = self.topology_preprocessor.transform(features)
        pairs = self.topology_preprocessor.gene_pairs
        genes = [g.name for g in pairs]
        if self.spec.target_gene not in genes:
            raise ValueError('Target absent from fitted topology')
        for i, (_, row) in enumerate(features.iterrows()):
            go = self.kinetic.predict_go_probability(row.to_dict()) if progression_override is None else progression_override
            kind = int(row['observation_type'])
            representation = posterior_transition_representation(
                self.topology, self.topology_params, np.r_[1., z[i]], genes,
                [row[g.primary] for g in pairs] if kind != 2 else None,
                [row[g.metastasis] for g in pairs] if kind in (2, 3) else None, kind,
                diagnosis_order=row.get('diag_order'), first_seeding=row.get('seeding_at_first_observation', -1),
                joint_snapshot=bool(row.get('joint_snapshot', False)), **asdict(self.inference_protocol))
            yield go, representation

    def _predict(self, features, *, progression_override=None):
        out=[]
        for go, representation in self._prediction_rows(features, progression_override=progression_override):
            key=(self.spec.compartment,self.spec.target_gene)
            probability=dict(zip(representation.accessible_events,representation.conditional_probabilities)).get(key,0.)
            out.append([go,probability,go*probability])
        return np.asarray(out).reshape(len(features),3)

    def predict_candidate_scores(self, features, *, progression_override=None):
        events=self.selected_event_schema.target_candidate_events(self.spec.compartment)
        rows=[]
        for go, representation in self._prediction_rows(features, progression_override=progression_override):
            conditional=dict(zip(representation.accessible_events,representation.conditional_probabilities))
            rows.append([go*conditional.get(event,0.) for event in events])
        return np.asarray(rows).reshape(len(features),len(events))


@dataclass
class FormalProEHNTrainer:
    requires_formal_folds = True
    spec: TargetRecoverySpec
    label_schema: ProgressionLabelSchema
    top_n_genes: int = 20
    kinetic_config: dict | None = None
    topology_config: dict | None = None
    covariate_protocol: FrozenCovariateProtocol | None = None
    configuration_source: str = 'prespecified'
    topology_candidates: tuple[dict, ...] = ()
    inference_protocol: PosteriorInferenceProtocol = field(default_factory=PosteriorInferenceProtocol)

    @classmethod
    def from_scientific_config(cls, spec, config):
        """Explicit protocol config: pending cohort templates intentionally cannot fit."""
        from .features import FrozenCovariateProtocol
        from .scientific_config import validate_model_parameters
        label = dict(config.get('label_protocol', {}))
        cov = dict(config.get('covariate_protocol', {}))
        if not label.get('frozen') or not cov.get('frozen'):
            raise ValueError('REQUIRES_PROTOCOL_FREEZE_BEFORE_RERUN')
        cohort = config['cohort']
        label.pop('source_variables', None)
        label['response_mapping'] = tuple(tuple(x) for x in (label.get('response_mapping') or ()))
        schema = ProgressionLabelSchema(cohort=cohort, **label)
        cov['topology'], cov['kinetic'] = tuple(cov['topology']), tuple(cov['kinetic'])
        protocol = FrozenCovariateProtocol(cohort=cohort, **cov).validate()
        kinetic = dict(config.get('kinetic', {}))
        topology = dict(config.get('topology', {}))
        validate_model_parameters(kinetic, topology, config['configuration_source'], config.get('topology_candidates', ()))
        event_config = dict(config['event_selection'])
        event_config['frozen_core_drivers'] = tuple(event_config['frozen_core_drivers'])
        event_protocol = EventSelectionProtocol(**event_config)
        spec = replace(spec, event_selection_protocol=event_protocol, event_schema_version=event_protocol.schema_version)
        return cls(spec, schema, int(topology.pop('top_n_genes', 20)), kinetic, topology, protocol,
            config['configuration_source'], tuple(config.get('topology_candidates', ())), PosteriorInferenceProtocol(**config['posterior_inference']))

    def fit(self, training, validation):
        require_frozen_label_protocol(self.label_schema)
        from .scientific_config import validate_model_parameters
        validate_model_parameters(dict(self.kinetic_config or {}), dict(self.topology_config or {}, top_n_genes=self.top_n_genes), self.configuration_source, self.topology_candidates)
        if self.spec.event_selection_protocol is None or self.spec.event_selection_protocol.top_n_events != self.top_n_genes:
            raise ValueError('Formal trainer requires the shared event selection protocol')
        if self.configuration_source == 'prespecified':
            if self.topology_candidates:
                raise ValueError('Prespecified mode cannot silently select candidates')
            fitted = self._fit_prespecified(training, validation)
            fitted.selection_metadata = dict(configuration_source='prespecified', topology_config=dict(self.topology_config or {}))
            return fitted
        if self.configuration_source != 'inner_validation' or not 1 <= len(self.topology_candidates) <= 8:
            raise ValueError('Declare prespecified configuration or 1–8 frozen inner-validation candidates')
        if any(not set(c) <= {'regularization_strength', 'l1_ratio'} for c in self.topology_candidates):
            raise ValueError('Selection supports only necessary topology regularization candidates')
        from .evaluation import select_on_inner_validation
        fitted_candidates = {}
        def fit_and_score(train, val, candidate):
            config = dict(self.topology_config or {}); config.update(candidate)
            fitted = replace(self, configuration_source='prespecified', topology_candidates=(), topology_config=config)._fit_prespecified(train, val)
            # Training-fitted gene vocabulary/scaler, validation observation likelihood.
            data = self.spec.training_frame(val)
            buckets, _, _, n, _, _ = build_topology_training_data(data, preprocessor=fitted.topology_preprocessor)
            import jax.numpy as jnp
            loss = sum(float(fitted.topology.bucket_loss(jnp.asarray(fitted.topology_params), kind, np_, nm_, jnp.asarray(genes), jnp.asarray(z)))
                for kind, np_, nm_, genes, z in buckets) / n
            fitted_candidates[tuple(sorted(candidate.items()))] = fitted
            return loss
        selected = select_on_inner_validation(training, validation, self.topology_candidates, fit_and_score,
            patient_id_column=self.spec.patient_id_column)
        fitted = fitted_candidates[tuple(sorted(selected.items()))]
        fitted.selection_metadata = dict(configuration_source='inner_validation', criterion='mean_validation_topology_nll',
            candidates=[dict(c) for c in self.topology_candidates], selected=dict(selected))
        return fitted

    def _fit_prespecified(self, training, validation):
        require_frozen_label_protocol(self.label_schema)
        ids = self.spec.patient_id_column
        if training[ids].isna().any() or validation[ids].isna().any() or set(training[ids]) & set(validation[ids]):
            raise ValueError('Training/validation patients must be disjoint and known')
        train_features = self.spec.features(training)
        val_features = self.spec.features(validation)
        if self.covariate_protocol is None or self.covariate_protocol.cohort != self.label_schema.cohort:
            raise ValueError('Matching frozen cohort covariate protocol required')
        self.covariate_protocol.validate(train_features).validate(val_features)
        used=set(self.covariate_protocol.topology)|set(self.covariate_protocol.kinetic)
        for masking in self.spec.masking_protocols:
            burden_fields={f'{name}_{masking.compartment}' for name in ('nMut','TMB','CNA','FGA','CNA_adjusted')}
            if used&burden_fields and (masking.burden_protocol is None or not masking.burden_protocol.frozen):
                raise ValueError('Selected genomic burdens require a frozen assay/coverage protocol')
        for raw, features in ((training, train_features), (validation, val_features)):
            features[ids] = raw[ids].to_numpy()
            features[self.spec.stop_label_column] = build_progression_label(raw, self.label_schema)
            if not {0, 1} <= set(features[self.spec.stop_label_column]):
                raise ValueError('Training and validation each require observed Stop and Go patients')
            features.attrs['label_provenance'] = {self.spec.stop_label_column: label_provenance(self.label_schema)}
        topology_data = self.spec.training_frame(training)
        prep = TopologyTrainingPreprocessor(self.top_n_genes, schema=TopologyCovariateSchema(self.covariate_protocol.topology)).fit(topology_data)
        selected_schema = select_event_schema(training, self.spec)
        prep.gene_pairs = [GenePair(g, f'P.{g} (M)', f'M.{g} (M)') for g in selected_schema.gene_names]
        gate = fit_kinetic_model(train_features, val_features, patient_id_column=ids,
                               label_column=self.spec.stop_label_column, label_schema=self.label_schema, model_config=dict(self.kinetic_config or {}, covariates=self.covariate_protocol.kinetic))
        buckets, ne, nf, ns, _, _ = build_topology_training_data(topology_data, preprocessor=prep)
        model, params, _ = fit_topology_model(buckets, ne, nf, ns, **dict(self.topology_config or {}))
        from dataclasses import asdict
        gate.scientific_training_protocol = dict(label_protocol=label_provenance(self.label_schema),
            covariates=asdict(self.covariate_protocol), event_schema_version=selected_schema.schema_version,
            event_loci=[(name, list(loci)) for name, loci in self.spec.event_loci],
            training_feature_provenance_version='leave-target-out-raw-loci-v1',
            event_selection=asdict(self.spec.event_selection_protocol),
            selected_gene_names=list(selected_schema.gene_names), posterior_inference=asdict(self.inference_protocol),
            target_masking_protocols=[asdict(p) for p in self.spec.masking_protocols])
        result = FormalProEHNPredictor(model, params, prep, gate, self.spec, self.inference_protocol)
        result.selected_event_schema = selected_schema
        result.training_patient_ids = tuple(map(str, training[ids]))
        result.validation_patient_ids = tuple(map(str, validation[ids]))
        return result

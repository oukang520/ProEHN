"""One training-only event selection contract shared by every comparison method."""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class EventSelectionProtocol:
    schema_id: str
    version: str
    external_driver_source: str
    frozen_core_drivers: tuple[str,...]
    recurrent_event_rule: str
    minimum_prevalence: float
    top_n_events: int
    training_only_frequency_selection: bool
    target_retention_rule: str

    def __post_init__(self):
        if not self.schema_id or not self.version or not self.external_driver_source:
            raise ValueError('Versioned event selection and explicit external driver source required')
        if self.recurrent_event_rule not in ('patient_prevalence','none') or not 0<=self.minimum_prevalence<=1 or self.top_n_events<1:
            raise ValueError('Unsupported frozen recurrent-event rule')
        if not self.training_only_frequency_selection or self.target_retention_rule!='always_include':
            raise ValueError('Formal target recovery requires training-only selection and target retention')
        if len(set(self.frozen_core_drivers))!=len(self.frozen_core_drivers):
            raise ValueError('Duplicate core drivers')

    @property
    def schema_version(self):
        return f'{self.schema_id}:{self.version}'


@dataclass(frozen=True)
class SelectedEventSchema:
    gene_names: tuple[str,...]
    selection_protocol_version: str
    schema_version: str

    def target_candidate_events(self, compartment):
        if compartment not in ('primary','metastasis'): raise ValueError('Unknown target compartment')
        return tuple((compartment,gene) for gene in self.gene_names)

    @property
    def candidate_events(self):
        return tuple((compartment,gene) for compartment in ('primary','metastasis') for gene in self.gene_names)


def select_event_schema(training, spec):
    """Independent-patient prevalence; no validation/test frame or outcome input."""
    from .preprocessing import find_gene_pairs
    protocol=spec.event_selection_protocol
    if protocol is None:
        raise ValueError('Shared EventSelectionProtocol required')
    frame=spec.training_frame(training)
    ids=training[spec.patient_id_column]
    if ids.isna().any() or not len(training):
        raise ValueError('Training patient identity required')
    pairs={g.name:g for g in find_gene_pairs(frame.columns)}
    required=list(dict.fromkeys((*protocol.frozen_core_drivers,spec.target_gene)))
    if not set(required)<=set(pairs) or len(required)>protocol.top_n_events:
        raise ValueError('Frozen core/target cannot fit the declared event space')
    frequencies={name:float(((frame[g.primary]==1)|(frame[g.metastasis]==1)).groupby(ids.to_numpy()).max().mean()) for name,g in pairs.items()}
    recurrent=[] if protocol.recurrent_event_rule=='none' else sorted(
        (name for name in pairs if name not in required and frequencies[name]>=protocol.minimum_prevalence),
        key=lambda name:(-frequencies[name],name))
    genes=tuple(required+recurrent[:protocol.top_n_events-len(required)])
    loci=[(name,list(columns)) for name,columns in spec.event_loci if any(name in (f'P.{g} (M)',f'M.{g} (M)') for g in genes)]
    identity=json.dumps(dict(protocol=asdict(protocol),genes=genes,loci=loci),sort_keys=True,separators=(',',':'))
    version=protocol.schema_version+':'+hashlib.sha256(identity.encode()).hexdigest()[:20]
    return SelectedEventSchema(genes,protocol.schema_version,version)


def project_event_frame(features, selected_schema, *, compartment=None):
    """Thin classic-method input: shared event set, optional single compartment."""
    if compartment not in (None,'primary','metastasis'):
        raise ValueError('Invalid compartment projection')
    prefixes=('P','M') if compartment is None else ('P',) if compartment=='primary' else ('M',)
    columns=[f'{p}.{g} (M)' for p in prefixes for g in selected_schema.gene_names]
    columns += [c for c in ('observation_type','diag_order','seeding_at_first_observation','joint_snapshot') if c in features]
    return features[columns].copy()

"""Explicit numerical configurations; no fallback to evolving Python defaults."""
import math

KINETIC_PARAMETERS=frozenset(('d_model','n_head_layers','dropout_rate','learning_rate','weight_decay',
    'num_epochs','patience','focal_gamma','calibration','random_seed'))
TOPOLOGY_PARAMETERS=frozenset(('top_n_genes','regularization_strength','l1_ratio','l2_floor',
    'log_rate_clip_min','log_rate_clip_max','optimizer_maxiter','random_seed'))


def validate_model_parameters(kinetic,topology,configuration_source,topology_candidates=()):
    for name,values,required in [('kinetic',kinetic,KINETIC_PARAMETERS),('topology',topology,TOPOLOGY_PARAMETERS)]:
        missing=required-set(values)
        if missing: raise ValueError(f'Explicit frozen {name} parameters missing: {sorted(missing)}')
        if any(not math.isfinite(float(values[key])) for key in required if key!='calibration'):
            raise ValueError('Model parameters must be finite')
    for key in ('d_model','num_epochs','patience'):
        if kinetic[key]<1 or int(kinetic[key])!=kinetic[key]: raise ValueError('Positive integer kinetic dimensions/budgets required')
    if kinetic['n_head_layers']<0 or int(kinetic['n_head_layers'])!=kinetic['n_head_layers'] or not 0<=kinetic['dropout_rate']<1 or kinetic['learning_rate']<=0 or kinetic['weight_decay']<0:
        raise ValueError('Invalid kinetic architecture/optimizer')
    if kinetic['focal_gamma']<0 or kinetic['calibration'] not in ('none','platt') or (kinetic['focal_gamma']>0 and kinetic['calibration']!='platt'):
        raise ValueError('Focal probability requires explicit calibration')
    if topology['top_n_genes']<1 or int(topology['top_n_genes'])!=topology['top_n_genes'] or topology['optimizer_maxiter']<1 or int(topology['optimizer_maxiter'])!=topology['optimizer_maxiter']:
        raise ValueError('Positive integer topology dimensions/budgets required')
    if topology['regularization_strength']<0 or topology['l2_floor']<0 or not 0<=topology['l1_ratio']<=1 or topology['log_rate_clip_min']>=topology['log_rate_clip_max']:
        raise ValueError('Invalid topology regularization/clip bounds')
    if any(int(c['random_seed'])!=c['random_seed'] or c['random_seed']<0 for c in (kinetic,topology)):
        raise ValueError('Frozen nonnegative integer seeds required')
    if configuration_source=='prespecified':
        if topology_candidates: raise ValueError('Prespecified configuration cannot contain a selection grid')
    elif configuration_source=='inner_validation':
        if not 1<=len(topology_candidates)<=8: raise ValueError('Freeze 1–8 necessary candidates')
        for candidate in topology_candidates:
            if not candidate or not set(candidate)<={'regularization_strength','l1_ratio'}:
                raise ValueError('Only explicit topology regularization candidates supported')
            validate_model_parameters(kinetic,dict(topology,**candidate),'prespecified')
    else: raise ValueError('Explicit configuration_source required')

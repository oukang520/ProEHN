from proehn.events import EventSelectionProtocol


def explicit_configs():
    kinetic=dict(d_model=2,n_head_layers=0,dropout_rate=0.,learning_rate=.001,
        weight_decay=0.,num_epochs=1,patience=1,focal_gamma=0.,calibration='none',random_seed=42)
    topology=dict(regularization_strength=.01,l1_ratio=1.,l2_floor=0.,
        log_rate_clip_min=-20.,log_rate_clip_max=20.,optimizer_maxiter=1,random_seed=42)
    return kinetic,topology


def event_protocol(n=20):
    return EventSelectionProtocol('synthetic-events','1','none; synthetic',(),'patient_prevalence',0.,n,True,'always_include')

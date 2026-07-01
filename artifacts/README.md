# ProEHN Artifacts

This directory is intentionally kept lightweight. Trained weights and cohort data are
not bundled with the method code.

Expected artifact names are defined in `configs/*.yaml`:

- `proehn_kinetic_<cohort>.msgpack`: Flax parameters for the kinetic gatekeeper.
- `proehn_kinetic_<cohort>_metadata.pkl`: fitted scalers, feature groups and model configuration.
- `proehn_topology_<cohort>.npz`: topology parameters, selected genes and topology covariates.

The `scripts/train_topology.py` command writes a topology `.npz` artifact directly.
The `scripts/run_paca_examples.py` command consumes the PACA topology artifact and,
for Fig.2A, the PACA kinetic artifact pair.

# ProEHN

ProEHN is a context-aware evolutionary hazard network for cancer progression.
The released code keeps the core method only: the kinetic gatekeeper, the
feature-modulated topology engine, and the final next-event synthesis layer.

## Method Summary

For patient `i`, ProEHN uses a bipartite genotype state
`x_i in {0,1}^{2n+1}`. The first `n` bits represent primary-tumor events, the
next `n` bits represent metastatic events, and the final bit represents
metastatic seeding.

The framework has three stages:

1. `K`: a kinetic gatekeeper estimates `P(Stop | x_i, z_i)`. The progression
   probability is `P(Go) = 1 - P(Stop)`.
2. `T`: a feature-modulated CTMC constructs a patient-specific transition matrix
   through `theta_i[j,l] = sum_k z_i[k] * W_theta[k,j,l]`.
3. Final synthesis converts relative topology hazards into absolute next-event
   risk:

```text
P(next=e | x_i,z_i) = P(Go | x_i,z_i) * lambda_e(x_i,z_i) / sum_{a in A(x_i)} lambda_a(x_i,z_i)
```

## Repository Layout

```text
ProEHN/
  proehn/
    kinetic.py        # Kinetic gatekeeper architecture and inference wrapper
    topology.py       # Feature-modulated CTMC topology model
    engine.py         # Unified ProEHN prediction API
    preprocessing.py  # Analysis-ready feature table assembly utilities
    training.py       # Kinetic and topology training routines
    ctmc/             # Kronecker-factorized CTMC likelihood kernels
  configs/            # Default and cohort-specific parameters
  scripts/            # Minimal command-line entry points
  docs/               # Method and release notes
  examples/           # Example patient JSON
  tests/              # Lightweight API checks
  artifacts/          # Expected location for trained weights
```

## Installation

```bash
cd ProEHN
python -m pip install -e .
```

JAX installation can be platform-specific. If GPU acceleration is required,
install the matching `jaxlib` wheel following the official JAX instructions, then
install this package in editable mode.

## Train the Kinetic Gatekeeper

```bash
python scripts/train_kinetic.py \
  --config configs/cohorts/paca.yaml \
  --data path/to/cohort_feature_table.csv \
  --params-out artifacts/proehn_kinetic_paca.msgpack \
  --metadata-out artifacts/proehn_kinetic_paca_metadata.pkl
```

The kinetic input table should be analysis-ready and contain the configured
stop-label column, paired mutation columns and numeric covariates. The script
does not derive labels from raw clinical fields.

## Train the Topology Engine

```bash
python scripts/train_topology.py \
  --config configs/cohorts/paca.yaml \
  --data path/to/cohort_feature_table.csv \
  --out artifacts/proehn_topology_paca.npz
```

The input table should contain paired mutation columns such as `P.KRAS (M)` and
`M.KRAS (M)`, plus `Seeding`, `type`, and `diag_order` when available. Missing
`type` and `diag_order` columns are filled with conservative defaults for paired
samples.

## Predict One Patient

```bash
python scripts/predict_patient.py \
  --config configs/cohorts/paca.yaml \
  --patient examples/patient_example.json \
  --top-k 10
```

If topology artifacts are absent, the output contains kinetic probabilities but
no ranked next-event hazards.

## PACA Example Experiments

Pure-data example experiments for manuscript Fig.2A, Fig.3A-D and Fig.5B-C are
available through:

```bash
python scripts/run_paca_examples.py \
  --topology-model artifacts/proehn_topology_paca.npz \
  --topology-data path/to/paca_topology_feature_table.csv \
  --kinetic-data path/to/paca_gatekeeper_feature_table.csv \
  --out-dir results/paca_examples
```

See [docs/PACA_EXAMPLE_EXPERIMENTS.md](docs/PACA_EXAMPLE_EXPERIMENTS.md) for
the exact output tables and data requirements.

## Artifact Convention

Artifacts are configured in YAML:

- `proehn_kinetic_<cohort>.msgpack`: Flax parameters for the gatekeeper.
- `proehn_kinetic_<cohort>_metadata.pkl`: scalers, feature groups and model config.
- `proehn_topology_<cohort>.npz`: CTMC parameters, selected gene names and feature names.

## Tests

```bash
pytest
```

The included tests are intentionally lightweight so they can run before private
cohort data and trained weights are available.

# ProEHN Project Logic

## Architecture

ProEHN separates cancer progression prediction into three orthogonal questions.

1. Is the tumor evolutionarily active?
   The kinetic gatekeeper estimates `P(Stop)` from primary, metastatic and shared
   patient-level features.

2. If it is active, where can evolution proceed next?
   The topology engine constructs patient-specific CTMC hazards from a
   feature-modulated epistatic tensor.

3. What is the absolute next-event risk?
   The inference engine normalizes accessible topology hazards and gates them by
   `P(Go) = 1 - P(Stop)`.

## Core Code Retained

- `proehn/kinetic.py`: gatekeeper architecture, masked focal loss and inference wrapper.
- `proehn/topology.py`: patient-conditioned CTMC parameterization and likelihood loss.
- `proehn/ctmc/`: Kronecker-factorized CTMC kernels and observation likelihoods.
- `proehn/preprocessing.py`: gene-pair extraction, label generation and bucket building.
- `proehn/training.py`: L-BFGS-B fitting for topology parameters.
- `proehn/engine.py`: unified patient-level prediction interface.

## Experimental Code Excluded

The cleaned release excludes:

- figure scripts and generated figures
- one-off validation scripts
- robustness/ablation plotting outputs
- debug files, logs, archives and compiled caches
- private or intermediate cohort matrices

The goal is to expose the reproducible method surface, not the exploratory
analysis history.

## Naming Policy

All public classes, files, configs and artifact names use the ProEHN project
name. Cohort-specific differences are represented as YAML parameters rather than
new method names.

## Source-Code Priority

When the manuscript and implementation differ slightly, this release follows the
current implementation. In particular:

- the gatekeeper output is interpreted as `P(Stop)`
- `P(Go)` is computed as `1 - P(Stop)`
- final next-event risk is computed by normalizing accessible CTMC hazards and
  multiplying by `P(Go)`
- cohort-specific thresholds and regularization choices are kept in config files

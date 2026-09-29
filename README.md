# ProEHN

ProEHN is a context-aware evolutionary hazard network for cancer progression.

## Web application

The research web interface is available at **[https://47.239.63.248/](https://47.239.63.248/)**.

## Model core

ProEHN combines three components:

- **Evolutionary kinetics (K):** estimates the probability of continued progression, `P(Go) = 1 - P(Stop)`, from the observed patient state and covariates.
- **Evolutionary topology (T):** uses a covariate-modulated continuous-time Markov chain to model accessible primary-tumor, metastatic-tumor, and seeding transitions.
- **Next-event synthesis:** scores each accessible event `e` as `P(Go) * lambda_e / sum_a lambda_a`, combining progression probability with the event's conditional topology probability.

## Runtime environment

- Python 3.10+
- JAX and JAXlib 0.4.20+
- Flax 0.8+ and Optax 0.2+
- NumPy 1.24+, pandas 2.0+, SciPy 1.10+, scikit-learn 1.3+, and PyYAML 6.0+

```bash
python -m pip install -e .
```

## Datasets

- **PACA-AU and MELA-AU:** [ICGC 25K legacy Release 28](https://docs.icgc-argo.org/docs/data-access/icgc-25k-data)
- **LUAD:** [AACR Project GENIE 19.0-public access page](https://aacrprojectgenie.org/data/)

Patient-level source data are not redistributed in this repository. Access and use them under the source repositories' terms.

## Core commands

```bash
python scripts/train_kinetic.py --config configs/cohorts/paca.yaml --data data.csv
python scripts/train_topology.py --config configs/cohorts/paca.yaml --data data.csv
python scripts/run_paca_examples.py --experiments proehn_metrics --topology-data examples/paca_processed_data.csv
```

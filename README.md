# ProEHN

ProEHN is a context-aware evolutionary hazard network for cancer progression.

## Install

```bash
python -m pip install -e .
```

## Core Commands

```bash
python scripts/train_kinetic.py --config configs/cohorts/paca.yaml --data data.csv
python scripts/train_topology.py --config configs/cohorts/paca.yaml --data data.csv
python scripts/run_paca_examples.py --experiments proehn_metrics --topology-data examples/paca_processed_data.csv
```

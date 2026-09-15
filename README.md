# ProEHN

ProEHN is a context-aware evolutionary hazard network for cancer progression.

## Web application

The research web interface is available at **[https://47.239.63.248/](https://47.239.63.248/)**.

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

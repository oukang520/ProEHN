# PACA Example Experiments

This release includes pure-data example experiments for selected PACA panels in
the manuscript. The scripts do not generate figures; they print tabular outputs
and can optionally save each result table as CSV.

## Panels Covered

### Fig.2A: Kinetic Gatekeeper

Command output tables:

- `fig2a_metrics`: sample counts, AUC-ROC, AUC-PR and P(Stop) quantiles.
- `fig2a_robustness`: AUC under genomic bit-flip noise and label noise.
- `fig2a_survival_summary`: median-split survival summary and log-rank p-value.
- `fig2a_km_curve`: Kaplan-Meier curve data for high- and low-progression-risk groups.
- `fig2a_partial_effect_proxy`: P(Stop) decile summaries for tabular partial-effect inspection.

### Fig.3A-D: Host-Modulated Topology

Command output tables:

- `fig3a_seeding_surface`: TMB/Age z-score grid and relative seeding-hazard change.
- `fig3b_feature_group_contribution`: feature-category contribution to seeding and local mutation.
- `fig3c_top_seeding_features`: top features ranked by relative seeding contribution.
- `fig3d_impact_matrix_long`: long-form feature-induced gene-to-gene impact matrix.

### Fig.5B-C: Evolutionary Trajectories

Command output tables:

- `fig5b_state_exit_probabilities`: state-conditioned one-step exit probabilities.
- `fig5b_state_exit_raw`: patient-level raw state-exit probabilities.
- `fig5c_multistep_trajectories`: recovered multi-step trajectory edges from the WT state.

### PACA Ablation

Command output tables:

- `paca_ablation_topk_accuracy`: Full ProEHN versus evolution-only Top-k accuracy across Go ratios.
- `paca_ablation_cindex`: Full ProEHN versus evolution-only survival C-index.

### PACA ProEHN Metrics

Command output tables:

- `paca_proehn_metrics`: cross-validated ProEHN AUC-ROC, AUC-PR,
  specificity at 90% sensitivity, and max MCC.
- `paca_proehn_context_genes`: selected context genes for each PACA target.

## Example Command

```bash
python scripts/run_paca_examples.py \
  --topology-model artifacts/proehn_topology_paca.npz \
  --topology-data path/to/paca_topology_feature_table.csv \
  --kinetic-params artifacts/proehn_kinetic_paca.msgpack \
  --kinetic-metadata artifacts/proehn_kinetic_paca_metadata.pkl \
  --kinetic-data path/to/paca_gatekeeper_feature_table.csv \
  --out-dir results/paca_examples
```

To run only topology-based examples:

```bash
python scripts/run_paca_examples.py \
  --experiments fig3 fig5 \
  --topology-model artifacts/proehn_topology_paca.npz \
  --topology-data path/to/paca_topology_feature_table.csv
```

To run only the PACA ablation tables:

```bash
python scripts/run_paca_examples.py \
  --experiments ablation \
  --topology-model artifacts/proehn_topology_paca.npz \
  --topology-data examples/paca_processed_data.csv \
  --kinetic-params artifacts/proehn_kinetic_paca.msgpack \
  --kinetic-metadata artifacts/proehn_kinetic_paca_metadata.pkl
```

To run only the cleaned PACA ProEHN metrics:

```bash
python scripts/run_paca_examples.py \
  --experiments proehn_metrics \
  --topology-data examples/paca_processed_data.csv
```

## Data Requirements

The topology examples expect a PACA feature table containing paired mutation
columns such as `P.KRAS (M)` and `M.KRAS (M)`, plus model covariates used during
topology training.

The Fig.2A gatekeeper example additionally expects kinetic artifacts and a PACA
feature table with one of the following label sources:

- `Patient_Label`
- `PT_label` and `MT_label`
- `PFS_days` with optional `PFS_days_is_Missing`

Survival summaries require an OS time column such as `OS_days` and a vital-status
column such as `donor_vital_status`.

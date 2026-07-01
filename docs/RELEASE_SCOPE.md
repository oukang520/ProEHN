# Release Scope

This folder is a cleaned method release for ProEHN. It deliberately excludes:

- figure-generation scripts
- ad hoc validation scripts
- intermediate `.png`, `.pdf`, `.csv`, `.zip`, `.log`, `.pyc` files
- cohort-specific exploratory debugging files

The retained code covers:

- kinetic gatekeeper architecture and inference wrapper
- feature-modulated CTMC topology likelihood
- core preprocessing for gene pairs, labels and CTMC buckets
- topology training with L-BFGS-B
- unified patient-level prediction API
- cohort-specific YAML parameter presets

Original source files remain untouched outside this release directory.

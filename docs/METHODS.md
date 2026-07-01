# ProEHN Core Method

This document describes the code-level method implemented in `proehn/`. It is
written to match the current source code, which is treated as authoritative when
minor manuscript-code differences exist.

## 1. State Space

For `n` canonical genomic events, each patient is encoded in a coupled
primary-metastatic state:

```text
x = [x_PT_1, ..., x_PT_n, x_MT_1, ..., x_MT_n, x_seed] in {0,1}^{2n+1}
```

The final event, `x_seed`, indicates metastatic seeding and is treated as the
bridge between localized primary-tumor evolution and metastatic evolution.

## 2. Kinetic Gatekeeper

The kinetic gatekeeper is a late-fusion MLP implemented in
`proehn/kinetic.py`.

Inputs are split into five groups:

- primary genomic events
- primary dynamic covariates
- metastatic genomic events
- metastatic dynamic covariates
- shared host-level covariates

The model outputs a single logit:

```text
y_hat = sigmoid(f_K(x_PT_gen, x_PT_dyn, x_MT_gen, x_MT_dyn, z_shared))
```

The code interprets `y_hat` as `P(Stop)`, so:

```text
P(Go) = 1 - P(Stop)
```

The training objective is masked focal loss with source-compatible handling of
unknown labels (`label = -1`):

```text
FL(p_t) = - (1 - p_t)^gamma log(p_t)
```

The decision rule is:

```text
Stasis      if P(Stop) > tau_stop
Progression otherwise
```

Default `tau_stop` is conservative (`0.90`) and cohort-specific YAML files keep
the source-code tuned values where they differed.

## 3. Feature-Modulated Topology

The topology engine is implemented in `proehn/topology.py`. It extends a CTMC
hazard model by allowing patient features to modulate the epistatic transition
matrix.

Given covariates with a leading intercept:

```text
z_i = [1, z_i1, ..., z_id]
```

the patient-specific parameters are:

```text
theta_i[j,l] = sum_k z_i[k] * W_theta[k,j,l]
log_dp_i[j]  = sum_k z_i[k] * W_dp[k,j]
log_dm_i[j]  = sum_k z_i[k] * W_dm[k,j]
```

Diagonal entries `theta_i[j,j]` are basal log hazards. Off-diagonal entries
`theta_i[j,l]` are epistatic effects from active regulator event `l` to target
event `j`.

For an accessible target event `j`, the code-level transition hazard is:

```text
lambda_j(x_i,z_i) = exp(theta_i[j,j] + sum_{l in active(x_i)} theta_i[j,l])
```

The seeding event is represented by index `n`.

## 4. Observation Likelihood

The CTMC likelihood kernels are in `proehn/ctmc/`. The code supports three
cross-sectional observation buckets:

- `type = 1`: primary-only observation
- `type = 2`: metastasis-only observation
- `type = 3`: paired primary-metastatic observation

Paired samples additionally use `diag_order`:

- `0`: simultaneous diagnosis
- `1`: primary-first diagnosis
- `2`: metastasis-first diagnosis

The CTMC generator is never explicitly materialized for large state spaces.
Instead, Kronecker-factorized matrix-vector products evaluate restricted
resolvent terms of the form:

```text
(D - Q)^(-1) p0
```

where `D` is the diagnosis-rate diagonal and `Q` is the feature-conditioned CTMC
generator.

## 5. Regularization

The topology objective is:

```text
L = - sum_i log P(x_i | z_i, W) + penalty(W)
```

The penalty is applied to feature-modulating dimensions, not to the basal
intercept rates:

```text
penalty = lambda * [alpha * ||W_feature||_1 + (1 - alpha + 0.1) * ||W_feature||_2^2]
```

This follows the source-code intent: preserve cohort-level basal hazards while
regularizing patient-specific modulation.

## 6. Final Risk Synthesis

For all currently accessible events `A(x_i)`, topology probabilities are
obtained from competing CTMC hazards:

```text
P_topo(next=e | x_i,z_i, Go) = lambda_e / sum_{a in A(x_i)} lambda_a
```

The final absolute next-event risk is:

```text
P(next=e | x_i,z_i) = P(Go | x_i,z_i) * P_topo(next=e | x_i,z_i, Go)
```

This is implemented in `ProEHNEngine.predict`.

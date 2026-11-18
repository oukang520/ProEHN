# Runtime environment

Use Python 3.11 (64-bit) on Windows, Linux or WSL2. CPU execution is supported.

```sh
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux / WSL2
source .venv/bin/activate
```

Install the pinned Python runtime:

```sh
python -m pip install --upgrade pip
python -m pip install -e .
```

Enable double precision before running Python:

```powershell
$env:JAX_ENABLE_X64="true"
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
```

```sh
export JAX_ENABLE_X64=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Optional survival-analysis dependencies:

```sh
python -m pip install -e ".[clinical]"
```

Optional MHN and official metMHN dependencies require Git and a compiler when wheels are unavailable:

```sh
python -m pip install -e ".[comparators]"
```

The R comparator adapters require R 4.5.1 with `Rscript` on `PATH`,
`Oncotree` 0.3.5, `hypertrapsct` 0.99.0 and `jsonlite` installed in the R library.
Install their pinned sources from R (a native compiler toolchain is needed when building packages):

```r
install.packages(c("remotes", "jsonlite"), repos = "https://cloud.r-project.org")
remotes::install_version("Rcpp", version = "1.1.2", upgrade = "never",
                        repos = "https://cloud.r-project.org")
remotes::install_version("Oncotree", version = "0.3.5", upgrade = "never",
                        repos = "https://cloud.r-project.org")
remotes::install_github("StochasticBiology/hypertrapsct@b70b459cb3ec717712fa914f5a1c91b02d5de186",
                       dependencies = NA, upgrade = "never")
```

Use `--rscript` to specify the R executable when it is outside `PATH`.
Use `R_LIBS_USER` for a custom R package-library directory.

```sh
python -c "import jax, proehn; print(proehn.__version__, jax.config.x64_enabled)"
```

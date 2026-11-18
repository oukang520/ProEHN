args <- commandArgs(trailingOnly = TRUE)
method <- args[1]
destination <- normalizePath(args[2], winslash = "/", mustWork = TRUE)

require_version <- function(package, version) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(paste("Missing R package:", package, "required version", version))
  }
  actual <- as.character(utils::packageVersion(package))
  if (actual != version) stop(paste(package, "requires", version, "found", actual))
}

if (method == "check") {
  require_version("Oncotree", "0.3.5")
  require_version("hypertrapsct", "0.99.0")
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Missing R package: jsonlite")
  cat("R comparator dependencies are available\n")
  quit(status = 0)
}

observations <- as.matrix(read.csv(file.path(destination, "training.csv"),
                                   check.names = FALSE))
genes <- colnames(observations)
if (method == "oncotree") {
  require_version("Oncotree", "0.3.5")
  tree <- Oncotree::oncotree.fit(as.data.frame(observations))
  saveRDS(tree, file.path(destination, "fit.rds"))
  latent <- Oncotree::distribution.oncotree(tree, with.probs = TRUE,
                  with.errors = FALSE, edge.weights = "estimated")
  stopifnot(min(latent$Prob) >= -1e-12, abs(sum(latent$Prob) - 1) < 1e-8)
  floor <- 0.5 / (nrow(observations) + 1)
  epos <- max(floor, min(1 - floor, as.numeric(tree$eps[1])))
  eneg <- max(floor, min(1 - floor, as.numeric(tree$eps[2])))
  observed <- expand.grid(rep(list(0:1), length(genes)))
  names(observed) <- genes
  latent_x <- as.matrix(latent[, genes])
  observed_x <- as.matrix(observed)
  probability <- numeric(nrow(observed))
  for (i in seq_len(nrow(latent_x))) {
    component <- rep(1, nrow(observed))
    for (j in seq_along(genes)) {
      one <- if (latent_x[i, j] == 1) 1 - eneg else epos
      component <- component * ifelse(observed_x[, j] == 1, one, 1 - one)
    }
    probability <- probability + latent$Prob[i] * component
  }
  observed$Probability <- probability
  write.csv(observed, file.path(destination, "states.csv"), row.names = FALSE)
  write.csv(data.frame(epos = epos, eneg = eneg, floor = floor,
      original_epos = tree$eps[1], original_eneg = tree$eps[2]),
      file.path(destination, "errors.csv"), row.names = FALSE)
} else if (method == "hypertraps_ct") {
  require_version("hypertrapsct", "0.99.0")
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Missing R package: jsonlite")
  seed <- as.integer(args[3])
  nominal_penalty <- as.numeric(args[4])
  initial <- scan(file.path(destination, "initial.txt"), quiet = TRUE)
  set.seed(seed)
  started <- proc.time()[3]
  fit <- hypertrapsct::HyperTraPS(observations, initialparams = initial,
      starttimes = rep(0, nrow(observations)),
      endtimes = rep(Inf, nrow(observations)), length = 3, model = 2,
      kernel = 2, lasso = nrow(observations) * nominal_penalty,
      sa = 1, seed = seed, walkers = 200, full_analysis = 0,
      limited_output = 0, output_transitions = 0,
      featurenames = genes, samplegap = 100)
  readout <- hypertrapsct::HyperTraPS(observations[1, , drop = FALSE],
      initialparams = as.numeric(fit$best), starttimes = 0, endtimes = Inf,
      length = 0, kernel = 2, sa = 1, walkers = 1,
      full_analysis = 0, limited_output = 1, output_transitions = 1)
  fit$dynamics <- readout$dynamics
  saveRDS(fit, file.path(destination, "fit.rds"))
  write.csv(fit$dynamics$states, file.path(destination, "states.csv"), row.names = FALSE)
  write.csv(fit$lik.traces, file.path(destination, "likelihood_trace.csv"), row.names = FALSE)
  write.csv(fit$posterior.samples, file.path(destination, "parameter_samples.csv"), row.names = FALSE)
  jsonlite::write_json(list(parameters = as.numeric(fit$best),
      bestlik = fit$bestlik, elapsed_seconds = unname(proc.time()[3] - started),
      nominal_penalty = nominal_penalty,
      native_integer_penalty = floor(nrow(observations) * nominal_penalty),
      seed = seed, proposals = 1000, walkers = 200, kernel = 2,
      width = 0.01, sample_gap = 100,
      time_windows = "0 to infinity; no observed times",
      algorithm = "Finite-budget simulated annealing point estimate"),
      file.path(destination, "native.json"), auto_unbox = TRUE, digits = 17)
} else {
  stop(paste("Unknown comparator:", method))
}

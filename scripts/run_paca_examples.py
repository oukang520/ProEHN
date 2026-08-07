"""Run PACA pure-data example experiments for selected ProEHN manuscript panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from proehn.mhn_benchmarks import DEFAULT_PACA_MHN_TARGETS, run_paca_mhn_benchmark
from proehn.paca_experiments import PACAExampleExperiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PACA example experiments as pure tabular outputs."
    )
    parser.add_argument(
        "--topology-model",
        default="artifacts/proehn_topology_paca.npz",
        help="PACA topology .npz artifact.",
    )
    parser.add_argument("--topology-data", required=True, help="PACA topology/trajectory feature CSV.")
    parser.add_argument(
        "--kinetic-params",
        default="artifacts/proehn_kinetic_paca.msgpack",
        help="PACA gatekeeper parameter file.",
    )
    parser.add_argument(
        "--kinetic-metadata",
        default="artifacts/proehn_kinetic_paca_metadata.pkl",
        help="PACA gatekeeper metadata pickle.",
    )
    parser.add_argument("--kinetic-data", default=None, help="PACA gatekeeper/survival feature CSV for Fig.2A.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["fig2a", "fig3", "fig5"],
        choices=["fig2a", "fig3", "fig5", "ablation", "mhn", "all"],
        help="Experiments to run.",
    )
    parser.add_argument("--out-dir", default=None, help="Optional directory for CSV outputs.")
    parser.add_argument("--preview-rows", type=int, default=12, help="Rows shown in terminal previews.")
    parser.add_argument("--grid-size", type=int, default=31, help="Grid size for Fig.3A surface data.")
    parser.add_argument("--n-sim", type=int, default=1000, help="Simulation size for PACA ablation Top-k accuracy.")
    parser.add_argument(
        "--mhn-targets",
        nargs="+",
        default=None,
        help="Primary mutation targets for the PACA MHN benchmark.",
    )
    parser.add_argument(
        "--mhn-context-genes",
        type=int,
        default=9,
        help="Number of context genes per MHN benchmark target.",
    )
    parser.add_argument(
        "--mhn-cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for the PACA MHN benchmark.",
    )
    parser.add_argument(
        "--mhn-maxit",
        type=int,
        default=5000,
        help="Maximum BFGS iterations for regularized MHN fitting.",
    )
    parser.add_argument(
        "--mhn-hypertraps-iterations",
        type=int,
        default=150,
        help="HyperTraPS MCMC iterations per fold.",
    )
    parser.add_argument(
        "--mhn-hypertraps-walks",
        type=int,
        default=15,
        help="HyperTraPS walks per likelihood estimate.",
    )
    return parser.parse_args()


def print_table(name: str, table: pd.DataFrame, preview_rows: int) -> None:
    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)
    if table.empty:
        print("[empty]")
        return
    print(table.head(preview_rows).to_string(index=False))
    if len(table) > preview_rows:
        print(f"... ({len(table)} rows total)")


def save_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path | None) -> None:
    if out_dir is None:
        return
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_path / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    selected = set(args.experiments)
    if "all" in selected:
        selected = {"fig2a", "fig3", "fig5", "ablation", "mhn"}

    runner = None
    if selected & {"fig2a", "fig3", "fig5", "ablation"}:
        runner = PACAExampleExperiments(
            topology_model_path=args.topology_model,
            topology_data_path=args.topology_data,
            kinetic_params_path=args.kinetic_params,
            kinetic_metadata_path=args.kinetic_metadata,
            kinetic_data_path=args.kinetic_data,
        )

    all_tables: dict[str, pd.DataFrame] = {}
    if "fig2a" in selected:
        assert runner is not None
        tables = runner.fig2a_kinetic_gatekeeper()
        all_tables.update(tables)
    if "fig3" in selected:
        assert runner is not None
        tables = runner.fig3_host_modulation(grid_size=args.grid_size)
        all_tables.update(tables)
    if "fig5" in selected:
        assert runner is not None
        tables = runner.fig5_trajectories()
        all_tables.update(tables)
    if "ablation" in selected:
        assert runner is not None
        tables = runner.ablation_full_vs_evolution(n_sim=args.n_sim)
        all_tables.update(tables)
    if "mhn" in selected:
        targets = args.mhn_targets if args.mhn_targets is not None else DEFAULT_PACA_MHN_TARGETS
        tables = run_paca_mhn_benchmark(
            pd.read_csv(args.topology_data),
            targets=targets,
            n_context_genes=args.mhn_context_genes,
            n_splits=args.mhn_cv_folds,
            mhn_maxit=args.mhn_maxit,
            hypertraps_iterations=args.mhn_hypertraps_iterations,
            hypertraps_walks=args.mhn_hypertraps_walks,
        )
        all_tables.update(tables)

    save_tables(all_tables, args.out_dir)
    for name, table in all_tables.items():
        print_table(name, table, args.preview_rows)

    if args.out_dir:
        print(f"\nCSV outputs written to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()

"""Inspect a saved ProEHN topology artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a ProEHN topology .npz artifact.")
    parser.add_argument("artifact", help="Path to the .npz artifact.")
    args = parser.parse_args()

    data = np.load(Path(args.artifact), allow_pickle=True)
    summary = {
        "artifact": args.artifact,
        "n_parameters": int(data["params"].size),
        "n_genes": int(len(data["gene_names"])),
        "genes": [str(x) for x in data["gene_names"].tolist()],
        "n_features": int(len(data["feature_names"])),
        "features": [str(x) for x in data["feature_names"].tolist()],
        "final_loss": float(data["final_loss"]) if "final_loss" in data else None,
        "n_samples": int(data["n_samples"]) if "n_samples" in data else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

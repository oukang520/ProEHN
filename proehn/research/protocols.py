"""Frozen event vocabularies and executable patient partition conventions."""

from pathlib import Path
import hashlib
import json
import numpy as np
import yaml

GENES = {
    "GENIE": (
        "TP53",
        "KRAS",
        "EGFR",
        "KEAP1",
        "STK11",
        "KMT2D",
        "RBM10",
        "ATM",
        "NF1",
        "RB1",
        "FAT1",
        "SMARCA4",
    ),
    "MELA": (
        "BRAF",
        "PTPRT",
        "PTPRD",
        "EPHA3",
        "NF1",
        "TP53",
        "FAT1",
        "KMT2C",
        "SETD2",
        "ATM",
        "EGFR",
        "ARID1A",
    ),
    "PACA": (
        "KRAS",
        "TP53",
        "ARID1A",
        "CSMD1",
        "KMT2C",
        "NRXN3",
        "ATM",
        "RBM10",
        "NRXN1",
        "PCDH15",
        "SMARCA4",
        "DMD",
    ),
}
FULL_OPTIMIZER = dict(
    regularization_strength=0.002,
    l1_ratio=0.0,
    l2_floor=0.0,
    optimizer_maxiter=1200,
    optimizer_bound=4.0,
    log_rate_clip_min=-5.0,
    log_rate_clip_max=5.0,
    optimizer_ftol=1e-8,
)
BASE_RIDGE = {"MELA": 0.025, "PACA": 0.005}
PUBLISHED_SNAPSHOT = {
    "GENIE": ("legacy",) * 5,
    "MELA": ("EO20", "legacy", "legacy", "EO20", "legacy"),
    "PACA": ("legacy", "E05", "legacy", "legacy", "legacy"),
}


def load_protocol(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_external_output(path):
    """Keep generated data and fitted artifacts outside the source checkout."""
    target = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    if target == repository or repository in target.parents:
        raise ValueError("Choose an output directory outside the source repository")
    return target


def patient_digest(frame):
    return hashlib.sha256(
        "\n".join(sorted(map(str, frame.patient_id))).encode()
    ).hexdigest()


def partition(frame, fold):
    """Outer test f, validation (f+1)%5, training the other three folds."""
    if fold not in range(5) or not frame.fold.isin(range(5)).all():
        raise ValueError("Patient folds must be integers 0..4")
    inner = (fold + 1) % 5
    return (
        frame[~frame.fold.isin([fold, inner])].copy(),
        frame[frame.fold.eq(inner)].copy(),
        frame[frame.fold.eq(fold)].copy(),
    )


def snapshot_frame(frame, genes):
    """PT if observed, otherwise MT; every adapter sees a primary-only record."""
    if frame.patient_id.isna().any() or frame.patient_id.astype(str).duplicated().any():
        raise ValueError("Exactly one row per patient is required")
    if not frame.observation_type.isin(
        [
            "primary_only",
            "metastasis_only",
            "paired",
            "primary_seeding_unknown",
            "primary_disseminated",
            "primary_no_dissemination",
            0,
            1,
            2,
            3,
            4,
        ]
    ).all():
        raise ValueError("Invalid observation_type")
    projected = frame.copy()
    mt = frame.observation_type.isin(["metastasis_only", 2]).to_numpy()
    projected["source_compartment"] = np.where(mt, "MT", "PT")
    values = np.empty((len(frame), len(genes)), int)
    for j, gene in enumerate(genes):
        values[:, j] = np.where(mt, frame["M." + gene], frame["P." + gene])
    if not np.isin(values, [0, 1]).all():
        raise ValueError("The common snapshot needs callable binary genotypes")
    for j, gene in enumerate(genes):
        projected["P." + gene] = values[:, j]
        projected["M." + gene] = np.nan
    projected["observation_type"] = "primary_only"
    projected["observed_seeding"] = np.nan
    projected["diag_order"] = 0
    projected["seeding_at_first_observation"] = -1
    return projected, values


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)

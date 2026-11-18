"""Raw source reconstruction with frozen genes and allele-level background exclusion."""

from collections import defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import numpy as np
import pandas as pd
from .protocols import GENES

GENIE_NONSYN = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
}
ICGC_NONSYN = {
    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "initiator_codon_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "inframe_insertion",
    "inframe_deletion",
    "protein_altering_variant",
    "disruptive_inframe_insertion",
    "disruptive_inframe_deletion",
}


def _update(variants, key, genes, mutated_genes, fully_mapped, nonsyn, depth, mutant):
    value = variants.setdefault(
        key,
        dict(
            sample=key[0],
            genes=set(),
            mutated_genes=set(),
            fully_mapped=True,
            nonsyn=False,
            depth=None,
            mutant=None,
        ),
    )
    value["genes"].update(genes)
    value["mutated_genes"].update(mutated_genes)
    value["fully_mapped"] &= fully_mapped
    value["nonsyn"] |= nonsyn
    try:
        depth, mutant = float(depth), float(mutant)
        if (
            np.isfinite(depth)
            and np.isfinite(mutant)
            and depth > 0
            and 0 <= mutant <= depth
        ):
            if value["depth"] is None or (depth, -mutant) > (
                value["depth"],
                -value["mutant"],
            ):
                value["depth"], value["mutant"] = depth, mutant
    except (ValueError, TypeError):
        pass


def background(variants, modeled_genes):
    keep = [
        v
        for v in variants
        if v["nonsyn"]
        and v["fully_mapped"]
        and v["genes"]
        and not set(v["genes"]) & set(modeled_genes)
    ]
    vaf = [v["mutant"] / v["depth"] for v in keep if v["depth"] is not None]
    return len(keep), float(np.mean(vaf)) if vaf else np.nan


def aggregate_icgc(rows, gene_map):
    variants = {}
    profiled = set()
    strategies = defaultdict(set)
    for row in rows:
        sid = row["icgc_sample_id"]
        profiled.add(sid)
        strategies[sid].add(row.get("sequencing_strategy", ""))
        key = (
            sid,
            row["chromosome"].removeprefix("chr"),
            row["chromosome_start"],
            row["chromosome_end"],
            row["mutated_from_allele"],
            row["mutated_to_allele"],
        )
        ids = [
            g.strip().split(".")[0]
            for g in row.get("gene_affected", "").replace(";", ",").split(",")
            if g.strip()
        ]
        consequences = set(
            row.get("consequence_type", "")
            .replace(";", ",")
            .replace("&", ",")
            .split(",")
        )
        nonsyn = bool(consequences & ICGC_NONSYN)
        mapped = {gene_map[g] for g in ids if g in gene_map}
        _update(
            variants,
            key,
            mapped,
            mapped if nonsyn else set(),
            bool(ids) and all(g in gene_map for g in ids),
            nonsyn,
            row.get("total_read_count"),
            row.get("mutant_allele_read_count"),
        )
    return [v for v in variants.values() if v["nonsyn"]], profiled, strategies


def aggregate_genie(chunks, selected):
    variants = {}
    for chunk in chunks:
        chunk = chunk[chunk.Tumor_Sample_Barcode.isin(selected)]
        chunk = chunk[chunk.FILTER.fillna("PASS").isin(["PASS", "."])]
        for row in chunk.to_dict("records"):
            sid = row["Tumor_Sample_Barcode"]
            key = (
                sid,
                str(row["Chromosome"]),
                row["Start_Position"],
                row["End_Position"],
                row["Reference_Allele"],
                row["Tumor_Seq_Allele2"],
            )
            genes = {
                g.strip()
                for g in str(row["Hugo_Symbol"]).replace(";", ",").split(",")
                if g.strip() and g != "nan"
            }
            nonsyn = row["Variant_Classification"] in GENIE_NONSYN
            try:
                depth = float(row["t_ref_count"]) + float(row["t_alt_count"])
            except (ValueError, TypeError):
                depth = None
            _update(
                variants,
                key,
                genes,
                genes if nonsyn else set(),
                bool(genes),
                nonsyn,
                depth,
                row["t_alt_count"],
            )
    return [v for v in variants.values() if v["nonsyn"]]


def _finish(samples, by_sample, genes):
    samples = samples.copy()
    for index, row in samples.iterrows():
        variants = by_sample.get(row.sample_id, [])
        calls = {gene for value in variants for gene in value["mutated_genes"]}
        count, vaf = background(variants, genes)
        samples.loc[index, "background_nMut"] = count
        samples.loc[index, "background_VAF_mean"] = vaf
        for gene in genes:
            samples.loc[index, gene] = int(gene in calls)
    patients = []
    for pid, group in samples.groupby("patient_id"):
        p = group[group.compartment.eq("P")]
        m = group[group.compartment.eq("M")]
        row = dict(
            patient_id=pid,
            fold=int(group.fold.iloc[0]),
            age=float(group.age.min()),
            observation_type=(
                "paired"
                if len(p) and len(m)
                else "primary_only" if len(p) else "metastasis_only"
            ),
            primary_sample_id="" if p.empty else p.sample_id.iloc[0],
            metastasis_sample_id="" if m.empty else m.sample_id.iloc[0],
            background_nMut=float(group.background_nMut.mean()),
            background_VAF_mean=float(group.background_VAF_mean.mean()),
        )
        for prefix, part in [("P", p), ("M", m)]:
            for gene in genes:
                row[f"{prefix}.{gene}"] = (
                    np.nan if part.empty else int(part[gene].iloc[0])
                )
            for field in ["background_nMut", "background_VAF_mean"]:
                row[f"{prefix}.{field}"] = (
                    np.nan if part.empty else float(part[field].iloc[0])
                )
        patients.append(row)
    return pd.DataFrame(patients), samples


def prepare_genie(*, sample_path, variant_path, panel_path, universe_path):
    """Use the released lung patient universe and 12 fixed genes, without reselection."""
    genes = GENES["GENIE"]
    universe = pd.read_csv(universe_path)
    id_column = next(
        (c for c in ["patientID", "patient_id", "PATIENT_ID"] if c in universe), None
    )
    if id_column is None:
        raise ValueError("Patient universe needs patientID or patient_id")
    samples = pd.read_csv(sample_path, sep="\t", comment="#", low_memory=False)
    samples = samples[
        samples.PATIENT_ID.isin(set(universe[id_column]))
        & samples.SAMPLE_TYPE.isin(["Primary", "Metastasis"])
        & samples.SAMPLE_CLASS.eq("Tumor")
    ].copy()
    samples["age"] = pd.to_numeric(
        samples.AGE_AT_SEQ_REPORT.astype(str).replace({">89": "89", "<18": "18"}),
        errors="coerce",
    )
    samples = samples.sort_values(
        ["PATIENT_ID", "SAMPLE_TYPE", "age", "SAMPLE_ID"], na_position="last"
    ).drop_duplicates(["PATIENT_ID", "SAMPLE_TYPE"])
    ids = sorted(samples.PATIENT_ID.unique())
    np.random.default_rng(20261106).shuffle(ids)
    folds = {pid: index % 5 for index, pid in enumerate(ids)}
    panel = pd.read_csv(
        panel_path, sep="\t", usecols=["Hugo_Symbol", "SEQ_ASSAY_ID", "includeInPanel"]
    )
    panel = panel[panel.includeInPanel.astype(str).str.lower().eq("true")]
    panelgenes = panel.groupby("SEQ_ASSAY_ID").Hugo_Symbol.agg(set).to_dict()
    samples = samples[
        samples.SEQ_ASSAY_ID.map(
            lambda assay: set(genes) <= panelgenes.get(assay, set())
        )
    ].copy()
    variants = aggregate_genie(
        pd.read_csv(variant_path, sep="\t", chunksize=100000, low_memory=False),
        set(samples.SAMPLE_ID),
    )
    by_sample = defaultdict(list)
    for variant in variants:
        by_sample[variant["sample"]].append(variant)
    samples = samples.rename(
        columns={"PATIENT_ID": "patient_id", "SAMPLE_ID": "sample_id"}
    )
    samples["compartment"] = samples.SAMPLE_TYPE.map(
        {"Primary": "P", "Metastasis": "M"}
    )
    samples["fold"] = samples.patient_id.map(folds)
    return _finish(samples, by_sample, genes)


def prepare_icgc(*, cohort, source, bindings_path, hgnc_path):
    """Read release28 donor-part files with their original separate header files."""
    if cohort not in ("MELA", "PACA"):
        raise ValueError("ICGC cohort must be MELA or PACA")
    source = Path(source)
    genes = GENES[cohort]
    hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str)
    hgnc = hgnc[hgnc.status.eq("Approved")]
    mapping = (
        hgnc.dropna(subset=["ensembl_gene_id"])
        .groupby("ensembl_gene_id")
        .symbol.agg(set)
    )
    gene_map = {
        key: next(iter(value)) for key, value in mapping.items() if len(value) == 1
    }
    bindings = pd.read_csv(bindings_path)
    bindings = bindings[bindings.cohort.eq(cohort)]
    headers = {
        kind: gzip.decompress((source / f"{kind}_header.tsv.gz").read_bytes())
        .decode()
        .strip()
        .split("\t")
        for kind in ["sample", "specimen", "ssm_open"]
    }

    def rows(directory, kind):
        for path in sorted((directory / kind).glob("part*.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                yield from csv.DictReader(
                    handle, delimiter="\t", fieldnames=headers[kind]
                )

    selected = []
    by_sample = defaultdict(list)
    for donor in bindings.itertuples():
        directory = source / donor.project_code / donor.icgc_donor_id
        variants, profiled, strategies = aggregate_icgc(
            rows(directory, "ssm_open"), gene_map
        )
        specimens = {
            row["icgc_specimen_id"]: row for row in rows(directory, "specimen")
        }
        candidates = []
        for row in rows(directory, "sample"):
            sid = row["icgc_sample_id"]
            specimen = specimens.get(row["icgc_specimen_id"], {})
            kind = specimen.get("specimen_type", "").lower()
            comp = (
                "P"
                if kind.startswith(("primary tumour", "primary tumor"))
                else (
                    "M"
                    if kind.startswith(("metastatic tumour", "metastatic tumor"))
                    else None
                )
            )
            if (
                comp is None
                or sid not in profiled
                or not set(strategies[sid]) & {"WGS", "WXS"}
            ):
                continue
            fold = (
                int(
                    hashlib.sha256(
                        f"20261106:{cohort}:{donor.icgc_donor_id}".encode()
                    ).hexdigest()[:16],
                    16,
                )
                % 5
            )
            candidates.append(
                dict(
                    patient_id=donor.icgc_donor_id,
                    project=donor.project_code,
                    sample_id=sid,
                    specimen_id=row["icgc_specimen_id"],
                    compartment=comp,
                    fold=fold,
                    specimen_interval=pd.to_numeric(
                        specimen.get("specimen_interval", ""), errors="coerce"
                    ),
                    age=pd.to_numeric(donor.donor_age_at_diagnosis, errors="coerce"),
                )
            )
        if not candidates:
            continue
        candidates = (
            pd.DataFrame(candidates)
            .drop_duplicates("sample_id")
            .sort_values(
                ["specimen_interval", "specimen_id", "sample_id"], na_position="last"
            )
            .drop_duplicates("compartment")
        )
        selected.extend(candidates.to_dict("records"))
        chosen = set(candidates.sample_id)
        for variant in variants:
            if variant["sample"] in chosen:
                by_sample[variant["sample"]].append(variant)
    if not selected:
        raise ValueError("No eligible raw genomic source records")
    frame, samples = _finish(pd.DataFrame(selected), by_sample, genes)
    endpoint = bindings.set_index("icgc_donor_id")
    for column in ["source_DFI_days", "source_DFI180_label"]:
        if column in endpoint:
            frame[column] = frame.patient_id.map(endpoint[column])
    return frame, samples

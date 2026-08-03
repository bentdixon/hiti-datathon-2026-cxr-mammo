"""
Cohort construction, end to end: two raw embedding stores -> one modelling
parquet with demo_/race_/tech_/mammo_/cxr_ prefixed feature blocks.

This used to be two stages -- merge_datasets.py building analysis_table /
model_table / fusion_table / model_features.json from the raw mammography and
chest-X-ray stores, then models/cohort.py re-reading those files and encoding
them into the block layout linear.py and nonlinear.py select on. They are
folded into one module and one process here, so a fresh checkout needs to point
at exactly two data locations -- --mammo-embed-dir and --cxr-embed-dir -- plus
two small metadata tables, and gets a modelling-ready parquet out the other
end without an intermediate contract file that could drift from the table it
describes.

Design, unchanged from merge_datasets.py:
- Eligibility (`valid`) is computed once, up front, from all four sources plus
  the female-only restriction, so no downstream table is filtered on a stale
  set. The 43 male patients are excluded: this is a screening-mammography
  cohort, they are not the target population, and 43 patients cannot support
  their own estimates. Results apply to women only.
- `index_date = max(mammo_date, cxr_date)`, the later of the two imaging
  studies, so no imaging feature ever postdates the label anchor. Labels are
  prevalence: a patient is positive for an outcome if any ICD code for it
  carries `dx_date <= index_date`.
- Mammography E is two independently L2-normalised 768-d blocks (FFDM,
  mean-pooled over L/R views of the index accession) concatenated to 1536-d.
  Chest-X-ray E is the 768-d `global` pooled vector, not normalised. Both are
  written raw; L2-normalisation and standardisation belong inside the CV fold,
  not here.
- `Race` is carried as a stratifier and is not a feature by default -- an
  explicit allow-list protects it, never "all columns except the labels".
  `--race-block` (below) is the deliberate override, done in one function
  (`encode_race`) so it is visible rather than implicit.
- `abs_dt`, the CXR<->mammogram gap, is kept as a covariate: it correlates
  negatively with all four outcomes (sicker patients see the health system
  more often, so their two studies land closer together), so it is a care
  intensity confounder the embeddings could otherwise be credited for.

Column prefixes downstream code selects on:
  demo_   age_at_index, bmi, bmi_missing, ethnicity indicators
  race_   the Race stratifier, one-hot, only if --race-block
  tech_   portable, abs_dt, screen -- acquisition/scheduling, not the patient
  mammo_  the mammography embedding block
  cxr_    the chest-X-ray embedding block (fusion arm only)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MAMMO_ENCODER_DEFAULT = "E"          # A..F available; E is two 768-d L2-normalised halves
CXR_ENCODER_DEFAULT = "e"            # matched letter, so mammo-vs-cxr is a modality comparison
CXR_NPZ_KEY = "global"               # (768,) pooled study vector; 'patches' (1369,768) unused

DX_MIN, DX_MAX = pd.Timestamp("1990-01-01"), pd.Timestamp("2025-12-31")
NOISE = {"--", "??", "Unspecified Diagnosis Code"}

OUTCOMES = ["t2dm", "htn", "hld", "ckd"]
WINDOW_DEFAULT = 730                 # |mammo - cxr| <= this; the only window, pre-specified

STRATIFIERS = ["Race", "index_date"]
NON_FEATURES = ["Race", "Age", "signed_dt", "density", "birads", "cxr_date",
                "index_mammo_date", "index_date", "anchor_is_cxr", "window_flag", "SOP"]
BASE_COVARIATES = ["age_at_index", "bmi", "bmi_missing", "portable", "abs_dt", "screen"]
ETH_REF = "Non-Hispanic"             # reference level, dropped from the one-hot block

DEMO_COVARIATES = {"age_at_index", "bmi", "bmi_missing"}
TECH_COVARIATES = {"portable", "abs_dt", "screen"}
DEMO_PREFIXES = ("eth_",)
RACE_COL = "Race"
RACE_MISSING_LEVEL = "Missing"
RACE_MIN_COUNT = 100


def code_masks(c):
    """ICD-9 + ICD-10 masks over a normalised (decimal-stripped, upper) code Series.

    t2dm: 250* excluding 5th digit 1/3 (type 1), plus E11*.
    htn:  40[1-5]*, I1[0-6]*.
    hld:  272[0-4]*, E78[0-5]*.
    ckd:  585*, N18*, Z992*, plus hypertensive CKD 40[34]*/I1[23]* -- those are
          CKD by definition (e.g. I12 hypertensive chronic kidney disease) and
          stay in htn too, since they genuinely are both.
    """
    t2dm = ((c.str.startswith("250") & ~(c.str.len().ge(5) & c.str[4].isin(["1", "3"])))
            | c.str.startswith("E11"))
    htn = c.str.match(r"^(40[1-5]|I1[0-6])")
    hld = c.str.match(r"^(272([0-4]|$)|E78([0-5]|$))")
    ckd = c.str.match(r"^(585|N18|Z992|40[34]|I1[23])")
    return {"t2dm": t2dm, "htn": htn, "hld": hld, "ckd": ckd}


# --------------------------------------------------------------------------- #
# 1. Load the two raw stores
# --------------------------------------------------------------------------- #

def load_raw(cxr_meta, cxr_icd, mammo_embed_dir, mammo_encoder):
    """
    Read the two metadata tables plus everything under --mammo-embed-dir.

    --mammo-embed-dir must contain magview.csv and
    embeddings/{ffdm,cview}/{encoder}_ffdm.parquet -- the layout the
    mammography embedding store is released in. --cxr-embed-dir (read
    separately, in build_cxr_block) is the corresponding per-patient npz root.
    """
    d = Path(mammo_embed_dir)
    cxr = pd.read_csv(cxr_meta)
    mag = pd.read_csv(d / "magview.csv")
    ffdm = pd.read_parquet(d / "embeddings" / "ffdm" / f"{mammo_encoder}_ffdm.parquet")
    cview = pd.read_parquet(d / "embeddings" / "cview" / f"{mammo_encoder}_cview.parquet")
    icd = pd.read_csv(cxr_icd)

    raw_dim = len(ffdm["emb"].iloc[0])
    print(f"CXR:     {len(cxr):>9,} images      {cxr['empi_anon'].nunique():>6,} patients")
    print(f"Magview: {len(mag):>9,} views       {mag['empi_anon'].nunique():>6,} patients")
    print(f"FFDM:    {len(ffdm):>9,} embeddings  {ffdm['empi_anon'].nunique():>6,} patients"
          f"   encoder {mammo_encoder}, {raw_dim}-d")
    print(f"CView:   {len(cview):>9,} embeddings  {cview['empi_anon'].nunique():>6,} patients")
    print(f"ICD:     {len(icd):>9,} rows        {icd['empi_anon'].nunique():>6,} patients")
    return cxr, mag, ffdm, cview, icd, raw_dim


# --------------------------------------------------------------------------- #
# 2. Eligibility, once, up front
# --------------------------------------------------------------------------- #

def eligible_patients(cxr, mag, ffdm, icd):
    """
    A patient is eligible only with a CXR, magview labels, an FFDM embedding,
    at least one usably dated ICD row, and Sex == 'F'. Fixed here, before any
    table is filtered, so every downstream frame stays mutually consistent.

    Returns the frozen patient index and the cleaned ICD frame (normalised
    code, parsed dx_date, a `dated` flag for plausible-date rows).
    """
    icd = icd[~icd["DIAGNOSIS_CODE"].astype(str).isin(NOISE)].copy()
    icd["code"] = icd["DIAGNOSIS_CODE"].astype(str).str.replace(".", "", regex=False).str.upper()
    icd["dx_date"] = pd.to_datetime(icd["DX_DATE_anon"], errors="coerce")
    icd["dated"] = icd["dx_date"].notna() & icd["dx_date"].between(DX_MIN, DX_MAX)

    with_icd = set(icd.loc[icd["dated"], "empi_anon"].unique())
    female = set(cxr.loc[cxr["Sex"] == "F", "empi_anon"].unique())
    imaging = set(cxr["empi_anon"]) & set(mag["empi_anon"]) & set(ffdm["empi_anon"])
    valid = imaging & with_icd & female

    patients = pd.Index(sorted(valid), name="empi_anon")
    print(f"\nCXR n mag n ffdm:        {len(imaging):,}")
    print(f"... n dated ICD:         {len(imaging & with_icd):,}")
    print(f"... n female (FINAL):    {len(patients):,}")
    print(f"males excluded here:     {len(imaging & with_icd) - len(patients):,}  "
          f"(screening-mammography cohort; too few to model, not the target population)")
    assert (cxr.loc[cxr["empi_anon"].isin(valid), "Sex"] == "F").all()
    return patients, icd


def check_ffdm_not_cview(ffdm, cview, mammo_encoder):
    """FFDM and CView are distinct embeddings at the same encoder, not duplicates.

    Only checked, never used past this: FFDM alone is the deliberate modality
    choice (CView also carries cancer_logit_* outcome-leakage columns)."""
    keys = ["empi_anon", "acc_anon", "side"]
    ffdm_s = ffdm.sort_values(keys).reset_index(drop=True)
    cview_s = cview.sort_values(keys).reset_index(drop=True)
    if not ffdm_s[keys].equals(cview_s[keys]):
        print("  WARNING: ffdm/cview row keys do not align; skipping the duplicate check")
        return
    e_ffdm = np.stack(ffdm_s["emb"].values).astype("float32")
    e_cview = np.stack(cview_s["emb"].values).astype("float32")
    identical = np.allclose(e_ffdm, e_cview)
    print(f"ffdm vs cview at encoder {mammo_encoder}: allclose={identical} "
          f"-> {'unexpected duplicate' if identical else 'distinct; using FFDM only'}")


# --------------------------------------------------------------------------- #
# 3. Index accession and anchor date
# --------------------------------------------------------------------------- #

def build_index_dates(cxr_f, mag_f, patients):
    """
    The index accession is the mammography accession nearest the CXR date,
    ties broken by lowest acc_anon for determinism. The anchor date is
    `max(mammo_date, cxr_date)`, the later of the two studies, so no imaging
    feature ever postdates the label anchor -- unlike anchoring on the
    mammogram alone, which would leave the CXR postdating the anchor for
    however many patients had the later CXR.
    """
    cxr_f = cxr_f.copy()
    mag_f = mag_f.copy()
    cxr_f["cxr_date"] = pd.to_datetime(cxr_f["StudyDate_anon"], errors="coerce")
    mag_f["mammo_date"] = pd.to_datetime(mag_f["study_date"], errors="coerce")

    mag_acc = mag_f.groupby(["empi_anon", "acc_anon"], as_index=False).agg(
        mammo_date=("mammo_date", "first"), screen=("screen", "max"))

    pairs = cxr_f[["empi_anon", "cxr_date"]].merge(mag_acc, on="empi_anon", how="inner")
    pairs["signed_dt"] = (pairs["mammo_date"] - pairs["cxr_date"]).dt.days
    pairs["abs_dt"] = pairs["signed_dt"].abs()

    nearest = (pairs.sort_values(["empi_anon", "abs_dt", "acc_anon"])
                    .groupby("empi_anon", as_index=False).first())
    nearest["index_date"] = nearest[["mammo_date", "cxr_date"]].max(axis=1)
    nearest["anchor_is_cxr"] = (nearest["index_date"].eq(nearest["cxr_date"])
                                 & nearest["signed_dt"].lt(0))
    nearest["days_cxr_to_index"] = (nearest["index_date"] - nearest["cxr_date"]).dt.days

    assert len(nearest) == len(patients)
    assert nearest["empi_anon"].is_unique
    assert (nearest["days_cxr_to_index"] == nearest["signed_dt"].clip(lower=0)).all()

    n_cxr = int(nearest["anchor_is_cxr"].sum())
    print(f"\nindex rows: {len(nearest):,}   "
          f"anchor is CXR (later study): {n_cxr:,} ({n_cxr / len(nearest) * 100:.1f}%)")
    return cxr_f, mag_f, nearest


# --------------------------------------------------------------------------- #
# 4. Anchored labels
# --------------------------------------------------------------------------- #

def build_labels(icd_f, nearest, patients):
    """
    Prevalence at the anchor: positive if any matching code has
    `dx_date <= index_date`. Every patient has >=1 dated ICD row by
    eligibility, so absence is a true negative, not a missing observation.
    """
    icd_f = icd_f.copy()
    nrst_map = nearest.set_index("empi_anon")
    icd_f["index_date"] = icd_f["empi_anon"].map(nrst_map["index_date"])

    masks_all = code_masks(icd_f["code"])
    keep = icd_f["dated"] & (icd_f["dx_date"] <= icd_f["index_date"])
    sub = icd_f[keep]
    sub_masks = {k: v[keep] for k, v in masks_all.items()}
    pos = {name: set(sub.loc[sub_masks[name], "empi_anon"]) for name in OUTCOMES}

    analysis = pd.DataFrame(index=patients)
    for name in OUTCOMES:
        analysis[name] = patients.isin(pos[name]).astype("int8")
    assert analysis[OUTCOMES].notna().all().all()

    print(f"\nprevalence at index (n={len(analysis):,}):")
    for name in OUTCOMES:
        print(f"  {name:>5}: {analysis[name].mean() * 100:>5.1f}%  ({analysis[name].sum():,})")

    ckd_pos = set(analysis.index[analysis["ckd"] == 1])
    htn_pos = set(analysis.index[analysis["htn"] == 1])
    if ckd_pos:
        print(f"  ckd/htn overlap: {len(ckd_pos & htn_pos) / len(ckd_pos) * 100:.1f}% of ckd "
              f"positives are also htn -- clinical comorbidity (HTN causes CKD), not a coding "
              f"artifact of the shared 40[34]/I1[23] codes; consider CKD-among-hypertensives as "
              f"a secondary analysis")
    return analysis


# --------------------------------------------------------------------------- #
# 5. Mammography vector
# --------------------------------------------------------------------------- #

def build_mammo_block(ffdm_f, nearest, patients, mammo_encoder, raw_dim):
    """
    Mean-pool the L/R FFDM embeddings of the index accession into one vector
    per patient, via an index-aligned join so a missing patient surfaces as
    NaN rather than silently shifting every row.

    Encoder E is not a plain vector: it is two 768-d blocks concatenated, each
    independently L2-normalised (half-norm 1.0, full norm sqrt(2)). Asserted
    here so a future encoder swap without that structure fails loudly rather
    than quietly changing what the features mean.
    """
    ffdm_idx = ffdm_f.merge(nearest[["empi_anon", "acc_anon"]], on=["empi_anon", "acc_anon"],
                             how="inner")
    raw = np.stack(ffdm_idx["emb"].values).astype("float32")
    if raw_dim == 1536:
        h1, h2 = raw[:, :768], raw[:, 768:]
        n1, n2 = np.linalg.norm(h1, axis=1), np.linalg.norm(h2, axis=1)
        assert np.allclose(n1, 1.0, atol=1e-4) and np.allclose(n2, 1.0, atol=1e-4), \
            "encoder E is expected to be two independently L2-normalised 768-d blocks"
        assert not np.allclose(h1, h2), "the two halves must not be duplicates"

    pooled = ffdm_idx.groupby("empi_anon")["emb"].apply(lambda s: np.mean(np.stack(s.values), axis=0))
    emb_dim = len(pooled.iloc[0])
    assert emb_dim == raw_dim, "pooling must not change dimensionality"

    mammo_cols = [f"mammo_{i:04d}" for i in range(emb_dim)]     # zero-padded: lexical == numeric order
    assert sorted(mammo_cols) == mammo_cols

    mammo = pd.DataFrame(np.stack(pooled.values), index=pooled.index,
                          columns=mammo_cols).astype("float32")
    mammo = mammo.reindex(patients)
    assert not mammo.isna().any().any(), "patient missing an index-accession embedding"

    pn = np.linalg.norm(mammo.values, axis=1)
    print(f"\nmammo block: {mammo.shape}   encoder {mammo_encoder}   "
          f"pooled norms {pn.min():.3f}-{pn.max():.3f}")
    return mammo, mammo_cols


# --------------------------------------------------------------------------- #
# 6. Covariates
# --------------------------------------------------------------------------- #

def build_covariates(analysis, cxr_f, mag_f, nearest, patients):
    """
    Age, Race, Ethnicity, bmi and portable come from the CXR record, so they
    are measured at the CXR date, not the anchor. age_at_index moves age onto
    the anchor timeline by adding the CXR-to-anchor gap, so it is never less
    than the raw CXR-time Age.

    Race is saved as a stratifier for disparity analysis and is not encoded
    here -- it stays a raw string, unusable as a numeric feature by accident.
    Promoting it to a feature is a separate, visible decision (encode_race,
    section 8).
    """
    cxr_pat = cxr_f.set_index("empi_anon").reindex(patients)
    nrst = nearest.set_index("empi_anon").reindex(patients)

    analysis["Age"] = cxr_pat["Age"].astype(float)
    analysis["age_at_index"] = (analysis["Age"] + nrst["days_cxr_to_index"] / 365.25).round(2)
    analysis["Race"] = cxr_pat["Race"]
    analysis["Ethnicity"] = cxr_pat["Ethnicity"]
    analysis["bmi"] = cxr_pat["BMI"].astype(float)
    analysis["portable"] = (cxr_pat["StudyDescription"].str.upper()
                             .str.contains("PORTABLE", na=False).astype("int8"))
    assert (cxr_pat["Sex"] == "F").all()     # female-only cohort; Sex itself is not saved

    analysis["cxr_date"] = cxr_pat["cxr_date"]
    analysis["index_mammo_date"] = nrst["mammo_date"]
    analysis["index_date"] = nrst["index_date"]
    analysis["anchor_is_cxr"] = nrst["anchor_is_cxr"].astype(bool)
    analysis["abs_dt"] = nrst["abs_dt"].astype(float)
    analysis["signed_dt"] = nrst["signed_dt"].astype(float)
    analysis["screen"] = nrst["screen"].astype(float)

    mag_idx = mag_f.merge(nearest[["empi_anon", "acc_anon"]], on=["empi_anon", "acc_anon"],
                           how="inner")
    mag_agg = mag_idx.groupby("empi_anon").agg(density=("density", "max"), birads=("birads", "max"))
    analysis["density"] = mag_agg["density"].reindex(patients).astype(float)
    analysis["birads"] = mag_agg["birads"].reindex(patients).astype(float)
    analysis["SOP"] = cxr_pat["SOP"]
    return analysis, nrst


def assemble_analysis_table(analysis, mammo, mammo_cols, window):
    """Join the mammo block on, flag the fit-cohort window, and check the anchor property."""
    analysis = analysis.join(mammo)
    analysis["window_flag"] = analysis["abs_dt"] <= window

    late_cxr = int((analysis["cxr_date"] > analysis["index_date"]).sum())
    late_mam = int((analysis["index_mammo_date"] > analysis["index_date"]).sum())
    assert late_cxr == 0 and late_mam == 0, "an imaging date fell after the label anchor"
    assert (analysis["index_date"] == analysis[["cxr_date", "index_mammo_date"]].max(axis=1)).all()
    assert [c for c in analysis.columns if c.startswith("mammo_")] == mammo_cols

    print(f"\nanalysis table: {analysis.shape}   "
          f"window_flag True: {analysis['window_flag'].sum():,} "
          f"({analysis['window_flag'].mean() * 100:.1f}%) -> the fit cohort")
    return analysis


# --------------------------------------------------------------------------- #
# 7. Model table -- abs_dt <= window, encoded, allow-listed
# --------------------------------------------------------------------------- #

def build_model_table(analysis, mammo_cols, window):
    """
    Cohort-filter and assemble the model matrix. Features are an explicit
    allow-list (covariates + mammo_cols), never "all columns except the
    labels" -- that idiom would silently pull in dates, SOP, unanchored Age,
    signed_dt, birads, density and Race.
    """
    m = analysis[analysis["abs_dt"] <= window].copy()

    m["bmi_missing"] = m["bmi"].isna().astype("int8")
    bmi_median = m["bmi"].median()
    m["bmi"] = m["bmi"].fillna(bmi_median)

    eth = pd.get_dummies(m["Ethnicity"], prefix="eth").drop(columns=[f"eth_{ETH_REF}"], errors="ignore")
    eth_cols = sorted(eth.columns)
    m = m.join(eth[eth_cols])

    covariates = BASE_COVARIATES + eth_cols
    features = covariates + mammo_cols
    out = m[OUTCOMES + STRATIFIERS + features]

    assert not out.drop(columns=STRATIFIERS).isna().any().any(), "no NaN may reach the model matrix"
    assert not set(features) & set(OUTCOMES) and not set(features) & set(STRATIFIERS)
    for bad in NON_FEATURES + ["Sex", "obesity", "Ethnicity"]:
        assert bad not in features, bad
    assert out[mammo_cols].dtypes.eq("float32").all()
    assert not pd.api.types.is_numeric_dtype(out["Race"])

    print(f"\ncohort: abs_dt <= {window} -> {len(out):,} patients "
          f"(from {len(analysis):,}, {len(out) / len(analysis) * 100:.1f}%)")
    print(f"features: {len(covariates)} covariates + {len(mammo_cols)} mammo = {len(features)}")
    print(f"bmi: {int(out['bmi_missing'].sum())} imputed at median {bmi_median:.1f}")

    for name in OUTCOMES:
        r = out["abs_dt"].corr(out[name])
        print(f"  abs_dt vs {name}: r = {r:+.4f}")
    return out, covariates, features


# --------------------------------------------------------------------------- #
# 8. Chest-X-ray block and fusion table
# --------------------------------------------------------------------------- #

def build_cxr_block(cxr_embed_dir, cxr_encoder, model_index, cxr_npz_key=CXR_NPZ_KEY):
    """
    Load the per-patient global CXR vector, one .npz per empi_anon under
    --cxr-embed-dir. Fusion is only honest on complete coverage of the fit
    cohort -- a partial join would make the CXR block a missingness indicator
    for "was this patient re-encoded", not a clinical signal -- so coverage is
    asserted before anything is loaded.
    """
    root = Path(cxr_embed_dir)
    avail = {int(p.name) for p in root.iterdir() if p.name.isdigit()}
    cov = len(set(model_index) & avail)
    print(f"\nCXR encoder {cxr_encoder.upper()}  {root}")
    print(f"  patients on disk: {len(avail):,}   in-window covered: {cov:,} / {len(model_index):,}")
    assert cov == len(model_index), "incomplete CXR coverage in-window -- do not build a fusion table"

    cxr_dim = None
    rows = {}
    for pid in model_index:
        with np.load(root / str(pid) / f"model-{cxr_encoder}_cxr.npz") as z:
            v = z[cxr_npz_key]
        if cxr_dim is None:
            cxr_dim = len(v)
        assert v.shape == (cxr_dim,), f"{pid}: expected ({cxr_dim},), got {v.shape}"
        rows[pid] = v

    cxr_cols = [f"cxr_{i:04d}" for i in range(cxr_dim)]
    assert sorted(cxr_cols) == cxr_cols

    cxr_emb = pd.DataFrame.from_dict(rows, orient="index", columns=cxr_cols).astype("float32")
    cxr_emb.index.name = model_index.name
    cxr_emb = cxr_emb.reindex(model_index)
    assert not cxr_emb.isna().any().any()
    assert np.isfinite(cxr_emb.values).all(), "non-finite value in the CXR block"

    n_norm = np.linalg.norm(cxr_emb.values, axis=1)
    print(f"  cxr block: {cxr_emb.shape}   norms {n_norm.min():.1f}-{n_norm.max():.1f}   "
          f"L2-normalised: {np.allclose(n_norm, 1.0, atol=1e-3)} -> stored raw, scaled in-fold")
    return cxr_emb, cxr_cols


def assemble_fusion_table(model, features, cxr_emb, cxr_cols):
    """Append the CXR block to the same rows, same order, mammo block first."""
    fusion_features = features + cxr_cols
    fusion = model.join(cxr_emb, how="left")

    assert fusion.index.equals(model.index)
    assert not fusion.drop(columns=STRATIFIERS).isna().any().any()
    for c in model.columns:
        assert fusion[c].equals(model[c]), c
    assert not set(fusion_features) & set(OUTCOMES) and not set(fusion_features) & set(STRATIFIERS)
    for bad in NON_FEATURES + ["Sex", "obesity", "Ethnicity"]:
        assert bad not in fusion_features, bad

    print(f"\nfusion: {fusion.shape}   features {len(features)} -> {len(fusion_features)} "
          f"(+{len(cxr_cols)} cxr)")
    return fusion, fusion_features


# --------------------------------------------------------------------------- #
# 9. Block encoding -- promote covariates and (optionally) Race to prefixed
#    feature blocks, the step formerly done by models/cohort.py on disk
# --------------------------------------------------------------------------- #

def encode_race(series, min_count=RACE_MIN_COUNT):
    """
    One-hot Race into a race_ feature block. Missing gets its own indicator
    rather than being folded into the reference level by get_dummies; levels
    below min_count are pooled into race_Other; the largest remaining level is
    the dropped reference, matching the Ethnicity convention.
    """
    raw = series.astype("object").where(series.notna(), RACE_MISSING_LEVEL).astype(str)
    counts = raw.value_counts()
    thin = [lvl for lvl, n in counts.items() if n < min_count and lvl != RACE_MISSING_LEVEL]
    level = raw.where(~raw.isin(thin), "Other")

    counts = level.value_counts()
    eligible = [lvl for lvl in counts.index if lvl != RACE_MISSING_LEVEL]
    if not eligible:
        raise ValueError("Race is missing for every patient; drop --race-block")
    reference = counts[eligible].idxmax()

    dummies = pd.get_dummies(level, prefix="race").drop(columns=[f"race_{reference}"]).astype("float32")
    return dummies, {"reference": reference, "n_missing": int(series.isna().sum()),
                      "levels": list(dummies.columns)}


def classify_covariate(name, extra_demo=(), extra_tech=()):
    if name in DEMO_COVARIATES or name in extra_demo or name.startswith(DEMO_PREFIXES):
        return "demo"
    if name in TECH_COVARIATES or name in extra_tech:
        return "tech"
    raise ValueError(f"covariate '{name}' is not classified; add it to DEMO_COVARIATES / "
                      f"TECH_COVARIATES, or pass --demo-extra {name} / --tech-extra {name}")


def to_modelling_table(df, covariates, embedding_cols, race_min_count=RACE_MIN_COUNT,
                        with_race=False, extra_demo=(), extra_tech=()):
    """
    Rename covariates into their demo_/tech_ blocks and, if requested,
    one-hot Race into its own race_ block. Embedding columns pass through
    unchanged.
    """
    rename = {c: f"{classify_covariate(c, extra_demo, extra_tech)}_{c}" for c in covariates}
    out = df[OUTCOMES + ["Race", "index_date"] + covariates + embedding_cols].rename(columns=rename)

    race_info = None
    if with_race:
        race_block, race_info = encode_race(df["Race"], race_min_count)
        out = out.join(race_block)

    prefixes = ("demo", "race", "tech", "mammo", "cxr")
    blocks = {p: [c for c in out.columns if c.startswith(p + "_")] for p in prefixes}
    print("\nblocks")
    for name, members in blocks.items():
        if members:
            print(f"  {name:6s} {len(members):>5}")
    if race_info:
        print(f"  race promoted to feature: reference {race_info['reference']!r} dropped, "
              f"{race_info['n_missing']} missing -> race_Missing")

    for prefix in prefixes:
        for c in blocks[prefix]:
            if not pd.api.types.is_numeric_dtype(out[c]):
                raise TypeError(f"{c} is {out[c].dtype}, not numeric")
    return out, blocks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_cohort(mammo_embed_dir, cxr_embed_dir, cxr_meta, cxr_icd,
                  mammo_encoder=MAMMO_ENCODER_DEFAULT, cxr_encoder=CXR_ENCODER_DEFAULT,
                  window=WINDOW_DEFAULT, mammo_only=False, with_race=False,
                  race_min_count=RACE_MIN_COUNT, extra_demo=(), extra_tech=()):
    """Raw sources -> the final block-prefixed modelling table(s).

    Returns {'mammo': df} or {'mammo': df, 'fusion': df}, each already in the
    layout linear.py / nonlinear.py read (demo_/race_/tech_/mammo_/cxr_).
    """
    cxr, mag, ffdm, cview, icd, raw_dim = load_raw(cxr_meta, cxr_icd, mammo_embed_dir, mammo_encoder)
    patients, icd = eligible_patients(cxr, mag, ffdm, icd)

    cxr_f = cxr[cxr["empi_anon"].isin(patients)].copy()
    mag_f = mag[mag["empi_anon"].isin(patients)].copy()
    ffdm_f = ffdm[ffdm["empi_anon"].isin(patients)].copy()
    cview_f = cview[cview["empi_anon"].isin(patients)].copy()
    icd_f = icd[icd["empi_anon"].isin(patients)].copy()
    assert cxr_f.groupby("empi_anon")["StudyDate_anon"].nunique().eq(1).all()
    assert cxr_f["empi_anon"].is_unique

    check_ffdm_not_cview(ffdm_f, cview_f, mammo_encoder)
    cxr_f, mag_f, nearest = build_index_dates(cxr_f, mag_f, patients)
    analysis = build_labels(icd_f, nearest, patients)
    mammo, mammo_cols = build_mammo_block(ffdm_f, nearest, patients, mammo_encoder, raw_dim)
    analysis, _ = build_covariates(analysis, cxr_f, mag_f, nearest, patients)
    analysis = assemble_analysis_table(analysis, mammo, mammo_cols, window)

    model, covariates, features = build_model_table(analysis, mammo_cols, window)
    mammo_out, _ = to_modelling_table(model, covariates, mammo_cols, race_min_count,
                                       with_race, extra_demo, extra_tech)
    tables = {"mammo": mammo_out}

    if not mammo_only:
        cxr_emb, cxr_cols = build_cxr_block(cxr_embed_dir, cxr_encoder, model.index)
        fusion, _ = assemble_fusion_table(model, features, cxr_emb, cxr_cols)
        fusion_out, _ = to_modelling_table(fusion, covariates, mammo_cols + cxr_cols,
                                            race_min_count, with_race, extra_demo, extra_tech)
        tables["fusion"] = fusion_out

    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mammo-embed-dir", required=True,
                     help="directory containing magview.csv and "
                          "embeddings/{ffdm,cview}/{encoder}_ffdm.parquet")
    ap.add_argument("--cxr-embed-dir", required=True,
                     help="directory holding one <empi_anon>/model-<encoder>_cxr.npz per patient")
    ap.add_argument("--cxr-meta", required=True, help="CXR metadata csv (Age/Race/Ethnicity/BMI/...)")
    ap.add_argument("--cxr-icd", required=True, help="ICD diagnosis codes csv")
    ap.add_argument("--mammo-encoder", default=MAMMO_ENCODER_DEFAULT)
    ap.add_argument("--cxr-encoder", default=CXR_ENCODER_DEFAULT)
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                     help="cohort cut: |mammo - cxr date| <= this many days")
    ap.add_argument("--mammo-only", action="store_true", help="skip the CXR block and fusion table")
    ap.add_argument("--race-block", dest="with_race", action="store_true",
                     help="promote Race from stratifier to its own feature block")
    ap.add_argument("--race-min-count", type=int, default=RACE_MIN_COUNT)
    ap.add_argument("--demo-extra", nargs="*", default=[])
    ap.add_argument("--tech-extra", nargs="*", default=[])
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    built = build_cohort(
        args.mammo_embed_dir, args.cxr_embed_dir, args.cxr_meta, args.cxr_icd,
        mammo_encoder=args.mammo_encoder, cxr_encoder=args.cxr_encoder, window=args.window,
        mammo_only=args.mammo_only, with_race=args.with_race, race_min_count=args.race_min_count,
        extra_demo=set(args.demo_extra), extra_tech=set(args.tech_extra))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in built.items():
        path = out / f"cohort_{name}.parquet"
        df.to_parquet(path)
        rt = pd.read_parquet(path)
        assert rt.shape == df.shape and rt.index.equals(df.index), "round-trip check failed"
        print(f"\nwrote {path}  {df.shape}")


if __name__ == "__main__":
    main()

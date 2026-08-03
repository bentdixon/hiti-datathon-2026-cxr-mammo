# Data

**No patient data, imaging, or embeddings are stored in this repository.**
Every path this pipeline reads is passed in at the command line
(`--mammo-embed-dir`, `--cxr-embed-dir`, `--cxr-meta`, `--cxr-icd`, or
`--cohort`) and none of it is committed, vendored, or cached to disk by any
script here. `.gitignore` also excludes every `*.parquet`/`*.csv` result file
this pipeline produces, so a clone of this repository ships code and
documentation only.

## Source cohort

The cohort is drawn from two paired Emory Healthcare imaging collections,
screening-mammography patients matched to a chest radiograph within the same
episode of care (see `cohort.py` for the exact eligibility and anchoring
rules — female-only, at least one dated diagnosis code, and a mammography
study within `--window` days of the CXR).

**Mammography — EMBED.** The mammography side is the *EMory BrEast imaging
Dataset* (EMBED): 3.4 million screening and diagnostic mammography images from
about 110,000 patients across four Emory-affiliated hospitals, 2013–2020, with
approximately equal Black and white representation and linked structured
imaging descriptors (BI-RADS, density) and pathologic outcomes. It is
maintained by the Emory HITI (Health Innovation and Translational Informatics)
Lab.

- Paper: Jeong et al., *The EMory BrEast imaging Dataset (EMBED): A Racially
  Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic
  Mammographic Images*, Radiology: Artificial Intelligence, 2023.
  https://pubs.rsna.org/doi/10.1148/ryai.220047
- A 20% public subset is released through the AWS Open Data Program, for
  non-commercial research use:
  https://registry.opendata.aws/emory-breast-imaging-dataset-embed/
- Data descriptor, sample notebooks, and access instructions:
  https://github.com/Emory-HITI/EMBED_Open_Data

**Chest X-ray — Emory-CXR.** The chest-radiograph side is a private Emory
Healthcare collection (reported elsewhere as ~226,000 studies from ~58,000
patients across five hospitals in the Emory system, 2019–2020), re-encoded by
the HITI Lab specifically to cover this mammography cohort by patient
identifier rather than through the smaller, disjoint `DS{1-6}_patient_*`
namespace used by any public Emory-CXR release. This re-encoding is what
`--cxr-embed-dir` and `--cxr-meta`/`--cxr-icd` point at; it is not itself a
public dataset, and this repository does not redistribute it.

Both embedding stores are read only through `cohort.py`'s CLI (see
[input contract](#input-contract-mammo-embed-dir---cxr-embed-dir) below).

## CHoRUS

Emory University is one of the fourteen data-acquisition sites in **CHoRUS**
(the "Collaborative Hospital Repository Uniting Standards for Equitable AI"),
one of four NIH Bridge2AI data-generation consortia. CHoRUS is building a
separate, much larger multimodal dataset — EHR (OMOP-standardised), radiology,
waveform, and clinical-note data from ICU/PICU/NICU admissions across its
member hospitals — accessed exclusively under a signed data-use agreement; and as of its
current release radiology imaging is explicitly limited.

- Consortium site: https://chorus4ai.org/
- Bridge2AI project page: https://bridge2ai.org/data-chorus/

## Input contract: `--mammo-embed-dir` / `--cxr-embed-dir`

`cohort.py` expects the following on disk, read only from the paths
passed at the command line:

```
<mammo-embed-dir>/
  magview.csv                              one row per (empi_anon, acc_anon):
                                            study_date, screen, density, birads
  embeddings/ffdm/<ENCODER>_ffdm.parquet   one row per (empi_anon, acc_anon, side):
                                            emb -- a length-D float vector
  embeddings/cview/<ENCODER>_cview.parquet same schema as ffdm, used only for
                                            the one-time FFDM-vs-CView check

<cxr-embed-dir>/
  <empi_anon>/model-<encoder>_cxr.npz      one directory per patient; the npz
                                            holds 'global' (768,), 'patches'
                                            (1369, 768, unused) and 'grid' (2,)

--cxr-meta   csv: empi_anon, Sex, StudyDate_anon, Age, Race, Ethnicity, BMI,
             StudyDescription, SOP -- one row per CXR study
--cxr-icd    csv: empi_anon, DIAGNOSIS_CODE, DX_DATE_anon -- one row per
             diagnosis code
```

`empi_anon` is the join key across every table; `acc_anon` identifies a
mammography accession. `<ENCODER>` (`--mammo-encoder`, default `E`) and
`<encoder>` (`--cxr-encoder`, default `e`) select which embedding model's
output is read — both letters must exist in the corresponding store. See
`cohort.py`'s module docstring for the full eligibility, anchoring and
labelling rules applied on top of these tables.

## Standing in for the data

Every entry point in this pipeline (`main.py`, `linear.py`,
`nonlinear.py`) falls back to an in-memory synthetic cohort of the same
shape when no data location is given, so the whole pipeline — cohort build
through both modelling arms — can be exercised without access to any of the
above. See the Run section of the top-level `README.md`.

`simulate.py` writes that same synthetic cohort to a parquet file instead
of holding it in memory, so the demo path and the real-data path become the
same shape of command:

```
uv run mammocxr-simulate --out cohort_simulated.parquet
uv run mammocxr-linear --cohort cohort_simulated.parquet --outdir results
uv run mammocxr-nonlinear --cohort cohort_simulated.parquet --outdir results
```

`--n`, `--d-mammo`, `--d-cxr` and `--seed` control the simulated cohort's size,
embedding widths, and reproducibility.

"""
Linear baseline for the mammography + CXR fusion question.

Design: prevalence at the time of imaging, not prediction. The index date is
the later of the two image dates, max(mammography date, chest radiograph date),
and the label is whether the risk factor is already coded for that patient as
of the index date. This is a detection study. Nothing here forecasts onset, and
no result should be described as risk of developing a condition.

Two consequences follow, and both are load-bearing:

  Denominators. Prevalent patients are the positives, so there is no
  outcome-specific exclusion. See eligible_mask.

  Confounding. Under a prediction design the acquisition covariates were a
  nuisance to adjust for. Under a detection design they are a shortcut: a
  portable film means an inpatient, an inpatient is sick, and a model can score
  well on prevalent comorbidity without reading any anatomy. The headline
  comparison is therefore demo+tech versus demo+tech+images, never a bare
  image block against nothing.

Blocks compared per outcome, where clinical = demo+race+tech:
  demo, race, demo+race, tech, demo+tech, clinical, mammo, cxr, mammo+cxr,
  clinical+mammo, clinical+cxr, clinical+mammo+cxr

Race and ethnicity are NOT features by default -- see DEFAULT_DEMOGRAPHICS.
They are analysed post hoc instead: the raw Race column rides along unprefixed,
in no block, and subgroups() reports per-group AUROC, AUPRC and calibration on
the same rows and the same out-of-fold predictions. `--demographics` controls
which demographic groups may be used as inputs, so the race and ethnicity arms
remain available for a sensitivity analysis without editing any block.

Blocks naming a disabled group are reduced rather than dropped, and names that
then coincide are fit once (clinical becomes age+bmi+tech when race and eth are
held out). Every reduction is printed, and performance.csv records the groups
actually fit, so a block label can never overstate what went into it.

Every block is an L2-penalised logistic regression. Each block is standardised
separately before concatenation, so a block with larger raw norms cannot absorb
the penalty budget. The penalty strength C is chosen by inner cross-validation
inside each outer fold, so no test patient influences its own model.

Reported per block: AUROC, AUPRC (with lift over prevalence), Brier score,
calibration slope and intercept. Reported per comparison: delta AUROC with a
paired bootstrap 95% interval, computed on the pooled out-of-fold predictions.
Reported per subgroup, across five stratifiers -- age (<50 / 50-64 / 65+), BMI
(<30 / >=30), race, imaging year (<=2017 / >2017) and the mammogram-to-CXR gap
(<=90d / 91-365d / 366-720d / >720d) -- AUROC with a bootstrap interval, AUPRC
against that subgroup's own prevalence, the calibration intercept, and the
within-subgroup fusion delta, which is the primary quantity of the three.

The subgroup analysis runs on one fitted block, the mammo+cxr imaging model. See
DEFAULT_SUBGROUP_BLOCKS and subgroup_deltas() for why it is deliberately narrow,
and subgroups() for why within-subgroup AUROC cannot be compared across levels
of a stratifier that is itself predictive.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .common import (DATE_COLS, DEFAULT_DEMOGRAPHICS, DEMOGRAPHIC_GROUPS,
                     EMBEDDING_GROUPS, MISSING_LEVEL, OUTCOMES, RESTRICTIONS, _levels,
                     apply_restriction, assign_groups, block_columns, cap_index_gap,
                     delta_auc, eligible_mask, gains_table, metrics, simulate, subgroups)

# Groups whose absence is normal rather than a degraded fit.
OPTIONAL_GROUPS = {"demo_other"}

# 'demo' stays available as the age + bmi + ethnicity bundle so the older
# comparisons still read the same way, but it is now composed of the finer
# groups rather than being the unit of selection.
DEMO = ["age", "bmi", "eth", "demo_other"]

# 'tech' is fit on its own because the acquisition covariates alone put a floor
# under every image block: whatever AUROC three columns of care-setting metadata
# reach is the part of prevalence detection that needs no imaging at all.
#
# 'race' and 'eth' appear in CLINICAL and in the blocks below so the sensitivity
# analysis is one flag away, but they are disabled by default and every block
# naming them silently reduces. Under the default they contribute nothing.
CLINICAL = DEMO + ["race", "tech"]

BLOCKS = {
    "age": ["age"],
    "bmi": ["bmi"],
    "race": ["race"],
    "age+bmi": ["age", "bmi"],
    "age+bmi+race": ["age", "bmi", "race"],
    "demo": DEMO,
    "demo+race": DEMO + ["race"],
    "tech": ["tech"],
    "clinical": CLINICAL,
    "mammo": ["mammo"],
    "cxr": ["cxr"],
    "mammo+cxr": ["mammo", "cxr"],
    # The demographics-adjusted probes. 'demo' is age + bmi (+ any demo_other)
    # under the default --demographics, so these are the imaging blocks with the
    # two clinical covariates and without the acquisition covariates that
    # 'clinical' adds. That distinction is what makes demo+mammo+cxr the natural
    # headline model: the tech block is a care-intensity proxy, so a gain over
    # 'clinical' is a gain over knowing how sick the patient's care pattern says
    # they are, which is a different and harder claim than a gain over age and BMI.
    "demo+mammo": DEMO + ["mammo"],
    "demo+cxr": DEMO + ["cxr"],
    "demo+mammo+cxr": DEMO + ["mammo", "cxr"],
    "clinical+mammo": CLINICAL + ["mammo"],
    "clinical+cxr": CLINICAL + ["cxr"],
    "clinical+mammo+cxr": CLINICAL + ["mammo", "cxr"],
}

# clinical+mammo+cxr is retained because it is the only block that answers
# whether CXR adds anything once demographics, race, acquisition and
# mammography are already in the model.
COMPARISONS = [
    ("mammo", "mammo+cxr"),
    ("cxr", "mammo+cxr"),
    # The three requested features, isolated one at a time.
    ("age", "age+bmi"),
    ("age+bmi", "age+bmi+race"),
    ("demo", "demo+race"),
    ("demo+race", "clinical"),
    ("demo", "demo+mammo+cxr"),
    ("demo+mammo", "demo+mammo+cxr"),
    ("demo+cxr", "demo+mammo+cxr"),
    ("clinical", "clinical+mammo"),
    ("clinical", "clinical+cxr"),
    ("clinical+mammo", "clinical+mammo+cxr"),
    ("clinical", "clinical+mammo+cxr"),
]

# The comparisons carried down into the subgroup analysis. Deliberately a subset:
# every pair costs a paired bootstrap per level per stratifier per outcome, and
# more importantly each one multiplies the number of intervals a reader can go
# fishing in. These three are the study's question -- does the second modality
# add anything, unadjusted and adjusted -- so they are the three pre-specified
# here rather than chosen once the subgroup tables exist.
SUBGROUP_COMPARISONS = [
    ("mammo", "mammo+cxr"),
    ("cxr", "mammo+cxr"),
    ("demo+mammo", "demo+mammo+cxr"),
    ("demo+cxr", "demo+mammo+cxr"),
    ("clinical+mammo", "clinical+mammo+cxr"),
]

# Which fitted blocks the subgroup analysis reports on. One block by default --
# the imaging model, a linear probe on the two embeddings and nothing else.
#
# That choice makes the subgroup tables a statement about the images. An
# adjusted block would fold age and BMI into the same predictions, so a bracket
# where performance dropped could not be attributed to the embeddings rather
# than to the covariates carrying less of the signal there.
#
# This is a reporting filter, not a fitting filter: use --blocks to change what
# gets fit. Keeping it to one block is the point. Per-group metrics are free,
# but every extra block multiplies the number of intervals in the tables, and a
# reader scanning fifteen blocks x five stratifiers x four outcomes for the
# subgroup where something reaches significance will always find one.
DEFAULT_SUBGROUP_BLOCKS = ("mammo+cxr",)


def _rescale(X, k):
    """Module-level so the FunctionTransformer stays picklable under n_jobs=-1."""
    return X * k


def make_pipeline(df, groups, class_weight=None, clinical_scale=1.0):
    """
    One StandardScaler per group, then a single penalised logistic regression.

    Scaling per group rather than per column changes nothing numerically --
    StandardScaler is per-column either way -- but it keeps the transformer
    layout aligned with the group layout, so a block's coefficients can be split
    back apart by group without recomputing anything.

    clinical_scale multiplies the standardised non-embedding features by a
    constant. One L2 budget is shared across every column, so in a fused block
    a handful of clinical features compete with thousands of embedding
    dimensions and get flattened -- measurably so: at 1,280 embedding columns
    BMI kept 3% of the coefficient it had in the clinical-only fit. Scaling a
    feature up by k means it needs a coefficient k times smaller for the same
    effect, so its penalty contribution falls by k squared. clinical_scale=1.0
    is the default and changes nothing.

    class_weight is None by default. Setting it to "balanced" raises AUPRC on the
    rare outcomes but destroys calibration: predicted probabilities no longer
    reflect prevalence, so Brier and calibration slope become uninterpretable.
    Use it only if you drop those two metrics.
    """
    assigned = assign_groups(df)
    transformers = []
    for g in groups:
        if not assigned.get(g):
            continue
        step = StandardScaler()
        if clinical_scale != 1.0 and g not in EMBEDDING_GROUPS:
            step = Pipeline([("std", step),
                             ("relief", FunctionTransformer(_rescale,
                                                            kw_args={"k": clinical_scale}))])
        transformers.append((g, step, assigned[g]))
    return Pipeline([
        ("scale", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(max_iter=5000, class_weight=class_weight)),
    ])


def crossval_predict(df, y, groups, n_outer=5, n_inner=3, seed=0, Cs=None, class_weight=None,
                     clinical_scale=1.0):
    """
    Pooled out-of-fold predicted probabilities, log-odds and coefficients.

    Coefficients are averaged over the outer folds and returned on the
    standardised scale, so they are directly comparable across features within
    a block and across values of clinical_scale. They exist to answer a question
    the AUROC cannot: whether age, BMI and race are still doing anything once
    2,300 embedding columns are sharing the same L2 budget, or whether the
    penalty has flattened them to nothing.
    """
    Cs = np.logspace(-4, 1, 12) if Cs is None else Cs
    cols = block_columns(df, groups)
    X = df[cols]
    proba = np.empty(len(y))
    logit = np.empty(len(y))
    chosen, coefs = [], []

    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        search = GridSearchCV(
            make_pipeline(df, groups, class_weight, clinical_scale),
            {"clf__C": Cs},
            cv=StratifiedKFold(n_inner, shuffle=True, random_state=seed),
            scoring="neg_log_loss",
            n_jobs=-1,
        )
        search.fit(X.iloc[tr], y[tr])
        best = search.best_estimator_
        proba[te] = best.predict_proba(X.iloc[te])[:, 1]
        logit[te] = best.decision_function(X.iloc[te])
        chosen.append(search.best_params_["clf__C"])
        coefs.append(best.named_steps["clf"].coef_[0])

    coef = pd.Series(np.mean(coefs, axis=0), index=cols, name="coef")
    if clinical_scale != 1.0:
        # A clinical feature was fed in as k * x_std, so its contribution is
        # (w * k) * x_std. Undo that here rather than at every read site, so a
        # reported coefficient always means the same thing.
        lookup = {c: g for g, cs in assign_groups(df).items() for c in cs}
        boosted = [c for c in cols if lookup.get(c) not in EMBEDDING_GROUPS]
        coef.loc[boosted] *= clinical_scale
    return proba, logit, chosen, coef


# ---------------------------------------------------------------------------
# Post-hoc subgroup stratifiers
#
# Every entry is a reporting axis only. Nothing here is ever a feature, and each
# is derived from the raw columns rather than the standardised design matrix, so
# the brackets mean what their labels say. That also means a stratifier stays
# available when the matching feature group is switched off: under the default
# --demographics age bmi, race and ethnicity are absent from every model and
# still fully reported here, which is the whole point of holding them out.
#
# Cutpoints are fixed and pre-specified rather than data-driven quantiles.
# Quantile bins move with the cohort, so a tertile computed on the full cohort
# and a tertile computed on the CKD-among-hypertensives denominator are not the
# same subgroup, and the two analyses could not be read side by side.
# ---------------------------------------------------------------------------

AGE_BINS = (0, 50, 65, 200)
AGE_LABELS = ("<50", "50-64", "65+")

# One cut at the WHO obesity threshold rather than the full five-class ladder.
# Two levels of roughly 2,600 patients each estimate a within-group AUROC that
# means something; five classes would put 'underweight' under 200 patients and
# below subgroup_min_n on every outcome, so it would be tabulated and never
# scored.
BMI_BINS = (0, 30, 200)
BMI_LABELS = ("<30", ">=30")

# Imaging year, from index_date = max(mammo_date, cxr_date).
#
# READ THE WARNING PRINTED BY THIS STRATIFIER BEFORE USING IT. merge_datasets.py
# excludes cxr_date, index_mammo_date and index_date from the feature set with
# the note "calendar dates are date-shifted; intervals only are valid". The dates
# in the container are *_anon and carry a de-identification offset, so the year
# read off index_date is not necessarily the year the image was taken and 2017
# is not necessarily a real calendar boundary. If the offset is a single global
# shift the split is still a valid before/after ordering with a mislabelled
# cutpoint; if it is per-patient the split is noise. The two cases are
# distinguishable from the printed date range: a cohort imaged over a few years
# that spans decades after shifting has been shifted per patient.
YEAR_CUT = 2017
YEAR_LABELS = (f"<={YEAR_CUT}", f">{YEAR_CUT}")

# Mammogram-to-CXR gap, in days, from the raw tech_abs_dt column.
#
# Two things ride on this axis. The mechanical one: the further apart the two
# studies, the staler the earlier modality is relative to the index date, so if
# fusion is real its gain should be largest in the tightest bracket. The
# confounding one: merge_datasets.py measures abs_dt as negatively correlated
# with all four outcomes, because sicker patients are in contact with the system
# more often and get imaged closer together, which makes the gap a proxy for
# care intensity. A fusion gain concentrated in the narrow bracket is therefore
# consistent with both stories and does not by itself distinguish them.
#
# A '>720d' level is included rather than folded into the top bracket. The
# cohort is capped at 730 days, so a handful of patients sit between 721 and 730
# and would otherwise be silently dropped from every row of the table.
GAP_BINS = (-1, 90, 365, 720, 10 ** 6)
GAP_LABELS = ("<=90d", "91-365d", "366-720d", ">720d")


_ANNOUNCED = set()


def _announce_once(key):
    """True the first time a given message key is seen in this process."""
    if key in _ANNOUNCED:
        return False
    _ANNOUNCED.add(key)
    return True


def _binned(df, col, bins, labels, right=False):
    if col not in df.columns:
        return None
    v = pd.to_numeric(df[col], errors="coerce")
    cut = pd.cut(v, bins=list(bins), labels=list(labels), right=right)
    lab = cut.astype(object).where(cut.notna(), MISSING_LEVEL).astype(str)
    return pd.Series(lab.to_numpy()), tuple(labels) + (MISSING_LEVEL,)


def _raw(df, col):
    if col not in df.columns:
        return None
    s = pd.Series(df[col]).astype(object)
    s = s.where(s.notna(), MISSING_LEVEL).astype(str).str.strip()
    return pd.Series(s.where(s != "", MISSING_LEVEL).to_numpy()), None


def _imaging_year(df, cut=YEAR_CUT):
    """
    Before/after split on the index date's calendar year, with the shift caveat.

    Falls back through DATE_COLS so the axis still works if a cohort table kept
    only the CXR date. The fallback is announced, because index_date and cxr_date
    are the same year for most patients but not all, and a silent substitution
    would make two runs disagree for no visible reason.
    """
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        d = pd.to_datetime(df[col], errors="coerce")
        if d.notna().sum() == 0:
            continue
        yr = d.dt.year
        span = f"{d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}"
        note = "" if col == DATE_COLS[0] else f", using '{col}' as '{DATE_COLS[0]}' is absent"
        # Labels are rebuilt once per outcome, on that outcome's filtered rows.
        # The caveat is the same every time and printing it eight times buries it.
        if _announce_once(f"imaging_year:{col}"):
            print(f"  imaging_year: cut at {cut} on '{col}' spanning {span}{note}.\n"
                  f"    CAUTION: these are date-shifted *_anon dates. If the shift is per "
                  f"patient, this axis is\n    not calendar time and should be dropped; if it is "
                  f"one global offset, the ordering holds\n    but '{cut}' is not the real year. "
                  f"Check the span above against how long the cohort was\n    actually collected "
                  f"before reading anything into it.", flush=True)
        lab = np.where(yr.isna(), MISSING_LEVEL,
                       np.where(yr <= cut, YEAR_LABELS[0], YEAR_LABELS[1]))
        return pd.Series(lab), YEAR_LABELS + (MISSING_LEVEL,)
    return None


STRATIFIERS = {
    "age": lambda df, race_col="Race": _binned(df, "demo_age_at_index", AGE_BINS, AGE_LABELS),
    "bmi": lambda df, race_col="Race": _binned(df, "demo_bmi", BMI_BINS, BMI_LABELS),
    "race": lambda df, race_col="Race": _raw(df, race_col),
    "imaging_year": lambda df, race_col="Race": _imaging_year(df),
    "index_gap": lambda df, race_col="Race": _binned(df, "tech_abs_dt", GAP_BINS, GAP_LABELS,
                                                     right=True),
}

DEFAULT_STRATIFIERS = tuple(STRATIFIERS)


def stratifier_labels(df, names=None, race_col="Race"):
    """
    Yield (name, labels, order) for each requested stratifier this table supports.

    Silently skipping an unavailable axis is the wrong default -- a missing BMI
    column would quietly turn six subgroup analyses into five -- so the caller
    gets the list of what was dropped and prints it.
    """
    names = DEFAULT_STRATIFIERS if names is None else tuple(names)
    found, absent = [], []
    for name in names:
        if name not in STRATIFIERS:
            raise SystemExit(f"unknown stratifier {name!r}; choose from {list(STRATIFIERS)}")
        got = STRATIFIERS[name](df, race_col)
        if got is None:
            absent.append(name)
            continue
        labels, order = got
        # A stratifier with one level is not a stratification; reporting it
        # duplicates the pooled row and pads the table.
        if labels.nunique() < 2:
            absent.append(name)
            continue
        found.append((name, labels, order))
    return found, absent


def subgroup_report(y, proba, logit, strat, min_n=100, n_boot=0, seed=0):
    """Long table: one row per (stratifier, level), stacked across stratifiers."""
    out = []
    for name, labels, order in strat:
        s = subgroups(y, proba, logit, labels, min_n, n_boot, seed, order)
        s.insert(0, "stratifier", name)
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def subgroup_deltas(y, p_small, p_large, strat, min_n=100, n_boot=1000, seed=0):
    """
    Within-subgroup delta AUROC, paired bootstrap on that subgroup's rows only.

    This, not the per-group AUROC, is the primary subgroup quantity. Both arms
    are scored on the same patients, so the case-mix compression that depresses
    within-bracket AUROC hits each equally and cancels out of the difference.
    A per-group AUROC of 0.62 in the 70+ bracket says little on its own; a fusion
    gain of +0.04 there against +0.00 under 50 is a subtyping result.

    The intervals are unadjusted for multiplicity and there are many of them
    (stratifiers x levels x comparisons x outcomes). Treat the pre-specified
    fusion comparisons as the analysis and everything else as description; an
    interval that just clears zero in one bracket of one outcome is what
    multiplicity produces on its own.
    """
    y = np.asarray(y)
    rows = []
    for name, labels, order in strat:
        g = labels.to_numpy()
        for lev in _levels(g, order):
            m = g == lev
            yy = y[m]
            n_pos = int(yy.sum())
            row = {"stratifier": name, "group": lev, "n": int(m.sum()),
                   "n_prevalent": n_pos,
                   "prevalence": float(yy.mean()) if m.any() else float("nan"),
                   "delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                   "p_gt_0": float("nan"), "note": ""}
            if m.sum() >= min_n and 0 < n_pos < m.sum():
                row.update(delta_auc(yy, p_small[m], p_large[m], n_boot, seed))
            else:
                row["note"] = f"n<{min_n} or single-class; not estimated"
            rows.append(row)
    return pd.DataFrame(rows)


def named_coefficients(coef, df, drop_embeddings=True):
    """
    Fitted coefficients labelled with the group each feature belongs to.

    Embedding weights are dropped by default: 2,300 of them say nothing a reader
    can use, while the handful of clinical weights are the ones that show
    whether age, BMI and race survived the shared penalty.
    """
    lookup = {c: g for g, cols in assign_groups(df).items() for c in cols}
    rows = [{"group": lookup.get(c, "unassigned"), "feature": c, "coef": float(v)}
            for c, v in coef.items()
            if not (drop_embeddings and lookup.get(c) in EMBEDDING_GROUPS)]
    return pd.DataFrame(rows)


def drop_empty_blocks(df, blocks, comparisons, demographics=DEFAULT_DEMOGRAPHICS):
    """
    Reduce each block to the groups this table has and this run enables.

    A demographic group is usable only if it has columns *and* it was requested
    in `demographics`. Everything else -- tech, the embeddings -- is usable
    whenever it has columns. Disabling a demographic is therefore handled by the
    same path as a column that was never built, which is why race can be dropped
    from the feature set without touching any block definition.

    A block is dropped only when none of its groups are usable. When some
    are, it is fit on those. Dropping the whole block instead would mean that
    excluding race -- the default -- deletes clinical and all three
    clinical+image blocks at once, leaving only the unadjusted ones. Excluding a
    demographic should cost that adjustment, not the entire adjusted arm.

    Reduction can collapse two names onto the same feature set: with no race
    block, 'clinical' is exactly 'demo+tech'. The duplicate is dropped rather
    than fit twice, comparisons naming it are redirected to the surviving name,
    and a comparison whose two sides collapse together is dropped because its
    delta is zero by construction.
    """
    def usable(g):
        if not block_columns(df, [g]):
            return False
        return g not in DEMOGRAPHIC_GROUPS or g in demographics

    # A group can be unusable for two quite different reasons, and conflating
    # them hides a real problem: a column that was never built is a data issue,
    # whereas a disabled demographic is this run's stated design.
    def why_unusable(g):
        return "absent" if not block_columns(df, [g]) else "disabled"

    reduced, by_key, canonical = {}, {}, {}
    empty, disabled, shrunk, merged = [], [], [], []

    for name, groups in blocks.items():
        present = [g for g in groups if usable(g)]
        if not present:
            reasons = {why_unusable(g) for g in groups}
            (disabled if reasons == {"disabled"} else empty).append(name)
            continue
        key = tuple(present)
        if key in by_key:
            canonical[name] = by_key[key]
            merged.append(f"{name} == {by_key[key]}")
            continue
        by_key[key] = name
        canonical[name] = name
        reduced[name] = present
        # demo_other is a catch-all that is empty in a well-formed table, so its
        # absence is the expected case and not worth reporting as degradation.
        if set(groups) - set(present) - OPTIONAL_GROUPS:
            shrunk.append(f"{name} -> {'+'.join(present)}")

    if empty:
        print(f"blocks with no columns in this table, dropped: {sorted(empty)}", flush=True)
    if disabled:
        print(f"blocks made empty by --demographics, dropped: {sorted(disabled)}", flush=True)
    if shrunk:
        print(f"blocks fit without an absent component: {shrunk}", flush=True)
    if merged:
        print(f"blocks that collapsed onto another and were not refit: {merged}", flush=True)

    kept_comparisons, seen = [], set()
    for a, b in comparisons:
        ca, cb = canonical.get(a), canonical.get(b)
        if ca is None or cb is None or ca == cb or (ca, cb) in seen:
            continue
        seen.add((ca, cb))
        kept_comparisons.append((ca, cb))

    return reduced, kept_comparisons


def run(df, outcomes=OUTCOMES, n_outer=5, n_inner=3, n_boot=2000, seed=0, class_weight=None,
        ignore_incident_eligibility=False, restrictions=None, race_col="Race",
        subgroup_min_n=100, clinical_scale=1.0, demographics=DEFAULT_DEMOGRAPHICS,
        stratifiers=DEFAULT_STRATIFIERS, subgroup_boot=1000,
        subgroup_blocks=DEFAULT_SUBGROUP_BLOCKS):
    demographics = tuple(demographics)
    blocks, comparisons = drop_empty_blocks(df, BLOCKS, COMPARISONS, demographics)
    restrictions = RESTRICTIONS if restrictions is None else restrictions
    rows, deltas, gains, subs, coef_rows, sg_deltas = [], [], [], [], [], []

    # A subgroup block that was never fit cannot be reported on, and silently
    # emitting an empty table would read as "no disparity" rather than "not run".
    sg_blocks = [b for b in subgroup_blocks if b in blocks]
    sg_missing = [b for b in subgroup_blocks if b not in blocks]
    # A within-subgroup delta belongs to the subgroup analysis only if its
    # augmented arm is one of the blocks that analysis is about.
    sg_pairs = [p for p in comparisons
                if p in set(SUBGROUP_COMPARISONS) and p[1] in sg_blocks]

    assigned = assign_groups(df)
    used = {g: c for g, c in assigned.items() if c
            and (g not in DEMOGRAPHIC_GROUPS or g in demographics)}
    print("features entering the regression: "
          + ", ".join(f"{g} ({len(c)})" for g, c in used.items()), flush=True)

    held = [g for g in DEMOGRAPHIC_GROUPS
            if g not in demographics and assigned.get(g)]
    if held:
        cols = [c for g in held for c in assigned[g]]
        print(f"held out of every model, by --demographics: {held} -> {cols}", flush=True)
    asked_absent = [g for g in demographics if not assigned.get(g)]
    if asked_absent:
        print(f"  WARNING: --demographics asked for {asked_absent} but this table has no such "
              f"columns, so no model uses them", flush=True)
    if assigned.get("demo_other"):
        print(f"  note: unrecognised covariates routed to demo_other and kept: "
              f"{assigned['demo_other']}", flush=True)

    print(f"\nsubgroup analysis reports on: {sg_blocks or 'nothing'}"
          + (f"; comparisons {sg_pairs}" if sg_pairs else ""), flush=True)
    if sg_missing:
        print(f"  WARNING: --subgroup-blocks named {sg_missing}, which this run does not fit. "
              f"Add them to --blocks or the subgroup tables will not cover them", flush=True)
    available, absent_strat = stratifier_labels(df, stratifiers, race_col)
    print("post-hoc subgroup axes (stratifiers only, never features): "
          + (", ".join(f"{n} ({labels.nunique()} levels)" for n, labels, _ in available)
             or "none"), flush=True)
    if absent_strat:
        print(f"  WARNING: requested but unavailable or single-level, so not reported: "
              f"{absent_strat}", flush=True)
    if race_col not in df.columns:
        print(f"  WARNING: no '{race_col}' column, so race cannot be stratified on -- "
              f"which is the only place race is used now", flush=True)

    for outcome in outcomes:
        sub = df[eligible_mask(df, outcome, ignore_incident_eligibility)]
        sub, restricted_by = apply_restriction(sub, outcome, restrictions)
        y = sub[outcome].to_numpy().astype(int)
        # Both tails matter now. Near-universal prevalence leaves nothing to
        # discriminate just as surely as near-zero prevalence does, and AUROC
        # stays defined in both, so neither is caught by an events-only guard.
        n_neg = len(y) - int(y.sum())
        if y.sum() < 30 or n_neg < 30:
            print(f"skipping {outcome}: {y.sum()} prevalent and {n_neg} unaffected among "
                  f"{len(y)} patients; need 30 of each", flush=True)
            continue
        label = f"{outcome} among {restricted_by}" if restricted_by else outcome
        print(f"\n=== {label}  n={len(y)}  prevalent at index={y.sum()} ({y.mean():.1%}) ===",
              flush=True)

        preds, logits, coefs = {}, {}, {}
        for name, groups in blocks.items():
            print(f"  fitting {name} ...", end="\r", flush=True)
            proba, logit, Cs, coef = crossval_predict(
                sub, y, groups, n_outer, n_inner, seed, class_weight=class_weight,
                clinical_scale=clinical_scale)
            preds[name] = proba
            logits[name] = logit
            coefs[name] = coef
            m = metrics(y, proba, logit, n_boot=subgroup_boot, seed=seed)
            g = gains_table(y, proba)
            g.insert(0, "block", name)
            g.insert(0, "outcome", outcome)
            gains.append(g)
            nc = named_coefficients(coef, sub)
            nc.insert(0, "block", name)
            nc.insert(0, "restricted_to", restricted_by or "")
            nc.insert(0, "outcome", outcome)
            coef_rows.append(nc)
            rows.append({"outcome": outcome, "restricted_to": restricted_by or "",
                         "block": name,
                         # The block name is the one requested; this is what was
                         # actually fit. They differ when a component was absent,
                         # and 'clinical' without race must not read as with it.
                         "groups_fit": "+".join(groups),
                         "n_patients": len(y),
                         "n_prevalent": int(y.sum()),
                         "n_features": len(block_columns(sub, groups)),
                         "C_median": float(np.median(Cs)), **m})
            ci = ("" if np.isnan(m["auroc_lo"])
                  else f" [{m['auroc_lo']:.3f}, {m['auroc_hi']:.3f}]")
            print(f"  {name:20s} AUROC {m['auroc']:.3f}{ci}  AUPRC {m['auprc']:.3f}"
                  f" ({m['auprc_lift']:.2f}x)"
                  f"  Brier {m['brier']:.3f}  cal {m['cal_slope']:.2f}/{m['cal_intercept']:+.2f}"
                  f"  |  recall@top10% {m['recall_at_top10pct']:.3f}"
                  f"  prec {m['precision_at_top10pct']:.3f}"
                  f"  sens@90spec {m['sens_at_90spec']:.3f}"
                  f"  recall@0.5 {m['recall_at_0.5']:.3f} (n={m['n_flagged_at_0.5']})", flush=True)

        for small, large in comparisons:
            d = delta_auc(y, preds[small], preds[large], n_boot, seed)
            deltas.append({"outcome": outcome, "restricted_to": restricted_by or "",
                           "baseline": small, "augmented": large,
                           "n_patients": len(y), "n_prevalent": int(y.sum()), **d})
            print(f"  {large} vs {small}: dAUROC {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]",
                  flush=True)

        best = max(preds, key=lambda k: roc_auc_score(y, preds[k]))
        print(f"\n  gains for best block ({best}), prevalence {y.mean():.1%}:", flush=True)
        gt = gains_table(y, preds[best])
        for _, r in gt.iterrows():
            print(f"    flag top {r['alert_rate']:.0%} ({int(r['n_flagged'])} patients) -> "
                  f"identify {int(r['true_positives_caught'])}/{int(r['total_positives'])} "
                  f"already coded = {r['recall']:.1%} recall, {r['precision']:.1%} precision, "
                  f"{r['lift']:.2f}x lift", flush=True)

        # The subgroup labels are rebuilt on `sub`, not on `df`. Eligibility and
        # the CKD-among-hypertensives restriction have already dropped rows, and
        # y/preds/logits are positional within `sub`, so a stratifier built on
        # the full frame would be silently misaligned.
        strat, _ = stratifier_labels(sub, stratifiers, race_col)

        for name in sg_blocks:
            s = subgroup_report(y, preds[name], logits[name], strat, subgroup_min_n,
                                subgroup_boot, seed)
            s.insert(0, "block", name)
            s.insert(0, "restricted_to", restricted_by or "")
            s.insert(0, "outcome", outcome)
            subs.append(s)

        for shown in subs[-len(sg_blocks):] if strat and sg_blocks else []:
            head = shown["block"].iloc[0]
            print(f"\n  performance by subgroup for {head}, pooled out-of-fold. Compare each "
                  f"AUROC\n  against the others in its own stratifier, not against the pooled "
                  f"figure: conditioning\n  on age or BMI removes that variable's spread and "
                  f"lowers within-bracket AUROC by itself.", flush=True)
            for axis in shown["stratifier"].unique():
                print(f"    [{axis}]", flush=True)
                for _, r in shown[shown["stratifier"] == axis].iterrows():
                    lead = (f"      {r['group']:<18} n={int(r['n']):>5,}  "
                            f"prev {r['prevalence']:>5.1%}")
                    if r["note"]:
                        print(f"{lead}   {r['note']}", flush=True)
                        continue
                    ci = ("" if np.isnan(r["auroc_lo"])
                          else f" [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}]")
                    print(f"{lead}   AUROC {r['auroc']:.3f}{ci}  "
                          f"AUPRC {r['auprc']:.3f} ({r['auprc_lift']:.2f}x)  "
                          f"cal {r['cal_intercept']:+.2f}", flush=True)
                est = shown[(shown["stratifier"] == axis)].dropna(subset=["auroc"])
                if len(est) > 1:
                    spread = est["auroc"].max() - est["auroc"].min()
                    lo = est.loc[est["auroc"].idxmin(), "group"]
                    print(f"      -> AUROC spread {spread:.3f} across estimable levels, "
                          f"worst '{lo}'", flush=True)

        # The primary subgroup quantity: is the fusion gain the same everywhere?
        # Paired within each subgroup, so case-mix compression cancels and these
        # numbers are comparable across levels in a way the AUROCs above are not.
        for small, large in sg_pairs:
            d = subgroup_deltas(y, preds[small], preds[large], strat, subgroup_min_n,
                                subgroup_boot, seed)
            d.insert(0, "augmented", large)
            d.insert(0, "baseline", small)
            d.insert(0, "restricted_to", restricted_by or "")
            d.insert(0, "outcome", outcome)
            sg_deltas.append(d)
            est = d.dropna(subset=["delta"])
            if est.empty:
                continue
            print(f"\n  {large} vs {small}, dAUROC within subgroup:", flush=True)
            for axis in est["stratifier"].unique():
                cell = est[est["stratifier"] == axis]
                parts = "  ".join(f"{r['group']} {r['delta']:+.3f}"
                                  f"{'*' if r['lo'] > 0 or r['hi'] < 0 else ''}"
                                  for _, r in cell.iterrows())
                print(f"    [{axis}] {parts}", flush=True)
            print(f"    (* = bootstrap interval excludes 0, unadjusted for the "
                  f"{len(est)} intervals in this comparison)", flush=True)

        # Whether age, BMI and race are still doing anything in the fused model
        # is not something AUROC can answer: one shared L2 budget across ~2,300
        # embedding columns can flatten a handful of clinical weights to nothing
        # while the headline AUROC barely moves. Printed side by side with the
        # clinical-only fit, where nothing competes with them, so the shrinkage
        # is a number rather than a suspicion.
        fused = next((b for b in ("clinical+mammo+cxr", "demo+mammo+cxr") if b in coefs), best)
        solo_nc = named_coefficients(coefs["clinical"], sub) if "clinical" in coefs else None
        both_nc = named_coefficients(coefs[fused], sub)
        # A pure-embedding block has no clinical weights at all, so both frames
        # come back empty and there is nothing to compare. That happens whenever
        # --blocks leaves out every adjusted arm; it is not an error.
        if solo_nc is not None and fused != "clinical" and not both_nc.empty:
            solo = solo_nc.set_index("feature")["coef"]
            both = both_nc.set_index("feature")["coef"]
            print(f"\n  clinical coefficients, standardised: clinical vs {fused}", flush=True)
            for feat in both.index:
                a, b = solo.get(feat, float("nan")), both[feat]
                line = f"    {feat:<24} {a:>+7.3f} -> {b:>+7.3f}"
                if a:
                    line += f"   {abs(b) / abs(a):>5.0%} retained"
                print(line, flush=True)
            wanted = both.index[both.index.str.startswith(("demo_age", "demo_bmi", "race_"))]
            dead = [f for f in wanted if abs(both[f]) < 1e-6]
            print(f"    -> age/bmi/race non-zero in {fused}: "
                  f"{len(wanted) - len(dead)}/{len(wanted)}"
                  + (f"; flattened: {dead}" if dead else ""), flush=True)

    if not rows:
        raise SystemExit("no outcome had 30 patients in both classes; nothing was fit")
    return (pd.DataFrame(rows), pd.DataFrame(deltas), pd.concat(gains, ignore_index=True),
            pd.concat(subs, ignore_index=True) if subs else pd.DataFrame(),
            pd.concat(coef_rows, ignore_index=True),
            pd.concat(sg_deltas, ignore_index=True) if sg_deltas else pd.DataFrame())


# Kept as a module attribute so scripts written against the old name still
# work; `common.simulate` is the canonical definition both arms share.
_simulate = simulate


def main():
    global BLOCKS, COMPARISONS

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None, help="parquet from cohort.py; omit to run on simulated data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--outcomes", nargs="+", default=OUTCOMES)
    ap.add_argument("--blocks", nargs="+", default=None,
                    help="subset of block names to fit; default is all")
    ap.add_argument("--n-outer", type=int, default=5)
    ap.add_argument("--n-inner", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    ap.add_argument("--max-index-gap", type=int, default=None,
                    help="drop patients whose two images are more than this many days apart; "
                         "the earlier modality is stale by that much relative to the index date")
    ap.add_argument("--ignore-incident-eligibility", action="store_true",
                    help="tolerate leftover <outcome>_eligible columns instead of raising")
    ap.add_argument("--no-restrictions", action="store_true",
                    help=f"fit every outcome on the full cohort, dropping the pre-specified "
                         f"secondary analyses {RESTRICTIONS}")
    ap.add_argument("--race-col", default="Race",
                    help="raw stratifier column used for subgroup reporting")
    ap.add_argument("--subgroup-min-n", type=int, default=100,
                    help="subgroups smaller than this are counted but not scored")
    ap.add_argument("--stratify-by", nargs="*", default=list(DEFAULT_STRATIFIERS),
                    choices=list(STRATIFIERS),
                    help=f"post-hoc subgroup axes: age {AGE_LABELS}, bmi {BMI_LABELS}, race, "
                         f"imaging_year {YEAR_LABELS}, index_gap {GAP_LABELS}. These are never "
                         f"features; an axis stays available when the matching feature group is "
                         f"off. Pass '--stratify-by' with no values to skip the subgroup analysis")
    ap.add_argument("--subgroup-blocks", nargs="*", default=list(DEFAULT_SUBGROUP_BLOCKS),
                    help=f"which fitted blocks the subgroup analysis reports on; default "
                         f"{list(DEFAULT_SUBGROUP_BLOCKS)}. Reporting only -- these must also be "
                         f"in --blocks to have been fit")
    ap.add_argument("--subgroup-boot", type=int, default=1000,
                    help="bootstrap resamples for block-level AUROC intervals in "
                         "performance.csv, subgroup AUROC intervals, and within-subgroup "
                         "deltas; 0 reports point estimates only")
    ap.add_argument("--demographics", nargs="*", default=list(DEFAULT_DEMOGRAPHICS),
                    choices=list(DEMOGRAPHIC_GROUPS),
                    help="which demographic groups may be used as FEATURES. Default "
                         "'age bmi': race and ethnicity are held out and analysed post hoc "
                         "as stratifiers instead. Pass '--demographics' with no values to "
                         "drop all four; pass 'age bmi eth race' to include everything. "
                         "Race remains the subgroup stratifier either way.")
    ap.add_argument("--clinical-scale", type=float, default=1.0,
                    help="multiply standardised age/bmi/race/eth/tech features by this, "
                         "reducing their share of the L2 penalty by its square. 1.0 changes "
                         "nothing; try 10 if the coefficient report shows them flattened")
    args = ap.parse_args()

    if args.blocks:
        unknown = [b for b in args.blocks if b not in BLOCKS]
        if unknown:
            raise SystemExit(f"unknown blocks {unknown}; choose from {list(BLOCKS)}")
        BLOCKS = {k: v for k, v in BLOCKS.items() if k in args.blocks}
        COMPARISONS = [(a, b) for a, b in COMPARISONS if a in BLOCKS and b in BLOCKS]

    df = _simulate() if args.cohort is None else pd.read_parquet(args.cohort)
    df = cap_index_gap(df, args.max_index_gap)
    perf, deltas, gains, subs, coefs, sg_deltas = run(
        df, args.outcomes, args.n_outer, args.n_inner, args.n_boot, args.seed,
        args.class_weight, args.ignore_incident_eligibility,
        restrictions={} if args.no_restrictions else RESTRICTIONS,
        race_col=args.race_col, subgroup_min_n=args.subgroup_min_n,
        clinical_scale=args.clinical_scale, demographics=tuple(args.demographics),
        stratifiers=tuple(args.stratify_by), subgroup_boot=args.subgroup_boot,
        subgroup_blocks=tuple(args.subgroup_blocks))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    perf.to_csv(out / "performance.csv", index=False)
    deltas.to_csv(out / "deltas.csv", index=False)
    gains.to_csv(out / "gains.csv", index=False)
    written = ["performance.csv", "deltas.csv", "gains.csv"]
    if not subs.empty:
        subs.to_csv(out / "subgroups.csv", index=False)
        written.append("subgroups.csv")
    if not sg_deltas.empty:
        sg_deltas.to_csv(out / "subgroup_deltas.csv", index=False)
        written.append("subgroup_deltas.csv")
    if not coefs.empty:
        coefs.to_csv(out / "coefficients.csv", index=False)
        written.append("coefficients.csv")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nwrote {', '.join(written)} to {out}")


if __name__ == "__main__":
    main()
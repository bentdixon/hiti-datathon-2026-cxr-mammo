"""
Shared layer for all model architectures to ensure equivalent accuracy metrics, 
training, data loading, etc.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

OUTCOMES = ["t2dm", "htn", "hld", "ckd"]

# Feature groups. Age, BMI and race are separate groups rather than one 'demo'
# bundle so each enters the final regression as its own named set of columns and
# can be reported, ablated and read off independently. cohort.py writes
# them all under a demo_ / race_ prefix, so the split happens here by exact
# column rather than by renaming anything upstream.
#
# Every feature column must match exactly one group, first match wins, and
# demo_other is a deliberate catch-all: a covariate added upstream lands there
# and stays in the model instead of being silently dropped from every block.
GROUPS = {
    "age": lambda c: c == "demo_age_at_index",
    "bmi": lambda c: c in ("demo_bmi", "demo_bmi_missing"),
    "eth": lambda c: c.startswith("demo_eth"),
    "race": lambda c: c.startswith("race_"),
    "tech": lambda c: c.startswith("tech_"),
    "mammo": lambda c: c.startswith("mammo_"),
    "cxr": lambda c: c.startswith("cxr_"),
    "demo_other": lambda c: c.startswith("demo_"),
}

EMBEDDING_GROUPS = ("mammo", "cxr")

# Date columns cohort.py carries for the imaging_year stratifier, most
# preferred first. They must never reach a model, and are checked by exact name
# before the prefix rules in GROUPS: 'cxr_date' would match the 'cxr' prefix
# rule and put a datetime in the CXR embedding block. cohort.py keeps all
# three out of every feature list for the same reason.
DATE_COLS = ("index_date", "cxr_date", "index_mammo_date")
NEVER_FEATURES = frozenset(DATE_COLS)

# Demographic groups whose inclusion as *features* is a decision, not a given.
DEMOGRAPHIC_GROUPS = ("age", "bmi", "eth", "race")

# Race and ethnicity are held out of the feature set and analysed post hoc.
#
# Both are self-reported social categories rather than measurements, so a model
# that uses them to detect prevalent disease is partly encoding differential
# access to diagnosis rather than physiology. cohort.py takes the same
# position for race -- it carries the column as a stratifier and asserts it out
# of every feature list on each run. Holding ethnicity out too makes that
# treatment consistent.
#
# Nothing about the post-hoc analysis changes. The raw Race column is carried
# unprefixed by cohort.py, so it is not in any block and subgroups() still
# reports per-group AUROC, AUPRC and calibration on the exact rows that were
# fit. Excluding race as an input arguably makes that reporting more important,
# not less: a model blind to race can still perform unequally across racial
# groups.
DEFAULT_DEMOGRAPHICS = ("age", "bmi")

# Pre-specified secondary analysis. cohort.py measures that 96.8% of CKD
# positives are also hypertensive, and that this survives rebuilding CKD from
# pure renal codes, so the nesting is comorbidity rather than a coding artifact.
# An apparent imaging signal for CKD may therefore be a signal for hypertension;
# restricting to hypertensives asks the motivated question instead, which of
# them has kidney disease.
RESTRICTIONS = {"ckd": "htn"}

MISSING_LEVEL = "Missing"
MIN_LOGIT_SD = 1e-3
GAINS_RATES = (0.05, 0.10, 0.20, 0.30, 0.50)

_GROUP_CACHE = {}


def assign_groups(df):
    """
    Map every feature column to exactly one group, first match in GROUPS wins.

    Non-feature columns -- the outcomes, the raw Race stratifier, any eligible
    flag -- match nothing and are simply absent from the result, which is what
    keeps them out of every model.

    NEVER_FEATURES is checked before GROUPS because prefix matching is not safe
    for the date columns: 'cxr_date' begins with 'cxr_' and would otherwise be
    filed in the CXR embedding block, putting a datetime into the design matrix
    of every block that names cxr. It is carried only as the imaging_year
    stratifier, so it is excluded by name here.

    Memoised on the column tuple. Block selection happens once per block per
    outer fold (linear arm) or once per architecture per fold (nonlinear arm),
    and re-scanning thousands of columns each time is pure waste.
    """
    key = tuple(df.columns)
    hit = _GROUP_CACHE.get(key)
    if hit is not None:
        return hit

    out = {g: [] for g in GROUPS}
    for c in key:
        if c in NEVER_FEATURES:
            continue
        for g, match in GROUPS.items():
            if match(c):
                out[g].append(c)
                break
    _GROUP_CACHE[key] = out
    return out


def block_columns(df, groups):
    """
    Columns of a block, in group order.

    Order matters and is not df column order. The linear arm's ColumnTransformer
    emits its transformers' outputs in the order they are declared, so the
    coefficient vector is laid out group by group; the nonlinear arm's clinical
    tower reads the same ordering so its offset coefficients stay comparable to
    the linear arm's. Labelling either one by any other ordering would silently
    attach the wrong name to every weight.
    """
    assigned = assign_groups(df)
    cols = []
    for g in groups:
        cols.extend(assigned.get(g, []))
    return cols


def eligible_mask(df, outcome, ignore_incident_eligibility=False):
    """
    The rows that contribute to `outcome`.

    Under an incident design eligibility was outcome-specific: a patient
    already hypertensive was dropped from the hypertension denominator but kept
    for CKD. Prevalence at the index date inverts that. The label is whether the
    condition is coded as of the image, so a prevalent patient is the positive
    case, not an ineligible one, and applying a `<outcome>_eligible` filter here
    would delete most of the signal being measured.

    The one eligibility question that survives is record coverage: did this
    patient have enough EHR contact before the index date that an absent code
    means an absent condition rather than absent data. That is a property of the
    patient, not of the outcome, so it is a single cohort-level `eligible`
    column.
    """
    per_outcome = f"{outcome}_eligible"
    if per_outcome in df.columns and not ignore_incident_eligibility:
        raise ValueError(
            f"'{per_outcome}' is an incident-design column: it excludes patients who "
            f"already had {outcome} at the index date, who are exactly the positives "
            f"under a prevalence design. Rebuild the cohort without it, or pass "
            f"--ignore-incident-eligibility to leave the column in place and disregard it."
        )
    if "eligible" in df.columns:
        return df["eligible"].to_numpy() == 1
    return np.ones(len(df), dtype=bool)


def apply_restriction(df, outcome, restrictions):
    """
    Restrict an outcome's denominator to a pre-specified subgroup.

    Distinct from eligibility, which is about whether a patient's record can be
    read at all. This is a narrower clinical question deliberately substituted
    for the broad one: CKD among hypertensives rather than CKD in everyone. It
    is pre-specified in RESTRICTIONS rather than chosen after seeing results,
    which is the only thing separating it from subgroup fishing.
    """
    col = restrictions.get(outcome)
    if col is None:
        return df, None
    if col not in df.columns:
        raise KeyError(f"restriction '{outcome} among {col}' needs a '{col}' column")
    return df[df[col] == 1], col


def cap_index_gap(df, max_days, col="tech_abs_dt"):
    """
    Drop patients whose two images are more than `max_days` apart.

    The index date is the later of the two images, so the earlier modality is
    stale by abs_dt days. Under an incident design that cost nothing: both
    images preceded the outcome either way. Under prevalence at the index date
    it is a real mismatch, because a condition first coded between the two
    images is present in the label while the earlier image was taken before
    there was anything to see. Capping the gap bounds how much of the label the
    earlier modality could not have observed.
    """
    if max_days is None:
        return df
    if col not in df.columns:
        print(f"--max-index-gap given but '{col}' is not in the table; no cap applied", flush=True)
        return df
    keep = df[col].abs() <= max_days
    print(f"index gap cap {max_days}d: keeping {int(keep.sum()):,} of {len(df):,} patients "
          f"(median gap {df[col].abs().median():.0f}d)", flush=True)
    return df[keep]


def calibration_slope(y, logit):
    """
    Refit y ~ a + b*logit. b = 1 is perfect; b < 1 means over-confident.

    NaN when the out-of-fold logit is near-constant. A small block -- race is
    three indicators -- can be penalised almost to the intercept, and regressing
    the outcome on a predictor with no spread returns an arbitrary large slope
    that reads as catastrophic miscalibration rather than as no signal. The
    AUROC alongside it already says there is nothing to calibrate.
    """
    if float(np.std(logit)) < MIN_LOGIT_SD:
        return float("nan")
    m = LogisticRegression(max_iter=5000, C=1e6).fit(logit.reshape(-1, 1), y)
    return float(m.coef_[0, 0])


def calibration_intercept(y, logit, iters=100, tol=1e-10):
    """
    Solve for a in y ~ sigmoid(a + logit), holding the slope at 1.

    Calibration-in-the-large: the constant shift needed to make mean predicted
    risk match observed prevalence. 0 is perfect, negative means the model is
    systematically too high. Slope alone cannot see this error, and it is the
    one that matters most here: prevalence labels sit at a far higher base rate
    than incident labels did, so a model carried over from an incident design
    will look well discriminated and still be badly off-centre.
    """
    a = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + logit)))
        hess = float(np.sum(p * (1 - p)))
        if hess < 1e-12:
            break
        step = float(np.sum(y - p)) / hess
        a += step
        if abs(step) < tol:
            break
    return float(a)


def _levels(g, order=None):
    """Level names present in `g`, in bracket order when one is declared."""
    present = set(g)
    if order is None:
        return sorted(present)
    named = [lv for lv in order if lv in present]
    return named + sorted(present - set(named))


def _auroc_ci(y, proba, n_boot, seed):
    """Percentile bootstrap interval for AUROC on one subgroup's rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], proba[i]))
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def subgroups(y, proba, logit, groups, min_n=100, n_boot=0, seed=0, order=None):
    """
    Per-group discrimination and calibration on the pooled out-of-fold predictions.

    A model can equalise its average performance while doing systematically
    worse inside one group, and a single pooled AUROC cannot show that. Both
    halves are needed: discrimination says whether the ranking is as good within
    the group, and the calibration intercept says whether the group's risks are
    shifted as a whole, which is the error that changes who crosses a threshold.

    Read `auroc` across groups with care. Conditioning on a stratifier removes
    that variable's spread from the subgroup, so a stratifier that is itself
    predictive -- age and BMI both are -- mechanically depresses within-bracket
    AUROC relative to the pooled figure. That is case-mix compression, not a
    disparity.

    Groups below min_n are reported with their counts but no metrics. An AUROC
    from 30 patients is noise, and printing it invites a disparity claim the
    data cannot support. cohort.py uses the same threshold to decide a
    racial group is too thin to report alone.
    """
    # Positional, not label-based: y/proba/logit are plain arrays in row order,
    # so groups is compared as an array and never joined on an index.
    g = pd.Series(groups).fillna(MISSING_LEVEL).astype(str).to_numpy()
    y = np.asarray(y)

    rows = []
    for name in _levels(g, order):
        m = g == name
        yy, pp, ll = y[m], proba[m], logit[m]
        n_pos = int(yy.sum())
        prev = float(yy.mean()) if m.any() else float("nan")
        row = {"group": name, "n": int(m.sum()), "n_prevalent": n_pos,
               "prevalence": prev,
               "auroc": float("nan"), "auroc_lo": float("nan"), "auroc_hi": float("nan"),
               "auprc": float("nan"), "auprc_lift": float("nan"),
               "cal_intercept": float("nan"), "note": ""}
        if m.sum() >= min_n and 0 < n_pos < m.sum():
            auprc = float(average_precision_score(yy, pp))
            row["auroc"] = float(roc_auc_score(yy, pp))
            row["auprc"] = auprc
            row["auprc_lift"] = auprc / prev
            row["cal_intercept"] = calibration_intercept(yy, ll)
            if n_boot:
                row["auroc_lo"], row["auroc_hi"] = _auroc_ci(yy, pp, n_boot, seed)
        else:
            row["note"] = f"n<{min_n} or single-class; not estimated"
        rows.append(row)

    out = pd.DataFrame(rows)
    # A fixed bracket order reads as a dose-response; an ad-hoc set does not, so
    # only the latter is re-sorted by size.
    return out if order else out.sort_values("n", ascending=False).reset_index(drop=True)


def recall_at_alert_rate(y, proba, rate):
    """
    Recall when the top `rate` fraction of patients by predicted score are flagged.

    This is the operating point a case-finding review actually chooses: capacity
    fixes how many charts can be pulled, not a probability cutoff. Returns recall
    (fraction of patients who already carry the code that are captured) and
    precision (fraction of flagged patients who carry it). A model with no signal
    captures `rate` of them, so compare recall against the alert rate itself.
    """
    k = max(round(rate * len(y)), 1)
    flagged = np.zeros(len(y), dtype=int)
    flagged[np.argsort(-proba)[:k]] = 1
    return (recall_score(y, flagged, zero_division=0),
            precision_score(y, flagged, zero_division=0))


def sensitivity_at_specificity(y, proba, target=0.90):
    """Recall at the threshold giving at least `target` specificity."""
    fpr, tpr, _ = roc_curve(y, proba)
    ok = fpr <= (1 - target)
    return float(tpr[ok].max()) if ok.any() else 0.0


def gains_table(y, proba, rates=GAINS_RATES):
    """
    Recall at a range of alert rates, with lift over flagging at random.

    Reads as: "flag the top R% of patients by predicted score and you capture
    RECALL of everyone whose record already carries the condition at the index
    date." Flagging R% at random captures R% of them, so lift = recall / R is
    the honest summary of how much the model buys.

    Lift degrades as prevalence rises even when discrimination is unchanged:
    at 40% prevalence the top 50% cannot exceed 1.25x however good the model is.
    Read lift next to the prevalence column, not on its own.
    """
    y = np.asarray(y)
    order = np.argsort(-proba)
    cum = np.cumsum(y[order])
    rows = []
    for r in rates:
        k = max(round(r * len(y)), 1)
        tp = int(cum[k - 1])
        rows.append({
            "alert_rate": r,
            "n_flagged": k,
            "true_positives_caught": tp,
            "total_positives": int(y.sum()),
            "recall": tp / max(int(y.sum()), 1),
            "precision": tp / k,
            "lift": (tp / max(int(y.sum()), 1)) / r,
        })
    return pd.DataFrame(rows)


def metrics(y, proba, logit, alert_rates=(0.10, 0.20), n_boot=0, seed=0):
    """
    Point estimates for one block/architecture on one outcome, plus an optional
    AUROC interval.

    `n_boot` adds `auroc_lo`/`auroc_hi`, an unpaired percentile bootstrap over
    patients. It answers "how precisely is this AUROC pinned down", which is not
    the question `delta_auc` answers: that one resamples the same patients for
    both arms, so its interval is much tighter than the overlap of two of these.
    Read these for reporting one fit, and the paired delta for deciding whether
    fusion helped. Columns are always present so the schema does not depend on
    the flag; they are NaN when `n_boot` is 0.
    """
    prevalence = float(np.mean(y))
    auprc = float(average_precision_score(y, proba))
    lo, hi = _auroc_ci(y, proba, n_boot, seed) if n_boot else (float("nan"),
                                                               float("nan"))
    out = {
        "auroc": float(roc_auc_score(y, proba)),
        "auroc_lo": lo,
        "auroc_hi": hi,
        "auprc": auprc,
        # A no-signal model scores AUPRC equal to prevalence. At the base rates
        # a prevalence design produces that floor is high enough that raw AUPRC
        # reads as strong performance on its own; the ratio is what moved.
        "auprc_lift": auprc / prevalence if prevalence > 0 else float("nan"),
        "brier": float(brier_score_loss(y, proba)),
        "brier_skill": (1 - brier_score_loss(y, proba) / (prevalence * (1 - prevalence))
                        if 0 < prevalence < 1 else float("nan")),
        "cal_slope": calibration_slope(y, logit),
        "cal_intercept": calibration_intercept(y, logit),
        "prevalence": prevalence,
    }

    hard = (proba >= 0.5).astype(int)
    out["recall_at_0.5"] = float(recall_score(y, hard, zero_division=0))
    out["precision_at_0.5"] = float(precision_score(y, hard, zero_division=0))
    out["n_flagged_at_0.5"] = int(hard.sum())

    out["sens_at_90spec"] = sensitivity_at_specificity(y, proba, 0.90)

    for rate in alert_rates:
        rec, prec = recall_at_alert_rate(y, proba, rate)
        out[f"recall_at_top{int(rate * 100)}pct"] = float(rec)
        out[f"precision_at_top{int(rate * 100)}pct"] = float(prec)

    return out


def delta_auc(y, p_small, p_large, n_boot=2000, seed=0):
    """Paired bootstrap over patients on the AUROC difference."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            d[b] = np.nan
            continue
        d[b] = roc_auc_score(y[i], p_large[i]) - roc_auc_score(y[i], p_small[i])
    d = d[~np.isnan(d)]
    out = {"delta": float(roc_auc_score(y, p_large) - roc_auc_score(y, p_small))}
    if d.size == 0:
        # n_boot=0, or every resample drew a single class. The point estimate is
        # still meaningful; the interval is not, and must not be invented.
        out.update({"lo": np.nan, "hi": np.nan, "p_gt_0": np.nan})
    else:
        out.update({
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_gt_0": float((d > 0).mean()),
        })
    return out


def simulate(n=6500, d_mammo=512, d_cxr=768, seed=0):
    """
    Shared adiposity/age latent in both modalities, a CXR-only axis, and a care
    intensity latent behind the acquisition covariates.

    Care intensity is the confound the prevalence design has to survive: sicker
    patients get portable films, get imaged closer together, and get their
    comorbidities coded. It is deliberately wired into both the tech block and
    every outcome, so the tech-only block reaches a non-trivial AUROC with no
    anatomy involved and the demo+tech baseline is a real one to beat. It also
    bleeds slightly into the CXR embedding, because a portable film looks
    different from a PA film.

    Base rates are prevalent-at-index, not incident, so they are several times
    higher than the rates this simulation produced under a prediction design.

    Race is generated with the cohort's rough composition (majority Black, and
    one level thin enough to exercise the pooling threshold), with a genuine
    effect on prevalence and a smaller one on care intensity. That combination
    is what makes the subgroup table say something: a model can use race to
    sharpen its average and still rank worse inside the smaller groups.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 10, n)
    adipose = rng.normal(size=n)
    cxr_only = rng.normal(size=n)

    race_levels = ["Black", "White", "Asian", "Other", "Pacific Islander"]
    race = rng.choice(race_levels, n, p=[0.55, 0.34, 0.06, 0.04, 0.01])
    race_effect = pd.Series(
        {"Black": 0.45, "White": 0.0, "Asian": -0.25, "Other": -0.1,
         "Pacific Islander": 0.2}).reindex(race).to_numpy()

    care = rng.normal(size=n) + 0.2 * race_effect

    mammo = adipose[:, None] * rng.normal(size=(1, d_mammo)) * 0.6 + rng.normal(size=(n, d_mammo))
    cxr = (adipose[:, None] * rng.normal(size=(1, d_cxr)) * 0.6
           + cxr_only[:, None] * rng.normal(size=(1, d_cxr)) * 0.5
           + care[:, None] * rng.normal(size=(1, d_cxr)) * 0.3
           + rng.normal(size=(n, d_cxr)))

    z = (age - 60) / 10
    logits = {
        "t2dm": -1.4 + 0.4 * z + 0.9 * adipose + 0.5 * cxr_only + 0.5 * care + 0.6 * race_effect,
        "htn": -0.1 + 0.8 * z + 0.6 * adipose + 0.6 * care + 0.8 * race_effect,
        "hld": -0.4 + 0.5 * z + 0.4 * adipose + 0.4 * care + 0.2 * race_effect,
        "ckd": -2.4 + 0.7 * z + 0.3 * adipose + 0.2 * cxr_only + 0.7 * care + 0.7 * race_effect,
    }
    portable = rng.binomial(1, 1 / (1 + np.exp(-(-1.4 + 1.2 * care))), n)
    # Days between the two images. The index date is the later of them, so this
    # is how stale the earlier modality is relative to the label.
    abs_dt = rng.integers(0, 800, n)

    # Index date, carried only so the imaging_year stratifier has something to
    # cut. Spread over 2014-2020 so both sides of the 2017 boundary are populated;
    # it is drawn independently of every outcome, so a year effect appearing in
    # the subgroup tables on simulated data is noise by construction and a useful
    # check on how much of one this design produces.
    index_date = (pd.Timestamp("2014-01-01")
                  + pd.to_timedelta(rng.integers(0, 365 * 7, n), unit="D"))

    # Encoded the way cohort.py encodes it: White as the reference level and
    # the thin Pacific Islander level pooled into Other, so the simulated table
    # has the same block structure a real one does.
    race_series = pd.Series(race).replace({"Pacific Islander": "Other"})
    race_dummies = pd.get_dummies(race_series, prefix="race").drop(
        columns=["race_White"]).astype(float)

    parts = [
        pd.DataFrame({k: rng.binomial(1, 1 / (1 + np.exp(-v))) for k, v in logits.items()}),
        pd.DataFrame({"Race": race, "index_date": index_date}),
        pd.DataFrame({"demo_age_at_index": age,
                      "demo_bmi": 28 + 4 * adipose + rng.normal(0, 3, n),
                      "demo_bmi_missing": np.zeros(n),
                      "tech_portable": portable.astype(float),
                      "tech_abs_dt": abs_dt.astype(float),
                      "tech_screen": rng.binomial(1, 0.8, n).astype(float)}),
        race_dummies,
        pd.DataFrame(mammo, columns=[f"mammo_{i}" for i in range(d_mammo)]),
        pd.DataFrame(cxr, columns=[f"cxr_{i}" for i in range(d_cxr)]),
    ]
    return pd.concat(parts, axis=1)

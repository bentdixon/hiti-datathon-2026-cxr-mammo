"""
Flatten subgroups.csv into a plain-reading accuracy table, one row per subgroup.

subgroups.csv is complete but wide, and its two traps are invisible in the raw
numbers. This adds a generated sentence per row plus an explicit comparability
flag, so the table can be read without re-deriving the caveats each time.

The two traps:

  Case-mix compression. Conditioning on a stratifier that itself predicts the
  outcome removes that variance from the within-level ranking problem, so the
  within-level AUROC falls for reasons that have nothing to do with model
  quality. Detected here by prevalence spread across the axis: if a stratifier
  sorts the outcome, it will show a wide spread, and its levels are then not
  comparable to each other. `comparable` records the verdict.

  Base-rate shrinkage. One pooled model pulls every subgroup toward the cohort
  base rate, so the calibration intercept tracks (subgroup prevalence - pooled
  prevalence) rather than anything specific to the subgroup. Reported, but
  worded as miscalibration rather than as a disparity.

Usage:
    python subgroup_reading.py --infile subgroups.csv --out subgroup_accuracy.csv
"""

import argparse

import pandas as pd

DISCRIMINATION = ((0.80, "strong"), (0.70, "moderate"), (0.60, "modest"),
                  (0.00, "weak"))

# Prevalence spread across an axis's scorable levels, above which the axis is
# treated as predictive of the outcome and its levels as non-comparable.
COMPRESSION_SPREAD = 0.10

# |calibration intercept| below which the sign is treated as noise.
CAL_NEGLIGIBLE = 0.10

OUTCOME_LABELS = {"t2dm": "type 2 diabetes", "htn": "hypertension",
                  "hld": "hyperlipidaemia", "ckd": "chronic kidney disease"}

AXIS_LABELS = {"age": "age", "bmi": "BMI", "race": "race",
               "imaging_year": "imaging year",
               "index_gap": "mammogram-to-CXR gap"}


def _describe(auroc):
    for bound, word in DISCRIMINATION:
        if auroc >= bound:
            return word
    return DISCRIMINATION[-1][1]


def build(infile, block=None):
    df = pd.read_csv(infile)
    if "stratifier" not in df.columns:
        raise SystemExit(
            f"{infile} predates the subgroup rewrite (no 'stratifier' column). "
            "Re-run linear.py and use the file from its --outdir.")
    if block:
        df = df[df["block"] == block]
    if df.empty:
        raise SystemExit(f"no rows for block={block!r}")

    df = df.copy()
    df["scored"] = df["auroc"].notna()

    # Comparability is a property of the axis within an outcome, so it is
    # computed once per (outcome, stratifier) and broadcast to its levels.
    rows = []
    for (outcome, axis), g in df.groupby(["outcome", "stratifier"], sort=False):
        scored = g[g["scored"]]
        spread = (scored["prevalence"].max() - scored["prevalence"].min()
                  if len(scored) > 1 else 0.0)
        predictive = spread > COMPRESSION_SPREAD
        for _, r in g.iterrows():
            rows.append(_row(r, outcome, axis, spread, predictive))
    return pd.DataFrame(rows)


def _row(r, outcome, axis, spread, predictive):
    out = {
        "outcome": outcome,
        "stratifier": axis,
        "group": r["group"],
        "n": int(r["n"]),
        "prevalence": round(float(r["prevalence"]), 3),
        "auroc": None if pd.isna(r["auroc"]) else round(float(r["auroc"]), 3),
        "auroc_lo": None if pd.isna(r["auroc_lo"]) else round(float(r["auroc_lo"]), 3),
        "auroc_hi": None if pd.isna(r["auroc_hi"]) else round(float(r["auroc_hi"]), 3),
        "auprc_lift": None if pd.isna(r["auprc_lift"]) else round(float(r["auprc_lift"]), 2),
        "cal_intercept": None if pd.isna(r["cal_intercept"]) else round(
            float(r["cal_intercept"]), 3),
        "comparable_across_levels": "" if pd.isna(r["auroc"]) else (
            "no" if predictive else "yes"),
        "prevalence_spread_on_axis": round(float(spread), 3),
    }

    label = OUTCOME_LABELS.get(outcome, outcome)
    axis_label = AXIS_LABELS.get(axis, axis)

    if pd.isna(r["auroc"]):
        out["reading"] = (
            f"{label}, {axis_label} = {r['group']}: n={int(r['n']):,}, too few "
            f"patients to score (threshold 100), so no accuracy estimate. "
            f"Counted in the cohort, excluded from comparison.")
        return out

    ci = f"{r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}"
    width = float(r["auroc_hi"]) - float(r["auroc_lo"])
    cal = float(r["cal_intercept"])

    text = (f"{label}, {axis_label} = {r['group']}: AUROC {r['auroc']:.3f} "
            f"(95% CI {ci}) on {int(r['n']):,} patients, {_describe(r['auroc'])} "
            f"discrimination, {r['auprc_lift']:.2f}x precision-recall lift over "
            f"this group's own {r['prevalence']:.0%} base rate.")

    if abs(cal) >= CAL_NEGLIGIBLE:
        text += (f" Risk is {'under' if cal > 0 else 'over'}-predicted here "
                 f"(intercept {cal:+.2f}), which shifts who clears a fixed "
                 f"threshold.")
    else:
        text += f" Calibration is close to correct (intercept {cal:+.2f})."

    if predictive:
        text += (f" Not comparable with other {axis_label} levels: {axis_label} "
                 f"itself predicts {label} (base rate varies {spread:.0%} across "
                 f"the axis), so within-level AUROC is compressed.")
    if width > 0.10:
        text += f" The interval is {width:.2f} wide, so treat this as imprecise."
    out["reading"] = text
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infile", default="subgroups.csv")
    ap.add_argument("--out", default="subgroup_accuracy.csv")
    ap.add_argument("--block", default=None,
                    help="restrict to one model block; default is every block present")
    args = ap.parse_args()

    table = build(args.infile, args.block)
    table.to_csv(args.out, index=False)
    scored = table["auroc"].notna().sum()
    print(f"wrote {args.out}  ({len(table)} rows, {scored} scored, "
          f"{len(table) - scored} below the n threshold)")


if __name__ == "__main__":
    main()

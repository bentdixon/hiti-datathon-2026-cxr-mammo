"""
Per-outcome, per-block headline table with intervals and a plain-English reading.

`performance.csv` carries one row per (outcome, block) with a dozen metrics, and
`deltas.csv` carries the paired fusion comparisons separately. Reading the two
side by side is what the write-up needs, so this joins them into one table and
attaches a sentence per row.

The sentence is generated from the numbers, not written by hand, so it cannot
drift out of step with the CSV when the pipeline is re-run. Thresholds for the
wording are declared in DISCRIMINATION and CALIBRATION below; they are reporting
conventions, not analysis choices, and nothing downstream depends on them.

Two intervals appear here and they answer different questions. `auroc_lo/hi` is
an unpaired bootstrap on one block: how precisely that AUROC is pinned down.
`delta_lo/hi` is the paired bootstrap from delta_auc: both arms resampled on the
same patients, so the between-arm correlation cancels. The paired interval is
much tighter, and it, not the overlap of two unpaired intervals, is what decides
whether fusion helped.

Usage:
    python block_report.py --indir results
    python block_report.py --indir . --out block_outcomes.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# AUROC floor -> descriptor. Read as "at least this value".
DISCRIMINATION = ((0.80, "strong"), (0.70, "moderate"), (0.60, "modest"),
                  (0.00, "weak"))

# |calibration intercept| ceiling -> descriptor.
CALIBRATION = ((0.10, "well calibrated"), (0.30, "mildly off-centre"),
               (float("inf"), "substantially off-centre"))

# Which single-modality arm each fused block should be reported against.
FUSION_BASELINES = {
    "mammo+cxr": ("mammo", "cxr"),
    "clinical+mammo+cxr": ("clinical+mammo", "clinical+cxr"),
    "demo+mammo+cxr": ("demo+mammo", "demo+cxr"),
}

LABELS = {
    "mammo": "mammography alone",
    "cxr": "CXR alone",
    "mammo+cxr": "mammography + CXR",
    "clinical": "clinical covariates alone",
    "clinical+mammo": "clinical + mammography",
    "clinical+cxr": "clinical + CXR",
    "clinical+mammo+cxr": "clinical + mammography + CXR",
    "demo+mammo": "demographics + mammography",
    "demo+cxr": "demographics + CXR",
    "demo+mammo+cxr": "demographics + mammography + CXR",
}

OUTCOME_LABELS = {"t2dm": "type 2 diabetes", "htn": "hypertension",
                  "hld": "hyperlipidaemia", "ckd": "chronic kidney disease"}


def _describe(table, value):
    for bound, word in table:
        if table is DISCRIMINATION and value >= bound:
            return word
        if table is CALIBRATION and abs(value) <= bound:
            return word
    return table[-1][1]


def _ci(lo, hi, digits=3):
    if lo is None or np.isnan(lo):
        return ""
    return f" (95% CI {lo:.{digits}f}-{hi:.{digits}f})"


def _signed_ci(lo, hi):
    if lo is None or np.isnan(lo):
        return ""
    return f" [{lo:+.3f}, {hi:+.3f}]"


def sentence(row, deltas):
    """One reading of a single (outcome, block) row, generated from its numbers."""
    outcome = OUTCOME_LABELS.get(row["outcome"], row["outcome"])
    if row.get("restricted_to"):
        outcome += f" among patients with {row['restricted_to']}"
    block = LABELS.get(row["block"], row["block"])

    parts = [
        (f"For {outcome} (n={int(row['n_patients']):,}, "
         f"{row['prevalence']:.0%} already coded at index), {block} ranks patients "
         f"with AUROC {row['auroc']:.3f}"
         f"{_ci(row.get('auroc_lo'), row.get('auroc_hi'))}, "
         f"{_describe(DISCRIMINATION, row['auroc'])} discrimination")
    ]

    # AUPRC is only interpretable against this outcome's own base rate, which
    # varies from 0.18 to 0.68 across the four; the raw value is not comparable.
    parts.append(
        f"precision-recall lift is {row['auprc_lift']:.2f}x over the base rate")

    # Below the "well calibrated" threshold the sign of the intercept is noise,
    # and naming a direction there reads as a finding when there isn't one.
    cal = row["cal_intercept"]
    word = _describe(CALIBRATION, cal)
    tail = "" if word == CALIBRATION[0][1] else (
        f", so it {'under-predicts' if cal > 0 else 'over-predicts'} risk on average")
    parts.append(f"and the model is {word} overall (intercept {cal:+.2f}{tail})")

    text = ", ".join(parts) + "."

    if row.get("sens_at_90spec") == row.get("sens_at_90spec"):
        text += (f" At 90% specificity it identifies "
                 f"{row['sens_at_90spec']:.0%} of prevalent cases.")

    # Fusion arms carry their incremental value; single-modality arms carry what
    # they were missing, so every row states its own place in the comparison.
    for baseline, augmented, d in deltas:
        if augmented != row["block"] and baseline != row["block"]:
            continue
        if d["outcome"] != row["outcome"] or d.get("restricted_to", "") != row.get(
                "restricted_to", ""):
            continue
        gain = f"{d['delta']:+.3f}{_signed_ci(d.get('lo'), d.get('hi'))}"
        clears = d.get("lo", float("nan")) > 0 or d.get("hi", float("nan")) < 0
        verdict = "excludes zero" if clears else "includes zero"
        if augmented == row["block"]:
            text += (f" Against {LABELS.get(baseline, baseline)} it gains "
                     f"{gain} AUROC; the interval {verdict}.")
        else:
            text += (f" Adding the other modality to it gains {gain} AUROC; "
                     f"the interval {verdict}.")
    return text


def build(indir, blocks=None, outcomes=None):
    indir = Path(indir)
    perf = pd.read_csv(indir / "performance.csv")
    dpath = indir / "deltas.csv"
    dels = pd.read_csv(dpath) if dpath.exists() else pd.DataFrame(
        columns=["outcome", "restricted_to", "baseline", "augmented",
                 "delta", "lo", "hi"])

    if "restricted_to" not in perf.columns:
        perf["restricted_to"] = ""
    perf["restricted_to"] = perf["restricted_to"].fillna("")
    if len(dels):
        dels["restricted_to"] = dels.get("restricted_to", "").fillna("")

    if blocks:
        missing = [b for b in blocks if b not in set(perf["block"])]
        if missing:
            raise SystemExit(f"not in performance.csv: {missing}. "
                             f"present: {sorted(set(perf['block']))}")
        perf = perf[perf["block"].isin(blocks)]
    if outcomes:
        perf = perf[perf["outcome"].isin(outcomes)]

    # Order rows so each outcome reads single-modality first, fused last.
    order = {b: i for i, b in enumerate(blocks)} if blocks else {}
    perf = perf.assign(_o=perf["block"].map(lambda b: order.get(b, 99)))
    perf = perf.sort_values(["outcome", "_o", "block"]).drop(columns="_o")

    drecs = [(r["baseline"], r["augmented"], r) for _, r in dels.iterrows()]

    keep = ["outcome", "restricted_to", "block", "n_patients", "n_prevalent",
            "prevalence", "auroc", "auroc_lo", "auroc_hi", "auprc",
            "auprc_lift", "brier", "cal_slope", "cal_intercept",
            "sens_at_90spec", "recall_at_top10pct", "precision_at_top10pct"]
    keep = [c for c in keep if c in perf.columns]
    out = perf[keep].copy()

    # Attach the fused-vs-baseline deltas as columns as well as prose, so the
    # table is usable without parsing the sentence.
    for label, side in (("vs_mammo", "mammo"), ("vs_cxr", "cxr")):
        col_d, col_lo, col_hi = f"delta_{label}", f"{label}_lo", f"{label}_hi"
        out[col_d] = np.nan
        out[col_lo] = np.nan
        out[col_hi] = np.nan
        for i, row in out.iterrows():
            base = FUSION_BASELINES.get(row["block"], ())
            if side not in base:
                continue
            m = dels[(dels["outcome"] == row["outcome"])
                     & (dels["baseline"] == side)
                     & (dels["augmented"] == row["block"])]
            if len(m):
                out.loc[i, [col_d, col_lo, col_hi]] = m.iloc[0][
                    ["delta", "lo", "hi"]].values

    out["reading"] = [sentence(r, drecs) for _, r in perf.iterrows()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=".",
                    help="directory holding performance.csv and deltas.csv")
    ap.add_argument("--out", default="block_outcomes.csv")
    ap.add_argument("--blocks", nargs="*", default=["mammo", "cxr", "mammo+cxr"],
                    help="blocks to report, in the order they should appear")
    ap.add_argument("--outcomes", nargs="*", default=None)
    args = ap.parse_args()

    table = build(args.indir, args.blocks, args.outcomes)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(table)} rows)\n")
    for _, r in table.iterrows():
        print(f"[{r['outcome']}/{r['block']}] {r['reading']}\n")


if __name__ == "__main__":
    main()

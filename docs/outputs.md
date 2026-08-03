# Output CSV reference

One row per table below; one entry per column. See [`methods.md`](methods.md)
for what each metric means statistically (calibration, paired vs. unpaired
bootstrap, etc.) — this file only documents *shape*: which column holds what,
and how it was computed. `restricted_to` and `outcome`/`block`/`arch`/etc. key
columns are listed once per table in the order they appear; metric columns are
grouped by what they describe rather than by position.

All tables are written by `linear.py` / `nonlinear.py` (or `main.py`, which
calls both) to `--outdir`. Nothing here is written by `cohort.py` itself.

## Linear arm (`linear.py`)

### `performance.csv`

One row per (outcome, block). Written whenever the linear arm runs.

| column | meaning |
|---|---|
| `outcome` | `t2dm` / `htn` / `hld` / `ckd` |
| `restricted_to` | empty, or the column the denominator was restricted to (currently only `htn`, for the CKD-among-hypertensives secondary analysis) |
| `block` | block name as requested (e.g. `clinical+mammo+cxr`) |
| `groups_fit` | the groups actually fit, `+`-joined — differs from `block` when a component group was absent from the table (e.g. `clinical` reduces to `age+bmi+tech` when race/eth are unavailable) |
| `n_patients` | eligible, restricted rows fit for this outcome |
| `n_prevalent` | positive cases within `n_patients` |
| `n_features` | design-matrix width for this block |
| `C_median` | median of the 5 outer folds' selected `LogisticRegression` `C` |
| `auroc`, `auroc_lo`, `auroc_hi` | pooled out-of-fold AUROC and its **unpaired** bootstrap interval (NaN bounds if `--subgroup-boot 0`) |
| `auprc`, `auprc_lift` | AUPRC, and the ratio to this outcome's own prevalence |
| `brier`, `brier_skill` | Brier score, and skill relative to a prevalence-only predictor |
| `cal_slope`, `cal_intercept` | calibration slope (1 = perfect) and intercept (0 = perfect); slope is NaN if the out-of-fold logit has near-zero spread |
| `prevalence` | positive rate in `n_patients` |
| `recall_at_0.5`, `precision_at_0.5`, `n_flagged_at_0.5` | operating point at a raw 0.5 probability threshold |
| `sens_at_90spec` | recall at the threshold giving ≥90% specificity |
| `recall_at_top10pct`, `precision_at_top10pct`, `recall_at_top20pct`, `precision_at_top20pct` | operating points at fixed alert rates |

### `deltas.csv`

One row per (outcome, pre-specified block comparison in `COMPARISONS`).

| column | meaning |
|---|---|
| `outcome`, `restricted_to` | as above |
| `baseline`, `augmented` | the smaller and larger block being compared (e.g. `mammo` → `mammo+cxr`) |
| `n_patients`, `n_prevalent` | as above |
| `delta` | `AUROC(augmented) - AUROC(baseline)` on the pooled predictions |
| `lo`, `hi` | 95% **paired** bootstrap interval on `delta` (same resampled patient indices score both arms each replicate) |
| `p_gt_0` | fraction of bootstrap replicates with `delta > 0` |

### `gains.csv`

One row per (outcome, block, alert rate in `GAINS_RATES = (0.05, 0.10, 0.20, 0.30, 0.50)`).

| column | meaning |
|---|---|
| `outcome`, `block` | as above |
| `alert_rate` | fraction of patients flagged, by predicted score, highest first |
| `n_flagged` | patient count corresponding to `alert_rate` |
| `true_positives_caught` | already-coded patients among those flagged |
| `total_positives` | total already-coded patients for this outcome |
| `recall` | `true_positives_caught / total_positives` |
| `precision` | `true_positives_caught / n_flagged` |
| `lift` | `recall / alert_rate` — how much better than flagging at random |

### `subgroups.csv`

One row per (outcome, block, stratifier, group level), for every block named
in `--subgroup-blocks` (default: `mammo+cxr` only) and every stratifier in
`--stratify-by` (default: all five — age, bmi, race, imaging_year,
index_gap).

| column | meaning |
|---|---|
| `outcome`, `restricted_to`, `block` | as above |
| `stratifier` | `age` / `bmi` / `race` / `imaging_year` / `index_gap` |
| `group` | the bracket/level within that stratifier (e.g. `50-64`, `Black`, `<=2017`) |
| `n`, `n_prevalent`, `prevalence` | as above, within this group only |
| `auroc`, `auroc_lo`, `auroc_hi`, `auprc`, `auprc_lift`, `cal_intercept` | as in `performance.csv`, computed on this group's rows only; NaN with a `note` if `n < --subgroup-min-n` (default 100) or the group is single-class |
| `note` | explanation when metrics were not estimated; empty otherwise |

**Read with care**: conditioning on a stratifier that itself predicts the
outcome (age, BMI) mechanically depresses within-bracket AUROC relative to the
pooled figure — that is case-mix compression, not evidence of a disparity. Use
`subgroup_deltas.csv` below for the comparison that survives this effect.

### `subgroup_deltas.csv`

One row per (outcome, pre-specified fusion comparison in
`SUBGROUP_COMPARISONS`, stratifier, group level). This, not `subgroups.csv`,
is the primary subgroup quantity — see `methods.md`.

| column | meaning |
|---|---|
| `outcome`, `restricted_to` | as above |
| `baseline`, `augmented` | the compared blocks, restricted to the pre-specified subset relevant to subgroup reporting |
| `stratifier`, `group` | as above |
| `n`, `n_prevalent`, `prevalence` | within this group |
| `delta`, `lo`, `hi`, `p_gt_0` | paired bootstrap on the within-group AUROC delta, same definition as `deltas.csv` but resampled within this group's rows only |
| `note` | present when `n < --subgroup-min-n` or the group is single-class |

### `coefficients.csv`

One row per (outcome, block, non-embedding feature). Embedding-column
coefficients are dropped by default (`named_coefficients(..., drop_embeddings=True)`)
— thousands of them say nothing a reader can act on.

| column | meaning |
|---|---|
| `outcome`, `restricted_to`, `block` | as above |
| `group` | the feature's group (`age`, `bmi`, `eth`, `race`, `tech`, or `demo_other`) |
| `feature` | column name |
| `coef` | fitted coefficient, mean over the 5 outer folds, on the **standardised, unscaled** footing — if `--clinical-scale != 1.0` was used, this has already been divided back so it means the same thing regardless of the flag |

## Nonlinear arm (`nonlinear.py`)

All `nl_*` tables share the same key columns at the front: `outcome`,
`restricted_to`, `arch` (`early`/`late`/`gated`), `config`
(`mammo`/`cxr`/`mammo+cxr`), `seed`. Every table has one row per
(outcome × arch × config × seed) unless noted otherwise.

### `nl_performance.csv`

| column | meaning |
|---|---|
| `outcome`, `restricted_to`, `arch`, `config`, `seed` | as above |
| `n_patients`, `n_prevalent` | as in the linear arm |
| `epochs_mean` | mean epochs run across the 5 outer folds |
| `early_stopped` | count of folds (of 5) that triggered early stopping rather than hitting `--max-epochs` |
| `dead_units` | mean fraction of near-zero-variance bottleneck activations across folds — a health check, not a metric of interest on its own |
| `seconds` | wall-clock time for this (arch, config, seed) fit |
| `auroc` … `precision_at_top20pct` | identical definitions to `performance.csv` |

### `nl_deltas.csv`

Same shape as `deltas.csv`, with `arch` and `seed` added — one row per
(outcome, arch, comparison in `COMPARISONS`, seed), since seeds are **not**
ensembled and each has its own paired-bootstrap delta.

### `nl_gains.csv`

Same shape as `gains.csv`, with `arch`, `config`, `seed` added.

### `nl_subgroups.csv`

Same shape as `subgroups.csv`, stratified on `Race` only (`--race-col`) rather
than all five axes, with `arch`, `config`, `seed` added; one row per (outcome,
arch, config, seed, race group).

### `nl_diagnostics.csv`

One row per (outcome, arch, config, seed, outer fold) — the only `nl_*` table
keyed by fold rather than pooled across folds.

| column | meaning |
|---|---|
| `outcome`, `arch`, `config`, `seed` | as above (note: no `restricted_to` in this table) |
| `fold` | outer fold index, 0-4 |
| `best_val_loss` | validation log-loss at the early-stopped (or final) epoch |
| `epochs_run` | epochs actually run before stopping |
| `stopped_early` | whether patience was exhausted before `--max-epochs` |
| `dead_units` | fraction of bottleneck units with near-zero variance on this fold's test rows |
| `mean_abs_act` | mean absolute bottleneck activation, a scale sanity check |
| `alpha` | `gated` architecture only: this fold's learned gate value if constant, else NaN (see `nl_gates.csv` for the per-patient distribution) |

### `nl_coefficients.csv`

`gated` architecture only (other architectures have no `wide` path). One row
per (outcome, arch, config, seed, clinical feature).

| column | meaning |
|---|---|
| `outcome`, `arch`, `config`, `seed` | as above |
| `feature` | clinical column name |
| `coef` | mean `wide`-path coefficient across the 5 outer folds |
| `source` | `"clinical offset (penalised LR, as linear.py)"` when `--no-clinical-offset` was *not* passed (the default), or the alternative jointly-learned path otherwise |
| `alpha_mean` | mean gate value across folds, when the gate collapsed to a near-constant (diagnostic, not a per-patient value) |

### `nl_gates.csv`

`gated` architecture only, and only for outcomes/configs where the gate
produced a finite value. One row per (outcome, arch, config, seed, group),
where `group` is `"ALL"` plus one row per race level with `n >=
--subgroup-min-n`.

| column | meaning |
|---|---|
| `outcome`, `arch`, `config`, `seed` | as above |
| `group` | `"ALL"`, or a race level |
| `n` | patients in this group |
| `gate_mean`, `gate_sd` | mean and sd of the per-patient gate value (fraction of the logit drawn from the mammography tower) in this group |
| `gate_p10`, `gate_p90` | 10th/90th percentile of the gate in this group |
| `collapsed` | `True` if `gate_sd < 0.01` — the gate has settled on one modality for every patient, which is not fusion and should be read as a null result on the gate, not as evidence both modalities contribute equally |
| `share_mammo`, `share_cxr` | mean relative contribution magnitude of each modality's logit contribution, when both are present (`CANON = ("mammo", "cxr")`) |

## Reporting scripts (not `linear.py`/`nonlinear.py` outputs, but consume them)

### `block_outcomes.csv` (from `block_report.py`)

One row per (outcome, block in `--blocks`, default `mammo cxr mammo+cxr`),
built by reading `performance.csv` and `deltas.csv` from `--indir`.

| column | meaning |
|---|---|
| `outcome`, `restricted_to`, `block` | carried through from `performance.csv` |
| `n_patients`, `n_prevalent`, `prevalence` | carried through |
| `auroc`, `auroc_lo`, `auroc_hi`, `auprc`, `auprc_lift`, `brier`, `cal_slope`, `cal_intercept` | carried through |
| `sens_at_90spec`, `recall_at_top10pct`, `precision_at_top10pct` | carried through |
| `delta_vs_mammo`, `vs_mammo_lo`, `vs_mammo_hi` | this block's paired-bootstrap AUROC delta against the `mammo`-only block, from `deltas.csv`, when `mammo` is one of this block's declared fusion baselines |
| `delta_vs_cxr`, `vs_cxr_lo`, `vs_cxr_hi` | same, against the `cxr`-only block |
| `reading` | plain-language sentence generated from this row's own numbers — never written by hand, so it cannot drift from the CSV it describes |

### `subgroup_accuracy.csv` (from `subgroup_reading.py`)

One row per (outcome, stratifier, group), flattening one block's rows from
`subgroups.csv` (`--block`, default: every block present) down to the columns
below, plus two generated ones.

| column | meaning |
|---|---|
| `outcome`, `stratifier`, `group`, `n`, `prevalence` | carried through from `subgroups.csv` |
| `auroc`, `auroc_lo`, `auroc_hi`, `auprc_lift`, `cal_intercept` | carried through from `subgroups.csv` |
| `prevalence_spread_on_axis` | max − min prevalence across this stratifier's levels — large values are the case-mix-compression warning sign |
| `comparable_across_levels` | `"no"` when `prevalence_spread_on_axis` is large enough that within-level AUROCs should not be compared to each other; empty when AUROC itself is NaN for this row |
| `reading` | plain-language sentence generated from the row's own numbers, including the comparability caveat when it applies — never written by hand, so it cannot drift from the CSV |

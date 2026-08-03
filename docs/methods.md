# Methods

## Cohort and eligibility

A patient enters the cohort if all of the following hold (`cohort.py`, `eligible_patients`):

- has a chest radiograph (CXR study),
- has mammography imaging,
- has an FFDM mammography embedding for at least one accession,
- has at least one ICD diagnosis row with a plausible, parseable date (1990-01-01 through 2025-12-31), and
- is female.

The female-only restriction is a modelling decision, not a data artefact: this is a screening-mammography cohort, and the \~43 excluded male patients cannot support their own estimates or a sex interaction term at this scale. Future work should explicitly explore male and gender diverse subgroup analyses.

**Index date.** `index_date = max(mammo_date, cxr_date)` — the later of the two imaging studies for that patient, so no imaging feature ever postdates the label anchor. The mammography accession used is the one nearest the CXR date (ties broken by lowest accession ID, for determinism).

**Fit-cohort window.** Only patients whose two studies fall within `--window` days of each other (default 730) enter the tables `linear.py` and `nonlinear.py` fit on. This is the only window; there is no sensitivity arm at a different cutoff pre-specified in this codebase.

## Outcomes: prevalence

`t2dm`, `htn`, `hld`, `ckd` are **prevalence at the index date**, not prediction of future onset. A patient is positive for an outcome if any ICD-9/ICD-10 code matching that outcome's pattern carries `dx_date <= index_date`; a patient with no such code is a true negative, because cohort eligibility already requires at least one dated diagnosis row (absence of the code means absence of the condition, not absence of the record). See `cohort.py`'s `code_masks` for the exact ICD patterns and `common.py`'s module docstring for why this changes what "eligible" and "positive" mean relative to an incident-onset design.

`ckd` has a pre-specified secondary analysis, **CKD among hypertensives** (`RESTRICTIONS = {"ckd": "htn"}` in `common.py`): a large majority of CKD-positive patients are also HTN-positive (`cohort.py`'s `build_labels` prints the exact overlap for the cohort actually built), read as a clinical comorbidity (hypertension is a principal cause of chronic kidney disease) rather than a labelling artefact of the codes the two outcomes happen to share (`40[34]`/`I1[23]` route to both). **Caveat:** an earlier version of the cohort-build script substantiated that "comorbidity, not artefact" reading by rebuilding CKD from pure-renal codes alone (`585`/`N18`/`Z992`, excluding the shared codes) and showing the overlap held up on that subset too; that rebuild-and-compare step did not carry over into the current `cohort.py` and is not run today, so treat the comorbidity conclusion as a finding from the original analysis rather than something this codebase re-verifies on every build.

## Feature blocks

Every feature column belongs to exactly one named group (`common.py`'s `GROUPS`): `age`, `bmi`, `eth`, `race`, `tech`, `mammo`, `cxr`, plus a `demo_other` catch-all so an upstream covariate never silently vanishes from every block. A **block** is a named union of groups (e.g. `mammo+cxr`, `clinical+mammo+cxr`); see `linear.py`'s `BLOCKS` for the full list and `nonlinear.py`'s `CONFIGS` for the three modality configurations (`mammo`, `cxr`, `mammo+cxr`) each architecture is fit on.

**Race and ethnicity are not features by default.** `--demographics` (default `age bmi`) controls which demographic groups may enter any model; race and ethnicity are held out and analysed only as post-hoc subgroup stratifiers — the raw `Race` column rides along unprefixed and unpenalised, used solely to compute per-group metrics. This is a modelling decision, not a data limitation: a model that used race/ethnicity to detect prevalent disease would partly be encoding differential access to diagnosis rather than physiology, and holding them out means the subgroup tables measure whether a race-*blind* model still performs unequally across racial groups, which is the harder and more informative question.

## Model arms

**Arm 1 — linear (`linear.py`).** One L2-penalised logistic regression per block. Each block's columns are standardised (`StandardScaler`) before concatenation, so a block with larger raw norms cannot absorb more of the shared penalty budget purely from scale.

**Arms 2-4 — nonlinear (`nonlinear.py`).** Three small neural architectures over the same frozen embeddings, each fit for `mammo`-only, `cxr`-only, and `mammo+cxr`, so the fusion delta is measured *within* an architecture rather than across model families:

- `early` — concatenate both embedding blocks, two hidden layers, then a classifier head with the clinical block concatenated in.
- `late` — a separate tower per modality, concatenated before the classifier head.
- `gated` — a separate tower per modality with a per-patient learned gate mixing their logit contributions, plus a **direct clinical path** (`wide`) that bypasses the deep network entirely and carries its own zero weight decay — the structural analogue of `--clinical-scale` below, and see `--clinical-scale` for why that matters.

All three read their clinical block from `clinical_columns()`, which is gated by the same `--demographics` flag as the linear arm, so a demographic held out of one arm is held out of both.

## Cross-validation

Both arms use the **same** `StratifiedKFold(n_outer=5, shuffle=True, random_state=seed)` split on the same rows, so a delta between the linear and nonlinear arms at a given seed compares two models on identical folds rather than on incidentally different data splits. Indexing is positional throughout in both arms.

**Linear arm — nested CV.** Inside each of the 5 outer folds, the regularisation strength `C` is chosen by `GridSearchCV` over `np.logspace(-4, 1, 12)`, itself validated by 3-fold inner `StratifiedKFold` scored on negative log-loss (`--n-inner`, default 3). The outer fold's test predictions come from the pipeline refit at the best inner `C` on the full outer-training data. No test patient's label ever contributes to the `C` selection that scores that same patient.

**Nonlinear arm — early stopping, not a second grid search.** Each outer training fold is split again (`train_test_split`, stratified, `val_frac` in `DEFAULT_HP`, default 0.2 — not currently exposed as its own CLI flag) into a fitting set and a validation set used only for early stopping (patience `--patience`, default 10, on validation log-loss — not AUPRC, which is too noisy at typical event counts to select an epoch on). The `gated` architecture's clinical offset (`fit_clinical_offset`) *does* run a real inner grid search, on the same `C` grid and the same `neg_log_loss` criterion as the linear arm, fit on the fitting rows only — so that offset is literally the linear arm's clinical model, not a separately-tuned one that happens to share columns.

In both arms, every preprocessing step (`StandardScaler` in the linear arm; row L2-normalisation + per-dimension `StandardScaler`, `FoldScaler`, in the nonlinear arm) is fit on the fold's training rows only and applied unchanged to that fold's validation/test rows — never fit on the full table.

## `--clinical-scale`

`--clinical-scale k` (linear arm only; default `1.0`, a no-op) multiplies the *standardised* non-embedding features (age, BMI, race, ethnicity, tech) by a constant `k` before they enter the shared logistic regression.

**Why it exists.** A fused block shares one L2 penalty budget across every column. When a handful of clinical features sit alongside thousands of embedding dimensions, the penalty flattens the clinical coefficients toward zero — measured directly in this codebase: at \~1,280 embedding columns, BMI retained only 3% of the coefficient it had in a clinical-only fit. That is a property of the shared penalty, not evidence that BMI stopped mattering clinically.

**Why scaling fixes it.** A feature scaled up by `k` needs a coefficient `k` times smaller to produce the same effect on the logit, so its penalty contribution (which is quadratic in the coefficient) falls by `k²`. Scaling relieves the clinical block from competing on equal footing against thousands of embedding columns for the same penalty budget — it does not touch the likelihood term, so a feature with no real signal still gets pushed toward a near-zero coefficient regardless of `k`.

**What it does not do.** It does not inflate a clinical feature's importance or manufacture signal: the data still determines the coefficient, only the penalty changes, and as `k → ∞` the clinical coefficients converge to their *unpenalised* maximum-likelihood values — never beyond. Coefficients are reported on their original (unscaled) footing regardless of `k`: after fitting, each affected coefficient is multiplied back by `k` before being returned (`crossval_predict`), specifically so a reported coefficient always means the same thing — "effect per one standard deviation of the raw feature" — independent of the scale flag used to fit it.

**Current limitation.** `k` is a hand-chosen CLI flag, not something tuned or validated. Nothing in this codebase currently reports a sensitivity curve over `k`, includes it in the inner-CV grid, or otherwise justifies a specific value beyond the illustrative `k=10` example in the module docstring.

## Metrics

Reported per block/architecture/outcome (`common.py`'s `metrics`): AUROC (with an optional unpaired bootstrap interval, `auroc_lo`/`auroc_hi`), AUPRC (with `auprc_lift` — the ratio to that outcome's own prevalence, since a no-signal model's AUPRC equals the base rate), Brier score and Brier skill score, calibration slope and intercept, plus operating-point summaries at a 0.5 probability threshold and at a fixed 90% specificity.

**Calibration intercept** (`calibration_intercept`) is calibration-in-the-large: the constant shift needed to make mean predicted risk match observed prevalence, holding the slope at 1. This is the error that matters most under a prevalence design, where base rates sit far higher than an incident-onset design would produce, and slope alone cannot see it.

**Calibration slope** (`calibration_slope`) refits `y ~ a + b * logit`; `b = 1` is perfect, `b < 1` is over-confident. Returned as `NaN` when the out-of-fold logit has near-zero spread (`MIN_LOGIT_SD = 1e-3`) — a heavily penalised small block (e.g. `race` alone) can be pushed almost to a constant, and regressing on a predictor with no spread would return an arbitrary large slope that reads as catastrophic miscalibration rather than "nothing to calibrate," which the AUROC alongside it already says.

**Gains** (`gains_table`) reports recall and precision at fixed alert rates (5/10/20/30/50% of patients flagged by predicted score, `GAINS_RATES`), read as "flag the top R% and capture this fraction of everyone already coded" — with lift (`recall / R`) as the honest summary of how much a model buys over flagging at random, which itself captures `R%` by construction.

## Paired vs. unpaired intervals

Two different bootstrap intervals appear in these tables and they answer different questions:

- **`auroc_lo`/`auroc_hi`** (in `performance.csv`/`nl_performance.csv`) is an **unpaired** percentile bootstrap over patients, resampling one block's predictions alone. It answers "how precisely is this block's own AUROC pinned down."
- **`delta_auc`** (in `deltas.csv`, `nl_deltas.csv`, `subgroup_deltas.csv`) is a **paired** bootstrap: the same resampled patient indices are used to score both the baseline and the augmented block on each replicate, so between-arm correlation in the resampling cancels out of the difference. This interval is much tighter than the gap between two unpaired intervals would suggest, and it — not the overlap of two `auroc_lo`/`auroc_hi` ranges — is what should decide whether fusion helped.

`subgroup_deltas` applies the same paired logic within one stratifier level at a time, which is the primary subgroup quantity for exactly this reason: both arms are scored on the same patients in that bracket, so case-mix compression (a stratifier that itself predicts the outcome mechanically depressing within-bracket AUROC) affects both arms equally and cancels out of the difference, even though it distorts the pooled per-block AUROC in that bracket.

Every interval reported anywhere in these tables is **unadjusted for multiplicity**. With five stratifiers × several levels × multiple comparisons × four outcomes, treat only the pre-specified fusion comparisons (`SUBGROUP_COMPARISONS` in `linear.py`) as confirmatory; everything else is description, and an interval that just clears zero in one bracket of one outcome is exactly what multiplicity produces on its own.

## What is reproducible, and what is not

**Linear arm — fully seeded, deterministic given the same inputs.** Every random draw (`StratifiedKFold` splits, `GridSearchCV`'s inner splits, every bootstrap) is driven by `--seed` via `np.random.default_rng`/ `random_state=seed`. Given the same cohort table and the same flags, the linear arm's outputs are bit-for-bit reproducible across runs and across machines (`scikit-learn`'s `LogisticRegression` with `lbfgs`, the default solver, is deterministic).

**Nonlinear arm — seeded, but not guaranteed bit-identical across hardware.** `torch.manual_seed(seed * 1000 + fold)` is set per fold, and numpy's RNG is seeded the same way as the linear arm. However, this codebase does **not** set `torch.use_deterministic_algorithms(True)` or pin cuDNN/MPS to deterministic kernels, so floating-point results from GPU- or MPS-accelerated convolution/matmul implementations can differ in the last few bits across runs, devices, and PyTorch versions, even at a fixed seed. In practice this means: re-running the nonlinear arm on the *same machine* with the same PyTorch/device should reproduce results very closely; comparing exact per-epoch numbers across different machines or `--device` choices (`cuda` vs `mps` vs `cpu`) is not guaranteed and small AUROC/loss differences at that level should not be read as a bug.

**Across seeds.** The nonlinear arm's `--seeds` are deliberately **not** ensembled — each seed produces its own out-of-fold prediction vector and its own `delta_auc`, and both variance sources are reported separately: `summarise_seeds` prints the AUROC mean/sd *across* seeds (initialisation variability), while each seed's own paired bootstrap interval captures *within-seed* patient-sampling variability. A fusion gain is only credible if it clears both, which is why `nl_performance.csv`/`nl_deltas.csv` keep every seed as its own row rather than pre-averaging them away.

**Data.** Nothing about reproducibility here extends to the real cohort itself — see [`docs/data.md`](data.md) for what the source data is and how to obtain it. The synthetic fallback (`common.simulate`, `simulate.py`) is itself seeded and fully reproducible, which is what makes it useful as a standing regression check independent of data access.
"""
Nonlinear fusion arms for the mammography + CXR question.

linear.py answers "does combining the two embeddings help" with penalised
logistic regression. This module asks the same question of three small neural
heads over the same frozen embeddings, so a null result there is not just an
artifact of having assumed linearity.

Three architectures, each fit for mammography alone, CXR alone, and fused, so
the fusion delta is measured *within* an architecture rather than across model
families:

  early   concat[mammo, cxr] -> 512 -> 128 -> classifier(+clinical)
  late    per-modality tower -> 128 each -> concat -> classifier(+clinical)
  gated   per-modality tower -> 128 each -> per-patient gated logit mix,
          plus a direct clinical path that bypasses the deep network

The third is the one designed here rather than specified. Its motivation is a
number measured in the linear arm: under a single shared L2 penalty across 2,304
embedding columns, BMI retained 3% of the coefficient it had without the
embeddings, age 20%, race_Black 18%. The clinical covariates were being crushed
by the penalty budget of features they were competing with. `gated` gives them
their own path to the logit with their own (zero) weight decay, so they cannot be
crushed at all, and the three architectures form a gradient on that defect:

  linear.py fused   clinical competes with 2,304 raw columns   severe
  early             competes with 128 hidden units             reduced
  late              competes with 256 hidden units             reduced
  gated             competes with nothing                      none by design

Race and ethnicity are not features by default. `--demographics` selects which
demographic groups may be used as inputs, defaulting to age and bmi, and it gates
all three architectures at once because they all read their clinical block from
clinical_columns(). Race is still carried as a raw column and still stratifies
the subgroup report, which is the post-hoc analysis; excluding it as an input
does not make that reporting less necessary, since a model blind to race can
still perform unequally across racial groups.

Everything about the evaluation comes from common.py -- the fold splits, the
eligibility rules, the metrics, the paired bootstrap, the subgroup breakdown --
which linear.py shares rather than exports. That is deliberate. A delta
between this module and linear.py is only meaningful if both were scored the
same way on the same patients in the same folds, and both now read that scoring
from one place rather than one importing it from the other.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from . import common
from .common import (DEFAULT_DEMOGRAPHICS, DEMOGRAPHIC_GROUPS, EMBEDDING_GROUPS,
                     OUTCOMES, RESTRICTIONS, apply_restriction, assign_groups,
                     cap_index_gap, delta_auc, eligible_mask, gains_table, metrics,
                     simulate, subgroups)

# Canonical modality order. Fixed here so the gate always means "weight on
# mammography" and never silently flips because a dict iterated differently.
CANON = ("mammo", "cxr")

CONFIGS = {
    "mammo": ("mammo",),
    "cxr": ("cxr",),
    "mammo+cxr": ("mammo", "cxr"),
}

ARCHS = ("early", "late", "gated")

# Within an architecture, so a gain is attributable to adding a modality rather
# than to the choice of architecture.
COMPARISONS = [("mammo", "mammo+cxr"), ("cxr", "mammo+cxr")]


def pick_device(requested=None):
    """cuda -> mps -> cpu. The server has GPUs; this machine has MPS."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clinical_columns(df, demographics=DEFAULT_DEMOGRAPHICS):
    """
    Non-embedding feature columns, in common.py's group order.

    Reuses common.block_columns so the clinical design matrix here is exactly
    the one the linear arm fits on, including its ordering -- which is what
    makes the gated arm's offset coefficients comparable to linear.py's
    clinical block.

    A demographic group is included only if `demographics` requests it. Race and
    ethnicity are excluded by default and analysed post hoc; see
    common.DEFAULT_DEMOGRAPHICS. This gates every architecture at once, because
    all three read their clinical block from here: the early and late arms
    concatenate it at the classifier, and the gated arm fits its offset on it.
    """
    groups = [g for g in common.GROUPS
              if g not in EMBEDDING_GROUPS
              and (g not in DEMOGRAPHIC_GROUPS or g in demographics)]
    return common.block_columns(df, groups)


def split_blocks(df, modalities, demographics=DEFAULT_DEMOGRAPHICS):
    """
    Split the frame into float32 arrays: one per requested embedding block, plus
    clinical. Returns (blocks, clinical, clinical_columns).
    """
    assigned = assign_groups(df)
    blocks = {}
    for m in CANON:
        if m in modalities:
            cols = assigned.get(m, [])
            if not cols:
                raise KeyError(f"no '{m}_*' columns in this table")
            blocks[m] = df[cols].to_numpy(dtype=np.float32)
    clin_cols = clinical_columns(df, demographics)
    clin = df[clin_cols].to_numpy(dtype=np.float32)
    return blocks, clin, clin_cols


class FoldScaler:
    """
    Row L2-normalise each embedding block, then standardise every column.

    Two separate steps, for two separate reasons.

    Row L2 first, because the blocks arrive with incomparable geometry:
    mammography-E rows have norm sqrt(2) (two independently unit-normalised 768-d
    halves) while CXR-E rows run 43 to 52. Feeding both into one shared first
    layer at those scales lets CXR dominate by raw magnitude before any learning
    happens. cohort.py's own manifest note says to normalise in-fold and
    this is that step. It is a per-row operation, so it uses no information from
    other patients and cannot leak.

    Per-dimension standardisation second, fitted on the fitting rows only. The
    clinical block is standardised but not row-normalised -- its columns are
    unrelated quantities (years, kg/m2, indicators), so a row norm across them
    would be meaningless.
    """

    def __init__(self):
        self.scalers = {}
        self.fit_rows = None

    @staticmethod
    def _row_l2(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, 1e-12)

    def _prep(self, name, X):
        return X if name == "clinical" else self._row_l2(X)

    def fit(self, blocks, n_rows):
        for name, X in blocks.items():
            self.scalers[name] = StandardScaler().fit(self._prep(name, X))
        self.fit_rows = n_rows
        return self

    def transform(self, blocks):
        if not self.scalers:
            raise RuntimeError("FoldScaler.transform before fit")
        return {name: self.scalers[name].transform(self._prep(name, X)).astype(np.float32)
                for name, X in blocks.items()}


class EarlyFusion(nn.Module):
    """
    Concatenate the embeddings, then two hidden layers, then classify.

    GeLU and dropout are applied after *both* hidden layers. Two consecutive
    linear maps with no activation between them compose to a single linear map,
    which would make the first projection pointless -- so the activation after
    W1 is load-bearing, not decoration.

    The clinical features join at the final layer, so they are concatenated to
    the 128-d bottleneck rather than being pushed through the trunk.
    """

    def __init__(self, dims, d_clin, hidden=512, bottleneck=128, dropout=0.3):
        super().__init__()
        self.names = [m for m in CANON if m in dims]
        d_emb = sum(dims[m] for m in self.names)
        self.trunk = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(bottleneck + d_clin, 1)

    def forward(self, blocks, clin, offset=None, want_extras=False):
        h = self.trunk(torch.cat([blocks[m] for m in self.names], 1))
        logit = self.head(torch.cat([h, clin], 1)).squeeze(1)
        return (logit, {"bottleneck": h}) if want_extras else logit


class LateFusion(nn.Module):
    """
    One tower per modality to 128 dimensions, concatenated, then classified.

    Each modality is encoded independently and fused at the representation level.
    GeLU and dropout are inside each tower: without a nonlinearity the tower and
    the classifier would again collapse into one linear map, making this a
    rank-128 logistic regression rather than a nonlinear model.
    """

    def __init__(self, dims, d_clin, bottleneck=128, dropout=0.3):
        super().__init__()
        self.names = [m for m in CANON if m in dims]
        self.towers = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(dims[m], bottleneck), nn.GELU(), nn.Dropout(dropout))
            for m in self.names})
        self.head = nn.Linear(bottleneck * len(self.names) + d_clin, 1)

    def forward(self, blocks, clin, offset=None, want_extras=False):
        z = [self.towers[m](blocks[m]) for m in self.names]
        logit = self.head(torch.cat(z + [clin], 1)).squeeze(1)
        return (logit, {"bottleneck": torch.cat(z, 1)}) if want_extras else logit


class GatedWideDeep(nn.Module):
    """
    logit = clinical . beta  +  [ g * f_mammo + (1 - g) * f_cxr ]  +  bias

    Two design choices, both deliberate.

    The clinical features get their own linear path straight to the logit. They
    do not pass through the towers, are not subject to dropout, and are given
    zero weight decay, so the embedding penalty cannot shrink them. This is the
    structural version of linear.py's --clinical-scale patch, and it makes the
    `wide` coefficients directly comparable to the linear arm's clinical
    coefficients.

    The gate mixes the two modalities' *logit contributions*, not their 128-d
    representations. Mixing representations would require z_mammo and z_cxr to
    occupy a shared aligned space so one head could read either, which is a
    strong assumption to impose at n≈5,000. Gating scalars instead lets each
    modality keep its own head.

    Each contribution is batch-normalised to zero mean and unit variance before
    gating, and that is not cosmetic -- without it the gate is not identifiable.
    `g * f_m + (1 - g) * f_c` with freely-scaled heads is over-parameterised: the
    model can halve g and double f_m for identical predictions. Verified on
    simulated data where the CXR block was replaced with pure noise: the
    un-normalised gate sat at 0.500 (sd 0.014) and never moved, because the model
    simply shrank the noise head instead of closing the gate. Fixing both
    contributions to unit variance removes that escape route and makes g the only
    free mixing parameter, so it has to move.

    Even so, `g` is reported alongside the *contribution share*
    mean|g*f_m| / (mean|g*f_m| + mean|(1-g)*f_c|), which measures what each
    modality actually contributed and stays interpretable regardless of
    parameterisation. Prefer the share when reading results.

    The clinical term is a FIXED OFFSET, not a jointly-trained layer, and that
    detail was forced by a measurement. The first version of this class learned
    `wide` jointly with the towers. It did not work: on simulated data where age
    and adiposity drive hypertension strongly and positively, the fitted wide
    coefficients came out at -0.008 for age and -0.052 for BMI. Being unpenalised
    is not enough to protect a feature. Both paths descend the same loss, the deep
    path has orders of magnitude more capacity, and it absorbs the age/adiposity
    signal through the embeddings -- which encode body habitus -- leaving the
    clinical path fitting a near-zero residual with arbitrary sign.

    So the clinical model is fitted first, on the training rows only, by ordinary
    penalised logistic regression with C chosen by inner CV on log-loss, exactly
    as linear.py does. Its decision function enters as an additive offset:

        logit = alpha * clinical_offset  +  gated imaging  +  bias

    Now three things are true that were not before. The clinical coefficients are
    literally linear.py's clinical coefficients, so comparing them is meaningful
    rather than aspirational. The imaging term is unambiguously *incremental over
    the clinical model*, which is the question the datathon actually asks. And
    `alpha` is diagnostic: it starts at 1.0, and drifting far from 1 means the
    clinical model needed rescaling once imaging was present.

    `--no-clinical-offset` restores the jointly-learned version for comparison.

    With one modality the gate is undefined, so it is omitted and g is reported
    as 1.0 -- the model is then the clinical offset plus a single imaging tower.
    """

    def __init__(self, dims, d_clin, bottleneck=128, dropout=0.3, use_offset=True):
        super().__init__()
        self.names = [m for m in CANON if m in dims]
        self.use_offset = use_offset
        self.towers = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(dims[m], bottleneck), nn.GELU(), nn.Dropout(dropout))
            for m in self.names})
        self.heads = nn.ModuleDict({m: nn.Linear(bottleneck, 1) for m in self.names})
        # affine=False: scale and shift would immediately undo the normalisation
        # and restore the identifiability problem it exists to remove.
        self.norms = (nn.ModuleDict({m: nn.BatchNorm1d(1, affine=False) for m in self.names})
                      if len(self.names) > 1 else None)
        self.gate = nn.Linear(bottleneck * len(self.names), 1) if len(self.names) > 1 else None
        if self.gate is not None:
            # Start at sigmoid(0) = 0.5 so neither modality is favoured before
            # training. Default init would start the gate at a random offset.
            nn.init.zeros_(self.gate.bias)
            nn.init.zeros_(self.gate.weight)
        if use_offset:
            self.wide = None
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.wide = nn.Linear(d_clin, 1)
            self.alpha = None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, blocks, clin, offset=None, want_extras=False):
        z = {m: self.towers[m](blocks[m]) for m in self.names}
        raw = {m: self.heads[m](z[m]) for m in self.names}            # (B, 1) each
        if self.norms is not None:
            raw = {m: self.norms[m](raw[m]) for m in self.names}
        f = {m: v.squeeze(1) for m, v in raw.items()}

        contrib = {}
        if self.gate is None:
            only = self.names[0]
            g = torch.ones_like(f[only])
            contrib[only] = f[only]
        else:
            g = torch.sigmoid(self.gate(torch.cat([z[m] for m in self.names], 1))).squeeze(1)
            first, second = self.names[0], self.names[1]
            contrib[first] = g * f[first]
            contrib[second] = (1.0 - g) * f[second]
        deep = sum(contrib.values())

        if self.use_offset:
            if offset is None:
                raise ValueError("GatedWideDeep(use_offset=True) needs a clinical offset")
            base = self.alpha * offset
        else:
            base = self.wide(clin).squeeze(1)

        logit = base + deep + self.bias
        if not want_extras:
            return logit
        return logit, {
            "gate": g,
            "contrib": contrib,
            "bottleneck": torch.cat([z[m] for m in self.names], 1),
            "alpha": (float(self.alpha) if self.alpha is not None else float("nan")),
        }


BUILDERS = {"early": EarlyFusion, "late": LateFusion, "gated": GatedWideDeep}


def fit_clinical_offset(clin_fit, y_fit, seed, n_inner=3):
    """
    Penalised logistic regression on the clinical block alone, returning the
    fitted model and its coefficients.

    C is chosen by inner cross-validation on log-loss, the same criterion and the
    same search grid linear.py uses, so the offset this produces is the linear
    arm's clinical model rather than a different one that happens to use the same
    columns. Fitted on the fitting rows only; the decision function is then
    evaluated on validation and test rows without refitting.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(
        LogisticRegression(max_iter=5000),
        {"C": np.logspace(-4, 1, 12)},
        cv=StratifiedKFold(n_inner, shuffle=True, random_state=seed),
        scoring="neg_log_loss", n_jobs=-1,
    )
    search.fit(clin_fit, y_fit)
    best = search.best_estimator_
    return best, best.coef_[0].copy(), float(search.best_params_["C"])


def param_groups(model, weight_decay):
    """
    Weight decay on the embedding path only.

    The clinical path and every bias are exempt. For `gated` that exemption is
    the whole point -- a penalised direct path would reintroduce the crushing it
    exists to prevent. For `early` and `late` the clinical weights live inside
    the final layer and cannot be separated from it, so there the exemption
    applies only to biases and the effect is smaller. That asymmetry is real and
    is why the architectures are expected to differ in how well the covariates
    survive.
    """
    decayed, exempt = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        free = (name.startswith("wide") or name.endswith("bias")
                or name in ("bias", "alpha"))
        (exempt if free else decayed).append(p)
    return [{"params": decayed, "weight_decay": weight_decay},
            {"params": exempt, "weight_decay": 0.0}]


def to_device(blocks, clin, y, device):
    out = {m: torch.from_numpy(v).to(device) for m, v in blocks.items()}
    return (out,
            torch.from_numpy(clin).to(device),
            torch.from_numpy(y.astype(np.float32)).to(device))


def train_one(model, train, val, device, lr, weight_decay, batch, max_epochs, patience, seed):
    """
    Fit with early stopping on validation log-loss.

    Log-loss rather than validation AUPRC. The validation slice is 16% of the
    cohort, which for CKD is roughly 130 events -- AUPRC over that few positives
    is noisy enough that selecting an epoch on it partly fits the validation
    split. Log-loss uses every validation patient's predicted probability, so it
    is far smoother, and it matches linear.py's inner criterion exactly
    (GridSearchCV(scoring="neg_log_loss")), keeping model selection identical
    across the two arms. AUPRC is still recorded per epoch for the log.
    """
    opt = torch.optim.AdamW(param_groups(model, weight_decay), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = len(train["y"])

    gen = torch.Generator()
    gen.manual_seed(seed)

    best_loss, best_state, bad, epochs_run = math.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            # BatchNorm cannot compute a variance from one row, and a singleton
            # gradient step is worthless anyway.
            if idx.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            out = model({m: v[idx] for m, v in train["blocks"].items()},
                        train["clin"][idx],
                        offset=None if train["offset"] is None else train["offset"][idx])
            lossf(out, train["y"][idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(val["blocks"], val["clin"], offset=val["offset"]),
                                val["y"]))
        epochs_run = epoch + 1

        if vloss < best_loss - 1e-6:
            best_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": best_loss, "epochs_run": epochs_run,
            "stopped_early": bad >= patience}


def crossval(df, y, arch, modalities, seed, hp, device, verbose=True):
    """
    Pooled out-of-fold predictions, matching linear.py's protocol exactly.

    Same StratifiedKFold(5, shuffle=True, random_state=seed) on the same rows, so
    a delta against the linear arm at the same seed compares two models on
    identical folds. Indexing is positional throughout, as in linear.py.

    Each outer training fold is split again to give early stopping a validation
    set. The scaler is fitted on the *fitting* rows only, not on the validation
    rows, so nothing outside the 64% used for gradient steps informs the
    preprocessing.
    """
    blocks_all, clin_all, clin_cols = split_blocks(
        df, modalities, hp.get("demographics", DEFAULT_DEMOGRAPHICS))
    dims = {m: v.shape[1] for m, v in blocks_all.items()}

    use_offset = arch == "gated" and hp.get("clinical_offset", True)

    n = len(y)
    proba = np.full(n, np.nan)
    logit = np.full(n, np.nan)
    gate = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    wide_coefs, diags, alphas, shares = [], [], [], []

    outer = StratifiedKFold(hp["n_outer"], shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(outer.split(np.zeros(n), y)):
        fit_i, val_i = train_test_split(
            tr, test_size=hp["val_frac"], stratify=y[tr], random_state=seed)

        scaler = FoldScaler().fit({m: blocks_all[m][fit_i] for m in dims}
                                  | {"clinical": clin_all[fit_i]}, len(fit_i))

        # The clinical model is fitted on the standardised fitting rows only, so
        # the offset carries no information from validation or test patients.
        if use_offset:
            cf = scaler.transform({"clinical": clin_all[fit_i]})["clinical"]
            lr_model, lr_coef, lr_C = fit_clinical_offset(cf, y[fit_i], seed)
            wide_coefs.append(lr_coef)

        def prep(idx):
            t = scaler.transform({m: blocks_all[m][idx] for m in dims}
                                 | {"clinical": clin_all[idx]})
            b, c, yy = to_device({m: t[m] for m in dims}, t["clinical"], y[idx], device)
            off = None
            if use_offset:
                off = torch.from_numpy(
                    lr_model.decision_function(t["clinical"]).astype(np.float32)).to(device)
            return {"blocks": b, "clin": c, "y": yy, "offset": off}

        train, val, test = prep(fit_i), prep(val_i), prep(te)

        torch.manual_seed(seed * 1000 + fold)
        kwargs = dict(hp.get("arch_kwargs", {}))
        if arch == "gated":
            kwargs["use_offset"] = use_offset
        model = BUILDERS[arch](dims, len(clin_cols),
                               dropout=hp["dropout"], **kwargs).to(device)

        info = train_one(model, train, val, device, hp["lr"], hp["weight_decay"],
                         hp["batch"], hp["max_epochs"], hp["patience"], seed * 1000 + fold)

        model.eval()
        with torch.no_grad():
            out, extras = model(test["blocks"], test["clin"], offset=test["offset"],
                                want_extras=True)
        out = out.float().cpu().numpy()

        # Written exactly once per test row; asserted after the loop.
        if covered[te].any():
            raise AssertionError(f"fold {fold} overlaps an earlier test fold")
        covered[te] = True
        logit[te] = out
        proba[te] = 1.0 / (1.0 + np.exp(-out))
        if "gate" in extras:
            gate[te] = extras["gate"].float().cpu().numpy()
        if "contrib" in extras and len(extras["contrib"]) > 1:
            mags = {m: float(v.abs().mean()) for m, v in extras["contrib"].items()}
            total = sum(mags.values())
            shares.append({m: (v / total if total else float("nan"))
                           for m, v in mags.items()})

        if not use_offset and getattr(model, "wide", None) is not None:
            wide_coefs.append(model.wide.weight.detach().float().cpu().numpy().ravel())
        if "alpha" in extras and np.isfinite(extras["alpha"]):
            alphas.append(extras["alpha"])

        # A bottleneck unit with no variance across the test fold is dead: it
        # contributes a constant and the capacity it represents is not being used.
        act = extras["bottleneck"].float().cpu().numpy()
        diags.append({"fold": fold, **info,
                      "dead_units": float((act.std(axis=0) < 1e-4).mean()),
                      "mean_abs_act": float(np.abs(act).mean()),
                      "alpha": extras.get("alpha", float("nan"))})
        if verbose:
            atxt = (f" alpha {extras['alpha']:.3f}"
                    if "alpha" in extras and np.isfinite(extras["alpha"]) else "")
            print(f"      fold {fold}: {info['epochs_run']:>3} epochs "
                  f"(early={info['stopped_early']}) val_loss {info['best_val_loss']:.4f} "
                  f"dead {diags[-1]['dead_units']:.1%}{atxt}", flush=True)

    if not covered.all():
        raise AssertionError(f"{(~covered).sum()} patients never received a prediction")

    coef = (pd.Series(np.mean(wide_coefs, axis=0), index=clin_cols)
            if wide_coefs else None)
    share = ({m: float(np.mean([s[m] for s in shares])) for m in shares[0]}
             if shares else None)
    return {"proba": proba, "logit": logit, "gate": gate, "share": share,
            "diagnostics": pd.DataFrame(diags), "wide_coef": coef,
            "coef_source": ("clinical offset (penalised LR, as linear.py)" if use_offset
                            else "jointly-learned wide layer"),
            "alpha_mean": float(np.mean(alphas)) if alphas else float("nan")}


def run(df, outcomes=OUTCOMES, archs=ARCHS, configs=None, seeds=(0, 1, 2), hp=None,
        device=None, n_boot=2000, restrictions=None, race_col="Race",
        subgroup_min_n=100, ignore_incident_eligibility=False):
    hp = {**DEFAULT_HP, **(hp or {})}
    configs = CONFIGS if configs is None else configs
    restrictions = RESTRICTIONS if restrictions is None else restrictions
    device = pick_device(device)
    print(f"device {device}  |  seeds {list(seeds)}  |  archs {list(archs)}  |  "
          f"configs {list(configs)}", flush=True)

    demographics = tuple(hp.get("demographics", DEFAULT_DEMOGRAPHICS))
    assigned = assign_groups(df)
    clin_cols = clinical_columns(df, demographics)
    print(f"clinical features ({len(clin_cols)}): {clin_cols}", flush=True)

    held = [g for g in DEMOGRAPHIC_GROUPS if g not in demographics and assigned.get(g)]
    if held:
        cols = [c for g in held for c in assigned[g]]
        print(f"held out of every architecture, by --demographics: {held} -> {cols}", flush=True)
    if race_col in df.columns:
        print(f"post-hoc only: '{race_col}' stratifies the subgroup report, not a feature",
              flush=True)
    for m in CANON:
        print(f"{m:6s} block: {len(assigned.get(m, []))} columns", flush=True)

    perf, deltas, gains, subs, gates, coefs, diags = [], [], [], [], [], [], []

    for outcome in outcomes:
        sub = df[eligible_mask(df, outcome, ignore_incident_eligibility)]
        sub, restricted_by = apply_restriction(sub, outcome, restrictions)
        y = sub[outcome].to_numpy().astype(int)
        n_neg = len(y) - int(y.sum())
        if y.sum() < 30 or n_neg < 30:
            print(f"skipping {outcome}: {y.sum()} prevalent, {n_neg} unaffected", flush=True)
            continue

        label = f"{outcome} among {restricted_by}" if restricted_by else outcome
        print(f"\n=== {label}  n={len(y)}  prevalent={y.sum()} ({y.mean():.1%}) ===", flush=True)

        for arch in archs:
            store = {}
            for cname, modalities in configs.items():
                for seed in seeds:
                    t0 = time.time()
                    print(f"  {arch}/{cname} seed {seed}", flush=True)
                    res = crossval(sub, y, arch, modalities, seed, hp, device)
                    store[(cname, seed)] = res

                    m = metrics(y, res["proba"], res["logit"])
                    d = res["diagnostics"]
                    perf.append({"outcome": outcome, "restricted_to": restricted_by or "",
                                 "arch": arch, "config": cname, "seed": seed,
                                 "n_patients": len(y), "n_prevalent": int(y.sum()),
                                 "epochs_mean": float(d["epochs_run"].mean()),
                                 "early_stopped": int(d["stopped_early"].sum()),
                                 "dead_units": float(d["dead_units"].mean()),
                                 "seconds": round(time.time() - t0, 1), **m})
                    print(f"    AUROC {m['auroc']:.3f}  AUPRC {m['auprc']:.3f} "
                          f"({m['auprc_lift']:.2f}x)  Brier {m['brier']:.3f}  "
                          f"cal {m['cal_slope']:.2f}/{m['cal_intercept']:+.2f}  "
                          f"[{round(time.time() - t0, 1)}s]", flush=True)

                    g = gains_table(y, res["proba"])
                    g.insert(0, "seed", seed)
                    g.insert(0, "config", cname)
                    g.insert(0, "arch", arch)
                    g.insert(0, "outcome", outcome)
                    gains.append(g)

                    if race_col in sub.columns:
                        s = subgroups(y, res["proba"], res["logit"], sub[race_col],
                                      subgroup_min_n)
                        for col, val in (("seed", seed), ("config", cname),
                                         ("arch", arch), ("restricted_to", restricted_by or ""),
                                         ("outcome", outcome)):
                            s.insert(0, col, val)
                        subs.append(s)

                    dd = res["diagnostics"].copy()
                    for col, val in (("seed", seed), ("config", cname), ("arch", arch),
                                     ("outcome", outcome)):
                        dd.insert(0, col, val)
                    diags.append(dd)

                    if res["wide_coef"] is not None:
                        c = res["wide_coef"].rename("coef").reset_index()
                        c.columns = ["feature", "coef"]
                        c["source"] = res["coef_source"]
                        c["alpha_mean"] = res["alpha_mean"]
                        for col, val in (("seed", seed), ("config", cname), ("arch", arch),
                                         ("outcome", outcome)):
                            c.insert(0, col, val)
                        coefs.append(c)

                    if np.isfinite(res["gate"]).any():
                        gr = gate_report(y, res["gate"], sub, race_col, outcome,
                                         arch, cname, seed, subgroup_min_n)
                        if res["share"]:
                            for m, v in res["share"].items():
                                gr[f"share_{m}"] = v
                        gates.append(gr)
                        if res["share"]:
                            print("    contribution share: "
                                  + "  ".join(f"{m} {v:.1%}"
                                              for m, v in res["share"].items()), flush=True)

                summarise_seeds(perf, outcome, arch, cname, seeds)

            # Deltas per seed. Seeds are not ensembled, so there is no single
            # averaged prediction vector to bootstrap; instead each seed gives
            # its own delta and both variance sources get reported.
            for small, large in COMPARISONS:
                if not all((c, s) in store for c in (small, large) for s in seeds):
                    continue
                per_seed = []
                for seed in seeds:
                    dl = delta_auc(y, store[(small, seed)]["proba"],
                                   store[(large, seed)]["proba"], n_boot, seed)
                    deltas.append({"outcome": outcome, "restricted_to": restricted_by or "",
                                   "arch": arch, "baseline": small, "augmented": large,
                                   "seed": seed, "n_patients": len(y),
                                   "n_prevalent": int(y.sum()), **dl})
                    per_seed.append(dl["delta"])
                print(f"  {arch}: {large} vs {small}  dAUROC per seed "
                      f"{np.round(per_seed, 3).tolist()}  "
                      f"mean {np.mean(per_seed):+.3f}  seed sd {np.std(per_seed):.3f}",
                      flush=True)

    if not perf:
        raise SystemExit("nothing was fit")

    return {
        "performance": pd.DataFrame(perf),
        "deltas": pd.DataFrame(deltas),
        "gains": pd.concat(gains, ignore_index=True) if gains else pd.DataFrame(),
        "subgroups": pd.concat(subs, ignore_index=True) if subs else pd.DataFrame(),
        "gates": pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(),
        "coefficients": pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame(),
        "diagnostics": pd.concat(diags, ignore_index=True) if diags else pd.DataFrame(),
    }


def gate_report(y, gate, sub, race_col, outcome, arch, config, seed, min_n):
    """
    Distribution of the modality gate, overall and by racial group.

    The sd matters as much as the mean. A gate that has collapsed to a constant
    is not fusing anything -- it has settled on one modality for every patient --
    and without the spread that is indistinguishable from a genuine finding that
    both modalities contribute equally.
    """
    rows = [{"group": "ALL", "n": len(gate), "gate_mean": float(np.mean(gate)),
             "gate_sd": float(np.std(gate)),
             "gate_p10": float(np.percentile(gate, 10)),
             "gate_p90": float(np.percentile(gate, 90)),
             "collapsed": bool(np.std(gate) < 0.01)}]

    if race_col in sub.columns:
        g = pd.Series(sub[race_col]).fillna("Missing").astype(str).to_numpy()
        for name in sorted(set(g)):
            m = g == name
            if m.sum() < min_n:
                continue
            rows.append({"group": name, "n": int(m.sum()),
                         "gate_mean": float(np.mean(gate[m])),
                         "gate_sd": float(np.std(gate[m])),
                         "gate_p10": float(np.percentile(gate[m], 10)),
                         "gate_p90": float(np.percentile(gate[m], 90)),
                         "collapsed": bool(np.std(gate[m]) < 0.01)})

    out = pd.DataFrame(rows)
    for col, val in (("seed", seed), ("config", config), ("arch", arch),
                     ("outcome", outcome)):
        out.insert(0, col, val)
    return out


def summarise_seeds(perf, outcome, arch, config, seeds):
    """Mean and sd of AUROC across seeds -- the initialisation variability."""
    rows = [r for r in perf if r["outcome"] == outcome and r["arch"] == arch
            and r["config"] == config]
    if len(rows) < 2:
        return
    a = np.array([r["auroc"] for r in rows])
    print(f"    -> {arch}/{config} AUROC {a.mean():.3f} +/- {a.std():.3f} "
          f"over {len(a)} seeds", flush=True)


DEFAULT_HP = {
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "dropout": 0.3,
    "batch": 256,
    "max_epochs": 200,
    "patience": 10,
    "n_outer": 5,
    "val_frac": 0.2,
    # gated only: fit the clinical model first and add it as a fixed offset,
    # rather than learning it jointly with the towers. See GatedWideDeep.
    "clinical_offset": True,
    # Which demographic groups may be features. Race and ethnicity are out by
    # default and analysed post hoc; see common.DEFAULT_DEMOGRAPHICS.
    "demographics": DEFAULT_DEMOGRAPHICS,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None,
                    help="parquet from cohort.py; omit to run on simulated data")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--outcomes", nargs="+", default=list(OUTCOMES))
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None, help="cuda / mps / cpu; default auto")
    ap.add_argument("--lr", type=float, default=DEFAULT_HP["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_HP["weight_decay"])
    ap.add_argument("--dropout", type=float, default=DEFAULT_HP["dropout"])
    ap.add_argument("--batch", type=int, default=DEFAULT_HP["batch"])
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_HP["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULT_HP["patience"])
    ap.add_argument("--n-outer", type=int, default=DEFAULT_HP["n_outer"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-index-gap", type=int, default=None,
                    help="drop patients whose two images are more than this many days apart")
    ap.add_argument("--no-restrictions", action="store_true",
                    help=f"drop the pre-specified secondary analyses {RESTRICTIONS}")
    ap.add_argument("--race-col", default="Race")
    ap.add_argument("--subgroup-min-n", type=int, default=100)
    ap.add_argument("--demographics", nargs="*", default=list(DEFAULT_DEMOGRAPHICS),
                    choices=list(DEMOGRAPHIC_GROUPS),
                    help="which demographic groups may be used as FEATURES, in every "
                         "architecture. Default 'age bmi': race and ethnicity are held out "
                         "and analysed post hoc as stratifiers. Race remains the subgroup "
                         "stratifier either way.")
    ap.add_argument("--ignore-incident-eligibility", action="store_true")
    ap.add_argument("--no-clinical-offset", dest="clinical_offset", action="store_false",
                    help="gated arch: learn the clinical path jointly with the towers "
                         "instead of fitting it first as a fixed offset. Measured to leave "
                         "age/BMI coefficients near zero with arbitrary sign; kept for "
                         "comparison only")
    ap.add_argument("--simulate-n", type=int, default=6500)
    ap.add_argument("--simulate-dims", type=int, nargs=2, default=[512, 768])
    args = ap.parse_args()

    if args.cohort is None:
        df = simulate(n=args.simulate_n, d_mammo=args.simulate_dims[0],
                      d_cxr=args.simulate_dims[1])
    else:
        df = pd.read_parquet(args.cohort)
    df = cap_index_gap(df, args.max_index_gap)

    hp = {k: getattr(args, k) for k in
          ("lr", "weight_decay", "dropout", "batch", "max_epochs", "patience", "n_outer",
           "clinical_offset")}
    hp["demographics"] = tuple(args.demographics)

    out_tables = run(
        df, outcomes=args.outcomes, archs=tuple(args.archs),
        configs={k: CONFIGS[k] for k in args.configs},
        seeds=tuple(args.seeds), hp=hp, device=args.device, n_boot=args.n_boot,
        restrictions={} if args.no_restrictions else RESTRICTIONS,
        race_col=args.race_col, subgroup_min_n=args.subgroup_min_n,
        ignore_incident_eligibility=args.ignore_incident_eligibility)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in out_tables.items():
        if table is None or table.empty:
            continue
        fn = f"nl_{name}.csv"
        table.to_csv(out / fn, index=False)
        written.append(fn)
    (out / "nl_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nwrote {', '.join(written)} to {out}")


if __name__ == "__main__":
    main()

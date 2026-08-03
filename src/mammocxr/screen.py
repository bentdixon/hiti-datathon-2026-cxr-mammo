"""
Label-free screen over the anonymised embedding blocks.

With twelve mammography embeddings (A-F x ffdm/cview) and seven CXR embeddings
(model-a .. model-g), picking a pair by downstream AUROC and then reporting that
pair's delta-AUROC is selection on the outcome. This module ranks encoders using
only the embedding matrices, never the labels, so the choice costs no label
information and inflates no p-value.

Input is a built feature table -- the parquet cohort.py writes, or the
cohort parquet from cohort.py -- and every embedding block in it is screened,
discovered by column prefix. Screening a candidate encoder therefore means
building its table first; there is no per-family loader here, because reading
the raw embedding stores is cohort.py's job and duplicating it would
give two paths to the same matrix.

RankMe (Garrido et al. 2023) is the primary criterion. For an n x d embedding
matrix Z with singular values s_1 >= ... >= s_d, define
    p_k = s_k / (sum_j s_j) + eps
and
    RankMe(Z) = exp( - sum_k p_k * log p_k ).
The bracketed quantity is the Shannon entropy of the normalised singular value
spectrum; exponentiating it gives an effective rank, in units of dimensions.
Plain English: how many directions the embedding actually uses. A 1024-dim
embedding with RankMe 40 has collapsed almost all of its capacity; one with
RankMe 600 is spreading information across many directions. Higher is better,
and RankMe tracks downstream linear-probe accuracy without needing labels.

Also reported:
  n_eff_99  number of principal components holding 99% of variance
  cond      condition number s_1 / s_min, large values flag near-degeneracy
  mean_norm mean L2 length of a row, a check on the stated normalisation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Column prefixes that mark an embedding block. cohort.py names the
# columns mammo_0.. and cxr_0.., and cohort.py carries those names through.
DEFAULT_PREFIXES = ("mammo_", "cxr_")


def singular_values(Z):
    """Singular values via the Gram matrix. On a 6500 x 1536 matrix this is
    a 1536 x 1536 eigendecomposition rather than a full SVD."""
    Z = np.asarray(Z, dtype=np.float64)
    ev = np.linalg.eigvalsh(Z.T @ Z if Z.shape[0] >= Z.shape[1] else Z @ Z.T)
    return np.sqrt(np.clip(ev, 0, None))[::-1]


def rankme_from_sv(s, eps=1e-7):
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def rankme(Z, eps=1e-7):
    return rankme_from_sv(singular_values(Z), eps)


def quality(Z, name):
    """One eigendecomposition on the raw matrix, one on the centred matrix."""
    Z = np.asarray(Z, dtype=np.float64)
    s_raw = singular_values(Z)
    s_cen = singular_values(Z - Z.mean(0, keepdims=True))

    rm = rankme_from_sv(s_raw)
    var = np.cumsum(s_cen ** 2) / (s_cen ** 2).sum()
    return {
        "embedding": name,
        "n": Z.shape[0],
        "dim": Z.shape[1],
        "rankme": round(rm, 1),
        "rankme_frac": round(rm / Z.shape[1], 3),
        "n_eff_99": int(np.searchsorted(var, 0.99) + 1),
        "cond": round(float(s_cen[0] / max(s_cen[-1], 1e-12)), 1),
        "mean_norm": round(float(np.linalg.norm(Z, axis=1).mean()), 3),
    }


def blocks(df, prefixes=DEFAULT_PREFIXES):
    """
    {block name: its columns}, in table order, for every prefix that is present.

    A prefix with no columns is skipped rather than reported as an empty block,
    so a mammography-only table screens cleanly.
    """
    found = {}
    for p in prefixes:
        cols = [c for c in df.columns if str(c).startswith(p)]
        if cols:
            found[p.rstrip("_")] = cols
    return found


def screen(df, prefixes=DEFAULT_PREFIXES, out_path=None):
    """
    Every embedding block in a built feature table, ranked.

    Results are written after each block, so a run interrupted partway still
    leaves a usable table.
    """
    import time

    jobs = list(blocks(df, prefixes).items())
    if not jobs:
        raise SystemExit(
            f"no embedding columns matching {list(prefixes)} in the table. "
            f"present: {list(df.columns)[:12]}...")

    rows = []
    t_all = time.time()
    for i, (name, cols) in enumerate(jobs, 1):
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        try:
            Z = df[cols].to_numpy(dtype=np.float64)
            q = quality(Z, name)
            if q["rankme"] < 2:
                print(f"    WARNING: rankme {q['rankme']} means one direction dominates the "
                      f"spectrum; check for a raw-scale column mixed into the matrix")
            rows.append({**q, "seconds": round(time.time() - t0, 1)})
            print(f"    dim {Z.shape[1]}, n {Z.shape[0]}, rankme {rows[-1]['rankme']}, "
                  f"{rows[-1]['seconds']}s", flush=True)
        except Exception as e:
            rows.append({"embedding": name, "error": f"{type(e).__name__}: {e}"})
            print(f"    failed: {type(e).__name__}: {e}", flush=True)

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

        done, elapsed = i, time.time() - t_all
        print(f"    {elapsed / 60:.1f} min elapsed, ETA {elapsed / done * (len(jobs) - done) / 60:.1f} min\n",
              flush=True)

    out = pd.DataFrame(rows)
    if "rankme" not in out.columns:
        print("every embedding failed to load; check paths")
        return out
    return out.sort_values("rankme", ascending=False, na_position="last")


def _demo():
    rng = np.random.default_rng(0)
    n = 2000
    healthy = rng.normal(size=(n, 256))
    collapsed = rng.normal(size=(n, 8)) @ rng.normal(size=(8, 256))
    anisotropic = rng.normal(size=(n, 256)) * np.exp(-np.arange(256) / 20)
    for name, Z in [("healthy", healthy), ("collapsed", collapsed), ("anisotropic", anisotropic)]:
        print(quality(Z, name))


def _read(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=None,
                    help="parquet or csv holding the embedding columns; omit to run --demo")
    ap.add_argument("--prefixes", nargs="+", default=list(DEFAULT_PREFIXES),
                    help="column prefixes marking each embedding block")
    ap.add_argument("--out", default="results/encoder_screen.csv")
    ap.add_argument("--demo", action="store_true", help="sanity check on synthetic matrices")
    args = ap.parse_args()

    if args.demo or args.table is None:
        _demo()
    else:
        table = screen(_read(args.table), tuple(args.prefixes), out_path=args.out)
        print(table.to_string(index=False))
        if "error" in table.columns and table["error"].notna().any():
            print("\nfailed to screen:", table.loc[table["error"].notna(), "embedding"].tolist())
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
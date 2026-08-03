"""
End to end: build the cohort from the two raw embedding stores, then fit the
linear and nonlinear arms over a grid of outcomes/blocks/architectures/seeds.

    uv run mammocxr \\
        --mammo-embed-dir /path/to/mammo/embeddings \\
        --cxr-embed-dir /path/to/cxr/embeddings \\
        --cxr-meta /path/to/cxr_metadata.csv \\
        --cxr-icd /path/to/cxr_icd.csv \\
        --outdir results

Omit all four data-location flags to run the grid on simulated data instead --
useful for exercising the whole pipeline without access to the real stores.

This is the single entry point for the two data locations the pipeline needs:
cohort.py takes --mammo-embed-dir and --cxr-embed-dir (plus the two small
metadata tables) and returns the block-prefixed modelling table(s) in memory,
which are then handed directly to linear.py and nonlinear.py -- no
intermediate parquet round-trip is required to go from raw data to results.
"""

import argparse
import json
from pathlib import Path

from . import linear, nonlinear
from .common import simulate
from .cohort import build_cohort, OUTCOMES as COHORT_OUTCOMES


def run_linear_grid(df, args, outdir):
    if args.blocks:
        unknown = [b for b in args.blocks if b not in linear.BLOCKS]
        if unknown:
            raise SystemExit(f"unknown --blocks {unknown}; choose from {list(linear.BLOCKS)}")
        linear.BLOCKS = {k: v for k, v in linear.BLOCKS.items() if k in args.blocks}
        linear.COMPARISONS = [(a, b) for a, b in linear.COMPARISONS
                              if a in linear.BLOCKS and b in linear.BLOCKS]

    perf, deltas, gains, subs, coefs, sg_deltas = linear.run(
        df, outcomes=args.outcomes, n_boot=args.linear_n_boot, seed=args.seed,
        subgroup_boot=args.subgroup_boot)

    outdir.mkdir(parents=True, exist_ok=True)
    perf.to_csv(outdir / "performance.csv", index=False)
    deltas.to_csv(outdir / "deltas.csv", index=False)
    gains.to_csv(outdir / "gains.csv", index=False)
    written = ["performance.csv", "deltas.csv", "gains.csv"]
    for name, table in (("subgroups.csv", subs), ("subgroup_deltas.csv", sg_deltas),
                        ("coefficients.csv", coefs)):
        if not table.empty:
            table.to_csv(outdir / name, index=False)
            written.append(name)
    print(f"\nlinear arm wrote {', '.join(written)} to {outdir}")
    return perf


def run_nonlinear_grid(df, args, outdir):
    configs = {k: nonlinear.CONFIGS[k] for k in args.configs}
    out_tables = nonlinear.run(
        df, outcomes=args.outcomes, archs=tuple(args.archs), configs=configs,
        seeds=tuple(args.seeds), n_boot=args.nonlinear_n_boot)

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in out_tables.items():
        if table is None or table.empty:
            continue
        fn = f"nl_{name}.csv"
        table.to_csv(outdir / fn, index=False)
        written.append(fn)
    print(f"nonlinear arm wrote {', '.join(written)} to {outdir}")
    return out_tables.get("performance")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    data = ap.add_argument_group("cohort data locations (all four, or none for simulated data)")
    data.add_argument("--mammo-embed-dir", default=None,
                      help="directory with magview.csv and embeddings/{ffdm,cview}/*.parquet")
    data.add_argument("--cxr-embed-dir", default=None,
                      help="directory with one <empi_anon>/model-<encoder>_cxr.npz per patient")
    data.add_argument("--cxr-meta", default=None, help="CXR metadata csv")
    data.add_argument("--cxr-icd", default=None, help="ICD diagnosis codes csv")
    data.add_argument("--mammo-only", action="store_true", help="skip the fusion arm entirely")
    data.add_argument("--race-block", dest="with_race", action="store_true",
                      help="promote Race from stratifier to its own feature block")

    grid = ap.add_argument_group("grid")
    grid.add_argument("--outcomes", nargs="+", default=list(COHORT_OUTCOMES))
    grid.add_argument("--blocks", nargs="+", default=None,
                      help="linear arm: subset of block names to fit; default is all")
    grid.add_argument("--archs", nargs="+", default=list(nonlinear.ARCHS), choices=list(nonlinear.ARCHS))
    grid.add_argument("--configs", nargs="+", default=list(nonlinear.CONFIGS),
                      choices=list(nonlinear.CONFIGS))
    grid.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                      help="nonlinear arm seeds; the linear arm's CV seed is --seed")
    grid.add_argument("--seed", type=int, default=0)
    grid.add_argument("--linear-n-boot", type=int, default=2000)
    grid.add_argument("--nonlinear-n-boot", type=int, default=2000)
    grid.add_argument("--subgroup-boot", type=int, default=1000)
    grid.add_argument("--skip-linear", action="store_true")
    grid.add_argument("--skip-nonlinear", action="store_true")

    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    data_flags = [args.mammo_embed_dir, args.cxr_embed_dir, args.cxr_meta, args.cxr_icd]
    if any(data_flags) and not all(data_flags):
        raise SystemExit("pass all four of --mammo-embed-dir/--cxr-embed-dir/--cxr-meta/--cxr-icd, "
                         "or none of them to run on simulated data")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if all(data_flags):
        tables = build_cohort(args.mammo_embed_dir, args.cxr_embed_dir, args.cxr_meta,
                              args.cxr_icd, mammo_only=args.mammo_only, with_race=args.with_race)
        for name, df in tables.items():
            df.to_parquet(outdir / f"cohort_{name}.parquet")
        df_mammo = tables["mammo"]
        df_fusion = tables.get("fusion", df_mammo)
    else:
        print("no data locations given; running the grid on simulated data")
        df_mammo = df_fusion = simulate()

    (outdir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    if not args.skip_linear:
        print("\n=== linear arm ===")
        run_linear_grid(df_fusion, args, outdir)

    if not args.skip_nonlinear:
        print("\n=== nonlinear arm ===")
        run_nonlinear_grid(df_fusion, args, outdir)


if __name__ == "__main__":
    main()

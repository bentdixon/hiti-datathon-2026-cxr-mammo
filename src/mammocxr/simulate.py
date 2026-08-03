"""
CLI for common.simulate() -- write a synthetic cohort parquet in the same
demo_/race_/tech_/mammo_/cxr_ block layout cohort.py produces.

Every entry point in this pipeline (main.py, linear.py, nonlinear.py) already
falls back to simulate() in memory when no cohort/data-location flags are
given, so this module adds nothing new to the fallback path. What it adds is a
first-class command: writing that same table to a parquet file puts the demo
path on equal footing with the real one -- `--cohort <path>` works identically
whether the path came from cohort.py or from here, and the simulated table can
be inspected, diffed, or handed to someone else without re-running Python.

    uv run mammocxr-simulate --out cohort_simulated.parquet
    uv run mammocxr-linear --cohort cohort_simulated.parquet --outdir results
"""

import argparse
from pathlib import Path

from .common import simulate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="cohort_simulated.parquet")
    ap.add_argument("--n", type=int, default=6500, help="number of simulated patients")
    ap.add_argument("--d-mammo", type=int, default=512, help="mammography embedding width")
    ap.add_argument("--d-cxr", type=int, default=768, help="chest X-ray embedding width")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = simulate(n=args.n, d_mammo=args.d_mammo, d_cxr=args.d_cxr, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {out}  {df.shape}  ({args.n} patients, "
          f"{args.d_mammo}-d mammo, {args.d_cxr}-d cxr, seed {args.seed})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_experiment.streamlit_dataset import build_and_export_streamlit_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "streamlit_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the precomputed dataset library for the Streamlit companion app.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the precomputed parquet/CSV files should be written.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Build a reduced validation grid instead of the full production library.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional process count for the full build. Defaults to a conservative auto value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets, output_paths, manifest = build_and_export_streamlit_dataset(
        output_dir=args.output_dir,
        smoke=args.smoke,
        workers=args.workers,
    )

    print("Streamlit dataset build complete.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(
        "Rows: "
        f"sweep={len(datasets['sweep_results'])}, "
        f"snapshots={len(datasets['portfolio_snapshots'])}, "
        f"summary={len(datasets['summary_metrics'])}"
    )
    print(
        "Build mode: "
        f"{'smoke' if manifest['smoke_mode'] else 'full'} "
        f"with {manifest['workers']} worker(s)."
    )
    print("Exported files:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

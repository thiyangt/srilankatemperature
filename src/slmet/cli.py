"""Command-line entry points: ``slmet <subcommand> ...``"""
from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_DATE_START


def _cmd_ingest(args):
    from .ingest import rename_and_copy
    report = rename_and_copy(args.src, args.dst, overwrite=args.overwrite)
    if args.report_csv:
        report.to_csv(args.report_csv, index=False)
    counts = report["status"].value_counts()
    print(counts.to_string())


def _cmd_missing(args):
    from .missing import find_missing_in_dir
    missing = find_missing_in_dir(args.dir, args.start, args.end)
    if args.out_csv:
        missing.to_csv(args.out_csv, index=False)
    print(f"{len(missing)} missing (date, measure) pairs out of the expected range")
    print(missing.to_string(index=False))


def _cmd_build_stations(args):
    from .stations import DEFAULT_REGISTRY_PATH, build_registry, discover_image_paths, save_registry
    paths = discover_image_paths(args.image_dir)
    df = build_registry(paths, max_images=args.max_images)
    out = args.out or DEFAULT_REGISTRY_PATH
    save_registry(df, out)
    print(f"Wrote {len(df)} stations to {out}")
    print(df.to_string(index=False))


def _cmd_build_dataset(args):
    from .dataset import build_dataset
    from .stations import load_registry
    registry = load_registry(args.stations) if args.stations else None
    df = build_dataset(args.image_dir, registry=registry, cache_path=args.out,
                       diagnostics=args.diagnostics)
    print(f"Extracted {len(df)} rows from {args.image_dir} -> {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="slmet", description=(
        "Extract tidy min/max temperature data from Sri Lanka Dept. of "
        "Meteorology map images."))
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("ingest", help="Rename/copy raw images into '<Measure> Temperature - YYYY-MM-DD.jpg'")
    sp.add_argument("src", help="Directory of raw downloaded images")
    sp.add_argument("dst", help="Destination directory for renamed copies")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--report-csv", default=None)
    sp.set_defaults(func=_cmd_ingest)

    sp = sub.add_parser("missing", help="List missing (date, measure) pairs in a date range")
    sp.add_argument("dir", help="Directory of renamed images")
    sp.add_argument("--start", default=DEFAULT_DATE_START)
    sp.add_argument("--end", required=True)
    sp.add_argument("--out-csv", default=None)
    sp.set_defaults(func=_cmd_missing)

    sp = sub.add_parser("build-stations", help="Build the canonical station registry from a corpus of images")
    sp.add_argument("image_dir")
    sp.add_argument("--out", default=None, help="Defaults to the package's bundled data/stations.csv")
    sp.add_argument("--max-images", type=int, default=60)
    sp.set_defaults(func=_cmd_build_stations)

    sp = sub.add_parser("build-dataset", help="Extract the full tidy dataset from a directory of images")
    sp.add_argument("image_dir")
    sp.add_argument("--out", required=True)
    sp.add_argument("--stations", default=None, help="Path to a station registry CSV")
    sp.add_argument("--diagnostics", action="store_true",
                    help="Also output label_ocr_guess, label_ocr_score and station_source")
    sp.set_defaults(func=_cmd_build_dataset)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

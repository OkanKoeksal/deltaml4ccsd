import argparse
import shutil
from pathlib import Path


def read_xyz_list(path: Path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise RuntimeError(f"No entries found in {path}")
    return lines


def resolve_existing_xyz(old_path_string: str, structures_dir: Path) -> Path:
    """
    processed_xyz_files_final.txt may contain old absolute paths.
    We resolve each entry by using the basename inside the current Structures directory.
    """
    old_path = Path(old_path_string)

    if old_path.exists():
        return old_path

    candidate = structures_dir / old_path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find structure:\n"
        f"  listed path: {old_path_string}\n"
        f"  tried basename path: {candidate}"
    )


def standardize_names(
    xyz_list_file: Path,
    structures_dir: Path,
    output_dir: Path,
    output_list_file: Path,
    prefix: str,
    start_index: int,
    mode: str,
    dry_run: bool,
):
    listed_paths = read_xyz_list(xyz_list_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_list_file.parent.mkdir(parents=True, exist_ok=True)

    new_paths = []
    seen_sources = set()

    for i, old_path_string in enumerate(listed_paths, start=start_index):
        source = resolve_existing_xyz(old_path_string, structures_dir)

        if source in seen_sources:
            raise RuntimeError(f"Duplicate source structure detected: {source}")
        seen_sources.add(source)

        new_name = f"{prefix}{i}.xyz"
        target = output_dir / new_name

        if target.exists() and target.resolve() != source.resolve():
            raise FileExistsError(f"Target file already exists: {target}")

        new_paths.append(target)

        print(f"{source}  ->  {target}")

        if dry_run:
            continue

        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "move":
            shutil.move(str(source), str(target))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if not dry_run:
        with open(output_list_file, "w") as f:
            for path in new_paths:
                f.write(str(path) + "\n")

    print()
    print(f"Number of structures processed: {len(listed_paths)}")
    print(f"Mode: {mode}")
    print(f"Dry run: {dry_run}")
    print(f"Output directory: {output_dir}")
    print(f"Output xyz list: {output_list_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standardize mixed xyz filenames using the row order in "
            "processed_xyz_files_final.txt."
        )
    )

    parser.add_argument(
        "--xyz-list",
        required=True,
        help="Path to processed_xyz_files_final.txt.",
    )
    parser.add_argument(
        "--structures-dir",
        required=True,
        help="Directory containing the current mixed-name xyz files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write standardized xyz files.",
    )
    parser.add_argument(
        "--output-list",
        required=True,
        help="Path to write updated xyz list with standardized names.",
    )
    parser.add_argument(
        "--prefix",
        default="structure",
        help="Prefix for standardized files. Default: structure",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for standardized names. Default: 0",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Use copy first. Move only after verification.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the operation. Without this, only a dry run is printed.",
    )

    args = parser.parse_args()

    standardize_names(
        xyz_list_file=Path(args.xyz_list),
        structures_dir=Path(args.structures_dir),
        output_dir=Path(args.output_dir),
        output_list_file=Path(args.output_list),
        prefix=args.prefix,
        start_index=args.start_index,
        mode=args.mode,
        dry_run=not args.execute,
    )


if __name__ == "__main__":
    main()